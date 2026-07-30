# SPDX-License-Identifier: Apache-2.0
"""S4 P2 integration: EnginePool coexistence on a real two-node formation.

Same single-process, two-``ClusterManager``, real-rank-process harness as
``test_cluster_serving.py`` (real ASGI transport between head and worker,
real loopback ring formation) -- but exercising the POOL path
(``EnginePool.get_engine``/``load_cluster_model``) instead of calling
``FormationManager.load`` directly, per S4's acceptance: pool-path
formation with a ``kind="cluster"`` entry, preview-equals-recorded-decision,
and LRU-pressure clean unform + reload.

Double-marked ``cluster`` + ``integration`` so the default unit gate
collects none of it. Hard timeouts everywhere; children are killed in
teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncIterator

import httpx
import pytest

from omlx.cluster import launcher
from omlx.cluster.client import ClusterClient
from omlx.cluster.manager import (
    ClusterManager,
    set_cluster_manager,
    set_engine_pool_getter,
)
from omlx.cluster.placement import (
    plan_placement,
    resolve_placement_inputs,
    worker_node_capacity,
)
from omlx.engine_pool import EnginePool

from .conftest import build_app, make_settings

pytestmark = [pytest.mark.cluster, pytest.mark.integration]

MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# The real model is ~730MB estimated; per-rank share (world_size=2, 1.15
# headroom) is ~420MB -- a ceiling comfortably between the two forces
# mode="distributed" under auto placement without fitting locally.
CEILING = 500_000_000
LOAD_TIMEOUT_S = 180.0
GEN_TIMEOUT_S = 60.0
JOIN_TIMEOUT_S = 10.0


class ASGIClusterClient(ClusterClient):
    """A control-plane client that reaches the head's ASGI app in-process."""

    def __init__(self, base_url: str, transport: httpx.ASGITransport) -> None:
        super().__init__(base_url)
        self._transport = transport

    def _build(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            follow_redirects=False,
            transport=self._transport,
        )


def _settings(tmp_path, role, port):
    return make_settings(
        tmp_path / role,
        role=role,
        data_plane_subnet="127.0.0.0/8",
        data_plane_address="127.0.0.1",
        data_plane_base_port=port,
        backend="ring",
    )


def _worker_active(head: ClusterManager) -> bool:
    return any(
        head.liveness(m.id) is not None and head.liveness(m.id).status == "active"
        for m in head.state.members
    )


async def _wait_for(predicate, timeout, interval=0.05):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met in time")


@contextlib.asynccontextmanager
async def two_node_with_pool(
    tmp_path,
) -> AsyncIterator[tuple[ClusterManager, ClusterManager, EnginePool, str]]:
    port = random.randint(43000, 46000)
    head = ClusterManager(_settings(tmp_path, "head", port))
    await head.start()
    set_cluster_manager(head)

    pool = EnginePool()
    pool._get_final_ceiling = lambda: CEILING
    model_dirs = [str(p) for p in head.global_settings.get_effective_model_dirs()]
    pool.discover_models(model_dirs)
    set_engine_pool_getter(lambda: pool)

    model_id = next(
        (mid for mid, e in pool._entries.items() if e.source_repo_id == MODEL_REPO),
        None,
    )
    if model_id is None:
        set_engine_pool_getter(None)
        set_cluster_manager(None)
        await head.stop()
        pytest.skip(f"{MODEL_REPO} not present in a discoverable model dir")

    app = build_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 40404))

    worker = ClusterManager(
        _settings(tmp_path, "worker", port),
        client_factory=lambda base_url: ASGIClusterClient(base_url, transport),
    )
    await worker.start()
    try:
        token = (await head.mint_bootstrap_token())["token"]
        await worker.local_join("http://head.test", token)
        await _wait_for(lambda: _worker_active(head), JOIN_TIMEOUT_S)
        # Let the worker's first real heartbeat carry its node_state (D1)
        # before any test drives placement off it.
        await _wait_for(
            lambda: head.node_state(head.state.members[0].id) is not None,
            JOIN_TIMEOUT_S,
        )
        yield head, worker, pool, model_id
    finally:
        set_engine_pool_getter(None)
        set_cluster_manager(None)
        with contextlib.suppress(Exception):
            await worker.stop()
        with contextlib.suppress(Exception):
            await head.stop()
        launcher.sweep_orphaned_ranks()


async def test_pool_path_forms_cluster_entry_and_serves(tmp_path):
    async with two_node_with_pool(tmp_path) as (head, worker, pool, model_id):
        engine = await asyncio.wait_for(pool.get_engine(model_id), LOAD_TIMEOUT_S)
        assert engine is not None

        entry = pool.get_entry(model_id)
        assert entry.kind == "cluster"
        assert entry.engine is engine
        assert pool.current_model_memory == entry.cluster_head_share

        # A second get_engine call is the zero-I/O fast path -- no reload.
        again = await pool.get_engine(model_id)
        assert again is engine

        out = await asyncio.wait_for(
            engine.generate("Hello", max_tokens=8, temperature=0.0), GEN_TIMEOUT_S
        )
        assert out.completion_tokens > 0
        assert out.text


async def test_explicit_load_records_decision_equal_to_preview(tmp_path):
    async with two_node_with_pool(tmp_path) as (head, worker, pool, model_id):
        # A preview taken before the load, on the still-quiesced head.
        entry = pool.get_entry(model_id)
        est_size, model_config = resolve_placement_inputs(entry.model_path)
        head_capacity = pool.head_capacity()
        workers = []
        for candidate in head.state.members:
            live = head.liveness(candidate.id)
            node_state = head.node_state(candidate.id)
            if live is not None and live.status == "active" and node_state is not None:
                workers.append(worker_node_capacity(candidate.id, node_state))
        preview = plan_placement(
            model_id=model_id,
            model_type=entry.model_type,
            est_size=est_size,
            model_config=model_config,
            head=head_capacity,
            workers=workers,
            prefer="auto",
        )
        assert preview.mode == "distributed"

        result = await asyncio.wait_for(
            head.load_distributed(model_id, prefer="auto"), LOAD_TIMEOUT_S
        )
        recorded = result["decision"]

        status = head.formation_status()
        assert status["jobs"][-1]["decision"] == recorded

        # The placement-determining fields match the pre-load preview
        # (S4 acceptance row 3's equality domain).
        for field in ("mode", "world_size", "per_rank_estimate", "divisible"):
            assert recorded[field] == getattr(preview, field)
        assert recorded["presence"] == preview.presence


async def test_lru_pressure_unforms_cleanly_then_reload_works(tmp_path):
    async with two_node_with_pool(tmp_path) as (head, worker, pool, model_id):
        engine = await asyncio.wait_for(pool.get_engine(model_id), LOAD_TIMEOUT_S)
        assert engine is not None
        entry = pool.get_entry(model_id)
        assert entry.kind == "cluster"
        head_share = entry.cluster_head_share

        # Simulate LRU pressure: mark the cluster entry idle-evictable and
        # let the out-of-lock driver unform it (mirrors what get_engine's
        # admission-eviction branch does when this entry is the LRU
        # victim -- exercised directly here for a deterministic trigger).
        async with pool._lock:
            pool._mark_pending_unload_locked(model_id, "lru pressure (test)")
            event = pool._cluster_unload_completion_event_locked(model_id)
            pool._wake_cluster_unload_driver_locked()
        await asyncio.wait_for(event.wait(), LOAD_TIMEOUT_S)

        unformed = pool.get_entry(model_id)
        assert unformed.kind == "local"
        assert unformed.engine is None
        assert unformed.estimated_size != head_share
        assert pool.current_model_memory == 0
        assert head.formation.active_engine(model_id) is None
        # Workers scrubbed: no lingering formation on the head side.
        assert head.formation._active_model is None

        # A reload works cleanly after the clean unform.
        reloaded = await asyncio.wait_for(pool.get_engine(model_id), LOAD_TIMEOUT_S)
        assert reloaded is not None
        assert pool.get_entry(model_id).kind == "cluster"


async def test_flag_off_zero_delta_role_off_never_reaches_placement(tmp_path):
    """`auto_placement=False` on the same wired head restores S3 behavior:
    a too-big model raises the plain too-big error, no formation reached.
    """
    from omlx.exceptions import ModelTooLargeError

    async with two_node_with_pool(tmp_path) as (head, worker, pool, model_id):
        head.settings.auto_placement = False
        with pytest.raises(ModelTooLargeError):
            await pool.get_engine(model_id)
        assert pool.get_entry(model_id).kind == "local"
        assert head.formation._active_model is None
