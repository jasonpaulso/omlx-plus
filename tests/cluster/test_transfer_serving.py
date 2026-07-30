# SPDX-License-Identifier: Apache-2.0
"""S5 P2 integration: model distribution over the REAL control plane.

Follows test_cluster_serving.py's two-process convention: a real head
ClusterManager and a real worker ClusterManager, the worker heartbeating to
the head over real HTTP (``ASGITransport`` through the FastAPI cluster
router, auth and all). Unlike that file, formation and transfer are driven
through a real ``EnginePool.load_cluster_model`` (S5 D5's own entry point,
not `head.formation.load` directly), so the presence-resolution pre-step,
the transfer job, and formation all run end-to-end over the authenticated
command/heartbeat channel.

Only the process-spawn boundaries are faked (formation's rank spawn on both
ends, and a transfer round's `mlx_lm.share` ring session) -- real ring/share
hardware is P3's rig job, not pytest's (mirrors test_pool_coexistence.py and
test_transfer.py's own precedent).

Double-marked ``cluster`` + ``integration`` so the default unit gate collects
none of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from omlx.cluster import launcher
from omlx.cluster.client import ClusterClient
from omlx.cluster.formation import FormationManager
from omlx.cluster.manager import (
    ClusterError,
    ClusterManager,
    WorkerCommandExecutor,
    set_cluster_manager,
    set_engine_pool_getter,
)
from omlx.cluster.transfer import TransferWorkerExecutor
from omlx.engine_pool import EnginePool

from .conftest import build_app, make_settings

pytestmark = [pytest.mark.cluster, pytest.mark.integration]

JOIN_TIMEOUT_S = 10.0
JOB_TIMEOUT_S = 30.0

# Same shape as test_pool_coexistence.py: world_size=2, divisible head counts.
_CONFIG = {"model_type": "llama", "num_attention_heads": 8, "num_key_value_heads": 8}
_HEAD_CEILING = 7_000_000
_WORKER_CEILING = 9_000_000


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


def _settings(tmp_path, role, port, **overrides):
    opts = {
        "data_plane_subnet": "127.0.0.0/8",
        "data_plane_address": "127.0.0.1",
        "data_plane_base_port": port,
        "backend": "ring",
    }
    opts.update(overrides)
    return make_settings(tmp_path / role, role=role, **opts)


class FakeLeader:
    def wait_ready(self, timeout=0.0):
        return {"event": "ready"}

    def stop(self):
        return None


class FakeEngine:
    async def start(self):
        return None

    async def stop(self):
        return None

    def get_stats(self):
        return {"engine_type": "cluster-distributed", "loaded": True}

    def has_active_requests(self):
        return False


class FakeRankCluster:
    def __init__(self) -> None:
        self._alive = True

    def any_alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False


def _fake_session_launcher(file_contents_by_round: list[dict[str, bytes]]):
    """Like test_transfer.py's fake, but takes an ORDERED list of
    per-round content maps -- round N (1-indexed by call count) delivers
    `file_contents_by_round[N-1]` (or the last entry, once exhausted), so a
    test can model "round 1 loses/corrupts a file, round 2 delivers it"
    without a real interrupted process.
    """
    calls = {"n": 0}

    class _FakeProcess:
        def wait(self, timeout=None):
            return 0

    class _FakeSessionLeader:
        def __init__(self):
            self.process = _FakeProcess()

    class _FakeSession:
        def __init__(self):
            self.leader = _FakeSessionLeader()

        def stop(self):
            pass

        def kill(self):
            pass

    def launcher_fn(*, rank, world_size, ips, base_port, argv_builder, **kwargs):
        idx = min(calls["n"], len(file_contents_by_round) - 1)
        calls["n"] += 1
        file_contents = file_contents_by_round[idx]
        argv = argv_builder(rank)
        manifest_path = Path(argv[argv.index("--manifest") + 1])
        staging_dir = Path(argv[argv.index("--root") + 1])
        entries = json.loads(manifest_path.read_text())
        for entry in entries:
            data = file_contents.get(entry["relative_path"])
            if data is None:
                continue
            target = staging_dir / entry["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return _FakeSession()

    return launcher_fn


# The fixture model's exact bytes, known upfront so a test can hand
# `_fake_session_launcher` the right content without round-tripping through
# the harness to discover it.
_MODEL_FILES: dict[str, bytes] = {
    "config.json": json.dumps(_CONFIG).encode(),
    "model.safetensors": b"W" * 10_000_000,
    "tokenizer.model": b"T" * 1024,
}


def _recording_launcher(inner, sink: list[set[str]]):
    """Wraps a fake session launcher to also record each round's requested
    relative-path subset -- the acceptance claim is "only the missing
    subset re-sent", which mtime alone doesn't prove (an `os.replace` of
    byte-identical content would also touch mtime).
    """

    def launcher_fn(*, rank, world_size, ips, base_port, argv_builder, **kwargs):
        real_argv_builder = argv_builder

        def _recording_argv_builder(rank_):
            argv = real_argv_builder(rank_)
            manifest_path = Path(argv[argv.index("--manifest") + 1])
            entries = json.loads(manifest_path.read_text())
            sink.append({e["relative_path"] for e in entries})
            return argv

        return inner(
            rank=rank,
            world_size=world_size,
            ips=ips,
            base_port=base_port,
            argv_builder=_recording_argv_builder,
            **kwargs,
        )

    return launcher_fn


def _make_head_pool(
    tmp_path, *, model_id: str = "target", ceiling: int = _HEAD_CEILING
):
    model_dir = tmp_path / "head" / "models" / model_id
    model_dir.mkdir(parents=True)
    for name, data in _MODEL_FILES.items():
        (model_dir / name).write_bytes(data)

    pool = EnginePool()
    pool._get_final_ceiling = lambda c=ceiling: c
    pool.discover_models(str(tmp_path / "head" / "models"))
    assert pool.get_entry(model_id) is not None
    return pool, model_dir


@contextlib.asynccontextmanager
async def two_node_transfer(
    tmp_path, *, session_launcher
) -> AsyncIterator[tuple[ClusterManager, ClusterManager, EnginePool, str, Path]]:
    port = random.randint(43000, 46000)
    pool, model_dir = _make_head_pool(tmp_path)
    model_id = "target"

    head = ClusterManager(_settings(tmp_path, "head", port))
    await head.start()
    set_cluster_manager(head)
    head._formation = FormationManager(
        head,
        spawn_leader_fn=lambda **kw: FakeLeader(),
        engine_factory=lambda **kw: FakeEngine(),
        model_resolver=lambda mid: (
            pool.get_entry(mid).model_path if pool.get_entry(mid) is not None else None
        ),
    )
    # head._transfer stays the real TransferManager `start()` created.

    app = build_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 40404))

    worker_settings = _settings(tmp_path, "worker", port)
    worker_root = worker_settings.get_effective_model_dirs()[0]
    worker_root.mkdir(parents=True, exist_ok=True)
    worker = ClusterManager(
        worker_settings,
        client_factory=lambda base_url: ASGIClusterClient(base_url, transport),
    )
    # `_worker_memory_ceiling()`/`TransferWorkerExecutor._pool_conflict` both
    # read the process-global `get_engine_pool()` -- point it at a bare
    # ceiling-only stub (never at `pool`, the HEAD's own pool: sharing that
    # would make `_pool_conflict` see the head's just-claimed
    # `is_loading=True` entry and falsely refuse TRANSFER_START, the exact
    # single-process trap this harness must avoid). The stub has no entries
    # of its own, so `_pool_conflict` always reads "not present" -- correct,
    # since this fake worker never actually loads anything.
    worker_pool_stub = EnginePool()
    worker_pool_stub._get_final_ceiling = lambda: _WORKER_CEILING
    set_engine_pool_getter(lambda: worker_pool_stub)
    await worker.start()
    # Replace the real (rank/subprocess-spawning) executor with one whose
    # spawn + transfer session are faked -- no real ring hardware needed
    # (mirrors test_worker_executor.py's FakeCluster + test_transfer.py's
    # fake session_launcher), BEFORE local_join so the heartbeat sender
    # `_start_heartbeat` wires to THIS executor. `start()` already started
    # the ORIGINAL executor's own command loop -- stop it before swapping,
    # and start the replacement's, or `deliver()` enqueues into a queue
    # nothing ever drains.
    old_executor = worker._executor
    worker._executor = WorkerCommandExecutor(
        worker.global_settings,
        spawn_fn=lambda prepared: FakeRankCluster(),
        local_addresses={"127.0.0.1"},
        transfer_executor=TransferWorkerExecutor(
            worker.global_settings, session_launcher=session_launcher
        ),
    )
    if old_executor is not None:
        await old_executor.stop()
    await worker._executor.start()
    try:
        token = (await head.mint_bootstrap_token())["token"]
        await worker.local_join("http://head.test", token)
        await _wait_for(lambda: _worker_active(head), JOIN_TIMEOUT_S)
        yield head, worker, pool, model_id, worker_root
    finally:
        set_cluster_manager(None)
        set_engine_pool_getter(None)
        with contextlib.suppress(Exception):
            await worker.stop()
        with contextlib.suppress(Exception):
            await head.stop()
        launcher.sweep_orphaned_ranks()


def _worker_active(head: ClusterManager) -> bool:
    return any(
        head.liveness(m.id) is not None and head.liveness(m.id).status == "active"
        for m in head.state.members
    )


async def _wait_for(predicate, timeout, interval=0.02):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met in time")


def _latest_transfer_job(head: ClusterManager) -> dict[str, Any] | None:
    assert head.transfer is not None
    jobs = head.transfer.snapshot()["jobs"]
    return jobs[-1] if jobs else None


# -- acceptance row 1: plain distributed load transfers then forms ------------


async def test_absent_on_worker_transfers_then_forms_zero_preemptive_action(tmp_path):
    session_launcher = _fake_session_launcher([dict(_MODEL_FILES)])
    async with two_node_transfer(tmp_path, session_launcher=session_launcher) as (
        head,
        worker,
        pool,
        model_id,
        worker_root,
    ):
        result = await asyncio.wait_for(
            pool.load_cluster_model(model_id, "auto"), JOB_TIMEOUT_S
        )
        assert result["status"] == "ready"
        assert result["decision"]["mode"] == "distributed"

        entry = pool.get_entry(model_id)
        assert entry.kind == "cluster"
        assert entry.engine is not None

        job = _latest_transfer_job(head)
        assert job is not None
        assert job["status"] == "done"
        for name in _MODEL_FILES:
            assert (worker_root / model_id / name).read_bytes() == _MODEL_FILES[name]


# -- acceptance row 2: interrupted transfer resumes file-granular -------------


async def test_interrupted_transfer_resumes_file_granular(tmp_path):
    # Round 1 "loses" tokenizer.model (models an interrupted/killed worker --
    # only config.json + model.safetensors land); the job stalls on the
    # still-missing entry and errors out after the round cap.
    partial = {k: v for k, v in _MODEL_FILES.items() if k != "tokenizer.model"}
    session_launcher = _fake_session_launcher([partial])
    async with two_node_transfer(tmp_path, session_launcher=session_launcher) as (
        head,
        worker,
        pool,
        model_id,
        worker_root,
    ):
        with pytest.raises(ClusterError):
            await asyncio.wait_for(
                pool.load_cluster_model(model_id, "auto"), JOB_TIMEOUT_S
            )
        first_job = _latest_transfer_job(head)
        assert first_job is not None
        assert first_job["status"] == "error"

        verified_before = worker_root / model_id / "model.safetensors"
        assert verified_before.exists()
        mtime_before = verified_before.stat().st_mtime_ns

        # Re-issue: this time the launcher delivers everything, but a real
        # worker's own `have` scan (D2's diff authority) must recognise the
        # already-verified files and never re-touch them -- only request the
        # still-missing subset.
        round_subsets: list[set[str]] = []
        assert worker.executor is not None
        worker.executor.transfer._session_launcher = _recording_launcher(
            _fake_session_launcher([dict(_MODEL_FILES)]), round_subsets
        )

        result = await asyncio.wait_for(
            pool.load_cluster_model(model_id, "auto"), JOB_TIMEOUT_S
        )
        assert result["status"] == "ready"

        second_job = _latest_transfer_job(head)
        assert second_job is not None
        assert second_job["status"] == "done"

        # The primary claim: round 1 of the re-issued job requests ONLY the
        # still-missing file -- config.json/model.safetensors are never
        # re-sent (mtime alone can't prove this: an os.replace of
        # byte-identical content would also touch it).
        assert round_subsets == [{"tokenizer.model"}]
        assert verified_before.stat().st_mtime_ns == mtime_before
        for name in _MODEL_FILES:
            assert (worker_root / model_id / name).read_bytes() == _MODEL_FILES[name]


# -- acceptance row 3: corrupted staged file is detected, deleted, re-fetched -


async def test_corrupted_staged_file_detected_deleted_refetched(tmp_path):
    corrupted = dict(_MODEL_FILES)
    corrupted["tokenizer.model"] = b"CORRUPTED-BYTES-WRONG-DIGEST"
    # Round 1 corrupts tokenizer.model; round 2+ delivers it correctly.
    session_launcher = _fake_session_launcher([corrupted, dict(_MODEL_FILES)])
    async with two_node_transfer(tmp_path, session_launcher=session_launcher) as (
        head,
        worker,
        pool,
        model_id,
        worker_root,
    ):
        result = await asyncio.wait_for(
            pool.load_cluster_model(model_id, "auto"), JOB_TIMEOUT_S
        )
        assert result["status"] == "ready"

        job = _latest_transfer_job(head)
        assert job is not None
        assert job["status"] == "done"
        assert job["rounds_completed"] >= 2

        # The corrupted bytes never reached the final dir -- what's there now
        # digest-matches the real content.
        for name in _MODEL_FILES:
            assert (worker_root / model_id / name).read_bytes() == _MODEL_FILES[name]


# -- acceptance row 4: abort mid-transfer leaves clean staging, no formation --


async def test_abort_mid_transfer_cleans_up_and_never_forms(tmp_path):
    session_launcher = _fake_session_launcher([dict(_MODEL_FILES)])
    async with two_node_transfer(tmp_path, session_launcher=session_launcher) as (
        head,
        worker,
        pool,
        model_id,
        worker_root,
    ):
        load_task = asyncio.create_task(pool.load_cluster_model(model_id, "auto"))

        await _wait_for(lambda: _latest_transfer_job(head) is not None, JOIN_TIMEOUT_S)
        job_id = _latest_transfer_job(head)["id"]
        assert head.transfer is not None
        await head.transfer.abort_transfer(job_id)

        with pytest.raises(ClusterError):
            await asyncio.wait_for(load_task, JOB_TIMEOUT_S)

        job = head.transfer.job(job_id)
        assert job is not None
        assert job.status == "aborted"

        entry = pool.get_entry(model_id)
        assert entry.is_loading is False
        assert entry.kind == "local"
        # Formation was never attempted: no engine, no active model.
        assert head.formation is not None
        assert head.formation.active_engine(model_id) is None
        assert head.formation.snapshot()["active_model"] is None
        # No file ever lands under the model's final dir (staging is
        # discarded, never partially promoted).
        assert not (worker_root / model_id).exists() or not any(
            (worker_root / model_id).iterdir()
        )
