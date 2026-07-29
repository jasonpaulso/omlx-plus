# SPDX-License-Identifier: Apache-2.0
"""P2 integration: full distributed serving on a single machine.

A real head daemon (installed as the process manager) and a real worker manager
run in one process; the worker heartbeats to the head over real HTTP
(``ASGITransport`` through the FastAPI cluster router, auth and all), the head's
formation job commands the worker over the D2 channel, and two loopback rank
processes form a ring, shard-load a small model, and serve through the
``ClusterEngine``.

Double-marked ``cluster`` + ``integration`` so the default unit gate collects
none of it. Hard timeouts everywhere; children are killed in teardown.
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
from omlx.cluster.manager import ClusterManager, set_cluster_manager

from .conftest import build_app, make_settings

pytestmark = [pytest.mark.cluster, pytest.mark.integration]

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
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


@contextlib.asynccontextmanager
async def two_node(tmp_path) -> AsyncIterator[tuple[ClusterManager, ClusterManager]]:
    port = random.randint(43000, 46000)
    head = ClusterManager(_settings(tmp_path, "head", port))
    await head.start()
    set_cluster_manager(head)

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
        yield head, worker
    finally:
        set_cluster_manager(None)
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


async def _wait_for(predicate, timeout, interval=0.05):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met in time")


# -- full lifecycle -----------------------------------------------------------


async def test_full_lifecycle(tmp_path):
    async with two_node(tmp_path) as (head, worker):
        assert head.formation is not None
        result = await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        assert result["status"] == "ready"
        engine = head.formation.active_engine(MODEL)
        assert engine is not None

        # generate
        out = await asyncio.wait_for(
            engine.generate("Hello", max_tokens=8, temperature=0.0), GEN_TIMEOUT_S
        )
        assert out.completion_tokens > 0
        assert out.text

        # stream
        chunks = []
        async for output in engine.stream_generate(
            "Count:", max_tokens=8, temperature=0.0
        ):
            chunks.append(output)
        assert chunks and chunks[-1].finished

        # abort (client disconnect mid-stream)
        stream = engine.stream_generate("Long", max_tokens=256, request_id="ab")
        await stream.__anext__()
        await stream.aclose()

        # the group survives an abort: another request still works
        again = await asyncio.wait_for(
            engine.generate("Hi", max_tokens=4, temperature=0.0), GEN_TIMEOUT_S
        )
        assert again.completion_tokens > 0

        # unload
        unloaded = await asyncio.wait_for(head.formation.unload(MODEL), LOAD_TIMEOUT_S)
        assert unloaded["status"] == "unloaded"
        assert head.formation.active_engine(MODEL) is None


# -- greedy parity dist-vs-single-node ----------------------------------------


async def test_greedy_parity_with_single_node(tmp_path):
    from omlx.engine.batched import BatchedEngine

    prompt = "The capital of France is"
    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None
        dist = await asyncio.wait_for(
            engine.generate(prompt, max_tokens=16, temperature=0.0), GEN_TIMEOUT_S
        )
        await asyncio.wait_for(head.formation.unload(MODEL), LOAD_TIMEOUT_S)

    single = BatchedEngine(MODEL)
    try:
        await single.start()
        ref = await single.generate(prompt, max_tokens=16, temperature=0.0)
    finally:
        await single.stop()

    # Greedy TP is bit-for-bit deterministic on the token stream; the two paths
    # detokenize/finalize trailing whitespace differently (the streaming rank
    # detokenizer vs the batched path), so normalise the trailing edge. Any real
    # mid-stream argmax divergence still fails this assertion.
    assert dist.text.rstrip() == ref.text.rstrip()
    assert dist.text.rstrip()


# -- presence gap fails, nothing forms ----------------------------------------


async def test_missing_model_fails_and_spawns_nothing(tmp_path):
    from omlx.cluster.manager import ClusterError

    async with two_node(tmp_path) as (head, worker):
        with pytest.raises(ClusterError):
            await asyncio.wait_for(
                head.formation.load("mlx-community/does-not-exist-42"), LOAD_TIMEOUT_S
            )
        assert head.formation.active_engine("mlx-community/does-not-exist-42") is None
        # No rank process is left behind.
        assert launcher.sweep_orphaned_ranks() == 0


# -- rank death mid-generation -> clean error + teardown ----------------------


async def test_rank_death_surfaces_clean_error(tmp_path):
    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None
        leader = head.formation._local
        assert leader is not None

        # Killing rank 0 (the head's own rank) closes its reply pipe, which the
        # engine surfaces at once as a clean RuntimeError rather than a hang.
        # (A worker-rank death is the harder cross-node case: the head's
        # deathwatch cannot see a wedged-but-alive rank 0, so that path is
        # bounded only by the generate-idle timeout in S2 — see s2-security
        # notes / S3.)
        with pytest.raises(RuntimeError):
            index = 0
            async for _output in engine.stream_generate(
                "Tell me a long story", max_tokens=256, temperature=0.0
            ):
                index += 1
                if index == 3:
                    for entry in leader.ranks:
                        entry.process.kill()
