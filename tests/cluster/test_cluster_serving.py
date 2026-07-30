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
from typing import Any

import httpx
import pytest

from omlx.cluster import launcher
from omlx.cluster.client import ClusterClient
from omlx.cluster.manager import ClusterManager, set_cluster_manager
from omlx.exceptions import SchedulerQueueFullError

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


async def test_worker_rank_death_propagates_degrades_and_re_forms(tmp_path):
    """S6 D1 rev2: the measured ~385s hang, closed.

    Killing the WORKER's own rank (not the head's rank 0, above) leaves the
    head's rank 0 blocked in the collective with NOTHING locally to notice
    -- only the worker's next heartbeat (`ranks_status()`) tells the head.
    Before this fix nothing ever told it; the formation job stayed
    "running" forever and the request only failed at the 600s
    generate-idle timeout. This must degrade + tear the formation down
    within a couple of heartbeat intervals instead, and a re-issued load
    must re-form cleanly.

    `_degrade_and_teardown` delegates the actual entry rollback/accounting
    to the S4 unload driver (`EnginePool.request_unload`); this stub
    mirrors exactly what that driver does (`_teardown_cluster_entry`
    calling `manager.formation.unload`) without pulling in the whole
    EnginePool, which this in-process harness never wires.
    """
    from omlx.cluster.manager import set_engine_pool_getter

    async with two_node(tmp_path) as (head, worker):
        teardown_calls: list[str] = []

        class _StubPool:
            async def request_unload(self, model_id: str) -> None:
                teardown_calls.append(model_id)
                await head.formation.unload(model_id)

        set_engine_pool_getter(lambda: _StubPool())
        try:
            await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
            engine = head.formation.active_engine(MODEL)
            assert engine is not None
            assert worker.executor is not None
            worker_cluster = worker.executor.cluster
            assert worker_cluster is not None

            for entry in worker_cluster.ranks:
                entry.process.kill()

            # Degrade + teardown from heartbeat propagation alone, bounded
            # well under the 600s generate-idle timeout that used to be the
            # only thing that ever noticed.
            await _wait_for(lambda: head.formation.active_engine(MODEL) is None, 30.0)
            assert teardown_calls == [MODEL]
            assert any(job.status == "degraded" for job in head.formation._jobs)
            assert any("degraded" in alarm for alarm in head.formation.alarms())

            reformed = await asyncio.wait_for(
                head.formation.load(MODEL), LOAD_TIMEOUT_S
            )
            assert reformed["status"] == "ready"
            again = await asyncio.wait_for(
                head.formation.active_engine(MODEL).generate(
                    "Hi", max_tokens=4, temperature=0.0
                ),
                GEN_TIMEOUT_S,
            )
            assert again.completion_tokens > 0
            await asyncio.wait_for(head.formation.unload(MODEL), LOAD_TIMEOUT_S)
        finally:
            set_engine_pool_getter(None)


# -- S3 P2: real scheduler-driven concurrent batching -------------------------
#
# Every test below drives ``engine.stream_generate``/``stream_chat`` directly
# against the live formation, the same idiom the rest of this file already
# uses (``test_full_lifecycle`` etc.) -- there is no HTTP route/engine-pool
# wiring anywhere in this file (only the control-plane join/heartbeat goes
# over ASGI), so these prove the D5 demux and the rank-0 Scheduler's real
# admission/backpressure over the actual pipe, not a mocked one.


async def test_two_concurrent_streams_interleave(tmp_path):
    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None

        async def collect(request_id, prompt):
            return [
                o
                async for o in engine.stream_generate(
                    prompt,
                    max_tokens=12,
                    temperature=0.0,
                    request_id=request_id,
                )
            ]

        outputs_a, outputs_b = await asyncio.wait_for(
            asyncio.gather(
                collect("interleave-a", "Count from one: one, two,"),
                collect("interleave-b", "The capital of France is"),
            ),
            GEN_TIMEOUT_S,
        )
        assert outputs_a and outputs_a[-1].finished
        assert outputs_b and outputs_b[-1].finished
        assert outputs_a[-1].completion_tokens > 0
        assert outputs_b[-1].completion_tokens > 0
        # Genuinely two different completions -- the demux did not cross the
        # streams (a request seeing the other's tokens would corrupt this).
        assert outputs_a[-1].text != outputs_b[-1].text


async def test_abort_one_stream_other_completes(tmp_path):
    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None

        keep_stream = engine.stream_generate(
            "The capital of France is",
            max_tokens=16,
            temperature=0.0,
            request_id="keep-alive",
        )
        abort_stream = engine.stream_generate(
            "Tell me a very long story",
            max_tokens=256,
            temperature=0.0,
            request_id="abort-target",
        )

        # Admit both concurrently (first token/chunk each) before acting, so
        # the abort genuinely lands on a request the scheduler is running
        # alongside the one that will keep going.
        await asyncio.wait_for(
            asyncio.gather(keep_stream.__anext__(), abort_stream.__anext__()),
            GEN_TIMEOUT_S,
        )
        await asyncio.wait_for(
            abort_stream.aclose(), GEN_TIMEOUT_S
        )  # client disconnect

        kept = []
        async for o in keep_stream:
            kept.append(o)
        assert kept and kept[-1].finished
        assert kept[-1].completion_tokens > 0


async def test_active_request_count_reflects_live_formation(tmp_path):
    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None
        assert engine.has_active_requests() is False
        assert engine.get_stats()["num_active_requests"] == 0

        stream = engine.stream_generate(
            "Tell me a very long story",
            max_tokens=64,
            temperature=0.0,
            request_id="active-count",
        )
        await asyncio.wait_for(stream.__anext__(), GEN_TIMEOUT_S)
        assert engine.has_active_requests() is True
        assert engine.get_stats()["num_active_requests"] == 1

        await stream.aclose()
        assert engine.has_active_requests() is False
        assert engine.get_stats()["num_active_requests"] == 0


async def test_cold_burst_is_refused_at_preflight_not_in_stream(tmp_path):
    """S3 acceptance row 4, in the ordering the live rig actually produces.

    The sibling test below submits straight into ``stream_generate``, which is
    the *backstop* path: it proves rank 0 still refuses, but by then a real
    route has committed to a ``StreamingResponse`` and the client can only be
    told in-stream, under HTTP 200. Row 4 asks for a 503, which only preflight
    can deliver.

    The ordering matters more than the count. Over HTTP, all 41 requests clear
    preflight before starlette iterates any response body, so the two phases
    here are separated on purpose — driving preflight and submission together
    from one task would let each request's ``_pending`` entry land before the
    next one preflights, which is the warm shape, not row 4's. The deployed
    ``_pending``-only gate passes that version of this test and still measured
    ``{200: 41}`` on the rig.

    Mirrors ``benchmarks/cluster_spike/s3_row4.py``: >=1 rejection AND >=1
    request still streaming. Rejecting the whole burst would satisfy the first
    clause alone while being worse than the defect.
    """
    num_submissions = 41
    max_tokens_burst = 4096

    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None

        # Phase 1 — every request preflights, cold: nothing has submitted yet.
        verdicts = await asyncio.gather(
            *(
                engine.preflight_chat([{"role": "user", "content": "Tell me a story"}])
                for _ in range(num_submissions)
            ),
            return_exceptions=True,
        )
        rejected = [v for v in verdicts if isinstance(v, SchedulerQueueFullError)]
        unexpected = [
            v
            for v in verdicts
            if isinstance(v, BaseException)
            and not isinstance(v, SchedulerQueueFullError)
        ]
        assert not unexpected, f"preflight raised something else: {unexpected}"
        assert rejected, (
            "expected at least one preflight rejection on a cold "
            f"{num_submissions}-request burst; got none"
        )
        assert len(rejected) < num_submissions, (
            "the whole burst was rejected — a gate that refuses everything "
            "passes 'a 503 happened' while being worse than the defect"
        )

        # Phase 2 — the admitted ones must still be servable. Preflight that
        # holds a slot forever would show up here as nothing streaming.
        admitted = [
            i for i, v in enumerate(verdicts) if not isinstance(v, BaseException)
        ]
        streams: dict[int, Any] = {}
        streaming: list[int] = []

        async def submit(i: int) -> None:
            stream = engine.stream_generate(
                "Tell me a very long, detailed story",
                max_tokens=max_tokens_burst,
                temperature=0.0,
                request_id=f"cold-{i}",
            )
            try:
                await stream.__anext__()
            except SchedulerQueueFullError:
                # rank 0's backstop: allowed, the preflight->submit race is
                # open by design and the burst can outrun a stepping loop.
                return
            streams[i] = stream
            streaming.append(i)

        tasks = [asyncio.create_task(submit(i)) for i in admitted]
        try:
            # Not gather(): only max_num_seqs of these ever emit a first token
            # within the test, the rest park in rank 0's waiting queue behind
            # 4096-token generations. Wait for the outcome under test.
            await _wait_for(lambda: bool(streaming), GEN_TIMEOUT_S)
            assert streaming, "expected admitted requests to stream"
        finally:
            for task in tasks:
                task.cancel()
            for stream in streams.values():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(stream.aclose(), GEN_TIMEOUT_S)


async def test_queue_full_rejects_while_earlier_requests_keep_streaming(tmp_path):
    """The pinned S3 recipe (s3-plan.md D7/Tests): max_num_seqs = 8 (the D6(b)
    rank-side default) makes the waiting-queue cap max(8*4, 32) = 32
    (scheduler.py:6750); since admission pops up to max_num_seqs requests out
    of ``waiting`` per step, >= cap + max_num_seqs + 1 = 41 concurrent
    submissions are required to guarantee at least one lands after the cap is
    hit. max_tokens is large enough that nothing finishes during the burst.
    """
    num_submissions = 41
    max_tokens_burst = 4096

    async with two_node(tmp_path) as (head, worker):
        await asyncio.wait_for(head.formation.load(MODEL), LOAD_TIMEOUT_S)
        engine = head.formation.active_engine(MODEL)
        assert engine is not None

        streams: list[Any] = [None] * num_submissions
        outcomes: list[str] = [""] * num_submissions

        async def submit(i: int) -> None:
            stream = engine.stream_generate(
                "Tell me a very long, detailed story",
                max_tokens=max_tokens_burst,
                temperature=0.0,
                request_id=f"burst-{i}",
            )
            try:
                await stream.__anext__()
            except SchedulerQueueFullError:
                outcomes[i] = "queue_full"
                return
            streams[i] = stream
            outcomes[i] = "streaming"

        tasks = [asyncio.create_task(submit(i)) for i in range(num_submissions)]
        try:
            # Deliberately NOT gather(): most of these submissions never settle
            # within the test, by design. Only max_num_seqs (8) requests reach
            # the running batch and emit a first token; the next 32 fill the
            # waiting queue and stay parked there behind 4096-token
            # generations -- that saturation is the precondition the burst
            # exists to create -- and only the overflow past the cap raises.
            # Waiting on all 41 would therefore always time out, whatever the
            # scheduler does. Wait for the two outcomes actually under test.
            await _wait_for(
                lambda: any(o == "queue_full" for o in outcomes)
                and any(o == "streaming" for o in outcomes),
                GEN_TIMEOUT_S,
            )

            rejected = [o for o in outcomes if o == "queue_full"]
            streaming = [o for o in outcomes if o == "streaming"]
            assert rejected, (
                "expected at least one SchedulerQueueFullError under the "
                f"{num_submissions}-submission burst; outcomes={outcomes}"
            )
            assert streaming, "expected earlier submissions to still be streaming"
        finally:
            # Disconnect every admitted stream so the formation tears down
            # cleanly rather than waiting out 4096-token generations.
            for task in tasks:
                task.cancel()
            for stream in streams:
                if stream is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(stream.aclose(), GEN_TIMEOUT_S)
