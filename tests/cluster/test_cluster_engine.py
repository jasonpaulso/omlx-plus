# SPDX-License-Identifier: Apache-2.0
"""ClusterEngine surface (S3 D5), driven over a mock rank pipe.

No ranks, no mlx: a mock cluster's ``write``/``read_reply`` pair replaces the
real ``LocalCluster``, so the engine's D5 demux -- concurrent streams sharing
one reply pipe, abort, queue-full, EOF fan-out -- can be exercised without a
formation. There is no ``RequestOutputCollector`` — the engine consumes the
rank-0 reply frames directly.
"""

from __future__ import annotations

import asyncio

import pytest

from omlx.cluster.engine import ClusterEngine, ClusterNonGoalError
from omlx.exceptions import SchedulerQueueFullError


class StubTokenizer:
    def encode(self, text):
        return [1, 2, 3]


class MockCluster:
    """``write`` records; ``read_reply`` drains one shared, pre-scripted
    frame list -- exactly what the real pipe looks like from the demux's
    point of view: one ordered stream of frames, routed by ``request_id``.

    Once the scripted frames run out, ``read_reply`` returns ``None``
    (matching the real idle-timeout contract) rather than raising, so the
    background reader task quietly idles instead of a spurious pipe-closed
    failure racing a slow-scheduled consumer that hasn't drained its last
    legitimate frame yet. Pass ``eof=True`` for tests that specifically want
    the pipe-closed path.
    """

    world_size = 2
    backend = "ring"

    def __init__(self, frames, *, eof: bool = False):
        self._frames = list(frames)
        self._eof = eof
        self.aborted: list[str] = []
        self.stopped = False
        self.written: list[dict] = []

    def write(self, payload):
        self.written.append(payload)

    def read_reply(self, timeout=None):
        if self._frames:
            return self._frames.pop(0)
        if self._eof:
            raise RuntimeError("rank 0 closed its reply channel")
        return None

    def abort(self, request_id=""):
        self.aborted.append(request_id)
        return True

    def stop(self):
        self.stopped = True


@pytest.fixture
async def make_engine():
    """Builds a ClusterEngine over a MockCluster and stops it (cancelling the
    D5 demux reader task) at teardown -- without this, the reader's
    background loop over a MockCluster with no more frames spins forever."""
    created: list[ClusterEngine] = []

    def _make(frames, **cluster_kwargs):
        engine = ClusterEngine(
            "m", cluster=MockCluster(frames, **cluster_kwargs), resolved_path="/x"
        )
        engine._loaded = True
        engine._tokenizer = StubTokenizer()
        created.append(engine)
        return engine

    yield _make
    for engine in created:
        await engine.stop()


def _chunks(request_id, text_parts, done):
    frames = [
        {"ok": True, "request_id": request_id, "chunk": part, "tokens": i + 1}
        for i, part in enumerate(text_parts)
    ]
    frames.append(done)
    return frames


NORMAL = _chunks(
    "r",
    ["Hel", "lo"],
    {
        "ok": True,
        "request_id": "r",
        "done": True,
        "text": "Hello",
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "finish_reason": "length",
    },
)


async def test_stream_generate_yields_chunks_then_final(make_engine):
    engine = make_engine(NORMAL)
    outputs = [
        o async for o in engine.stream_generate("hi", max_tokens=8, request_id="r")
    ]
    assert "".join(o.new_text for o in outputs[:-1]) == "Hello"
    final = outputs[-1]
    assert final.finished is True
    assert final.finish_reason == "length"
    assert final.text == "Hello"
    assert final.completion_tokens == 2


async def test_generate_returns_final_only(make_engine):
    engine = make_engine(NORMAL)
    output = await engine.generate("hi", max_tokens=8, request_id="r")
    assert output.text == "Hello"
    assert output.finish_reason == "length"
    assert output.completion_tokens == 2


async def test_stream_chat_applies_template_then_streams(make_engine):
    engine = make_engine(NORMAL)
    outputs = [
        o
        async for o in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=8, request_id="r"
        )
    ]
    assert outputs[-1].finished is True
    assert outputs[-1].text == "Hello"


async def test_client_disconnect_forwards_an_abort(make_engine):
    engine = make_engine(NORMAL)
    cluster = engine._cluster

    stream = engine.stream_generate("hi", max_tokens=8, request_id="r")
    await stream.__anext__()
    await stream.aclose()  # client disconnect
    assert "r" in cluster.aborted


async def test_stop_finalization_trims_trailing_stop(make_engine):
    frames = _chunks(
        "r",
        ["ab", "c#"],
        {
            "ok": True,
            "request_id": "r",
            "done": True,
            "text": "abc#",
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "finish_reason": "stop",
        },
    )
    engine = make_engine(frames)
    output = await engine.generate("hi", max_tokens=8, stop=["#"], request_id="r")
    assert output.finish_reason == "stop"
    assert output.text == "abc"


async def test_rank_error_frame_raises(make_engine):
    frames = [{"ok": False, "request_id": "r", "error": "rank died"}]
    engine = make_engine(frames)
    with pytest.raises(RuntimeError, match="rank died"):
        [o async for o in engine.stream_generate("hi", request_id="r")]


async def test_queue_full_error_frame_raises_typed_exception(make_engine):
    """D5: a queue_full-coded error frame must reconstruct
    SchedulerQueueFullError, not a generic RuntimeError, so the existing
    FastAPI handler (server.py's scheduler_queue_full_handler) maps it to a
    real 503 the same way single-node's own SchedulerQueueFullError does."""
    frames = [
        {
            "ok": False,
            "request_id": "r",
            "error": "Scheduler waiting queue full: 32 >= 32",
            "code": "queue_full",
            "current_depth": 32,
            "max_depth": 32,
        }
    ]
    engine = make_engine(frames)
    with pytest.raises(SchedulerQueueFullError) as excinfo:
        [o async for o in engine.stream_generate("hi", request_id="r")]
    assert excinfo.value.current_depth == 32
    assert excinfo.value.max_depth == 32


def test_stats_and_type(make_engine):
    engine = make_engine(NORMAL)
    engine._model_type_value = "llama"
    stats = engine.get_stats()
    assert stats["engine_type"] == "cluster-distributed"
    assert stats["world_size"] == 2
    assert engine.model_type == "llama"
    assert engine.get_cache_stats() is None


# -- S3 D5: demux over one shared reply pipe ----------------------------------


async def test_two_concurrent_requests_interleave_correctly(make_engine):
    """Frames for two requests arrive interleaved on the one reply pipe; the
    demux must route each frame to the request it names, not to whichever
    stream_generate call started first."""
    frames = [
        {"ok": True, "request_id": "a", "chunk": "A1", "tokens": 1},
        {"ok": True, "request_id": "b", "chunk": "B1", "tokens": 1},
        {"ok": True, "request_id": "a", "chunk": "A2", "tokens": 2},
        {
            "ok": True,
            "request_id": "b",
            "done": True,
            "text": "B1B2",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "finish_reason": "length",
        },
        {
            "ok": True,
            "request_id": "a",
            "done": True,
            "text": "A1A2",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "finish_reason": "length",
        },
    ]
    engine = make_engine(frames)

    async def collect(request_id):
        return [
            o
            async for o in engine.stream_generate(
                "hi", max_tokens=8, request_id=request_id
            )
        ]

    outputs_a, outputs_b = await asyncio.gather(collect("a"), collect("b"))
    assert outputs_a[-1].text == "A1A2"
    assert outputs_b[-1].text == "B1B2"
    assert all(o.new_text in ("A1", "A2", "") for o in outputs_a)
    assert all(o.new_text in ("B1", "B2", "") for o in outputs_b)


async def test_abort_one_leaves_the_other_streaming(make_engine):
    frames = [
        {"ok": True, "request_id": "keep", "chunk": "K1", "tokens": 1},
        {
            "ok": True,
            "request_id": "abort-me",
            "done": True,
            "text": "",
            "prompt_tokens": 1,
            "completion_tokens": 0,
            "finish_reason": "abort",
        },
        {
            "ok": True,
            "request_id": "keep",
            "done": True,
            "text": "K1K2",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "finish_reason": "length",
        },
    ]
    engine = make_engine(frames)

    async def collect(request_id):
        return [
            o
            async for o in engine.stream_generate(
                "hi", max_tokens=8, request_id=request_id
            )
        ]

    kept, aborted = await asyncio.gather(collect("keep"), collect("abort-me"))
    assert kept[-1].text == "K1K2"
    assert kept[-1].finish_reason == "length"
    assert aborted[-1].finish_reason == "abort"


async def test_active_request_count_reflects_in_flight_streams(make_engine):
    frames = [
        {
            "ok": True,
            "request_id": "r",
            "done": True,
            "text": "x",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "finish_reason": "length",
        }
    ]
    engine = make_engine(frames)
    assert engine.has_active_requests() is False
    assert engine.get_stats()["num_active_requests"] == 0

    stream = engine.stream_generate("hi", max_tokens=8, request_id="r")
    await stream.__anext__()  # submits, registers the pending queue
    assert engine.has_active_requests() is True
    assert engine.get_stats()["num_active_requests"] == 1

    async for _ in stream:
        pass
    assert engine.has_active_requests() is False
    assert engine.get_stats()["num_active_requests"] == 0


async def test_reply_pipe_eof_fails_every_pending_stream(make_engine):
    """The reader dying (rank 0's pipe closed) must fan the failure out to
    EVERY request waiting on it, not just the one whose frame happened to be
    read last -- otherwise a concurrent stream hangs to its idle timeout
    instead of failing promptly."""
    engine = make_engine([], eof=True)

    async def collect(request_id):
        return [
            o
            async for o in engine.stream_generate(
                "hi", max_tokens=8, request_id=request_id
            )
        ]

    with pytest.raises(RuntimeError):
        await asyncio.gather(collect("x"), collect("y"))


# -- S3 non-goals: VLM / SpecPrefill rejected with a clear error -------------


async def test_preflight_chat_rejects_multimodal_content(make_engine):
    engine = make_engine(NORMAL)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
    ]
    with pytest.raises(ClusterNonGoalError):
        await engine.preflight_chat(messages)


async def test_preflight_chat_rejects_specprefill(make_engine):
    engine = make_engine(NORMAL)
    with pytest.raises(ClusterNonGoalError):
        await engine.preflight_chat(
            [{"role": "user", "content": "hi"}], specprefill=True
        )


async def test_preflight_chat_allows_plain_text(make_engine):
    engine = make_engine(NORMAL)
    await engine.preflight_chat([{"role": "user", "content": "hi"}])


async def test_preflight_completion_rejects_specprefill(make_engine):
    engine = make_engine(NORMAL)
    with pytest.raises(ClusterNonGoalError):
        await engine.preflight_completion("hi", specprefill=True)


# -- head-side waiting-queue gate (row 4 carry-forward) ----------------------


class TestPreflightQueueGate:
    """Rank 0's own ``add_request`` cap cannot reach a streaming client: by
    the time the generate command crosses the pipe the route has returned a
    ``StreamingResponse`` and HTTP 200 is on the wire, so the rejection
    arrives as an SSE ``error`` event (S3 P3 acceptance row 4, both
    backends). The head therefore gates in preflight, from its own view of
    what rank 0 is holding.
    """

    @staticmethod
    def _fill(engine, count):
        for i in range(count):
            engine._pending[f"inflight-{i}"] = asyncio.Queue()

    async def test_admits_below_rank_capacity(self, make_engine):
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        # One preflight per engine: a second one against the same 39 in-flight
        # requests is the 41st request overall, and preflight now reserves the
        # slot it checked for, so that one is correctly refused.
        chat_engine = make_engine(NORMAL)
        self._fill(chat_engine, rank_inflight_capacity() - 1)
        await chat_engine.preflight_chat([{"role": "user", "content": "hi"}])

        completion_engine = make_engine(NORMAL)
        self._fill(completion_engine, rank_inflight_capacity() - 1)
        await completion_engine.preflight_completion("hi")

    async def test_rejects_at_rank_capacity(self, make_engine):
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        engine = make_engine(NORMAL)
        self._fill(engine, rank_inflight_capacity())
        with pytest.raises(SchedulerQueueFullError) as excinfo:
            await engine.preflight_chat([{"role": "user", "content": "hi"}])
        # Reported the way rank 0 would: everything past the running batch is
        # queue depth, so a client sees the same numbers as on single-node.
        assert excinfo.value.current_depth == 32
        assert excinfo.value.max_depth == 32

    async def test_completion_path_is_gated_too(self, make_engine):
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        engine = make_engine(NORMAL)
        self._fill(engine, rank_inflight_capacity())
        with pytest.raises(SchedulerQueueFullError):
            await engine.preflight_completion("hi")

    async def test_gate_runs_before_the_non_goal_checks(self, make_engine):
        """A saturated formation should answer 503, not a non-goal 400 — the
        request is retryable and nothing about it was inspected yet."""
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        engine = make_engine(NORMAL)
        self._fill(engine, rank_inflight_capacity())
        with pytest.raises(SchedulerQueueFullError):
            await engine.preflight_chat(
                [{"role": "user", "content": "hi"}], specprefill=True
            )

    def test_capacity_derives_from_one_cap_definition(self):
        from omlx.cluster.scheduler_config import (
            rank_inflight_capacity,
            rank_max_num_seqs,
        )
        from omlx.scheduler import waiting_queue_capacity

        max_num_seqs = rank_max_num_seqs()
        assert rank_inflight_capacity() == max_num_seqs + waiting_queue_capacity(
            max_num_seqs
        )
        # The plan's burst recipe: cap + max_num_seqs + 1 submissions is what
        # it takes to force a rejection, so the 41st of 41 is the one refused.
        assert rank_inflight_capacity() + 1 == 41

    async def test_cold_burst_is_refused_before_any_generator_runs(self, make_engine):
        """The shape the live rig measured, and the reason the first fix did
        nothing there.

        On a cold burst every request preflights before starlette iterates any
        response body, so ``_pending`` -- filled inside ``stream_generate`` --
        is still empty when the gate reads it. Gating on ``_pending`` alone
        therefore admits all 41 and the cap is only hit later, in-stream,
        under HTTP 200. Note what this test does *not* do: it does not
        pre-fill anything. That is the whole point.
        """
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        engine = make_engine(NORMAL)
        capacity = rank_inflight_capacity()

        admitted = 0
        rejected = 0
        for _ in range(capacity + 1):
            try:
                await engine.preflight_chat([{"role": "user", "content": "hi"}])
            except SchedulerQueueFullError:
                rejected += 1
            else:
                admitted += 1

        assert engine._pending == {}, "no generator ran: this is the cold shape"
        assert admitted == capacity
        assert rejected == 1

    async def test_admission_hands_the_reservation_back(self, make_engine):
        """A reservation covers preflight->submit only. Once the request is in
        ``_pending`` it is counted there, and counting it twice would shrink
        the formation's usable capacity by one per request.
        """
        engine = make_engine(NORMAL)
        await engine.preflight_chat([{"role": "user", "content": "hi"}])
        assert engine._reserved_slots() == 1

        outputs = [
            o async for o in engine.stream_generate("hi", max_tokens=8, request_id="r")
        ]

        assert outputs[-1].finished is True
        assert engine._reserved_slots() == 0
        assert engine._pending == {}

    async def test_reservation_is_released_when_the_request_never_admits(
        self, make_engine
    ):
        """``stream_generate`` can fail or return before reaching ``_pending``
        (empty prompt, a non-goal, a dead pipe). The slot has to come back on
        those paths too, or a stream of malformed requests would wall off the
        formation for a full TTL.
        """
        engine = make_engine(NORMAL)
        await engine.preflight_chat([{"role": "user", "content": "hi"}])

        engine._tokenizer = type("Empty", (), {"encode": lambda self, text: []})()
        outputs = [o async for o in engine.stream_generate("", request_id="empty")]

        # Exactly the empty-prompt early return, not an ordinary completion:
        # one empty terminal output and nothing submitted.
        assert len(outputs) == 1
        assert outputs[0].text == "" and outputs[0].finished is True
        assert engine._pending == {}
        assert engine._reserved_slots() == 0

    async def test_abandoned_reservations_expire(self, make_engine, monkeypatch):
        """The one path no explicit release can cover: a request that passes
        preflight and never reaches ``stream_generate`` at all -- a client
        that disconnects after the route committed. The TTL sweep is what
        keeps those from accumulating into a permanent capacity loss.
        """
        import omlx.cluster.engine as engine_module
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        monkeypatch.setattr(engine_module, "_RESERVATION_TTL_S", 0.05)
        engine = make_engine(NORMAL)
        for _ in range(rank_inflight_capacity()):
            await engine.preflight_chat([{"role": "user", "content": "hi"}])
        with pytest.raises(SchedulerQueueFullError):
            await engine.preflight_chat([{"role": "user", "content": "hi"}])

        await asyncio.sleep(0.06)
        await engine.preflight_chat([{"role": "user", "content": "hi"}])
        assert engine._reserved_slots() == 1

    async def test_reservations_and_inflight_share_one_ceiling(self, make_engine):
        """Occupancy is what rank 0 holds plus what is on its way there. Half
        of each must still add up to the same ceiling, or the gate would
        double-count during churn and reject early.
        """
        from omlx.cluster.scheduler_config import rank_inflight_capacity

        capacity = rank_inflight_capacity()
        engine = make_engine(NORMAL)
        self._fill(engine, capacity // 2)
        for _ in range(capacity - capacity // 2):
            await engine.preflight_chat([{"role": "user", "content": "hi"}])
        with pytest.raises(SchedulerQueueFullError):
            await engine.preflight_chat([{"role": "user", "content": "hi"}])

    async def test_backstop_frame_is_logged_head_side(self, make_engine, caplog):
        """The preflight->submit race stays open, so rank 0 can still refuse
        a request after the response is committed. That path was invisible:
        no head-side log line at all, which is half of what made row 4 hard
        to see from the daemon.
        """
        import logging

        frames = [
            {
                "ok": False,
                "request_id": "late",
                "error": "Scheduler waiting queue full: 32 >= 32",
                "code": "queue_full",
                "current_depth": 32,
                "max_depth": 32,
            }
        ]
        engine = make_engine(frames)
        with (
            caplog.at_level(logging.WARNING, logger="omlx.cluster.engine"),
            pytest.raises(SchedulerQueueFullError),
        ):
            [o async for o in engine.stream_generate("hi", request_id="late")]
        assert any("waiting queue full" in r.getMessage() for r in caplog.records)
