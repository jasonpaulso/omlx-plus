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
