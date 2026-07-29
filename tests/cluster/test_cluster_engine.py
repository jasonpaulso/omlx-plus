# SPDX-License-Identifier: Apache-2.0
"""ClusterEngine surface (D6), driven over a mock rank pipe.

No ranks, no mlx: a mock cluster yields canned reply frames so the engine's
streaming, non-streaming, client-disconnect abort, and stop finalization can be
exercised without a formation. There is no RequestOutputCollector — the engine
consumes the rank-0 reply frames directly.
"""

from __future__ import annotations

import pytest

from omlx.cluster.engine import ClusterEngine


class StubTokenizer:
    def encode(self, text):
        return [1, 2, 3]


class MockCluster:
    world_size = 2
    backend = "ring"

    def __init__(self, frames):
        self._frames = frames
        self.aborted: list[str] = []
        self.stopped = False

    def stream(self, payload, timeout=None):
        yield from self._frames

    def abort(self, request_id=""):
        self.aborted.append(request_id)
        return True

    def stop(self):
        self.stopped = True


def _engine(frames):
    engine = ClusterEngine("m", cluster=MockCluster(frames), resolved_path="/x")
    engine._loaded = True
    engine._tokenizer = StubTokenizer()
    return engine


def _chunks(text_parts, done):
    frames = [
        {"ok": True, "request_id": "r", "chunk": part, "tokens": i + 1}
        for i, part in enumerate(text_parts)
    ]
    frames.append(done)
    return frames


NORMAL = _chunks(
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


async def test_stream_generate_yields_chunks_then_final():
    engine = _engine(NORMAL)
    outputs = [o async for o in engine.stream_generate("hi", max_tokens=8)]
    assert "".join(o.new_text for o in outputs[:-1]) == "Hello"
    final = outputs[-1]
    assert final.finished is True
    assert final.finish_reason == "length"
    assert final.text == "Hello"
    assert final.completion_tokens == 2


async def test_generate_returns_final_only():
    engine = _engine(NORMAL)
    output = await engine.generate("hi", max_tokens=8)
    assert output.text == "Hello"
    assert output.finish_reason == "length"
    assert output.completion_tokens == 2


async def test_stream_chat_applies_template_then_streams():
    engine = _engine(NORMAL)
    outputs = [
        o
        async for o in engine.stream_chat(
            [{"role": "user", "content": "hi"}], max_tokens=8
        )
    ]
    assert outputs[-1].finished is True
    assert outputs[-1].text == "Hello"


async def test_client_disconnect_forwards_an_abort():
    cluster = MockCluster(NORMAL)
    engine = ClusterEngine("m", cluster=cluster, resolved_path="/x")
    engine._loaded = True
    engine._tokenizer = StubTokenizer()

    stream = engine.stream_generate("hi", max_tokens=8, request_id="r")
    await stream.__anext__()
    await stream.aclose()  # client disconnect
    assert "r" in cluster.aborted


async def test_stop_finalization_trims_trailing_stop():
    frames = _chunks(
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
    engine = _engine(frames)
    output = await engine.generate("hi", max_tokens=8, stop=["#"])
    assert output.finish_reason == "stop"
    assert output.text == "abc"


async def test_rank_error_frame_raises():
    frames = [{"ok": False, "request_id": "r", "error": "rank died"}]
    engine = _engine(frames)
    with pytest.raises(RuntimeError, match="rank died"):
        [o async for o in engine.stream_generate("hi")]


def test_stats_and_type():
    engine = _engine(NORMAL)
    engine._model_type_value = "llama"
    stats = engine.get_stats()
    assert stats["engine_type"] == "cluster-distributed"
    assert stats["world_size"] == 2
    assert engine.model_type == "llama"
    assert engine.get_cache_stats() is None
