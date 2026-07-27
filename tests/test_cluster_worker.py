# SPDX-License-Identifier: Apache-2.0
"""The rank worker's control plane: channels, dispatch, and the pipe formats.

The serving loop itself lives in `omlx/cluster/batching.py` and is tested in
`test_cluster_batching.py`; what is pinned here is everything around it - that
commands are never stranded in a buffer, that signals name their request, and
that a worker losing its daemon shuts down rather than lingering.
"""

from __future__ import annotations

import json
import os

import pytest

from omlx.cluster.protocol import GenerationSpec, StopTextBuffer
from omlx.cluster.worker import CommandReader, ControlChannel, Worker


# =============================================================================
# Stop sequences
# =============================================================================


class TestStopTextBuffer:
    def test_passes_everything_through_without_stops(self):
        buffer = StopTextBuffer([])
        assert buffer.push("hello ") == "hello "
        assert buffer.push("world") == "world"
        assert buffer.hit is None
        assert buffer.text == "hello world"

    def test_holds_back_a_possible_partial_match(self):
        buffer = StopTextBuffer(["END"])
        # "EN" could still become "END", so it is not safe to emit yet.
        assert buffer.push("done EN") == "done "
        assert buffer.hit is None

    def test_releases_the_held_back_tail_when_it_was_not_a_stop(self):
        buffer = StopTextBuffer(["END"])
        buffer.push("done EN")
        assert buffer.push("ough") == "ENough"
        assert buffer.text == "done ENough"

    def test_truncates_at_a_stop_that_straddles_two_tokens(self):
        buffer = StopTextBuffer(["<|im_end|>"])
        assert buffer.push("answer<|im_") == "answer"
        assert buffer.push("end|> trailing") == ""
        assert buffer.hit == "<|im_end|>"
        assert buffer.text == "answer"

    def test_flush_emits_nothing_after_a_hit(self):
        buffer = StopTextBuffer(["STOP"])
        buffer.push("a STOP b")
        assert buffer.flush() == ""
        assert buffer.text == "a "

    def test_flush_releases_a_dangling_partial(self):
        buffer = StopTextBuffer(["STOP"])
        buffer.push("almost ST")
        assert buffer.flush() == "ST"
        assert buffer.text == "almost ST"

    def test_earliest_stop_wins(self):
        buffer = StopTextBuffer(["B", "AB"])
        assert buffer.push("xxABxx") == "xxA"
        assert buffer.hit == "B"


# =============================================================================
# The out-of-band signal channel
# =============================================================================


class TestControlChannel:
    def test_no_fd_yields_nothing(self):
        assert ControlChannel(None).take_events() == []

    def test_a_quiet_pipe_yields_nothing(self):
        read_fd, write_fd = os.pipe()
        try:
            assert ControlChannel(read_fd).take_events() == []
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_an_abort_names_its_request(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = ControlChannel(read_fd)
            os.write(write_fd, b'{"op": "abort", "request_id": "r1"}\n')
            events = channel.take_events()
            assert events == [{"op": "abort", "request_id": "r1"}]
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_events_are_consumed_not_latched(self):
        """A late abort for a finished request must not end the next one -
        scoping by request id replaced the old drain-between-runs dance."""
        read_fd, write_fd = os.pipe()
        try:
            channel = ControlChannel(read_fd)
            os.write(write_fd, b'{"op": "abort", "request_id": "r1"}\n')
            assert len(channel.take_events()) == 1
            assert channel.take_events() == []
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_daemon_that_went_away_aborts_everything(self):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            events = ControlChannel(read_fd).take_events()
            assert events == [{"op": "abort"}]
        finally:
            os.close(read_fd)

    def test_a_partial_line_is_not_an_event_yet(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = ControlChannel(read_fd)
            os.write(write_fd, b'{"op": "abo')
            assert channel.take_events() == []
            os.write(write_fd, b'rt"}\n')
            assert channel.take_events() == [{"op": "abort"}]
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_unparseable_signals_are_ignored(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = ControlChannel(read_fd)
            os.write(write_fd, b"not json at all\n")
            assert channel.take_events() == []
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_several_signals_arrive_in_order(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = ControlChannel(read_fd)
            os.write(
                write_fd,
                b'{"op": "abort", "request_id": "a"}\n'
                b'{"op": "abort", "request_id": "b"}\n',
            )
            ids = [e.get("request_id") for e in channel.take_events()]
            assert ids == ["a", "b"]
        finally:
            os.close(read_fd)
            os.close(write_fd)


# =============================================================================
# The command channel
# =============================================================================


class TestCommandReader:
    def test_readline_returns_one_line(self):
        read_fd, write_fd = os.pipe()
        try:
            reader = CommandReader(read_fd)
            os.write(write_fd, b'{"op": "ping"}\n')
            assert reader.readline() == '{"op": "ping"}'
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_eof_is_an_empty_string(self):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            assert CommandReader(read_fd).readline() == ""
        finally:
            os.close(read_fd)

    def test_drain_does_not_block_on_a_quiet_pipe(self):
        read_fd, write_fd = os.pipe()
        try:
            assert CommandReader(read_fd).drain_lines() == []
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_burst_of_commands_is_not_stranded(self):
        """Two commands in one read must both come out - the failure mode a
        buffered readline + select mix produces is the second one sitting in
        a Python buffer while select reports an empty pipe."""
        read_fd, write_fd = os.pipe()
        try:
            reader = CommandReader(read_fd)
            os.write(write_fd, b"one\ntwo\n")
            assert reader.readline() == "one"
            assert reader.drain_lines() == ["two"]
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_drain_keeps_a_partial_line_for_later(self):
        read_fd, write_fd = os.pipe()
        try:
            reader = CommandReader(read_fd)
            os.write(write_fd, b"whole\npart")
            assert reader.drain_lines() == ["whole"]
            os.write(write_fd, b"ial\n")
            assert reader.drain_lines() == ["partial"]
        finally:
            os.close(read_fd)
            os.close(write_fd)


# =============================================================================
# Dispatch
# =============================================================================


class FakeWorld:
    def __init__(self, rank: int, size: int = 2) -> None:
        self.rank = rank
        self.size = size

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


class FakeSession:
    """A leader-only stand-in: broadcast hands back what rank 0 read."""

    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def broadcast(self, obj):
        return obj

    def agree_int(self, value: int) -> int:
        return value


class ScriptedControl:
    """A command channel that replays a script, then reports EOF."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    def readline(self) -> str:
        return self._lines.pop(0) if self._lines else ""

    def drain_lines(self) -> list[str]:
        return []


class RecordingBatch:
    def __init__(self, shutdown: bool = False) -> None:
        self.served: list[dict] = []
        self._shutdown = shutdown

    def serve(self, event) -> bool:
        self.served.append(event)
        return self._shutdown


def build_worker(*, rank: int = 0) -> tuple[Worker, list[dict]]:
    """A Worker with the collective replaced; replies land in the list."""
    worker = Worker.__new__(Worker)
    world = FakeWorld(rank)
    worker.world = world
    worker.session = FakeSession(world)
    worker.config = type("Cfg", (), {"model_path": "fake", "seed": 0})()
    worker.model = object()
    worker.tokenizer = None
    worker.batch = None
    worker.signals = ControlChannel(None)
    worker._commands = None

    replies: list[dict] = []
    worker._reply = lambda payload, flush=True: replies.append(payload)
    return worker, replies


class TestWorkerDispatch:
    def test_generate_before_load_is_refused(self):
        worker, replies = build_worker()
        worker.run(
            ScriptedControl(
                [json.dumps({"op": "generate", "request_id": "r1", "prompt_ids": [1]})]
            )
        )
        assert replies == [
            {
                "ok": False,
                "request_id": "r1",
                "error": "generate called before load",
            }
        ]

    def test_a_generate_enters_the_batch_loop(self):
        worker, _ = build_worker()
        worker.batch = RecordingBatch()
        command = {"op": "generate", "request_id": "r1", "prompt_ids": [1]}
        worker.run(ScriptedControl([json.dumps(command)]))
        assert worker.batch.served == [command]

    def test_shutdown_from_inside_the_batch_ends_the_run(self):
        worker, _ = build_worker()
        worker.batch = RecordingBatch(shutdown=True)
        worker.run(
            ScriptedControl(
                [
                    json.dumps({"op": "generate", "prompt_ids": [1]}),
                    json.dumps({"op": "ping"}),  # must never be reached
                ]
            )
        )
        assert len(worker.batch.served) == 1

    def test_eof_means_shutdown(self):
        worker, replies = build_worker()
        worker.run(ScriptedControl([]))
        assert replies == []

    def test_ping_answers_with_the_rank(self):
        worker, replies = build_worker()
        worker.run(ScriptedControl([json.dumps({"op": "ping"})]))
        assert replies == [{"ok": True, "rank": 0}]

    def test_an_unknown_op_is_reported_not_fatal(self):
        worker, replies = build_worker()
        worker.run(ScriptedControl([json.dumps({"op": "dance"})]))
        assert replies[0]["ok"] is False
        assert "dance" in replies[0]["error"]


# =============================================================================
# The wire format
# =============================================================================


class TestGenerationSpec:
    def test_round_trips(self):
        spec = GenerationSpec(
            prompt_ids=[1, 2, 3],
            max_tokens=7,
            temperature=0.5,
            stop=["END"],
            stop_token_ids=[9],
            seed=42,
            request_id="r-1",
        )
        assert GenerationSpec.from_dict(spec.to_dict()) == spec

    def test_ignores_fields_it_does_not_know(self):
        spec = GenerationSpec.from_dict(
            {"op": "generate", "prompt_ids": [1], "future_field": True}
        )
        assert spec.prompt_ids == [1]
