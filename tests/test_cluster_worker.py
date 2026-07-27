# SPDX-License-Identifier: Apache-2.0
"""The rank worker's decode loop, and the two channels that drive it.

The loop is where the lockstep rule lives, so the tests care less about the
text produced than about *who decided* to stop and whether every rank would
have left the loop on the same step. `FakeSession` records every `agree_int`
for exactly that reason.
"""

from __future__ import annotations

import os

import pytest

from omlx.cluster.protocol import (
    STEP_ABORT,
    STEP_CONTINUE,
    STEP_EOS,
    STEP_STOP_TEXT,
    GenerationSpec,
    StopTextBuffer,
)
from omlx.cluster.worker import AbortChannel, Worker


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
# The out-of-band abort channel
# =============================================================================


class TestAbortChannel:
    def test_no_fd_never_aborts(self):
        assert AbortChannel(None).poll() is False

    def test_quiet_pipe_does_not_abort(self):
        read_fd, write_fd = os.pipe()
        try:
            assert AbortChannel(read_fd).poll() is False
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_abort_signal_is_seen(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = AbortChannel(read_fd)
            os.write(write_fd, b'{"op": "abort"}\n')
            assert channel.poll() is True
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_abort_stays_true_once_seen(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = AbortChannel(read_fd)
            os.write(write_fd, b'{"op": "abort"}\n')
            assert channel.poll() is True
            # Nothing new arrives, but the run is still aborted.
            assert channel.poll() is True
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_daemon_that_went_away_counts_as_abort(self):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        try:
            assert AbortChannel(read_fd).poll() is True
        finally:
            os.close(read_fd)

    def test_partial_line_is_not_an_abort_yet(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = AbortChannel(read_fd)
            os.write(write_fd, b'{"op": "abo')
            assert channel.poll() is False
            os.write(write_fd, b'rt"}\n')
            assert channel.poll() is True
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_unparseable_signal_is_ignored(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = AbortChannel(read_fd)
            os.write(write_fd, b"not json at all\n")
            assert channel.poll() is False
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_drain_discards_a_stale_abort(self):
        read_fd, write_fd = os.pipe()
        try:
            channel = AbortChannel(read_fd)
            os.write(write_fd, b'{"op": "abort"}\n')
            channel.drain()
            assert channel.poll() is False
        finally:
            os.close(read_fd)
            os.close(write_fd)


# =============================================================================
# The decode loop
# =============================================================================


class FakeWorld:
    def __init__(self, rank: int, size: int = 2) -> None:
        self.rank = rank
        self.size = size

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


class FakeSession:
    """Stands in for the collective, recording what was agreed.

    `agree_int` returns rank 0's value, which is what the real `all_sum`-based
    implementation does. Followers are simulated by feeding a scripted list of
    leader verdicts instead.
    """

    def __init__(self, world: FakeWorld, leader_verdicts: list[int] | None = None):
        self.world = world
        self.agreed: list[int] = []
        self._scripted = list(leader_verdicts or [])
        self.seeds: list[int] = []

    def agree_int(self, value: int) -> int:
        agreed = value if self.world.is_leader else (
            self._scripted.pop(0) if self._scripted else STEP_CONTINUE
        )
        self.agreed.append(agreed)
        return agreed

    def seed_everyone(self, seed: int) -> int:
        self.seeds.append(seed)
        return seed


class FakeDetokenizer:
    """One token, one word - enough to exercise segmentation and stops."""

    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = pieces
        self.last_segment = ""
        self.finalized = False

    def reset(self) -> None:
        self.last_segment = ""

    def add_token(self, token_id: int) -> None:
        self.last_segment = self._pieces.get(token_id, f"<{token_id}>")

    def finalize(self) -> None:
        self.finalized = True


class FakeTokenizer:
    def __init__(self, pieces: dict[int, str], eos: int = 99) -> None:
        self.detokenizer = FakeDetokenizer(pieces)
        self.eos_token_id = eos


def build_worker(
    monkeypatch,
    token_ids: list[int],
    pieces: dict[int, str],
    *,
    rank: int = 0,
    leader_verdicts: list[int] | None = None,
    abort_fd: int | None = None,
) -> Worker:
    """A Worker with the collective and the model replaced.

    `Worker.__init__` joins a real mlx collective, which a unit test must not
    do - a distributed session cannot be torn down inside a process. Building
    the object directly is the point of the seam.
    """
    import importlib

    import mlx.core as mx

    worker = Worker.__new__(Worker)
    world = FakeWorld(rank)
    worker.world = world
    worker.session = FakeSession(world, leader_verdicts)
    worker.config = type("Cfg", (), {"model_path": "fake", "seed": 0})()
    worker.model = object()
    worker.tokenizer = FakeTokenizer(pieces)
    worker.abort = AbortChannel(abort_fd)

    def fake_generate_step(prompt, model, *, max_tokens, sampler, logits_processors):
        # mlx-lm 0.31.3 yields a plain int; older versions yield an mx.array.
        # Alternate between them so the worker keeps handling both.
        for index, token_id in enumerate(token_ids[:max_tokens]):
            yield (token_id if index % 2 else mx.array(token_id)), None

    # Patched on the module object, not by dotted path: `mlx_lm.generate` is
    # also the name of a re-exported *function*, so the string form resolves
    # to that and not to the module the worker imports from.
    monkeypatch.setattr(
        importlib.import_module("mlx_lm.generate"),
        "generate_step",
        fake_generate_step,
    )
    return worker


class TestWorkerGenerate:
    def test_streams_text_and_reports_usage(self, monkeypatch):
        worker = build_worker(
            monkeypatch, [1, 2, 3], {1: "one ", 2: "two ", 3: "three"}
        )
        payloads = list(
            worker.generate(GenerationSpec(prompt_ids=[7, 8], max_tokens=8))
        )

        chunks = [p["chunk"] for p in payloads if "chunk" in p]
        final = payloads[-1]
        assert "".join(chunks) == "one two three"
        assert final["done"] is True
        assert final["text"] == "one two three"
        assert final["prompt_tokens"] == 2
        assert final["completion_tokens"] == 3
        assert final["finish_reason"] == "length"

    def test_eos_ends_the_run_and_is_not_output(self, monkeypatch):
        worker = build_worker(monkeypatch, [1, 99, 2], {1: "hi ", 2: "nope"})
        payloads = list(worker.generate(GenerationSpec(prompt_ids=[7])))

        final = payloads[-1]
        assert final["text"] == "hi "
        assert final["completion_tokens"] == 1
        assert final["finish_reason"] == "stop"
        assert worker.session.agreed == [STEP_CONTINUE, STEP_EOS]

    def test_stop_string_truncates_and_counts_its_token(self, monkeypatch):
        worker = build_worker(
            monkeypatch, [1, 2, 3], {1: "keep ", 2: "END now", 3: "unreached"}
        )
        payloads = list(
            worker.generate(GenerationSpec(prompt_ids=[7], stop=["END"]))
        )

        final = payloads[-1]
        assert final["text"] == "keep "
        assert final["finish_reason"] == "stop"
        assert final["completion_tokens"] == 2
        assert worker.session.agreed == [STEP_CONTINUE, STEP_STOP_TEXT]

    def test_abort_stops_mid_run(self, monkeypatch):
        read_fd, write_fd = os.pipe()
        try:
            worker = build_worker(
                monkeypatch,
                [1, 2, 3],
                {1: "a", 2: "b", 3: "c"},
                abort_fd=read_fd,
            )
            stream = worker.generate(GenerationSpec(prompt_ids=[7]))
            assert next(stream)["chunk"] == "a"
            os.write(write_fd, b'{"op": "abort"}\n')
            final = list(stream)[-1]

            assert final["finish_reason"] == "abort"
            assert final["completion_tokens"] == 1
            assert worker.session.agreed[-1] == STEP_ABORT
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_stale_abort_does_not_end_the_next_run(self, monkeypatch):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b'{"op": "abort"}\n')
            worker = build_worker(
                monkeypatch, [1, 2], {1: "a", 2: "b"}, abort_fd=read_fd
            )
            final = list(worker.generate(GenerationSpec(prompt_ids=[7])))[-1]
            assert final["completion_tokens"] == 2
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_a_follower_produces_no_output_but_steps_alike(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            [1, 2, 3],
            {1: "a", 2: "b", 3: "c"},
            rank=1,
            leader_verdicts=[STEP_CONTINUE, STEP_EOS],
        )
        payloads = list(worker.generate(GenerationSpec(prompt_ids=[7])))

        assert payloads == []
        # It left the loop on rank 0's verdict, not on anything it could see.
        assert worker.session.agreed == [STEP_CONTINUE, STEP_EOS]

    def test_a_follower_never_decides_to_stop_by_itself(self, monkeypatch):
        """A follower's own contribution is always "continue".

        This is the divergence rule in one assertion: whatever the follower's
        local detokenizer or abort channel might have said, it contributes
        `STEP_CONTINUE` and takes rank 0's answer.
        """
        read_fd, write_fd = os.pipe()
        os.close(write_fd)  # a closed pipe reads as abort - on rank 0 only
        try:
            worker = build_worker(
                monkeypatch,
                [99, 99],
                {},
                rank=1,
                leader_verdicts=[STEP_CONTINUE, STEP_CONTINUE],
                abort_fd=read_fd,
            )
            list(worker.generate(GenerationSpec(prompt_ids=[7], max_tokens=2)))
            assert worker.session.agreed == [STEP_CONTINUE, STEP_CONTINUE]
        finally:
            os.close(read_fd)

    def test_request_seed_is_agreed_before_sampling(self, monkeypatch):
        worker = build_worker(monkeypatch, [1], {1: "x"})
        list(worker.generate(GenerationSpec(prompt_ids=[7], seed=1234)))
        assert worker.session.seeds == [1234]

    def test_no_seed_leaves_the_launch_seed_alone(self, monkeypatch):
        worker = build_worker(monkeypatch, [1], {1: "x"})
        list(worker.generate(GenerationSpec(prompt_ids=[7])))
        assert worker.session.seeds == []

    def test_zero_is_a_real_seed(self, monkeypatch):
        """`seed=0` must not be read as "no seed"."""
        worker = build_worker(monkeypatch, [1], {1: "x"})
        list(worker.generate(GenerationSpec(prompt_ids=[7], seed=0)))
        assert worker.session.seeds == [0]

    def test_an_empty_prompt_is_refused(self, monkeypatch):
        worker = build_worker(monkeypatch, [1], {1: "x"})
        with pytest.raises(ValueError, match="prompt token"):
            list(worker.generate(GenerationSpec(prompt_ids=[])))

    def test_generate_before_load_is_refused(self, monkeypatch):
        worker = build_worker(monkeypatch, [1], {1: "x"})
        worker.model = None
        with pytest.raises(RuntimeError, match="before load"):
            list(worker.generate(GenerationSpec(prompt_ids=[7])))


# =============================================================================
# The wire format
# =============================================================================


class TestGenerationSpec:
    def test_round_trips(self):
        spec = GenerationSpec(
            prompt_ids=[1, 2],
            max_tokens=16,
            temperature=0.5,
            stop=["END"],
            seed=7,
            request_id="abc",
        )
        assert GenerationSpec.from_dict(spec.to_dict()) == spec

    def test_ignores_fields_it_does_not_know(self):
        spec = GenerationSpec.from_dict(
            {"prompt_ids": [1], "op": "generate", "invented_later": True}
        )
        assert spec.prompt_ids == [1]
