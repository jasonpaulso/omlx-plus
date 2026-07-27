# SPDX-License-Identifier: Apache-2.0
"""The lockstep batching loop.

The property that matters is not the text produced but that every rank makes
identical batching decisions from the event stream alone. The fakes are built
for that: a `FakeGenerator` whose output is a pure function of what was
inserted, and a linked pair of sessions that lets a real leader and a real
follower run the loop against each other with nothing shared but the
collective's semantics.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

from omlx.cluster.batching import BatchConfig, BatchLoop
from omlx.cluster.protocol import CMD_GENERATE, CMD_SHUTDOWN, SIGNAL_ABORT

# Token id whose script entry means "the state machine matched a stop token".
EOS = 99


# =============================================================================
# Fakes
# =============================================================================


class FakeWorld:
    def __init__(self, rank: int, size: int) -> None:
        self.rank = rank
        self.size = size

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


class SoloSession:
    """A world of one: every agreement is with yourself."""

    def __init__(self) -> None:
        self.world = FakeWorld(0, 1)

    def agree_int(self, value: int) -> int:
        return value

    def broadcast(self, obj):
        return obj


class LinkedSession:
    """One side of a two-rank collective built from queues.

    Ordering is what makes this honest: each rank issues the same sequence of
    collective calls, so pairing them FIFO reproduces exactly what `all_sum`
    over a real transport would agree on.
    """

    def __init__(self, world: FakeWorld, channel: queue.Queue) -> None:
        self.world = world
        self._channel = channel

    def agree_int(self, value: int) -> int:
        if self.world.is_leader:
            self._channel.put(("int", value))
            return value
        kind, agreed = self._channel.get(timeout=5)
        assert kind == "int"
        return agreed

    def broadcast(self, obj):
        if self.world.is_leader:
            self._channel.put(("obj", obj))
            return obj
        kind, agreed = self._channel.get(timeout=5)
        assert kind == "obj"
        return agreed


class FakeGenerator:
    """Continuous batching whose output is a pure function of its inputs.

    Scripts are keyed by the prompt: each step yields the next scripted token
    for every active sequence, in insertion order, exactly as the real
    generator steps its batch. `EOS` in a script plays the part of the token
    state machine matching a stop token; exhausting `max_tokens` produces
    `length`, both mirroring mlx-lm's `BatchGenerator` semantics.
    """

    def __init__(self, model, config, scripts: dict[tuple, list[int]]) -> None:
        self.scripts = scripts
        self.uid_count = 0
        self.active: dict[int, dict] = {}
        self.inserted: list[tuple] = []
        self.removed: list[int] = []
        self.steps = 0

    def insert(self, prompts, max_tokens=None, caches=None, all_tokens=None,
               samplers=None, logits_processors=None, state_machines=None):
        uids = []
        # The loop hands the generator `[[last_token]]` plus the full prompt
        # as `all_tokens` (prefill happened outside); scripts key on the
        # full prompt.
        keys = all_tokens if all_tokens else prompts
        for key_source, limit in zip(keys, max_tokens):
            key = tuple(key_source)
            self.active[self.uid_count] = {
                "tokens": list(self.scripts[key]),
                "position": 0,
                "max_tokens": limit,
            }
            self.inserted.append((self.uid_count, key, limit))
            uids.append(self.uid_count)
            self.uid_count += 1
        return uids

    def remove(self, uids, return_prompt_caches=False):
        for uid in uids:
            self.active.pop(uid, None)
            self.removed.append(uid)
        return {}

    def next(self):
        self.steps += 1
        responses = []
        finished = []
        for uid in sorted(self.active):
            seq = self.active[uid]
            token = seq["tokens"][seq["position"]]
            seq["position"] += 1
            if token == EOS:
                finish = "stop"
            elif (
                seq["position"] >= seq["max_tokens"]
                or seq["position"] >= len(seq["tokens"])
            ):
                finish = "length"
            else:
                finish = None
            responses.append(
                SimpleNamespace(uid=uid, token=token, finish_reason=finish)
            )
            if finish is not None:
                finished.append(uid)
        for uid in finished:
            self.active.pop(uid, None)
        return [], responses


class FakeDetokenizer:
    """One token, one word; `last_segment` consumes, like the real one."""

    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = pieces
        self._pending = ""

    def add_token(self, token_id: int) -> None:
        self._pending += self._pieces.get(token_id, f"<{token_id}>")

    def finalize(self) -> None:
        pass

    @property
    def last_segment(self) -> str:
        segment, self._pending = self._pending, ""
        return segment


class FakeTokenizer:
    """Hands out a *fresh* detokenizer per access, as the real wrapper does."""

    def __init__(self, pieces: dict[int, str], eos: int = EOS) -> None:
        self._pieces = pieces
        self.eos_token_id = eos

    @property
    def detokenizer(self) -> FakeDetokenizer:
        return FakeDetokenizer(self._pieces)


def submit(request_id: str, prompt: list[int], **kwargs) -> dict:
    return {
        "op": CMD_GENERATE,
        "request_id": request_id,
        "prompt_ids": prompt,
        "temperature": 0.0,
        **kwargs,
    }


def build_loop(
    scripts: dict[tuple, list[int]],
    pieces: dict[int, str],
    *,
    session=None,
    events: list[list[dict]] | None = None,
):
    """A leader-side BatchLoop wired to fakes; returns (loop, replies, gen)."""
    replies: list[dict] = []
    feed = list(events or [])
    generators: list[FakeGenerator] = []

    def factory(model, config):
        generator = FakeGenerator(model, config, scripts)
        generators.append(generator)
        return generator

    loop = BatchLoop(
        session or SoloSession(),
        model=object(),
        tokenizer=FakeTokenizer(pieces),
        config=BatchConfig(),
        reply=replies.append,
        gather_events=lambda: feed.pop(0) if feed else [],
        generator_factory=factory,
        prefill=lambda ids: None,
    )
    return loop, replies, generators


def done_replies(replies):
    return [r for r in replies if r.get("done")]


def chunks_for(replies, request_id):
    return "".join(
        r["chunk"]
        for r in replies
        if r.get("chunk") and r["request_id"] == request_id
    )


# =============================================================================
# One request, start to finish
# =============================================================================


class TestSingleRequest:
    PIECES = {1: "Red", 2: ", Blue", 3: ", Yellow"}

    def test_streams_chunks_and_reports_usage(self):
        loop, replies, _ = build_loop(
            {(7, 8): [1, 2, 3]}, self.PIECES
        )
        shutdown = loop.serve(submit("r1", [7, 8], max_tokens=16))

        assert shutdown is False
        assert chunks_for(replies, "r1") == "Red, Blue, Yellow"
        [done] = done_replies(replies)
        assert done["request_id"] == "r1"
        assert done["finish_reason"] == "length"
        assert done["prompt_tokens"] == 2
        assert done["completion_tokens"] == 3
        assert done["text"] == "Red, Blue, Yellow"

    def test_a_stop_token_ends_the_run_and_is_not_output(self):
        loop, replies, _ = build_loop({(7,): [1, EOS]}, self.PIECES)
        loop.serve(submit("r1", [7], max_tokens=16))

        [done] = done_replies(replies)
        assert done["finish_reason"] == "stop"
        assert done["text"] == "Red"
        assert done["completion_tokens"] == 1

    def test_a_stop_string_truncates_and_counts_its_token(self):
        pieces = {1: "value", 2: " END", 3: " ignored"}
        loop, replies, generators = build_loop({(7,): [1, 2, 3]}, pieces)
        loop.serve(submit("r1", [7], max_tokens=16, stop=["END"]))

        [done] = done_replies(replies)
        assert done["finish_reason"] == "stop"
        assert done["text"] == "value "
        assert done["completion_tokens"] == 2
        # The sequence was still live in the generator; the loop must evict it.
        assert generators[0].removed == [0]

    def test_max_tokens_is_honored(self):
        loop, replies, _ = build_loop({(7,): [1, 2, 3]}, self.PIECES)
        loop.serve(submit("r1", [7], max_tokens=2))

        [done] = done_replies(replies)
        assert done["finish_reason"] == "length"
        assert done["completion_tokens"] == 2

    def test_an_empty_prompt_is_refused(self):
        loop, replies, generators = build_loop({}, {})
        loop.serve(submit("r1", []))

        assert replies[0]["ok"] is False
        assert "prompt token" in replies[0]["error"]
        assert generators == []  # nothing was ever admitted


# =============================================================================
# Requests joining and leaving a running batch
# =============================================================================


class TestContinuousBatching:
    PIECES = {1: "a", 2: "b", 3: "c", 4: "x", 5: "y"}

    def test_a_second_request_joins_mid_flight(self):
        scripts = {(7,): [1, 2, 3], (8,): [4, 5]}
        # The second request arrives while the first is on its second step.
        events = [[], [submit("r2", [8], max_tokens=16)]]
        loop, replies, generators = build_loop(
            scripts, self.PIECES, events=events
        )
        loop.serve(submit("r1", [7], max_tokens=16))

        assert chunks_for(replies, "r1") == "abc"
        assert chunks_for(replies, "r2") == "xy"
        assert {d["request_id"] for d in done_replies(replies)} == {"r1", "r2"}
        # Both really were in the batch at once.
        assert generators[0].inserted[0][0] != generators[0].inserted[1][0]

    def test_an_abort_evicts_one_and_leaves_the_rest(self):
        scripts = {(7,): [1, 2, 3], (8,): [4, 5]}
        events = [
            [submit("r2", [8], max_tokens=16)],
            [{"op": SIGNAL_ABORT, "request_id": "r1"}],
        ]
        loop, replies, generators = build_loop(
            scripts, self.PIECES, events=events
        )
        loop.serve(submit("r1", [7], max_tokens=16))

        dones = {d["request_id"]: d for d in done_replies(replies)}
        assert dones["r1"]["finish_reason"] == "abort"
        assert dones["r2"]["finish_reason"] == "length"
        assert chunks_for(replies, "r2") == "xy"

    def test_an_abort_for_an_unknown_request_is_a_noop(self):
        events = [[{"op": SIGNAL_ABORT, "request_id": "ghost"}]]
        loop, replies, _ = build_loop({(7,): [1, 2]}, self.PIECES, events=events)
        loop.serve(submit("r1", [7], max_tokens=16))

        [done] = done_replies(replies)
        assert done["request_id"] == "r1"

    def test_an_abort_with_no_id_clears_the_whole_batch(self):
        scripts = {(7,): [1, 2, 3], (8,): [4, 5]}
        events = [
            [submit("r2", [8], max_tokens=16)],
            [{"op": SIGNAL_ABORT}],
        ]
        loop, replies, _ = build_loop(scripts, self.PIECES, events=events)
        loop.serve(submit("r1", [7], max_tokens=16))

        reasons = {d["request_id"]: d["finish_reason"] for d in done_replies(replies)}
        assert reasons == {"r1": "abort", "r2": "abort"}

    def test_shutdown_mid_batch_aborts_and_reports(self):
        events = [[{"op": CMD_SHUTDOWN}]]
        loop, replies, _ = build_loop({(7,): [1, 2, 3]}, self.PIECES, events=events)
        shutdown = loop.serve(submit("r1", [7], max_tokens=16))

        assert shutdown is True
        [done] = done_replies(replies)
        assert done["finish_reason"] == "abort"

    def test_a_ping_mid_batch_is_answered_and_serving_continues(self):
        events = [[{"op": "ping"}]]
        loop, replies, _ = build_loop({(7,): [1, 2]}, self.PIECES, events=events)
        loop.serve(submit("r1", [7], max_tokens=16))

        assert {"ok": True, "rank": 0} in replies
        assert done_replies(replies)


# =============================================================================
# Lockstep: a real leader and follower, linked only by the collective
# =============================================================================


class TestLockstep:
    def test_both_ranks_make_identical_batching_decisions(self):
        """The whole correctness argument, exercised end to end: the follower
        sees nothing but broadcast events, and must insert, step and evict
        exactly as the leader does."""
        scripts = {(7,): [1, 2, 3, 4], (8,): [5, 6]}
        pieces = {n: f"t{n}" for n in range(1, 7)}
        channel: queue.Queue = queue.Queue()

        leader_session = LinkedSession(FakeWorld(0, 2), channel)
        follower_session = LinkedSession(FakeWorld(1, 2), channel)

        first = submit("r1", [7], max_tokens=16)
        leader_events = [
            [submit("r2", [8], max_tokens=16)],
            [{"op": SIGNAL_ABORT, "request_id": "r1"}],
        ]
        leader, leader_replies, leader_gens = build_loop(
            scripts, pieces, session=leader_session, events=leader_events
        )

        follower_replies: list[dict] = []
        follower_gens: list[FakeGenerator] = []

        def follower_factory(model, config):
            generator = FakeGenerator(model, config, scripts)
            follower_gens.append(generator)
            return generator

        follower = BatchLoop(
            follower_session,
            model=object(),
            tokenizer=FakeTokenizer(pieces),
            config=BatchConfig(),
            reply=follower_replies.append,
            gather_events=lambda: pytest.fail(
                "a follower must never gather events locally"
            ),
            generator_factory=follower_factory,
            prefill=lambda ids: None,
        )

        follower_thread = threading.Thread(
            target=follower.serve, args=(dict(first),), daemon=True
        )
        follower_thread.start()
        leader.serve(first)
        follower_thread.join(timeout=5)
        assert not follower_thread.is_alive()

        # Identical decisions: same admissions in the same order, same
        # evictions, same number of steps.
        assert follower_gens[0].inserted == leader_gens[0].inserted
        assert follower_gens[0].removed == leader_gens[0].removed
        assert follower_gens[0].steps == leader_gens[0].steps
        # Stronger still: the follower computed the *identical* reply stream.
        # (In production `Worker._reply` discards it - rank-gating is the
        # worker's job, not the loop's - but computing it identically is the
        # lockstep property itself.)
        assert follower_replies == leader_replies


# =============================================================================
# Configuration
# =============================================================================


class TestBatchConfig:
    def test_defaults(self):
        config = BatchConfig.from_command({})
        assert config.completion_batch_size == 8

    def test_prefill_is_always_serial(self):
        """Padded multi-prompt prefill deadlocks a sharded model over ring;
        no configuration may reintroduce it."""
        assert BatchConfig.from_command({}).prefill_batch_size == 1
        assert BatchConfig.from_command({"max_batch_size": 32}).prefill_batch_size == 1

    def test_the_leaders_size_wins(self):
        config = BatchConfig.from_command({"max_batch_size": 3})
        assert config.completion_batch_size == 3

    def test_a_nonsense_size_becomes_the_floor(self):
        assert BatchConfig.from_command({"max_batch_size": 0}).completion_batch_size == 8
        assert BatchConfig.from_command({"max_batch_size": -2}).completion_batch_size == 1
