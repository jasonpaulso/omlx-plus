# SPDX-License-Identifier: Apache-2.0
"""Continuous batching, in lockstep across every rank.

Under tensor parallelism every rank must run the same forward pass, so
batching two requests means every rank agreeing, at every step, on which
sequences advance. The scheduler divergence audit
(`docs/cluster-scheduler-divergence-audit.md`) worked out what that costs for
oMLX's own scheduler: six memory-admission gates and a prefix-cache lookup,
each a place where a rank could branch on state only it can see.

This module takes the other exit. mlx-lm's `BatchGenerator` is a continuous
batcher whose every decision is a deterministic function of its inputs:
admission order comes from a deque, batch caps come from configuration, stop
detection runs a token-id state machine, and sampling draws from an RNG the
ranks keep synchronised on logits that tensor parallelism has already
all-reduced. Give every rank an identical `BatchGenerator` and an identical
event stream, and they stay in lockstep with **no per-token collective at
all** - the serial loop's per-step verdict agreement disappears.

What is left to synchronise is exactly the state only rank 0 can see: request
arrivals and aborts. Each iteration begins with one small collective agreeing
how many such events exist, and a broadcast of them only when there are any.
Everything downstream - who joins the batch, who leaves, which token is drawn -
follows identically on every rank.

The two divergence classes the audit flagged are absent by construction rather
than fixed: no rank ever consults local memory (admission is capped by the
agreed `max_batch_size`, not by a machine-local reading), and there is no
prefix cache in cluster mode (every request starts from a fresh cache).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from omlx.cluster.protocol import (
    CMD_GENERATE,
    CMD_PING,
    CMD_SHUTDOWN,
    SIGNAL_ABORT,
    GenerationSpec,
    StopTextBuffer,
)

logger = logging.getLogger(__name__)


def make_generator(model: Any, config: "BatchConfig") -> Any:
    """A real mlx-lm `BatchGenerator`. Tests inject a fake instead.

    """
    from mlx_lm.generate import BatchGenerator

    return BatchGenerator(
        model,
        completion_batch_size=config.completion_batch_size,
        prefill_batch_size=config.prefill_batch_size,
        prefill_step_size=config.prefill_step_size,
    )


@dataclass
class BatchConfig:
    """Batching knobs, agreed by every rank.

    These arrive in the leader's `load` command rather than being read from
    each node's own settings - two nodes configured differently would build
    generators that admit differently, which is exactly the divergence this
    module exists to prevent.
    """

    completion_batch_size: int = 8
    # The generator must NEVER process a prompt. Every configuration that
    # routes prompt processing through mlx-lm's PromptProcessingBatch on a
    # tensor-sharded model deadlocks the ring - measured 0/5 across padded
    # batches, serial prefill, decode-overlap, and mlx-lm's own server stream
    # combo (mlx 0.32.0 / mlx-lm 0.31.3, pure mlx-lm, no oMLX code). Decode
    # batching alone measured 5/5. So prompts are prefilled by hand in
    # `BatchLoop._admit` and sequences enter the generator with their cache
    # already built; this setting is kept at 1 purely as a backstop.
    prefill_batch_size: int = 1
    prefill_step_size: int = 2048

    @classmethod
    def from_command(cls, command: dict[str, Any]) -> "BatchConfig":
        size = max(1, int(command.get("max_batch_size") or 8))
        return cls(completion_batch_size=size)


@dataclass
class _Sequence:
    """One request's rank-local bookkeeping while it is in the batch."""

    request_id: str
    detokenizer: Any
    buffer: StopTextBuffer
    prompt_tokens: int
    generated: int = 0


class BatchLoop:
    """The lockstep serving loop every rank runs while requests are active.

    Entered when the idle control loop broadcasts the first `generate`, left
    when the last sequence finishes. Followers run it with a `reply` that
    discards and a `gather_events` that is never called - their events arrive
    through the collective.
    """

    def __init__(
        self,
        session: Any,
        model: Any,
        tokenizer: Any,
        config: BatchConfig,
        *,
        reply: Callable[[dict[str, Any]], None],
        gather_events: Callable[[], list[dict[str, Any]]],
        generator_factory: Callable[[Any, BatchConfig], Any] = make_generator,
        prefill: Callable[[list[int]], Any] | None = None,
    ) -> None:
        self._session = session
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._reply = reply
        self._gather = gather_events
        self._factory = generator_factory
        self._prefill = prefill or self._prefill_by_hand
        self._generator: Any = None
        self._sequences: dict[int, _Sequence] = {}

    # -- the loop ----------------------------------------------------------

    def serve(self, first_event: dict[str, Any]) -> bool:
        """Serve until the batch drains. True means shutdown was requested.

        `first_event` is the command that woke the idle loop; every rank
        already holds it (it arrived by broadcast), so it is applied without
        being synchronised again.
        """
        pending: list[dict[str, Any]] = [first_event]
        while True:
            events = self._sync(pending)
            pending = []
            if self._apply(events):
                self._abort("")  # daemon is leaving; nobody is listening
                return True
            if not self._sequences:
                return False
            self._step()

    def _sync(self, pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Agree this step's events. One collective; two when events exist.

        The drain first is a correctness requirement, not a flush for
        tidiness. The ring backend requires every rank to hand it collectives
        in one global order, and the generator pipelines compute with
        `async_eval` - so between steps the generation stream may still be
        scheduling model collectives while this control collective is issued
        from another thread, and the interleaving resolves differently on
        different ranks. Draining leaves a single in-flight source: model,
        then control, in program order on every rank. (Putting the control
        collective on the model's stream instead is not possible: ring
        `AllReduce` has no GPU implementation and throws when forced onto a
        GPU stream.)
        """
        self._drain()
        extra: list[dict[str, Any]] = []
        if self._session.world.is_leader:
            extra = list(self._gather())
        count = self._session.agree_int(len(extra))
        if count:
            extra = self._session.broadcast(
                extra if self._session.world.is_leader else None
            )
        return list(pending) + list(extra)

    def _drain(self) -> None:
        """Wait out every in-flight op on the generator's stream."""
        stream = getattr(self._generator, "stream", None)
        if stream is None:
            return
        import mlx.core as mx

        mx.synchronize(stream)

    def _apply(self, events: list[dict[str, Any]]) -> bool:
        for event in events:
            op = event.get("op")
            if op == CMD_SHUTDOWN:
                return True
            if op == CMD_GENERATE:
                self._admit(event)
            elif op == SIGNAL_ABORT:
                self._abort(str(event.get("request_id") or ""))
            elif op == CMD_PING:
                self._reply({"ok": True, "rank": self._session.world.rank})
            else:
                self._reply({"ok": False, "error": f"unknown op {op!r}"})
        return False

    # -- admission and eviction --------------------------------------------

    def _admit(self, event: dict[str, Any]) -> None:
        import mlx.core as mx
        from mlx_lm.generate import SequenceStateMachine
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        spec = GenerationSpec.from_dict(event)
        if not spec.prompt_ids:
            self._reply(
                {
                    "ok": False,
                    "request_id": spec.request_id,
                    "error": "generate needs at least one prompt token",
                }
            )
            return

        # A request-pinned seed re-seeds the shared stream. Every rank applies
        # it at the same point in the same order, so they stay identical - and
        # unlike the serial loop no collective is needed to agree on it,
        # because the spec itself was broadcast.
        if spec.seed is not None:
            mx.random.seed(int(spec.seed))

        sampler = make_sampler(
            temp=spec.temperature,
            top_p=spec.top_p,
            min_p=spec.min_p,
            top_k=spec.top_k,
        )
        processors = make_logits_processors(
            repetition_penalty=spec.repetition_penalty,
            repetition_context_size=spec.repetition_context_size,
            presence_penalty=spec.presence_penalty,
            frequency_penalty=spec.frequency_penalty,
        )
        stop_ids = sorted(set(spec.stop_token_ids) | self._eos_ids())
        machine = SequenceStateMachine(
            {"normal": [([token], None) for token in stop_ids]}
        )

        # Prefill by hand, then hand the generator a sequence that is already
        # one token from decoding. The generator's own prompt processing
        # deadlocks a sharded world in every shape it offers (see
        # `BatchConfig`); a plain forward per chunk, fully evaluated while
        # nothing else is in flight (the caller drained before applying
        # events), is the shape measured stable - and it is identical on
        # every rank, so lockstep holds.
        generator = self._ensure_generator()
        cache = self._prefill(list(spec.prompt_ids))
        [uid] = generator.insert(
            [[spec.prompt_ids[-1]]],
            max_tokens=[spec.max_tokens],
            caches=[cache],
            all_tokens=[list(spec.prompt_ids)],
            samplers=[sampler],
            logits_processors=[processors or []],
            state_machines=[machine],
        )
        self._sequences[uid] = _Sequence(
            request_id=spec.request_id,
            # Each access constructs a fresh streaming detokenizer, so
            # sequences do not share offset state.
            detokenizer=self._tokenizer.detokenizer,
            buffer=StopTextBuffer(spec.stop),
            prompt_tokens=len(spec.prompt_ids),
        )

    def _abort(self, request_id: str) -> None:
        """Remove one request - or all of them, for an empty id."""
        if request_id:
            uids = [
                uid
                for uid, seq in self._sequences.items()
                if seq.request_id == request_id
            ]
        else:
            uids = list(self._sequences)
        if not uids:
            return
        self._remove(uids)
        for uid in uids:
            self._finish(uid, "abort")

    # -- stepping ----------------------------------------------------------

    def _step(self) -> None:
        _, generated = self._generator.next()
        stopped_on_text: list[int] = []
        for response in generated:
            seq = self._sequences.get(response.uid)
            if seq is None:
                continue

            if response.finish_reason == "stop":
                # A stop token. Never detokenized - its text would land in
                # the output it is supposed to end - and not counted, matching
                # the serial loop. The generator already evicted it.
                self._finish(response.uid, "stop")
                continue

            token = response.token
            token_id = int(token.item() if hasattr(token, "item") else token)
            seq.detokenizer.add_token(token_id)
            chunk = seq.buffer.push(seq.detokenizer.last_segment)
            seq.generated += 1

            if seq.buffer.hit is not None:
                # A stop *string*, produced by a real token whose text the
                # buffer truncated at the match. Still in the generator's
                # batch, so it needs an explicit eviction - on every rank.
                if chunk:
                    self._chunk(seq, chunk)
                stopped_on_text.append(response.uid)
                self._finish(response.uid, "stop")
                continue

            if chunk:
                self._chunk(seq, chunk)

            if response.finish_reason == "length":
                seq.detokenizer.finalize()
                tail = seq.buffer.push(seq.detokenizer.last_segment)
                tail += seq.buffer.flush()
                if seq.buffer.hit is not None:
                    self._finish(response.uid, "stop")
                    continue
                if tail:
                    self._chunk(seq, tail)
                self._finish(response.uid, "length")

        if stopped_on_text:
            self._remove(stopped_on_text)

    # -- plumbing ----------------------------------------------------------

    def _chunk(self, seq: _Sequence, text: str) -> None:
        self._reply(
            {
                "ok": True,
                "request_id": seq.request_id,
                "chunk": text,
                "tokens": seq.generated,
            }
        )

    def _finish(self, uid: int, finish_reason: str) -> None:
        seq = self._sequences.pop(uid, None)
        if seq is None:
            return
        self._reply(
            {
                "ok": True,
                "request_id": seq.request_id,
                "done": True,
                "text": seq.buffer.text,
                "prompt_tokens": seq.prompt_tokens,
                "completion_tokens": seq.generated,
                "finish_reason": finish_reason,
            }
        )

    def _remove(self, uids: list[int]) -> None:
        """Evict sequences - on the generator's own stream, which is load-
        bearing. A natural finish is filtered *inside* `next()`, under the
        generation stream; an eviction from here would otherwise slice the
        surviving sequences' caches on the default stream, and the next
        forward deadlocks on the cross-stream residue - measured 3/3 as
        "abort lands, then the whole batch wedges" whenever another sequence
        was still decoding.
        """
        generator = self._ensure_generator()
        stream = getattr(generator, "stream", None)
        if stream is None:  # test fakes have no stream
            generator.remove(uids)
            return
        import mlx.core as mx

        with mx.stream(stream):
            generator.remove(uids)

    def _prefill_by_hand(self, prompt_ids: list[int]) -> Any:
        """A fresh cache holding everything but the last prompt token."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self._model)
        tokens = prompt_ids[:-1]
        step = self._config.prefill_step_size
        for start in range(0, len(tokens), step):
            chunk = tokens[start : start + step]
            self._model(mx.array(chunk)[None], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def _ensure_generator(self) -> Any:
        if self._generator is None:
            self._generator = self._factory(self._model, self._config)
        return self._generator

    def _eos_ids(self) -> set[int]:
        eos = getattr(self._tokenizer, "eos_token_ids", None)
        if eos:
            return set(eos)
        single = getattr(self._tokenizer, "eos_token_id", None)
        return {single} if single is not None else set()

    @property
    def active(self) -> int:
        return len(self._sequences)
