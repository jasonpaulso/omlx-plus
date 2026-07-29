# SPDX-License-Identifier: Apache-2.0
"""S3 forward-replay: the mechanism that carries every model invocation from
rank 0 to the rest of a tensor-parallel formation under continuous batching.

Why this exists (see `discovery/spec/s3-plan.md` D2 and
`discovery/analysis/s3-interface-audit.md` for the evidence): the oMLX
``Scheduler`` does not route every model call through its batch generator —
prefill (both the single-shot and chunked paths) calls ``self.model(...)``
directly, and the generator receives an already-prefilled cache at insert
time. A collective mismatch across ranks is a hang, not a wrong answer, so
every one of those call sites must reach every rank identically. The fix is
architectural, not a generator variant: rank 0's ``Scheduler`` is constructed
with a :class:`LeaderModelProxy` as its model. Every ``__call__`` the
scheduler (or the generator below) makes broadcasts a forward op first, then
delegates to the real sharded model; followers never decide anything, they
just replay the op with :class:`FollowerReplayer`.

:class:`TPBatchGenerator` is the leader-only continuous-batching engine that
conforms to the four methods the scheduler consumes from mlx-lm's
``BatchGenerator`` (``insert``, ``next_generated``, ``remove``,
``extract_cache``). It wraps a single persistent ``mlx_lm.generate
.GenerationBatch`` — never ``PromptProcessingBatch``, which deadlocked under
TP collectives in the prior attempt (salvage evidence) — because
``GenerationBatch._step()`` already does exactly what D2 needs: one
deterministic forward per call, over a cache merged across every running row
via mlx-lm's own cache-batching machinery.

Tag lifetime (the cache-tag registry) has two phases, because merging a
cache (``mlx_lm.generate._merge_caches``) builds new objects — a per-request
``KVCache`` instance stops carrying live state the moment its row enters the
batch, so a tag's weakref anchor must migrate:

* **standalone** (prefill, before ``insert()``): anchored to one ``KVCache``
  element of the per-request cache list.
* **batched** (after ``insert()``): anchored to the owning
  :class:`TPBatchGenerator` instance itself. The scheduler's three
  generator-destroying recovery branches (``scheduler.py:10194,10214,10246``)
  all drop ``self.batch_generator`` — the only strong reference to that
  instance — so every tag anchored to it becomes collectible in one shot.

Per-row finish/abort is not weakref-mediated at all: :meth:`TPBatchGenerator
.remove` reports it to the proxy directly and deterministically.
"""

from __future__ import annotations

import itertools
import time
import weakref
from dataclasses import dataclass
from typing import Any, Protocol

from omlx.cluster.protocol import (
    PHASE_ADMIT,
    PHASE_BATCHED,
    PHASE_STANDALONE,
    ProtocolError,
    RankOp,
)


class Broadcaster(Protocol):
    """The one method LeaderModelProxy/FollowerReplayer need from a session.

    Matches ``DistributedSession.broadcast_json`` (``rank_worker.py``)
    exactly, so production code passes a real session and unit tests pass a
    trivial stand-in with no ``mx.distributed`` dependency.
    """

    def broadcast_json(self, obj: Any | None) -> Any: ...


def _sync() -> None:
    """Drain the model stream before a broadcast (see module docstring and
    ``DistributedSession.broadcast_json``'s docstring): every rank must hand
    the backend collectives in one global order, so any lazy op left over
    from a prior forward must be materialised first.
    """
    import mlx.core as mx

    mx.synchronize()


# -- the leader-side model wrapper --------------------------------------------


@dataclass
class _StandaloneEntry:
    cache_id: int
    anchor: Any  # weakref.ref


class LeaderModelProxy:
    """Wraps rank 0's real (possibly TP-sharded) model.

    Two ways a call arrives:

    * Scheduler's own direct prefill calls pass a single per-request cache
      list. The tag is derived from that list's identity — assigned on
      first sight, reused across the same request's later chunks (the list
      object is never replaced mid-prefill; confirmed in the interface
      audit).
    * :class:`TPBatchGenerator`'s batched decode calls arm the proxy first
      via :meth:`arm`, handing it the full ordered tag list the merged
      cache's rows are about to use — a merged cache carries no per-row
      identity of its own, so this is the only channel for that path.

    Everything except ``__call__`` delegates to the wrapped model via
    ``__getattr__`` (``make_cache``, ``register_rope_delta``,
    ``clear_vlm_position_state``, ...) so the whole surface of ``hasattr``
    checks the scheduler makes keeps working transparently.
    """

    def __init__(self, model: Any, session: Broadcaster) -> None:
        self._model = model
        self._session = session
        self._tag_source = itertools.count(1)
        self._standalone: dict[int, _StandaloneEntry] = {}
        self._standalone_by_cache_id: dict[int, int] = {}
        self._batched: dict[int, Any] = {}  # tag -> weakref.ref(owner)
        self._armed: list[int] | None = None
        self._armed_owner: Any | None = None
        self._armed_phase: str = PHASE_BATCHED
        self._pending_release: list[int] = []
        self.tax_samples: list[float] = []

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes LeaderModelProxy itself doesn't define
        # (this class defines no __slots__, so __getattr__ only fires on a
        # genuine miss).
        return getattr(self._model, name)

    # -- tag lifecycle -------------------------------------------------------

    def tag_for_cache(self, cache_list: list[Any]) -> int:
        """Assign (or look up) a tag for a standalone per-request cache list."""
        if not cache_list:
            raise ProtocolError("cannot tag an empty cache list")
        cache_id = id(cache_list)
        tag = self._standalone_by_cache_id.get(cache_id)
        if tag is not None:
            return tag
        tag = next(self._tag_source)
        self._standalone[tag] = _StandaloneEntry(
            cache_id=cache_id, anchor=weakref.ref(cache_list[0])
        )
        self._standalone_by_cache_id[cache_id] = tag
        return tag

    def arm(self, tags: list[int], owner: Any, phase: str = PHASE_BATCHED) -> None:
        """Declare the row order the next ``__call__`` will use (decode path).

        Single-use: consumed and cleared by the next ``__call__``. ``phase``
        controls only the wire framing, not tag-registry bookkeeping (the
        call is always promoted to the batched anchor below): a caller that
        arms a tag for a self-contained forward over that row's own cache —
        not a statement about the running batch's full membership — must
        pass ``phase=PHASE_STANDALONE``, or a follower reconciling a later
        genuinely-batched op's tag list will read this call's single-tag
        payload as "the batch is now just this row" and drop every other
        row (see :meth:`TPBatchGenerator.insert`).
        """
        self._armed = list(tags)
        self._armed_owner = owner
        self._armed_phase = phase

    def release_tags(self, tags: list[int]) -> list[int]:
        """Explicit, deterministic release (``TPBatchGenerator.remove()``).

        Deregisters immediately and queues the tags to ride the next
        broadcast's ``release`` field (or an explicit :meth:`flush` if the
        caller knows no further forward is coming this step).
        """
        released = []
        for tag in tags:
            entry = self._standalone.pop(tag, None)
            if entry is not None:
                del self._standalone_by_cache_id[entry.cache_id]
                released.append(tag)
                continue
            if self._batched.pop(tag, None) is not None:
                released.append(tag)
        self._pending_release.extend(released)
        return released

    def _drain_pending(self) -> list[int]:
        pending, self._pending_release = self._pending_release, []
        return pending

    def _sweep(self) -> list[int]:
        """Detect caches abandoned without going through ``release_tags``
        (the scheduler's three generator-destroying recovery branches) and
        deregister them. Called at every op boundary (D2).
        """
        dead = [
            tag for tag, entry in self._standalone.items() if entry.anchor() is None
        ]
        dead += [tag for tag, ref in self._batched.items() if ref() is None]
        for tag in dead:
            entry = self._standalone.pop(tag, None)
            if entry is not None:
                del self._standalone_by_cache_id[entry.cache_id]
            self._batched.pop(tag, None)
        return dead

    def _promote(self, tags: list[int], owner: Any) -> None:
        owner_ref = weakref.ref(owner)
        for tag in tags:
            entry = self._standalone.pop(tag, None)
            if entry is not None:
                del self._standalone_by_cache_id[entry.cache_id]
            self._batched[tag] = owner_ref

    # -- the intercepted call --------------------------------------------------

    def __call__(self, tokens: Any, cache: Any = None, **kwargs: Any) -> Any:
        if self._armed is not None:
            tags = self._armed
            owner = self._armed_owner
            phase = self._armed_phase
            self._armed = None
            self._armed_owner = None
            self._armed_phase = PHASE_BATCHED
            self._promote(tags, owner)
        elif cache is not None:
            tags = [self.tag_for_cache(cache)]
            phase = PHASE_STANDALONE
        else:
            raise ProtocolError(
                "LeaderModelProxy called with neither armed tags nor a "
                "resolvable cache"
            )

        token_ids = tokens.tolist()
        released = self._sweep() + self._drain_pending()
        op = RankOp(tags=tags, token_ids=token_ids, release=released, phase=phase)

        _sync()
        t0 = time.perf_counter()
        self._session.broadcast_json(op.to_dict())
        self.tax_samples.append((time.perf_counter() - t0) * 1000.0)

        result = self._model(tokens, cache=cache, **kwargs)
        # Force this forward's TP collectives to run before any later call
        # can race ahead and issue a different collective (a new broadcast,
        # or the next op's own forward) out of step with every follower —
        # who forces the identical pair (its own output plus its own cache
        # state) immediately after its own model call in
        # FollowerReplayer.apply(). Evaluating cache state alone is not
        # enough: cache writes come from each layer's local K/V projection,
        # before that layer's own attention-output/MLP all-reduce, so the
        # *last* layer's all-reduce is never on cache state's dependency
        # path — only `result` (which feeds the replicated lm_head) forces
        # it. Skipping either half here would make this rank issue a
        # different collective sequence than a follower that always forces
        # both, and that mismatch deadlocks the next collective.
        import mlx.core as mx

        if cache is not None:
            mx.eval(result, [c.state for c in cache])
        else:
            mx.eval(result)
        return result

    def flush(self, tags: list[int] | None = None) -> None:
        """Broadcast a release-only op when no forward is coming this step
        (``TPBatchGenerator.remove()`` emptied the running batch entirely).
        """
        released = self._sweep() + self._drain_pending()
        if not released:
            return
        tags = list(tags or [])
        op = RankOp(
            tags=tags,
            token_ids=[[] for _ in tags],
            release=released,
            phase=PHASE_BATCHED,
        )
        _sync()
        self._session.broadcast_json(op.to_dict())


# -- the follower's replay half -----------------------------------------------


class FollowerReplayer:
    """A follower's obedient half of the forward-replay contract (D2).

    Never decides anything — it replays whatever forward op the leader
    broadcasts, on its own model shard, maintaining cache state per tag.
    """

    def __init__(self, model: Any, make_cache: Any) -> None:
        self._model = model
        self._make_cache = make_cache
        self._standalone: dict[int, list[Any]] = {}
        # Tag -> already batch-wrapped, already-stepped cache produced by a
        # PHASE_ADMIT forward, waiting to be folded into the running batch by
        # a later PHASE_BATCHED op (mirrors GenerationBatch.extend() folding
        # an already-stepped delta batch into the persistent one).
        self._admitted: dict[int, list[Any]] = {}
        self._batch_tags: list[int] = []
        self._batch_cache: list[Any] = []

    def apply(self, op: RankOp) -> None:
        import mlx.core as mx
        from mlx_lm.generate import _merge_caches

        if op.phase == PHASE_STANDALONE:
            tag = op.tags[0]
            cache = self._standalone.get(tag)
            if cache is None:
                cache = self._make_cache()
                self._standalone[tag] = cache
            tokens = mx.array(op.token_ids)
            out = self._model(tokens, cache=cache)
            # Evaluate the model's own output, not just cache state: cache
            # writes come from each layer's *local* K/V projection, before
            # that layer's attention-output/MLP all-reduce (D4's
            # "sharded-to-all" shard_linear placement); the last layer's own
            # all-reduce is never on cache state's dependency path, only on
            # the final hidden state that feeds the (unsharded, replicated)
            # lm_head. Forcing cache state alone silently skips that
            # collective — this rank would never issue it, while the leader
            # eventually does (its result feeds sampling), a per-rank
            # collective-count mismatch that deadlocks the next op.
            mx.eval(out, [c.state for c in cache])
        elif op.phase == PHASE_ADMIT:
            tag = op.tags[0]
            cache = self._standalone.pop(tag, None)
            if cache is None:
                cache = self._make_cache()
            wrapped = _merge_caches([cache])
            tokens = mx.array(op.token_ids)
            out = self._model(tokens, cache=wrapped)
            mx.eval(out, [c.state for c in wrapped])
            self._admitted[tag] = wrapped
        elif op.phase == PHASE_BATCHED:
            self._reconcile_batched(op.tags)
            if op.tags:
                tokens = mx.array(op.token_ids)
                out = self._model(tokens, cache=self._batch_cache)
                mx.eval(out, [c.state for c in self._batch_cache])
        else:
            raise ProtocolError(f"FollowerReplayer: unhandled phase {op.phase!r}")

        self._release(op.release)

    def _reconcile_batched(self, tags: list[int]) -> None:
        from mlx_lm.generate import _extend_cache, _merge_caches

        keep_idx = [i for i, t in enumerate(self._batch_tags) if t in tags]
        if len(keep_idx) < len(self._batch_tags):
            if keep_idx:
                keep = _index_array(keep_idx)
                for c in self._batch_cache:
                    c.filter(keep)
            else:
                # Filtering a batch cache down to zero rows is not a case
                # mlx-lm's own .filter() handles (BatchKVCache.filter()
                # reduce-mins an empty array); drop the batch outright,
                # mirroring GenerationBatch.filter()'s own empty-keep path.
                self._batch_cache = []
            self._batch_tags = [self._batch_tags[i] for i in keep_idx]

        new_tags = [t for t in tags if t not in self._batch_tags]
        for tag in new_tags:
            # A tag admitted via PHASE_ADMIT already carries a wrapped,
            # once-stepped cache — reuse it rather than re-wrapping (it is
            # no longer the raw per-layer list _merge_caches expects) or
            # rebuilding it fresh (which would silently lose that step).
            wrapped = self._admitted.pop(tag, None)
            if wrapped is None:
                cache = self._standalone.pop(tag, None)
                if cache is None:
                    cache = self._make_cache()
                wrapped = _merge_caches([cache])
            self._batch_cache = _extend_cache(self._batch_cache, wrapped)
            self._batch_tags.append(tag)

        if self._batch_tags != tags:
            raise ProtocolError(
                "follower batch order desync: have "
                f"{self._batch_tags}, expected {tags}"
            )

    def _release(self, tags: list[int]) -> None:
        for tag in tags:
            self._standalone.pop(tag, None)
            self._admitted.pop(tag, None)
            if tag in self._batch_tags:
                idx = self._batch_tags.index(tag)
                keep_idx = [i for i in range(len(self._batch_tags)) if i != idx]
                if keep_idx:
                    keep = _index_array(keep_idx)
                    for c in self._batch_cache:
                        c.filter(keep)
                else:
                    self._batch_cache = []
                self._batch_tags.pop(idx)


def _index_array(indices: list[int]) -> Any:
    """mlx-lm's batch cache ``.filter()`` indexes with an mx.array; an empty
    Python list coerces to float32 by default (no ints to infer from),
    which ``mx`` fancy-indexing then rejects. Pin the dtype explicitly.
    """
    import mlx.core as mx

    return mx.array(indices, dtype=mx.int32)


# -- the leader-only continuous-batching engine -------------------------------


@dataclass
class Response:
    """``BatchGenerator.Response`` shim consumed by ``_process_batch_responses``
    (``scheduler.py:8955``): same field surface as the private
    ``_VLMMTPResponse`` shim already in that file (``uid``, ``token``,
    ``finish_reason``, ``logprobs``).
    """

    uid: int
    token: int
    finish_reason: str | None = None
    logprobs: Any = None


def _argmax_fallback(logprobs: Any) -> Any:
    import mlx.core as mx

    return mx.argmax(logprobs, axis=-1)


class TPBatchGenerator:
    """Leader-only continuous-batching engine driving TP decode (D2).

    Conforms to the four methods ``Scheduler`` consumes from mlx-lm's
    ``BatchGenerator`` (``insert``, ``next_generated``, ``remove``,
    ``extract_cache``). Internally wraps one persistent ``mlx_lm.generate
    .GenerationBatch`` and keeps ``self._tags`` in lockstep with its row
    order, so ``LeaderModelProxy`` always knows which tag each row is.

    ``extract_cache`` is unreachable in cluster mode: both scheduler call
    sites are gated behind ``self.block_aware_cache is not None``
    (``scheduler.py:5483,5823``), and D6(b) forces that ``None`` in rank
    processes.
    """

    def __init__(self, proxy: LeaderModelProxy) -> None:
        from mlx_lm.generate import GenerationBatch

        self._proxy = proxy
        # GenerationBatch's stub types `model` as an nn.Module; LeaderModelProxy
        # is a deliberate duck-typed stand-in (it forwards everything __call__
        # doesn't intercept via __getattr__), not a real Module subclass.
        self._gb = GenerationBatch.empty(
            model=proxy,  # type: ignore[arg-type]
            fallback_sampler=_argmax_fallback,
        )
        self._tags: list[int] = []

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int] | None = None,
        caches: list[list[Any]] | None = None,
        all_tokens: list[list[int]] | None = None,
        samplers: list[Any] | None = None,
        logits_processors: list[list[Any]] | None = None,
        state_machines: list[Any] | None = None,
    ) -> list[int]:
        import mlx.core as mx
        from mlx_lm.generate import GenerationBatch, SequenceStateMachine

        n = len(prompts)
        if caches is None or any(c is None for c in caches):
            raise ProtocolError(
                "TPBatchGenerator.insert requires a prefilled cache per row "
                "(cluster prefill always hands one in before insert)"
            )
        max_tokens = list(max_tokens) if max_tokens else [256] * n
        all_tokens = list(all_tokens) if all_tokens else [[] for _ in range(n)]
        samplers = list(samplers) if samplers else [None] * n
        logits_processors = (
            list(logits_processors) if logits_processors else [[] for _ in range(n)]
        )
        state_machines = (
            list(state_machines)
            if state_machines
            else [SequenceStateMachine() for _ in range(n)]
        )

        uids: list[int] = []
        for i in range(n):
            tag = self._proxy.tag_for_cache(caches[i])
            # This is GenerationBatch.__init__'s own synchronous first step,
            # run over a freshly batch-wrapped copy of this row's own cache
            # before it ever joins the shared running batch — not a
            # statement about that batch's membership, and not a plain
            # per-request forward either (mlx-lm always wraps the cache
            # here, even for one row). Wire it as PHASE_ADMIT (see
            # protocol.py and LeaderModelProxy.arm's docstring) so a
            # follower reconciling the *next* genuinely batched op neither
            # reads this single tag as "the batch is now just this row"
            # (dropping every other running row) nor replays it against an
            # unwrapped cache (a shape mismatch against the leader's).
            self._proxy.arm([tag], owner=self, phase=PHASE_ADMIT)
            delta = GenerationBatch(
                model=self._proxy,  # type: ignore[arg-type]  # see __init__
                uids=[tag],
                inputs=mx.array([prompts[i][-1]]),
                prompt_cache=_merge_one(caches[i]),
                tokens=[list(all_tokens[i])],
                samplers=[samplers[i]],
                fallback_sampler=_argmax_fallback,
                logits_processors=[list(logits_processors[i] or [])],
                state_machines=[state_machines[i]],
                max_tokens=[max_tokens[i]],
            )
            self._gb.extend(delta)
            self._tags.append(tag)
            uids.append(tag)
        return uids

    def next_generated(self) -> list[Response]:
        if not self._gb.uids:
            return []
        self._proxy.arm(list(self._tags), owner=self)
        tags_before = list(self._tags)
        responses = self._gb.next()

        finished = [
            tags_before[i]
            for i, r in enumerate(responses)
            if r.finish_reason is not None
        ]
        self._tags = [
            tags_before[i] for i, r in enumerate(responses) if r.finish_reason is None
        ]
        if finished:
            self._proxy.release_tags(finished)

        return [
            Response(
                uid=tags_before[i],
                token=r.token,
                finish_reason=r.finish_reason,
                logprobs=None,
            )
            for i, r in enumerate(responses)
        ]

    def remove(self, uids: list[int]) -> None:
        target = set(uids)
        present = [t for t in uids if t in self._gb.uids]
        if not present:
            return
        keep_idx = [i for i, t in enumerate(self._gb.uids) if t not in target]
        # GenerationBatch.filter() wants a plain list (it iterates it as
        # Python indices AND guards the empty case itself); only the raw
        # per-layer cache .filter() below needs an explicit-dtype mx.array.
        self._gb.filter(keep_idx)
        self._tags = [t for t in self._tags if t not in target]
        self._proxy.release_tags(present)
        if not self._gb.uids:
            self._proxy.flush(tags=[])

    def extract_cache(self, uids: list[int]) -> dict[int, Any]:
        raise NotImplementedError(
            "TPBatchGenerator.extract_cache is unreachable in cluster mode: "
            "D6(b) forces block_aware_cache=None in rank processes, and both "
            "scheduler call sites (scheduler.py:5483,5823) are gated behind "
            "block_aware_cache is not None."
        )


def _merge_one(cache_list: list[Any]) -> list[Any]:
    """Merge a single per-request cache list into GenerationBatch's expected
    batch-cache shape (a batch of one row)."""
    from mlx_lm.generate import _merge_caches

    merged: list[Any] = _merge_caches([cache_list])
    return merged
