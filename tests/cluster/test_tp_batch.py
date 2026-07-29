# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the S3 forward-replay mechanism (tp_batch.py).

Every test here runs at world_size=1 by necessity (no real second process),
which means ordering, follower reconciliation, and cross-rank shape
agreement are structurally invisible to this file — see
``discovery/analysis/s3-interface-audit.md`` section 7.
``tests/cluster/test_rank_batch.py`` (two real local rank processes) is the
integration test that actually exercises the broadcast.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from omlx.cluster.protocol import (
    PHASE_ADMIT,
    PHASE_BATCHED,
    PHASE_STANDALONE,
    ProtocolError,
    RankOp,
)
from omlx.cluster.tp_batch import (
    FollowerReplayer,
    LeaderModelProxy,
    TPBatchGenerator,
)

VOCAB = 32
N_LAYERS = 2
N_KV_HEADS = 1
HEAD_DIM = 4
BOOSTED_TOKEN = 1


class _FakeSession:
    """Records every broadcast; returns the payload unchanged, matching
    ``DistributedSession.broadcast_json``'s world_size==1 behaviour."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def broadcast_json(self, obj):
        self.sent.append(obj)
        return obj


def _identity_model(tokens, cache=None, **kwargs):
    return tokens


class _FakeCacheElement:
    """A plain weakref-able stand-in for one layer's KVCache — good enough
    for tag registry tests that never call merge/filter on it (bare
    ``object()`` instances are not weakref-able). ``state`` is a trivial
    mx-eval-able value: LeaderModelProxy.__call__ evals cache state on every
    call (to force TP collectives on a real cache), and these tests exercise
    that same call path."""

    @property
    def state(self) -> Any:
        return mx.array([])


def _fake_cache_list(n: int = 2) -> list[object]:
    return [_FakeCacheElement() for _ in range(n)]


class _FakeOwner:
    """A plain weakref-able stand-in for a TPBatchGenerator instance
    (``SimpleNamespace``, like bare ``object()``, has no ``__weakref__``
    slot)."""


# -- LeaderModelProxy: broadcast no-op at world_size=1, tag bookkeeping ------


def test_proxy_prefill_call_assigns_one_tag_and_broadcasts():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    cache = _fake_cache_list()

    proxy(mx.array([[10, 11, 12]]), cache=cache)

    assert len(session.sent) == 1
    op = RankOp.from_dict(session.sent[0])
    assert op.phase == PHASE_STANDALONE
    assert op.token_ids == [[10, 11, 12]]
    assert len(op.tags) == 1


def test_proxy_reuses_tag_for_same_cache_list_identity():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    cache = _fake_cache_list()

    proxy(mx.array([[1]]), cache=cache)
    proxy(mx.array([[2]]), cache=cache)

    tag_a = RankOp.from_dict(session.sent[0]).tags[0]
    tag_b = RankOp.from_dict(session.sent[1]).tags[0]
    assert tag_a == tag_b


def test_proxy_distinct_cache_lists_get_distinct_tags():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)

    proxy(mx.array([[1]]), cache=_fake_cache_list())
    proxy(mx.array([[2]]), cache=_fake_cache_list())

    tag_a = RankOp.from_dict(session.sent[0]).tags[0]
    tag_b = RankOp.from_dict(session.sent[1]).tags[0]
    assert tag_a != tag_b


def test_proxy_armed_call_uses_given_tags_and_phase_batched():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    owner = _FakeOwner()

    proxy.arm([5, 6], owner=owner)
    proxy(mx.array([[7], [8]]))  # no cache kwarg — decode-shaped call

    op = RankOp.from_dict(session.sent[0])
    assert op.phase == PHASE_BATCHED
    assert op.tags == [5, 6]
    assert op.token_ids == [[7], [8]]


def test_proxy_arm_with_admit_phase_overrides_wire_framing():
    """TPBatchGenerator.insert()'s delta forward arms with phase=PHASE_ADMIT
    (not the arm() default of PHASE_BATCHED): it is a self-contained forward
    over one row's own cache, not a statement about the running batch's
    membership, and the wire framing must say so (see LeaderModelProxy.arm's
    docstring and protocol.py's PHASE_ADMIT documentation)."""
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    owner = _FakeOwner()

    proxy.arm([9], owner=owner, phase=PHASE_ADMIT)
    proxy(mx.array([[3]]))

    op = RankOp.from_dict(session.sent[0])
    assert op.phase == PHASE_ADMIT
    assert op.tags == [9]


def test_proxy_raises_without_armed_tags_or_cache():
    proxy = LeaderModelProxy(_identity_model, _FakeSession())
    with pytest.raises(ProtocolError):
        proxy(mx.array([[1]]))


def test_proxy_delegates_unknown_attributes_to_wrapped_model():
    model = SimpleNamespace(make_cache=lambda: "the-cache")
    proxy = LeaderModelProxy(model, _FakeSession())
    assert proxy.make_cache() == "the-cache"


# -- tag-lifetime: sweep + explicit release ----------------------------------


def test_sweep_releases_standalone_cache_dropped_without_removal():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    cache = _fake_cache_list()
    proxy(mx.array([[1]]), cache=cache)  # tag assigned, standalone phase
    tag = RankOp.from_dict(session.sent[0]).tags[0]

    del cache
    gc.collect()

    # Next call is unrelated (a different cache); its op's `release` must
    # report the swept tag, and the tag registry must never reuse it.
    proxy(mx.array([[2]]), cache=_fake_cache_list())
    op = RankOp.from_dict(session.sent[1])
    assert tag in op.release
    assert tag not in op.tags


def test_sweep_does_not_release_a_live_batched_row():
    """Regression for the merge-invalidates-the-anchor finding (advisor
    review, s3-interface-audit.md section 3): once a standalone cache is
    promoted into a batch (arm() + a call), dropping the ORIGINAL prefill
    cache object must NOT cause a release — the tag's anchor has migrated to
    the owning generator, which is still alive.
    """
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    cache = _fake_cache_list()

    proxy(mx.array([[1]]), cache=cache)  # standalone: prefill
    tag = RankOp.from_dict(session.sent[0]).tags[0]

    owner = _FakeOwner()
    proxy.arm([tag], owner=owner)
    proxy(mx.array([[2]]))  # promotes tag: standalone -> batched(owner)

    # Drop every reference to the original prefill cache list AND its
    # element — this is exactly what scheduler.py's recovery branches do
    # (request.prompt_cache = None) once a row is running.
    del cache
    gc.collect()

    proxy(mx.array([[3]]), cache=_fake_cache_list())
    op = RankOp.from_dict(session.sent[2])
    assert tag not in op.release, "a live batched row must not be swept"


def test_sweep_releases_batched_row_once_owner_dies():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    cache = _fake_cache_list()
    proxy(mx.array([[1]]), cache=cache)
    tag = RankOp.from_dict(session.sent[0]).tags[0]

    owner = _FakeOwner()
    proxy.arm([tag], owner=owner)
    proxy(mx.array([[2]]))

    del owner
    gc.collect()

    proxy(mx.array([[3]]), cache=_fake_cache_list())
    op = RankOp.from_dict(session.sent[2])
    assert tag in op.release


def test_explicit_release_is_deterministic_not_gc_dependent():
    session = _FakeSession()
    proxy = LeaderModelProxy(_identity_model, session)
    cache = _fake_cache_list()
    proxy(mx.array([[1]]), cache=cache)
    tag = RankOp.from_dict(session.sent[0]).tags[0]

    proxy.arm([tag], owner=_FakeOwner())
    proxy(mx.array([[2]]))

    released = proxy.release_tags([tag])
    assert released == [tag]

    # Rides the NEXT broadcast's release field without needing a GC pass.
    proxy(mx.array([[3]]), cache=_fake_cache_list())
    op = RankOp.from_dict(session.sent[2])
    assert tag in op.release


# -- FollowerReplayer: replay a recorded op sequence -------------------------


def _stub_model_and_make_cache():
    from mlx_lm.models.cache import KVCache

    def model(tokens, cache=None, **kwargs):
        if cache:
            b, t = tokens.shape
            keys = mx.zeros((b, N_KV_HEADS, t, HEAD_DIM))
            values = mx.zeros((b, N_KV_HEADS, t, HEAD_DIM))
            for c in cache:
                c.update_and_fetch(keys, values)
        b, t = tokens.shape
        rows = [
            [100.0 if v == BOOSTED_TOKEN else 0.0 for v in range(VOCAB)]
            for _ in range(t)
        ]
        return mx.array([rows for _ in range(b)])

    def make_cache():
        return [KVCache() for _ in range(N_LAYERS)]

    return model, make_cache


def test_follower_replays_standalone_then_batched_ops():
    model, make_cache = _stub_model_and_make_cache()
    replayer = FollowerReplayer(model, make_cache)

    replayer.apply(RankOp(tags=[1], token_ids=[[5, 6]], phase=PHASE_STANDALONE))
    assert 1 in replayer._standalone
    assert replayer._batch_tags == []

    replayer.apply(RankOp(tags=[1], token_ids=[[7]], phase=PHASE_BATCHED))
    assert replayer._batch_tags == [1]
    assert 1 not in replayer._standalone


def test_follower_admit_does_not_touch_running_batch():
    """PHASE_ADMIT's single tag must never be read as "the batch is now just
    this row": TPBatchGenerator.insert()'s delta forward for a *second* row
    must not disturb a first row that is already merged into the running
    batch. Regression for the P1 rig hang: with PHASE_ADMIT folded into
    PHASE_BATCHED's wire framing, receiving tag 2's admit op while tag 1 was
    the running batch's only member made _reconcile_batched() read `tags=[2]`
    as complete membership, dropping tag 1's cache entirely — surfacing three
    ops later as an order desync (`[2, 1]` vs `[1, 2]`) and, on the real rig,
    a follower crash that hung the leader's next collective.
    """
    model, make_cache = _stub_model_and_make_cache()
    replayer = FollowerReplayer(model, make_cache)

    replayer.apply(RankOp(tags=[1], token_ids=[[5, 6]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1], token_ids=[[7]], phase=PHASE_ADMIT))
    assert replayer._batch_tags == []
    assert 1 in replayer._admitted

    replayer.apply(RankOp(tags=[1], token_ids=[[8]], phase=PHASE_BATCHED))
    assert replayer._batch_tags == [1]
    assert 1 not in replayer._admitted

    replayer.apply(RankOp(tags=[2], token_ids=[[9, 10]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[2], token_ids=[[11]], phase=PHASE_ADMIT))
    # Row 1 must be untouched by row 2's isolated admit forward.
    assert replayer._batch_tags == [1]
    assert 2 in replayer._admitted

    replayer.apply(RankOp(tags=[1, 2], token_ids=[[9], [12]], phase=PHASE_BATCHED))
    assert replayer._batch_tags == [1, 2]


def test_follower_admits_second_row_alongside_first():
    model, make_cache = _stub_model_and_make_cache()
    replayer = FollowerReplayer(model, make_cache)

    replayer.apply(RankOp(tags=[1], token_ids=[[5]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1], token_ids=[[6]], phase=PHASE_BATCHED))

    replayer.apply(RankOp(tags=[2], token_ids=[[9]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1, 2], token_ids=[[7], [10]], phase=PHASE_BATCHED))

    assert replayer._batch_tags == [1, 2]


def test_follower_filters_finished_row_by_absence():
    model, make_cache = _stub_model_and_make_cache()
    replayer = FollowerReplayer(model, make_cache)

    replayer.apply(RankOp(tags=[1], token_ids=[[5]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1], token_ids=[[6]], phase=PHASE_BATCHED))
    replayer.apply(RankOp(tags=[2], token_ids=[[9]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1, 2], token_ids=[[7], [10]], phase=PHASE_BATCHED))

    # Row 1 finishes: the next batched op no longer lists it.
    replayer.apply(RankOp(tags=[2], token_ids=[[11]], phase=PHASE_BATCHED, release=[1]))
    assert replayer._batch_tags == [2]


def test_follower_rejects_order_desync():
    model, make_cache = _stub_model_and_make_cache()
    replayer = FollowerReplayer(model, make_cache)
    replayer.apply(RankOp(tags=[1], token_ids=[[5]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1], token_ids=[[6]], phase=PHASE_BATCHED))
    replayer.apply(RankOp(tags=[2], token_ids=[[9]], phase=PHASE_STANDALONE))
    replayer.apply(RankOp(tags=[1, 2], token_ids=[[7], [10]], phase=PHASE_BATCHED))
    assert replayer._batch_tags == [1, 2]

    # Same membership, swapped order: simple filter-then-append reconciliation
    # cannot reproduce this without reordering rows in place, which it never
    # does — this must be caught loudly rather than silently mis-shaping the
    # next merged forward.
    with pytest.raises(ProtocolError):
        replayer.apply(RankOp(tags=[2, 1], token_ids=[[11], [12]], phase=PHASE_BATCHED))


def test_follower_release_frees_standalone_cache():
    model, make_cache = _stub_model_and_make_cache()
    replayer = FollowerReplayer(model, make_cache)
    replayer.apply(RankOp(tags=[1], token_ids=[[5]], phase=PHASE_STANDALONE))
    assert 1 in replayer._standalone

    replayer.apply(RankOp(tags=[], token_ids=[], phase=PHASE_BATCHED, release=[1]))
    assert 1 not in replayer._standalone


# -- TPBatchGenerator conformance, driven by the real Scheduler --------------


def _stub_scheduler_model():
    from mlx_lm.models.cache import KVCache

    class _StubModel:
        def __init__(self):
            self.config = SimpleNamespace(num_hidden_layers=N_LAYERS)

        def make_cache(self):
            return [KVCache() for _ in range(N_LAYERS)]

        def __call__(self, tokens, cache=None, **kwargs):
            if cache:
                b, t = tokens.shape
                keys = mx.zeros((b, N_KV_HEADS, t, HEAD_DIM))
                values = mx.zeros((b, N_KV_HEADS, t, HEAD_DIM))
                for c in cache:
                    c.update_and_fetch(keys, values)
            b, t = tokens.shape
            rows = [
                [100.0 if v == BOOSTED_TOKEN else 0.0 for v in range(VOCAB)]
                for _ in range(t)
            ]
            return mx.array([rows for _ in range(b)])

    return _StubModel()


def _build_cluster_scheduler(mock_tokenizer):
    from omlx.scheduler import Scheduler, SchedulerConfig

    session = _FakeSession()
    model = _stub_scheduler_model()
    proxy = LeaderModelProxy(model, session)
    factory = lambda sampling_params: TPBatchGenerator(proxy)  # noqa: E731
    scheduler = Scheduler(
        model=proxy,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(),
        batch_generator_factory=factory,
    )
    return scheduler, proxy, session


def _make_request(request_id: str, prompt_ids: list[int], max_tokens: int):
    from omlx.request import Request, SamplingParams

    request = Request(
        request_id=request_id,
        prompt=prompt_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    request.prompt_token_ids = list(prompt_ids)
    request.num_prompt_tokens = len(prompt_ids)
    return request


def test_factory_injection_builds_tp_batch_generator(mock_tokenizer):
    scheduler, proxy, session = _build_cluster_scheduler(mock_tokenizer)
    scheduler.add_request(_make_request("r1", [3, 4, 5], max_tokens=2))
    scheduler.step()
    assert isinstance(scheduler.batch_generator, TPBatchGenerator)


def test_conformance_prefill_then_decode_produces_boosted_token(mock_tokenizer):
    scheduler, proxy, session = _build_cluster_scheduler(mock_tokenizer)
    scheduler.add_request(_make_request("r1", [3, 4, 5], max_tokens=2))

    output = scheduler.step()
    assert any(o.request_id == "r1" for o in output.outputs)

    # The prefill call site (scheduler.py direct self.model(...)) went
    # through the proxy: at least one standalone-phase op was broadcast.
    assert any(RankOp.from_dict(op).phase == PHASE_STANDALONE for op in session.sent)
    r1 = next(o for o in output.outputs if o.request_id == "r1")
    assert r1.new_token_ids == [BOOSTED_TOKEN]


def test_conformance_finish_at_max_tokens_removes_and_releases_tag(mock_tokenizer):
    scheduler, proxy, session = _build_cluster_scheduler(mock_tokenizer)
    scheduler.add_request(_make_request("r1", [3, 4, 5], max_tokens=1))

    output = scheduler.step()
    r1 = next(o for o in output.outputs if o.request_id == "r1")
    assert r1.finished
    assert r1.finish_reason == "length"
    assert scheduler.batch_generator._tags == []


def test_conformance_admit_second_request_mid_decode(mock_tokenizer):
    scheduler, proxy, session = _build_cluster_scheduler(mock_tokenizer)
    scheduler.add_request(_make_request("r1", [3, 4, 5], max_tokens=4))
    scheduler.step()  # r1 prefilled + first decode token

    scheduler.add_request(_make_request("r2", [6, 7], max_tokens=4))
    output = scheduler.step()  # r1 continues, r2 prefills + admits

    ids = {o.request_id for o in output.outputs}
    assert "r1" in ids
    assert "r2" in ids
    assert len(scheduler.batch_generator._tags) == 2


def test_conformance_abort_mid_decode(mock_tokenizer):
    scheduler, proxy, session = _build_cluster_scheduler(mock_tokenizer)
    scheduler.add_request(_make_request("r1", [3, 4, 5], max_tokens=8))
    scheduler.step()
    assert scheduler.batch_generator._tags != []

    # Deferred abort semantics (D4): abort_request() only marks it; the
    # scheduler applies it at the START of the next step() (_process_pending
    # _aborts, scheduler.py:10092) — before next_generated() runs, so the
    # row is gone from the generator with no further decode for it. oMLX
    # does not synthesize a RequestOutput for a client-initiated abort (the
    # caller already knows it aborted); the observable effect is scheduler
    # state, not an output.outputs entry.
    assert scheduler.abort_request("r1") is True
    scheduler.step()

    assert "r1" not in scheduler.running
    assert "r1" not in scheduler.request_id_to_uid
    assert scheduler.batch_generator._tags == []
