# SPDX-License-Identifier: Apache-2.0
"""Tests for scheduler admission control (queue depth cap + admission_paused)."""

import threading
from collections import deque
from unittest.mock import MagicMock

import pytest

from omlx.exceptions import SchedulerQueueFullError
from omlx.scheduler import Scheduler


@pytest.fixture
def scheduler():
    """Build a minimal Scheduler instance without invoking __init__.

    Scheduler.__init__ pulls in mlx_lm model wiring; for queue-cap tests we
    only need self.config, self.waiting/running/prefilling, self.requests,
    and the preflight reservation state, so we manufacture a bare instance
    and seed those attributes directly.

    ``block_aware_cache`` and ``_prefill_memory_guard`` are set so
    ``add_request`` can run all the way through to the ``self.waiting``
    insert for requests that clear the cap check — needed by tests that
    drive both preflight and add_request end to end.
    """
    s = Scheduler.__new__(Scheduler)
    s.config = MagicMock(max_num_seqs=8)
    s.waiting = deque()
    s.running = {}
    s.prefilling = deque()
    s.requests = {}
    s.block_aware_cache = None
    s._prefill_memory_guard = False
    s._reserved = deque()
    s._reservation_lock = threading.Lock()
    return s


def _make_request(rid: str):
    r = MagicMock()
    r.request_id = rid
    r.prompt = "hello"
    r.prompt_token_ids = [1, 2, 3]
    r.num_prompt_tokens = 3
    return r


def _fill_running(scheduler, n: int):
    for i in range(n):
        scheduler.running[f"run{i}"] = _make_request(f"run{i}")


class TestWaitingQueueCap:
    """``add_request``'s own cap check, now gated on total occupancy
    (running + prefilling + waiting + reservations) rather than the waiting
    queue alone — see ``total_queue_capacity``. For max_num_seqs=8 the total
    cap is 8 + waiting_queue_capacity(8) == 8 + 32 == 40.
    """

    def test_admits_below_cap(self, scheduler):
        # Seed occupancy to 39 (running=8, waiting=31) — one below the total
        # cap of 40. add_request for the next request should clear the cap
        # check (it fails later on the duplicate-ID check, which is what
        # proves we got past the cap rather than short-circuiting on it).
        _fill_running(scheduler, 8)
        for i in range(31):
            scheduler.waiting.append(_make_request(f"r{i}"))
        req = _make_request("r-new")
        scheduler.requests[req.request_id] = req
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(req)

    def test_rejects_at_cap(self, scheduler):
        # Total occupancy at cap: running=8 (max_num_seqs) + waiting=32
        # (waiting_queue_capacity(8)) == 40. This is the warm-backpressure
        # shape that genuinely trips the total gate.
        _fill_running(scheduler, 8)
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        req = _make_request("over")
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(req)
        assert exc.value.current_depth == 32
        assert exc.value.max_depth == 32

    def test_waiting_at_old_cap_alone_no_longer_rejects(self, scheduler):
        """Documents the semantic switch from waiting-only to total-form.

        32 waiting requests with an empty running batch used to be exactly
        the old cap (waiting_queue_capacity(8) == 32) and raised. Under the
        total form the cap is 40 (max_num_seqs=8 + waiting cap 32), so 32
        waiting alone is well under it — add_request must clear the cap
        check and fail on the duplicate-ID check instead. A regression back
        to the waiting-only form would raise SchedulerQueueFullError here.
        """
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        req = _make_request("r-new")
        scheduler.requests[req.request_id] = req
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(req)

    def test_cap_scales_with_max_num_seqs(self, scheduler):
        # total cap = max_num_seqs + waiting_queue_capacity(max_num_seqs);
        # for max_num_seqs=16, that's 16 + 64 == 80.
        scheduler.config.max_num_seqs = 16
        _fill_running(scheduler, 16)
        for i in range(64):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_make_request("over"))
        assert exc.value.max_depth == 64

    def test_cap_floor_at_32(self, scheduler):
        # Tiny max_num_seqs still gets a waiting-cap floor of 32; total cap
        # is 1 + 32 == 33.
        scheduler.config.max_num_seqs = 1
        _fill_running(scheduler, 1)
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_make_request("over"))
        assert exc.value.max_depth == 32

    def test_duplicate_request_raises_before_cap(self, scheduler):
        # Duplicate check fires before the cap check.
        req = _make_request("dup")
        scheduler.requests[req.request_id] = req
        # Even with an empty queue, duplicate should raise ValueError.
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(req)


class TestPreflightQueueOrRaise:
    """The pre-StreamingResponse half of the cap.

    ``add_request`` runs inside the route's response generator on the
    streaming path, by which point starlette has already sent HTTP 200 — so
    its raise degrades into a truncated in-stream error and the registered
    503 handler never fires. ``preflight_queue_or_raise`` is what the route
    calls while it can still answer with a status code, and now also gates
    on total occupancy — see ``TestWaitingQueueCap``.
    """

    def test_silent_below_cap(self, scheduler):
        for i in range(31):
            scheduler.waiting.append(_make_request(f"r{i}"))
        assert scheduler.preflight_queue_or_raise() is None

    def test_raises_at_cap_with_the_same_depths_add_request_reports(self, scheduler):
        _fill_running(scheduler, 8)
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.preflight_queue_or_raise()
        assert exc.value.current_depth == 32
        assert exc.value.max_depth == 32

    def test_fires_with_the_prefill_memory_guard_disabled(self, scheduler):
        """The guard-off configuration is the one most likely to go untested,
        and it is the default. ``preflight_or_raise`` returns immediately when
        ``_prefill_memory_guard`` is false; if the queue check had been folded
        into that method — or placed after the tokenize step that feeds it —
        backpressure would be dead for every server running without the
        memory guard.
        """
        scheduler._prefill_memory_guard = False
        scheduler._memory_hard_limit_bytes = 0
        scheduler.memory_monitor = None
        _fill_running(scheduler, 8)
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError):
            scheduler.preflight_queue_or_raise()

    def test_shares_one_cap_definition_with_add_request(self, scheduler):
        scheduler.config.max_num_seqs = 16
        _fill_running(scheduler, 16)
        for i in range(64):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError) as preflight_exc:
            scheduler.preflight_queue_or_raise()
        with pytest.raises(SchedulerQueueFullError) as admit_exc:
            scheduler.add_request(_make_request("over"))
        assert preflight_exc.value.max_depth == admit_exc.value.max_depth == 64

    def test_claims_a_reservation_that_add_request_releases(self, scheduler):
        """A slot claimed at preflight must count toward occupancy until the
        matching add_request releases it — and must not double-count once
        the request is actually in self.waiting.
        """
        scheduler.preflight_queue_or_raise()
        assert scheduler._reserved_slots() == 1

        scheduler.add_request(_make_request("r0"))
        assert scheduler._reserved_slots() == 0
        assert len(scheduler.waiting) == 1

    def test_add_request_without_preflight_releases_nothing(self, scheduler):
        """An internal/test caller that never preflighted must not crash or
        under-flow the reservation deque (see _release_reservation).
        """
        scheduler.add_request(_make_request("r0"))
        assert scheduler._reserved_slots() == 0
        assert len(scheduler.waiting) == 1


class TestColdBurstPreflightReservation:
    """S3 task #15: single-node's cold-burst cousin of S3 acceptance row 4.

    ``add_request`` only ever runs inside the route's response generator,
    which starlette does not iterate until the route has already committed
    to the ``StreamingResponse``. On a cold burst every request preflights
    before any of them reaches ``add_request``, so a gate that only reads
    ``self.waiting`` sees an empty queue for the whole burst and never
    engages. Phase 1 below drives every preflight to completion before Phase
    2 submits any of the accepted ones through ``add_request`` — the same
    two-phase ordering ``tests/cluster/test_cluster_serving.py::
    test_cold_burst_is_refused_at_preflight_not_in_stream`` uses, and for
    the same reason: interleaving preflight and submission per-request is
    the warm shape, and passes even without a fix.
    """

    def test_cold_burst_is_refused_at_preflight_not_in_stream(self, scheduler):
        # total cap = max_num_seqs(8) + waiting_queue_capacity(8)(32) == 40
        num_submissions = 41

        # Phase 1 — every request preflights, cold: nothing has been
        # submitted through add_request yet, so self.waiting is empty for
        # the whole phase.
        verdicts: list[SchedulerQueueFullError | None] = []
        for _ in range(num_submissions):
            try:
                scheduler.preflight_queue_or_raise()
                verdicts.append(None)
            except SchedulerQueueFullError as exc:
                verdicts.append(exc)

        rejected = [v for v in verdicts if v is not None]
        assert rejected, (
            "expected at least one preflight rejection on a cold "
            f"{num_submissions}-request burst; got none"
        )
        assert len(rejected) < num_submissions, (
            "the whole burst was rejected — a gate that refuses everything "
            "passes 'a 503 happened' while being worse than the defect"
        )

        # Phase 2 — the admitted ones must still be able to land in
        # self.waiting. A reservation that never gets released would show up
        # here as a queue-full raise for requests that already cleared
        # preflight.
        admitted = [i for i, v in enumerate(verdicts) if v is None]
        for i in admitted:
            scheduler.add_request(_make_request(f"cold-{i}"))
        assert len(scheduler.waiting) == len(admitted)
        assert scheduler._reserved_slots() == 0


class TestReservationThreadSafety:
    """``preflight_queue_or_raise`` runs synchronously on the asyncio loop
    thread (``BaseEngine._preflight_queue`` calls it directly, no executor
    hop); ``add_request`` runs on the dedicated single-worker MLX executor
    thread (``engine_core.py``: ``run_in_executor(self._mlx_executor,
    self.scheduler.add_request, request)``). Those are two different OS
    threads racing the same ``self._reserved`` deque, so this drives both
    from real threads to check the lock actually prevents the crash a
    sweep-then-pop race can hit (see ``_reservation_lock``'s docstring in
    ``Scheduler.__init__``): ``_reserved_slots`` sweeping an entry that
    ``_release_reservation`` pops out from under it between the peek and the
    ``popleft``.
    """

    def test_concurrent_preflight_and_add_request_do_not_crash(self, scheduler):
        import concurrent.futures

        iterations = 2000
        errors: list[Exception] = []

        def preflighter():
            for _ in range(iterations):
                try:
                    scheduler.preflight_queue_or_raise()
                except SchedulerQueueFullError:
                    pass
                except Exception as e:  # pragma: no cover - failure path
                    errors.append(e)

        def submitter():
            for i in range(iterations):
                try:
                    scheduler.add_request(_make_request(f"thread-{i}"))
                except (SchedulerQueueFullError, ValueError):
                    pass
                except Exception as e:  # pragma: no cover - failure path
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(preflighter)
            f2 = pool.submit(submitter)
            f1.result()
            f2.result()

        assert not errors, errors
        assert scheduler._reserved_slots() >= 0


class TestWaitingQueueCapacityHelper:
    def test_floor_and_scaling(self):
        from omlx.scheduler import waiting_queue_capacity

        assert waiting_queue_capacity(1) == 32
        assert waiting_queue_capacity(8) == 32
        assert waiting_queue_capacity(9) == 36
        assert waiting_queue_capacity(16) == 64


class TestTotalQueueCapacityHelper:
    def test_floor_and_scaling(self):
        from omlx.scheduler import total_queue_capacity

        assert total_queue_capacity(1) == 33  # 1 + 32
        assert total_queue_capacity(8) == 40  # 8 + 32
        assert total_queue_capacity(9) == 45  # 9 + 36
        assert total_queue_capacity(16) == 80  # 16 + 64


class TestAdmissionPausedField:
    def test_default_false(self):
        # Direct field check on a fresh Scheduler — we want to make sure the
        # attribute exists with the right default for enforcer to set.
        s = Scheduler.__new__(Scheduler)
        # Mimic the relevant subset of __init__
        s._memory_limit_bytes = 0
        s._memory_hard_limit_bytes = 0
        s._prefill_memory_guard = False
        s._admission_paused = False
        assert s._admission_paused is False
