# SPDX-License-Identifier: Apache-2.0
"""Builds the ``omlx.scheduler.SchedulerConfig`` a rank-0 process constructs
its real ``Scheduler`` with (S3 D6(b)).

Two things this seam is responsible for, both named in
``discovery/spec/s3-plan.md`` D6(b):

* **The paged-SSD cache stack must never initialize in a rank process.**
  ``block_aware_cache`` is only built when ``paged_ssd_cache_dir`` is set
  (``scheduler.py:1835,1862``; ``cache/factory.py``) — this helper forces it
  ``None`` unconditionally, *regardless of what the caller passes in*, so a
  rank process can never accidentally inherit the daemon's own cache
  directory. This is the "always-miss" stub: cross-rank cache-status is not
  implemented yet (SharedHotCacheBudget/SSD-dir cross-talk risk moot because
  the stack never turns on here), and this function is where a future
  implementation would plug in.
* **max_num_seqs matches the app-wide default**, not ``SchedulerConfig``'s
  own dataclass default of 256. Single-node engines get ``max_num_seqs=8``
  from ``omlx.config.SchedulerConfig`` via ``settings.py``; there is no
  equivalent settings path into a rank process (D8: no new spawn/lifecycle
  wiring), so this helper hard-codes the same value a rank process would get
  if that path existed. This is what makes the plan's queue-full test recipe
  (``max_num_seqs=8`` => waiting-queue cap of 32) apply to the cluster path
  with no new configuration surface.
"""

from __future__ import annotations

from typing import Any

# The app-wide single-node default (omlx/config.py's SchedulerConfig,
# max_num_seqs=8) -- kept in sync manually since rank processes have no path
# to the daemon's settings object (D8).
_DEFAULT_MAX_NUM_SEQS = 8


def rank_max_num_seqs() -> int:
    """The ``max_num_seqs`` rank 0's ``Scheduler`` actually runs with.

    ``rank_worker.py`` calls ``build_rank_scheduler_config()`` with no
    overrides, so this is exact rather than a guess -- which is what lets the
    head daemon reason about rank-0 admission limits without a round trip.
    """
    return _DEFAULT_MAX_NUM_SEQS


def rank_inflight_capacity() -> int:
    """How many requests rank 0 can hold at once: ``max_num_seqs`` in the
    running batch plus a full waiting queue.

    ``ClusterEngine`` gates on this in preflight. It cannot call
    ``Scheduler.preflight_queue_or_raise`` -- the scheduler is in another
    process -- so it reproduces the ceiling from the same
    ``waiting_queue_capacity`` definition instead of a second literal.
    """
    from omlx.scheduler import waiting_queue_capacity

    max_num_seqs = rank_max_num_seqs()
    return max_num_seqs + waiting_queue_capacity(max_num_seqs)


def build_rank_scheduler_config(**overrides: Any) -> Any:
    """Build a ``scheduler.SchedulerConfig`` for a rank-0 process.

    ``overrides`` are forwarded to the dataclass constructor (e.g.
    ``max_num_seqs`` for a test that wants a different cap), but the cache
    fields are always forced off afterward -- an override attempt on those
    fields does not survive, by design (the always-miss stub).
    """
    from omlx.scheduler import SchedulerConfig

    overrides.setdefault("max_num_seqs", _DEFAULT_MAX_NUM_SEQS)
    config = SchedulerConfig(**overrides)

    # Force cache-off regardless of what was passed in or what the daemon's
    # own settings say (D6(b)): no paged SSD cache, no shared hot-cache
    # budget, in a rank process, ever.
    config.paged_ssd_cache_dir = None
    config.hot_cache_only = False
    config.hot_cache_max_size = 0
    config.hot_cache_budget = None

    return config
