# Scheduler divergence audit

Prerequisite for making `omlx/scheduler.py` (10,930 lines) rank-aware.

Under lockstep, every rank runs the same scheduling decisions on different
machines. The rule is that **no rank may branch on state only it can see**. This
audit enumerates every place `scheduler.py` does.

The result is much narrower than the file size suggests: **the divergence
surface is essentially one thing — memory-gated admission.**

## Class B: machine-local memory — THE problem

`_current_usage_bytes()` (line 3738) is the source. It reads
`mx.get_active_memory()` (3747) and `get_phys_footprint()` (3754), both of
which describe *this machine only*. An M5 Max with 128 GB and an M3 Ultra with
96 GB return different numbers for identical work.

It feeds six decision gates:

| line | gate |
|---|---|
| 3227 | `elif current > self._memory_limit_bytes` |
| 4265 | `elif current > self._memory_limit_bytes` |
| 7650 | `if estimated > hard_limit` — prefill admission |
| 7781 | `if current + peak > self._memory_hard_limit_bytes` |
| 7858 | `if peak and current + peak > self._memory_hard_limit_bytes` |
| 8067 | `if current > self._memory_limit_bytes` |

Concretely: two ranks can reach **different admission decisions for the same
request**, at which point one steps the batch and the other does not, and the
collective deadlocks or produces garbage.

`_memory_hard_limit_bytes` is itself derived per-machine (set by the process
memory enforcer), so both sides of every comparison diverge.

~18 further call sites of `_current_usage_bytes()` are advisory (logging,
telemetry, throttle sampling) and matter only if they later feed a gate.

## Class A: wall-clock — one real site

11 uses of `time.monotonic()` / `perf_counter()`. Almost all are bookkeeping:
`generation_started_at` and `last_activity_at` are **written and never read**
for any decision in this file (verified — the only non-assignment reference is
another assignment at 8694). Divergent timestamps across ranks are harmless.

The exception is **line 3813**: `_memory_admission_blocked_since` plus a
timeout, which gates admission. It belongs to the memory-admission family
above rather than being a separate problem.

## Class C: RNG — already safe

Two real sites (4326, 8576), both `mx.random.seed(request.sampling_params.seed)`
— seeded from the *request*, not from machine state, so every rank derives the
same stream provided the request is broadcast. `mlx_adapter.seed_everyone()`
covers the global seed.

One `uuid4()` at 7779 generates a preflight diagnostic request id. Rank-local
and never compared, but it should not be allowed to leak into anything ordered.

## Class D: iteration order — safe

Python dicts preserve insertion order, and the hot paths iterate dicts
(`self.running.values()`, `self.requests.items()`), not sets. Order is
identical across ranks as long as insertion order is — which follows from
broadcasting admission. No bare `set` iteration was found in a decision path.

## Class E: cache lookups — not yet audited

49 grep hits on `prefix_cache` / `cache_hit` / `lookup` / `reuse`. A local
prefix-cache hit is a textbook divergence source: rank 0 hits, rank 1 misses,
and they disagree about how many tokens to prefill. This class was **not**
worked through and is the remaining piece of the audit.

## Implication for the fix

The design note "route every cache decision through rank 0" generalises: the
scheduler needs one seam where *memory-derived admission* is decided on rank 0
and broadcast, rather than each rank computing it locally. That is a much
smaller change than making the whole scheduler rank-aware, because the other
divergence classes are already benign.

Suggested shape: make `_current_usage_bytes()` and the hard/soft limits
cluster-aware — on a follower, return the leader's broadcast value instead of
the local reading — so the six gates keep their existing logic and
automatically agree. Verify by asserting all ranks reach the same admission
verdict for a fixed request sequence.

## Method

Grep by divergence class, then read every hit in context. Counts:
timing 11, local-memory 40, RNG 5, iteration 14, cache 49. Reading the hits
that actually gate a decision is a few hundred lines, not the whole file.
