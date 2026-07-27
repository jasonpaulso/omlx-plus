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

## Class E: cache lookups — one function, one field

49 grep hits, and they collapse to a single entry point:
**`_prepare_prefix_cache_for_request()` (line 6349)**, called once per
admission at line 8111. Everything else in the class is either its helpers, its
logging, or teardown.

The function's only divergent output is **`request.remaining_tokens`**. That
field is what line 8117 turns into the prefill's token count, so it decides the
*shape of a forward pass*. Two ranks that disagree about it do not produce
slightly different answers; they issue different collectives and hang.

Everything it writes is per-request and derived from the same lookup:

| field | set at | consequence of divergence |
|---|---|---|
| `remaining_tokens` | 6391, 6419, 6429, 6448, 6470, 6476, 6479 | different prefill length — **fatal** |
| `prompt_cache` | 6386 | different KV state entering the pass |
| `block_table`, `shared_prefix_blocks` | 6387, 6389 | paged bookkeeping only |
| `cached_tokens` | 6388 | reported usage only |

Three separate ways to reach a different `remaining_tokens` on two machines,
all of them ordinary operation rather than edge cases:

1. **The block store is per-node.** `fetch_cache` (6356) reads this machine's
   paged/SSD cache. Rank 0 served the previous turn and has the blocks; rank 1
   was idle and does not.
2. **Reconstruction can fail locally** (6466) and silently downgrade a hit to a
   miss on one rank — `reconstruct_cache` also truncates `block_table`
   in place on partial validity (6375, 6453).
3. **`_bypass_hot_cache_under_pressure()` (6364)** is Class B wearing a Class E
   hat: it reads `_current_usage_bytes()`, so the *memory* divergence decides
   whether blocks are preloaded, and the non-preloaded path can reconstruct a
   different number of tokens.

`_find_pending_store_for_lookup` (6233) waits on this node's in-flight store
jobs before looking up, which is a fourth path to the same field and is also
wall-clock dependent.

**How it must be fixed.** Not by broadcasting the reconstructed cache — that is
gigabytes per turn. The seam is the *decision*: rank 0 performs the lookup,
agrees `len(remaining_tokens)` through the collective, and any rank whose local
lookup produced a different number discards its cache and prefills the agreed
count from scratch. Correct, cheap (one int per admission), and it degrades to
"nobody uses the cache" only when the ranks genuinely disagree.

**Until that exists, cluster mode runs with no prefix cache at all** — see
`omlx/cluster/engine.py`, which starts every request from a fresh
`make_prompt_cache`. That is why the current serving path is safe without any
of this.

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

## Status

The audit is complete: all five classes have been worked through. Two of them
need code before the scheduler can run inside a rank worker — **memory-gated
admission** (Class B, six gates) and **the prefix-cache lookup** (Class E, one
field). The other three are already safe.

Neither is on the critical path for what ships today. The current cluster
serves one request at a time from a fresh cache
(`omlx/cluster/worker.py`), which is why it needs no part of this. This
document is the specification for the batched successor.
