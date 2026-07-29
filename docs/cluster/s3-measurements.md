# S3 measurements — continuous batching across a 2-node TP formation

Slice S3 acceptance evidence, taken on the live 2-Mac rig. This document is the
P3 deliverable: pass/fail is readable here without re-running anything.

Companion docs: `bringup.md` (rig setup), `s2-measurements.md` (S2 acceptance —
context only; **no S2 number is the S3 gate**, see "Why S2's throughput figure is
not the baseline" below).

Status: **rows 1, 2, 3, 5, 6 PASS on both backends. Row 4 fails as written on
both: the queue-full cap does fire at exactly 32, but it reaches the client as
an in-stream SSE error under HTTP 200, never as a 503.** That is the slice's
one real defect and its main carry-forward. E4's stop condition did not fire on
either backend.

Row 4 update (2026-07-29): the defect turned out to be shared with single-node,
not cluster-specific. A first fix landed at the preflight seam and **was
re-measured on the rig: it did not close row 4** — the gate counted requests
that had not entered the engine yet, so it never engaged on a cold burst. The
rejection *is* now logged head-side. See "Row 4 re-measure, ring" below.

Row 4 second fix (2026-07-29, later): preflight now **reserves** the slot it
checked for, on the cluster path only, so occupancy counts requests on their way
to rank 0 as well as those it already holds. **Rig-unverified — the rig has not
been touched since the re-measure above.** Local evidence is in "Reservation
fix" below; the rig re-run is what would flip this row to PASS.

---

## Rig

| Item | Value |
| --- | --- |
| Head | M5 Max, 128 GB |
| Worker | M3 Ultra, 96 GB (`Jasons-Mac-Studio.local`) |
| Link | Thunderbolt 5, static TB IP aliases (S0 recipe); data plane `10.0.2.1` ↔ `10.0.2.2`, **not** the `192.168.5.x` LAN |
| Commit under test | `decc52a9` — both machines, verified by `git rev-parse` on each |
| Model | `MiniMax-M2.7-3bit` (93.23 GB, 3-bit MoE) |
| Backends | ring, then jaccl — pinned explicitly, `negotiated_backend == requested` asserted per row |

> **The E10 handshake does not check the checkout.** `compare_versions()`
> (`omlx/cluster/versions.py:126`) compares `omlx.__version__` — a static literal,
> `0.5.3` — plus the `mlx` version and mlx-lm's version and commit id. The
> "commit-level" part is *mlx-lm's* commit, read from PEP 610 `direct_url.json`;
> the omlx repo's own HEAD is never exchanged. A worker sitting on an older omlx
> commit therefore joins cleanly and then runs mismatched TP protocol code. Both
> checkouts must be confirmed with `git rev-parse` before forming; the handshake
> will not do it for you. (Corrects an earlier program note claiming E10 would
> hard-reject a stale worker.)

---

## Pinned measurement protocol (D7)

Fixed before the rig session so the gate is decidable in advance and *can fail*.
No parameter was left to run-time choice.

**Prompt.** One fixed prompt, verbatim in
`benchmarks/cluster_spike/s3_prompt.txt`. Measured token count against the
MiniMax-M2.7-3bit tokenizer:

| Measure | Tokens |
| --- | --- |
| Raw text | 722 |
| Chat-templated (`add_generation_prompt=True`) | 761 |
| Server-reported `prompt_tokens` | _(filled from the run)_ |

The ≥512-token pin matters: it makes each per-request prefill gap under
serialization far exceed the ~137 ms discrimination threshold, so a FIFO run
scores decisively below baseline instead of ambiguously near it.

**`max_tokens` = 128. N = 4.**

**Formulas** (from SSE token-arrival timestamps; window boundaries are token
*arrival* times, never submit times, so inter-request prefill/queue gaps fall
inside the concurrent window):

```
baseline  = (completion_tokens - 1) / (last_arrival - first_arrival)     # single request
aggregate = (Σ completion_tokens over the 4) / (last arrival across ALL
                                                - first arrival across ALL)
```

**Pass ⇔ `aggregate ≥ baseline`, per backend.** Both runs happen in the same
session at S3 tip.

**No-retry rule.** The first completed measurement per backend is the recorded
result. A run may be repeated only if it errors out (infra failure), never
because the number is unfavourable; any repeat is logged in this document.

### Why the obvious formula was rejected

The intuitive aggregate — summing each request's individual decode rate — is
*maximized* by FIFO serialization: each request in isolation decodes at full
speed and the idle waiting time never enters any denominator. It cannot detect
the regression the gate exists for.

Verified empirically against synthetic fixtures through the real scoring script
before the rig session, so the gate is known to discriminate:

| Synthetic case | Chosen formula | Rejected sum-of-rates formula |
| --- | --- | --- |
| True continuous batching (4 overlapping streams @ 7 tok/s, baseline 10) | 27.99 tok/s → **PASS** (2.80×) | 28.00 tok/s → pass |
| FIFO serialization (4 sequential @ 10 tok/s + 5 s gaps) | 7.75 tok/s → **FAIL** (0.775×) | 40.00 tok/s → *false pass* (4.0×) |

The rejected formula rates the serialized system **4× better than baseline**.
That is the failure mode this protocol exists to catch.

### Why S2's throughput figure is not the baseline

`s2-measurements.md` reports ≈4.4 tok/s, but that number is prefill-inclusive
and disclaimed in that document itself, and no jaccl tok/s row exists there — so
no S2 scalar is comparable to an S3 decode-rate measurement. The baseline is
therefore re-measured in-session at S3 tip, per backend. S2 figures appear below
only as context, never as a gate.

---

## Harness

Capture and scoring are deliberately separate programs, so the gate arithmetic
can be recomputed from raw data without re-running the rig:

| File | Role |
| --- | --- |
| `benchmarks/cluster_spike/s3_prompt.txt` | the pinned prompt, verbatim |
| `benchmarks/cluster_spike/s3_measure.py` | drives SSE, dumps **raw per-token arrival timestamps** to JSON (`single` / `concurrent` / `abort` / `flood`) |
| `benchmarks/cluster_spike/s3_compute.py` | applies the D7 throughput formula (row 5) to a raw dump; exit 0 = PASS |
| `benchmarks/cluster_spike/s3_tax.py` | applies the E4 coordination-tax gate (row 6) to a snapshot pair + dump; exit 0 = PASS |

Stdlib only, so it runs from any checkout without a venv.

### Row 6: why the tax reading needs a snapshot pair

`engine_stats.last_tax` from `GET /v1/cluster/models/status` is **cumulative, not
per-request**. Rank 0's `LeaderModelProxy.tax_samples` is created once per model
load and never reset (`tp_batch.py:145`, appended `:261`), so `_tax_summary`
averages every broadcast since load — the baseline run and every prefill
included. Reading it straight after the concurrent run would not be "tax under
batch"; it would be a lifetime average.

Since `avg_ms × steps` is an exact sum, snapshotting either side of a run
recovers the windowed figure with no product-code change:

```
sum_ms        = avg_after × steps_after - avg_before × steps_before
steps         = steps_after - steps_before
per-token tax = sum_ms / completion_tokens_in_window
```

Per-*token*, not per-step: one broadcast serves the whole batch, so a per-step
figure is not comparable to a budget expressed per token.

**Gated on both windows — single (batch=1) and concurrent (batch=4) — and both
must pass.** Batch=1 is the strict case, since the tax amortizes over one token
rather than four. Gating only the concurrent window would be picking the lenient
number after the fact, which is precisely what the no-retry rule exists to
prevent.

Falsifiability checked against synthetic fixtures through the real script before
the rig session:

| Synthetic case | Windowed per-token tax | Verdict |
| --- | --- | --- |
| Healthy window (1024 ms over 512 tokens) | 2.000 ms/token | **PASS** (19.5% of budget) |
| Over-budget window (6144 ms over 512 tokens) | 12.000 ms/token | **STOP** (116.9%) |
| Bad window hidden behind 10 000 cheap prior steps | 12.000 ms/token | **STOP** — cumulative avg reads a healthy 1.0016 ms/step and would have passed |

The third row is the reason for the snapshot pair: the naive cumulative read
hides a window that is 117 % of budget.

---

## Run order (pinned)

Per backend, in this order. Ordering is part of the protocol, not a convenience:

1. Form → assert `negotiated_backend == requested` (**row 1**)
2. `single` → D7 baseline + batch=1 tax window
3. `concurrent` → D7 aggregate + batch=4 tax window (**rows 5, 6**)
4. `abort` (**row 3**)
5. `flood` (**row 4**) — **last**

`flood` runs last because aborts are deferred (`abort_request()` only records the
ID; it applies at the next `step()`) and the burst parks 32 requests in `waiting`
with `max_tokens=4096`. A client disconnect does not guarantee the scheduler has
drained, and that residue must not land inside a no-retry timing window.
`num_active_requests` is checked back to zero between runs.

**Row 2 (join mid-generation) is read out of the `concurrent` dump**, not a
separate run: with the one-admission-per-step throttle, requests 2–4 necessarily
join a batch that is already decoding, and the interleaved arrival timestamps are
the evidence. Fewer runs, less residue.

Raw dumps and status snapshots are written to a scratchpad outside the repo —
command lines carry `--api-key`, and nothing captured is meant for git except the
numbers transcribed into this document.

---

## Acceptance matrix

Per backend, on MiniMax-M2.7-3bit.

| # | Row | ring | jaccl |
| --- | --- | --- | --- |
| 1 | `negotiated_backend == requested` | **PASS** | **PASS** |
| 2 | Join mid-generation (admit a request into a live batch) | **PASS** | **PASS** |
| 3 | Abort mid-batch (client disconnect; sibling unaffected) | **PASS** | **PASS** |
| 4 | Queue-full 503 (`max_num_seqs=8`, 41 concurrent, long `max_tokens`) | **FAIL (letter)** | **FAIL (letter)** |
| 5 | D7 throughput gate (aggregate ≥ baseline) | **PASS** 1.893× | **PASS** 1.010× |
| 6 | E4 decode-path tax under batch ≤ 10.268 ms/token | **PASS** 0.995 / 0.330 | **PASS** 0.471 / 0.283 |

Row 4 recipe is the real hard-floored cap, not a knob: `max_waiting =
max(max_num_seqs * 4, 32)` with a hard floor of 32 and no config override
(`scheduler.py:6766`). Because admission pops from `waiting`
(`scheduler.py:8391`), up to `max_num_seqs` submissions leave the queue, so the
burst is `max(max_num_seqs*4, 32) + max_num_seqs + 1 = 41` at `max_num_seqs=8`.
The assertion is index-free: **at least one submission receives 503 while
earlier ones continue streaming.**

Row 6 budget: 10 % of the S0-measured 102.682 ms/token single-token latency =
**10.268 ms/token**. E4's stop condition is live — if the measured tax exceeds
the budget, work stops and the measurements go to the user; this slice cannot
waive it. Decode tax is measured per *step* (one broadcast serves the whole
batch), so per-token tax is expected to fall as batch size grows.

---

## Results

### ring — 2026-07-29, commit `cec04f05`, formation `world_size 2`

**The ring column spans two formations, not one.** Rows 1, 2, 3, 5 and 6 come
from the first ring session below. Row 4 comes from a second, freshly formed
ring session run after jaccl, once the capture defect was fixed. Rows 5 and 6
were deliberately not re-run in that second session — the numbers below stand,
and re-running them for tidiness is exactly what the no-retry rule forbids.

Formation: `active_model MiniMax-M2.7-3bit`, `loaded: true`, `world_size: 2`,
`negotiated_backend: "ring"` — **row 1 PASS**. Rank 1 loaded its shard on the
Studio (`50 340 050 944` shard param bytes). `prompt_tokens: 761` on every
request, matching the pinned templated count exactly.

**Row 5 — D7 throughput.**

| Quantity | Value |
| --- | --- |
| baseline (single) | **19.4029 tok/s** — (128−1) tokens / 6.5454 s |
| aggregate (4 concurrent) | **36.7373 tok/s** — 512 tokens / 13.9368 s window |
| ratio | **1.893×** |
| verdict | **PASS** (`aggregate ≥ baseline`) |

Token counts come from server-reported `usage`, not arrival counts. The rejected
sum-of-rates formula would have reported 44.09 tok/s on the same data — recorded
only to show the chosen formula is the discriminating one.

**Row 6 — E4 coordination tax.** Windowed via the snapshot pair, both gated:

| Window | Steps | Tokens | tok/step | ms/step | **ms/token** | % of budget |
| --- | --- | --- | --- | --- | --- | --- |
| single (batch=1) | 145 | 128 | 0.88 | 0.8780 | **0.9946** | 9.69 % |
| concurrent (batch=4) | 139 | 512 | 3.68 | 1.2162 | **0.3302** | 3.22 % |

Both **PASS** against the 10.268 ms/token budget; the E4 stop condition did not
fire. Per-token tax falls as the batch grows, as predicted — one broadcast
serves the whole batch. The 3.68 tok/step in the concurrent window is
independent confirmation that real batching occurred, not serialization.

**Row 2 — join mid-generation. PASS.** Read out of the concurrent dump: the four
requests reached their first token at 1.30 / 2.54 / 3.79 / 5.77 s, so requests
2–4 were admitted into a batch that was already decoding (one admission per
step). All four then completed 128 tokens.

**Row 3 — abort mid-batch. PASS.** The victim disconnected after 8 tokens; the
sibling completed its full 128 (`finish_reason: length`) unaffected, and
`num_active_requests` returned to 0 afterwards.

**Row 4 — queue-full. Not established on ring; see the jaccl result below,
which settles the mechanism.** All 41 submissions were accepted across three
burst attempts. At the time this was read as "the cap was never reached",
because a `grep` for `queue_full` / `max_depth` / `current_depth` over the
head's `server.log` and stdout returned nothing.

**That inference was wrong, and the grep was never discriminating.** The jaccl
run later produced a real queue-full rejection, and the same grep *still*
returns zero — the head does not log these frames at all. The ring bursts also
ran before the capture was fixed to record in-stream `error` payloads, so a
rejection would have been silently dropped (a rejected request would look
exactly like one that merely never started: status 200, no tokens, no error).

Ring row 4 is therefore **unknown from the runs above**, not "never reached".

**Ring row 4, re-measured.** After the jaccl session the rig was torn down,
switched back to `backend=ring`, re-formed with fresh daemons and a scrubbed
ghost, and the burst was re-run with the fixed capture. Result identical to
jaccl: request `f22` received

```
{"message": "Scheduler waiting queue full: 32 >= 32", "type": "server_error"}
```

again with HTTP status 200, 40 of 41 streaming, no 503. **Ring row 4 = FAIL on
the letter, same mechanism as jaccl.** The defect is backend-independent, as a
route-layer problem should be.

### jaccl — 2026-07-29, commit `1250a804`, fresh daemons

Fresh daemons on both nodes (PD-leak: one jaccl session per process), ghost
member scrubbed, re-joined. `negotiated_backend: "jaccl"`, `loaded: true`,
`world_size: 2`, no alarms — **row 1 PASS**. RDMA enabled on both nodes.

**Row 5 — D7 throughput.**

| Quantity | Value |
| --- | --- |
| baseline (single) | **23.9013 tok/s** — (128−1) / 5.3135 s |
| aggregate (4 concurrent) | **24.1512 tok/s** — 512 tokens / 21.1998 s window |
| ratio | **1.010×** |
| verdict | **PASS**, but with almost no margin |

Recorded as measured under the no-retry rule. The contrast with ring is the
interesting part and is not smoothed over: jaccl's *single-stream* baseline is
faster (23.90 vs 19.40 tok/s), but its batching gain is ~nil (1.010× vs ring's
1.893×). Per-request decode under load ran 6.2–10.0 tok/s against ring's
9.4–13.5, and TTFT under load was markedly worse (4.0–12.4 s vs 1.3–5.8 s). So
on this rig jaccl buys single-stream latency and gives back concurrency
throughput. One observation, one run — not a trend, and worth a dedicated
comparison before anyone designs around it. The TTFT gap is the part most likely
to matter downstream: it is user-visible latency, not a throughput ratio.

**Row 6 — E4 coordination tax.** Both windows **PASS**; stop condition did not
fire.

| Window | Steps | Tokens | tok/step | ms/step | **ms/token** | % of budget |
| --- | --- | --- | --- | --- | --- | --- |
| single (batch=1) | 130 | 128 | 0.98 | 0.4633 | **0.4705** | 4.58 % |
| concurrent (batch=4) | 139 | 512 | 3.68 | 1.0436 | **0.2833** | 2.76 % |

Lower than ring at both batch sizes (0.9946 / 0.3302), consistent with RDMA and
with S2's ring-vs-jaccl ordering.

**Row 2 — PASS.** TTFTs 4.01 / 6.72 / 9.41 / 12.40 s: requests admitted into an
already-decoding batch.

**Row 3 — PASS.** Victim disconnected at 8 tokens; sibling completed 128
(`finish_reason: length`); `num_active_requests` back to 0.

**Row 4 — queue-full. The cap fires, but not as an HTTP 503.**

With the fixed capture, request `f17` received a real rejection:

```
{"message": "Scheduler waiting queue full: 32 >= 32", "type": "server_error"}
```

`32 >= 32` is exactly the configured cap — `build_rank_scheduler_config` pins
`max_num_seqs=8`, giving `max(8*4, 32) = 32` (`scheduler.py:6766`). So the
backpressure mechanism works end to end on the live cluster path: rank 0 raised
`SchedulerQueueFullError`, replied `code="queue_full"` (`rank_worker.py:430`),
and the daemon surfaced it to the client.

**But it arrived with HTTP status 200, as an in-stream SSE error event, not as
a 503.** The row as written ("at least one submission receives 503") is
therefore **FAIL on the letter and PASS on the substance**, and the gap is
structural rather than incidental: the cluster route returns a
`StreamingResponse` before the generate command reaches rank 0, so by the time
the rejection comes back the response headers are already sent and
`server.py:708`'s `SchedulerQueueFullError` → 503 handler can no longer fire.

Consequences worth carrying forward, since a client cannot treat this as a
normal 503:

- The rejection is invisible to any client that only inspects status codes, and
  to anything that retries on 503. It is also **not logged on the head at all** —
  the grep above finds nothing even for this confirmed rejection, so an operator
  has no server-side record that backpressure occurred.
- 40 of 41 requests still streamed at least one token although only 8 can run
  concurrently, so slots do churn during the burst; the queue reaching its cap
  at all is what makes the rejection meaningful.

Not fixed here — it is a route-layer change (peek the first frame before
committing to a `StreamingResponse`, or surface queue depth pre-admission), and
S3's scope is the batching path. Filed as the slice's main carry-forward.

**Follow-up (2026-07-29): the diagnosis above is right about the mechanism and
wrong about the blame.** Chasing the fix showed this is not a cluster defect at
all — single-node `stream: true` degrades identically, and always has.
`BatchedEngine.preflight_chat` only ever ran the *prefill memory* check
(`scheduler.py`'s `preflight_or_raise`); nothing anywhere checked queue depth
before the route committed. `Scheduler.add_request` raises from inside the
response generator (`batched.py:817`), and starlette sends
`http.response.start` before it iterates — the same 200-commit this row hit.
The registered 503 handler could therefore only ever answer `stream: false`.
The cluster path was matching single-node semantics exactly, which is what
spec-S3 asked of it.

The fix takes the second of the two options named above — surface queue depth
pre-admission — at the preflight seam all four LLM routes already await:
`Scheduler.preflight_queue_or_raise` for in-process engines, and a head-side
gate on rank-0 in-flight count for `ClusterEngine`, which cannot reach that
scheduler. Both derive their cap from one `waiting_queue_capacity` definition.
No `server.py` change was needed. The head-side log line the second bullet
above asks for now exists on the backstop path.

**Row 4 remains rig-unverified.** The fix is covered by unit and two-rank
integration tests, but this row was measured on the live 2-Mac pair on both
backends and only a rig re-run can close it. Re-measure with the same `flood`
recipe (41 concurrent, `max_tokens` large enough that nothing completes) and
assert an actual HTTP 503 — do not mark it PASS from the test suite.

### Row 4 re-measure protocol — pinned before the run

Written before any daemon was started, so the rules cannot be chosen to fit a
number. Same shape as the D7 protocol.

**Gate.** `benchmarks/cluster_spike/s3_row4.py`, run over the `flood` dump.
PASS ⇔ **at least one HTTP 503 AND at least one request streaming ≥1 token**.
Both clauses are load-bearing: a preflight gate that rejected the whole burst
would satisfy "a 503 happened" while being strictly worse than the defect it
replaces, so that shape must fail. In-stream `queue_full` errors are counted
and reported but are **not** a failure — the preflight→submit race is open by
design, so the backstop staying reachable is expected. Proven falsifiable
before rig time with `s3_row4.py --selftest`: five synthetic shapes, including
the pre-fix one (40 streaming + 1 in-stream error, no 503) which must FAIL and
the over-aggressive one (all 503) which must also FAIL.

**No-retry.** A flood that produces no 503 is a **result**, not an infra
error. The first completed run per backend is the number. A re-run is
permitted only when the *formation* failed — model load error, rank death,
`negotiated_backend != requested` — and any re-run is logged with its reason.
Row 4 is a status-code assertion, not a timing measurement, so machine-noise
arguments do not apply to it.

**Preconditions, checked by hand each session.** Both checkouts at the same
SHA compared as strings (E10 does not check the omlx checkout — a stale worker
joins cleanly and runs mismatched TP code); ghost members scrubbed before
forming; `negotiated_backend == requested` asserted per backend; jaccl gets
freshly started daemons (PD-leak is one session per process).

### Row 4 re-measure, ring, 2026-07-29 — still FAIL

Run at `b3d39ad6`, both checkouts verified identical by `git rev-parse`, both
venvs confirmed importing the new symbols. Formation: MiniMax-M2.7-3bit across
the pair, `negotiated_backend: ring` == requested, smoke generation served
(24 tokens, 6.55 s) before the burst. 41 concurrent, `max_tokens=4096`.

```
status codes         : {200: 41}
HTTP 503 rejections  : 0
streamed >=1 token   : 40
in-stream queue_full : 1
ROW 4 GATE : FAIL
```

Identical to the pre-fix shape. **The fix does not close row 4, and the reason
is structural in the fix itself, not in the rig.**

`ClusterEngine._preflight_queue` gates on `len(self._pending)`, but `_pending`
is populated at `cluster/engine.py:350` — inside `stream_generate`, which the
route only iterates *after* it has committed to the `StreamingResponse`. On a
cold simultaneous burst all 41 requests run preflight before any generator
starts, so the counter the gate reads is still empty and the gate never fires.
Confirmed from the head log: zero preflight 503s, exactly one backstop warning.

The same reasoning applies to the single-node path: `Scheduler.waiting` is
filled by `add_request`, which is likewise inside the generator. So
`preflight_queue_or_raise` only fires when the queue is **already** at cap when
a later request preflights — sustained backpressure, which is the production
shape — and never on a cold burst, which is precisely row 4's recipe. The unit
tests pass because they pre-fill the queue before calling preflight; that is a
real behaviour, just not the one this row measures.

What the fix *did* close: the rejection is now logged head-side, which was the
second half of row 4's complaint ("not logged on the head at all"). The
warning fired exactly once, naming the request id and `32/32`.

Closing this row needs the head to reserve the slot at preflight time rather
than count generators that have not started — the reservation approach
consciously rejected when the fix was designed, on the reasoning that
over-admitting by a few was immaterial. That reasoning was wrong for a cold
burst: the gate does not over-admit by a few, it does not engage at all.
Reservation needs a preflight→submit identity, and the route passes
`request_id` from an `x-request-id` header that is usually absent, so it is a
design change rather than a patch. Left for a decision.

**jaccl not run — a decision, not an omission.** The defect is in the head's
route/engine layer and cannot vary with the collective backend; ring is
decisive and a jaccl session would cost a fresh-daemon cycle to reproduce a
foreordained result. If the reservation lands, re-run both.

### Reservation fix — local evidence, rig re-run still owed

The design change the section above asked for, scoped to the cluster path.

**It needs no preflight→submit identity.** That was the blocker on paper, and it
dissolves once reservations are counted rather than matched: slots are
interchangeable, so preflight appends one and `stream_generate` drops one when
it takes its place in `_pending`. Occupancy is `len(_pending) + reservations`
against the same `rank_inflight_capacity()` ceiling (40 = 8 running + 32
waiting), so nothing is double-counted at the handover — the release happens in
the same synchronous frame as the insert.

| Path | Behaviour |
| --- | --- |
| Cold burst of 41 | 40 reserve, the 41st is refused **at preflight** → 503 |
| Admitted request | releases its slot as it enters `_pending` |
| Fails before `_pending` (empty prompt, non-goal, dead pipe) | releases in `finally` |
| Never reaches `stream_generate` (client vanished after preflight) | expires after `_RESERVATION_TTL_S` (30 s) |
| `stream_generate` with no preceding preflight (internal caller, direct test) | releases nothing when there is nothing to release |

The failure mode is chosen deliberately: an extra release relaxes the gate
toward the previous behaviour, whereas a missed release would hold a slot no
request owns. Reservations never outlive the TTL, so no leak is permanent.

**Not applied to single-node.** `Scheduler.preflight_queue_or_raise` compares
against the *waiting* cap only, while the cluster gate compares against a total
(running + waiting). Adding reservations there requires switching it to the
total form — a behaviour change to the default configuration, with its own
tests to move. Row 4 is a cluster-path measurement; single-node's identical
cold-burst hole is a separate slice, and it is still open.

Local gates at this commit:

| Gate | Result |
| --- | --- |
| `pytest -m cluster` | **22 passed** (3:40) — 21 baseline + the new cold-burst test |
| New tests vs stashed source | **6 of 6 fail without the fix** — see the caveat below |
| Default unit gate | **7534 passed, 2 failed** — the known GLM numerical pair, zero delta |
| black / ruff / mypy | zero delta (mypy 665 errors / 81 files both sides) |
| `s3_row4.py --selftest` | the predicted `{200: 40, 503: 1}` shape scores **PASS**; all 5 shapes behave as specified |

**Don't over-credit the 6.** Three of the five unit tests touch `_reserved_slots`
or `_RESERVATION_TTL_S`, which do not exist pre-fix, so they fail on
`AttributeError` rather than on behaviour. The weight is carried by
`test_cold_burst_is_refused_before_any_generator_runs` and
`test_reservations_and_inflight_share_one_ceiling` — public surface only, failing
because no rejection happens — and by the two-rank test.

The two-rank test (`test_cold_burst_is_refused_at_preflight_not_in_stream`)
deliberately separates preflight from submission into two phases. That is the
whole point, and it was checked rather than assumed: a single-phase variant
(preflight and submit interleaved per task) was run against the **pre-fix**
engine and reported `{streaming: 8, pending: 32, preflight_503: 1}` — it passes
without the fix, because each request's `_pending` entry lands before the next
one preflights. That is the warm shape the previous fix's pre-filled unit tests
also measured, and it is why the rig disagreed with them. The two-phase test
asserts the same two clauses as the row-4 gate.

**Still rig-unverified.** The rig was not touched in this session. Closing row 4
means re-running `flood` on ring and jaccl at this commit, under the pinned
protocol above (no-retry included).

Rig left clean: model unloaded, member scrubbed, both daemons stopped,
8910/8911 free, `backend=ring` and roles intact on both nodes, daily-driver
`:8899` (PID 87617) and Studio `:8888`/`:8889` untouched throughout.

## Repeats and anomalies

Every repeat below is logged per the no-retry rule. **No repeat was triggered by
an unfavourable number**: in each case the run produced no computable value for
its gate, and each fix was committed before the re-run, with no result in hand.
The D7 and E4 scalars above come from the first run that could compute them, and
were never re-run.

1. **`single`, run 1 — instrumentation defect, discarded.** MiniMax-M2.7 is a
   reasoning model: it streams tokens into `delta.reasoning_content` and emits
   `delta.content` only at the end. The harness counted `content` alone, so a
   128-token generation recorded **one** arrival and the baseline span was zero
   — the D7 gate was not computable. Fixed in `cec04f05` (count both), re-run.
2. **`flood`, attempt 1 — released slots.** Disconnecting after 3 tokens freed
   each slot; with one admission per step the queue drained as fast as it
   filled. All 41 returned 200 over 122 s (submit spread 2 ms, first-token
   spread 121 s). Fixed in `145b3469` (hold connections open).
3. **`flood`, attempt 2 — unbounded, killed at 10 min.** The per-thread 180 s
   hold compounded because admitted requests generate thousands of tokens and
   the queued ones never report. Replaced with one global deadline in
   `9737dfc6`. My ssh session to the worker dropped during this attempt; the
   worker daemon and rank process both survived and the formation stayed
   healthy (member `active`, seq 231, 0 alarms) — only the pipe died.
4. **`flood`, attempt 3 — inconclusive, see row 4.** Also revealed that the
   capture silently dropped in-stream `error` payloads (no `choices` key), so a
   late-arriving `queue_full` would have been invisible. Fixed before jaccl so
   the same burst cannot produce a false negative there.
5. **`flood` on ring, re-measured after jaccl** (row 4 only). The three ring
   bursts above predate the error-event fix and could not observe a rejection;
   the jaccl run proved rejections are real and unlogged, so ring was re-formed
   from scratch and re-run to get symmetric evidence. This is a repeat for a
   defective instrument, not for an unfavourable number — and it *added* a
   failing row rather than removing one. Ring's rows 5 and 6 were **not**
   re-run and remain the originals.

### Rig facts corrected during the session

- **Head is an M5 Max**, not an M4 Max as previously recorded.
- **The E10 handshake does not check the checkout** — see the note under Rig.
  Both SHAs were confirmed by hand with `git rev-parse` instead.
- **`omlx` on `PATH` is a uv tool install**, not the repo. Starting it wrote its
  own schema over `~/omlx-cluster-dev/settings.json` and reset four cluster
  fields (`role` → `off`, `data_plane_subnet`, `data_plane_address`,
  `rdma_device` → empty), which silently disables the entire cluster control
  plane — every `/v1/cluster/*` route 404s. Restored by hand and re-verified
  through the repo parser; the rig must be driven with `.venv/bin/omlx`.
- One worker join attempt failed every heartbeat with "All connection attempts
  failed" while `curl` and a hand-rolled `ClusterClient` from the same host
  succeeded. Restarting the worker attached to a live ssh session fixed it. The
  cause is **unexplained** — an attached-vs-orphaned theory was ruled out when a
  later orphaned worker heartbeated normally.
