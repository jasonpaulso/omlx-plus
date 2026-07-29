# S3 measurements — continuous batching across a 2-node TP formation

Slice S3 acceptance evidence, taken on the live 2-Mac rig. This document is the
P3 deliverable: pass/fail is readable here without re-running anything.

Companion docs: `bringup.md` (rig setup), `s2-measurements.md` (S2 acceptance —
context only; **no S2 number is the S3 gate**, see "Why S2's throughput figure is
not the baseline" below).

Status: **IN PROGRESS** — protocol pinned, rig run pending.

---

## Rig

| Item | Value |
| --- | --- |
| Head | M4 Max, 128 GB |
| Worker | M3 Ultra, 96 GB (`Jasons-Mac-Studio.local`) |
| Link | Thunderbolt 5, static TB IP aliases (S0 recipe) |
| Commit under test | `4de9fd01` — both machines (E10 enforces commit-level handshake) |
| Model | `MiniMax-M2.7-3bit` (93.23 GB, 3-bit MoE) |
| Backends | ring, then jaccl — pinned explicitly, `negotiated_backend == requested` asserted per row |

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
| `benchmarks/cluster_spike/s3_compute.py` | applies the D7 formula to a raw dump; exit 0 = PASS |

Stdlib only, so it runs from any checkout without a venv.

---

## Acceptance matrix

Per backend, on MiniMax-M2.7-3bit.

| # | Row | ring | jaccl |
| --- | --- | --- | --- |
| 1 | `negotiated_backend == requested` | _pending_ | _pending_ |
| 2 | Join mid-generation (admit a request into a live batch) | _pending_ | _pending_ |
| 3 | Abort mid-batch (client disconnect; sibling unaffected) | _pending_ | _pending_ |
| 4 | Queue-full 503 (`max_num_seqs=8`, 41 concurrent, long `max_tokens`) | _pending_ | _pending_ |
| 5 | D7 throughput gate (aggregate ≥ baseline) | _pending_ | _pending_ |
| 6 | E4 decode-path tax under batch ≤ 10.268 ms/token | _pending_ | _pending_ |

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

_Pending the rig session._

## Repeats and anomalies

_None yet._
