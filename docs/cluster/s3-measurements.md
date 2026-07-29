# S3 measurements — continuous batching across a 2-node TP formation

Slice S3 acceptance evidence, taken on the live 2-Mac rig. This document is the
P3 deliverable: pass/fail is readable here without re-running anything.

Companion docs: `bringup.md` (rig setup), `s2-measurements.md` (S2 acceptance —
context only; **no S2 number is the S3 gate**, see "Why S2's throughput figure is
not the baseline" below).

Status: **ring 5 of 6 rows PASS; row 4 NOT DEMONSTRATED. jaccl pending.**
Five passing rows are five passing rows — this is not a ring pass as a whole.
See row 4 for what was and was not established.

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
| 1 | `negotiated_backend == requested` | **PASS** | _pending_ |
| 2 | Join mid-generation (admit a request into a live batch) | **PASS** | _pending_ |
| 3 | Abort mid-batch (client disconnect; sibling unaffected) | **PASS** | _pending_ |
| 4 | Queue-full 503 (`max_num_seqs=8`, 41 concurrent, long `max_tokens`) | **NOT DEMONSTRATED** | _pending_ |
| 5 | D7 throughput gate (aggregate ≥ baseline) | **PASS** 1.893× | _pending_ |
| 6 | E4 decode-path tax under batch ≤ 10.268 ms/token | **PASS** 0.995 / 0.330 | _pending_ |

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

**Row 4 — queue-full. NOT DEMONSTRATED.** Not a failure: the condition was never
reached, so the assertion was never put to the system. Across three burst
attempts all 41 submissions were accepted, and **no `queue_full` frame was ever
emitted** — `grep` over the head's `server.log` and stdout for
`queue_full` / `max_depth` / `current_depth` returns nothing for the whole
session. (The single `503` in that log is a token count, not a status code.)

The mechanism is wired correctly — rank 0 raises `SchedulerQueueFullError` and
replies `code="queue_full"` (`rank_worker.py:430`), `build_rank_scheduler_config`
pins `max_num_seqs=8` giving a waiting cap of 32 (`scheduler.py:6766`) — and it
is covered by the passing unit test. What is missing is a client-side way to
*hold* 41 requests resident on the live path: rank 0 admits one request per step,
and requests terminate on their own (one finished naturally at 503 tokens,
`finish_reason=stop`, despite `max_tokens=4096`), so slots free at least as fast
as the queue fills. 40 of 41 streamed at least one token where only 8 can run
concurrently, which is direct evidence of that churn.

Establishing this row needs a way to pin requests resident — a prompt that
reliably generates to the `max_tokens` ceiling, or an admission pause — and that
is a harness question, not an S3 code question. Left open rather than reported
either way.

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
