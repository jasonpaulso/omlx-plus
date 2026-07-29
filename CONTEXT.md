# Session Context — 2026-07-29 (late) — ROW 4 CLOSED on the rig, both backends

**Status:** the second row-4 fix landed and was measured on the live 2-Mac rig. Preflight now
*reserves* the slot it checked for on the cluster path, so the gate engages on a cold burst
instead of reading a counter that is still empty. **Ring and jaccl both `{200: 40, 503: 1}`,
gate PASS, one flood each, no re-runs.** All six S3 acceptance rows now PASS on both backends.

Branch tip `df100432` (code + tests) plus this doc commit. **Rig left clean** — daemons down,
8910/8911 free, `backend=ring` restored and roles intact on both nodes, member scrubbed, model
unloaded; daily-driver `:8899` (PID 87617) and Studio `:8888`/`:8889` (947/926) untouched.

## What was wrong, and what it needed

The first fix (`87fb0e2e`) gated on `len(self._pending)`. `_pending` is filled inside
`stream_generate`, which starlette only iterates *after* the route commits to the
`StreamingResponse`. On a cold burst all 41 requests preflight before any generator body runs,
so the counter is empty and the gate does nothing. Measured on the rig at `b3d39ad6`:
`{200: 41}`, zero 503s — identical to pre-fix.

The recorded blocker was that reservation "needs a preflight→submit identity, and the route's
`request_id` comes from an `x-request-id` header that is usually absent." **That blocker was
not real.** It only exists if reservations are *matched* to requests. Counted instead, slots
are interchangeable: preflight appends one, `stream_generate` drops one when it takes its place
in `_pending`. No identity anywhere.

## What landed (`df100432`)

`omlx/cluster/engine.py` only. Occupancy = `len(_pending) + live reservations`, against the
unchanged `rank_inflight_capacity()` ceiling (40 = 8 running + 32 waiting).

| Piece | Note |
| --- | --- |
| `self._reserved: deque[float]` | monotonic expiry deadlines, oldest first |
| `_reserved_slots()` | sweeps expired, then counts |
| `_release_reservation()` | pops the oldest; no-op when empty |
| `_preflight_queue()` | checks occupancy, then claims — same synchronous frame, nothing awaits between |
| `stream_generate` | `try/finally` from entry to the `_pending` insert, so every pre-admission exit returns the slot |
| `_RESERVATION_TTL_S = 30.0` | covers the one path no explicit release can: a request that never reaches `stream_generate` at all |

Deliberate choices worth not re-litigating:

- **Counting, not identity.** An extra release (a `stream_generate` that never preflighted)
  relaxes the gate toward the old behaviour; a missed release would hold a slot no request
  owns. The cheaper failure mode was chosen on purpose.
- **No lock.** Preflight and the `_pending` insert both run on the event loop and neither awaits
  between check and claim, so check-then-add is already atomic.
- **Single-node deliberately untouched.** It has the same cold-burst hole, but
  `Scheduler.preflight_queue_or_raise` compares against the *waiting* cap while the cluster gate
  compares against a total (running + waiting). Adding reservations there means switching it to
  the total form — a behaviour change to the default configuration that moves the tests
  `87fb0e2e` added. Row 4 is a cluster measurement; that is a separate slice, **still open**.

## Verification

| Gate | Result |
| --- | --- |
| **Rig, ring** | `{200: 40, 503: 1}`, 40 streaming, 0 in-stream, **PASS** |
| **Rig, jaccl** | `{200: 40, 503: 1}`, 40 streaming, 0 in-stream, **PASS** |
| `pytest -m cluster` | 22 passed (3:49) — 21 baseline + the new two-rank cold-burst test |
| Falsifiability | 6 of 6 new tests fail against stashed source — but see below |
| Default unit gate | 7534 passed, 2 failed — the known GLM numerical pair, zero delta |
| black / ruff / mypy | zero delta (mypy 665 errors / 81 files both sides) |
| `s3_row4.py --selftest` | predicted `{200: 40, 503: 1}` scores PASS; all 5 shapes as specified |

Zero backstop warnings on both rig runs: the 503 came from preflight, before the response
committed, and none of the 40 admitted requests was refused in-stream.

**Don't over-credit the 6.** Three of the five unit tests reference `_reserved_slots` /
`_RESERVATION_TTL_S`, which do not exist pre-fix, so they die on `AttributeError`, not on
behaviour. The weight is on `test_cold_burst_is_refused_before_any_generator_runs`,
`test_reservations_and_inflight_share_one_ceiling` (public surface only) and the two-rank test.

The two-rank test separates preflight from submission into **two phases**, and that is the point
rather than a convenience. Verified, not assumed: a single-phase variant (preflight and submit
interleaved in one task) run against the **pre-fix** engine reported
`{streaming: 8, pending: 32, preflight_503: 1}` — it passes without the fix, because each
request's `_pending` entry lands before the next one preflights. That is the warm shape the
previous fix's pre-filled unit tests measured, and why they disagreed with the rig. That probe
was temporary and is deleted.

**Fresh-context independence not obtained, fourth session running.** Every gate here was run by
the main session; the session-level "do not call the AgentTool unless the user requested it" is
more specific than the CLAUDE.md orchestration policy, so no `verifier` was dispatched. Treat
the evidence as self-reported. The rig numbers are the exception in kind — they come from the
pinned gate over a raw dump, recomputable without re-running anything.

## S3 slice acceptance — where it actually stands

Checked against `discovery/spec/s3-plan.md` §Acceptance, not against "the rows are green":

| # | Item | State |
| --- | --- | --- |
| 1 | Concurrent matrix on the live pair + D7 throughput gate | **met** — 6/6 rows, both backends |
| 2 | Scheduler dynamics intact, "sole edit = the D4 inert seam" | **met in spirit, deviated in letter** — `87fb0e2e` added `waiting_queue_capacity()` and `preflight_queue_or_raise()` to `scheduler.py`. Recorded then and still true; `df100432` added nothing there |
| 3 | E4 tax ≤ 10.268 ms/token | **met** — stop condition never fired |
| 4 | D6 cache seams | **met** |
| 5 | All unit + cluster tests green at final commit | **met** |
| 6 | `docs/cluster/s3-measurements.md` readable without re-running | **met** |
| 7 | **Fresh verifier CONFIRMED at both boundaries (P2 code, P3 rig)** | **NOT met** |

**#7 stays open on acceptance item 7 alone.** Four sessions running, no fresh-context verifier:
the two dispatched ones went idle without reporting, and since then the session-level "do not
call the AgentTool unless the user requested it" has been the operative rule. Closing the slice
needs either that verifier or an explicit decision to waive the item — a call for the user, not
for a session that would be verifying its own work.

## Next

1. **#7 — S3 completion**, blocked only on acceptance item 7 above.
2. Single-node cold-burst hole (same defect, different accounting) — its own slice.
3. Optional: dedicated ring-vs-jaccl comparison, driven by the TTFT gap (4.0–12.4 s vs
   1.3–5.8 s), not the throughput ratio.
4. S6 carry-forwards unchanged: rejoin-without-leave accumulates ghost members;
   worker-rank-death propagation.

## Rig operation, confirmed again this session

- Studio synced by **`git bundle`**, never a push — `origin` has no `feat/cluster-v1`.
- Start the worker **attached**: `ssh -o ServerAliveInterval=15 ... 'cd ~/Developer/Repos/omlx
  && ulimit -n 65536 && export OMLX_BASE_PATH=$HOME/omlx-cluster-dev && exec .venv/bin/omlx
  serve --host 0.0.0.0 --port 8911 --base-path $HOME/omlx-cluster-dev'` from a background call.
- The worker keeps its old head URL across restarts but its member credential goes stale →
  heartbeat 401. Re-join: mint on the head with `.venv/bin/omlx cluster token --port 8910
  --api-key <head key>`, then POST `{head_url, token}` to the **worker's own**
  `/v1/cluster/local/join` with the **Studio's** api_key (keys are not shared).
- E10 does **not** check the omlx checkout. Compare both SHAs by hand *and* import the new
  symbols in both venvs — a stale worker does not crash, it silently reproduces the old number.
- Use `.venv/bin/omlx`; the PATH `omlx` is a uv tool install that rewrites `settings.json` and
  disables the cluster plane.

---
