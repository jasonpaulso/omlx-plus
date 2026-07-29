# Session Context — 2026-07-29 (late) — row 4: reservation fix landed, rig re-run owed

**Status:** the second row-4 fix is in. Preflight now *reserves* the slot it checked for on
the cluster path, so the gate engages on a cold burst instead of reading a counter that is
still empty. Proven locally against a real two-rank formation; **not yet measured on the
rig**, which is the only thing that can flip row 4 to PASS. S3 (#7) stays open on that.

Branch tip: see `git log`. **Rig untouched this session** — no daemons started, nothing
ssh'd. Daily-driver `:8899` (PID 87617) and Studio `:8888`/`:8889` never approached.

## What was wrong, and what it needed

The first fix (`87fb0e2e`) gated on `len(self._pending)`. `_pending` is filled inside
`stream_generate`, which starlette only iterates *after* the route commits to the
`StreamingResponse`. On a cold burst all 41 requests preflight before any generator body
runs, so the counter is empty and the gate does nothing. Measured on the rig at `b3d39ad6`:
`{200: 41}`, zero 503s — identical to pre-fix.

The recorded blocker was that reservation "needs a preflight→submit identity, and the
route's `request_id` comes from an `x-request-id` header that is usually absent."
**That blocker is not real.** It only exists if reservations are *matched* to requests.
Counted instead, slots are interchangeable: preflight appends one, `stream_generate` drops
one when it takes its place in `_pending`. No identity anywhere.

## What landed

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
- **No lock.** Preflight and the `_pending` insert both run on the event loop and neither
  awaits between check and claim, so check-then-add is already atomic.
- **Single-node deliberately untouched.** It has the same cold-burst hole, but
  `Scheduler.preflight_queue_or_raise` compares against the *waiting* cap while the cluster
  gate compares against a total (running + waiting). Adding reservations there means
  switching it to the total form — a behaviour change to the default configuration that
  moves the tests `87fb0e2e` added. Row 4 is a cluster measurement; that is a separate
  slice, still open.

## Verification

| Gate | Result |
| --- | --- |
| `pytest -m cluster` | **22 passed** (3:40) — 21 baseline + the new two-rank cold-burst test |
| Falsifiability | **6 of 6 new tests fail against stashed source** — but see below |
| Default unit gate | **7534 passed, 2 failed** — the known GLM numerical pair, zero delta |
| black / ruff / mypy | zero delta (mypy 665 errors / 81 files both sides) |
| `s3_row4.py --selftest` | predicted `{200: 40, 503: 1}` scores **PASS**; all 5 shapes as specified |

**Don't over-credit the 6**, the same way the last session's 17-of-20 needed reading twice.
Three of the five unit tests reference `_reserved_slots` / `_RESERVATION_TTL_S`, which do
not exist pre-fix, so they die on `AttributeError`, not on behaviour. The weight is on
`test_cold_burst_is_refused_before_any_generator_runs`,
`test_reservations_and_inflight_share_one_ceiling` (public surface only) and the two-rank
test.

The two-rank test separates preflight from submission into **two phases**, and that is the
point rather than a convenience. Verified, not assumed: a single-phase variant (preflight
and submit interleaved in one task) run against the **pre-fix** engine reported
`{streaming: 8, pending: 32, preflight_503: 1}` — it passes without the fix, because each
request's `_pending` entry lands before the next one preflights. That is the warm shape the
previous fix's pre-filled unit tests measured, and why they disagreed with the rig. The
probe was temporary and is deleted; the committed test keeps the two-phase shape and
asserts what the row-4 gate asserts (≥1 rejection AND ≥1 still streaming).

**Fresh-context independence not obtained, fourth session running.** Every gate here was run
by the main session; the session-level "do not call the AgentTool unless the user requested
it" is more specific than the CLAUDE.md orchestration policy, so no `verifier` was
dispatched. Treat the evidence as self-reported.

## Next

1. **Rig re-run of row 4** at this commit — ring, then jaccl. Pinned protocol is in
   `docs/cluster/s3-measurements.md` ("Row 4 re-measure protocol"), no-retry included.
   Expected shape `{200: 40, 503: 1}`; in-stream backstop hits are tolerated by the gate.
2. **#7 — S3 completion**, blocked only on that run.
3. Single-node cold-burst hole (same defect, different accounting) — its own slice.
4. Optional: dedicated ring-vs-jaccl comparison, driven by the TTFT gap (4.0–12.4 s vs
   1.3–5.8 s), not the throughput ratio.

**Rig operation, hard-won:** start the worker **attached** —
`ssh -o ServerAliveInterval=15 ... 'cd ... && ulimit -n 65536 && exec .venv/bin/omlx serve ...'`
from a background call. A `nohup`-orphaned worker fails every control-plane call with
"All connection attempts failed" while curl/httpx/ClusterClient from the same host all
succeed; cause still unknown (a Tailscale network extension is the untested lead). Also: the
api_key is **not** shared between nodes, there is no `omlx cluster join` verb (the worker
calls its own `/v1/cluster/local/join`), head LAN is `192.168.4.68`, and E10 does **not**
check the omlx checkout — compare both SHAs by hand.

---
