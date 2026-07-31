# S6 P3 — Resilience matrix + acceptance anchors (live rig)

Rig: M5 Max 128 GB head (`10.0.2.1`, `:8910`) + M3 Ultra 96 GB worker (`Jasons-Mac-Studio.local`,
`10.0.2.2`, `:8911`), TB5 data plane, ring backend unless stated. Code at `46031823`
(both checkouts + venvs probed for the symbols under test, per rig discipline).
Raw captures: `benchmarks/cluster_spike/s6_dumps/` (timelines are epoch-stamped JSONL from a
1 Hz status poller; stream captures are raw SSE bodies with start/end stamps).
All matrix rows and anchor cells were measured at `46031823`; the two defect fixes below
landed after the cells at `03ef2387` and were then live-verified on a fresh formation
(preview parity: `mode=distributed`, empty reasons; vision probe: clean 400
`cluster_non_goal`; text against the same formation: 200).

**Status: COMPLETE. Resilience matrix rows 1–5 PASS (row 2 re-proven post-fix at
`46031823`); both D4 anchors PASS (capacity best cell 66.7 tok/s ≥ 43; speedup 1.439× ≥ 1.3×);
E4 tax PASS on every cell. Two real defects found by the live run are fixed on this branch:
the placement preview ignored `allow_text_only_distribution`, and vision requests against a
text-only-distributed model were silently served as text (route conversions strip image
parts before the engine guard) — both now route-guarded with pre-fix-failing tests.**
(The mid-run Thunderbolt link loss was physical; after replug the link was re-proven with a
30/30 TCP probe streak over 4 minutes before any measurement resumed.)

## Resilience matrix

| # | Row | Result | Evidence |
|---|---|---|---|
| 1 | Rank kill mid-flood (worker rank SIGKILL, daemon alive) | **PASS** | `s6_dumps/resilience/rowA-*` |
| 2 | Full worker drop mid-flood (daemon + rank SIGKILL) | **FAIL at `b6271845` → fixed `46031823` → re-proof PASS** | `s6_dumps/resilience/rowB-*` (pre-fix), `rowB2-*` (post-fix) |
| 3 | Member recovery after worker restart | **PASS** | timeline in CONTEXT/session log |
| 4 | Rejoin dedup: refusal (active) + supersede (lost) + old credential dead | **PASS** | session captures |
| 5 | Head restart recovery + re-form | **PASS** | session captures |

### Row 1 — rank kill mid-flood (D1 dead-rank report path)

MiniMax-M2.7-3bit formed distributed (ring, world 2, 42 s). 4 concurrent streams
(max_tokens=256), worker rank SIGKILLed ~4 s in (epoch 1785473335.48).

- Head learned via the worker's dead-rank heartbeat report at **t+4.1 s** (alarm:
  `formation ... degraded (worker reported dead rank(s) [1]); tearing down`). Pre-S6 this
  path did not exist and the equivalent hang measured ~385 s (S5 residuals).
- All 4 streams terminated with a clean in-stream error
  (`data: {"error": {"message": "rank 0 closed its reply channel", ...}}` then `data: [DONE]`)
  at t+23.7 s — delivered by the teardown killing rank 0. No hangs.
- Teardown complete at t+24.5 s; `active_model` null; local model (Llama-3.2-1B) served
  normally throughout and after.
- Re-issued distributed load formed clean in 42.3 s (job `c5d4b75e9907`) — the worker's
  stale post-kill ranks payload did NOT abort the fresh formation (P1c R1 fix, live).

### Row 2 — full worker drop mid-flood (the gap)

Same flood shape; worker daemon AND rank SIGKILLed together (epoch 1785473542.98),
simulating a machine loss.

Measured at `b6271845`: the head marked the member `lost` at t+21.0 s (member_timeout 20 s)
— and did nothing else. Formation stayed `ready` ≥2 min (poller end), `active_model` still
set; all 4 streams hung on keepalives until the client-side 120 s cap (no server error, no
`[DONE]`); fresh requests hung (10 s probe, empty). Head rank 0 alive but wedged: mlx ring
logged `[ring] Receiving from socket 3 failed with errno 54` floods then
`Too many send/recv errors. Aborting...` WITHOUT the process exiting, so the engine's
pipe-EOF detection never fired. Neither of D1's two triggers (worker dead-rank report,
engine EOF) can see this failure class: the reporter is dead and the EOF never happens.

Fix (`46031823`, D1 amendment recorded in the plan): the liveness scrub's member→`lost`
transition now notifies the formation manager, which runs the same degrade path as a
dead-rank report (formation-scoped, idempotent). The fix also un-queued scrub's lost-marking
from the E6 command queue — a formation blocked in `wait_ready` monopolizes that queue, so a
queued scrub could never have fired during exactly the hang it needs to break (proven by a
pre-fix test that deadlocked: `asyncio.TimeoutError` on `scrub()` behind a blocked queue).
Expiry/prune still rides the queue (persisted-state mutation, D3 serialization).

Second sighting, same session: an orphaned worker daemon (control ssh dropped by a transient
"No route to host") wedged its heartbeats → member `lost` 307 s → same do-nothing at
`b6271845`. One fix covers both.

**Re-proof at `46031823` (2026-07-31, `s6_dumps/resilience/rowB2-*`): PASS.** Gate stated
before the run, with the pre-fix rowB dump as the proven must-fail shape. Same flood shape
(4 streams, MiniMax ring), worker daemon 39563 + rank 40184 SIGKILLed at epoch
1785523808.97:

- member `lost` at **kill+20.3 s** (member_timeout 20 s, exact);
- formation **degraded + teardown done by kill+34 s** via the new liveness-scrub trigger
  (pre-fix: still `ready` at +120 s);
- all 4 streams got the clean in-stream error + `[DONE]` at **kill+30.6 s**, curl exit 0
  (pre-fix: hung to the client-side cap);
- fresh MiniMax request post-teardown: **200 in 8.8 s, served head-local** — auto-placement
  fell back to a local load (98 GB fits the idle head), strictly better than the gate's
  "prompt clean error" expectation; recorded as the S4 fallback working as designed;
- head healthy throughout; head-local Llama served normally.

### Row 3 — member recovery

Worker daemon restarted with persisted identity while its member row was `lost`: member
returned `active` on the first heartbeat, same member id, no ghost created (D2 behavior, live).

### Row 4 — rejoin dedup (D3/D3-sec live)

- Same-name join against an ACTIVE member: **409** with the named error ("cannot be
  superseded: it is active (a member becomes replaceable after 20s of silence)...");
  member untouched.
- Identity-wiped worker (cluster.json `worker` cleared — the S2 "rejoin without leave"
  shape), member `lost`: fresh join **superseded** it — new member id (`b7f44ee2…`
  replacing `88507141…`), exactly one member remained, and a heartbeat signed with the old
  member secret returned **401**. Ghost-accumulation carry-forward closed on the live pair.

### Row 5 — head restart recovery

Head daemon killed and restarted with no formation resident. The persisted member returned
`active` on the head's first state read (worker's failure-backoff heartbeat loop
re-established liveness with no operator action); a re-issued distributed load reached
`ready` in 39.5 s. (The subsequent smoke was aborted by the orphaned-worker event above —
re-form itself is the row's claim and is proven; the smoke re-runs with the Row 2 re-proof.)

## Anchors (D4) — BOTH PASS (2026-07-31)

All cells first-attempt (`attempt=1`, no retries anywhere); dumps in
`benchmarks/cluster_spike/s6_dumps/`; scored by the pinned `s6_score.py`
(selftest re-run OK before and after the counting amendment below).

### Scorer amendment (recorded BEFORE the speedup cells ran)

`s6_score.py::completion_tokens` used `max(usage.completion_tokens, (text_len+reasoning_len)//4)`.
The capacity dumps exposed a double-count: oMLX streams MiniMax decode as `reasoning_content`
and then re-emits the full text as a final `content` block, so the chars floor counted the
same characters twice (observed `text_len == reasoning_len == 694` exactly, while
`usage == arrivals == 128`; a non-stream probe measured a sane 5.41 chars/token). Amended to:
server usage is authoritative when present; chars/4 only as fallback for absent usage. The
capacity dumps predate the amendment and are scored under the corrected rule — the verdict is
**PASS under both countings** (inflated best was 178.3 tok/s; corrected best 66.7; gate 43).
Numbers below are the corrected ones.

### Capacity anchor — MiniMax-M2.7-3bit distributed, gate ≥ 43 tok/s: **PASS**

| Backend | Batch | Decode rate | E4 tax (budget 10.268 ms/tok) |
|---|---|---|---|
| ring | 1 | 24.23 tok/s | PASS 0.71 ms/tok |
| ring | 4 | 48.96 tok/s (aggregate) | PASS 0.41 ms/tok |
| jaccl | 1 | **46.04 tok/s** | PASS 0.61 ms/tok |
| jaccl | 4 | **66.74 tok/s (aggregate)** | PASS 0.38 ms/tok |

Best cell 66.74 ≥ 43 → PASS; notably the jaccl batch-1 single-request rate (46.0) clears the
gate alone — vs ~23.9 measured in the S3 spike era.

### Speedup anchor — Qwen3.6-27B-bf16 under the text-only opt-in, gate ≥ 1.3×: **PASS**

Model id `mlx-community--Qwen3.6-27B-bf16` (the id present on BOTH nodes — head bare-folder
copy `Qwen3.6-27B-bf16` measured identically, 10.30 vs 10.26 tok/s, dumps kept as
provenance). Worker copy verified complete pre-run (11/11 shards, 54.71 GB). One 32-token
warmup ran before the single-node cells (page-in; recorded, not a cell).

| Node mode | Batch | Rate |
|---|---|---|
| single-node | 1 | 10.26 tok/s |
| single-node | 4 | 18.67 tok/s (aggregate) |
| distributed (jaccl) | 1 | 17.29 tok/s |
| distributed (jaccl) | 4 | **26.88 tok/s (aggregate)** |

Best distributed 26.88 vs best single 18.67 = **1.439× ≥ 1.3×** → PASS. Every contributing
record hit the 256-token steady-state floor. (Ring distributed cells not needed — the gate
passed on jaccl; ring remains available headroom.)

## Defects found by this run (both fixed on this branch, after the cells)

1. **Preview ignored the opt-in** — `GET /v1/cluster/placement` called `plan_placement`
   without `allow_text_only_distribution`, so it reported "vlm … not eligible" for a model
   the real load path happily distributed (observed live: preview said `local`/ineligible,
   the load formed distributed seconds later). Fixed in `cluster/routes.py`; parity +
   polarity tests added (fail pre-fix).
2. **Vision request against a text-only-distributed model was silently served as text**
   (HTTP 200, image dropped, `prompt_tokens` counted text only) — the fork ruling requires a
   clear error. The engine-level guard (`_reject_if_multimodal` in `preflight_chat`) never
   fires on the live path because the non-VLM message conversions (`extract_text_content`,
   `convert_anthropic_to_internal(preserve_images=False)`,
   `convert_responses_input_to_messages`) strip image parts BEFORE preflight; the P1c test
   suite faked preflight raising, so it proved handler wiring, not the path. Fixed with a
   route-level guard on the RAW payload in all three routes (chat completions, `/v1/messages`,
   `/v1/responses`); route-level tests with an inert preflight fail pre-fix (4/4) and a
   post-fix live probe on the formed rig returned the clean 400.
