# S5 acceptance — model distribution on the live 2-Mac rig

2026-07-30. Head: M5 Max 128 GB (`10.0.2.1`), worker: M3 Ultra 96 GB Studio
(`10.0.2.2`), Thunderbolt data plane, ring backend, daemons `:8910`/`:8911`
under `~/omlx-cluster-dev`. Item 1 captured at `bc9d1469`; items 2–4 at
`7f9576bf` (two rig-found fixes landed mid-run, below). Verdict recomputable
without the rig:

```
python3 benchmarks/cluster_spike/s5_score.py benchmarks/cluster_spike/s5_dumps/s5_acceptance_dump.json
```

Scorer pinned and selftested (`--selftest`: 8 must-fail shapes fail, good dump
passes) **before** any rig time. Raw captures in
`benchmarks/cluster_spike/s5_dumps/`.

## Verdict

| # | Acceptance item | Result |
|---|---|---|
| 1 | Head-only model transferred + clustered, zero preemptive action | **PASS** |
| 2 | Kill worker daemon mid-flight → file-granular resume, verified files not re-sent | **PASS** |
| 3 | Corrupted file detected by digest, deleted, re-fetched | **PASS** |
| 4 | HF fan-out at the head's pinned 40-hex revision, no unverified file in final dir | **PASS** |
| — | CL5-16 wedged-peer probe (residual row) | SKIP (see below) |

## The journeys

**Item 1** — `gpt-oss-120b-Fable-5-Distilled` (59 GB, 13 shards, bare-folder,
head-only; peer-only source so D6 auto-picks with no prompt). One
`POST /v1/cluster/models/load {"prefer": "distributed"}` returned HTTP 200
`ready` in 103.7 s covering manifest build → TRANSFER_START → have-scan (empty)
→ 2 rounds → 22/22 digest-verified → formation. Decision recorded with reason
`"model absent on <member>; transfer required (S5)"`. Worker discovery logged
exactly the head's id (`type: llm`). Transfer throughput ≈ **1.26 GB/s**
(50.5 GB in 40 s) over the TB ring. Post-form stream: 48 tokens at 48.7 tok/s.

**Item 2** — worker copy holed (5 shards deleted, 1 corrupted in place), then a
fresh load; `kill -9` of the worker daemon 12 s into the round with 16 files
have-verified. Load failed HTTP 424 (`worker did not ack in time`), job
`error 16/22`, rollback clean (`active_model: None`, entry re-loadable). After
worker restart + rejoin, the re-issued load: have-scan re-verified the 16
survivors, **one round re-sent exactly the 5 missing shards + the corrupt
shard**, `done 22/22`, formation `ready` in 98 s. Evidence (mtime_ns/size
capture before + after): all 16 verified files byte-identical and absent from
the resent set.

**Item 3** — the corrupt shard (1 MB of zeros at offset 100 MB, size unchanged —
digest-only corruption): observed sha `e19176…` ≠ manifest `fe5e23…` before
resume; dropped from `have` by the scan; re-fetched; final sha == manifest sha.
Corrupted bytes never entered a *new* final-dir file (the in-place-corrupted
file was atomically replaced by a verified copy).

**Item 4** — `mlx-community--Qwen2.5-0.5B-Instruct-4bit`, hub-cache-resident on
the head at revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, absent on the
worker. Omitted `source` → typed `choice_required` (HTTP 200, both viable).
Explicit `source: "hf"` → worker HFDownloader pull → digest verify vs the
head's snapshot-rooted manifest → job `done 10/10` → formation `ready` in 37 s
→ 32-token stream. Worker final dir listing == the 10 digest-verified manifest
entries exactly (no `.cache/**`, nothing unverified — asserted by listing).
Revision pinning evidence: worker files digest-match the head's snapshot
manifest (equivalent-or-stronger than a logged revision; the downloader does
not log the revision — minor residual).

## Rig-found defects, fixed mid-run (each with a failing-first test)

1. **`bc9d1469` (P3a): the head never spawned its rank-0 `--role src`
   session.** Every round: worker rank 1 up, 10 s ring-join timeout, killed —
   `3 consecutive failed/no-progress rounds`. Every unit/integration test
   faked the session layer worker-side only, so no test ever needed a head
   rank — a fake-seam blind spot only the rig could expose. Fixed in
   `_drive_rounds` (spawn before TRANSFER_ROUND, stop in `finally`,
   spawn-bound counts as a stalled round).
2. **`7f9576bf` (P3b): presence meant "discoverable", not "complete".** A
   partially transferred dir (config + index landed) still discovered, so the
   re-issued load skipped the transfer pre-step entirely — the resume path was
   structurally unreachable — and the TP rank *lazy-loaded the holed dir
   without error* (`load_model(lazy=True, strict=False)`), producing a `ready`
   formation that served nothing. Fixed two-ways: `_scan_models_present`
   omits index-incomplete models (worker log line observed live), and
   `shard_and_load` raises `TPIncompleteModelError` before the lazy load
   (observed live: rank died 4 s after spawn on the holed dir).

## Wedged-peer probe (CL5-16 residual row — recorded, not scored)

A real dst rank was `SIGSTOP`'d on three attempts. What the rig proved: **no
wedge ever held anything** — every attempt reached a terminal state with the
single-active gate released (one in-job recovery: stopped rank killed in round
teardown, rounds 2–3 re-sent the remainder, job `done 22/22`; one clean
`error` after the round cap with a successful re-issue afterward). What the
rig did *not* prove: the 30 s min-progress watchdog specifically was never
demonstrably the killer — faster guards (10 s ring-join timeout, round
teardown) won every race. The watchdog's evidence remains its full-path unit
tests (stalled-fake downloader/round, gate-release-via-`finally` rows).

## Residuals (recorded, not fixed here)

- **Transfer-beside-live-formation is unreachable via the only trigger
  surface**: the load path 409s on an active formation before the transfer
  pre-step, so the D4 "resident formation keeps serving during a transfer"
  provision and data-plane port coexistence cannot be exercised end-to-end
  until a standalone transfer/repair surface exists. Port non-overlap stays
  pinned by the CL5-17 settings assertion + tests.
- **Presence staleness ≤65 s** (worker 60 s inventory rescan + 5 s beat): a
  load inside the window after external file changes skips the transfer;
  P3b's rank guard turns this loud (formation error), not silent. Digest
  authority stays with the transfer have-scan by design.
- **Rank-death propagation (known S6 item), now measured**: worker deathwatch
  reported the dead rank in 4 s; the head's formation hung ~385 s before
  failing (`rank 0 closed its channel before reporting ready`) — twice. S6's
  heartbeat rank-status field + degrade path is the fix.
- **Round-retry port-grace collision**: after a killed round, immediate retry
  rounds can join-timeout while the predecessor session dies its 10 s grace
  (observed: 3 rounds burned in ~60 s → job error; re-issue succeeded). S6
  candidate: retry backoff > kill grace, or per-round port rotation.
- Worker `/v1/models` listing lags post-transfer discovery (pool list cache);
  discovery, node_state, and formation all resolve the new model correctly.
- Double cluster-load surfaces `ModelLoadingError` as an unhandled 500 on
  `/v1/cluster/models/load` (route mapping gap).
- HF watchdog cancel + `to_thread(snapshot_download)`: staging can be
  rmtree'd under a still-writing thread (code-read finding from the P2d
  verifier; orphan sweep is the backstop).
- Most local inventory classifies `vlm` and is refused distributed placement
  (Qwen3.6/Qwen3.5/gemma/Ornith/KAT…). S6 must investigate the classifier —
  the Qwen3.6-27B ≥1.3× anchor is unrunnable while this holds.

## Rig hygiene

Left clean and verified: daemons down, ports `8910`/`8911` free, no rank or
transfer processes on either node, staging swept, members scrubbed, settings
untouched (never edited this run), must-not-touch PIDs unchanged (local
`:8899` = 87617; Studio `:8888`/`:8889` = 947/926). Worker inventory ends
richer: `gpt-oss-120b-Fable-5-Distilled` (complete, digest-verified) and
`mlx-community--Qwen2.5-0.5B-Instruct-4bit` now on the Studio; nothing was
lost from either machine.
