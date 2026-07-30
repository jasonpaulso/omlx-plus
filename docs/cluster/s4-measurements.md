# S4 acceptance — auto-placement + pool coexistence, live 2-Mac rig

2026-07-30. Final SHA `cd872c0c`, ring backend, M5 Max 128 GB head (`10.0.2.1`) + M3 Ultra
96 GB Studio worker (`10.0.2.2`), daemons `:8910`/`:8911` under `~/omlx-cluster-dev`.
Verdict computed by the pinned scorer `benchmarks/cluster_spike/s4_score.py` over the raw
capture `benchmarks/cluster_spike/s4_dumps/s4_final_dump.json` — recomputable without the rig:

```
PASS  row3 preview==recorded
PASS  row1 mixed workload
PASS  row2 evict/pin interplay
VERDICT: PASS
```

The gate was proven falsifiable BEFORE rig time: `s4_score.py --selftest` feeds five
must-fail dumps (mismatched decision, errored stream, serialized window, evicted pinned
model, surviving cluster entry) — all FAIL — plus one good dump — PASS.

## What each row measured

**Row 3 — preview matches actual.** `GET /v1/cluster/placement?model=MiniMax-M2.7-3bit`
immediately before a load equals the `FormationJob.decision` read back from
`GET /v1/cluster/models/status` after it, over the pinned domain (`mode`, `world_size`,
`per_rank_estimate`, `divisible`, `presence`, `fits[*].ok`). Final pair captured around the
pass-4 reload (head under pressure: Qwen resident, decision `distributed / 2 / 60427446322`,
`requires_eviction=true`); an earlier quiesced-head pair at `c0be9205` matched identically.

**Row 1 — mixed workload.** MiniMax-M2.7-3bit (93 GB, 3-bit MoE) distributed via the
*plain* `POST /v1/models/{id}/load` auto-placement path + Llama-3.2-1B-4bit resident
single-node on the head (pinned) + Llama-3.2-1B-4bit resident single-node on the worker.
Three concurrent streams, one per surface, in one overlapping window at `cd872c0c`:
head-minimax 193 tokens / head-llama 63 / worker-llama 59, zero errors. The head pool
carried the formation as a real EngineEntry: `current_model_memory = 61 157 494 439` =
head share (60 427 446 322) + Llama (730 048 117), exactly.

**Row 2 — eviction/pinning interplay.** With the formation + pinned Llama resident, an
over-ceiling local load (Qwen3.6-27B-bf16, 53.5 GB) succeeded in ONE call: the LRU victim
was the cluster entry (never the pinned model), the formation unformed cleanly (unload job
`done`, worker rank procs 0, accounting restored exactly), Qwen loaded (`58 179 334 863` =
Qwen + pinned Llama). The pin survived a daemon restart (auto-preload). Reload leg: with
Qwen resident, placement returned `distributed + requires_eviction=true`, and one plain
load evicted Qwen, waited for memory settle, and formed MiniMax — generation through the
reloaded formation verified (32/32 tokens).

## Three defects this run found and fixed (each: rig repro → fix → pre-fix-failing test → re-measure)

| # | Symptom on the rig | Root cause | Fix |
| --- | --- | --- | --- |
| 1 | Loading a 0.73 GB model evicted the 93 GB formation | Rank-child memory is in neither `omlx_phys` nor free, so the dynamic ceiling collapses by the shard size while the pool also counts the head share — double-count makes every resident formation "over ceiling" | `c0be9205` — admission ceilings credit resident `cluster_head_share` (locals already get this basis via `omlx_phys`) |
| 2 | After a correct eviction, the triggering load still failed ("only 45.96 GB reclaimable"), succeeding 15 s later | Freed rank-child memory takes seconds to become reclaimable; the retry loop burned all attempts inside the window (local unloads have a settle barrier; cluster teardown didn't) | `80725884` — bounded settle wait (≤20 s, outside the pool lock) after a cluster-victim teardown |
| 3 | Reloading the formation with a non-pinned local model resident returned `mode=reject` | The distributed branch's head fit had no eviction awareness (`requires_eviction` existed only for the local rule) and `get_engine` treats reject as terminal | `cd872c0c` — `NodeCapacity.evictable_memory`, one `requires_eviction` definition for both branches, pre-formation LRU eviction of local victims |

All three are placement/admission *arithmetic against live memory dynamics* — invisible to
the unit suite's static-ceiling mocks (two pre-existing tests had even encoded defect #1 as
expected behavior). The formation/teardown machinery itself (P2 + P2b) behaved correctly in
every pass, including the failure passes.

## Evidence trail (per-row SHAs, stated rather than smoothed)

- Pass 2 (`c0be9205`): row 3 quiesced pair PASS; coexistence PASS (the defect-1 fix's live
  proof); row-1 window PASS-shaped (instrument note below).
- Pass 3 (`80725884`): row-2 trigger PASS (defect-2 fix's live proof); formation-as-victim +
  pinned-survivor + worker-scrub evidence is from this pass.
- Pass 4 (`cd872c0c`, final): row-1 window re-run; row-3 pair re-captured under pressure;
  reload leg PASS (defect-3 fix's live proof — itself an evict→settle→form cycle).

Instrument note: the first row-1 window read 2 "tokens" for the Llamas — SSE **events**
under-count when the server coalesces tokens into one chunk. Fixed to prefer the stream's
`stream_options.include_usage` final chunk with a chars/4 floor (`s4_window.py`); the
2-token readings were disproven by a 59-token non-stream probe of the same prompt.

## Rig disposition

Left clean and verified: daemons down, `:8910`/`:8911` free, 0 rank procs, member scrubbed,
MiniMax unformed, Llama unpinned, `backend=ring` + roles intact in both nodes' settings,
models kept (S6 anchors), daily drivers untouched (local `:8899` PID 87570; Studio
`:8888`/`:8889` PIDs 933/926 — verified before and after; note these PIDs rotated since the
S3 docs were written).

## S4 slice acceptance vs `discovery/spec/s4-plan.md`

| # | Item | State |
| --- | --- | --- |
| 1 | Mixed workload on the live pair | **met** (row 1) |
| 2 | Eviction/pinning interplay + clean unform + reload | **met** (row 2; via 3 fix cycles, each re-measured) |
| 3 | Preview matches actual over the pinned domain | **met** (row 3) |
| 4 | Flag-off zero delta; suites green unmodified except the named touchpoint rewrite | **met** at every commit (unit gate 7646/2 known-GLM at final SHA) |
| 5 | New unit + cluster tests green; linters baseline-delta zero | **met** (582 pool/cluster unit tests; `pytest -m cluster` 26 passed at `80aafa06` pre-P3) |
| 6 | This document, readable without re-running | **met** |
| 7 | Fresh verifier at both boundaries | **code boundary: met** (`verify-s4-p2` CONFIRMED with an owner-tracked lock instrument; P2b closed its residuals). **Rig boundary: the pinned-scorer-over-raw-dump form** (recomputable independence), same disposition as S3 — not a second model's eyes on the rig run itself |
