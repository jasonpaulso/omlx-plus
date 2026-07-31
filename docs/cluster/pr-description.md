# PR description draft — Distributed Serving (cluster v1)

**Status: READY (not opened, not pushed).** Live-rig acceptance is complete —
`docs/cluster/s6-measurements.md` records the full resilience matrix (5/5 PASS) and both
D4 anchors passing (capacity best cell 66.7 tok/s ≥ 43; speedup 1.439× ≥ 1.3×). Updated
from branch `feat/cluster-v1` at `37f3c38f`; the `cluster-v1-pr` branch is cut from that
tip with the fork-only exclusions below. Opening the PR / pushing anywhere remains a
user decision.

---

## Title

`feat(cluster): distributed serving for two-node Apple Silicon clusters`

## Summary

Adds an opt-in distributed-serving mode: two (or more) oMLX nodes on the same Thunderbolt
link can form a `head` + `worker(s)` cluster and serve one model tensor-parallel across
their combined memory, for models too large for a single machine. Default behavior
(`cluster.role = "off"`) is completely unaffected — the cluster package is inert, no new
routes are reachable, no background task runs.

- Cluster control plane: bootstrap-token join, heartbeats, membership state, credential
  revocation on leave/removal/supersede.
- Placement: a plain model load decides local-vs-distributed automatically from model size,
  shard divisibility, and node capacity — distributed serving is not a separate API surface.
- Transfer: a model absent from a worker is synced before formation (peer copy from the
  head, or HuggingFace fan-out), resumable and digest-verified, file-granular.
- Formation: tensor-parallel serving over a `ring` (TCP) or `jaccl` (RDMA) collective
  backend, coexisting with the head's own locally-pinned models under the existing
  LRU/eviction accounting.
- Hardening: worker rank-death detection degrades a formation to single-node instead of
  hanging; head-restart recovery; rejoin dedup so a restarted worker's old credential is
  revoked instead of accumulating as a ghost member.

## Architecture summary

```
head node                              worker node
├── ClusterManager (control plane)     ├── ClusterManager (control plane)
│   ├── membership + heartbeats  <-----│   heartbeats, join/leave
│   ├── bootstrap token issuance
│   ├── placement (plan_placement)
│   ├── transfer orchestration   ----->│   HFDownloader / peer receive,
│   │   (peer or HF fan-out,           │   digest verify, resumable
│   │    manifest + digest verify)
│   └── formation (TP over ring/jaccl) │   TP worker rank(s)
│
└── EnginePool (unchanged single-node  └── EnginePool (unchanged single-node
    LRU/TTL/pinning machinery;             machinery; also runs standalone
    a formation is one more EngineEntry)   models the worker serves itself)
```

New code lives under `omlx/cluster/` (manager, formation, placement, transfer, heartbeat,
routes, auth, tp) plus settings/CLI/route wiring; existing scheduler/engine/pool code is
touched only at the narrow points a distributed model needs (formation teardown trigger,
placement call on load, memory accounting for a formation's share).

## What works (evidence)

Each slice was measured on a real 2-Mac rig (Thunderbolt-linked Apple Silicon, one pair per
run) with a pinned, falsifiable scorer committed alongside the measurement doc:

- `docs/cluster/bringup.md` — S0: collective transport spike, per-step coordination tax on
  both backends well under the 10% budget.
- `docs/cluster/s2-measurements.md`, `docs/cluster/s2-security-notes.md` — S2: control-plane
  bring-up (join/heartbeat/credentials) + a security pass on the bootstrap-token/trust model.
- `docs/cluster/s3-measurements.md` — S3: real TP-sharded serving through the batched
  engine, concurrent-throughput gate (`s3_compute.py`), coordination-tax re-measurement under
  load (`s3_tax.py`).
- `docs/cluster/s4-measurements.md` — S4: auto-placement decision correctness, mixed
  workload (distributed model + pinned local models, three concurrent streams), eviction/pin
  interplay — scored by `s4_score.py`, selftested against five must-fail shapes first.
- `docs/cluster/s5-measurements.md` — S5: model transfer (peer + HF fan-out), mid-transfer
  kill + file-granular resume, digest-mismatch re-fetch — scored by `s5_score.py`.
- `docs/cluster/s6-measurements.md` — S6: resilience matrix (rank kill, worker-daemon kill
  mid-flood, head restart, rejoin dedup/supersede — 5/5 PASS on the live pair) and the two
  acceptance anchors, both PASS (capacity: best cell 66.7 tok/s ≥ 43, jaccl batch-4;
  speedup: 1.439× ≥ 1.3×), scored by `s6_score.py` over committed raw dumps and
  independently re-confirmed by a fresh-context verification pass.

## Honest residuals (recorded, not fixed in this PR)

- **Transfer-beside-live-formation is unreachable via the only trigger surface**: the load
  path 409s on an active formation before the transfer pre-step, so "a formation keeps
  serving during a transfer" cannot be exercised end-to-end until a standalone
  transfer/repair surface exists.
- **Presence staleness up to ~65s**: a load inside the window after an external file change
  on the worker (outside the API) can skip the transfer step; a resulting holed model fails
  loudly (`TPIncompleteModelError`) rather than serving silently.
- **Worker `/v1/models` listing can lag briefly after a transfer** (pool list cache);
  discovery, node state, and formation all resolve the new model correctly in the meantime.
- **HF-watchdog cancel vs `to_thread(snapshot_download)` staging race**: staging can in
  principle be removed under a still-writing background thread; the orphan sweep is the
  backstop, not a fix.
- **Multimodal checkpoints stay refused for distributed placement by default** —
  distributing a vision-language model text-only would silently drop served capability.
  Eligibility does NOT trust the declared `language_model_only` flag alone: it
  cross-checks the checkpoint's own safetensors index for real vision-tower weights
  (`has_vision_tower_weights()`, prefix set derived from mlx_vlm's source), so a
  mislabeled checkpoint is refused with the real reason named. An explicit
  `cluster.allow_text_only_distribution` opt-in (default OFF) permits text-only
  distribution of any vlm-classified checkpoint; image/audio-bearing requests against a
  model distributed that way are rejected with a clean named 400
  (`cluster_non_goal`) at the route layer — on all three API surfaces (chat completions,
  `/v1/messages`, `/v1/responses`) — never silently served as text. Proper multimodal
  distribution (vision tower head-resident, text decoder sharded) is deferred to a
  future version by explicit decision.
- **Models with fewer KV heads than the world size cannot distribute** — the divisibility
  gate follows mlx-lm's tensor-parallel sharding, which splits KV heads across ranks.
  MQA/MLA-family checkpoints (e.g. DeepSeek V4's single latent KV head,
  `num_key_value_heads: 1`) are therefore refused with a named reason, even when memory
  would otherwise allow it. The standard remedy — replicate the KV projection on every
  rank and shard only the query heads — is a recorded follow-on, not implemented here.
  Relatedly, v1's equal-share TP means the weakest node's ceiling caps the largest
  distributable model (the head's surplus memory is unused); memory-weighted unequal
  sharding is out of scope for v1.

## Tests

- Unit suite: `pytest -m "not slow and not integration"` (default CI gate) — new cluster
  code covered by `tests/cluster/`.
- Cluster suite: `pytest -m cluster` — two-node-shaped tests that don't require live
  hardware (fakes/mocks at the transport boundary).
- Live-rig acceptance: manual, `benchmarks/cluster_spike/` — raw SSE captures + pinned
  scorer scripts per slice (`s3_compute.py`, `s3_tax.py`, `s4_score.py`, `s5_score.py`,
  `s6_score.py`), each with a `--selftest` proving the gate can actually fail before it is
  trusted to pass. `s6_anchors.py` + `s6_score.py` are the S6 acceptance-anchor harness
  (capacity: MiniMax-M2.7-3bit distributed, best of ring/jaccl x batch1/batch4 >= 43 tok/s
  — measured best 66.7 tok/s, jaccl batch-4, PASS; speedup: Qwen3.6-27B-bf16 under
  `cluster.allow_text_only_distribution`, distributed best >= 1.3x single-node best —
  measured 1.439x, PASS; full tables in `docs/cluster/s6-measurements.md`).
- Linters: `black`, `ruff`, `mypy omlx` (the latter does not cover `benchmarks/`).

## Upstream drift (facts only, as of this draft)

- Upstream remote: `https://github.com/jundot/omlx.git`, default branch `main`
  (`git remote show upstream` -> `HEAD branch: main`).
- Fork point (`git merge-base upstream/main HEAD`): `9595001e` (2026-07-28 22:44:17 +0900).
- Divergence (`git rev-list --left-right --count upstream/main...HEAD`): upstream is **54**
  commits ahead of the fork point, this branch is **57** commits ahead of it. This branch
  has never rebased onto upstream's newer history; rebasing is a separate decision for the
  repo owner, not performed here.
- This PR would target upstream `main` and add only `omlx/cluster/*`, cluster-related
  additions to `omlx/settings.py`/`omlx/cli.py`/`omlx/server.py`/route registration,
  `tests/cluster/*`, `docs/cluster/*`, and `benchmarks/cluster_spike/*`. Fork-only files
  (`CLAUDE.md`, `CONTEXT.md`, the `.gitignore` discovery line) are excluded from the
  eventual `cluster-v1-pr` branch, not from this repo.

## Rollout

Fully opt-in via `cluster.role` (default `off`); no behavior change for any existing
single-node deployment. No new required dependencies. No data migration.
