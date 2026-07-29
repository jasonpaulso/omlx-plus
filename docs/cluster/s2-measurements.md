# S2 P3 — Live-Rig Acceptance Measurements

Date: 2026-07-28. Rig: local M4 Max 128GB (head, `10.0.2.1:8910`) + Mac Studio M3 Ultra 96GB
(worker, `10.0.2.2:8911`), TB5 link, code at commit `7e021f08` (`feat/cluster-v1`, S2 P1+P2+P3-fix)
on both nodes. This is the tracked S2 P3 deliverable per `discovery/spec/s2-plan.md`.

## Result summary vs S2 acceptance

| # | Acceptance item | Result |
|---|---|---|
| 1a | MiniMax-M2.7-3bit loads across the pair, generates, streams, aborts cleanly (real HTTP, ring) | **PASS** |
| 1b | Small-model + Qwen greedy parity, dist-vs-single-node, ring | **PASS** (byte-identical / matches after documented trailing-whitespace normalization) |
| 1c | MiniMax loads/generates/streams/aborts cleanly, jaccl (real HTTP, RDMA) | **PASS** — jaccl formation succeeded on the real RDMA link (negotiated_backend=jaccl, world_size=2); greedy output correct ("Paris. The capital of the United Kingdom is London…"); mid-stream abort clean (formation stayed ready, follow-up request succeeded) |
| 2 | E4 tax ≤ 10% of 102.682 ms/token, ring | **PASS** — 0.457 ms/token, 4.45% of budget |
| 2 | Same, jaccl | **PASS** — 0.262 ms/token, 2.55% of budget (300-step decode; lower than ring, consistent with RDMA) |
| MiniMax capacity | 93GB 3-bit MoE shards across 96GB Studio + 128GB local without OOM | **PASS** (operational evidence; precise per-rank memory bytes not captured — see caveat) |

**Headline: BOTH backends pass every acceptance item. MiniMax (93GB 3-bit MoE) loads and serves
across the 96GB Studio on ring AND on the real RDMA/jaccl link; the real-protocol E4 tax clears the
10% budget with large headroom on both (ring 4.45%, jaccl 2.55%). The jaccl half was completed in a
follow-up run after the jaccl-launcher-wiring slice (commit 4b0f0d49) landed and deployed — see the
updated jaccl section below.**

_Superseded note (kept for history): at the time of the ring run, the P2-landed launcher/formation code
hard-rejected any backend other than `ring` before formation was attempted, and a dedicated
jaccl-launcher-wiring slice was in flight to close that gap — see the jaccl section below for
why this doc deliberately does not record the (soon-to-be-deleted) 400 as acceptance evidence.**

## E4 tax re-measurement (Task per D9)

- Model: `mlx-community--Qwen3.6-27B-bf16` (local id resolves via the HF cache; see model-id note
  below — this is the SAME on-disk model as the S0 baseline, D9's named acceptance model).
- Backend: **ring**, explicitly pinned in both nodes' `cluster.backend` setting (never `auto`).
  `negotiated_backend` in the response matched the requested `ring` — no silent fallback.
- Greedy (`temperature=0.0`), 300-step decode (`max_tokens=300`, all 300 emitted,
  `finish_reason="length"`), read from `GET /v1/cluster/models/status` → `engine_stats.last_tax`
  immediately after the request (this endpoint's `engine_stats` field is new — see "P2 gap found and
  fixed" below).

```
steps: 300
avg ms/token: 0.457148
p50 ms/token: 0.448417
p90 ms/token: 0.486792
negotiated_backend: ring
```

**10% budget = 10.2682 ms/token (from the S0 baseline, `docs/cluster/bringup.md`).**
**Ratio: 0.457148 / 10.2682 = 0.0445 → 4.45% of budget consumed. PASS**, with large headroom.
As a fraction of the raw baseline: 0.445% of 102.682 ms/token — higher than S0's spike-level ring
figure (0.15%, smoke-scale synthetic skew) but still comfortably under budget, consistent with S0's
own prediction that the real rank-0-drives protocol under real heterogeneous skew would raise the
number without necessarily breaching it.

jaccl: **0.262 ms/token, 2.55% of budget — PASS** (measured in the follow-up jaccl run; see the jaccl section below).

## MiniMax-M2.7-3bit capacity acceptance

- Load: `POST /v1/cluster/models/load {"model": "MiniMax-M2.7-3bit"}` → `status: "ready"` in ~40s
  (job `28213c3740a6`; `head_presence` → `register_engine` steps all clean, no retries).
- Generate (`/v1/chat/completions`, non-streaming): 32 completion tokens in 7.34s total (includes
  prefill of a 47-token prompt) ≈ 4.4 tok/s including prefill. Not a pure decode-only figure.
- Stream (`/v1/chat/completions`, `stream: true`): SSE chunks delivered incrementally
  (`reasoning_content` then `content` deltas, terminated by `finish_reason: "length"` then
  `[DONE]`) — streaming confirmed working.
- Abort: a real mid-stream client disconnect (curl piped to `head -N`, closing the pipe early —
  genuine TCP disconnect, not a synthetic signal) during a 512-token generation. Formation state
  stayed `"ready"` immediately after; a follow-up request on the same formation succeeded normally
  (8 tokens, 1.11s). **Clean abort, formation survives.**
- Unload: `POST /v1/cluster/models/unload` → `status: "unloaded"` in ~29s, rank processes exited
  cleanly on both nodes (confirmed via `ps`, zero `rank_worker` processes remaining).

**Memory reality check (the P1 caveat this run was meant to resolve): PASS, with a measurement
caveat.** The 93GB 3-bit MoE model loaded, generated, streamed, and unloaded without any crash, OOM,
or swap-thrash symptom on either node — including the 96GB Studio, which holds one full rank's
shard. This is real operational evidence the model fits. However, I did **not** obtain a precise
per-rank memory-bytes figure: `ps`'s RSS column reported only ~4.9GB for each rank process, which
almost certainly **undercounts** MLX's Metal-backed unified-memory allocations (a known macOS
accounting quirk — GPU-resident buffers are not fully reflected in classic process RSS). `get_stats()`
carries no memory field. A precise number would need `mx.get_peak_memory()` read from inside the rank
process (not currently exposed) or a Metal-aware profiling tool run during the load — out of scope
for this Bash-driven P3 pass. Treat "no OOM, no crash, formation stayed ready through a full
generate/stream/abort/unload cycle" as the acceptance evidence, not a specific GB number.

## Greedy parity spot-checks (ring)

Both comparisons use the raw, non-chat-templated path on both sides for a fair comparison
(`/v1/completions` on the distributed side — which routes to `engine.generate()`, same as the P2
integration test — vs `omlx.engine.batched.BatchedEngine.generate()` called directly, single-node,
same prompt/params). An earlier attempt compared `/v1/chat/completions` (templated) against
`BatchedEngine.generate()` (raw) and produced completely different text — that was a methodology
error on my part (comparing two different code paths), not a parity failure; corrected before
recording the result below.

**Qwen3.6-27B-bf16**, prompt `"The capital of France is"`, `max_tokens=16`, `temperature=0.0`:
- Distributed: `"Paris.\n\n<think>\n\n</think>\n\nThat is correct. Paris is the capital and"`
- Single-node: `"Paris.\n\n<think>\n\n</think>\n\nThat is correct. Paris is the capital and"`
- **Byte-identical. PASS.**

**Llama-3.2-1B-Instruct-4bit**, same prompt/params:
- Distributed: `"Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\n"`
- Single-node: `"Paris.\nThe capital of France is Paris.\nThe capital of France is Paris."`
- Differ only in a trailing newline — matches P2's own documented finding
  (`tests/cluster/test_cluster_serving.py`: "the two paths detokenize/finalize trailing whitespace
  differently"). After `.rstrip()` both sides are identical. **PASS** (no mid-stream argmax
  divergence — a real divergence would survive `.rstrip()` and fail this check).

jaccl: **measured — PASS** (0.262 ms/token, 2.55% of budget; see the jaccl section below).

## jaccl: measured on the real RDMA link — PASS

The jaccl-launcher-wiring slice (commit `4b0f0d49`, merged after the ring run) removed the three
ring-only hard-gates and wired the existing `hostfile.py` jaccl builders (`jaccl_env`, the IBV
device-matrix writer, coordinator port) through the launcher/formation/manager spawn path. It was
deployed to both nodes; both daemons were restarted with `cluster.backend="jaccl"` explicitly pinned
(never `auto`, per the D9 integrity rule), and the jaccl acceptance run followed:

- **Formation on jaccl succeeded** — `POST /v1/cluster/models/load {MiniMax-M2.7-3bit}` returned
  `{"status":"ready","negotiated_backend":"jaccl"}`; status confirmed `world_size=2`,
  `negotiated_backend=jaccl`, `loaded=true`, no alarms. The RDMA group formed on the real Thunderbolt
  link (device matrix `[[null,"rdma_en2"],["rdma_en4",null]]`, coordinator `10.0.2.1:41200`).
- **Generation correct** — greedy `/v1/completions` on "The capital of France is" →
  `"Paris. The capital of the United Kingdom is London. The capital of Germany is"` (coherent, correct).
- **Clean abort** — a mid-stream request killed at the client: formation stayed `ready` with no
  alarms, and an immediate follow-up request succeeded (not wedged).
- **E4 tax on jaccl** — 300-step greedy decode, read through `engine_stats.last_tax` on the status
  endpoint: **avg 0.262 ms/token, p50 0.244, p90 0.325**. Against the 10.268 ms/token budget = **2.55%.
  PASS**, with even more headroom than ring (4.45%). jaccl's lower real-protocol overhead vs ring is
  consistent with the RDMA-vs-TCP advantage S0 observed at the collective level.

The negotiated backend was verified to equal the requested backend on the run (a ring-fallback under
an explicit jaccl request would have failed the run per the wiring slice's no-silent-fallback rule,
not been recorded as jaccl). Both backends now satisfy acceptance items 1 and 2 in full.

_Historical note: at the ring-run stage the P2-landed code (`be739ffa`) hard-rejected any non-ring
backend with an explicit 400 before formation — a fail-loud gate, not a silent ring substitution.
That 400 was deliberately never captured as "acceptance evidence" because it came from the exact
hard-gate `4b0f0d49` deleted; the real evidence is the jaccl PASS recorded above._

## P2 gap found and fixed during this session

`GET /v1/cluster/models/status` originally returned only `{active_model, jobs, alarms}` —
`ClusterEngine.get_stats()` (which computes the exact D9 `{steps, avg_ms, p50_ms, p90_ms,
negotiated_backend}` figures) was reachable only as an in-process Python call, with no HTTP surface.
Flagged mid-session; fixed in a new commit (`7e021f08`, additive, not amending `be739ffa`/P1/P2) that
adds `snap["engine_stats"] = engine.get_stats()` to `ClusterFormation.snapshot()` when a model is
loaded. Verified live in every measurement above — the `engine_stats.last_tax`/`negotiated_backend`
fields in this doc all came from this endpoint post-fix.

## Model-id conventions — verified directly, not assumed

D10's prose implies an `org/name` slash convention throughout; the actual `discover_models_from_dirs`
behavior is more particular and was verified with `python -c "..."` against both nodes' real
`get_effective_model_dirs()` before any load call (avoided several dead-end load attempts):

- Plain directories under a configured `model.model_dirs` entry (e.g. `~/Models/mlx-community/X`)
  register with the **bare folder name**, no org prefix: `"MiniMax-M2.7-3bit"`, not
  `"mlx-community/MiniMax-M2.7-3bit"`.
- HF-cache entries (`~/.cache/huggingface/hub/models--Org--Name/...`) register with the cache
  directory's **double-dash** convention: `"mlx-community--Qwen3.6-27B-bf16"`, not
  `"mlx-community/Qwen3.6-27B-bf16"`. This id is an omlx-internal string only — passing it to
  `mlx_lm`/`huggingface_hub` directly (e.g. via `BatchedEngine(model_name)`) fails HF's repo-id
  validator (`Cannot have -- or .. in repo_id`); the single-node reference runs in this doc used the
  resolved filesystem snapshot path instead.
- **A real cross-node trap, found and avoided before any load attempt**: the local machine has a
  pre-existing, unrelated plain-dir copy of Qwen bf16 at `~/Models/mlx-community/Qwen3.6-27B-bf16`
  (not created by this session) that resolves to the bare id `"Qwen3.6-27B-bf16"` and would shadow
  the HF-cache entry locally. The Studio has no such plain-dir copy — only the HF-cache entry from
  this session's transfer, id `"mlx-community--Qwen3.6-27B-bf16"`. Using the bare id would have
  loaded fine on the head and then failed the worker's own CL2-02 presence check with a **false**
  "absent" (an id-string mismatch, not real absence). All Qwen calls in this doc use the double-dash
  id, verified to resolve identically and correctly on both nodes.

## `mlx_lm.share` transfer measurements (D10)

Both transfers used a `share_hostfile.json` (`{"backend":"ring","envs":[],"hosts":[{"ssh":"127.0.0.1","ips":["10.0.2.1"]},{"ssh":"Jasons-Mac-Studio.local","ips":["10.0.2.2"]}]}`).

**MiniMax-M2.7-3bit**: `--path ~/Models/mlx-community/MiniMax-M2.7-3bit --dst
~/Models/mlx-community/MiniMax-M2.7-3bit` (plain directory, worked on the first attempt). 93GB, 94
files, ~20s wall, ~4.8GB/s. Landed correctly, verified via `du -sh` + non-zero file sizes.

**Qwen3.6-27B-bf16**: hit two real bugs, both worth recording for S5's manifest-transfer design:
1. `--model mlx-community/Qwen3.6-27B-bf16` failed: `hf_repo_to_path` → `snapshot_download(...,
   local_files_only=True)` → `IncompleteSnapshotError` — the local HF cache entry has real weight
   blobs but is missing 2 metadata files (`.gitattributes`, `README.md`) that `huggingface_hub`'s
   completeness check requires even though `omlx`'s own `discover_models_from_dirs` is satisfied
   with far less (just `config.json` + weights).
2. Worked around #1 with `--path`/`--dst` pointed directly at the local snapshot directory
   (`~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-bf16/snapshots/<rev>`) — this
   **landed all 22 files at 0 bytes** on the Studio. Root cause: `mlx_lm/share.py`'s
   `DirectoryEntry.from_path`/the receiver's materialization (`share.py:52,194-195`) treats a
   symlink entry as "recreate this symlink," storing/replaying `path.readlink()`'s target string
   rather than following it — since the snapshot dir's `*.safetensors` entries are symlinks into
   `../../blobs/<hash>` (HF's default on-disk layout), and `blobs/` sits outside the transferred
   `snapshots/<rev>` root, the receiver got symlinks pointing at blob hashes that were never shipped.
   Fixed by `cp -RL` dereferencing the snapshot directory into a plain temporary copy first (APFS
   clone-on-write makes this ~free — real elapsed time 26.9s wall including checksums, no meaningful
   extra disk used), then sharing that plain copy with `--dst` pointed at the real target snapshot
   path. Cleaned the broken 0-byte partial transfer off the Studio before retrying; removed the local
   temp copy after. Re-run: 51GB, 22 files, ~12s wall, ~4.7GB/s. Landed correctly, verified via
   `du -sh` + non-zero file sizes on a sampled file.

**Both bugs are `mlx_lm.share`-level (vendored, pinned at `ab1806e`), not `omlx` code.** S5's
manifest-transfer design (`{relative_path, size, sha256}`, per S0's own note) should account for
both: (a) don't rely on `huggingface_hub`'s completeness semantics for presence checks, and (b)
either dereference symlinks before transfer or ship blob content out-of-root when the source is an
HF-cache-style symlink farm.

## Rig state left behind

- **TB aliases**: already up before this session started (not re-applied — verified present,
  untouched). Left up (S2/S3 want them). Rollback if ever needed:
  `sudo ifconfig en2 -alias 10.0.2.1` (local); `ssh Jasons-Mac-Studio.local 'sudo ifconfig bridge0
  -alias 10.0.2.2'` (Studio).
> **Update after the jaccl follow-up run (orchestrating session):** the cluster daemons were restarted
> on `4b0f0d49` (jaccl wired) for the jaccl acceptance, then **cleanly stopped** — both cluster ports
> (:8910, :8911) confirmed DOWN, zero `rank_worker`/`omlx serve` on either node. The daily-driver
> oMLX.app server (PID 87617, port :8899) was verified alive and untouched throughout. `cluster.backend`
> restored to `"ring"` on both nodes (known-good idle default). Studio checkout advanced to `4b0f0d49`.
> The "PID 84865" mystery process P3 flagged was transient — gone (with its parent) by the time it was
> checked, nothing killed. Section below reflects P3's end-of-ring-run state; deltas are as noted here.

- **Studio git checkout**: `feat/cluster-v1` @ `4b0f0d49` (jaccl wiring), left checked out (S3 needs this code; prior
  branch was `feat/cluster-distributed-serving`, recorded for rollback:
  `ssh Jasons-Mac-Studio.local 'cd ~/Developer/Repos/omlx && git checkout
  feat/cluster-distributed-serving'`). Three untracked S0 spike files that collided with checkout
  (byte-identical to the tracked versions) were moved to `~/omlx-cluster-dev-backup/` on the Studio
  before checkout — restorable via `cp ~/omlx-cluster-dev-backup/*.py
  ~/Developer/Repos/omlx/benchmarks/cluster_spike/` if ever needed (they'd be untracked again on the
  old branch, matching their original state).
- **Transferred models**: left on the Studio (acceptance anchors for S6, per the brief's own
  suggestion) — `~/Models/mlx-community/MiniMax-M2.7-3bit` (93GB) and
  `~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-bf16/snapshots/e3d7ee20c3abdb072783ea696b2fe044aa85bf89`
  (51GB). Removal commands if ever needed: `rm -rf
  ~/Models/mlx-community/MiniMax-M2.7-3bit`; `rm -rf
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-bf16/snapshots/e3d7ee20c3abdb072783ea696b2fe044aa85bf89`.
- **Settings**: `~/omlx-cluster-dev/settings.json` written on both nodes (head port 8910, worker port
  8911, separate from the daily-driver `~/.omlx` base path and its bundled server ports). Left in
  place; harmless idle config.
- **Formation/rank processes**: swept — confirmed `active_model: null`, zero `rank_worker` processes
  on either node after the final unload.
- **Cluster-dev daemons** (`omlx serve` on 10.0.2.1:8910 and 10.0.2.2:8911): were **not started by
  this executor** (started via the orchestrating session's own background Bash, since a persistent
  server is a long-running process this executor's contract forbids starting directly) and were
  **not stopped by this executor**, for the same reason — stopping them is the owning session's
  action to take cleanly. PIDs at time of writing: local `omlx-server` 62609 (listening on 8910,
  parent zsh 62602) plus an ssh-tunnelled Studio launcher (parent zsh 62724/62731); Studio
  `omlx-server` 26250. **An unexplained second local `omlx-server` process (PID 84865, parent 84863,
  not listening on port 8910 or any port this executor identified) was also observed** — not started
  intentionally by this executor, origin unknown, not touched; flagged for the orchestrating session
  to investigate and clean up alongside the two intentional daemons.
