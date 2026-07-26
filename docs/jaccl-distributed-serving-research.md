# Distributed inference on Apple Silicon — state of play & a turn-key JACCL design for oMLX

*2026-07-26. Sources verified against mlx `main` and mlx-lm v0.31.3 source (cloned today), MLX 0.32.0 docs, WWDC26 session 233, Apple TN3205.*

## 1. Your read is correct — and here's exactly where the line is

Distributed inference **is** shipped in mlx-lm today. What's shipped is the *math and the model plumbing*. What's absent is everything oMLX would call product: discovery, coordination, lifecycle, cache, multi-model, failure handling. That gap is precisely the opportunity.

### What MLX core provides (v0.32)

Four backends behind one API (`mx.distributed.init(backend=...)`): `ring` (TCP, always available), `jaccl` (RDMA over TB5), `jaccl-ring`, `mpi`, `nccl`. First successful init is sticky. Ops: `all_sum/all_max/all_min/all_gather/send/recv/sum_scatter`; all no-ops at world size 1.

**JACCL** is vendored *inside* the mlx repo (`mlx/distributed/jaccl/lib`) — C++ lib, mesh + ring topologies, single-digit-µs latency vs ~300µs TCP. Requirements are hard:

- **macOS 26.2+** (RDMA over Thunderbolt introduced there; TN3205 is the authoritative technote)
- **Thunderbolt 5 machines** (M3 Ultra, M4 Pro/Max class and later; TB4 has no RDMA)
- RDMA enabled once per machine in **Recovery mode**: `rdma_ctl enable`, verify with `ibv_devices`
- **Full mesh cabling** for the `jaccl` mesh backend (N·(N−1)/2 cables — 6 cables for 4 nodes); `jaccl-ring` for ring cabling. RDMA over TB **cannot route** — the app layer forwards, hence topology matters.
- TN3205 constraints worth knowing: verbs send/recv only, max 10 UC queue pairs, 16 MB max message, 4095 outstanding WRs.

### The init contract is trivially small (the turn-key key insight)

`mlx.launch` is *nothing but SSH + env vars + exec*. From `mlx/_distributed_utils/launch.py`, a JACCL rank process needs exactly:

```
MLX_RANK=<n>
MLX_JACCL_COORDINATOR=<rank0_ip>:<port>     # TCP bootstrap: GID/QP exchange, then RDMA takes over
MLX_IBV_DEVICES=<path to json>              # N×N matrix of rdma_enX device names, null on diagonal
MLX_JACCL_RING=1                            # optional: ring instead of mesh
```

(`JACCL_*` variants also accepted.) The ring backend needs just `MLX_HOSTFILE` + `MLX_RANK`. **oMLX can own process spawning on each node and set these directly — no SSH, no shared filesystem, no `mlx.launch` at all.** The launcher is not a dependency; it's a reference implementation.

### What `mlx.distributed_config` does (worth porting, not shelling out to)

From `mlx/_distributed_utils/config.py`: SSHes into each host, runs `system_profiler SPThunderboltDataType` to build a TB connectivity matrix, checks for `rdma_*` devices, then picks the backend by decision tree — RDMA+mesh → `jaccl`, RDMA+ring → `jaccl-ring`, else TCP `ring` — assigns link-local IPs (`--auto-setup` with sudo configures interfaces), and writes the hostfile:

```json
{
  "backend": "jaccl",
  "envs": ["MLX_METAL_FAST_SYNCH=1"],
  "hosts": [
    {"ssh": "host1", "ips": ["192.168.1.10"], "rdma": [null, "rdma_en5", "rdma_en4"]},
    {"ssh": "host2", "ips": ["192.168.1.11"], "rdma": ["rdma_en5", null, "rdma_en3"]},
    {"ssh": "host3", "ips": ["192.168.1.12"], "rdma": ["rdma_en4", "rdma_en3", null]}
  ]
}
```

The connectivity/mesh/ring analysis functions (`extract_connectivity`, `make_connectivity_matrix`, `check_valid_mesh`, `extract_rings`) are clean, self-contained logic. In an oMLX cluster each node reports its *own* profiler + ibv data over the control plane, so the SSH layer drops out entirely.

## 2. What mlx-lm provides (v0.31.3)

**`sharded_load(repo, pipeline_group, tensor_group)`** — the workhorse. Lazy-loads config, decides shardability, then downloads/loads **only the weights the local rank needs**:

- **Tensor parallel** (default): model implements `shard(group)` — 18 families today: llama, qwen2/3/3.5, deepseek v2/v3/v3.2, glm4_moe(+lite), gpt_oss, kimi_k25, minimax, step3p5, longcat_flash(+ngram), exaone_moe, ministral3, iquestloopcoder. MoE expert parallelism included.
- **Pipeline parallel** (`--pipeline`): model exposes `model.pipeline(group)`; layers split by depth, send/recv between adjacent ranks.

**`mlx_lm.server` is already a distributed server**, and its loop is the blueprint:

- rank 0 runs the HTTP server; ranks 1+ block in a worker loop (`run()` splits on `group.rank()`)
- requests are broadcast rank0→all via `_share_object`: **pickle bytes shipped through `all_sum` collectives** (size then payload)
- `TimeBudget` keeps ranks' scheduling loops in lockstep by periodically all-summing loop times
- RNG seed synced via collective so per-rank sampling is bit-identical (TP replicates final logits after all_reduce, so each rank samples the same token)

**Its limitations (your feature list, effectively):** single model per process; no adapters or draft models in distributed mode (explicitly raised); **request cancellation is `raise NotImplementedError()`** in the distributed path; simple LRU prompt cache, not paged/tiered; orchestration assumes `mlx.launch` + same script path everywhere. `BatchGenerator` itself is distributed-agnostic — correctness comes entirely from identical lockstep scheduling + deterministic sampling across ranks.

WWDC26 (session 233) demoed exactly this stack: `mlx_lm.chat` on Kimi K2.6 (1T) across 4× M3 Ultra via `mlx.launch --hostfile`, ~3× throughput vs one machine; TP as default, PP for the very largest models.

## 3. Field-reported gotchas (from mlx issues #3207, #2944)

These belong in oMLX's runtime, not its docs:

1. **Metal command-buffer timeout (~5 s, unconfigurable)** kills a rank blocked in a long GPU-stream `recv`. Workaround: long-wait send/recv on `stream=mx.cpu`. Matters for any idle-follower design.
2. **Post-init barrier**: run an `all_sum` on a non-trivial array (e.g. `mx.ones(10)`) right after `init()` to force RDMA resource setup before real traffic.
3. **Protection-domain exhaustion**: repeated JACCL init/teardown cycles in-process exhaust kernel PDs; recovery is a **reboot**. Consequence: one distributed session per process lifetime — model swap = respawn worker processes, never re-init.
4. **Thunderbolt Bridge must be disabled** (bridge0 conflicts with RDMA); it resurrects after reboot. oMLX should detect and offer the fix.
5. `MLX_METAL_FAST_SYNCH=1` is the recommended env for distributed runs.
6. Bonus: JACCL send/recv doubles as a **3.5+ GB/s file transfer** between nodes (~23× rsync over 10GbE) — useful for weight replication.

## 4. Prior art on turn-key: exo 1.0

exo (exo-explore) is active — v1.0.71, Apr 2026, 44.7k stars: zero-config auto-discovery, topology-aware auto-parallelism, day-0 RDMA-over-TB5 support, OpenAI/Anthropic-compatible API. It validates the turn-key UX you're after, but it's a whole separate stack — no tiered KV cache, no multi-model pool, none of oMLX's serving depth. The differentiator for oMLX isn't discovery per se; it's *discovery + the existing scheduler/cache stack made rank-aware*.

## 5. Proposed shape for oMLX cluster mode

Every box already runs oMLX (menubar app / brew service). That's the asset — the cluster control plane is the oMLX daemon itself.

**Discovery & formation.** Advertise `_omlx._tcp` via Bonjour (NetService — free on macOS; `utils/network.py` already does interface/alias detection). Peers appear in the admin UI; "form cluster" is one click (or automatic with a shared cluster key in settings). No SSH ever.

**Topology probe.** Each node self-reports `system_profiler SPThunderboltDataType`, its `rdma_*` device list, RAM, chip, and model inventory over the control plane. The leader ports `distributed_config`'s matrix/mesh/ring logic to compute: backend (`jaccl` → `jaccl-ring` → TCP `ring` fallback — the fallback also gives you TB4/older-macOS support for free, just slower), rank assignment, and the `MLX_IBV_DEVICES` matrix. Admin UI renders the detected TB graph and names the missing cable when mesh is incomplete (`distributed_config` even has DOT output to crib from).

**Setup automation.** Preflight per node: RDMA enabled? (else show the one-time Recovery-mode instruction) · Thunderbolt Bridge active? (offer disable) · macOS ≥ 26.2, TB5 present? · model shardable? (`hasattr(model, "shard")` check against the 18 families). Set `MLX_METAL_FAST_SYNCH=1` automatically.

**Launch.** Leader picks a coordinator port; each node's daemon spawns a **worker subprocess** with the four env vars and the right model path. Rank 0 lives on the leader and runs the normal oMLX API/admin/scheduler; followers run a thin engine that joins the collective loop. `sharded_load` per rank pulls only that rank's safetensors — the downloader can pre-stage shards per node (later: JACCL file-transfer for LAN-speed replication instead of N× internet downloads).

**Serving loop.** v1 adopts the proven mlx-lm pattern: rank 0 makes *all* scheduling decisions (admission, batch composition, prefill segmentation, eviction) and broadcasts them; every rank steps `BatchGenerator` in lockstep; seeds synced. Consider moving scheduling metadata onto the TCP control plane instead of the pickle-over-`all_sum` trick — it spends GPU collectives on control flow and couples scheduler latency to the interconnect — but that's an optimization, not a v1 requirement.

**KV cache (the real integration work).** Under TP each rank holds the KV for its head-shard, so the paged cache "just works" per-process — *iff every rank's block table mutates identically*. Route every cache decision through rank 0's broadcast; nothing rank-local may depend on local-only state. SSD cold tier: fold `(world_size, rank, parallelism scheme)` into the block signature — the existing incompatible-block/janitor machinery then handles invalidation across topology changes for free. Prefix-reuse lookups happen on rank 0 only, results broadcast.

**Gate off in distributed v1** (all have rank-divergence or upstream blockers): speculative/dflash/MTP, draft models, adapters, VLM/embedding/reranker engines (no sharding support — they keep serving *locally* per node, which multi-model makes natural). Cancellation needs a design upstream punted on: broadcast an abort set at batch-step boundaries via the control plane — oMLX's scheduler is actually better positioned to do this than mlx-lm's loop.

**Multi-model.** Simplest v1: a cluster session pins exactly one distributed model; each node's EnginePool continues serving its local small models alongside (memory enforcer already arbitrates). Cluster-wide lockstep load/unload of multiple distributed models is a v2 problem — remember PD exhaustion means swap = worker respawn, so make respawn cheap rather than re-init clever.

**Failure handling.** JACCL has zero fault tolerance; a dead rank hangs collectives and the 5 s Metal timeout then kills peers. Control-plane heartbeats detect peer loss; response is tear down and respawn the distributed engine (local models unaffected). Don't attempt in-process recovery — see PD exhaustion.

### Suggested build order

1. **Spike** (2 nodes, ring-TCP first, then JACCL if both boxes qualify): worker subprocess + env-var launch + `sharded_load` + lockstep `BatchGenerator`, no cache. Proves the loop on your hardware. *(Check hardware first: JACCL needs TB5 on both ends — Studio M3 Ultra qualifies; a pre-M4 MacBook Pro does not, though TCP ring still works.)*
2. Bonjour discovery + topology probe + preflight UI (valuable standalone even before serving works).
3. Rank-aware scheduler/paged-cache integration (the deep work).
4. SSD-tier shard signatures, abort protocol, shard pre-staging, jaccl-ring fallback polish.

One upstream note: mlx-lm's distributed serving surface is moving fast (this is Apple's current headline feature), so pin the mlx-lm version per release and treat `sharded_load`/server-loop internals as unstable API — same discipline as the existing fork-seam rules.

## Sources

- [MLX Distributed Communication docs (0.32.0)](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [WWDC26 Session 233 — Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/)
- [Apple TN3205 — Low-latency communication with RDMA over Thunderbolt (summary)](https://blog.massapi.com/posts/2026-03-18-1623-tn3205-low-latency-communication-with-rdma-over-thunderbolt/)
- [mlx PR #2808 — Thunderbolt RDMA communications backend](https://github.com/ml-explore/mlx/pull/2808)
- [mlx issue #3207 — RDMA file transfer over TB5 with JACCL (field guide)](https://github.com/ml-explore/mlx/issues/3207)
- [mlx issue #2944 — RDMA/JACCL bugs](https://github.com/ml-explore/mlx/issues/2944)
- [mlx-lm repo](https://github.com/ml-explore/mlx-lm) · [DeepWiki: mlx-lm distributed execution](https://deepwiki.com/ml-explore/mlx-lm/7.5-distributed-execution)
- [exo](https://github.com/exo-explore/exo)
- [byteiota — MLX + JACCL over TB5](https://byteiota.com/mlx-jaccl-thunderbolt-distributed-training/) · [byteiota — multi-Mac LLM clusters](https://byteiota.com/mlx-distributed-training-with-jaccl-multi-mac-llm-clusters-explained/)
- Source inspected directly: `mlx/_distributed_utils/{launch,config,common}.py`, `mlx/distributed/jaccl/lib` (JACCL README + jaccl.cpp), `mlx_lm/{server,utils,generate}.py` @ v0.31.3
