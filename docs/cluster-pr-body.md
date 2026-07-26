# Cluster serving: run one model across several Macs

Adds an `omlx/cluster/` package that shards a single model across several
machines using `mlx.distributed`, so a model too large for any one Mac fits on
the group.

**Off by default** (`cluster.enabled = false`). The daemon's only contact with
the package is `bootstrap.install()` at startup, which returns immediately when
the setting is off. Single-node behaviour is unchanged.

**Scope:** this is the foundation — the launch contract, the rank worker,
topology detection, preflight and discovery, all hardware-verified. It does
**not** include scheduler or KV-cache integration, and no HTTP request is
served by a cluster yet. See "What is NOT verified" for the precise boundary.

## Why this shape

MLX ships the mathematics and the model plumbing for distributed inference.
What it does not ship is the product around it: discovery, coordination,
lifecycle, failure handling. oMLX already runs a daemon on every machine, which
is exactly the missing piece — so the cluster control plane is the daemon
itself.

`mlx.launch` is only SSH plus environment variables plus exec. oMLX sets those
variables itself, which removes SSH, the shared-filesystem assumption, and the
identical-script-path assumption in one go. The launcher is treated as a
reference implementation, not a dependency.

Two design decisions are worth calling out because they are not obvious:

**Rank 0 is a child process, not the API process.** A distributed session
cannot be torn down and re-created inside a process — repeated init/teardown
exhausts kernel protection domains and the only recovery is a reboot. If the
daemon held the session, swapping the distributed model would restart the whole
server and evict every *local* model serving alongside it. Making respawn the
supported path keeps the daemon's uptime independent of the cluster's.

**Every `mlx_lm` distributed call goes through one adapter module.** That
surface is Apple's current headline feature and is moving fast, so
`sharded_load` and the server-loop internals are treated as unstable API with a
single blast radius.

## What is verified on hardware

Measured on an M5 Max MacBook Pro and an M3 Ultra Mac Studio, macOS 27.0,
mlx 0.32.0 / mlx-lm 0.31.3.

| Claim | Evidence |
|---|---|
| Tensor-parallel sharding is real | 51M llama → 34.09M/rank at world 2, 25.24M/rank at world 4 |
| Ranks stay in lockstep | identical tokens on every rank at world 1, 2 and 4 |
| Sampling is bit-identical | synced seed → identical draws `0.339957, 0.229705, 0.980378` |
| Ranks launch without `mlx.launch` | own hostfile + env vars, no SSH |
| **Two real machines serve one model** | MacBook rank 0 + Studio rank 1, loads in 3.2 s, output byte-identical to single-node |
| Discovery finds peers | MacBook found the Studio in 10 s, read back "Apple M3 Ultra, 96GB" |
| Preflight is correct both ways | MacBook → `jaccl`; Studio → `ring` naming all three blockers |
| Topology matches real cabling | domain-UUID cross-match verified bidirectionally on a real cable |
| Idle followers survive | 7 s blocked `recv` on `stream=mx.cpu` completes |

122 unit tests. Topology tests use real `system_profiler` captures from the two
cabled machines. The suite is mutation-checked — breaking the `bridge0`
conflict rule fails exactly the test that covers it.

## What is NOT verified

**JACCL has not been run.** It requires RDMA armed from the Recovery OS on
every node; only one of the two available machines has it. The JACCL code path
is written, unit-tested and dormant. Everything hardware-verified above used
the TCP `ring` backend, which exercises the same launch contract, the same
lockstep loop, and the same sharding.

**The bus-to-RDMA-device mapping is positional.** It matches reality on the one
cabled machine (`bus_0 → en1 → rdma_en1`, confirmed against the only active
interface) but a multi-cable machine is unverified. A missing device yields a
null matrix entry and downgrades the backend rather than launching a run that
would hang.

**Scheduler and paged-cache integration is not in this PR.** The worker runs
its own lockstep greedy decode loop, independent of oMLX's scheduler. Nothing
routes an HTTP request to a cluster yet: `bootstrap.install()` starts peer
discovery and no more. Concretely, still to build:

- rank-aware admission, batch composition, prefill segmentation and eviction,
  all decided on rank 0 and broadcast;
- the paged KV cache made lockstep-safe, and `(world_size, rank, parallelism)`
  folded into the SSD cold-tier block signature so a 4-node cluster's blocks
  are not reused by a 2-node one;
- the abort protocol, request routing, and the admin UI.

The prerequisite for that work is an audit of every local-only divergence
source in the 10.9k-line scheduler — local memory pressure, wall-clock
decisions, local cache hits, iteration order. That audit has not been done.

## Findings that contradict the common understanding

These were read out of the shipped binaries and confirmed by running them.

**Neither `ring` nor `jaccl` supports `Group.split()`.** Verified on `ring`
with four ranks; the same error string is in `libmlx.dylib` for `jaccl`. So
`sharded_load`'s `pipeline_group` and `tensor_group` can never both be real
subgroups — a run is tensor-parallel across the whole world *or*
pipeline-parallel across the whole world. **2D tensor × pipeline parallelism is
impossible on these transports.** Only MPI can split.

**`rdma_ctl enable` outside Recovery prints an error and still exits 0.** Parse
the output, never the exit status.

**`ibv_devices` succeeds with RDMA disarmed** and prints only its header, so an
empty device list is the signal rather than a failure.

**Internet Sharing over Thunderbolt is what creates `bridge0`** — the
documented RDMA conflict. Turning Internet Sharing off drops the bridge's IP
addresses but leaves the Thunderbolt interfaces enslaved, so the conflict
outlives the setting that caused it.

**Two different files are both called a "hostfile".** What
`mlx.distributed_config` writes is a cluster description; what `MLX_HOSTFILE`
must contain for `ring` is `[["ip:port"], …]` in rank order. Confusing them
fails with a JSON parse error inside `init()`.

**Rank 0 must be listening before peers connect,** or the peer dies with
`[ring] Couldn't connect (error: 65)` — which reads exactly like a firewall
fault and is not one.

## Correctness argument

Rank 0 makes every decision and broadcasts it; all ranks step together. The
invariant:

> No rank may branch on state only it can see.

Local free memory, local cache hits, wall-clock time and set iteration order
are all divergence sources. Sampling is deliberately done on every rank rather
than broadcast — logits are already all-reduced and the RNG is synced, so each
rank independently draws the same token, one collective cheaper.

## Review notes

- `omlx/cluster/` is self-contained; no existing module is modified.
- Reviewing order: `hostfile.py` (the contract) → `mlx_adapter.py` (the
  constraints) → `worker.py` (the loop) → `launcher.py` → `topology.py` /
  `preflight.py` / `discovery.py`.
- `docs/cluster-serving.md` carries the operational detail.

## Follow-ups

1. Rank-aware scheduler and paged-cache integration, including folding
   `(world_size, rank, parallelism scheme)` into the SSD block signature so a
   4-node cluster's cold blocks are not reused by a 2-node one.
2. Abort protocol — broadcast an abort set at batch-step boundaries. mlx-lm
   raises `NotImplementedError` here; oMLX's scheduler is better placed to do
   it.
3. Admin UI for the topology graph and preflight blockers.
4. JACCL validation once a second machine has RDMA armed, then shard
   pre-staging over JACCL's send/recv file transfer.
