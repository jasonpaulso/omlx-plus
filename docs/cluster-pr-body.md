# Cluster serving: run one model across several Macs

Adds an `omlx/cluster/` package that shards a single model across several
machines using `mlx.distributed`, so a model too large for any one Mac fits on
the group.

**Off by default** (`cluster.enabled = false`). The daemon's only contact with
the package is `bootstrap.install()` at startup, which returns immediately when
the setting is off. Single-node behaviour is unchanged.

**Scope:** `/v1/chat/completions` is served by a cluster, one request at a
time. It does **not** include the rank-aware scheduler, so a cluster does not
batch and does not use the prefix cache — both are correctness, not tuning; see
"One request at a time". "What is NOT verified" gives the precise boundary.

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
| **A cluster serves an OpenAI request** | `POST /v1/chat/completions` on the MacBook → "Red, Blue, Yellow", `finish_reason: stop`, weights sharded across both Macs |
| Formation is quick enough to be automatic | 12 s cold (discovery → preflight → topology → both ranks → sharded load); 1.0 s for the next request |
| Streaming works token by token | SSE deltas arrive one token at a time from the remote shard |
| Abort frees the cluster | client disconnects mid-2000-token run; the next request is served ~3 s later |

202 unit tests. Topology tests use real `system_profiler` captures from the two
cabled machines, and the `dns-sd` parser tests a line captured verbatim at
09:04. The suite is mutation-checked — making a follower decide for itself
instead of taking rank 0's verdict fails five tests, and removing the abort on
client disconnect fails a sixth.

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

**The rank-aware scheduler is not in this PR.** The worker runs its own
lockstep decode loop with real sampling, stop sequences and aborts, but it is
not oMLX's scheduler — so a cluster serves **one request at a time** and starts
every request from a fresh KV cache. Both are correctness rather than tuning:
under tensor parallelism every rank must run the same forward pass, so batching
requires every rank to agree at every step which sequences advance, and a local
prefix-cache hit is exactly the state ranks may not branch on. Concretely,
still to build:

- rank-aware admission, batch composition, prefill segmentation and eviction,
  all decided on rank 0 and broadcast;
- the paged KV cache made lockstep-safe, and `(world_size, rank, parallelism)`
  folded into the SSD cold-tier block signature so a 4-node cluster's blocks
  are not reused by a 2-node one;
- batching on top of that, which is the only thing standing between this and
  a cluster that is also fast under load.

The prerequisite for that work — an audit of every local-only divergence source
in the 10.9k-line scheduler — is in `docs/cluster-scheduler-divergence-audit.md`.
It found the surface far narrower than expected: **memory-gated admission is
essentially the whole problem.** `_current_usage_bytes()` reads
`mx.get_active_memory()` and `get_phys_footprint()`, both machine-local, and
feeds six admission gates, so two ranks can decide differently about the same
request and deadlock the collective. Wall-clock, RNG and iteration order are
already benign for the reasons given there. The cache class is also now audited
and turns out to be one function and one field: `remaining_tokens`, which
decides the shape of a forward pass, so two ranks that disagree about it do not
produce slightly different answers - they issue different collectives and hang.

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

**Do not probe a rank's port to find out whether it is listening.** The backend
accepts the probe, takes it for the peer it is waiting on, and the real peer's
handshake never completes — both ranks then time out with error 60 on a network
where nothing is wrong. A `nc` from the other machine *succeeds*, and by
succeeding breaks the next attempt too. Readiness is read from the process
table instead.

**A Mac with a Thunderbolt cable publishes several A records** for its `.local`
name, and the first is routinely a link-local `169.254.x.x` on a bridge nobody
serves on. Resolving that name from inside a daemon also takes over a minute —
long enough for rank 0's connect window to expire while an HTTP request is
still trying to leave the machine. Peers are addressed by an IPv4 address
chosen by evidence: whichever candidate accepts a connection on their daemon
port.

**`dns-sd` pads the hour with a space, not a zero.** Its browse output reads
` 9:04:13.991` before 10:00, so a two-digit-hour pattern discards every line
and peer discovery finds nothing for the first ten hours of every day. It looks
exactly like a LAN with nobody on it.

## Correctness argument

Rank 0 makes every decision and broadcasts it; all ranks step together. The
invariant:

> No rank may branch on state only it can see.

Local free memory, local cache hits, wall-clock time and set iteration order
are all divergence sources. Sampling is deliberately done on every rank rather
than broadcast — logits are already all-reduced and the RNG is synced, so each
rank independently draws the same token, one collective cheaper.

## Review notes

- `omlx/cluster/` is self-contained apart from three small hunks: two lines in
  `server.py` (install/shutdown), a `ClusterEngine` branch and a skipped memory
  ceiling in `engine_pool.py`, and an English fallback in the admin `t()`.
- Reviewing order: `hostfile.py` (the contract) → `mlx_adapter.py` (the
  constraints) → `worker.py` (the loop) → `launcher.py` → `topology.py` /
  `preflight.py` / `discovery.py`.
- `docs/cluster-serving.md` carries the operational detail.

## Follow-ups

1. Rank-aware scheduler and paged-cache integration, including folding
   `(world_size, rank, parallelism scheme)` into the SSD block signature so a
   4-node cluster's cold blocks are not reused by a 2-node one.
2. Batching, which the scheduler work unlocks — the abort protocol,
   request routing, and the admin panel are in this PR.
3. A liveness timeout on a formed cluster. JACCL has no fault tolerance by
   design, so a rank that dies leaves rank 0 blocked in a collective and the
   request hangs until the client gives up.
4. JACCL validation once a second machine has RDMA armed, then shard
   pre-staging over JACCL's send/recv file transfer.
