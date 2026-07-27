# Cluster serving: run one model across several Macs

Adds an `omlx/cluster/` package that shards a single model across several
machines using `mlx.distributed`, so a model too large for any one Mac fits on
the group.

**Off by default** (`cluster.enabled = false`). The daemon's only contact with
the package is `bootstrap.install()` at startup, which returns immediately when
the setting is off. Single-node behaviour is unchanged.

**Scope:** `/v1/chat/completions` is served by a cluster with **continuous
batching** — concurrent requests join a shared batch that every rank steps in
lockstep — and **rank death fails fast**: every daemon runs a deathwatch, so a
dead rank fails in-flight requests in seconds and the next request re-forms
the cluster. There is still no prefix cache on this path (a local cache hit is
exactly the state ranks may not branch on); "What is NOT in this PR" gives the
precise boundary.

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
| Formation is quick enough to be automatic | 12 s cold (discovery → preflight → topology → both ranks → sharded load); ~2 s for the next request |
| Streaming works token by token | SSE deltas arrive one token at a time from the remote shard |
| **Requests batch across the two machines** | 3 concurrent API requests, ~400 tokens total, wall time = the longest one alone — **0.8 s on `jaccl`**, 39.5 s on `ring` — all outputs correct |
| A request joins a running batch | submitted mid-decode of another request, admitted between steps, both stream interleaved |
| Per-request abort leaves the batch serving | one of two concurrent requests aborted at 20 tokens with `finish_reason: abort`; the other ran to completion |
| Client disconnect frees only its slot | dropped a streaming 2000-token request at 5 s; ranks idle within seconds, an immediate follow-up served in 3 s |
| **A dead rank fails the request in seconds** | Studio's rank killed mid-generation: the in-flight request failed **1.7-2.1 s** later naming the dead rank; the next request re-formed and served in 7-10 s — verified on both `ring` and `jaccl` |
| Stop strings and seeded sampling hold in lockstep | `stop: ["gamma"]` truncated mid-stream; `temperature 0.8, seed 42` produced coherent output |
| JACCL (RDMA over Thunderbolt) forms and serves batched | `auto` forms directly on `jaccl` and the whole battery above runs on it; device selection resolves the physical receptacle, because a Studio Display daisy-chained to both Macs puts a second live RDMA port on each machine and position picked the display's port |

258 cluster unit tests, including a lockstep pair test: a real leader and
follower loop linked only by queue-built collective semantics must make
identical admissions, evictions and step counts — and compute the identical
reply stream. Topology tests use real `system_profiler` captures from the two
cabled machines, and the `dns-sd` parser tests a line captured verbatim at
09:04.

## How batching stays correct — and what it dodges

Batching runs mlx-lm's `BatchGenerator` in lockstep on every rank
(`omlx/cluster/batching.py`): identical generators, identical event streams,
**no per-token collective at all**. Each step costs one small collective
agreeing how many events rank 0 is holding (request arrivals, aborts — the
only state peers cannot see), and a broadcast of them only when there are any.
Everything downstream — who joins, who leaves, which token is drawn — follows
identically on every rank, because admission order, batch caps, token-id stop
machines and the synchronised RNG are all deterministic functions of the
agreed inputs.

Three upstream landmines were measured on the way (mlx 0.32.0 / mlx-lm
0.31.3, pure mlx-lm reproductions with no oMLX code), and the design routes
around each:

- **The generator's prompt processing deadlocks a sharded world in every
  shape it offers** — padded multi-prompt batches, serial prefill, prefill
  overlapping decode, even mlx-lm's own server stream configuration: 0/5
  each, while decode-only batching measured 5/5. So prompts are prefilled by
  hand (one plain forward per chunk, fully evaluated) and sequences enter the
  generator one token from decoding.
- **Control collectives racing in-flight model collectives deadlock the
  ring** — each stream has its own issuing thread, so the cross-rank order
  diverges. The loop drains the generation stream before each per-step sync.
  (Moving the sync onto the model's stream is not an option: ring `AllReduce`
  has no GPU implementation.)
- **Evicting a sequence outside the generation stream wedges the survivors** —
  a natural finish filters inside `next()` under the generation stream, so
  evictions are performed under that same stream.

## What is NOT in this PR

**The scheduler embed.** The cluster batches, but it is not oMLX's scheduler,
so there is **no prefix cache** on this path (every request starts from a
fresh cache) and no memory-aware admission (the batch is capped by
`cluster.max_batch_size`, default 8, agreed by every rank via the load
command). The audit that prices out the scheduler route — six machine-local
memory gates and one cache-lookup field, each able to deadlock the collective
— is `docs/cluster-scheduler-divergence-audit.md`; it remains the
specification for the day the cluster wants prefix-cache reuse and
memory-aware admission, including folding `(world_size, rank, parallelism)`
into the SSD block signature.

**Rings of more than two RDMA machines.** Device selection is no longer
guesswork — each node resolves its cabled buses to devices through the
physical receptacle, verified on machines carrying two live Thunderbolt links
each — but a `jaccl-ring` of three or more Macs has never been formed. A
missing device yields a null matrix entry and downgrades the backend rather
than launching a run that would hang.

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
  constraints) → `batching.py` (the lockstep loop and its correctness
  argument) → `worker.py` (the control plane around it) → `launcher.py`
  (supervision and the deathwatch) → `manager.py` (formation and the reply
  router) → `topology.py` / `preflight.py` / `discovery.py`.
- `docs/cluster-serving.md` carries the operational detail.

## Follow-ups

1. Scheduler embed for prefix-cache reuse and memory-aware admission
   (specified in `docs/cluster-scheduler-divergence-audit.md`), including
   folding `(world_size, rank, parallelism scheme)` into the SSD block
   signature so a 4-node cluster's cold blocks are not reused by a 2-node one.
2. Upstream the mlx-lm findings: sharded-world deadlocks in
   `PromptProcessingBatch` (all shapes) and in cross-stream
   eviction/collective ordering, each with a two-rank reproduction.
3. Shard pre-staging over JACCL's send/recv file transfer.
