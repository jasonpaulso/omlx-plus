# Cluster serving

Running one model across several Macs, so a model that does not fit on any
single machine fits on the group.

The feature is off by default. With `cluster.enabled = false` the daemon's only
contact with `omlx/cluster/` is `bootstrap.install()` at startup, which returns
immediately; single-node behaviour is unchanged.

`/v1/chat/completions` is served by a cluster today, one request at a time. The
rank-aware scheduler that would let a cluster batch is specified in
`docs/cluster-scheduler-divergence-audit.md` and is not built; the section
"What one-at-a-time means" below says exactly what that costs.

## What this is built on

MLX ships the mathematics and the model plumbing for distributed inference.
What it does not ship is the product around it: discovery, coordination,
lifecycle, cache, failure handling. oMLX already runs a daemon on every
machine, which is exactly the missing piece — the cluster control plane is the
daemon itself, so there is no SSH, no shared filesystem, and no
`mlx.launch`.

`mlx.launch` is only SSH plus environment variables plus exec. oMLX sets those
variables itself. The launcher is a reference implementation, not a dependency.

## Transports

| Backend | Needs | Latency |
|---|---|---|
| `jaccl` | RDMA over Thunderbolt 5, full mesh cabling | single-digit µs |
| `jaccl-ring` | RDMA over TB5, ring cabling | single-digit µs |
| `ring` | nothing but an IP route | ~300 µs |

`ring` is the fallback and works on any Mac over wifi or ethernet. That is not
only a degraded mode — it is what makes the feature usable on Thunderbolt 4 and
older machines, and what the whole stack is developed against.

## Measured constraints

Everything below was measured on mlx 0.32.0 / mlx-lm 0.31.3 against an M5 Max
MacBook Pro and an M3 Ultra Mac Studio, not taken from documentation. Several
contradict what the surrounding literature implies.

**Neither backend supports group splitting.** `Group.split()` raises
"Group split not supported" on `ring` (verified with four ranks) and the same
string is present in `libmlx.dylib` for `jaccl`. So `sharded_load`'s
`pipeline_group` and `tensor_group` can never both be real subgroups: a run is
tensor-parallel across the whole world *or* pipeline-parallel across the whole
world. **2D tensor × pipeline parallelism is impossible on these transports.**
Only MPI can split, and MPI is not offered here.

**One distributed session per process, forever.** Repeated init/teardown
exhausts kernel protection domains and the only recovery is a reboot. There is
therefore no teardown path in the code. Every rank — including rank 0 — is a
child process of a daemon, so a model swap kills children and spawns new ones
while the daemon keeps its uptime, its admin UI, and its local models.

**Long waits must use the CPU stream.** A `send`/`recv` that blocks longer than
the ~5 s Metal command-buffer timeout kills the waiting rank on the default GPU
stream. Verified: a 7 s idle `recv` completes on `stream=mx.cpu`. Every
potentially-blocking collective is pinned there.

**Init needs a real barrier.** RDMA resources are allocated lazily, so an
`all_sum` over a non-trivial array is issued straight after `init()`.

**Rank 0 must be listening before peers connect.** The ring backend's connect
window expires and the peer dies with `[ring] Couldn't connect (error: 60)`.
Start order is not incidental; the launcher waits for rank 0's socket.

**Stale ranks poison the next run.** A worker left alive from a failed run
holds the ring port, and the next rank 0 silently fails to own it. Teardown
escalates to kill for this reason.

**`ring` recv works only between direct neighbours**, and `jaccl` has no
`sum_scatter`.

**Never probe a rank's port to see whether it is up.** The ring backend accepts
the probe, takes it for the peer it is waiting on, and the real peer's
handshake never completes; both ranks then sit there until they time out with
`[ring] Couldn't connect (error: 60)`, on a network where nothing is wrong.
Readiness is read from the process table instead (`LocalCluster._is_listening`).
This cost an afternoon to find, because a `nc` from the other machine
*succeeds* - and by succeeding, breaks the next attempt too.

**A Mac with a Thunderbolt cable publishes several A records** for its `.local`
name, and the first is routinely a link-local `169.254.x.x` on a bridge nobody
serves on. A rank handed that address binds where its peers cannot see it -
error 60 again. Addresses are chosen by evidence: whichever candidate accepts a
TCP connection on the peer's *daemon* port.

**Resolving a `.local` name from inside the daemon takes over a minute.**
Long enough that rank 0 spent its whole connect window waiting for an HTTP
request to leave the machine. The control plane addresses peers by IP.

**Two "hostfiles" exist and are not the same file.** What
`mlx.distributed_config` writes is a cluster description
(`{"backend": ..., "hosts": [...]}`). What `MLX_HOSTFILE` must contain for the
ring backend is `[["ip:port"], ["ip:port"]]` in rank order. Passing one where
the other is expected fails with a JSON parse error inside `init()`.

## The launch contract

A rank process needs only environment variables. Both `MLX_`- and
`JACCL_`-prefixed spellings are accepted; oMLX emits the `MLX_` forms.

```
ring:    MLX_RANK, MLX_HOSTFILE
jaccl:   MLX_RANK, MLX_JACCL_COORDINATOR=<rank0_ip:port>,
         MLX_IBV_DEVICES=<path to N×N json>, MLX_JACCL_RING=1 (ring cabling)
always:  MLX_METAL_FAST_SYNCH=1
```

The worker environment is scrubbed of the daemon's own `OMLX_BASE_URL` and
`OMLX_API_KEY` so a worker can never be pointed back at the server that spawned
it, and of any stale distributed variables so it cannot half-inherit a previous
topology.

## Topology discovery

Each node reports its own `system_profiler SPThunderboltDataType` over the
control plane; the leader assembles the graph. Nodes are matched by Thunderbolt
**domain UUID**: a bus lists its peer's own domain UUID, so an edge exists when
that UUID equals another node's bus domain. Verified bidirectionally on a real
cable:

```
macbook bus_0  domain 817DCFA4…  sees peer domain 28CA4C30…
studio  bus_1  domain 28CA4C30…  sees peer domain 817DCFA4…
```

Displays, drives and docks report a null peer domain and drop out of the graph
by themselves, so no device allow-list is needed.

From the graph: every pair cabled → `jaccl`; a Hamiltonian cycle →
`jaccl-ring`; anything else, or any node not RDMA-ready → TCP `ring`. RDMA over
Thunderbolt cannot route, so a missing cable downgrades the whole cluster
rather than costing one hop.

## Preflight

RDMA over Thunderbolt needs macOS 26.2+, TB5 silicon, RDMA armed once per
machine from the **Recovery OS**, and Thunderbolt Bridge switched off.

Two traps worth knowing:

- `rdma_ctl enable` outside Recovery prints an error **and still exits 0**.
  Parse the output, never the exit status.
- `ibv_devices` succeeds with RDMA disarmed and prints only its header, so an
  empty device list is the signal, not a failure.
- Enabling **Internet Sharing over Thunderbolt** is what creates `bridge0`.
  Turning Internet Sharing back off drops the bridge's IP addresses but leaves
  the Thunderbolt interfaces enslaved, so the RDMA conflict outlives the
  setting that caused it. The bridge also returns after a reboot, which is why
  preflight runs on every cluster formation rather than once at setup.

## Serving a request

```
POST /v1/chat/completions          leader daemon
        |                               |
        v                               v
   ClusterEngine  ---- token ids ---> rank 0 worker  <=== collective ===> rank 1
        ^                               |
        +--------- text chunks ---------+
```

`ClusterEngine` subclasses `BatchedEngine`, so chat templating, tool
conversion, Harmony preprocessing and thinking kwargs are the *same code* that
serves local models — all of it is tokenizer work, and none of it cares where
the weights are. What is replaced is only the two methods that touch weights:
`start` loads a tokenizer instead of a model, and `generate` /`stream_generate`
push token ids down a pipe instead of into a local scheduler.

The prompt crosses the pipe **already tokenized**. The daemon holds the
tokenizer it used to render the chat template, and re-encoding the string
inside each worker would be one more chance for the ranks to disagree, for
nothing.

The engine is selected by model id: `cluster.model` names the one model this
fleet shards. Every other model on every node keeps loading and serving
locally, out of the same pool, at the same time.

**Local memory admission is skipped for that model, deliberately.** Its weights
never enter the daemon process — rank 0 is a child process holding one shard —
so checking it against this machine's ceiling would refuse exactly the models
clustering exists to serve. The trade is that the operator, not the pool, owns
the headroom for the shard.

### What one-at-a-time means

`ClusterManager.stream` holds a lock for the whole generation. This is
correctness, not a tuning parameter: under tensor parallelism every rank must
run the same forward pass, so overlapping two requests means every rank
agreeing, at every step, which sequences advance. That is the rank-aware
scheduler, and it does not exist yet. A second concurrent request would not be
slower, it would be wrong — so it queues.

There is no prefix cache on this path either, for the same reason: a local
cache hit is exactly the state ranks may not branch on. Each request starts
from a fresh `make_prompt_cache`.

## Aborting a running request

Rank 0's stdin is occupied for the whole of a generation — it is inside the
decode loop and cannot go back and read another command — so an abort needs
somewhere else to arrive. It gets a second inherited pipe, polled between
tokens, and the result is agreed through the collective like any other stop
decision. Peers never hear about the abort directly; they hear rank 0's verdict.

A client that disconnects mid-stream therefore does stop the cluster, which
matters more here than locally: until those ranks leave the decode loop,
nothing else can be served at all.

## The control plane

Every daemon serves the peer half:

| route | who calls it | what it does |
|---|---|---|
| `POST /cluster/report` | a forming leader | topology, RDMA state, *and* whether this node has the model and an interpreter that can import oMLX |
| `POST /cluster/ranks/start` | a forming leader | spawn this node's ranks |
| `POST /cluster/ranks/stop` | a leader, or recovery | kill them |
| `GET /admin/api/cluster/status` | the admin UI | formation state, backend, rank order, peers |
| `GET /admin/api/cluster/preflight` | the admin UI | this node's checklist |
| `POST /admin/api/cluster/teardown` | an operator | stop everything, everywhere |

The peer routes authenticate with the shared `cluster_key`, not the node's API
key. They are not the same permission: a cluster key says "you may make this
machine a rank", not "you may use this machine's models".

`/cluster/report` answers the model and interpreter questions *before* anything
is spawned on purpose. Both failures are otherwise invisible: a rank whose
interpreter cannot import oMLX dies before it listens, and the **other**
machine reports `[ring] Couldn't connect (error: 65)` — which reads as a
firewall fault and is not one.

## Correctness under lockstep

Rank 0 makes every scheduling decision and broadcasts it; all ranks step
together. The rule that keeps this correct:

> No rank may branch on state only it can see.

Local free memory, a local cache hit, wall-clock time, and set iteration order
are all divergence sources. Anything derived from them must be decided by rank
0 and broadcast, or agreed through a collective.

Sampling is deliberately done on every rank rather than on rank 0 and
broadcast. Under tensor parallelism the final logits are already all-reduced,
and `seed_everyone` synchronises the RNG, so every rank independently draws the
same token — one collective cheaper per token. Verified: two ranks produce
identical draws (`0.339957, 0.229705, 0.980378`) and identical output text.

## What is gated off

Speculative decoding, dflash, MTP, draft models, and adapters all have
rank-divergence or upstream blockers. VLM, embedding and reranker engines have
no sharding support and keep serving **locally** on each node, which multi-model
makes natural.

## Failure handling

JACCL has no fault tolerance: a dead rank leaves peers blocked in a collective
until the Metal timeout kills them too. The response is to tear the session
down and respawn, never to recover in-process — protection-domain exhaustion
makes in-process recovery a reboot risk. Local models are unaffected
throughout.

## What has actually been run

Two Macs on a LAN - an M5 Max MacBook Pro as rank 0 and an M3 Ultra Mac Studio
as rank 1 - serving `mlx-community/Llama-3.2-1B-Instruct-4bit` tensor-sharded
over the TCP `ring` backend:

```
POST /v1/chat/completions  "Name three colours, comma separated."
  -> "Red, Blue, Yellow"   finish_reason "stop", 43 prompt / 5 completion
```

Formation (discovery, preflight, topology, both ranks, sharded load) takes
about 12s cold; the second request on the formed cluster returns in 1.0s.
Streaming arrives token by token. A client that disconnects mid-generation
aborts the run, and the next request is served ~3s later - the ranks left the
decode loop rather than finishing 2000 tokens nobody was reading.

Reproduce with two daemons that have `cluster.enabled`, the same
`cluster.cluster_key`, and the same model id on their own disks. One trap that
is *not* oMLX's: starting a peer daemon with `nohup` over an ssh session that
then closes leaves it unable to complete TCP connections from its children.
Start it with `ssh -f` and no `nohup`.

## Testing without a second Mac

Ranks are addressed by `ip:port`, and colocated ranks share an IP. A full
cluster therefore runs on one machine over `127.0.0.1`, which is how most of
this was developed:

```python
from omlx.cluster.launcher import LocalCluster

c = LocalCluster(model_path="…", world_size=2, backend="ring")
c.start(ranks=[0, 1])
c.command({"op": "load"})
for reply in c.stream({"op": "generate", "prompt": "…", "max_tokens": 32}):
    ...
c.stop()
```
