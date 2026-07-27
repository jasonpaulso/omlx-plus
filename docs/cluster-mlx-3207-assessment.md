# Assessment: mlx#3207 and its bridge0 comment

Two documents, read against our implementation on 2026-07-27 (mlx 0.32.0,
macOS 27.0 on both nodes):

- **The issue** (`ml-explore/mlx#3207`, closed): a guide to RDMA file transfer
  over Thunderbolt 5 with JACCL, 3.5–3.8 GB/s, plus three workarounds and a
  macOS bridge fix.
- **The comment** (`qubitcontracting`, 2026-04-03): bridge0 does *not* need to
  be destroyed; assigning IPs to the individual Thunderbolt interfaces is
  enough, and JACCL coexists with bridge0 at 7.4 GB/s across three nodes.

Their hardware is a 3-node TB5 full mesh on macOS 26.4 / mlx 0.31.1 with SIP
disabled. We are on 0.32.0 and macOS 27.0, two nodes.

## What is verified here, and what is not

**Verified in our tree.** We already apply their Workaround 1. Every collective
in `omlx/cluster/mlx_adapter.py` passes `stream=mx.cpu`, so the Metal command
buffer timeout that kills their receivers cannot reach ours. We arrived at this
independently; it is the same fix.

**Verified as not applicable.** Workaround for `#3149` — consecutive send/recv
with differing shapes producing wrong data or hangs — does not touch the
serving path, because that path uses `all_sum` only and never `send`/`recv`. It
*would* apply the moment we move a file with mlx primitives.

**Verified still open.** `#3142` (GPU locking with `METAL_FAST_SYNCH=1` and
JACCL) and `#3149` are both open today, against a newer mlx than theirs. These
are live, not historical.

**Not verifiable here.** The bridge0 claim itself. Neither node currently has a
`bridge0`, and creating one is a network settings change — the operator's, not
ours. Their evidence is decent (PORT_ACTIVE on individual devices, a three-way
`init()` succeeding, throughput matching the bridge-destroyed case) and their
mechanism is plausible: RDMA reaches the interfaces through `ibv_*`, below the
bridge's network stack. It is one report, on a different macOS, and our own
preflight docstring records enslavement behaviour "observed on macOS 27.0" that
their macOS 26.4 may not share.

## The conflict this exposes in our own code

This is the finding that matters immediately, and it is ours, not theirs.

`preflight.rdma_ready` requires `not self.bridged_interfaces`. A node whose
Thunderbolt interfaces sit in `bridge0` is declared not RDMA-capable, and
`best_backend` demotes the cluster to the TCP ring. Their numbers for what that
costs: a 235B model "completely unusable" on ring, fine on JACCL.

Meanwhile the transfer path we just shipped tells the operator that enabling
**Thunderbolt Bridge** is what gives the cable an IP address and makes a
model transfer fast. Following that advice creates `bridge0`, which trips the
gate, which drops inference to TCP.

So we currently hand the operator a choice between a fast transfer and a fast
cluster. If the comment is right, that choice is an artifact of our gate rather
than a property of the hardware.

## The change to make, if it holds

Stop using bridge membership as the gate and use the evidence we already
collect. `preflight.run()` already reads `ibv_devinfo` and records
`rdma_active_devices` — devices in `PORT_ACTIVE`. That is ground truth for
whether RDMA can carry traffic; bridge membership is a proxy for it, and the
comment is a report of the proxy being wrong.

Concretely: drop `not self.bridged_interfaces` from `rdma_ready`, keep the
`thunderbolt_bridge_off` check as a **warning** with its remedy text, and let
`rdma_active_devices` decide. The downside if the comment is wrong for macOS 27
is bounded and already handled — formation attempts jaccl, hangs its connect
window, and falls back to ring (`_may_fall_back`), which is the behaviour we
already see and already log.

The safest sequence is to have the operator enable Thunderbolt Bridge on both
nodes and read our own preflight: if `rdma_active_devices` is non-empty with
`bridged_interfaces` also non-empty, the comment holds on our OS and the gate is
provably too strict. That is a five-minute check and it needs no code change to
run.

## The durability finding

The one worth taking most seriously is not about bridges.

> Each `mx.distributed.init()` / teardown cycle allocates and (incompletely)
> releases RDMA protection domains. The kernel has a hard limit. ... the only
> fix is to **reboot the affected node**.

Our design re-forms the cluster more often than a training job would: on every
model load after an idle teardown, on rank-death recovery, and — since the
configuration surface landed — on every configuration change, because applying
one tears the cluster down and stands it back up.

If protection domains leak per formation and are not reclaimed when the rank
process exits, then a long-lived node accumulates them until JACCL stops
initialising, and the recovery is a reboot. That is a slow-burn failure that
would present as "clustering worked for a week and then stopped", which is the
worst shape a bug can have.

Two things follow regardless of whether the leak is confirmed:

1. **Do not re-form for changes that do not require it.** The hot-apply path
   currently tears down unconditionally. A change to `max_batch_size` or the
   discovery interval does not need the ranks restarted the way a change of
   model or backend does. Narrowing that is free and reduces cycle count.
2. **Count formations since boot and show it.** If the limit is real, the
   operator needs to see the number climbing before it is hit, not discover it
   at the failure.

Confirming the leak means forming and tearing down in a loop until JACCL
refuses. The failure mode of that experiment is a reboot of both machines, so
it is the operator's call to run, not something to slip into a verification
pass.

## For the transfer path

Their measured ceiling, on hardware like ours:

| Path | Throughput |
|---|---|
| RDMA over TB5 via JACCL | 3.5–3.8 GB/s |
| rsync over TB5 **IP** | 300–500 MB/s |
| rsync over 10GbE | 100–200 MB/s |
| our HTTP transfer over the LAN, measured | ~40 MB/s |

Two separate gains are available, and they are not the same size. Giving the
Thunderbolt link an IP address moves us from ~40 MB/s toward the 300–500 MB/s
row — roughly 10x, and it needs no code from us. Going to the RDMA row is
another 10x on top, and it means moving bytes with `mx.distributed.send`/`recv`
inside a single JACCL session, which drags in `#3149` (open), the
protection-domain limit above, and a second code path that has to coexist with
the one serving inference on the same devices.

The first is worth taking now. The second is a project, and it should not start
until the protection-domain question has an answer, because it is precisely the
"separate JACCL session per file" shape their guide warns hits the limit
quickly.

Their other operational notes worth keeping: Mac Studio **Receptacle 1** was
unreliable for RDMA and moving the cable fixed it — which is adjacent to the
receptacle bug we already fixed in `af838796` — and macOS re-enables Thunderbolt
Bridge on every reboot and OS update, which is why our check runs on every
formation rather than once at setup.
