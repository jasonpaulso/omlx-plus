# S0 — Rig Bring-up + Evidence Spike

Reference cluster: local M4 Max 128GB (`Jasons-MacBook-Pro-2.local`) + Mac Studio M3 Ultra 96GB
(`Jasons-Mac-Studio.local`, alias `192.168.5.28`), TB5 cable, `rdma_ctl` enabled both ends.
Date: 2026-07-28. Scripts: `discovery/spike/`. This file is the tracked S0 deliverable per the
program spec (`discovery/spec/cluster-v1-spec.md`, slice S0).

## Result summary vs S0 acceptance

| # | Acceptance item | Result |
|---|---|---|
| a | Per-step broadcast tax, both transports, on TP-sharded smoke model mid-decode, vs 10% budget | **PASS both** (ring ~0.15%, jaccl ~0.29% of baseline — see below; ring re-measured on an idle machine after an initial contaminated run) |
| a | Idle-rank lower bound reported separately | done, both backends, both well under budget |
| b | `mlx_lm.share` throughput + interruption behavior | done — ~4.6 GB/s; no resume, restarts from byte zero |
| c | Working ring AND jaccl init recipes | both recorded below, both formed cleanly on first attempt |

**Headline: S0's per-step broadcast tax is a small fraction of the 10% budget on both transports.**
Per E4's own stop condition, this is a spike-level lower bound (smoke-scale 1B model, synthetic
skew) — S2 re-measures under the real rank-0-drives protocol and real TP load, where heterogeneous
straggler skew will be present and is expected to raise the number. Nothing here should be read as
"S2's number will also pass" — only that nothing in the transport layer itself precludes it.

## E4 baseline (Task B)

- Model: `mlx-community/Qwen3.6-27B-bf16`. **Note:** the spec names "dense FP16"; the on-disk
  artifact is bf16. Recorded as bf16 per the actual measurement; not treated as a blocker.
- Machine: local M4 Max, single-node, batch 1, decode-only (prefill excluded from timing).
- Greedy sampling (`temp=0.0`), 320 decode-step steady-state window (16-step warmup discarded), first 256+ included.
- Script: `discovery/spike/baseline_decode.py`

```
avg ms/token: 102.682
p50 ms/token: 102.011
p90 ms/token: 105.715
tok/s (avg): 9.74
```

**10% budget = 10.2682 ms/token.**

*Caveat (found in verification): the measurement script double-prefilled the prompt into the KV
cache (~2× prompt-length KV, ~40 vs ~20 tokens, against 320+ decode steps), slightly inflating the
baseline and thus slightly shrinking the tax ratios — i.e., biasing toward PASS by a negligible
margin. The script (`discovery/spike/baseline_decode.py`) is fixed for future runs; the recorded
figure stands with this disclosure since the pass/fail outcome is insensitive to it.*

## Collective + broadcast measurements (Task C + D)

Smoke model for TP sharding: `mlx-community/Llama-3.2-1B-Instruct-4bit` (present on both machines;
`shard()`-capable per `mlx_lm.models.llama.Model`). Hand-prefilled, batch=1 — **not** routed through
`PromptProcessingBatch`/`BatchGenerator` (salvage pitfall #1: 0/5 configurations of that path
survived TP collectives on the prior attempt).

Script: `discovery/spike/collective_spike.py` (one process per rank, one `init()` per process
lifetime, all measurements for a backend in a single run — see "process lifecycle" note below).

| Metric | Ring (idle-machine re-run) | JACCL |
|---|---|---|
| `all_sum` latency, 1KB (avg / p50 / p90 ms) | 0.3424 / 0.2862 / 0.6401 | 0.2162 / 0.1875 / 0.3256 |
| `all_sum` latency, 4KB | 0.2652 / 0.2615 / 0.3033 | 0.2216 / 0.1821 / 0.3428 |
| `all_sum` latency, 16KB | 0.2572 / 0.2503 / 0.2846 | 0.2344 / 0.2085 / 0.3445 |
| **Idle-rank broadcast, lower bound** (avg / p50 / p90 ms) | 0.3583 / 0.3516 / 0.3761 | 0.1765 / 0.1708 / 0.2108 |
| TP decode, no per-step broadcast (ms/token) | 5.9281 | 2.7083 |
| TP decode, with per-step broadcast (ms/token) | 6.0802 | 3.0038 |
| TP decode, no-broadcast repeat (drift check, ms/token) | 5.6689 | (not run — see note) |
| **TP decode broadcast overhead — primary figure** (ms/token) | **0.1521** | **0.2955** |
| Overhead as % of E4 baseline (102.682 ms/token) | 0.15% | 0.29% |
| vs 10% budget (10.2682 ms/token) | **PASS** | **PASS** |

The "idle-rank" figure is the synthetic lower bound (tight loop, no compute in between); the
"TP decode broadcast overhead" figure is the primary S0 acceptance figure — it's measured
interleaved into a real hand-prefilled TP-sharded decode loop, so arrival skew between the two
ranks is present in the number, per spec (a). Both figures, both transports, are reported.

**Contamination caught and corrected**: the first ring run above was executed while
`baseline_decode.py` (Task B, a 51GB model load) was still running concurrently on the same
machine — the original ring TP-decode figures (5.69 / 6.11 ms/token, overhead 0.4194) were recorded
under that contention and are **not** the numbers in this table. Ring was re-run after the baseline
finished and the machine was idle; the table above is that clean re-run. The re-run also adds a
third pass (no-broadcast again, after the with-broadcast pass) specifically to check for monotonic
drift contaminating whichever measurement runs later in process lifetime: the drift check shows
pass-3 **faster** than pass-1 (5.67 vs 5.93 ms/token, delta -0.26ms) — the opposite direction from
contamination, i.e. no drift artifact, and the ~0.15ms broadcast overhead is at the noise floor of
this measurement (run-to-run variance on a sub-millisecond figure is real; treat the ring overhead
figure as "on the order of 0.15-0.4 ms/token, comfortably sub-1%" rather than a precise point value).
JACCL was not re-run idle (it ran immediately after the first ring run, close to when the baseline
load was finishing — its own numbers may carry some residual contention that was not independently
verified); its pass/fail conclusion is unaffected regardless, since 0.29% has enormous headroom
under the 10% budget. Re-running jaccl would spend one of the two clean attempts budgeted for it in
the spec's stop condition — not spent here since it doesn't change any pass/fail outcome.

Broadcast payload: a mock per-step message (32 token ids + a composition delta dict), pickled and
shipped via the `all_sum`-of-pickled-bytes technique salvaged from `mlx_adapter.py`'s
`DistributedSession.broadcast` (two collectives: size-then-payload, both pinned `stream=mx.cpu`).
Per that code's own note, the caller drains the model stream (`mx.synchronize()`) before issuing
the broadcast collective, to avoid racing in-flight model compute across ranks.

JACCL's collective latency and idle-broadcast figures are consistently lower than ring's, and its
raw TP-decode throughput (2.71 ms/token, no broadcast) is roughly 2x faster than ring's clean-run
figure (5.93 ms/token) — a gap that survives the idle re-run, so it looks like a real RDMA-vs-TCP
difference on this rig rather than an artifact, though jaccl itself wasn't independently re-verified
idle (see above). Treat the magnitude as indicative, not a precise ratio.

## Working init recipes

### Static TB IPs (performed once, both machines)

Local active TB interface: `en2` (self-assigned `169.254.72.202` before this change, no `bridge0`
present locally). Remote active TB interface: `en4`, enslaved to `bridge0` (`bridge0` had only an
IPv6 link-local address, no IPv4, before this change).

```
# local (M4 Max)
sudo ifconfig en2 alias 10.0.2.1 netmask 255.255.255.0

# remote (M3 Ultra) -- alias on bridge0, NOT en4 directly, since en4 is a bridge member
ssh Jasons-Mac-Studio.local 'sudo ifconfig bridge0 alias 10.0.2.2 netmask 255.255.255.0'
```

Verified: `ping 10.0.2.2` from local, sub-ms RTT (0.5-0.66ms), 0% loss. Used `alias` (additive) not
address replacement, so the pre-existing 169.254 addresses are untouched and rollback is exact
(`sudo ifconfig en2 -alias 10.0.2.1` / same on `bridge0` remotely). These aliases are non-persistent
across reboot/link-bounce by design (configd may reclaim them) — correct for a spike, not meant to
survive past it. **`bridge0` was not destroyed or reconfigured beyond adding this one alias.**

Per salvage note: static private IPs on the TB interfaces are the correct reading of CL-09's
"link-scoped hostfile entries" — NOT `169.254.x.x` link-local, which broke ring connects on the
prior branch.

### Ring backend

Rank 0 (listener) must start before rank 1 (peer) — a peer starting early burns its connect window
(salvage pitfall #3). `MLX_HOSTFILE` is the flat per-rank-link-address list (distinct from the
`{"backend":..., "hosts":[...]}` cluster-description format `mlx.distributed_config`/`mlx_lm.share`
use — see `hostfile.py`'s module docstring on the two file shapes).

```
# hostfile.json: [["10.0.2.1:41100"], ["10.0.2.2:41100"]]

# rank 0 (local)
MLX_RANK=0 MLX_HOSTFILE=hostfile.json MLX_METAL_FAST_SYNCH=1 OMLX_CLUSTER_BACKEND=ring \
  python discovery/spike/collective_spike.py

# rank 1 (remote), started ~3s after rank 0
ssh Jasons-Mac-Studio.local '... MLX_RANK=1 MLX_HOSTFILE=hostfile.json MLX_METAL_FAST_SYNCH=1 \
  OMLX_CLUSTER_BACKEND=ring python discovery/spike/collective_spike.py'
```

Full orchestration: `discovery/spike/run_ring.sh`. Formed cleanly on the first attempt.

### JACCL backend

`MLX_IBV_DEVICES` is a matrix keyed by rdma device name per node; RDMA over TB cannot route, so
off-diagonal entries name the device that reaches that peer. Device naming convention (from the
`Hostfile` docstring's own example): `rdma_<interface>`. On this rig: local device `rdma_en2`
(local active TB iface), remote device `rdma_en4` (remote active TB iface).

```
# ibv_devices.json: [[null, "rdma_en2"], ["rdma_en4", null]]

# rank 0 (local, coordinator)
MLX_RANK=0 MLX_JACCL_COORDINATOR=10.0.2.1:41200 MLX_IBV_DEVICES=ibv_devices.json \
  MLX_METAL_FAST_SYNCH=1 OMLX_CLUSTER_BACKEND=jaccl python discovery/spike/collective_spike.py

# rank 1 (remote), started ~3s after rank 0
ssh Jasons-Mac-Studio.local '... MLX_RANK=1 MLX_JACCL_COORDINATOR=10.0.2.1:41200 \
  MLX_IBV_DEVICES=ibv_devices.json MLX_METAL_FAST_SYNCH=1 OMLX_CLUSTER_BACKEND=jaccl \
  python discovery/spike/collective_spike.py'
```

Full orchestration: `discovery/spike/run_jaccl.sh`. **Formed cleanly on the first attempt** — no
second attempt was needed, so the 2-clean-attempt budget was not exhausted. `bridge0` enslavement
of the remote TB interface did not prevent RDMA formation.

**Process-lifecycle discipline honored**: each backend run is a single process per rank for its
entire lifetime (init → barrier → all measurements → exit); JACCL was never re-init'd within a
running process. If a future run needs to retry jaccl, it must be a fresh process, not a retry loop
inside one.

## `mlx_lm.share` probe (Task E)

`mlx_lm.share`'s CLI (verified at the pinned mlx-lm commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc`)
self-launches across hosts via `mlx._distributed_utils.launch.launch_ring`/`launch_jaccl`, driven by
a **cluster-description** hostfile (`{"backend":..., "hosts":[{"ssh":..., "ips":[...]}]}`), not the
raw per-rank `MLX_HOSTFILE` array used above. Invocation:

```
python -m mlx_lm share --model mlx-community/Qwen3.6-27B-OptiQ-4bit --hostfile share_hostfile.json
```

**Gotcha found and worked around**: the launcher builds its remote command using this process's
`sys.executable` (this machine's venv path) and ships the *same* absolute path/cwd to every rank,
including the remote one — an identical-absolute-path assumption across hosts (exactly the
assumption `omlx/cluster/hostfile.py`'s own docstring calls out as the reason it avoids `mlx.launch`
entirely). The two repos here live at different absolute paths (`.../Runners/omlx` locally vs
`.../Repos/omlx` on the Studio). Fix: a single directory symlink on the remote,
`/Users/jasonschulz/Developer/Runners/omlx -> /Users/jasonschulz/Developer/Repos/omlx`, so the
shipped cwd/python path resolves correctly on both ends without touching either real repo. Left in
place (harmless, trivially removable: `rm /Users/jasonschulz/Developer/Runners/omlx` on the Studio).
Also: the local rank's `ssh` entry in the hostfile must be the literal string `127.0.0.1` — mlx's
`RemoteProcess` special-cases exactly that string to skip SSH; the machine's own `.local` mDNS
hostname does **not** resolve to itself over SSH on this rig (confirmed: `ssh <own-.local-hostname>`
from itself fails with exit 255) and is not a valid substitute.

**Throughput**: transferred `mlx-community/Qwen3.6-27B-OptiQ-4bit` (22GB, present on local only —
genuine presence gap, remote had no cached copy) node-to-node over the TB link.

```
real 0m8.402s (wall, includes process spawn/teardown)
per-blob transfer speed reported by the tool: ~4.6-4.8 GB/s
```

Landed correctly on the remote (`du -sh` confirmed 22G at the expected HF cache path,
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-OptiQ-4bit`).

**Interruption test**: removed the transferred copy from the remote, re-ran the same transfer, and
sent `SIGKILL` to the orchestrating process ~3s into the ~8s transfer (mid-flight, before the final
file was received).

Findings (killed with `SIGKILL` ~3s into an ~8s transfer, twice; each finding below states its own
evidence basis — *observed* vs *established by code reading at the pinned commit*):
- **No resumability.** Reading `mlx_lm/share.py`: on the receiving rank, every file is written into
  a `tempfile.TemporaryDirectory()`, and the destination path is only `mkdir`'d and the temp dir
  `os.rename`'d into place **after every file in the manifest has been received in full**. A kill
  mid-flight leaves the real destination path **never created** — confirmed empirically both times:
  after the kill, `~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-OptiQ-4bit` did not
  exist on the receiver at all.
- **Temp-artifact leakage — two distinct classes, corrected after fresh-context verification.**
  *(An earlier revision of this bullet claimed SIGKILL-orphaned `TemporaryDirectory` leftovers were
  observed via a before/after `$TMPDIR` diff; verification refuted that observation — the files in
  question were something else, below — and this bullet now states what the evidence supports.)*
  - **Observed**: every `mlx_lm.share` *launch* leaves two `mktemp` regular files (a generated
    hostfile and a pidfile, tens of bytes) in the receiver's `$TMPDIR` — written by mlx's remote
    bootstrap (`mlx/_distributed_utils/launch.py:115,142`) and never cleaned up, kill or no kill.
    Four launches → four pairs, timestamps matching launches, not kills. A real but tiny per-launch
    leak.
  - **Code-reading hypothesis, not observed**: Python's `TemporaryDirectory` cleanup runs only on
    normal/exception unwind, so a SIGKILL that lands *after* file bytes start landing should orphan
    the staging directory with partial data. On the tested runs no such `tmpXXXXXXXX` artifact was
    found on the receiver afterward (the kills may have landed before the receiving rank created
    it). S5's cleanup design should assume both artifact classes exist and sweep for both.
- **Re-running restarts from byte zero, not a resume — directly observed.** After the interrupted
  attempt, a fresh transfer of the same model was run to completion: same ~8.4s wall time, same
  per-file progression (all 41 files, full size, from `0%`) as the very first (uninterrupted) run.
  No partial data was reused — consistent with the destination path never having been created for
  the interrupted attempt to leave anything behind.
- **No integrity check** beyond implicit trust in a fully-received manifest: there is no digest
  verification in `share.py` at this pin. This is exactly the gap S5's `{relative_path, size, sha256}`
  manifest + staging + atomic-move design (CL-13) is meant to close — `mlx_lm.share`'s own
  `TemporaryDirectory` + atomic-`rename` pattern is reusable as the mechanism, but per-file digest
  verification, chunk-granular resumability, and temp-dir cleanup-on-kill are net-new orchestration
  on top of it, matching the program spec's framing (only the orchestration is net-new; the transfer
  primitive is maintainer-sanctioned as-is).

Cleanup: the transferred 22GB copy was removed from the remote's HF cache after the probe, to leave
the rig's model inventory as found (it was not present there before this task).

## Versions / pins (both machines)

| | Local (M4 Max) | Remote (M3 Ultra) |
|---|---|---|
| Python | 3.12 (venv freshly created for this task) | 3.12 (existing `uv`-managed venv) |
| `mlx` | 0.32.0 | 0.32.0 |
| `mlx-lm` | 0.31.3 @ commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc` | 0.31.3 @ commit `ab1806e8f5d6aa035973af194a1b9198ab4754dc` |

Pins matched exactly on both machines — **no reconciliation was needed.** Confirmed by reading each
side's installed `mlx_lm-*.dist-info/direct_url.json` (`vcs_info.commit_id`), not just the version
string (the version string alone can't distinguish commits at the same release).

Local venv created at `/Users/jasonschulz/Developer/Runners/omlx/.venv` with
`python3.12 -m venv .venv && pip install -e .` (no dev extras). Remote repo (checked out to the
scrapped `feat/cluster-distributed-serving` branch — untouched, not readded to this program) already
had a matching `uv`-managed venv; reused as-is.

## Rig-facts discrepancy noted

The task's rig facts state both machines have `mlx-community/MiniMax-M2.7-3bit` on disk. On the
local machine this is **not true**: the HF cache entry exists but is a stub (`refs/` only, no
`blobs/`, no `snapshots/`) — never actually downloaded. Not a blocker for S0 (no S0 task needs
MiniMax; it's an S6 acceptance anchor). Flagging for whoever picks up S5/S6.

## Rollback

Full pre-change state dumped to `discovery/spike/rollback/` before any change:
`local-ifconfig-before.txt`, `local-netstat-before.txt`, `local-hwports-before.txt`,
`remote-ifconfig-before.txt`, `remote-netstat-before.txt`, `remote-hwports-before.txt`.

**Daemon/process sweep** (logged before killing; killed only oMLX-cluster-attempt artifacts, left
every daily-driver process running):

| Machine | PID | Command | Action |
|---|---|---|---|
| local | 14519 | `ssh -f studio cd ~/Developer/Repos/omlx && OMLX_BASE_PATH=$HOME/omlx-cluster-dev ... omlx serve` | killed (launcher for the pair below) |
| remote | 3284 | `uv run omlx serve` (started 12:03AM, `OMLX_BASE_PATH=~/omlx-cluster-dev`) | killed (SIGTERM then SIGKILL — didn't die on first signal) |
| remote | 3285 | `omlx-server`, listening on 8901 | killed (SIGKILL) |
| remote | 9560 | `dns-sd -R ... _omlx._tcp ... port=8901` (advert for the pair above) | killed |

Left alone (daily drivers, out of scope): local 87570/87617 (`oMLX.app` + bundled server), remote
933/947 (`oMLX.app` + bundled server), remote 921 (`mlx_lm.server`).

**No stale ring/jaccl ranks or port-pollution were found from any prior distributed-serving attempt**
— the pollution pitfall (salvage note #4) didn't materialize on this rig; noted as a finding, not a
gap.

**Network changes** (both reversible, both additive-alias, neither touched by removal during this
session — intentionally left up since S2 will want the same static IPs):
- local: `sudo ifconfig en2 alias 10.0.2.1 netmask 255.255.255.0` → rollback: `sudo ifconfig en2 -alias 10.0.2.1`
- remote: `sudo ifconfig bridge0 alias 10.0.2.2 netmask 255.255.255.0` → rollback: `sudo ifconfig bridge0 -alias 10.0.2.2`
- `bridge0` itself was not created, destroyed, or had members added/removed.

**Filesystem change**: remote symlink `/Users/jasonschulz/Developer/Runners/omlx ->
/Users/jasonschulz/Developer/Repos/omlx` (enables `mlx_lm.share`'s same-absolute-path launcher
assumption to resolve on both machines). Rollback: `rm /Users/jasonschulz/Developer/Runners/omlx` on
the Studio.

**Processes at end of this task**: none of the spike's own processes are left running on either
machine (all rank processes exit on their own at the end of each script; no daemons were started).
Verified by re-checking `ps aux` for `collective_spike`/`baseline_decode`/`mlx_lm share` on both
machines after the last run.
