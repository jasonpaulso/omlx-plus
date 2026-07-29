# S2 security notes — distributed-serving command channel (E9)

These are the honest residuals of the S2 head→worker command channel (D2), the
CL2 review dispositions as *implemented*, and the operator settings that widen
or narrow the trust model. Read this before enabling `cluster.role` on a network
you do not fully control.

The governing principle (from the CL2 review's ordering framing):
**authenticating the head does not bound what a compromised head can do.** The
control that stops head-compromise or head-impersonation from becoming worker
code execution is *local confinement on the worker*, never response signing.
Signing is defence in depth on top of confinement — never a substitute.

## The command channel rides the heartbeat, in plaintext (re-rated CL-05)

In S1 the head's heartbeat response was a liveness ack, so plaintext was
accepted. Under D2 the same response body can carry formation commands. The
no-TLS-in-v1 decision stands, but the residual is larger and must be stated
plainly:

- A command-bearing heartbeat response is an **unauthenticated instruction on a
  plaintext LAN**. The CL2-05 HMAC (below) bounds *impersonation* — a party that
  does not hold the member digest cannot forge a commands-bearing response the
  worker will act on — but it does **not** provide confidentiality, and it does
  not stop an on-path attacker from **suppressing** a legitimate response
  (CL2-06 residual, below). TLS remains the real fix and is deferred past v1
  (the seam exists: `client.py` `tls_context`, `state.py` peer-cert field).
- Because confinement is the real control, a forged or impersonated head still
  cannot make the worker load an arbitrary path, run injected env, or carry the
  ring to a foreign host — see the confinement section.

## CL2-05 response HMAC — what it does and does not protect

When a heartbeat response carries commands it is signed:
`response_key = HMAC(member_digest, "omlx.cluster.command-response.v1")`, then
`sig = HMAC(response_key, canonical(commands) | epoch | seq)`, compared with
`compare_keys`. The worker derives the same key from its own secret's digest;
the head holds that digest already (the S1 bearer-verifier property, `CL-04`),
so **no new secret is stored at rest**.

- **At-rest consequence.** A read of the head's `cluster.json` (0o600) exposes
  the member *digests*, which are enough to sign command responses — i.e. it
  enables **head-impersonation-to-workers**, NOT worker-impersonation-to-head
  (the worker's own secret is never derivable from the digest). This is the
  same directionality S1 documented, now with a command-channel consequence.
- **Domain separation.** The response key is derived from the digest with a
  fixed label, so it is never the bearer verifier itself — a signature can
  never be replayed as a member credential or vice-versa.
- **Replay/echo binding (CL2-06).** The MAC covers the head-echoed `epoch` and
  `seq`, and the worker discards any commands-bearing response that does not
  echo the exact heartbeat it just sent. Every command additionally carries a
  head-minted `(job_id, step)`; a re-delivered pair is a no-op ack and never
  spawns twice.
- **Residual: teardown suppression.** An on-path attacker can *drop* a
  commands-bearing response (it cannot forge one). Dropping a `teardown` leaves
  a rank holding its ring port and weights. This is detectable, not silent: the
  head arms a CL2-06 alarm when it tears a formation down, and any later
  authenticated `job_update` that still reports that formation as live raises
  the alarm (surfaced in formation status / the dashboard).

## Worker-side confinement — the primary control

Every command the worker applies is confined against the worker's OWN settings,
in `WorkerCommandExecutor` (`omlx/cluster/manager.py`):

- **CL2-01 — no env crosses the wire.** The command schema carries typed
  scalars only (`extra="forbid"`); an env-shaped field makes the whole command
  fail closed. The rank env is built locally from an allowlist
  (`hostfile.local_worker_env`), so `PYTHONPATH`/`DYLD_INSERT_LIBRARIES`/etc.
  can never be injected.
- **CL2-02 — model id, never a path.** A command carries a model IDENTIFIER; the
  worker resolves it against its own `get_effective_model_dirs()`. A
  head-supplied path can never reach the loader; an absent model is a named
  error and nothing spawns (S5 has no auto-download).
- **CL2-03 — the worker owns its hostfile.** Every peer address is re-validated
  against the worker's own `data_plane_subnet` via the D7 predicate, and the
  worker computes its *own* rank's entry from its *own* configured address —
  never accepting a head-supplied address for itself.
- **CL2-04 — fail closed.** Unknown kind, unknown field, or off-version schema
  is rejected and reported (as a `job_update`, logged with a key fingerprint),
  never ignored. E10 skew rejection at join keeps head and worker on identical
  code, so ignore-unknown is never needed.
- **CL2-09 — bounded spawns.** At most one live formation per worker, refused
  while one is live — the worker's own exhaustion accounting, independent of any
  head-supplied limit.
- **CL2-10 — bounded numerics.** Every head-supplied number is bounded
  (heartbeat interval ceiling, world-size ceiling, port range) and non-numeric
  is rejected.
- **CL2-12 — fail closed with no config.** A worker with no own
  `data_plane_subnet`/`data_plane_address` **refuses to form** — it never
  degrades to trusting head-supplied values.

## Availability residual — a worker-rank death is not propagated in S2

A rank death is surfaced cleanly only when the daemon that *owns* that rank can
see it. The head's deathwatch watches its own rank 0; killing rank 0 closes its
reply pipe and the engine raises at once. But if the **worker's** rank dies
mid-generation, the head's rank 0 is left blocked-but-alive inside the next
collective — the head's deathwatch cannot distinguish "wedged" from "working",
and S2 does not propagate the worker daemon's own observation of the death back
to the head. That path is therefore bounded only by the generate-idle timeout
(`GENERATE_IDLE_TIMEOUT_S`, 600s): the request eventually fails and the
formation can be torn down, but not promptly. Cross-node death propagation
(worker daemon → head, via an out-of-band `job_update` that triggers teardown)
is an S3 item. Operators should treat a stuck distributed request as a signal to
`unload` the formation.

## CL2-07 — a compromised worker can lie

The head attributes every `job_update` to the AUTHENTICATED member and ignores
any member/rank id in the update body. A member secret is still a full member
credential: a **compromised worker** that holds a valid secret can report false
status ("ready" when not, or lie about presence) and, being inside the ring
(CL-09 trust model), can read activations and inject tensors. This is inside the
accepted S2 threat model — the ring is trusted once formed — and is called out
here rather than silently assumed. It is not mitigated in v1.

## D7 operator settings — and when they are dangerous

New `ClusterSettings` fields (env `OMLX_CLUSTER_*`, CLI `--cluster-*`):

| Setting | Meaning |
|---|---|
| `data_plane_subnet` | CIDR of the Thunderbolt link subnet (e.g. `10.0.2.0/24`). **Unset ⇒ formation refuses.** This IS the CL-09 link-scope predicate. |
| `data_plane_address` | This node's own address inside that subnet. Validated bound to a local interface before it is reported/used. |
| `backend` | `ring` \| `jaccl` \| `auto`. P2 forms `ring` only; `jaccl` lands in P3. |
| `data_plane_base_port` | First ring listening port (default 41100). Each node uses its own. |
| `rdma_device` | RDMA device for the jaccl backend (P3). |
| `allow_routable_data_plane` | **Dangerous.** Accepts a data-plane address *outside* `data_plane_subnet`. |

`allow_routable_data_plane` is the one to be careful with. Default-deny link
scoping is the whole of CL-09's mitigation: it keeps model activations and
tokens on the point-to-point Thunderbolt link and refuses to carry them to any
routable/management address. Turning it on lets the ring bind an address the
subnet predicate would otherwise reject — appropriate only when the operator has
deliberately placed the data plane on a routable network they trust end-to-end,
and never on a shared or hostile LAN. `169.254.0.0/16` (link-local) is rejected
*always*, with no override, because it breaks ring connects (salvage pitfall 2).
Loopback additionally requires `allow_loopback` (single-host test mode).
