# Cluster: configuration and management surfaces

The serving path works. Two Macs form a cluster, shard a model, batch requests
across it, and notice within seconds when a rank dies. What does not work is
*arriving* at that state without a text editor.

This is the plan for the operator-facing half.

## The gap, stated plainly

Today, turning clustering on means editing `settings.json` by hand on both
machines, inventing a shared secret, typing it identically twice, guessing a
model id that exists on both, and restarting both daemons. The admin UI shows a
read-only panel — and only *after* all of that already worked.

There is a circularity at the centre of it: `bootstrap.install()` returns early
when `cluster.enabled` is False, and again when `cluster_key` is empty. Both
returns sit above the `include_router` calls. So on a default install
`/admin/api/cluster/status` does not exist, the panel never loads, and no
configuration surface placed behind that prefix could ever be the thing that
turns clustering on. Fix that first; everything else depends on it.

## What we are not copying

A screenshot of another implementation of this feature is doing the rounds. Its
information architecture leads with detection — "Waiting for another Mac",
"Ports detected: 3", three diagnostic *Run check* buttons — because at that
stage it cannot form a cluster, so hardware detection is the only thing it can
truthfully show.

Ours forms one. The page should lead with formation state and nodes, and put
capability checks below, where a thing you consult when something is wrong
belongs.

One idea there is worth taking: a **memory budget** figure. Display-only. The
engine pool deliberately bypasses this daemon's memory ceiling for a cluster
model (`engine_pool.py`, `_is_cluster_model`), because the weights are never in
this process; a fleet-capacity number in the UI is what replaces the refusal the
operator no longer gets.

One idea there is actively harmful: a **peer IP route check**. A TCP probe of a
collective's port is not passive — it is accepted *as* the peer and poisons the
handshake, and on a jaccl coordinator it kills rank 0 outright. We have this
written down as a gotcha because it cost us an afternoon. Reachability, if we
show it, is measured against the peer's *daemon* HTTP port or read from the
process table, never by connecting to a ring port.

## Slices

Each slice is independently shippable and independently useful.

### 1. Make the surface reachable (prerequisite)

Split the two routers in `bootstrap.install()`:

- `configure()` and `admin_router` register unconditionally at daemon start.
  `admin_router` is already behind `require_admin`, and every handler already
  tolerates `manager is None` / `discovery is None` — that is the disabled
  state, and it is a state worth being able to *read*.
- `peer_router` stays gated on `enabled AND cluster_key`. That is the half that
  authenticates with the shared key and spawns rank processes on this machine;
  it has no business existing on a node that has not opted in.

`_installed` currently gates both together, so this needs a second flag. On a
*live* enable the peer routes must be added at that moment, which means the
config handler needs the live `app` — `request.app` — rather than the
`install(None, settings)` shape.

Acceptance: on a default (`enabled=false`) install, `GET
/admin/api/cluster/status` returns 200 with `enabled: false` and `POST
/cluster/report` 404s. After a live enable, `/cluster/report` authenticates
instead of 404ing.

### 2. Configuration, written and hot-applied

`GET` and `POST /admin/api/cluster/config` over the `cluster.*` namespace:
`enabled`, `cluster_key`, `backend`, `model`, `pipeline`, `max_batch_size`,
`discovery_interval_seconds`.

Writes go through the live `GlobalSettings` object and persist through it. Not a
second writer to `settings.json` — that file already has an admin-form writer
and an ssh-side writer, and a third would make the last one to save win.

Hot-apply is `shutdown()` then `install()`, not a field write. `ClusterDiscovery`
captures the port and poll interval at construction, so mutating settings alone
changes nothing until something reconstructs it.

`shutdown()` tears down ranks, so a config write is refused (409) while the
cluster is formed and serving. Tearing down someone's in-flight generation to
apply a settings change is not a trade the operator asked for.

### 3. The Cluster tab

Promote the panel out of the Status tab into its own top-level nav entry, in the
order the operator actually needs it:

1. **State** — formed or not, which model, which transport and *why* that one
   (`status.reason` already carries this), rank order. Rank order is the
   cabling for jaccl-ring, not a presentation choice.
2. **Nodes** — this Mac and every peer: chip, RAM, version, rank, whether it
   joined. Fleet memory budget as a sum, against the selected model's size.
3. **Configuration** — the form from slice 2.
4. **This node's checks** — the existing preflight list, collapsed by default.

### 4. Pairing

Two nodes are in the same cluster when their keys match. The advertisement
carries a truncated digest of the key, never the key, so a mismatch is
detectable without disclosing anything.

`_reconcile_one` retains every oMLX peer it sees regardless of fingerprint, and
`matches_fingerprint()` is defined but never called. So "that Mac is running
oMLX clustering with a different key" is already in the data — it just has never
been surfaced. That is the whole pairing UX: generate a key here, copy it there,
and until it matches, say so in as many words instead of showing an empty peer
list.

### 5. Model selection

The model dropdown is sourced from this node's local models. Validation that the
peer has it too comes from `POST /cluster/report` with a model set, on demand
behind a button — never on the status poll. That call scans every model
directory, which on a machine with a large external volume takes tens of
seconds.

No new peer endpoint. Two independently-upgraded Macs will disagree about
`protocol.py` long before they disagree about anything else.

## Verification

The default path is `enabled=false`. That is the configuration being shipped and
the one every new surface has to be correct in, so it is the one to test — not
just the enabled box.

Live verification runs against the dev daemons on `:8901`, not against the
installed `.app` on `:8888`, while the serving PR is still open.
