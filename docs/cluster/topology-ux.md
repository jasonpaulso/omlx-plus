# Topology & UX direction — cluster v-1.5 ("seamless")

**Status: DIRECTION AGREED (Jason, 2026-07-31). Not yet planned or scheduled; next
session picks this up.** This note pins the diagnosis, the target model, the options
considered, and the slice order so the next session doesn't re-derive any of it.

## The requirement we drifted from

The product requirement for clustering was a **seamless oMLX experience** for an
audience of one (two Macs, one operator). Upstreaming remains a goal but is not *the*
goal yet — and before any upstream PR, cluster-vs-no-cluster must be transparent and
must not require hand-editing JSON. The v1 opt-in posture (`cluster.role` default
`off`, settings-file-only enablement, hidden dashboard panel) was written for the
upstream PR contract ("no behavior change for any existing deployment") and leaked
into our own deployment's UX. That was drift, not a decision.

Reference experience: **EXO.** EXO has internal coordinator/worker asymmetry, but the
user never sees it — every node's dashboard offers the same actions (create/delete
instances, downloads, chat), whether the node is in a cluster of N, a cluster of 1, or
no cluster at all.

## Diagnosis

v1 conflated two independent axes:

1. **Internal coordination topology** — someone must own membership truth. Our
   head/worker split with token join is a deliberate, audited design (single-writer
   membership, no consensus, explicit trust boundary). This part is fine.
2. **User-facing capability model** — what you can do from whichever dashboard you're
   on. This is where v1 surfaced axis 1 into the UX: the role you hand-wrote into
   JSON decides what your dashboard can do.

EXO hides axis 1 and makes axis 2 symmetric. That is the target.

## Target model

> Every node boots as head-of-itself ("cluster of 1"). "Cluster" is a **dashboard
> verb**, not an identity: *Add node* / *Join* / *Leave* are actions. Role is an
> implementation detail the user never names. The upstream opt-in flag survives as
> packaging (default for upstream), not as our UX.

Three code facts make this shorter than it looks:

- **Cluster-of-1 already works.** Coexistence was a v1 design requirement: a head
  with zero members serves its local models exactly like a plain server (S4). So
  "every node is head-of-itself" is `role=head` everywhere, supported today.
- **Join can be an action, not an identity.** The bootstrap-token join flow exists
  end-to-end; what's missing is UI on both ends. "Add node" on box A mints a token;
  accepting it on box B makes B a worker *internally* — plumbing the user never sees.
- **Dashboard symmetry is a proxy problem, not a rearchitecture.** A worker's admin
  UI can offer the same actions by forwarding operator calls to the head over the
  member channel that already exists.

The inference API needs **no** changes: serving was never split. `/v1/cluster` is the
node-to-node control plane (join/heartbeat/transfer), analogous to `/admin`; clients
use the same `/v1/*` endpoints and `auto_placement` (default on) decides local vs
distributed per load.

## The one real engineering gap under "seamless"

When both boxes serve their own local models *and* participate in a formation, the
head's placement math does not see the worker's independent local load:
`ProcessMemoryEnforcer` is process-local, and a distributed shard on the worker is
invisible to every ceiling (recorded S4 finding; related follow-ons: KV-head
replication, unequal sharding). Heartbeats already carry `node_state`
(total_memory, memory_ceiling, models_present), so the carrier exists — placement
needs to consume live worker load, not just capacity. This is the substantive item;
everything else is UI and defaults.

## Options considered

| Option | Verdict |
| --- | --- |
| A. Do nothing; document the settings key | Rejected — feature stays invisible; fails the requirement. |
| B. Symmetric dashboard over the existing asymmetric control plane (this note) | **Chosen.** Keeps the audited S5/S6 join-security model intact. |
| C. Peerless mesh à la EXO (auto-discovery, election, no configured roles) | **V-NEXT.** Reopens the join trust model (EXO trusts the LAN; we deliberately don't — bootstrap tokens, credential revocation) for little experience gain over one-click join. |

## Slice order (proposed, unplanned)

1. **Head-everywhere defaults for our deployment** — both boxes get `role=head` in
   their `~/.omlx/settings.json`; panel becomes visible; each box is a cluster of 1.
   Zero code. (Interim hand-edit — the last one — retired by slice 2.)
2. **Dashboard enable/toggle, restart-bound** — `cluster.role` (off/head only) in the
   global-settings modal + POST handler, "requires restart" like existing
   restart-bound settings, paired with the existing `/admin/api/server/restart`.
   Worker is never offered here — worker-ness comes only from the join flow.
3. **Join/leave UX** — mint token ("Add node") and accept token ("Join") from the
   dashboard on both ends; leave/remove likewise. Wraps the existing
   `/v1/cluster` join flow; no protocol changes.
4. **Worker dashboard symmetry** — proxy operator actions from a joined node's admin
   UI to the head over the member channel; joined node's panel shows the same
   cluster state and verbs as the head's.
5. **Placement sees worker load** — extend `node_state` consumption so placement
   accounts for the worker's own resident models / live memory, closing the gap
   above. Likely the only slice needing rig time.

Slices 1–3 are independent of 4–5 and deliver most of the felt UX. Order within 4–5
can flip if daily-driver clustering (both boxes serving locally while formed) starts
before the proxy work.

## Non-goals (unchanged from v1 decisions)

- Peerless auto-discovery / leader election (option C) — v-next.
- Multimodal distribution (exo-style head-resident vision tower) — v-next (Jason,
  2026-07-30).
- Live (no-restart) role activation — not needed once the toggle exists; revisit only
  if restart friction proves real.
