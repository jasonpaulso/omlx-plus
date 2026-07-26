# SPDX-License-Identifier: Apache-2.0
"""Thunderbolt topology discovery and backend selection.

`mlx.distributed_config` answers the same question by SSHing into every host
and running `system_profiler` remotely. oMLX already has a daemon on each node
and a control plane between them, so each node reports its *own* Thunderbolt
data and the leader assembles the picture. The SSH layer disappears; the graph
analysis is what is worth keeping.

How two machines are matched up
-------------------------------
`system_profiler SPThunderboltDataType` gives each Thunderbolt bus a
`domain_uuid_key`, and lists any connected peer under `_items` along with *that
peer's own* domain UUID. So an edge exists when one node's bus reports a peer
domain that equals another node's bus domain. It is symmetric and needs no
cable labelling.

Verified on an M5 Max MacBook Pro cabled to an M3 Ultra Mac Studio:

    macbook bus_0  domain 817DCFA4...  sees peer domain 28CA4C30...
    studio  bus_1  domain 28CA4C30...  sees peer domain 817DCFA4...

which is exactly the edge (macbook.bus_0) <-> (studio.bus_1).

Non-Mac Thunderbolt devices - displays, drives, docks - report a null peer
domain, so they drop out of the graph without needing a device allow-list.

What the graph decides
----------------------
- Every pair connected directly -> full mesh -> `jaccl`.
- Connected only as a cycle -> `jaccl-ring`.
- Anything else, or any node that is not RDMA-ready -> TCP `ring`, which needs
  no Thunderbolt at all.

RDMA over Thunderbolt cannot route: a packet will not hop through an
intermediate Mac. That is why a missing cable downgrades the whole cluster
rather than costing one hop.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from itertools import combinations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bus:
    """One Thunderbolt bus on one machine."""

    name: str
    domain_uuid: str
    # Domain UUID of the Mac on the other end, if a Mac is plugged in.
    peer_domain_uuid: str | None = None
    peer_model: str | None = None


@dataclass
class NodeReport:
    """What a single node tells the leader about itself."""

    node_id: str
    buses: list[Bus] = field(default_factory=list)
    rdma_devices: list[str] = field(default_factory=list)
    rdma_ready: bool = False

    @property
    def domains(self) -> set[str]:
        return {b.domain_uuid for b in self.buses}


def probe_local(node_id: str) -> NodeReport:
    """Run `system_profiler` on this machine and parse it into a report."""
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPThunderboltDataType"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("topology: system_profiler failed: %s", exc)
        return NodeReport(node_id=node_id)
    return NodeReport(node_id=node_id, buses=parse_buses(out))


def parse_buses(profiler_json: str) -> list[Bus]:
    """Parse `system_profiler -json SPThunderboltDataType` output."""
    try:
        payload = json.loads(profiler_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("topology: could not parse system_profiler output")
        return []

    buses: list[Bus] = []
    for entry in payload.get("SPThunderboltDataType", []):
        domain = entry.get("domain_uuid_key")
        if not domain:
            continue
        peer_domain = None
        peer_model = None
        for item in entry.get("_items", []):
            # A Mac reports its own domain UUID; a display or drive does not.
            if item.get("domain_uuid_key"):
                peer_domain = item["domain_uuid_key"]
                peer_model = item.get("device_name_key")
                break
        buses.append(
            Bus(
                name=entry.get("_name", ""),
                domain_uuid=domain,
                peer_domain_uuid=peer_domain,
                peer_model=peer_model,
            )
        )
    return buses


def connectivity(reports: list[NodeReport]) -> set[frozenset[str]]:
    """Undirected edges between node ids, derived from domain UUID matching.

    An edge is recorded when either direction sees it. In practice both sides
    report it, but a node whose profiler data is stale or missing should
    degrade the graph rather than silently drop a real cable.
    """
    by_domain = {
        domain: report.node_id for report in reports for domain in report.domains
    }
    edges: set[frozenset[str]] = set()
    for report in reports:
        for bus in report.buses:
            if not bus.peer_domain_uuid:
                continue
            peer = by_domain.get(bus.peer_domain_uuid)
            if peer is not None and peer != report.node_id:
                edges.add(frozenset((report.node_id, peer)))
    return edges


def is_full_mesh(node_ids: list[str], edges: set[frozenset[str]]) -> bool:
    """Whether every pair of nodes has a direct cable."""
    if len(node_ids) < 2:
        return True
    return all(frozenset(pair) in edges for pair in combinations(node_ids, 2))


def missing_cables(
    node_ids: list[str], edges: set[frozenset[str]]
) -> list[tuple[str, str]]:
    """Pairs with no direct cable, for the admin UI to name explicitly."""
    return [
        (a, b) for a, b in combinations(sorted(node_ids), 2) if frozenset((a, b)) not in edges
    ]


def find_ring(node_ids: list[str], edges: set[frozenset[str]]) -> list[str] | None:
    """A Hamiltonian cycle over the nodes, or None.

    Cluster sizes here are small - a handful of Macs on a desk - so an
    exhaustive search is both fast enough and easier to trust than a heuristic.
    """
    n = len(node_ids)
    if n < 3:
        # Two nodes joined by one cable are a degenerate ring; MLX treats that
        # as a mesh, so do not claim a ring here.
        return None

    adjacency = {node: set() for node in node_ids}
    for edge in edges:
        a, b = tuple(edge)
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    start = node_ids[0]
    path = [start]
    seen = {start}

    def extend() -> bool:
        if len(path) == n:
            return start in adjacency[path[-1]]
        for nxt in sorted(adjacency[path[-1]]):
            if nxt in seen:
                continue
            path.append(nxt)
            seen.add(nxt)
            if extend():
                return True
            path.pop()
            seen.remove(nxt)
        return False

    return list(path) if extend() else None


def ibv_matrix(
    reports: list[NodeReport], order: list[str]
) -> list[list[str | None]]:
    """The `MLX_IBV_DEVICES` matrix: `m[i][j]` reaches node `j` from node `i`.

    Null on the diagonal, and null wherever two nodes have no cable.

    The bus-to-device mapping is positional: the Nth Thunderbolt bus
    corresponds to the Nth RDMA device, both taken in sorted order. That held
    on the hardware available here - the single cabled bus mapped to the single
    active Thunderbolt interface, `bus_0` -> `en1` -> `rdma_en1` - but a
    multi-cable machine has not been verified. When a node cannot supply a
    device for a link, the entry is null and the caller falls back off `jaccl`
    rather than launching a run that would hang.
    """
    by_id = {report.node_id: report for report in reports}
    domain_owner = {
        domain: report.node_id for report in reports for domain in report.domains
    }

    matrix: list[list[str | None]] = []
    for src in order:
        row: list[str | None] = []
        report = by_id.get(src)
        for dst in order:
            if src == dst or report is None:
                row.append(None)
                continue
            device = None
            for index, bus in enumerate(sorted(report.buses, key=lambda b: b.name)):
                if not bus.peer_domain_uuid:
                    continue
                if domain_owner.get(bus.peer_domain_uuid) != dst:
                    continue
                if index < len(sorted(report.rdma_devices)):
                    device = sorted(report.rdma_devices)[index]
                break
            row.append(device)
        matrix.append(row)
    return matrix


@dataclass
class Plan:
    """The leader's decision about how to run this cluster."""

    backend: str
    order: list[str]
    edges: set[frozenset[str]]
    missing: list[tuple[str, str]]
    reason: str


def plan(reports: list[NodeReport]) -> Plan:
    """Choose a backend and a rank order for these nodes.

    Rank order matters for `jaccl-ring`, where consecutive ranks must be
    physically adjacent. For the mesh and TCP backends any order works, so
    node id order is used to keep runs reproducible.
    """
    node_ids = sorted(report.node_id for report in reports)
    edges = connectivity(reports)
    missing = missing_cables(node_ids, edges)

    if len(node_ids) < 2:
        return Plan("ring", node_ids, edges, missing, "single node")

    not_ready = sorted(r.node_id for r in reports if not r.rdma_ready)
    if not_ready:
        return Plan(
            "ring",
            node_ids,
            edges,
            missing,
            "RDMA is not ready on " + ", ".join(not_ready),
        )

    if is_full_mesh(node_ids, edges):
        return Plan("jaccl", node_ids, edges, [], "every pair is cabled directly")

    ring = find_ring(node_ids, edges)
    if ring is not None:
        return Plan("jaccl-ring", ring, edges, missing, "nodes form a cable ring")

    return Plan(
        "ring",
        node_ids,
        edges,
        missing,
        "Thunderbolt cabling is neither a full mesh nor a ring; "
        + str(len(missing))
        + " pair(s) not connected",
    )
