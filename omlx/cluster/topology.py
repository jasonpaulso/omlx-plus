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
import re
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
    # The physical receptacle this bus's cable is in, from system_profiler.
    receptacle: str | None = None
    # The RDMA device serving that receptacle, resolved locally by
    # `annotate_rdma_devices`. This is the ground truth for JACCL device
    # selection: receptacle N -> hardware port "Thunderbolt N" -> enX ->
    # rdma_enX. Guessing it positionally picked the Studio Display's port
    # over the actual cable on a machine with two live Thunderbolt links.
    rdma_device: str | None = None


@dataclass
class NodeReport:
    """What a single node tells the leader about itself."""

    node_id: str
    buses: list[Bus] = field(default_factory=list)
    rdma_devices: list[str] = field(default_factory=list)
    # The subset of `rdma_devices` whose port is PORT_ACTIVE - i.e. the ones
    # with a live cable. Empty when the node predates port-state reporting.
    active_rdma_devices: list[str] = field(default_factory=list)
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
    return NodeReport(node_id=node_id, buses=annotate_rdma_devices(parse_buses(out)))


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
        receptacle = (entry.get("receptacle_1_tag") or {}).get("receptacle_id_key")
        buses.append(
            Bus(
                name=entry.get("_name", ""),
                domain_uuid=domain,
                peer_domain_uuid=peer_domain,
                peer_model=peer_model,
                receptacle=str(receptacle) if receptacle is not None else None,
            )
        )
    return buses


def thunderbolt_interfaces() -> dict[str, str]:
    """Receptacle number -> `enX`, from `networksetup -listallhardwareports`.

    The hardware port named "Thunderbolt N" serves receptacle N; its device
    is the `enX` whose RDMA twin is `rdma_enX`. Measured on both test
    machines (MacBook receptacle 3 -> "Thunderbolt 3" -> en2; Studio
    receptacle 2 -> "Thunderbolt 2" -> en3), each carrying the live cable.
    """
    try:
        out = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("topology: networksetup failed: %s", exc)
        return {}

    mapping: dict[str, str] = {}
    port = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            name = line.split(":", 1)[1].strip()
            match = re.fullmatch(r"Thunderbolt (\d+)", name)
            port = match.group(1) if match else None
        elif line.startswith("Device:") and port is not None:
            mapping[port] = line.split(":", 1)[1].strip()
            port = None
    return mapping


def annotate_rdma_devices(buses: list[Bus]) -> list[Bus]:
    """Resolve each cabled bus's RDMA device from its receptacle."""
    from dataclasses import replace

    interfaces = None
    annotated: list[Bus] = []
    for bus in buses:
        if bus.peer_domain_uuid is not None and bus.receptacle is not None:
            if interfaces is None:
                interfaces = thunderbolt_interfaces()
            interface = interfaces.get(bus.receptacle)
            if interface:
                bus = replace(bus, rdma_device=f"rdma_{interface}")
        annotated.append(bus)
    return annotated


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


def unattributed_device(report: NodeReport) -> str | None:
    """The RDMA device of a node that cannot name its peer on any bus.

    A Thunderbolt hub between two Macs - a Studio Display with an upstream and
    a downstream port, say - is enumerated asymmetrically: the Mac downstream
    reports the far Mac as its bus peer, the Mac upstream reports no peer at
    all. Its buses therefore attribute no device to any link, and the matrix
    entry falls through to null even though RDMA is live on both ends
    (measured 2026-07-27: a hand-written symmetric matrix formed a jaccl world
    over exactly that cabling and completed a collective).

    Only a single active device is trusted here. With several, position is the
    only discriminator and there is no bus to index by, so guessing would be
    worse than the honest null. A wrong guess is bounded anyway: the launch
    fails, and `auto` falls back to ring.
    """
    devices = set(report.rdma_devices)
    active = sorted(d for d in report.active_rdma_devices if d in devices)
    return active[0] if len(active) == 1 else None


def ibv_matrix(
    reports: list[NodeReport], order: list[str]
) -> list[list[str | None]]:
    """The `MLX_IBV_DEVICES` matrix: `m[i][j]` reaches node `j` from node `i`.

    Null on the diagonal, and null wherever two nodes have no cable.

    Device selection prefers link state over position. A machine enumerates
    an RDMA device for *every* Thunderbolt port, but only the port the cable
    is in reports `PORT_ACTIVE` - and handing JACCL a `PORT_DOWN` device
    fails at protection-domain allocation with an error that reads like a
    driver fault (observed 2026-07-27: `[jaccl] Couldn't allocate protection
    domain` because the positional pick chose a down device out of three).

    - The bus knows its device (`bus.rdma_device`, resolved from the physical
      receptacle): that is the ground truth and wins outright. Position and
      even link state can lie - a Studio Display daisy-chained to both Macs
      puts a second, wrong PORT_ACTIVE device on each machine, and the
      positional pick chose the display's port over the cable (measured
      2026-07-27: both ranks hung ~60s in jaccl init, formation fell back).
    - Exactly one active device: that is the cable; use it for every edge.
    - Several active devices and no receptacle mapping: the Nth cabled bus
      takes the Nth active device, both in sorted order - positional between
      live cables, never a dead port.
    - No port-state information (a peer running older code): the original
      positional map over all devices.

    When a node cannot supply a device for a link, the entry is null and the
    caller falls back off `jaccl` rather than launching a run that would hang.
    """
    by_id = {report.node_id: report for report in reports}
    domain_owner = {
        domain: report.node_id for report in reports for domain in report.domains
    }

    def pick(report: NodeReport, bus: Bus, cable_index: int) -> str | None:
        devices = sorted(report.rdma_devices)
        if bus.rdma_device and bus.rdma_device in devices:
            return bus.rdma_device
        active = sorted(d for d in report.active_rdma_devices if d in devices)
        if len(active) == 1:
            return active[0]
        pool = active or devices
        return pool[cable_index] if cable_index < len(pool) else None

    matrix: list[list[str | None]] = []
    for src in order:
        row: list[str | None] = []
        report = by_id.get(src)
        cabled: list[Bus] = []
        if report is not None:
            cabled = [
                bus
                for bus in sorted(report.buses, key=lambda b: b.name)
                if bus.peer_domain_uuid
                and domain_owner.get(bus.peer_domain_uuid) not in (None, src)
            ]
        for dst in order:
            if src == dst or report is None:
                row.append(None)
                continue
            device = None
            for cable_index, bus in enumerate(cabled):
                if domain_owner.get(bus.peer_domain_uuid) != dst:
                    continue
                device = pick(report, bus, cable_index)
                break
            if device is None and not cabled:
                device = unattributed_device(report)
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
