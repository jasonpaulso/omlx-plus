# SPDX-License-Identifier: Apache-2.0
"""Forming a cluster, and owning it once formed.

This is the leader's half of the control plane. It answers four questions and
then gets out of the way:

- *who* - which discovered peers are joining, and in what rank order
- *how* - which transport the cabling and RDMA state actually allow
- *where* - each node resolves the model on its own disk, from an id
- *when* - rank 0 must be listening before any peer is told to start

Peers are driven over HTTP against their own daemons (`omlx/cluster/routes.py`),
which is why there is no SSH here and no assumption that the nodes share a
filesystem or install path.

Formation is deliberately all-or-nothing. A cluster that came up on three of
four machines is not a smaller cluster, it is four machines waiting inside
`mx.distributed.init()` for someone who is never arriving, so every failure
path tears the whole thing down.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

from omlx.cluster import hostfile, preflight, topology
from omlx.cluster.launcher import LocalCluster, resolve_python
from omlx.cluster.protocol import CMD_GENERATE, CMD_LOAD, GenerationSpec

logger = logging.getLogger(__name__)

# A peer that cannot answer a control call in this long is not going to make a
# usable cluster member either. Generous because `/cluster/report` scans every
# model directory, which on an external volume takes tens of seconds.
PEER_TIMEOUT_S = 180.0
# Loading a shard reads weights off disk on every node at once.
LOAD_TIMEOUT_S = 900.0
# How long a formation waits for Bonjour to answer before calling the fleet
# empty. A browse plus a resolve is several seconds, and the first request
# after a daemon restart arrives well inside that.
PEER_DISCOVERY_GRACE_S = 20.0


class ClusterFormationError(RuntimeError):
    """Formation failed. The cluster has already been torn back down."""


@dataclass
class NodeSlot:
    """One node's place in the formed cluster."""

    node_id: str
    host: str
    port: int
    rank: int
    is_local: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterStatus:
    """What the admin UI and `/cluster/status` report."""

    enabled: bool = False
    formed: bool = False
    model: str = ""
    backend: str = ""
    reason: str = ""
    world_size: int = 0
    nodes: list[NodeSlot] = field(default_factory=list)
    missing_cables: list[tuple[str, str]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    error: str = ""
    busy: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nodes"] = [n.to_dict() for n in self.nodes]
        data["missing_cables"] = [list(pair) for pair in self.missing_cables]
        return data


class PeerClient:
    """Blocking HTTP to one peer daemon's cluster routes.

    Blocking on purpose: every caller already runs in a worker thread, and the
    control plane is a handful of calls per formation, not a hot path.
    """

    def __init__(self, host: str, port: int, cluster_key: str) -> None:
        self.host = host
        self.port = port
        self._key = cluster_key

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        import httpx

        response = httpx.post(
            f"{self.base_url}{path}",
            json=payload or {},
            headers={"X-Cluster-Key": self._key},
            timeout=PEER_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json()


def local_ip() -> str:
    """This machine's address on the route peers will use to reach it.

    Opening a UDP socket to a peer is the only reliable way to learn which of
    several interfaces (wifi, ethernet, Thunderbolt bridge) the kernel would
    actually pick; hostname resolution routinely returns the wrong one.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: routed, never answered
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def resolve_ipv4(host: str, port: int) -> str:
    """The address of `host` that this node can actually reach it on.

    Bonjour resolves peers to `.local` names, and the ring backend's hostfile
    wants `ip:port` - it is not a name resolver. Naming is not the hard part
    though: a Mac with a Thunderbolt cable attached publishes **several** A
    records, and the first one is routinely a link-local `169.254.x.x` that
    belongs to a bridge nobody is serving on. A rank handed that address binds
    somewhere its peers cannot see and the run dies with
    `[ring] Couldn't connect (error: 60)` - a timeout that reads like a
    firewall fault and is not one.

    So the address is chosen by *evidence*: whichever candidate accepts a TCP
    connection on the peer's daemon port is a route that demonstrably works.
    Link-local addresses are skipped outright; they are never the answer for a
    node whose daemon is on the LAN.
    """
    try:
        info = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        raise ClusterFormationError(f"cannot resolve {host!r} to an IPv4 address: {exc}")

    candidates: list[str] = []
    for entry in info:
        address = entry[4][0]
        if address.startswith("169.254.") or address.startswith("127."):
            continue
        if address not in candidates:
            candidates.append(address)
    if not candidates:
        raise ClusterFormationError(f"{host!r} has no routable IPv4 address")

    for address in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2.0)
            if probe.connect_ex((address, port)) == 0:
                return address

    # Nothing answered. The daemon may be mid-restart rather than unreachable,
    # so take the first routable address rather than refusing to form.
    logger.warning(
        "cluster: no candidate address for %s answered on port %d; using %s",
        host,
        port,
        candidates[0],
    )
    return candidates[0]


def resolve_model_path(settings: Any, model_id: str) -> str:
    """Find `model_id` on *this* machine's disks.

    Every node resolves the model itself. The leader cannot send a path: the
    peer may keep its models on an external volume, under a different user, or
    simply somewhere else, and a path that happens to be right on the leader
    would be a silent trap everywhere else.
    """
    from omlx.model_discovery import discover_models_from_dirs

    dirs = settings.get_effective_model_dirs()
    discovered = discover_models_from_dirs([d for d in dirs])
    found = discovered.get(model_id)
    if found is None:
        raise ClusterFormationError(
            f"model {model_id!r} is not present on {socket.gethostname()}; "
            "every node needs its own copy of the weights"
        )
    return found.model_path


class ClusterManager:
    """The leader's cluster: formation, teardown, and one request at a time.

    Serialization is not a simplification to be removed later by tuning. Under
    tensor parallelism every rank must run the same forward pass, so batching
    two requests means every rank agreeing, every step, on which sequences
    advance - that is the rank-aware scheduler described in
    `docs/cluster-scheduler-divergence-audit.md`. Until it exists, a second
    concurrent request would not be slower, it would be wrong.
    """

    def __init__(
        self,
        settings: Any,
        peers: Callable[[], list[Any]] | None = None,
    ) -> None:
        self._settings = settings
        self._peers_fn = peers or (lambda: [])
        self._cluster: LocalCluster | None = None
        self._slots: list[NodeSlot] = []
        self._backend = ""
        self._reason = ""
        self._model_id = ""
        self._error = ""
        self._missing: list[tuple[str, str]] = []
        self._blockers: list[str] = []
        # Guards the rank-0 pipe. One request in flight, cluster-wide.
        self._pipe = threading.Lock()
        self._busy = False

    # -- state -------------------------------------------------------------

    @property
    def settings(self) -> Any:
        """The daemon's settings, as this manager sees them."""
        return self._settings

    @property
    def formed(self) -> bool:
        return self._cluster is not None

    def status(self) -> ClusterStatus:
        cluster = getattr(self._settings, "cluster", None)
        return ClusterStatus(
            enabled=bool(cluster is not None and cluster.enabled),
            formed=self.formed,
            model=self._model_id,
            backend=self._backend,
            reason=self._reason,
            world_size=len(self._slots),
            nodes=list(self._slots),
            missing_cables=list(self._missing),
            blockers=list(self._blockers),
            error=self._error,
            busy=self._busy,
        )

    # -- formation ---------------------------------------------------------

    def form(self, model_id: str) -> ClusterStatus:
        """Bring a cluster up for `model_id`. Blocking; call it off-loop."""
        if self.formed:
            if self._model_id == model_id:
                return self.status()
            self.teardown()

        self._error = ""
        try:
            self._form(model_id)
        except Exception as exc:  # noqa: BLE001 - reported, never half-formed
            self._error = str(exc)
            logger.exception("cluster: formation failed")
            self.teardown()
            raise ClusterFormationError(str(exc)) from exc
        return self.status()

    def _form(self, model_id: str) -> None:
        cluster_settings = getattr(self._settings, "cluster", None)
        if cluster_settings is None or not cluster_settings.enabled:
            raise ClusterFormationError("cluster.enabled is false")
        if not cluster_settings.cluster_key:
            raise ClusterFormationError("cluster.cluster_key is not set")

        model_path = resolve_model_path(self._settings, model_id)
        python = resolve_python()

        clients = self._await_peers(cluster_settings)

        reports, local_id = self._collect_reports(clients, model_id)
        chosen = topology.plan(reports)
        backend = (
            chosen.backend
            if cluster_settings.backend == "auto"
            else cluster_settings.backend
        )
        self._backend = backend
        self._reason = (
            chosen.reason
            if cluster_settings.backend == "auto"
            else f"pinned to {backend} by settings"
        )
        self._missing = list(chosen.missing)

        order = self._rank_order(chosen.order, local_id)
        slots: list[NodeSlot] = []
        for rank, node_id in enumerate(order):
            if node_id == local_id:
                slots.append(
                    NodeSlot(
                        node_id=node_id,
                        host=local_ip(),
                        port=int(self._settings.server.port),
                        rank=rank,
                        is_local=True,
                    )
                )
            else:
                client = clients[node_id]
                slots.append(
                    NodeSlot(
                        node_id=node_id,
                        host=resolve_ipv4(client.host, client.port),
                        port=client.port,
                        rank=rank,
                    )
                )
        self._slots = slots
        self._model_id = model_id

        ips = [slot.host for slot in slots]
        launch: dict[str, Any] = {}
        if backend != "ring":
            launch["coordinator"] = (
                f"{ips[0]}:{hostfile.DEFAULT_JACCL_COORDINATOR_PORT}"
            )
            launch["ibv_devices"] = topology.ibv_matrix(reports, order)

        # Rank 0 first, and listening, before any peer is told to start. The
        # ring backend gives a connecting peer a bounded retry window; a peer
        # that starts first burns it against a socket that does not exist yet
        # and dies with an error that reads like a firewall fault.
        self._cluster = LocalCluster(
            model_path=model_path,
            world_size=len(slots),
            backend=backend,
            pipeline=bool(cluster_settings.pipeline),
            python=python,
        )
        self._cluster.start(ranks=[0], ips=ips, **launch)
        ready_port = (
            hostfile.DEFAULT_RING_BASE_PORT
            if backend == "ring"
            else hostfile.DEFAULT_JACCL_COORDINATOR_PORT
        )
        # Probe the address rank 0 actually binds - the one in the hostfile.
        # `127.0.0.1` never answers: the ring backend binds the specific
        # interface it was given, so a loopback probe silently falls through to
        # the grace-period path and the readiness check checks nothing.
        if not self._cluster.wait_until_ready(ready_port, host=ips[0]):
            raise ClusterFormationError("rank 0 died before the world formed")

        for slot in slots[1:]:
            clients[slot.node_id].post(
                "/cluster/ranks/start",
                {
                    "model": model_id,
                    "backend": backend,
                    "world_size": len(slots),
                    "ranks": [slot.rank],
                    "ips": ips,
                    "pipeline": bool(cluster_settings.pipeline),
                    "coordinator": launch.get("coordinator"),
                    "ibv_devices": launch.get("ibv_devices"),
                },
            )

        reply = self._cluster.command({"op": CMD_LOAD}, timeout=LOAD_TIMEOUT_S)
        if not reply.get("ok"):
            raise ClusterFormationError(reply.get("error", "load failed"))
        logger.info(
            "cluster: formed %d ranks on %s for %s",
            len(slots),
            backend,
            model_id,
        )

    def _await_peers(self, cluster_settings: Any) -> dict[str, PeerClient]:
        """Wait for Bonjour to answer before declaring the fleet empty.

        Discovery is a poll, not a subscription, and a browse plus a resolve
        takes several seconds. The first request after a daemon restart
        therefore arrives before any peer is known - and because the engine
        pool marks a failed load sticky, failing instantly turns "Bonjour has
        not answered yet" into "this model is broken until you reload".
        """
        interval = float(
            getattr(cluster_settings, "discovery_interval_seconds", 5.0) or 5.0
        )
        deadline = time.monotonic() + max(PEER_DISCOVERY_GRACE_S, interval * 3)
        while True:
            clients = self._peer_clients(cluster_settings.cluster_key)
            if clients:
                return clients
            if time.monotonic() >= deadline:
                raise ClusterFormationError(
                    "no peers are visible; a cluster of one is just a local model"
                )
            time.sleep(min(1.0, interval))

    def _peer_clients(self, key: str) -> dict[str, PeerClient]:
        return {
            peer.info.node_id: PeerClient(peer.host, peer.info.port, key)
            for peer in self._peers_fn()
        }

    def _collect_reports(
        self, clients: dict[str, PeerClient], model_id: str
    ) -> tuple[list[topology.NodeReport], str]:
        """Gather every node's own view of itself.

        Each peer is also asked to resolve the model before anything is
        spawned. A peer that is missing the weights fails here, with a sentence
        naming the peer - rather than during `init()`, where it presents as the
        whole world hanging.
        """
        from omlx.cluster.discovery import default_node_id

        local_id = default_node_id()
        local_pre = preflight.run()
        self._blockers = [f"{local_id}: {c.detail}" for c in local_pre.blockers()]
        local = topology.probe_local(local_id)
        local.rdma_devices = list(local_pre.rdma_devices)
        local.rdma_ready = local_pre.rdma_ready
        reports = [local]

        for node_id, client in clients.items():
            payload = client.post("/cluster/report", {"model": model_id})
            if not payload.get("has_model"):
                raise ClusterFormationError(
                    f"{node_id} does not have {model_id!r} on disk"
                )
            if payload.get("python_error"):
                raise ClusterFormationError(
                    f"{node_id} cannot spawn a rank: {payload['python_error']}"
                )
            reports.append(_report_from_dict(node_id, payload))
            self._blockers += [f"{node_id}: {b}" for b in payload.get("blockers", [])]
        return reports, local_id

    @staticmethod
    def _rank_order(order: list[str], local_id: str) -> list[str]:
        """Rotate the planned order so this node is rank 0.

        Rank 0 has to be the node holding the engine - it is the only one with
        a pipe to a daemon that has an HTTP request waiting. For `jaccl-ring`
        the order is a cycle, and rotating a cycle keeps every rank next to the
        neighbours it was cabled to.
        """
        if local_id not in order:
            return [local_id] + [n for n in order if n != local_id]
        pivot = order.index(local_id)
        return order[pivot:] + order[:pivot]

    # -- serving -----------------------------------------------------------

    def stream(self, spec: GenerationSpec) -> Iterator[dict[str, Any]]:
        """Run one generation, yielding the worker's replies.

        Holds the pipe lock for the whole generation: see the class docstring
        for why that is correctness rather than throughput policy.
        """
        cluster = self._cluster
        if cluster is None:
            raise ClusterFormationError("no cluster is formed")

        with self._pipe:
            self._busy = True
            try:
                for reply in cluster.stream({"op": CMD_GENERATE, **spec.to_dict()}):
                    if not reply.get("ok", False):
                        raise RuntimeError(reply.get("error", "cluster generate failed"))
                    yield reply
            finally:
                self._busy = False

    def abort(self) -> bool:
        """Ask a running generation to stop. False when nothing is running."""
        cluster = self._cluster
        if cluster is None:
            return False
        return cluster.abort()

    # -- teardown ----------------------------------------------------------

    def teardown(self) -> None:
        """Stop every rank, local and remote. Safe to call when not formed."""
        cluster, self._cluster = self._cluster, None
        slots, self._slots = self._slots, []

        key = getattr(getattr(self._settings, "cluster", None), "cluster_key", "")
        for slot in slots:
            if slot.is_local:
                continue
            try:
                PeerClient(slot.host, slot.port, key).post("/cluster/ranks/stop")
            except Exception:  # noqa: BLE001 - a peer we cannot reach is killed
                logger.warning(  # by its own daemon's stale-rank sweep
                    "cluster: could not stop ranks on %s", slot.node_id
                )

        if cluster is not None:
            try:
                cluster.stop()
            except Exception:  # noqa: BLE001
                logger.exception("cluster: local teardown was not clean")
        self._model_id = ""
        self._backend = ""


def _report_from_dict(node_id: str, payload: dict[str, Any]) -> topology.NodeReport:
    """Rebuild a peer's `NodeReport` from its JSON."""
    buses = [
        topology.Bus(
            name=b.get("name", ""),
            domain_uuid=b.get("domain_uuid", ""),
            peer_domain_uuid=b.get("peer_domain_uuid"),
            peer_model=b.get("peer_model"),
        )
        for b in payload.get("buses", [])
    ]
    return topology.NodeReport(
        node_id=node_id,
        buses=buses,
        rdma_devices=list(payload.get("rdma_devices", [])),
        rdma_ready=bool(payload.get("rdma_ready", False)),
    )


async def form_async(manager: ClusterManager, model_id: str) -> ClusterStatus:
    """`form()` from the event loop, without blocking it for minutes."""
    return await asyncio.to_thread(manager.form, model_id)
