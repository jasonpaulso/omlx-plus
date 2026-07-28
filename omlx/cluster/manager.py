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
import queue
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

from omlx.cluster import hostfile, preflight, topology
from omlx.cluster.launcher import DeathWatch, LocalCluster, resolve_python
from omlx.cluster.protocol import CMD_GENERATE, CMD_LOAD, GenerationSpec

logger = logging.getLogger(__name__)

# A peer that cannot answer a control call in this long is not going to make a
# usable cluster member either. Generous because `/cluster/report` scans every
# model directory, which on an external volume takes tens of seconds.
PEER_TIMEOUT_S = 180.0
# Loading a shard reads weights off disk on every node at once.
LOAD_TIMEOUT_S = 900.0
# The longest a generation may go without a single reply. Generous, because a
# long prefill is silent - but finite, because a rank that dies mid-collective
# leaves rank 0 blocked inside mlx with no way to notice, and the request would
# otherwise hang until the client gives up.
GENERATE_IDLE_TIMEOUT_S = 600.0
# How long a formation waits for Bonjour to answer before calling the fleet
# empty. A browse plus a resolve is several seconds, and the first request
# after a daemon restart arrives well inside that.
PEER_DISCOVERY_GRACE_S = 20.0
# A liveness poll is one process-table read on the far side; a peer that needs
# longer than this to answer it is not healthy in any sense that matters.
ALIVE_POLL_TIMEOUT_S = 3.0


class ClusterFormationError(RuntimeError):
    """Formation failed. The cluster has already been torn back down."""


class ReplyRouter(threading.Thread):
    """Routes rank 0's replies to the requests that own them.

    The worker batches, so replies for several requests interleave on one
    pipe. Exactly one thread may read that pipe - the reader owns buffered
    bytes - and this is it; requests register a queue under their id and
    consume from that. EOF means rank 0 is gone (the deathwatch's kill lands
    here too), and every waiting request is told at once.
    """

    CLOSED = object()

    def __init__(self, reader: Any) -> None:
        super().__init__(name="cluster-replies", daemon=True)
        self._reader = reader
        self._lock = threading.Lock()
        self._queues: dict[str, queue.SimpleQueue] = {}
        self._closed = False

    def register(self, request_id: str) -> queue.SimpleQueue:
        with self._lock:
            if self._closed:
                q: queue.SimpleQueue = queue.SimpleQueue()
                q.put(self.CLOSED)
                return q
            q = self._queues.setdefault(request_id, queue.SimpleQueue())
            return q

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._queues.pop(request_id, None)

    def run(self) -> None:
        import json

        while True:
            line = self._reader.readline(None)
            if not line:
                break
            try:
                reply = json.loads(line)
            except ValueError:
                logger.warning("cluster: unparseable reply line, ignoring")
                continue
            request_id = str(reply.get("request_id") or "")
            with self._lock:
                q = self._queues.get(request_id)
            if q is not None:
                q.put(reply)
            elif not reply.get("ok", False):
                # An error nobody is waiting for is still worth a log line.
                logger.warning(
                    "cluster: worker error with no owner: %s",
                    reply.get("error", reply),
                )
        with self._lock:
            self._closed = True
            waiting = list(self._queues.values())
            self._queues.clear()
        for q in waiting:
            q.put(self.CLOSED)


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

    def get_json(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        """GET one of the peer's cluster routes."""
        import httpx

        response = httpx.get(
            f"{self.base_url}{path}",
            headers={"X-Cluster-Key": self._key},
            timeout=PEER_TIMEOUT_S if timeout is None else timeout,
        )
        response.raise_for_status()
        return response.json()

    def alive_ranks(self) -> list[int]:
        """Which ranks the peer's daemon says are running, right now."""
        import httpx

        response = httpx.get(
            f"{self.base_url}/cluster/ranks/alive",
            headers={"X-Cluster-Key": self._key},
            timeout=ALIVE_POLL_TIMEOUT_S,
        )
        response.raise_for_status()
        return [int(r) for r in response.json().get("ranks", [])]


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


def resolve_model_repo(settings: Any, model_id: str) -> str:
    """The HuggingFace repo `model_id` came from, or `""` if it has none.

    Only a model still backed by its HF cache entry can be fetched onto
    another node. A locally quantised or renamed model has no repo to pull
    from, and the honest answer is to say so rather than to offer a download
    that cannot start.
    """
    from omlx.model_discovery import discover_models_from_dirs

    discovered = discover_models_from_dirs(list(settings.get_effective_model_dirs()))
    found = discovered.get(model_id)
    return getattr(found, "source_repo_id", None) or "" if found else ""


class ClusterManager:
    """The leader's cluster: formation, serving, teardown.

    Serving is concurrent: the worker runs continuous batching in lockstep
    across the ranks (`omlx/cluster/batching.py`), so requests join and leave
    a shared batch rather than queueing on a lock. This side's job is to
    multiplex them onto rank 0's pipe - commands down stdin, replies routed
    back by request id.
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
        self._active = 0
        # Guards formation itself: after a fast-fail teardown, every queued
        # request tries to re-form at once, and they must take turns rather
        # than race half-built clusters against each other.
        self._form_lock = threading.Lock()
        self._watch: DeathWatch | None = None
        self._router: ReplyRouter | None = None

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
            busy=self._active > 0,
        )

    # -- formation ---------------------------------------------------------

    def form(self, model_id: str) -> ClusterStatus:
        """Bring a cluster up for `model_id`. Blocking; call it off-loop."""
        with self._form_lock:
            return self._form_locked(model_id)

    def _form_locked(self, model_id: str) -> ClusterStatus:
        if self.formed:
            if self._model_id == model_id:
                return self.status()
            self.teardown()

        self._error = ""
        try:
            self._form(model_id)
        except Exception as exc:  # noqa: BLE001 - reported, never half-formed
            # Read the attempted backend before tearing down; teardown clears it.
            attempted = self._backend
            self.teardown()
            if not self._may_fall_back(attempted):
                self._error = str(exc)
                logger.exception("cluster: formation failed")
                raise ClusterFormationError(str(exc)) from exc

            # `auto` means the fastest transport that *works*, not the fastest
            # the cabling suggests. RDMA can still fail at init for reasons
            # preflight cannot fully predict (the 2026-07-27 failure blamed on
            # an exhausted protection-domain pool turned out to be a PORT_DOWN
            # device - now selected against - but the class remains), and TCP
            # `ring` needs nothing but an IP route. Refusing to serve because
            # the fast path is unavailable would be the wrong answer.
            logger.warning(
                "cluster: %s failed to form (%s); retrying on TCP ring",
                attempted,
                exc,
            )
            fallback = f"{attempted} failed to start ({exc}); fell back to ring"
            try:
                self._form(model_id, force_backend="ring")
            except Exception as retry:  # noqa: BLE001
                self._error = str(retry)
                logger.exception("cluster: ring fallback also failed")
                self.teardown()
                raise ClusterFormationError(str(retry)) from retry
            self._reason = fallback
        return self.status()

    def _may_fall_back(self, attempted: str) -> bool:
        """True when the operator asked for `auto` and RDMA was what failed."""
        cluster = getattr(self._settings, "cluster", None)
        return bool(
            cluster
            and cluster.backend == "auto"
            and attempted in ("jaccl", "jaccl-ring")
        )

    def _form(self, model_id: str, *, force_backend: str | None = None) -> None:
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
        if force_backend is not None:
            backend, reason = force_backend, f"retrying on {force_backend}"
        elif cluster_settings.backend == "auto":
            backend, reason = chosen.backend, chosen.reason
        else:
            backend = cluster_settings.backend
            reason = f"pinned to {backend} by settings"
        self._backend = backend
        self._reason = reason
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
                        host=client.host,
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
        if not self._cluster.wait_until_ready(ready_port):
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
                    # The follower's own deathwatch calls back to this daemon;
                    # rank 0's ring address is this machine, but the daemon
                    # port is not derivable from it.
                    "leader_port": int(self._settings.server.port),
                },
            )

        # The watch starts before the load, because the load is where a death
        # is most expensive: every rank is blocked in the collective reading
        # weights, and without the watch a peer that dies here costs the whole
        # load timeout before anyone notices.
        self._start_watch(clients)

        reply = self._cluster.command(
            {
                "op": CMD_LOAD,
                # Batching is configured by the leader for every rank; a rank
                # reading its own settings could admit differently and diverge.
                "max_batch_size": int(
                    getattr(cluster_settings, "max_batch_size", 8) or 8
                ),
            },
            timeout=LOAD_TIMEOUT_S,
        )
        if not reply.get("ok"):
            raise ClusterFormationError(reply.get("error", "load failed"))

        # Rank 0 has always reported the world it actually joined; nothing read
        # it, and `status()` reports the world we *planned*. When a backend
        # fails to come up, mlx hands back a one-process group, so rank 0 loads
        # the whole model and serves alone while every log line says the fleet
        # formed. Fail here instead, which puts `auto` onto the ring fallback.
        joined = reply.get("world_size")
        if joined != len(slots):
            raise ClusterFormationError(
                f"rank 0 joined a world of {joined}, not {len(slots)}: "
                f"the {backend} backend did not form across every node"
            )

        # From here on, replies interleave across requests; the router owns
        # the read side of the pipe. Nothing may call `command()` again on
        # this cluster.
        self._router = ReplyRouter(self._cluster.reply_reader())
        self._router.start()
        logger.info(
            "cluster: formed %d ranks on %s for %s",
            len(slots),
            backend,
            model_id,
        )

    def _start_watch(self, clients: dict[str, PeerClient]) -> None:
        """Watch every rank, local and remote, from formation onward."""
        cluster = self._cluster
        if cluster is None:
            return
        checks: list[tuple[str, Any]] = [
            ("local rank 0", lambda: bool(cluster.alive_ranks()))
        ]
        for slot in self._slots:
            if slot.is_local:
                continue
            client = clients.get(slot.node_id)
            if client is None:
                continue
            checks.append(
                (
                    f"rank {slot.rank} on {slot.node_id}",
                    _peer_alive_check(client, slot.rank),
                )
            )
        self._watch = DeathWatch(checks, self._on_rank_death)
        self._watch.start()

    def _on_rank_death(self, label: str, reason: str) -> None:
        """A rank is gone. Kill ours, then tear the formation down.

        Runs on the deathwatch thread. The kill comes first because it is the
        part that matters: closing rank 0's reply pipe is what makes the
        in-flight request fail now rather than at the idle timeout. Requests
        queued behind it then find no cluster formed and fail immediately too -
        and the next fresh request re-forms.
        """
        # The callback runs *on* the watch thread, which makes staleness
        # checkable by identity: a watch the manager no longer owns belongs to
        # a formation that is already gone, and acting on it would kill the
        # healthy formation that replaced it.
        if threading.current_thread() is not self._watch:
            return
        self._error = f"{label} died ({reason}); the cluster was torn down"
        cluster = self._cluster
        if cluster is not None:
            cluster.kill()
        self.teardown()

    def alive_local_ranks(self) -> list[int]:
        """This daemon's own leader-side ranks that are still running."""
        cluster = self._cluster
        return cluster.alive_ranks() if cluster is not None else []

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
        """Peers, addressed by IPv4 rather than by the name Bonjour gave.

        The control plane talks to the address, not the `.local` name, because
        resolving one from inside the daemon costs **over a minute** per call -
        long enough that rank 0 spent its entire connect window waiting for an
        HTTP request to leave the machine, and died with a timeout that looked
        like the peer's fault. Discovery has already resolved the name once;
        doing it again per request is both slow and a second chance to pick a
        link-local address.

        Peers advertising a different key are skipped. Discovery deliberately
        keeps them - the admin UI needs to be able to say "that Mac has a
        different key" rather than show an empty list - but including one here
        would send it our key, take a 403 from `/cluster/report`, and fail the
        whole formation. One unrelated oMLX install on the LAN would otherwise
        be enough to stop this cluster forming at all. The fingerprint is not
        an authorisation decision: `verify_cluster_key` still compares the key
        itself, and a peer filtered here is only ever excluded.
        """
        from omlx.cluster.discovery import matches_fingerprint

        clients: dict[str, PeerClient] = {}
        for peer in self._peers_fn():
            peer_fingerprint = getattr(peer.info, "key_fingerprint", "")
            if peer_fingerprint and not matches_fingerprint(key, peer_fingerprint):
                logger.info(
                    "cluster: ignoring %s at %s - it advertises a different "
                    "cluster key",
                    peer.info.node_id,
                    peer.host,
                )
                continue
            try:
                host = resolve_ipv4(peer.host, peer.info.port)
            except ClusterFormationError as exc:
                logger.warning("cluster: skipping peer %s: %s", peer.host, exc)
                continue
            clients[peer.info.node_id] = PeerClient(host, peer.info.port, key)
        return clients

    def _collect_reports(
        self, clients: dict[str, PeerClient], model_id: str
    ) -> tuple[list[topology.NodeReport], str]:
        """Gather every node's own view of itself.

        Each peer is also asked to resolve the model before anything is
        spawned. A peer that is missing the weights fails here, with a sentence
        naming the peer - rather than during `init()`, where it presents as the
        whole world hanging.
        """
        import httpx

        from omlx.cluster.discovery import default_node_id

        local_id = default_node_id()
        local_pre = preflight.run()
        self._blockers = [f"{local_id}: {c.detail}" for c in local_pre.blockers()]
        local = topology.probe_local(local_id)
        local.rdma_devices = list(local_pre.rdma_devices)
        local.active_rdma_devices = list(local_pre.rdma_active_devices)
        local.rdma_ready = local_pre.rdma_ready
        reports = [local]

        for node_id, client in clients.items():
            try:
                payload = client.post("/cluster/report", {"model": model_id})
            except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as exc:
                # Name the peer and the route; the raw errno is useless in a
                # log. The macOS hint is earned: Local Network privacy denies
                # third-party interpreters (python) while Apple binaries
                # (curl, nc) still get through, so "the peer answers curl but
                # not oMLX" reads like a network fault and is a permission.
                raise ClusterFormationError(
                    f"cannot reach {node_id} at {client.host}:{client.port}: "
                    f"{exc}. If other tools on this machine reach the peer "
                    "while oMLX cannot, macOS Local Network privacy has "
                    "likely denied this interpreter - check System Settings "
                    "> Privacy & Security > Local Network."
                ) from exc
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

        Concurrent by design: the request joins the worker's batch, and its
        replies come back through the router under its request id. The idle
        timeout is a backstop - the deathwatch is what normally notices a
        dead rank, in seconds.
        """
        cluster = self._cluster
        router = self._router
        if cluster is None or router is None:
            raise ClusterFormationError(self._error or "no cluster is formed")

        if not spec.request_id:
            spec.request_id = uuid.uuid4().hex
        replies = router.register(spec.request_id)
        self._active += 1
        try:
            cluster.submit({"op": CMD_GENERATE, **spec.to_dict()})
            while True:
                try:
                    reply = replies.get(timeout=GENERATE_IDLE_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError(
                        f"rank 0 sent nothing for {GENERATE_IDLE_TIMEOUT_S:.0f}s; "
                        "the collective is most likely blocked on a rank that died"
                    ) from None
                if reply is ReplyRouter.CLOSED:
                    raise RuntimeError(
                        self._error or "rank 0 closed its reply channel"
                    )
                if not reply.get("ok", False):
                    raise RuntimeError(reply.get("error", "cluster generate failed"))
                yield reply
                if reply.get("done"):
                    return
        except (OSError, ValueError) as exc:
            # Writing into a dead process's stdin is the symptom; if the
            # deathwatch recorded the cause, that is the error worth raising.
            if self._error:
                raise RuntimeError(self._error) from exc
            raise
        finally:
            self._active -= 1
            router.unregister(spec.request_id)

    def abort(self, request_id: str = "") -> bool:
        """Stop one request - or all of them, for an empty id."""
        cluster = self._cluster
        if cluster is None:
            return False
        return cluster.abort(request_id)

    # -- teardown ----------------------------------------------------------

    def teardown(self) -> None:
        """Stop every rank, local and remote. Safe to call when not formed."""
        watch, self._watch = self._watch, None
        if watch is not None:
            watch.stop()

        # The router thread needs no explicit stop: killing the ranks below
        # closes the pipe, it sees EOF, and it wakes every waiting request.
        self._router = None

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


def _peer_alive_check(client: PeerClient, rank: int) -> Callable[[], bool | None]:
    """A deathwatch check for one remote rank.

    A reachable daemon that does not list the rank is a definitive death; a
    daemon that cannot be reached at all might just be a LAN blip, so that is
    only a strike.
    """

    def check() -> bool | None:
        try:
            return rank in client.alive_ranks()
        except Exception:  # noqa: BLE001 - unreachable, not (yet) dead
            return None

    return check


def _report_from_dict(node_id: str, payload: dict[str, Any]) -> topology.NodeReport:
    """Rebuild a peer's `NodeReport` from its JSON."""
    buses = [
        topology.Bus(
            name=b.get("name", ""),
            domain_uuid=b.get("domain_uuid", ""),
            peer_domain_uuid=b.get("peer_domain_uuid"),
            peer_model=b.get("peer_model"),
            receptacle=b.get("receptacle"),
            rdma_device=b.get("rdma_device"),
        )
        for b in payload.get("buses", [])
    ]
    return topology.NodeReport(
        node_id=node_id,
        buses=buses,
        rdma_devices=list(payload.get("rdma_devices", [])),
        active_rdma_devices=list(payload.get("active_rdma_devices", [])),
        rdma_ready=bool(payload.get("rdma_ready", False)),
    )


async def form_async(manager: ClusterManager, model_id: str) -> ClusterStatus:
    """`form()` from the event loop, without blocking it for minutes."""
    return await asyncio.to_thread(manager.form, model_id)
