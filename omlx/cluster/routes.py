# SPDX-License-Identifier: Apache-2.0
"""The cluster control plane, served by every oMLX daemon.

Two audiences, deliberately separated:

- `/cluster/*` is **peer-to-peer**. A leader forming a cluster calls these on
  the other machines. They are authenticated with the shared `cluster_key`
  rather than the node's API key, because the two are not the same permission:
  a cluster key says "you may make this machine a rank", not "you may use this
  machine's models".
- `/admin/api/cluster/*` is **operator-facing**, behind the daemon's normal
  API key, and is what the admin UI reads.

The peer half is what makes SSH unnecessary. Each node already runs a daemon
with credentials, a model directory and a process supervisor; asking it over
HTTP to spawn its own rank uses all three, and never assumes the two machines
share a filesystem, a username or an install path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from omlx.admin.auth import require_admin
from omlx.cluster import preflight, topology
from omlx.cluster.launcher import DeathWatch, LocalCluster, resolve_python

logger = logging.getLogger(__name__)

peer_router = APIRouter(prefix="/cluster", tags=["cluster"])
admin_router = APIRouter(
    prefix="/admin/api/cluster",
    tags=["cluster"],
    dependencies=[Depends(require_admin)],
)

# Set by bootstrap.install(). Kept as callables so the routes never hold a
# stale settings object across a reload.
_get_settings: Callable[[], Any] | None = None
_get_manager: Callable[[], Any] | None = None

# This node's ranks when it is a *follower*. The leader's ranks live in its
# ClusterManager; a follower has no manager because it decides nothing.
_follower: LocalCluster | None = None
# The follower's own deathwatch on itself and on the leader. Without it, a
# leader that dies leaves this node's rank blocked in a collective holding
# its shard of the weights until someone notices by hand.
_follower_watch: DeathWatch | None = None

# What `/cluster/report` worked out about this node, keyed by model id.
#
# Not an optimisation. Resolving a model means scanning every model directory,
# which on a machine with a large external volume takes tens of seconds - and
# `/cluster/ranks/start` is called with rank 0 already up and inside the ring
# backend's bounded connect window. Doing that work at report time, before rank
# 0 exists, is the difference between a cluster forming and rank 0 timing out
# against a peer that was still listing its disk.
_resolved: dict[str, str] = {}
_resolved_python: str = ""

# This machine's own description, for the operator surface. Cached because
# resolving it shells out to `sysctl`, and the admin UI polls status every few
# seconds; none of it changes while the process lives.
_local_node: dict[str, Any] | None = None


def reset_resolved() -> None:
    """Forget what `/cluster/report` resolved.

    Called on shutdown, which includes the shutdown half of a configuration
    change. A path resolved under one set of model directories is not a path
    under the next set.
    """
    global _resolved_python
    _resolved.clear()
    _resolved_python = ""


def configure(
    get_settings: Callable[[], Any], get_manager: Callable[[], Any]
) -> None:
    """Wire the routes to the daemon's state."""
    global _get_settings, _get_manager
    _get_settings, _get_manager = get_settings, get_manager


def _settings() -> Any:
    if _get_settings is None:
        raise HTTPException(status_code=503, detail="cluster routes are not configured")
    return _get_settings()


def verify_cluster_key(x_cluster_key: str = Header(default="")) -> bool:
    """Authenticate a peer.

    Compared in constant time, and a node with clustering off refuses outright
    rather than leaking whether a key would have matched.
    """
    import hmac

    cluster = getattr(_settings(), "cluster", None)
    if cluster is None or not cluster.enabled or not cluster.cluster_key:
        raise HTTPException(status_code=403, detail="clustering is not enabled here")
    if not hmac.compare_digest(x_cluster_key, cluster.cluster_key):
        raise HTTPException(status_code=403, detail="cluster key mismatch")
    return True


# =============================================================================
# Peer control plane
# =============================================================================


@peer_router.post("/report", dependencies=[Depends(verify_cluster_key)])
async def report(payload: dict[str, Any]) -> dict[str, Any]:
    """Everything the leader needs to plan a cluster that includes this node.

    Deliberately answers the two questions that otherwise fail late and
    unreadably: does this machine have the weights, and can its interpreter
    import oMLX. A missing model or a wrong interpreter surfaces here, named,
    instead of as the entire world hanging inside `mx.distributed.init()`.
    """
    import asyncio

    settings = _settings()
    model_id = payload.get("model", "")

    def _gather() -> dict[str, Any]:
        from omlx.cluster.discovery import default_node_id
        from omlx.cluster.manager import resolve_model_path

        checks = preflight.run()
        local = topology.probe_local(default_node_id())

        global _resolved_python

        has_model, python_error = False, ""
        if model_id:
            try:
                _resolved[model_id] = resolve_model_path(settings, model_id)
                has_model = True
            except Exception as exc:  # noqa: BLE001
                _resolved.pop(model_id, None)
                logger.info("cluster: peer cannot serve %s: %s", model_id, exc)
        try:
            _resolved_python = resolve_python()
        except Exception as exc:  # noqa: BLE001
            _resolved_python = ""
            python_error = str(exc)

        return {
            "node_id": local.node_id,
            "buses": [
                {
                    "name": b.name,
                    "domain_uuid": b.domain_uuid,
                    "peer_domain_uuid": b.peer_domain_uuid,
                    "peer_model": b.peer_model,
                    "receptacle": b.receptacle,
                    # Resolved locally: only this node can map its own
                    # receptacles to RDMA devices.
                    "rdma_device": b.rdma_device,
                }
                for b in local.buses
            ],
            "rdma_devices": list(checks.rdma_devices),
            "active_rdma_devices": list(checks.rdma_active_devices),
            "rdma_ready": checks.rdma_ready,
            "blockers": [c.detail for c in checks.blockers()],
            "has_model": has_model,
            "python_error": python_error,
            "chip": checks.chip,
            "macos": list(checks.macos),
        }

    return await asyncio.to_thread(_gather)


@peer_router.post("/ranks/start", dependencies=[Depends(verify_cluster_key)])
async def start_ranks(payload: dict[str, Any]) -> dict[str, Any]:
    """Spawn this node's ranks for a cluster the leader has already started.

    Only ever called *after* rank 0 is listening. A peer that starts first
    burns the ring backend's bounded connect window against a socket that does
    not exist yet and dies with an error that reads like a firewall fault.
    """
    import asyncio

    settings = _settings()
    if _follower is not None:
        await asyncio.to_thread(_stop_follower)

    def _spawn() -> dict[str, Any]:
        global _follower
        from omlx.cluster.manager import resolve_model_path

        model_id = payload["model"]
        cluster = LocalCluster(
            # Both of these were resolved at report time, before rank 0
            # existed. Falling back to resolving here is correct but slow, and
            # slow here means rank 0 times out waiting.
            model_path=_resolved.get(model_id)
            or resolve_model_path(settings, model_id),
            world_size=int(payload["world_size"]),
            backend=payload.get("backend", "ring"),
            pipeline=bool(payload.get("pipeline", False)),
            python=_resolved_python or resolve_python(),
        )
        extra: dict[str, Any] = {}
        if payload.get("coordinator"):
            extra["coordinator"] = payload["coordinator"]
        if payload.get("ibv_devices"):
            extra["ibv_devices"] = payload["ibv_devices"]
        cluster.start(
            ranks=[int(r) for r in payload["ranks"]],
            ips=list(payload["ips"]),
            **extra,
        )
        _follower = cluster
        _start_follower_watch(cluster, payload)
        return {"ok": True, "ranks": [r.rank for r in cluster.ranks]}

    try:
        return await asyncio.to_thread(_spawn)
    except Exception as exc:  # noqa: BLE001 - the leader needs the reason
        logger.exception("cluster: could not start ranks")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@peer_router.get("/ranks/alive", dependencies=[Depends(verify_cluster_key)])
async def ranks_alive() -> dict[str, Any]:
    """The ranks running on this machine, read from the process table.

    This is what a deathwatch polls, so it must stay cheap - one process poll
    per rank, no disk, no subprocess - and it must never probe a collective's
    port to find out (a probe is not passive; it poisons the handshake).
    Covers both roles: follower ranks and, on the leader, its own rank 0.
    """
    ranks: list[int] = []
    if _follower is not None:
        ranks += _follower.alive_ranks()
    manager = _get_manager() if _get_manager is not None else None
    if manager is not None:
        ranks += manager.alive_local_ranks()
    return {"ranks": sorted(set(ranks))}


@peer_router.post("/ranks/stop", dependencies=[Depends(verify_cluster_key)])
async def stop_ranks() -> dict[str, Any]:
    """Kill this node's ranks.

    Always safe to call, and worth calling even when the leader thinks nothing
    is running: a worker left alive from a failed run holds the ring port, and
    the *next* rank 0 then silently fails to own it.
    """
    import asyncio

    stopped = await asyncio.to_thread(_stop_follower)
    return {"ok": True, "stopped": stopped}


@peer_router.post("/models/download", dependencies=[Depends(verify_cluster_key)])
async def start_model_download(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch a model onto this node, so it can join a cluster that needs it.

    Reuses the daemon's own downloader rather than inventing a transfer: the
    weights land in this node's model directory, named the way this node's
    discovery expects, and the existing task list can be polled for progress.
    The leader never sends bytes - it sends a repo id, and this node pulls.
    """
    from omlx.admin import routes as admin_routes

    downloader = admin_routes._hf_downloader
    if downloader is None:
        raise HTTPException(status_code=503, detail="downloader is not initialised")

    repo_id = str(payload.get("repo_id", "")).strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")

    try:
        task = await downloader.start_download(repo_id, payload.get("hf_token"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "task": task.to_dict()}


@peer_router.get("/models/download/{task_id}", dependencies=[Depends(verify_cluster_key)])
async def model_download_status(task_id: str) -> dict[str, Any]:
    """Progress of a download this node was asked to start."""
    from omlx.admin import routes as admin_routes

    downloader = admin_routes._hf_downloader
    if downloader is None:
        raise HTTPException(status_code=503, detail="downloader is not initialised")

    for task in downloader.get_tasks():
        if task.get("task_id") == task_id:
            return {"task": task}
    raise HTTPException(status_code=404, detail="no such download task")


def _start_follower_watch(cluster: LocalCluster, payload: dict[str, Any]) -> None:
    """Watch this node's own ranks, and the leader that owns them.

    The leader's address is rank 0's ring IP - rank 0 is always on the
    leader's machine - but its daemon port has to be told to us, so older
    leaders that do not send one simply get no leader check.
    """
    global _follower_watch

    checks: list[tuple[str, Any]] = [
        ("this node's ranks", lambda: bool(cluster.alive_ranks()))
    ]
    leader_port = payload.get("leader_port")
    ips = list(payload.get("ips") or [])
    key = getattr(getattr(_settings(), "cluster", None), "cluster_key", "")
    if leader_port and ips and key:
        checks.append(
            ("the leader", _leader_alive_check(ips[0], int(leader_port), key))
        )

    def on_death(label: str, reason: str) -> None:
        import threading

        # Runs on the watch thread; a watch that is no longer the current one
        # belongs to a formation `ranks/start` has already replaced.
        if threading.current_thread() is not _follower_watch:
            return
        logger.error(
            "cluster: %s %s; stopping this node's ranks", label, reason
        )
        cluster.kill()
        _stop_follower()

    _follower_watch = DeathWatch(checks, on_death)
    _follower_watch.start()


def _leader_alive_check(host: str, port: int, key: str):
    """True while the leader's daemon still runs a rank 0."""
    import httpx

    from omlx.cluster.manager import ALIVE_POLL_TIMEOUT_S

    def check() -> bool | None:
        try:
            response = httpx.get(
                f"http://{host}:{port}/cluster/ranks/alive",
                headers={"X-Cluster-Key": key},
                timeout=ALIVE_POLL_TIMEOUT_S,
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - unreachable is a strike, not a death
            return None
        return 0 in response.json().get("ranks", [])

    return check


def _stop_follower() -> int:
    global _follower, _follower_watch
    watch, _follower_watch = _follower_watch, None
    if watch is not None:
        watch.stop()
    cluster, _follower = _follower, None
    if cluster is None:
        return 0
    count = len(cluster.ranks)
    cluster.stop()
    return count


def shutdown_follower() -> None:
    """Called from bootstrap when the daemon itself is going away."""
    try:
        _stop_follower()
    except Exception:  # noqa: BLE001 - never block shutdown
        logger.exception("cluster: follower ranks did not stop cleanly")


# =============================================================================
# Operator surface
# =============================================================================


def _local_description() -> dict[str, Any]:
    """This machine, as a peer would see it."""
    global _local_node

    if _local_node is None:
        from omlx import __version__
        from omlx.cluster.discovery import (
            default_chip,
            default_hostname,
            default_node_id,
            default_ram_gb,
        )

        _local_node = {
            "node_id": default_node_id(),
            "hostname": default_hostname(),
            "chip": default_chip(),
            "ram_gb": default_ram_gb(),
            "version": str(__version__),
        }
    return dict(_local_node)


@admin_router.get("/status")
async def cluster_status() -> dict[str, Any]:
    """Formation state, chosen backend, rank order and any blockers.

    Answers on a node where clustering has never been enabled, because that is
    the node the operator is looking at when they come to turn it on. The
    cluster key is deliberately absent: this is polled every few seconds, and
    a secret does not belong in a heartbeat.
    """
    import asyncio

    from omlx.cluster import bootstrap

    cluster = getattr(_settings(), "cluster", None)
    manager = _get_manager() if _get_manager is not None else None
    status = (
        manager.status().to_dict()
        if manager is not None
        else {"enabled": False, "formed": False}
    )

    # Read from settings rather than from the manager: with clustering off
    # there is no manager, and "off" is precisely what the UI needs to render.
    status["enabled"] = bool(cluster and cluster.enabled)
    status["configured"] = bool(cluster and cluster.enabled and cluster.cluster_key)

    local = await asyncio.to_thread(_local_description)
    local["port"] = _settings().server.port
    status["local"] = local

    key = cluster.cluster_key if cluster else ""
    status["peers"] = [
        {
            "node_id": p.info.node_id,
            "hostname": p.info.hostname,
            "host": p.host,
            "port": p.info.port,
            "chip": p.info.chip,
            "ram_gb": p.info.ram_gb,
            "version": p.info.version,
            # A peer advertising a different key will never join. Saying so is
            # the difference between a pairing mistake and an empty list.
            # Display only - the authorisation decision stays with
            # `verify_cluster_key`, which compares the key itself.
            "key_match": _fingerprint_matches(key, p.info.key_fingerprint),
        }
        for p in bootstrap.peers()
    ]
    status["follower_ranks"] = [] if _follower is None else [
        r.rank for r in _follower.ranks
    ]
    return status


def _fingerprint_matches(key: str, peer_fingerprint: str) -> bool | None:
    """Whether a peer advertises our key. None when we have no key to compare."""
    from omlx.cluster.discovery import matches_fingerprint

    if not key or not peer_fingerprint:
        return None
    return matches_fingerprint(key, peer_fingerprint)


@admin_router.get("/preflight")
async def cluster_preflight() -> dict[str, Any]:
    """This node's own capability report, for the admin UI's checklist."""
    import asyncio

    checks = await asyncio.to_thread(preflight.run)
    return {
        "macos": list(checks.macos),
        "chip": checks.chip,
        "rdma_enabled": checks.rdma_enabled,
        "rdma_devices": checks.rdma_devices,
        "active_rdma_devices": checks.rdma_active_devices,
        "bridged_interfaces": checks.bridged_interfaces,
        "tb_max_gbps": checks.tb_max_gbps,
        "rdma_ready": checks.rdma_ready,
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail, "remedy": c.remedy}
            for c in checks.checks
        ],
    }


@admin_router.post("/teardown")
async def cluster_teardown() -> dict[str, Any]:
    """Stop every rank, everywhere. The recovery path for a wedged run."""
    import asyncio

    manager = _get_manager() if _get_manager is not None else None
    if manager is not None:
        await asyncio.to_thread(manager.teardown)
    await asyncio.to_thread(_stop_follower)
    return {"ok": True}


# =============================================================================
# Configuration
# =============================================================================

BACKENDS = ("auto", "ring", "jaccl", "jaccl-ring")

# Longer than the daemon's own API key minimum, and deliberately so. Every node
# broadcasts a truncated digest of this key over Bonjour to the whole LAN, so a
# short one can be recovered offline and then used to spawn processes here. A
# generated key is 32 bytes; a typed one has to be non-trivial.
KEY_MIN_LENGTH = 16


class ClusterConfigRequest(BaseModel):
    """A partial update. Every field left unset keeps its current value."""

    enabled: bool | None = None
    cluster_key: str | None = None
    backend: str | None = None
    model: str | None = None
    pipeline: bool | None = None
    max_batch_size: int | None = None
    discovery_interval_seconds: float | None = None


def _require_real_auth() -> None:
    """Refuse configuration writes when admin auth has been switched off.

    `auth.skip_api_key_verification` makes `require_admin` return True for
    every caller. That is the operator's choice for the rest of the admin API,
    but this endpoint is different in kind: writing a cluster key and enabling
    clustering is what causes `peer_router` to be served, and `peer_router`
    spawns processes on this machine. Turning an unauthenticated config write
    into unauthenticated remote execution is not a trade anyone opted into by
    skipping key verification.
    """
    auth = getattr(_settings(), "auth", None)
    if auth is not None and getattr(auth, "skip_api_key_verification", False):
        raise HTTPException(
            status_code=403,
            detail=(
                "Cluster configuration cannot be changed while "
                "auth.skip_api_key_verification is enabled. Configure an API "
                "key first, or edit cluster settings in settings.json."
            ),
        )


def _validate_cluster_key(key: str) -> None:
    """The same wire constraints as the daemon's API key, plus a length floor.

    The ASCII rule is not cosmetic: this key is sent as the `X-Cluster-Key`
    header, which the ASGI layer decodes as latin-1, so a non-ASCII key can
    never be matched over the wire and would fail as a silent 403 on every
    peer call.
    """
    if not key:
        return
    if len(key) < KEY_MIN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Cluster key must be at least {KEY_MIN_LENGTH} characters",
        )
    if any(c.isspace() for c in key):
        raise HTTPException(
            status_code=400, detail="Cluster key must not contain whitespace"
        )
    if not key.isprintable() or not key.isascii():
        raise HTTPException(
            status_code=400,
            detail="Cluster key must contain only printable ASCII characters",
        )


def _config_dict(cluster: Any) -> dict[str, Any]:
    return {
        "enabled": cluster.enabled,
        "cluster_key": cluster.cluster_key,
        "backend": cluster.backend,
        "model": cluster.model,
        "pipeline": cluster.pipeline,
        "max_batch_size": cluster.max_batch_size,
        "discovery_interval_seconds": cluster.discovery_interval_seconds,
    }


@admin_router.get("/config")
async def get_cluster_config() -> dict[str, Any]:
    """The cluster settings, for the admin form.

    Returns the key in the clear, matching how `GET /admin/api/global-settings`
    already treats the daemon's API key: the operator has to be able to read it
    off one Mac to type it into the other, and this endpoint is fetched on
    demand rather than polled.
    """
    cluster = getattr(_settings(), "cluster", None)
    if cluster is None:
        raise HTTPException(status_code=503, detail="settings are not loaded")
    config = _config_dict(cluster)
    config["backends"] = list(BACKENDS)
    config["key_min_length"] = KEY_MIN_LENGTH
    return config


@admin_router.post("/key")
async def generate_cluster_key() -> dict[str, str]:
    """Mint a key for pairing. Not saved until the form is saved."""
    import secrets

    _require_real_auth()
    return {"cluster_key": secrets.token_urlsafe(32)}


@admin_router.post("/config")
async def set_cluster_config(
    payload: ClusterConfigRequest, request: Request
) -> dict[str, Any]:
    """Write the cluster settings and apply them without a restart.

    Applying means tearing the cluster down and standing it back up, because
    discovery reads its port and interval once at construction and a formed
    cluster holds a manager built from the old settings. That is why a cluster
    with a request in flight is refused rather than quietly interrupted.
    """
    from omlx.cluster import bootstrap

    _require_real_auth()

    settings = _settings()
    cluster = getattr(settings, "cluster", None)
    if cluster is None:
        raise HTTPException(status_code=503, detail="settings are not loaded")

    manager = _get_manager() if _get_manager is not None else None
    if manager is not None:
        status = manager.status()
        if status.formed and status.busy:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The cluster is serving a request. Wait for it to finish, "
                    "or tear the cluster down first."
                ),
            )

    # A follower has no manager - it decides nothing - so the check above does
    # not cover it, and applying here would kill this node's ranks out from
    # under a leader that is mid-request. Stricter than the leader rule, and
    # deliberately: this node cannot see whether the request it is serving a
    # shard of is still in flight, so holding a rank at all is enough.
    if _follower is not None and _follower.alive_ranks():
        raise HTTPException(
            status_code=409,
            detail=(
                "This Mac is holding ranks for a cluster led by another Mac. "
                "Changing the configuration here would stop them mid-request. "
                "Tear the cluster down from the leader first."
            ),
        )

    if payload.backend is not None and payload.backend not in BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend: {payload.backend!r} (expected one of {list(BACKENDS)})",
        )
    if payload.max_batch_size is not None and not 1 <= payload.max_batch_size <= 64:
        raise HTTPException(
            status_code=400, detail="max_batch_size must be between 1 and 64"
        )
    if payload.discovery_interval_seconds is not None and not (
        1.0 <= payload.discovery_interval_seconds <= 300.0
    ):
        raise HTTPException(
            status_code=400,
            detail="discovery_interval_seconds must be between 1 and 300",
        )
    if payload.cluster_key is not None:
        _validate_cluster_key(payload.cluster_key)

    # Enabling without a key would start discovery, advertise nothing anyone
    # can authenticate against, and leave the operator staring at a peer list
    # that never fills in.
    key = cluster.cluster_key if payload.cluster_key is None else payload.cluster_key
    enabled = cluster.enabled if payload.enabled is None else payload.enabled
    if enabled and not key:
        raise HTTPException(
            status_code=400,
            detail="Set a cluster key before enabling clustering.",
        )

    previous_model = cluster.model
    for name, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(cluster, name, value)

    settings.save()

    advertising = await bootstrap.reapply(
        request.app, settings, previous_model=previous_model
    )
    logger.info(
        "cluster: configuration applied (enabled=%s, advertising=%s)",
        cluster.enabled,
        advertising,
    )
    return {"ok": True, "advertising": advertising, "config": _config_dict(cluster)}


@admin_router.post("/peers/check")
async def check_peers(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask every peer whether it can serve a model, on demand.

    Behind a button rather than on the status poll on purpose: with a model set
    this makes each peer scan its model directories, which on a machine with a
    large external volume takes tens of seconds.
    """
    import asyncio

    from omlx.cluster import bootstrap
    from omlx.cluster.manager import PeerClient, resolve_model_repo

    # A write in effect: it makes every matched peer scan its model
    # directories, which takes tens of seconds each.
    _require_real_auth()

    cluster = getattr(_settings(), "cluster", None)
    if cluster is None or not cluster.cluster_key:
        raise HTTPException(status_code=400, detail="no cluster key is configured")

    model_id = str(payload.get("model", ""))
    key = cluster.cluster_key

    def _check(peer: Any) -> dict[str, Any]:
        # Same rule as formation: a peer advertising a different key is not
        # called at all. Calling it would hand our key to a machine that is
        # not in this cluster, and return a raw 403 where the page already
        # knows the answer.
        if _fingerprint_matches(key, peer.info.key_fingerprint) is False:
            return {
                "node_id": peer.info.node_id,
                "ok": False,
                "key_match": False,
                "error": "this node advertises a different cluster key",
            }
        client = PeerClient(peer.host, peer.info.port, key)
        try:
            report = client.post("/cluster/report", {"model": model_id})
        except Exception as exc:  # noqa: BLE001 - the operator wants the reason
            return {"node_id": peer.info.node_id, "ok": False, "error": str(exc)}
        return {
            "node_id": peer.info.node_id,
            "ok": True,
            "has_model": report.get("has_model", False),
            "rdma_ready": report.get("rdma_ready", False),
            "blockers": report.get("blockers", []),
            "python_error": report.get("python_error", ""),
        }

    # Off the event loop: resolving a repo scans every model directory, which
    # is the same tens of seconds this endpoint sits behind a button for.
    repo_id = (
        await asyncio.to_thread(resolve_model_repo, _settings(), model_id)
        if model_id
        else ""
    )
    results = await asyncio.gather(
        *(asyncio.to_thread(_check, peer) for peer in bootstrap.peers())
    )
    return {
        "model": model_id,
        # What a missing model can be fixed with. Empty means the model has no
        # HuggingFace origin - locally quantised or renamed - so there is
        # nothing for a peer to pull and the operator has to copy it across
        # themselves.
        "repo_id": repo_id,
        "peers": list(results),
    }


# =============================================================================
# Getting the model onto a peer
# =============================================================================

# Downloads this node has asked peers to run, keyed by node id. Kept so
# progress can be polled without the browser having to remember task ids
# across a reload.
_peer_downloads: dict[str, dict[str, Any]] = {}


def _peer_by_node_id(node_id: str) -> Any:
    from omlx.cluster import bootstrap

    for peer in bootstrap.peers():
        if peer.info.node_id == node_id:
            return peer
    return None


@admin_router.post("/peers/download")
async def start_peer_downloads(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask the named peers to fetch the cluster model.

    The leader sends a repo id, never bytes. Each peer pulls into its own
    model directory, which is the only place its discovery will look and the
    only path its rank could load from.
    """
    import asyncio

    from omlx.cluster.manager import PeerClient, resolve_model_repo

    _require_real_auth()

    cluster = getattr(_settings(), "cluster", None)
    if cluster is None or not cluster.cluster_key:
        raise HTTPException(status_code=400, detail="no cluster key is configured")

    model_id = str(payload.get("model", "")).strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")

    repo_id = await asyncio.to_thread(resolve_model_repo, _settings(), model_id)
    if not repo_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{model_id!r} has no source repository - it was quantised or "
                "renamed locally - so a peer has nothing to download. Copy the "
                "model directory to the other Mac by hand."
            ),
        )

    node_ids = [str(n) for n in (payload.get("node_ids") or [])]
    key = cluster.cluster_key

    def _start(node_id: str) -> dict[str, Any]:
        peer = _peer_by_node_id(node_id)
        if peer is None:
            return {"node_id": node_id, "ok": False, "error": "peer is no longer visible"}
        if _fingerprint_matches(key, peer.info.key_fingerprint) is False:
            return {
                "node_id": node_id,
                "ok": False,
                "error": "this node advertises a different cluster key",
            }
        client = PeerClient(peer.host, peer.info.port, key)
        try:
            reply = client.post("/cluster/models/download", {"repo_id": repo_id})
        except Exception as exc:  # noqa: BLE001 - the operator needs the reason
            # A peer too old to know this route answers 404, which reads as a
            # missing model rather than a missing feature unless we say so.
            return {
                "node_id": node_id,
                "ok": False,
                "error": f"{exc} (a peer running an older oMLX cannot be sent a download)",
            }
        task = reply.get("task", {})
        _peer_downloads[node_id] = {
            "task_id": task.get("task_id", ""),
            "repo_id": repo_id,
            "model": model_id,
        }
        return {"node_id": node_id, "ok": True, "task": task}

    results = await asyncio.gather(
        *(asyncio.to_thread(_start, node_id) for node_id in node_ids)
    )
    return {"model": model_id, "repo_id": repo_id, "peers": list(results)}


@admin_router.get("/peers/downloads")
async def peer_download_progress() -> dict[str, Any]:
    """Progress of every download this node started on a peer."""
    import asyncio

    from omlx.cluster.manager import PeerClient

    cluster = getattr(_settings(), "cluster", None)
    key = cluster.cluster_key if cluster else ""
    if not key or not _peer_downloads:
        return {"downloads": []}

    def _poll(node_id: str, record: dict[str, Any]) -> dict[str, Any]:
        peer = _peer_by_node_id(node_id)
        base = {"node_id": node_id, **record}
        if peer is None:
            return {**base, "ok": False, "error": "peer is no longer visible"}
        client = PeerClient(peer.host, peer.info.port, key)
        try:
            reply = client.get_json(
                f"/cluster/models/download/{record['task_id']}"
            )
        except Exception as exc:  # noqa: BLE001
            return {**base, "ok": False, "error": str(exc)}
        return {**base, "ok": True, "task": reply.get("task", {})}

    results = await asyncio.gather(
        *(
            asyncio.to_thread(_poll, node_id, record)
            for node_id, record in list(_peer_downloads.items())
        )
    )
    return {"downloads": list(results)}


@admin_router.post("/peers/downloads/clear")
async def clear_peer_downloads() -> dict[str, Any]:
    """Forget finished downloads so the page stops showing them."""
    _peer_downloads.clear()
    return {"ok": True}


@admin_router.post("/form")
async def form_cluster(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Form the cluster now, rather than waiting for the next request.

    Formation is otherwise a side effect of loading the sharded model, which
    means the first request after the weights land pays for it. Having it as
    an action of its own is also what lets the page finish a peer download and
    go straight to a running cluster.
    """
    from omlx.cluster.manager import form_async

    _require_real_auth()

    manager = _get_manager() if _get_manager is not None else None
    if manager is None:
        raise HTTPException(status_code=400, detail="clustering is not enabled here")

    cluster = getattr(_settings(), "cluster", None)
    model_id = str((payload or {}).get("model") or (cluster.model if cluster else ""))
    if not model_id:
        raise HTTPException(status_code=400, detail="no sharded model is configured")

    try:
        status = await form_async(manager, model_id)
    except Exception as exc:  # noqa: BLE001 - the reason is the whole point
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "status": status.to_dict()}
