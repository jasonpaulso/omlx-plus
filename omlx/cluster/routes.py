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

from fastapi import APIRouter, Depends, Header, HTTPException

from omlx.admin.auth import require_admin
from omlx.cluster import preflight, topology
from omlx.cluster.launcher import LocalCluster, resolve_python

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
        return {"ok": True, "ranks": [r.rank for r in cluster.ranks]}

    try:
        return await asyncio.to_thread(_spawn)
    except Exception as exc:  # noqa: BLE001 - the leader needs the reason
        logger.exception("cluster: could not start ranks")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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


def _stop_follower() -> int:
    global _follower
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


@admin_router.get("/status")
async def cluster_status() -> dict[str, Any]:
    """Formation state, chosen backend, rank order and any blockers."""
    from omlx.cluster import bootstrap

    manager = _get_manager() if _get_manager is not None else None
    status = (
        manager.status().to_dict()
        if manager is not None
        else {"enabled": False, "formed": False}
    )
    status["peers"] = [
        {
            "node_id": p.info.node_id,
            "host": p.host,
            "port": p.info.port,
            "chip": p.info.chip,
            "ram_gb": p.info.ram_gb,
            "version": p.info.version,
        }
        for p in bootstrap.peers()
    ]
    status["follower_ranks"] = [] if _follower is None else [
        r.rank for r in _follower.ranks
    ]
    return status


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
