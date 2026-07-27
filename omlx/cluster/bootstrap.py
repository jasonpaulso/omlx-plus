# SPDX-License-Identifier: Apache-2.0
"""The one seam between the oMLX daemon and cluster mode.

Everything the server needs to know about clustering is `install()` and
`shutdown()`. Keeping it to two calls means `server.py` gains two small hunks
rather than being threaded through with cluster logic, and it gives the feature
a single place to be switched off.

Nothing here does real work while `cluster.enabled` is False. `mlx.distributed`
and the discovery backend are imported inside `install()` rather than at module
scope, so a disabled cluster costs one attribute lookup at startup.

The two routers are installed on different conditions, and the difference is
the whole security boundary:

- `admin_router` is behind the daemon's own admin auth and is registered
  unconditionally. It is *how* clustering gets turned on, so gating it on
  clustering being on would make the feature unreachable from the UI - which
  is exactly the state this file used to be in.
- `peer_router` authenticates with the shared cluster key and spawns rank
  processes on this machine. It appears only once the operator has opted in
  with a key.

Neither router can be removed once added: FastAPI has no route removal. That is
survivable because `verify_cluster_key` reads live settings on every call, so
disabling clustering leaves the peer paths present but fail-closed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_discovery: Any = None
_manager: Any = None
# The live GlobalSettings. Held so `configure()` can hand the routes a getter
# rather than a captured object - a hot-apply that ever swapped the settings
# object would otherwise leave `verify_cluster_key` comparing against a key the
# operator has already rotated away.
_settings: Any = None
# Tracked separately. One flag for both would mean the admin router's
# registration at startup suppressed the peer router's registration on a later
# live enable, and the leader's `/cluster/report` would 404 instead of forming.
_admin_installed = False
_peer_installed = False


def install(app: Any, settings: Any) -> bool:
    """Start cluster participation if it is enabled and configured.

    Returns True when the node is now advertising itself to peers. A missing
    cluster key is treated as "not configured" rather than an error: peers
    would have nothing to authenticate against, and silently forming an
    unauthenticated cluster on whatever LAN the machine is attached to is not
    a reasonable default.

    Safe to call again on a running daemon - that is how a configuration change
    is applied without a restart - provided `shutdown()` ran first.
    """
    global _discovery, _manager, _settings, _admin_installed, _peer_installed

    _settings = settings

    # Unconditional: the operator surface has to exist on a node that has never
    # heard of clustering, because it is the thing that turns clustering on.
    # Every handler on it already tolerates a None manager and no discovery.
    from .routes import admin_router, configure

    configure(lambda: _settings, lambda: _manager)
    if app is not None and not _admin_installed:
        app.include_router(admin_router)
        _admin_installed = True

    cluster = getattr(settings, "cluster", None)
    if cluster is None or not cluster.enabled:
        return False

    if not cluster.cluster_key:
        logger.warning(
            "cluster: enabled but no cluster_key is set; not advertising. "
            "Set cluster.cluster_key to the same value on every node."
        )
        return False

    try:
        from omlx import __version__

        from .discovery import ClusterDiscovery, default_node_id
        from .manager import ClusterManager
        from .routes import peer_router

        _manager = ClusterManager(settings, peers)

        # Added once per process, but on the first *enable* rather than at
        # startup - which may be now, on a live config write, long after the
        # app was built.
        if app is not None and not _peer_installed:
            app.include_router(peer_router)
            _peer_installed = True

        _discovery = ClusterDiscovery(
            node_id=default_node_id(),
            port=settings.server.port,
            version=str(__version__),
            cluster_key=cluster.cluster_key,
            poll_interval=cluster.discovery_interval_seconds,
        )
        _discovery.start()
    except Exception:  # noqa: BLE001 - clustering must never block startup
        logger.exception("cluster: failed to start discovery; continuing single-node")
        _discovery = None
        return False

    logger.info("cluster: discovery started, advertising to peers")
    return True


def shutdown() -> None:
    """Stop advertising, and kill every rank this node owns.

    Ranks outlive their daemon otherwise, and a worker left holding the ring
    port makes the *next* cluster formation fail with a connect error that
    reads like a firewall fault.
    """
    global _discovery, _manager

    if _manager is not None:
        try:
            _manager.teardown()
        except Exception:  # noqa: BLE001 - never block shutdown
            logger.exception("cluster: leader teardown was not clean")
        _manager = None

    try:
        from .routes import reset_resolved, shutdown_follower

        shutdown_follower()
        # Model paths resolved at report time are only valid for the model
        # directories that were configured when they were resolved. Carrying
        # them across a settings change would spawn a worker on a stale path.
        reset_resolved()
    except Exception:  # noqa: BLE001
        logger.exception("cluster: follower teardown was not clean")

    if _discovery is None:
        return
    try:
        _discovery.stop()
    except Exception:  # noqa: BLE001 - never block shutdown
        logger.exception("cluster: discovery did not stop cleanly")
    finally:
        _discovery = None


async def reapply(app: Any, settings: Any, *, previous_model: str = "") -> bool:
    """Apply a changed cluster configuration without restarting the daemon.

    A field write alone changes nothing that matters: `ClusterDiscovery` reads
    the port and poll interval at construction, and a formed cluster holds a
    `ClusterManager` built from the old settings. So the whole cycle runs -
    including dropping any pooled engine that still points at the manager
    `shutdown()` is about to discard.

    That eviction is not housekeeping. `ClusterEngine` holds its manager by
    reference, and while clustering is disabled the pool will not rebuild the
    engine either, so a stale one would keep answering requests through a torn
    down cluster until something else evicted it.
    """
    await _evict_cluster_engine(previous_model)
    shutdown()
    return install(app, settings)


async def _evict_cluster_engine(model_id: str) -> None:
    """Drop the pooled engine for `model_id`, if one is loaded."""
    if not model_id:
        return
    try:
        from omlx.server import _server_state

        pool = _server_state.engine_pool
        if pool is None:
            return
        await pool._unload_engine(model_id)
    except Exception:  # noqa: BLE001 - a config write must not fail on this
        logger.exception("cluster: could not unload the cluster engine for %s", model_id)


def peers() -> list[Any]:
    """Currently visible peers, or empty when clustering is not running."""
    return list(_discovery.peers) if _discovery is not None else []


def manager() -> Any:
    """The leader-side `ClusterManager`, or None when clustering is off."""
    return _manager


def serves_cluster_model(model_id: str) -> bool:
    """True when `model_id` is the model this fleet shards across its nodes.

    The engine pool asks this before applying its local memory ceiling. A
    cluster model's weights are never in this process, so admitting it against
    the daemon's own footprint would refuse every model clustering exists to
    serve.
    """
    if _manager is None:
        return False
    cluster = getattr(_manager.settings, "cluster", None)
    return bool(
        cluster and cluster.enabled and cluster.model and cluster.model == model_id
    )


def build_engine(
    *,
    model_id: str,
    model_path: str,
    trust_remote_code: bool,
    model_settings: Any,
) -> Any:
    """A `ClusterEngine` for the cluster model, or None for everything else.

    Returning None is the common case and is how every other model in the pool
    keeps loading locally while one of them is served by the whole fleet.
    """
    if not serves_cluster_model(model_id):
        return None

    from .engine import ClusterEngine

    logger.info("cluster: %s will be served across the cluster", model_id)
    return ClusterEngine(
        model_name=model_path,
        model_id=model_id,
        manager=_manager,
        trust_remote_code=trust_remote_code,
        model_settings=model_settings,
    )
