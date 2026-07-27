# SPDX-License-Identifier: Apache-2.0
"""The one seam between the oMLX daemon and cluster mode.

Everything the server needs to know about clustering is `install()` and
`shutdown()`. Keeping it to two calls means `server.py` gains two small hunks
rather than being threaded through with cluster logic, and it gives the feature
a single place to be switched off.

Nothing here does real work while `cluster.enabled` is False. In particular the
heavy imports - `mlx.distributed`, FastAPI route registration, the discovery
backend - happen inside `install()` rather than at module scope, so a disabled
cluster costs one attribute lookup at startup and adds no routes at all.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_discovery: Any = None
_manager: Any = None
_installed = False


def install(app: Any, settings: Any) -> bool:
    """Start cluster participation if it is enabled and configured.

    Returns True when the node is now advertising itself to peers. A missing
    cluster key is treated as "not configured" rather than an error: peers
    would have nothing to authenticate against, and silently forming an
    unauthenticated cluster on whatever LAN the machine is attached to is not
    a reasonable default.
    """
    global _discovery, _manager, _installed

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
        from .routes import admin_router, configure, peer_router

        _manager = ClusterManager(settings, peers)
        configure(lambda: settings, lambda: _manager)

        # Routes are added once per process. FastAPI has no remove, and a
        # settings reload that re-ran install() would otherwise stack
        # duplicate paths.
        if not _installed and app is not None:
            app.include_router(peer_router)
            app.include_router(admin_router)
            _installed = True

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
        from .routes import shutdown_follower

        shutdown_follower()
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
