# SPDX-License-Identifier: Apache-2.0
"""The one seam between the oMLX daemon and cluster mode.

Everything the server needs to know about clustering is `install()` and
`shutdown()`. Keeping it to two calls means `server.py` gains two small hunks
rather than being threaded through with cluster logic, and it gives the feature
a single place to be switched off.

Nothing here does real work while `cluster.enabled` is False. In particular the
heavy imports - `mlx.distributed`, the discovery backend - happen inside
`install()` rather than at module scope, so a disabled cluster costs one
attribute lookup at startup.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_discovery: Any = None


def install(settings: Any) -> bool:
    """Start cluster participation if it is enabled and configured.

    Returns True when the node is now advertising itself to peers. A missing
    cluster key is treated as "not configured" rather than an error: peers
    would have nothing to authenticate against, and silently forming an
    unauthenticated cluster on whatever LAN the machine is attached to is not
    a reasonable default.
    """
    global _discovery

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
    """Stop advertising and browsing. Safe to call when never installed."""
    global _discovery

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
