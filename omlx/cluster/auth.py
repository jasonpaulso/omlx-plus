# SPDX-License-Identifier: Apache-2.0
"""Cluster authentication dependencies (E7).

These dependencies reuse oMLX's auth *primitives* (``compare_keys``,
``fingerprint_key``) but none of its credential classes and none of its
escape hatches: ``auth.skip_api_key_verification`` and the
"no API key configured means allow" rule that ``verify_api_key`` applies
(``omlx/server.py:305-313``) are both ignored here. Every dependency fails
closed.

The privilege boundary runs one way. An admin session or the main API key
may drive cluster endpoints; a member secret never satisfies the operator
tier, and no cluster credential is accepted anywhere outside this router.
Sub-keys are rejected outright (CL-02).

Each dependency is mounted router-level on a per-tier sub-router, so a new
endpoint cannot exist outside a tier and therefore cannot be added without
auth.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from ..admin.auth import compare_keys, fingerprint_key, verify_session
from .credentials import bootstrap_token_matches, verify_secret
from .manager import ClusterManager, get_cluster_manager
from .state import Member

logger = logging.getLogger(__name__)


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return ""
    return header[7:]


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"}
    )


def _not_found() -> HTTPException:
    # A disabled role answers exactly like a server without the feature.
    return HTTPException(status_code=404, detail="Not Found")


def _active_manager() -> ClusterManager:
    manager = get_cluster_manager()
    if manager is None or manager.role == "off":
        raise _not_found()
    return manager


def _match_member(manager: ClusterManager, token: str) -> Member | None:
    if not token:
        return None
    for member in manager.state.members:
        digest = manager.state.member_digests.get(member.id)
        if digest and verify_secret(token, digest):
            return member
    return None


async def require_cluster_enabled() -> ClusterManager:
    """404 every cluster route unless a cluster role is active."""
    return _active_manager()


async def require_head_role() -> ClusterManager:
    """404 head-facing routes on a node that is not the head."""
    manager = _active_manager()
    if manager.role != "head":
        raise _not_found()
    return manager


async def require_worker_role() -> ClusterManager:
    """404 worker-local routes on a node that is not a worker."""
    manager = _active_manager()
    if manager.role != "worker":
        raise _not_found()
    return manager


async def require_cluster_member(request: Request) -> Member:
    """Authenticate a per-member secret against the head's stored digests."""
    manager = _active_manager()
    token = _bearer_token(request)
    if not token:
        raise _unauthorized("Cluster member credential required")
    member = _match_member(manager, token)
    if member is None:
        logger.warning(
            "Rejected cluster member credential (fp=%s)", fingerprint_key(token)
        )
        raise _unauthorized("Invalid cluster member credential")
    request.state.cluster_member = member
    return member


async def require_cluster_operator(request: Request) -> bool:
    """Require an admin session or the main API key.

    Sub-keys are refused (a sub-key is an inference credential), and no
    bypass flag is honored. The no-API-key-configured state is unreachable
    here: a cluster role refuses to start the server without one.
    """
    manager = _active_manager()
    if verify_session(request):
        return True
    token = _bearer_token(request)
    configured = manager.global_settings.auth.api_key
    if token and configured and compare_keys(token, configured):
        return True
    if token:
        logger.warning(
            "Rejected cluster operator credential (fp=%s)", fingerprint_key(token)
        )
    raise _unauthorized("Cluster operator credential required")


async def require_bootstrap_token(request: Request) -> bool:
    """Authenticate the TTL-bounded bootstrap join token."""
    manager = _active_manager()
    token = _bearer_token(request)
    if not token:
        raise _unauthorized("Bootstrap join token required")
    if not bootstrap_token_matches(manager.state.bootstrap, token):
        logger.warning("Rejected bootstrap join token (fp=%s)", fingerprint_key(token))
        raise _unauthorized("Invalid or expired bootstrap join token")
    return True


async def require_cluster_operator_or_member(request: Request) -> bool:
    """Read tier: either an operator or an admitted member may read state.

    The disjunction only widens the read tier — a member secret still
    never satisfies :func:`require_cluster_operator`.
    """
    manager = _active_manager()
    member = _match_member(manager, _bearer_token(request))
    if member is not None:
        request.state.cluster_member = member
        return True
    return await require_cluster_operator(request)
