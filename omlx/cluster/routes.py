# SPDX-License-Identifier: Apache-2.0
"""``/v1/cluster/*`` control-plane API.

The router is always mounted; the role check inside it decides whether
anything is reachable. With ``cluster.role="off"`` every route answers 404
and no state or credential is exposed. (The paths still appear in
``/openapi.json`` and wrong-method requests draw 405, so a prober can
fingerprint that the feature exists — but not interact with it.)

Composition matters for both E6 and E7. One parent router carries the
enabled check, and each auth tier gets its own sub-router whose
``dependencies`` are evaluated in order: role first, then credentials. A
route cannot be declared outside a tier, so an endpoint added later cannot
be left unauthenticated, and a request to a role that is not this node's
role gets 404 before any credential is examined.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .auth import (
    require_bootstrap_token,
    require_cluster_enabled,
    require_cluster_member,
    require_cluster_operator,
    require_cluster_operator_or_member,
    require_head_role,
    require_worker_role,
)
from .manager import ClusterError, ClusterManager, get_cluster_manager
from .state import Member


class JoinRequest(BaseModel):
    """Worker → head join payload.

    The body carries only the port: the address is taken from the request
    socket so a peer cannot supply one (CL-10).
    """

    port: int = Field(ge=1, le=65535)
    versions: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


class HeartbeatRequest(BaseModel):
    seq: int = Field(ge=0)
    epoch: str
    # Worker->head rank status (D2). Absent = S1 behaviour. The head attributes
    # every update to the authenticated member and ignores any member/rank id
    # carried in the update bodies (CL2-07).
    job_updates: list[dict[str, Any]] = Field(default_factory=list)


class LocalJoinRequest(BaseModel):
    head_url: str
    token: str


class DistributedModelRequest(BaseModel):
    model: str


def _manager() -> ClusterManager:
    manager = get_cluster_manager()
    if manager is None or manager.role == "off":
        raise HTTPException(status_code=404, detail="Not Found")
    return manager


def _http_error(exc: ClusterError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


router = APIRouter(
    prefix="/v1/cluster",
    tags=["cluster"],
    dependencies=[Depends(require_cluster_enabled)],
)

# Tier routers. Role check precedes the credential check in every list.
_state_router = APIRouter(
    dependencies=[
        Depends(require_head_role),
        Depends(require_cluster_operator_or_member),
    ]
)
_operator_router = APIRouter(
    dependencies=[Depends(require_head_role), Depends(require_cluster_operator)]
)
_join_router = APIRouter(
    dependencies=[Depends(require_head_role), Depends(require_bootstrap_token)]
)
_member_router = APIRouter(
    dependencies=[Depends(require_head_role), Depends(require_cluster_member)]
)
_local_router = APIRouter(
    prefix="/local",
    dependencies=[Depends(require_worker_role), Depends(require_cluster_operator)],
)


@_state_router.get("/state")
async def get_cluster_state() -> dict[str, Any]:
    """Read cluster membership and liveness. No credential material."""
    return _manager().snapshot()


@_operator_router.post("/token")
async def mint_token() -> dict[str, Any]:
    """Mint or renew the bootstrap join token. The value is returned once."""
    try:
        return await _manager().mint_bootstrap_token()
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_operator_router.delete("/token")
async def revoke_token() -> dict[str, Any]:
    """Invalidate the current bootstrap join token."""
    try:
        return await _manager().revoke_bootstrap_token()
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_operator_router.delete("/members/{member_id}")
async def remove_member(member_id: str) -> dict[str, Any]:
    """Force-remove a member and revoke its secret."""
    try:
        return await _manager().remove_member(member_id)
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_operator_router.post("/models/load")
async def load_distributed_model(body: DistributedModelRequest) -> dict[str, Any]:
    """Stand a tensor-parallel model up across the pair (head, operator tier)."""
    try:
        return await _manager().load_distributed(body.model)
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_operator_router.post("/models/unload")
async def unload_distributed_model(body: DistributedModelRequest) -> dict[str, Any]:
    """Tear a distributed formation down (head, operator tier)."""
    try:
        return await _manager().unload_distributed(body.model)
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_operator_router.get("/models/status")
async def distributed_status() -> dict[str, Any]:
    """Read-only formation/job state (head, operator tier)."""
    try:
        return _manager().formation_status()
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_join_router.post("/join")
async def join_cluster(request: Request, body: JoinRequest) -> dict[str, Any]:
    """Admit a worker. Address comes from the socket, never from the body."""
    peer_host = request.client.host if request.client else ""
    if not peer_host:
        raise HTTPException(status_code=400, detail="Could not determine peer address")
    try:
        return await _manager().join(
            peer_host=peer_host,
            port=body.port,
            name=body.name,
            versions=body.versions,
        )
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_member_router.post("/heartbeat")
async def heartbeat(
    body: HeartbeatRequest,
    member: Annotated[Member, Depends(require_cluster_member)],
) -> dict[str, Any]:
    """Record liveness for the authenticated member."""
    try:
        return _manager().record_heartbeat(
            member, seq=body.seq, epoch=body.epoch, job_updates=body.job_updates
        )
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_member_router.post("/leave")
async def leave_cluster(
    member: Annotated[Member, Depends(require_cluster_member)],
) -> dict[str, Any]:
    """Leave the cluster: the caller's secret is revoked."""
    try:
        return await _manager().member_leave(member)
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_local_router.post("/join")
async def local_join(body: LocalJoinRequest) -> dict[str, Any]:
    """Drive this worker's join handshake against a head."""
    try:
        return await _manager().local_join(body.head_url, body.token)
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_local_router.post("/leave")
async def local_leave() -> dict[str, Any]:
    """Leave the cluster from the worker side."""
    try:
        return await _manager().local_leave()
    except ClusterError as exc:
        raise _http_error(exc) from exc


@_local_router.get("/status")
async def local_status() -> dict[str, Any]:
    """Pull-based worker status."""
    return _manager().local_status()


for _tier in (
    _state_router,
    _operator_router,
    _join_router,
    _member_router,
    _local_router,
):
    router.include_router(_tier)
