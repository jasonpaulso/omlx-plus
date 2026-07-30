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

from typing import Annotated, Any, Literal

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
from .manager import ClusterError, ClusterManager, get_cluster_manager, get_engine_pool
from .placement import plan_placement, resolve_placement_inputs, worker_node_capacity
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
    # S4 D1: advisory capacity/inventory. Typed as `Any` (not `dict | None`)
    # so any shape a worker sends is accepted by the model — the leniency
    # binding rule (D2b) is enforced by `MemberNodeState.parse`, not by
    # rejecting the request here; a malformed value must never fail the
    # heartbeat's liveness path.
    node_state: Any = None
    # S5 D1b: a SIBLING channel to job_updates, not an ack -- see
    # `ClusterManager.record_heartbeat`. Bounds are enforced there
    # (CL5-04), not by this model, so an oversized batch is a clean drop
    # rather than a validation error.
    transfer_updates: list[dict[str, Any]] = Field(default_factory=list)


class LocalJoinRequest(BaseModel):
    head_url: str
    token: str


class DistributedModelRequest(BaseModel):
    model: str
    # S4 D3: /v1/cluster/models/load becomes prefer=distributed placement +
    # formation; exposing the knob lets a caller force `local`/`auto` too
    # rather than only ever attempting distributed.
    prefer: Literal["auto", "local", "distributed"] = "distributed"


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
        return await _manager().load_distributed(body.model, prefer=body.prefer)
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


def compute_placement_preview(
    manager: ClusterManager,
    model: str,
    prefer: Literal["auto", "local", "distributed"],
) -> dict[str, Any]:
    """Dry-run placement preview (S4 D3): zero side effects, no formation.

    Shared by ``GET /v1/cluster/placement`` and the admin dashboard's
    ``GET /admin/api/cluster/placement`` proxy (D6) so the two never drift.
    """
    pool = get_engine_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Engine pool is not available")
    entry = pool.get_entry(model)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    est_size, model_config = resolve_placement_inputs(entry.model_path)
    head = pool.head_capacity()
    workers = []
    for candidate in manager.state.members:
        live = manager.liveness(candidate.id)
        node_state = manager.node_state(candidate.id)
        if live is not None and live.status == "active" and node_state is not None:
            workers.append(worker_node_capacity(candidate.id, node_state))
    decision = plan_placement(
        model_id=model,
        model_type=entry.model_type,
        est_size=est_size,
        model_config=model_config,
        head=head,
        workers=workers,
        prefer=prefer,
    )
    return decision.to_dict()


@_operator_router.get("/placement")
async def preview_placement(
    model: str, prefer: Literal["auto", "local", "distributed"] = "auto"
) -> dict[str, Any]:
    """Dry-run placement preview (S4 D3): zero side effects, no formation."""
    return compute_placement_preview(_manager(), model, prefer)


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
            member,
            seq=body.seq,
            epoch=body.epoch,
            job_updates=body.job_updates,
            node_state=body.node_state,
            transfer_updates=body.transfer_updates,
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
