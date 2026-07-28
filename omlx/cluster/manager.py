# SPDX-License-Identifier: Apache-2.0
"""The cluster runtime: membership, credentials, liveness and jobs.

The head is the single serialized owner of every cluster mutation (E6):
join, leave, remove, token operations and scrub expiry all run as commands
on one queue, and each command persists ``cluster.json`` after applying,
so two concurrent formation sequences are impossible by construction.

Liveness is deliberately not part of that: heartbeats mutate an in-memory
map only. A head restart therefore starts with no liveness at all and
reports persisted members as ``lost`` until their next heartbeat arrives,
which self-heals within one interval whether or not the worker restarted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import socket
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..settings import ClusterSettings, GlobalSettings
from .client import ClusterClient, ClusterClientError, validate_head_url
from .credentials import (
    cluster_state_path,
    digest_secret,
    generate_secret,
    load_state,
    mint_bootstrap_token,
    save_state,
)
from .heartbeat import HeartbeatSender
from .queue import ClusterCommandQueue
from .state import (
    ClusterState,
    Member,
    MemberLiveness,
    WorkerIdentity,
    parse_member_address,
)
from .versions import VersionInfo, collect_versions, compare_versions

logger = logging.getLogger(__name__)

JOIN_PATH = "/v1/cluster/join"
LEAVE_PATH = "/v1/cluster/leave"
MEMBER_ID_BYTES = 8
MAX_MEMBER_NAME_LENGTH = 64

_cluster_manager: ClusterManager | None = None


def get_cluster_manager() -> ClusterManager | None:
    """Return the active cluster manager, or None when the role is off."""
    return _cluster_manager


def set_cluster_manager(manager: ClusterManager | None) -> None:
    """Install (or clear) the process-wide cluster manager."""
    global _cluster_manager
    _cluster_manager = manager


class ClusterError(Exception):
    """A cluster operation failed with a specific HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ClusterManager:
    """Owns cluster state for one node, in whichever role it was started."""

    def __init__(
        self,
        global_settings: GlobalSettings,
        *,
        state_path: Path | None = None,
        client_factory: Callable[[str], ClusterClient] | None = None,
    ) -> None:
        self.global_settings = global_settings
        self.state_path = state_path or cluster_state_path(global_settings.base_path)
        self._client_factory = client_factory or (lambda url: ClusterClient(url))
        self._state = ClusterState()
        self._liveness: dict[str, MemberLiveness] = {}
        self._queue = ClusterCommandQueue()
        self._scrub_task: asyncio.Task[None] | None = None
        self._heartbeat: HeartbeatSender | None = None
        self._versions: VersionInfo | None = None
        self._started = False

    # ---- basic accessors -------------------------------------------------

    @property
    def role(self) -> str:
        return self.global_settings.cluster.role

    @property
    def state(self) -> ClusterState:
        return self._state

    @property
    def settings(self) -> ClusterSettings:
        return self.global_settings.cluster

    @property
    def versions(self) -> VersionInfo:
        if self._versions is None:
            self._versions = collect_versions()
        return self._versions

    def liveness(self, member_id: str) -> MemberLiveness | None:
        return self._liveness.get(member_id)

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Load state and start the role's background tasks.

        Refuses to start any cluster role without a configured API key.
        Without one the operator tier is either wide open (the
        unauthenticated ``POST /admin/api/setup-api-key`` mints an admin
        session exactly when no key is set) or locked out (admin login
        refuses without a key). There is no opt-out: configuring a key is
        one line of operator config.
        """
        if self.role == "off":
            return
        if not self.global_settings.auth.api_key:
            raise RuntimeError(
                f"cluster.role={self.role!r} requires auth.api_key to be configured. "
                "Cluster control-plane endpoints are operator-gated and refuse to "
                "run on a server with no API key. Set one with "
                "`omlx serve --api-key ...`, the OMLX_API_KEY environment "
                "variable, or the admin settings page, then restart."
            )
        self._state = load_state(self.state_path)
        await self._queue.start()
        if self.role == "head":
            self._scrub_task = asyncio.create_task(self._scrub_loop())
        elif self.role == "worker" and self._state.worker is not None:
            await self._start_heartbeat(self._state.worker)
        self._started = True
        logger.info("Cluster manager started (role=%s)", self.role)

    async def stop(self) -> None:
        if self._scrub_task is not None:
            self._scrub_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scrub_task
            self._scrub_task = None
        if self._heartbeat is not None:
            await self._heartbeat.stop()
            self._heartbeat = None
        await self._queue.stop()
        self._started = False

    # ---- head commands ---------------------------------------------------

    async def mint_bootstrap_token(self) -> dict[str, Any]:
        """Mint or renew the bootstrap join token. Value returned once."""

        async def _apply() -> dict[str, Any]:
            token, record = mint_bootstrap_token(
                float(self.settings.bootstrap_token_ttl_s)
            )
            self._persist(replace(self._state, bootstrap=record))
            logger.info(
                "Bootstrap join token minted (expires_at=%s)", record.expires_at
            )
            return {
                "token": token,
                "expires_at": record.expires_at,
                "ttl_s": float(self.settings.bootstrap_token_ttl_s),
            }

        return await self._queue.submit("mint_bootstrap_token", _apply)

    async def revoke_bootstrap_token(self) -> dict[str, Any]:
        """Invalidate the current bootstrap join token."""

        async def _apply() -> dict[str, Any]:
            existed = self._state.bootstrap is not None
            self._persist(replace(self._state, bootstrap=None))
            logger.info("Bootstrap join token revoked (existed=%s)", existed)
            return {"revoked": existed}

        return await self._queue.submit("revoke_bootstrap_token", _apply)

    async def join(
        self,
        *,
        peer_host: str,
        port: int,
        name: str | None,
        versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Admit a worker: E10 version check, socket-derived address, mint secret."""

        async def _apply() -> dict[str, Any]:
            remote = VersionInfo.from_dict(versions)
            mismatch = compare_versions(self.versions, remote)
            if mismatch is not None:
                raise ClusterError(409, mismatch)
            if not 1 <= port <= 65535:
                raise ClusterError(400, f"Invalid member port: {port}")
            try:
                address = parse_member_address(
                    peer_host, allow_loopback=bool(self.settings.allow_loopback)
                )
            except ValueError as exc:
                raise ClusterError(400, str(exc)) from exc
            member = Member(
                id=secrets.token_hex(MEMBER_ID_BYTES),
                address=address,
                port=port,
                name=_sanitize_name(name),
                versions=remote,
                joined_at=time.time(),
                peer_cert_fingerprint=None,
            )
            secret = generate_secret()
            digests = dict(self._state.member_digests)
            digests[member.id] = digest_secret(secret)
            self._persist(
                replace(
                    self._state,
                    members=self._state.members + (member,),
                    member_digests=digests,
                )
            )
            logger.info(
                "Cluster member joined: id=%s endpoint=%s name=%s",
                member.id,
                member.endpoint,
                member.name,
            )
            return {
                "member_id": member.id,
                "member_secret": secret,
                "heartbeat_interval_s": float(self.settings.heartbeat_interval_s),
                "member_timeout_s": float(self.settings.member_timeout_s),
            }

        return await self._queue.submit("join", _apply)

    async def remove_member(self, member_id: str) -> dict[str, Any]:
        """Remove a member and revoke its secret."""

        async def _apply() -> dict[str, Any]:
            member = self._state.member(member_id)
            if member is None:
                raise ClusterError(404, f"Unknown cluster member: {member_id}")
            digests = dict(self._state.member_digests)
            digests.pop(member_id, None)
            self._persist(
                replace(
                    self._state,
                    members=tuple(m for m in self._state.members if m.id != member_id),
                    member_digests=digests,
                )
            )
            self._liveness.pop(member_id, None)
            logger.info("Cluster member removed: id=%s", member_id)
            return {"member_id": member_id, "removed": True}

        return await self._queue.submit("remove_member", _apply)

    def record_heartbeat(
        self, member: Member, *, seq: int, epoch: str
    ) -> dict[str, Any]:
        """Record liveness for a heartbeat. Touches no persisted state.

        A new epoch is accepted and resets the sequence — the request is
        already authenticated by the member secret, so an epoch change is
        a restart, not an unauthenticated replay. Inside a live epoch the
        sequence must strictly increase.
        """
        if not epoch:
            raise ClusterError(400, "Heartbeat epoch must not be empty")
        now = time.time()
        current = self._liveness.get(member.id)
        if current is not None and current.epoch == epoch and seq <= current.last_seq:
            raise ClusterError(
                409,
                f"Heartbeat sequence {seq} is not greater than {current.last_seq} "
                f"for epoch {epoch}",
            )
        revived = current is not None and current.status == "lost"
        self._liveness[member.id] = MemberLiveness(
            epoch=epoch, last_seq=seq, last_heartbeat_at=now, status="active"
        )
        if revived:
            logger.info("Cluster member revived: id=%s", member.id)
        return {
            "member_id": member.id,
            "status": "active",
            "heartbeat_interval_s": float(self.settings.heartbeat_interval_s),
        }

    async def member_leave(self, member: Member) -> dict[str, Any]:
        """Handle a member's own leave: revoke its secret."""
        return await self.remove_member(member.id)

    async def scrub(self) -> list[str]:
        """Mark members past the timeout as lost. Never revokes a credential.

        A timeout is a liveness statement, not a trust decision: a resumed
        heartbeat revives the member. Revocation belongs to explicit leave
        and operator removal.
        """

        async def _apply() -> list[str]:
            now = time.time()
            timeout = float(self.settings.member_timeout_s)
            expired: list[str] = []
            for member in self._state.members:
                live = self._liveness.get(member.id)
                if live is None or live.status != "active":
                    continue
                if now - live.last_heartbeat_at > timeout:
                    self._liveness[member.id] = replace(live, status="lost")
                    expired.append(member.id)
                    logger.warning(
                        "Cluster member timed out: id=%s endpoint=%s",
                        member.id,
                        member.endpoint,
                    )
            return expired

        return await self._queue.submit("scrub", _apply)

    # ---- worker commands -------------------------------------------------

    async def local_join(self, head_url: str, token: str) -> dict[str, Any]:
        """Perform the join handshake against the head and persist the credential."""

        async def _apply() -> dict[str, Any]:
            if not token:
                raise ClusterError(400, "A bootstrap join token is required")
            try:
                base_url = validate_head_url(head_url)
            except ValueError as exc:
                raise ClusterError(400, str(exc)) from exc
            payload = {
                "port": int(self.global_settings.server.port),
                "name": _sanitize_name(self.settings.node_name or socket.gethostname()),
                "versions": self.versions.to_dict(),
            }
            client = self._client_factory(base_url)
            try:
                reply = await client.post_json(JOIN_PATH, token=token, payload=payload)
            except ClusterClientError as exc:
                raise ClusterError(exc.status_code or 502, str(exc)) from exc
            member_id = str(reply.get("member_id") or "")
            secret = str(reply.get("member_secret") or "")
            if not member_id or not secret:
                raise ClusterError(502, "Head returned an incomplete join response")
            identity = WorkerIdentity(
                member_id=member_id,
                secret=secret,
                head_url=base_url,
                joined_at=time.time(),
            )
            self._persist(replace(self._state, worker=identity))
            interval = float(
                reply.get("heartbeat_interval_s") or self.settings.heartbeat_interval_s
            )
            await self._start_heartbeat(identity, interval_s=interval)
            logger.info("Joined cluster head %s as member %s", base_url, member_id)
            return {
                "member_id": member_id,
                "head_url": base_url,
                "heartbeat_interval_s": interval,
            }

        return await self._queue.submit("local_join", _apply)

    async def local_leave(self) -> dict[str, Any]:
        """Leave the cluster and drop the local credential."""

        async def _apply() -> dict[str, Any]:
            identity = self._state.worker
            if identity is None:
                raise ClusterError(400, "This node is not a member of a cluster")
            if self._heartbeat is not None:
                await self._heartbeat.stop()
                self._heartbeat = None
            notified = True
            client = self._client_factory(identity.head_url)
            try:
                await client.post_json(LEAVE_PATH, token=identity.secret, payload={})
            except ClusterClientError as exc:
                # The local credential is dropped either way: a worker that
                # cannot reach the head must still be able to leave.
                notified = False
                logger.warning("Leave notification to head failed: %s", exc)
            self._persist(replace(self._state, worker=None))
            logger.info("Left cluster head %s", identity.head_url)
            return {
                "member_id": identity.member_id,
                "head_url": identity.head_url,
                "head_notified": notified,
            }

        return await self._queue.submit("local_leave", _apply)

    def local_status(self) -> dict[str, Any]:
        """Pull-based worker status (CL-14: the head never calls back)."""
        identity = self._state.worker
        return {
            "enabled": True,
            "role": self.role,
            "joined": identity is not None,
            "member_id": identity.member_id if identity else None,
            "head_url": identity.head_url if identity else None,
            "joined_at": identity.joined_at if identity else None,
            "heartbeat": self._heartbeat.status() if self._heartbeat else None,
            "versions": self.versions.to_dict(),
        }

    # ---- observability ---------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """State for the API and dashboard. Never contains credential material."""
        members = []
        for member in self._state.members:
            live = self._liveness.get(member.id)
            entry = member.to_dict()
            entry["status"] = live.status if live else "lost"
            entry["last_heartbeat_at"] = live.last_heartbeat_at if live else None
            entry["last_seq"] = live.last_seq if live else None
            entry["epoch"] = live.epoch if live else None
            members.append(entry)
        bootstrap = self._state.bootstrap
        snapshot: dict[str, Any] = {
            "enabled": True,
            "role": self.role,
            "members": members,
            "member_count": len(members),
            "active_count": sum(1 for m in members if m["status"] == "active"),
            "heartbeat_interval_s": float(self.settings.heartbeat_interval_s),
            "member_timeout_s": float(self.settings.member_timeout_s),
            "bootstrap_token": {
                "configured": bootstrap is not None,
                "expires_at": bootstrap.expires_at if bootstrap else None,
            },
            "jobs": [job.to_dict() for job in self._state.jobs],
            "versions": self.versions.to_dict(),
        }
        if self.role == "worker":
            snapshot["local"] = self.local_status()
        return snapshot

    # ---- internals -------------------------------------------------------

    def _persist(self, state: ClusterState) -> None:
        save_state(self.state_path, state)
        self._state = state

    async def _start_heartbeat(
        self, identity: WorkerIdentity, *, interval_s: float | None = None
    ) -> None:
        if self._heartbeat is not None:
            await self._heartbeat.stop()
        self._heartbeat = HeartbeatSender(
            identity,
            interval_s=(
                interval_s
                if interval_s is not None
                else float(self.settings.heartbeat_interval_s)
            ),
            client_factory=self._client_factory,
        )
        await self._heartbeat.start()

    async def _scrub_loop(self) -> None:
        interval = max(0.1, float(self.settings.heartbeat_interval_s))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.scrub()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.warning("Cluster scrub failed: %s", exc)


def _sanitize_name(name: str | None) -> str:
    """Keep member names printable and bounded — they reach logs and the UI."""
    if not name:
        return ""
    cleaned = "".join(ch for ch in name if ch.isprintable())
    return cleaned.strip()[:MAX_MEMBER_NAME_LENGTH]
