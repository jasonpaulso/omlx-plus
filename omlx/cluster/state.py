# SPDX-License-Identifier: Apache-2.0
"""Typed cluster state.

Two kinds of state live here and they are deliberately kept apart:

* :class:`ClusterState` is the persisted document (``cluster.json``) —
  membership identity, credential digests, the bootstrap token record and
  the worker's own credential.
* :class:`MemberLiveness` is runtime-only. Heartbeats mutate nothing on
  disk, so a head restart starts with empty liveness and members report
  ``lost`` until their next heartbeat arrives.

Member addresses are parsed ``ipaddress`` objects, never free strings: a
peer-supplied address string eventually reaches a generated hostfile, and
a typed address cannot carry an injected line (CL-10).
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any, Literal

from .versions import VersionInfo

STATE_VERSION = 1

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
ClusterRole = Literal["off", "head", "worker"]
MemberStatus = Literal["active", "lost"]


@dataclass(frozen=True)
class Member:
    """A worker's persisted identity as admitted by the head."""

    id: str
    address: IPAddress
    port: int
    name: str
    versions: VersionInfo
    joined_at: float
    # TLS seam (CL-05): always None in v1. The control plane is plaintext
    # HTTP, so there is no peer certificate to pin yet.
    peer_cert_fingerprint: str | None = None

    @property
    def endpoint(self) -> str:
        if isinstance(self.address, ipaddress.IPv6Address):
            return f"[{self.address}]:{self.port}"
        return f"{self.address}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "address": str(self.address),
            "port": self.port,
            "name": self.name,
            "versions": self.versions.to_dict(),
            "joined_at": self.joined_at,
            "peer_cert_fingerprint": self.peer_cert_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Member:
        fingerprint = data.get("peer_cert_fingerprint")
        return cls(
            id=str(data["id"]),
            address=ipaddress.ip_address(str(data["address"])),
            port=int(data["port"]),
            name=str(data.get("name") or ""),
            versions=VersionInfo.from_dict(data.get("versions") or {}),
            joined_at=float(data.get("joined_at") or 0.0),
            peer_cert_fingerprint=str(fingerprint) if fingerprint else None,
        )


@dataclass(frozen=True)
class MemberLiveness:
    """Runtime-only liveness for one member. Never persisted."""

    epoch: str
    last_seq: int
    last_heartbeat_at: float
    status: MemberStatus = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "last_seq": self.last_seq,
            "last_heartbeat_at": self.last_heartbeat_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class MemberNodeState:
    """A worker's self-reported capacity/inventory (S4 D1).

    Rides the heartbeat as an optional field, never persisted, and lives on
    the liveness side of the identity-vs-liveness split alongside
    :class:`MemberLiveness`: it is advisory only, used exclusively for
    placement scoring, and never consulted for auth or liveness decisions.

    ``memory_ceiling`` is the worker's own ``get_final_ceiling()`` verbatim
    (0 when its memory guard is disabled) — placement's capacity-unknown
    rule treats that 0 as "unknown", so no fallback is applied here the way
    the head's own capacity gets one.
    """

    total_memory: int
    memory_ceiling: int
    models_present: dict[str, int]
    received_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memory": self.total_memory,
            "memory_ceiling": self.memory_ceiling,
            "models_present": dict(self.models_present),
            "received_at": self.received_at,
        }

    @classmethod
    def parse(cls, data: Any, *, received_at: float) -> MemberNodeState | None:
        """Lenient parse: any shape mismatch drops the field, never raises.

        Advisory data must never fail the heartbeat's liveness path (D2b),
        so every failure mode here — wrong top-level type, missing/
        non-numeric fields, a non-dict ``models_present`` — returns None
        rather than propagating.
        """
        if not isinstance(data, dict):
            return None
        try:
            total_memory = int(data["total_memory"])
            memory_ceiling = int(data["memory_ceiling"])
            raw_present = data.get("models_present") or {}
            if not isinstance(raw_present, dict):
                return None
            models_present = {str(k): int(v) for k, v in raw_present.items()}
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            total_memory=total_memory,
            memory_ceiling=memory_ceiling,
            models_present=models_present,
            received_at=received_at,
        )


@dataclass(frozen=True)
class FileManifestEntry:
    """One head-pinned file in a transfer manifest (CL-13 seam, used in S5)."""

    relative_path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileManifestEntry:
        return cls(
            relative_path=str(data["relative_path"]),
            size=int(data["size"]),
            sha256=str(data["sha256"]),
        )


@dataclass(frozen=True)
class TransferJob:
    """A resumable job record. S1 only reserves the shape."""

    id: str
    kind: str
    status: str
    created_at: float
    manifest: tuple[FileManifestEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "manifest": [entry.to_dict() for entry in self.manifest],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferJob:
        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind") or ""),
            status=str(data.get("status") or ""),
            created_at=float(data.get("created_at") or 0.0),
            manifest=tuple(
                FileManifestEntry.from_dict(entry)
                for entry in data.get("manifest") or []
            ),
        )


@dataclass(frozen=True)
class BootstrapTokenRecord:
    """Digest and expiry of the current bootstrap join token.

    The token value itself is returned once at mint time and never stored.
    """

    digest: str
    created_at: float
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BootstrapTokenRecord:
        return cls(
            digest=str(data["digest"]),
            created_at=float(data.get("created_at") or 0.0),
            expires_at=float(data.get("expires_at") or 0.0),
        )


@dataclass(frozen=True)
class WorkerIdentity:
    """The worker's own membership credential.

    Held in plaintext by necessity — the worker has to present it on every
    heartbeat — which is why ``cluster.json`` is written 0o600.
    """

    member_id: str
    secret: str
    head_url: str
    joined_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "secret": self.secret,
            "head_url": self.head_url,
            "joined_at": self.joined_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerIdentity:
        return cls(
            member_id=str(data["member_id"]),
            secret=str(data["secret"]),
            head_url=str(data["head_url"]),
            joined_at=float(data.get("joined_at") or 0.0),
        )


@dataclass(frozen=True)
class ClusterState:
    """The whole persisted cluster document."""

    members: tuple[Member, ...] = ()
    member_digests: dict[str, str] = field(default_factory=dict)
    bootstrap: BootstrapTokenRecord | None = None
    worker: WorkerIdentity | None = None
    jobs: tuple[TransferJob, ...] = ()

    def member(self, member_id: str) -> Member | None:
        for candidate in self.members:
            if candidate.id == member_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "members": [member.to_dict() for member in self.members],
            "member_digests": dict(self.member_digests),
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap else None,
            "worker": self.worker.to_dict() if self.worker else None,
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterState:
        bootstrap = data.get("bootstrap")
        worker = data.get("worker")
        return cls(
            members=tuple(
                Member.from_dict(member) for member in data.get("members") or []
            ),
            member_digests={
                str(key): str(value)
                for key, value in (data.get("member_digests") or {}).items()
            },
            bootstrap=(
                BootstrapTokenRecord.from_dict(bootstrap)
                if isinstance(bootstrap, dict)
                else None
            ),
            worker=(
                WorkerIdentity.from_dict(worker) if isinstance(worker, dict) else None
            ),
            jobs=tuple(TransferJob.from_dict(job) for job in data.get("jobs") or []),
        )


def parse_member_address(value: str, *, allow_loopback: bool = False) -> IPAddress:
    """Parse and gate a peer address derived from the request socket.

    Unspecified and multicast addresses are never valid member endpoints.
    Loopback is only valid in the single-host test mode enabled by
    ``cluster.allow_loopback`` (CL-10).
    """
    address = ipaddress.ip_address(value)
    if address.is_unspecified:
        raise ValueError(f"Unspecified address is not a valid member address: {value}")
    if address.is_multicast:
        raise ValueError(f"Multicast address is not a valid member address: {value}")
    if address.is_loopback and not allow_loopback:
        raise ValueError(
            f"Loopback address rejected: {value} "
            "(set cluster.allow_loopback for single-host testing)"
        )
    return address
