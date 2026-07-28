# SPDX-License-Identifier: Apache-2.0
"""Version identity exchanged during the cluster join handshake (E10).

Collectives and tensor-parallel shard layouts are only valid between
identical stacks, so a skewed worker has to fail at join rather than
inside a collective. mlx-lm is pinned to a git commit, so its version
string alone cannot detect commit skew — the commit id is read from the
PEP 610 ``direct_url.json`` metadata the installer records for VCS
installs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackageVersion:
    """A distribution version plus its VCS commit when installed from git."""

    version: str
    commit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "commit_id": self.commit_id}

    @classmethod
    def from_dict(cls, data: Any) -> PackageVersion:
        if not isinstance(data, dict):
            return cls(version=str(data or ""))
        commit_id = data.get("commit_id")
        return cls(
            version=str(data.get("version") or ""),
            commit_id=str(commit_id) if commit_id else None,
        )

    def describe(self) -> str:
        if self.commit_id:
            return f"{self.version}@{self.commit_id}"
        return f"{self.version}@unknown"


@dataclass(frozen=True)
class VersionInfo:
    """The stack identity a node advertises when joining a cluster."""

    omlx: str
    mlx: str
    mlx_lm: PackageVersion

    def to_dict(self) -> dict[str, Any]:
        return {
            "omlx": self.omlx,
            "mlx": self.mlx,
            "mlx_lm": self.mlx_lm.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionInfo:
        return cls(
            omlx=str(data.get("omlx") or ""),
            mlx=str(data.get("mlx") or ""),
            mlx_lm=PackageVersion.from_dict(data.get("mlx_lm")),
        )

    def describe(self) -> str:
        return f"omlx={self.omlx} mlx={self.mlx} mlx-lm={self.mlx_lm.describe()}"


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        logger.warning("Distribution %s is not installed; version unknown", name)
        return ""


def vcs_commit_id(name: str) -> str | None:
    """Return the git commit a distribution was installed from, if any.

    Reads the PEP 610 ``direct_url.json`` the installer writes for direct
    URL installs. Returns None for index installs, which legitimately have
    no commit id.
    """
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    try:
        raw = dist.read_text("direct_url.json")
    except OSError:
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs_info = parsed.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    return str(commit_id) if commit_id else None


def collect_versions() -> VersionInfo:
    """Collect this node's stack identity."""
    from .._version import __version__

    return VersionInfo(
        omlx=__version__,
        mlx=_distribution_version("mlx"),
        mlx_lm=PackageVersion(
            version=_distribution_version("mlx-lm"),
            commit_id=vcs_commit_id("mlx-lm"),
        ),
    )


def compare_versions(local: VersionInfo, remote: VersionInfo) -> str | None:
    """Return a rejection reason if the two stacks differ, else None.

    ``omlx`` and ``mlx`` versions must match exactly. ``mlx-lm`` must match
    on version, and on commit id when both sides report one. Neither side
    reporting a commit id (index installs) falls back to a version-only
    comparison; exactly one side reporting one is a provenance mismatch and
    is rejected rather than silently downgraded.
    """
    if local.omlx != remote.omlx or local.mlx != remote.mlx:
        return _mismatch(local, remote)
    if local.mlx_lm.version != remote.mlx_lm.version:
        return _mismatch(local, remote)
    local_commit = local.mlx_lm.commit_id
    remote_commit = remote.mlx_lm.commit_id
    if (local_commit is None) != (remote_commit is None):
        return _mismatch(local, remote)
    if local_commit is not None and local_commit != remote_commit:
        return _mismatch(local, remote)
    return None


def _mismatch(local: VersionInfo, remote: VersionInfo) -> str:
    return (
        "Version skew rejected: collectives require identical stacks. "
        f"head has [{local.describe()}], joining node has [{remote.describe()}]"
    )
