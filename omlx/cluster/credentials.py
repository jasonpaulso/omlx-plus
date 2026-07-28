# SPDX-License-Identifier: Apache-2.0
"""Cluster credential machinery and the ``cluster.json`` store.

Cluster credentials live in their own file, never in ``settings.json``:
settings are written non-atomically with the process umask and hold the
inference keys, and a cluster credential must never become an inference
credential (CL-02/CL-04). ``cluster.json`` is written atomically and is
0o600 before it is visible under its final name.

The head stores only SHA-256 digests of member secrets, so a stolen head
state file yields no usable credential. The worker necessarily stores its
own secret in plaintext — it has to present it on every heartbeat.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path

from ..admin.auth import compare_keys
from .state import BootstrapTokenRecord, ClusterState

logger = logging.getLogger(__name__)

CLUSTER_STATE_FILENAME = "cluster.json"
SECRET_BYTES = 32
EPOCH_BYTES = 8


def cluster_state_path(base_path: Path) -> Path:
    """Return the path of the cluster state file for a base path."""
    return Path(base_path) / CLUSTER_STATE_FILENAME


def generate_secret() -> str:
    """Mint a 256-bit credential."""
    return secrets.token_hex(SECRET_BYTES)


def generate_epoch() -> str:
    """Mint a worker runtime epoch identifier."""
    return secrets.token_hex(EPOCH_BYTES)


def digest_secret(secret: str) -> str:
    """Return the SHA-256 digest stored in place of a credential."""
    return hashlib.sha256(secret.encode("utf-8", "surrogatepass")).hexdigest()


def verify_secret(provided: str, expected_digest: str) -> bool:
    """Constant-time check of a presented secret against a stored digest."""
    if not provided or not expected_digest:
        return False
    return compare_keys(digest_secret(provided), expected_digest)


def mint_bootstrap_token(
    ttl_s: float, *, now: float | None = None
) -> tuple[str, BootstrapTokenRecord]:
    """Mint a bootstrap join token, returning the value and its record.

    The value is the only copy; the caller returns it to the operator once
    and keeps the record.
    """
    issued_at = time.time() if now is None else now
    token = generate_secret()
    record = BootstrapTokenRecord(
        digest=digest_secret(token),
        created_at=issued_at,
        expires_at=issued_at + ttl_s,
    )
    return token, record


def bootstrap_token_matches(
    record: BootstrapTokenRecord | None, provided: str, *, now: float | None = None
) -> bool:
    """Check a presented bootstrap token against the current record and TTL."""
    if record is None or not provided:
        return False
    checked_at = time.time() if now is None else now
    if record.is_expired(checked_at):
        return False
    return verify_secret(provided, record.digest)


def load_state(path: Path) -> ClusterState:
    """Load cluster state, returning empty state when the file is absent.

    A corrupt file is not silently reset: refusing here surfaces the
    problem while the membership and credential digests are still on disk.
    """
    if not path.exists():
        return ClusterState()
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return ClusterState.from_dict(data)


def save_state(path: Path, state: ClusterState) -> None:
    """Write cluster state atomically with 0o600 permissions.

    The temp file is created 0o600 by ``mkstemp`` and lands in the same
    directory, so the rename is atomic and the credentials are never
    world-readable, not even for the width of the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".cluster-", suffix=".json"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
