# SPDX-License-Identifier: Apache-2.0
"""The per-round transfer rank script (D1/D2/R3c).

A fresh 2-rank ring session runs this module once per round: rank 0 is
always the head (the source of every peer transfer), rank 1 is always the
worker (the destination). Both ends get the round's ordered file list from
the daemon that spawned them (a plain JSON file on local disk, itself built
from the AUTHENTICATED control-plane exchange, per D1) -- nothing here reads
anything peer-supplied off the collective except raw file bytes and their
declared chunk length.

Unlike ``rank_worker.py`` this is a batch job: it joins the collective,
moves exactly the round's files, and exits. There is no persistent
command/reply pipe protocol -- the owning daemon supervises it as an
ordinary subprocess (wait for exit, kill on the round deadline) and does
its own digest verification and ``os.replace`` into the final directory
afterward (``omlx/cluster/transfer.py``); this script only ever writes into
a FRESH staging directory the daemon handed it, never the final model dir.

Digest verification, path-containment re-checks, and the final atomic move
are deliberately NOT here -- they live in the long-lived, more auditable
daemon code path (``transfer.py``), reusing the same
:mod:`omlx.cluster.manifest` validator every other manifest consumer does.
This script's only job is moving bytes safely.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Vendored from `mlx_lm.share.share_file` (CHUNK_SIZE + the receiver half),
# per CL5-15/CL5-08: the stock receiver allocates `mx.zeros(peer_chunk_size,
# ...)` from a PEER-SUPPLIED size with no bound, and opens its destination
# with a plain `open(path, "wb")` that follows an existing symlink and
# happily overwrites. This module vendors just the receiver, bounding both;
# the sender half is unmodified upstream (`mlx_lm.share.share_file`) -- it
# only ever reads a locally-resolved, already-manifest-validated file of a
# size WE control, so it carries none of CL5-15's risk.
CHUNK_SIZE = 100 * 1024 * 1024


class TransferRankError(RuntimeError):
    """A round failed inside the rank process."""


def _open_staging_target(path: Path) -> int:
    """Open ``path`` for writing a fresh file, never following a symlink and
    never overwriting anything already there (CL5-08): fresh-per-round
    staging means this exact path must not already exist. Parent
    directories are created here, by us -- never by trusting an existing
    (possibly attacker-controlled) directory tree.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)


def _check_chunk_bounds(chunk_size: int, total_so_far: int, expected_size: int) -> int:
    """Pure bound check (CL5-15), split out so it's testable without mlx: a
    peer-declared ``chunk_size`` over :data:`CHUNK_SIZE`, or a running total
    that would exceed the manifest's declared ``expected_size``, is fatal.
    Returns the new running total.
    """
    if chunk_size > CHUNK_SIZE:
        raise TransferRankError(
            f"peer-supplied chunk size {chunk_size} exceeds the {CHUNK_SIZE} bound"
        )
    total = total_so_far + chunk_size
    if total > expected_size:
        raise TransferRankError(
            f"received {total} bytes, exceeding the manifest's declared size "
            f"{expected_size}"
        )
    return total


def receive_file_bounded(
    dst_path: Path, *, expected_size: int, group: Any, mx: Any
) -> int:
    """Vendored + bounded receiver half of ``mlx_lm.share.share_file``
    (CL5-15/CL5-08). Returns the number of bytes written; raises
    :class:`TransferRankError` (and removes the partial file) on any bound
    violation.

    Caps a single chunk at ``CHUNK_SIZE`` and the running total at
    ``expected_size`` (the manifest's declared size for this entry) --
    an oversized peer-declared chunk size raises before ever allocating a
    buffer that size. This cannot, by itself, prevent every failure mode: the
    size collective must still be joined every iteration (skipping it would
    desync the ring and hang the peer instead of failing cleanly), so a peer
    that keeps sending valid-looking chunk sizes forever is caught by the
    daemon's external per-round deadline/watchdog (CL5-16), not here.
    """
    from functools import partial

    all_sum = partial(mx.distributed.all_sum, group=group)
    fd = _open_staging_target(dst_path)
    total = 0
    try:
        with os.fdopen(fd, "wb") as f:
            data = None
            chunk_size = int(all_sum(0).item())
            if chunk_size > 0:
                total = _check_chunk_bounds(chunk_size, total, expected_size)
                data = all_sum(mx.zeros(chunk_size, dtype=mx.uint8))
                mx.eval(data)

            while chunk_size > 0:
                next_data = None
                chunk_size = int(all_sum(0).item())
                if chunk_size > 0:
                    total = _check_chunk_bounds(chunk_size, total, expected_size)
                    next_data = all_sum(mx.zeros(chunk_size, dtype=mx.uint8))
                    mx.async_eval(next_data)
                # Invariant: the while condition only holds once `data` was
                # set (either just above, pre-loop, or by a prior iteration
                # when chunk_size was also > 0) -- it is never None here.
                assert data is not None
                f.write(bytes(data))
                data = next_data
    except BaseException:
        with contextlib.suppress(OSError):
            dst_path.unlink()
        raise
    return total


def _load_round_manifest(path: Path) -> list[dict[str, Any]]:
    """The round's ordered ``{relative_path, size, sha256}`` list, re-run
    through the same manifest validator as every other manifest consumer
    (this file is written locally by the daemon that spawned us, but stays
    untrusted-input discipline regardless -- CL5-06/CL5-14).
    """
    from omlx.cluster.manifest import validate_received_manifest

    raw = json.loads(Path(path).read_text())
    return [entry.to_dict() for entry in validate_received_manifest(raw)]


def _run(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from omlx.cluster.hostfile import BACKEND_VAR

    backend = os.environ.get(BACKEND_VAR) or "ring"
    group = mx.distributed.init(strict=True, backend=backend)
    rank = int(group.rank())
    size = int(group.size())
    if size != 2:
        logger.error("transfer session must be exactly 2 ranks, got %d", size)
        return 1
    expected_rank = 0 if args.role == "src" else 1
    if rank != expected_rank:
        logger.error("role %s expects rank %d, got %d", args.role, expected_rank, rank)
        return 1

    entries = _load_round_manifest(Path(args.manifest))
    root = Path(args.root)

    if args.role == "src":
        from mlx_lm.share import share_file

        for entry in entries:
            share_file(root, entry["relative_path"], 0, group)
        return 0

    for entry in entries:
        dst_path = root / entry["relative_path"]
        try:
            receive_file_bounded(
                dst_path, expected_size=entry["size"], group=group, mx=mx
            )
        except TransferRankError as exc:
            logger.error("round entry %s failed: %s", entry["relative_path"], exc)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for a spawned transfer-round rank process."""
    parser = argparse.ArgumentParser(prog="omlx-cluster-transfer-rank")
    parser.add_argument("--role", choices=["src", "dst"], required=True)
    parser.add_argument(
        "--manifest", required=True, help="path to this round's ordered manifest JSON"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="src: the resolved model dir to read from; dst: a fresh staging dir",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return _run(args)
    except Exception:  # noqa: BLE001 - reported, the daemon reads our exit code
        logger.exception("cluster: transfer rank failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
