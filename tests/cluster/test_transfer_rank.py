# SPDX-License-Identifier: Apache-2.0
"""The transfer rank script's mlx-free parts (CL5-08/CL5-15).

Mirrors rank_worker.py's precedent: the mlx.distributed-touching collective
loop is not unit tested here (that needs real mlx, exercised on the live
rig, P3); the vendored bound-check and the staging-target open discipline
are pure Python and tested directly.
"""

from __future__ import annotations

import os

import pytest

from omlx.cluster import transfer_rank as tr

# -- CL5-08: fresh-staging open discipline ------------------------------------


def test_open_staging_target_refuses_existing_path(tmp_path):
    path = tmp_path / "a" / "b.safetensors"
    fd = tr._open_staging_target(path)
    os.close(fd)
    with pytest.raises(FileExistsError):
        tr._open_staging_target(path)


def test_open_staging_target_refuses_symlink(tmp_path):
    real = tmp_path / "real.safetensors"
    real.write_bytes(b"x")
    link = tmp_path / "evil.safetensors"
    os.symlink(real, link)
    with pytest.raises(OSError):
        tr._open_staging_target(link)


def test_open_staging_target_creates_parents_itself(tmp_path):
    path = tmp_path / "nested" / "deep" / "file.json"
    fd = tr._open_staging_target(path)
    os.close(fd)
    assert path.parent.is_dir()


def test_open_staging_target_writes_0600(tmp_path):
    import stat

    path = tmp_path / "file.json"
    fd = tr._open_staging_target(path)
    os.close(fd)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# -- CL5-15: chunk/cumulative bounds ------------------------------------------


def test_check_chunk_bounds_accepts_within_bounds():
    assert tr._check_chunk_bounds(10, 0, 100) == 10
    assert tr._check_chunk_bounds(10, 10, 100) == 20


def test_check_chunk_bounds_rejects_oversized_chunk():
    with pytest.raises(tr.TransferRankError, match="bound"):
        tr._check_chunk_bounds(tr.CHUNK_SIZE + 1, 0, 10**12)


def test_check_chunk_bounds_rejects_cumulative_overflow():
    with pytest.raises(tr.TransferRankError, match="declared size"):
        tr._check_chunk_bounds(50, 60, 100)


# -- round manifest loading is re-validated -----------------------------------


def test_load_round_manifest_rejects_disallowed_entry(tmp_path):
    import json

    manifest_path = tmp_path / "round-manifest.json"
    manifest_path.write_text(
        json.dumps([{"relative_path": "evil.py", "size": 1, "sha256": "0" * 64}])
    )
    from omlx.cluster.manifest import ManifestError

    with pytest.raises(ManifestError):
        tr._load_round_manifest(manifest_path)


def test_load_round_manifest_returns_ordered_dicts(tmp_path):
    import json

    manifest_path = tmp_path / "round-manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"relative_path": "b.json", "size": 1, "sha256": "0" * 64},
                {"relative_path": "a.json", "size": 1, "sha256": "0" * 64},
            ]
        )
    )
    entries = tr._load_round_manifest(manifest_path)
    assert [e["relative_path"] for e in entries] == ["b.json", "a.json"]


# -- argv parsing --------------------------------------------------------------


def test_main_rejects_missing_required_args():
    with pytest.raises(SystemExit):
        tr.main([])
