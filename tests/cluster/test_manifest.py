# SPDX-License-Identifier: Apache-2.0
"""S5 manifest builder + validator + cache (D3/D3a, CL5-06/07/09/11/12/14)."""

from __future__ import annotations

import os
import stat

import pytest

from omlx.cluster import manifest as m
from omlx.cluster.state import FileManifestEntry


def _write(root, relative, data=b"x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# -- D3a: builder filters, never fatal ----------------------------------------


def test_build_manifest_filters_dotfiles_extensionless_and_original(tmp_path):
    _write(tmp_path, "model.safetensors")
    _write(tmp_path, "config.json")
    _write(tmp_path, ".gitattributes")
    _write(tmp_path, ".cache/huggingface/download/model.safetensors.metadata")
    _write(tmp_path, "noext")
    _write(tmp_path, "original/consolidated.00.pth")

    entries = m.build_manifest(tmp_path)
    assert {e.relative_path for e in entries} == {"model.safetensors", "config.json"}
    assert all(isinstance(e, FileManifestEntry) for e in entries)


def test_build_manifest_rejects_explicit_extensions_silently(tmp_path):
    _write(tmp_path, "config.json")
    _write(tmp_path, "run.py")
    _write(tmp_path, "weights.bin")
    entries = m.build_manifest(tmp_path)
    assert {e.relative_path for e in entries} == {"config.json"}


def test_build_manifest_digests_are_deterministic(tmp_path):
    _write(tmp_path, "config.json", data=b"hello world")
    e1 = m.build_manifest(tmp_path)
    e2 = m.build_manifest(tmp_path)
    assert e1 == e2
    assert e1[0].sha256 == __import__("hashlib").sha256(b"hello world").hexdigest()


# -- D3a: worker-side validator rejects by name --------------------------------


def test_validate_received_manifest_rejects_disallowed_extension():
    with pytest.raises(m.ManifestError, match="allowlist"):
        m.validate_received_manifest(
            [{"relative_path": "evil.py", "size": 1, "sha256": "0" * 64}]
        )


def test_validate_received_manifest_rejects_traversal():
    for bad in ("../etc/passwd", "/etc/passwd", "a/../../b.json", "a\\b.json"):
        with pytest.raises(m.ManifestError):
            m.validate_received_manifest(
                [{"relative_path": bad, "size": 1, "sha256": "0" * 64}]
            )


def test_validate_received_manifest_rejects_bad_sha256_shape():
    with pytest.raises(m.ManifestError, match="sha256"):
        m.validate_received_manifest(
            [{"relative_path": "a.json", "size": 1, "sha256": "not-hex"}]
        )
    with pytest.raises(m.ManifestError, match="sha256"):
        m.validate_received_manifest(
            [{"relative_path": "a.json", "size": 1, "sha256": "0" * 63}]
        )


def test_validate_received_manifest_rejects_casefold_nfc_collision():
    # "A.JSON" and "a.json" collide after casefold.
    with pytest.raises(m.ManifestError, match="duplicate"):
        m.validate_received_manifest(
            [
                {"relative_path": "A.json", "size": 1, "sha256": "0" * 64},
                {"relative_path": "a.json", "size": 1, "sha256": "0" * 64},
            ]
        )


def test_validate_received_manifest_rejects_empty_manifest():
    with pytest.raises(m.ManifestError):
        m.validate_received_manifest([])


def test_validate_received_manifest_accepts_a_filtered_head_manifest(tmp_path):
    _write(tmp_path, "model.safetensors")
    _write(tmp_path, "config.json")
    entries = m.build_manifest(tmp_path)
    validated = m.validate_received_manifest([e.to_dict() for e in entries])
    assert validated == entries


# -- CL5-11: bounds -------------------------------------------------------------


def test_validate_received_manifest_rejects_oversized_file(monkeypatch):
    with pytest.raises(m.ManifestError, match="ceiling"):
        m.validate_received_manifest(
            [
                {
                    "relative_path": "a.json",
                    "size": m.MAX_FILE_BYTES + 1,
                    "sha256": "0" * 64,
                }
            ]
        )


def test_validate_received_manifest_rejects_too_many_entries(monkeypatch):
    monkeypatch.setattr(m, "MAX_ENTRIES", 2)
    entries = [
        {"relative_path": f"a{i}.json", "size": 1, "sha256": "0" * 64} for i in range(3)
    ]
    with pytest.raises(m.ManifestError, match="entries"):
        m.validate_received_manifest(entries)


# -- CL5-07: symlink resolution -------------------------------------------------


def test_build_manifest_resolves_hub_cache_symlinks_within_repo_root(tmp_path):
    repo_root = tmp_path / "models--org--repo"
    blobs = repo_root / "blobs"
    snap = repo_root / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(b"hello")
    os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
    _write(snap, "config.json")

    entries = m.build_manifest(snap)
    assert {e.relative_path for e in entries} == {"model.safetensors", "config.json"}
    # No hub-cache internal structure ships -- the manifest is flat.
    assert all("/" not in e.relative_path for e in entries)


def test_build_manifest_rejects_symlink_escaping_repo_root(tmp_path):
    repo_root = tmp_path / "models--org--repo"
    snap = repo_root / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"nope")
    os.symlink(outside, snap / "evil.safetensors")

    with pytest.raises(m.ManifestError, match="outside the repo root"):
        m.build_manifest(snap)


def test_build_manifest_rejects_escaping_symlink_for_plain_local_dir(tmp_path):
    # A plain (non hub-cache-shaped) model dir has no repo root to escape
    # into -- any escaping symlink is rejected outright.
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"nope")
    os.symlink(outside, model_dir / "weights.safetensors")

    with pytest.raises(m.ManifestError):
        m.build_manifest(model_dir)


# -- CL5-12: content-addressed cache --------------------------------------------


def test_cache_roundtrip_and_invalidation_on_mtime_change(tmp_path):
    model_dir = tmp_path / "model"
    cache_dir = tmp_path / "cache"
    _write(model_dir, "config.json", data=b"v1")

    first = m.cached_or_build_manifest(model_dir, cache_dir)
    cache_files = list(cache_dir.iterdir())
    assert len(cache_files) == 1
    assert stat.S_IMODE(cache_files[0].stat().st_mode) == 0o600

    second = m.cached_or_build_manifest(model_dir, cache_dir)
    assert second == first

    # Touch the file: a changed mtime invalidates the key and produces a
    # second cache entry (the digest itself is unchanged here, but the key
    # is mtime/inode-derived so it still misses).
    os.utime(model_dir / "config.json", None)
    third = m.cached_or_build_manifest(model_dir, cache_dir)
    assert third == first  # same content -> same manifest
    assert len(list(cache_dir.iterdir())) == 2  # but a new cache entry


def test_cache_treats_a_corrupted_cache_file_as_a_miss(tmp_path):
    model_dir = tmp_path / "model"
    cache_dir = tmp_path / "cache"
    _write(model_dir, "config.json")
    cache_dir.mkdir()
    # A hand-edited cache file with an illegal entry must not become a
    # trusted manifest -- it's parsed through the same validator as any
    # other untrusted input and simply misses.
    bogus = cache_dir / "bogus.manifest-cache.json"
    bogus.write_text(
        '{"key": "whatever", "entries": [{"relative_path": "x.py", '
        '"size": 1, "sha256": "' + "0" * 64 + '"}]}'
    )
    manifest = m.cached_or_build_manifest(model_dir, cache_dir)
    assert {e.relative_path for e in manifest} == {"config.json"}


# -- resolve_under / realpath containment --------------------------------------


def test_resolve_under_joins_lexically():
    entry = FileManifestEntry(relative_path="a/b.json", size=1, sha256="0" * 64)
    assert m.resolve_under(entry, "/root") == __import__("pathlib").Path(
        "/root/a/b.json"
    )


def test_assert_realpath_contained_rejects_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = root / "escaped"
    os.symlink(outside, escaped)
    with pytest.raises(m.ManifestError):
        m.assert_realpath_contained(escaped, root)


def test_assert_realpath_contained_accepts_contained_path(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    inner = root / "sub" / "file.json"
    inner.write_text("{}")
    assert m.assert_realpath_contained(inner, root) == inner.resolve()
