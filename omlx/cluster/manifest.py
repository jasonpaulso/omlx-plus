# SPDX-License-Identifier: Apache-2.0
"""File manifests for S5 model distribution (D3/D3a).

A manifest is the head's pinned description of one model's files:
``{relative_path, size, sha256}`` per file, built over the RESOLVED model
dir — for a hub-cache entry that is the snapshot dir, never the repo root
(``blobs/``, ``refs/``, ``snapshots/`` never ship; CL5-07). The shape is S1's
orphaned CL-13 seam, :class:`~omlx.cluster.state.FileManifestEntry`, used
as-is (D3) — this module does not define a competing entry type.

:func:`validate_entry` is the single validating constructor. Nothing outside
it produces a trusted ``FileManifestEntry``, and every stat/open/hash/replace
a manifest entry can reach goes through :func:`resolve_under` on its result
-- that is CL5-06's "single validating entry type" low and half of CL5-12's
"cache is pure optimization; disk manifest is untrusted input".

Two very different failure disciplines share :func:`validate_entry` (D3a,
rev4):

* :func:`build_manifest` (head-side) FILTERS -- a file that fails the
  allowlist, escapes the repo root, or collides after casefold+NFC is
  silently excluded. Every real model dir contains dotfiles, extensionless
  files, and an ``original/**`` subtree; that is normal, not an attack.
* :func:`validate_received_manifest` (worker-side) REJECTS -- the same
  violation surviving in a manifest that already crossed the wire from the
  head is evidence of a hostile or broken head, so it is a named, fatal
  error, never silently dropped.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from .state import FileManifestEntry

# -- CL5-14: the manifest allowlist -------------------------------------------

# A served MLX model is exactly these file kinds. Anything else -- code,
# pickled/torch checkpoints, native libraries, dotfiles, extensionless files,
# `original/**` -- is excluded by the builder and rejected by the validator.
ALLOWED_EXTENSIONS = frozenset(
    {
        ".safetensors",
        ".json",
        ".txt",
        ".md",
        ".model",
        ".jinja",
        ".npz",
        ".tiktoken",
    }
)

# Named explicitly for the "explicit rejection" half of CL5-14 (the allowlist
# above is what's actually enforced; this set documents the ones an operator
# would otherwise expect to see and never should).
EXPLICITLY_REJECTED_EXTENSIONS = frozenset(
    {".py", ".bin", ".pt", ".pth", ".so", ".dylib"}
)

# -- CL5-11: exhaustion bounds -------------------------------------------------

MAX_ENTRIES = 100_000
MAX_PATH_LENGTH = 400
MAX_FILE_BYTES = 200 * 1024**3  # 200 GiB/file ceiling -- generous, still bounded
MAX_TOTAL_BYTES = 2 * 1024**4  # 2 TiB/manifest ceiling

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_HASH_CHUNK = 4 * 1024 * 1024


class ManifestError(ValueError):
    """A manifest entry, or a whole manifest, violated CL5-06/07/09/11/14."""


def validate_entry(relative_path: Any, size: Any, sha256: Any) -> FileManifestEntry:
    """Validate one candidate entry, raising :class:`ManifestError` on any
    violation, and return the resulting :class:`FileManifestEntry`.

    The single function both the filtering builder and the rejecting
    validator route every entry through (D3a) -- there is no other way to
    obtain a ``FileManifestEntry`` whose ``relative_path`` violates
    traversal/NUL/backslash/allowlist/length rules or whose ``size``/
    ``sha256`` fail their shape checks.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise ManifestError("relative_path must be a non-empty string")
    if len(relative_path) > MAX_PATH_LENGTH:
        raise ManifestError(
            f"relative_path exceeds {MAX_PATH_LENGTH} characters: {relative_path!r}"
        )
    if "\x00" in relative_path:
        raise ManifestError(f"relative_path contains a NUL byte: {relative_path!r}")
    if "\\" in relative_path:
        raise ManifestError(f"relative_path contains a backslash: {relative_path!r}")
    normalized = unicodedata.normalize("NFC", relative_path)
    posix = PurePosixPath(normalized)
    if posix.is_absolute():
        raise ManifestError(f"relative_path must not be absolute: {relative_path!r}")
    parts = posix.parts
    if not parts:
        raise ManifestError(f"relative_path must not be empty: {relative_path!r}")
    for part in parts:
        if part in ("", ".", ".."):
            raise ManifestError(
                f"relative_path contains an illegal segment {part!r}: "
                f"{relative_path!r}"
            )
        if part.startswith("."):
            raise ManifestError(
                f"relative_path contains a dotfile segment: {relative_path!r}"
            )
    if parts[0] == "original":
        raise ManifestError(f"'original/**' is excluded: {relative_path!r}")
    suffix = posix.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ManifestError(
            f"extension {suffix!r} is not on the manifest allowlist: "
            f"{relative_path!r}"
        )
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ManifestError(f"size must be a non-negative integer: {size!r}")
    if size > MAX_FILE_BYTES:
        raise ManifestError(
            f"size {size} exceeds the {MAX_FILE_BYTES} per-file ceiling"
        )
    if not isinstance(sha256, str) or not _SHA256_RE.match(sha256):
        raise ManifestError(
            f"sha256 must match ^[0-9a-f]{{64}}$ (no algorithm field, ever): "
            f"{sha256!r}"
        )
    return FileManifestEntry(relative_path=str(posix), size=size, sha256=sha256)


def resolve_under(entry: FileManifestEntry, root: Path) -> Path:
    """The only way a manifest entry becomes a concrete filesystem path.

    ``entry.relative_path`` was already validated traversal-free (no ``..``,
    no absolute form), so this join cannot lexically escape ``root``. It does
    NOT re-check the filesystem -- the entry may not exist yet (staging,
    pre-write). Callers that then create/open/replace the result MUST
    separately re-validate realpath containment after the filesystem
    operation (CL5-08): a symlinked ancestor swapped in between validation
    and use is not caught by a lexical join alone.
    """
    return Path(root) / entry.relative_path


def assert_realpath_contained(path: Path, root: Path) -> Path:
    """Re-check that ``path`` (which now exists) really resolves under
    ``root`` (CL5-08's "re-validate realpath-under-root at open AND at
    os.replace"). Raises :class:`ManifestError` on escape.
    """
    root_real = Path(root).resolve()
    path_real = Path(path).resolve()
    try:
        path_real.relative_to(root_real)
    except ValueError as exc:
        raise ManifestError(
            f"{path} resolves outside {root} after creation (symlink swap?)"
        ) from exc
    return path_real


def _hub_cache_repo_root(snapshot_dir: Path) -> Path | None:
    """The repo root for a hub-cache ``.../snapshots/<hash>`` snapshot dir.

    ``None`` when ``snapshot_dir`` is not shaped like one (a plain local
    model directory has no repo root notion, so an escaping symlink there
    has nothing legitimate to escape TO).
    """
    if snapshot_dir.parent.name == "snapshots":
        return snapshot_dir.parent.parent
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# -- D3: head-side manifest builder (filters, CL5-07 symlink resolution) -----


def build_manifest(model_dir: Path) -> tuple[FileManifestEntry, ...]:
    """Build a manifest over the RESOLVED model dir (D3a rev5: snapshot-rooted
    for both sources -- this is the flat serving tree, never a hub-cache
    repo root).

    Every symlink is resolved; a target outside the repo root is a hard
    error (CL5-07) -- ``../../blobs/<hash>`` from a snapshot dir is the
    legitimate hub-cache shape and is allowed, anything else is not. A file
    (symlinked or not) that fails the CL5-14 allowlist/traversal/dedup
    checks is silently excluded, never fatal (D3a).
    """
    model_dir = Path(model_dir).resolve()
    seen_norm: set[str] = set()
    entries: list[FileManifestEntry] = []
    total_bytes = 0
    for path in sorted(model_dir.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(model_dir)
        posix = relative.as_posix()

        if path.is_symlink():
            target = path.resolve()
            if not _is_within(target, model_dir):
                repo_root = _hub_cache_repo_root(model_dir)
                if repo_root is None or not _is_within(target, repo_root):
                    raise ManifestError(
                        f"symlink {posix!r} resolves outside the repo root: "
                        f"{target}"
                    )
            if not target.is_file():
                continue
            stat_target = target
        else:
            if not path.is_file():
                continue
            stat_target = path

        try:
            size = stat_target.stat().st_size
        except OSError:
            continue
        try:
            shaped = validate_entry(posix, size, "0" * 64)
        except ManifestError:
            continue  # D3a: filtered, never fatal

        norm_key = unicodedata.normalize("NFC", shaped.relative_path).casefold()
        if norm_key in seen_norm:
            continue
        seen_norm.add(norm_key)

        entry = FileManifestEntry(
            relative_path=shaped.relative_path,
            size=size,
            sha256=_sha256_file(stat_target),
        )
        entries.append(entry)
        total_bytes += entry.size
        if len(entries) > MAX_ENTRIES:
            raise ManifestError(f"manifest exceeds {MAX_ENTRIES} entries")
        if total_bytes > MAX_TOTAL_BYTES:
            raise ManifestError(f"manifest exceeds {MAX_TOTAL_BYTES} total bytes")
    return tuple(entries)


# -- D3a: worker-side manifest validator (rejects by name) -------------------


def validate_received_manifest(raw_entries: Any) -> tuple[FileManifestEntry, ...]:
    """Validate a manifest that crossed the wire from the head.

    Unlike :func:`build_manifest`, every violation here is fatal and named
    (D3a): the head already filtered its own manifest, so a violation that
    survived is evidence of a hostile or broken head.
    """
    if not isinstance(raw_entries, (list, tuple)):
        raise ManifestError("manifest must be a list of entries")
    if len(raw_entries) > MAX_ENTRIES:
        raise ManifestError(f"manifest exceeds {MAX_ENTRIES} entries")
    if not raw_entries:
        raise ManifestError("manifest has no entries")

    seen_norm: set[str] = set()
    entries: list[FileManifestEntry] = []
    total_bytes = 0
    for raw in raw_entries:
        relative_path: Any
        size: Any
        sha256: Any
        if isinstance(raw, FileManifestEntry):
            relative_path, size, sha256 = raw.relative_path, raw.size, raw.sha256
        elif isinstance(raw, dict):
            relative_path = raw.get("relative_path")
            size = raw.get("size")
            sha256 = raw.get("sha256")
        else:
            raise ManifestError(f"manifest entry has an unexpected shape: {raw!r}")
        entry = validate_entry(relative_path, size, sha256)

        norm_key = unicodedata.normalize("NFC", entry.relative_path).casefold()
        if norm_key in seen_norm:
            raise ManifestError(
                f"duplicate manifest entry after casefold+NFC: "
                f"{entry.relative_path!r}"
            )
        seen_norm.add(norm_key)
        total_bytes += entry.size
        entries.append(entry)

    if total_bytes > MAX_TOTAL_BYTES:
        raise ManifestError(f"manifest exceeds {MAX_TOTAL_BYTES} total bytes")
    return tuple(entries)


# -- CL5-12: content-addressed manifest cache ---------------------------------

CACHE_SUFFIX = ".manifest-cache.json"


def _walk_stat_meta(model_dir: Path) -> list[tuple[str, int, int, int]]:
    """Ordered ``(relative_path, size, st_mtime_ns, st_ino)`` over every file
    a manifest build would see (symlinks stat'd through their target, same
    as :func:`build_manifest`) -- the cheap fingerprint the cache key hashes
    without re-hashing file contents.
    """
    meta: list[tuple[str, int, int, int]] = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        try:
            st = path.stat()  # follows symlinks, matching build_manifest
        except OSError:
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        meta.append((relative, st.st_size, st.st_mtime_ns, st.st_ino))
    return meta


def _cache_key(model_dir: Path, meta: list[tuple[str, int, int, int]]) -> str:
    """Hash of ordered ``(relative_path, size, st_mtime_ns, st_ino)`` plus the
    resolved realpath and count (CL5-12) -- invalidates on any rename,
    resize, retouch, or inode change, and on the model moving.
    """
    digest = hashlib.sha256()
    digest.update(str(model_dir).encode())
    digest.update(str(len(meta)).encode())
    for relative_path, size, mtime_ns, ino in meta:
        digest.update(f"{relative_path}\0{size}\0{mtime_ns}\0{ino}\0".encode())
    return digest.hexdigest()


def cached_or_build_manifest(
    model_dir: Path, cache_dir: Path
) -> tuple[FileManifestEntry, ...]:
    """The manifest for ``model_dir``, from the cache when the key still
    matches, else built fresh and cached.

    The cache is a pure optimization (CL5-12): a cache file is parsed
    through the SAME entry validator as any other untrusted input, so a
    corrupted or hand-edited cache file can only ever produce a validation
    failure (treated as a cache miss), never an unvalidated entry.
    """
    model_dir = Path(model_dir).resolve()
    meta = _walk_stat_meta(model_dir)
    key = _cache_key(model_dir, meta)
    cache_path = Path(cache_dir) / f"{key}{CACHE_SUFFIX}"

    cached = _read_cache(cache_path, expected_key=key)
    if cached is not None:
        return cached

    manifest = build_manifest(model_dir)
    _write_cache(cache_path, key, manifest)
    return manifest


def _read_cache(
    cache_path: Path, *, expected_key: str
) -> tuple[FileManifestEntry, ...] | None:
    try:
        raw = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("key") != expected_key:
        return None
    try:
        return validate_received_manifest(raw.get("entries") or [])
    except ManifestError:
        return None


def _write_cache(
    cache_path: Path, key: str, manifest: tuple[FileManifestEntry, ...]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key, "entries": [entry.to_dict() for entry in manifest]}
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    os.replace(tmp_path, cache_path)
