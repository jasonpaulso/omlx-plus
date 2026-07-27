# SPDX-License-Identifier: Apache-2.0
"""Is a model on disk actually all there?

A download that stops halfway leaves a directory that looks like a model and
lists like a model. Nothing said otherwise until it was loaded, and then it
said `Missing 1621 parameters` - a message about tensors, several layers away
from the cause, which is that some of the files are simply not there.

The check is nearly free because the answer is in the filenames.
`model-00006-of-00126.safetensors` states how many shards the set is supposed
to have, so counting the directory answers it without opening a single file.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SHARD_RE = re.compile(r"-of-(\d+)\.safetensors$")

# Keyed by path, holding the directory mtime it was computed against. A model
# directory changes when files land in it, so this stays correct across a
# download finishing without anyone having to invalidate it.
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()


def census(model_path: str) -> dict[str, Any]:
    """How many weight files a model has, and how many it should have.

    `expected` is 0 when the question does not apply - nothing on disk, or a
    naming scheme with no count in it. `complete` is True in that case, since
    reporting "incomplete" on no evidence would flag every single-file model.
    """
    directory = Path(model_path) if model_path else None
    if directory is None or not directory.is_dir():
        return {"present": 0, "expected": 0, "complete": True, "checked": False}

    try:
        mtime = directory.stat().st_mtime
    except OSError:
        return {"present": 0, "expected": 0, "complete": True, "checked": False}

    with _lock:
        cached = _cache.get(model_path)
        if cached is not None and cached[0] == mtime:
            return dict(cached[1])

    result = _census_uncached(directory)
    with _lock:
        _cache[model_path] = (mtime, result)
    return dict(result)


def _census_uncached(directory: Path) -> dict[str, Any]:
    indexed = _census_from_index(directory)
    if indexed is not None:
        return indexed

    try:
        shards = sorted(directory.glob("*.safetensors"))
    except OSError:
        return {"present": 0, "expected": 0, "complete": True, "checked": False}

    if not shards:
        return {"present": 0, "expected": 0, "complete": True, "checked": False}

    match = _SHARD_RE.search(shards[0].name)
    expected = int(match.group(1)) if match else len(shards)
    present = len(shards)
    return {
        "present": present,
        "expected": expected,
        "complete": present >= expected,
        "checked": True,
    }


def _census_from_index(directory: Path) -> dict[str, Any] | None:
    """Census against `model.safetensors.index.json`, if there is one.

    Preferred over counting filenames, because the index names every file the
    loader's weight map needs - and it is checked by **existence**, not by
    count. Some models reference a file in a subdirectory
    (`optiq/optiq_vision.safetensors`), which a top-level glob never sees; that
    read as one file short and reported two perfectly good models as broken.
    """
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return None
    try:
        weight_map = json.loads(index.read_text()).get("weight_map", {})
    except (OSError, ValueError):
        return None
    referenced = sorted(set(weight_map.values()))
    if not referenced:
        return None
    present = sum(1 for name in referenced if (directory / name).is_file())
    return {
        "present": present,
        "expected": len(referenced),
        "complete": present == len(referenced),
        "checked": True,
    }


def describe(model_path: str) -> dict[str, Any]:
    """`census`, plus a sentence for anything that is not complete."""
    result = census(model_path)
    if result["complete"] or not result["checked"]:
        result["detail"] = ""
    else:
        result["detail"] = (
            f"{result['present']} of {result['expected']} weight files are on disk"
        )
    return result
