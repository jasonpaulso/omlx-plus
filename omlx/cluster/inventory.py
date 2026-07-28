# SPDX-License-Identifier: Apache-2.0
"""Which of this node's models could actually be served by a cluster.

Three questions, and none is answered by oMLX's own model type. A model is a
candidate when mlx-lm can split its architecture across ranks, when the
weights on disk are all there, and when mlx-lm can be trusted with the
checkpoint at all.

The first two were being guessed at. The picker filtered on
`model_type == "llm"`, which hid every large MoE on the machine - they
classify as `vlm` - while offering models whose architecture has no `shard` at
all. And nothing checked completeness, so a directory holding 6 of 126 shards
looked like a 4 GB model rather than a fragment of a 90 GB one.

The third question is the one that has no error to go by. A cluster serves
through mlx-lm; the rest of oMLX serves most of these checkpoints through
mlx-vlm. Where the two implementations disagree, nothing raises: the model
loads, the ranks form, and the answers come back fluent and wrong.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import struct
import threading
from pathlib import Path
from typing import Any

from omlx import model_integrity

logger = logging.getLogger(__name__)

_shardable: dict[str, bool] = {}
_mtp: dict[str, tuple[float, bool]] = {}
_mtp_lock = threading.Lock()


def supports_tensor_sharding(architecture: str) -> bool:
    """Whether mlx-lm can split `architecture` across ranks.

    Decided by asking mlx-lm, not by a list kept here: a family gains `shard`
    upstream and this starts answering True without a release of ours. Only
    the one module named by the config is imported, and the answer is cached -
    importing all of mlx-lm's families costs seconds and prints installation
    advice for the ones needing extra packages.

    A config's `model_type` is not always a module name. mlx-lm keeps a
    `MODEL_REMAPPING` table and consults it in `_get_classes()` before
    importing, so asking for the module directly answers False for every
    remapped family - `minimax_m2` (-> `minimax`), `kimi_k2` and
    `joyai_llm_flash` (-> `deepseek_v3`), `mistral` and `iquestcoder`
    (-> `llama`), `llava` (-> `mistral3`) all shard, and all of them were
    being hidden from the cluster picker. Reported 2026-07-27 against a
    MiniMax-M2.7 checkpoint that exo advertises as distributable.
    """
    if not architecture:
        return False
    if architecture in _shardable:
        return _shardable[architecture]

    answer = False
    try:
        import importlib

        try:
            from mlx_lm.utils import MODEL_REMAPPING
        except ImportError:  # the table moved; the direct name is still right
            MODEL_REMAPPING = {}
        module_name = MODEL_REMAPPING.get(architecture, architecture)

        # mlx-lm prints to stdout on some optional-dependency misses.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            module = importlib.import_module(f"mlx_lm.models.{module_name}")
        model = getattr(module, "Model", None)
        answer = model is not None and hasattr(model, "shard")
    except BaseException:  # noqa: BLE001 - an unknown family is simply not shardable
        answer = False

    _shardable[architecture] = answer
    return answer


def carries_mtp_weights(model_path: str) -> bool:
    """Whether the checkpoint carries multi-token-prediction weights.

    mlx-lm ignores them - they are not part of its module tree, so they load
    as unused - and then generates nonsense from the weights it did take.
    Measured 2026-07-28 on one Mac, single process, no cluster involved:

        Ornith-1.0-35B-oQ8e        qwen3_5_moe, 0 mtp weights   -> "Paris, a
                                   city renowned for its rich history..."
        Macaron-V1-Tall-oQ8e-mtp   qwen3_5_moe, 42 mtp weights  -> "-in坎t店铺
                                   经营者format的..."
        ThinkingCap-...-oQ8e-mtp   qwen3_5, 29 mtp weights      -> "'+.**"

    Same mlx-lm module for the first two, same quantisation, same box: the MTP
    heads are the only difference, and they separate correct from garbage on
    both the dense and the MoE architecture. oMLX's own engine serves all
    three correctly, which is why this cannot be left to surface as an error -
    there is no error. The cluster returns fluent nonsense at full speed, and
    on the first report of it the model looked "successfully loaded".

    Why ignoring 29 unused tensors should corrupt the other 1847 is not
    established. The rule here is the measurement, not an explanation of it.
    """
    directory = Path(model_path) if model_path else None
    if directory is None or not directory.is_dir():
        return False

    try:
        mtime = directory.stat().st_mtime
    except OSError:
        return False

    with _mtp_lock:
        cached = _mtp.get(model_path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    answer = _has_mtp_uncached(directory)
    with _mtp_lock:
        _mtp[model_path] = (mtime, answer)
    return answer


def _is_mtp_key(name: str) -> bool:
    return ".mtp." in name or name.startswith("mtp.")


def _has_mtp_uncached(directory: Path) -> bool:
    """Weight names only. Reading headers beats loading 30 GB to ask."""
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except (OSError, ValueError):
            return False
        return any(_is_mtp_key(name) for name in weight_map)

    # No index: a safetensors file states its own tensor names in a JSON
    # header, so the names cost one seek rather than a load.
    try:
        files = sorted(directory.glob("*.safetensors"))
    except OSError:
        return False
    for path in files:
        try:
            with path.open("rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
        except (OSError, ValueError, struct.error):
            continue
        if any(_is_mtp_key(name) for name in header):
            return True
    return False


def shard_census(model_path: str) -> tuple[int, int]:
    """`(present, expected)` safetensors shards for a model directory.

    `(0, 0)` when the question does not apply - a directory with no shards at
    all, or one whose files carry no count to compare against.
    """
    result = model_integrity.census(model_path)
    return (result["present"], result["expected"])


def _format_size(model: dict[str, Any]) -> str:
    """A human size, formatted here when the pool only reported bytes.

    `engine_pool.get_status()` carries the raw counts; the pre-formatted
    strings are added further up, in the admin route this does not go through.
    """
    formatted = model.get("actual_size_formatted") or model.get(
        "estimated_size_formatted"
    )
    if formatted:
        return str(formatted)
    size = model.get("actual_size") or model.get("estimated_size") or 0
    if not size:
        return ""
    from omlx.model_discovery import format_size

    return format_size(int(size))


def describe(model: dict[str, Any]) -> dict[str, Any]:
    """Judge one model from the admin model list as a cluster candidate."""
    architecture = model.get("config_model_type") or ""
    shardable = supports_tensor_sharding(architecture)
    model_path = model.get("model_path") or ""
    present, expected = shard_census(model_path)
    complete = expected == 0 or present >= expected
    mtp = carries_mtp_weights(model_path)

    if not complete:
        reason = f"only {present} of {expected} weight files are on disk"
    elif not shardable:
        reason = (
            f"mlx-lm cannot split a {architecture!r} model across ranks"
            if architecture
            else "this model has no architecture mlx-lm can split"
        )
    elif mtp:
        reason = (
            "this checkpoint carries multi-token-prediction weights, which "
            "mlx-lm ignores and then generates nonsense from the rest - one "
            "Mac serves it correctly, a cluster cannot"
        )
    else:
        reason = ""

    return {
        "id": model.get("id", ""),
        "display_name": model.get("display_name") or model.get("id", ""),
        "architecture": architecture,
        "size": model.get("estimated_size") or 0,
        "size_formatted": _format_size(model),
        "shardable": shardable,
        "complete": complete,
        "shards_present": present,
        "shards_expected": expected,
        "mtp": mtp,
        "eligible": shardable and complete and not mtp,
        "reason": reason,
    }


def candidates(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every model worth offering as the sharded model, biggest first.

    Helpers are dropped outright - a speculative-decode companion is never the
    model a cluster exists to serve. Everything else is described rather than
    hidden: a model that cannot be sharded, or is half-downloaded, is exactly
    what the operator needs told, and silently omitting it is how someone ends
    up selecting a model that could never have formed.

    Biggest first because that is the ordering of interest: a cluster exists
    for the models that do not fit on one machine.
    """
    described = [
        describe(model)
        for model in models
        if not model.get("is_helper") and model.get("model_type") != "audio_stt"
    ]
    described.sort(key=lambda d: (not d["eligible"], -d["size"]))
    return described
