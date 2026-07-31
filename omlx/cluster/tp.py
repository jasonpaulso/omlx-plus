# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel model loading — the single place oMLX touches mlx-lm's
sharding API.

mlx-lm's distributed serving surface is moving fast and is treated as unstable
API: when it moves, this file is the blast radius. Everything asserted here was
read out of mlx-lm 0.31.3 (`sharded_load`'s tensor-group branch) and confirmed
by loading, not taken from documentation.

The load recipe is deliberately hand-rolled rather than a call to
`sharded_load`, for two reasons this slice needs:

* the head-count divisibility check has to run *before* `model.shard(group)`,
  which halves `n_heads` in place with no divisibility guard of its own; and
* the ``mx.eval`` boundary that materialises only the post-shard slice is the
  exact thing the P1 memory gate measures, so it must be under our control.

The recipe: load lazily, check shardability and divisibility, `shard(group)`,
then ``mx.eval`` the parameters — which realises only this rank's slice,
because ``shard`` has already replaced every full weight with its slice. That
is what must keep a 93 GB model inside a 96 GB node.
"""

from __future__ import annotations

import importlib
import io
import logging
import pkgutil
import resource
from collections.abc import Callable, Iterable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class TPIncompleteModelError(RuntimeError):
    """The model dir's own safetensors index references weight files that
    are not on disk (e.g. a partially transferred or externally holed
    copy). Raised BEFORE the lazy ``load_model`` -- mlx-lm loads with
    ``strict=False``, so missing shards would otherwise materialise as a
    silently wrong model that only misbehaves at first forward (S5 P3 rig
    finding)."""


class TPUnsupportedError(RuntimeError):
    """A model does not implement tensor-parallel ``shard(group)``."""


class TPDivisibilityError(RuntimeError):
    """A model's head count is not divisible by the world size."""


# -- capability probe --------------------------------------------------------

# A source of (module_name, module) pairs. Injectable so the probe can be
# exercised against stub modules without importing mlx-lm.
ModuleSource = Callable[[], Iterable[tuple[str, Any]]]


def _iter_mlx_lm_model_modules() -> Iterable[tuple[str, Any]]:
    """Import every `mlx_lm.models.*` module, yielding (name, module).

    Some model modules print installation hints on import; that noise is kept
    out of the daemon's own stdout. A module that fails to import (optional
    dependency, bad import) is skipped rather than fatal.
    """
    import mlx_lm.models

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        for module in pkgutil.iter_modules(mlx_lm.models.__path__):
            try:
                loaded = importlib.import_module(f"mlx_lm.models.{module.name}")
            except BaseException:  # noqa: BLE001 - optional deps, bad imports
                continue
            yield module.name, loaded


def tensor_parallel_architectures(
    iter_modules: ModuleSource | None = None,
) -> frozenset[str]:
    """Model families whose ``Model`` class implements ``shard(group)``.

    Derived by inspecting the installed mlx-lm rather than hard-coded: the set
    moves with every mlx-lm release, and a stale allow-list is worse than none.
    This is only a fast answer for preflight; the authoritative check is
    :func:`supports_tensor_parallel` on the loaded model.

    ``iter_modules`` defaults to walking ``mlx_lm.models``; tests inject a stub
    source of (name, module) pairs.
    """
    source = iter_modules or _iter_mlx_lm_model_modules
    found: set[str] = set()
    for name, loaded in source():
        model_cls = getattr(loaded, "Model", None)
        if model_cls is not None and hasattr(model_cls, "shard"):
            found.add(name)
    return frozenset(found)


def supports_tensor_parallel(model: Any) -> bool:
    """The authoritative per-model check: does this instance expose ``shard``?"""
    return hasattr(model, "shard")


# -- divisibility ------------------------------------------------------------

_HEAD_KEYS = ("num_attention_heads", "n_heads")
_KV_HEAD_KEYS = ("num_key_value_heads", "n_kv_heads")


def _first_present(config: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return int(value)
    return None


def check_divisibility(config: dict[str, Any], world_size: int) -> None:
    """Refuse a shard that would silently mis-split attention heads.

    ``Model.shard`` does ``n_heads //= world_size`` with no guard; an
    indivisible head count yields a wrong-shaped attention on every rank and a
    corrupt forward pass. Surfaced here as a clear load-time error instead.

    S6 D0: a ``language_model_only`` multimodal wrapper config (the vision
    tower shipped off) carries its head counts under a nested ``text_config``
    rather than top-level -- without this fallback the top-level lookup comes
    back ``None`` and this lenient-passes a config it never actually checked.
    Only consulted when the top-level key is absent, per field.
    """
    if world_size <= 1:
        return
    nested = config.get("text_config")
    nested = nested if isinstance(nested, dict) else None

    heads = _first_present(config, _HEAD_KEYS)
    if heads is None and nested is not None:
        heads = _first_present(nested, _HEAD_KEYS)
    if heads is not None and heads % world_size != 0:
        raise TPDivisibilityError(
            f"attention heads ({heads}) not divisible by world size ({world_size})"
        )
    kv_heads = _first_present(config, _KV_HEAD_KEYS)
    if kv_heads is None and nested is not None:
        kv_heads = _first_present(nested, _KV_HEAD_KEYS)
    if kv_heads is not None and kv_heads % world_size != 0:
        raise TPDivisibilityError(
            f"key/value heads ({kv_heads}) not divisible by world size "
            f"({world_size})"
        )


# -- memory measurement ------------------------------------------------------


def measure_param_bytes(model: Any) -> int:
    """Total bytes held by this model instance's parameters."""
    from mlx.utils import tree_flatten

    total = 0
    for entry in tree_flatten(model.parameters()):
        nbytes = getattr(entry[1], "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
    return total


def peak_process_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is bytes on Darwin (kilobytes on Linux); this runs on Apple
    Silicon, so it is bytes — verified on the rig before the gate was trusted.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


@dataclass(frozen=True)
class ShardResult:
    """A loaded, sharded model plus the numbers the memory gate reports."""

    model: Any
    tokenizer: Any
    config: dict[str, Any]
    post_shard_param_bytes: int


def missing_weight_files(model_dir: Any) -> list[str]:
    """Weight files the model dir's own metadata says should exist but
    don't -- empty when complete (or when there is no index to check
    against and at least one ``*.safetensors`` is present).

    Existence-only by design: cheap stat calls suitable for a heartbeat
    inventory scan. Digest-level integrity stays the transfer have-scan's
    job (D2); this catches holes, not bitrot.
    """
    import json as _json
    from pathlib import Path as _Path

    root = _Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = _json.loads(index.read_text()).get("weight_map", {})
        except (OSError, ValueError):
            return []  # unreadable index: let the loader surface it
        names = sorted(set(weight_map.values()))
        return [name for name in names if not (root / name).is_file()]
    if (root / "config.json").is_file() and not any(root.glob("*.safetensors")):
        return ["model.safetensors"]
    return []


# S6 P1c/D0: the `self.<name> = ...` vision-submodule attributes actually
# used across mlx_vlm's installed model architectures (grepped from the
# dependency this fork loads through, not guessed) -- `vision_tower`/
# `vision_model` cover the large majority (Qwen's family among them, which
# is what the `language_model_only` eligibility flag exists for);
# `vision_encoder`/`visual`/`vision` cover the rest.
VISION_WEIGHT_PREFIXES = (
    "vision_tower.",
    "vision_model.",
    "vision_encoder.",
    "visual.",
    "vision.",
)


def has_vision_tower_weights(model_dir: Any) -> bool:
    """Does this model dir's own weight index actually SHIP vision-tower
    parameters, regardless of what its config declares?

    S6 D0's `language_model_only: true` eligibility flag claims the vision
    tower was stripped before distribution; this is the cross-check that
    catches a MISLABELED checkpoint (flag true, weights say otherwise) --
    something `plan_placement` alone (pure, config-only) cannot see.
    Existence/name check only, mirroring :func:`missing_weight_files`'s
    shape exactly: reads the safetensors index's ``weight_map`` KEYS (the
    parameter names, never the shard filenames) and never loads a weight.
    False (not eligible-by-omission) when there is no index to check --
    the loader's own missing-file guard is what surfaces that case.
    """
    import json as _json
    from pathlib import Path as _Path

    root = _Path(model_dir)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return False
    try:
        weight_map = _json.loads(index.read_text()).get("weight_map", {})
    except (OSError, ValueError):
        return False
    return any(
        name.startswith(prefix)
        for name in weight_map
        for prefix in VISION_WEIGHT_PREFIXES
    )


def shard_and_load(model_path: str, group: Any) -> ShardResult:
    """Load only this rank's tensor-parallel slice of ``model_path``.

    Mirrors mlx-lm's own ``sharded_load`` tensor-group recipe, with the
    divisibility guard inserted before ``shard`` and the materialisation
    boundary made explicit:

    1. resolve the model locally (download only if absent),
    2. lazy-load to inspect shardability and config,
    3. refuse a non-shardable model or an indivisible head count,
    4. ``shard(group)`` — replace every full weight with this rank's slice,
    5. ``mx.eval`` the parameters — realise only that slice,
    6. a cpu-stream ``all_sum`` barrier so ranks do not skew into a timeout.
    """
    import mlx.core as mx
    from mlx_lm.utils import _download, load_model, load_tokenizer

    resolved = _download(model_path)
    missing = missing_weight_files(resolved)
    if missing:
        raise TPIncompleteModelError(
            f"{model_path} is missing {len(missing)} weight file(s) its own "
            f"index references (first: {missing[0]!r}); refusing to load a "
            "silently incomplete model"
        )
    model, config = load_model(resolved, lazy=True, strict=False)

    if not supports_tensor_parallel(model):
        raise TPUnsupportedError(
            f"{model_path} does not implement Model.shard(group); "
            "tensor parallelism is not available for this architecture"
        )
    check_divisibility(config, int(group.size()))

    model.shard(group)
    mx.eval(model.parameters())
    # Match sharded_load: keep ranks in step so a slow shard does not read as a
    # dead peer to the others. Pinned to the cpu stream (Metal timeout).
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))

    tokenizer = load_tokenizer(
        resolved,
        {"trust_remote_code": False},
        eos_token_ids=config.get("eos_token_id"),
    )
    param_bytes = measure_param_bytes(model)
    logger.info(
        "cluster: rank %d loaded %s tensor-parallel (%d shard param bytes)",
        int(group.rank()),
        model_path,
        param_bytes,
    )
    return ShardResult(
        model=model,
        tokenizer=tokenizer,
        config=config,
        post_shard_param_bytes=param_bytes,
    )
