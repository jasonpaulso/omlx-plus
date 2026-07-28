# SPDX-License-Identifier: Apache-2.0
"""Which models a cluster could actually serve.

Both questions here were previously guessed at by the picker, and both guesses
were wrong in the same direction: it hid the large models a cluster exists for
and offered ones that could never have formed.
"""

from __future__ import annotations

import pytest

from omlx.cluster import inventory


def _model(**overrides):
    base = {
        "id": "big-model",
        "display_name": "big-model",
        "model_path": "",
        "model_type": "llm",
        "config_model_type": "qwen3_5_moe",
        "estimated_size": 10,
        "estimated_size_formatted": "10 B",
        "is_helper": False,
    }
    base.update(overrides)
    return base


def _write_shards(directory, present, expected):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(1, present + 1):
        (directory / f"model-{i:05d}-of-{expected:05d}.safetensors").write_bytes(b"x")
    return str(directory)


def test_a_half_downloaded_model_is_not_a_candidate(tmp_path):
    """6 of 126 shards looked like a 4 GB model rather than a fragment of a
    90 GB one, right up until formation failed on missing parameters."""
    path = _write_shards(tmp_path / "partial", present=6, expected=126)

    described = inventory.describe(_model(model_path=path))

    assert described["complete"] is False
    assert described["eligible"] is False
    assert described["shards_present"] == 6
    assert described["shards_expected"] == 126
    assert "6 of 126" in described["reason"]


def test_a_fully_downloaded_model_is_complete(tmp_path):
    path = _write_shards(tmp_path / "whole", present=8, expected=8)

    described = inventory.describe(_model(model_path=path))

    assert described["complete"] is True
    assert described["eligible"] is True
    assert described["reason"] == ""


def test_a_single_file_model_is_not_reported_incomplete(tmp_path):
    """No `-of-N` suffix means there is nothing to compare against; guessing
    would flag every single-shard model as broken."""
    directory = tmp_path / "single"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"x")

    described = inventory.describe(_model(model_path=str(directory)))

    assert described["complete"] is True
    assert described["shards_expected"] == 1


@pytest.mark.parametrize("architecture", ["llama", "qwen3_5_moe", "gpt_oss"])
def test_architectures_mlx_lm_can_split_are_eligible(architecture):
    assert inventory.supports_tensor_sharding(architecture) is True


@pytest.mark.parametrize("architecture", ["gemma4", "nanbeige", ""])
def test_architectures_mlx_lm_cannot_split_are_refused(architecture):
    """The model the operator had selected was one of these: the weights can
    be everywhere and the cluster still never forms."""
    assert inventory.supports_tensor_sharding(architecture) is False


@pytest.mark.parametrize(
    "architecture",
    ["minimax_m2", "mistral", "kimi_k2", "joyai_llm_flash", "iquestcoder"],
)
def test_remapped_architectures_are_resolved_before_asking(architecture):
    """A config's `model_type` is not always a module name.

    mlx-lm consults `MODEL_REMAPPING` in `_get_classes()` before importing, so
    importing `mlx_lm.models.<model_type>` directly answers False for every
    remapped family that in fact shards. Reported 2026-07-27 against a
    MiniMax-M2.7 checkpoint (`model_type: minimax_m2` -> `mlx_lm.models.minimax`,
    which defines `shard`) that exo also advertises as tensor-splittable.
    """
    assert inventory.supports_tensor_sharding(architecture) is True


def test_remapping_does_not_invent_support():
    """Remapped families that land on a module without `shard` stay refused."""
    for architecture in ("falcon_mamba", "phi-msft", "qwen2_5_vl", "llava"):
        assert inventory.supports_tensor_sharding(architecture) is False


def test_an_unshardable_model_says_so_rather_than_vanishing(tmp_path):
    path = _write_shards(tmp_path / "gemma", present=7, expected=7)

    described = inventory.describe(
        _model(id="gemma-4-31B", config_model_type="gemma4", model_path=path)
    )

    assert described["eligible"] is False
    assert described["complete"] is True
    assert "gemma4" in described["reason"]


def test_candidates_lead_with_the_big_models(tmp_path):
    """The picker used to filter on oMLX's `model_type`, which classifies the
    large MoEs as `vlm` and hid every one of them - the exact models a cluster
    exists to serve."""
    path = _write_shards(tmp_path / "ok", present=1, expected=1)
    models = [
        _model(id="small", model_type="llm", estimated_size=1, model_path=path),
        _model(id="huge-moe", model_type="vlm", estimated_size=71_000, model_path=path),
        _model(id="helper", is_helper=True, estimated_size=999_999, model_path=path),
        _model(id="voice", model_type="audio_stt", estimated_size=5, model_path=path),
    ]

    ids = [c["id"] for c in inventory.candidates(models)]

    assert ids == ["huge-moe", "small"]


def test_an_ineligible_model_is_listed_after_the_usable_ones(tmp_path):
    usable = _write_shards(tmp_path / "usable", present=1, expected=1)
    broken = _write_shards(tmp_path / "broken", present=1, expected=9)
    models = [
        _model(id="broken", estimated_size=99_999, model_path=broken),
        _model(id="usable", estimated_size=1, model_path=usable),
    ]

    described = inventory.candidates(models)

    assert [c["id"] for c in described] == ["usable", "broken"]
    assert described[0]["eligible"] is True
    assert described[1]["eligible"] is False
