# SPDX-License-Identifier: Apache-2.0
"""Which models a cluster could actually serve.

Both questions here were previously guessed at by the picker, and both guesses
were wrong in the same direction: it hid the large models a cluster exists for
and offered ones that could never have formed.
"""

from __future__ import annotations

import json
import struct

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


def _write_index(directory, weight_names):
    """A model directory whose index names `weight_names`, files and all."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model-00001-of-00001.safetensors").write_bytes(b"x")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {n: "model-00001-of-00001.safetensors" for n in weight_names}}
        )
    )
    return str(directory)


def _write_single_file(directory, tensor_names):
    """One safetensors file with a real header, no index alongside it."""
    directory.mkdir(parents=True, exist_ok=True)
    header = json.dumps({n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in tensor_names}).encode()
    with (directory / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(b"\x00\x00\x00\x00")
    return str(directory)


def test_a_checkpoint_with_mtp_heads_is_refused(tmp_path):
    """Measured 2026-07-28, one Mac, single process, no cluster: Macaron-V1-
    Tall-oQ8e-mtp (42 mtp weights) generated '-in坎t店铺经营者format的...' while
    Ornith-1.0-35B-oQ8e - same mlx-lm module, same quantisation, no mtp -
    answered 'Paris, a city renowned for its rich history'. mlx-lm ignores the
    MTP tensors and mis-loads the rest, and nothing raises: the cluster forms,
    serves at full speed, and returns fluent nonsense.
    """
    path = _write_index(
        tmp_path / "mtp",
        [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.mtp.fc.weight",
            "language_model.mtp.layers.0.input_layernorm.weight",
        ],
    )

    described = inventory.describe(_model(id="Macaron-V1-Tall-oQ8e-mtp", model_path=path))

    assert described["mtp"] is True
    assert described["eligible"] is False
    assert described["complete"] is True
    assert described["shardable"] is True
    assert "multi-token-prediction" in described["reason"]


def test_the_same_model_without_mtp_heads_is_eligible(tmp_path):
    """The control arm. Blocking every checkpoint that merely looks like these
    would take out the large VL MoEs a cluster exists for - Ornith is one, and
    it is correct through mlx-lm."""
    path = _write_index(
        tmp_path / "plain",
        [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "vision_tower.blocks.0.attn.qkv.weight",
        ],
    )

    described = inventory.describe(_model(id="Ornith-1.0-35B-oQ8e", model_path=path))

    assert described["mtp"] is False
    assert described["eligible"] is True
    assert described["reason"] == ""


def test_mtp_is_found_in_a_single_file_model_too(tmp_path):
    """No index means the tensor names live in the safetensors header. Reading
    it costs a seek; not reading it would leave the gate open on exactly the
    checkpoints it exists to catch."""
    path = _write_single_file(tmp_path / "solo", ["model.layers.0.mlp.up_proj.weight", "mtp.fc.weight"])

    assert inventory.carries_mtp_weights(path) is True


def test_a_single_file_model_without_mtp_is_not_flagged(tmp_path):
    path = _write_single_file(tmp_path / "solo-plain", ["model.layers.0.mlp.up_proj.weight"])

    assert inventory.carries_mtp_weights(path) is False


def test_a_missing_directory_is_not_called_mtp(tmp_path):
    """No evidence is not evidence of a problem: a model this node does not
    hold must not be refused as though it were broken."""
    assert inventory.carries_mtp_weights(str(tmp_path / "nowhere")) is False
    assert inventory.carries_mtp_weights("") is False


def test_the_mtp_answer_is_recomputed_when_the_directory_changes(tmp_path):
    """Cached on directory mtime, like the census - a download finishing must
    not leave a stale answer behind."""
    directory = tmp_path / "growing"
    _write_index(directory, ["language_model.model.layers.0.self_attn.q_proj.weight"])
    assert inventory.carries_mtp_weights(str(directory)) is False

    _write_index(directory, ["language_model.mtp.fc.weight"])
    import os

    os.utime(directory, (0, 0))

    assert inventory.carries_mtp_weights(str(directory)) is True


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
