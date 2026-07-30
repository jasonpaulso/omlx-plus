# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TP capability probe, divisibility guard, and tax math.

These stay mlx-free: the probe is exercised against stub modules and the
divisibility check against plain config dicts.
"""

from __future__ import annotations

import types

import pytest

from omlx.cluster.rank_worker import _tax_summary
from omlx.cluster.tp import (
    TPDivisibilityError,
    check_divisibility,
    missing_weight_files,
    supports_tensor_parallel,
    tensor_parallel_architectures,
)

# -- capability probe against stub modules -----------------------------------


class _ShardableModel:
    def shard(self, group=None):  # noqa: D401 - stub
        return None


class _PlainModel:
    pass


def _stub_modules():
    yield "llama", types.SimpleNamespace(Model=_ShardableModel)
    yield "plain", types.SimpleNamespace(Model=_PlainModel)
    yield "no_model", types.SimpleNamespace()


def test_probe_finds_only_shardable_architectures():
    found = tensor_parallel_architectures(_stub_modules)
    assert found == frozenset({"llama"})


def test_supports_tensor_parallel_on_instance():
    assert supports_tensor_parallel(_ShardableModel()) is True
    assert supports_tensor_parallel(_PlainModel()) is False


# -- divisibility ------------------------------------------------------------


def test_divisibility_accepts_divisible_heads():
    check_divisibility({"num_attention_heads": 32, "num_key_value_heads": 8}, 2)


def test_divisibility_rejects_indivisible_attention_heads():
    with pytest.raises(TPDivisibilityError):
        check_divisibility({"num_attention_heads": 32}, 3)


def test_divisibility_rejects_indivisible_kv_heads():
    with pytest.raises(TPDivisibilityError):
        check_divisibility({"num_attention_heads": 8, "num_key_value_heads": 6}, 4)


def test_divisibility_noop_for_world_size_one():
    # A single-rank world never shards; an odd head count is fine.
    check_divisibility({"num_attention_heads": 7}, 1)


def test_divisibility_noop_when_config_silent():
    check_divisibility({}, 4)


# -- S6 D0: text_config fallback (language_model_only checkpoints) -----------


def test_divisibility_falls_back_to_nested_text_config():
    # No top-level heads (a language_model_only wrapper carries them nested)
    # -- the divisible nested count must still pass, not lenient-pass on
    # top-level absence.
    config = {"text_config": {"num_attention_heads": 24, "num_key_value_heads": 4}}
    check_divisibility(config, 2)


def test_divisibility_rejects_a_nested_indivisible_config():
    # Pre-fix this lenient-passed (top-level heads=None); it must now FAIL.
    config = {"text_config": {"num_attention_heads": 25, "num_key_value_heads": 5}}
    with pytest.raises(TPDivisibilityError):
        check_divisibility(config, 2)


def test_divisibility_prefers_top_level_over_nested():
    config = {
        "num_attention_heads": 8,
        "text_config": {"num_attention_heads": 7},
    }
    # Top-level 8 is divisible by 2; the (differing) nested value must not
    # be consulted once the top-level key is present.
    check_divisibility(config, 2)


def test_divisibility_ignores_a_non_dict_text_config():
    check_divisibility({"text_config": "not-a-dict"}, 4)


# -- D9 tax accumulator math -------------------------------------------------


def test_tax_summary_empty():
    assert _tax_summary([]) == {
        "steps": 0,
        "avg_ms": 0.0,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
    }


def test_tax_summary_statistics():
    summary = _tax_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["steps"] == 4
    assert summary["avg_ms"] == pytest.approx(2.5)
    # Nearest-rank percentiles over the sorted samples.
    assert summary["p50_ms"] == pytest.approx(3.0)
    assert summary["p90_ms"] == pytest.approx(4.0)


# -- weight-file completeness (S5 P3 rig finding) -----------------------------


def _write_index(root, weight_map):
    import json

    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_missing_weight_files_complete_dir_is_empty(tmp_path):
    _write_index(tmp_path, {"a.w": "model-00001-of-00001.safetensors"})
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"0")
    assert missing_weight_files(tmp_path) == []


def test_missing_weight_files_reports_holes_from_the_models_own_index(tmp_path):
    _write_index(
        tmp_path,
        {
            "a.w": "model-00001-of-00002.safetensors",
            "b.w": "model-00002-of-00002.safetensors",
        },
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"0")
    assert missing_weight_files(tmp_path) == ["model-00002-of-00002.safetensors"]


def test_missing_weight_files_no_index_with_single_file_is_empty(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"0")
    assert missing_weight_files(tmp_path) == []


def test_missing_weight_files_config_only_dir_is_incomplete(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert missing_weight_files(tmp_path) == ["model.safetensors"]


def test_missing_weight_files_unreadable_index_defers_to_loader(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{not json")
    assert missing_weight_files(tmp_path) == []
