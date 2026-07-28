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
