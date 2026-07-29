# SPDX-License-Identifier: Apache-2.0
"""S3 D6(b): the rank-side SchedulerConfig helper always forces cache off."""

from __future__ import annotations

from omlx.cluster.scheduler_config import build_rank_scheduler_config


def test_paged_ssd_cache_forced_off_by_default():
    config = build_rank_scheduler_config()
    assert config.paged_ssd_cache_dir is None


def test_paged_ssd_cache_forced_off_despite_configured_dir():
    """Even if a caller hands in the daemon's own configured cache dir, the
    helper strips it -- rank processes must never initialize the paged-SSD
    stack (D6(b))."""
    config = build_rank_scheduler_config(
        paged_ssd_cache_dir="/configured/cache/dir",
        hot_cache_max_size=1024,
        hot_cache_only=True,
    )
    assert config.paged_ssd_cache_dir is None
    assert config.hot_cache_only is False
    assert config.hot_cache_max_size == 0
    assert config.hot_cache_budget is None


def test_max_num_seqs_defaults_to_app_wide_default():
    """8 matches omlx.config.SchedulerConfig's app-wide default (the value
    every single-node engine gets via settings.py), so the plan's queue-full
    test recipe (cap = max(max_num_seqs * 4, 32) = 32) applies unchanged --
    no new cluster-specific config knob."""
    config = build_rank_scheduler_config()
    assert config.max_num_seqs == 8


def test_max_num_seqs_override_is_respected():
    config = build_rank_scheduler_config(max_num_seqs=16)
    assert config.max_num_seqs == 16


def test_other_overrides_pass_through():
    config = build_rank_scheduler_config(model_name="mlx-community/x")
    assert config.model_name == "mlx-community/x"
