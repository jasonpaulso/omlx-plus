# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/cluster/placement`` — the S4 D3 dry-run preview endpoint."""

from __future__ import annotations

import json

import pytest

from omlx.cluster.manager import set_engine_pool_getter
from omlx.engine_pool import EnginePool

from .conftest import (
    MAIN_API_KEY,
    SUB_KEY,
    bearer,
    build_app,
    http_client,
    make_settings,
    running_manager,
)

PLACEMENT = "/v1/cluster/placement"


@pytest.fixture
def small_model_dir(tmp_path):
    model_dir = tmp_path / "models" / "model-a"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path / "models"


@pytest.fixture
def pool_getter(small_model_dir):
    """Install a real EnginePool (with `head_capacity()` reachable) as the
    injected getter, and clean it up so tests never leak process-wide state."""
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 10 * 1024**3
    pool.discover_models(str(small_model_dir))
    set_engine_pool_getter(lambda: pool)
    try:
        yield pool
    finally:
        set_engine_pool_getter(None)


async def test_requires_operator(tmp_path, pool_getter):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            assert (
                await client.get(PLACEMENT, params={"model": "model-a"})
            ).status_code == 401
            resp = await client.get(
                PLACEMENT, params={"model": "model-a"}, headers=bearer(SUB_KEY)
            )
            assert resp.status_code == 401


async def test_404s_on_worker(tmp_path, pool_getter):
    settings = make_settings(tmp_path / "w", role="worker")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "model-a"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 404


async def test_unknown_model_is_404(tmp_path, pool_getter):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "does-not-exist"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 404


async def test_no_pool_getter_installed_is_503(tmp_path):
    set_engine_pool_getter(None)
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "model-a"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 503


async def test_preview_exercises_head_capacity_via_the_accessor(tmp_path, pool_getter):
    """A locally-fitting small model previews as local, sourcing the head's
    side of the decision from `EnginePool.head_capacity()` (D2b/D3)."""
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "model-a"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["mode"] == "local"
            assert body["world_size"] == 1
            assert body["fits"]["head"]["ceiling"] == 10 * 1024**3
            assert body["divisible"] is True


async def test_prefer_query_param_is_honored(tmp_path, pool_getter):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "model-a", "prefer": "distributed"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 200
            body = resp.json()
            # No members joined: known-eligible model, no capacity data at
            # all -> capacity-unknown rule rejects under prefer=distributed.
            assert body["mode"] == "reject"


async def test_zero_side_effects(tmp_path, pool_getter):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings) as manager:
        before = manager.state
        app = build_app()
        async with http_client(app) as client:
            await client.get(
                PLACEMENT,
                params={"model": "model-a"},
                headers=bearer(MAIN_API_KEY),
            )
        assert manager.state == before
        assert pool_getter.get_entry("model-a").engine is None


@pytest.fixture
def vlm_model_dir(tmp_path):
    """A 'vlm'-classified checkpoint (carries vision_config) with divisible
    text_config heads -- the text-only-distribution opt-in domain."""
    model_dir = tmp_path / "models" / "vlm-a"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "vision_config": {"depth": 1},
                "text_config": {
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                },
            }
        )
    )
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)
    return tmp_path / "models"


@pytest.fixture
def vlm_pool_getter(vlm_model_dir):
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 10 * 1024**3
    pool.discover_models(str(vlm_model_dir))
    set_engine_pool_getter(lambda: pool)
    try:
        yield pool
    finally:
        set_engine_pool_getter(None)


async def test_preview_honors_text_only_opt_in(tmp_path, vlm_pool_getter):
    """Preview must pass cluster.allow_text_only_distribution to
    plan_placement like the real load paths do (engine_pool.py). Pre-fix the
    preview hardcoded the default (False) and reported
    "model_type='vlm' is not eligible for distributed placement" for a model
    the load path would text-only-distribute -- breaking the S4
    preview==decision invariant on the opt-in domain (observed live
    2026-07-31 against the formed rig)."""
    settings = make_settings(
        tmp_path / "h", role="head", allow_text_only_distribution=True
    )
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "vlm-a", "prefer": "distributed"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert not any(
                "not eligible" in reason for reason in body.get("reasons", [])
            ), body


async def test_preview_default_off_still_reports_ineligible(tmp_path, vlm_pool_getter):
    """Polarity guard for the test above: with the opt-in at its default
    (False) the same preview must still name the vlm ineligibility."""
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(
                PLACEMENT,
                params={"model": "vlm-a", "prefer": "distributed"},
                headers=bearer(MAIN_API_KEY),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert any(
                "not eligible" in reason for reason in body.get("reasons", [])
            ), body
