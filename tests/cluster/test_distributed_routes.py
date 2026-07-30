# SPDX-License-Identifier: Apache-2.0
"""The distributed model endpoints: operator tier, head role (D8)."""

from __future__ import annotations

import json

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

LOAD = "/v1/cluster/models/load"
UNLOAD = "/v1/cluster/models/unload"
STATUS = "/v1/cluster/models/status"


async def test_endpoints_require_operator(tmp_path):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            # No credential -> 401.
            assert (await client.post(LOAD, json={"model": "m"})).status_code == 401
            # A sub-key is an inference credential, never operator.
            resp = await client.post(LOAD, json={"model": "m"}, headers=bearer(SUB_KEY))
            assert resp.status_code == 401


async def test_endpoints_404_on_worker(tmp_path):
    settings = make_settings(tmp_path / "w", role="worker")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            for path in (LOAD, UNLOAD):
                resp = await client.post(
                    path, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
                )
                assert resp.status_code == 404
            assert (
                await client.get(STATUS, headers=bearer(MAIN_API_KEY))
            ).status_code == 404


async def test_load_without_a_worker_is_424(tmp_path):
    """S4 D4: /v1/cluster/models/load now routes through
    ``pool.load_cluster_model``, so the pool must be wired (production
    always injects it via ``init_server``). With no active worker, the
    reject now comes from ``plan_placement`` itself (no cluster members
    available) rather than formation's own worker check -- same 424
    status, a placement reason attached instead of formation's raw
    message (S4 D3's documented semantics change for this route).
    """
    settings = make_settings(
        tmp_path / "h",
        role="head",
        data_plane_subnet="10.0.2.0/24",
        data_plane_address="10.0.2.1",
    )
    model_dir = tmp_path / "models" / "m"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)

    pool = EnginePool()
    pool._get_final_ceiling = lambda: 1  # too small to fit locally
    pool.discover_models(str(tmp_path / "models"))
    set_engine_pool_getter(lambda: pool)
    try:
        async with running_manager(settings):
            app = build_app()
            async with http_client(app) as client:
                resp = await client.post(
                    LOAD, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
                )
                assert resp.status_code == 424
                assert "no cluster members" in resp.json()["detail"]
    finally:
        set_engine_pool_getter(None)


async def test_load_maps_model_loading_error_to_409_not_500(tmp_path):
    """S6 rider: the S5 rig observed a bare, unhandled `ModelLoadingError`
    (already-being-loaded) reach the client as a 500. It must map to the
    same 409 the local /v1/models/load surface uses for the same condition.
    """
    settings = make_settings(
        tmp_path / "h",
        role="head",
        data_plane_subnet="10.0.2.0/24",
        data_plane_address="10.0.2.1",
    )
    model_dir = tmp_path / "models" / "m"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)

    pool = EnginePool()
    pool.discover_models(str(tmp_path / "models"))
    pool.get_entry("m").is_loading = True  # a concurrent load already claimed it
    set_engine_pool_getter(lambda: pool)
    try:
        async with running_manager(settings):
            app = build_app()
            async with http_client(app) as client:
                resp = await client.post(
                    LOAD, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
                )
                assert resp.status_code == 409
                assert "already being loaded" in resp.json()["detail"]
    finally:
        set_engine_pool_getter(None)


async def test_status_reports_formation(tmp_path):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(STATUS, headers=bearer(MAIN_API_KEY))
            assert resp.status_code == 200
            body = resp.json()
            assert body["active_model"] is None
            assert body["jobs"] == []


async def test_status_carries_a_sibling_transfer_jobs_field(tmp_path):
    """S5 P2: `/v1/cluster/models/status` surfaces TransferJob rows (summarized
    -- no manifest/have payload) as a sibling of formation's own `jobs`."""
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.get(STATUS, headers=bearer(MAIN_API_KEY))
            assert resp.status_code == 200
            body = resp.json()
            assert body["transfer"] == {"jobs": []}


async def test_load_body_accepts_an_explicit_source(tmp_path):
    """S5 D6: `source` on the load body reaches `load_distributed` -- with
    no active worker the call still 424s (same as the sourceless case), but
    the body must be ACCEPTED (not a 422 validation error) and the reject
    reason must be placement's, not a `source`-shaped complaint.
    """
    settings = make_settings(
        tmp_path / "h",
        role="head",
        data_plane_subnet="10.0.2.0/24",
        data_plane_address="10.0.2.1",
    )
    model_dir = tmp_path / "models" / "m"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)

    pool = EnginePool()
    pool._get_final_ceiling = lambda: 1
    pool.discover_models(str(tmp_path / "models"))
    set_engine_pool_getter(lambda: pool)
    try:
        async with running_manager(settings):
            app = build_app()
            async with http_client(app) as client:
                resp = await client.post(
                    LOAD,
                    json={"model": "m", "source": "peer"},
                    headers=bearer(MAIN_API_KEY),
                )
                assert resp.status_code == 424
                assert "no cluster members" in resp.json()["detail"]

                bad = await client.post(
                    LOAD,
                    json={"model": "m", "source": "not-a-real-source"},
                    headers=bearer(MAIN_API_KEY),
                )
                assert bad.status_code == 422
    finally:
        set_engine_pool_getter(None)
