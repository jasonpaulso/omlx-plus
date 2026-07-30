# SPDX-License-Identifier: Apache-2.0
"""S4 D6: the admin dashboard's cluster write-op proxies -- thin
operator-gated delegations to the same handlers the ``/v1/cluster/*``
routes call, not HTTP self-calls. Mutations are POST-only (CL-12).
"""

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
    running_manager,
)

LOAD = "/admin/api/cluster/models/load"
UNLOAD = "/admin/api/cluster/models/unload"
PLACEMENT = "/admin/api/cluster/placement"


def _wired_pool(tmp_path, *, ceiling: int = 1) -> EnginePool:
    model_dir = tmp_path / "models" / "m"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)

    pool = EnginePool()
    pool._get_final_ceiling = lambda c=ceiling: c
    pool.discover_models(str(tmp_path / "models"))
    set_engine_pool_getter(lambda: pool)
    return pool


async def test_endpoints_require_operator(head_settings):
    async with running_manager(head_settings):
        app = build_app(with_admin=True)
        async with http_client(app) as client:
            assert (await client.post(LOAD, json={"model": "m"})).status_code == 401
            resp = await client.post(LOAD, json={"model": "m"}, headers=bearer(SUB_KEY))
            assert resp.status_code == 401
            assert (await client.get(f"{PLACEMENT}?model=m")).status_code == 401


async def test_endpoints_404_on_worker(worker_settings):
    async with running_manager(worker_settings):
        app = build_app(with_admin=True)
        async with http_client(app) as client:
            resp = await client.post(
                LOAD, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
            )
            assert resp.status_code == 404
            resp = await client.get(
                f"{PLACEMENT}?model=m", headers=bearer(MAIN_API_KEY)
            )
            assert resp.status_code == 404


async def test_load_proxy_delegates_to_load_distributed(head_settings, tmp_path):
    async with running_manager(head_settings):
        # No worker joined -> placement rejects with "no cluster members".
        _wired_pool(tmp_path)
        app = build_app(with_admin=True)
        async with http_client(app) as client:
            resp = await client.post(
                LOAD, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
            )
        assert resp.status_code == 424
        assert "no cluster members" in resp.json()["detail"]
        set_engine_pool_getter(None)


async def test_unload_proxy_delegates_to_unload_distributed(head_settings, tmp_path):
    async with running_manager(head_settings):
        _wired_pool(tmp_path)
        app = build_app(with_admin=True)
        async with http_client(app) as client:
            resp = await client.post(
                UNLOAD, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
            )
        # No active formation -> the pool's own 404.
        assert resp.status_code == 404
        set_engine_pool_getter(None)


async def test_placement_proxy_matches_the_v1_preview(head_settings, tmp_path):
    from omlx.cluster.routes import router as cluster_router

    async with running_manager(head_settings):
        _wired_pool(tmp_path)
        app = build_app(with_admin=True)
        app.include_router(cluster_router)
        async with http_client(app) as client:
            admin_resp = await client.get(
                f"{PLACEMENT}?model=m&prefer=auto", headers=bearer(MAIN_API_KEY)
            )
            v1_resp = await client.get(
                "/v1/cluster/placement?model=m&prefer=auto",
                headers=bearer(MAIN_API_KEY),
            )
        assert admin_resp.status_code == 200
        assert admin_resp.json() == v1_resp.json()
        set_engine_pool_getter(None)


async def test_placement_proxy_unknown_model_is_404(head_settings, tmp_path):
    async with running_manager(head_settings):
        _wired_pool(tmp_path)
        app = build_app(with_admin=True)
        async with http_client(app) as client:
            resp = await client.get(
                f"{PLACEMENT}?model=missing", headers=bearer(MAIN_API_KEY)
            )
        assert resp.status_code == 404
        set_engine_pool_getter(None)


async def test_get_on_load_path_is_not_allowed(head_settings, tmp_path):
    """CL-12: mutations are POST-only."""
    async with running_manager(head_settings):
        _wired_pool(tmp_path)
        app = build_app(with_admin=True)
        async with http_client(app) as client:
            resp = await client.get(LOAD, headers=bearer(MAIN_API_KEY))
        assert resp.status_code == 405
        set_engine_pool_getter(None)
