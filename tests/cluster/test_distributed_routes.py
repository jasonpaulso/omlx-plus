# SPDX-License-Identifier: Apache-2.0
"""The distributed model endpoints: operator tier, head role (D8)."""

from __future__ import annotations

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
    settings = make_settings(
        tmp_path / "h",
        role="head",
        data_plane_subnet="10.0.2.0/24",
        data_plane_address="10.0.2.1",
    )
    async with running_manager(settings):
        app = build_app()
        async with http_client(app) as client:
            resp = await client.post(
                LOAD, json={"model": "m"}, headers=bearer(MAIN_API_KEY)
            )
            assert resp.status_code == 424
            assert "worker" in resp.json()["detail"]


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
