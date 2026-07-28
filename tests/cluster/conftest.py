# SPDX-License-Identifier: Apache-2.0
"""Fixtures for cluster control-plane tests.

Route tests drive the app through ``httpx.ASGITransport`` rather than
``fastapi.testclient.TestClient``: the manager owns asyncio tasks and
futures created on the test's event loop, and TestClient runs the app on a
separate loop, which would make every queued command cross loops.
"""

from __future__ import annotations

import contextlib
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from omlx.cluster.client import ClusterClient
from omlx.cluster.manager import ClusterManager, set_cluster_manager
from omlx.settings import GlobalSettings

MAIN_API_KEY = "test-main-api-key"
SUB_KEY = "test-sub-api-key"


def make_settings(tmp_path, role: str = "head", **overrides: Any) -> GlobalSettings:
    """Build settings for a node in the given cluster role."""
    from omlx.settings import SubKeyEntry

    settings = GlobalSettings(base_path=tmp_path)
    settings.auth.api_key = MAIN_API_KEY
    settings.auth.sub_keys = [SubKeyEntry(key=SUB_KEY, name="sub")]
    settings.cluster.role = role
    settings.cluster.allow_loopback = True
    settings.cluster.heartbeat_interval_s = 0.05
    settings.cluster.member_timeout_s = 0.2
    for key, value in overrides.items():
        setattr(settings.cluster, key, value)
    return settings


@contextlib.asynccontextmanager
async def running_manager(settings: GlobalSettings, **kwargs: Any):
    """Start a manager, install it as the process-wide one, and clean up."""
    manager = ClusterManager(settings, **kwargs)
    await manager.start()
    set_cluster_manager(manager)
    try:
        yield manager
    finally:
        set_cluster_manager(None)
        await manager.stop()


def build_app(*, with_admin: bool = False) -> FastAPI:
    """Build an app carrying the cluster router (optionally admin too)."""
    from omlx.cluster.routes import router as cluster_router

    app = FastAPI()
    app.include_router(cluster_router)
    if with_admin:
        from omlx.admin.routes import router as admin_router

        app.include_router(admin_router)
    return app


@contextlib.asynccontextmanager
async def http_client(app: FastAPI, peer: tuple[str, int] = ("10.1.2.3", 40404)):
    """An async client whose requests arrive from ``peer``."""
    transport = httpx.ASGITransport(app=app, client=peer)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://cluster.test"
    ) as client:
        yield client


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class FakeClusterClient(ClusterClient):
    """Control-plane client that answers from a canned script.

    Subclasses the real client so URL validation still runs — a test that
    passes a bogus head URL fails the same way production would.
    """

    def __init__(self, base_url: str, replies: dict[str, Any] | None = None) -> None:
        super().__init__(base_url)
        self.replies: dict[str, Any] = replies or {}
        self.calls: list[tuple[str, str, str, dict[str, Any] | None]] = []

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls.append((method, path, token, payload))
        reply = self.replies.get(path)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(payload)
        return reply if isinstance(reply, dict) else {}


@pytest.fixture
def head_settings(tmp_path):
    return make_settings(tmp_path / "head", role="head")


@pytest.fixture
def worker_settings(tmp_path):
    return make_settings(tmp_path / "worker", role="worker")
