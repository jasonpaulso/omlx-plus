# SPDX-License-Identifier: Apache-2.0
"""The E1 engine_pool touchpoint is flag-gated: it resolves a distributed
engine only on an active head with a live formation, and is otherwise inert —
so ``cluster.role=off`` takes the identical path it always did.
"""

from __future__ import annotations

from omlx.cluster.manager import set_cluster_manager
from omlx.engine_pool import _resolve_cluster_engine

from .conftest import make_settings, running_manager


def test_no_cluster_manager_resolves_none():
    set_cluster_manager(None)
    assert _resolve_cluster_engine("any-model") is None


async def test_worker_role_resolves_none(tmp_path):
    settings = make_settings(tmp_path / "w", role="worker")
    async with running_manager(settings):
        assert _resolve_cluster_engine("any-model") is None


async def test_head_without_active_engine_resolves_none(tmp_path):
    settings = make_settings(tmp_path / "h", role="head")
    async with running_manager(settings):
        # A head with a formation manager but no active formation returns None,
        # so get_engine falls through to the normal local-load path.
        assert _resolve_cluster_engine("any-model") is None


async def test_head_with_active_engine_resolves_it(tmp_path):
    settings = make_settings(tmp_path / "h", role="head")
    sentinel = object()

    class _FakeFormation:
        def active_engine(self, model_id):
            return sentinel if model_id == "formed" else None

        async def stop(self):
            return None

    async with running_manager(settings) as manager:
        manager._formation = _FakeFormation()
        assert _resolve_cluster_engine("formed") is sentinel
        assert _resolve_cluster_engine("other") is None
