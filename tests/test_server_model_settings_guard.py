# SPDX-License-Identifier: Apache-2.0
"""The SpecPrefill fallbacks in ``/v1/chat/completions`` must guard on ``ms``.

``server.py`` read per-model settings into ``ms`` and then wrote::

    elif _server_state.settings_manager and ms.specprefill_keep_pct is not None:

which tests the wrong object: ``ms`` is what gets dereferenced, and
``get_model_settings_for_request`` returns ``None`` for a falsy model id or a
manager that has none to give. With a manager installed and ``ms`` ``None``
that is an ``AttributeError`` mid-route, i.e. a 500 with a traceback.

It was masked rather than safe. ``ModelSettingsManager.get_settings`` returns a
default ``ModelSettings()`` instead of ``None`` when a model has no entry, and
an empty model id is rejected by engine resolution well before this line — so
no production request is known to reach it. Both are invariants of code
elsewhere, though, and neither is what the guard claims to check. Eight lines
above, the same ``ms`` is correctly guarded with ``if ms:``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from omlx.exceptions import SchedulerQueueFullError


def _post_chat_with_no_model_settings():
    """Drive the real route with a settings manager installed but ``ms`` None.

    ``preflight_chat`` raises a typed error the route already maps, so the
    status code tells us where control got to: 503 means the SpecPrefill block
    was crossed intact, 500 means it raised on the way through.
    """
    import omlx.server as srv

    async def _raising_preflight(*args, **kwargs):
        raise SchedulerQueueFullError(current_depth=32, max_depth=32)

    engine = MagicMock()
    engine.preflight_chat = AsyncMock(side_effect=_raising_preflight)
    engine.start = AsyncMock()
    engine.count_chat_tokens = MagicMock(return_value=16)
    engine.model_type = "llama"

    async def _get_engine_for_model(model_id, *, lease=None):
        return engine

    pool = MagicMock()
    pool.get_entry = MagicMock(return_value=None)
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()

    original_get_engine = srv.get_engine_for_model
    original_overrides = dict(srv.app.dependency_overrides)
    original_pool = srv._server_state.engine_pool
    original_settings_manager = srv._server_state.settings_manager
    try:
        srv.app.dependency_overrides[srv.verify_api_key] = lambda: True
        srv.get_engine_for_model = _get_engine_for_model  # type: ignore[assignment]
        srv._server_state.engine_pool = pool
        # The precondition: a manager IS installed (so the old guard's clause
        # is truthy) while the model resolves to no settings.
        srv._server_state.settings_manager = MagicMock()
        with (
            TestClient(srv.app, raise_server_exceptions=False) as client,
            patch.object(srv, "resolve_model_id", lambda name: name),
            patch.object(srv, "validate_context_window", lambda *a, **k: None),
            patch.object(srv, "get_model_settings_for_request", lambda *a, **k: None),
        ):
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "max_tokens": 8,
                },
            )
    finally:
        srv.get_engine_for_model = original_get_engine
        srv._server_state.engine_pool = original_pool
        srv._server_state.settings_manager = original_settings_manager
        srv.app.dependency_overrides.clear()
        srv.app.dependency_overrides.update(original_overrides)


class TestSpecPrefillFallbackGuardsOnMs:
    def test_absent_model_settings_do_not_500(self):
        resp = _post_chat_with_no_model_settings()
        assert (
            resp.status_code != 500
        ), f"route raised on the way through: {resp.text[:400]}"

    def test_route_reaches_its_own_error_mapping(self):
        resp = _post_chat_with_no_model_settings()
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"
