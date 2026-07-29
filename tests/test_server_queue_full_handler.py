# SPDX-License-Identifier: Apache-2.0
"""Verify SchedulerQueueFullError maps to HTTP 503 + Retry-After in server.py."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.exceptions import SchedulerQueueFullError


def _build_test_app():
    """Build a minimal FastAPI app that re-uses the same exception handler.

    Importing omlx.server would pull in heavy server-state init. We pluck the
    handler function out of the module and register it against a fresh app
    so the test stays fast and free of state.
    """
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(
        SchedulerQueueFullError, srv.scheduler_queue_full_handler
    )

    @app.get("/v1/raise")
    def raise_queue_full():
        raise SchedulerQueueFullError(current_depth=32, max_depth=32)

    @app.get("/health/raise")
    def raise_queue_full_health():
        raise SchedulerQueueFullError(current_depth=33, max_depth=32)

    return app


class TestQueueFullHandler:
    def test_returns_503(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.status_code == 503

    def test_has_retry_after_header(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.headers.get("Retry-After") == "1"

    def test_api_route_uses_openai_error_body(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        # _openai_error_body wraps in {"error": {...}}
        assert "error" in body
        assert "queue full" in body["error"]["message"].lower()
        # Depth numbers surface to the client
        assert "32/32" in body["error"]["message"]

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/health/raise")
        body = resp.json()
        assert "detail" in body
        assert "queue full" in body["detail"].lower()


class TestStreamingChatReaches503:
    """The handler above only proves the response body. This proves the
    wiring that made row 4 of the S3 acceptance matrix fail on the live rig.

    ``Scheduler.add_request`` raises the queue-full error from inside the
    route's response generator (``batched.py:817``), and starlette emits
    ``http.response.start`` with status 200 before it ever iterates that
    generator — so on ``stream: true`` the rejection could only ever reach
    the client as a truncated/in-stream error, never as the 503 the handler
    exists to send. The fix moves the check into the preflight seam the route
    already awaits before committing to the ``StreamingResponse``; this test
    asserts the status code a retrying client actually branches on.
    """

    def _run(self, *, stream: bool):
        from unittest.mock import AsyncMock, MagicMock, patch

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

        fake_pool = MagicMock()
        fake_pool.get_entry = MagicMock(return_value=None)
        fake_pool.preload_pinned_models = AsyncMock()
        fake_pool.check_ttl_expirations = AsyncMock()
        fake_pool.shutdown = AsyncMock()

        original_get_engine = srv.get_engine_for_model
        original_overrides = dict(srv.app.dependency_overrides)
        original_engine_pool = srv._server_state.engine_pool
        original_settings_manager = srv._server_state.settings_manager
        try:
            srv.app.dependency_overrides[srv.verify_api_key] = lambda: True
            srv.get_engine_for_model = _get_engine_for_model  # type: ignore[assignment]
            srv._server_state.engine_pool = fake_pool
            # Pinned so this test does not depend on what an earlier test
            # left on the shared server state. (It used to matter more: the
            # route's SpecPrefill fallbacks dereferenced ``ms`` behind a
            # ``settings_manager`` check, so an inherited manager turned this
            # into a 500. That guard is fixed — see
            # test_server_model_settings_guard.py — but pinning both sides of
            # the pair still keeps the run deterministic.)
            srv._server_state.settings_manager = None
            with (
                TestClient(srv.app, raise_server_exceptions=False) as client,
                patch.object(srv, "resolve_model_id", lambda name: name),
                patch.object(srv, "validate_context_window", lambda *a, **k: None),
                patch.object(
                    srv, "get_model_settings_for_request", lambda *a, **k: None
                ),
            ):
                return client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": stream,
                        "max_tokens": 8,
                    },
                )
        finally:
            srv.get_engine_for_model = original_get_engine
            srv._server_state.engine_pool = original_engine_pool
            srv._server_state.settings_manager = original_settings_manager
            srv.app.dependency_overrides.clear()
            srv.app.dependency_overrides.update(original_overrides)

    def test_streaming_request_gets_503_not_a_200_sse_error(self):
        resp = self._run(stream=True)
        assert (
            resp.status_code == 503
        ), f"expected 503, got {resp.status_code}: {resp.text[:400]}"
        assert resp.headers.get("Retry-After") == "1"
        assert "queue full" in resp.json()["error"]["message"].lower()

    def test_non_streaming_request_still_gets_503(self):
        resp = self._run(stream=False)
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"
