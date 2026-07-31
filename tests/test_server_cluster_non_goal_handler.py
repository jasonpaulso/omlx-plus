# SPDX-License-Identifier: Apache-2.0
"""Verify ClusterNonGoalError maps to HTTP 400 in server.py (S6 P1c item 6).

Before this fix there was no registered handler for `ClusterNonGoalError`
(a `ValueError` subclass): a vision request against a cluster-formed model
-- including one distributed TEXT-ONLY under
`cluster.allow_text_only_distribution` -- fell through to the catch-all
`Exception` handler and surfaced as an unhandled 500. Never a silent
degrade, but also never the clear, named error the request deserves.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster.engine import ClusterNonGoalError

_MESSAGE = (
    "VLM requests are not supported on a distributed cluster instance (S3 "
    "non-goal); serve this model single-node."
)


def _build_test_app():
    """Same minimal-app pattern as test_server_queue_full_handler.py:
    importing omlx.server pulls in heavy server-state init, so pluck the
    handler out and register it fresh against an empty app."""
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(ClusterNonGoalError, srv.cluster_non_goal_error_handler)

    @app.get("/v1/raise")
    def raise_non_goal():
        raise ClusterNonGoalError(_MESSAGE)

    @app.get("/health/raise")
    def raise_non_goal_health():
        raise ClusterNonGoalError(_MESSAGE)

    return app


class TestClusterNonGoalHandler:
    def test_returns_400_not_500(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.status_code == 400

    def test_api_route_uses_openai_error_body(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        assert "error" in body
        assert "distributed cluster" in body["error"]["message"].lower()

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/health/raise")
        body = resp.json()
        assert "detail" in body


class TestStreamingChatVisionRequestReaches400:
    """The handler above only proves the response body shape. This proves
    the wiring end to end, the same way test_server_queue_full_handler.py's
    `TestStreamingChatReaches503` proves its own handler is actually wired
    where a real request hits it: a vision-bearing chat request against a
    cluster-formed model must reach the client as a clean 400, never a
    truncated in-stream error and never an unhandled 500.

    `ClusterEngine.preflight_chat` (S3, unconditional for every
    cluster-formed model, `_reject_if_multimodal`) is what actually raises
    `ClusterNonGoalError` for a real image-bearing request -- this fakes
    that exact outcome at the engine boundary rather than re-driving a real
    formation, per the fake-performs-peer's-role guidance: assert the
    OUTBOUND HTTP surface, not any internal state.
    """

    def _run(self, *, stream: bool):
        from unittest.mock import AsyncMock, MagicMock, patch

        import omlx.server as srv

        async def _raising_preflight(*args, **kwargs):
            raise ClusterNonGoalError(_MESSAGE)

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
                        "model": "qwen-distributed-text-only",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "what is this?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "data:image/png;base64,x"},
                                    },
                                ],
                            }
                        ],
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

    def test_streaming_vision_request_gets_400_not_500(self):
        resp = self._run(stream=True)
        assert (
            resp.status_code == 400
        ), f"expected 400, got {resp.status_code}: {resp.text[:400]}"
        assert "distributed cluster" in resp.json()["error"]["message"].lower()

    def test_non_streaming_vision_request_gets_400_not_500(self):
        resp = self._run(stream=False)
        assert resp.status_code == 400
        assert "distributed cluster" in resp.json()["error"]["message"].lower()
