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


class TestRouteGuardSeesRawParts:
    """The class above fakes `preflight_chat` raising — it proves handler
    wiring, not the path. Live rig probe 2026-07-31 refuted the path: the
    non-VLM conversions strip image parts BEFORE preflight, so a vision
    request against a text-only-distributed model was silently served as
    text (HTTP 200, image dropped). These tests drive the REAL routes with
    an engine that `isinstance`-passes ClusterEngine and an INERT preflight:
    only the route-level `_reject_cluster_multimodal` guard can produce the
    400, so each test fails on the pre-fix source.
    """

    def _cluster_engine_fake(self):
        from unittest.mock import AsyncMock, MagicMock

        from omlx.cluster.engine import ClusterEngine

        # Subclass so plain class attributes shadow ClusterEngine's read-only
        # properties; __new__ skips the real __init__ (no formation needed).
        # isinstance(engine, ClusterEngine) — what the route guard checks —
        # still holds.
        fake_cls = type(
            "FakeClusterEngine",
            (ClusterEngine,),
            {"model_type": "llama", "tokenizer": MagicMock()},
        )
        engine = fake_cls.__new__(fake_cls)
        engine.preflight_chat = AsyncMock(return_value=None)  # inert on purpose
        engine.preflight_completion = AsyncMock(return_value=None)
        engine.start = AsyncMock()
        engine.count_chat_tokens = MagicMock(return_value=16)
        return engine

    def _post(self, path: str, payload: dict):
        from unittest.mock import AsyncMock, MagicMock, patch

        import omlx.server as srv

        engine = self._cluster_engine_fake()

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
                return client.post(path, json=payload)
        finally:
            srv.get_engine_for_model = original_get_engine
            srv._server_state.engine_pool = original_engine_pool
            srv._server_state.settings_manager = original_settings_manager
            srv.app.dependency_overrides.clear()
            srv.app.dependency_overrides.update(original_overrides)

    def test_chat_completions_image_part_400s_via_route_guard(self):
        resp = self._post(
            "/v1/chat/completions",
            {
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
                "max_tokens": 8,
            },
        )
        assert (
            resp.status_code == 400
        ), f"expected 400, got {resp.status_code}: {resp.text[:400]}"
        assert "distributed cluster" in resp.json()["error"]["message"].lower()

    def test_chat_completions_text_only_does_not_trip_guard(self):
        resp = self._post(
            "/v1/chat/completions",
            {
                "model": "qwen-distributed-text-only",
                "messages": [{"role": "user", "content": "plain text"}],
                "max_tokens": 8,
            },
        )
        if resp.status_code == 400:
            assert (
                "distributed cluster" not in resp.json()["error"]["message"].lower()
            ), f"text-only request tripped the multimodal guard: {resp.text[:400]}"

    def test_anthropic_messages_image_block_400s_via_route_guard(self):
        resp = self._post(
            "/v1/messages",
            {
                "model": "qwen-distributed-text-only",
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "x",
                                },
                            },
                        ],
                    }
                ],
            },
        )
        assert (
            resp.status_code == 400
        ), f"expected 400, got {resp.status_code}: {resp.text[:400]}"
        assert "distributed cluster" in resp.text.lower()

    def test_responses_input_image_400s_via_route_guard(self):
        resp = self._post(
            "/v1/responses",
            {
                "model": "qwen-distributed-text-only",
                "max_output_tokens": 16,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "what is this?"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,x",
                            },
                        ],
                    }
                ],
            },
        )
        assert (
            resp.status_code == 400
        ), f"expected 400, got {resp.status_code}: {resp.text[:400]}"
        assert "distributed cluster" in resp.text.lower()


class TestRejectClusterMultimodalHelper:
    def test_non_cluster_engine_is_ignored(self):
        from unittest.mock import MagicMock

        import omlx.server as srv

        srv._reject_cluster_multimodal(
            MagicMock(),
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        )  # no raise

    def test_cluster_engine_with_text_only_content_passes(self):
        import omlx.server as srv
        from omlx.cluster.engine import ClusterEngine

        engine = ClusterEngine.__new__(ClusterEngine)
        srv._reject_cluster_multimodal(
            engine,
            [
                {"role": "user", "content": "plain"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        )  # no raise
