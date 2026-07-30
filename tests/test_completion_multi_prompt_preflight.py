# SPDX-License-Identifier: Apache-2.0
"""Tests for the multi-prompt /v1/completions preflight loop in
``create_completion`` (task #20).

The route claims one queue reservation per prompt
(``for prompt in prompts: await engine.preflight_completion(...)``). Two
leak shapes:

- A mid-loop preflight failure (e.g. ``PrefillMemoryExceededError`` on
  prompt k) leaked the k-1 already-claimed reservations for the earlier
  prompts until the 30s TTL swept them.
- ``stream=true`` with more than one prompt generates ONLY ``prompts[0]``
  (see the ``StreamingResponse`` branch), so ``prompts[1:]``'s reservations
  ALWAYS leaked -- nothing downstream ever consumes them.

These tests drive ``create_completion`` directly with a mocked engine,
following ``tests/test_disconnect_guard.py``'s pattern of calling the
server-module function under test rather than going through a live
FastAPI/TestClient stack.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import omlx.server as server
from omlx.api.openai_models import CompletionRequest
from omlx.exceptions import PrefillMemoryExceededError


def _make_mock_engine(num_prompts_before_raise: int | None = None):
    """A mock LLM engine with a tokenizer and a ``preflight_completion``
    that optionally raises on the Nth call (1-indexed), simulating a
    mid-loop ``PrefillMemoryExceededError``.
    """
    engine = MagicMock()
    engine.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    engine._release_queue_reservation = MagicMock()

    if num_prompts_before_raise is None:
        engine.preflight_completion = AsyncMock(return_value=None)
    else:
        call_count = {"n": 0}

        async def _preflight(prompt, request_id=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == num_prompts_before_raise:
                raise PrefillMemoryExceededError(
                    message="too big for the ceiling",
                    request_id=request_id,
                    estimated_bytes=1,
                    limit_bytes=1,
                )

        engine.preflight_completion = AsyncMock(side_effect=_preflight)

    return engine


def _make_http_request():
    request = MagicMock()
    request.headers.get = MagicMock(return_value=None)
    return request


@pytest.mark.asyncio
class TestMultiPromptPreflightReservationLeak:
    async def test_mid_loop_failure_releases_prior_claims(self, monkeypatch):
        """4 prompts, preflight fails on the 3rd: the 2 already-claimed
        reservations for prompts 1-2 must be released before re-raising.
        """
        engine = _make_mock_engine(num_prompts_before_raise=3)
        monkeypatch.setattr(
            server, "get_engine_for_model", AsyncMock(return_value=engine)
        )

        request = CompletionRequest(
            model="test-model",
            prompt=["p1", "p2", "p3", "p4"],
            stream=False,
        )

        with pytest.raises(PrefillMemoryExceededError):
            await server.create_completion(request, _make_http_request(), True)

        assert engine.preflight_completion.call_count == 3
        assert engine._release_queue_reservation.call_count == 2

    async def test_stream_true_preflights_only_first_prompt(self, monkeypatch):
        """3 prompts, stream=true: the route only ever generates
        prompts[0] (StreamingResponse branch), so only prompts[0] should be
        preflighted -- claiming reservations for prompts[1:] would leak
        them forever since nothing downstream consumes those prompts.
        """
        engine = _make_mock_engine()
        monkeypatch.setattr(
            server, "get_engine_for_model", AsyncMock(return_value=engine)
        )

        request = CompletionRequest(
            model="test-model",
            prompt=["p1", "p2", "p3"],
            stream=True,
        )

        response = await server.create_completion(
            request, _make_http_request(), True
        )

        assert response is not None
        assert engine.preflight_completion.call_count == 1
