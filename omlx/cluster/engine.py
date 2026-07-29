# SPDX-License-Identifier: Apache-2.0
"""The head-daemon engine for a tensor-parallel formation (D6).

``ClusterEngine`` is a :class:`BatchedEngine` whose weights do not live in this
process: the model is sharded across the rank child processes, and this engine
only holds the tokenizer (to apply the chat template and encode prompts) and the
pipe to rank 0. Chat templating, tool conversion and the rest of the request
shaping stay in the inherited :class:`BatchedEngine` layer; the only things this
subclass replaces are the four generation entry points, which route a templated
token stream through the rank-0 pipe instead of a local scheduler.

Single-request rule (S2): one active generation at a time, FIFO — the rank loop
serves one request, and S3 replaces this with real scheduler integration. There
is no ``RequestOutputCollector`` and no ``EngineCore``; this engine *is* the
serving layer for a distributed model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

from ..engine.base import GenerationOutput
from ..engine.batched import BatchedEngine
from .launcher import LocalCluster
from .protocol import GenerationSpec

logger = logging.getLogger(__name__)

# Idle timeout for a single reply frame from rank 0 (salvage generate-idle
# value): a rank that dies mid-collective leaves rank 0 blocked, and this is
# what turns that into a clean error instead of a hang.
GENERATE_IDLE_TIMEOUT_S = 600.0


def _final_trim(text: str, stops: list[str]) -> str:
    """Strip one trailing stop string from the terminal text (D6 final trim)."""
    for stop in stops:
        if stop and text.endswith(stop):
            return text[: -len(stop)]
    return text


class ClusterEngine(BatchedEngine):
    """A :class:`BatchedEngine` backed by a distributed rank formation."""

    def __init__(
        self,
        model_name: str,
        *,
        cluster: LocalCluster,
        resolved_path: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(model_name, trust_remote_code=trust_remote_code)
        self._cluster = cluster
        self._resolved_path = resolved_path or model_name
        self._model_type_value: str | None = None
        self._request_lock = asyncio.Lock()
        # Last request's D9 coordination-tax summary, from rank 0's done frame.
        self._last_tax: dict[str, Any] | None = None

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Load only the tokenizer; the weights live in the rank processes."""
        if self._loaded:
            return
        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()
        self._tokenizer, self._model_type_value = await loop.run_in_executor(
            get_mlx_executor(), self._load_tokenizer_and_type
        )
        self._loaded = True

    def _load_tokenizer_and_type(self) -> tuple[Any, str | None]:
        from mlx_lm.utils import _download, load_tokenizer

        from ..utils.tokenizer import get_tokenizer_config

        resolved = _download(self._resolved_path)
        tokenizer_config = get_tokenizer_config(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        tokenizer = load_tokenizer(resolved, tokenizer_config)
        model_type: str | None = None
        try:
            with open(Path(resolved) / "config.json", encoding="utf-8") as handle:
                raw = json.load(handle)
            candidate = raw.get("model_type")
            model_type = candidate if isinstance(candidate, str) else None
        except (OSError, ValueError):
            model_type = None
        return tokenizer, model_type

    async def stop(self) -> None:
        """Tear the formation down. Idempotent — the formation manager may also
        stop the same cluster; ``LocalCluster.stop`` is safe to call twice.
        """
        cluster = self._cluster
        if cluster is not None:
            await asyncio.get_running_loop().run_in_executor(None, cluster.stop)
        self._loaded = False

    @property
    def model_type(self) -> str | None:
        return self._model_type_value

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "cluster-distributed",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "world_size": self._cluster.world_size,
            # The backend the formation actually formed on (D9 negotiated
            # backend), and the last request's coordination-tax summary.
            "negotiated_backend": self._cluster.backend,
            "last_tax": self._last_tax,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        # Prefix/SSD cache is not wired for the distributed path (envelope
        # seam: always-miss).
        return None

    def has_active_requests(self) -> bool:
        return self._request_lock.locked()

    async def preflight_chat(self, *args: Any, **kwargs: Any) -> None:
        # No local scheduler and no prefill memory guard on the distributed
        # path — the base no-op is the correct behaviour here.
        return None

    async def preflight_completion(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ---- generation ------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        final: GenerationOutput | None = None
        async for output in self.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        ):
            final = output
        if final is None:
            return GenerationOutput(text="", finish_reason="stop")
        return GenerationOutput(
            text=final.text,
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
            finish_reason=final.finish_reason,
        )

    async def stream_generate(  # type: ignore[override]
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        request_id = str(kwargs.get("request_id") or uuid.uuid4())
        spec = self._build_spec(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            request_id=request_id,
            kwargs=kwargs,
        )
        if not spec.prompt_ids:
            yield GenerationOutput(text="", finished=True, finish_reason="stop")
            return

        payload = {"op": "generate", "spec": spec.to_dict()}
        prompt_tokens = len(spec.prompt_ids)

        # FIFO: one active generation through the rank loop at a time (S2).
        async with self._request_lock:
            frames = self._pump(payload)
            text = ""
            completion = 0
            finished_normally = False
            try:
                async for frame in frames:
                    if not frame.get("ok", False):
                        raise RuntimeError(frame.get("error") or "rank error")
                    if frame.get("done"):
                        finished_normally = True
                        self._last_tax = frame.get("tax")
                        # Stop detection ran in the rank loop; a final trim in
                        # the engine (D6) strips any trailing stop string left
                        # on the terminal text — a no-op when the rank already
                        # removed it.
                        final_text = _final_trim(frame.get("text", text), spec.stop)
                        yield GenerationOutput(
                            text=final_text,
                            new_text="",
                            prompt_tokens=int(
                                frame.get("prompt_tokens", prompt_tokens)
                            ),
                            completion_tokens=int(
                                frame.get("completion_tokens", completion)
                            ),
                            finished=True,
                            finish_reason=frame.get("finish_reason", "stop"),
                        )
                        break
                    chunk = frame.get("chunk", "")
                    text += chunk
                    completion = int(frame.get("tokens", completion))
                    yield GenerationOutput(
                        text=text,
                        new_text=chunk,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion,
                        finished=False,
                        finish_reason=None,
                    )
            except GeneratorExit:
                # Client disconnected: forward the abort over the rank-0 control
                # pipe (it lands as a delta at the next step) and re-raise.
                self._cluster.abort(request_id)
                raise
            finally:
                if not finished_normally:
                    self._cluster.abort(request_id)
                await frames.aclose()

    def _build_spec(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        stop: list[str] | None,
        request_id: str,
        kwargs: dict[str, Any],
    ) -> GenerationSpec:
        prompt_ids = list(self.tokenizer.encode(prompt))
        frequency_penalty = kwargs.get("frequency_penalty")
        return GenerationSpec(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty and repetition_penalty != 1.0
                else None
            ),
            presence_penalty=(presence_penalty or None),
            frequency_penalty=(frequency_penalty or None),
            stop=list(stop or []),
            stop_token_ids=list(kwargs.get("stop_token_ids") or []),
            seed=kwargs.get("seed"),
            request_id=request_id,
        )

    async def _pump(
        self, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Drive the blocking rank-0 pipe generator from the event loop.

        ``LocalCluster.stream`` is a synchronous select-loop generator; running
        it on a worker thread and handing frames back over an asyncio queue
        keeps the event loop free while a decode step blocks in a collective.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def run() -> None:
            try:
                for frame in self._cluster.stream(
                    payload, timeout=GENERATE_IDLE_TIMEOUT_S
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, frame)
            except Exception as exc:  # noqa: BLE001 - surfaced to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        worker = loop.run_in_executor(None, run)
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Let the worker thread drain the generator to completion (after an
            # abort, rank 0 still emits its terminal frame) so the pipe is clean
            # for the next request.
            await worker
