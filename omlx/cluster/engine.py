# SPDX-License-Identifier: Apache-2.0
"""The engine that makes a cluster look like a model.

`/v1/chat/completions` does not know or care that its model is spread over two
machines. Everything above this file - routing, the OpenAI adapters, tool-call
parsing, the admin UI - talks to a `BaseEngine`, so a cluster becomes servable
by being one.

Subclassing `BatchedEngine` rather than `BaseEngine` is a deliberate choice:
chat templating, Harmony preprocessing, tool conversion, partial-message
handling and thinking kwargs are all *tokenizer* work, identical whether the
weights live in this process or across a LAN. Inheriting them means the cluster
path cannot drift from the local path, and it keeps the diff off
`omlx/engine/batched.py`, which is upstream's file.

What is overridden is exactly the part that differs: `start` loads a tokenizer
instead of a model, and `generate`/`stream_generate` push token ids down a pipe
to rank 0 instead of into a local scheduler.

Concurrency
-----------
Requests batch. The worker runs continuous batching in lockstep across the
ranks (`omlx/cluster/batching.py`), so concurrent requests join a shared batch
and stream back interleaved, each under its own request id.

No prefix cache, though: a cache hit is local state, rank 0 would hit where
rank 1 missed, and they would disagree about how many tokens to prefill. Each
cluster request starts from a fresh KV cache.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, AsyncIterator

from omlx.cluster.protocol import GenerationSpec
from omlx.engine.base import GenerationOutput
from omlx.engine.batched import BatchedEngine

logger = logging.getLogger(__name__)

# Sentinel pushed onto the bridge queue when the worker's reply stream ends.
_DONE = object()


class ClusterEngine(BatchedEngine):
    """A `BaseEngine` whose forward passes happen on several machines."""

    def __init__(
        self,
        model_name: str,
        model_id: str,
        manager: Any,
        *,
        trust_remote_code: bool = False,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            model_settings=model_settings,
        )
        self._model_id = model_id
        self._manager = manager
        self._config: dict[str, Any] = {}
        self._active = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Form the cluster, and load only the tokenizer here.

        The leader daemon never holds the weights. Its rank-0 *worker* does,
        as a child process, and so does every peer's - that separation is what
        lets a model swap kill and respawn ranks without restarting the daemon
        or evicting the local models it is serving alongside the cluster.
        """
        if self._loaded:
            return

        from pathlib import Path

        from mlx_lm.tokenizer_utils import load as load_tokenizer

        from ..utils.tokenizer import get_tokenizer_config

        self._tokenizer = await asyncio.to_thread(
            load_tokenizer,
            # mlx-lm indexes into this with `/`, so it must be a Path.
            Path(self._model_name),
            get_tokenizer_config(
                self._model_name, trust_remote_code=self._trust_remote_code
            ),
        )
        self._config = await asyncio.to_thread(_read_config, self._model_name)
        await asyncio.to_thread(self._manager.form, self._model_id)
        self._loaded = True
        logger.info(
            "cluster: engine ready for %s across %d ranks",
            self._model_id,
            self._manager.status().world_size,
        )

    async def stop(self) -> None:
        if not self._loaded:
            return
        await asyncio.to_thread(self._manager.teardown)
        self._tokenizer = None
        self._loaded = False

    # -- what a cluster cannot do ------------------------------------------

    @property
    def model_type(self) -> str | None:
        """Read from `config.json`, since there is no local model object."""
        value = self._config.get("model_type")
        return value if isinstance(value, str) else None

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    @property
    def grammar_compiler(self):
        """Structured output is not wired through the cluster path yet.

        Returning None makes the API layer refuse a schema rather than quietly
        ignore it and return unconstrained text.
        """
        return None

    async def preflight_chat(self, *args: Any, **kwargs: Any) -> None:
        """No local scheduler, so no local memory ceiling to check against.

        The gate that matters for a cluster is whether the shards fit on their
        own machines, which is decided at formation.
        """
        if not self._loaded:
            await self.start()

    async def preflight_completion(self, *args: Any, **kwargs: Any) -> None:
        if not self._loaded:
            await self.start()

    # -- generation --------------------------------------------------------

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
        """Non-streaming: drain the stream and return the last state."""
        final: GenerationOutput | None = None
        async for output in self.stream_generate(
            prompt=prompt,
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
            raise RuntimeError("cluster produced no output")
        return final

    async def stream_generate(
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
        if not self._manager.formed:
            # The deathwatch tore the cluster down under us - a rank died.
            # In-flight requests failed fast; this one arrived after, and
            # re-forming is a better answer than reporting a death it did not
            # witness. `form` serializes concurrent attempts itself.
            await asyncio.to_thread(self._manager.form, self._model_id)

        import uuid

        spec = GenerationSpec(
            prompt_ids=self._tokenizer.encode(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            # mlx-lm reads "no penalty" as None, not as the neutral 1.0/0.0 the
            # OpenAI surface uses; passing the neutral value builds a processor
            # that runs on every token to change nothing.
            repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
            presence_penalty=presence_penalty or None,
            frequency_penalty=kwargs.get("frequency_penalty") or None,
            stop=list(stop or []),
            seed=kwargs.get("seed"),
            # The batch serves several requests at once; the id is how this
            # one's replies and its abort find their way back to it.
            request_id=uuid.uuid4().hex,
        )

        text = ""
        prompt_tokens = len(spec.prompt_ids)
        completion_tokens = 0
        finished = False
        self._active += 1
        try:
            async for reply in self._bridge(spec):
                if reply.get("done"):
                    finished = True
                    yield GenerationOutput(
                        text=reply.get("text", text),
                        new_text="",
                        prompt_tokens=reply.get("prompt_tokens", prompt_tokens),
                        completion_tokens=reply.get(
                            "completion_tokens", completion_tokens
                        ),
                        finished=True,
                        finish_reason=reply.get("finish_reason", "stop"),
                    )
                    break

                chunk = reply.get("chunk", "")
                if not chunk:
                    continue
                text += chunk
                completion_tokens = reply.get("tokens", completion_tokens + 1)
                yield GenerationOutput(
                    text=text,
                    new_text=chunk,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    finished=False,
                )
        except GeneratorExit:
            logger.info("cluster: client disconnected, aborting the run")
            raise
        finally:
            self._active -= 1
            if not finished:
                # This request's sequence is still in the batch, burning a
                # slot and steps on every rank. Only this one is evicted;
                # the rest of the batch keeps serving.
                await asyncio.to_thread(self._manager.abort, spec.request_id)

    async def _bridge(self, spec: GenerationSpec) -> AsyncIterator[dict[str, Any]]:
        """Pump the worker's blocking pipe into the event loop.

        A thread rather than a task: the read side is a pipe owned by a
        subprocess, and the alternative - polling it from the loop - would
        either add latency per token or spin.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def pump() -> None:
            try:
                for reply in self._manager.stream(spec):
                    loop.call_soon_threadsafe(queue.put_nowait, reply)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the loop
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        threading.Thread(target=pump, name="cluster-stream", daemon=True).start()

        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    # -- observability -----------------------------------------------------

    def has_active_requests(self) -> bool:
        return self._active > 0

    def get_stats(self) -> dict[str, Any]:
        status = self._manager.status()
        return {
            "engine_type": "cluster",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "backend": status.backend,
            "world_size": status.world_size,
            "nodes": [n.node_id for n in status.nodes],
            "active_requests": self._active,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        """No prefix cache in cluster mode; see the module docstring."""
        return None

    async def abort_all_requests(self) -> int:
        return 1 if await asyncio.to_thread(self._manager.abort) else 0


def _read_config(model_path: str) -> dict[str, Any]:
    """`config.json` for the sharded model, read on the leader only."""
    import json
    from pathlib import Path

    try:
        return json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
