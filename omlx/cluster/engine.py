# SPDX-License-Identifier: Apache-2.0
"""The head-daemon engine for a tensor-parallel formation (D5/D6).

``ClusterEngine`` is a :class:`BatchedEngine` whose weights do not live in this
process: the model is sharded across the rank child processes, and this engine
only holds the tokenizer (to apply the chat template and encode prompts) and the
pipe to rank 0. Chat templating, tool conversion and the rest of the request
shaping stay in the inherited :class:`BatchedEngine` layer; the only things this
subclass replaces are the four generation entry points, which route a templated
token stream through the rank-0 pipe instead of a local scheduler.

Multiplexing (S3 D5): rank 0 now runs the real ``Scheduler`` and can drive
several requests concurrently, so this engine is a multiplexing pipe client,
not a FIFO gate. One writer lock (``LocalCluster.write``'s own) serialises
stdin; a single background reader task demultiplexes rank 0's one reply pipe
by ``request_id`` into a per-request ``asyncio.Queue``, so concurrent HTTP
streams genuinely interleave instead of queueing behind each other. There is
no ``RequestOutputCollector`` and no ``EngineCore``; this engine *is* the
serving layer for a distributed model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from ..api.utils import clean_special_tokens
from ..engine.base import GenerationOutput
from ..engine.batched import BatchedEngine
from ..exceptions import SchedulerQueueFullError
from .launcher import LocalCluster
from .protocol import GenerationSpec

logger = logging.getLogger(__name__)

# How long a preflight reservation stays counted if its request never reaches
# ``stream_generate``. Every path that *does* reach it releases explicitly (see
# ``_preflight_queue``), so this only covers a request abandoned between the two
# — a client that disconnects after preflight but before starlette iterates the
# body, or a route raising in between. Long enough that no real
# preflight->submit gap (chat templating + encode) can expire early; short
# enough that a leak cannot hold slots for a meaningful part of a burst.
_RESERVATION_TTL_S = 30.0

# Idle timeout for a single reply frame from rank 0 (salvage generate-idle
# value): a rank that dies mid-collective leaves rank 0 blocked, and this is
# what turns that into a clean error instead of a hang. Also the demux
# reader's poll interval while genuinely idle -- it just loops again.
GENERATE_IDLE_TIMEOUT_S = 600.0

# S3 non-goal (spec S3): VLM and SpecPrefill requests are rejected with a
# clear error rather than silently generating text-only / unsparsified output
# -- there is no image encoding or draft-model wiring in a rank process.
_MULTIMODAL_CONTENT_TYPES = frozenset(
    {"image_url", "input_image", "image", "input_audio", "video_url"}
)


class ClusterNonGoalError(ValueError):
    """A request touched a feature S3 explicitly does not support on a
    distributed instance (VLM, SpecPrefill/spec-decode). Distinct from
    ``SchedulerQueueFullError``: there is nothing to retry.
    """


def _reject_if_multimodal(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in _MULTIMODAL_CONTENT_TYPES:
                raise ClusterNonGoalError(
                    "VLM requests are not supported on a distributed cluster "
                    "instance (S3 non-goal); serve this model single-node."
                )


def raise_if_multimodal_messages(messages: list[dict[str, Any]]) -> None:
    """Route-level entry for the S3/S6 multimodal non-goal guard.

    The server's non-VLM message conversion (``extract_text_content``,
    ``convert_anthropic_to_internal(preserve_images=False)``,
    ``convert_responses_input_to_messages``) strips image/audio parts BEFORE
    ``ClusterEngine.preflight_chat`` runs, so the preflight guard alone can
    never see them — a vision request against a text-only-distributed model
    would be silently served as text. Routes must call this on the RAW
    request messages when the resolved engine is a ClusterEngine.
    """
    _reject_if_multimodal(messages)


def _reject_if_specprefill(kwargs: dict[str, Any]) -> None:
    if kwargs.get("specprefill"):
        raise ClusterNonGoalError(
            "SpecPrefill is not supported on a distributed cluster instance "
            "(S3 non-goal); no draft model is wired into a rank process."
        )


def _final_trim(text: str, stops: list[str]) -> str:
    """Strip one trailing stop string from the terminal text (D6 final trim)."""
    for stop in stops:
        if stop and text.endswith(stop):
            return text[: -len(stop)]
    return text


def _error_from_frame(frame: dict[str, Any]) -> Exception:
    """Translate a rank-0 error frame into the typed exception it means.

    ``code == "queue_full"`` reconstructs ``SchedulerQueueFullError`` so it
    propagates through the same registered FastAPI handler single-node uses
    (D5) -- everything else is a generic rank/generation failure.
    """
    if frame.get("code") == "queue_full":
        # Head-side visibility for the backstop path. The preflight gate
        # (``_preflight_queue``) turns most of these into a clean 503 before
        # the route commits, so a frame arriving here means a request slipped
        # through the preflight->submit race -- worth a line, because by this
        # point the client can only be told in-stream, under HTTP 200.
        logger.warning(
            "cluster: rank 0 rejected %s after the response was committed — "
            "waiting queue full (%s/%s); the client sees an in-stream error, "
            "not a 503",
            frame.get("request_id"),
            frame.get("current_depth"),
            frame.get("max_depth"),
        )
        return SchedulerQueueFullError(
            current_depth=int(frame.get("current_depth", 0)),
            max_depth=int(frame.get("max_depth", 0)),
        )
    return RuntimeError(frame.get("error") or "rank error")


class ClusterEngine(BatchedEngine):
    """A :class:`BatchedEngine` backed by a distributed rank formation."""

    def __init__(
        self,
        model_name: str,
        *,
        cluster: LocalCluster,
        resolved_path: str | None = None,
        trust_remote_code: bool = False,
        on_rank_death: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(model_name, trust_remote_code=trust_remote_code)
        self._cluster = cluster
        self._resolved_path = resolved_path or model_name
        self._model_type_value: str | None = None
        # S6 D1: fired (sync, fire-and-forget) when the reply-pipe reader
        # dies -- the head-side signal that lets FormationManager degrade a
        # SERVING formation instead of only failing in-flight requests.
        self._on_rank_death = on_rank_death
        # D5 multiplexing: one demux reader task fans rank 0's single reply
        # pipe out to a per-request queue by request_id; the writer side
        # serialises on LocalCluster's own stdin lock (write() takes it).
        self._pending: dict[str, asyncio.Queue[Any]] = {}
        # Slots claimed by requests that passed preflight but have not reached
        # ``stream_generate`` yet; monotonic expiry deadlines, oldest first.
        self._reserved: deque[float] = deque()
        self._reader_task: asyncio.Task[None] | None = None
        self._reader_error: BaseException | None = None
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
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None

    @property
    def model_type(self) -> str | None:
        return self._model_type_value

    def get_stats(self) -> dict[str, Any]:
        # D5 named a live on-demand "stats" op over the pipe; that would mean
        # a synchronous round trip inside a method every caller invokes
        # directly off the event loop (routes.py's /state handler, formation
        # .snapshot() -- neither offloads to an executor), which would block
        # or (since the demux reader that could answer it also only runs on
        # that same loop) deadlock. get_stats() stays purely local instead;
        # `num_active_requests` is exact (it's just the demux's own pending
        # count) and needs no round trip. The rank-0 "stats" op still exists
        # on the wire (rank_worker.py) for a future async-safe caller.
        return {
            "engine_type": "cluster-distributed",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "world_size": self._cluster.world_size,
            # The backend the formation actually formed on (D9 negotiated
            # backend), the last completed request's coordination-tax
            # summary, and how many requests the scheduler is juggling now.
            "negotiated_backend": self._cluster.backend,
            "last_tax": self._last_tax,
            "num_active_requests": len(self._pending),
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        # Prefix/SSD cache is not wired for the distributed path (envelope
        # seam: always-miss).
        return None

    def has_active_requests(self) -> bool:
        return bool(self._pending)

    def _reserved_slots(self) -> int:
        """Live preflight reservations, expiring stale ones first."""
        now = time.monotonic()
        while self._reserved and self._reserved[0] <= now:
            self._reserved.popleft()
        return len(self._reserved)

    def _release_reservation(self) -> None:
        """Give back one slot claimed by ``_preflight_queue``.

        Counting, not identity: reservations are interchangeable, so the
        oldest is dropped rather than a specific request's. A
        ``stream_generate`` that never preflighted (an internal caller, or a
        test driving the engine directly) therefore releases nothing when the
        deque is empty, and at worst returns a slot one request early —
        which relaxes the gate toward the pre-reservation behaviour instead
        of holding slots that no request owns.
        """
        if self._reserved:
            self._reserved.popleft()

    def _preflight_queue(self) -> None:
        """Reject a request rank 0 has no room for, before the route commits
        to a ``StreamingResponse``.

        Rank 0's ``Scheduler.preflight_queue_or_raise`` is unreachable from
        here (another process), and asking over the pipe would mean a round
        trip per request against a loop that is busy stepping. The head does
        not need to ask: it is rank 0's only submitter, so ``_pending`` --
        requests written and not yet terminal -- is what rank 0 is holding,
        modulo the frames still in flight between the two.

        **``_pending`` alone is not enough, and the live rig proved it.** It is
        filled inside ``stream_generate``, which starlette only iterates
        *after* the route has committed to the ``StreamingResponse``. Under a
        cold burst — S3 acceptance row 4's recipe — every request preflights
        before any generator body runs, so the counter is still empty when the
        gate reads it and the gate never engages. The original design
        consciously accepted "over-admitting by a few"; on a cold burst it
        does not over-admit by a few, it does nothing at all.

        So preflight *reserves* the slot it just checked for, and
        ``stream_generate`` hands the reservation back when it takes its place
        in ``_pending`` (or when it fails before getting there). Occupancy is
        the sum of the two: requests rank 0 holds plus requests on their way
        to it.

        Still approximate, deliberately: ``_pending`` lags rank 0 by one frame,
        so this can reject a little early under churn. ``add_request`` on rank
        0 stays the authority; the frame it sends back on a miss is the
        backstop (``_error_from_frame``), and it is logged there.
        """
        from .scheduler_config import rank_inflight_capacity, rank_max_num_seqs

        capacity = rank_inflight_capacity()
        occupancy = len(self._pending) + self._reserved_slots()
        if occupancy < capacity:
            # Claim the slot in the same synchronous frame that checked for
            # it. Nothing awaits in between, so no second preflight can see
            # this one's check without also seeing its claim.
            self._reserved.append(time.monotonic() + _RESERVATION_TTL_S)
            return
        max_num_seqs = rank_max_num_seqs()
        # Report it the way rank 0 would: everything past the running batch
        # is queue depth, so the numbers a client sees match single-node's.
        raise SchedulerQueueFullError(
            current_depth=max(0, occupancy - max_num_seqs),
            max_depth=capacity - max_num_seqs,
        )

    async def preflight_chat(
        self, messages: list[dict[str, Any]] | None = None, *args: Any, **kwargs: Any
    ) -> None:
        # No prefill memory guard on the distributed path -- but the
        # waiting-queue gate and S3's non-goals (VLM, SpecPrefill) are cheap
        # to catch here, before the route wraps the response in a
        # StreamingResponse.
        self._preflight_queue()
        if messages:
            _reject_if_multimodal(messages)
        _reject_if_specprefill(kwargs)

    async def preflight_completion(self, *args: Any, **kwargs: Any) -> None:
        self._preflight_queue()
        _reject_if_specprefill(kwargs)

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
        # Everything up to the ``_pending`` insert is the window a preflight
        # reservation covers: once this request is in ``_pending`` it is
        # counted there instead, and if it never gets there the slot must go
        # back rather than wait out ``_RESERVATION_TTL_S``. Both exits run the
        # release, hence the try/finally rather than a call per path.
        try:
            if not self._loaded:
                await self.start()
            _reject_if_specprefill(kwargs)
            await self._ensure_reader()

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

            prompt_tokens = len(spec.prompt_ids)
            queue: asyncio.Queue[Any] = asyncio.Queue()
            # Own queue submitted+consumed here (D5): no per-request lock, so
            # concurrent stream_generate calls interleave naturally, each
            # reading only the frames the demux routed to its own request_id.
            self._pending[request_id] = queue
        finally:
            self._release_reservation()

        loop = asyncio.get_running_loop()
        payload = {"op": "generate", "spec": spec.to_dict()}
        try:
            await loop.run_in_executor(None, self._cluster.write, payload)
        except Exception:
            self._pending.pop(request_id, None)
            raise

        text = ""
        completion = 0
        finished_normally = False
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                frame = item
                if not frame.get("ok", False):
                    raise _error_from_frame(frame)
                if frame.get("done"):
                    finished_normally = True
                    self._last_tax = frame.get("tax")
                    # Stop detection ran in the scheduler; a final trim in
                    # the engine (D6) strips any trailing stop string left
                    # on the terminal text — a no-op when it was already
                    # removed. clean_special_tokens matches BatchedEngine's
                    # own finalization (batched.py:748,827) so the terminal
                    # text is identical to single-node's, not just the token
                    # stream (S3 Acceptance 1's greedy-parity test also
                    # compares rendered text, not only tokens).
                    final_text = clean_special_tokens(
                        _final_trim(frame.get("text", text), spec.stop)
                    )
                    yield GenerationOutput(
                        text=final_text,
                        new_text="",
                        prompt_tokens=int(frame.get("prompt_tokens", prompt_tokens)),
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
            self._pending.pop(request_id, None)

    # ---- D5 demux ----------------------------------------------------------

    async def _ensure_reader(self) -> None:
        """Start the single background reader task, once, lazily.

        Every ``stream_generate`` call shares this one task: rank 0 has
        exactly one reply pipe, so exactly one reader may ever be draining it
        (a second reader would steal frames belonging to the first).
        """
        if self._reader_task is not None:
            return
        if self._reader_error is not None:
            raise self._reader_error
        loop = asyncio.get_running_loop()
        self._reader_task = loop.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                frame = await loop.run_in_executor(
                    None, self._cluster.read_reply, GENERATE_IDLE_TIMEOUT_S
                )
            except Exception as exc:  # noqa: BLE001 - fanned out below
                logger.warning("cluster: reply pipe reader stopped: %s", exc)
                self._reader_error = exc
                self._fail_all_pending(exc)
                if self._on_rank_death is not None:
                    # Fire-and-forget (never awaited inline): the callback's
                    # own teardown may call this engine's `stop()`, which
                    # cancels `self._reader_task` -- that IS this task, so
                    # awaiting here would be cancelling ourselves mid-chain.
                    try:
                        self._on_rank_death(str(exc))
                    except Exception:  # noqa: BLE001 - never break the reader
                        logger.exception("cluster: on_rank_death callback failed")
                return
            if frame is None:
                # Idle timeout with the pipe still open: nothing to route,
                # keep waiting for the next frame.
                continue
            self._dispatch(frame)

    def _dispatch(self, frame: dict[str, Any]) -> None:
        rid = frame.get("request_id")
        if not rid:
            # "ready" (consumed by wait_ready() before this reader ever
            # starts) and the unknown-op error frame both carry no
            # request_id; nothing here is waiting on either.
            logger.debug("cluster: dropping request-id-less frame: %s", frame)
            return
        queue = self._pending.get(rid)
        if queue is None:
            logger.debug("cluster: dropping frame for unknown request %s", rid)
            return
        queue.put_nowait(frame)

    def _fail_all_pending(self, exc: BaseException) -> None:
        for queue in list(self._pending.values()):
            queue.put_nowait(exc)

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
