# SPDX-License-Identifier: Apache-2.0
"""
Engine pool for oMLX multi-model serving.

This module manages multiple model engines with LRU-based eviction
when memory limits are exceeded. It supports:

- Pre-load memory checking to ensure models fit before loading
- LRU eviction of least recently used models
- Model pinning to keep specific models always loaded
- BatchedEngine for all LLM models (continuous batching)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .cluster.placement import NodeCapacity, PlacementDecision
    from .model_settings import ModelSettingsManager

import mlx.core as mx

from .engine import BaseEngine, BatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine.sts import STSEngine
from .engine.stt import STTEngine
from .engine.tts import TTSEngine
from .engine.vlm import VLMBatchedEngine
from .engine_core import get_mlx_executor
from .exceptions import (
    DEFAULT_CEILING_ADVICE,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
    describe_ceiling_binding,
)
from .model_discovery import discover_models, format_size
from .scheduler import SchedulerConfig
from .utils.proc_memory import get_phys_footprint

logger = logging.getLogger(__name__)


@dataclass
class EngineEntry:
    """Per-model state in the engine pool."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: Literal[
        "llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"
    ]  # Model type
    engine_type: Literal[
        "batched",
        "simple",
        "embedding",
        "reranker",
        "vlm",
        "audio_stt",
        "audio_tts",
        "audio_sts",
    ]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    actual_size: int | None = None  # Observed process-memory delta after load settles
    config_model_type: str = (
        ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    )
    thinking_default: bool | None = (
        None  # True if model thinks by default, False if not, None if unknown
    )
    preserve_thinking_default: bool | None = (
        None  # True when template supports preserve_thinking (Qwen 3.6+)
    )
    model_context_length: int | None = (
        None  # Declared context length from config.json (None if unknown)
    )
    source_type: str = "local"
    source_repo_id: str | None = None
    is_helper: bool = False  # Speculative-decoding drafter (dFlash/Assistant/MTP)
    engine: (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
        | None
    ) = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_loading: bool = False  # Prevent concurrent loads
    loading_started_at: float | None = None  # Timestamp when current load started
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Set by memory enforcer to abort in-progress load
    in_use: int = 0  # in-flight acquire/use lease count; never evict while > 0
    abort_requested: bool = False  # Set under hard pressure for leased requests
    pending_unload_reason: str | None = None  # Unload as soon as leases/activity drain
    runtime_settings_signature: tuple[tuple[str, str], ...] | None = None
    load_failed: bool = False  # Sticky until the next discovery refresh
    load_failure_message: str | None = None
    load_failure_at: float | None = None
    # S4 D4: cluster entries are first-class EngineEntry rows -- `engine` is
    # the ClusterEngine once formed, `estimated_size` becomes the head's
    # per-rank share while formed. `kind` is the only signal callers need;
    # nothing outside engine_pool.py should branch on `is_helper`-style
    # sentinels for this.
    kind: Literal["local", "cluster"] = "local"
    cluster_head_share: int | None = None  # est_size / world_size * headroom
    # The pre-form estimated_size, stashed at claim time so unform can
    # restore it exactly rather than re-deriving it from a fresh discovery.
    cluster_original_estimated_size: int | None = None
    cluster_teardown_in_progress: bool = False  # claimed by the unload driver
    # Explicit operator/dashboard unload ignores pin and busy state
    # (matches _unload_engine's immediate-abort semantics for local
    # entries); background marks (TTL, admission eviction) do not set this.
    cluster_unload_force: bool = False
    # S4 P2b: consecutive `formation.unload` failures for this entry's
    # driver-owned teardown. Reset to 0 on a successful teardown or a fresh
    # operator unload (`_force_mark_cluster_unload_locked`).
    cluster_teardown_failures: int = 0
    # Set once `cluster_teardown_failures` hits `_CLUSTER_TEARDOWN_MAX_FAILURES`:
    # the entry stays kind="cluster" (never silently reverts to a loadable
    # local row) but the driver stops auto-retrying it. Only an explicit
    # operator unload (which resets the counters above) clears this.
    cluster_teardown_quarantined: bool = False


# S4 D4: bounded outer retry for get_engine's cluster branches (rev4/5/6).
_GET_ENGINE_MAX_ATTEMPTS = 3
# Added to the formation command timeout for a bounded wait on cluster
# teardown (request_unload, admission eviction of a cluster victim).
_CLUSTER_UNLOAD_MARGIN_S = 15.0
_CLUSTER_UNLOAD_FALLBACK_TIMEOUT_S = 30.0
# S4 P2b: consecutive `formation.unload` failures before the driver stops
# auto-retrying a cluster entry's teardown and quarantines it instead.
_CLUSTER_TEARDOWN_MAX_FAILURES = 3


class PlacementStaleError(Exception):
    """A precomputed :class:`~omlx.cluster.placement.PlacementDecision`
    no longer holds at claim time (S4 D4 rev5): another load changed the
    pool's occupancy between the attempt that computed it and the attempt
    that tried to act on it. The outer ``get_engine`` retry recomputes a
    fresh decision on the next attempt rather than acting on a stale one.
    """


class _ClusterVictimPendingError(Exception):
    """Internal control-flow signal, not a real error: ``get_engine``'s
    admission eviction picked a formed cluster entry as its LRU victim.
    Raised while the pool lock is held so the enclosing ``async with``
    releases it before the caller awaits the unload driver's completion
    event -- the binding invariant is no formation await under the lock.
    """

    def __init__(self, model_id: str, event: asyncio.Event) -> None:
        super().__init__(model_id)
        self.model_id = model_id
        self.event = event


class _ClusterLoadPendingError(Exception):
    """Internal control-flow signal: placement chose ``distributed``.
    Raised under the lock so it releases before ``load_cluster_model`` runs
    its own a/b/c lock choreography outside it.
    """

    def __init__(self, decision: PlacementDecision) -> None:
        super().__init__(decision.mode)
        self.decision = decision


class _ConcurrentVariantRaceError(Exception):
    """Internal control-flow signal (S4 P2b): the cluster-gate branch of
    ``_get_engine_attempt`` released ``self._lock`` for placement I/O and,
    on re-acquiring it, found the entry loaded with a runtime-settings
    variant that does not match what this attempt requested -- a
    concurrent request reloaded it in the window. Raised under the lock so
    the caller never serves the wrong variant; the outer ``get_engine``
    retry loop (bound already exists) re-evaluates against the now-current
    entry, which will unload/reload to the requested variant.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.model_id = model_id


class EnginePool:
    """
    Manages multiple model engines with LRU-based memory management.

    Features:
    - Pre-load memory checking (evict before load, not after)
    - LRU eviction when memory limit is exceeded
    - Model pinning to prevent eviction
    - Automatic engine type selection based on model type
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initialize the engine pool.

        Args:
            scheduler_config: Configuration for BatchedEngine schedulers

        Note:
            Pre-load admission consults `enforcer.get_final_ceiling()` via
            the `_get_final_ceiling` callback set by `server.init_server()`.
            When that reads 0 (memory guard disabled), the pool falls back
            to `enforcer.get_admission_ceiling()` via `_get_admission_ceiling`
            for best-effort LRU eviction (#2290). Idle-model eviction
            starts at `enforcer.get_admission_soft_target()` via
            `_get_admission_soft_target` so the old model is unloaded
            before the new weights allocate (#2319). Until the callbacks
            are wired up the pool admits unconditionally.
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._current_model_memory = 0
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._get_final_ceiling: object | None = None  # Set by server
        self._get_admission_ceiling: object | None = None  # Set by server
        self._get_admission_soft_target: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks
        self._load_seconds_per_gb_ema: float | None = None
        self._load_time_observations: int = 0
        self._lease_release_tasks: set[asyncio.Task[None]] = set()
        # S4 D4: the one out-of-lock cluster teardown driver, created lazily
        # on the first cluster mark. `_cluster_unload_events` lets callers
        # (request_unload, admission eviction) await one entry's teardown.
        self._cluster_unload_task: asyncio.Task[None] | None = None
        self._cluster_unload_wake: asyncio.Event | None = None
        self._cluster_unload_events: dict[str, asyncio.Event] = {}
        self.configure_hot_cache_budget()

    @property
    def current_model_memory(self) -> int:
        """Current memory used by loaded models in bytes."""
        return self._current_model_memory

    def configure_hot_cache_budget(self) -> None:
        """Ensure loaded schedulers share one process-wide hot cache budget."""
        hot_max = int(getattr(self._scheduler_config, "hot_cache_max_size", 0) or 0)
        if hot_max <= 0:
            self._scheduler_config.hot_cache_budget = None
            return

        current = getattr(self._scheduler_config, "hot_cache_budget", None)
        if current is not None and getattr(current, "max_bytes", None) == hot_max:
            return

        from .cache.paged_ssd_cache import SharedHotCacheBudget

        self._scheduler_config.hot_cache_budget = SharedHotCacheBudget(hot_max)

    def _current_ceiling(self) -> int:
        """Resolve the current memory ceiling via the enforcer callback.

        Returns 0 when no callback is wired up (treated by callers as
        "no limit").
        """
        cb = self._get_final_ceiling
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def _fallback_admission_ceiling(self) -> int:
        """Best-effort admission ceiling used when `_current_ceiling()` is 0.

        Wired to `enforcer.get_admission_ceiling`, which keeps returning
        the static ceiling while the memory guard is disabled so a model
        swap still evicts LRU models instead of overcommitting physical
        memory (#2290). Returns 0 when no callback is wired up (standalone
        pools admit unconditionally).
        """
        cb = self._get_admission_ceiling
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def head_capacity(self, *, node_id: str = "head") -> NodeCapacity:
        """This pool's own capacity as an S4 D2 placement input.

        Read-only; the only piece of `omlx/cluster/` that ever touches the
        pool's memory-ceiling internals directly (D2b). Ceiling resolution
        mirrors the admission cascade a load already uses
        (`_current_ceiling()` then `_fallback_admission_ceiling()`,
        `:928-932`) — D2's capacity-unknown rule treats a ceiling of 0 as
        "unknown", so applying the same fallback here means the head's own
        placement input is never spuriously unknown just because the guard
        is disabled. Also applies the S4 P3 head-share correction
        (`_cluster_head_share_total()`): a resident formation's head shard
        is invisible to the phys footprint the dynamic ceiling is derived
        from, so the ceiling alone under-reports headroom for a LOCAL fit
        against an already-resident formation. `plan_placement` folds this
        `NodeCapacity` into `nodes = [head, *workers]` for the distributed
        branch too, so the head's own per-rank `NodeFit` there gets the
        same credit (correct: that fit also nets the resident share against
        the collapsed ceiling). Only a *worker's* ceiling is unaffected --
        that capacity comes from `_cluster_worker_capacities()`, not here.
        """
        from .cluster.placement import NodeCapacity

        ceiling = self._current_ceiling()
        if ceiling <= 0:
            ceiling = self._fallback_admission_ceiling()
        if ceiling > 0:
            ceiling += self._cluster_head_share_total()
        models_present = {
            model_id: entry.estimated_size for model_id, entry in self._entries.items()
        }
        return NodeCapacity(
            node_id=node_id,
            memory_ceiling=max(0, ceiling),
            current_model_memory=self._current_model_memory,
            models_present=models_present,
        )

    def _admission_soft_target(self) -> int:
        """Soft watermark that pre-load eviction targets (#2319).

        Wired to `enforcer.get_admission_soft_target`. Eviction of idle
        LRU models starts once the projected total exceeds this, so an
        old model is unloaded *before* the new weights allocate instead
        of after the first request's prefill guard fires. Returns 0 when
        no callback is wired up (callers fall back to the ceiling).
        """
        cb = self._get_admission_soft_target
        if cb is None:
            return 0
        try:
            return int(cb())
        except Exception:  # noqa: BLE001
            return 0

    def _cluster_head_share_total(self) -> int:
        """Sum of `cluster_head_share` across resident (formed) cluster
        entries (S4 P3 rig finding).

        A formed cluster entry's rank-0 shard is wired in the rank child
        process, not this one, so it never shows up in the phys footprint
        the enforcer's dynamic ceiling is derived from -- the ceiling
        collapses by the shard size as if that memory simply vanished.
        Meanwhile `_current_model_memory` still counts the same shard via
        `entry.cluster_head_share` (`:1319`). Left alone, that is a double
        subtraction: once off the ceiling, once off the usage side. Every
        admission-side ceiling comparison adds this back to restore the
        basis convention local models already get, where a model's memory
        is inside the phys footprint the ceiling is drawn from, not
        subtracted from the ceiling itself.
        """
        return sum(
            entry.cluster_head_share or 0
            for entry in self._entries.values()
            if entry.kind == "cluster" and entry.engine is not None
        )

    def _ceiling_binding_and_advice(
        self, *, ceiling: int, current: int, tail: str
    ) -> tuple[str | None, str | None]:
        """Name the ceiling that refused this load and the knob that moves it.

        A load refusal used to say "free system memory or lower
        memory_guard_tier" with no breakdown, which is the wrong knob twice
        over: the tier ladder runs safe → balanced → aggressive, so lowering
        it shrinks the ceiling further, and on a dynamic-bound machine the
        static / metal caps the user is likely to be shown elsewhere have
        room to spare. Reuses the scheduler's advice ladder so both
        rejection paths name the same constraint.

        Returns ``(None, None)`` when the enforcer is not wired up or its
        breakdown is unreadable; callers fall back to the generic advice.
        """
        enforcer = self._process_memory_enforcer
        getter = (
            getattr(enforcer, "get_ceiling_breakdown", None)
            if enforcer is not None
            else None
        )
        if not callable(getter):
            return None, None
        try:
            breakdown = getter()
            static = int(breakdown["static"])
            dynamic = int(breakdown["dynamic"])
            metal_cap = int(breakdown["metal_cap"])
        except Exception:  # noqa: BLE001
            return None, None
        if max(static, dynamic, metal_cap) <= 0:
            return None, None
        return describe_ceiling_binding(
            static=static,
            dynamic=dynamic,
            metal_cap=metal_cap,
            tier=str(getattr(enforcer, "memory_guard_tier", "") or ""),
            current=current,
            fmt=format_size,
            tail=tail,
        )

    def _wake_process_memory_enforcer(self, *, active: bool = False) -> None:
        enforcer = self._process_memory_enforcer
        wake = getattr(enforcer, "wake", None) if enforcer is not None else None
        if callable(wake):
            wake(active=active)

    @staticmethod
    def _invalidate_cluster_node_state_cache() -> None:
        """S4 D1 residual: force this worker's next heartbeat to re-scan its
        model dirs after a local load/unload, rather than waiting up to 60s
        for the cached inventory to expire on its own. A no-op on a head or
        a node with no cluster role (the manager import/lookup is cheap and
        already used elsewhere in this module).
        """
        from .cluster.manager import get_cluster_manager

        manager = get_cluster_manager()
        if manager is not None and manager.role == "worker":
            manager.invalidate_node_state_cache()

    @staticmethod
    def _canonical_signature_value(value: object) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return repr(value)

    def _engine_runtime_signature(
        self,
        model_id: str,
        runtime_settings: object | None = None,
        *,
        loaded_engine: object | None = None,
    ) -> tuple[tuple[str, str], ...] | None:
        settings = runtime_settings
        if settings is None and self._settings_manager is not None:
            get_settings = getattr(self._settings_manager, "get_settings", None)
            if callable(get_settings):
                settings = get_settings(model_id)
        if settings is None:
            return None

        to_dict = getattr(settings, "to_dict", None)
        data = to_dict() if callable(to_dict) else {}
        entry = self._entries.get(model_id)
        is_diffusion = bool(entry and self._entry_is_diffusion_model(entry))
        loaded_engine_name = (
            type(loaded_engine).__name__ if loaded_engine is not None else None
        )

        def has_value(key: str) -> bool:
            value = data.get(key)
            return value is not None and value != ""

        def normalized_index_cache_freq() -> int | None:
            value = data.get("index_cache_freq")
            try:
                freq = int(value) if value is not None else None
            except (TypeError, ValueError):
                return None
            return freq if freq is not None and freq >= 2 else None

        signature: list[tuple[str, str]] = []

        def add(key: str, value: object) -> None:
            signature.append((key, self._canonical_signature_value(value)))

        # Security/load gates.
        add("trust_remote_code", bool(data.get("trust_remote_code", False)))
        add("index_cache_freq", normalized_index_cache_freq())

        # Load-time model variants. Dependent fields only matter when their
        # feature is active; stale draft paths or tuning defaults must not
        # force a reload when the corresponding feature is disabled.
        mtp_active = bool(data.get("mtp_enabled", False))
        add("mtp_enabled", mtp_active)

        turboquant_active = bool(data.get("turboquant_kv_enabled", False))
        add("turboquant_kv_enabled", turboquant_active)
        if turboquant_active:
            add("turboquant_kv_bits", data.get("turboquant_kv_bits", 4))
            add("turboquant_skip_last", data.get("turboquant_skip_last", True))

        specprefill_active = bool(data.get("specprefill_enabled", False)) and has_value(
            "specprefill_draft_model"
        )
        add("specprefill_enabled", specprefill_active)
        if specprefill_active:
            add("specprefill_draft_model", data.get("specprefill_draft_model"))
            add("specprefill_keep_pct", data.get("specprefill_keep_pct", 0.2))
            add("specprefill_threshold", data.get("specprefill_threshold"))

        dflash_active = (
            bool(data.get("dflash_enabled", False))
            and has_value("dflash_draft_model")
            and not is_diffusion
        )
        if loaded_engine_name is not None:
            dflash_active = loaded_engine_name == "DFlashEngine"
        add("dflash_enabled", dflash_active)
        if dflash_active:
            add("dflash_draft_model", data.get("dflash_draft_model"))
            add(
                "dflash_draft_quant_enabled",
                bool(data.get("dflash_draft_quant_enabled", False)),
            )
            if data.get("dflash_draft_quant_enabled", False):
                add(
                    "dflash_draft_quant_weight_bits",
                    data.get("dflash_draft_quant_weight_bits", 4),
                )
                add(
                    "dflash_draft_quant_activation_bits",
                    data.get("dflash_draft_quant_activation_bits", 16),
                )
                add(
                    "dflash_draft_quant_group_size",
                    data.get("dflash_draft_quant_group_size", 64),
                )
            add("dflash_max_ctx", data.get("dflash_max_ctx"))
            add("dflash_in_memory_cache", data.get("dflash_in_memory_cache", True))
            add(
                "dflash_in_memory_cache_max_entries",
                data.get("dflash_in_memory_cache_max_entries", 4),
            )
            add(
                "dflash_in_memory_cache_max_bytes",
                data.get("dflash_in_memory_cache_max_bytes"),
            )
            add("dflash_ssd_cache", bool(data.get("dflash_ssd_cache", False)))
            if data.get("dflash_ssd_cache", False):
                add(
                    "dflash_ssd_cache_max_bytes", data.get("dflash_ssd_cache_max_bytes")
                )
            add("dflash_draft_window_size", data.get("dflash_draft_window_size"))
            add("dflash_draft_sink_size", data.get("dflash_draft_sink_size"))
            add("dflash_verify_mode", data.get("dflash_verify_mode"))

        vlm_mtp_active = bool(data.get("vlm_mtp_enabled", False)) and has_value(
            "vlm_mtp_draft_model"
        )
        if loaded_engine is not None and vlm_mtp_active:
            drafter = getattr(loaded_engine, "vlm_mtp_drafter", None)
            if callable(drafter):
                drafter = drafter()
            vlm_mtp_active = drafter is not None
        add("vlm_mtp_enabled", vlm_mtp_active)
        if vlm_mtp_active:
            add("vlm_mtp_draft_model", data.get("vlm_mtp_draft_model"))
            add("vlm_mtp_draft_block_size", data.get("vlm_mtp_draft_block_size"))

        return tuple(signature)

    @property
    def model_count(self) -> int:
        """Total number of discovered models."""
        return len(self._entries)

    @property
    def loaded_model_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.engine is not None)

    async def apply_embedding_batch_size(self, batch_size: int) -> None:
        """Apply embedding batch size to future and currently loaded embedding engines."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("embedding batch size must be > 0")

        async with self._lock:
            self._scheduler_config.embedding_batch_size = batch_size
            for entry in list(self._entries.values()):
                engine = entry.engine if entry is not None else None
                if isinstance(engine, EmbeddingEngine):
                    engine._batch_size = batch_size

    def discover_models(
        self, model_dirs: str | list[str], pinned_models: list[str] | None = None
    ) -> None:
        """
        Discover models in the specified directory or directories.

        Args:
            model_dirs: Path or list of paths to directories containing model subdirectories
            pinned_models: List of model IDs to pin (never evict)
        """
        from pathlib import Path

        from .model_discovery import discover_models_from_dirs

        if isinstance(model_dirs, str):
            dirs = [Path(model_dirs)]
        else:
            dirs = [Path(d) for d in model_dirs]

        if len(dirs) == 1:
            discovered = discover_models(dirs[0])
        else:
            discovered = discover_models_from_dirs(dirs)

        pinned_set = set(pinned_models or [])

        for model_id, info in discovered.items():
            existing = self._entries.get(model_id)
            if existing is not None and (
                existing.engine is not None or existing.is_loading
            ):
                # Loaded or loading model: preserve runtime state, only
                # update the pinned flag. Replacing an in-flight entry would
                # orphan the engine the load attaches on completion (#2307).
                existing.is_pinned = model_id in pinned_set
            else:
                # New or unloaded model: create fresh entry
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    config_model_type=getattr(info, "config_model_type", ""),
                    thinking_default=getattr(info, "thinking_default", None),
                    preserve_thinking_default=getattr(
                        info, "preserve_thinking_default", None
                    ),
                    model_context_length=getattr(info, "model_context_length", None),
                    source_type=getattr(info, "source_type", "local"),
                    source_repo_id=getattr(info, "source_repo_id", None),
                    is_helper=getattr(info, "is_helper", False),
                    is_pinned=model_id in pinned_set,
                )

            if model_id in pinned_set:
                logger.info(f"Pinned model: {model_id}")

        # Remove entries no longer discovered and neither loaded nor loading
        discovered_ids = set(discovered.keys())
        stale = [
            mid
            for mid in self._entries
            if mid not in discovered_ids
            and self._entries[mid].engine is None
            and not self._entries[mid].is_loading
        ]
        for mid in stale:
            del self._entries[mid]

        # Warn about pinned models not found
        found_models = set(self._entries.keys())
        for model_id in pinned_set:
            if model_id not in found_models:
                logger.warning(f"Pinned model not found: {model_id}")

        logger.info(f"Discovered {len(self._entries)} models")

    _MODEL_TYPE_TO_ENGINE: dict[str, str] = {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
    }

    @staticmethod
    def _entry_is_diffusion_model(entry: EngineEntry) -> bool:
        model_type = (entry.config_model_type or "").lower().replace("-", "_")
        return model_type == "diffusion_gemma"

    def apply_settings_overrides(
        self, settings_manager: "ModelSettingsManager"
    ) -> None:
        """Apply model_type_override from persisted settings to discovered entries."""
        for model_id, entry in self._entries.items():
            settings = settings_manager.get_settings(model_id)
            if settings.model_type_override:
                entry.model_type = settings.model_type_override
                entry.engine_type = self._MODEL_TYPE_TO_ENGINE.get(
                    settings.model_type_override, "batched"
                )
                logger.info(
                    f"Applied model_type override for {model_id}: "
                    f"type={entry.model_type}, engine={entry.engine_type}"
                )

    def get_model_ids(self) -> list[str]:
        """Get list of all discovered model IDs."""
        return list(self._entries.keys())

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return [mid for mid, e in self._entries.items() if e.engine is not None]

    def get_entry(self, model_id: str) -> EngineEntry | None:
        """Get entry for a specific model, or None if not found."""
        return self._entries.get(model_id)

    def _clear_load_failure(self, entry: EngineEntry) -> None:
        entry.load_failed = False
        entry.load_failure_message = None
        entry.load_failure_at = None

    def _mark_load_failure(self, entry: EngineEntry, exc: BaseException) -> None:
        entry.load_failed = True
        entry.load_failure_message = str(exc) or type(exc).__name__
        entry.load_failure_at = time.time()

    def _raise_if_model_path_missing_locked(
        self, model_id: str, entry: EngineEntry
    ) -> None:
        """Drop stale unloaded entries whose backing model directory vanished."""
        model_path = Path(entry.model_path)
        if model_path.exists() and (model_path / "config.json").exists():
            return

        if entry.engine is None:
            self._entries.pop(model_id, None)
        available = [mid for mid in self._entries if mid != model_id]
        raise ModelNotFoundError(model_id, available)

    def _raise_if_load_failed(self, model_id: str, entry: EngineEntry) -> None:
        if not entry.load_failed:
            return
        detail = entry.load_failure_message or "previous load attempt failed"
        logger.warning(
            "Skipping load retry for '%s' after cached failure: %s",
            model_id,
            detail,
        )
        raise ModelUnavailableError(
            model_id,
            f"Model '{model_id}' is unavailable after a previous load failure: {detail}. "
            "Reload models after fixing the files to retry.",
        )

    def set_pinned(self, model_id: str, pinned: bool) -> bool:
        """
        Set the pinned status for a model.

        Args:
            model_id: The model ID to update
            pinned: Whether to pin (True) or unpin (False) the model

        Returns:
            True if successful, False if model not found.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False
        entry.is_pinned = pinned
        return True

    def _case_insensitive_entry_match(self, name: str) -> str | None:
        """Find a model entry matching *name* case-insensitively.

        Returns the actual model_id if found, None otherwise.
        """
        lower = name.lower()
        for mid in self._entries:
            if mid.lower() == lower:
                return mid
        return None

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        """Resolve a model alias to its actual model_id (directory name).

        Tries exact match in _entries first, then case-insensitive match,
        then exposed profile model IDs, then scans model settings for alias
        match. If those fail and input contains a provider prefix (e.g.
        "omlx/my-model"), strips the prefix and retries. Returns the
        original string if no match found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        all_settings = None
        if settings_manager is not None:
            # Exposed profiles resolve to the physical model they overlay
            # (handles provider prefixes internally).
            if hasattr(settings_manager, "get_exposed_profile_source_model_id"):
                profile_source = settings_manager.get_exposed_profile_source_model_id(
                    model_id_or_alias
                )
                if profile_source is not None:
                    return profile_source
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if ms.model_alias and ms.model_alias == model_id_or_alias:
                    return mid

        # Strip provider prefix (e.g. "omlx/qwen3.5-35b" -> "qwen3.5-35b")
        if "/" in model_id_or_alias:
            stripped = model_id_or_alias.split("/", 1)[1]
            if stripped in self._entries:
                return stripped
            ci_match = self._case_insensitive_entry_match(stripped)
            if ci_match is not None:
                return ci_match
            if all_settings is not None:
                for mid, ms in all_settings.items():
                    if ms.model_alias and ms.model_alias == stripped:
                        return mid

        return model_id_or_alias

    @staticmethod
    def _entry_has_active_requests(entry: EngineEntry) -> bool:
        engine = entry.engine
        if engine is None:
            return False
        has_active_requests = getattr(engine, "has_active_requests", None)
        if not callable(has_active_requests):
            return False
        try:
            return has_active_requests() is True
        except Exception:
            return True

    def _entry_is_busy(self, entry: EngineEntry) -> bool:
        return entry.in_use > 0 or self._entry_has_active_requests(entry)

    def _raise_if_reload_busy(self, entry: EngineEntry, operation: str) -> None:
        if self._entry_is_busy(entry):
            raise ModelBusyError(entry.model_id, operation)

    @staticmethod
    def _engine_has_usable_tokenizer(engine: object) -> bool:
        tokenizer = getattr(engine, "tokenizer", None)
        return tokenizer is not None and callable(getattr(tokenizer, "encode", None))

    def _validate_llm_engine_ready(self, model_id: str, engine: object | None) -> None:
        if engine is None:
            raise ModelLoadingError(
                model_id,
                f"Model '{model_id}' did not return a loaded engine.",
            )
        llm_engine_types = [BaseEngine]
        if isinstance(VLMBatchedEngine, type):
            llm_engine_types.append(VLMBatchedEngine)
        if isinstance(engine, tuple(llm_engine_types)) and not (
            self._engine_has_usable_tokenizer(engine)
        ):
            raise ModelLoadingError(
                model_id,
                f"Model '{model_id}' loaded without a usable tokenizer.",
            )

    def _mark_pending_unload_locked(
        self,
        model_id: str,
        reason: str,
        *,
        abort_requested: bool = False,
    ) -> bool:
        """Mark a loaded non-pinned model for unload once it is no longer busy.

        Caller must hold ``self._lock``. Returns True when a pending marker was
        installed. The method deliberately does not unload by itself; call
        ``_unload_pending_if_idle_locked`` after abort/release state changes.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.engine is None or entry.is_loading or entry.is_pinned:
            return False
        entry.pending_unload_reason = reason
        if abort_requested:
            entry.abort_requested = True
        if entry.kind == "cluster":
            # S4 D4: TTL and enforcer background marks reach here too;
            # wake the driver so a mark-only cluster entry actually gets
            # torn down instead of sitting marked until something else
            # happens to wake it.
            self._wake_cluster_unload_driver_locked()
        return True

    def _find_pending_unload_ready_locked(self) -> str | None:
        candidates: list[tuple[float, str]] = []
        for mid, entry in self._entries.items():
            if not entry.pending_unload_reason:
                continue
            if (
                entry.engine is None
                or entry.is_loading
                or entry.is_pinned
                or entry.kind == "cluster"  # S4 D4: the driver handles these
                or self._entry_is_busy(entry)
            ):
                continue
            candidates.append((entry.last_access, mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _unload_pending_if_idle_locked(self, model_id: str) -> bool:
        """Unload a pending model if all leases and active requests have drained.

        Caller must hold ``self._lock``.
        """
        entry = self._entries.get(model_id)
        if entry is None or entry.engine is None or not entry.pending_unload_reason:
            return False

        if entry.kind == "cluster":
            # S4 D4 rev5: cluster teardown always awaits a formation command
            # round-trip, which can never happen under this lock (and some
            # callers here -- _release_engine_lease, the enforcer's
            # emergency busy-victim path -- reach this directly, bypassing
            # _find_pending_unload_ready_locked's own cluster exclusion).
            # Preserve the marker and hand off to the out-of-lock driver
            # instead of falling into _unload_engine, which raises for
            # kind="cluster".
            self._wake_cluster_unload_driver_locked()
            return False

        if entry.is_loading or entry.is_pinned or self._entry_is_busy(entry):
            return False

        reason = entry.pending_unload_reason
        entry.pending_unload_reason = None
        entry.abort_requested = False
        logger.warning(
            "Unloading pending model '%s' after activity drained (%s)",
            model_id,
            reason,
        )
        await self._unload_engine(model_id)
        return True

    def is_abort_requested(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        entry = self._entries.get(model_id)
        return bool(entry and entry.abort_requested)

    # ---- S4 D4: cluster entry coexistence ---------------------------------
    #
    # Cluster models are EngineEntry rows like any other (kind="cluster"),
    # with the pool as the single owner of their create/bind/teardown. The
    # binding lock-ordering rule: self._lock is never held across an E6
    # queue submit or any await on a formation job. Load runs as (a) claim
    # under the lock, (b) submit+await formation outside it, (c) bind under
    # the lock again. Teardown never runs inline: the only code that
    # unforms is `_cluster_unload_driver_loop`, an out-of-lock task the
    # pool owns, woken whenever a cluster entry is marked.

    def _force_mark_cluster_unload_locked(
        self, entry: EngineEntry, reason: str
    ) -> None:
        """Explicit operator/dashboard unload: ignores pin and busy state,
        matching `_unload_engine`'s documented immediate-abort semantics for
        local entries. Distinct from `_mark_pending_unload_locked`, whose
        pin/busy refusal is for background marks (TTL, admission eviction).

        Also clears a prior quarantine (S4 P2b): an explicit unload is the
        documented manual remedy for a divergence left by repeated
        `formation.unload` failures, so it gets a fresh set of retries.
        """
        entry.pending_unload_reason = reason
        entry.cluster_unload_force = True
        entry.cluster_teardown_failures = 0
        entry.cluster_teardown_quarantined = False
        self._wake_cluster_unload_driver_locked()

    def _cluster_unload_completion_event_locked(self, model_id: str) -> asyncio.Event:
        event = self._cluster_unload_events.get(model_id)
        if event is None:
            event = asyncio.Event()
            self._cluster_unload_events[model_id] = event
        return event

    def _signal_cluster_unload_complete_locked(self, model_id: str) -> None:
        event = self._cluster_unload_events.pop(model_id, None)
        if event is not None:
            event.set()

    def _wake_cluster_unload_driver_locked(self) -> None:
        """Create the driver task lazily on first cluster mark; wake it."""
        if self._cluster_unload_wake is None:
            self._cluster_unload_wake = asyncio.Event()
        if self._cluster_unload_task is None or self._cluster_unload_task.done():
            self._cluster_unload_task = asyncio.get_running_loop().create_task(
                self._cluster_unload_driver_loop(), name="cluster-unload-driver"
            )
        self._cluster_unload_wake.set()

    async def _drain_cluster_unload_driver(self) -> None:
        """Cancel and await the driver task (shutdown, mirrors
        `_drain_lease_release_tasks`)."""
        task, self._cluster_unload_task = self._cluster_unload_task, None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _claim_cluster_teardown_locked(self) -> str | None:
        """Caller holds `self._lock`. Pick one marked, idle, unclaimed
        `kind="cluster"` entry and claim it (`cluster_teardown_in_progress`)
        so a second driver iteration never double-claims it. Force-marked
        entries (explicit operator/dashboard unload) are claimed even while
        busy, matching `_unload_engine`'s immediate-abort semantics.
        """
        for mid, entry in self._entries.items():
            if entry.kind != "cluster" or entry.cluster_teardown_in_progress:
                continue
            if entry.cluster_teardown_quarantined:
                # S4 P2b: stop auto-retrying after repeated failures; an
                # explicit operator unload clears this and re-enters the
                # claim pool.
                continue
            if not entry.pending_unload_reason or entry.is_loading:
                continue
            if not entry.cluster_unload_force and self._entry_is_busy(entry):
                continue
            entry.cluster_teardown_in_progress = True
            return mid
        return None

    async def _teardown_cluster_entry(self, model_id: str) -> None:
        """Unform one claimed cluster entry outside the pool lock.

        On success, restore it to a plain unloaded `kind="local"` row and
        signal anything awaiting its completion event.

        On failure (S4 P2b): `formation.unload` raising must not revert the
        entry to a loadable local state -- pool and formation would then
        diverge (formation may still hold live ranks / an active_model)
        while a fresh local load of the same model_id becomes possible.
        The entry stays `kind="cluster"` and `_current_model_memory` keeps
        counting its share (the head's memory is still resident); the
        claim is released so the driver's own retry loop picks it back up.
        After `_CLUSTER_TEARDOWN_MAX_FAILURES` consecutive failures the
        entry is quarantined -- no further automatic retries, and
        `get_engine`/`load_cluster_model` refuse it with a clear pointer at
        the manual remedy (`POST /v1/cluster/models/unload`, which also
        clears the quarantine).
        """
        from .cluster.manager import get_cluster_manager

        unload_ok = False
        try:
            manager = get_cluster_manager()
            if manager is not None and manager.formation is not None:
                await manager.formation.unload(model_id)
            unload_ok = True
        except Exception:
            logger.exception("cluster: driver teardown failed for '%s'", model_id)
        finally:
            async with self._lock:
                entry = self._entries.get(model_id)
                if entry is not None and entry.kind == "cluster":
                    if unload_ok:
                        if entry.engine is not None:
                            self._current_model_memory = max(
                                0, self._current_model_memory - entry.estimated_size
                            )
                        entry.engine = None
                        entry.kind = "local"
                        if entry.cluster_original_estimated_size is not None:
                            entry.estimated_size = entry.cluster_original_estimated_size
                        entry.cluster_original_estimated_size = None
                        entry.cluster_head_share = None
                        entry.cluster_teardown_in_progress = False
                        entry.pending_unload_reason = None
                        entry.cluster_unload_force = False
                        entry.cluster_teardown_failures = 0
                        entry.cluster_teardown_quarantined = False
                        entry.last_access = 0.0
                        self._signal_cluster_unload_complete_locked(model_id)
                    else:
                        entry.cluster_teardown_in_progress = False
                        entry.cluster_teardown_failures += 1
                        if (
                            entry.cluster_teardown_failures
                            >= _CLUSTER_TEARDOWN_MAX_FAILURES
                        ):
                            entry.cluster_teardown_quarantined = True
                            logger.error(
                                "cluster: teardown for '%s' failed %d consecutive "
                                "times -- quarantining the entry; pool and "
                                "formation state may have diverged. Retry with "
                                "POST /v1/cluster/models/unload.",
                                model_id,
                                entry.cluster_teardown_failures,
                            )
                        # Do not signal completion here: an awaiter
                        # (unload_cluster_model / admission eviction) times
                        # out via `_await_cluster_teardown`'s own bound
                        # instead of being told the unload succeeded when
                        # it did not.
                else:
                    # Entry vanished or is no longer a cluster row (should
                    # not happen -- nothing else deletes/reverts a cluster
                    # entry mid-teardown); still release any waiter.
                    self._signal_cluster_unload_complete_locked(model_id)
            self._wake_process_memory_enforcer()

    async def _cluster_unload_driver_loop(self) -> None:
        assert self._cluster_unload_wake is not None
        while True:
            await self._cluster_unload_wake.wait()
            self._cluster_unload_wake.clear()
            while True:
                async with self._lock:
                    model_id = self._claim_cluster_teardown_locked()
                if model_id is None:
                    break
                try:
                    await self._teardown_cluster_entry(model_id)
                except Exception:
                    # A wedged request_unload/get_engine caller is worse
                    # than a logged failure -- keep the driver alive so a
                    # later wake can retry.
                    logger.exception("cluster: unload driver iteration failed")

    def _cluster_unload_bound(self) -> float:
        """Bound for awaiting a cluster teardown: the formation command
        timeout plus a margin (S4 D4 rev5), or a conservative fallback when
        no formation is wired up (should not happen for a real call, but a
        bound must exist either way).
        """
        from .cluster.manager import get_cluster_manager

        manager = get_cluster_manager()
        formation = manager.formation if manager is not None else None
        base = (
            formation.command_timeout
            if formation is not None
            else _CLUSTER_UNLOAD_FALLBACK_TIMEOUT_S
        )
        return float(base) + _CLUSTER_UNLOAD_MARGIN_S

    async def _await_cluster_teardown(
        self, model_id: str, event: asyncio.Event
    ) -> None:
        from .cluster.manager import ClusterFormationError

        bound = self._cluster_unload_bound()
        try:
            await asyncio.wait_for(event.wait(), timeout=bound)
        except TimeoutError as exc:
            raise ClusterFormationError(
                f"cluster teardown of '{model_id}' did not complete within "
                f"{bound:.0f}s"
            ) from exc

    def _claim_cluster_entry_locked(
        self, entry: EngineEntry, decision: PlacementDecision
    ) -> None:
        """Caller holds `self._lock`. Step (a) of D4's load choreography:
        mark the entry as forming. `estimated_size`/`_current_model_memory`
        are NOT touched here -- discovery's refresh already preserves any
        entry with `is_loading=True`, so nothing needs the accounting until
        bind (step c) actually succeeds.
        """
        entry.is_loading = True
        entry.loading_started_at = time.monotonic()
        entry.kind = "cluster"
        entry.cluster_original_estimated_size = entry.estimated_size
        entry.cluster_head_share = decision.per_rank_estimate

    def _revalidate_cluster_decision_locked(
        self, entry: EngineEntry, decision: PlacementDecision
    ) -> bool:
        """Claim-time revalidation of a precomputed decision (S4 D4 rev5):
        only the volatile predicate -- does the head still have room -- is
        rechecked, under the lock, with no I/O. Divisibility and worker fit
        do not change between the attempt that computed the decision and
        the attempt that acts on it, so they are not re-derived.
        """
        head = self.head_capacity()
        projected = head.current_model_memory + decision.per_rank_estimate
        return head.capacity_known and projected <= head.memory_ceiling

    def _rollback_cluster_entry_locked(self, model_id: str) -> None:
        """Caller holds `self._lock`. Undo a claim that never reached bind
        (formation submit/await raised, or bind found no engine)."""
        entry = self._entries.get(model_id)
        if entry is None or entry.kind != "cluster":
            return
        if entry.engine is not None:
            self._current_model_memory = max(
                0, self._current_model_memory - entry.estimated_size
            )
        entry.engine = None
        entry.kind = "local"
        if entry.cluster_original_estimated_size is not None:
            entry.estimated_size = entry.cluster_original_estimated_size
        entry.cluster_original_estimated_size = None
        entry.cluster_head_share = None
        entry.is_loading = False
        entry.loading_started_at = None

    @staticmethod
    def _raise_if_cluster_quarantined(entry: EngineEntry) -> None:
        """S4 P2b: a cluster entry whose teardown failed
        `_CLUSTER_TEARDOWN_MAX_FAILURES` times in a row is left quarantined
        (kind stays "cluster" so accounting isn't silently dropped, but
        nothing may treat it as loaded or reload it out from under a
        possibly still-live formation). Caller holds `self._lock`.
        """
        if entry.kind == "cluster" and entry.cluster_teardown_quarantined:
            raise ModelUnavailableError(
                entry.model_id,
                f"Model '{entry.model_id}' is quarantined after "
                f"{entry.cluster_teardown_failures} consecutive failed "
                "cluster teardown attempts; the engine pool and cluster "
                "formation state may have diverged. Retry with POST "
                "/v1/cluster/models/unload before loading this model again.",
            )

    async def load_cluster_model(
        self,
        model_id: str,
        prefer: Literal["auto", "local", "distributed"],
        *,
        decision: PlacementDecision | None = None,
    ) -> dict[str, Any]:
        """S4 D4: the ONE owner of cluster entry create/bind.

        Two callers: the routes path (`decision=None`) computes its own
        single placement decision internally; `get_engine`'s on-demand path
        supplies the decision it already computed, revalidated (not
        recomputed) at claim time -- a stale precomputed decision raises
        `PlacementStaleError` rather than silently acting on outdated occupancy.

        Lock-ordering: (a) claim under the lock, (b) submit+await the
        formation job outside it, (c) bind under the lock again. Never
        awaits formation while holding `self._lock`.
        """
        from .cluster.manager import ClusterError, get_cluster_manager
        from .cluster.placement import plan_placement, resolve_placement_inputs

        manager = get_cluster_manager()
        if manager is None or manager.role != "head" or manager.formation is None:
            raise ClusterError(
                404, "distributed formation is only available on the head"
            )

        if decision is None:
            async with self._lock:
                entry = self._entries.get(model_id)
                if entry is None:
                    raise ModelNotFoundError(model_id, list(self._entries.keys()))
                self._raise_if_cluster_quarantined(entry)
                if entry.engine is not None:
                    entry.last_access = time.time()
                    return {"model": model_id, "status": "ready", "decision": None}
                if entry.is_loading:
                    raise ModelLoadingError(model_id)
                model_path = entry.model_path
                model_type = entry.model_type
                est_size = entry.estimated_size

            # I/O outside the lock (S4 rev5 NB2): a single computation for
            # this call, driving the job that records it.
            _unused_est, model_config = resolve_placement_inputs(model_path)
            workers = self._cluster_worker_capacities(manager)

            async with self._lock:
                entry = self._entries.get(model_id)
                if entry is None:
                    raise ModelNotFoundError(model_id, list(self._entries.keys()))
                self._raise_if_cluster_quarantined(entry)
                if entry.engine is not None:
                    entry.last_access = time.time()
                    return {"model": model_id, "status": "ready", "decision": None}
                if entry.is_loading:
                    raise ModelLoadingError(model_id)
                decision = plan_placement(
                    model_id=model_id,
                    model_type=model_type,
                    est_size=est_size,
                    model_config=model_config,
                    head=self.head_capacity(),
                    workers=workers,
                    prefer=prefer,
                )
                if decision.mode != "distributed":
                    raise ClusterError(
                        424,
                        "; ".join(decision.reasons) or "placement rejected",
                    )
                self._claim_cluster_entry_locked(entry, decision)
        else:
            async with self._lock:
                entry = self._entries.get(model_id)
                if entry is None:
                    raise ModelNotFoundError(model_id, list(self._entries.keys()))
                self._raise_if_cluster_quarantined(entry)
                if entry.engine is not None:
                    entry.last_access = time.time()
                    return {
                        "model": model_id,
                        "status": "ready",
                        "decision": decision.to_dict(),
                    }
                if entry.is_loading:
                    raise ModelLoadingError(model_id)
                if not self._revalidate_cluster_decision_locked(entry, decision):
                    raise PlacementStaleError(model_id)
                self._claim_cluster_entry_locked(entry, decision)

        # (b) outside the lock: submit + await the formation job.
        try:
            result = await manager.formation.load(model_id, decision=decision)
        except BaseException:
            async with self._lock:
                self._rollback_cluster_entry_locked(model_id)
            raise

        # (c) re-acquire: bind the engine, clear is_loading.
        async with self._lock:
            entry = self._entries.get(model_id)
            engine = manager.formation.active_engine(model_id)
            if entry is None or engine is None:
                self._rollback_cluster_entry_locked(model_id)
                raise ClusterError(
                    500,
                    f"formation reported ready but no engine registered for "
                    f"{model_id!r}",
                )
            entry.engine = engine
            entry.estimated_size = entry.cluster_head_share or entry.estimated_size
            self._current_model_memory += entry.cluster_head_share or 0
            entry.is_loading = False
            entry.loading_started_at = None
            entry.last_access = time.time()

        result = dict(result)
        result["decision"] = decision.to_dict()
        return result

    async def unload_cluster_model(self, model_id: str) -> dict[str, Any]:
        """S4 D4: explicit distributed unload (`manager.unload_distributed`'s
        delegate target). Force-marks (ignores pin/busy, matching
        `_unload_engine`'s immediate-abort semantics) and awaits the
        driver's completion event, bounded by the formation command timeout
        + margin.
        """
        from .cluster.manager import ClusterError

        async with self._lock:
            entry = self._entries.get(model_id)
            if (
                entry is None
                or entry.kind != "cluster"
                or (entry.engine is None and not entry.is_loading)
            ):
                raise ClusterError(
                    404, f"no active distributed formation for {model_id!r}"
                )
            if entry.is_loading:
                raise ClusterError(409, f"model {model_id!r} is still forming")
            self._force_mark_cluster_unload_locked(entry, "operator unload")
            event = self._cluster_unload_completion_event_locked(model_id)

        await self._await_cluster_teardown(model_id, event)
        return {"model": model_id, "status": "unloaded"}

    async def request_unload(self, model_id: str) -> None:
        """S4 D4: the single out-of-pool unload dispatcher, kind-routed.

        Every caller outside `engine_pool.py` that used to call
        `_unload_engine` directly now calls this instead, so unloading a
        formed model gets a real driver-driven teardown (correct accounting
        restored) instead of `_unload_engine`'s raise.
        """
        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None or entry.engine is None:
                return
            kind = entry.kind
        if kind == "cluster":
            await self.unload_cluster_model(model_id)
            return
        await self._unload_engine(model_id)

    async def get_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        _lease: bool = False,
        runtime_settings: object | None = None,
    ) -> (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
    ):
        """
        Get or load engine for the specified model.

        This method implements pre-load memory checking:
        1. Check if model is already loaded -> return immediately
        2. Check if model is too large for memory limit -> raise error
        3. Evict LRU models until there's enough space
        4. Load the model
        5. Return the engine

        S4 D4/D5: this is a bounded outer retry loop (max
        ``_GET_ENGINE_MAX_ATTEMPTS``) around one attempt's lock scope(s).
        When a cluster head has ``cluster.auto_placement`` on, an attempt
        that needs to load may resolve to a distributed placement, or may
        pick a formed cluster entry as its LRU eviction victim; either way
        the pool releases its lock before awaiting formation (never holds
        it across an E6 queue submit or a formation command round-trip),
        then this loop retries. A retry that finds nothing to do raises
        the same errors this always raised.

        Args:
            model_id: The model ID to get engine for
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.
            runtime_settings: Optional transient settings used for this engine
                load. When its engine-construction signature differs from the
                currently loaded engine, the old engine is unloaded and the new
                variant is loaded without mutating persisted model settings.

        Returns:
            The loaded engine (BaseEngine for LLM, EmbeddingEngine for embeddings)

        Raises:
            ModelNotFoundError: If model is not discovered
            ModelTooLargeError: If model exceeds memory limit
            InsufficientMemoryError: If can't free enough memory (all pinned)
            ModelLoadingError: If model is already being loaded
        """
        last_decision: PlacementDecision | None = None
        for _attempt in range(_GET_ENGINE_MAX_ATTEMPTS):
            try:
                return await self._get_engine_attempt(
                    model_id,
                    force_lm=force_lm,
                    _lease=_lease,
                    runtime_settings=runtime_settings,
                )
            except _ClusterVictimPendingError as pending:
                await self._await_cluster_teardown(pending.model_id, pending.event)
            except _ClusterLoadPendingError as pending:
                last_decision = pending.decision
                with suppress(PlacementStaleError):
                    await self.load_cluster_model(
                        model_id, "auto", decision=pending.decision
                    )
            except _ConcurrentVariantRaceError:
                pass  # entry now reflects a concurrent load; next attempt re-evaluates
        detail = ""
        if last_decision is not None and last_decision.reasons:
            detail = f" Last placement decision: {'; '.join(last_decision.reasons)}."
        raise ModelLoadingError(
            model_id,
            f"Model '{model_id}' did not settle into a loaded state after "
            f"{_GET_ENGINE_MAX_ATTEMPTS} attempts; retry the request.{detail}",
        )

    async def _get_engine_attempt(
        self,
        model_id: str,
        *,
        force_lm: bool,
        _lease: bool,
        runtime_settings: object | None,
    ):
        """One ``get_engine`` attempt. Raises ``_ClusterVictimPendingError`` /
        ``_ClusterLoadPendingError`` (caught by the outer retry loop in
        ``get_engine``) when a cluster branch needs an out-of-lock await.
        """
        async with self._lock:
            entry = self._entries.get(model_id)
            if not entry:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))
            self._raise_if_cluster_quarantined(entry)
            unloaded_for_admission = False

            # Already loaded - just update access time. Cluster entries take
            # a zero-I/O fast path: the kind guard (S4 D4 rev3) skips the
            # whole variant/signature block below, since a formed model
            # never carries a runtime_settings_signature to reconcile.
            if entry.engine is not None:
                if entry.kind == "cluster":
                    if runtime_settings is not None:
                        from .cluster.engine import ClusterNonGoalError

                        raise ClusterNonGoalError(
                            f"Model '{model_id}' is served by a distributed "
                            "formation; runtime settings overrides are not "
                            "supported for a distributed model."
                        )
                    entry.last_access = time.time()
                    if _lease:
                        entry.in_use += 1
                    return entry.engine

                expected_signature = self._engine_runtime_signature(
                    model_id,
                    runtime_settings,
                )
                if (
                    expected_signature is not None
                    and entry.runtime_settings_signature is not None
                    and entry.runtime_settings_signature != expected_signature
                ) or (
                    runtime_settings is not None
                    and entry.runtime_settings_signature is None
                ):
                    self._raise_if_reload_busy(
                        entry,
                        "reload runtime settings variant",
                    )
                    logger.info(
                        "Runtime settings variant changed for %s; "
                        "unloading before reload.",
                        model_id,
                    )
                    await self._unload_engine(model_id)
                    unloaded_for_admission = True
                # If force_lm requested but current engine is VLM, unload and reload
                if (
                    entry.engine is not None
                    and force_lm
                    and isinstance(entry.engine, VLMBatchedEngine)
                ):
                    self._raise_if_reload_busy(entry, "reload as LM")
                    logger.info(
                        f"Unloading VLM engine for {model_id} "
                        f"(force_lm=True, reloading as LM)"
                    )
                    await self._unload_engine(model_id)
                    unloaded_for_admission = True
                elif entry.engine is not None:
                    self._validate_llm_engine_ready(model_id, entry.engine)
                    if entry.runtime_settings_signature is None:
                        entry.runtime_settings_signature = expected_signature
                    entry.last_access = time.time()
                    if _lease:
                        entry.in_use += 1
                    return entry.engine

            self._raise_if_model_path_missing_locked(model_id, entry)
            self._raise_if_load_failed(model_id, entry)
            if entry.is_loading:
                raise ModelLoadingError(model_id)

            cluster_manager = self._cluster_placement_gate()
            if cluster_manager is None:
                return await self._admit_and_load_locked(
                    model_id,
                    entry,
                    force_lm=force_lm,
                    runtime_settings=runtime_settings,
                    unloaded_for_admission=unloaded_for_admission,
                    allow_cluster_victim=False,
                    lease=_lease,
                )

            # Cluster gate active and a load decision is needed. est_size is
            # free (discovery already computed it); the config.json read
            # below is I/O and must never run under this lock (S4 rev5 NB2).
            model_path = entry.model_path
            model_type = entry.model_type
            est_size = entry.estimated_size

        from .cluster.placement import plan_placement, resolve_placement_inputs

        _unused_est, model_config = resolve_placement_inputs(model_path)
        workers = self._cluster_worker_capacities(cluster_manager)

        async with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))
            if entry.engine is not None:
                # Raced with a concurrent loader while the lock was released
                # for the config.json read above; the entry is now loaded.
                # S4 P2b: if this attempt asked for a specific runtime
                # settings variant, don't silently serve whatever variant
                # the racing loader landed -- let the outer retry
                # re-evaluate against the now-current entry instead.
                expected_signature = self._engine_runtime_signature(
                    model_id, runtime_settings
                )
                if (
                    expected_signature is not None
                    and entry.runtime_settings_signature is not None
                    and entry.runtime_settings_signature != expected_signature
                ) or (
                    runtime_settings is not None
                    and entry.runtime_settings_signature is None
                ):
                    raise _ConcurrentVariantRaceError(model_id)
                entry.last_access = time.time()
                if _lease:
                    entry.in_use += 1
                return entry.engine
            if entry.is_loading:
                raise ModelLoadingError(model_id)

            decision = plan_placement(
                model_id=model_id,
                model_type=model_type,
                est_size=est_size,
                model_config=model_config,
                head=self.head_capacity(),
                workers=workers,
                prefer="auto",
            )
            if decision.mode == "reject":
                raise InsufficientMemoryError(
                    required=est_size,
                    current=self._current_model_memory,
                    message=(
                        f"Cannot place '{model_id}' locally or distributed: "
                        f"{'; '.join(decision.reasons) or 'placement rejected'}"
                    ),
                )
            if decision.mode == "distributed":
                raise _ClusterLoadPendingError(decision)

            # mode == "local": exactly today's admission/eviction/load body,
            # now allowed to pick a formed cluster entry as its victim.
            return await self._admit_and_load_locked(
                model_id,
                entry,
                force_lm=force_lm,
                runtime_settings=runtime_settings,
                unloaded_for_admission=unloaded_for_admission,
                allow_cluster_victim=True,
                lease=_lease,
            )

    async def _admit_and_load_locked(
        self,
        model_id: str,
        entry: EngineEntry,
        *,
        force_lm: bool,
        runtime_settings: object | None,
        unloaded_for_admission: bool,
        allow_cluster_victim: bool,
        lease: bool,
    ):
        """Caller holds ``self._lock``. Today's admission/eviction/load body
        (unchanged local-only semantics), factored out so both the
        no-cluster-gate path and the cluster ``mode="local"`` path share it
        (S4 D4). May raise ``_ClusterVictimPendingError`` when
        ``allow_cluster_victim`` is True and the LRU victim is a formed
        cluster entry -- the pool marks it and hands off to the out-of-lock
        teardown driver instead of calling ``_unload_engine`` on it.
        """
        # Pre-load admission against the memory ceiling from the
        # process memory enforcer (min of static and dynamic). Try
        # evicting LRU non-pinned models first; if the model still
        # cannot fit after evicting everything available, raise.
        #
        # Eviction starts at the enforcer's *soft* watermark, not the
        # ceiling (#2319): the soft..ceiling band is exactly the
        # hard-pressure zone, and a second model admitted into it kept
        # both models resident through the load (swapping for minutes)
        # only to have the first request's prefill guard evict the old
        # one anyway. Evicting down to the same soft target *before*
        # the new weights allocate fixes the ordering; refusing a load
        # still requires exceeding the ceiling.
        #
        # ceiling == 0 means the guard is disabled or the enforcer is
        # not wired up. Eviction on model swap must not die with the
        # guard (#2290): fall back to the best-effort admission
        # ceiling (static, guard-independent) and keep evicting, but
        # never refuse the load under it — with the guard off the
        # user opted out of hard limits.
        ceiling = self._current_ceiling()
        best_effort = False
        if ceiling <= 0:
            ceiling = self._fallback_admission_ceiling()
            best_effort = ceiling > 0
        if ceiling > 0:
            # S4 P3: undo the dynamic ceiling's double subtraction of a
            # resident formation's head shard (`_cluster_head_share_total`).
            head_share_total = self._cluster_head_share_total()
            ceiling += head_share_total
            soft_target = self._admission_soft_target()
            if soft_target > 0:
                soft_target += head_share_total
            evict_target = min(soft_target, ceiling) if soft_target > 0 else ceiling
            evicted_any = unloaded_for_admission
            while True:
                # Consult the tracked accumulator alongside live memory:
                # after a model settles or idles, mx.get_active_memory() and
                # the process footprint can read well below the model's true
                # resident size, while _current_model_memory still reflects
                # the committed total. Using only live memory lets a second
                # large model load without evicting the first, over-
                # committing past the ceiling (#1623).
                current = max(
                    mx.get_active_memory(),
                    get_phys_footprint(),
                    self._current_model_memory,
                )
                projected = current + entry.estimated_size
                if projected <= evict_target:
                    break
                victim = self._find_lru_victim(include_cluster=allow_cluster_victim)
                if victim is not None:
                    victim_entry = self._entries[victim]
                    if victim_entry.kind == "cluster":
                        self._mark_pending_unload_locked(victim, "admission eviction")
                        event = self._cluster_unload_completion_event_locked(victim)
                        self._wake_cluster_unload_driver_locked()
                        raise _ClusterVictimPendingError(victim, event)
                    logger.info(
                        f"Evicting '{victim}' to fit '{model_id}' "
                        f"under the admission soft target "
                        f"({format_size(projected)} > "
                        f"{format_size(evict_target)})"
                    )
                    await self._unload_engine(victim)
                    evicted_any = True
                    continue
                if projected <= ceiling:
                    # Above the soft target with nothing left to
                    # evict, but still under the ceiling: admit. The
                    # soft target only decides when eviction starts
                    # (#2319); refusal keeps the ceiling-only
                    # contract.
                    if evict_target < ceiling:
                        logger.info(
                            f"Admitting '{model_id}' above the "
                            f"admission soft target with no idle "
                            f"model left to evict "
                            f"({format_size(projected)} > "
                            f"{format_size(evict_target)}, ceiling "
                            f"{format_size(ceiling)})"
                        )
                    break
                failure_current = current
                failure_projected = projected
                failure_label = "current"

                if evicted_any:
                    # Nothing else to evict after unloading at least one
                    # model in this get_engine() call. Before failing,
                    # re-test against the *tracked committed* baseline.
                    # The phys_footprint term folded into `current` is the
                    # macOS kernel ledger, which can still count
                    # reclaimable residue from models we just evicted.
                    # Pinned/in-use models that could not be evicted remain
                    # counted in _current_model_memory, preserving the
                    # #1623 undercount guard. Without a local eviction,
                    # keep trusting phys_footprint because it may be
                    # unrelated process pressure rather than model residue.
                    committed = max(mx.get_active_memory(), self._current_model_memory)
                    committed_projected = committed + entry.estimated_size
                    if committed_projected <= ceiling:
                        logger.info(
                            f"Admitting '{model_id}': committed baseline "
                            f"{format_size(committed_projected)} fits ceiling "
                            f"{format_size(ceiling)} "
                            f"(live footprint {format_size(projected)} included "
                            "reclaimable residue from evicted models)"
                        )
                        break
                    failure_current = committed
                    failure_projected = committed_projected
                    failure_label = "committed"

                if best_effort:
                    # Memory guard is off: evicting was all we could
                    # do. Admit over the static ceiling instead of
                    # refusing, matching the unguarded no-hard-limit
                    # contract.
                    logger.warning(
                        f"Loading '{model_id}' past the static memory "
                        f"ceiling with the memory guard disabled "
                        f"(projected {format_size(failure_projected)} > "
                        f"ceiling {format_size(ceiling)}, "
                        f"{failure_label} baseline) and nothing left to "
                        f"evict; the system may swap heavily."
                    )
                    break

                # Still over budget under the applicable baseline. Use
                # ModelTooLargeError when the model alone exceeds the
                # ceiling (no chance of fitting), InsufficientMemoryError
                # when current usage leaves no room.
                if entry.estimated_size > ceiling:
                    binding, advice = self._ceiling_binding_and_advice(
                        ceiling=ceiling,
                        current=failure_current,
                        tail="use a smaller model",
                    )
                    raise ModelTooLargeError(
                        model_id,
                        entry.estimated_size,
                        ceiling,
                        binding=binding,
                        advice=advice,
                    )
                binding, advice = self._ceiling_binding_and_advice(
                    ceiling=ceiling,
                    current=failure_current,
                    tail="unload another model",
                )
                label = f"{binding} memory ceiling" if binding else "memory ceiling"
                raise InsufficientMemoryError(
                    required=entry.estimated_size,
                    current=failure_current,
                    message=(
                        f"Cannot load {model_id}: projected memory "
                        f"{format_size(failure_projected)} would exceed "
                        f"the {label} {format_size(ceiling)} "
                        f"({failure_label}: {format_size(failure_current)}, "
                        f"model: {format_size(entry.estimated_size)}). "
                        f"{advice or DEFAULT_CEILING_ADVICE}."
                    ),
                )

        # Now load the model
        await self._load_engine(
            model_id,
            force_lm=force_lm,
            runtime_settings=runtime_settings,
        )

        loaded = self._entries[model_id]
        self._validate_llm_engine_ready(model_id, loaded.engine)
        if lease:
            loaded.in_use += 1
        return loaded.engine

    def _cluster_placement_gate(self) -> Any | None:
        """The active ``ClusterManager`` when auto-placement should run for
        this ``get_engine`` call (S4 D5): head role + ``cluster.auto_placement``
        + an initialized formation. ``None`` otherwise, so the caller takes
        exactly today's local-only path (``role=off``/worker/flag-off are
        byte-for-byte unchanged).
        """
        from .cluster.manager import get_cluster_manager

        manager = get_cluster_manager()
        if manager is None or manager.role != "head":
            return None
        if not bool(getattr(manager.settings, "auto_placement", True)):
            return None
        if manager.formation is None:
            return None
        return manager

    @staticmethod
    def _cluster_worker_capacities(manager: Any) -> list[NodeCapacity]:
        from .cluster.placement import worker_node_capacity

        workers = []
        for candidate in manager.state.members:
            live = manager.liveness(candidate.id)
            node_state = manager.node_state(candidate.id)
            if live is not None and live.status == "active" and node_state is not None:
                workers.append(worker_node_capacity(candidate.id, node_state))
        return workers

    async def _release_engine_lease(self, model_id: str) -> None:
        async with self._lock:
            e = self._entries.get(model_id)
            if e is not None and e.in_use > 0:
                e.in_use -= 1
            await self._unload_pending_if_idle_locked(model_id)

    def _finish_lease_release_task(self, task: asyncio.Task[None]) -> None:
        self._lease_release_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning("Engine lease release task was cancelled")
        except Exception:
            logger.exception("Engine lease release task failed")

    async def _drain_lease_release_tasks(self) -> None:
        while self._lease_release_tasks:
            tasks = tuple(self._lease_release_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def release_engine(self, model_id: str) -> None:
        """Release one in-use lease even if the caller is cancelled.

        ASGI disconnect cancellation can arrive while this release is waiting
        for the pool lock. Run the lock-taking operation in its own task so the
        lease still drains after the cancelled request task exits.
        """
        task = asyncio.create_task(
            self._release_engine_lease(model_id),
            name=f"engine-lease-release:{model_id}",
        )
        self._lease_release_tasks.add(task)
        task.add_done_callback(self._finish_lease_release_task)
        await asyncio.shield(task)

    async def unload_if_idle_unpinned(self, model_id: str) -> bool:
        """Unload a loaded engine only when it is idle and not pinned."""
        async with self._lock:
            entry = self._entries.get(model_id)
            if (
                entry is None
                or entry.engine is None
                or entry.is_loading
                or entry.is_pinned
                or entry.in_use > 0
            ):
                return False

            if self._entry_has_active_requests(entry):
                entry.last_access = time.time()
                return False

            if entry.kind == "cluster":
                # S4 D4: fire-and-forget mark -- the entry is confirmed idle
                # and unpinned above, so the mark always installs. The driver
                # does the actual (out-of-lock) teardown; this call site
                # (api/markitdown_pdf_fallback.py) doesn't need to await it.
                self._mark_pending_unload_locked(model_id, "unload_if_idle_unpinned")
                return True

            await self._unload_engine(model_id)
            return True

    @asynccontextmanager
    async def acquire(self, model_id: str, force_lm: bool = False):
        """Acquire an engine with an atomic in-use lease.

        The lease is taken under the pool lock at acquire time and always
        released in finally, so the engine cannot be evicted mid-request even
        on exception.
        """
        engine = await self.get_engine(model_id, force_lm=force_lm, _lease=True)
        try:
            yield engine
        finally:
            await self.release_engine(model_id)

    def _find_lru_victim(self, *, include_cluster: bool = False) -> str | None:
        """
        Find the least recently used non-pinned loaded model.

        Skips models with active inference requests to avoid interrupting
        in-flight generation. ``include_cluster`` (S4 D4) defaults to False
        so every existing caller -- the enforcer's pressure loop included --
        is unchanged; only ``get_engine``'s admission eviction opts in,
        since a formed cluster entry needs the out-of-lock teardown driver
        rather than ``_unload_engine``.

        Returns:
            Model ID of the LRU victim, or None if no evictable model found
        """
        candidates = []
        for mid, e in self._entries.items():
            if e.engine is None or e.is_pinned:
                continue
            if not include_cluster and e.kind == "cluster":
                continue
            if e.in_use > 0:
                continue
            if self._entry_has_active_requests(e):
                logger.debug(f"Skipping victim '{mid}': has active requests")
                continue
            candidates.append((e.last_access, mid))
        if not candidates:
            return None
        candidates.sort()  # Sort by last_access (oldest first)
        return candidates[0][1]

    async def _unload_other_dflash_engines(self, model_id: str) -> None:
        """Unload other idle DFlash engines before starting a new one.

        dflash-mlx installs target hooks on shared Python classes and owns a
        process-global runtime cache manager, so multiple loaded DFlash engines
        can leak state across model switches.
        """
        victims: list[str] = []
        blocked: list[str] = []
        for mid, e in self._entries.items():
            if mid == model_id or e.engine is None:
                continue
            if type(e.engine).__name__ != "DFlashEngine":
                continue
            if e.is_loading or e.in_use > 0:
                blocked.append(mid)
                continue
            try:
                if e.engine.has_active_requests():
                    blocked.append(mid)
                    continue
            except AttributeError:
                pass
            if e.is_pinned:
                blocked.append(f"{mid} (pinned)")
                continue
            victims.append(mid)

        if blocked:
            raise RuntimeError(
                "Cannot load DFlash model "
                f"'{model_id}' while another DFlash engine is active: "
                f"{', '.join(blocked)}"
            )

        for victim in victims:
            logger.info(
                "Unloading DFlash model '%s' before loading '%s' because "
                "dflash runtime hooks/cache are process-global",
                victim,
                model_id,
            )
            await self._unload_engine(victim)

    @staticmethod
    def _resolve_scheduler_from_engine(engine: object) -> object | None:
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            return scheduler
        try:
            return engine._engine.engine.scheduler  # type: ignore[attr-defined]
        except AttributeError:
            return None

    def _is_idle_for_prefill_eviction(self, entry: EngineEntry) -> bool:
        engine = entry.engine
        if engine is None or entry.is_pinned or entry.is_loading or entry.in_use > 0:
            return False
        if entry.kind == "cluster":
            # S4 D4: ineligible -- unforming a live formation for transient
            # prefill headroom is disproportionate.
            return False
        if self._entry_has_active_requests(entry):
            return False

        scheduler = self._resolve_scheduler_from_engine(engine)
        if scheduler is None:
            return True
        for attr in ("running", "waiting", "prefilling", "requests"):
            value = getattr(scheduler, attr, None)
            if value:
                return False
        return True

    def _find_lru_prefill_eviction_victim(self, *, exclude_model_id: str) -> str | None:
        candidates = []
        for mid, entry in self._entries.items():
            if mid == exclude_model_id:
                continue
            if self._is_idle_for_prefill_eviction(entry):
                candidates.append((entry.last_access, mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    async def _evict_idle_lru_for_prefill(
        self,
        exclude_model_id: str,
        eviction_request: object,
    ) -> bool:
        """Evict idle LRU models until the requested prefill step should fit."""
        target = int(getattr(eviction_request, "target_cap_bytes", 0) or 0)
        predicted = int(getattr(eviction_request, "predicted_transient_bytes", 0) or 0)
        request_id = str(getattr(eviction_request, "request_id", ""))
        if target <= 0 or predicted <= 0:
            return False

        evicted_any = False
        reclaim_attempted = False
        async with self._lock:
            while True:
                current = max(
                    mx.get_active_memory(),
                    get_phys_footprint(),
                    self._current_model_memory,
                )
                if current + predicted <= target:
                    # A model eviction and/or the pooled-buffer reclaim below
                    # created enough headroom; signal the caller to re-admit.
                    # reclaim_attempted stands in for "the reclaim freed
                    # memory": the helper's own footprint delta is
                    # process-wide and can be masked by concurrent
                    # allocation, but reaching this check with headroom
                    # after an attempt means admission will now succeed.
                    return evicted_any or reclaim_attempted

                victim = self._find_lru_prefill_eviction_victim(
                    exclude_model_id=exclude_model_id
                )
                if victim is None:
                    # No idle model left to evict -- the "No idle model
                    # evicted" case that used to reject outright even when
                    # tens of GB were reclaimable. Try the cheaper reclaim
                    # once before giving up: return MLX's pooled Metal buffers
                    # (freed by finished requests but still cached, so
                    # get_phys_footprint stays high) to the OS on the
                    # requesting engine's own MLX thread, then let the loop
                    # re-measure.
                    if not reclaim_attempted:
                        reclaim_attempted = True
                        await self._reclaim_pooled_buffers_for_prefill(
                            exclude_model_id, request_id
                        )
                        # Re-measure regardless of the reported delta: the
                        # helper measures a process-wide footprint, so
                        # concurrent allocation on another engine can mask a
                        # real reclaim as 0 bytes freed. The loop re-checks
                        # the target with a fresh reading; reclaim_attempted
                        # keeps this branch from running twice.
                        continue
                    if evicted_any:
                        logger.info(
                            "Prefill eviction for request %s stopped with no "
                            "more idle victims (current=%s, predicted=%s, "
                            "target=%s)",
                            request_id,
                            format_size(current),
                            format_size(predicted),
                            format_size(target),
                        )
                    return evicted_any

                logger.info(
                    "Evicting idle model '%s' for prefill headroom on '%s' "
                    "(request=%s, projected=%s > target=%s)",
                    victim,
                    exclude_model_id,
                    request_id,
                    format_size(current + predicted),
                    format_size(target),
                )
                await self._unload_engine(victim)
                evicted_any = True

    @staticmethod
    def _resolve_engine_core_from_engine(engine: object) -> object | None:
        """Resolve the EngineCore owning an entry's scheduler and MLX thread."""
        if getattr(engine, "scheduler", None) is not None:
            return engine
        try:
            return engine._engine.engine  # type: ignore[attr-defined]
        except AttributeError:
            return None

    async def _reclaim_pooled_buffers_for_prefill(
        self, model_id: str, request_id: str
    ) -> int:
        """Return MLX's pooled Metal buffers to the OS; report bytes freed.

        A warm server's resident baseline creeps between requests: finished
        requests free their KV / activation arrays into MLX's buffer *cache*
        (retained for reuse) rather than handing the pages back to the OS, so
        ``get_phys_footprint`` -- the resident figure the prefill guard reads
        -- stays high even though the bytes are reclaimable.

        The clear runs on the requesting engine's own MLX thread through the
        scheduler's ``_reclaim_prefill_headroom``, which synchronizes that
        engine's generation stream under ``_mx_buffer_access_lock`` before
        clearing (issues #300, #888, #1106) -- clearing from any other thread
        could release cached buffers still referenced by in-flight
        ``mx.async_eval`` command buffers. The engine's step loop is parked
        awaiting this eviction callback, so its executor is free. A failing
        reclaim is contained (#435 class): the request is then rejected
        exactly as if nothing had been reclaimable. ``clear_cache`` releases
        only unused cached buffers, so arrays an in-flight request still
        references are never touched, and the shared hot / prefix cache is
        left intact (it is SSD-backed and reclaimed separately by the
        enforcer).

        Returns:
            Bytes handed back to the OS (``get_phys_footprint`` delta, >= 0).
        """
        entry = self._entries.get(model_id)
        engine = entry.engine if entry is not None else None
        core = (
            self._resolve_engine_core_from_engine(engine)
            if engine is not None
            else None
        )
        scheduler = getattr(core, "scheduler", None)
        reclaim = getattr(scheduler, "_reclaim_prefill_headroom", None)
        executor = getattr(core, "_mlx_executor", None)
        if not callable(reclaim) or executor is None:
            # Engine mid-teardown (executor dropped at close) or an engine
            # shape without the scheduler helper: skip -- reject as before.
            return 0

        def _reclaim_on_engine_thread() -> None:
            gc.collect()
            reclaim()

        before = get_phys_footprint()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(executor, _reclaim_on_engine_thread)
        except Exception as e:
            logger.warning(
                "Pooled-buffer reclaim failed for prefill request %s: %s",
                request_id,
                e,
            )
            return 0
        freed = max(0, before - get_phys_footprint())
        if freed > 0:
            logger.info(
                "Reclaimed %s of pooled Metal buffers for prefill request %s "
                "(no idle model to evict)",
                format_size(freed),
                request_id,
            )
        return freed

    def _other_entries_serving(self, model_id: str) -> bool:
        """True when any other entry is serving or loading.

        Used by the settle barrier in ``_unload_engine``: the barrier's
        freed-memory check is a delta of the process-global
        ``mx.get_active_memory()`` gauge, which only measures THIS unload
        while no other engine is allocating concurrently. A loading entry
        (``is_loading=True``, ``engine`` still None) allocates weights at
        full speed, so it must count as concurrent activity too — else the
        barrier burns its rounds against a gauge that can even read
        negative and logs a bogus timeout (#2312).
        """
        # Snapshot the items: admin unload routes call _unload_engine without
        # the pool lock, so discover_models() can mutate _entries mid-iteration.
        for mid, e in list(self._entries.items()):
            if mid == model_id:
                continue
            if e.is_loading:
                return True
            if e.engine is None:
                continue
            if e.in_use > 0:
                return True
            if self._entry_has_active_requests(e):
                return True
        return False

    async def _unload_engine(self, model_id: str) -> None:
        """
        Immediately stop and unload an engine with memory settle barrier.

        After stopping the engine, polls mx.get_active_memory() to verify
        Metal buffers are actually reclaimed before updating the memory
        tracking counter.

        Args:
            model_id: The model ID to unload
        """
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            return
        if entry.kind == "cluster":
            # S4 D4 rev4: assert-unreachable, never a silent no-op. Every
            # caller of this method is guarded by kind dispositions
            # (request_unload's dispatch, _find_lru_victim's default
            # include_cluster=False, the prefill/enforcer selection filters,
            # the variant-branch kind guard) -- reaching here with a cluster
            # entry is a bug, and it must be loud rather than silently
            # skipping teardown.
            raise RuntimeError(
                f"_unload_engine() reached a cluster entry ('{model_id}'); "
                "cluster teardown must go through request_unload() / "
                "unload_cluster_model() / the unload driver (S4 D4), never "
                "this method directly"
            )

        logger.info(f"Unloading model: {model_id} (immediate abort)")
        pre_unload_active = mx.get_active_memory()

        try:
            await entry.engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping engine for {model_id}: {e}")

        # #1595: the immediate-abort stop() above tears the engine down without the normal
        # per-request completion callbacks, so a non-streaming engine's active_requests
        # counter can leak a phantom count (a stale engine then looks permanently busy).
        # Reset it on teardown so has_active_requests() and the status API stay consistent.
        reset = getattr(entry.engine, "_reset_activity_tracking", None)
        if callable(reset):
            try:
                reset()
            except Exception as e:
                logger.warning(f"Error resetting activity counter for {model_id}: {e}")

        # Yield to the event loop before dropping the engine reference.
        #
        # When abort_all_requests() fires before _unload_engine(), it sets
        # asyncio Events for each active request.  Server-side streaming
        # generators are then scheduled in the asyncio ready queue, but they
        # cannot run until the event loop gets control.  EngineCore.close()
        # (called inside stop()) blocks the event loop with synchronous
        # .result() calls on the MLX executor -- scheduler.shutdown() and
        # scheduler.deep_reset() -- so those generators are still suspended
        # when stop() returns.
        #
        # If we set entry.engine = None and call gc.collect() immediately,
        # the generators are still alive with a local 'engine' variable
        # referencing the BatchedEngine, keeping its refcount above zero.
        # The model's ~20 GB of MLX weight tensors therefore remain "active"
        # in Metal memory, the settle barrier times out, and subsequent load
        # attempts fail with 507 because the ceiling is still exceeded.
        #
        # A few asyncio.sleep(0) calls drain the ready queue -- generator
        # tear-down is at most a few frames deep -- so that by the time we
        # clear entry.engine and run gc.collect(), no coroutine frame holds
        # a stale engine reference.
        for _ in range(5):
            await asyncio.sleep(0)

        # Clear engine reference before settle barrier
        entry.engine = None
        entry.last_access = 0.0
        entry.actual_size = None
        entry.abort_requested = False
        entry.pending_unload_reason = None
        entry.runtime_settings_signature = None

        # Force garbage collection to release memory.
        # Run mx.clear_cache on the global MLX executor to avoid concurrent
        # Metal operations with running engines. See issue #85.
        # Synchronize before clearing to prevent releasing Metal buffers
        # still referenced by in-flight command buffers. See issue #300.
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )

        # Memory settle barrier: poll actual freed memory instead of
        # trusting the cumulative _current_model_memory estimate.
        # Scale tolerance with model size: estimated_size includes a 5%
        # overhead factor (model_discovery.py) that may not be reflected in
        # actual freed memory. Use 2 GB floor for small models. See #768.
        settle_tolerance = max(2 * 1024**3, int(entry.estimated_size * 0.05))
        min_expected_freed = max(0, entry.estimated_size - settle_tolerance)
        settled = False
        settle_indeterminate = False
        for _settle_round in range(10):
            active_now = mx.get_active_memory()
            actual_freed = pre_unload_active - active_now
            if actual_freed >= min_expected_freed:
                settled = True
                logger.debug(
                    f"Settle round {_settle_round + 1} for '{model_id}': "
                    f"freed={format_size(actual_freed)} "
                    f"(need>={format_size(min_expected_freed)}) - settled"
                )
                break
            if self._other_entries_serving(model_id):
                # actual_freed is a delta of the process-global MLX gauge,
                # so while another engine allocates (prefill/KV growth) the
                # amount freed by THIS unload is unmeasurable — the delta can
                # even read negative. Burning settle rounds here serializes
                # gc/synchronize/clear_cache against live decode for seconds,
                # under memory pressure, with the enforcer holding the pool
                # lock. Bail out instead: pre-load admission re-reads the
                # live gauge, so nothing downstream trusts this sample.
                settle_indeterminate = True
                logger.info(
                    f"Settle for '{model_id}' indeterminate under concurrent "
                    f"activity (freed={format_size(actual_freed)}, "
                    f"need>={format_size(min_expected_freed)}); skipping "
                    f"settle wait"
                )
                break
            logger.debug(
                f"Settle round {_settle_round + 1} for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)}) - retry"
            )
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )

        # Release memory tracking AFTER barrier
        self._current_model_memory -= entry.estimated_size

        if settled:
            logger.info(
                f"Unloaded model: {model_id}, "
                f"freed={format_size(actual_freed)} "
                f"(expected>={format_size(min_expected_freed)}), "
                f"active_memory: {format_size(active_now)} (settled)"
            )
        elif settle_indeterminate:
            # Settle wait skipped (logged above). Emergency reclaim is
            # deliberately skipped too: its gc + synchronize + clear_cache
            # rounds would stall the live engines that made the measurement
            # indeterminate in the first place. Recovery is not lost:
            # _wake_process_memory_enforcer() below triggers an immediate
            # enforcer re-poll, and pre-load admission re-reads the live gauge
            # alongside the tracked accumulator (the #1623 max() in
            # get_engine), so any unreleased memory stays visible to both.
            pass
        else:
            # Barrier timed out - try emergency reclaim
            logger.warning(
                f"Settle barrier timed out for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)})"
            )
            for _ in range(3):
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                await asyncio.sleep(1.0)
            active_after = mx.get_active_memory()
            if active_after > self._current_model_memory + 5 * 1024**3:
                logger.error(
                    f"Emergency reclaim failed for '{model_id}': "
                    f"active_memory={format_size(active_after)} "
                    f"exceeds safe threshold "
                    f"({format_size(self._current_model_memory + 5 * 1024**3)})"
                )
            else:
                logger.info(
                    f"Emergency reclaim succeeded: "
                    f"active_memory={format_size(active_after)}"
                )

        self._wake_process_memory_enforcer()
        self._invalidate_cluster_node_state_cache()

    def _schedule_failed_load_reclaim(
        self, model_id: str, pre_load_memory: int
    ) -> None:
        """Reclaim memory left behind by a failed model load.

        When a load raises partway through (e.g. weights loaded, processor
        construction failed), the weights are often reachable only via the
        propagating exception's traceback frames. Spawn a background task
        that waits briefly for the exception to be handled and dropped, then
        runs gc + synchronize + clear_cache rounds until the live memory
        reading returns near its pre-load level (or rounds are exhausted).
        """

        async def _reclaim() -> None:
            loop = asyncio.get_running_loop()
            # 2 GB slack over the pre-load level mirrors the unload settle
            # barrier's small-model tolerance floor.
            target = pre_load_memory + 2 * 1024**3
            current = 0
            for _round in range(6):
                await asyncio.sleep(0.5 if _round == 0 else 1.0)
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                current = max(mx.get_active_memory(), get_phys_footprint())
                if current <= target:
                    logger.info(
                        f"Reclaimed memory after failed load of '{model_id}': "
                        f"current={format_size(current)} "
                        f"(pre-load={format_size(pre_load_memory)})"
                    )
                    self._wake_process_memory_enforcer()
                    return
            logger.warning(
                f"Post-failed-load reclaim for '{model_id}' did not settle: "
                f"current={format_size(current)} "
                f"(pre-load={format_size(pre_load_memory)}). A server restart "
                f"may be required to release the leaked memory."
            )
            self._wake_process_memory_enforcer()

        # Keep a reference so the task isn't garbage-collected mid-flight;
        # one slot suffices -- a newer failure supersedes the previous task.
        self._failed_load_reclaim_task = asyncio.get_running_loop().create_task(
            _reclaim()
        )

    async def _load_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        runtime_settings: object | None = None,
    ) -> None:
        """
        Load an engine for the specified model.

        Args:
            model_id: The model ID to load
            force_lm: Force loading as BatchedEngine even for VLM models.

        Raises:
            ModelLoadingError: If model is already being loaded
        """
        entry = self._entries[model_id]
        if entry.is_loading:
            raise ModelLoadingError(model_id)

        entry.is_loading = True
        entry.loading_started_at = time.monotonic()
        self._wake_process_memory_enforcer(active=True)
        load_started_at = entry.loading_started_at
        load_completed = False
        entry_detached = False
        entry.abort_loading = False
        pre_load_memory = max(mx.get_active_memory(), get_phys_footprint())
        try:
            effective_type = entry.engine_type
            if force_lm and effective_type == "vlm":
                effective_type = "batched"
                logger.info(f"Loading model as LM (force_lm=True): {model_id}")
            else:
                logger.info(f"Loading model: {model_id}")

            # Retrieve per-model settings for post-load transforms
            model_settings = runtime_settings
            if model_settings is None and self._settings_manager is not None:
                model_settings = self._settings_manager.get_settings(model_id)

            # Wire the correct model_id / model_path into the shared scheduler
            # config so every engine (Batched/VLM/DFlash/Embedding) sees the
            # right values when it builds `SchedulerConfig` internally.
            self._scheduler_config.model_name = model_id
            self._scheduler_config.model_path = entry.model_path

            # Native MTP forces LM-only dispatch even for VLM models. Vision
            # encoder weights are ignored because the patched mtp_forward only
            # exists on the language model path. mtp_enabled was already
            # validated as mutually exclusive with dflash in
            # metal-knowledge: with the mlx-vlm runtime MTP patch (see
            # omlx/patches/mlx_vlm_mtp/qwen35_moe_vlm_runtime.py) VLM models
            # can run MTP natively while keeping vision intact. The old
            # force-LM-dispatch shortcut here is obsolete for patched
            # model families; let VLMBatchedEngine handle MTP-enabled VLMs.
            pass

            # Check if DFlash is enabled -- takes priority over engine type
            # since DFlash has its own model loading pipeline
            engine = None
            if model_settings is not None:
                dflash_enabled = getattr(model_settings, "dflash_enabled", False)
                dflash_draft = getattr(model_settings, "dflash_draft_model", None)
                if (
                    dflash_enabled
                    and dflash_draft
                    and self._entry_is_diffusion_model(entry)
                ):
                    logger.warning(
                        "DFlash is not supported for diffusion models; "
                        "loading %s with its native VLM engine",
                        model_id,
                    )
                elif dflash_enabled and dflash_draft:
                    try:
                        from .engine.dflash import DFlashEngine

                        engine = DFlashEngine(
                            model_name=entry.model_path,
                            draft_model_path=dflash_draft,
                            draft_quant_enabled=getattr(
                                model_settings, "dflash_draft_quant_enabled", False
                            ),
                            draft_quant_weight_bits=getattr(
                                model_settings, "dflash_draft_quant_weight_bits", 4
                            ),
                            draft_quant_activation_bits=getattr(
                                model_settings, "dflash_draft_quant_activation_bits", 16
                            ),
                            draft_quant_group_size=getattr(
                                model_settings, "dflash_draft_quant_group_size", 64
                            ),
                            model_settings=model_settings,
                            fallback_engine_type=effective_type,
                            scheduler_config=self._scheduler_config,
                            omlx_ssd_cache_dir=getattr(
                                self._scheduler_config, "paged_ssd_cache_dir", None
                            ),
                        )
                        logger.info(
                            f"DFlash enabled for {model_id}, draft={dflash_draft}"
                        )
                    except ImportError:
                        logger.warning(
                            f"DFlash enabled for {model_id} but dflash-mlx is not installed. "
                            f"Falling back to default engine."
                        )
                    except Exception as e:
                        logger.warning(
                            f"DFlash init failed for {model_id}: {e}. "
                            f"Falling back to default engine."
                        )

            # Per-model trust_remote_code (security opt-in, issue #926).
            # When unset, defaults to False -- repos with custom modeling_*.py
            # will fail to load until the user explicitly toggles this on
            # in the admin UI's model settings modal.
            trc = (
                bool(getattr(model_settings, "trust_remote_code", False))
                if model_settings
                else False
            )

            async def prefill_eviction_callback(
                eviction_request: object,
                *,
                _model_id: str = model_id,
            ) -> bool:
                return await self._evict_idle_lru_for_prefill(
                    exclude_model_id=_model_id,
                    eviction_request=eviction_request,
                )

            # Create engine based on engine type (if DFlash not active)
            if engine is None:
                if effective_type == "embedding":
                    engine = EmbeddingEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                    )
                elif effective_type == "reranker":
                    engine = RerankerEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                    )
                elif effective_type == "vlm":
                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                elif entry.engine_type == "audio_stt":
                    engine = STTEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_tts":
                    engine = TTSEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_sts":
                    engine = STSEngine(
                        model_name=entry.model_path,
                        config_model_type=entry.config_model_type,
                    )
                else:
                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        prefill_eviction_callback=prefill_eviction_callback,
                    )

            _is_dflash_engine = (
                engine is not None and type(engine).__name__ == "DFlashEngine"
            )
            if _is_dflash_engine:
                await self._unload_other_dflash_engines(model_id)

            try:
                await engine.start()
            except Exception as start_error:
                if _is_dflash_engine:
                    # DFlash engine failed to start -- fall back to the
                    # model's natural engine type (VLM or Batched)
                    logger.warning(
                        f"DFlash start failed for {model_id}: {start_error}. "
                        f"Falling back to {effective_type} engine."
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    if effective_type == "vlm":
                        engine = VLMBatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                            prefill_eviction_callback=prefill_eviction_callback,
                        )
                    else:
                        engine = BatchedEngine(
                            model_name=entry.model_path,
                            trust_remote_code=trc,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                            prefill_eviction_callback=prefill_eviction_callback,
                        )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"DFlash load failed: {start_error}; "
                            f"{effective_type} fallback also failed: {fallback_error}"
                        ) from start_error
                    logger.info(
                        f"Successfully loaded {model_id} as {effective_type} "
                        f"(fallback from DFlash)"
                    )

                elif force_lm and entry.engine_type == "vlm":
                    # force_lm created a BatchedEngine but mlx-lm can't
                    # load this VLM model -- fall back to VLMBatchedEngine.
                    logger.warning(
                        f"LM loading failed for VLM model {model_id} "
                        f"(force_lm=True), falling back to VLM engine: "
                        f"{start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    engine = VLMBatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"LM load failed (force_lm=True): {start_error}; "
                            f"VLM fallback also failed: {fallback_error}"
                        ) from start_error

                    logger.info(
                        f"Successfully loaded {model_id} as VLM "
                        f"(fallback from force_lm)"
                    )
                elif entry.engine_type == "vlm":
                    # VLM loading failed -- fall back to LLM (BatchedEngine)
                    logger.warning(
                        f"VLM loading failed for {model_id}, "
                        f"falling back to LLM: {start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    engine = BatchedEngine(
                        model_name=entry.model_path,
                        trust_remote_code=trc,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                        prefill_eviction_callback=prefill_eviction_callback,
                    )
                    try:
                        await engine.start()
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"VLM load failed: {start_error}; "
                            f"LLM fallback also failed: {fallback_error}"
                        ) from start_error

                    entry.model_type = "llm"
                    entry.engine_type = "batched"
                    logger.info(
                        f"Successfully loaded {model_id} as LLM " f"(fallback from VLM)"
                    )
                else:
                    raise

            # Check if memory enforcer requested abort during loading
            if entry.abort_loading:
                logger.warning(f"Model load aborted by memory enforcer: {model_id}")
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(f"Error stopping aborted engine for {model_id}: {e}")
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    model_id,
                    f"Model '{model_id}' load aborted: process memory limit exceeded",
                )

            self._validate_llm_engine_ready(model_id, engine)
            entry.engine = engine
            entry.last_access = time.time()
            self._current_model_memory += entry.estimated_size
            load_completed = True
            self._clear_load_failure(entry)

            # VLM MTP: load MTP drafter (gemma4_assistant or qwen3_5_mtp) and attach to engine.
            # Fail-soft -- drafter load issues never block the target engine.
            if (
                model_settings is not None
                and getattr(model_settings, "vlm_mtp_enabled", False)
                and getattr(model_settings, "vlm_mtp_draft_model", None)
                and hasattr(engine, "set_vlm_mtp_drafter")
            ):
                drafter_id = model_settings.vlm_mtp_draft_model
                drafter_entry = self._entries.get(drafter_id)
                drafter_path = drafter_entry.model_path if drafter_entry else drafter_id

                def _load_drafter_sync(path: str = drafter_path):
                    from .speculative.vlm_mtp import load_vlm_mtp_drafter

                    return load_vlm_mtp_drafter(path)

                loop = asyncio.get_running_loop()
                try:
                    drafter = await loop.run_in_executor(
                        get_mlx_executor(), _load_drafter_sync
                    )
                except Exception as e:
                    logger.warning(
                        f"VLM MTP drafter load raised for {model_id} "
                        f"(drafter={drafter_id}): {e} -- toggle ignored"
                    )
                    drafter = None
                if drafter is not None:
                    engine.set_vlm_mtp_drafter(drafter)
                    logger.info(f"VLM MTP enabled for {model_id}, drafter={drafter_id}")
                else:
                    logger.warning(
                        f"VLM MTP toggle on for {model_id} but drafter "
                        f"load failed; toggle ignored"
                    )

            entry.runtime_settings_signature = self._engine_runtime_signature(
                model_id,
                model_settings,
                loaded_engine=engine,
            )

            # Propagate memory limit to new engine's scheduler
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer._propagate_memory_limit()

            # Release intermediate Metal buffers from model loading.
            # mlx_lm.load() creates large temporaries (weight transforms,
            # quantization intermediates) that stay in the Metal buffer pool
            # because mx.set_cache_limit(total_mem) prevents automatic release.
            # Without this, memory stays at ~2x model size until the first
            # inference request triggers a clear. (#429)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )

            post_load_memory = max(mx.get_active_memory(), get_phys_footprint())
            observed_delta = max(0, post_load_memory - pre_load_memory)
            entry.actual_size = observed_delta or entry.estimated_size

            # Registry consistency check: a lockless mutator (a
            # discover_models() rescan or the runtime model-directory
            # reload) may have replaced or dropped this model's entry at an
            # await point above. The engine was then attached to a stale
            # object unreachable from the pool; keeping it running would
            # strand the weights until a process restart (#2307).
            if self._entries.get(model_id) is not entry:
                entry_detached = True
                logger.warning(
                    f"Registry entry for '{model_id}' changed during load; "
                    f"releasing the freshly loaded engine "
                    f"({format_size(entry.estimated_size)})"
                )
                entry.engine = None
                self._current_model_memory = max(
                    0, self._current_model_memory - entry.estimated_size
                )
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(
                        f"Error stopping orphaned engine for {model_id}: {e}"
                    )
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    model_id,
                    f"Model '{model_id}' was removed or replaced while it "
                    "was loading; the loaded engine was released. Retry "
                    "the request.",
                )

            logger.info(
                f"Loaded model: {model_id} "
                f"(actual: {format_size(entry.actual_size)}, "
                f"estimated: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        except Exception as exc:
            # A failed load can leave tens of GB of just-loaded weights
            # reachable only through the propagating exception's traceback
            # frames (loader-internal locals). Running gc/clear_cache
            # synchronously here is useless -- the exception is still alive
            # in the caller. Schedule a deferred reclaim that runs after the
            # exception has been handled and dropped, so the buffers are
            # actually released; otherwise the process footprint stays
            # inflated and the memory-ceiling admission check rejects all
            # subsequent loads until a server restart.
            self._schedule_failed_load_reclaim(model_id, pre_load_memory)
            if not entry.abort_loading and not entry_detached:
                self._mark_load_failure(entry, exc)
                logger.exception(
                    "Model load failed for '%s'; caching failure until next discovery refresh",
                    model_id,
                )
                raise ModelUnavailableError(
                    model_id,
                    f"Model '{model_id}' failed to load: {entry.load_failure_message}. "
                    "Reload models after fixing the files to retry.",
                ) from exc
            raise
        finally:
            if (
                load_completed
                and load_started_at is not None
                and entry.estimated_size > 0
            ):
                elapsed = max(0.0, time.monotonic() - load_started_at)
                size_gb = entry.estimated_size / (1024**3)
                if size_gb > 0 and elapsed > 0:
                    sample = elapsed / size_gb
                    if self._load_seconds_per_gb_ema is None:
                        self._load_seconds_per_gb_ema = sample
                    else:
                        self._load_seconds_per_gb_ema = (
                            self._load_seconds_per_gb_ema * 0.9 + sample * 0.1
                        )
                    self._load_time_observations += 1
                    logger.debug(
                        f"Observed model load speed: {sample:.2f}s/GB "
                        f"for {model_id} ({elapsed:.1f}s, {format_size(entry.estimated_size)}); "
                        f"EMA={self._load_seconds_per_gb_ema:.2f}s/GB"
                    )
            entry.is_loading = False
            entry.loading_started_at = None
            entry.abort_loading = False
            self._wake_process_memory_enforcer()
            self._invalidate_cluster_node_state_cache()

    async def preload_pinned_models(self) -> None:
        """
        Preload all pinned models at startup.

        This ensures pinned models are always available.
        """
        pinned_models = [
            model_id for model_id, e in self._entries.items() if e.is_pinned
        ]

        for model_id in pinned_models:
            try:
                logger.info(f"Preloading pinned model: {model_id}")
                await self.get_engine(model_id)
            except Exception as e:
                logger.error(f"Failed to preload pinned model {model_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown all engines gracefully."""
        await self._drain_lease_release_tasks()
        # S4 D4: drain the cluster unload driver FIRST, mirroring
        # _drain_lease_release_tasks above -- server.py's shutdown sequence
        # already runs cluster_manager.stop() (-> formation.stop()) before
        # engine_pool.shutdown(), so any formation is down by the time we
        # get here; a still-running driver task would just be racing a
        # teardown against one that already happened.
        await self._drain_cluster_unload_driver()
        async with self._lock:
            for model_id in list(self._entries.keys()):
                entry = self._entries.get(model_id)
                if entry and entry.engine is not None:
                    if entry.kind == "cluster":
                        entry.engine = None
                        continue
                    try:
                        await self._unload_engine(model_id)
                    except Exception as e:
                        logger.error(f"Error unloading {model_id} during shutdown: {e}")

        logger.info("Engine pool shutdown complete")

    def get_status(self) -> dict:
        """
        Get pool status for monitoring endpoints.

        Returns:
            Dictionary with pool status information
        """
        return {
            "final_ceiling": self._current_ceiling(),
            "current_model_memory": self._current_model_memory,
            "model_count": len(self._entries),
            "loaded_count": sum(
                1 for e in self._entries.values() if e.engine is not None
            ),
            "load_seconds_per_gb_estimate": self._load_seconds_per_gb_ema,
            "load_time_observations": self._load_time_observations,
            "models": [
                {
                    "id": mid,
                    "model_path": e.model_path,
                    "loaded": e.engine is not None,
                    "is_loading": e.is_loading,
                    "loading_started_at": e.loading_started_at,
                    "estimated_size": e.estimated_size,
                    "actual_size": e.actual_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "is_helper": e.is_helper,
                    "thinking_default": e.thinking_default,
                    "preserve_thinking_default": e.preserve_thinking_default,
                    "source_type": e.source_type,
                    "source_repo_id": e.source_repo_id,
                    "last_access": e.last_access if e.last_access > 0 else None,
                }
                for mid, e in sorted(self._entries.items())
            ],
        }

    async def check_ttl_expirations(
        self,
        settings_manager: ModelSettingsManager,
        global_idle_timeout_seconds: int | None = None,
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are skipped and their last_access is refreshed.
        Suppressed during benchmark runs via _suppress_ttl flag.

        Args:
            settings_manager: The settings manager to read TTL values from.
            global_idle_timeout_seconds: Global idle timeout fallback (None = no global TTL).

        Returns:
            List of model IDs that were unloaded.
        """
        if self._suppress_ttl:
            return []

        now = time.time()
        expired: list[str] = []

        async with self._lock:
            for model_id, entry in self._entries.items():
                if entry.engine is None or entry.is_loading or entry.is_pinned:
                    continue

                settings = settings_manager.get_settings(model_id)
                effective_ttl = settings.ttl_seconds
                if effective_ttl is None:
                    effective_ttl = global_idle_timeout_seconds
                if effective_ttl is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < effective_ttl:
                    continue

                # Check if model has active requests
                has_active = entry.engine.has_active_requests() or entry.in_use > 0

                if has_active:
                    entry.last_access = now
                    continue

                logger.info(
                    f"TTL expired for model '{model_id}' "
                    f"(idle {idle_time:.0f}s > ttl {effective_ttl}s)"
                )
                if entry.kind == "cluster":
                    # S4 D4: eligible via mark-only -- the out-of-lock driver
                    # does the actual teardown.
                    self._mark_pending_unload_locked(model_id, "ttl expired")
                else:
                    await self._unload_engine(model_id)
                expired.append(model_id)

        return expired
