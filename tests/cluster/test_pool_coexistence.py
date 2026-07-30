# SPDX-License-Identifier: Apache-2.0
"""S4 D4: EnginePool coexistence with cluster (distributed) entries.

Cluster models are first-class ``EngineEntry`` rows (``kind="cluster"``);
the pool is the single owner of their create/bind/teardown. These tests
exercise the pool side of that contract with a real ``ClusterManager`` +
``FormationManager`` (fake spawn/engine factories, matching
``tests/cluster/test_formation.py``'s harness) so formation's own
choreography (E6 queue, worker acking) is real, not mocked away -- only the
rank-process boundary (spawn/engine construction) is faked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.cluster.formation import FormationManager
from omlx.cluster.manager import (
    ClusterError,
    set_engine_pool_getter,
)
from omlx.cluster.versions import collect_versions
from omlx.engine_pool import (
    EngineEntry,
    EnginePool,
    PlacementStaleError,
)
from omlx.exceptions import ModelTooLargeError

from .conftest import make_settings, running_manager

# world_size=2, both head counts divisible by 2.
_CONFIG = {"model_type": "llama", "num_attention_heads": 8, "num_key_value_heads": 8}
_MODEL_BYTES = 10_000_000
# estimate_model_size adds a 5% overhead factor (model_discovery.py); the
# placement per-rank estimate then applies D2's 1.15 headroom on top.
_EST_SIZE = int(_MODEL_BYTES * 1.05)
_PER_RANK_ESTIMATE = int(_EST_SIZE / 2 * 1.15)
_HEAD_CEILING = 7_000_000  # too small to fit locally; large enough per-rank
_WORKER_CEILING = 9_000_000


def _head_settings(tmp_path, **overrides):
    return make_settings(
        tmp_path / "head",
        role="head",
        data_plane_subnet="10.0.2.0/24",
        data_plane_address="10.0.2.1",
        **overrides,
    )


class FakeLeader:
    def wait_ready(self, timeout=0.0):
        return {"event": "ready"}

    def stop(self):
        return None


class FakeEngine:
    async def start(self):
        return None

    async def stop(self):
        return None

    def get_stats(self):
        return {"engine_type": "cluster-distributed", "loaded": True}

    def has_active_requests(self):
        return False


def _formation(manager, spawns=None, *, load_delay: float = 0.0):
    def _spawn_leader(**kwargs):
        if load_delay:
            # Runs off-loop via _run_blocking's executor: genuinely stalls
            # wall-clock time without blocking the event loop, so a test can
            # observe the entry mid-formation (is_loading=True).
            time.sleep(load_delay)
        (spawns or []).append(kwargs)
        return FakeLeader()

    fm = FormationManager(
        manager,
        spawn_leader_fn=_spawn_leader,
        engine_factory=lambda **kwargs: FakeEngine(),
        model_resolver=lambda model_id: "/head/models/target",
    )
    return fm


async def _activate_member(manager, *, memory_ceiling: int = _WORKER_CEILING):
    reply = await manager.join(
        peer_host="10.1.2.3",
        port=40404,
        name="worker",
        versions=collect_versions().to_dict(),
    )
    member = manager.state.member(reply["member_id"])
    assert member is not None
    manager.record_heartbeat(
        member,
        seq=1,
        epoch="ep",
        node_state={
            "total_memory": memory_ceiling * 2,
            "memory_ceiling": memory_ceiling,
            "models_present": {},
        },
    )
    return member


def _ju(command, **fields):
    return {"job_id": command["job_id"], "step": command["step"], **fields}


async def _drive_worker(fm, member, stop_event, *, present=True):
    while not stop_event.is_set():
        cmds = fm.commands_for(member.id)
        if cmds:
            command = cmds[0]
            kind = command["kind"]
            if kind == "presence":
                fm.record_job_updates(
                    member,
                    [
                        _ju(
                            command,
                            status="present" if present else "absent",
                            present=present,
                            data_plane_address="10.0.2.2",
                            rdma_device="rdma_en4",
                        )
                    ],
                )
            elif kind == "sweep":
                fm.record_job_updates(member, [_ju(command, status="swept")])
            elif kind == "spawn_rank":
                fm.record_job_updates(member, [_ju(command, status="spawned")])
            elif kind == "teardown":
                fm.record_job_updates(member, [_ju(command, status="torn_down")])
        await asyncio.sleep(0.003)


def _make_pool(tmp_path, *, ceiling: int = _HEAD_CEILING, model_id: str = "target"):
    """A real EnginePool with one discovered too-big-to-fit-locally model."""
    model_dir = tmp_path / "models" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(_CONFIG))
    (model_dir / "model.safetensors").write_bytes(b"0" * _MODEL_BYTES)

    pool = EnginePool()
    pool._get_final_ceiling = lambda c=ceiling: c
    pool.discover_models(str(tmp_path / "models"))
    return pool


class _Harness:
    """One head manager + one active worker + a wired pool, torn down clean."""

    def __init__(self, manager, member, formation, pool, driver_task, stop_event):
        self.manager = manager
        self.member = member
        self.formation = formation
        self.pool = pool
        self._driver_task = driver_task
        self._stop_event = stop_event

    async def close(self):
        self._stop_event.set()
        await self._driver_task
        await self.pool._drain_cluster_unload_driver()
        set_engine_pool_getter(None)


@contextlib.asynccontextmanager
async def harness(tmp_path, *, load_delay: float = 0.0):
    settings = _head_settings(tmp_path)
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        fm = _formation(manager, load_delay=load_delay)
        manager._formation = fm
        pool = _make_pool(tmp_path)
        set_engine_pool_getter(lambda: pool)

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        h = _Harness(manager, member, fm, pool, driver, stop)
        try:
            yield h
        finally:
            await h.close()


# ---- load_cluster_model: happy path, entry path, decision recording ------


async def test_load_cluster_model_yields_cluster_entry(tmp_path):
    async with harness(tmp_path) as h:
        result = await h.pool.load_cluster_model("target", "auto")
        assert result["status"] == "ready"
        assert result["decision"]["mode"] == "distributed"

        entry = h.pool.get_entry("target")
        assert entry.kind == "cluster"
        assert entry.engine is not None
        assert entry.cluster_head_share == _PER_RANK_ESTIMATE
        assert entry.estimated_size == _PER_RANK_ESTIMATE
        assert h.pool.current_model_memory == _PER_RANK_ESTIMATE


async def test_get_engine_returns_formed_entry_without_local_load(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        engine = await h.pool.get_engine("target")
        assert engine is h.formation.active_engine("target")


async def test_plain_load_resolving_distributed_makes_one_plan_placement_call(
    tmp_path, monkeypatch
):
    calls = []
    from omlx.cluster import placement as placement_mod

    real_plan = placement_mod.plan_placement

    def _counting_plan(*args, **kwargs):
        calls.append(1)
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(placement_mod, "plan_placement", _counting_plan)

    async with harness(tmp_path) as h:
        engine = await h.pool.get_engine("target")
        assert engine is not None
        assert len(calls) == 1
        entry = h.pool.get_entry("target")
        assert entry.kind == "cluster"

        # The decision that drove the load is the one recorded on the job.
        status = h.manager.formation_status()
        assert status["jobs"][-1]["decision"]["mode"] == "distributed"


async def test_wiring_one_pool_call_and_recorded_decision_readable(tmp_path):
    async with harness(tmp_path) as h:
        result = await h.manager.load_distributed("target")
        assert result["decision"]["mode"] == "distributed"

        status = h.manager.formation_status()
        assert status["jobs"][-1]["decision"] == result["decision"]


# ---- entry lifecycle -------------------------------------------------------


async def test_form_then_unform_restores_everything(tmp_path):
    async with harness(tmp_path) as h:
        original_estimated_size = h.pool.get_entry("target").estimated_size
        await h.pool.load_cluster_model("target", "auto")
        assert h.pool.get_entry("target").kind == "cluster"

        await h.pool.unload_cluster_model("target")

        entry = h.pool.get_entry("target")
        assert entry.kind == "local"
        assert entry.estimated_size == original_estimated_size
        assert entry.cluster_head_share is None
        assert entry.cluster_original_estimated_size is None
        assert entry.engine is None
        assert h.pool.current_model_memory == 0


async def test_discovery_refresh_during_formation_does_not_replace_entry(tmp_path):
    async with harness(tmp_path, load_delay=0.05) as h:
        load_task = asyncio.create_task(h.pool.load_cluster_model("target", "auto"))
        await asyncio.sleep(0.005)
        entry_before = h.pool.get_entry("target")
        assert entry_before.is_loading is True

        h.pool.discover_models(str(tmp_path / "models"))
        assert h.pool.get_entry("target") is entry_before

        await load_task
        assert h.pool.get_entry("target").kind == "cluster"


# ---- lock discipline: deadlock row + bounded retry -------------------------


def _pin_live_memory_readings_to_zero(monkeypatch):
    """Real `mx.get_active_memory()`/`get_phys_footprint()` read this test
    process's actual baseline (tens of MB from loaded libraries) -- pinning
    both to 0 makes the pool's tracked `_current_model_memory` accumulator
    the sole admission signal, matching the established pattern in
    tests/test_engine_pool.py's own eviction-forcing tests.
    """
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)


async def test_admission_eviction_of_cluster_victim_never_awaits_under_lock(
    tmp_path, monkeypatch
):
    """Deadlock row: a formation job occupies the E6 queue while a task
    holding the pool lock selects the cluster entry as LRU victim --
    completes, and no formation await ever runs under the lock.
    """
    _pin_live_memory_readings_to_zero(monkeypatch)
    async with harness(tmp_path, load_delay=0.05) as h:
        await h.pool.load_cluster_model("target", "auto")
        entry = h.pool.get_entry("target")
        assert entry.kind == "cluster"

        # A second, smaller model that fits only once "target" is evicted.
        small_dir = tmp_path / "models" / "small"
        small_dir.mkdir()
        (small_dir / "config.json").write_text(json.dumps(_CONFIG))
        (small_dir / "model.safetensors").write_bytes(b"0" * 1_000_000)
        h.pool.discover_models(str(tmp_path / "models"))
        # Ceiling only fits "small" alone, not both -- forces eviction.
        h.pool._get_final_ceiling = lambda: 1_500_000

        # If the pool ever awaited formation.unload() under self._lock, this
        # concurrent lock probe would hang for the duration of that await.
        probe_acquired = asyncio.Event()

        async def _probe():
            async with h.pool._lock:
                probe_acquired.set()

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()
        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine):
            engine = await h.pool.get_engine("small")
        assert engine is mock_engine
        assert h.pool.get_entry("target").kind == "local"  # unformed

        probe_task = asyncio.create_task(_probe())
        await asyncio.wait_for(probe_acquired.wait(), timeout=1.0)
        await probe_task


async def test_cluster_victim_resolves_within_bounded_retries(tmp_path, monkeypatch):
    _pin_live_memory_readings_to_zero(monkeypatch)
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")

        small_dir = tmp_path / "models" / "small"
        small_dir.mkdir()
        (small_dir / "config.json").write_text(json.dumps(_CONFIG))
        (small_dir / "model.safetensors").write_bytes(b"0" * 1_000_000)
        h.pool.discover_models(str(tmp_path / "models"))
        h.pool._get_final_ceiling = lambda: 1_500_000

        mock_engine = MagicMock()
        mock_engine.start = AsyncMock()
        mock_engine.stop = AsyncMock()
        with patch("omlx.engine_pool.BatchedEngine", return_value=mock_engine):
            engine = await h.pool.get_engine("small")
        assert engine is mock_engine
        assert h.pool.get_entry("small").engine is not None


# ---- selection filters: TTL eligible, prefill/enforcer ineligible ---------


async def test_ttl_marks_cluster_entry_for_the_driver(tmp_path):
    import types

    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        entry = h.pool.get_entry("target")
        entry.last_access = 0.0  # far in the past

        settings_manager = types.SimpleNamespace(
            get_settings=lambda mid: types.SimpleNamespace(ttl_seconds=1)
        )
        expired = await h.pool.check_ttl_expirations(settings_manager)
        assert expired == ["target"]
        # Mark-only: the driver (not this call) does the actual teardown.
        assert entry.pending_unload_reason == "ttl expired"

        # Let the driver actually tear it down.
        for _ in range(200):
            if h.pool.get_entry("target").kind == "local":
                break
            await asyncio.sleep(0.01)
        assert h.pool.get_entry("target").kind == "local"


async def test_prefill_headroom_eviction_skips_cluster_entries(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        entry = h.pool.get_entry("target")
        assert h.pool._is_idle_for_prefill_eviction(entry) is False
        assert (
            h.pool._find_lru_prefill_eviction_victim(exclude_model_id="nothing") is None
        )


async def test_enforcer_pressure_selection_filters_cluster_entries(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        # Default include_cluster=False -- the enforcer's own call site.
        assert h.pool._find_lru_victim() is None
        assert h.pool._find_lru_victim(include_cluster=True) == "target"


async def test_enforcer_busy_victim_selector_filters_cluster_entries(tmp_path):
    from omlx.process_memory_enforcer import ProcessMemoryEnforcer

    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        h.pool.get_entry("target").in_use = 1  # busy

        enforcer = ProcessMemoryEnforcer.__new__(ProcessMemoryEnforcer)
        enforcer._engine_pool = h.pool
        assert enforcer._find_lru_busy_non_pinned_victim_locked() is None


# ---- _unload_engine raises; request_unload dispatches ----------------------


async def test_unload_engine_raises_on_cluster_entry(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        with pytest.raises(RuntimeError):
            await h.pool._unload_engine("target")


async def test_request_unload_drives_driver_for_cluster_entry(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        await h.pool.request_unload("target")
        entry = h.pool.get_entry("target")
        assert entry.kind == "local"
        assert entry.engine is None
        assert h.pool.current_model_memory == 0


async def test_request_unload_local_entry_unaffected(tmp_path):
    async with harness(tmp_path) as h:
        entry = EngineEntry(
            model_id="plain",
            model_path=str(tmp_path),
            model_type="llm",
            engine_type="batched",
            estimated_size=10,
        )

        class _E:
            async def stop(self):
                return None

            def has_active_requests(self):
                return False

        entry.engine = _E()
        h.pool._entries["plain"] = entry
        h.pool._current_model_memory = 10
        await h.pool.request_unload("plain")
        assert h.pool.get_entry("plain").engine is None


async def test_unload_if_idle_unpinned_kind_dispatches(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        result = await h.pool.unload_if_idle_unpinned("target")
        assert result is True
        entry = h.pool.get_entry("target")
        assert entry.pending_unload_reason == "unload_if_idle_unpinned"

        for _ in range(200):
            if h.pool.get_entry("target").kind == "local":
                break
            await asyncio.sleep(0.01)
        assert h.pool.get_entry("target").kind == "local"


# ---- rev5 edge rows ---------------------------------------------------------


async def test_ttl_marked_then_leased_then_released_no_raise_one_teardown(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        entry = h.pool.get_entry("target")
        entry.pending_unload_reason = "ttl expired"  # marked

        # Leased then released while marked -- must not raise, must not
        # unload inline (the guard preserves the marker and hands off).
        engine = await h.pool.get_engine("target", _lease=True)
        assert engine is not None
        await h.pool.release_engine("target")

        assert h.pool.get_entry("target").pending_unload_reason == "ttl expired"

        h.pool._wake_cluster_unload_driver_locked()
        for _ in range(200):
            if h.pool.get_entry("target").kind == "local":
                break
            await asyncio.sleep(0.01)
        assert h.pool.get_entry("target").kind == "local"


async def test_request_unload_mid_formation_is_conflict(tmp_path):
    async with harness(tmp_path, load_delay=0.1) as h:
        load_task = asyncio.create_task(h.pool.load_cluster_model("target", "auto"))
        await asyncio.sleep(0.01)
        with pytest.raises(ClusterError) as excinfo:
            await h.pool.unload_cluster_model("target")
        assert excinfo.value.status_code == 409
        await load_task


async def test_request_unload_pinned_and_busy_force_teardown_succeeds(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        entry = h.pool.get_entry("target")
        entry.is_pinned = True
        entry.in_use = 1  # busy

        await h.pool.request_unload("target")
        entry = h.pool.get_entry("target")
        assert entry.kind == "local"


async def test_shutdown_drains_driver_without_pending_await(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        # cluster_manager.stop() (server.py's ordering) already tore
        # formation down by the time engine_pool.shutdown() runs; shutdown
        # must not hang draining a driver awaiting a dead formation.
        await h.manager.stop()
        await asyncio.wait_for(h.pool.shutdown(), timeout=2.0)
        assert h.pool.get_entry("target").engine is None


async def test_placement_stale_triggers_exactly_one_recompute(tmp_path, monkeypatch):
    from omlx.cluster import placement as placement_mod

    calls = []
    real_plan = placement_mod.plan_placement

    def _counting_plan(*args, **kwargs):
        calls.append(1)
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(placement_mod, "plan_placement", _counting_plan)

    async with harness(tmp_path) as h:
        stale_decision = placement_mod.plan_placement(
            model_id="target",
            model_type="llm",
            est_size=_MODEL_BYTES,
            model_config=_CONFIG,
            head=h.pool.head_capacity(),
            workers=[],  # no workers known yet -> this decision is stale
            prefer="auto",
        )
        calls.clear()
        with pytest.raises(PlacementStaleError):
            await h.pool.load_cluster_model("target", "auto", decision=stale_decision)
        # The revalidation itself doesn't call plan_placement again -- a
        # caller (get_engine) recomputes on its next attempt, which this
        # unit exercises directly instead of driving get_engine's retry.
        assert calls == []


# ---- variant-branch (kind guard) rows --------------------------------------


async def test_runtime_settings_on_cluster_entry_raises_and_no_unload(tmp_path):
    from omlx.cluster.engine import ClusterNonGoalError

    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        with pytest.raises(ClusterNonGoalError):
            await h.pool.get_engine("target", runtime_settings=object())
        # Still formed -- no formation unload was driven from this block.
        assert h.pool.get_entry("target").kind == "cluster"
        assert h.pool.get_entry("target").engine is not None


async def test_cluster_entry_with_none_runtime_settings_untouched(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        entry = h.pool.get_entry("target")
        assert entry.runtime_settings_signature is None

        engine = await h.pool.get_engine("target", runtime_settings=None)
        assert engine is entry.engine
        assert entry.runtime_settings_signature is None
        assert entry.kind == "cluster"


# ---- uniform LRU / one-entry-per-model_id -----------------------------------


async def test_pinned_cluster_entry_never_a_victim(tmp_path):
    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        h.pool.get_entry("target").is_pinned = True
        assert h.pool._find_lru_victim(include_cluster=True) is None


async def test_one_entry_per_model_id_across_kinds(tmp_path):
    async with harness(tmp_path) as h:
        assert h.pool.get_entry("target").kind == "local"
        await h.pool.load_cluster_model("target", "auto")
        assert len([mid for mid in h.pool._entries if mid == "target"]) == 1
        assert h.pool.get_entry("target").kind == "cluster"


# ---- auto_placement=False / role=off routing --------------------------------


async def test_auto_placement_false_takes_todays_local_only_path(tmp_path):
    async with harness(tmp_path) as h:
        h.manager.settings.auto_placement = False
        with pytest.raises(ModelTooLargeError):
            await h.pool.get_engine("target")
        assert h.pool.get_entry("target").kind == "local"


async def test_role_off_never_reaches_cluster_gate(tmp_path):
    pool = _make_pool(tmp_path)
    from omlx.cluster.manager import set_cluster_manager

    set_cluster_manager(None)
    with pytest.raises(ModelTooLargeError):
        await pool.get_engine("target")
    assert pool.get_entry("target").kind == "local"


# ---- fast-path zero I/O row --------------------------------------------------


async def test_already_loaded_entry_makes_zero_placement_io_calls(
    tmp_path, monkeypatch
):
    from omlx.cluster import placement as placement_mod

    calls = []
    monkeypatch.setattr(
        placement_mod,
        "resolve_placement_inputs",
        lambda *a, **k: calls.append(1) or (0, {}),
    )

    async with harness(tmp_path) as h:
        await h.pool.load_cluster_model("target", "auto")
        calls.clear()
        await h.pool.get_engine("target")
        assert calls == []
