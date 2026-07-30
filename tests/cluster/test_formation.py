# SPDX-License-Identifier: Apache-2.0
"""Head-side formation job (D8): presence-gap failure, single-formation
idempotence, and the CL2-06 teardown-suppression alarm.

The worker is faked: a small driver resolves each published command with the
reply a confined worker would send, so the head-side sequencing is exercised
without spawning ranks.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

import pytest

from omlx.cluster.formation import FormationManager
from omlx.cluster.manager import ClusterError, set_engine_pool_getter
from omlx.cluster.state import Member
from omlx.cluster.versions import collect_versions

from .conftest import make_settings, running_manager


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
        # snapshot() surfaces the live engine's stats once a model is active.
        return {"engine_type": "cluster-distributed", "loaded": True}


def _formation(manager, spawns):
    return FormationManager(
        manager,
        spawn_leader_fn=lambda **kwargs: (spawns.append(kwargs) or FakeLeader()),
        engine_factory=lambda **kwargs: FakeEngine(),
        model_resolver=lambda model_id: "/head/models/target",
    )


async def _activate_member(manager) -> Member:
    reply = await manager.join(
        peer_host="10.1.2.3",
        port=40404,
        name="worker",
        versions=collect_versions().to_dict(),
    )
    member = manager.state.member(reply["member_id"])
    assert member is not None
    manager.record_heartbeat(member, seq=1, epoch="ep")
    return member


async def _pending(fm, member, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cmds = fm.commands_for(member.id)
        if cmds:
            return cmds[0]
        await asyncio.sleep(0.005)
    raise TimeoutError("no command was published")


def _ju(command, **fields):
    return {"job_id": command["job_id"], "step": command["step"], **fields}


async def _drive_worker(fm, member, stop_event, *, present=True):
    """Answer published commands as a confined worker would, until stopped."""
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


# -- presence gap -------------------------------------------------------------


async def test_presence_gap_fails_and_spawns_nothing(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation(manager, spawns)
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=False))
        try:
            with pytest.raises(ClusterError) as excinfo:
                await fm.load("target")
            assert excinfo.value.status_code == 424
            assert "absent" in excinfo.value.detail
            assert spawns == []
        finally:
            stop.set()
            await driver


# -- happy path + single-formation idempotence --------------------------------


async def test_load_forms_then_refuses_a_second(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation(manager, spawns)
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            result = await fm.load("target")
            assert result["status"] == "ready"
            assert fm.active_engine("target") is not None
            # rank 0 spawned once with the head's own address in slot 0.
            assert len(spawns) == 1
            assert spawns[0]["ips"] == ["10.0.2.1", "10.0.2.2"]

            with pytest.raises(ClusterError) as excinfo:
                await fm.load("other")
            assert excinfo.value.status_code == 409

            unload = await fm.unload("target")
            assert unload["status"] == "unloaded"
            assert fm.active_engine("target") is None
        finally:
            stop.set()
            await driver


# -- D7 backend resolution + auto fallback ------------------------------------


def _candidates_for(backend: str) -> list[str]:
    fm = FormationManager.__new__(FormationManager)
    fm._manager = types.SimpleNamespace(settings=types.SimpleNamespace(backend=backend))
    return fm._backend_candidates()


def test_backend_candidates_per_setting():
    assert _candidates_for("ring") == ["ring"]
    assert _candidates_for("jaccl") == ["jaccl"]
    # auto is a resolver that tries jaccl first, then ring.
    assert _candidates_for("auto") == ["jaccl", "ring"]


def _formation_with_spawn(manager, spawns, *, fail_on=None):
    """A formation whose fake leader spawn fails for one named backend."""

    def spawn(*, backend, **kwargs):
        spawns.append({"backend": backend, **kwargs})
        if backend == fail_on:
            raise RuntimeError(f"{backend} PD exhausted")
        return FakeLeader()

    return FormationManager(
        manager,
        spawn_leader_fn=spawn,
        engine_factory=lambda **kwargs: FakeEngine(),
        model_resolver=lambda model_id: "/head/models/target",
    )


async def test_auto_falls_back_to_ring_when_jaccl_fails(tmp_path):
    settings = _head_settings(tmp_path, backend="auto", rdma_device="rdma_en2")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation_with_spawn(manager, spawns, fail_on="jaccl")
        manager._formation = fm
        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            result = await fm.load("target")
            # Reported backend reflects what ACTUALLY formed, not the request.
            assert result["negotiated_backend"] == "ring"
            assert fm.snapshot()["jobs"][-1]["negotiated_backend"] == "ring"
            # jaccl was attempted first (with a real matrix), then ring.
            assert [s["backend"] for s in spawns] == ["jaccl", "ring"]
            assert spawns[0]["ibv_devices"] == [
                [None, "rdma_en2"],
                ["rdma_en4", None],
            ]
            assert spawns[1]["ibv_devices"] is None
            # An intra-job fallback teardown must NOT arm the CL2-06 alarm.
            assert fm.alarms() == []
        finally:
            stop.set()
            await driver


class _FailReadyLeader:
    """A leader that spawns fine but whose barrier fails on the first attempt —
    the sequence where the jaccl rank is live on the worker and the ring retry
    is refused (CL2-09) unless the fallback teardown lands first."""

    def __init__(self, fail: bool) -> None:
        self._fail = fail

    def wait_ready(self, timeout: float = 0.0):
        if self._fail:
            raise RuntimeError("jaccl barrier timed out")
        return {"event": "ready"}

    def stop(self):
        return None


async def _recording_driver(fm, member, seen, stop_event):
    """Like _drive_worker but records the ordered sequence of command kinds,
    deduped so a command seen across polls counts once."""
    last = None
    while not stop_event.is_set():
        cmds = fm.commands_for(member.id)
        if cmds:
            command = cmds[0]
            key = (command["job_id"], command["step"])
            if key != last:
                last = key
                seen.append(command["kind"])
            kind = command["kind"]
            if kind == "presence":
                fm.record_job_updates(
                    member,
                    [
                        _ju(
                            command,
                            status="present",
                            present=True,
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


async def test_auto_tears_worker_down_between_attempts(tmp_path):
    # The jaccl rank spawns on the worker, then the head's barrier fails: the
    # ring retry must be preceded by a worker teardown so CL2-09 does not refuse
    # the second spawn (fresh process per attempt, salvage pitfall 6).
    settings = _head_settings(tmp_path, backend="auto", rdma_device="rdma_en2")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []

        def spawn(*, backend, **kwargs):
            spawns.append({"backend": backend, **kwargs})
            return _FailReadyLeader(fail=(backend == "jaccl"))

        fm = FormationManager(
            manager,
            spawn_leader_fn=spawn,
            engine_factory=lambda **kwargs: FakeEngine(),
            model_resolver=lambda model_id: "/head/models/target",
        )
        manager._formation = fm
        seen: list[str] = []
        stop = asyncio.Event()
        driver = asyncio.create_task(_recording_driver(fm, member, seen, stop))
        try:
            result = await fm.load("target")
            assert result["negotiated_backend"] == "ring"
            assert [s["backend"] for s in spawns] == ["jaccl", "ring"]
            # A teardown landed between the worker's two spawn_rank commands.
            first = seen.index("spawn_rank")
            second = seen.index("spawn_rank", first + 1)
            assert "teardown" in seen[first + 1 : second]
            # The intra-job teardown must not arm the CL2-06 suppression alarm.
            assert fm.alarms() == []
        finally:
            stop.set()
            await driver


async def test_explicit_jaccl_failure_does_not_fall_back(tmp_path):
    settings = _head_settings(tmp_path, backend="jaccl", rdma_device="rdma_en2")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation_with_spawn(manager, spawns, fail_on="jaccl")
        manager._formation = fm
        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            with pytest.raises(RuntimeError) as excinfo:
                await fm.load("target")
            # The real jaccl error, never a silent ring substitution.
            assert "jaccl" in str(excinfo.value)
            assert [s["backend"] for s in spawns] == ["jaccl"]
            job = fm._jobs[-1]
            assert job.status == "failed"
            assert job.negotiated_backend == ""
        finally:
            stop.set()
            await driver


async def test_explicit_jaccl_success_reports_jaccl_and_builds_matrix(tmp_path):
    settings = _head_settings(tmp_path, backend="jaccl", rdma_device="rdma_en2")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation_with_spawn(manager, spawns, fail_on=None)
        manager._formation = fm
        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            result = await fm.load("target")
            assert result["negotiated_backend"] == "jaccl"
            # The head builds the full S0-recipe matrix from typed state.
            assert spawns[0]["backend"] == "jaccl"
            assert spawns[0]["ibv_devices"] == [
                [None, "rdma_en2"],
                ["rdma_en4", None],
            ]
        finally:
            stop.set()
            await driver


async def test_jaccl_without_head_rdma_device_fails(tmp_path):
    # backend pinned jaccl but the head has no rdma_device: an explicit request
    # that cannot form fails rather than silently forming ring.
    settings = _head_settings(tmp_path, backend="jaccl")  # no rdma_device
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation_with_spawn(manager, spawns, fail_on=None)
        manager._formation = fm
        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            with pytest.raises(ClusterError):
                await fm.load("target")
            assert spawns == []  # nothing formed on any backend
        finally:
            stop.set()
            await driver


async def test_unload_unknown_model_is_404(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        fm = _formation(manager, [])
        manager._formation = fm
        with pytest.raises(ClusterError) as excinfo:
            await fm.unload("nope")
        assert excinfo.value.status_code == 404


# -- CL2-06 teardown-suppression alarm ----------------------------------------


def _member() -> Member:
    import ipaddress

    return Member(
        id="m-1",
        address=ipaddress.ip_address("10.1.2.3"),
        port=40404,
        name="w",
        versions=collect_versions(),
        joined_at=1.0,
    )


def test_alarm_fires_when_worker_reports_a_torn_down_formation(tmp_path):
    fm = FormationManager.__new__(FormationManager)
    # Minimal state for the alarm path (no manager needed).
    fm._acks = {}
    fm._torn_down_jobs = {"job-x"}
    fm._alarms = []

    member = _member()
    fm.record_job_updates(member, [{"job_id": "job-x", "step": 3, "status": "spawned"}])
    assert fm.alarms()
    assert "tore down" in fm.alarms()[0]


async def test_head_restart_no_stale_job_resurrection_and_reform_works(tmp_path):
    """S6 D2 head-restart behavior -- TESTED here rather than re-engineered.

    Correction, recorded: the plan's original wording ("the CL2-06
    torn-down-job alarm still fires on a member reporting a pre-restart
    job") does not hold and was not implemented -- a genuinely fresh
    `FormationManager` has an empty `_torn_down_jobs`, so nothing lets that
    alarm fire for a job id from before the restart (it only fires within
    one formation's own lifetime; `test_alarm_fires_when_worker_reports_a_
    torn_down_formation`, above, is that guard). What restart-safety
    actually means: a pre-restart job never resurrects, is never alarmed
    on, and a re-issued load re-forms cleanly through the fresh manager.
    """
    settings = _head_settings(tmp_path, backend="ring")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)

        # "After restart": a FRESH FormationManager, exactly what
        # ClusterManager.start() constructs on a real process restart
        # (formation is deliberately runtime-only, formation.py:56).
        spawns: list = []
        fm = _formation(manager, spawns)
        manager._formation = fm
        assert fm._jobs == []
        assert fm._torn_down_jobs == set()

        # The worker's next heartbeat still references a pre-restart job id
        # (e.g. a queued update that never got a chance to be drained
        # before the head died).
        manager.record_heartbeat(
            member,
            seq=2,
            epoch="ep",
            job_updates=[
                {"job_id": "pre-restart-job-id", "step": 1, "status": "spawned"}
            ],
        )
        assert fm.alarms() == []  # nothing to resurrect, nothing to alarm on
        assert fm._jobs == []
        assert manager.liveness(member.id).status == "active"

        # A re-issued load re-forms cleanly through the fresh manager.
        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            result = await fm.load("target")
            assert result["status"] == "ready"
        finally:
            stop.set()
            await driver


def test_torn_down_ack_does_not_alarm(tmp_path):
    fm = FormationManager.__new__(FormationManager)
    fm._acks = {}
    fm._torn_down_jobs = {"job-x"}
    fm._alarms = []

    member = _member()
    fm.record_job_updates(
        member, [{"job_id": "job-x", "step": 1, "status": "torn_down"}]
    )
    assert fm.alarms() == []


def test_snapshot_surfaces_engine_stats_when_loaded(tmp_path):
    # The E4 coordination-tax summary must be readable through the status
    # endpoint (formation.snapshot), not only in-process (D9).
    fm = FormationManager.__new__(FormationManager)
    fm._jobs = []
    fm._alarms = []

    class _FakeEngine:
        def get_stats(self):
            return {
                "negotiated_backend": "ring",
                "last_tax": {"steps": 256, "avg_ms": 0.16},
            }

    # No model loaded: no engine_stats key.
    fm._active_model = None
    fm._engines = {}
    assert "engine_stats" not in fm.snapshot()

    # Model loaded: engine stats surfaced verbatim.
    fm._active_model = "m"
    fm._engines = {"m": _FakeEngine()}
    snap = fm.snapshot()
    assert snap["engine_stats"]["negotiated_backend"] == "ring"
    assert snap["engine_stats"]["last_tax"]["steps"] == 256


# -- S6 D1: rank-death propagation + degrade ----------------------------------


class _BlockingLeader:
    """Simulates rank 0 stuck in `wait_ready`'s barrier: blocks until
    `kill()`/`stop()` is called, then raises exactly as a closed reply
    channel would (`launcher.py`'s real `wait_ready`, line ~726)."""

    def __init__(self) -> None:
        self._killed = threading.Event()

    def wait_ready(self, timeout: float = 0.0):
        if not self._killed.wait(timeout=timeout):
            raise TimeoutError("test-only: wait_ready timed out without a kill")
        raise RuntimeError("rank 0 closed its channel before reporting ready")

    def kill(self) -> None:
        self._killed.set()

    def stop(self) -> None:
        self._killed.set()


async def test_dead_rank_report_kills_an_in_progress_formation_directly(tmp_path):
    """rev2/D1, rev3/B1 -- pins the measured ~385s hang closed.

    Uses a REAL `ClusterCommandQueue` (via `running_manager`): `fm.load`
    runs AS the queued op, blocked inside `wait_ready`. An implementation
    that submitted this abort to the same queue instead of killing the
    in-flight `LocalCluster` directly would deadlock -- the queue cannot run
    a second op until the first (this one) finishes -- and this test would
    time out rather than observe a fast rollback.
    """
    settings = _head_settings(tmp_path, backend="ring")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        leader = _BlockingLeader()
        fm = FormationManager(
            manager,
            spawn_leader_fn=lambda **kwargs: (spawns.append(kwargs) or leader),
            engine_factory=lambda **kwargs: FakeEngine(),
            model_resolver=lambda model_id: "/head/models/target",
        )
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        try:
            load_task = asyncio.create_task(fm.load("target"))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if fm._local is not None and fm._active_model is None:
                    break
                await asyncio.sleep(0.005)
            assert fm._local is leader, "formation never reached the in-progress state"

            fm.handle_dead_rank(member.id, "test: rank 1 process exited")

            with pytest.raises((RuntimeError, ClusterError)):
                await asyncio.wait_for(load_task, timeout=5.0)

            assert fm._local is None  # the pool's rollback guard ran
            assert fm._jobs[-1].status == "failed"
        finally:
            stop.set()
            await driver


async def test_dead_rank_report_leaves_a_healthy_formation_untouched(tmp_path):
    """No formation in progress and no active model: a dead-rank report is a
    no-op (nothing to kill, nothing to degrade)."""
    fm = FormationManager.__new__(FormationManager)
    fm._local = None
    fm._active_model = None
    fm.handle_dead_rank("m-1", "spurious")  # must not raise


class _RequestUnloadStubPool:
    """Stands in for `EnginePool.request_unload` (the S4 unload driver):
    records the call and delegates straight to the SAME `FormationManager`'s
    `unload`, exactly what the real driver's `_teardown_cluster_entry` does.
    """

    def __init__(self, fm: FormationManager) -> None:
        self._fm = fm
        self.calls: list[str] = []

    async def request_unload(self, model_id: str) -> None:
        self.calls.append(model_id)
        await self._fm.unload(model_id)


async def test_serving_formation_degrades_and_tears_down_on_dead_rank_report(
    tmp_path,
):
    settings = _head_settings(tmp_path, backend="ring")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation(manager, spawns)
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        stub_pool = _RequestUnloadStubPool(fm)
        set_engine_pool_getter(lambda: stub_pool)
        try:
            result = await fm.load("target")
            assert result["status"] == "ready"

            fm.handle_dead_rank(member.id, "test: rank 1 process exited")

            deadline = time.monotonic() + 2.0
            while (
                time.monotonic() < deadline and fm.active_engine("target") is not None
            ):
                await asyncio.sleep(0.01)

            assert fm.active_engine("target") is None
            assert stub_pool.calls == ["target"]
            assert any(job.status == "degraded" for job in fm._jobs)
            assert any("degraded" in alarm for alarm in fm.alarms())
        finally:
            set_engine_pool_getter(None)
            stop.set()
            await driver


async def test_worker_heartbeat_ranks_field_reaches_formation_end_to_end(tmp_path):
    """The FULL wire path, not a shortcut: a worker's heartbeat reporting a
    dead rank must flow through `ClusterManager.record_heartbeat`'s ranks
    parsing into the REAL `FormationManager.handle_dead_rank` -- the exact
    gap the S5 rig hit (worker deathwatch knew; the head learned nothing).
    """
    settings = _head_settings(tmp_path, backend="ring")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation(manager, spawns)
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        stub_pool = _RequestUnloadStubPool(fm)
        set_engine_pool_getter(lambda: stub_pool)
        try:
            result = await fm.load("target")
            assert result["status"] == "ready"

            # The worker's own next heartbeat, exactly the payload
            # `WorkerCommandExecutor.ranks_status()` would produce.
            manager.record_heartbeat(
                member,
                seq=2,
                epoch="ep",
                ranks={"alive": [], "dead": [1]},
            )

            deadline = time.monotonic() + 2.0
            while (
                time.monotonic() < deadline and fm.active_engine("target") is not None
            ):
                await asyncio.sleep(0.01)

            assert fm.active_engine("target") is None
            assert stub_pool.calls == ["target"]
            assert any(job.status == "degraded" for job in fm._jobs)
        finally:
            set_engine_pool_getter(None)
            stop.set()
            await driver


async def test_dead_rank_report_is_idempotent_for_a_serving_model(tmp_path):
    """A heartbeat dead-rank report and an engine EOF signal can both fire
    for the same SERVING model; the teardown must run once, not twice."""
    settings = _head_settings(tmp_path, backend="ring")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        fm = _formation(manager, spawns)
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        stub_pool = _RequestUnloadStubPool(fm)
        set_engine_pool_getter(lambda: stub_pool)
        try:
            result = await fm.load("target")
            assert result["status"] == "ready"

            fm.handle_dead_rank(member.id, "signal one")
            fm.handle_dead_rank(member.id, "signal two")

            deadline = time.monotonic() + 2.0
            while (
                time.monotonic() < deadline and fm.active_engine("target") is not None
            ):
                await asyncio.sleep(0.01)

            assert fm.active_engine("target") is None
            assert stub_pool.calls == ["target"]  # not twice
        finally:
            set_engine_pool_getter(None)
            stop.set()
            await driver


class _CapturingEngine(FakeEngine):
    """Records the `on_rank_death` callback `_load` wired in, so a test can
    invoke it exactly as `ClusterEngine`'s reader loop would."""

    def __init__(self, **kwargs) -> None:
        self.on_rank_death = kwargs.get("on_rank_death")


async def test_load_wires_a_working_on_rank_death_callback(tmp_path):
    """End-to-end: `_load` must actually pass a callback into the engine
    that, when invoked (as `ClusterEngine`'s reply-pipe reader would on an
    EOF), degrades and tears the formation down -- not merely accept the
    kwarg."""
    settings = _head_settings(tmp_path, backend="ring")
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        spawns: list = []
        engines: list[_CapturingEngine] = []

        def factory(**kwargs):
            engine = _CapturingEngine(**kwargs)
            engines.append(engine)
            return engine

        fm = FormationManager(
            manager,
            spawn_leader_fn=lambda **kwargs: (spawns.append(kwargs) or FakeLeader()),
            engine_factory=factory,
            model_resolver=lambda model_id: "/head/models/target",
        )
        manager._formation = fm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(fm, member, stop, present=True))
        stub_pool = _RequestUnloadStubPool(fm)
        set_engine_pool_getter(lambda: stub_pool)
        try:
            result = await fm.load("target")
            assert result["status"] == "ready"
            assert engines[0].on_rank_death is not None

            engines[0].on_rank_death("simulated rank-0 pipe EOF")

            deadline = time.monotonic() + 2.0
            while (
                time.monotonic() < deadline and fm.active_engine("target") is not None
            ):
                await asyncio.sleep(0.01)

            assert fm.active_engine("target") is None
            assert stub_pool.calls == ["target"]
        finally:
            set_engine_pool_getter(None)
            stop.set()
            await driver
