# SPDX-License-Identifier: Apache-2.0
"""S5 P2: EnginePool.load_cluster_model's presence-resolution pre-step (D5).

Combines test_pool_coexistence.py's FormationManager harness (fake
spawn_leader_fn/engine_factory/model_resolver -- no real ring hardware) with
test_transfer.py's TransferManager<->TransferWorkerExecutor round-trip
driving (fake session_launcher -- no real mlx ring session) into one worker
command driver, so the whole D5 pre-step (transfer job) -> formation
sequence runs through real EnginePool/ClusterManager/FormationManager/
TransferManager/TransferWorkerExecutor objects, matching both harnesses'
own "only the process-spawn boundary is faked" discipline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

import pytest

from omlx.cluster.formation import FormationManager
from omlx.cluster.manager import ClusterError, set_engine_pool_getter
from omlx.cluster.protocol import parse_command
from omlx.cluster.transfer import TransferManager, TransferWorkerExecutor
from omlx.cluster.versions import collect_versions
from omlx.engine_pool import EnginePool

from .conftest import make_settings, running_manager

# world_size=2, both head counts divisible by 2 (matches test_pool_coexistence.py).
_CONFIG = {"model_type": "llama", "num_attention_heads": 8, "num_key_value_heads": 8}
_MODEL_BYTES = 10_000_000
_EST_SIZE = int(_MODEL_BYTES * 1.05)
_PER_RANK_ESTIMATE = int(_EST_SIZE / 2 * 1.15)
_HEAD_CEILING = 7_000_000  # too small to fit locally; large enough per-rank
_WORKER_CEILING = 9_000_000
_REVISION = "a" * 40  # a syntactically valid 40-hex commit sha


def _head_settings(tmp_path, **overrides):
    opts = {
        "role": "head",
        "data_plane_subnet": "10.0.2.0/24",
        "data_plane_address": "10.0.2.1",
        # These tests drive a fake worker directly (no real HeartbeatSender
        # loop keeping the member alive) and can take longer than the
        # conftest default's 0.2s member_timeout_s -- a long enough ceiling
        # here means the member never goes "lost" mid-test.
        "member_timeout_s": 30.0,
    }
    opts.update(overrides)
    return make_settings(tmp_path / "head", **opts)


def _worker_settings(tmp_path, **overrides):
    opts = {"data_plane_subnet": "10.0.2.0/24", "data_plane_address": "10.0.2.2"}
    opts.update(overrides)
    return make_settings(tmp_path / "worker", role="worker", **opts)


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


def _formation(manager, pool):
    def _spawn_leader(**kwargs):
        return FakeLeader()

    return FormationManager(
        manager,
        spawn_leader_fn=_spawn_leader,
        engine_factory=lambda **kwargs: FakeEngine(),
        model_resolver=lambda model_id: (
            pool.get_entry(model_id).model_path
            if pool.get_entry(model_id) is not None
            else None
        ),
    )


async def _activate_member(manager, *, memory_ceiling: int = _WORKER_CEILING):
    reply = await manager.join(
        peer_host="10.0.2.9",
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


def _fake_session_launcher(file_contents: dict[str, bytes]):
    """Mirrors test_transfer.py's fake: simulates a completed round by
    writing each round entry's bytes directly into the staging dir the
    executor handed us, standing in for a real transfer_rank subprocess.
    """

    class _FakeProcess:
        def wait(self, timeout=None):
            return 0

    class _FakeSessionLeader:
        def __init__(self):
            self.process = _FakeProcess()

    class _FakeSession:
        def __init__(self):
            self.leader = _FakeSessionLeader()

        def stop(self):
            pass

        def kill(self):
            pass

    def launcher(*, rank, world_size, ips, base_port, argv_builder, **kwargs):
        argv = argv_builder(rank)
        manifest_path = Path(argv[argv.index("--manifest") + 1])
        staging_dir = Path(argv[argv.index("--root") + 1])
        entries = json.loads(manifest_path.read_text())
        for entry in entries:
            data = file_contents.get(entry["relative_path"])
            if data is None:
                continue
            target = staging_dir / entry["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return _FakeSession()

    return launcher


def _make_pool(tmp_path, *, hub_cache: bool, ceiling: int = _HEAD_CEILING):
    """A real EnginePool with one too-big-to-fit-locally model.

    ``hub_cache=False`` -- a bare local folder (peer-only, no
    ``source_repo_id``): ``<models>/target/``.
    ``hub_cache=True`` -- a hub-cache layout (both peer and HF viable,
    D6/CL5-07's "'--' form" id): ``<models>/models--mlx-community--target/
    snapshots/<40-hex>/`` with ``refs/main``, discovered as
    ``mlx-community--target``.
    """
    models_root = tmp_path / "models"
    config = json.dumps(_CONFIG).encode()
    weights = b"0" * _MODEL_BYTES

    if hub_cache:
        repo_dir = models_root / "models--mlx-community--target"
        snapshot_dir = repo_dir / "snapshots" / _REVISION
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_bytes(config)
        (snapshot_dir / "model.safetensors").write_bytes(weights)
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(_REVISION)
        model_id = "mlx-community--target"
        model_dir = snapshot_dir
    else:
        model_dir = models_root / "target"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_bytes(config)
        (model_dir / "model.safetensors").write_bytes(weights)
        model_id = "target"

    pool = EnginePool()
    pool._get_final_ceiling = lambda c=ceiling: c
    pool.discover_models(str(models_root))
    assert pool.get_entry(model_id) is not None, "fixture did not discover as expected"
    return pool, model_id, model_dir


def _mirror_file_contents(model_dir: Path) -> dict[str, bytes]:
    return {
        "config.json": (model_dir / "config.json").read_bytes(),
        "model.safetensors": (model_dir / "model.safetensors").read_bytes(),
    }


async def _drive_worker(
    fm: FormationManager,
    tm: TransferManager,
    worker_executor: TransferWorkerExecutor,
    member,
    worker_root: Path,
    model_id: str,
    stop_event: asyncio.Event,
) -> None:
    """Drains BOTH formation and transfer commands each tick (mirrors
    test_pool_coexistence.py's `_drive_worker` + test_transfer.py's own).
    Presence is answered by an ACTUAL filesystem check under
    ``worker_root`` rather than a static flag -- a transfer that lands
    files there must make the very next presence check see them, exactly
    like the real worker's fresh-per-command discovery scan (advisor
    finding: `_default_resolve_model` never caches).
    """
    seen_transfer: set[tuple[str, int]] = set()
    while not stop_event.is_set():
        for command in fm.commands_for(member.id):
            kind = command["kind"]
            if kind == "presence":
                present = (worker_root / model_id / "config.json").exists()
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
        for wire in tm.commands_for(member.id):
            key = (wire["job_id"], wire["step"])
            if key in seen_transfer:
                continue
            seen_transfer.add(key)
            ack = await worker_executor.dispatch(parse_command(wire))
            tm.record_transfer_updates(member, [ack])
        drained = worker_executor.pending_transfer_updates()
        if drained:
            tm.record_transfer_updates(member, drained)
        await asyncio.sleep(0.003)


class _Harness:
    def __init__(
        self, manager, member, pool, model_id, worker_root, driver_task, stop_event
    ):
        self.manager = manager
        self.member = member
        self.pool = pool
        self.model_id = model_id
        self.worker_root = worker_root
        self._driver_task = driver_task
        self._stop_event = stop_event

    async def close(self):
        self._stop_event.set()
        await self._driver_task
        await self.pool._drain_cluster_unload_driver()
        set_engine_pool_getter(None)


@contextlib.asynccontextmanager
async def harness(tmp_path, *, hub_cache: bool = False):
    settings = _head_settings(tmp_path)
    pool, model_id, model_dir = _make_pool(tmp_path, hub_cache=hub_cache)
    async with running_manager(settings) as manager:
        member = await _activate_member(manager)
        manager._formation = _formation(manager, pool)
        manager._transfer = TransferManager(manager)
        # NOT set_engine_pool_getter(lambda: pool): these tests call
        # `pool.load_cluster_model(...)` directly (no need for the head's
        # own registry lookup), and the global getter is also what
        # TransferWorkerExecutor._pool_conflict reads on the worker side --
        # in-process, that would be THIS SAME head pool, so the just-claimed
        # (is_loading=True) entry would false-positive CL5-10's "already
        # loaded/loading on this node" refusal. Leaving it unset is correct
        # here (advisor-flagged trap); route-level tests that need
        # get_engine_pool() wire it in their own harness instead.
        worker_settings = _worker_settings(tmp_path)
        worker_root = worker_settings.get_effective_model_dirs()[0]
        worker_root.mkdir(parents=True, exist_ok=True)
        launcher = _fake_session_launcher(_mirror_file_contents(model_dir))
        worker_executor = TransferWorkerExecutor(
            worker_settings, session_launcher=launcher
        )

        stop = asyncio.Event()
        driver = asyncio.create_task(
            _drive_worker(
                manager._formation,
                manager._transfer,
                worker_executor,
                member,
                worker_root,
                model_id,
                stop,
            )
        )
        h = _Harness(manager, member, pool, model_id, worker_root, driver, stop)
        try:
            yield h
        finally:
            await h.close()


async def _wait_for_terminal_job(manager, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = manager.transfer.snapshot()["jobs"]
        if jobs and jobs[-1]["status"] in ("done", "error", "aborted"):
            return jobs[-1]
        await asyncio.sleep(0.005)
    raise TimeoutError("transfer job never reached a terminal status")


async def _wait_for_any_job(manager, *, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = manager.transfer.snapshot()["jobs"]
        if jobs:
            return jobs[-1]["id"]
        await asyncio.sleep(0.005)
    raise TimeoutError("no transfer job was registered in time")


# ---- happy path: worker absent -> transfer -> forms -------------------------


async def test_absent_worker_transfers_then_forms_zero_preemptive_action(tmp_path):
    async with harness(tmp_path) as h:
        result = await h.pool.load_cluster_model(h.model_id, "auto")
        assert result["status"] == "ready"
        assert result["decision"]["mode"] == "distributed"

        entry = h.pool.get_entry(h.model_id)
        assert entry.kind == "cluster"
        assert entry.engine is not None

        job = await _wait_for_terminal_job(h.manager)
        assert job["status"] == "done"
        assert job["source"] == "peer"
        final = h.worker_root / h.model_id / "config.json"
        assert final.exists()


# ---- D6 source selection ----------------------------------------------------


async def test_ambiguous_no_source_no_choice_flag_auto_picks_peer(tmp_path):
    """The implicit on-demand path (`allow_source_choice` unset) never asks
    -- both a peer transfer and an HF fan-out are viable (hub-cache
    fixture) but it deterministically picks peer (E5)."""
    async with harness(tmp_path, hub_cache=True) as h:
        result = await h.pool.load_cluster_model(h.model_id, "auto")
        assert result["status"] == "ready"
        job = await _wait_for_terminal_job(h.manager)
        assert job["source"] == "peer"


async def test_ambiguous_with_choice_flag_returns_choice_required_and_rolls_back(
    tmp_path,
):
    async with harness(tmp_path, hub_cache=True) as h:
        result = await h.pool.load_cluster_model(
            h.model_id, "auto", allow_source_choice=True
        )
        assert result["status"] == "choice_required"
        assert result["peer_viable"] is True
        assert result["hf_viable"] is True

        entry = h.pool.get_entry(h.model_id)
        assert entry.is_loading is False
        assert entry.kind == "local"
        assert entry.cluster_head_share is None

        # No transfer job was ever started for the ambiguous call.
        assert h.manager.transfer.snapshot()["jobs"] == []

        # An explicit source bypasses the ambiguity and proceeds.
        result2 = await h.pool.load_cluster_model(
            h.model_id, "auto", source="peer", allow_source_choice=True
        )
        assert result2["status"] == "ready"


async def test_explicit_hf_source_not_viable_raises(tmp_path):
    async with harness(tmp_path, hub_cache=False) as h:
        with pytest.raises(ClusterError) as excinfo:
            await h.pool.load_cluster_model(h.model_id, "auto", source="hf")
        assert excinfo.value.status_code == 424

        entry = h.pool.get_entry(h.model_id)
        assert entry.is_loading is False
        assert entry.kind == "local"


# ---- D5 rollback rows ---------------------------------------------------


async def test_rollback_on_raising_transfer_then_subsequent_load_proceeds(
    tmp_path, monkeypatch
):
    from omlx.cluster import transfer as transfer_mod

    # `_resolve_cluster_presence` always supplies a `cache_dir`, so
    # `TransferManager._build_manifest` always goes through
    # `manifest_mod.cached_or_build_manifest` -- patch THAT (not the
    # constructor-injectable `manifest_builder`, which that codepath never
    # reaches) to force a raise during manifest build.
    def _raising(*_args, **_kwargs):
        raise RuntimeError("manifest build boom")

    async with harness(tmp_path) as h:
        monkeypatch.setattr(
            transfer_mod.manifest_mod, "cached_or_build_manifest", _raising
        )
        with pytest.raises(RuntimeError, match="boom"):
            await h.pool.load_cluster_model(h.model_id, "auto")

        entry = h.pool.get_entry(h.model_id)
        assert entry.is_loading is False
        assert entry.kind == "local"
        assert entry.cluster_head_share is None
        assert entry.cluster_original_estimated_size is None

        # No transfer job was ever registered (the raise happens before
        # `TransferJob` creation) -- the operation gate must be clear too.
        assert h.manager.transfer.snapshot()["jobs"] == []

        # Un-break manifest building and confirm a subsequent load proceeds.
        monkeypatch.undo()
        result = await h.pool.load_cluster_model(h.model_id, "auto")
        assert result["status"] == "ready"


async def test_rollback_on_aborted_transfer_then_subsequent_load_proceeds(tmp_path):
    async with harness(tmp_path) as h:
        load_task = asyncio.create_task(h.pool.load_cluster_model(h.model_id, "auto"))
        job_id = await _wait_for_any_job(h.manager)
        await h.manager.transfer.abort_transfer(job_id)

        with pytest.raises(ClusterError) as excinfo:
            await load_task
        assert excinfo.value.status_code == 424

        job = h.manager.transfer.job(job_id)
        assert job is not None
        assert job.status == "aborted"

        entry = h.pool.get_entry(h.model_id)
        assert entry.is_loading is False
        assert entry.kind == "local"
        assert entry.cluster_head_share is None

        result = await h.pool.load_cluster_model(h.model_id, "auto")
        assert result["status"] == "ready"
