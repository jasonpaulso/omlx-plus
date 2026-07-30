# SPDX-License-Identifier: Apache-2.0
"""S5 model distribution: head TransferManager + worker TransferWorkerExecutor.

Real head/worker objects are wired together with a hand-rolled polling glue
loop (mirrors test_formation.py's `_drive_worker`) so the whole D1-D6
protocol round-trip runs, with only the process-spawn boundary
(`launch_transfer_session`) faked -- forming a real mlx ring session needs
real hardware and is the P3 live-rig's job, not pytest's.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest

from omlx.cluster.manifest import ManifestError
from omlx.cluster.protocol import parse_command
from omlx.cluster.state import Member
from omlx.cluster.transfer import (
    TransferError,
    TransferManager,
    TransferWorkerExecutor,
    resolve_transfer_destination,
)
from omlx.cluster.versions import collect_versions

from .conftest import make_settings, running_manager


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# resolve_transfer_destination (CL5-06)
# =============================================================================


def test_resolve_transfer_destination_single_segment(tmp_path):
    dest = resolve_transfer_destination("my-model", tmp_path)
    assert dest == (tmp_path / "my-model").resolve()


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "..",
        ".hidden",
        "a/../b",
        "/abs",
        "a\\b",
        "a\x00b",
        "bad*char",
        "a" * 300,
    ],
)
def test_resolve_transfer_destination_rejects_invalid_ids(tmp_path, bad_id):
    with pytest.raises(ManifestError):
        resolve_transfer_destination(bad_id, tmp_path)


def test_resolve_transfer_destination_refuses_two_segment_ids(tmp_path):
    # CL5-07/R3 deviation: discover_models_from_dirs drops the org prefix on
    # a two-level layout (model_discovery.py:1338-1340), so a two-segment
    # destination could never be rediscovered back to this exact id --
    # refused by name rather than silently written.
    with pytest.raises(ManifestError, match="2 path segments"):
        resolve_transfer_destination("org/name", tmp_path)


def test_resolve_transfer_destination_refuses_existing_symlink(tmp_path):
    import os

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-model"
    os.symlink(real, link)
    with pytest.raises(ManifestError, match="symlink"):
        resolve_transfer_destination("linked-model", tmp_path)


# =============================================================================
# TransferWorkerExecutor -- TRANSFER_START
# =============================================================================


def _worker_settings(tmp_path, **overrides):
    opts = {"data_plane_subnet": "10.0.2.0/24", "data_plane_address": "10.0.2.2"}
    opts.update(overrides)
    return make_settings(tmp_path / "worker", role="worker", **opts)


def _start_command(**over):
    base = {
        "kind": "transfer_start",
        "schema_version": 3,
        "job_id": "j1",
        "step": 1,
        "model_id": "target-model",
        "manifest": [
            {"relative_path": "config.json", "size": 2, "sha256": _sha(b"{}")}
        ],
        "source": "peer",
        "epoch": "ep1",
        "repair": False,
    }
    base.update(over)
    return base


async def test_transfer_start_accepts_and_reports_have(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    executor = TransferWorkerExecutor(settings)
    command = parse_command(_start_command())
    ack = await executor.dispatch(command)
    assert ack["status"] == "accepted"
    # The ack carries the worker's own DATA-PLANE address (never
    # member.endpoint's control-plane one) so the head can address a round
    # session's ring -- mirrors PresenceCommand's reply for formation.
    assert ack["data_plane_address"] == "10.0.2.2"

    # The have-scan is an owned task; wait for its report.
    for _ in range(100):
        updates = executor.pending_transfer_updates()
        if updates:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("no have report arrived")
    assert updates[0]["status"] == "have"
    assert updates[0]["have"] == []  # nothing on disk yet


async def test_transfer_start_rejects_epoch_mismatch(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    executor.set_epoch("current-epoch")
    ack = await executor.dispatch(parse_command(_start_command(epoch="stale-epoch")))
    assert ack["status"] == "rejected"
    assert "epoch" in ack["detail"]


async def test_transfer_start_rejects_bad_manifest_entry(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    bad = _start_command(
        manifest=[{"relative_path": "evil.py", "size": 1, "sha256": "0" * 64}]
    )
    ack = await executor.dispatch(parse_command(bad))
    assert ack["status"] == "rejected"


async def test_transfer_start_rejects_two_segment_model_id(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    ack = await executor.dispatch(parse_command(_start_command(model_id="org/name")))
    assert ack["status"] == "rejected"
    assert "2 path segments" in ack["detail"]


async def test_transfer_start_rejects_loaded_destination(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)

    class _FakeEntry:
        engine = object()
        is_loading = False

    class _FakePool:
        def get_entry(self, model_id):
            return _FakeEntry() if model_id == "target-model" else None

    from omlx.cluster import manager as manager_mod

    manager_mod.set_engine_pool_getter(lambda: _FakePool())
    try:
        ack = await executor.dispatch(parse_command(_start_command()))
    finally:
        manager_mod.set_engine_pool_getter(None)
    assert ack["status"] == "rejected"
    assert "loaded" in ack["detail"]


async def test_transfer_start_hf_requires_repo_id_and_40_hex_revision(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    ack = await executor.dispatch(
        parse_command(
            _start_command(source="hf", hf_repo_id="org/name", hf_revision="main")
        )
    )
    assert ack["status"] == "rejected"
    assert "40-hex" in ack["detail"]


async def test_transfer_start_hf_disabled_by_settings(tmp_path):
    settings = _worker_settings(tmp_path, allow_hf_transfer=False)
    executor = TransferWorkerExecutor(settings)
    ack = await executor.dispatch(
        parse_command(
            _start_command(source="hf", hf_repo_id="org/name", hf_revision="a" * 40)
        )
    )
    assert ack["status"] == "rejected"
    assert "allow_hf_transfer" in ack["detail"]


# =============================================================================
# _compute_have: only digest-verified files count
# =============================================================================


async def test_scan_have_only_counts_digest_verified_files(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    dest = root / "target-model"
    dest.mkdir(parents=True)
    (dest / "config.json").write_bytes(b"{}")  # matches _sha(b"{}")
    (dest / "corrupt.json").write_bytes(b"WRONG")  # will not match

    executor = TransferWorkerExecutor(settings)
    command = parse_command(
        _start_command(
            manifest=[
                {"relative_path": "config.json", "size": 2, "sha256": _sha(b"{}")},
                {
                    "relative_path": "corrupt.json",
                    "size": 5,
                    "sha256": _sha(b"RIGHT"),
                },
            ]
        )
    )
    await executor.dispatch(command)
    for _ in range(100):
        updates = executor.pending_transfer_updates()
        if updates:
            break
        await asyncio.sleep(0.005)
    assert updates[0]["have"] == ["config.json"]


async def test_scan_have_repair_ignores_existing_files(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    dest = root / "target-model"
    dest.mkdir(parents=True)
    (dest / "config.json").write_bytes(b"{}")

    executor = TransferWorkerExecutor(settings)
    command = parse_command(_start_command(repair=True))
    await executor.dispatch(command)
    for _ in range(100):
        updates = executor.pending_transfer_updates()
        if updates:
            break
        await asyncio.sleep(0.005)
    assert updates[0]["have"] == []


# =============================================================================
# TRANSFER_ROUND + finalize (digest verify, move, delete-on-mismatch)
# =============================================================================


def _fake_session_launcher(file_contents: dict[str, bytes]):
    """Simulates a completed round: writes each round entry's bytes
    directly into the staging dir the executor handed us (standing in for a
    real transfer_rank subprocess), and returns a fake session whose local
    rank has already "exited".
    """

    class _FakeProcess:
        def wait(self, timeout=None):
            return 0

    class _FakeLeader:
        def __init__(self):
            self.process = _FakeProcess()

    class _FakeSession:
        def __init__(self):
            self.leader = _FakeLeader()
            self.stop_called = False
            self.kill_called = False

        def stop(self):
            self.stop_called = True

        def kill(self):
            self.kill_called = True

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


def _round_command(**over):
    base = {
        "kind": "transfer_round",
        "schema_version": 3,
        "job_id": "j1",
        "step": 2,
        "subset": ["config.json"],
        "peers": ["10.0.2.1", "10.0.2.2"],
        "base_port": 41164,
    }
    base.update(over)
    return base


async def test_round_moves_verified_file_into_final_dir(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    content = b"{}"
    launcher = _fake_session_launcher({"config.json": content})
    executor = TransferWorkerExecutor(settings, session_launcher=launcher)

    start_ack = await executor.dispatch(parse_command(_start_command()))
    assert start_ack["status"] == "accepted"
    round_ack = await executor.dispatch(parse_command(_round_command()))
    assert round_ack["status"] == "accepted"

    for _ in range(200):
        updates = [u for u in executor.pending_transfer_updates()]
        if any(u.get("status") == "round_done" for u in updates):
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("round never completed")

    final_file = root / "target-model" / "config.json"
    assert final_file.read_bytes() == content


async def test_round_deletes_digest_mismatch_never_moves(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    # The manifest declares config.json's sha256 as _sha(b"{}"), but the
    # fake session will deliver the WRONG bytes.
    launcher = _fake_session_launcher({"config.json": b"corrupted-bytes"})
    executor = TransferWorkerExecutor(settings, session_launcher=launcher)

    await executor.dispatch(parse_command(_start_command()))
    await executor.dispatch(parse_command(_round_command()))

    for _ in range(200):
        updates = executor.pending_transfer_updates()
        done = [u for u in updates if u.get("status") == "round_done"]
        if done:
            result = done[0]
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("round never completed")

    assert result["transferred"] == []
    final_file = root / "target-model" / "config.json"
    assert not final_file.exists()


async def test_round_rejects_unknown_job():
    settings = make_settings(Path("/tmp"), role="worker")
    executor = TransferWorkerExecutor(settings)
    ack = await executor.dispatch(parse_command(_round_command(job_id="ghost")))
    assert ack["status"] == "rejected"


async def test_round_rejects_subset_outside_manifest(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    await executor.dispatch(parse_command(_start_command()))
    ack = await executor.dispatch(
        parse_command(_round_command(subset=["not-in-manifest.json"]))
    )
    assert ack["status"] == "rejected"
    assert "outside the job's manifest" in ack["detail"]


async def test_round_rejects_out_of_scope_peer(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    await executor.dispatch(parse_command(_start_command()))
    ack = await executor.dispatch(
        parse_command(_round_command(peers=["10.0.2.1", "8.8.8.8"]))
    )
    assert ack["status"] == "rejected"


# =============================================================================
# TRANSFER_ABORT
# =============================================================================


async def test_abort_cancels_owned_task_and_marks_aborted(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)

    async def _never_finishes(*_args, **_kwargs):
        await asyncio.sleep(1000)

    executor = TransferWorkerExecutor(settings)
    await executor.dispatch(parse_command(_start_command()))
    job = executor._jobs["j1"]
    # Replace the owned task with one that hangs, to exercise cancellation.
    job.task.cancel()
    job.task = asyncio.create_task(_never_finishes())

    abort_ack = await executor.dispatch(
        parse_command(
            {
                "kind": "transfer_abort",
                "schema_version": 3,
                "job_id": "j1",
                "step": 99,
            }
        )
    )
    assert abort_ack["status"] == "aborted"
    assert job.status == "aborted"
    assert job.task.cancelled()


async def test_abort_unknown_job_is_rejected(tmp_path):
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    ack = await executor.dispatch(
        parse_command(
            {
                "kind": "transfer_abort",
                "schema_version": 3,
                "job_id": "ghost",
                "step": 1,
            }
        )
    )
    assert ack["status"] == "rejected"


# =============================================================================
# HF path (D6)
# =============================================================================


async def test_hf_required_entry_missing_is_terminal(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)

    async def _hf_download_writes_nothing(*_args, **_kwargs):
        return None

    executor = TransferWorkerExecutor(
        settings, hf_downloader=_hf_download_writes_nothing
    )
    command = parse_command(
        _start_command(source="hf", hf_repo_id="org/name", hf_revision="a" * 40)
    )
    ack = await executor.dispatch(command)
    assert ack["status"] == "accepted"

    for _ in range(200):
        updates = executor.pending_transfer_updates()
        errors = [u for u in updates if u.get("status") == "error"]
        if errors:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("HF job never reported terminal error")
    assert errors[0]["code"] == "hf_source_incomplete"


async def test_hf_success_moves_required_files(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    content = b"{}"

    async def _hf_download_writes_file(
        repo_id, target_dir, *, revision, ignore_patterns
    ):
        (Path(target_dir) / "config.json").write_bytes(content)

    executor = TransferWorkerExecutor(settings, hf_downloader=_hf_download_writes_file)
    command = parse_command(
        _start_command(source="hf", hf_repo_id="org/name", hf_revision="a" * 40)
    )
    await executor.dispatch(command)

    for _ in range(200):
        updates = executor.pending_transfer_updates()
        done = [u for u in updates if u.get("status") == "done"]
        if done:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("HF job never completed")
    assert (root / "target-model" / "config.json").read_bytes() == content


# =============================================================================
# TransferManager: single-active gate (D4)
# =============================================================================


def _head_settings(tmp_path, **overrides):
    return make_settings(
        tmp_path / "head",
        role="head",
        data_plane_subnet="10.0.2.0/24",
        data_plane_address="10.0.2.1",
        **overrides,
    )


async def test_operation_gate_refuses_concurrent_formation_and_transfer(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        manager.acquire_operation_gate("transfer", "t1")
        with pytest.raises(Exception) as excinfo:
            manager.acquire_operation_gate("formation", "f1")
        assert excinfo.value.status_code == 409
        assert "t1" in excinfo.value.detail

        manager.release_operation_gate("transfer", "t1")
        # Now the formation side can claim it.
        manager.acquire_operation_gate("formation", "f1")
        with pytest.raises(Exception) as excinfo2:
            manager.acquire_operation_gate("transfer", "t1")
        assert excinfo2.value.status_code == 409
        manager.release_operation_gate("formation", "f1")


async def test_operation_gate_release_is_a_noop_for_a_stale_owner(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        manager.acquire_operation_gate("transfer", "t1")
        # A stale release (wrong job_id) must not free someone else's slot.
        manager.release_operation_gate("transfer", "wrong-id")
        with pytest.raises(Exception) as excinfo:
            manager.acquire_operation_gate("formation", "f1")
        assert excinfo.value.status_code == 409


# =============================================================================
# TransferManager <-> TransferWorkerExecutor end-to-end round-trip
# =============================================================================


async def _activate_member(manager) -> Member:
    # Inside the 10.0.2.0/24 data-plane subnet the round-peer link-scope
    # check (CL2-03-style, mirrored in TransferWorkerExecutor) requires.
    reply = await manager.join(
        peer_host="10.0.2.9",
        port=40404,
        name="worker",
        versions=collect_versions().to_dict(),
    )
    member = manager.state.member(reply["member_id"])
    assert member is not None
    manager.record_heartbeat(member, seq=1, epoch="ep1")
    return member


async def _drive_worker(head, worker, member, stop_event):
    seen: set[tuple[str, int]] = set()
    while not stop_event.is_set():
        for wire in head.commands_for(member.id):
            key = (wire["job_id"], wire["step"])
            if key in seen:
                continue
            seen.add(key)
            command = parse_command(wire)
            ack = await worker.dispatch(command)
            head.record_transfer_updates(member, [ack])
        drained = worker.pending_transfer_updates()
        if drained:
            head.record_transfer_updates(member, drained)
        await asyncio.sleep(0.003)


async def test_diff_authority_round_shrinks_subset_to_done(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)

        source_dir = tmp_path / "source" / "target-model"
        source_dir.mkdir(parents=True)
        (source_dir / "config.json").write_bytes(b"{}")
        (source_dir / "model.safetensors").write_bytes(b"weights!")

        worker_settings = _worker_settings(tmp_path)
        worker_root = worker_settings.get_effective_model_dirs()[0]
        worker_root.mkdir(parents=True, exist_ok=True)
        launcher = _fake_session_launcher(
            {
                "config.json": b"{}",
                "model.safetensors": b"weights!",
            }
        )
        worker = TransferWorkerExecutor(worker_settings, session_launcher=launcher)

        tm = TransferManager(manager)
        manager._transfer = tm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(tm, worker, member, stop))
        try:
            job = await tm.start_transfer(
                "target-model",
                member=member,
                local_path=str(source_dir),
                source="peer",
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                current = tm.job(job.id)
                if current is not None and current.status in ("done", "error"):
                    break
                await asyncio.sleep(0.01)
            final_job = tm.job(job.id)
            assert final_job is not None
            assert final_job.status == "done", final_job.error
            assert set(final_job.have) == {"config.json", "model.safetensors"}
        finally:
            stop.set()
            await driver

        final_config = worker_root / "target-model" / "config.json"
        assert final_config.read_bytes() == b"{}"


async def test_round_peers_use_data_plane_not_control_plane_address(tmp_path):
    # A worker that never reported a data-plane address (e.g. misconfigured)
    # must fail the job with a named error, never send member.endpoint's
    # control-plane address into a round command.
    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)

        source_dir = tmp_path / "source" / "target-model"
        source_dir.mkdir(parents=True)
        (source_dir / "config.json").write_bytes(b"{}")

        worker_settings = _worker_settings(tmp_path, data_plane_address="")
        worker_root = worker_settings.get_effective_model_dirs()[0]
        worker_root.mkdir(parents=True, exist_ok=True)
        worker = TransferWorkerExecutor(worker_settings)

        tm = TransferManager(manager)
        manager._transfer = tm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(tm, worker, member, stop))
        try:
            job = await tm.start_transfer(
                "target-model", member=member, local_path=str(source_dir), source="peer"
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                current = tm.job(job.id)
                if current is not None and current.status in ("done", "error"):
                    break
                await asyncio.sleep(0.01)
            final_job = tm.job(job.id)
            assert final_job is not None
            assert final_job.status == "error"
            assert "data-plane address" in final_job.error
        finally:
            stop.set()
            await driver


async def test_round_cap_gives_up_after_no_progress(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "ROUND_CAP", 2)

    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)

        source_dir = tmp_path / "source" / "target-model"
        source_dir.mkdir(parents=True)
        (source_dir / "config.json").write_bytes(b"{}")

        worker_settings = _worker_settings(tmp_path)
        worker_root = worker_settings.get_effective_model_dirs()[0]
        worker_root.mkdir(parents=True, exist_ok=True)
        # The fake launcher delivers nothing -- every round makes zero
        # progress, so the round cap must fire.
        launcher = _fake_session_launcher({})
        worker = TransferWorkerExecutor(worker_settings, session_launcher=launcher)

        tm = TransferManager(manager)
        manager._transfer = tm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(tm, worker, member, stop))
        try:
            job = await tm.start_transfer(
                "target-model",
                member=member,
                local_path=str(source_dir),
                source="peer",
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                current = tm.job(job.id)
                if current is not None and current.status in ("done", "error"):
                    break
                await asyncio.sleep(0.01)
            final_job = tm.job(job.id)
            assert final_job is not None
            assert final_job.status == "error"
            assert "consecutive" in final_job.error
        finally:
            stop.set()
            await driver


# =============================================================================
# TransferManager.abort_transfer (S5 P2: completes D4's abort mechanism --
# P1 wired TRANSFER_ABORT's worker-side handling but never a head-side
# caller that publishes it)
# =============================================================================


async def test_abort_transfer_marks_head_and_worker_job_aborted(tmp_path):
    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)

        source_dir = tmp_path / "source" / "target-model"
        source_dir.mkdir(parents=True)
        (source_dir / "config.json").write_bytes(b"{}")

        worker_settings = _worker_settings(tmp_path)
        worker_root = worker_settings.get_effective_model_dirs()[0]
        worker_root.mkdir(parents=True, exist_ok=True)
        launcher = _fake_session_launcher({"config.json": b"{}"})
        worker = TransferWorkerExecutor(worker_settings, session_launcher=launcher)

        tm = TransferManager(manager)
        manager._transfer = tm

        stop = asyncio.Event()
        driver = asyncio.create_task(_drive_worker(tm, worker, member, stop))
        try:
            job = await tm.start_transfer(
                "target-model", member=member, local_path=str(source_dir), source="peer"
            )
            aborted = await tm.abort_transfer(job.id)
            assert aborted.status == "aborted"
            assert tm.job(job.id).status == "aborted"

            # The single-active gate must be released -- a fresh operation
            # can proceed immediately, not wedged behind the aborted job.
            manager.acquire_operation_gate("formation", "f1")
            manager.release_operation_gate("formation", "f1")

            # Idempotent: aborting an already-terminal job is a safe no-op.
            again = await tm.abort_transfer(job.id)
            assert again.status == "aborted"
        finally:
            stop.set()
            await driver

        with pytest.raises(TransferError) as excinfo:
            await tm.abort_transfer("no-such-job")
        assert excinfo.value.status_code == 404
