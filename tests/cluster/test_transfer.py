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
import os
import time
import types
from pathlib import Path

import pytest

from omlx.cluster.manifest import ManifestError
from omlx.cluster.protocol import parse_command
from omlx.cluster.state import FileManifestEntry, Member, TransferJob
from omlx.cluster.transfer import (
    TransferError,
    TransferManager,
    TransferWorkerExecutor,
    _JobRuntime,
    _WorkerTransferJob,
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
# CL5-08: symlinked-ancestor destination is refused BEFORE os.replace
# =============================================================================


def test_finalize_round_refuses_symlinked_ancestor_before_replace(tmp_path):
    """Pins the actual defect: the pre-fix code called `os.replace` before
    validating containment, so `rename(2)`'s own directory-component symlink
    following would already have written the file OUTSIDE the model root by
    the time the (post-replace-only) check ran. The fix validates realpath
    containment before `os.replace`, so a pre-planted symlinked ancestor is
    refused before anything is written through it."""
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, final_dir / "evil")  # "evil" resolves OUTSIDE final_dir

    staging_dir = tmp_path / "staging"
    content = b"{}"
    staged = staging_dir / "evil" / "payload.json"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(content)

    entry = FileManifestEntry(
        relative_path="evil/payload.json", size=len(content), sha256=_sha(content)
    )
    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    job = _WorkerTransferJob(
        job_id="j1",
        model_id="m",
        manifest=(entry,),
        source="peer",
        epoch="",
        final_dir=final_dir,
    )

    newly_have = executor._finalize_round(job, [entry], staging_dir)

    assert newly_have == []
    assert not (outside / "payload.json").exists()  # never wrote outside the root
    assert not staged.exists()  # staged copy was cleaned up, never moved


# =============================================================================
# CL5-11: staging volume guards (free space, total quota, orphan sweep)
# =============================================================================


async def test_round_rejects_when_staging_volume_lacks_free_space(
    tmp_path, monkeypatch
):
    from omlx.cluster import transfer as transfer_mod

    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    await executor.dispatch(parse_command(_start_command()))

    monkeypatch.setattr(
        transfer_mod.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(total=0, used=0, free=1),
    )
    ack = await executor.dispatch(parse_command(_round_command()))
    assert ack["status"] == "rejected"
    assert "CL5-11" in ack["detail"]


async def test_round_rejects_when_total_staging_quota_exceeded(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    await executor.dispatch(parse_command(_start_command()))

    monkeypatch.setattr(transfer_mod, "MAX_TOTAL_STAGING_BYTES", 1)
    ack = await executor.dispatch(parse_command(_round_command()))
    assert ack["status"] == "rejected"
    assert "quota" in ack["detail"]


def test_sweep_orphaned_staging_dirs_removes_only_stale_ones(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    monkeypatch.setattr(transfer_mod.tempfile, "gettempdir", lambda: str(tmp_path))

    stale = tmp_path / "omlx-transfer-staging-stale"
    stale.mkdir()
    stale_file = stale / "f.bin"
    stale_file.write_bytes(b"x")
    old = time.time() - 10_000
    os.utime(stale_file, (old, old))

    fresh = tmp_path / "omlx-transfer-hf-staging-fresh"
    fresh.mkdir()
    (fresh / "f.bin").write_bytes(b"x")

    unrelated = tmp_path / "not-a-staging-dir"
    unrelated.mkdir()

    removed = transfer_mod.sweep_orphaned_staging_dirs(older_than_s=60.0)

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_executor_construction_sweeps_orphaned_staging_dirs(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    calls: list[dict] = []
    monkeypatch.setattr(
        transfer_mod,
        "sweep_orphaned_staging_dirs",
        lambda **kw: calls.append(kw) or 0,
    )
    TransferWorkerExecutor(_worker_settings(tmp_path))
    assert len(calls) == 1


# =============================================================================
# CL5-11: worker-side new-job rate limit
# =============================================================================


async def test_new_transfer_job_is_rate_limited(tmp_path):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    executor = TransferWorkerExecutor(settings)

    first = await executor.dispatch(parse_command(_start_command(job_id="j1")))
    assert first["status"] == "accepted"

    second = await executor.dispatch(parse_command(_start_command(job_id="j2")))
    assert second["status"] == "rejected"
    assert "rate limited" in second["detail"]


async def test_redelivered_start_for_the_same_job_is_exempt_from_the_rate_limit(
    tmp_path,
):
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    executor = TransferWorkerExecutor(settings)

    first = await executor.dispatch(parse_command(_start_command(job_id="j1")))
    assert first["status"] == "accepted"

    # A redelivered START for the SAME job_id must never be rejected as a
    # rate-limited "new" job -- idempotent redelivery is exempt.
    again = await executor.dispatch(parse_command(_start_command(job_id="j1")))
    assert again["status"] == "accepted"


async def test_rate_limit_clears_after_the_window(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "NEW_JOB_RATE_LIMIT_S", 0.01)
    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)
    executor = TransferWorkerExecutor(settings)

    first = await executor.dispatch(parse_command(_start_command(job_id="j1")))
    assert first["status"] == "accepted"

    await asyncio.sleep(0.02)
    second = await executor.dispatch(parse_command(_start_command(job_id="j2")))
    assert second["status"] == "accepted"


# =============================================================================
# CL5-16: minimum-progress watchdog
# =============================================================================


class _BoundedBlockingProcess:
    """Stands in for a wedged rank process: `.wait()` blocks well past any
    watchdog under test, but is bounded so a leaked background thread (the
    watchdog's `run_in_executor` call cannot truly cancel a blocking
    syscall) does not hang the test session."""

    def wait(self, timeout=None):
        import threading

        threading.Event().wait(timeout=0.3)
        return None


class _FakeTransferSession:
    def __init__(self, process):
        self.leader = types.SimpleNamespace(process=process)
        self.kill_called = False
        self.stop_called = False

    def kill(self):
        self.kill_called = True

    def stop(self):
        self.stop_called = True


async def test_progress_watchdog_kills_a_stalled_round(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "MIN_PROGRESS_INTERVAL_S", 0.02)
    monkeypatch.setattr(transfer_mod, "MIN_PROGRESS_STRIKES", 2)

    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()  # never grows -- zero progress every poll

    cluster = _FakeTransferSession(_BoundedBlockingProcess())
    start = time.monotonic()
    await executor._wait_round_deadline(cluster, staging_dir)
    elapsed = time.monotonic() - start

    assert cluster.kill_called is True
    assert elapsed < 1.0  # event-driven -- nowhere near ROUND_DEADLINE_S


async def test_progress_watchdog_does_not_kill_a_growing_round(tmp_path, monkeypatch):
    from omlx.cluster import transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "MIN_PROGRESS_INTERVAL_S", 0.02)
    monkeypatch.setattr(transfer_mod, "MIN_PROGRESS_STRIKES", 2)

    settings = _worker_settings(tmp_path)
    executor = TransferWorkerExecutor(settings)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    class _GrowsThenExitsProcess:
        def wait(self, timeout=None):
            import time as time_mod

            for i in range(20):
                time_mod.sleep(0.01)
                (staging_dir / f"chunk-{i}").write_bytes(b"x" * 4096)
            return 0

    cluster = _FakeTransferSession(_GrowsThenExitsProcess())
    await executor._wait_round_deadline(cluster, staging_dir)

    assert cluster.kill_called is False


async def test_watchdog_kill_still_releases_the_session_via_finally(
    tmp_path, monkeypatch
):
    """End-to-end through `_run_round`: the watchdog firing must still hit
    `_run_round`'s `finally: cluster.stop()` (the single-active gate release
    on the worker side) and let the round complete with a terminal update,
    never leaving the job hung."""
    from omlx.cluster import transfer as transfer_mod

    monkeypatch.setattr(transfer_mod, "MIN_PROGRESS_INTERVAL_S", 0.02)
    monkeypatch.setattr(transfer_mod, "MIN_PROGRESS_STRIKES", 2)

    settings = _worker_settings(tmp_path)
    root = settings.get_effective_model_dirs()[0]
    root.mkdir(parents=True, exist_ok=True)

    session = _FakeTransferSession(_BoundedBlockingProcess())

    def launcher(*, rank, world_size, ips, base_port, argv_builder, **kwargs):
        argv_builder(rank)  # discover the staging dir/manifest path; write nothing
        return session

    executor = TransferWorkerExecutor(settings, session_launcher=launcher)
    await executor.dispatch(parse_command(_start_command()))
    await executor.dispatch(parse_command(_round_command()))

    for _ in range(200):
        updates = executor.pending_transfer_updates()
        done = [u for u in updates if u.get("status") == "round_done"]
        if done:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("round never completed after the watchdog fired")

    assert session.kill_called is True
    assert session.stop_called is True


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


# =============================================================================
# CL5-04/05: head-side unbounded growth (pending_results, _jobs/_runtime)
# =============================================================================


async def test_pending_results_bounded_under_a_flood_of_out_of_range_steps(tmp_path):
    from omlx.cluster.transfer import MAX_PENDING_RESULTS_PER_JOB

    async with running_manager(_head_settings(tmp_path)) as manager:
        member = await _activate_member(manager)
        tm = TransferManager(manager)
        tm._jobs["j1"] = TransferJob(
            id="j1", kind="transfer", status="running", created_at=time.time()
        )
        tm._runtime["j1"] = _JobRuntime(member_id=member.id)

        # The verifier's probe shape: 200 ordinary updates + 50 out-of-range
        # steps, none of which anyone is awaiting -- every one lands in
        # `pending_results`.
        updates = [
            {"job_id": "j1", "step": step, "status": "have", "have": []}
            for step in range(1, 201)
        ] + [
            {"job_id": "j1", "step": step, "status": "have", "have": []}
            for step in range(10_000, 10_050)
        ]
        tm.record_transfer_updates(member, updates)

        assert len(tm._runtime["j1"].pending_results) <= MAX_PENDING_RESULTS_PER_JOB


async def test_finished_jobs_are_pruned_to_the_cap(tmp_path):
    from omlx.cluster.transfer import MAX_FINISHED_JOBS

    async with running_manager(_head_settings(tmp_path)) as manager:
        tm = TransferManager(manager)
        total = MAX_FINISHED_JOBS + 5
        for i in range(total):
            job_id = f"job-{i}"
            tm._jobs[job_id] = TransferJob(
                id=job_id, kind="transfer", status="running", created_at=float(i)
            )
            tm._runtime[job_id] = _JobRuntime(member_id="m")
            tm._finish(job_id, "done")

        assert len(tm._jobs) == MAX_FINISHED_JOBS
        assert len(tm._runtime) == MAX_FINISHED_JOBS
        # Oldest (lowest created_at) are the ones pruned; most recent kept.
        assert "job-0" not in tm._jobs
        assert f"job-{total - 1}" in tm._jobs


async def test_a_still_running_job_is_never_pruned(tmp_path):
    """Only TERMINAL jobs count against the cap -- a job still in flight
    must never be evicted just because many older finished jobs exist."""
    from omlx.cluster.transfer import MAX_FINISHED_JOBS

    async with running_manager(_head_settings(tmp_path)) as manager:
        tm = TransferManager(manager)
        tm._jobs["running"] = TransferJob(
            id="running", kind="transfer", status="running", created_at=0.0
        )
        tm._runtime["running"] = _JobRuntime(member_id="m")

        for i in range(MAX_FINISHED_JOBS + 5):
            job_id = f"job-{i}"
            tm._jobs[job_id] = TransferJob(
                id=job_id, kind="transfer", status="running", created_at=float(i + 1)
            )
            tm._runtime[job_id] = _JobRuntime(member_id="m")
            tm._finish(job_id, "done")

        assert "running" in tm._jobs
        assert tm._jobs["running"].status == "running"
