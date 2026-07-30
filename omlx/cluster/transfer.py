# SPDX-License-Identifier: Apache-2.0
"""S5 model distribution: head orchestration + worker execution (D1-D6).

Two halves live in this one file because they are two ends of the same
protocol and nothing else touches either: :class:`TransferManager` (head)
drives resumable, file-granular transfer jobs -- it is the diff authority,
runs rounds over a fresh 2-rank ring session per round (peer source) or
delegates to the worker's own HF downloader (D6), and enforces the D4
single-active cluster-operation gate together with
:class:`~omlx.cluster.formation.FormationManager`.
:class:`TransferWorkerExecutor` (worker) applies TRANSFER_* commands under
the same CL2-style confinement `WorkerCommandExecutor` applies to formation
commands -- every command is untrusted input, and the actual I/O always runs
as an OWNED TASK off the command dispatch path so a minutes-long round never
stalls the heartbeat/command loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import manifest as manifest_mod
from .launcher import TransferSpawnBoundError, launch_transfer_session
from .manifest import ManifestError
from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    TransferAbortCommand,
    TransferRoundCommand,
    TransferSource,
    TransferStartCommand,
    command_to_wire,
    make_job_update,
    make_transfer_update,
)
from .state import FileManifestEntry, Member, TransferJob

if TYPE_CHECKING:
    from ..settings import GlobalSettings
    from .manager import ClusterManager

logger = logging.getLogger(__name__)

# D2: give up after this many consecutive failed/no-progress rounds.
ROUND_CAP = 3
# D1b/CL5-16: wall-clock deadline for one round + the daemon-side watchdog
# that kills a session past it, releasing the gate in `finally`.
ROUND_DEADLINE_S = 1800.0
MIN_PROGRESS_INTERVAL_S = 5.0

# CL5-14, mirrored from the manifest builder's ignore precedent: the HF
# path's DETERMINISTIC ignore list (D3a/CL5-01) -- never the
# `model_info`-conditional one `hf_downloader.start_download` computes for
# the admin UI (which vanishes on a metadata failure).
HF_IGNORE_PATTERNS = ["*.bin", "original/**", "consolidated.*.pth"]

_MODEL_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_MODEL_ID_LENGTH = 200
# CL5-03: the cluster HF path requires a full 40-hex commit sha.
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class TransferError(Exception):
    """A transfer operation failed with a specific HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_transfer_destination(model_id: str, root: Path) -> Path:
    """CL5-06: the single, validating id->path mapping for a transfer
    destination.

    ``model_id`` is validated as 1-2 segments, strict charset, no dotfiles/
    ``..``/absolute/NUL/backslash, length-capped. A 2-segment id is then
    explicitly refused (a documented, reported plan deviation -- see the
    module docstring header note): ``discover_models_from_dirs`` drops the
    org prefix on an organized two-level layout
    (``model_discovery.py:1338-1340``, and its own docstring at
    ``:1270-1276`` states this), so a two-level destination could never be
    rediscovered back to the head's id, and every real caller
    (``EnginePool``/``load_cluster_model``) only ever supplies a
    slash-free discovery key anyway -- writing an undiscoverable directory
    would be strictly worse than refusing.

    Returns the resolved destination path. Does not create it, and does not
    check the filesystem beyond refusing an existing symlink at that exact
    path.
    """
    if not model_id or len(model_id) > MAX_MODEL_ID_LENGTH:
        raise ManifestError("model id is empty or exceeds the length cap")
    if "\x00" in model_id or "\\" in model_id:
        raise ManifestError("model id contains a NUL byte or backslash")
    segments = model_id.split("/")
    if len(segments) > 2:
        raise ManifestError(
            f"model id {model_id!r} has more than 2 path segments; refused"
        )
    for segment in segments:
        if not segment or segment in (".", "..") or segment.startswith("."):
            raise ManifestError(f"model id segment {segment!r} is invalid")
        if not _MODEL_ID_SEGMENT_RE.match(segment):
            raise ManifestError(
                f"model id segment {segment!r} has disallowed characters"
            )
    if len(segments) == 2:
        raise ManifestError(
            f"model id {model_id!r} has 2 path segments; the S5 destination "
            "mapping only supports single-segment ids that "
            "discover_models_from_dirs can resolve back to this exact id "
            "(a two-level org/name layout drops its prefix on rediscovery -- "
            "deliberate plan deviation, reported)"
        )
    root = Path(root).resolve()
    candidate = root / model_id
    if candidate.exists() and candidate.is_symlink():
        raise ManifestError(f"destination {candidate} is a symlink; refused")
    return candidate


# =============================================================================
# Head side
# =============================================================================


@dataclass
class _JobRuntime:
    """Ephemeral, non-persisted-shape state for one head-side job -- kept
    separate from the frozen :class:`~omlx.cluster.state.TransferJob`
    snapshot the rest of the system reads."""

    task: asyncio.Task[None] | None = None
    member_id: str = ""
    stalled_rounds: int = 0
    round_updates: dict[int, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict
    )


class TransferManager:
    """Head-side orchestration: diff authority, rounds, source selection,
    the single-active gate, and transfer-session spawning (D1/D2/D4/D6).
    """

    def __init__(
        self,
        manager: ClusterManager,
        *,
        manifest_builder: Callable[[Path], tuple[FileManifestEntry, ...]] | None = None,
        session_launcher: Callable[..., Any] | None = None,
        hf_downloader: Callable[..., Any] | None = None,
        python: str | None = None,
    ) -> None:
        self._manager = manager
        self._manifest_builder = manifest_builder or self._default_build_manifest
        self._session_launcher = session_launcher or launch_transfer_session
        self._hf_downloader = hf_downloader
        self._python = python
        self._pending: dict[str, tuple[dict[str, Any], ...]] = {}
        self._acks: dict[tuple[str, int], asyncio.Future[dict[str, Any]]] = {}
        self._jobs: dict[str, TransferJob] = {}
        self._runtime: dict[str, _JobRuntime] = {}

    # ---- read side (heartbeat + dashboard) --------------------------------

    def commands_for(self, member_id: str) -> list[dict[str, Any]]:
        return [dict(command) for command in self._pending.get(member_id, ())]

    def snapshot(self) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in list(self._jobs.values())[-10:]]}

    def job(self, job_id: str) -> TransferJob | None:
        return self._jobs.get(job_id)

    async def stop(self) -> None:
        for runtime in list(self._runtime.values()):
            if runtime.task is not None and not runtime.task.done():
                runtime.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runtime.task

    # ---- worker transfer updates (D1b) -------------------------------------

    def record_transfer_updates(
        self, member: Member, updates: list[dict[str, Any]]
    ) -> None:
        """Resolve acks and accumulate progress from a member's
        AUTHENTICATED transfer updates. NOT the same as an ack future
        resolving alone -- ``have``/``round_done``/``error`` reports are
        routed to whichever round future is awaiting them, and every update
        is also folded into the job's accumulated ``updates``.
        """
        for update in updates:
            if not isinstance(update, dict):
                continue
            job_id = str(update.get("job_id") or "")
            step_raw = update.get("step")
            step = int(step_raw) if isinstance(step_raw, int) else 0
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs[job_id] = replace(job, updated_at=time.time())
            key = (job_id, step)
            ack_future = self._acks.get(key)
            if ack_future is not None and not ack_future.done():
                ack_future.set_result(update)
                continue
            runtime = self._runtime.get(job_id)
            if runtime is not None:
                round_future = runtime.round_updates.get(step)
                if round_future is not None and not round_future.done():
                    round_future.set_result(update)

    # ---- job creation (E6 queue, D4 gate) ----------------------------------

    async def start_transfer(
        self,
        model_id: str,
        *,
        member: Member,
        local_path: str,
        source: str,
        hf_repo_id: str | None = None,
        hf_revision: str | None = None,
        cache_dir: Path | None = None,
    ) -> TransferJob:
        """Register a job and return once it is queued -- the actual I/O
        runs as an owned task, never blocking this call or the E6 queue
        beyond registration (D4).
        """
        return await self._manager._queue.submit(
            "transfer_start",
            lambda: self._start(
                model_id,
                member=member,
                local_path=local_path,
                source=source,
                hf_repo_id=hf_repo_id,
                hf_revision=hf_revision,
                cache_dir=cache_dir,
            ),
        )

    async def _start(
        self,
        model_id: str,
        *,
        member: Member,
        local_path: str,
        source: str,
        hf_repo_id: str | None,
        hf_revision: str | None,
        cache_dir: Path | None,
    ) -> TransferJob:
        job_id = secrets.token_hex(6)
        self._manager.acquire_operation_gate("transfer", job_id)
        try:
            manifest = await self._build_manifest(Path(local_path), cache_dir)
        except Exception:
            self._manager.release_operation_gate("transfer", job_id)
            raise
        job = TransferJob(
            id=job_id,
            kind="transfer",
            status="running",
            created_at=time.time(),
            manifest=manifest,
            model_id=model_id,
            member_id=member.id,
            source=source,
            updated_at=time.time(),
        )
        self._jobs[job_id] = job
        self._runtime[job_id] = _JobRuntime(member_id=member.id)
        task = asyncio.create_task(
            self._drive(job_id, member, source, hf_repo_id, hf_revision)
        )
        self._runtime[job_id].task = task
        return job

    async def _build_manifest(
        self, local_path: Path, cache_dir: Path | None
    ) -> tuple[FileManifestEntry, ...]:
        loop = asyncio.get_running_loop()
        if cache_dir is not None:
            return await loop.run_in_executor(
                None, manifest_mod.cached_or_build_manifest, local_path, cache_dir
            )
        return await loop.run_in_executor(None, self._manifest_builder, local_path)

    def _default_build_manifest(self, model_dir: Path) -> tuple[FileManifestEntry, ...]:
        return manifest_mod.build_manifest(model_dir)

    # ---- the owned task: rounds (peer) or HF fan-out delegation ------------

    async def _drive(
        self,
        job_id: str,
        member: Member,
        source: str,
        hf_repo_id: str | None,
        hf_revision: str | None,
    ) -> None:
        try:
            epoch = self._member_epoch(member)
            start_ack = await self._start_command(
                member, job_id, source, hf_repo_id, hf_revision, epoch
            )
            if start_ack.get("status") == "rejected":
                self._finish(job_id, "error", error=str(start_ack.get("detail") or ""))
                return
            if source == "hf":
                await self._drive_hf(job_id, member)
            else:
                await self._drive_rounds(job_id, member)
        except asyncio.CancelledError:
            self._finish(job_id, "aborted")
            raise
        except Exception as exc:  # noqa: BLE001 - recorded on the job
            logger.warning("cluster: transfer job %s failed: %s", job_id, exc)
            self._finish(job_id, "error", error=str(exc))
        finally:
            self._manager.release_operation_gate("transfer", job_id)

    def _member_epoch(self, member: Member) -> str:
        live = self._manager.liveness(member.id)
        return live.epoch if live is not None else ""

    async def _start_command(
        self,
        member: Member,
        job_id: str,
        source: str,
        hf_repo_id: str | None,
        hf_revision: str | None,
        epoch: str,
    ) -> dict[str, Any]:
        job = self._jobs[job_id]
        wire = command_to_wire(
            TransferStartCommand(
                schema_version=PROTOCOL_VERSION,
                job_id=job_id,
                step=1,
                model_id=job.model_id,
                manifest=[entry.to_dict() for entry in job.manifest],
                source=TransferSource(source),
                epoch=epoch,
                hf_repo_id=hf_repo_id,
                hf_revision=hf_revision,
            )
        )
        return await self._send_and_ack(member, wire)

    async def _send_and_ack(
        self, member: Member, wire: dict[str, Any], *, timeout: float = 30.0
    ) -> dict[str, Any]:
        key = (str(wire["job_id"]), int(wire["step"]))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._acks[key] = future
        self._pending[member.id] = (wire,)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            return make_job_update(
                key[0], key[1], status="rejected", detail="worker did not ack in time"
            )
        finally:
            self._acks.pop(key, None)
            self._pending.pop(member.id, None)

    async def _await_round_result(
        self, job_id: str, step: int, *, timeout: float
    ) -> dict[str, Any] | None:
        runtime = self._runtime[job_id]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        runtime.round_updates[step] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            return None
        finally:
            runtime.round_updates.pop(step, None)

    async def _drive_rounds(self, job_id: str, member: Member) -> None:
        job = self._jobs[job_id]
        manifest_by_path = {entry.relative_path: entry for entry in job.manifest}
        have: set[str] = set()
        step = 1
        stalled = 0
        while have != set(manifest_by_path):
            subset = sorted(set(manifest_by_path) - have)
            step += 1
            wire = command_to_wire(
                TransferRoundCommand(
                    schema_version=PROTOCOL_VERSION,
                    job_id=job_id,
                    step=step,
                    subset=subset,
                    peers=self._round_peers(member),
                    base_port=self._round_base_port(),
                )
            )
            ack = await self._send_and_ack(member, wire)
            if ack.get("status") == "rejected":
                self._finish(job_id, "error", error=str(ack.get("detail") or ""))
                return
            result = await self._await_round_result(
                job_id, step, timeout=ROUND_DEADLINE_S + 60.0
            )
            if result is None or result.get("status") == "round_error":
                stalled += 1
            else:
                new_have = set(result.get("have") or [])
                stalled = 0 if new_have - have else stalled + 1
                have = new_have
            self._jobs[job_id] = replace(
                self._jobs[job_id],
                have=tuple(sorted(have)),
                rounds_completed=step - 1,
                updated_at=time.time(),
            )
            if stalled >= ROUND_CAP:
                self._finish(
                    job_id,
                    "error",
                    error=f"{ROUND_CAP} consecutive failed/no-progress rounds",
                )
                return
        self._finish(job_id, "done")

    async def _drive_hf(self, job_id: str, member: Member) -> None:
        """HF fan-out (D6): a single TRANSFER_START already carried the repo
        id/revision; the worker's owned task does the whole download+verify
        +move and reports the outcome as one transfer update.
        """
        result = await self._await_round_result(job_id, 1, timeout=ROUND_DEADLINE_S * 4)
        if result is None:
            self._finish(job_id, "error", error="worker never reported an outcome")
            return
        have = tuple(sorted(result.get("have") or []))
        self._jobs[job_id] = replace(
            self._jobs[job_id], have=have, updated_at=time.time()
        )
        if result.get("status") == "done":
            self._finish(job_id, "done")
        else:
            self._finish(job_id, "error", error=str(result.get("detail") or ""))

    def _finish(self, job_id: str, status: str, *, error: str = "") -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        self._jobs[job_id] = replace(
            job, status=status, error=error, updated_at=time.time()
        )

    def _round_peers(self, member: Member) -> list[str]:
        head_addr = self._manager.settings.data_plane_address
        return [head_addr, member.endpoint.rsplit(":", 1)[0]]

    def _round_base_port(self) -> int:
        from ..settings import transfer_base_port

        return transfer_base_port(self._manager.settings)


# =============================================================================
# Worker side
# =============================================================================


@dataclass
class _WorkerTransferJob:
    job_id: str
    model_id: str
    manifest: tuple[FileManifestEntry, ...]
    source: str
    epoch: str
    final_dir: Path
    status: str = "running"
    have: set[str] = field(default_factory=set)
    task: asyncio.Task[None] | None = None
    hf_repo_id: str | None = None
    hf_revision: str | None = None


class TransferWorkerExecutor:
    """Worker-side confinement + execution of TRANSFER_* commands (D1b/D2/D6).

    Mirrors ``WorkerCommandExecutor``'s discipline: every command is
    untrusted input, confined against this node's OWN settings, and the
    actual (potentially minutes-long) transfer work always runs as an
    OWNED TASK off the command dispatch path -- the ack this returns is
    always fast.
    """

    def __init__(
        self,
        global_settings: GlobalSettings,
        *,
        session_launcher: Callable[..., Any] | None = None,
        hf_downloader: Callable[..., Any] | None = None,
        python: str | None = None,
    ) -> None:
        self._global_settings = global_settings
        self._session_launcher = session_launcher or launch_transfer_session
        self._hf_downloader = hf_downloader or self._default_hf_download
        self._python = python
        self._jobs: dict[str, _WorkerTransferJob] = {}
        self._updates: list[dict[str, Any]] = []
        self._current_epoch: str | None = None

    def set_epoch(self, epoch: str) -> None:
        self._current_epoch = epoch

    def pending_transfer_updates(self) -> list[dict[str, Any]]:
        updates, self._updates = self._updates, []
        return updates

    def record_local_update(self, update: dict[str, Any]) -> None:
        """Queue an update for the next heartbeat. Used both by owned
        background tasks reporting progress and, from
        ``WorkerCommandExecutor``, to re-emit a replayed ack."""
        self._updates.append(update)

    def _emit(self, job_id: str, step: int, *, status: str, **extra: Any) -> None:
        self.record_local_update(
            make_transfer_update(job_id, step, status=status, **extra)
        )

    async def dispatch(self, command: Any) -> dict[str, Any]:
        if isinstance(command, TransferStartCommand):
            return self._do_start(command)
        if isinstance(command, TransferRoundCommand):
            return self._do_round(command)
        if isinstance(command, TransferAbortCommand):
            return await self._do_abort(command)
        raise ProtocolError(f"unhandled transfer command {command!r}")

    # ---- TRANSFER_START -----------------------------------------------

    def _do_start(self, command: TransferStartCommand) -> dict[str, Any]:
        if (
            self._current_epoch is not None
            and command.epoch
            and command.epoch != self._current_epoch
        ):
            return make_job_update(
                command.job_id,
                command.step,
                status="rejected",
                detail="job epoch does not match this worker's current epoch (CL5-10)",
            )
        try:
            manifest = manifest_mod.validate_received_manifest(command.manifest)
        except ManifestError as exc:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail=str(exc)
            )
        try:
            final_dir = self._resolve_destination(command.model_id)
        except ManifestError as exc:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail=str(exc)
            )

        pool_conflict = self._pool_conflict(command.model_id)
        if pool_conflict is not None:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail=pool_conflict
            )

        if command.source == TransferSource.HF:
            if not command.hf_repo_id or not _COMMIT_SHA_RE.match(
                command.hf_revision or ""
            ):
                return make_job_update(
                    command.job_id,
                    command.step,
                    status="rejected",
                    detail="HF transfer requires hf_repo_id and a 40-hex hf_revision",
                )
            if not self._global_settings.cluster.allow_hf_transfer:
                return make_job_update(
                    command.job_id,
                    command.step,
                    status="rejected",
                    detail="cluster.allow_hf_transfer is disabled on this node",
                )

        job = _WorkerTransferJob(
            job_id=command.job_id,
            model_id=command.model_id,
            manifest=manifest,
            source=command.source.value,
            epoch=command.epoch,
            final_dir=final_dir,
            hf_repo_id=command.hf_repo_id,
            hf_revision=command.hf_revision,
        )
        self._jobs[command.job_id] = job
        if command.source == TransferSource.HF:
            job.task = asyncio.create_task(self._run_hf_download(job, command.step))
        else:
            job.task = asyncio.create_task(
                self._scan_have(job, command.step, repair=command.repair)
            )
        return make_job_update(command.job_id, command.step, status="accepted")

    def _resolve_destination(self, model_id: str) -> Path:
        roots = self._global_settings.get_effective_model_dirs()
        if not roots:
            raise ManifestError("this node has no configured model dirs")
        return resolve_transfer_destination(model_id, roots[0])

    def _pool_conflict(self, model_id: str) -> str | None:
        """CL5-10: refuse a destination that is loaded/loading in this
        node's own pool."""
        from .manager import get_engine_pool

        pool = get_engine_pool()
        if pool is None:
            return None
        entry = pool.get_entry(model_id)
        if entry is not None and (entry.engine is not None or entry.is_loading):
            return f"model {model_id!r} is loaded or loading on this node"
        return None

    async def _scan_have(
        self, job: _WorkerTransferJob, step: int, *, repair: bool
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            have = await loop.run_in_executor(None, self._compute_have, job, repair)
        except OSError as exc:
            job.status = "error"
            self._emit(job.job_id, step, status="round_error", detail=str(exc))
            return
        job.have = set(have)
        full = {entry.relative_path for entry in job.manifest}
        if job.have == full:
            job.status = "done"
        self._emit(job.job_id, step, status="have", have=sorted(job.have))

    def _compute_have(self, job: _WorkerTransferJob, repair: bool) -> list[str]:
        if repair:
            return []
        have: list[str] = []
        for entry in job.manifest:
            path = manifest_mod.resolve_under(entry, job.final_dir)
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                if path.stat().st_size != entry.size:
                    continue
                if _sha256_file(path) != entry.sha256:
                    continue
            except OSError:
                continue
            have.append(entry.relative_path)
        return have

    # ---- TRANSFER_ROUND (peer source) ----------------------------------

    def _do_round(self, command: TransferRoundCommand) -> dict[str, Any]:
        job = self._jobs.get(command.job_id)
        if job is None:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail="unknown job"
            )
        if job.status in ("done", "error", "aborted"):
            return make_job_update(
                command.job_id,
                command.step,
                status="rejected",
                detail=f"job already {job.status}",
            )
        from .hostfile import LinkScopeError

        try:
            peers = self._validate_peers(command.peers)
        except (ManifestError, LinkScopeError) as exc:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail=str(exc)
            )
        by_path = {entry.relative_path: entry for entry in job.manifest}
        subset_entries = [by_path[p] for p in command.subset if p in by_path]
        if len(subset_entries) != len(command.subset):
            return make_job_update(
                command.job_id,
                command.step,
                status="rejected",
                detail="round subset references entries outside the job's manifest",
            )
        job.task = asyncio.create_task(
            self._run_round(job, command.step, subset_entries, peers, command.base_port)
        )
        return make_job_update(command.job_id, command.step, status="accepted")

    def _validate_peers(self, peers: list[str]) -> list[str]:
        from .hostfile import require_link_scope

        cs = self._global_settings.cluster
        if len(peers) != 2:
            raise ManifestError("a transfer session is always exactly 2 peers")
        for address in peers:
            require_link_scope(
                address,
                data_plane_subnet=cs.data_plane_subnet,
                allow_routable_data_plane=cs.allow_routable_data_plane,
                allow_loopback=cs.allow_loopback,
            )
        return list(peers)

    async def _run_round(
        self,
        job: _WorkerTransferJob,
        step: int,
        entries: list[FileManifestEntry],
        peers: list[str],
        base_port: int,
    ) -> None:
        staging_dir = Path(tempfile.mkdtemp(prefix="omlx-transfer-staging-"))
        manifest_path = staging_dir / "round-manifest.json"
        manifest_path.write_text(json.dumps([entry.to_dict() for entry in entries]))

        def argv_builder(_rank: int) -> list[str]:
            return [
                "--role",
                "dst",
                "--manifest",
                str(manifest_path),
                "--root",
                str(staging_dir),
            ]

        cs = self._global_settings.cluster
        cluster = None
        try:
            cluster = self._session_launcher(
                rank=1,
                world_size=2,
                ips=peers,
                base_port=base_port,
                argv_builder=argv_builder,
                data_plane_subnet=cs.data_plane_subnet,
                allow_routable_data_plane=cs.allow_routable_data_plane,
                allow_loopback=cs.allow_loopback,
                python=self._python,
            )
        except TransferSpawnBoundError as exc:
            self._emit(job.job_id, step, status="round_error", detail=str(exc))
            shutil.rmtree(staging_dir, ignore_errors=True)
            return

        try:
            await self._wait_round_deadline(cluster)
        finally:
            cluster.stop()

        newly_have = self._finalize_round(job, entries, staging_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        job.have |= set(newly_have)
        full = {entry.relative_path for entry in job.manifest}
        if job.have == full:
            job.status = "done"
        self._emit(
            job.job_id,
            step,
            status="round_done",
            have=sorted(job.have),
            transferred=newly_have,
        )

    async def _wait_round_deadline(self, cluster: Any) -> None:
        """CL5-16: a wall-clock deadline that kills a wedged session; the
        gate release happens in ``finally`` regardless (`_run_round`'s
        ``cluster.stop()``, `_drive`'s gate release on the head side)."""
        leader = cluster.leader
        if leader is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, leader.process.wait),
                timeout=ROUND_DEADLINE_S,
            )
        except TimeoutError:
            logger.warning("cluster: transfer round exceeded its deadline; killing")
            cluster.kill()

    def _finalize_round(
        self,
        job: _WorkerTransferJob,
        entries: list[FileManifestEntry],
        staging_dir: Path,
    ) -> list[str]:
        """D2/CL5-08: digest-verify each staged file and atomically
        ``os.replace`` verified ones into the final dir. A mismatch deletes
        the staged file, never moves it."""
        newly_have: list[str] = []
        for entry in entries:
            staged = manifest_mod.resolve_under(entry, staging_dir)
            if not staged.is_file():
                continue
            try:
                manifest_mod.assert_realpath_contained(staged, staging_dir)
                if staged.stat().st_size != entry.size or (
                    _sha256_file(staged) != entry.sha256
                ):
                    with contextlib.suppress(OSError):
                        staged.unlink()
                    continue
                final_path = manifest_mod.resolve_under(entry, job.final_dir)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, final_path)
                manifest_mod.assert_realpath_contained(final_path, job.final_dir)
                newly_have.append(entry.relative_path)
            except (ManifestError, OSError) as exc:
                logger.warning(
                    "cluster: round entry %s failed to finalize: %s",
                    entry.relative_path,
                    exc,
                )
                with contextlib.suppress(OSError):
                    staged.unlink()
        return newly_have

    # ---- HF fan-out (D6) -------------------------------------------------

    async def _run_hf_download(self, job: _WorkerTransferJob, step: int) -> None:
        staging_dir = Path(tempfile.mkdtemp(prefix="omlx-transfer-hf-staging-"))
        try:
            required = {entry.relative_path for entry in job.manifest}
            missing = required
            for _attempt in range(2):  # D6: one re-fetch on divergence, then terminal
                try:
                    await self._hf_downloader(
                        job.hf_repo_id,
                        staging_dir,
                        revision=job.hf_revision,
                        ignore_patterns=HF_IGNORE_PATTERNS,
                    )
                except Exception as exc:
                    job.status = "error"
                    self._emit(
                        job.job_id,
                        step,
                        status="error",
                        detail=f"HF download failed: {exc}",
                    )
                    return
                moved, missing = self._finalize_hf(job, staging_dir)
                job.have = set(moved)
                if not missing:
                    break
            if missing:
                job.status = "error"
                self._emit(
                    job.job_id,
                    step,
                    status="error",
                    code="hf_source_incomplete",
                    detail=f"required entries missing or divergent: {sorted(missing)}",
                    have=sorted(job.have),
                )
                return
            job.status = "done"
            self._emit(job.job_id, step, status="done", have=sorted(job.have))
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _finalize_hf(
        self, job: _WorkerTransferJob, staging_dir: Path
    ) -> tuple[list[str], set[str]]:
        moved: list[str] = []
        missing: set[str] = set()
        for entry in job.manifest:
            if entry.relative_path in job.have:
                moved.append(entry.relative_path)
                continue
            staged = manifest_mod.resolve_under(entry, staging_dir)
            if not staged.is_file():
                missing.add(entry.relative_path)
                continue
            try:
                manifest_mod.assert_realpath_contained(staged, staging_dir)
                if staged.stat().st_size != entry.size or (
                    _sha256_file(staged) != entry.sha256
                ):
                    missing.add(entry.relative_path)
                    continue
                final_path = manifest_mod.resolve_under(entry, job.final_dir)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, final_path)
                manifest_mod.assert_realpath_contained(final_path, job.final_dir)
                moved.append(entry.relative_path)
            except (ManifestError, OSError):
                missing.add(entry.relative_path)
        return moved, missing

    async def _default_hf_download(
        self,
        repo_id: str | None,
        target_dir: Path,
        *,
        revision: str | None,
        ignore_patterns: list[str] | None,
    ) -> None:
        from ..admin.hf_downloader import download_model_to_dir

        # "TRANSFER_START never carries hf_token" (D7 low): the worker uses
        # its own settings for it. There is no persisted HF token setting
        # (the admin UI always passes one explicitly, interactively) --
        # the worker's own environment is the non-interactive analog, and
        # HF_TOKEN is already on the rank-spawn env allowlist for the same
        # reason (`hostfile.ENV_ALLOWLIST`).
        await download_model_to_dir(
            repo_id or "",
            target_dir,
            revision=revision or "",
            hf_token=os.environ.get("HF_TOKEN", ""),
            ignore_patterns=ignore_patterns,
        )

    # ---- TRANSFER_ABORT ----------------------------------------------

    async def _do_abort(self, command: TransferAbortCommand) -> dict[str, Any]:
        job = self._jobs.get(command.job_id)
        if job is None:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail="unknown job"
            )
        task = job.task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        job.status = "aborted"
        return make_job_update(command.job_id, command.step, status="aborted")
