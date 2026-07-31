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
from .launcher import (
    DEFAULT_STOP_GRACE_S,
    TransferSpawnBoundError,
    launch_transfer_session,
)
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
# S6 rider: a killed/wedged round's session is stopped with
# `LocalCluster.stop()`'s own default grace before it escalates to SIGKILL
# and the OS actually releases the round's ring port. A respawn sleep
# shorter than that grace races the still-closing predecessor for the same
# port -- the S5 rig lost 3 join-timeout rounds in ~60s to exactly this
# collision. The retry sleep must exceed it -- pinned to the SAME constant
# `LocalCluster.stop()` defaults to (S6 P1c/R3), not a second literal that
# could silently drift from it.
ROUND_RETRY_BACKOFF_S = DEFAULT_STOP_GRACE_S + 2.0
# CL5-04/05: head-side unbounded-growth bounds. `pending_results` buffers a
# RESULT that arrived before `_await_round_result` registered a future for
# its step (S5 P2 completion, see `_JobRuntime`); a flood of updates for
# steps nobody is waiting on (bogus or a stale worker) would otherwise grow
# it forever. Evict-oldest on overflow, same discipline as
# `WorkerCommandExecutor.MAX_TRACKED_JOBS`'s LRU (manager.py).
MAX_PENDING_RESULTS_PER_JOB = 64
# `_jobs`/`_runtime` themselves are the second half of the same gap: a
# long-lived head daemon never drops a finished job's record, so its history
# grows without bound. Mirrors `FormationManager.snapshot`'s last-10
# convention -- only TERMINAL jobs are ever pruned, oldest-CREATED-first
# (`_prune_finished` sorts by `created_at`, not by when a job finished).
MAX_FINISHED_JOBS = 10
# S5 P2/D4: the step number a head-initiated TRANSFER_ABORT always uses.
# `WorkerCommandExecutor.MAX_STEPS_PER_JOB` (manager.py) is 4096; this sits
# right at that ceiling (still accepted -- the worker only rejects a step
# strictly greater) and no round or start command will ever reach it, so it
# never collides with -- and is never treated as a replay of -- a real
# round step.
ABORT_STEP = 4096
# D1b/CL5-16: wall-clock deadline for one round + the daemon-side watchdog
# that kills a session past it, releasing the gate in `finally`. A SECOND,
# tighter watchdog rides the same poll: a round with zero staging-dir growth
# for `MIN_PROGRESS_STRIKES` consecutive polls is declared stalled (not
# merely slow) and killed well before the wall clock -- a wedged-but-silent
# peer would otherwise run the full deadline before anything notices.
ROUND_DEADLINE_S = 1800.0
MIN_PROGRESS_INTERVAL_S = 5.0
MIN_PROGRESS_STRIKES = 6  # 6 * MIN_PROGRESS_INTERVAL_S == 30s of zero progress

# CL5-14, mirrored from the manifest builder's ignore precedent: the HF
# path's DETERMINISTIC ignore list (D3a/CL5-01) -- never the
# `model_info`-conditional one `hf_downloader.start_download` computes for
# the admin UI (which vanishes on a metadata failure).
HF_IGNORE_PATTERNS = ["*.bin", "original/**", "consolidated.*.pth"]

# -- CL5-11: staging exhaustion bounds ----------------------------------------
#
# Every one of these guards the SAME resource -- the worker's local staging
# volume -- from three different angles: not enough room for what's about to
# land (free-space precheck), too much already sitting there across every
# job at once (a hard total cap, independent of how much free space there
# happens to be), and stale directories nobody is coming back to (a crashed
# process, or a head that went silent for good).

# Headroom kept beyond the round/job's own byte cost: catches "wrote 99% of
# a shard, then ENOSPC mid-file" at admission instead of partway through a
# multi-GB write. Not derived from the model -- a generous constant is the
# point, not exact accounting.
STAGING_FREE_SPACE_MARGIN_BYTES = 1 * 1024**3  # 1 GiB
# Ceiling on bytes any staging directory tree may hold at once, across every
# job this worker is running. A spacious disk should not let a single
# misbehaving or greedy head balloon local staging without bound.
MAX_TOTAL_STAGING_BYTES = 200 * 1024**3  # 200 GiB
# A staging dir untouched for longer than this is assumed orphaned -- either
# the worker restarted mid-round (in-memory job state lost, so nothing will
# ever finish or clean it up) or the head that owned it went silent well
# past any live round's own deadline. Comfortably longer than
# ROUND_DEADLINE_S so a slow-but-live round is never swept.
STAGING_ORPHAN_AGE_S = ROUND_DEADLINE_S + 300.0
# The two staging-directory prefixes this worker ever creates (peer rounds,
# HF fan-out) -- the only directories the orphan sweep and the quota scan
# will ever touch.
_STAGING_DIR_PREFIXES = ("omlx-transfer-staging-", "omlx-transfer-hf-staging-")

# CL5-11: minimal worker-side flood guard -- a genuinely new transfer job
# (never one already known, so an idempotent redelivery is exempt) is
# refused when the last ACCEPTED new job landed less than this many seconds
# ago. One node forming/tearing down transfer sessions in a tight loop is
# exactly the exhaustion CL2-09's formation bound already guards against for
# formations; this is the same discipline for transfers.
NEW_JOB_RATE_LIMIT_S = 5.0

# S5 P2d: the ONE legitimate retry the head gives a job whose TRANSFER_START
# lost the race against CL5-11's guard (e.g. an aborted-then-immediately-
# retried load, which genuinely is a new job to the worker -- the aborted
# job never made it into `self._jobs`). Reuses `job_id` but sends this
# DIFFERENT step number rather than step 1 again: replaying the exact same
# `(job_id, step)` would just re-emit the cached rejection (CL2-06) instead
# of re-running `_do_start`'s rate-limit check against the now-later clock.
# Never collides with a round step (those start at 2 and only increase) or
# `ABORT_STEP`.
TRANSFER_START_RETRY_STEP = -1
# Added to the worker-reported `retry_after_s` before the head sleeps, so a
# retry issued right at the edge of the window doesn't lose the race to
# clock/scheduling jitter and get rejected a second time for the same
# reason.
_RATE_LIMIT_RETRY_EPSILON_S = 0.25

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


# -- CL5-11: staging volume guards (free space, total quota, orphan sweep) ---


def _staging_dirs() -> list[Path]:
    """Every one of this worker's own staging directories currently on
    disk, matched by prefix under the system temp dir -- the same location
    `tempfile.mkdtemp` uses for both round and HF-fan-out staging."""
    tmp = Path(tempfile.gettempdir())
    try:
        entries = list(tmp.iterdir())
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if entry.is_dir() and entry.name.startswith(_STAGING_DIR_PREFIXES)
    ]


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                with contextlib.suppress(OSError):
                    total += file_path.stat().st_size
    except OSError:
        return total
    return total


def _total_staging_bytes() -> int:
    """Bytes currently sitting in every one of this worker's staging
    directories, across every job -- read straight off disk rather than
    tracked in memory, so it stays honest across process restarts and
    partial writes."""
    return sum(_dir_bytes(entry) for entry in _staging_dirs())


def _check_staging_capacity(required_bytes: int) -> None:
    """CL5-11: refuse to stage more bytes than the volume can hold, or more
    than this worker's total staging quota allows.

    Two independent guards checked together before staging begins for one
    round or one HF fan-out: free space on the staging volume (with a
    margin, so a full disk fails loudly at admission rather than partway
    through a multi-GB write), and a hard ceiling on the total this worker
    will ever stage across every job at once (a spacious disk should not
    let a single misbehaving or greedy head balloon staging without bound).
    """
    usage = shutil.disk_usage(tempfile.gettempdir())
    needed = required_bytes + STAGING_FREE_SPACE_MARGIN_BYTES
    if usage.free < needed:
        raise ManifestError(
            f"staging volume has {usage.free} bytes free, need {needed} "
            f"({required_bytes} for this transfer + "
            f"{STAGING_FREE_SPACE_MARGIN_BYTES} margin) (CL5-11)"
        )
    in_flight = _total_staging_bytes()
    if in_flight + required_bytes > MAX_TOTAL_STAGING_BYTES:
        raise ManifestError(
            f"staging quota exceeded: {in_flight} bytes already staged + "
            f"{required_bytes} for this transfer > {MAX_TOTAL_STAGING_BYTES} "
            "cap (CL5-11)"
        )


def sweep_orphaned_staging_dirs(*, older_than_s: float | None = None) -> int:
    """Remove staging directories nobody is coming back to (CL5-11).

    Age-based rather than liveness-based -- a staging directory is not a
    process, so there is nothing to poll the way `launcher.sweep_orphaned_
    ranks` polls the process table. A directory whose newest file is older
    than the threshold is assumed to belong to either a worker that
    restarted mid-round (in-memory job state lost, so nothing will ever
    finish or clean it up) or a round whose head went silent well past its
    own deadline. Safe to call at any time, including concurrently with a
    genuinely live round elsewhere on the same machine: the threshold is
    comfortably longer than any live round's own deadline, so a directory
    still being written to is never mistaken for an orphan.
    """
    threshold = STAGING_ORPHAN_AGE_S if older_than_s is None else older_than_s
    now = time.time()
    removed = 0
    for entry in _staging_dirs():
        try:
            newest = max(
                (p.stat().st_mtime for p in entry.rglob("*") if p.is_file()),
                default=entry.stat().st_mtime,
            )
        except OSError:
            continue
        if now - newest < threshold:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
        logger.warning("cluster: swept orphaned transfer staging dir %s", entry)
    return removed


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
    # The worker's own DATA-PLANE address (Thunderbolt-link, distinct from
    # `member.endpoint`'s control-plane address), reported on the
    # TRANSFER_START ack -- mirrors formation's PresenceCommand exchange
    # (`_build_ips`), since a round session must ring-connect over the data
    # plane, never the join/heartbeat address.
    worker_data_plane_address: str = ""
    # The resolved model dir the manifest was built over (snapshot dir for
    # hub-cache sources) -- the head's per-round src session reads entries
    # relative to this root.
    source_root: str = ""
    # S5 P2 completion: a RESULT (have/round_done/round_error/done/error)
    # that arrives before `_await_round_result` has registered a future for
    # its step -- an owned round task on the worker can complete faster
    # than the head's own coroutine resumes past its ack await -- is
    # buffered here instead of being dropped. `_await_round_result` checks
    # this first.
    pending_results: dict[int, dict[str, Any]] = field(default_factory=dict)


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

    # ---- abort (D4/P2: completes the abort mechanism P1 left unwired) -----

    async def abort_transfer(self, job_id: str) -> TransferJob:
        """Cancel an in-flight job: stop the head's own owned task first
        (so nothing else races a fresh command onto the single-occupancy
        member slot -- D1b), THEN publish `TRANSFER_ABORT` so the worker
        cancels its owned task and discards staging. Idempotent: a job
        already terminal is a no-op that just returns its current record.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise TransferError(404, f"unknown transfer job {job_id!r}")
        if job.status in ("done", "error", "aborted"):
            return job
        runtime = self._runtime.get(job_id)
        if runtime is not None and runtime.task is not None and not runtime.task.done():
            runtime.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runtime.task
        # `_drive`'s own CancelledError handler already finished the job and
        # released the gate (above); still tell the worker so its owned
        # task/staging are cleaned up -- a head-side cancel is purely local.
        member = self._manager.state.member(job.member_id)
        if member is not None:
            wire = command_to_wire(
                TransferAbortCommand(
                    schema_version=PROTOCOL_VERSION, job_id=job_id, step=ABORT_STEP
                )
            )
            with contextlib.suppress(Exception):
                await self._send_and_ack(member, wire, timeout=10.0)
        self._finish(job_id, "aborted")
        self._manager.release_operation_gate("transfer", job_id)
        return self._jobs[job_id]

    # ---- worker transfer updates (D1b) -------------------------------------

    # S5 P2 completion (P1 routing defect, fixed here): the ONLY statuses a
    # command's own direct ack ever carries -- everything else is a RESULT
    # (an asynchronous report from the owned task, possibly sharing its
    # step number with the command that started it -- `_scan_have`'s
    # "have" report rides the TRANSFER_START step). The split has to be on
    # STATUS, never on step: a redelivered/stale "accepted" ack for a round
    # step (CL2-06 idempotent redelivery, re-emitted after the head already
    # popped its ack future) must never fall through and be mistaken for
    # that round's real "have"/"round_done" result -- that misroute is what
    # produced a bogus empty `have` and an extra, rejected round in
    # practice.
    _ACK_STATUSES = frozenset({"accepted", "rejected", "aborted"})

    def record_transfer_updates(
        self, member: Member, updates: list[dict[str, Any]]
    ) -> None:
        """Resolve acks and accumulate progress from a member's
        AUTHENTICATED transfer updates. NOT the same as an ack future
        resolving alone -- ``have``/``round_done``/``error`` reports are
        routed to whichever round future is awaiting them (or buffered on
        the runtime if none is registered yet -- a fast owned task can
        report before the head's own coroutine resumes far enough to
        register one), and every update also refreshes the job's
        ``updated_at``.
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
            status = str(update.get("status") or "")
            key = (job_id, step)
            if status in self._ACK_STATUSES:
                ack_future = self._acks.get(key)
                if ack_future is not None and not ack_future.done():
                    ack_future.set_result(update)
                # A stale/redelivered ack with no live future is correctly
                # dropped -- it must NEVER be treated as a round result.
                continue
            runtime = self._runtime.get(job_id)
            if runtime is None:
                continue
            round_future = runtime.round_updates.get(step)
            if round_future is not None and not round_future.done():
                round_future.set_result(update)
            else:
                pending = runtime.pending_results
                is_new_step = step not in pending
                pending[step] = update
                # CL5-04: bound the buffer -- a flood of results for steps
                # nobody is awaiting (bogus or a stale/misbehaving worker)
                # would otherwise grow this without limit. Evict the oldest
                # buffered step (dict insertion order); overwriting an
                # already-buffered step never grows the dict, so it never
                # evicts.
                if is_new_step and len(pending) > MAX_PENDING_RESULTS_PER_JOB:
                    oldest_step = next(iter(pending))
                    if oldest_step != step:
                        del pending[oldest_step]
                        logger.warning(
                            "cluster: transfer job %s pending_results exceeded "
                            "%d; evicted step %d",
                            job_id,
                            MAX_PENDING_RESULTS_PER_JOB,
                            oldest_step,
                        )

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
        self._runtime[job_id] = _JobRuntime(
            member_id=member.id, source_root=str(local_path)
        )
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
            start_ack, start_step = await self._start_with_rate_limit_retry(
                member, job_id, source, hf_repo_id, hf_revision, epoch
            )
            if start_ack.get("status") == "rejected":
                self._finish(job_id, "error", error=str(start_ack.get("detail") or ""))
                return
            self._runtime[job_id].worker_data_plane_address = str(
                start_ack.get("data_plane_address") or ""
            )
            if source == "hf":
                await self._drive_hf(job_id, member, start_step)
            else:
                # D2: the START step's own worker-side have-scan (`_scan_have`)
                # IS the diff authority's starting point -- seed round 1 from
                # it rather than always requesting the full manifest (which
                # would silently defeat resume: a re-issued job's already-
                # digest-verified files would be re-requested and re-written
                # every time). Bounded + falls back to an empty seed on
                # timeout/failure -- that degrades to today's (correct, just
                # wasteful) behavior rather than blocking the job.
                initial_have = await self._await_initial_have(job_id, start_step)
                await self._drive_rounds(job_id, member, initial_have=initial_have)
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

    async def _start_with_rate_limit_retry(
        self,
        member: Member,
        job_id: str,
        source: str,
        hf_repo_id: str | None,
        hf_revision: str | None,
        epoch: str,
    ) -> tuple[dict[str, Any], int]:
        """S5 P2d: retry a CL5-11 rate-limited TRANSFER_START EXACTLY ONCE.

        The worker's flood guard stays exactly as strict as it is (no
        exemption, no constant change) -- this is the head-side legitimate
        path for the one shape that guard produces against a genuinely new
        job that simply lost a race against another recent new job (e.g. an
        aborted-then-immediately-retried load: the aborted job never made
        it into the worker's `_jobs`, so the retry is genuinely new too).

        A rejection is retried only if it additively carries
        ``retry_after_s`` (the worker's own remaining-window hint) -- any
        other rejection reason (epoch mismatch, bad manifest, pool
        conflict, ...) is terminal on the first try, fail-closed, exactly
        as before. The wait is bounded to never exceed the guard's own
        window (+1s slack): ``retry_after_s`` is itself already bounded by
        ``NEW_JOB_RATE_LIMIT_S``, so this is a belt-and-braces cap against a
        malformed/adversarial value, not a real-world ceiling. The retry
        re-sends on ``TRANSFER_START_RETRY_STEP`` -- never step 1 again --
        so it is a fresh ``(job_id, step)`` pair rather than a replay of the
        rejected attempt (CL2-06 would otherwise just re-emit the cached
        rejection instead of re-running the worker's check against the
        now-later clock). A second rejection -- for any reason, including
        the same guard firing again -- is terminal.

        Returns the winning ack alongside the step it was sent on, so the
        caller awaits the matching "have"/HF-outcome report on that step.
        """
        ack = await self._start_command(
            member, job_id, source, hf_repo_id, hf_revision, epoch, step=1
        )
        if ack.get("status") != "rejected":
            return ack, 1
        retry_after_s = ack.get("retry_after_s")
        if not isinstance(retry_after_s, (int, float)) or retry_after_s <= 0:
            return ack, 1
        wait_s = min(
            float(retry_after_s) + _RATE_LIMIT_RETRY_EPSILON_S,
            NEW_JOB_RATE_LIMIT_S + 1.0,
        )
        await asyncio.sleep(wait_s)
        retry_ack = await self._start_command(
            member,
            job_id,
            source,
            hf_repo_id,
            hf_revision,
            epoch,
            step=TRANSFER_START_RETRY_STEP,
        )
        return retry_ack, TRANSFER_START_RETRY_STEP

    async def _start_command(
        self,
        member: Member,
        job_id: str,
        source: str,
        hf_repo_id: str | None,
        hf_revision: str | None,
        epoch: str,
        *,
        step: int = 1,
    ) -> dict[str, Any]:
        job = self._jobs[job_id]
        wire = command_to_wire(
            TransferStartCommand(
                schema_version=PROTOCOL_VERSION,
                job_id=job_id,
                step=step,
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
        # A fast owned task can report before this coroutine even gets here
        # (e.g. a round that completes between the ack resolving and this
        # call) -- `record_transfer_updates` buffers that case rather than
        # dropping it; check for it before waiting on a fresh future.
        buffered = runtime.pending_results.pop(step, None)
        if buffered is not None:
            return buffered
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        runtime.round_updates[step] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            return None
        finally:
            runtime.round_updates.pop(step, None)

    async def _await_initial_have(self, job_id: str, start_step: int) -> set[str]:
        """D2: await the START step's own "have" report (`_scan_have`) --
        the worker's fresh digest-verified scan of its final dir, and the
        diff authority's actual starting point. ``start_step`` is whichever
        step the winning TRANSFER_START actually landed on (1, or
        ``TRANSFER_START_RETRY_STEP`` after a P2d rate-limit retry) -- the
        worker's have-scan reports on that SAME step (`_scan_have` is
        started with `command.step`). Falls back to an empty seed (today's
        pre-fix behavior: correct, just wasteful -- round 1 requests the
        full manifest) on timeout or any non-"have" result, never blocking
        or failing the job over this.
        """
        result = await self._await_round_result(job_id, start_step, timeout=300.0)
        if result is None or result.get("status") != "have":
            return set()
        return set(result.get("have") or [])

    async def _drive_rounds(
        self, job_id: str, member: Member, *, initial_have: set[str] | None = None
    ) -> None:
        job = self._jobs[job_id]
        manifest_by_path = {entry.relative_path: entry for entry in job.manifest}
        peers = self._round_peers(job_id)
        if peers is None:
            self._finish(
                job_id,
                "error",
                error="worker never reported a data-plane address; cannot form "
                "a round session",
            )
            return
        have: set[str] = set(initial_have or ()) & set(manifest_by_path)
        if have:
            self._jobs[job_id] = replace(
                self._jobs[job_id], have=tuple(sorted(have)), updated_at=time.time()
            )
        step = 1
        stalled = 0
        while have != set(manifest_by_path):
            subset = sorted(set(manifest_by_path) - have)
            step += 1
            # The head is rank 0 (`--role src`) of the round's 2-rank ring
            # session; the worker spawns rank 1 (`--role dst`) on receiving
            # TRANSFER_ROUND. Spawn our side FIRST so rank 0 is joinable the
            # moment the worker's rank comes up.
            try:
                src_session, src_tmp = self._launch_src_session(
                    job_id, [manifest_by_path[p] for p in subset], peers
                )
            except TransferSpawnBoundError:
                stalled += 1
                self._jobs[job_id] = replace(
                    self._jobs[job_id],
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
                await asyncio.sleep(ROUND_RETRY_BACKOFF_S)
                continue
            try:
                wire = command_to_wire(
                    TransferRoundCommand(
                        schema_version=PROTOCOL_VERSION,
                        job_id=job_id,
                        step=step,
                        subset=subset,
                        peers=peers,
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
            finally:
                src_session.stop()
                shutil.rmtree(src_tmp, ignore_errors=True)
            if result is None or result.get("status") == "round_error":
                failed_round = True
                stalled += 1
            else:
                failed_round = False
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
            if failed_round:
                # S6 rider: the round's (possibly just-killed) session needs
                # the same stop-grace margin before a respawn, or the retry
                # races the same port collision the SpawnBoundError branch
                # above guards against.
                await asyncio.sleep(ROUND_RETRY_BACKOFF_S)
        self._finish(job_id, "done")

    async def _drive_hf(self, job_id: str, member: Member, start_step: int) -> None:
        """HF fan-out (D6): a single TRANSFER_START already carried the repo
        id/revision; the worker's owned task does the whole download+verify
        +move and reports the outcome as one transfer update on
        ``start_step`` -- whichever step the winning TRANSFER_START actually
        landed on (see `_await_initial_have`'s same note).
        """
        result = await self._await_round_result(
            job_id, start_step, timeout=ROUND_DEADLINE_S * 4
        )
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
        self._prune_finished()

    def _prune_finished(self) -> None:
        """CL5-04/05: cap retained job records so ``_jobs``/``_runtime``
        don't grow without bound across a long-lived daemon's history --
        mirrors ``FormationManager``'s last-10 convention. Only TERMINAL
        jobs (``done``/``error``/``aborted``) are ever pruned; a job still
        running is never touched no matter how old."""
        terminal = [
            job
            for job in self._jobs.values()
            if job.status in ("done", "error", "aborted")
        ]
        if len(terminal) <= MAX_FINISHED_JOBS:
            return
        terminal.sort(key=lambda job: job.created_at)
        for job in terminal[: len(terminal) - MAX_FINISHED_JOBS]:
            self._jobs.pop(job.id, None)
            self._runtime.pop(job.id, None)

    def _round_peers(self, job_id: str) -> list[str] | None:
        """``[head_data_plane_address, worker_data_plane_address]`` -- NEVER
        ``member.endpoint`` (the control-plane join/heartbeat address, a
        different network entirely from the Thunderbolt data-plane link a
        round's ring session must form over). Mirrors formation's
        ``_build_ips``, which sources the same address from a
        ``PresenceCommand`` reply rather than membership state.
        """
        worker_addr = self._runtime[job_id].worker_data_plane_address
        if not worker_addr:
            return None
        head_addr = self._manager.settings.data_plane_address
        return [head_addr, worker_addr]

    def _round_base_port(self) -> int:
        from ..settings import transfer_base_port

        return transfer_base_port(self._manager.settings)

    def _launch_src_session(
        self,
        job_id: str,
        entries: list[FileManifestEntry],
        peers: list[str],
    ) -> tuple[Any, Path]:
        """Spawn the head's rank-0 ``--role src`` session for one round.

        Returns ``(session, tmp_dir)``; the caller stops the session and
        removes ``tmp_dir`` (holding the round's subset manifest) when the
        round ends, success or not.
        """
        source_root = self._runtime[job_id].source_root
        tmp = Path(tempfile.mkdtemp(prefix="omlx-transfer-src-"))
        manifest_path = tmp / "round-manifest.json"
        manifest_path.write_text(json.dumps([e.to_dict() for e in entries]))

        def argv_builder(_rank: int) -> list[str]:
            return [
                "--role",
                "src",
                "--manifest",
                str(manifest_path),
                "--root",
                source_root,
            ]

        cs = self._manager.settings
        try:
            session = self._session_launcher(
                rank=0,
                world_size=2,
                ips=peers,
                base_port=self._round_base_port(),
                argv_builder=argv_builder,
                data_plane_subnet=cs.data_plane_subnet,
                allow_routable_data_plane=cs.allow_routable_data_plane,
                allow_loopback=cs.allow_loopback,
                python=self._python,
            )
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return session, tmp


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
        # CL5-11: last time a genuinely NEW job was accepted -- the flood
        # guard `_do_start` checks new job_ids against.
        self._last_new_job_at: float = 0.0
        # CL5-11: a fresh executor is this worker's "just (re)started" point
        # -- sweep whatever staging directories survived a prior crash. Age-
        # gated (see `sweep_orphaned_staging_dirs`), so this is safe even
        # though tests construct many executors sharing the same temp dir.
        sweep_orphaned_staging_dirs()

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
        is_new_job = command.job_id not in self._jobs
        if is_new_job:
            elapsed = time.time() - self._last_new_job_at
            if elapsed < NEW_JOB_RATE_LIMIT_S:
                # S5 P2d: `retry_after_s` is additive wire surface (plain
                # dict, no `extra="forbid"` model on a job update -- see
                # `make_job_update`) -- a machine-readable hint for the
                # head's ONE-retry path (`TransferManager.
                # _start_with_rate_limit_retry`). The guard itself is
                # UNCHANGED: this only tells a legitimate caller how long
                # to wait, it never widens or skips the window.
                return make_job_update(
                    command.job_id,
                    command.step,
                    status="rejected",
                    retry_after_s=max(NEW_JOB_RATE_LIMIT_S - elapsed, 0.0),
                    detail=(
                        f"a new transfer job was accepted {elapsed:.1f}s ago; "
                        f"rate limited to one new job per {NEW_JOB_RATE_LIMIT_S:.0f}s "
                        "(CL5-11)"
                    ),
                )
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
            # CL5-11: the HF fan-out stages the whole manifest in one shot
            # (no round-by-round subsetting), so the capacity check runs
            # against the full manifest total here rather than at round
            # start the way the peer path's `_do_round` does.
            try:
                _check_staging_capacity(sum(entry.size for entry in manifest))
            except ManifestError as exc:
                return make_job_update(
                    command.job_id, command.step, status="rejected", detail=str(exc)
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
        if is_new_job:
            self._last_new_job_at = time.time()
        if command.source == TransferSource.HF:
            job.task = asyncio.create_task(self._run_hf_download(job, command.step))
        else:
            job.task = asyncio.create_task(
                self._scan_have(job, command.step, repair=command.repair)
            )
        return make_job_update(
            command.job_id,
            command.step,
            status="accepted",
            data_plane_address=self._reportable_address(),
        )

    def _reportable_address(self) -> str:
        """This node's own data-plane address (D1/R3c), reported on the
        TRANSFER_START ack so the head knows where to ring-connect for a
        round -- the SAME field a ``PresenceCommand`` reply carries for
        formation, never `member.endpoint`'s control-plane address.
        """
        return self._global_settings.cluster.data_plane_address

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
        try:
            _check_staging_capacity(sum(entry.size for entry in subset_entries))
        except ManifestError as exc:
            return make_job_update(
                command.job_id, command.step, status="rejected", detail=str(exc)
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
            await self._wait_round_deadline(cluster, staging_dir)
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

    async def _wait_round_deadline(self, cluster: Any, staging_dir: Path) -> None:
        """CL5-16: two watchdogs guard one round, either one killing the
        session; the gate release happens in ``finally`` regardless
        (`_run_round`'s ``cluster.stop()``, `_drive`'s gate release on the
        head side). Thin wrapper over `_wait_with_progress_watchdog` --
        the SAME shared machinery the HF path's `_run_hf_download` reuses
        for its own (wall-clock-free) minimum-progress watchdog.
        """
        leader = cluster.leader
        if leader is None:
            return
        loop = asyncio.get_running_loop()
        wait_task = loop.run_in_executor(None, leader.process.wait)
        deadline = loop.time() + ROUND_DEADLINE_S
        await self._wait_with_progress_watchdog(
            wait_task, staging_dir, deadline=deadline, kill=cluster.kill
        )

    async def _wait_with_progress_watchdog(
        self,
        wait_task: asyncio.Future[Any],
        staging_dir: Path,
        *,
        deadline: float | None,
        kill: Callable[[], Any],
    ) -> bool:
        """CL5-16: shared minimum-progress watchdog behind both the round
        path (`_wait_round_deadline`, wall clock + min-progress) and the HF
        path (`_run_hf_download`, min-progress only -- ``deadline=None``,
        since download duration scales with model size and a wall-clock
        ceiling would be the wrong bound there).

        Polls ``staging_dir``'s total bytes every ``MIN_PROGRESS_INTERVAL_S``;
        ``MIN_PROGRESS_STRIKES`` consecutive polls with zero growth calls
        ``kill()`` and returns ``True``. If ``deadline`` is not None, also
        calls ``kill()`` and returns ``True`` once the wall clock passes it
        (the round path's ``ROUND_DEADLINE_S``). Returns ``False`` if
        ``wait_task`` completes on its own before either watchdog fires --
        callers use this to distinguish "killed by the watchdog" from "the
        underlying work finished/raised normally" without relying on
        ``asyncio.CancelledError`` (which a WATCHDOG-cancelled
        ``wait_task`` and an EXTERNALLY-cancelled caller task would
        otherwise both raise, indistinguishably).
        """
        loop = asyncio.get_running_loop()
        last_bytes = -1
        stalled_polls = 0
        try:
            while True:
                if deadline is not None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        logger.warning(
                            "cluster: transfer round exceeded its deadline; killing"
                        )
                        kill()
                        return True
                    poll_timeout = min(MIN_PROGRESS_INTERVAL_S, remaining)
                else:
                    poll_timeout = MIN_PROGRESS_INTERVAL_S
                done, _pending = await asyncio.wait({wait_task}, timeout=poll_timeout)
                if wait_task in done:
                    return False
                current_bytes = _dir_bytes(staging_dir)
                if current_bytes > last_bytes:
                    last_bytes = current_bytes
                    stalled_polls = 0
                    continue
                stalled_polls += 1
                if stalled_polls >= MIN_PROGRESS_STRIKES:
                    logger.warning(
                        "cluster: transfer made no progress for %.0fs; killing",
                        MIN_PROGRESS_INTERVAL_S * MIN_PROGRESS_STRIKES,
                    )
                    kill()
                    return True
        finally:
            wait_task.cancel()

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
                # CL5-08: validate BEFORE touching the destination -- a
                # symlinked ancestor already in place (planted before this
                # round even started) must be refused before `mkdir` walks
                # through it and before `os.replace` writes through it, not
                # merely detected after the fact. The leaf write itself
                # stays `os.replace` (never an open()-based copy), which
                # does not follow a symlink AT the final component even if
                # one is raced in later; the second check below still
                # guards that narrower TOCTOU window.
                manifest_mod.assert_realpath_contained(final_path, job.final_dir)
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
                download_task = asyncio.create_task(
                    self._hf_downloader(
                        job.hf_repo_id,
                        staging_dir,
                        revision=job.hf_revision,
                        ignore_patterns=HF_IGNORE_PATTERNS,
                    )
                )
                # CL5-16: the SAME minimum-progress watchdog the round path
                # uses -- no wall-clock deadline here (download duration
                # scales with model size, unlike a bounded round), but a
                # download that stages zero new bytes for
                # MIN_PROGRESS_STRIKES consecutive polls is cancelled
                # rather than left to wedge the head's D4 single-active
                # gate (`_await_round_result`'s own 2-hour backstop in
                # `_drive_hf` would otherwise be the only thing catching
                # this, far later than a live watchdog would).
                killed = await self._wait_with_progress_watchdog(
                    download_task,
                    staging_dir,
                    deadline=None,
                    kill=download_task.cancel,
                )
                if killed:
                    with contextlib.suppress(asyncio.CancelledError):
                        await download_task
                    job.status = "error"
                    self._emit(
                        job.job_id,
                        step,
                        status="error",
                        detail=(
                            "HF download made no staging progress; "
                            "cancelled by the min-progress watchdog (CL5-16)"
                        ),
                    )
                    return
                try:
                    await download_task
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
                # CL5-08: same pre-replace validation as `_finalize_round` --
                # see its comment.
                manifest_mod.assert_realpath_contained(final_path, job.final_dir)
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
