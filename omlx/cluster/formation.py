# SPDX-License-Identifier: Apache-2.0
"""Head-side formation orchestration (D8): the E6 job that stands a
tensor-parallel model up across the pair and tears it down.

Every step runs inside one queued operation (E6), so two formation sequences
cannot interleave and the ``commands_for`` snapshot a heartbeat reads is never
torn (CL2-08). Head→worker work is published as an immutable per-member command
snapshot and delivered on the next heartbeat response; the worker's reply comes
back as an authenticated ``job_update`` (CL2-07) that resolves the awaiting step.

The CL2-06 teardown-suppression alarm lives here: once the head tears a
formation down it remembers the job id, and a subsequent authenticated
``job_update`` that still reports that formation as live is raised as an alarm —
the documented, detectable residual of on-path teardown suppression.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .hostfile import ibv_matrix_from_devices, require_link_scope
from .launcher import DEFAULT_JOIN_TIMEOUT_S, LocalCluster, sweep_orphaned_ranks
from .manager import ClusterError
from .placement import PlacementDecision
from .protocol import (
    PROTOCOL_VERSION,
    Backend,
    PresenceCommand,
    SpawnRankCommand,
    SweepCommand,
    TeardownCommand,
    command_to_wire,
)
from .state import Member

if TYPE_CHECKING:
    from .engine import ClusterEngine
    from .manager import ClusterManager

logger = logging.getLogger(__name__)

# Ceiling for rank 0 to finish loading and barrier (implies the worker's rank
# also loaded). The real load is seconds-to-minutes; this is the failure ceiling.
LOAD_TIMEOUT_S = 900.0


@dataclass
class FormationJob:
    """A runtime, observable formation job. Not persisted — a distributed
    formation does not survive a head restart (rank children die with it).
    """

    id: str
    kind: str
    model: str
    created_at: float
    status: str = "running"
    error: str = ""
    # The backend formation ACTUALLY formed on — never the requested setting.
    # Under `auto` this is what survived (jaccl, or ring after a fallback), so a
    # reader sees the real transport without re-running (D9). Empty until formed.
    negotiated_backend: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    _step_counter: int = 0
    # S4 D3: the pool-computed PlacementDecision that drove this job, when
    # one exists (a load job driven through EnginePool.load_cluster_model).
    # Recorded exactly once by the caller that computed it -- this field
    # only stores, never recomputes.
    decision: dict[str, Any] | None = None

    def next_step(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def mark(self, name: str, status: str, detail: str = "") -> None:
        self.steps.append(
            {"name": name, "status": status, "detail": detail, "at": time.time()}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "model": self.model,
            "status": self.status,
            "error": self.error,
            "negotiated_backend": self.negotiated_backend,
            "steps": list(self.steps),
            "created_at": self.created_at,
            "decision": self.decision,
        }


class FormationManager:
    """Drives formation/teardown for one head. One active formation (S2)."""

    def __init__(
        self,
        manager: ClusterManager,
        *,
        spawn_leader_fn: Callable[..., LocalCluster] | None = None,
        engine_factory: Callable[..., ClusterEngine] | None = None,
        model_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        self._manager = manager
        self._spawn_leader_fn = spawn_leader_fn or self._default_spawn_leader
        self._engine_factory = engine_factory or self._default_engine
        self._model_resolver = model_resolver or self._default_resolve_model
        self._pending: dict[str, tuple[dict[str, Any], ...]] = {}
        self._acks: dict[tuple[str, int], asyncio.Future[dict[str, Any]]] = {}
        self._engines: dict[str, ClusterEngine] = {}
        self._local: LocalCluster | None = None
        self._active_model: str | None = None
        self._jobs: list[FormationJob] = []
        self._torn_down_jobs: set[str] = set()
        self._alarms: list[str] = []

    # ---- read side (heartbeat + dashboard) -------------------------------

    def commands_for(self, member_id: str) -> list[dict[str, Any]]:
        """The immutable command snapshot for a member (CL2-08)."""
        return [dict(command) for command in self._pending.get(member_id, ())]

    def active_engine(self, model_id: str) -> ClusterEngine | None:
        return self._engines.get(model_id)

    def alarms(self) -> list[str]:
        return list(self._alarms)

    def snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "active_model": self._active_model,
            "jobs": [job.to_dict() for job in self._jobs[-10:]],
            "alarms": list(self._alarms),
        }
        # Surface the live engine's stats (D9 coordination-tax summary +
        # negotiated backend) so the E4 re-measurement is readable through the
        # operator-tier status endpoint, not only in-process. No credential
        # material; read-only.
        if self._active_model is not None:
            engine = self._engines.get(self._active_model)
            if engine is not None:
                snap["engine_stats"] = engine.get_stats()
        return snap

    # ---- worker job updates (CL2-07 attribution, CL2-06 alarm) -----------

    def record_job_updates(self, member: Member, updates: list[dict[str, Any]]) -> None:
        """Resolve awaiting steps from a member's authenticated updates.

        The updates are attributed to ``member`` (already authenticated by the
        heartbeat route); any member/rank id in the update bodies is ignored
        (CL2-07). An update that still reports a torn-down formation as live is
        the CL2-06 suppression alarm.
        """
        for update in updates:
            if not isinstance(update, dict):
                continue
            job_id = str(update.get("job_id") or "")
            step = update.get("step")
            step = int(step) if isinstance(step, int) else 0
            if job_id in self._torn_down_jobs and update.get("status") != "torn_down":
                self._raise_alarm(
                    f"member {member.id} still reports formation job {job_id} that "
                    f"the head tore down (status={update.get('status')!r})"
                )
            future = self._acks.get((job_id, step))
            if future is not None and not future.done():
                future.set_result(update)

    def _raise_alarm(self, message: str) -> None:
        logger.error("cluster: FORMATION ALARM: %s", message)
        self._alarms.append(message)

    # ---- formation / teardown jobs (E6 queue) ----------------------------

    async def load(
        self, model_id: str, *, decision: PlacementDecision | None = None
    ) -> dict[str, Any]:
        return await self._manager._queue.submit(
            "cluster_load", lambda: self._load(model_id, decision=decision)
        )

    async def unload(self, model_id: str) -> dict[str, Any]:
        return await self._manager._queue.submit(
            "cluster_unload", lambda: self._unload(model_id)
        )

    async def _load(
        self, model_id: str, *, decision: PlacementDecision | None = None
    ) -> dict[str, Any]:
        if self._active_model is not None:
            raise ClusterError(
                409,
                f"a distributed formation is already active for "
                f"{self._active_model!r}; unload it first",
            )
        job = self._new_job("load", model_id)
        # S4 D3: recorded once, here, by whichever caller supplied it --
        # EnginePool.load_cluster_model computes it exactly once per call
        # and this job is where a reader (GET /v1/cluster/models/status)
        # finds it.
        job.decision = decision.to_dict() if decision is not None else None
        # S5 D4: the single-active cluster-operation gate -- refuses (409)
        # while a transfer is in flight, so a formation and a transfer
        # sequence can never interleave.
        self._manager.acquire_operation_gate("formation", job.id)
        try:
            member = self._require_active_member()
            # Step 1: model present on the head's OWN dirs (id-only).
            local_path = self._model_resolver(model_id)
            if local_path is None:
                raise ClusterError(
                    424,
                    f"model {model_id!r} is not present on the head; S5 has no "
                    "auto-download",
                )
            job.mark("head_presence", "present")
            # Step 1b: model present on the worker (check_model round-trip).
            presence = await self._command(
                member, self._presence_cmd(job, model_id), job, "check_model"
            )
            if not presence.get("present"):
                raise ClusterError(
                    424,
                    f"model {model_id!r} is absent on worker {member.id}; S5 has "
                    "no auto-download — nothing was spawned",
                )
            worker_addr = str(presence.get("data_plane_address") or "")
            worker_rdma = str(presence.get("rdma_device") or "")
            ips = self._build_ips(worker_addr)
            # Step 2: sweep orphaned ranks on both nodes.
            sweep_orphaned_ranks()
            await self._command(member, self._sweep_cmd(job), job, "sweep")
            # Steps 3-5: form on the negotiated backend, falling back per D7.
            negotiated = await self._form_with_fallback(
                job, member, model_id, local_path, ips, worker_rdma
            )
            job.negotiated_backend = negotiated
            # Step 6: register the engine so get_engine yields it. A successful
            # _form_with_fallback leaves self._local set to the formed cluster.
            assert self._local is not None
            engine = self._engine_factory(
                model_id=model_id, cluster=self._local, resolved_path=local_path
            )
            await engine.start()
            self._engines[model_id] = engine
            self._active_model = model_id
            job.status = "ready"
            job.mark("register_engine", "ready")
            return {
                "model": model_id,
                "status": "ready",
                "job_id": job.id,
                "negotiated_backend": negotiated,
            }
        except BaseException as exc:  # noqa: BLE001 - recorded, formation aborted
            job.status = "failed"
            job.error = str(exc)
            job.mark("failed", "error", str(exc))
            await self._abort_formation()
            raise
        finally:
            self._manager.release_operation_gate("formation", job.id)

    async def _unload(self, model_id: str) -> dict[str, Any]:
        if self._active_model != model_id:
            raise ClusterError(404, f"no active distributed formation for {model_id!r}")
        job = self._new_job("unload", model_id)
        self._manager.acquire_operation_gate("formation", job.id)
        try:
            # Arm the CL2-06 alarm before the teardown reaches the worker:
            # after this, any worker update still reporting the formation as
            # live is the on-path suppression residual.
            self._torn_down_jobs.add(job.id)
            member = self._active_member_or_none()
            if member is not None:
                try:
                    await self._command(
                        member, self._teardown_cmd(job), job, "teardown"
                    )
                except ClusterError as exc:
                    # Best-effort: the local teardown still runs.
                    job.mark("teardown_command", "error", str(exc))
            engine = self._engines.pop(model_id, None)
            if engine is not None:
                await engine.stop()
            await self._abort_formation()
            job.status = "done"
            job.mark("unloaded", "done")
            return {"model": model_id, "status": "unloaded", "job_id": job.id}
        finally:
            self._manager.release_operation_gate("formation", job.id)

    async def stop(self) -> None:
        for engine in list(self._engines.values()):
            try:
                await engine.stop()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                logger.warning("cluster: engine stop failed", exc_info=True)
        self._engines.clear()
        await self._abort_formation()

    # ---- command round-trip ----------------------------------------------

    async def _command(
        self,
        member: Member,
        wire: dict[str, Any],
        job: FormationJob,
        step_name: str,
    ) -> dict[str, Any]:
        key = (str(wire["job_id"]), int(wire["step"]))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._acks[key] = future
        # Publish the immutable snapshot the heartbeat will read (CL2-08).
        self._pending[member.id] = (wire,)
        job.mark(step_name, "sent")
        try:
            update = await asyncio.wait_for(future, timeout=self._command_timeout)
        except TimeoutError as exc:
            job.mark(step_name, "timeout")
            raise ClusterError(
                504, f"worker {member.id} did not answer {step_name} in time"
            ) from exc
        finally:
            self._acks.pop(key, None)
            # Marking the command delivered is itself a mutation inside this
            # queued operation (CL2-08).
            self._pending.pop(member.id, None)
        status = update.get("status")
        job.mark(step_name, str(status), str(update.get("detail") or ""))
        if status in ("error", "rejected"):
            raise ClusterError(
                424,
                f"worker {member.id} refused {step_name}: {update.get('detail')}",
            )
        return update

    # ---- command builders -------------------------------------------------

    def _presence_cmd(self, job: FormationJob, model_id: str) -> dict[str, Any]:
        return command_to_wire(
            PresenceCommand(
                schema_version=PROTOCOL_VERSION,
                job_id=job.id,
                step=job.next_step(),
                model_id=model_id,
            )
        )

    def _sweep_cmd(self, job: FormationJob) -> dict[str, Any]:
        return command_to_wire(
            SweepCommand(
                schema_version=PROTOCOL_VERSION, job_id=job.id, step=job.next_step()
            )
        )

    def _teardown_cmd(self, job: FormationJob) -> dict[str, Any]:
        return command_to_wire(
            TeardownCommand(
                schema_version=PROTOCOL_VERSION, job_id=job.id, step=job.next_step()
            )
        )

    def _spawn_cmd(
        self,
        job: FormationJob,
        model_id: str,
        ips: list[str],
        backend: str,
        ibv_devices: list[list[str | None]] | None = None,
    ) -> dict[str, Any]:
        return command_to_wire(
            SpawnRankCommand(
                schema_version=PROTOCOL_VERSION,
                job_id=job.id,
                step=job.next_step(),
                rank=1,
                world_size=len(ips),
                backend=Backend(backend),
                model_id=model_id,
                peers=list(ips),
                base_port=int(self._settings.data_plane_base_port),
                ibv_devices=ibv_devices,
            )
        )

    # ---- helpers ----------------------------------------------------------

    @property
    def _settings(self) -> Any:
        return self._manager.settings

    @property
    def _command_timeout(self) -> float:
        interval = float(self._settings.heartbeat_interval_s)
        return max(30.0, interval * 6.0)

    @property
    def command_timeout(self) -> float:
        """Public alias (S4 D4): the bound `EnginePool.request_unload`/the
        cluster unload driver add their own margin to.
        """
        return self._command_timeout

    def _new_job(self, kind: str, model: str) -> FormationJob:
        job = FormationJob(
            id=secrets.token_hex(6), kind=kind, model=model, created_at=time.time()
        )
        self._jobs.append(job)
        return job

    def _active_members(self) -> list[Member]:
        active: list[Member] = []
        for member in self._manager.state.members:
            live = self._manager.liveness(member.id)
            if live is not None and live.status == "active":
                active.append(member)
        return active

    def _require_active_member(self) -> Member:
        """The single active worker, or a ClusterError. Never returns None."""
        active = self._active_members()
        if not active:
            raise ClusterError(
                424, "no active worker to form with (S2 needs one head + one worker)"
            )
        if len(active) > 1:
            raise ClusterError(
                409,
                "S2 forms across exactly one worker; multiple active members present",
            )
        return active[0]

    def _active_member_or_none(self) -> Member | None:
        """The single active worker if there is exactly one, else None (teardown
        tolerates a departed worker).
        """
        active = self._active_members()
        return active[0] if len(active) == 1 else None

    def _build_ips(self, worker_addr: str) -> list[str]:
        cs = self._settings
        head_addr = cs.data_plane_address
        if not cs.data_plane_subnet or not head_addr:
            raise ClusterError(
                400,
                "the head has no cluster.data_plane_subnet/data_plane_address "
                "configured; formation refuses",
            )
        if not worker_addr:
            raise ClusterError(
                424, "the worker did not report a data-plane address; cannot form"
            )
        ips = [head_addr, worker_addr]  # rank 0 = head, rank 1 = worker
        for address in ips:
            # Head-side operator-error catch; the worker re-validates its own.
            require_link_scope(
                address,
                data_plane_subnet=cs.data_plane_subnet,
                allow_routable_data_plane=cs.allow_routable_data_plane,
                allow_loopback=cs.allow_loopback,
            )
        return ips

    def _backend_candidates(self) -> list[str]:
        """The backends to attempt, in order (D7).

        ``ring``/``jaccl`` pin one transport — an explicit request that fails to
        form fails the job with the real error, never a silent substitution.
        ``auto`` tries jaccl first and falls back to ring, so a reader is a
        resolver that tries jaccl then ring.
        """
        backend = self._settings.backend
        if backend == "ring":
            return ["ring"]
        if backend == "jaccl":
            return ["jaccl"]
        if backend == "auto":
            return ["jaccl", "ring"]
        raise ClusterError(400, f"unknown cluster.backend {backend!r}")

    async def _form_with_fallback(
        self,
        job: FormationJob,
        member: Member,
        model_id: str,
        local_path: str,
        ips: list[str],
        worker_rdma: str,
    ) -> str:
        """Attempt each candidate backend; return the one that formed.

        Each attempt spawns FRESH rank processes on both nodes; a failed attempt
        is torn down completely — head local then worker — before the next
        candidate spawns (salvage pitfall 6: one distributed session per process,
        a fresh process per formation attempt, and the worker's CL2-09 slot must
        be free before it will spawn again). The last candidate's failure
        propagates the real error: an explicit ``jaccl`` never silently becomes
        ring, and an ``auto`` that also fails on ring reports that failure.
        """
        candidates = self._backend_candidates()
        for index, backend in enumerate(candidates):
            is_last = index == len(candidates) - 1
            try:
                await self._form_once(
                    job, member, model_id, local_path, ips, worker_rdma, backend
                )
                return backend
            except (
                Exception
            ) as exc:  # noqa: BLE001 - recorded; the real error re-raises
                job.mark(f"form_{backend}", "failed", str(exc))
                # Fresh process per attempt: tear the failed attempt down on both
                # nodes before the next candidate. NOT a torn-down-job mark — this
                # is an intra-job retry, not an operator unload, so it must not
                # arm the CL2-06 suppression alarm.
                await self._abort_formation()
                await self._teardown_worker(member, job)
                if is_last:
                    raise
                logger.warning(
                    "cluster: %s formation failed (%s); falling back to ring (auto)",
                    backend,
                    exc,
                )
        # Unreachable: _backend_candidates never returns an empty list.
        raise ClusterError(500, "no backend candidate formed")

    async def _form_once(
        self,
        job: FormationJob,
        member: Member,
        model_id: str,
        local_path: str,
        ips: list[str],
        worker_rdma: str,
        backend: str,
    ) -> None:
        """One formation attempt on ``backend`` (steps 3-5). Raises on failure."""
        ibv = self._ibv_matrix(backend, worker_rdma)
        # Step 3: spawn rank 0 (head child) and wait until it is listening.
        self._local = await self._run_blocking(
            self._spawn_leader_fn,
            model_path=local_path,
            ips=ips,
            backend=backend,
            base_port=int(self._settings.data_plane_base_port),
            ibv_devices=ibv,
        )
        job.mark(f"spawn_leader_{backend}", "listening")
        # Step 4: command the worker to spawn its rank.
        await self._command(
            member,
            self._spawn_cmd(job, model_id, ips, backend, ibv),
            job,
            "spawn_rank",
        )
        # Step 5: rank 0 reports ready — the barrier implies the worker's rank
        # also loaded its shard.
        await self._run_blocking(self._local.wait_ready, timeout=LOAD_TIMEOUT_S)
        job.mark(f"ranks_ready_{backend}", "ready")

    def _ibv_matrix(
        self, backend: str, worker_rdma: str
    ) -> list[list[str | None]] | None:
        """The full jaccl device matrix from both nodes' typed settings (None for
        ring). Rank 0 is the head, rank 1 the worker.
        """
        if backend != "jaccl":
            return None
        head_rdma = self._settings.rdma_device
        if not head_rdma:
            raise ClusterError(
                400,
                "jaccl requested but the head has no cluster.rdma_device configured",
            )
        if not worker_rdma:
            raise ClusterError(
                424, "jaccl requested but the worker reported no rdma_device"
            )
        return ibv_matrix_from_devices([head_rdma, worker_rdma])

    async def _teardown_worker(self, member: Member, job: FormationJob) -> None:
        """Command the worker to tear its rank down (best-effort, between
        fallback attempts and on failure).

        Catches broadly on purpose: this runs inside a fallback's exception
        handler, so a teardown error must NEVER replace the real formation error
        that is about to be re-raised (the acceptance pins a true jaccl failure).
        """
        try:
            await self._command(member, self._teardown_cmd(job), job, "teardown")
        except (
            Exception
        ) as exc:  # noqa: BLE001 - best-effort; never mask the real error
            job.mark("teardown_command", "error", str(exc))

    async def _abort_formation(self) -> None:
        local, self._local = self._local, None
        self._active_model = None
        if local is not None:
            await self._run_blocking(local.stop)

    async def _run_blocking(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(**kwargs))

    def _default_spawn_leader(
        self,
        *,
        model_path: str,
        ips: list[str],
        backend: str,
        base_port: int,
        ibv_devices: list[list[str | None]] | None = None,
    ) -> LocalCluster:
        cs = self._settings
        cluster = LocalCluster(
            model=model_path,
            world_size=len(ips),
            backend=backend,
            base_port=base_port,
            # The head's own rank 0 is operator-initiated and E6-serialised, so
            # it does not claim the worker's CL2-09 exhaustion slot.
            enforce_spawn_bound=False,
        )
        cluster.start(
            [0],
            ips=ips,
            ibv_devices=ibv_devices,
            data_plane_subnet=cs.data_plane_subnet,
            allow_routable_data_plane=cs.allow_routable_data_plane,
            allow_loopback=cs.allow_loopback,
        )
        cluster.start_deathwatch()
        if not cluster.wait_until_ready(timeout=DEFAULT_JOIN_TIMEOUT_S):
            cluster.stop()
            raise ClusterError(
                500, "rank 0 exited before the world formed; refusing to continue"
            )
        return cluster

    def _default_engine(
        self, *, model_id: str, cluster: LocalCluster, resolved_path: str
    ) -> ClusterEngine:
        from .engine import ClusterEngine

        return ClusterEngine(model_id, cluster=cluster, resolved_path=resolved_path)

    def _default_resolve_model(self, model_id: str) -> str | None:
        from ..model_discovery import discover_models_from_dirs

        dirs = self._manager.global_settings.get_effective_model_dirs()
        discovered = discover_models_from_dirs(dirs)
        entry = discovered.get(model_id)
        if entry is None:
            for candidate in discovered.values():
                if candidate.source_repo_id == model_id:
                    entry = candidate
                    break
        return entry.model_path if entry is not None else None
