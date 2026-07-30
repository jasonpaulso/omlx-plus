# SPDX-License-Identifier: Apache-2.0
"""The cluster runtime: membership, credentials, liveness and jobs.

The head is the single serialized owner of every cluster mutation (E6):
join, leave, remove, token operations and scrub expiry all run as commands
on one queue, and each command persists ``cluster.json`` after applying,
so two concurrent formation sequences are impossible by construction.

Liveness is deliberately not part of that: heartbeats mutate an in-memory
map only. A head restart therefore starts with no liveness at all and
reports persisted members as ``lost`` until their next heartbeat arrives,
which self-heals within one interval whether or not the worker restarted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import socket
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .formation import FormationManager
    from .transfer import TransferManager, TransferWorkerExecutor

from ..admin.auth import fingerprint_key
from ..model_discovery import discover_models
from ..settings import ClusterSettings, GlobalSettings
from ..utils.psutil_compat import get_total_memory
from .client import ClusterClient, ClusterClientError, validate_head_url
from .credentials import (
    cluster_state_path,
    digest_secret,
    generate_secret,
    load_state,
    mint_bootstrap_token,
    save_state,
    sign_command_response,
)
from .heartbeat import HeartbeatSender
from .hostfile import LinkScopeError, require_link_scope
from .launcher import SpawnBoundError
from .protocol import (
    Backend,
    PresenceCommand,
    ProtocolError,
    SpawnRankCommand,
    SweepCommand,
    TeardownCommand,
    TransferAbortCommand,
    TransferRoundCommand,
    TransferStartCommand,
    make_job_update,
    parse_command,
)
from .queue import ClusterCommandQueue
from .state import (
    ClusterState,
    Member,
    MemberLiveness,
    MemberNodeState,
    WorkerIdentity,
    parse_member_address,
)
from .versions import VersionInfo, collect_versions, compare_versions

logger = logging.getLogger(__name__)

JOIN_PATH = "/v1/cluster/join"
LEAVE_PATH = "/v1/cluster/leave"
MEMBER_ID_BYTES = 8
MAX_MEMBER_NAME_LENGTH = 64
# CL2-10: a ceiling on the head-supplied heartbeat interval so a hostile join
# reply cannot set it absurdly high (silencing the worker) or be non-numeric.
MAX_HEARTBEAT_INTERVAL_S = 3600.0
# CL2-10: an upper bound on a head-commanded world size — each rank loads a
# multi-GB shard, so an unbounded world is machine-level exhaustion.
MAX_WORLD_SIZE = 64

_cluster_manager: ClusterManager | None = None


def get_cluster_manager() -> ClusterManager | None:
    """Return the active cluster manager, or None when the role is off."""
    return _cluster_manager


def set_cluster_manager(manager: ClusterManager | None) -> None:
    """Install (or clear) the process-wide cluster manager."""
    global _cluster_manager
    _cluster_manager = manager


# S4 D4/D2b: nothing in `omlx/cluster/` reaches into `_server_state` — the
# placement preview endpoint (D3) and the worker's own node_state (D1) both
# need the local EnginePool, so `server.init_server()` injects a getter here,
# mirroring the admin `configure(pool_getter=...)` precedent
# (admin/routes.py:1106,1133-1135).
_get_engine_pool: Callable[[], Any] | None = None


def get_engine_pool() -> Any | None:
    """Return this process's EnginePool, or None before init_server() runs."""
    getter = _get_engine_pool
    if getter is None:
        return None
    return getter()


def set_engine_pool_getter(getter: Callable[[], Any] | None) -> None:
    """Install (or clear) the injected EnginePool accessor."""
    global _get_engine_pool
    _get_engine_pool = getter


class ClusterError(Exception):
    """A cluster operation failed with a specific HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ClusterFormationError(Exception):
    """A worker refused to act on a head command (CL2-03/09/10/12)."""


class ModelNotPresentError(Exception):
    """A command named a model absent from this node's own model dirs (CL2-02)."""


@dataclass
class _PreparedSpawn:
    """The validated, locally-derived inputs for one rank spawn.

    Nothing here is trusted verbatim from the head: ``model_path`` is resolved
    against this node's own model dirs (CL2-02), ``ips`` has the worker's own
    address in its own rank slot and every entry re-validated against this
    node's own subnet (CL2-03), ``base_port`` is this node's own setting.
    ``ibv_devices`` (jaccl only, else None) has this rank's own row computed
    from this node's OWN ``rdma_device`` — never a head-supplied one (CL2-03).
    """

    rank: int
    world_size: int
    backend: str
    model_path: str
    ips: list[str]
    base_port: int
    seed: int
    ibv_devices: list[list[str | None]] | None = None


class WorkerCommandExecutor:
    """Applies head->worker heartbeat commands under full CL2 confinement.

    Every command is untrusted input: a compromised or impersonated head must
    never become code execution on the worker (the CL2 review's ordering
    framing — worker-side confinement is the control, not head authentication).
    The executor therefore, for every command:

    * fails closed on an unknown kind/field or off-version schema (CL2-04),
    * resolves a model IDENTIFIER against its OWN model dirs — never a path
      (CL2-02),
    * re-validates every hostfile entry against its OWN data-plane settings and
      computes its own rank's entry from its own address (CL2-03),
    * builds the rank env locally from an allowlist — no env crosses the wire
      (CL2-01, enforced in the launcher),
    * refuses to form with no own data-plane config (CL2-12),
    * bounds itself to one live formation (CL2-09),
    * treats a re-delivered ``(job_id, step)`` as a no-op ack (CL2-06).

    Commands are applied off the heartbeat path on an internal queue: formation
    is minutes-scale, so a rank spawn must not stall the liveness loop.
    """

    # S5 CL5-05: LRU bound on how many distinct job_ids' (job_id, step) replay
    # keys this worker keeps -- unbounded across a worker's lifetime, this
    # would grow forever (many more steps per transfer job than any
    # formation job ever had). Also doubles as CL5-11's round-cap backstop:
    # a step beyond the per-job ceiling is refused outright.
    MAX_TRACKED_JOBS = 64
    MAX_STEPS_PER_JOB = 4096

    def __init__(
        self,
        global_settings: GlobalSettings,
        *,
        spawn_fn: Callable[[_PreparedSpawn], Any] | None = None,
        model_resolver: Callable[[str], tuple[str, int] | None] | None = None,
        local_addresses: set[str] | None = None,
        transfer_executor: TransferWorkerExecutor | None = None,
    ) -> None:
        self._global_settings = global_settings
        self._spawn_fn = spawn_fn
        self._model_resolver = model_resolver or self._default_resolve_model
        self._local_addresses = local_addresses
        self._applied: dict[tuple[str, int], dict[str, Any]] = {}
        self._seen: set[tuple[str, int]] = set()
        self._job_order: OrderedDict[str, None] = OrderedDict()  # CL5-05 LRU
        self._updates: list[dict[str, Any]] = []
        self._cluster: Any = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        if transfer_executor is None:
            from .transfer import TransferWorkerExecutor

            transfer_executor = TransferWorkerExecutor(global_settings)
        self._transfer = transfer_executor

    @property
    def settings(self) -> ClusterSettings:
        return self._global_settings.cluster

    @property
    def cluster(self) -> Any:
        """The active local formation, or None."""
        return self._cluster

    @property
    def transfer(self) -> Any:
        """The worker-side transfer executor (S5 D1b)."""
        return self._transfer

    def set_epoch(self, epoch: str) -> None:
        """Wired from the heartbeat's own epoch (CL5-10): a transfer job is
        bound to the epoch that was live when it started, and refuses a
        command echoing a stale one.
        """
        self._transfer.set_epoch(epoch)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._teardown_local()

    # ---- heartbeat integration -------------------------------------------

    def deliver(self, commands: list[Any]) -> None:
        """Enqueue verified commands, deduped by ``(job_id, step)`` (CL2-06).

        Called from the heartbeat loop after the response signature has been
        checked, so it must stay fast: it only dedups and enqueues, never
        spawns. A re-delivered command that already produced an ack re-emits
        that ack and never spawns again. A step beyond ``MAX_STEPS_PER_JOB``
        for its job_id is refused outright, never enqueued (CL5-05/CL5-11).
        """
        for raw in commands:
            if not isinstance(raw, dict):
                continue
            key = self._replay_key(raw)
            if key is not None and key in self._seen:
                self._job_order.move_to_end(key[0])
                prior = self._applied.get(key)
                if prior is not None:
                    self._append_update(prior, self._channel_for_raw(raw))
                continue
            if key is not None:
                job_id, step = key
                if step > self.MAX_STEPS_PER_JOB:
                    logger.warning(
                        "cluster: rejecting command for job %s: step %d exceeds "
                        "the %d ceiling (CL5-05/CL5-11)",
                        job_id,
                        step,
                        self.MAX_STEPS_PER_JOB,
                    )
                    continue
                self._seen.add(key)
                self._track_job(job_id)
            self._queue.put_nowait(raw)

    def _track_job(self, job_id: str) -> None:
        """CL5-05: LRU-evict the oldest job_id's replay keys once more than
        ``MAX_TRACKED_JOBS`` distinct job_ids have been seen."""
        if job_id in self._job_order:
            self._job_order.move_to_end(job_id)
            return
        self._job_order[job_id] = None
        if len(self._job_order) > self.MAX_TRACKED_JOBS:
            oldest, _ = self._job_order.popitem(last=False)
            for key in [k for k in self._seen if k[0] == oldest]:
                self._seen.discard(key)
                self._applied.pop(key, None)

    def pending_job_updates(self) -> list[dict[str, Any]]:
        """Drain the job updates accumulated since the last heartbeat."""
        updates, self._updates = self._updates, []
        return updates

    def pending_transfer_updates(self) -> list[dict[str, Any]]:
        """Drain the transfer updates accumulated since the last heartbeat
        (S5 D1b) -- a SIBLING channel to job updates, not the same list."""
        return self._transfer.pending_transfer_updates()

    def _channel_for_raw(self, raw: dict[str, Any]) -> str:
        kind = raw.get("kind")
        if kind in ("transfer_start", "transfer_round", "transfer_abort"):
            return "transfer"
        return "job"

    def _append_update(self, update: dict[str, Any], channel: str) -> None:
        if channel == "transfer":
            self._transfer.record_local_update(update)
        else:
            self._updates.append(update)

    # ---- command application (off the heartbeat path) --------------------

    async def _run(self) -> None:
        while True:
            raw = await self._queue.get()
            channel = self._channel_for_raw(raw)
            try:
                update = await self._apply(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported, never crash the loop
                logger.warning("cluster: worker command failed: %s", exc)
                update = make_job_update(
                    str(raw.get("job_id") or ""),
                    self._safe_step(raw),
                    status="error",
                    detail=str(exc),
                )
            key = self._replay_key(raw)
            if key is not None:
                self._applied[key] = update
            self._append_update(update, channel)

    async def _apply(self, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            command = parse_command(raw)  # CL2-04: fail closed on anything unexpected
        except ProtocolError as exc:
            logger.warning(
                "cluster: rejected head command (fp=%s): %s",
                fingerprint_key(json.dumps(raw, sort_keys=True, default=str)),
                exc,
            )
            return make_job_update(
                str(raw.get("job_id") or ""),
                self._safe_step(raw),
                status="rejected",
                detail=str(exc),
            )
        try:
            return await self._dispatch(command)
        except (
            ClusterFormationError,
            ModelNotPresentError,
            SpawnBoundError,
            LinkScopeError,
        ) as exc:
            # A confinement refusal travels back as a job_update — never
            # swallowed — and is logged (CL2-04 note).
            logger.warning("cluster: refused %s command: %s", command.kind.value, exc)
            return make_job_update(
                command.job_id, command.step, status="error", detail=str(exc)
            )

    async def _dispatch(self, command: Any) -> dict[str, Any]:
        if isinstance(command, PresenceCommand):
            return self._do_presence(command)
        if isinstance(command, SweepCommand):
            from .launcher import sweep_orphaned_ranks

            killed = sweep_orphaned_ranks()
            return make_job_update(
                command.job_id, command.step, status="swept", killed=killed
            )
        if isinstance(command, TeardownCommand):
            await self._teardown_local()
            return make_job_update(command.job_id, command.step, status="torn_down")
        if isinstance(
            command, (TransferStartCommand, TransferRoundCommand, TransferAbortCommand)
        ):
            return await self._transfer.dispatch(command)
        # The command union is closed; anything else is a schema/parse invariant
        # violation. Fail closed rather than assert (which -O compiles out on the
        # spawn dispatch path).
        if not isinstance(command, SpawnRankCommand):
            raise ProtocolError(f"unhandled command kind {command!r}")
        return await self._do_spawn(command)

    def _do_presence(self, command: PresenceCommand) -> dict[str, Any]:
        resolved = self._model_resolver(command.model_id)
        present = resolved is not None
        return make_job_update(
            command.job_id,
            command.step,
            status="present" if present else "absent",
            model_id=command.model_id,
            present=present,
            resolved_size=(resolved[1] if resolved is not None else 0),
            data_plane_address=self._reportable_address(),
            # The worker's own RDMA device, so the head can build a complete
            # jaccl matrix (mirrors data_plane_address). The worker still
            # recomputes its own matrix row from this same setting (CL2-03).
            rdma_device=self.settings.rdma_device,
        )

    async def _do_spawn(self, command: SpawnRankCommand) -> dict[str, Any]:
        prepared = self._prepare_spawn(command)  # raises on any CL2 violation
        spawn = self._spawn_fn or self._default_spawn
        loop = asyncio.get_running_loop()
        cluster = await loop.run_in_executor(None, spawn, prepared)
        self._cluster = cluster
        return make_job_update(
            command.job_id,
            command.step,
            status="spawned",
            rank=prepared.rank,
            world_size=prepared.world_size,
            backend=prepared.backend,
        )

    def _prepare_spawn(self, command: SpawnRankCommand) -> _PreparedSpawn:
        cs = self.settings
        # CL2-12: an unreviewed node with no own data-plane config must never
        # degrade to trusting head-supplied values.
        if not cs.data_plane_subnet or not cs.data_plane_address:
            raise ClusterFormationError(
                "this node has no cluster.data_plane_subnet/data_plane_address "
                "configured; refusing to form (CL2-12)"
            )
        # CL2-09: the worker's own exhaustion accounting — one live formation.
        if self._cluster is not None and self._cluster.any_alive():
            raise SpawnBoundError(
                "a formation is already live on this worker; refusing a second "
                "spawn (CL2-09)"
            )
        # CL2-10: bound the head-supplied topology relationships (the scalars
        # themselves are bounded by the Pydantic schema).
        if command.world_size > MAX_WORLD_SIZE:
            raise ClusterFormationError(
                f"world_size {command.world_size} exceeds the {MAX_WORLD_SIZE} "
                "ceiling"
            )
        if command.rank >= command.world_size:
            raise ClusterFormationError(
                f"rank {command.rank} is not < world_size {command.world_size}"
            )
        if len(command.peers) != command.world_size:
            raise ClusterFormationError(
                f"{len(command.peers)} peer addresses != world_size "
                f"{command.world_size}"
            )
        own = cs.data_plane_address
        if not self._address_is_local(own):
            raise ClusterFormationError(
                f"configured data_plane_address {own} is not bound to any local "
                "interface (D7)"
            )
        # CL2-02: resolve the id against this node's OWN model dirs; a
        # head-supplied path can never reach the loader.
        resolved = self._model_resolver(command.model_id)
        if resolved is None:
            raise ModelNotPresentError(
                f"model {command.model_id!r} is not present in this node's model "
                "dirs; S5 has no auto-download"
            )
        model_path, _size = resolved
        # CL2-03: compute this rank's own entry from its own address, and
        # re-validate every entry against this node's own subnet.
        ips: list[str] = []
        for index, peer in enumerate(command.peers):
            address = own if index == command.rank else peer
            require_link_scope(
                address,
                data_plane_subnet=cs.data_plane_subnet,
                allow_routable_data_plane=cs.allow_routable_data_plane,
                allow_loopback=cs.allow_loopback,
            )
            ips.append(address)
        backend = self._resolve_backend(command.backend)
        ibv_devices = self._own_ibv_matrix(command) if backend == "jaccl" else None
        return _PreparedSpawn(
            rank=command.rank,
            world_size=command.world_size,
            backend=backend,
            model_path=model_path,
            ips=ips,
            base_port=int(cs.data_plane_base_port),
            seed=command.seed,
            ibv_devices=ibv_devices,
        )

    def _own_ibv_matrix(self, command: SpawnRankCommand) -> list[list[str | None]]:
        """The jaccl matrix this rank launches with, own row from own device.

        The head supplies a full matrix (device names for peers this node cannot
        observe). The worker keeps those peer rows but computes its OWN rank's
        row from its OWN ``cluster.rdma_device`` — a head-supplied device name
        for its own rank is never trusted (CL2-03), and a node with no configured
        device refuses to form (the CL2-12 discipline, extended to jaccl).
        """
        own_device = self.settings.rdma_device
        if not own_device:
            raise ClusterFormationError(
                "jaccl formation requested but this node has no "
                "cluster.rdma_device configured; refusing to form (CL2-12)"
            )
        size = command.world_size
        supplied = command.ibv_devices
        if (
            supplied is None
            or len(supplied) != size
            or any(len(row) != size for row in supplied)
        ):
            raise ClusterFormationError(
                f"jaccl spawn command carried no {size}x{size} ibv device matrix"
            )
        matrix = [list(row) for row in supplied]
        rank = command.rank
        matrix[rank] = [None if j == rank else own_device for j in range(size)]
        return matrix

    def _resolve_backend(self, backend: Backend) -> str:
        if backend == Backend.RING:
            return "ring"
        if backend == Backend.JACCL:
            return "jaccl"
        raise ClusterFormationError(f"unsupported backend {backend.value!r}")

    def _default_spawn(self, prepared: _PreparedSpawn) -> Any:
        from .launcher import LocalCluster

        cs = self.settings
        cluster = LocalCluster(
            model=prepared.model_path,
            world_size=prepared.world_size,
            backend=prepared.backend,
            base_port=prepared.base_port,
            seed=prepared.seed,
        )
        cluster.start(
            [prepared.rank],
            ips=prepared.ips,
            ibv_devices=prepared.ibv_devices,
            data_plane_subnet=cs.data_plane_subnet,
            allow_routable_data_plane=cs.allow_routable_data_plane,
            allow_loopback=cs.allow_loopback,
        )
        cluster.start_deathwatch()
        return cluster

    async def _teardown_local(self) -> None:
        cluster, self._cluster = self._cluster, None
        if cluster is None:
            return
        await asyncio.get_running_loop().run_in_executor(None, cluster.stop)

    def _default_resolve_model(self, model_id: str) -> tuple[str, int] | None:
        from ..model_discovery import discover_models_from_dirs

        dirs = self._global_settings.get_effective_model_dirs()
        discovered = discover_models_from_dirs(dirs)
        entry = discovered.get(model_id)
        if entry is None:
            for candidate in discovered.values():
                if candidate.source_repo_id == model_id:
                    entry = candidate
                    break
        if entry is None:
            return None
        return entry.model_path, int(entry.estimated_size)

    def _reportable_address(self) -> str:
        address = self.settings.data_plane_address
        if not address:
            return ""
        if not self._address_is_local(address):
            logger.warning(
                "cluster: configured data_plane_address %s is not bound to a "
                "local interface; not reporting it",
                address,
            )
            return ""
        return address

    def _address_is_local(self, address: str) -> bool:
        if not address:
            return False
        if self._local_addresses is not None:
            return address in self._local_addresses
        try:
            import psutil

            for infos in psutil.net_if_addrs().values():
                for info in infos:
                    if info.address == address:
                        return True
            return False
        except Exception:  # noqa: BLE001 - the subnet predicate is the real gate
            return True

    @staticmethod
    def _replay_key(raw: dict[str, Any]) -> tuple[str, int] | None:
        job_id = raw.get("job_id")
        step = raw.get("step")
        if isinstance(job_id, str) and job_id and isinstance(step, int):
            return (job_id, step)
        return None

    @staticmethod
    def _safe_step(raw: dict[str, Any]) -> int:
        step = raw.get("step")
        return step if isinstance(step, int) else 0


class ClusterManager:
    """Owns cluster state for one node, in whichever role it was started."""

    def __init__(
        self,
        global_settings: GlobalSettings,
        *,
        state_path: Path | None = None,
        client_factory: Callable[[str], ClusterClient] | None = None,
    ) -> None:
        self.global_settings = global_settings
        self.state_path = state_path or cluster_state_path(global_settings.base_path)
        self._client_factory = client_factory or (lambda url: ClusterClient(url))
        self._state = ClusterState()
        self._liveness: dict[str, MemberLiveness] = {}
        # S4 D1: advisory, liveness-side. Never persisted, never consulted
        # for auth or liveness — placement scoring only.
        self._node_state: dict[str, MemberNodeState] = {}
        # S4 D1: this worker's own inventory scan, cached at most 60s so the
        # heartbeat (every few seconds) does not re-scan the model dirs on
        # every beat.
        self._node_state_cache: dict[str, int] | None = None
        self._node_state_cache_at: float = 0.0
        self._queue = ClusterCommandQueue()
        self._scrub_task: asyncio.Task[None] | None = None
        self._heartbeat: HeartbeatSender | None = None
        self._versions: VersionInfo | None = None
        self._started = False
        # Worker role: applies head commands under CL2 confinement.
        self._executor: WorkerCommandExecutor | None = None
        # Head role: drives formation jobs on the E6 queue (D8), set in start().
        self._formation: FormationManager | None = None
        # Head role: drives transfer jobs (S5 D4), set in start().
        self._transfer: TransferManager | None = None
        # S5 D4: the single-active cluster-operation gate -- one flag + owner,
        # checked by both FormationManager's and TransferManager's queued
        # ops, so a formation and a transfer sequence can never interleave.
        self._active_operation: tuple[str, str] | None = None  # (kind, job_id)

    # ---- basic accessors -------------------------------------------------

    @property
    def role(self) -> str:
        return self.global_settings.cluster.role

    @property
    def state(self) -> ClusterState:
        return self._state

    @property
    def settings(self) -> ClusterSettings:
        return self.global_settings.cluster

    @property
    def versions(self) -> VersionInfo:
        if self._versions is None:
            self._versions = collect_versions()
        return self._versions

    def liveness(self, member_id: str) -> MemberLiveness | None:
        return self._liveness.get(member_id)

    def node_state(self, member_id: str) -> MemberNodeState | None:
        """The member's most recently reported node_state (S4 D1), or None."""
        return self._node_state.get(member_id)

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Load state and start the role's background tasks.

        Refuses to start any cluster role without a configured API key.
        Without one the operator tier is either wide open (the
        unauthenticated ``POST /admin/api/setup-api-key`` mints an admin
        session exactly when no key is set) or locked out (admin login
        refuses without a key). There is no opt-out: configuring a key is
        one line of operator config.
        """
        if self.role == "off":
            return
        if not self.global_settings.auth.api_key:
            raise RuntimeError(
                f"cluster.role={self.role!r} requires auth.api_key to be configured. "
                "Cluster control-plane endpoints are operator-gated and refuse to "
                "run on a server with no API key. Set one with "
                "`omlx serve --api-key ...`, the OMLX_API_KEY environment "
                "variable, or the admin settings page, then restart."
            )
        # CL5-17: loud, fail-fast assertion that the derived transfer port
        # range does not collide with the formation ring range or the jaccl
        # coordinator port. Applies to both roles -- either could spawn a
        # transfer-session rank.
        from ..settings import assert_transfer_ports_non_overlapping

        assert_transfer_ports_non_overlapping(self.settings)

        self._state = load_state(self.state_path)
        await self._queue.start()
        if self.role == "head":
            from .formation import FormationManager
            from .transfer import TransferManager

            self._formation = FormationManager(self)
            self._transfer = TransferManager(self)
            self._scrub_task = asyncio.create_task(self._scrub_loop())
        elif self.role == "worker":
            self._executor = WorkerCommandExecutor(self.global_settings)
            await self._executor.start()
            if self._state.worker is not None:
                await self._start_heartbeat(self._state.worker)
        self._started = True
        logger.info("Cluster manager started (role=%s)", self.role)

    async def stop(self) -> None:
        if self._scrub_task is not None:
            self._scrub_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scrub_task
            self._scrub_task = None
        if self._heartbeat is not None:
            await self._heartbeat.stop()
            self._heartbeat = None
        if self._formation is not None:
            await self._formation.stop()
            self._formation = None
        if self._transfer is not None:
            await self._transfer.stop()
            self._transfer = None
        if self._executor is not None:
            await self._executor.stop()
            self._executor = None
        await self._queue.stop()
        self._started = False

    @property
    def formation(self) -> FormationManager | None:
        """The head's formation manager, or None off the head role."""
        return self._formation

    @property
    def transfer(self) -> TransferManager | None:
        """The head's transfer manager, or None off the head role."""
        return self._transfer

    @property
    def executor(self) -> WorkerCommandExecutor | None:
        """The worker's command executor, or None off the worker role."""
        return self._executor

    # ---- S5 D4: the single-active cluster-operation gate ------------------

    def acquire_operation_gate(self, kind: str, job_id: str) -> None:
        """Claim the single-active-operation slot, or refuse with a 409
        naming the in-flight job (D4 -- "concurrent formation/transfer
        sequences are impossible by construction"). Symmetric: a formation
        op refuses while a transfer is active, and vice versa.
        """
        if self._active_operation is not None:
            active_kind, active_job = self._active_operation
            raise ClusterError(
                409,
                f"a {active_kind} operation ({active_job}) is already in "
                f"flight; {kind} operations must wait",
            )
        self._active_operation = (kind, job_id)

    def release_operation_gate(self, kind: str, job_id: str) -> None:
        """Release the slot, but only if it is still held by ``(kind, job_id)``
        -- a release from a stale/already-superseded caller is a no-op.
        """
        if self._active_operation == (kind, job_id):
            self._active_operation = None

    # ---- head commands ---------------------------------------------------

    async def mint_bootstrap_token(self) -> dict[str, Any]:
        """Mint or renew the bootstrap join token. Value returned once."""

        async def _apply() -> dict[str, Any]:
            token, record = mint_bootstrap_token(
                float(self.settings.bootstrap_token_ttl_s)
            )
            self._persist(replace(self._state, bootstrap=record))
            logger.info(
                "Bootstrap join token minted (expires_at=%s)", record.expires_at
            )
            return {
                "token": token,
                "expires_at": record.expires_at,
                "ttl_s": float(self.settings.bootstrap_token_ttl_s),
            }

        return await self._queue.submit("mint_bootstrap_token", _apply)

    async def revoke_bootstrap_token(self) -> dict[str, Any]:
        """Invalidate the current bootstrap join token."""

        async def _apply() -> dict[str, Any]:
            existed = self._state.bootstrap is not None
            self._persist(replace(self._state, bootstrap=None))
            logger.info("Bootstrap join token revoked (existed=%s)", existed)
            return {"revoked": existed}

        return await self._queue.submit("revoke_bootstrap_token", _apply)

    async def join(
        self,
        *,
        peer_host: str,
        port: int,
        name: str | None,
        versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Admit a worker: E10 version check, socket-derived address, mint secret."""

        async def _apply() -> dict[str, Any]:
            remote = VersionInfo.from_dict(versions)
            mismatch = compare_versions(self.versions, remote)
            if mismatch is not None:
                raise ClusterError(409, mismatch)
            if not 1 <= port <= 65535:
                raise ClusterError(400, f"Invalid member port: {port}")
            try:
                address = parse_member_address(
                    peer_host, allow_loopback=bool(self.settings.allow_loopback)
                )
            except ValueError as exc:
                raise ClusterError(400, str(exc)) from exc
            member = Member(
                id=secrets.token_hex(MEMBER_ID_BYTES),
                address=address,
                port=port,
                name=_sanitize_name(name),
                versions=remote,
                joined_at=time.time(),
                peer_cert_fingerprint=None,
            )
            secret = generate_secret()
            digests = dict(self._state.member_digests)
            digests[member.id] = digest_secret(secret)
            self._persist(
                replace(
                    self._state,
                    members=self._state.members + (member,),
                    member_digests=digests,
                )
            )
            logger.info(
                "Cluster member joined: id=%s endpoint=%s name=%s",
                member.id,
                member.endpoint,
                member.name,
            )
            return {
                "member_id": member.id,
                "member_secret": secret,
                "heartbeat_interval_s": float(self.settings.heartbeat_interval_s),
                "member_timeout_s": float(self.settings.member_timeout_s),
            }

        return await self._queue.submit("join", _apply)

    async def remove_member(self, member_id: str) -> dict[str, Any]:
        """Remove a member and revoke its secret."""

        async def _apply() -> dict[str, Any]:
            member = self._state.member(member_id)
            if member is None:
                raise ClusterError(404, f"Unknown cluster member: {member_id}")
            digests = dict(self._state.member_digests)
            digests.pop(member_id, None)
            self._persist(
                replace(
                    self._state,
                    members=tuple(m for m in self._state.members if m.id != member_id),
                    member_digests=digests,
                )
            )
            self._liveness.pop(member_id, None)
            self._node_state.pop(member_id, None)
            logger.info("Cluster member removed: id=%s", member_id)
            return {"member_id": member_id, "removed": True}

        return await self._queue.submit("remove_member", _apply)

    def record_heartbeat(
        self,
        member: Member,
        *,
        seq: int,
        epoch: str,
        job_updates: list[dict[str, Any]] | None = None,
        node_state: Any = None,
        transfer_updates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record liveness for a heartbeat. Touches no persisted state.

        A new epoch is accepted and resets the sequence — the request is
        already authenticated by the member secret, so an epoch change is
        a restart, not an unauthenticated replay. Inside a live epoch the
        sequence must strictly increase.

        The optional ``job_updates`` are attributed to the AUTHENTICATED
        ``member`` and any member/rank id in the update bodies is ignored
        (CL2-07). When the head has formation OR transfer work queued for
        this member the reply carries ``commands`` plus a signature over the
        commands and the echoed epoch+seq (CL2-05/CL2-06); S1 heartbeats (no
        updates, no pending commands) get exactly the S1 reply.

        The optional ``node_state`` (S4 D1) is parsed leniently: a malformed
        or absent value simply means this member has no capacity data for
        placement — it never fails the heartbeat, and liveness is recorded
        either way.

        The optional ``transfer_updates`` (S5 D1b) are a SIBLING of
        ``job_updates``, not an ack -- a TRANSFER_ROUND command's immediate
        ack only confirms the round started, and per-file progress/terminal
        state arrive here, possibly several heartbeats later, and are
        ACCUMULATED on the TransferJob rather than resolving a single-shot
        future. Bounded, not truncated (CL5-04): an oversized batch is
        dropped whole and logged, never partially applied.
        """
        if not epoch:
            raise ClusterError(400, "Heartbeat epoch must not be empty")
        now = time.time()
        current = self._liveness.get(member.id)
        if current is not None and current.epoch == epoch and seq <= current.last_seq:
            raise ClusterError(
                409,
                f"Heartbeat sequence {seq} is not greater than {current.last_seq} "
                f"for epoch {epoch}",
            )
        revived = current is not None and current.status == "lost"
        self._liveness[member.id] = MemberLiveness(
            epoch=epoch, last_seq=seq, last_heartbeat_at=now, status="active"
        )
        if revived:
            logger.info("Cluster member revived: id=%s", member.id)

        if node_state is not None:
            parsed = MemberNodeState.parse(node_state, received_at=now)
            if parsed is not None:
                self._node_state[member.id] = parsed
            else:
                logger.debug(
                    "cluster: dropping malformed node_state from member %s", member.id
                )

        if job_updates and self._formation is not None:
            self._formation.record_job_updates(member, job_updates)

        if transfer_updates and self._transfer is not None:
            bounded = _bound_transfer_updates(transfer_updates)
            if bounded is None:
                logger.warning(
                    "cluster: dropping oversized transfer_updates batch from "
                    "member %s (CL5-04)",
                    member.id,
                )
            else:
                self._transfer.record_transfer_updates(member, bounded)

        reply: dict[str, Any] = {
            "member_id": member.id,
            "status": "active",
            "heartbeat_interval_s": float(self.settings.heartbeat_interval_s),
        }
        commands: list[dict[str, Any]] = []
        if self._formation is not None:
            commands += self._formation.commands_for(member.id)
        if self._transfer is not None:
            commands += self._transfer.commands_for(member.id)
        if commands:
            digest = self._state.member_digests.get(member.id, "")
            reply["commands"] = commands
            reply["command_epoch"] = epoch
            reply["command_seq"] = seq
            reply["command_sig"] = sign_command_response(
                digest, commands, epoch=epoch, seq=seq
            )
        return reply

    async def member_leave(self, member: Member) -> dict[str, Any]:
        """Handle a member's own leave: revoke its secret."""
        return await self.remove_member(member.id)

    # ---- distributed formation (D8) --------------------------------------

    async def load_distributed(
        self,
        model_id: str,
        *,
        prefer: str = "distributed",
        source: Literal["peer", "hf"] | None = None,
    ) -> dict[str, Any]:
        """Stand a tensor-parallel model up across the pair (head only).

        S4 D4: guard-and-delegate. The pool is the single owner of cluster
        entry create/bind (it drives formation, not the other way around),
        so this only validates the request and hands off to
        `EnginePool.load_cluster_model` rather than calling
        `FormationManager.load` directly.

        S5 D6: `source` ("peer"/"hf") is this explicit operator surface's
        own choice when the worker lacks the model and a presence-
        resolution transfer is needed; `allow_source_choice=True` here (and
        only here -- never from `get_engine`'s implicit on-demand path)
        lets an ambiguous case (both sources viable, no `source` given)
        answer with a `choice_required` reply instead of guessing.
        """
        if self._formation is None:
            raise ClusterError(
                404, "distributed formation is only available on the head"
            )
        if not model_id:
            raise ClusterError(400, "a model id is required")
        pool = get_engine_pool()
        if pool is None:
            raise ClusterError(503, "the engine pool is not available")
        result: dict[str, Any] = await pool.load_cluster_model(
            model_id, prefer, source=source, allow_source_choice=True
        )
        return result

    async def unload_distributed(self, model_id: str) -> dict[str, Any]:
        """Tear a distributed formation down (head only). Guard-and-delegate
        to `EnginePool.unload_cluster_model` (S4 D4)."""
        if self._formation is None:
            raise ClusterError(
                404, "distributed formation is only available on the head"
            )
        if not model_id:
            raise ClusterError(400, "a model id is required")
        pool = get_engine_pool()
        if pool is None:
            raise ClusterError(503, "the engine pool is not available")
        result: dict[str, Any] = await pool.unload_cluster_model(model_id)
        return result

    def formation_status(self) -> dict[str, Any]:
        """Read-only formation/job state (head only).

        S5 P2: carries a sibling `transfer` field (summarized TransferJob
        rows -- D5's presence-resolution pre-step and any operator-driven
        transfer) alongside formation's own `jobs`, so a caller reads both
        surfaces from the one status endpoint.
        """
        if self._formation is None:
            raise ClusterError(
                404, "distributed formation is only available on the head"
            )
        snap = self._formation.snapshot()
        if self._transfer is not None:
            snap["transfer"] = {
                "jobs": [
                    _transfer_job_summary(job)
                    for job in self._transfer.snapshot()["jobs"]
                ]
            }
        return snap

    async def scrub(self) -> list[str]:
        """Mark members past the timeout as lost. Never revokes a credential.

        A timeout is a liveness statement, not a trust decision: a resumed
        heartbeat revives the member. Revocation belongs to explicit leave
        and operator removal.
        """

        async def _apply() -> list[str]:
            now = time.time()
            timeout = float(self.settings.member_timeout_s)
            expired: list[str] = []
            for member in self._state.members:
                live = self._liveness.get(member.id)
                if live is None or live.status != "active":
                    continue
                if now - live.last_heartbeat_at > timeout:
                    self._liveness[member.id] = replace(live, status="lost")
                    expired.append(member.id)
                    logger.warning(
                        "Cluster member timed out: id=%s endpoint=%s",
                        member.id,
                        member.endpoint,
                    )
            return expired

        return await self._queue.submit("scrub", _apply)

    # ---- worker commands -------------------------------------------------

    async def local_join(self, head_url: str, token: str) -> dict[str, Any]:
        """Perform the join handshake against the head and persist the credential."""

        async def _apply() -> dict[str, Any]:
            if not token:
                raise ClusterError(400, "A bootstrap join token is required")
            try:
                base_url = validate_head_url(head_url)
            except ValueError as exc:
                raise ClusterError(400, str(exc)) from exc
            payload = {
                "port": int(self.global_settings.server.port),
                "name": _sanitize_name(self.settings.node_name or socket.gethostname()),
                "versions": self.versions.to_dict(),
            }
            client = self._client_factory(base_url)
            try:
                reply = await client.post_json(JOIN_PATH, token=token, payload=payload)
            except ClusterClientError as exc:
                raise ClusterError(exc.status_code or 502, str(exc)) from exc
            member_id = str(reply.get("member_id") or "")
            secret = str(reply.get("member_secret") or "")
            if not member_id or not secret:
                raise ClusterError(502, "Head returned an incomplete join response")
            identity = WorkerIdentity(
                member_id=member_id,
                secret=secret,
                head_url=base_url,
                joined_at=time.time(),
            )
            self._persist(replace(self._state, worker=identity))
            interval = _bounded_interval(
                reply.get("heartbeat_interval_s"),
                default=float(self.settings.heartbeat_interval_s),
            )
            await self._start_heartbeat(identity, interval_s=interval)
            logger.info("Joined cluster head %s as member %s", base_url, member_id)
            return {
                "member_id": member_id,
                "head_url": base_url,
                "heartbeat_interval_s": interval,
            }

        return await self._queue.submit("local_join", _apply)

    async def local_leave(self) -> dict[str, Any]:
        """Leave the cluster and drop the local credential."""

        async def _apply() -> dict[str, Any]:
            identity = self._state.worker
            if identity is None:
                raise ClusterError(400, "This node is not a member of a cluster")
            if self._heartbeat is not None:
                await self._heartbeat.stop()
                self._heartbeat = None
            notified = True
            client = self._client_factory(identity.head_url)
            try:
                await client.post_json(LEAVE_PATH, token=identity.secret, payload={})
            except ClusterClientError as exc:
                # The local credential is dropped either way: a worker that
                # cannot reach the head must still be able to leave.
                notified = False
                logger.warning("Leave notification to head failed: %s", exc)
            self._persist(replace(self._state, worker=None))
            logger.info("Left cluster head %s", identity.head_url)
            return {
                "member_id": identity.member_id,
                "head_url": identity.head_url,
                "head_notified": notified,
            }

        return await self._queue.submit("local_leave", _apply)

    def local_status(self) -> dict[str, Any]:
        """Pull-based worker status (CL-14: the head never calls back)."""
        identity = self._state.worker
        return {
            "enabled": True,
            "role": self.role,
            "joined": identity is not None,
            "member_id": identity.member_id if identity else None,
            "head_url": identity.head_url if identity else None,
            "joined_at": identity.joined_at if identity else None,
            "heartbeat": self._heartbeat.status() if self._heartbeat else None,
            "versions": self.versions.to_dict(),
        }

    # ---- observability ---------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """State for the API and dashboard. Never contains credential material."""
        members = []
        for member in self._state.members:
            live = self._liveness.get(member.id)
            entry = member.to_dict()
            entry["status"] = live.status if live else "lost"
            entry["last_heartbeat_at"] = live.last_heartbeat_at if live else None
            entry["last_seq"] = live.last_seq if live else None
            entry["epoch"] = live.epoch if live else None
            members.append(entry)
        bootstrap = self._state.bootstrap
        snapshot: dict[str, Any] = {
            "enabled": True,
            "role": self.role,
            "members": members,
            "member_count": len(members),
            "active_count": sum(1 for m in members if m["status"] == "active"),
            "heartbeat_interval_s": float(self.settings.heartbeat_interval_s),
            "member_timeout_s": float(self.settings.member_timeout_s),
            "bootstrap_token": {
                "configured": bootstrap is not None,
                "expires_at": bootstrap.expires_at if bootstrap else None,
            },
            "jobs": [job.to_dict() for job in self._state.jobs],
            "versions": self.versions.to_dict(),
        }
        if self.role == "worker":
            snapshot["local"] = self.local_status()
        if self._formation is not None:
            snapshot["formation"] = self._formation.snapshot()
        if self._transfer is not None:
            # S5 P2: sibling of "formation" (dashboard/CLI read both from
            # this one snapshot poll -- D6's compose-not-duplicate letter).
            snapshot["transfer"] = {
                "jobs": [
                    _transfer_job_summary(job)
                    for job in self._transfer.snapshot()["jobs"]
                ]
            }
        return snapshot

    # ---- internals -------------------------------------------------------

    def _persist(self, state: ClusterState) -> None:
        save_state(self.state_path, state)
        self._state = state

    # S4 D1: worker-side node_state assembly, attached to every heartbeat.
    _NODE_STATE_RESCAN_INTERVAL_S = 60.0

    def invalidate_node_state_cache(self) -> None:
        """Force the next heartbeat to re-scan model dirs (after a local load/unload)."""
        self._node_state_cache = None

    def _worker_memory_ceiling(self) -> int:
        """This node's own ``get_final_ceiling()``, verbatim (no fallback).

        Reuses the EnginePool's already-wired enforcer callback
        (`engine_pool.py:201-213`) via the injected getter rather than a
        second wiring path; deliberately does NOT chain the pool's
        admission fallback (`_fallback_admission_ceiling`) — D2's
        capacity-unknown rule needs the raw 0 when this worker's guard is
        off, not a substituted estimate.
        """
        pool = get_engine_pool()
        if pool is None:
            return 0
        try:
            return int(pool._current_ceiling())
        except Exception:  # noqa: BLE001 - advisory, never blocks the beat
            return 0

    def _scan_models_present(self) -> dict[str, int]:
        """model_id -> size_bytes for models physically on this node's disk.

        Cached for `_NODE_STATE_RESCAN_INTERVAL_S` so a 5s heartbeat does
        not re-scan the model dirs on every beat.
        """
        now = time.time()
        if (
            self._node_state_cache is not None
            and now - self._node_state_cache_at < self._NODE_STATE_RESCAN_INTERVAL_S
        ):
            return self._node_state_cache
        present: dict[str, int] = {}
        for model_dir in self.global_settings.get_effective_model_dirs():
            try:
                scanned = discover_models(model_dir)
            except (OSError, ValueError):
                # A configured-but-not-yet-created directory (e.g. the
                # default base_path/models before a model is ever added)
                # must not abort the whole scan -- other configured dirs
                # (the HF cache) still need scanning, and this is advisory
                # inventory, not a hard dependency.
                continue
            for model_id, info in scanned.items():
                present.setdefault(model_id, info.estimated_size)
        self._node_state_cache = present
        self._node_state_cache_at = now
        return present

    def _collect_node_state(self) -> dict[str, Any] | None:
        """Worker-side node_state payload for the heartbeat (D1).

        Advisory only: any failure returns None (omitting the field, S1's
        heartbeat shape) rather than raising into HeartbeatSender.
        """
        try:
            return {
                "total_memory": int(get_total_memory()),
                "memory_ceiling": self._worker_memory_ceiling(),
                "models_present": self._scan_models_present(),
            }
        except Exception as exc:  # noqa: BLE001 - advisory, never blocks the beat
            logger.debug("cluster: node_state collection failed: %s", exc)
            return None

    async def _start_heartbeat(
        self, identity: WorkerIdentity, *, interval_s: float | None = None
    ) -> None:
        if self._heartbeat is not None:
            await self._heartbeat.stop()
        executor = self._executor
        self._heartbeat = HeartbeatSender(
            identity,
            interval_s=(
                interval_s
                if interval_s is not None
                else float(self.settings.heartbeat_interval_s)
            ),
            client_factory=self._client_factory,
            command_sink=executor.deliver if executor is not None else None,
            job_updates_provider=(
                executor.pending_job_updates if executor is not None else None
            ),
            transfer_updates_provider=(
                executor.pending_transfer_updates if executor is not None else None
            ),
            node_state_provider=self._collect_node_state,
        )
        if executor is not None:
            # CL5-10: a transfer job is bound to the epoch live when it
            # started; the executor needs to know its own current one.
            executor.set_epoch(self._heartbeat.epoch)
        await self._heartbeat.start()

    async def _scrub_loop(self) -> None:
        interval = max(0.1, float(self.settings.heartbeat_interval_s))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.scrub()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.warning("Cluster scrub failed: %s", exc)


# S5 CL5-04: bounds for one heartbeat's `transfer_updates` batch.
MAX_TRANSFER_UPDATES_PER_BEAT = 50
MAX_TRANSFER_UPDATE_STRING_LENGTH = 4096
MAX_TRANSFER_UPDATE_LIST_ENTRIES = 100_000


def _bound_transfer_updates(
    raw: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """CL5-04: bound one heartbeat's ``transfer_updates`` batch.

    Rejects (returns ``None`` for the WHOLE batch) rather than truncating any
    single oversized field -- silently truncating a ``have`` list or an
    error string could hide data loss instead of surfacing it.
    """
    if not isinstance(raw, list) or len(raw) > MAX_TRANSFER_UPDATES_PER_BEAT:
        return None
    for update in raw:
        if not isinstance(update, dict):
            return None
        for value in update.values():
            if (
                isinstance(value, str)
                and len(value) > MAX_TRANSFER_UPDATE_STRING_LENGTH
            ):
                return None
            if isinstance(value, list):
                if len(value) > MAX_TRANSFER_UPDATE_LIST_ENTRIES:
                    return None
                if any(
                    isinstance(item, str)
                    and len(item) > MAX_TRANSFER_UPDATE_STRING_LENGTH
                    for item in value
                ):
                    return None
    return raw


# S5 P2: a status-surface summary of one TransferJob. Deliberately drops the
# full `manifest`/`have` arrays (those can be tens of thousands of entries
# for a large model) down to counts -- both `/v1/cluster/models/status` and
# the admin dashboard's 5s snapshot poll only need progress, not the whole
# file-list payload.
def _transfer_job_summary(job_dict: dict[str, Any]) -> dict[str, Any]:
    manifest = job_dict.get("manifest") or []
    have = job_dict.get("have") or []
    return {
        "id": job_dict.get("id"),
        "kind": job_dict.get("kind"),
        "status": job_dict.get("status"),
        "model_id": job_dict.get("model_id"),
        "member_id": job_dict.get("member_id"),
        "source": job_dict.get("source"),
        "created_at": job_dict.get("created_at"),
        "updated_at": job_dict.get("updated_at"),
        "rounds_completed": job_dict.get("rounds_completed"),
        "error": job_dict.get("error"),
        "files_total": len(manifest),
        "files_verified": len(have),
    }


def _bounded_interval(value: Any, *, default: float) -> float:
    """Clamp a head-supplied heartbeat interval, rejecting non-numeric (CL2-10).

    A hostile or broken join reply must not be able to silence the worker with
    an absurd interval, or crash it with a non-numeric one.
    """
    try:
        interval = float(value)
    except (TypeError, ValueError):
        return default
    if interval <= 0:
        return default
    return min(interval, MAX_HEARTBEAT_INTERVAL_S)


def _sanitize_name(name: str | None) -> str:
    """Keep member names printable and bounded — they reach logs and the UI."""
    if not name:
        return ""
    cleaned = "".join(ch for ch in name if ch.isprintable())
    return cleaned.strip()[:MAX_MEMBER_NAME_LENGTH]
