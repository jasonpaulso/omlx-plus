# SPDX-License-Identifier: Apache-2.0
"""Spawning and supervising rank processes from a daemon.

The daemon that owns rank 0 talks to it over pipes; a peer daemon owns its own
ranks and drives them implicitly, because rank 0 broadcasts every decision over
the collective. Nothing here uses SSH, and nothing assumes the nodes share a
filesystem or have oMLX installed at the same path.

Ordering matters at startup. Every rank blocks inside ``mx.distributed.init()``
until the whole world arrives, so a slow rank holds up the rest and a rank that
never starts hangs them until they are killed. Readiness is therefore read from
the process table, never by connecting to a ring port (salvage pitfall 3: a TCP
probe is accepted as a peer and poisons the handshake).

This module is mlx-free: the unit gate exercises the env build, the spawn bound
and the sweep without touching MLX.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omlx.cluster import hostfile

logger = logging.getLogger(__name__)

# The rank process entry points, matched in the process table by the sweep.
WORKER_MODULE = "omlx.cluster.rank_worker"
# S5: the per-round transfer rank script (D1/R3c) -- a distinct entry point,
# never rank_worker, spawned by `launch_transfer_session`.
TRANSFER_MODULE = "omlx.cluster.transfer_rank"


def _sweepable_modules() -> tuple[str, str]:
    """``(WORKER_MODULE, TRANSFER_MODULE)``, read live rather than frozen at
    import time -- a module-level constant tuple built once would bake in
    whatever these two names pointed to at import, and a caller that
    monkeypatches ``WORKER_MODULE`` (e.g. tests, to run a test-only entry
    point) would never see the sweep follow it."""
    return (WORKER_MODULE, TRANSFER_MODULE)


# init() blocks until the whole world joins; past this we assume someone never
# arrives and tear the run down rather than hanging forever.
DEFAULT_JOIN_TIMEOUT_S = 120

# How often a DeathWatch looks, and how many unreachable polls it tolerates
# before declaring a node gone. A definitive answer ("your rank is not
# running") fires at once; unreachability gets patience because the control
# plane shares the LAN and a wifi blink must not kill a healthy formation.
DEATHWATCH_INTERVAL_S = 2.0
DEATHWATCH_STRIKES = 5


class SpawnBoundError(RuntimeError):
    """A spawn was refused because a formation is already live (CL2-09)."""


# -- CL2-09: at most one live formation per machine --------------------------

_spawn_lock = threading.Lock()
_active_cluster: LocalCluster | None = None


def _register_formation(cluster: LocalCluster) -> None:
    """Claim this machine's single formation slot, or refuse (CL2-09).

    The bound is the worker's own accounting, not a head-supplied limit: each
    rank loads a multi-GB shard, so an unbounded spawn is machine-level
    exhaustion. A second formation is refused while one is live.
    """
    global _active_cluster
    with _spawn_lock:
        if _active_cluster is not None and _active_cluster.any_alive():
            raise SpawnBoundError(
                "a cluster formation is already live on this machine; "
                "refusing to spawn a second (CL2-09)"
            )
        _active_cluster = cluster


def _release_formation(cluster: LocalCluster) -> None:
    global _active_cluster
    with _spawn_lock:
        if _active_cluster is cluster:
            _active_cluster = None


# -- S5 R3c: transfer sessions get their OWN single-slot bound ---------------
#
# Deliberately a separate lock/singleton from the formation slot above: a
# live formation (serving requests) and one transfer session must coexist on
# the same machine, so a transfer must never claim the formation's slot. A
# SECOND concurrent transfer session is still refused -- each round is its
# own fresh 2-rank ring session (D1/D2), and running two at once would mean
# two rank processes racing for the same transfer ports.


class TransferSpawnBoundError(RuntimeError):
    """A transfer session spawn was refused: one is already live (R3c)."""


_transfer_spawn_lock = threading.Lock()
_active_transfer_session: LocalCluster | None = None


def _register_transfer_session(cluster: LocalCluster) -> None:
    global _active_transfer_session
    with _transfer_spawn_lock:
        if (
            _active_transfer_session is not None
            and _active_transfer_session.any_alive()
        ):
            raise TransferSpawnBoundError(
                "a transfer session is already live on this machine; "
                "refusing a second concurrent one (R3c)"
            )
        _active_transfer_session = cluster


def _release_transfer_session(cluster: LocalCluster) -> None:
    global _active_transfer_session
    with _transfer_spawn_lock:
        if _active_transfer_session is cluster:
            _active_transfer_session = None


# -- liveness ----------------------------------------------------------------


class DeathWatch(threading.Thread):
    """Notices a dead rank in seconds, instead of at the idle timeout.

    Without this, a rank that dies leaves every other rank blocked inside a
    collective — mlx has no fault tolerance — and the only thing that ends the
    wait is a minutes-long idle timeout. The watch polls liveness and fires
    ``on_death`` once; the owner then kills its local ranks, which closes
    rank 0's reply pipe and turns "blocked for ten minutes" into "failed in
    seconds".

    Each check returns True (alive), False (definitively dead — fires at once),
    or None (could not tell — counts a strike, fires after ``strikes`` misses).
    Liveness is read from process tables, never by connecting to a ring port.
    """

    def __init__(
        self,
        checks: list[tuple[str, Callable[[], bool | None]]],
        on_death: Callable[[str, str], None],
        *,
        interval: float | None = None,
        strikes: int | None = None,
    ) -> None:
        super().__init__(name="cluster-deathwatch", daemon=True)
        self._checks = checks
        self._on_death = on_death
        self._interval = DEATHWATCH_INTERVAL_S if interval is None else interval
        self._strikes = DEATHWATCH_STRIKES if strikes is None else strikes
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Stand the watch down. Safe to call from the watch's own callback."""
        self._stop_event.set()
        if threading.current_thread() is not self:
            self.join(timeout=self._interval * 2)

    def run(self) -> None:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        misses = {label: 0 for label, _ in self._checks}
        while not self._stop_event.wait(self._interval):
            for label, check in self._checks:
                try:
                    verdict = check()
                except Exception:  # noqa: BLE001 - a broken check is a miss
                    verdict = None
                if verdict is True:
                    misses[label] = 0
                    continue
                if verdict is None:
                    misses[label] += 1
                    if misses[label] < self._strikes:
                        continue
                    reason = f"unreachable for {misses[label]} checks"
                else:
                    reason = "reported dead"
                if self._stop_event.is_set():
                    return
                self._stop_event.set()
                logger.error("cluster: deathwatch: %s %s", label, reason)
                self._on_death(label, reason)
                return


def _parent_is_alive(pid: int | None) -> bool:
    """True when ``pid`` is a live process other than init.

    A rank reparented to launchd (ppid 1) lost its daemon and is an orphan by
    definition; anything else still has somebody responsible for it.
    """
    if pid is None or pid <= 1:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001
        return False


def sweep_orphaned_ranks() -> int:
    """Kill rank processes on this machine that no live daemon is driving.

    A worker left alive by a failed run holds its ring port, and the next
    rank 0 then quietly fails to own it — the whole cluster dies with a connect
    timeout that looks like a network fault. A restarted daemon has no handle
    on the children it left behind; only the process table remembers them.

    **Our own children are never swept.** Two daemons on one machine is exactly
    how the stack is tested without a second Mac, and a peer forming its rank
    must not kill the leader's rank 0 next to it. Only reparented ranks — whose
    parent is gone — are orphans.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        return 0

    ours = os.getpid()
    killed = 0
    sweepable = _sweepable_modules()
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if not any(module in cmdline for module in sweepable):
            continue
        parent = proc.info.get("ppid")
        if parent == ours:
            continue
        if _parent_is_alive(parent):
            continue
        try:
            proc.kill()
            killed += 1
            logger.warning("cluster: killed orphaned rank process %d", proc.info["pid"])
        except Exception:  # noqa: BLE001 - already gone, or not ours
            continue
    return killed


# -- pipe I/O primitives -----------------------------------------------------


class ReplyReader:
    """Reads rank 0's newline-delimited replies with an idle timeout.

    Owns its buffering rather than using the ``Popen`` text wrapper, because
    the timeout has to be honest: ``select`` reports on the file descriptor,
    and a buffered reader routinely pulls several replies out of the pipe at
    once — so a wrapper-based implementation would sit in ``select`` waiting for
    bytes that had already arrived and time out with the answer in its hand.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""

    def readline(self, timeout: float | None) -> str | None:
        """The next reply line, ``""`` at EOF, or ``None`` when it timed out."""
        import select

        deadline = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self._buffer:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if (
                    remaining <= 0
                    or not select.select([self._fd], [], [], remaining)[0]
                ):
                    return None
            chunk = os.read(self._fd, 65536)
            if not chunk:
                return ""
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace")


class CommandReader:
    """Rank 0's command channel, with the buffering owned here.

    The serving loop needs two read shapes from the same pipe: block until the
    next command while idle, and drain whatever arrived between steps while
    serving. A buffered file object cannot provide both — its ``readline``
    reads ahead, and a later ``select`` then reports an empty pipe while
    commands sit in the Python buffer.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""
        self._eof = False

    def readline(self) -> str:
        """The next command line, blocking. ``""`` once the pipe is closed."""
        import select

        while b"\n" not in self._buffer:
            if self._eof:
                return ""
            select.select([self._fd], [], [])
            if not self._fill():
                return ""
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace")

    def drain_lines(self) -> list[str]:
        """Every complete line already arrived, without blocking."""
        import select

        while not self._eof and select.select([self._fd], [], [], 0)[0]:
            if not self._fill():
                break
        lines: list[str] = []
        while b"\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition(b"\n")
            lines.append(line.decode("utf-8", "replace"))
        return lines

    def _fill(self) -> bool:
        try:
            chunk = os.read(self._fd, 65536)
        except (BlockingIOError, InterruptedError):
            return True
        except OSError:
            chunk = b""
        if not chunk:
            self._eof = True
            return False
        self._buffer += chunk
        return True


class ControlChannel:
    """Rank 0's out-of-band read side, drained between decode steps.

    Signals name the request they concern, so a late abort for a request that
    already finished is a no-op by construction. A closed pipe means the daemon
    is gone, reported as an abort of everything — there is nobody left to stream
    to.
    """

    def __init__(self, fd: int | None) -> None:
        self._fd = fd
        self._buffer = b""
        self._closed = False

    def take_events(self) -> list[dict[str, Any]]:
        """Parsed abort signals that have arrived since the last call."""
        import select

        if self._fd is None:
            return []
        if self._closed:
            return [{"op": "abort"}]

        while select.select([self._fd], [], [], 0)[0]:
            try:
                chunk = os.read(self._fd, 4096)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                self._closed = True
                break
            if not chunk:
                self._closed = True
                break
            self._buffer += chunk

        events: list[dict[str, Any]] = []
        while b"\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition(b"\n")
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                logger.warning("cluster: unparseable control signal, ignoring")
                continue
            if isinstance(event, dict) and event.get("op") == "abort":
                events.append(event)

        if self._closed:
            events.append({"op": "abort"})
        return events


def resolve_python(candidate: str | None = None) -> str:
    """Pick an interpreter that can actually ``import omlx``.

    A peer daemon spawns ranks from whatever cwd and environment launchd gave
    it, and an interpreter that cannot see oMLX fails in the least obvious way:
    the rank dies immediately with ``ModuleNotFoundError``, nobody listens on
    its ring port, and *the other machine* reports ``[ring] Couldn't connect``
    — which reads as a firewall fault and is not one.
    """
    python = candidate or sys.executable
    probe = subprocess.run(
        [python, "-c", "import omlx, sys; sys.stdout.write(omlx.__file__)"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"{python} cannot import omlx: {probe.stderr.strip() or 'no output'}"
        )
    return python


# -- the rank processes a daemon owns ----------------------------------------


@dataclass
class RankProcess:
    """A spawned rank and the pipe to it, if it is rank 0."""

    rank: int
    process: subprocess.Popen
    # Write end of the out-of-band signal pipe. Only rank 0 gets one; peers
    # learn about an abort through the collective, from rank 0.
    control_w: int | None = None
    replies: ReplyReader | None = None

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


@dataclass
class LocalCluster:
    """Rank processes this daemon is responsible for.

    Usually one — a node runs a single rank. Several ranks on one machine is
    supported because it is how the whole stack is tested without a second Mac.
    """

    model: str
    world_size: int
    backend: str = "ring"
    base_port: int = hostfile.DEFAULT_RING_BASE_PORT
    seed: int = 0
    python: str = field(default_factory=lambda: sys.executable)
    metrics_dir: str | Path | None = None
    # CL2-09 is the WORKER's own exhaustion accounting: a worker spawning a
    # head-commanded rank claims the single per-machine formation slot. The
    # head forming its own rank 0 is operator-initiated and serialised by the
    # E6 queue, so it does not hold the worker's slot — and colocating both in
    # one process (the single-host integration test) needs exactly one of them
    # to hold it. The head constructs with this False; the worker leaves it True.
    enforce_spawn_bound: bool = True
    # S5 R3c: which entry point a spawned rank runs, and how its argv is
    # built. Defaults reproduce the formation shape exactly (rank_worker,
    # `--model`/`--seed`/`--control-fd`/`--metrics-path`); a non-default
    # `module` is never used without a matching `argv_builder` (transfer
    # sessions run `transfer_rank`, whose argv shape is unrelated).
    #
    # Defaults to None, resolved to `WORKER_MODULE` at spawn time by
    # `_resolve_module`, rather than binding `WORKER_MODULE` directly here:
    # a dataclass field's plain-value default is captured once, at class
    # definition (import) time. A caller that monkeypatches the module-level
    # `WORKER_MODULE` afterwards (tests run a test-only entry point this
    # way) would never reach an already-baked default -- resolving at use
    # time is what makes the patch effective.
    module: str | None = None
    argv_builder: Callable[[int], list[str]] | None = None
    ranks: list[RankProcess] = field(default_factory=list)
    _workdir: Path | None = None
    _deathwatch: DeathWatch | None = None
    _stdin_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Per-backend launch context, written once in ``_start`` and read by each
    # ``_spawn_one``: ring sets ``_hostfile_path``; jaccl sets ``_ibv_path``
    # (the MLX_IBV_DEVICES matrix file) and ``_coordinator`` ("<rank0_ip>:port").
    _hostfile_path: Path | None = None
    _ibv_path: Path | None = None
    _coordinator: str | None = None

    def any_alive(self) -> bool:
        return any(r.alive for r in self.ranks)

    def start(
        self,
        ranks: list[int],
        *,
        ips: list[str],
        ibv_devices: list[list[str | None]] | None = None,
        data_plane_subnet: str | None = None,
        allow_routable_data_plane: bool = False,
        allow_loopback: bool = False,
    ) -> None:
        """Spawn the given ranks on this machine.

        ``ips`` is the whole cluster's data-plane address list in rank order —
        every node needs it, because the ring hostfile and the jaccl coordinator
        both describe all ranks. For the jaccl backend ``ibv_devices`` is the
        ``MLX_IBV_DEVICES`` matrix (``matrix[i][j]`` names node ``i``'s device to
        node ``j``); it is ignored for the ring backend. When
        ``data_plane_subnet`` is given, every address is put through the D7
        predicate before it can enter the launch context (CL2-03); with the
        subnet unset the caller has already validated them.
        """
        if self.enforce_spawn_bound:
            _register_formation(self)
        try:
            self._start(
                ranks,
                ips=ips,
                ibv_devices=ibv_devices,
                data_plane_subnet=data_plane_subnet,
                allow_routable_data_plane=allow_routable_data_plane,
                allow_loopback=allow_loopback,
            )
        except BaseException:
            self.kill()
            if self.enforce_spawn_bound:
                _release_formation(self)
            raise

    def _start(
        self,
        ranks: list[int],
        *,
        ips: list[str],
        ibv_devices: list[list[str | None]] | None,
        data_plane_subnet: str | None,
        allow_routable_data_plane: bool,
        allow_loopback: bool,
    ) -> None:
        sweep_orphaned_ranks()

        if data_plane_subnet is not None:
            for ip in ips:
                hostfile.require_link_scope(
                    ip,
                    data_plane_subnet=data_plane_subnet,
                    allow_routable_data_plane=allow_routable_data_plane,
                    allow_loopback=allow_loopback,
                )

        if self.backend not in ("ring", "jaccl"):
            raise ValueError(
                f"unsupported backend {self.backend!r}; expected ring or jaccl"
            )
        # World size is the hostfile / coordinator peer-list length; a mismatch
        # would form a smaller world than the caller believes (the singleton-
        # group trap strict=True exists to prevent).
        if len(ips) != self.world_size:
            raise ValueError(
                f"world_size {self.world_size} != {len(ips)} data-plane addresses"
            )

        self._workdir = Path(tempfile.mkdtemp(prefix="omlx-cluster-"))
        if self.backend == "ring":
            addresses = hostfile.ring_addresses(ips, self.base_port)
            self._hostfile_path = hostfile.write_ring_hostfile(
                self._workdir / "hosts.json", addresses
            )
        else:  # jaccl: a TCP coordinator bootstraps the RDMA queue-pair exchange
            if ibv_devices is None:
                raise ValueError("the jaccl backend requires an ibv device matrix")
            self._ibv_path = hostfile.write_ibv_devices(
                self._workdir / "ibv.json", ibv_devices
            )
            self._coordinator = f"{ips[0]}:{hostfile.DEFAULT_JACCL_COORDINATOR_PORT}"

        # Rank 0 must bind its listening socket before any peer connects: a peer
        # that starts first burns its connect window and dies with error 65,
        # which reads as a firewall fault and is not one (salvage pitfall 3). So
        # spawn rank 0 first, wait until it is listening, then spawn the rest.
        ordered = sorted(ranks)
        for index, rank in enumerate(ordered):
            self._spawn_one(rank)
            more_ranks_pending = rank == 0 and index + 1 < len(ordered)
            if more_ranks_pending and not self.wait_until_ready(
                timeout=DEFAULT_JOIN_TIMEOUT_S
            ):
                raise RuntimeError(
                    "rank 0 exited before the world formed; refusing to "
                    "spawn peers against a dead listener"
                )

    def _resolve_module(self) -> str:
        """The rank entry-point module, resolved at spawn time (S5 fix).

        See the ``module`` field's docstring: resolving here rather than
        reading a frozen dataclass default is what makes a monkeypatched
        ``WORKER_MODULE`` actually take effect.
        """
        return self.module if self.module is not None else WORKER_MODULE

    def _build_argv(self, rank: int, *, control_r: int | None) -> list[str]:
        """The spawned rank's argv, apart from actually spawning it.

        Factored out of ``_spawn_one`` so the module-resolution/argv shape
        is assertable without spawning a subprocess or opening a pipe --
        ``control_r`` is the (already-opened, or ``None``) read end of the
        control pipe rank 0 gets; the caller decides whether to open one.
        """
        module = self._resolve_module()
        if self.argv_builder is not None:
            # A non-rank_worker entry point (e.g. transfer_rank): the caller
            # owns the whole argv shape, including any inheritable fds it
            # wants -- rank_worker's control-pipe convention is specific to
            # its own daemon-abort protocol and does not apply generically.
            return [self.python, "-m", module, *self.argv_builder(rank)]
        argv = [
            self.python,
            "-m",
            module,
            "--model",
            self.model,
            "--seed",
            str(self.seed),
        ]
        if self.metrics_dir is not None:
            argv += [
                "--metrics-path",
                str(Path(self.metrics_dir) / f"rank-{rank}.json"),
            ]
        if control_r is not None:
            argv += ["--control-fd", str(control_r)]
        return argv

    def _spawn_one(self, rank: int) -> None:
        env = hostfile.local_worker_env(
            dict(os.environ),
            rank=rank,
            backend=self.backend,
            hostfile=self._hostfile_path,
            coordinator=self._coordinator,
            ibv_devices=self._ibv_path,
        )
        control_r = control_w = None
        if self.argv_builder is None and rank == 0:
            control_r, control_w = os.pipe()
            os.set_inheritable(control_r, True)
        argv = self._build_argv(rank, control_r=control_r)

        try:
            process = subprocess.Popen(
                argv,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # worker logs go to the daemon's stderr
                text=True,
                bufsize=1,
                pass_fds=(control_r,) if control_r is not None else (),
            )
        except BaseException:
            if control_w is not None:
                os.close(control_w)
            raise
        finally:
            if control_r is not None:
                os.close(control_r)

        self.ranks.append(RankProcess(rank=rank, process=process, control_w=control_w))
        logger.info("cluster: spawned rank %d (pid %d)", rank, process.pid)

    # -- readiness ---------------------------------------------------------

    def wait_until_ready(self, *, timeout: float = 60.0) -> bool:
        """Block until it is safe to start peer ranks on other machines.

        Readiness is read from the process table, never by connecting (salvage
        pitfall 3). Nothing observable is not treated as failure — colocated
        ranks may already be past init — so a short grace lets the caller
        proceed. The only real failure is a rank that died.
        """
        # Rank 0 listens on the ring base port; under jaccl it listens on the
        # coordinator port (the TCP bootstrap for the RDMA queue-pair exchange).
        port = (
            self.base_port
            if self.backend == "ring"
            else hostfile.DEFAULT_JACCL_COORDINATOR_PORT
        )
        grace_deadline = time.monotonic() + min(timeout, 5.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not all(r.alive for r in self.ranks):
                logger.error("cluster: a rank exited before the world formed")
                return False
            if self._is_listening(port):
                return True
            if time.monotonic() > grace_deadline:
                return True
            time.sleep(0.25)
        return all(r.alive for r in self.ranks)

    def wait_ready(self, *, timeout: float = 180.0) -> dict:
        """Block until rank 0 reports its shard is loaded and serving.

        Consumes the unsolicited ``ready`` frame rank 0 emits once, so a later
        :meth:`stream` starts against a clean reply channel. Raises if rank 0
        dies or stays silent past ``timeout`` (the shard load can be minutes).
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        reader = self._reply_reader(leader)
        line = reader.readline(timeout)
        if line is None:
            raise RuntimeError(f"rank 0 did not report ready within {timeout:.0f}s")
        if not line:
            raise RuntimeError("rank 0 closed its channel before reporting ready")
        frame: dict = json.loads(line)
        if frame.get("event") != "ready":
            raise RuntimeError(f"rank 0's first frame was not 'ready': {frame}")
        return frame

    def _is_listening(self, port: int) -> bool:
        """True when one of this node's ranks holds a listening socket on ``port``.

        Read-only by construction; a TCP connect to a ring port is not passive
        (salvage pitfall 3). The interpreter that ends up listening is not
        always the direct child, so the whole subtree is inspected.
        """
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil is a hard dependency
            return False

        for entry in self.ranks:
            try:
                proc = psutil.Process(entry.process.pid)
                candidates = [proc] + proc.children(recursive=True)
            except Exception:  # noqa: BLE001 - the rank may have just exited
                continue
            for candidate in candidates:
                try:
                    connections = candidate.net_connections(kind="tcp")
                except Exception:  # noqa: BLE001 - permissions, or it exited
                    continue
                for conn in connections:
                    if conn.status == "LISTEN" and conn.laddr.port == port:
                        return True
        return False

    # -- deathwatch --------------------------------------------------------

    def start_deathwatch(
        self,
        on_death: Callable[[str, str], None] | None = None,
        *,
        interval: float | None = None,
    ) -> DeathWatch:
        """Watch the local ranks, killing the formation if one dies.

        Started before the load command so a rank that dies in the expensive
        load or decode window is caught in seconds (salvage pitfall 5). The
        default ``on_death`` kills the formation, which closes rank 0's reply
        pipe — that EOF is what surfaces as a clean error at the pipe.
        """

        def check_rank(entry: RankProcess) -> Callable[[], bool | None]:
            return lambda: entry.alive

        checks = [(f"rank-{entry.rank}", check_rank(entry)) for entry in self.ranks]
        watch = DeathWatch(
            checks,
            on_death or (lambda _label, _reason: self.kill()),
            interval=interval,
        )
        watch.start()
        self._deathwatch = watch
        return watch

    # -- the rank-0 pipe ---------------------------------------------------

    @property
    def leader(self) -> RankProcess | None:
        """The rank-0 process, if this machine owns it."""
        return next((r for r in self.ranks if r.rank == 0), None)

    def _reply_reader(self, leader: RankProcess) -> ReplyReader:
        assert leader.process.stdout is not None
        if leader.replies is None:
            leader.replies = ReplyReader(leader.process.stdout.fileno())
        return leader.replies

    def command(self, payload: dict, *, timeout: float | None = None) -> dict:
        """Send one command to rank 0 and read one reply."""
        return next(self.stream(payload, timeout=timeout))

    def stream(self, payload: dict, *, timeout: float | None = None):
        """Send one command and yield replies until ``done`` or a failure.

        ``timeout`` is an *idle* timeout — the longest this waits for the next
        reply. A rank that dies mid-collective leaves rank 0 blocked in mlx with
        no way to notice; without this the caller hangs until it gives up.

        One request at a time only: this reads every reply off rank 0's pipe
        until ``done``, so a second concurrent caller would steal frames that
        belong to this one. S2 callers (single generation in flight) and tests
        that drive exactly one request use this; S3's continuous-batching
        ``ClusterEngine`` uses :meth:`write` / :meth:`read_reply` directly
        instead, through its own demux (D5) — several requests genuinely
        interleave on the same pipe there.
        """
        self.write(payload)
        while True:
            reply = self.read_reply(timeout)
            if reply is None:
                raise RuntimeError(
                    f"rank 0 sent nothing for {timeout:.0f}s; the collective is "
                    "most likely blocked on a rank that died"
                )
            yield reply
            if reply.get("done") or not reply.get("ok", False):
                return

    def write(self, payload: dict) -> None:
        """Write one command line to rank 0's stdin.

        Serialised by the same lock every writer uses, so two concurrent
        submissions (D5's multiplexing client) never interleave partial JSON
        lines on the wire.
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        proc = leader.process
        assert proc.stdin is not None
        with self._stdin_lock:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

    def read_reply(self, timeout: float | None = None) -> dict | None:
        """Block for the next reply frame from rank 0's single reply pipe.

        Returns ``None`` on an idle timeout (the caller decides what that
        means — S2's :meth:`stream` treats it as fatal; D5's demux reader
        just loops). Raises ``RuntimeError`` once the pipe is closed (EOF) —
        there is exactly one rank-0 reply pipe per formation, so this is the
        only read path every consumer, single- or multi-request, shares.
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        assert leader.process.stdout is not None
        reader = self._reply_reader(leader)
        line = reader.readline(timeout)
        if line is None:
            return None
        if not line:
            raise RuntimeError("rank 0 closed its reply channel")
        frame: dict = json.loads(line)
        return frame

    def abort(self, request_id: str = "") -> bool:
        """Ask a running generation to stop, out of band.

        Names the request; an empty id aborts everything. The worker drains
        this pipe between steps and the eviction rides the broadcast, so every
        rank drops the same sequence on the same step.
        """
        leader = self.leader
        if leader is None or leader.control_w is None or not leader.alive:
            return False
        payload = json.dumps({"op": "abort", "request_id": request_id})
        try:
            os.write(leader.control_w, payload.encode() + b"\n")
        except OSError:
            logger.warning("cluster: abort signal could not be delivered")
            return False
        return True

    # -- teardown ----------------------------------------------------------

    def alive_ranks(self) -> list[int]:
        return [r.rank for r in self.ranks if r.alive]

    def kill(self) -> None:
        """Hard-kill every local rank, no shutdown handshake.

        Killing rank 0 closes the reply pipe, which is what unblocks a caller
        waiting on a dead cluster.
        """
        if self._deathwatch is not None:
            self._deathwatch.stop()
            self._deathwatch = None
        for entry in self.ranks:
            if entry.alive:
                entry.process.kill()
        _release_formation(self)
        _release_transfer_session(self)

    def stop(self, *, timeout: float = 10.0) -> None:
        """Shut every local rank down, politely then not.

        A rank blocked in a collective waiting for a peer that will never
        arrive does not respond to its shutdown command, so escalation to kill
        is the normal path during a failure teardown, not an edge case.
        """
        if self._deathwatch is not None:
            self._deathwatch.stop()
            self._deathwatch = None
        for entry in self.ranks:
            if entry.alive and entry.process.stdin is not None:
                try:
                    entry.process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    entry.process.stdin.flush()
                except (BrokenPipeError, ValueError):
                    pass

        for entry in self.ranks:
            try:
                entry.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("cluster: rank %d ignored shutdown, killing", entry.rank)
                entry.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    entry.process.wait(timeout=timeout)
            finally:
                if entry.control_w is not None:
                    with contextlib.suppress(OSError):
                        os.close(entry.control_w)
                    entry.control_w = None

        self.ranks.clear()
        _release_formation(self)
        _release_transfer_session(self)


# -- S5 R3c: spawning one node's rank of a transfer-session round ------------


def launch_transfer_session(
    *,
    rank: int,
    world_size: int,
    ips: list[str],
    base_port: int,
    argv_builder: Callable[[int], list[str]],
    data_plane_subnet: str | None,
    allow_routable_data_plane: bool = False,
    allow_loopback: bool = False,
    python: str | None = None,
) -> LocalCluster:
    """Spawn this node's rank of one transfer-session round (D1/D2/R3c).

    Reuses :class:`LocalCluster`'s hostfile/link-scope/join-timeout/
    deathwatch machinery with a parameterized entry module -- ``transfer_rank``,
    never ``rank_worker`` -- so a transfer session gets the same salvage-pitfall
    protections (rank 0 must listen before a peer connects, a dead rank is
    noticed in seconds) without duplicating them. Always ring (D1: no RDMA
    need for a transfer).

    Transfer sessions claim the DEDICATED slot from
    :func:`_register_transfer_session`, distinct from the formation slot: a
    live formation and one transfer session coexist on the same machine; a
    second concurrent transfer session raises :class:`TransferSpawnBoundError`.
    The caller is responsible for releasing the session once its local
    process has exited or been torn down (``cluster.stop()``/``cluster.kill()``
    release both slots unconditionally, so this need not be called twice).
    """
    cluster = LocalCluster(
        model="",
        world_size=world_size,
        backend="ring",
        base_port=base_port,
        python=python or sys.executable,
        module=TRANSFER_MODULE,
        argv_builder=argv_builder,
        # This function owns its own (distinct) slot -- never the formation's.
        enforce_spawn_bound=False,
    )
    _register_transfer_session(cluster)
    try:
        cluster.start(
            [rank],
            ips=ips,
            data_plane_subnet=data_plane_subnet,
            allow_routable_data_plane=allow_routable_data_plane,
            allow_loopback=allow_loopback,
        )
    except BaseException:
        _release_transfer_session(cluster)
        raise
    return cluster
