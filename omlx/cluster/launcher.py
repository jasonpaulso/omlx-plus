# SPDX-License-Identifier: Apache-2.0
"""Spawning and supervising rank processes from the daemon.

The leader daemon owns its local rank-0 worker and talks to it over a pipe.
Peer daemons own their own workers and are told what to spawn over the control
plane. Nothing here uses SSH, and nothing assumes the nodes share a filesystem
or have oMLX installed at the same path.

Ordering matters at startup. Every rank blocks inside `mx.distributed.init()`
until the whole world has arrived, so a worker that is slow to start holds up
all of them, and a worker that never starts hangs the rest until they are
killed. `LocalCluster.start` therefore treats "not everyone joined in time" as
a normal outcome with a clean teardown, not as an exception to bubble up.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from omlx.cluster import hostfile

logger = logging.getLogger(__name__)

# init() blocks until the whole world joins; past this we assume someone is
# never arriving and tear the run down rather than hanging a request forever.
DEFAULT_JOIN_TIMEOUT_S = 120

# How often a DeathWatch looks, and how many unreachable polls in a row it
# tolerates before declaring a node gone. A definitive answer ("your rank is
# not running") fires immediately; unreachability gets patience because the
# control plane shares the LAN with everything else on the machine, and a
# wifi blink must not kill a healthy formation.
DEATHWATCH_INTERVAL_S = 2.0
DEATHWATCH_STRIKES = 5


class DeathWatch(threading.Thread):
    """Notices a dead rank in seconds, instead of at the idle timeout.

    Without this, a rank that dies leaves every other rank blocked inside a
    collective - mlx has no fault tolerance - and the only thing that ends the
    wait is the generate idle timeout (minutes) or the load timeout (longer).
    The watch polls liveness and fires `on_death` once, at which point the
    owner kills its local ranks; killing them closes the reply pipe, which is
    what turns "blocked for ten minutes" into "failed in seconds".

    Each check returns True (alive), False (definitively dead - fires at
    once), or None (could not tell - counts a strike, fires after
    `strikes` consecutive misses). Liveness is read from process tables and
    daemon HTTP, never by connecting to a collective's port: a TCP probe of a
    ring port is not passive, it poisons the handshake.
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
        # The liveness poll is an HTTP request every couple of seconds for as
        # long as a cluster is formed; httpx logging each one at INFO turns
        # the daemon log into a heartbeat monitor.
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


@dataclass
class RankProcess:
    """A spawned rank and the pipe to it, if it is rank 0."""

    rank: int
    process: subprocess.Popen
    node_id: str = "local"
    # Write end of the out-of-band signal pipe. Only rank 0 gets one; peers
    # learn about an abort through the collective, from rank 0.
    control_w: int | None = None
    # Reply reader, created on first use. Rank 0 only.
    replies: "ReplyReader | None" = None

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


def resolve_python(candidate: str | None = None) -> str:
    """Pick an interpreter that can actually `import omlx`.

    A peer daemon spawns ranks from whatever cwd and environment launchd gave
    it, and an interpreter that cannot see oMLX fails in the least obvious way
    available: the rank dies immediately with `ModuleNotFoundError`, nobody is
    listening on its ring port, and *the other machine* reports
    `[ring] Couldn't connect (error: 65)` - which reads as a firewall or
    routing fault and is not one. Resolving this before spawning turns a
    cross-machine mystery into a local error message.
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


WORKER_MODULE = "omlx.cluster.worker"


class ReplyReader:
    """Reads rank 0's newline-delimited replies with an idle timeout.

    Owns its buffering rather than using the `Popen` text wrapper, because the
    timeout has to be honest. `select` reports on the *file descriptor*, and a
    buffered reader routinely pulls several replies out of the pipe in one go -
    so a wrapper-based implementation would sit in `select` waiting for bytes
    that had already arrived and time out with the answer in its hand.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""

    def readline(self, timeout: float | None) -> str | None:
        """The next reply line, `""` at EOF, or `None` when it timed out."""
        import select

        deadline = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self._buffer:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self._fd], [], [], remaining)[0]:
                    return None
            chunk = os.read(self._fd, 65536)
            if not chunk:
                return ""
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace")


def sweep_orphaned_ranks() -> int:
    """Kill rank processes on this machine that no live daemon is driving.

    A worker left alive by a failed run holds its ring port, and the next
    rank 0 then quietly fails to own it - the whole cluster dies with a connect
    timeout that looks like a network fault. Every teardown path escalates to
    kill for this reason, but a daemon that was itself restarted has no handle
    on the children it left behind, and only the process table remembers them.

    **Our own children are never swept.** Two daemons on one machine is exactly
    how the stack is tested without a second Mac, and a peer forming its rank
    must not kill the leader's rank 0 sitting next to it. A rank whose parent
    is still alive belongs to somebody; only the reparented ones are orphans.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        return 0

    ours = os.getpid()
    killed = 0
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if WORKER_MODULE not in cmdline:
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


def _parent_is_alive(pid: int | None) -> bool:
    """True when `pid` is a live process other than init.

    A rank reparented to launchd (ppid 1) lost its daemon and is an orphan by
    definition; anything else still has somebody responsible for it.
    """
    if pid is None or pid <= 1:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001
        return False


@dataclass
class LocalCluster:
    """Rank processes this daemon is responsible for.

    Usually one - a node runs a single rank. Several ranks on one machine is
    supported because it is how the whole stack is tested without a second Mac.
    """

    model_path: str
    world_size: int
    backend: str = "ring"
    pipeline: bool = False
    seed: int = 0
    python: str = field(default_factory=lambda: sys.executable)
    ranks: list[RankProcess] = field(default_factory=list)
    _workdir: Path | None = None
    # Several requests write commands down the same stdin now that the worker
    # batches; interleaved partial lines would be parsed as garbage.
    _stdin_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(
        self,
        ranks: list[int],
        *,
        ips: list[str] | None = None,
        coordinator: str | None = None,
        ibv_devices: list[list[str | None]] | None = None,
    ) -> None:
        """Spawn the given ranks on this machine.

        `ips` is the whole cluster's address list in rank order - every node
        needs it, not just its own entry, because the ring hostfile describes
        all ranks.
        """
        sweep_orphaned_ranks()
        self._workdir = Path(tempfile.mkdtemp(prefix="omlx-cluster-"))

        launch_kwargs: dict = {}
        if self.backend == "ring":
            addresses = hostfile.ring_addresses(ips or ["127.0.0.1"] * self.world_size)
            launch_kwargs["hostfile"] = hostfile.write_ring_hostfile(
                self._workdir / "hosts.json", addresses
            )
        else:
            if coordinator is None or ibv_devices is None:
                raise ValueError(
                    f"backend {self.backend} needs a coordinator and ibv matrix"
                )
            launch_kwargs["coordinator"] = coordinator
            launch_kwargs["ibv_devices"] = hostfile.write_ibv_devices(
                self._workdir / "ibv.json", ibv_devices
            )

        for rank in ranks:
            spec = hostfile.build(
                backend=self.backend,
                rank=rank,
                world_size=self.world_size,
                **launch_kwargs,
            )
            argv = [
                self.python,
                "-m",
                "omlx.cluster.worker",
                "--model",
                self.model_path,
                "--seed",
                str(self.seed),
            ]
            if self.pipeline:
                argv.append("--pipeline")

            # Rank 0 gets a second, out-of-band pipe. Its stdin is occupied for
            # the whole of a generation - it is inside the decode loop and
            # cannot go back and read another command - so an abort that has to
            # reach a *running* request needs somewhere else to arrive.
            control_r = control_w = None
            if rank == 0:
                control_r, control_w = os.pipe()
                os.set_inheritable(control_r, True)
                argv += ["--control-fd", str(control_r)]

            try:
                process = subprocess.Popen(
                    argv,
                    env=hostfile.scrubbed_parent_env() | spec.env,
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
                # The child owns the read end now; keeping it open here would
                # mean the worker never sees the pipe close when we go away.
                if control_r is not None:
                    os.close(control_r)

            self.ranks.append(
                RankProcess(rank=rank, process=process, control_w=control_w)
            )
            logger.info("cluster: spawned rank %d (pid %d)", rank, process.pid)

    def wait_until_ready(self, port: int, *, timeout: float = 60.0) -> bool:
        """Block until it is safe to start peer ranks on other machines.

        Peers must not be told to start before this returns. The ring backend
        gives a connecting peer a bounded retry window; a peer that starts
        first burns it against a socket that does not exist yet and dies with
        `[ring] Couldn't connect (error: 65)`, which reads like a firewall or
        routing fault and is not one.

        **Readiness is read from the process table, never by connecting.** A
        TCP probe of the ring port is not a passive observation: the backend
        accepts it, takes it for the peer it is waiting on, and the handshake
        is then poisoned for the real peer - both ranks sit there until they
        time out with error 60, on a network where nothing is wrong. That cost
        an afternoon; do not reintroduce a connect here.

        Nothing observable is not treated as failure - the ranks may already be
        past init - so a short grace period lets the caller proceed. The only
        real failure is a rank that died, which is checked directly.
        """
        grace_deadline = time.monotonic() + min(timeout, 5.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not all(r.alive for r in self.ranks):
                logger.error("cluster: a rank exited before the world formed")
                return False
            if self._is_listening(port):
                return True
            if time.monotonic() > grace_deadline:
                # Never saw the socket, but nothing has died: the ranks are
                # most likely already past init because they are colocated.
                return True
            time.sleep(0.25)
        return all(r.alive for r in self.ranks)

    def _is_listening(self, port: int) -> bool:
        """True when one of this node's ranks holds a listening socket on `port`.

        Read-only by construction. The interpreter that ends up listening is
        not always the direct child - a launcher shim re-execs - so the whole
        subtree is inspected.
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

    @property
    def leader(self) -> RankProcess | None:
        """The rank-0 process, if this machine owns it."""
        return next((r for r in self.ranks if r.rank == 0), None)

    def command(self, payload: dict, *, timeout: float | None = None) -> dict:
        """Send one command to rank 0 and read one reply.

        Only meaningful on the node that owns rank 0; peer daemons drive their
        workers implicitly, because rank 0 broadcasts every command over the
        collective.
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        return next(self.stream(payload, timeout=timeout))

    def submit(self, payload: dict) -> None:
        """Write one command down rank 0's stdin, without reading a reply.

        The serving path: replies come back tagged with the request id they
        belong to and are routed by the manager's reply router, so writers
        never touch the read side.
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        proc = leader.process
        assert proc.stdin is not None
        with self._stdin_lock:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

    def reply_reader(self) -> "ReplyReader":
        """Rank 0's reply stream, created on first use and then shared.

        Shared deliberately: the reader owns buffered bytes, and a second
        reader on the same descriptor would split replies between the two.
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        assert leader.process.stdout is not None
        if leader.replies is None:
            leader.replies = ReplyReader(leader.process.stdout.fileno())
        return leader.replies

    def stream(self, payload: dict, *, timeout: float | None = None):
        """Send one command and yield replies until `done`.

        `timeout` is an *idle* timeout - the longest this will wait for the
        next reply, not for the whole command. It matters because the failure
        it catches is not rare: a rank that dies mid-collective leaves rank 0
        blocked inside mlx with no way to notice, and without this the HTTP
        request hangs until the client gives up. JACCL has no fault tolerance
        by design, so the timeout is the only thing standing between a dead
        peer and a wedged daemon.
        """
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        proc = leader.process
        assert proc.stdin is not None and proc.stdout is not None

        with self._stdin_lock:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

        if leader.replies is None:
            leader.replies = ReplyReader(proc.stdout.fileno())

        while True:
            line = leader.replies.readline(timeout)
            if line is None:
                raise RuntimeError(
                    f"rank 0 sent nothing for {timeout:.0f}s; the collective is "
                    "most likely blocked on a rank that died"
                )
            if not line:
                raise RuntimeError("rank 0 closed its reply channel")
            reply = json.loads(line)
            yield reply
            if reply.get("done") or not reply.get("ok", False):
                return

    def alive_ranks(self) -> list[int]:
        """The ranks on this machine whose processes are still running."""
        return [r.rank for r in self.ranks if r.alive]

    def kill(self) -> None:
        """Hard-kill every local rank, no shutdown handshake.

        The fast-fail path. A rank blocked in a collective ignores its
        shutdown command by definition - it is not reading commands - and the
        polite escalation in `stop()` would spend its whole timeout learning
        that. Killing rank 0 closes the reply pipe, which is what unblocks a
        request waiting on a dead cluster.
        """
        for entry in self.ranks:
            if entry.alive:
                entry.process.kill()

    def abort(self, request_id: str = "") -> bool:
        """Ask a running generation to stop, out of band.

        Names the request, so aborting one stream leaves the rest of the
        batch running; an empty id aborts everything. Returns False when the
        signal could not be delivered. The worker drains this pipe between
        steps and the eviction rides the event broadcast, so every rank drops
        the same sequence on the same step; nothing here reaches the peers
        directly.
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

    def stop(self, *, timeout: float = 10.0) -> None:
        """Shut every local rank down, politely then not.

        A rank blocked in a collective waiting for a peer that will never
        arrive does not respond to its shutdown command, so the escalation to
        kill is the normal path during a failure teardown, not an edge case.
        """
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
                entry.process.wait(timeout=timeout)
            finally:
                if entry.control_w is not None:
                    try:
                        os.close(entry.control_w)
                    except OSError:
                        pass
                    entry.control_w = None

        self.ranks.clear()
