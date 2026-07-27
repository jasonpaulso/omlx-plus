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
import time
from dataclasses import dataclass, field
from pathlib import Path

from omlx.cluster import hostfile

logger = logging.getLogger(__name__)

# init() blocks until the whole world joins; past this we assume someone is
# never arriving and tear the run down rather than hanging a request forever.
DEFAULT_JOIN_TIMEOUT_S = 120


@dataclass
class RankProcess:
    """A spawned rank and the pipe to it, if it is rank 0."""

    rank: int
    process: subprocess.Popen
    node_id: str = "local"
    # Write end of the out-of-band signal pipe. Only rank 0 gets one; peers
    # learn about an abort through the collective, from rank 0.
    control_w: int | None = None

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

    def wait_until_ready(
        self, port: int, host: str = "127.0.0.1", *, timeout: float = 60.0
    ) -> bool:
        """Block until it is safe to start peer ranks on other machines.

        Peers must not be told to start before this returns. The ring backend
        gives a connecting peer a bounded retry window; a peer that starts
        first burns it against a socket that does not exist yet and dies with
        `[ring] Couldn't connect (error: 65)`, which reads like a firewall or
        routing fault and is not one.

        The port being *closed* is ambiguous - it means either "not open yet"
        or "already connected and moved on", and those cannot be told apart
        from outside. So a closed port is not treated as failure. The only
        real failure is a rank that died, which is checked directly. This
        returns as soon as the socket appears, and otherwise waits out a short
        grace period before letting the caller proceed.
        """
        import socket

        grace_deadline = time.monotonic() + min(timeout, 5.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not all(r.alive for r in self.ranks):
                logger.error("cluster: a rank exited before the world formed")
                return False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(1.0)
                if probe.connect_ex((host, port)) == 0:
                    return True
            if time.monotonic() > grace_deadline:
                # Never saw the socket, but nothing has died: the ranks are
                # most likely already past init because they are colocated.
                return True
            time.sleep(0.25)
        return all(r.alive for r in self.ranks)

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

    def stream(self, payload: dict, *, timeout: float | None = None):
        """Send one command and yield replies until `done`."""
        leader = self.leader
        if leader is None:
            raise RuntimeError("this node does not own rank 0")
        proc = leader.process
        assert proc.stdin is not None and proc.stdout is not None

        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("rank 0 closed its reply channel")
            reply = json.loads(line)
            yield reply
            if reply.get("done") or not reply.get("ok", False):
                return

    def abort(self) -> bool:
        """Ask a running generation to stop, out of band.

        Returns False when there is nothing to abort. The worker polls this
        pipe between tokens and agrees the decision through the collective, so
        every rank leaves the decode loop on the same step; nothing here
        reaches the peers directly.
        """
        leader = self.leader
        if leader is None or leader.control_w is None or not leader.alive:
            return False
        try:
            os.write(leader.control_w, b'{"op": "abort"}\n')
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
