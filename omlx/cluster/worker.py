# SPDX-License-Identifier: Apache-2.0
"""The rank process: one per machine, joined to the collective.

Why the oMLX daemon does not join the collective itself
-------------------------------------------------------
It would be simpler for the leader's API process to be rank 0 directly. It is
also wrong. A distributed session cannot be torn down and re-created inside a
process - repeated init/teardown exhausts kernel protection domains and the
only recovery is a reboot. If the API process held the session then swapping
the distributed model would mean restarting the whole server, which would also
evict every *local* model the node is serving alongside the cluster.

So every rank, including rank 0, is a child process. The daemon keeps its
scheduler, its admin UI, its local EnginePool and its uptime; a model swap
kills these children and spawns new ones. Respawn is cheap by design, because
the alternative is not allowed to exist.

Shape of a run
--------------
    leader daemon                        peer daemon
         |  json lines over a pipe            |
      rank 0 worker  <==== mlx collective ====>  rank 1 worker

Rank 0 is the only rank that talks to a daemon. Every decision it makes is
broadcast over the collective, and all ranks execute the same decision in
lockstep. No rank may ever branch on something only it can see - local free
memory, a local cache hit, wall-clock time - or the ranks silently diverge and
produce garbage.

Serving is continuous batching, run in lockstep: see
`omlx/cluster/batching.py` for the loop and its correctness argument. The
worker's own job is the control plane around it - reading commands without
double-buffering them, keeping the out-of-band signal pipe drained, and owning
the model shard.

Failure
-------
JACCL has no fault tolerance: a dead rank leaves its peers blocked in a
collective until the daemons' deathwatch kills them. There is no in-process
recovery here on purpose. A worker that loses its daemon exits, the daemon
that loses a worker tears the session down and respawns it, and local models
carry on serving throughout.
"""

from __future__ import annotations

import json
import logging
import os
import select
import sys
from dataclasses import dataclass
from typing import Any

from omlx.cluster.batching import BatchConfig, BatchLoop
from omlx.cluster.protocol import (
    CMD_GENERATE,
    CMD_LOAD,
    CMD_PING,
    CMD_SHUTDOWN,
    SIGNAL_ABORT,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CMD_GENERATE",
    "CMD_LOAD",
    "CMD_PING",
    "CMD_SHUTDOWN",
    "CommandReader",
    "ControlChannel",
    "Worker",
    "WorkerConfig",
    "main",
]


@dataclass
class WorkerConfig:
    """Everything the worker needs that is not in the mlx environment."""

    model_path: str
    pipeline: bool = False
    seed: int = 0
    control_fd: int | None = None


class CommandReader:
    """Rank 0's command channel, with the buffering owned here.

    The batch loop needs two read shapes from the same pipe: block until the
    next command while idle, and drain whatever has arrived between steps
    while serving. A buffered file object cannot provide both - its
    `readline` reads ahead, and a later `select` on the descriptor then
    reports an empty pipe while commands sit in the Python buffer. Commands
    stranded like that would admit a request only when the *next* event
    happened to arrive.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""
        self._eof = False

    def readline(self) -> str:
        """The next command line, blocking. `""` once the pipe is closed."""
        while b"\n" not in self._buffer:
            if self._eof:
                return ""
            select.select([self._fd], [], [])
            if not self._fill():
                return ""
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace")

    def drain_lines(self) -> list[str]:
        """Every complete line that has already arrived, without blocking."""
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
    """Rank 0's out-of-band read side, drained between batch steps.

    Signals name the request they concern, so a late abort for a request that
    already finished is a no-op by construction rather than a hazard to be
    drained away. A closed pipe means the daemon is gone, which is reported as
    an abort of everything - there is nobody left to stream to.
    """

    def __init__(self, fd: int | None) -> None:
        self._fd = fd
        self._buffer = b""
        self._closed = False

    def take_events(self) -> list[dict[str, Any]]:
        """Parsed signals that have arrived since the last call."""
        if self._fd is None:
            return []
        if self._closed:
            return [{"op": SIGNAL_ABORT}]

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
            if isinstance(event, dict) and event.get("op") == SIGNAL_ABORT:
                events.append(event)

        if self._closed:
            events.append({"op": SIGNAL_ABORT})
        return events


class Worker:
    """One rank. Owns the session, the model shard, and the batch loop."""

    def __init__(self, config: WorkerConfig) -> None:
        from omlx.cluster.mlx_adapter import DistributedSession

        self.config = config
        self.session = DistributedSession()
        self.world = self.session.world
        self.model: Any = None
        self.tokenizer: Any = None
        self.batch: BatchLoop | None = None
        self.signals = ControlChannel(
            config.control_fd if self.world.is_leader else None
        )
        self._commands: CommandReader | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load this rank's shard and agree that every rank managed it.

        The `load` command also carries the batching configuration, so every
        rank builds its generator from the leader's settings rather than its
        own - two nodes configured differently would admit differently.
        """
        from omlx.cluster.mlx_adapter import load_sharded, parallelism_support

        model, tokenizer = load_sharded(
            self.config.model_path, self.session, pipeline=self.config.pipeline
        )
        self.model, self.tokenizer = model, tokenizer

        tensor_ok, pipeline_ok = parallelism_support(model)
        wanted = pipeline_ok if self.config.pipeline else tensor_ok

        # Every rank must have succeeded. A partial load is worse than none:
        # the ranks that did load would block forever waiting for the one that
        # did not.
        if not self.session.all_agree(wanted):
            raise RuntimeError(
                f"{self.config.model_path} does not support "
                f"{'pipeline' if self.config.pipeline else 'tensor'} parallelism "
                "on every rank"
            )

        self.session.seed_everyone(self.config.seed)
        self.batch = BatchLoop(
            self.session,
            model,
            tokenizer,
            BatchConfig.from_command(command or {}),
            reply=self._reply,
            gather_events=self._gather_events,
        )
        return {
            "rank": self.world.rank,
            "world_size": self.world.size,
            "tensor_parallel": tensor_ok,
            "pipeline_parallel": pipeline_ok,
        }

    # -- control loop ------------------------------------------------------

    def run(self, control: Any = None) -> None:
        """Serve commands until told to stop.

        Rank 0 reads newline-delimited JSON from `control` (its daemon's
        pipe); every other rank blocks in `broadcast` waiting to be told what
        rank 0 read. Both paths reach the same dispatch with the same command.
        A `generate` hands the loop over to the batch loop, which keeps
        serving - and keeps reading commands - until the batch drains.
        """
        if control is None:
            control = CommandReader(sys.stdin.fileno())
        self._commands = control

        while True:
            if self.world.is_leader:
                line = control.readline()
                command = json.loads(line) if line else {"op": CMD_SHUTDOWN}
            else:
                command = None

            command = self.session.broadcast(command)
            if command is None or command.get("op") == CMD_SHUTDOWN:
                logger.info("cluster: rank %d shutting down", self.world.rank)
                return

            if command.get("op") == CMD_GENERATE:
                if self.batch is None:
                    self._reply(
                        {
                            "ok": False,
                            "request_id": str(command.get("request_id") or ""),
                            "error": "generate called before load",
                        }
                    )
                    continue
                if self.batch.serve(command):
                    logger.info(
                        "cluster: rank %d shutting down", self.world.rank
                    )
                    return
                continue

            try:
                self._dispatch(command)
            except Exception as exc:  # noqa: BLE001 - report, do not die silently
                logger.exception("cluster: rank %d failed a command", self.world.rank)
                self._reply({"ok": False, "error": str(exc)})

    def _dispatch(self, command: dict[str, Any]) -> None:
        op = command.get("op")
        if op == CMD_PING:
            self._reply({"ok": True, "rank": self.world.rank})
        elif op == CMD_LOAD:
            self._reply({"ok": True, **self.load(command)})
        else:
            self._reply({"ok": False, "error": f"unknown op {op!r}"})

    def _gather_events(self) -> list[dict[str, Any]]:
        """Everything that arrived while the batch loop was stepping.

        Leader only, called between steps: commands from the daemon's pipe
        and abort signals from the out-of-band channel, in that order, each
        already in arrival order.
        """
        events: list[dict[str, Any]] = []
        if self._commands is not None:
            for line in self._commands.drain_lines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    logger.warning("cluster: unparseable command, ignoring")
        events += self.signals.take_events()
        return events

    def _reply(self, payload: dict[str, Any], *, flush: bool = True) -> None:
        """Only rank 0 has a daemon listening; the rest stay quiet."""
        if not self.world.is_leader:
            return
        sys.stdout.write(json.dumps(payload) + "\n")
        if flush:
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Entry point for a spawned rank process."""
    import argparse

    parser = argparse.ArgumentParser(prog="omlx-cluster-worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-fd",
        type=int,
        default=None,
        help="inherited read end of the daemon's out-of-band signal pipe",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # A wedged rank is opaque from outside (py-spy needs root); SIGUSR2 makes
    # it explain itself. Stacks go to stderr, which the daemon already owns.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR2, all_threads=True)

    # Logs go to stderr; stdout is the reply channel and must stay clean JSON.
    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    worker = Worker(
        WorkerConfig(
            model_path=args.model,
            pipeline=args.pipeline,
            seed=args.seed,
            control_fd=args.control_fd,
        )
    )
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
