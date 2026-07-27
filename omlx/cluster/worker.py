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

That rule is the whole correctness argument, and this loop obeys it in three
places worth naming:

- the prompt arrives as **token ids** in the broadcast command, so no rank
  tokenizes anything;
- sampling is deliberately *not* broadcast. Logits are already all-reduced
  under tensor parallelism and `seed_everyone` synchronises the RNG, so every
  rank draws the same token independently - one collective cheaper per token;
- **stopping** is a rank-0 decision (`agree_int`), because detokenized text and
  the abort pipe are things only rank 0 can see.

One request at a time
---------------------
There is no batching here. Continuous batching means the scheduler decides,
every step, which sequences advance - and under tensor parallelism that
decision must be identical on every rank or the forward passes do not line up.
That needs `omlx/scheduler.py` running inside every worker with its
memory-derived admission gates made cluster-aware; see
`docs/cluster-scheduler-divergence-audit.md`. Until then a cluster serves
requests serially, which is the honest shape for a model that only exists
because it did not fit on one machine.

Failure
-------
JACCL has no fault tolerance: a dead rank leaves its peers blocked in a
collective until the Metal timeout kills them too. There is no in-process
recovery here on purpose. A worker that loses its daemon exits, the daemon that
loses a worker tears the session down and respawns it, and local models carry
on serving throughout.
"""

from __future__ import annotations

import json
import logging
import os
import select
import sys
from dataclasses import dataclass
from typing import Any, Iterator

from omlx.cluster.protocol import (
    CMD_GENERATE,
    CMD_LOAD,
    CMD_PING,
    CMD_SHUTDOWN,
    FINISH_REASON,
    SIGNAL_ABORT,
    STEP_ABORT,
    STEP_CONTINUE,
    STEP_EOS,
    STEP_STOP_TEXT,
    GenerationSpec,
    StopTextBuffer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CMD_GENERATE",
    "CMD_LOAD",
    "CMD_PING",
    "CMD_SHUTDOWN",
    "AbortChannel",
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


class AbortChannel:
    """Rank 0's out-of-band read side, polled between tokens.

    Non-blocking by construction: a decode loop that blocked here would stall
    every other rank waiting in the next collective. A closed pipe means the
    daemon is gone, which is treated as an abort - there is nobody left to
    stream to.
    """

    def __init__(self, fd: int | None) -> None:
        self._fd = fd
        self._buffer = b""
        self._closed = False
        self._aborted = False

    def poll(self) -> bool:
        """True once an abort has been signalled (or the daemon vanished).

        Latching, not edge-triggered: the loop polls once per token and the
        answer must not depend on which poll happened to read the bytes.
        """
        if self._fd is None:
            return False
        if self._aborted or self._closed:
            return True

        while select.select([self._fd], [], [], 0)[0]:
            try:
                chunk = os.read(self._fd, 4096)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                self._closed = True
                return True
            if not chunk:
                self._closed = True
                return True
            self._buffer += chunk

        if b"\n" not in self._buffer:
            return False

        lines, _, self._buffer = self._buffer.rpartition(b"\n")
        for line in lines.split(b"\n"):
            if not line.strip():
                continue
            try:
                if json.loads(line).get("op") == SIGNAL_ABORT:
                    self._aborted = True
                    return True
            except (ValueError, AttributeError):
                logger.warning("cluster: unparseable control signal, ignoring")
        return False

    def drain(self) -> None:
        """Discard anything buffered, so a stale abort cannot end the next run."""
        self._buffer = b""
        self._aborted = False
        if self._fd is None or self._closed:
            return
        while select.select([self._fd], [], [], 0)[0]:
            try:
                if not os.read(self._fd, 4096):
                    self._closed = True
                    return
            except OSError:
                self._closed = True
                return


class Worker:
    """One rank. Owns the session, the model, and the lockstep loop."""

    def __init__(self, config: WorkerConfig) -> None:
        from omlx.cluster.mlx_adapter import DistributedSession

        self.config = config
        self.session = DistributedSession()
        self.world = self.session.world
        self.model: Any = None
        self.tokenizer: Any = None
        self.abort = AbortChannel(config.control_fd if self.world.is_leader else None)

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load this rank's shard and agree that every rank managed it."""
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
        return {
            "rank": self.world.rank,
            "world_size": self.world.size,
            "tensor_parallel": tensor_ok,
            "pipeline_parallel": pipeline_ok,
        }

    # -- generation --------------------------------------------------------

    def generate(self, spec: GenerationSpec) -> Iterator[dict[str, Any]]:
        """Decode in lockstep, yielding streaming payloads on rank 0.

        Followers run the identical loop and discard their output. They have to
        run it: under tensor parallelism each rank owns a slice of every
        attention head, so a forward pass only completes if all of them step
        together. That is also why the loop must not `break` early on one rank
        - `_step_verdict` is the single place a decision to stop is agreed.
        """
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        if self.model is None:
            raise RuntimeError("generate called before load")

        # A seed pinned by the request beats the launch seed, but every rank
        # has to take the same one, so it goes through the collective.
        if spec.seed is not None:
            self.session.seed_everyone(int(spec.seed))

        self.abort.drain()

        if not spec.prompt_ids:
            raise ValueError("generate needs at least one prompt token")

        detokenizer = self._detokenizer()
        buffer = StopTextBuffer(spec.stop)
        stop_ids = set(spec.stop_token_ids) | self._eos_ids()
        leader = self.world.is_leader

        generated = 0
        verdict = STEP_CONTINUE
        steps = generate_step(
            mx.array(spec.prompt_ids),
            self.model,
            max_tokens=spec.max_tokens,
            sampler=self._sampler(spec),
            logits_processors=self._logits_processors(spec),
        )

        for token, _ in steps:
            token_id = int(token.item())
            text = ""
            local = STEP_CONTINUE

            if leader:
                # Order matters. A stop token is never detokenized, or its
                # text would land in the output it is supposed to end.
                if self.abort.poll():
                    local = STEP_ABORT
                elif token_id in stop_ids:
                    local = STEP_EOS
                else:
                    detokenizer.add_token(token_id)
                    text = buffer.push(detokenizer.last_segment)
                    if buffer.hit is not None:
                        local = STEP_STOP_TEXT

            verdict = self.session.agree_int(local)

            # A stop token and an aborted step produce no output at all, so
            # they are not counted either. A stop *string* was produced by a
            # real token whose text the buffer truncated at the match.
            if verdict in (STEP_EOS, STEP_ABORT):
                break

            generated += 1
            if text:
                yield {"chunk": text, "tokens": generated}
            if verdict == STEP_STOP_TEXT:
                break

        if leader:
            detokenizer.finalize()
            tail = buffer.flush() if verdict == STEP_CONTINUE else ""
            if tail:
                yield {"chunk": tail, "tokens": generated}
            yield {
                "done": True,
                "text": buffer.text,
                "prompt_tokens": len(spec.prompt_ids),
                "completion_tokens": generated,
                "finish_reason": FINISH_REASON.get(verdict, "length"),
            }

    def _sampler(self, spec: GenerationSpec):
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=spec.temperature,
            top_p=spec.top_p,
            min_p=spec.min_p,
            top_k=spec.top_k,
        )

    def _logits_processors(self, spec: GenerationSpec):
        from mlx_lm.sample_utils import make_logits_processors

        processors = make_logits_processors(
            repetition_penalty=spec.repetition_penalty,
            repetition_context_size=spec.repetition_context_size,
            presence_penalty=spec.presence_penalty,
            frequency_penalty=spec.frequency_penalty,
        )
        return processors or None

    def _detokenizer(self):
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        return detokenizer

    def _eos_ids(self) -> set[int]:
        eos = getattr(self.tokenizer, "eos_token_ids", None)
        if eos:
            return set(eos)
        single = getattr(self.tokenizer, "eos_token_id", None)
        return {single} if single is not None else set()

    # -- control loop ------------------------------------------------------

    def run(self, control: Any = None) -> None:
        """Serve commands until told to stop.

        Rank 0 reads newline-delimited JSON from `control` (its daemon's pipe);
        every other rank blocks in `broadcast` waiting to be told what rank 0
        read. Both paths reach the same dispatch with the same command.
        """
        control = control if control is not None else sys.stdin

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
            self._reply({"ok": True, **self.load()})
        elif op == CMD_GENERATE:
            for payload in self.generate(GenerationSpec.from_dict(command)):
                self._reply({"ok": True, **payload})
        else:
            self._reply({"ok": False, "error": f"unknown op {op!r}"})

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
