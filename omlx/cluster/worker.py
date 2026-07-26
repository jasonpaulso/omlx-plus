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
produce garbage. That rule is the whole correctness argument, and it is why
`_advance` takes its token count from `agree_int` rather than from `len()` of
anything local.

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
import sys
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Control messages. Rank 0 receives these from its daemon and broadcasts them
# verbatim, so every rank runs the same branch of the same loop.
CMD_LOAD = "load"
CMD_GENERATE = "generate"
CMD_PING = "ping"
CMD_SHUTDOWN = "shutdown"


@dataclass
class WorkerConfig:
    """Everything the worker needs that is not in the mlx environment."""

    model_path: str
    pipeline: bool = False
    seed: int = 0


class Worker:
    """One rank. Owns the session, the model, and the lockstep loop."""

    def __init__(self, config: WorkerConfig) -> None:
        from omlx.cluster.mlx_adapter import DistributedSession

        self.config = config
        self.session = DistributedSession()
        self.world = self.session.world
        self.model: Any = None
        self.tokenizer: Any = None

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

    def generate(self, prompt: str, max_tokens: int) -> Iterator[str]:
        """Greedy decode in lockstep, yielding text on rank 0.

        Followers run the identical loop and discard their output. They have to
        run it: under tensor parallelism each rank owns a slice of every
        attention head, so a forward pass only completes if all of them step
        together.

        The loop deliberately samples on every rank rather than sampling on
        rank 0 and broadcasting the token. Logits are already all-reduced, the
        RNG is already synchronised, so local sampling is both correct and one
        collective cheaper per token.
        """
        from mlx_lm.models.cache import make_prompt_cache

        import mlx.core as mx

        if self.model is None:
            raise RuntimeError("generate called before load")

        # The prompt must come from the broadcast command, never from a
        # rank-local source, or the ranks tokenize different text.
        ids = mx.array(self.tokenizer.encode(prompt))[None]
        cache = make_prompt_cache(self.model)

        y = ids
        for _ in range(max_tokens):
            logits = self.model(y, cache=cache)[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
            token_id = int(token.item())

            # Rank 0 owns the stop decision so a divergent local detokenizer
            # cannot end the run on one rank only.
            stop = token_id in self._eos_ids()
            if self.session.agree_int(1 if stop else 0):
                break

            if self.world.is_leader:
                yield self.tokenizer.decode([token_id])
            y = token[None]

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
            for chunk in self.generate(
                command["prompt"], int(command.get("max_tokens", 128))
            ):
                self._reply({"ok": True, "chunk": chunk}, flush=True)
            self._reply({"ok": True, "done": True})
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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # Logs go to stderr; stdout is the reply channel and must stay clean JSON.
    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    worker = Worker(
        WorkerConfig(model_path=args.model, pipeline=args.pipeline, seed=args.seed)
    )
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
