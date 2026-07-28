# SPDX-License-Identifier: Apache-2.0
"""The rank process: one per machine, joined to the tensor-parallel collective.

Why the oMLX daemon does not join the collective itself
-------------------------------------------------------
A distributed session cannot be torn down and re-created inside a process —
repeated init/teardown exhausts kernel protection domains and the only recovery
is a reboot (salvage pitfall 6). If the API process held the session, swapping
the distributed model would mean restarting the whole server and evicting every
*local* model the node serves alongside the cluster. So every rank, including
rank 0, is a child process; a model swap kills these children and spawns new
ones. One ``init()`` per process lifetime (S0 discipline).

Shape of a run
--------------
    head daemon                          worker daemon
        |  json lines over a pipe             |
     rank 0 process  <=== mlx collective ===>  rank 1 process

Rank 0 is the only rank that talks to a daemon. It samples, and every decision
it makes — the token, whether to stop, an abort — is broadcast to every rank in
one per-step message (D4). No rank ever branches on something only it can see,
or the ranks silently diverge and produce garbage.

The generation loop is oMLX's own, NOT mlx-lm's ``BatchGenerator``: every shape
of that path deadlocked a tensor-sharded ring in the prior attempt (salvage
pitfall 1). Prompts are prefilled by hand in chunks; decode runs one token at a
time with a single control collective per step.

Failure
-------
mlx has no fault tolerance: a dead rank leaves its peers blocked in a collective
until a daemon's deathwatch kills them. A rank that loses its daemon (reparented
to launchd) exits on its own so its peers fail fast rather than hang.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

from omlx.cluster.launcher import CommandReader, ControlChannel, DeathWatch
from omlx.cluster.protocol import (
    DELTA_ABORT,
    DELTA_FINISH,
    GenerationSpec,
    StepMessage,
    StopTextBuffer,
    chunk_frame,
    done_frame,
    error_frame,
)

logger = logging.getLogger(__name__)

# Hand-prefill chunk size (salvage-measured stable shape under TP collectives).
PREFILL_STEP = 2048


class DistributedSession:
    """A joined collective and the operations the rank loop performs on it.

    Constructing this calls ``mx.distributed.init(strict=True)``, which reads
    the environment prepared by the launcher. ``strict=True`` because the
    default returns a *singleton group* when no backend comes up, and every
    layer above would read that as success: rank 0 loads the whole model and
    the peer sits idle holding nothing. There is intentionally no ``close()``.
    """

    def __init__(self) -> None:
        import mlx.core as mx

        from omlx.cluster.hostfile import BACKEND_VAR

        backend = os.environ.get(BACKEND_VAR) or "any"
        self._group = mx.distributed.init(strict=True, backend=backend)
        self.rank = int(self._group.rank())
        self.size = int(self._group.size())
        self.is_leader = self.rank == 0
        self.barrier()
        logger.info("cluster: joined collective as rank %d of %d", self.rank, self.size)

    @property
    def group(self) -> Any:
        return self._group

    def barrier(self) -> None:
        """Synchronise every rank, forcing lazy RDMA setup on the first call."""
        import mlx.core as mx

        mx.eval(mx.distributed.all_sum(mx.ones(10), group=self._group, stream=mx.cpu))

    def seed_everyone(self, seed: int) -> int:
        """Agree on one RNG seed so any sampling is reproducible across ranks."""
        import mlx.core as mx

        chosen = mx.array([seed if self.is_leader else 0], dtype=mx.int64)
        chosen = mx.distributed.all_sum(chosen, group=self._group, stream=mx.cpu)
        mx.eval(chosen)
        agreed = int(chosen.item())
        mx.random.seed(agreed)
        return agreed

    def broadcast_json(self, obj: Any | None) -> Any:
        """Send a JSON-serialisable object from rank 0 to every rank.

        JSON bytes are shipped through two ``all_sum`` collectives (size then
        payload): ranks other than 0 contribute zeros, so the sum is rank 0's
        payload. NOT pickle — an on-link attacker who can inject into the ring
        (CL-09) would otherwise get arbitrary code execution on every rank
        (D4). Both collectives are pinned ``stream=mx.cpu``: ring AllReduce has
        no GPU implementation, and the cpu pin is also what lets an idle rank
        block here past Metal's ~5 s command-buffer timeout.

        Callers that interleave with model compute must drain the model's
        stream first (``mx.synchronize``); the backend requires every rank to
        hand it collectives in one global order.
        """
        import mlx.core as mx

        if self.size == 1:
            return obj

        if self.is_leader:
            payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
            size = mx.array([len(payload)], dtype=mx.int64)
        else:
            payload = b""
            size = mx.array([0], dtype=mx.int64)

        size = mx.distributed.all_sum(size, group=self._group, stream=mx.cpu)
        mx.eval(size)
        length = int(size.item())
        if length == 0:
            return None

        if self.is_leader:
            buf = mx.array(list(payload), dtype=mx.uint32)
        else:
            buf = mx.zeros(length, dtype=mx.uint32)
        buf = mx.distributed.all_sum(buf, group=self._group, stream=mx.cpu)
        mx.eval(buf)
        return json.loads(bytes(bytearray(buf.tolist())).decode("utf-8"))


def _tax_summary(samples: list[float]) -> dict[str, Any]:
    """Per-step broadcast wall-time summary for the D9 re-measurement."""
    if not samples:
        return {"steps": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}
    ordered = sorted(samples)
    count = len(ordered)

    def pct(p: float) -> float:
        idx = min(count - 1, max(0, int(round(p * (count - 1)))))
        return ordered[idx]

    return {
        "steps": count,
        "avg_ms": sum(ordered) / count,
        "p50_ms": pct(0.50),
        "p90_ms": pct(0.90),
    }


class Rank:
    """One rank: owns the session, its model shard, and the decode loop."""

    def __init__(self, model_path: str, seed: int, control_fd: int | None) -> None:
        self.model_path = model_path
        self.seed = seed
        self.session = DistributedSession()
        self.signals = ControlChannel(control_fd if self.session.is_leader else None)
        self.model: Any = None
        self.tokenizer: Any = None
        self.eos_ids: set[int] = set()
        self._commands: CommandReader | None = None

    # -- lifecycle ---------------------------------------------------------

    def load(self, metrics_path: str | None) -> None:
        """Shard-load this rank's slice and record the memory-gate numbers.

        The deathwatch over the parent daemon is started BEFORE this expensive
        window (salvage pitfall 5): a rank orphaned during the load exits at
        once rather than sitting in a half-formed collective.
        """
        import mlx.core as mx

        from omlx.cluster import tp

        self._start_parent_watch()

        result = tp.shard_and_load(self.model_path, self.session.group)
        self.model = result.model
        self.tokenizer = result.tokenizer
        self.eos_ids = self._collect_eos_ids()
        self.session.seed_everyone(self.seed)

        if metrics_path:
            self._write_metrics(
                metrics_path,
                post_shard_param_bytes=result.post_shard_param_bytes,
                mx_peak_bytes=int(mx.get_peak_memory()),
                rss_peak_bytes=tp.peak_process_bytes(),
            )

    def _start_parent_watch(self) -> None:
        """Exit fast if the daemon that spawned this rank goes away.

        A rank reparented to launchd (ppid 1) has lost its daemon; nobody is
        reading its replies and it must not keep a collective open. ``os._exit``
        because a rank blocked in a collective will not unwind cleanly.
        """

        def parent_alive() -> bool:
            return os.getppid() > 1

        watch = DeathWatch(
            [("parent", parent_alive)],
            lambda _label, _reason: os._exit(1),
        )
        watch.start()

    def _write_metrics(self, path: str, **numbers: int) -> None:
        payload = {"rank": self.session.rank, **numbers}
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)

    def _collect_eos_ids(self) -> set[int]:
        eos = getattr(self.tokenizer, "eos_token_ids", None)
        if eos:
            return set(eos)
        single = getattr(self.tokenizer, "eos_token_id", None)
        return {single} if single is not None else set()

    # -- control loop ------------------------------------------------------

    def serve(self) -> None:
        """Read commands (rank 0 from its pipe, followers via broadcast).

        A ``generate`` hands control to the decode loop until the request ends;
        everything else is a one-shot dispatch. Rank 0 emits a ``ready`` frame
        once loaded so the daemon knows the shard is up.
        """
        self._commands = CommandReader(sys.stdin.fileno())
        if self.session.is_leader:
            self._reply({"ok": True, "event": "ready", "rank": self.session.rank})

        while True:
            if self.session.is_leader:
                line = self._commands.readline()
                command = json.loads(line) if line else {"op": "shutdown"}
            else:
                command = None

            command = self.session.broadcast_json(command)
            if command is None or command.get("op") == "shutdown":
                logger.info("cluster: rank %d shutting down", self.session.rank)
                return

            op = command.get("op")
            if op == "generate":
                self._generate(command.get("spec") or {})
            elif op == "ping":
                self._reply({"ok": True, "rank": self.session.rank})
            else:
                self._reply(error_frame("", f"unknown op {op!r}"))

    # -- the D3 decode loop ------------------------------------------------

    def _generate(self, spec_dict: dict[str, Any]) -> None:
        """Run one request end to end, in lockstep across every rank."""
        import mlx.core as mx

        spec = GenerationSpec.from_dict(spec_dict)
        if not spec.prompt_ids:
            self._reply(error_frame(spec.request_id, "prompt has no tokens"))
            return

        cache = self._prefill(spec.prompt_ids)

        leader = self.session.is_leader
        rid = spec.request_id
        stop_set = set(spec.stop_token_ids) | self.eos_ids

        sampler = processors = None
        detok = stopbuf = None
        if leader:
            from omlx.utils.sampling import make_sampler

            sampler = make_sampler(
                temp=spec.temperature,
                top_p=spec.top_p,
                min_p=spec.min_p,
                top_k=spec.top_k,
            )
            processors = self._build_processors(spec)
            detok = self.tokenizer.detokenizer
            stopbuf = StopTextBuffer(spec.stop)

        current = spec.prompt_ids[-1]
        all_ids = list(spec.prompt_ids)
        completion = 0
        tax_samples: list[float] = []
        pending_stop = False
        finish_reason = "stop"
        step = 0

        while True:
            logits = self.model(mx.array([current])[None], cache=cache)[:, -1, :]

            if leader:
                next_id, decided_done, finish_reason, deltas, emit = self._decide(
                    logits,
                    sampler,
                    processors,
                    all_ids,
                    stop_set,
                    pending_stop,
                    completion,
                    spec.max_tokens,
                )
                mx.eval(logits, [c.state for c in cache])
                payload: dict[str, Any] | None = StepMessage(
                    step=step,
                    tokens={rid: next_id},
                    deltas=deltas,
                    done=decided_done,
                ).to_dict()
            else:
                mx.eval(logits, [c.state for c in cache])
                payload = None

            # D9 tax window: drain the model stream, then the two-collective
            # broadcast. Sampling is already realised (above), so it is not
            # counted here.
            t0 = time.perf_counter()
            mx.synchronize()
            received = self.session.broadcast_json(payload)
            if leader:
                tax_samples.append((time.perf_counter() - t0) * 1000.0)

            message = StepMessage.from_dict(received)
            token = message.tokens[rid]
            all_ids.append(token)

            if leader:
                assert stopbuf is not None
                pending_stop = self._emit(rid, token, completion, emit, detok, stopbuf)
                if emit and message.done and finish_reason == "length":
                    self._flush_tail(detok, stopbuf, rid, completion)
                if emit:
                    completion += 1

            step += 1
            if message.done:
                break
            current = token

        if leader:
            self._reply(
                done_frame(
                    rid,
                    text=stopbuf.text if stopbuf is not None else "",
                    prompt_tokens=len(spec.prompt_ids),
                    completion_tokens=completion,
                    finish_reason=finish_reason,
                    tax=_tax_summary(tax_samples),
                )
            )

    def _decide(
        self,
        logits: Any,
        sampler: Any,
        processors: list[Any] | None,
        all_ids: list[int],
        stop_set: set[int],
        pending_stop: bool,
        completion: int,
        max_tokens: int,
    ) -> tuple[int, bool, str, list[dict[str, Any]], bool]:
        """Rank 0's per-step decision: sample, then decide whether to stop.

        Returns ``(next_id, done, finish_reason, deltas, emit_token)``. A stop
        *token* is never emitted; an abort or a deferred stop-string hit ends
        the request without consuming this step's token as output.
        """
        import mlx.core as mx

        scored = logits
        if processors:
            context = mx.array(all_ids)
            for processor in processors:
                scored = processor(context, scored)
        logprobs = scored - mx.logsumexp(scored, axis=-1, keepdims=True)
        next_id = int(sampler(logprobs).item())

        aborted = any(e.get("op") == "abort" for e in self.signals.take_events())
        if pending_stop:
            return (
                next_id,
                True,
                "stop",
                [{"op": DELTA_FINISH, "reason": "stop"}],
                False,
            )
        if aborted:
            return next_id, True, "abort", [{"op": DELTA_ABORT}], False
        if next_id in stop_set:
            return (
                next_id,
                True,
                "stop",
                [{"op": DELTA_FINISH, "reason": "stop"}],
                False,
            )
        if completion + 1 >= max_tokens:
            return (
                next_id,
                True,
                "length",
                [{"op": DELTA_FINISH, "reason": "length"}],
                True,
            )
        return next_id, False, "", [], True

    def _emit(
        self,
        rid: str,
        token: int,
        completion: int,
        emit: bool,
        detok: Any,
        stopbuf: StopTextBuffer,
    ) -> bool:
        """Detokenize and stream one token; return whether a stop string hit.

        A stop string is detected from rank 0's text only, so it takes effect
        on the *next* step's broadcast (deferred one step, like an abort).
        """
        if not emit:
            return False
        detok.add_token(token)
        chunk = stopbuf.push(detok.last_segment)
        if chunk:
            self._reply(chunk_frame(rid, chunk, completion + 1))
        return stopbuf.hit is not None

    def _flush_tail(
        self,
        detok: Any,
        stopbuf: StopTextBuffer,
        rid: str,
        completion: int,
    ) -> None:
        detok.finalize()
        tail = stopbuf.push(detok.last_segment) + stopbuf.flush()
        if tail:
            self._reply(chunk_frame(rid, tail, completion))

    def _prefill(self, prompt_ids: list[int]) -> Any:
        """A fresh cache holding everything but the last prompt token.

        Chunked and fully evaluated on every rank while nothing else is in
        flight, so the sharded forward's collectives are issued in the same
        global order on every rank.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        tokens = prompt_ids[:-1]
        for start in range(0, len(tokens), PREFILL_STEP):
            chunk = tokens[start : start + PREFILL_STEP]
            # Eval the forward output too, not just the cache state: the final
            # layer's all-reduce feeds only the (discarded) logits and could
            # otherwise stay pending across ranks until a later synchronize.
            out = self.model(mx.array(chunk)[None], cache=cache)
            mx.eval(out, [c.state for c in cache])
        return cache

    def _build_processors(self, spec: GenerationSpec) -> list[Any]:
        if not any(
            (
                spec.repetition_penalty,
                spec.presence_penalty,
                spec.frequency_penalty,
            )
        ):
            return []
        from mlx_lm.sample_utils import make_logits_processors

        return list(
            make_logits_processors(
                repetition_penalty=spec.repetition_penalty,
                repetition_context_size=spec.repetition_context_size,
                presence_penalty=spec.presence_penalty,
                frequency_penalty=spec.frequency_penalty,
            )
        )

    def _reply(self, payload: dict[str, Any]) -> None:
        """Only rank 0 has a daemon listening; the rest stay quiet."""
        if not self.session.is_leader:
            return
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Entry point for a spawned rank process."""
    parser = argparse.ArgumentParser(prog="omlx-cluster-rank")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-fd",
        type=int,
        default=None,
        help="inherited read end of the daemon's out-of-band abort pipe",
    )
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    # A wedged rank is opaque from outside; SIGUSR2 dumps every thread's stack
    # to stderr, which the daemon already owns.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR2, all_threads=True)

    # Logs go to stderr; stdout is the reply channel and must stay clean JSON.
    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rank = Rank(args.model, args.seed, args.control_fd)
    rank.load(args.metrics_path)
    rank.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
