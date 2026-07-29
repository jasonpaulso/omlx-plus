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

Rank 0 is the only rank that talks to a daemon. Since S3 it runs oMLX's real
``Scheduler`` (unmodified, D4's single inert seam), constructed with a
:class:`~omlx.cluster.tp_batch.LeaderModelProxy` standing in for its model:
every model invocation the scheduler makes — prefill, admission, batched
decode — is broadcast to every rank as a :class:`~omlx.cluster.protocol.RankOp`
*before* it runs locally, so the whole formation's collective sequence stays
in lockstep. No rank ever branches on something only it can see, or the ranks
silently diverge and produce garbage. Rank 0 is the only rank that decides
anything (admits, samples, aborts, finishes); every other rank is a pure
:class:`~omlx.cluster.tp_batch.FollowerReplayer` — see
``discovery/spec/s3-plan.md`` D2/D4 and ``discovery/analysis/s3-interface-audit.md``
for the evidence this design rests on.

Failure
-------
mlx has no fault tolerance: a dead rank leaves its peers blocked in a collective
until a daemon's deathwatch kills them. A rank that loses its daemon (reparented
to launchd) exits on its own so its peers fail fast rather than hang.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from typing import Any

from omlx.cluster.launcher import CommandReader, ControlChannel, DeathWatch
from omlx.cluster.protocol import (
    GenerationSpec,
    RankOp,
    chunk_frame,
    done_frame,
    error_frame,
)

logger = logging.getLogger(__name__)


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

    # -- control loop ------------------------------------------------------

    def serve(self) -> None:
        """Dispatch to the leader's scheduler-driven loop, or a follower's
        pure replay loop (S3 D4). The two share nothing but the session:
        rank 0 owns the real ``Scheduler`` and every decision it makes;
        every other rank only ever replays a broadcast ``RankOp``.
        """
        if self.session.is_leader:
            self._serve_leader()
        else:
            self._serve_follower()

    # -- leader: the D4 scheduler-driven serve loop -------------------------

    def _serve_leader(self) -> None:
        from omlx.cluster.scheduler_config import build_rank_scheduler_config
        from omlx.cluster.tp_batch import LeaderModelProxy, TPBatchGenerator
        from omlx.scheduler import Scheduler

        self._commands = CommandReader(sys.stdin.fileno())
        proxy = LeaderModelProxy(self.model, self.session)
        scheduler = Scheduler(
            model=proxy,
            tokenizer=self.tokenizer,
            config=build_rank_scheduler_config(),
            batch_generator_factory=lambda _sp: TPBatchGenerator(proxy),
        )
        self._reply({"ok": True, "event": "ready", "rank": 0})

        # D2's deterministic gc-sweep trigger: a cache dropped on one of the
        # scheduler's three generator-destroying recovery branches only
        # becomes collectible when `batch_generator` itself is replaced (the
        # proxy's batched-phase tag anchor is a weakref to that instance) --
        # forcing a collect on identity-change-or-None here, before the next
        # op, is what makes a follower's release arrive promptly instead of
        # at arbitrary GC time.
        last_batch_generator: Any = scheduler.batch_generator

        # A batch generator's very first joint forward for two brand-new
        # requests admitted before either has ever taken a decode step is
        # not the pattern LeaderModelProxy/FollowerReplayer are proven
        # against -- only "join an already-stepped batch" is
        # (tests/cluster/test_rank_batch.py's admit-mid-decode case).
        # drain_lines() happily hands back every generate command that
        # landed in the pipe between iterations, so two HTTP requests
        # arriving close together would otherwise both get admitted before
        # step() ever runs once -- including the very first request, whose
        # solo admission used to skip straight back to the top of the loop
        # with no step() in between. Cap fresh admissions to one per
        # iteration and always step() right after admitting one, so every
        # later admission always joins a batch that already has step
        # history. Any extra generate lines wait one more iteration (one
        # step, a few tens of ms).
        pending_lines: list[str] = []

        while True:
            self._drain_aborts(scheduler)

            lines = pending_lines
            pending_lines = []
            if not lines:
                if scheduler.has_requests():
                    lines = self._commands.drain_lines()
                else:
                    line = self._commands.readline()
                    if not line:
                        self._shutdown_followers()
                        return
                    lines = [line]

            admitted_fresh = False
            for index, line in enumerate(lines):
                # Only a "generate" that actually reaches add_request() (not
                # one rejected with queue_full or a bad spec, which touch no
                # scheduler state) needs a step() before the next one -- do
                # not defer those, or a rejection-heavy burst would pay one
                # needless step() per rejected line.
                if json.loads(line).get("op") == "generate" and admitted_fresh:
                    pending_lines = lines[index:]
                    break
                keep_serving, admitted = self._handle_leader_command(
                    scheduler, proxy, line
                )
                if not keep_serving:
                    self._shutdown_followers()
                    return
                if admitted:
                    admitted_fresh = True

            if scheduler.has_requests():
                output = scheduler.step()
                self._emit_step_outputs(output, proxy)

            current_batch_generator = scheduler.batch_generator
            if (
                current_batch_generator is not last_batch_generator
                or current_batch_generator is None
            ):
                gc.collect()
            last_batch_generator = current_batch_generator

    def _shutdown_followers(self) -> None:
        """Broadcast the sentinel that ends every follower's replay loop."""
        logger.info("cluster: rank %d shutting down", self.session.rank)
        self.session.broadcast_json(None)

    def _handle_leader_command(
        self, scheduler: Any, proxy: Any, line: str
    ) -> tuple[bool, bool]:
        """Dispatch one pipe command.

        Returns ``(keep_serving, admitted)``: ``keep_serving`` is ``False``
        on ``shutdown``; ``admitted`` is ``True`` only when a ``generate``
        command actually added a request to the scheduler (a rejection
        touches no scheduler state, so it does not count).
        """
        command = json.loads(line)
        op = command.get("op")
        if op == "shutdown":
            return False, False
        if op == "generate":
            admitted = self._handle_generate(scheduler, command.get("spec") or {})
            return True, admitted
        if op == "stats":
            self._reply(self._stats_frame(scheduler, proxy))
        else:
            self._reply(error_frame("", f"unknown op {op!r}"))
        return True, False

    def _handle_generate(self, scheduler: Any, spec_dict: dict[str, Any]) -> bool:
        """Admit one ``generate`` command. Returns ``True`` iff the request
        actually reached ``scheduler.add_request()``.
        """
        from omlx.exceptions import SchedulerQueueFullError
        from omlx.request import Request, SamplingParams

        spec = GenerationSpec.from_dict(spec_dict)
        if not spec.prompt_ids:
            self._reply(error_frame(spec.request_id, "prompt has no tokens"))
            return False

        request = Request(
            request_id=spec.request_id,
            prompt=list(spec.prompt_ids),
            sampling_params=SamplingParams(
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                top_p=spec.top_p,
                top_k=spec.top_k,
                min_p=spec.min_p,
                repetition_penalty=(
                    spec.repetition_penalty
                    if spec.repetition_penalty is not None
                    else 1.0
                ),
                presence_penalty=spec.presence_penalty or 0.0,
                frequency_penalty=spec.frequency_penalty or 0.0,
                stop=list(spec.stop),
                stop_token_ids=list(spec.stop_token_ids),
                seed=spec.seed,
            ),
        )
        # GenerationSpec always carries pre-tokenized ids (the daemon owns
        # the tokenizer it used for the chat template, S2 idiom) -- setting
        # these directly makes add_request() skip its own tokenize path
        # (scheduler.py:6776-6781).
        request.prompt_token_ids = list(spec.prompt_ids)
        request.num_prompt_tokens = len(spec.prompt_ids)

        try:
            scheduler.add_request(request)
        except SchedulerQueueFullError as exc:
            self._reply(
                error_frame(
                    spec.request_id,
                    str(exc),
                    code="queue_full",
                    current_depth=exc.current_depth,
                    max_depth=exc.max_depth,
                )
            )
            return False
        except Exception as exc:  # noqa: BLE001 - surfaced to the daemon
            logger.exception("cluster: add_request failed for %s", spec.request_id)
            self._reply(error_frame(spec.request_id, str(exc)))
            return False
        return True

    def _drain_aborts(self, scheduler: Any) -> None:
        """Apply every pending abort signal (S2 idiom, request-id-routed).

        ``abort_request()`` only enqueues; the actual removal happens inside
        the next ``step()``, and the scheduler never surfaces an abort
        completion through ``step()``'s own outputs (unlike a model-decided
        finish) -- so this synthesizes the terminal frame itself, guarded on
        the request genuinely being live right now. An empty ``request_id``
        (control pipe closed -- nobody left to stream to) aborts every
        running and waiting request.
        """
        for event in self.signals.take_events():
            rid = event.get("request_id") or ""
            targets = (
                [rid]
                if rid
                else list(scheduler.running) + [r.request_id for r in scheduler.waiting]
            )
            for request_id in targets:
                if request_id in scheduler.requests:
                    scheduler.abort_request(request_id)
                    self._reply(
                        done_frame(
                            request_id,
                            text="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            finish_reason="abort",
                        )
                    )

    def _emit_step_outputs(self, output: Any, proxy: Any) -> None:
        for out in output.outputs:
            rid = out.request_id
            if out.finished:
                if out.finish_reason == "error":
                    self._reply(error_frame(rid, out.error or "generation error"))
                else:
                    self._reply(
                        done_frame(
                            rid,
                            text=out.output_text,
                            prompt_tokens=out.prompt_tokens,
                            completion_tokens=out.completion_tokens,
                            finish_reason=out.finish_reason or "stop",
                            tax=_tax_summary(proxy.tax_samples),
                        )
                    )
            elif out.new_text:
                self._reply(chunk_frame(rid, out.new_text, out.completion_tokens))

    def _stats_frame(self, scheduler: Any, proxy: Any) -> dict[str, Any]:
        stats = scheduler.get_stats()
        stats["tax"] = _tax_summary(proxy.tax_samples)
        return {"ok": True, "event": "stats", "stats": stats}

    # -- follower: pure forward-replay loop (S3 D2) -------------------------

    def _serve_follower(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        # Import omlx.scheduler for its module-level monkeypatches
        # (mlx_lm.generate._merge_caches/_extend_cache singleton-passthrough,
        # KVCache.filter/.extract) -- without this a follower's single-row
        # cache merges take mlx-lm's unpatched path while the leader (which
        # always imports Scheduler) takes the patched one, growing the two
        # ranks' batch cache buffers to different physical sizes and
        # deadlocking the sharded forward the first time two rows genuinely
        # merge (S3 P1 audit finding 2; see tp_batch.py's module docstring).
        import omlx.scheduler  # noqa: F401
        from omlx.cluster.tp_batch import FollowerReplayer

        replayer = FollowerReplayer(self.model, lambda: make_prompt_cache(self.model))
        while True:
            # Sync discipline (audit section 8): drain this rank's own model
            # stream before handing the backend the next broadcast so every
            # rank issues collectives in one global order.
            mx.synchronize()
            payload = self.session.broadcast_json(None)
            if payload is None:
                logger.info("cluster: rank %d shutting down", self.session.rank)
                return
            replayer.apply(RankOp.from_dict(payload))

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
