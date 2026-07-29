# SPDX-License-Identifier: Apache-2.0
"""Test-only rank entry point for the S3 P1 integration test
(``test_rank_batch.py``).

Why this file exists instead of driving ``omlx.cluster.rank_worker``: P1's
scope is ``tp_batch.py``, ``protocol.py``, and the single named seam in
``scheduler.py`` — rewriting ``rank_worker.py``'s ``Rank._generate()`` FIFO
loop into the D4 continuous-batching serve loop is P2's job (see the S3 plan's
Execution topology). There is therefore no production entry point yet that
drives a real ``Scheduler`` + ``TPBatchGenerator`` + ``LeaderModelProxy``
across two rank processes. This module is that entry point for the test only.

It is spawned by ``omlx.cluster.launcher.LocalCluster`` with
``launcher.WORKER_MODULE`` monkeypatched to this module's dotted path for the
duration of the test — ``LocalCluster``'s spawn/hostfile/deathwatch/teardown
machinery and its generic newline-JSON pipe transport (``stream``/``command``/
``abort``/``stop``) are reused completely unchanged; only the process this
module names on the other end of the pipe differs from production.

Rank 0 constructs a real ``Scheduler`` (unmodified, D4 seam only) with a
``LeaderModelProxy`` as its model and a ``batch_generator_factory`` that
builds ``TPBatchGenerator``. Every rank > 0 runs a ``FollowerReplayer`` loop.
The command/reply frame shapes here are private to this test file — the only
part of the real wire protocol exercised is ``DistributedSession
.broadcast_json`` (unchanged) carrying ``RankOp`` (P1-owned).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from omlx.cluster.launcher import CommandReader, ControlChannel, DeathWatch
from omlx.cluster.protocol import RankOp, chunk_frame, done_frame, error_frame
from omlx.cluster.rank_worker import DistributedSession
from omlx.cluster.tp_batch import FollowerReplayer, LeaderModelProxy, TPBatchGenerator

logger = logging.getLogger(__name__)


def _start_parent_watch() -> None:
    """Exit fast if the daemon (here: the test process) that spawned this
    rank goes away — same discipline as ``rank_worker.Rank._start_parent_watch``.
    """

    def parent_alive() -> bool:
        return os.getppid() > 1

    DeathWatch([("parent", parent_alive)], lambda _label, _reason: os._exit(1)).start()


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _run_leader(
    session: DistributedSession, model: Any, tokenizer: Any, control_fd: int | None
) -> None:
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig

    signals = ControlChannel(control_fd)
    proxy = LeaderModelProxy(model, session)
    scheduler = Scheduler(
        model=proxy,
        tokenizer=tokenizer,
        config=SchedulerConfig(),
        batch_generator_factory=lambda _sp: TPBatchGenerator(proxy),
    )

    commands = CommandReader(sys.stdin.fileno())
    _reply({"ok": True, "event": "ready", "rank": 0})

    def handle(cmd: dict[str, Any]) -> bool:
        """Returns False when the loop must exit (shutdown)."""
        op = cmd.get("op")
        if op == "shutdown":
            # Tell every follower to stop looping too — broadcast_json is the
            # only channel that reaches them (D3).
            session.broadcast_json(None)
            return False
        if op == "generate":
            spec = cmd["spec"]
            # Pre-tokenized ids for tests that only care about scheduler
            # dynamics; a plain-text prompt (tokenized here, on rank 0, with
            # the real loaded tokenizer) for the greedy-parity test, so both
            # sides of the comparison tokenize identically.
            prompt_ids = spec.get("prompt_ids")
            if prompt_ids is None:
                prompt_ids = tokenizer.encode(spec["prompt"])
            request = Request(
                request_id=spec["request_id"],
                prompt=prompt_ids,
                sampling_params=SamplingParams(
                    max_tokens=spec.get("max_tokens", 64),
                    temperature=0.0,
                    stop=spec.get("stop") or [],
                    stop_token_ids=spec.get("stop_token_ids") or [],
                ),
            )
            request.prompt_token_ids = list(prompt_ids)
            request.num_prompt_tokens = len(request.prompt_token_ids)
            try:
                scheduler.add_request(request)
            except Exception as exc:  # noqa: BLE001
                logger.exception("add_request failed")
                _reply(error_frame(spec["request_id"], str(exc)))
            return True
        _reply(error_frame("", f"unknown op {op!r}"))
        return True

    while True:
        for event in signals.take_events():
            rid = event.get("request_id") or ""
            targets = (
                [rid]
                if rid
                else list(scheduler.running) + [r.request_id for r in scheduler.waiting]
            )
            for request_id in targets:
                # abort_request() always returns True (unconditionally
                # enqueued) and the actual removal is deferred to the next
                # step() — Scheduler never surfaces an abort completion
                # through step()'s outputs (unlike a model-decided finish),
                # so the harness emits the terminal frame itself, guarded on
                # the request genuinely being live right now.
                if request_id in scheduler.requests:
                    scheduler.abort_request(request_id)
                    _reply(
                        done_frame(
                            request_id,
                            text="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            finish_reason="abort",
                        )
                    )

        if scheduler.has_requests():
            for line in commands.drain_lines():
                if not handle(json.loads(line)):
                    return
            output = scheduler.step()
            for out in output.outputs:
                rid = out.request_id
                if out.finished:
                    _reply(
                        done_frame(
                            rid,
                            text=out.output_text,
                            prompt_tokens=out.prompt_tokens,
                            completion_tokens=out.completion_tokens,
                            finish_reason=out.finish_reason or "stop",
                        )
                    )
                elif out.new_text:
                    _reply(chunk_frame(rid, out.new_text, out.completion_tokens))
        else:
            line = commands.readline()
            if not line:
                return
            if not handle(json.loads(line)):
                return


def _run_follower(session: DistributedSession, model: Any) -> None:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    # Importing omlx.scheduler (never otherwise needed on a follower rank)
    # runs its module-level monkeypatches: mlx_lm.generate._merge_caches /
    # _extend_cache are globally replaced, and KVCache gains .filter /
    # .extract if upstream lacks them. The leader rank always imports
    # Scheduler and gets these for free; without this import a follower's
    # single-row cache merges take mlx-lm's *unpatched* path (always a full
    # BatchKVCache wrap) while the leader's take the patched "singleton
    # passthrough" path (stays a raw KVCache until genuinely merged with a
    # second row) — same tag, same logical offset, but a different cache
    # object lineage with a different buffer-growth history on each rank.
    # That surfaced as a real bug: by the time two rows actually merge, the
    # two ranks' buffers had grown to different physical sizes even though
    # `offset`/`left_padding` matched, and the sharded forward over those
    # mismatched buffers deadlocked mid-collective.
    import omlx.scheduler  # noqa: F401

    replayer = FollowerReplayer(model, lambda: make_prompt_cache(model))
    while True:
        mx.synchronize()
        payload = session.broadcast_json(None)
        if payload is None:
            return
        replayer.apply(RankOp.from_dict(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="s3-rank-batch-test-worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-fd", type=int, default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR2, all_threads=True)
    logging.basicConfig(
        level=args.log_level.upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from omlx.cluster import tp

    _start_parent_watch()
    session = DistributedSession()
    result = tp.shard_and_load(args.model, session.group)
    session.seed_everyone(args.seed)

    if session.is_leader:
        _run_leader(session, result.model, result.tokenizer, args.control_fd)
    else:
        _run_follower(session, result.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
