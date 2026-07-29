# SPDX-License-Identifier: Apache-2.0
"""P1 integration: two real local rank processes driving the S3 forward-replay
Scheduler (no HTTP, no formation job, no ClusterEngine — those are S1/P2).

Rank 0 runs a real ``Scheduler`` (unmodified beyond the D4 seam) with a
``LeaderModelProxy`` as its model and a ``batch_generator_factory`` building
``TPBatchGenerator``; rank 1 runs a ``FollowerReplayer`` loop. Both are driven
by ``tests/cluster/_s3_rank_worker.py`` — a test-only entry point, since the
production rewrite of ``rank_worker.py``'s serve loop is P2 scope (see
``discovery/spec/s3-plan.md``, Execution topology). ``LocalCluster``'s
spawn/hostfile/deathwatch/teardown machinery is reused completely unchanged
via a monkeypatched ``WORKER_MODULE``.

Double-marked ``cluster`` + ``integration`` so the default unit gate
(``-m "not slow and not integration"``) collects none of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from collections.abc import Iterator

import pytest

from omlx.cluster import launcher
from omlx.cluster.launcher import LocalCluster

pytestmark = [pytest.mark.cluster, pytest.mark.integration]

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
READY_TIMEOUT_S = 180.0
IDLE_TIMEOUT_S = 60.0


_WORKER_MODULE = "tests.cluster._s3_rank_worker"


@contextlib.contextmanager
def formation() -> Iterator[LocalCluster]:
    """Form two loopback ranks driving the S3 test-only entry point.

    ``_s3_rank_worker.__name__`` is not used here: pytest's rootless test
    collection (``tests/`` has no ``__init__.py``) imports this test module
    itself as ``cluster.test_rank_batch``, which would make the worker
    subprocess's ``-m`` import fail (no ``PYTHONPATH`` reaches it — CL2-01).
    The literal dotted path resolves correctly because the launcher spawns
    from the repo-root CWD, where ``tests`` is importable as an implicit
    PEP 420 namespace package.
    """
    original = launcher.WORKER_MODULE
    launcher.WORKER_MODULE = _WORKER_MODULE
    base_port = random.randint(45000, 48000)
    cluster = LocalCluster(
        model=MODEL, world_size=2, backend="ring", base_port=base_port
    )
    try:
        cluster.start(
            [0, 1],
            ips=["127.0.0.1", "127.0.0.1"],
            data_plane_subnet="127.0.0.0/8",
            allow_loopback=True,
        )
        cluster.wait_ready(timeout=READY_TIMEOUT_S)
        yield cluster
    finally:
        cluster.stop(timeout=10)
        launcher.WORKER_MODULE = original


class _Demux:
    """Reads rank 0's reply stream, routing frames by ``request_id`` so two
    concurrent generations can be driven over the one pipe — the real
    ``LocalCluster.stream()`` helper assumes one request in flight at a time
    (S2's own FIFO shape), which this test deliberately violates.
    """

    def __init__(self, cluster: LocalCluster) -> None:
        self._cluster = cluster
        self._reader = cluster._reply_reader(cluster.leader)
        self._buffered: dict[str, list[dict]] = {}

    def send_generate(self, request_id: str, *, max_tokens: int, **spec_kwargs) -> None:
        spec = {"request_id": request_id, "max_tokens": max_tokens, **spec_kwargs}
        leader = self._cluster.leader
        assert leader is not None and leader.process.stdin is not None
        payload = json.dumps({"op": "generate", "spec": spec}) + "\n"
        with self._cluster._stdin_lock:
            leader.process.stdin.write(payload)
            leader.process.stdin.flush()

    def next_frame(self, request_id: str, timeout: float = IDLE_TIMEOUT_S) -> dict:
        buffered = self._buffered.get(request_id)
        if buffered:
            return buffered.pop(0)
        while True:
            line = self._reader.readline(timeout)
            if line is None:
                raise RuntimeError(f"idle timeout waiting for {request_id!r}")
            if not line:
                raise RuntimeError("rank 0 closed its reply channel")
            frame = json.loads(line)
            rid = frame.get("request_id")
            if rid == request_id:
                return frame
            self._buffered.setdefault(rid, []).append(frame)

    def drain_until_done(
        self, request_ids: list[str], timeout: float = IDLE_TIMEOUT_S
    ) -> dict[str, list[dict]]:
        pending = set(request_ids)
        collected: dict[str, list[dict]] = {rid: [] for rid in request_ids}
        for rid in list(pending):
            buffered = self._buffered.pop(rid, [])
            collected[rid].extend(buffered)
            if buffered and buffered[-1].get("done"):
                pending.discard(rid)
        while pending:
            line = self._reader.readline(timeout)
            if line is None:
                raise RuntimeError(f"idle timeout waiting for {sorted(pending)}")
            if not line:
                raise RuntimeError("rank 0 closed its reply channel")
            frame = json.loads(line)
            rid = frame.get("request_id")
            if rid in collected:
                collected[rid].append(frame)
                if frame.get("done"):
                    pending.discard(rid)
            else:
                self._buffered.setdefault(rid, []).append(frame)
        return collected


# -- form -> shard-load -> admit B mid-decode -> both complete ---------------


def test_admit_mid_decode_both_complete():
    with formation() as cluster:
        demux = _Demux(cluster)
        demux.send_generate("A", prompt_ids=[1, 2, 3], max_tokens=12)

        # Read a couple of A's chunks first, proving it is genuinely decoding
        # (not just queued) before B is admitted mid-stream.
        for _ in range(2):
            frame = demux.next_frame("A")
            assert frame["ok"]

        demux.send_generate("B", prompt_ids=[4, 5], max_tokens=8)

        frames = demux.drain_until_done(["A", "B"])
        assert frames["A"][-1]["done"] and frames["A"][-1]["finish_reason"] == "length"
        assert frames["B"][-1]["done"] and frames["B"][-1]["finish_reason"] == "length"
        assert frames["A"][-1]["completion_tokens"] == 12
        assert frames["B"][-1]["completion_tokens"] == 8


# -- abort A at a step boundary while B continues to completion --------------


def test_abort_mid_batch_other_continues():
    with formation() as cluster:
        demux = _Demux(cluster)
        demux.send_generate("A", prompt_ids=[1, 2, 3], max_tokens=64)
        demux.next_frame("A")
        demux.send_generate("B", prompt_ids=[4, 5], max_tokens=16)
        demux.next_frame("B")

        assert cluster.abort("A") is True

        frames = demux.drain_until_done(["A", "B"])
        assert frames["A"][-1]["finish_reason"] == "abort"
        assert frames["B"][-1]["finish_reason"] == "length"
        assert frames["B"][-1]["completion_tokens"] == 16


# -- greedy parity: batched (single request) == single-node ------------------


def test_greedy_parity_batched_vs_single_node():
    with formation() as cluster:
        demux = _Demux(cluster)
        demux.send_generate("solo", prompt="The capital of France is", max_tokens=8)
        frames = demux.drain_until_done(["solo"])
        dist_text = frames["solo"][-1]["text"]
        assert dist_text

    from omlx.engine.batched import BatchedEngine

    async def _single_node() -> str:
        engine = BatchedEngine(MODEL)
        try:
            await engine.start()
            out = await engine.generate(
                "The capital of France is", max_tokens=8, temperature=0.0
            )
            return out.text
        finally:
            await engine.stop()

    single_text = asyncio.run(_single_node())
    # S2's own parity test (test_cluster_serving.py::
    # test_greedy_parity_with_single_node) normalises only the trailing
    # edge, because it streams through the production BatchedEngine path on
    # both sides. This test's distributed side streams through
    # _s3_rank_worker.py instead — a minimal test-only entry point (P2 owns
    # the production rewrite) that concatenates raw chunk text without the
    # engine's leading-space suppression on the first generated token, so a
    # leading-edge difference here is a test-harness formatting detail, not
    # a token-sequence divergence. Normalise both edges; a real mid-stream
    # argmax divergence still fails this assertion.
    assert dist_text.strip() == single_text.strip()


# -- teardown ------------------------------------------------------------


def test_clean_teardown_leaves_no_orphans():
    with formation() as cluster:
        assert cluster.any_alive()
    assert launcher.sweep_orphaned_ranks() == 0
