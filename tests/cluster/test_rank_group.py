# SPDX-License-Identifier: Apache-2.0
"""Integration test: two local rank processes driven through the launcher/pipe.

No HTTP, no formation job, no ClusterEngine — every component here is P1's.
Two colocated ranks form a ring on loopback, shard-load a small model, decode,
apply an abort at a step boundary, and tear down; plus orphan sweep and a
rank-death-mid-decode caught by the deathwatch as a clean error at the pipe.

Double-marked ``cluster`` + ``integration`` so the default unit gate
(``-m "not slow and not integration"``) collects none of it.
"""

from __future__ import annotations

import contextlib
import random
import subprocess
import sys
import time
from collections.abc import Iterator

import psutil
import pytest

from omlx.cluster.launcher import (
    WORKER_MODULE,
    LocalCluster,
    sweep_orphaned_ranks,
)
from omlx.cluster.protocol import GenerationSpec

pytestmark = [pytest.mark.cluster, pytest.mark.integration]

# Small, shard-capable, present on disk (S0). A 1B model cannot expose the
# transient full-materialisation memory peak — that is the 27B memory gate's
# job — but it is enough to prove the loop forms, decodes, aborts, and tears
# down.
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

READY_TIMEOUT_S = 180.0
IDLE_TIMEOUT_S = 30.0


@contextlib.contextmanager
def formation(metrics_dir=None) -> Iterator[LocalCluster]:
    """Form two loopback ranks; always tear them down."""
    base_port = random.randint(42000, 45000)
    cluster = LocalCluster(
        model=MODEL,
        world_size=2,
        backend="ring",
        base_port=base_port,
        metrics_dir=metrics_dir,
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


def _generate(cluster: LocalCluster, spec: GenerationSpec) -> list[dict]:
    return list(
        cluster.stream(
            {"op": "generate", "spec": spec.to_dict()}, timeout=IDLE_TIMEOUT_S
        )
    )


# -- form -> decode -> abort -> teardown -------------------------------------


def test_group_forms_decodes_and_reports_tax():
    with formation() as cluster:
        spec = GenerationSpec(prompt_ids=[1, 2, 3], max_tokens=8, request_id="r1")
        frames = _generate(cluster, spec)

        done = frames[-1]
        assert done["done"] is True
        assert done["finish_reason"] == "length"
        assert done["completion_tokens"] == 8
        # Every earlier frame is a streamed chunk for the same request.
        assert all(f["request_id"] == "r1" for f in frames)
        # D9 tax accounting is present and per-step.
        tax = done["tax"]
        assert tax["steps"] == 8
        assert tax["avg_ms"] >= 0.0


def test_abort_applies_at_a_step_boundary():
    with formation() as cluster:
        spec = GenerationSpec(prompt_ids=[1, 2, 3], max_tokens=64, request_id="r2")
        done = None
        for index, frame in enumerate(
            cluster.stream(
                {"op": "generate", "spec": spec.to_dict()}, timeout=IDLE_TIMEOUT_S
            )
        ):
            if index == 3:
                assert cluster.abort("r2") is True
            if frame.get("done"):
                done = frame

        assert done is not None
        assert done["finish_reason"] == "abort"
        # The abort landed well before the 64-token cap.
        assert done["completion_tokens"] < 64


def test_second_generation_reuses_the_group():
    # The ranks stay alive across requests (one init per process lifetime).
    with formation() as cluster:
        first = _generate(
            cluster, GenerationSpec(prompt_ids=[1, 2, 3], max_tokens=4, request_id="a")
        )
        second = _generate(
            cluster, GenerationSpec(prompt_ids=[4, 5, 6], max_tokens=4, request_id="b")
        )
        assert first[-1]["done"] and second[-1]["done"]
        assert second[-1]["request_id"] == "b"


def test_multi_chunk_prefill():
    # A prompt longer than PREFILL_STEP (2048) exercises the chunked
    # hand-prefill loop running more than once, on both ranks in lockstep.
    with formation() as cluster:
        spec = GenerationSpec(
            prompt_ids=list(range(1, 2500)), max_tokens=4, request_id="p1"
        )
        done = _generate(cluster, spec)[-1]
        assert done["prompt_tokens"] == 2499
        assert done["completion_tokens"] == 4
        assert done["finish_reason"] == "length"


def test_stop_string_ends_the_request():
    # A stop string is detected from rank 0's text and applied one step later
    # (the deferred-stop path). Greedy decode of this prompt emits "#" first.
    with formation() as cluster:
        spec = GenerationSpec(
            prompt_ids=[1, 2, 3], max_tokens=16, stop=["#"], request_id="s1"
        )
        done = _generate(cluster, spec)[-1]
        assert done["finish_reason"] == "stop"
        assert "#" not in done["text"]


# -- rank death mid-decode -> deathwatch -> clean error at the pipe ----------


def test_rank_death_midflight_surfaces_a_clean_error():
    with formation() as cluster:
        cluster.start_deathwatch(interval=0.2)
        spec = GenerationSpec(prompt_ids=[1, 2, 3], max_tokens=256, request_id="die")

        with pytest.raises(RuntimeError):
            for index, _frame in enumerate(
                cluster.stream(
                    {"op": "generate", "spec": spec.to_dict()}, timeout=IDLE_TIMEOUT_S
                )
            ):
                if index == 3:
                    # Kill the peer rank mid-decode. Rank 0 blocks in the next
                    # collective; the deathwatch notices and kills the
                    # formation, closing rank 0's pipe — a clean error, not a
                    # ten-minute hang.
                    peer = next(r for r in cluster.ranks if r.rank == 1)
                    peer.process.kill()


# -- orphan sweep ------------------------------------------------------------


def test_sweep_leaves_our_own_ranks_alone():
    with formation() as cluster:
        # Our ranks are live children of this process, never orphans.
        assert sweep_orphaned_ranks() == 0
        assert cluster.any_alive()


def test_sweep_kills_a_reparented_orphan():
    # A rank whose parent daemon has exited (reparented to launchd) holds its
    # ring port and must be swept. Build one: a short-lived parent spawns a
    # module-named sleeper, then exits, orphaning it.
    parent_code = (
        "import subprocess, sys;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)',"
        f" {WORKER_MODULE!r}])"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    parent.wait(timeout=10)
    time.sleep(0.5)  # let the orphan reparent

    try:
        assert sweep_orphaned_ranks() >= 1
    finally:
        # Best-effort: nothing named like the orphan should survive.
        for proc in psutil.process_iter(["cmdline"]):
            cmdline = proc.info.get("cmdline") or []
            if WORKER_MODULE in cmdline and "time.sleep(20)" in " ".join(cmdline):
                with contextlib.suppress(Exception):
                    proc.kill()
