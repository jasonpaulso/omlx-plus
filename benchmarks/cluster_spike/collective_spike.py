"""S0 Tasks C+D: bring up a distributed session (ring or jaccl) and measure:
  1. small-collective latency (all_sum on a few-KB array), steady state
  2. idle-rank per-step broadcast lower bound (mock token ids + composition delta)
  3. the same broadcast interleaved into a real TP-sharded decode loop of a
     small model, vs the same loop without the broadcast

Backend/topology is entirely env-var driven (MLX_HOSTFILE / MLX_JACCL_COORDINATOR
etc, set by the launching shell) -- this script itself is backend-agnostic.

ONE session per process lifetime. Do not re-run this process to "retry" a
step; if something goes wrong, kill it and start a completely fresh process
(see bringup.md pitfall notes on JACCL protection-domain leaks).

Usage: run identically on both ranks (rank 0 = this machine / leader, rank 1
= worker), env vars set per-rank beforehand. Output is only meaningful from
rank 0 (it prints the timings); rank 1 participates silently.
"""
from __future__ import annotations

import pickle
import sys
import time

import mlx.core as mx
from mlx_lm import load as lm_load

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from dist_session import DistributedSession  # noqa: E402

SMALL_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
COLLECTIVE_ITERS = 200
COLLECTIVE_WARMUP = 20
IDLE_BROADCAST_ITERS = 200
IDLE_BROADCAST_WARMUP = 20
DECODE_TOKENS = 64
DECODE_WARMUP = 8


def rprint(session, *args):
    if session.world.is_leader:
        print(*args)


def measure_collective_latency(session: DistributedSession, nbytes: int):
    for _ in range(COLLECTIVE_WARMUP):
        mx.eval(session.all_sum_latency(nbytes))
    times = []
    for _ in range(COLLECTIVE_ITERS):
        t0 = time.perf_counter()
        mx.eval(session.all_sum_latency(nbytes))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "nbytes": nbytes,
        "avg_ms": sum(times) / len(times),
        "p50_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
    }


def mock_step_payload(step: int):
    # a "few hundred bytes": mock token ids (batch of ~32) + a composition delta
    return {
        "step": step,
        "token_ids": list(range(32)),
        "composition_delta": {"admit": [step % 7], "finish": [], "abort": []},
    }


def measure_idle_broadcast(session: DistributedSession):
    for i in range(IDLE_BROADCAST_WARMUP):
        session.broadcast(mock_step_payload(i) if session.world.is_leader else None)
    times = []
    for i in range(IDLE_BROADCAST_ITERS):
        t0 = time.perf_counter()
        session.broadcast(mock_step_payload(i) if session.world.is_leader else None)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "avg_ms": sum(times) / len(times),
        "p50_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
    }


def measure_tp_decode(session: DistributedSession, with_broadcast: bool):
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import sharded_load

    model, tokenizer = sharded_load(SMALL_MODEL, tensor_group=session.group)
    sampler = make_sampler(temp=0.0)
    cache = make_prompt_cache(model)

    prompt_text = "Explain the concept of distributed systems in a few sentences."
    if session.world.is_leader:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt = None
    # TP forward pass is collective: every rank must feed identical input
    # tokens regardless of the with_broadcast flag (that flag only gates the
    # *per-step* control broadcast measured below).
    prompt = session.broadcast(prompt)
    prompt = mx.array(prompt)

    # hand-prefill (batch=1); do NOT use PromptProcessingBatch/BatchGenerator
    # under TP -- 0/5 configs survived in the prior attempt (salvage pitfall #1)
    logits = model(prompt[None], cache=cache)[:, -1, :]
    tok = sampler(logits)
    mx.eval(tok)

    times = []
    for i in range(DECODE_TOKENS + DECODE_WARMUP):
        t0 = time.perf_counter()
        if with_broadcast:
            # drain model stream before the control-plane collective (mlx_adapter
            # broadcast docstring: ops here race in-flight model compute)
            mx.synchronize()
            _ = session.broadcast(mock_step_payload(i) if session.world.is_leader else None)
        logits = model(tok[None], cache=cache)[:, -1, :]
        tok = sampler(logits)
        mx.eval(tok)
        t1 = time.perf_counter()
        if i >= DECODE_WARMUP:
            times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "avg_ms_per_token": sum(times) / len(times),
        "p50_ms_per_token": times[len(times) // 2],
    }


def main():
    session = DistributedSession()
    rprint(session, f"=== joined: rank {session.world.rank}/{session.world.size} backend={__import__('os').environ.get('OMLX_CLUSTER_BACKEND')} ===")

    # 1. collective latency sweep, few-KB sizes
    for nbytes in (1024, 4096, 16384):
        r = measure_collective_latency(session, nbytes)
        rprint(session, f"[collective all_sum] nbytes={nbytes} avg={r['avg_ms']:.4f}ms p50={r['p50_ms']:.4f}ms p90={r['p90_ms']:.4f}ms")

    # 2. idle-rank per-step broadcast lower bound
    r = measure_idle_broadcast(session)
    rprint(session, f"[idle broadcast] avg={r['avg_ms']:.4f}ms p50={r['p50_ms']:.4f}ms p90={r['p90_ms']:.4f}ms")

    # 3. TP decode loop, without then with broadcast, then a third no-broadcast
    # pass to check for monotonic drift (cache warming / memory pressure)
    # contaminating whichever measurement runs later in process lifetime.
    r_without = measure_tp_decode(session, with_broadcast=False)
    rprint(session, f"[TP decode, no broadcast] avg={r_without['avg_ms_per_token']:.4f}ms/token")
    r_with = measure_tp_decode(session, with_broadcast=True)
    rprint(session, f"[TP decode, with broadcast] avg={r_with['avg_ms_per_token']:.4f}ms/token")
    r_without2 = measure_tp_decode(session, with_broadcast=False)
    rprint(session, f"[TP decode, no broadcast, repeat] avg={r_without2['avg_ms_per_token']:.4f}ms/token")

    if session.world.is_leader:
        overhead = r_with["avg_ms_per_token"] - r_without["avg_ms_per_token"]
        drift = r_without2["avg_ms_per_token"] - r_without["avg_ms_per_token"]
        print(f"[TP decode broadcast overhead] {overhead:.4f} ms/token")
        print(f"[TP decode drift check, no-broadcast pass 1 vs 3] delta={drift:.4f} ms/token")

    rprint(session, "=== DONE ===")


if __name__ == "__main__":
    main()
