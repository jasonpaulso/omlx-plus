"""S0 Task B: single-node batch-1 decode-only ms/token baseline.

Qwen3.6-27B-bf16, batch 1, decode-only (prefill excluded), >=256 tokens steady state.
Run: python benchmarks/cluster_spike/baseline_decode.py
"""
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen3.6-27B-bf16"
NUM_DECODE_TOKENS = 320  # >=256, some headroom for steady-state window
PROMPT = "Explain the concept of distributed systems in a few sentences."


def main():
    print(f"Loading {MODEL} ...")
    model, tokenizer = load(MODEL)

    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = mx.array(prompt)

    sampler = make_sampler(temp=0.0)  # greedy, deterministic, no extra RNG cost variance

    # --- Prefill (excluded from timing) ---
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)

    def step(y, cache):
        logits = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        tok = sampler(logits)
        return tok

    # prefill: feed full prompt once, get first token
    first_logits = model(prompt[None], cache=cache)[:, -1, :]
    first_tok = sampler(first_logits)
    mx.eval(first_tok)

    # --- Decode loop, timed ---
    tok = first_tok
    times = []
    tokens_generated = 0
    warmup = 16
    for i in range(NUM_DECODE_TOKENS + warmup):
        t0 = time.perf_counter()
        logits = model(tok[None], cache=cache)[:, -1, :]
        tok = sampler(logits)
        mx.eval(tok)
        t1 = time.perf_counter()
        tokens_generated += 1
        if i >= warmup:
            times.append(t1 - t0)

    times_ms = [t * 1000 for t in times]
    avg = sum(times_ms) / len(times_ms)
    times_sorted = sorted(times_ms)
    p50 = times_sorted[len(times_sorted) // 2]
    p90 = times_sorted[int(len(times_sorted) * 0.9)]

    print("\n=== E4 BASELINE RESULT ===")
    print(f"model: {MODEL} (bf16)")
    print(f"batch: 1, decode-only (prefill excluded), steady-state tokens: {len(times)}")
    print(f"avg ms/token: {avg:.3f}")
    print(f"p50 ms/token: {p50:.3f}")
    print(f"p90 ms/token: {p90:.3f}")
    print(f"tok/s (avg): {1000.0/avg:.2f}")


if __name__ == "__main__":
    main()
