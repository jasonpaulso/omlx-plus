#!/usr/bin/env python3
"""Apply the E4 coordination-tax gate (row 6) to a measured window.

`last_tax` from `GET /v1/cluster/models/status` is CUMULATIVE: rank 0's
`LeaderModelProxy.tax_samples` is created once per model load and never reset
(tp_batch.py:145, appended at :261), so `_tax_summary` averages every broadcast
since the model loaded -- baseline run, prefills and all. Reading it straight
after the concurrent run would not be "tax under batch".

Because `avg_ms * steps` is an exact sum, a snapshot either side of a run
recovers the windowed average without touching product code:

  sum_ms   = avg_after * steps_after - avg_before * steps_before
  steps    = steps_after - steps_before
  per-token tax = sum_ms / completion_tokens_in_window

Per-token, not per-step: one broadcast serves the whole batch, so per-step tax
is not comparable to the E4 budget, which is 10% of the S0-measured
102.682 ms/token single-token latency.

  Pass <=> per-token tax <= 10.268 ms/token.

The gate is applied to BOTH the single (batch=1) and concurrent (batch=4)
windows, and both must pass. Batch=1 is the strict case -- the tax amortizes
over one token instead of four -- so gating only the concurrent window would be
choosing the lenient number. Pinned before the run, per the D7 no-retry rule.
"""

from __future__ import annotations

import argparse
import json
import sys

BUDGET_MS_PER_TOKEN = 10.268  # 10% of S0's 102.682 ms/token


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def tax_point(path):
    """Read (steps, sum_ms) out of a models/status snapshot.

    `last_tax` is null until the first request completes, which is the correct
    reading for a pre-run snapshot: zero steps, zero accumulated time.
    """
    snap = load(path)
    stats = snap.get("engine_stats") or {}
    tax = stats.get("last_tax")
    if not tax:
        return 0, 0.0
    steps = int(tax.get("steps") or 0)
    return steps, float(tax.get("avg_ms") or 0.0) * steps


def completion_tokens(rec):
    usage = rec.get("usage") or {}
    if usage.get("completion_tokens"):
        return int(usage["completion_tokens"])
    return len(rec["arrivals"])


def window_tokens(dump):
    return sum(
        completion_tokens(r)
        for r in dump["records"]
        if r["status"] == 200 and r["arrivals"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. 'ring / concurrent (batch=4)'")
    ap.add_argument("--before", required=True, help="models/status snapshot pre-run")
    ap.add_argument("--after", required=True, help="models/status snapshot post-run")
    ap.add_argument("--dump", required=True, help="raw dump from s3_measure.py")
    args = ap.parse_args()

    steps_b, sum_b = tax_point(args.before)
    steps_a, sum_a = tax_point(args.after)
    steps = steps_a - steps_b
    sum_ms = sum_a - sum_b
    tokens = window_tokens(load(args.dump))

    if steps <= 0:
        raise SystemExit(f"no broadcast steps in window (before={steps_b} after={steps_a})")
    if tokens <= 0:
        raise SystemExit("no completion tokens in window")

    per_token = sum_ms / tokens
    per_step = sum_ms / steps
    passed = per_token <= BUDGET_MS_PER_TOKEN

    print(f"{args.label}")
    print(f"  window   : {steps} broadcast steps, {tokens} completion tokens "
          f"({tokens / steps:.2f} tok/step)")
    print(f"  tax      : {sum_ms:.2f} ms total, {per_step:.4f} ms/step")
    print(f"  per-token: {per_token:.4f} ms/token  (budget {BUDGET_MS_PER_TOKEN})")
    print(f"  headroom : {100.0 * per_token / BUDGET_MS_PER_TOKEN:.2f}% of budget")
    print(f"  E4 GATE  : {'PASS' if passed else 'STOP'} "
          f"(pass <=> per-token tax <= {BUDGET_MS_PER_TOKEN})")
    if not passed:
        print("  E4 stop condition FIRED -- halt the slice and report. Not waivable.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
