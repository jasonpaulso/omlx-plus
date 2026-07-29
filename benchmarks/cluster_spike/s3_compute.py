#!/usr/bin/env python3
"""Apply the D7 throughput gate to raw dumps from s3_measure.py.

The formula is transcribed verbatim from the approved S3 plan (revision 4/5),
and is deliberately NOT the intuitive one -- a sum of per-request decode rates
is maximized by FIFO serialization and therefore cannot fail:

  baseline  = (completion_tokens - 1) / (last_arrival - first_arrival)
              for the single request

  aggregate = (sum of completion_tokens across the N requests)
              / (last arrival across ALL requests - first arrival across ALL)

Window boundaries are token-ARRIVAL times, not submit times, so inter-request
prefill/queue gaps fall inside the concurrent window.

  Pass <=> aggregate >= baseline, per backend.
"""

from __future__ import annotations

import argparse
import json
import sys


def completion_tokens(rec):
    """Plan's `completion_tokens`, preferring server-reported usage."""
    usage = rec.get("usage") or {}
    if usage.get("completion_tokens"):
        return int(usage["completion_tokens"]), "usage"
    return len(rec["arrivals"]), "arrival-count"


def baseline_rate(rec):
    n, src = completion_tokens(rec)
    span = rec["arrivals"][-1] - rec["arrivals"][0]
    if span <= 0:
        raise SystemExit("baseline arrival span is zero; cannot compute")
    return (n - 1) / span, n, span, src


def aggregate_rate(records):
    live = [r for r in records if r["status"] == 200 and r["arrivals"]]
    if not live:
        raise SystemExit("no successful streaming requests in dump")
    total = 0
    srcs = set()
    for r in live:
        n, src = completion_tokens(r)
        total += n
        srcs.add(src)
    first = min(r["arrivals"][0] for r in live)
    last = max(r["arrivals"][-1] for r in live)
    span = last - first
    if span <= 0:
        raise SystemExit("concurrent arrival window is zero; cannot compute")
    return total / span, total, span, len(live), sorted(srcs)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="raw dump from `single` mode")
    ap.add_argument("--concurrent", required=True, help="raw dump from `concurrent` mode")
    args = ap.parse_args()

    single = load(args.single)
    conc = load(args.concurrent)

    srec = single["records"][0]
    if srec["status"] != 200 or not srec["arrivals"]:
        raise SystemExit(f"single-request run did not stream: status={srec['status']}")

    base, bn, bspan, bsrc = baseline_rate(srec)
    agg, total, aspan, nlive, asrcs = aggregate_rate(conc["records"])

    backend = single["meta"].get("backend") or conc["meta"].get("backend") or "?"
    passed = agg >= base

    print(f"backend: {backend}")
    print(f"  baseline : {base:.4f} tok/s  "
          f"[({bn} - 1) tokens / {bspan:.4f} s, source={bsrc}]")
    print(f"  aggregate: {agg:.4f} tok/s  "
          f"[{total} tokens over {nlive} requests / {aspan:.4f} s window, "
          f"source={','.join(asrcs)}]")
    print(f"  ratio    : {agg / base:.3f}x")
    print(f"  D7 GATE  : {'PASS' if passed else 'FAIL'} (pass <=> aggregate >= baseline)")

    # Cross-check: the serialization-blind formula the plan rejected. Printed
    # only so the doc can show why the chosen formula is the discriminating one.
    live = [r for r in conc["records"] if r["status"] == 200 and r["arrivals"]]
    naive = 0.0
    for r in live:
        n, _ = completion_tokens(r)
        span = r["arrivals"][-1] - r["arrivals"][0]
        if span > 0:
            naive += (n - 1) / span
    print(f"  (rejected sum-of-rates formula, context only: {naive:.4f} tok/s)")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
