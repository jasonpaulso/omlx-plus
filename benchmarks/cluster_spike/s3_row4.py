#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The S3 acceptance row-4 gate, over a `flood` dump from s3_measure.py.

Row 4 as written (s3-plan.md, P2/P3): "at least one submission receives 503
while earlier ones continue streaming" -- index-free, so it asserts the two
outcomes coexist, not which request got which.

There was no gate for this before: the 2026-07-29 run judged row 4 by eye from
s3_measure.py's printed status-code histogram plus a manual grep. That is what
let the first ring session record "the cap was never reached" when the real
story was an in-stream error the parser dropped. The rule lives here now.

PASS requires BOTH:
  * at least one record with HTTP 503
  * at least one record that actually streamed (HTTP 200 with >= 1 arrival)

The second clause is not a formality. A head-side preflight gate that is too
aggressive would reject the whole burst -- every request 503, nothing serving
-- which satisfies "a 503 happened" while being strictly worse than the defect
it replaced. That shape must FAIL, and `--selftest` pins it.

In-stream `queue_full` errors are counted and reported but are NOT a failure:
the preflight->submit race is open by design, so rank 0 can still refuse a
request after the response is committed. Seeing a few means the backstop
fired. Seeing them *instead of* any 503 is the pre-fix defect, and the 503
clause already fails that.

Usage:
    python3 s3_row4.py --flood <dump.json>
    python3 s3_row4.py --selftest      # no rig needed; proves the gate can fail
"""

import argparse
import json
import sys


def classify(records):
    """Split a flood dump's records into the outcomes row 4 talks about."""
    rejected_503 = [r for r in records if r.get("status") == 503]
    streaming = [
        r for r in records if r.get("status") == 200 and (r.get("arrivals") or [])
    ]
    # Committed-then-refused: HTTP 200, but the body carried a queue-full
    # error instead of tokens. This is the shape row 4 originally measured.
    in_stream_queue_full = []
    for r in records:
        err = r.get("stream_error")
        if not err:
            continue
        blob = json.dumps(err).lower()
        if "queue" in blob and "full" in blob:
            in_stream_queue_full.append(r)
    other = [
        r
        for r in records
        if r.get("status") not in (200, 503) or r.get("status") is None
    ]
    return rejected_503, streaming, in_stream_queue_full, other


def report(records, *, backend="?", label="flood"):
    rejected, streaming, in_stream, other = classify(records)
    codes = {}
    for r in records:
        codes[r.get("status")] = codes.get(r.get("status"), 0) + 1

    passed = bool(rejected) and bool(streaming)

    print(f"backend: {backend}  ({label}, n={len(records)})")
    print(f"  status codes            : {codes}")
    print(f"  HTTP 503 rejections     : {len(rejected)}")
    print(f"  streamed >=1 token      : {len(streaming)}")
    print(f"  in-stream queue_full    : {len(in_stream)}  (backstop; not a failure)")
    if other:
        print(f"  other/transport failures: {len(other)}")
    print(
        f"  ROW 4 GATE : {'PASS' if passed else 'FAIL'} "
        f"(pass <=> >=1 HTTP 503 AND >=1 request streaming)"
    )
    if not passed:
        if not rejected and in_stream:
            print(
                "    -> the cap fired but only in-stream, under HTTP 200. "
                "This is the original row-4 defect."
            )
        elif not rejected:
            print("    -> no rejection at all: the waiting queue never hit the cap.")
        elif not streaming:
            print(
                "    -> everything was rejected and nothing served: the "
                "admission gate is too aggressive, which is worse than the "
                "defect it replaces."
            )
    return passed


def _selftest():
    """Prove the gate discriminates, before any rig time is spent on it.

    Mirrors how the D7 throughput gate was proven falsifiable: run the shapes
    that must pass and the shapes that must fail, and check the verdicts.
    """
    streamed = {"status": 200, "arrivals": [1.0, 1.1, 1.2], "stream_error": None}

    def committed_then_refused():
        return {
            "status": 200,
            "arrivals": [],
            "stream_error": {"message": "Scheduler waiting queue full: 32 >= 32"},
        }

    cases = [
        (
            "pre-fix shape (40 streaming + 1 in-stream queue_full, no 503)",
            [dict(streamed) for _ in range(40)] + [committed_then_refused()],
            False,
        ),
        (
            "post-fix shape (40 streaming + 1 clean 503)",
            [dict(streamed) for _ in range(40)] + [{"status": 503, "arrivals": []}],
            True,
        ),
        (
            "over-aggressive gate (all 503, nothing served)",
            [{"status": 503, "arrivals": []} for _ in range(41)],
            False,
        ),
        (
            "cap never reached (all streaming)",
            [dict(streamed) for _ in range(41)],
            False,
        ),
        (
            "mixed: 503 plus a backstop firing, still a pass",
            [dict(streamed) for _ in range(39)]
            + [committed_then_refused(), {"status": 503, "arrivals": []}],
            True,
        ),
    ]

    ok = True
    for name, records, expected in cases:
        print(f"--- selftest: {name}")
        got = report(records, backend="selftest", label=name)
        verdict = "ok" if got is expected else "SELFTEST FAILED"
        if got is not expected:
            ok = False
        print(f"  expected {'PASS' if expected else 'FAIL'} -> {verdict}\n")
    print("selftest:", "all cases behaved as specified" if ok else "BROKEN")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flood", help="raw JSON dump from s3_measure.py `flood` mode")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="run the gate against synthetic shapes; no rig required",
    )
    args = ap.parse_args()

    if args.selftest:
        return 0 if _selftest() else 1
    if not args.flood:
        ap.error("--flood is required unless --selftest is given")

    with open(args.flood, encoding="utf-8") as fh:
        dump = json.load(fh)
    backend = dump.get("meta", {}).get("backend") or "?"
    return 0 if report(dump["records"], backend=backend) else 1


if __name__ == "__main__":
    sys.exit(main())
