#!/usr/bin/env python3
"""S6 P3 acceptance-anchor measurement driver — pinned BEFORE the rig run.

Captures RAW SSE token-arrival timestamps (reusing `s3_measure.stream_request`
/ `run_parallel`, the house convention: capture and arithmetic stay apart so a
gate can be recomputed without re-running the rig) for ONE cell of ONE anchor
per invocation, plus a before/after E4 coordination-tax snapshot from
`GET /v1/cluster/models/status` in the SAME dump. `s6_score.py` applies the
D4 gates; this script only measures and records.

D4 pins two anchors (`discovery/spec/s6-plan.md`):

  capacity anchor  — MiniMax-M2.7-3bit, distributed, per backend (ring,
                      jaccl): single-request decode rate AND batch-4
                      wall-clock aggregate. Decode-only, >=512-token pinned
                      prompt, max_tokens=128 (fixed, enforced below).
  speedup anchor   — a caller-supplied model (NEVER hardcoded here — the
                      anchor model is under user decision, plan acceptance
                      row 2): best single-node cell vs best distributed
                      cell, batch-1 and batch-4, >=256-token steady state
                      (max_tokens>=256, enforced below).

A "cell" is the tuple (anchor, model, node_mode, backend, batch). One dump
file = one cell = one invocation. The no-retry rule ("first completed
measurement per cell is the number; repeats only on infra error, logged in
the dump") is RECORDED here via `--attempt`/`--infra-error` and ENFORCED by
`s6_score.py` across a set of dumps that share a cell key — this driver never
refuses to write a dump; it only tells the truth about what attempt it is.

Dump schema (consumed by s6_score.py):
{
  "meta": {
    "anchor": "capacity" | "speedup",
    "model": str,
    "node_mode": "distributed" | "single_node",
    "backend": "ring" | "jaccl" | "n/a",
    "batch": 1 | 4,
    "max_tokens": int,
    "prompt_file": str, "prompt_chars": int, "prompt_token_floor": int,
    "attempt": int,                  # 1 = first measurement of this cell
    "infra_error": str | null,       # non-empty iff attempt > 1 and legit
    "tax_before": {"steps": int, "avg_ms": float} | null,
    "tax_after": {"steps": int, "avg_ms": float} | null,
    "tax_note": str | null,          # why tax is null (e.g. no cluster role)
    "url": str, "wall_start": float, "wall_end": float
  },
  "records": [ ... same shape as s3_measure.py's records ... ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_measure  # noqa: E402  (reuse raw SSE capture, house convention)

CLOCK = time.perf_counter
DEFAULT_PROMPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "s3_prompt.txt"
)
CAPACITY_MAX_TOKENS = 128
CAPACITY_MIN_PROMPT_TOKENS = 512
SPEEDUP_MIN_MAX_TOKENS = 256


def _tax_snapshot(url, api_key, timeout=10):
    """Read {steps, avg_ms} from `GET /v1/cluster/models/status`.

    Returns (tax_dict_or_None, note). `last_tax` is null until the first
    request completes against an active formation (correct zero-point for a
    pre-run snapshot); a single-node run (no cluster role, or role=off) 404s
    -- that is not an error, just "no tax to report", recorded as a note
    rather than aborting the measurement.
    """
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/cluster/models/status", method="GET"
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    ctx = ssl.create_default_context() if url.startswith("https") else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            snap = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return None, f"models/status HTTP {exc.code} (no active cluster formation?)"
    except (urllib.error.URLError, OSError) as exc:
        return None, f"models/status unreachable: {type(exc).__name__}: {exc}"
    tax = (snap.get("engine_stats") or {}).get("last_tax")
    if not tax:
        return {"steps": 0, "avg_ms": 0.0}, None
    return {
        "steps": int(tax.get("steps") or 0),
        "avg_ms": float(tax.get("avg_ms") or 0.0),
    }, None


def build_body(model, prompt, max_tokens):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("anchor", choices=["capacity", "speedup"])
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8910")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--node-mode", required=True, choices=["distributed", "single_node"]
    )
    ap.add_argument("--backend", default="n/a", choices=["ring", "jaccl", "n/a"])
    ap.add_argument("--batch", type=int, required=True, choices=[1, 4])
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="1 = first measurement of this cell; >1 must pair with --infra-error",
    )
    ap.add_argument(
        "--infra-error",
        default="",
        help="required (non-empty) when --attempt > 1: the infra failure that "
        "justifies re-measuring this cell (D4 no-retry rule)",
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True, help="raw JSON dump path")
    args = ap.parse_args()

    if args.anchor == "capacity":
        if args.node_mode != "distributed":
            raise SystemExit("capacity anchor is distributed-only (D4)")
        if args.backend not in ("ring", "jaccl"):
            raise SystemExit("capacity anchor requires --backend ring|jaccl (D4)")
        if args.max_tokens != CAPACITY_MAX_TOKENS:
            raise SystemExit(
                f"capacity anchor pins max_tokens={CAPACITY_MAX_TOKENS} (D4), "
                f"got {args.max_tokens}"
            )
    else:  # speedup
        if args.max_tokens < SPEEDUP_MIN_MAX_TOKENS:
            raise SystemExit(
                f"speedup anchor requires max_tokens>={SPEEDUP_MIN_MAX_TOKENS} "
                f"for steady-state (D4), got {args.max_tokens}"
            )
        if args.node_mode == "single_node" and args.backend != "n/a":
            raise SystemExit("single_node speedup cells have no backend (use n/a)")
        if args.node_mode == "distributed" and args.backend not in ("ring", "jaccl"):
            raise SystemExit("distributed speedup cells require --backend ring|jaccl")

    if args.attempt > 1 and not args.infra_error.strip():
        raise SystemExit(
            "--attempt > 1 requires a non-empty --infra-error (D4 no-retry rule); "
            "the scorer FAILS a retry recorded without one"
        )

    with open(args.prompt_file, encoding="utf-8") as fh:
        prompt = fh.read()
    prompt_token_floor = len(prompt) // 4
    if args.anchor == "capacity" and prompt_token_floor < CAPACITY_MIN_PROMPT_TOKENS:
        raise SystemExit(
            f"capacity anchor requires a >={CAPACITY_MIN_PROMPT_TOKENS}-token "
            f"pinned prompt (D4); {args.prompt_file} floors at "
            f"{prompt_token_floor} tokens (chars//4)"
        )

    tax_before, tax_note_before = _tax_snapshot(args.url, args.api_key)

    body = build_body(args.model, prompt, args.max_tokens)
    if args.batch == 1:
        records = [
            s3_measure.stream_request(
                args.url.rstrip("/") + "/v1/chat/completions",
                args.api_key,
                body,
                "single",
                timeout=args.timeout,
            )
        ]
    else:
        url = args.url.rstrip("/") + "/v1/chat/completions"
        records = s3_measure.run_parallel(
            [
                (
                    lambda i=i: s3_measure.stream_request(
                        url, args.api_key, dict(body), f"c{i}", timeout=args.timeout
                    )
                )
                for i in range(4)
            ]
        )

    tax_after, tax_note_after = _tax_snapshot(args.url, args.api_key)
    tax_note = tax_note_before or tax_note_after

    meta = {
        "anchor": args.anchor,
        "model": args.model,
        "node_mode": args.node_mode,
        "backend": args.backend,
        "batch": args.batch,
        "max_tokens": args.max_tokens,
        "prompt_file": args.prompt_file,
        "prompt_chars": len(prompt),
        "prompt_token_floor": prompt_token_floor,
        "attempt": args.attempt,
        "infra_error": args.infra_error.strip() or None,
        "tax_before": tax_before,
        "tax_after": tax_after,
        "tax_note": tax_note,
        "url": args.url,
        "wall_start": time.time(),
    }
    meta["wall_end"] = time.time()
    dump = {"meta": meta, "records": records}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=2)

    ok = sum(1 for r in records if r["status"] == 200)
    cell = (
        f"{args.anchor}/{args.model}/{args.node_mode}/{args.backend}/batch{args.batch}"
    )
    print(f"cell={cell} attempt={args.attempt} n={len(records)} ok={ok}")
    for r in records:
        print(
            f"  {r['label']}: status={r['status']} tokens_arrivals={len(r['arrivals'])} "
            f"usage={r['usage']} finish={r['finish_reason']} "
            f"err={(r['error'] or '')[:120]}"
        )
    print(f"tax_before={tax_before} tax_after={tax_after} note={tax_note}")
    print(f"raw dump -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
