#!/usr/bin/env python3
"""S6 P3 acceptance-anchor scorer — pinned BEFORE the rig run (DoD rule).

Reads one raw dump per cell (JSON, produced by `s6_anchors.py`) and prints
PASS/FAIL per gate. The verdict must be recomputable from the dumps without
re-running the rig.

A "cell" is the tuple (anchor, model, node_mode, backend, batch) carried in
each dump's `meta`. Two gates, per the D4 protocol:

  capacity gate: for ONE model (the capacity anchor, MiniMax-M2.7-3bit —
    never hardcoded here, read from the dumps), distributed cells
    {ring,jaccl} x {batch1,batch4} must ALL be present. Gate = the BEST of
    those 4 cells' decode rates >= 43.0 tok/s.

  speedup gate: for ONE model (the speedup anchor — parameterized, whatever
    model the caller measured), single_node cells must cover {batch1,batch4}
    and distributed cells (any backend(s)) must cover {batch1,batch4}. Gate =
    best distributed cell's rate >= 1.3 x best single_node cell's rate, and
    every contributing record hit >=256 completion tokens (steady state).

"Best" is a max across DISTINCT cells. Within one cell it is never the best
attempt — it is the first attempt that actually completed (D4's no-retry
rule): a cell with more than one dump is scored from the earliest dump (by
`attempt`) that streamed successfully; later dumps are provenance only, and
if a later dump's `infra_error` is empty, or an earlier attempt in the same
cell already completed, that is itself a scored violation (a "repeat for a
better number" shape), independent of what the retried number turns out to
be.

Missing cells FAIL the gate they belong to; they are never silently skipped
(the speedup anchor's model is still a user decision at time of writing --
use `--gate capacity` to score the capacity anchor alone; the tool prints
plainly that speedup was excluded, never an implicit pass).

E4 coordination-tax is re-checked per cell when a dump carries tax_before/
tax_after (capacity anchor cells always do; single-node speedup cells do
not -- no cluster role, printed as n/a): per-token tax must stay under the
S0 budget (`s3_tax.BUDGET_MS_PER_TOKEN`). This is a live stop condition, not
part of either anchor's pass/fail row, but it adds to the failure count
because it is not waivable per AUTONOMY.md.

Dump schema: see s6_anchors.py's module docstring (produced by that script).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_tax import BUDGET_MS_PER_TOKEN  # noqa: E402 (reuse the pinned E4 budget)

CAPACITY_GATE_TOK_S = 43.0
SPEEDUP_RATIO = 1.3
SPEEDUP_STEADY_STATE_TOKENS = 256


def completion_tokens(rec: dict) -> int:
    """Prefer server usage; floor via chars/4 -- SSE arrival-count under-counts
    coalesced/reasoning tokens (see s5_stream_probe.py convention)."""
    usage = rec.get("usage") or {}
    reported = int(usage.get("completion_tokens") or 0)
    floor = int(rec.get("text_len", 0) or 0) + int(rec.get("reasoning_len", 0) or 0)
    return max(reported, floor // 4)


def live_records(dump: dict) -> list[dict]:
    return [r for r in dump["records"] if r["status"] == 200 and r["arrivals"]]


def cell_key(meta: dict) -> tuple:
    return (
        meta["anchor"],
        meta["model"],
        meta["node_mode"],
        meta["backend"],
        meta["batch"],
    )


def cell_rate(dump: dict) -> dict:
    """One cell's decode rate: (n-1)/span for batch=1 (baseline formula),
    Sum(tokens)/window for batch=4 (S3 D7 aggregate formula). Pre-validates
    spans instead of raising, so a zero-span shape FAILs cleanly rather than
    aborting the whole scorer."""
    records = live_records(dump)
    if not records:
        return {"ok": False, "reason": "no successful streaming requests"}
    batch = dump["meta"]["batch"]
    if batch == 1:
        rec = records[0]
        n = completion_tokens(rec)
        span = rec["arrivals"][-1] - rec["arrivals"][0]
        if span <= 0:
            return {"ok": False, "reason": "zero arrival span"}
        return {
            "ok": True,
            "rate": (n - 1) / span,
            "tokens": n,
            "span": span,
            "records": records,
        }
    total = sum(completion_tokens(r) for r in records)
    first = min(r["arrivals"][0] for r in records)
    last = max(r["arrivals"][-1] for r in records)
    span = last - first
    if span <= 0:
        return {"ok": False, "reason": "zero concurrent window"}
    return {
        "ok": True,
        "rate": total / span,
        "tokens": total,
        "span": span,
        "records": records,
    }


def resolve_cell(dumps: list[dict]) -> tuple[dict | None, list[str]]:
    """Apply the D4 no-retry rule across dumps sharing one cell key.

    Returns (canonical_rate_or_None, violation_messages). The canonical rate
    is the first dump (by `attempt`) that completed -- never the fastest.
    """
    violations = []
    ordered = sorted(dumps, key=lambda d: d["meta"]["attempt"])
    canonical = None
    any_prior_completed = False
    for d in ordered:
        attempt = d["meta"]["attempt"]
        infra_error = d["meta"].get("infra_error")
        if attempt > 1:
            if not infra_error:
                violations.append(
                    f"attempt {attempt} has no infra_error recorded "
                    "(D4 no-retry rule violation)"
                )
            if any_prior_completed:
                violations.append(
                    f"attempt {attempt} repeats a cell whose earlier attempt "
                    "already completed -- retries are only for a failed attempt"
                )
        rate = cell_rate(d)
        if rate["ok"]:
            any_prior_completed = True
            if canonical is None:
                canonical = rate
    return canonical, violations


def tax_check(dump: dict, tokens: int) -> tuple[str, str]:
    """('pass'|'fail'|'n/a', message) for one cell's E4 per-token tax."""
    before = dump["meta"].get("tax_before")
    after = dump["meta"].get("tax_after")
    if not before or not after:
        return "n/a", dump["meta"].get("tax_note") or "no tax snapshot captured"
    steps = after["steps"] - before["steps"]
    if steps <= 0 or tokens <= 0:
        return "n/a", f"no broadcast steps in window (before={before} after={after})"
    sum_ms = after["avg_ms"] * after["steps"] - before["avg_ms"] * before["steps"]
    per_token = sum_ms / tokens
    ok = per_token <= BUDGET_MS_PER_TOKEN
    return (
        "pass" if ok else "fail",
        f"{per_token:.4f} ms/token (budget {BUDGET_MS_PER_TOKEN}, {steps} steps)",
    )


def _group(dumps: list[dict], anchor: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for d in dumps:
        if d["meta"]["anchor"] != anchor:
            continue
        groups.setdefault(cell_key(d["meta"]), []).append(d)
    return groups


def score_capacity(dumps: list[dict]) -> tuple[int, list[str]]:
    lines = []
    failures = 0
    cap_dumps = [d for d in dumps if d["meta"]["anchor"] == "capacity"]
    if not cap_dumps:
        return 1, ["FAIL capacity: no capacity dumps provided"]
    models = {d["meta"]["model"] for d in cap_dumps}
    if len(models) != 1:
        return 1, [f"FAIL capacity: dumps span multiple models {sorted(models)}"]
    model = models.pop()
    groups = _group(cap_dumps, "capacity")
    best = None
    for backend in ("ring", "jaccl"):
        for batch in (1, 4):
            key = ("capacity", model, "distributed", backend, batch)
            cell_dumps = groups.get(key)
            if not cell_dumps:
                lines.append(
                    f"FAIL capacity cell backend={backend} batch={batch}: missing"
                )
                failures += 1
                continue
            canonical, violations = resolve_cell(cell_dumps)
            for v in violations:
                lines.append(f"FAIL capacity cell backend={backend} batch={batch}: {v}")
                failures += 1
            if canonical is None:
                lines.append(
                    f"FAIL capacity cell backend={backend} batch={batch}: "
                    "no dump in this cell completed"
                )
                failures += 1
                continue
            lines.append(
                f"      capacity cell backend={backend} batch={batch}: "
                f"{canonical['rate']:.4f} tok/s ({canonical['tokens']} tok / "
                f"{canonical['span']:.4f}s)"
            )
            if best is None or canonical["rate"] > best[0]:
                best = (canonical["rate"], backend, batch, canonical)
            tax_status, tax_msg = tax_check(cell_dumps[0], canonical["tokens"])
            if tax_status == "fail":
                lines.append(
                    f"FAIL capacity cell backend={backend} batch={batch} E4 tax: {tax_msg}"
                )
                failures += 1
            elif tax_status == "pass":
                lines.append(
                    f"      capacity cell backend={backend} batch={batch} E4 tax: PASS {tax_msg}"
                )
            else:
                lines.append(
                    f"      capacity cell backend={backend} batch={batch} E4 tax: n/a ({tax_msg})"
                )
    if best is None:
        lines.append("FAIL capacity GATE: no cell produced a usable rate")
        failures += 1
    else:
        passed = best[0] >= CAPACITY_GATE_TOK_S
        lines.append(
            f"{'PASS' if passed else 'FAIL'} capacity GATE: best cell "
            f"{best[0]:.4f} tok/s (backend={best[1]} batch={best[2]}) "
            f"{'>=' if passed else '<'} {CAPACITY_GATE_TOK_S} tok/s"
        )
        if not passed:
            failures += 1
    return failures, lines


def score_speedup(dumps: list[dict]) -> tuple[int, list[str]]:
    lines = []
    failures = 0
    spd_dumps = [d for d in dumps if d["meta"]["anchor"] == "speedup"]
    if not spd_dumps:
        return 1, ["FAIL speedup: no speedup dumps provided"]
    models = {d["meta"]["model"] for d in spd_dumps}
    if len(models) != 1:
        return 1, [f"FAIL speedup: dumps span multiple models {sorted(models)}"]
    model = models.pop()

    def best_for(
        node_mode: str,
    ) -> tuple[tuple[float | None, dict | None, list[str]], int]:
        batches_seen: set[int] = set()
        local_lines: list[str] = []
        local_failures: list[int] = []
        best: tuple[float, int, str] | None = None
        for d in spd_dumps:
            m = d["meta"]
            if m["node_mode"] != node_mode:
                continue
            batches_seen.add(m["batch"])
        for batch in (1, 4):
            if batch not in batches_seen:
                local_lines.append(f"FAIL speedup {node_mode} batch={batch}: missing")
                local_failures.append(1)
                continue
            group_dumps = [
                d
                for d in spd_dumps
                if d["meta"]["node_mode"] == node_mode and d["meta"]["batch"] == batch
            ]
            by_cell: dict[tuple, list[dict]] = {}
            for d in group_dumps:
                by_cell.setdefault(cell_key(d["meta"]), []).append(d)
            for key, cell_dumps in by_cell.items():
                canonical, violations = resolve_cell(cell_dumps)
                for v in violations:
                    local_lines.append(
                        f"FAIL speedup {node_mode} batch={batch} backend={key[3]}: {v}"
                    )
                    local_failures.append(1)
                if canonical is None:
                    local_lines.append(
                        f"FAIL speedup {node_mode} batch={batch} backend={key[3]}: "
                        "no dump in this cell completed"
                    )
                    local_failures.append(1)
                    continue
                low = [
                    completion_tokens(r)
                    for r in canonical["records"]
                    if completion_tokens(r) < SPEEDUP_STEADY_STATE_TOKENS
                ]
                if low:
                    local_lines.append(
                        f"FAIL speedup {node_mode} batch={batch} backend={key[3]}: "
                        f"steady-state violation, record(s) below "
                        f"{SPEEDUP_STEADY_STATE_TOKENS} tokens: {low}"
                    )
                    local_failures.append(1)
                    continue
                local_lines.append(
                    f"      speedup {node_mode} batch={batch} backend={key[3]}: "
                    f"{canonical['rate']:.4f} tok/s"
                )
                if best is None or canonical["rate"] > best[0]:
                    best = (canonical["rate"], batch, key[3])
        return (best[0] if best else None, {"detail": best}, local_lines), sum(
            local_failures
        )

    (single_best, _single_detail, single_lines), single_fail = best_for("single_node")
    lines.extend(single_lines)
    failures += single_fail
    (dist_best, _dist_detail, dist_lines), dist_fail = best_for("distributed")
    lines.extend(dist_lines)
    failures += dist_fail

    if single_best is None or dist_best is None:
        lines.append(
            "FAIL speedup GATE: single_node or distributed best is unavailable"
        )
        failures += 1
    else:
        ratio = dist_best / single_best
        passed = ratio >= SPEEDUP_RATIO
        lines.append(
            f"{'PASS' if passed else 'FAIL'} speedup GATE: model={model} "
            f"distributed_best={dist_best:.4f} tok/s single_best={single_best:.4f} tok/s "
            f"ratio={ratio:.3f}x {'>=' if passed else '<'} {SPEEDUP_RATIO}x"
        )
        if not passed:
            failures += 1
    return failures, lines


def score(dumps: list[dict], gates: tuple[str, ...] = ("capacity", "speedup")) -> int:
    total_failures = 0
    for gate in ("capacity", "speedup"):
        if gate not in gates:
            print(f"SKIP {gate} gate (excluded by --gate {','.join(gates)})")
            continue
        fn = score_capacity if gate == "capacity" else score_speedup
        failures, lines = fn(dumps)
        for line in lines:
            print(line)
        total_failures += failures
    print(
        f"VERDICT: {'PASS' if total_failures == 0 else f'FAIL ({total_failures} rows)'}"
    )
    return total_failures


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return cast(dict, json.load(fh))


# ---------------------------------------------------------------------------
# selftest: synthetic dumps, no filesystem, must-fail shapes per house rule.
# ---------------------------------------------------------------------------


def _mk_single(
    anchor,
    model,
    node_mode,
    backend,
    rate,
    tokens,
    attempt=1,
    infra_error=None,
    tax_before=None,
    tax_after=None,
):
    span = (tokens - 1) / rate
    dt = span / (tokens - 1)
    arrivals = [i * dt for i in range(tokens)]
    rec = {
        "label": "single",
        "status": 200,
        "error": None,
        "arrivals": arrivals,
        "text_len": tokens * 4,
        "reasoning_len": 0,
        "usage": {"completion_tokens": tokens},
        "finish_reason": "stop",
        "aborted": False,
    }
    return {
        "meta": {
            "anchor": anchor,
            "model": model,
            "node_mode": node_mode,
            "backend": backend,
            "batch": 1,
            "max_tokens": 128 if anchor == "capacity" else 256,
            "prompt_file": "s3_prompt.txt",
            "prompt_chars": 3840,
            "prompt_token_floor": 960,
            "attempt": attempt,
            "infra_error": infra_error,
            "tax_before": tax_before,
            "tax_after": tax_after,
            "tax_note": None,
            "url": "http://test",
            "wall_start": 0.0,
            "wall_end": 1.0,
        },
        "records": [rec],
    }


def _mk_batch4(
    anchor,
    model,
    node_mode,
    backend,
    rate,
    tokens_each,
    attempt=1,
    infra_error=None,
    tax_before=None,
    tax_after=None,
):
    total = tokens_each * 4
    span = total / rate
    dt = span / (tokens_each - 1) if tokens_each > 1 else 0
    arrivals = [i * dt for i in range(tokens_each)]
    if arrivals:
        arrivals[-1] = span  # pin the aggregate window's last arrival exactly
    records = []
    for i in range(4):
        records.append(
            {
                "label": f"c{i}",
                "status": 200,
                "error": None,
                "arrivals": list(arrivals),
                "text_len": tokens_each * 4,
                "reasoning_len": 0,
                "usage": {"completion_tokens": tokens_each},
                "finish_reason": "stop",
                "aborted": False,
            }
        )
    return {
        "meta": {
            "anchor": anchor,
            "model": model,
            "node_mode": node_mode,
            "backend": backend,
            "batch": 4,
            "max_tokens": 128 if anchor == "capacity" else 256,
            "prompt_file": "s3_prompt.txt",
            "prompt_chars": 3840,
            "prompt_token_floor": 960,
            "attempt": attempt,
            "infra_error": infra_error,
            "tax_before": tax_before,
            "tax_after": tax_after,
            "tax_note": None,
            "url": "http://test",
            "wall_start": 0.0,
            "wall_end": 1.0,
        },
        "records": records,
    }


def _good_capacity_dumps(rate=50.0):
    return [
        _mk_single("capacity", "minimax", "distributed", "ring", rate, 128),
        _mk_batch4("capacity", "minimax", "distributed", "ring", rate + 5, 128),
        _mk_single("capacity", "minimax", "distributed", "jaccl", rate + 2, 128),
        _mk_batch4("capacity", "minimax", "distributed", "jaccl", rate + 8, 128),
    ]


def _good_speedup_dumps(single_rate=20.0, dist_rate=30.0):
    return [
        _mk_single("speedup", "qwen", "single_node", "n/a", single_rate, 256),
        _mk_batch4("speedup", "qwen", "single_node", "n/a", single_rate + 2, 256),
        _mk_single("speedup", "qwen", "distributed", "ring", dist_rate, 256),
        _mk_batch4("speedup", "qwen", "distributed", "ring", dist_rate + 2, 256),
    ]


def _selftest() -> int:
    cases: list[tuple[str, list[dict], tuple[str, ...], bool]] = []

    cases.append(
        (
            "all-good capacity+speedup",
            _good_capacity_dumps() + _good_speedup_dumps(),
            ("capacity", "speedup"),
            True,
        )
    )

    low = _good_capacity_dumps(rate=10.0)  # every cell well under 43 tok/s
    cases.append(("capacity best-cell below threshold", low, ("capacity",), False))

    weak_speedup = [
        _mk_single("speedup", "qwen", "single_node", "n/a", 30.0, 256),
        _mk_batch4("speedup", "qwen", "single_node", "n/a", 30.0, 256),
        _mk_single("speedup", "qwen", "distributed", "ring", 32.0, 256),  # ratio 1.07x
        _mk_batch4("speedup", "qwen", "distributed", "ring", 32.0, 256),
    ]
    cases.append(("speedup ratio below 1.3x", weak_speedup, ("speedup",), False))

    missing = _good_capacity_dumps()[:-1]  # drop jaccl batch4
    cases.append(
        ("capacity missing cell (jaccl batch4)", missing, ("capacity",), False)
    )

    retry_bad = _good_capacity_dumps()
    retry_dump = _mk_single(
        "capacity",
        "minimax",
        "distributed",
        "ring",
        60.0,
        128,
        attempt=2,
        infra_error=None,
    )
    cases.append(
        (
            "capacity no-retry violation (retry with no infra_error)",
            retry_bad + [retry_dump],
            ("capacity",),
            False,
        )
    )

    retry_after_success = _good_capacity_dumps()
    retry_dump2 = _mk_single(
        "capacity",
        "minimax",
        "distributed",
        "ring",
        90.0,
        128,
        attempt=2,
        infra_error="wanted a better number",
    )
    cases.append(
        (
            "capacity no-retry violation (retry after a completed attempt)",
            retry_after_success + [retry_dump2],
            ("capacity",),
            False,
        )
    )

    unsteady = [
        _mk_single("speedup", "qwen", "single_node", "n/a", 20.0, 256),
        _mk_batch4("speedup", "qwen", "single_node", "n/a", 20.0, 256),
        _mk_single("speedup", "qwen", "distributed", "ring", 40.0, 200),  # below 256
        _mk_batch4("speedup", "qwen", "distributed", "ring", 40.0, 256),
    ]
    cases.append(
        (
            "speedup steady-state violation (200 < 256 tokens)",
            unsteady,
            ("speedup",),
            False,
        )
    )

    hi_tax = _good_capacity_dumps()
    hi_tax[0] = copy.deepcopy(hi_tax[0])
    hi_tax[0]["meta"]["tax_before"] = {"steps": 0, "avg_ms": 0.0}
    hi_tax[0]["meta"]["tax_after"] = {"steps": 10, "avg_ms": 200.0}  # way over budget
    cases.append(("capacity E4 tax over budget", hi_tax, ("capacity",), False))

    bad = 0
    for name, dumps, gates, want_pass in cases:
        print(f"--- selftest: {name} (expect {'PASS' if want_pass else 'FAIL'})")
        failures = score(dumps, gates)
        ok = (failures == 0) == want_pass
        if not ok:
            print(f"SELFTEST BROKEN on: {name}")
            bad += 1
        print()
    print(
        f"SELFTEST: {'OK — gates are falsifiable' if bad == 0 else f'BROKEN ({bad})'}"
    )
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump", action="append", default=[], help="raw dump path (repeatable)"
    )
    ap.add_argument(
        "--gate",
        default="all",
        choices=["all", "capacity", "speedup"],
        help="score only this gate (default: both)",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    if not args.dump:
        raise SystemExit(
            "--dump is required (repeat for multiple cells), or use --selftest"
        )
    dumps = [load(p) for p in args.dump]
    gates = ("capacity", "speedup") if args.gate == "all" else (args.gate,)
    return 1 if score(dumps, gates) else 0


if __name__ == "__main__":
    sys.exit(main())
