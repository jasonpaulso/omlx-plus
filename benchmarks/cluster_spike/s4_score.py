#!/usr/bin/env python3
"""S4 P3 acceptance scorer — pinned BEFORE the rig run (DoD falsifiability rule).

Reads a raw capture dump (JSON, schema below) and prints PASS/FAIL per row.
The verdict must be recomputable from the dump without re-running the rig.

Dump schema (all payloads are raw endpoint responses, captured verbatim):
{
  "sha": str, "captured_at": str,
  "row3": {                       # quiesced head, BEFORE the row-1 window
    "quiesced_loaded_models": [str, ...],
    "preview": {PlacementDecision},          # GET /v1/cluster/placement
    "recorded": {PlacementDecision}          # FormationJob.decision via GET /v1/cluster/models/status
  },
  "row1": {
    "loads": {"minimax": {...}, "head_llama": {...}, "worker_llama": {...}},  # raw load responses
    "window": {"streams": [
      {"surface": str, "started_at": float, "first_token_at": float|null,
       "completed_at": float|null, "completion_tokens": int, "error": str|null}
    ]}
  },
  "row2": {
    "pre_evict_cluster_active": bool,        # cluster entry present before trigger
    "evict_trigger_load_ok": bool,           # the over-ceiling local load succeeded
    "post_evict_cluster_active": bool,       # cluster entry present after trigger
    "worker_formation_scrubbed": bool,       # worker reports no live formation/ranks
    "pinned_model": str,
    "pinned_model_loaded_after_trigger": bool,
    "reload": {"preview_mode": str, "status_ready": bool}
  }
}
"""

import json
import sys

DECISION_EQ_FIELDS = ("mode", "world_size", "per_rank_estimate", "divisible")


def _decision_key(d: dict) -> tuple:
    fits_ok = tuple(sorted((k, v.get("ok")) for k, v in (d.get("fits") or {}).items()))
    presence = tuple(sorted((d.get("presence") or {}).items()))
    return tuple(d.get(f) for f in DECISION_EQ_FIELDS) + (fits_ok, presence)


def score_row3(row: dict) -> tuple[bool, str]:
    p, r = row["preview"], row["recorded"]
    if p.get("mode") != "distributed":
        return False, f"preview mode {p.get('mode')!r} != distributed"
    if _decision_key(p) != _decision_key(r):
        return False, f"preview/recorded diverge: {_decision_key(p)} vs {_decision_key(r)}"
    return True, "preview == recorded over placement-determining fields"


def score_row1(row: dict) -> tuple[bool, str]:
    streams = row["window"]["streams"]
    surfaces = {s["surface"] for s in streams}
    want = {"head-minimax", "head-llama", "worker-llama"}
    if surfaces != want:
        return False, f"surfaces {surfaces} != {want}"
    for s in streams:
        if s["error"] is not None:
            return False, f"{s['surface']} errored: {s['error']}"
        if s["completed_at"] is None or s["completion_tokens"] < 16:
            return False, f"{s['surface']} incomplete ({s['completion_tokens']} tokens)"
    # True concurrency: every stream starts before the earliest completion.
    if max(s["started_at"] for s in streams) >= min(s["completed_at"] for s in streams):
        return False, "windows do not overlap (serialized, not concurrent)"
    return True, "3 surfaces streamed concurrently to completion"


def score_row2(row: dict) -> tuple[bool, str]:
    if not row["pre_evict_cluster_active"]:
        return False, "cluster entry absent before trigger (probe invalid)"
    if not row["evict_trigger_load_ok"]:
        return False, "over-ceiling trigger load failed"
    if row["post_evict_cluster_active"]:
        return False, "cluster entry survived the trigger (LRU never evicted it)"
    if not row["worker_formation_scrubbed"]:
        return False, "worker still reports a live formation after unform"
    if not row["pinned_model_loaded_after_trigger"]:
        return False, f"pinned model {row['pinned_model']} was evicted"
    rl = row["reload"]
    if rl["preview_mode"] != "distributed" or not rl["status_ready"]:
        return False, f"reload failed: preview={rl['preview_mode']} ready={rl['status_ready']}"
    return True, "evict/pin interplay + clean unform + reload all held"


def score(dump: dict) -> int:
    rows = [("row3 preview==recorded", score_row3, dump["row3"]),
            ("row1 mixed workload", score_row1, dump["row1"]),
            ("row2 evict/pin interplay", score_row2, dump["row2"])]
    failures = 0
    for name, fn, payload in rows:
        ok, why = fn(payload)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {why}")
        failures += 0 if ok else 1
    print(f"VERDICT: {'PASS' if failures == 0 else f'FAIL ({failures} rows)'}")
    return failures


def _selftest() -> int:
    good = {
        "row3": {"preview": {"mode": "distributed", "world_size": 2, "per_rank_estimate": 5,
                             "divisible": True, "presence": {"a": True}, "fits": {"a": {"ok": True}}},
                 "recorded": {"mode": "distributed", "world_size": 2, "per_rank_estimate": 5,
                              "divisible": True, "presence": {"a": True}, "fits": {"a": {"ok": True}}}},
        "row1": {"window": {"streams": [
            {"surface": "head-minimax", "started_at": 0.0, "first_token_at": 1.0,
             "completed_at": 10.0, "completion_tokens": 32, "error": None},
            {"surface": "head-llama", "started_at": 0.5, "first_token_at": 1.0,
             "completed_at": 9.0, "completion_tokens": 32, "error": None},
            {"surface": "worker-llama", "started_at": 0.7, "first_token_at": 1.2,
             "completed_at": 8.0, "completion_tokens": 32, "error": None}]}},
        "row2": {"pre_evict_cluster_active": True, "evict_trigger_load_ok": True,
                 "post_evict_cluster_active": False, "worker_formation_scrubbed": True,
                 "pinned_model": "llama", "pinned_model_loaded_after_trigger": True,
                 "reload": {"preview_mode": "distributed", "status_ready": True}},
    }
    import copy
    cases = []
    d = copy.deepcopy(good); d["row3"]["recorded"]["mode"] = "local"
    cases.append(("mismatched preview/recorded", d, False))
    d = copy.deepcopy(good); d["row1"]["window"]["streams"][1]["error"] = "boom"
    cases.append(("one stream errored", d, False))
    d = copy.deepcopy(good)
    for i, s in enumerate(d["row1"]["window"]["streams"]):
        s["started_at"], s["completed_at"] = i * 20.0, i * 20.0 + 10.0
    cases.append(("serialized (non-overlapping) window", d, False))
    d = copy.deepcopy(good); d["row2"]["pinned_model_loaded_after_trigger"] = False
    cases.append(("pinned model evicted", d, False))
    d = copy.deepcopy(good); d["row2"]["post_evict_cluster_active"] = True
    cases.append(("cluster entry survived trigger", d, False))
    cases.append(("all-good dump", good, True))

    bad = 0
    for name, dump, want_pass in cases:
        print(f"--- selftest: {name} (expect {'PASS' if want_pass else 'FAIL'})")
        failures = score(dump)
        ok = (failures == 0) == want_pass
        if not ok:
            print(f"SELFTEST BROKEN on: {name}")
            bad += 1
    print(f"SELFTEST: {'OK — gate is falsifiable' if bad == 0 else f'BROKEN ({bad})'}")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    with open(sys.argv[1]) as f:
        sys.exit(1 if score(json.load(f)) else 0)
