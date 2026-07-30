#!/usr/bin/env python3
"""S5 P3 acceptance scorer — pinned BEFORE the rig run (DoD falsifiability rule).

Reads a raw capture dump (JSON, schema below) and prints PASS/FAIL per row.
The verdict must be recomputable from the dump without re-running the rig.

Dump schema:
{
  "sha": str, "captured_at": str,
  "item1": {                # transfer + form, zero preemptive action
    "model_id": str,
    "manifest_entry_count": int,
    "load_accepted": bool,             # plain distributed load API call accepted
    "preemptive_user_actions": [str],  # anything the operator had to do on the worker first; must be []
    "job_reached_done": bool,          # transfer job terminal state == done on the successful journey
    "formation_ready": bool,
    "post_form_stream_tokens": int,
    "worker_discovered_id": str,       # id discovery returns on the worker after transfer
    "worker_final_dir_file_count": int
  },
  "item2": {                # kill worker mid-flight, file-granular resume
    "killed_mid_flight": bool,
    "verified_before_kill": [{"path": str, "size": int, "mtime_ns": int}, ...],
    "verified_after_resume": [{"path": str, "size": int, "mtime_ns": int}, ...],
    "resent_paths": [str, ...],        # files re-sent by post-resume rounds
    "exempt_paths": [str, ...]         # deliberately corrupted files (item3) — excluded from the no-resend/mtime checks
  },
  "item3": {                # corruption detected by digest, re-fetched
    "corrupted_path": str,
    "corrupted_sha_observed": str,
    "manifest_sha": str,
    "final_sha_after_resume": str,
    "refetched": bool
  },
  "item4": {                # HF fan-out path
    "both_viable_choice_required": bool,  # omitted source produced a typed choice_required
    "explicit_source": str,               # must be "hf"
    "revision": str,                      # worker download revision
    "head_revision": str,                 # head's resolved revision
    "required_set_count": int,
    "final_dir_listing": [str, ...],
    "digest_verified_paths": [str, ...],
    "formation_ready": bool
  },
  "residual_cl5_16": {      # OPTIONAL row — wedged-peer watchdog probe
    "wedged": bool,
    "watchdog_fired_before_deadline": bool,
    "gate_released": bool
  }
}
"""

import json
import re
import sys

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def score_item1(row: dict) -> tuple[bool, str]:
    if not row["load_accepted"]:
        return False, "distributed load API call was not accepted"
    if row["preemptive_user_actions"] != []:
        return False, f"operator had to act first: {row['preemptive_user_actions']}"
    if row["manifest_entry_count"] <= 0:
        return False, f"manifest_entry_count {row['manifest_entry_count']} <= 0"
    if not row["job_reached_done"]:
        return False, "transfer job did not reach terminal state done"
    if not row["formation_ready"]:
        return False, "formation not ready after transfer"
    if row["post_form_stream_tokens"] < 16:
        return (
            False,
            f"post-form stream produced {row['post_form_stream_tokens']} tokens (<16)",
        )
    if row["worker_discovered_id"] != row["model_id"]:
        return False, (
            f"worker discovered id {row['worker_discovered_id']!r} "
            f"!= model_id {row['model_id']!r}"
        )
    if row["worker_final_dir_file_count"] != row["manifest_entry_count"]:
        return False, (
            f"worker final dir has {row['worker_final_dir_file_count']} files "
            f"!= manifest {row['manifest_entry_count']}"
        )
    return True, "transfer + form completed with zero preemptive operator action"


def score_item2(row: dict) -> tuple[bool, str]:
    if not row["killed_mid_flight"]:
        return False, "worker was not actually killed mid-flight"
    before = row["verified_before_kill"]
    if not before:
        return False, "verified_before_kill is empty (nothing to check)"
    resent = row["resent_paths"]
    if not resent:
        return False, "resent_paths is empty (resume never re-sent anything)"
    exempt = set(row["exempt_paths"])
    after_by_path = {e["path"]: e for e in row["verified_after_resume"]}
    for e in before:
        if e["path"] in exempt:
            continue
        after = after_by_path.get(e["path"])
        if after is None:
            return False, f"{e['path']} missing from verified_after_resume"
        if after["size"] != e["size"] or after["mtime_ns"] != e["mtime_ns"]:
            return False, f"{e['path']} changed after resume (size/mtime mismatch)"
        if e["path"] in resent:
            return False, f"{e['path']} was unnecessarily re-sent on resume"
    return (
        True,
        "pre-kill files survived resume untouched; only missing/corrupt data re-sent",
    )


def score_item3(row: dict) -> tuple[bool, str]:
    if row["corrupted_sha_observed"] == row["manifest_sha"]:
        return (
            False,
            "corrupted_sha_observed == manifest_sha (corruption probe was invalid)",
        )
    if row["final_sha_after_resume"] != row["manifest_sha"]:
        return False, (
            f"final sha {row['final_sha_after_resume']} != manifest sha {row['manifest_sha']}"
        )
    if not row["refetched"]:
        return False, "corrupted file was not refetched"
    return (
        True,
        "digest mismatch detected and corrupted file refetched to match manifest",
    )


def score_item4(row: dict) -> tuple[bool, str]:
    if not row["both_viable_choice_required"]:
        return False, "omitted source did not force a typed choice_required"
    if row["explicit_source"] != "hf":
        return False, f"explicit_source {row['explicit_source']!r} != 'hf'"
    if not REVISION_RE.match(row["revision"]):
        return False, f"revision {row['revision']!r} is not a 40-char hex commit sha"
    if row["revision"] != row["head_revision"]:
        return False, (
            f"worker revision {row['revision']!r} != head revision {row['head_revision']!r}"
        )
    if len(row["digest_verified_paths"]) != row["required_set_count"]:
        return False, (
            f"{len(row['digest_verified_paths'])} digest-verified paths "
            f"!= required set count {row['required_set_count']}"
        )
    final = set(row["final_dir_listing"])
    verified = set(row["digest_verified_paths"])
    extra = final - verified
    missing = verified - final
    if extra:
        return False, f"unverified file present in final dir: {sorted(extra)}"
    if missing:
        return False, f"required file missing from final dir: {sorted(missing)}"
    return (
        True,
        "HF fan-out: explicit source honored, pinned revision, full digest-verified set present",
    )


def score_residual_cl5_16(row: dict) -> tuple[bool, str]:
    if not row["wedged"]:
        return False, "peer was not actually wedged (probe invalid)"
    if not row["watchdog_fired_before_deadline"]:
        return False, "watchdog did not fire before deadline"
    if not row["gate_released"]:
        return False, "gate was not released after watchdog fired"
    return True, "wedged peer detected by watchdog and gate released before deadline"


def score(dump: dict) -> int:
    rows = [
        ("item1 transfer+form, zero preemptive action", score_item1, dump["item1"]),
        ("item2 kill mid-flight, file-granular resume", score_item2, dump["item2"]),
        ("item3 corruption detected by digest, refetched", score_item3, dump["item3"]),
        ("item4 HF fan-out path", score_item4, dump["item4"]),
    ]
    failures = 0
    for name, fn, payload in rows:
        ok, why = fn(payload)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {why}")
        failures += 0 if ok else 1
    if "residual_cl5_16" in dump:
        ok, why = score_residual_cl5_16(dump["residual_cl5_16"])
        print(
            f"{'PASS' if ok else 'FAIL'}  residual_cl5_16 wedged-peer watchdog: {why}"
        )
        failures += 0 if ok else 1
    else:
        print("SKIP residual_cl5_16 (row not captured)")
    print(f"VERDICT: {'PASS' if failures == 0 else f'FAIL ({failures} rows)'}")
    return failures


def _selftest() -> int:
    good = {
        "item1": {
            "model_id": "llama-70b",
            "manifest_entry_count": 5,
            "load_accepted": True,
            "preemptive_user_actions": [],
            "job_reached_done": True,
            "formation_ready": True,
            "post_form_stream_tokens": 32,
            "worker_discovered_id": "llama-70b",
            "worker_final_dir_file_count": 5,
        },
        "item2": {
            "killed_mid_flight": True,
            "verified_before_kill": [
                {"path": "a/model.safetensors", "size": 100, "mtime_ns": 111},
                {"path": "a/config.json", "size": 10, "mtime_ns": 222},
                {"path": "a/corrupt.bin", "size": 50, "mtime_ns": 333},
            ],
            "verified_after_resume": [
                {"path": "a/model.safetensors", "size": 100, "mtime_ns": 111},
                {"path": "a/config.json", "size": 10, "mtime_ns": 222},
                {"path": "a/corrupt.bin", "size": 50, "mtime_ns": 999},
                {"path": "a/missing.bin", "size": 20, "mtime_ns": 555},
            ],
            "resent_paths": ["a/missing.bin"],
            "exempt_paths": ["a/corrupt.bin"],
        },
        "item3": {
            "corrupted_path": "a/corrupt.bin",
            "corrupted_sha_observed": "deadbeef",
            "manifest_sha": "cafef00d",
            "final_sha_after_resume": "cafef00d",
            "refetched": True,
        },
        "item4": {
            "both_viable_choice_required": True,
            "explicit_source": "hf",
            "revision": "a" * 40,
            "head_revision": "a" * 40,
            "required_set_count": 3,
            "final_dir_listing": ["x", "y", "z"],
            "digest_verified_paths": ["x", "y", "z"],
            "formation_ready": True,
        },
        "residual_cl5_16": {
            "wedged": True,
            "watchdog_fired_before_deadline": True,
            "gate_released": True,
        },
    }
    import copy

    cases = []
    d = copy.deepcopy(good)
    d["item1"]["worker_discovered_id"] = "wrong-id"
    cases.append(("item1 worker discovered id mismatch", d, False))

    d = copy.deepcopy(good)
    d["item1"]["preemptive_user_actions"] = ["copied files manually"]
    cases.append(("item1 preemptive operator action required", d, False))

    d = copy.deepcopy(good)
    d["item2"]["resent_paths"].append("a/model.safetensors")
    cases.append(("item2 non-exempt file unnecessarily re-sent", d, False))

    d = copy.deepcopy(good)
    d["item2"]["verified_after_resume"][0]["mtime_ns"] = 9999999
    cases.append(("item2 non-exempt file mtime changed after resume", d, False))

    d = copy.deepcopy(good)
    d["item2"]["verified_before_kill"] = []
    cases.append(("item2 verified_before_kill empty", d, False))

    d = copy.deepcopy(good)
    d["item3"]["final_sha_after_resume"] = "stillbad"
    cases.append(("item3 final sha never converged to manifest", d, False))

    d = copy.deepcopy(good)
    d["item4"]["final_dir_listing"].append("extra_unverified_file")
    cases.append(("item4 unverified file present in final dir", d, False))

    d = copy.deepcopy(good)
    d["item4"]["revision"] = "main"
    cases.append(("item4 revision not pinned to a commit sha", d, False))

    cases.append(("all-good dump", good, True))

    d = copy.deepcopy(good)
    del d["residual_cl5_16"]
    cases.append(("residual_cl5_16 absent (SKIP, still PASS)", d, True))

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
