#!/usr/bin/env python
"""CPU acceptance evaluation of the frozen P1 run (criteria A1-A13 of the P1 spec)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.core_p1 import LAYER, VERSION_P1, selected_gate


def evaluate(out: Path) -> dict:
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != VERSION_P1 or digest({k: v for k, v in manifest.items()
                                                     if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid P1 manifest")
    panel = json.loads((out / "panel.json").read_text())
    config = manifest["resolved_config"]
    tol = config["acceptance"]
    nfp = config["num_forced_prefix"]
    runtime = json.loads((out / "runtime.json").read_text()) if (out / "runtime.json").exists() else {}
    expected = [(i, r["utterance_id"], float(a)) for i, r in enumerate(panel["rows"]) for a in config["alphas"]]
    records, missing = {}, []
    for i, uid, a in expected:
        p = out / "rows" / f"{i:03d}_alpha{a}.json"
        if not p.exists():
            missing.append(f"{uid}|alpha={a}")
            continue
        rec = json.loads(p.read_text())
        if rec.get("manifest_hash") != manifest["manifest_hash"] or rec.get("identity") != f"{uid}|alpha={a}":
            raise ValueError(f"stale shard {p}")
        records[(uid, a)] = rec
    ok = {k: v for k, v in records.items() if v.get("status") == "ok"}
    zero = [v for (u, a), v in ok.items() if a == 0.0]
    dose = [v for (u, a), v in ok.items() if a > 0.0]
    steps0 = [s for v in zero for s in v["steps"]]
    steps1 = [s for v in dose for s in v["steps"]]
    all_steps = steps0 + steps1
    edits = [s for s in steps1 if s["edit_applied"]]
    fail = {}

    # A1 zero-dose identity: hook installed at alpha=0 is a bitwise no-op, tokens = unsteered argmax
    a1 = bool(zero) and all(s["f3_bitwise_equals_f1"] and s["steered_next"] == s["unsteered_next"]
                            and not s["edit_applied"] for s in steps0)
    # A2 finite, sample-varying direction
    dir_ok = [s for s in all_steps if s["direction_status"] == "ok"]
    cos_prev = [s["cos_to_previous_direction"] for s in dir_ok if s.get("cos_to_previous_direction") is not None]
    a2 = (bool(dir_ok) and all(math.isfinite(s["direction_norm"]) for s in dir_ok)
          and any(c < tol["direction_variation_cos_max"] for c in cos_prev))
    # A3 tiny/nonfinite direction fallback never edits
    dir_fail = [s for s in all_steps if s["direction_status"] != "ok"]
    a3 = all(not s["edit_applied"] for s in dir_fail)
    # A4 at least one real edit
    a4 = any(s["g"] > 0 and s["direction_status"] == "ok" and s["query_index"] >= nfp
             and s["hook_audit"] and s["hook_audit"]["edit_norm"] > 0 for s in edits)
    # A5 site: every audit record at layer 24, at the current query position
    audits = [s["hook_audit"] for s in all_steps if s.get("hook_audit")]
    a5 = bool(audits) and all(a["layer"] == LAYER for a in audits) and \
        all(s["hook_audit"]["abs_pos"] == s["query_index"] for s in all_steps if s.get("hook_audit"))
    # A6 sign and hook/specified-edit consistency. v1.1 instrument: the specified edit is
    # replicated at the site's actual precision (apply_steering on the recorded site); v1 compared
    # against float64, which bf16 cannot meet below its resolution (attempt 1, job 54770).
    ref_key = "replica_edit_norm" if tol.get("a6_reference") == "site_precision_replica" else "reference_edit_norm"
    a6 = bool(edits) and all(
        s["reference_cos_edit_d"] is not None and s["reference_cos_edit_d"] > 0 and
        s.get(ref_key) is not None and
        abs(s["hook_audit"]["edit_norm"] - s[ref_key])
        <= tol["reference_edit_rel_tol"] * max(s[ref_key], 1e-12) for s in edits)
    # A7 gate recomputed from logged components; fallbacks have g=0 and no edit
    def gate_ok(s):
        if s["fallback_reason"] is not None:
            return s["g"] == 0.0 and not s["edit_applied"]
        return abs(s["g"] - selected_gate(s["local_support"], s["baseline"])) <= tol["gate_abs_tol"]
    a7 = all(gate_ok(s) for s in all_steps)
    # A8 NormPreserve on edits; zero edit when nothing is edited at the query
    a8 = bool(edits) and all(
        abs(s["hook_audit"]["post_norm"] - s["hook_audit"]["pre_norm"])
        <= tol["normpreserve_rel_tol"] * s["hook_audit"]["pre_norm"] for s in edits) and \
        all(s["hook_audit"]["edit_norm"] == 0.0 for s in all_steps
            if s.get("hook_audit") and not s["edit_applied"])
    # A9 prefix integrity; forced prefix never edited
    a9 = all(s["input_is_prompt_plus_prefix"] for s in all_steps) and \
        all(p >= nfp for v in ok.values() for p in v["edited_positions"])
    # A10 isolation: runner asserts no hooks around F1/F2 and after F3 (a leak aborts the shard);
    # all shards completed, so every assertion passed.
    a10 = len(ok) == len(expected)
    # A11 continuation
    a11 = bool(dose) and all(v["terminated"] in ("eos", "cap") and isinstance(v["text"], str) for v in dose)
    # A12 serialization
    a12 = not missing and len(ok) == len(expected)
    # A13 cost
    a13 = bool(runtime.get("elapsed_sec")) and bool(runtime.get("peak_vram_bytes"))
    checks = {"A1_zero_dose_identity": a1, "A2_finite_varying_direction": a2,
              "A3_tiny_direction_fallback": a3, "A4_real_edit": a4, "A5_site": a5,
              "A6_sign": a6, "A7_gate": a7, "A8_normpreserve": a8, "A9_prefix_integrity": a9,
              "A10_isolation": a10, "A11_continuation": a11, "A12_serialization": a12,
              "A13_cost": a13}
    # Diagnostics (no pass rule): agreement of alpha=0 decode with the R2 generate() baseline
    r2_rows = {}
    for p in sorted((ROOT / "results/inference_cf/p0_r2/rows").glob("*.json")):
        s = json.loads(p.read_text())
        r2_rows[s["identity"]] = s.get("baseline_content_ids")
    b0_match = {v["utterance_id"]: v["tokens"] == r2_rows.get(v["utterance_id"]) for v in zero}
    changed = {}
    for v in dose:
        z = records.get((v["utterance_id"], 0.0))
        changed[v["utterance_id"]] = None if z is None else v["tokens"] != z["tokens"]
    summary = {"schema": VERSION_P1, "manifest_hash": manifest["manifest_hash"],
               "verdict": "P1_CAUSAL_ACCEPTANCE_PASS" if all(checks.values()) else "P1_BLOCKED",
               "checks": checks,
               "counts": {"expected_records": len(expected), "ok_records": len(ok),
                          "missing": missing,
                          "failures": [f"{k}: {v.get('reason')}" for k, v in records.items()
                                       if v.get("status") != "ok"],
                          "steps_alpha0": len(steps0), "steps_alpha1": len(steps1),
                          "edited_steps": len(edits),
                          "gate_positive_steps_alpha1": sum(s["g"] > 0 for s in steps1),
                          "forced_prefix_blocked": sum(s["forced_prefix_blocked"] for s in steps1),
                          "direction_fail": len(dir_fail),
                          "fallbacks": {r: sum(s["fallback_reason"] == r for s in all_steps)
                                        for r in sorted({s["fallback_reason"] for s in all_steps
                                                         if s["fallback_reason"]})}},
               "edit_statistics": {
                   "gate": [min(s["g"] for s in edits), max(s["g"] for s in edits)] if edits else None,
                   "direction_norm": [min(s["direction_norm"] for s in dir_ok),
                                      max(s["direction_norm"] for s in dir_ok)] if dir_ok else None,
                   "realized_edit_norm": [min(s["hook_audit"]["edit_norm"] for s in edits),
                                          max(s["hook_audit"]["edit_norm"] for s in edits)] if edits else None,
                   "relative_edit": [min(s["hook_audit"]["edit_norm"] / s["hook_audit"]["pre_norm"] for s in edits),
                                     max(s["hook_audit"]["edit_norm"] / s["hook_audit"]["pre_norm"] for s in edits)] if edits else None,
                   "max_normpreserve_rel_error": max((abs(s["hook_audit"]["post_norm"] - s["hook_audit"]["pre_norm"])
                                                      / s["hook_audit"]["pre_norm"] for s in edits), default=None),
                   "min_cos_edit_d": min((s["reference_cos_edit_d"] for s in edits), default=None),
                   "max_rel_error_hook_vs_replica": max((abs(s["hook_audit"]["edit_norm"] - s["replica_edit_norm"])
                                                         / max(s["replica_edit_norm"], 1e-12) for s in edits
                                                         if s.get("replica_edit_norm") is not None), default=None),
                   # descriptive: float64 reference agreement for edits well above bf16 resolution
                   "max_rel_error_hook_vs_float64_above_resolution": max(
                       (abs(s["hook_audit"]["edit_norm"] - s["reference_edit_norm"]) / s["reference_edit_norm"]
                        for s in edits if s["reference_edit_norm"] >= 10 * s["hook_audit"]["pre_norm"] * 2 ** -8),
                       default=None),
                   "min_cos_between_consecutive_directions": min(cos_prev) if cos_prev else None},
               "diagnostics": {"alpha0_equals_r2_generate_tokens": b0_match,
                               "alpha0_generate_agreement": (sum(b0_match.values()) / len(b0_match)
                                                             if b0_match else None),
                               "alpha1_token_sequence_changed": changed},
               "runtime": {k: runtime.get(k) for k in ("job_id", "gpu", "elapsed_sec", "peak_vram_bytes")},
               "claim_boundary": "computational/causal validity of the intervention pipeline only; no ASR-improvement claim"}
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_cf/p1_r1")
    args = parser.parse_args()
    s = evaluate(ROOT / args.out)
    print(json.dumps({"verdict": s["verdict"], "checks": s["checks"]}, indent=2))
