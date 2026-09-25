#!/usr/bin/env python
"""CPU acceptance of the Pre-P2 cached-equivalence run (criteria CE1-CE8, frozen spec)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from csasr.inference_cf.core import atomic_json, digest

NEAR_TIE = 0.25


def frac(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def evaluate(out: Path) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    shards = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
    expected = len(m["utterances"]) * len(m["layers"]) * len(m["alphas"])
    rt = json.loads((out / "runtime.json").read_text()) if (out / "runtime.json").exists() else {}
    comps = [(s, c) for s in shards for c in s["comparisons"]]
    zero = [s for s in shards if s["alpha"] == 0.0]
    dose = [(s, c) for s, c in comps if s["alpha"] > 0.0]
    # CE1
    ce1 = bool(zero) and all(all(st["f3_bitwise_equals_b"] for st in s["steps"]) and
                             s["tokens"] == s["cached_greedy_tokens"] for s in zero)
    # CE2
    ce2 = all(s["lineage_ok"] and [st["query_index"] for st in s["steps"]] ==
              list(range(3, 3 + len(s["steps"]))) for s in shards)
    # CE3
    ce3 = all(s["distinct_caches"] for s in shards)
    # CE4
    fb_same = [c["fb"][0] == c["fb"][1] for _, c in comps]
    both_w = [c for _, c in comps if c["window"][0] and c["window"][1]]
    win_same = [c["window"][0] == c["window"][1] for c in both_w]
    rb = [abs(c["R_B"][0] - c["R_B"][1]) <= 0.02 for _, c in comps
          if c["R_B"][0] is not None and c["R_B"][1] is not None]
    e_same_win = [abs(c["E"][0] - c["E"][1]) <= 0.02 for c in both_w
                  if c["window"][0] == c["window"][1] and c["E"][0] is not None and c["E"][1] is not None]
    gd = [abs(c["g"][0] - c["g"][1]) <= 0.02 for _, c in comps]
    ce4_stats = {"fallback_identical": frac(fb_same), "window_identical": frac(win_same),
                 "R_B_within_0.02": frac(rb), "E_within_0.02_same_window": frac(e_same_win),
                 "g_within_0.02": frac(gd)}
    ce4 = (ce4_stats["fallback_identical"] >= .99 and ce4_stats["window_identical"] >= .95 and
           ce4_stats["R_B_within_0.02"] >= .99 and ce4_stats["E_within_0.02_same_window"] >= .99 and
           ce4_stats["g_within_0.02"] >= .95)
    exceed = [{"id": s["identity"], "t": c["t"], "g": c["g"], "window": c["window"], "E": c["E"], "R_B": c["R_B"]}
              for s, c in comps if abs(c["g"][0] - c["g"][1]) > 0.02]
    # CE5
    status_same = [c["dir_status"][0] == c["dir_status"][1] for _, c in comps]
    ok_dirs = [c for _, c in comps if c["dir_status"] == ["ok", "ok"]]
    cos_ok = [c["dir_cos"] >= .99 for c in ok_dirs]
    norm_ok = [abs(c["dir_norm"][0] - c["dir_norm"][1]) / c["dir_norm"][1] <= .02 for c in ok_dirs]
    ce5_stats = {"status_identical": frac(status_same), "cos_ge_0.99": frac(cos_ok),
                 "norm_within_2pct": frac(norm_ok),
                 "min_cos": min((c["dir_cos"] for c in ok_dirs), default=None)}
    ce5 = ce5_stats["status_identical"] == 1.0 and ce5_stats["cos_ge_0.99"] >= .99 and ce5_stats["norm_within_2pct"] >= .99
    # CE6
    edit_same = [c["edit"][0] == c["edit"][1] for _, c in dose]
    cached_edits = [c for _, c in dose if c["edit"][0]]
    sign_ok = all(c["reference_cos"] is not None and c["reference_cos"] > 0 for c in cached_edits)
    replica_ok = all(abs(c["audit"][0]["edit_norm"] - c["replica_edit_norm"]) <= .05 * max(c["replica_edit_norm"], 1e-12)
                     for c in cached_edits)
    np_ok = all(abs(c["audit"][0]["post_norm"] - c["audit"][0]["pre_norm"]) <= 1e-2 * c["audit"][0]["pre_norm"]
                for c in cached_edits)
    common = [c for c in cached_edits if c["edit"][1] and abs(c["g"][0] - c["g"][1]) <= .02]
    common_ok = [abs(c["audit"][0]["edit_norm"] - c["audit"][1]["edit_norm"]) <= .05 * c["audit"][1]["edit_norm"]
                 for c in common]
    disagreements = [{"id": s["identity"], "t": c["t"], "argmax": c["argmax"], "margin": c["margin"],
                      "near_tie": min(c["margin"]) <= NEAR_TIE}
                     for s, c in comps if c["argmax"][0] != c["argmax"][1]]
    non_near = [x for x in disagreements if not x["near_tie"]]
    ce6_stats = {"edit_applied_identical": frac(edit_same), "cached_edits": len(cached_edits),
                 "sign_ok": sign_ok, "replica_ok": replica_ok, "normpreserve_ok": np_ok,
                 "common_edits": len(common), "common_edit_norm_within_5pct": frac(common_ok),
                 "argmax_disagreements": disagreements, "non_near_tie_disagreements": len(non_near),
                 "max_abs_dlogit": max((c["max_abs_dlogit"] for _, c in comps), default=None)}
    ce6 = (bool(cached_edits) and (ce6_stats["edit_applied_identical"] or 0) >= .97 and sign_ok and replica_ok
           and np_ok and (ce6_stats["common_edit_norm_within_5pct"] or 0) >= .95 and not non_near)
    # CE7: every shard written means every in-loop assertion passed; runtime completed after final assert
    ce7 = len(shards) == expected and rt.get("status") == "completed"
    # CE8
    ce8 = all(rt.get(k) is not None for k in ("elapsed_sec", "peak_vram_bytes", "cached_sec", "audio_sec"))
    checks = {"CE1_zero_dose_identity": ce1, "CE2_positions_lineage": ce2, "CE3_branch_isolation": ce3,
              "CE4_gate_agreement": ce4, "CE5_direction_agreement": ce5,
              "CE6_intervention_equivalence": ce6, "CE7_hook_isolation": ce7, "CE8_runtime": ce8}
    b0_match = frac(s["tokens"] == s["r2_generate_tokens"] for s in zero)
    summary = {"schema": m["schema"], "manifest_hash": m["manifest_hash"],
               "verdict": "CACHED_EQUIVALENCE: PASS" if all(checks.values()) and len(shards) == expected
               else "CACHED_EQUIVALENCE: BLOCK",
               "checks": checks, "shards": len(shards), "expected": expected,
               "steps_compared": len(comps), "ce4": ce4_stats, "g_exceedances": exceed,
               "ce5": ce5_stats, "ce6": ce6_stats,
               "diagnostics": {"alpha0_cached_equals_r2_generate": b0_match},
               "runtime": rt}
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/inference_cf/pre_p2_ce")
    s = evaluate(ROOT / ap.parse_args().out)
    print(json.dumps({"verdict": s["verdict"], "checks": s["checks"]}, indent=2))
