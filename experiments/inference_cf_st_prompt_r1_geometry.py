#!/usr/bin/env python
"""ST-PROMPT-R1-A pre-pulse geometry coverage gate (frozen spec 'Physical site and energy'): a SEPARATE process run inside
the A allocation after passive capture and before any pulse. It reads only the reference-free CPU geometry ledger and
the panel's offline evaluation-membership STRATA/dialogue labels (never reference targets, competitors or outcomes) and
requires, for EVERY primary and random arm, >= 54/60 reachable cells and >= 12 dialogues in EACH stratum. Failure =
STOP before outcomes for explicit human dose revision (no pruning, no ladder change). The pulse runner never sees strata;
it only reads this file's verdict."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json, file_hash

CONFIG = "configs/inference_cf/st_prompt_r1.json"
PANEL = "docs/inference_cf/ST_PROMPT_R1_PANEL.json"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")


def coverage(ledger: dict, membership: list[dict], arm_ids: list[str], e: dict) -> dict:
    out = {}
    for aid in arm_ids:
        n = defaultdict(int)
        dl = defaultdict(set)
        for j, p in enumerate(membership):
            if ledger[f"q{j:03d}|{aid}"]["reachable"]:
                n[p["stratum"]] += 1
                dl[p["stratum"]].add(p["dialogue_id"])
        out[aid] = {"reachable": {s: n[s] for s in STRATA}, "dialogues": {s: len(dl[s]) for s in STRATA},
                    "pass": all(n[s] >= e["minimum_valid_per_stratum"] and len(dl[s]) >= e["minimum_valid_dialogues_per_stratum"] for s in STRATA)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = ROOT / args.out
    if (run / "geometry_gate.json").exists():
        raise FileExistsError("gate exists")
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    cap = json.loads((run / "capture" / "capture.json").read_text())
    if file_hash(run / "capture" / "geometry_ledger.json") != cap["files"]["geometry_ledger.json"]:
        raise ValueError("ledger changed")
    ledger = json.loads((run / "capture" / "geometry_ledger.json").read_text())
    plan = json.loads((ROOT / "results/inference_cf/st_prompt_r1/plan_sealed.json").read_text())
    if [(q["utterance_id"], q["t"]) for q in plan["runtime"]["runtime_queries"]] != [(p["utterance_id"], p["t"]) for p in P["evaluation_membership"]]:
        raise ValueError("membership order")
    cov = coverage(ledger, P["evaluation_membership"], [a["id"] for a in plan["arms"]], cfg["eligibility"])
    ok = all(v["pass"] for v in cov.values()) and len(cov) == 36
    doc = {"schema": "st_prompt_r1_geometry_gate_v1", "verdict": "GEOMETRY_COVERAGE_PASS" if ok else "STOP_FOR_HUMAN_DOSE_REVISION",
           "arms": cov, "ledger_sha256": cap["files"]["geometry_ledger.json"], "reference_targets_used": False,
           "failing_arms": [a for a, v in cov.items() if not v["pass"]]}
    atomic_json(run / "geometry_gate.json", doc)
    print(json.dumps({"verdict": doc["verdict"], "failing": doc["failing_arms"]}))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
