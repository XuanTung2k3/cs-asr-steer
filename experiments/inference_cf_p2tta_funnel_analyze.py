#!/usr/bin/env python
"""P2-TTA-FUNNEL primary evaluation (frozen contract ff75e1a). Evaluator side; references are opened only behind
each stage's committed independent gate.

r-evaluate: TTA0-R. Requires the committed PASS_TO_P2_TTA0_R_EVALUATION gate and re-verified sealed bytes, then runs
the ORIGINAL TTA0 evaluator/decision (`inference_cf_p2tta0_analyze.analyze`, thresholds = original decision section)
with only the live complete-gradient tolerance re-adjudicated at 0.02. Stage label mapped to the R vocabulary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json

CONFIG = "configs/inference_cf/p2_tta_funnel.json"
R_DIR = "results/inference_cf/p2tta_funnel/tta0_r"
R_GATE = f"{R_DIR}/gate_audit.json"
R_MAP = {"P2_TTA0_INVALID": "P2_TTA0_R_STILL_INVALID", "P2_TTA0_OBJECTIVE_SELECTED": "P2_TTA0_R_OBJECTIVE_SELECTED",
         "P2_TTA0_NO_VIABLE_OBJECTIVE": "P2_TTA0_R_NO_VIABLE_OBJECTIVE"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gate_ok(rel: str, verdict: str) -> bool:
    if not (ROOT / rel).exists():
        return False
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel) and json.loads((ROOT / rel).read_text())["verdict"] == verdict


def r_evaluate() -> dict:
    c = json.loads((ROOT / CONFIG).read_text())
    if not gate_ok(R_GATE, "PASS_TO_P2_TTA0_R_EVALUATION"):
        raise PermissionError("TTA0-R gate missing/uncommitted/not PASS: references stay closed")
    if any(sha(ROOT / p) != h for p, h in c["TTA0_R"]["sealed_sha256"].items()):
        raise ValueError("sealed TTA0 bytes changed")
    import experiments.inference_cf_p2tta0_analyze as an
    if an.TH != c["original_tta0_decision"]["thresholds"]:
        raise ValueError("original TTA0 thresholds differ from master copy")
    res = an.analyze(ROOT / "results/inference_cf/p2tta0/run1", rel_grad_tol=c["TTA0_R"]["relative_gradient_tolerance"])
    seal = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta0/pseudo_sealed.json").read_text())["rows"]}
    rows = [json.loads((ROOT / f"results/inference_cf/p2tta0/run1/rows/{i:02d}.json").read_text()) for i in range(20)]
    differing = [u for u in res["per_utterance"] if seal[u["utterance_id"]]["y_B"] != seal[u["utterance_id"]]["y_A"]]
    res["auto_forced_differing_cases"] = [
        {"utterance_id": u["utterance_id"], **{k: {"changed_vs_B0": not r["objectives"][k]["vs_B0_FORCED"]["equal"],
                                                   "lev_to_AUTO": r["objectives"][k]["vs_B0_AUTO"]["levenshtein"]} for k in ("A1", "A2")},
         "lev_B0_to_AUTO": None} for u in differing for r in rows if r["identity"] == u["utterance_id"]]
    for d in res["auto_forced_differing_cases"]:
        from csasr.inference_cf.episodic_tta import levenshtein
        s = seal[d["utterance_id"]]
        d["lev_B0_to_AUTO"] = levenshtein(s["y_B"], s["y_A"])
    res["r_label"] = R_MAP[res["label"]]
    res["gate_sha256"] = sha(ROOT / R_GATE)
    res["repaired_relative_gradient_tolerance"] = c["TTA0_R"]["relative_gradient_tolerance"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("r-evaluate").add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable(r_evaluate())
    atomic_json(out, res)
    print(json.dumps({"r_label": res["r_label"], "selected": res["selected"], "labels": res["labels"], "integrity": res["integrity"],
                      "metrics": {k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()},
                      "points": {k: v["point"] for k, v in res["objectives"].items()},
                      "safety": {k: v["safety"] for k, v in res["objectives"].items()},
                      "useful": {k: v["useful"] for k, v in res["objectives"].items()},
                      "cb": res["objectives"].get("A1", {}).get("confirmation_bias"),
                      "differing": res["auto_forced_differing_cases"]}, indent=1))


if __name__ == "__main__":
    main()
