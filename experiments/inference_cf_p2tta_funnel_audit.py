#!/usr/bin/env python
"""Independent P2-TTA-FUNNEL auditor (frozen contract ff75e1a: P2_TTA_FUNNEL_SPEC.md, P2_TTA_FUNNEL_CODEX_DESIGN.md,
configs/inference_cf/p2_tta_funnel.json). Does NOT import any primary analysis/decision/objective code
(``*_analyze``, ``csasr.inference_cf.episodic_tta``, the funnel runner); reuses the independent TTA0 auditor and
locked canonical primitives only.

r-gate -> PASS_TO_P2_TTA0_R_EVALUATION | P2_TTA0_R_STILL_INVALID   (no reference access, no model, no adaptation)
r-post -> P2_TTA0_R_AUDIT: PASS | BLOCK
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import experiments.inference_cf_p2tta0_audit as tau

FREEZE = "ff75e1a6b524415d8071749554b66c09f8dd2ad2"
CONFIG = "configs/inference_cf/p2_tta_funnel.json"
FROZEN = ("docs/inference_cf/P2_TTA_FUNNEL_SPEC.md", "docs/inference_cf/P2_TTA_FUNNEL_CODEX_DESIGN.md", CONFIG,
          "docs/inference_cf/P2_TTA_MAP_PANEL24.json", "docs/inference_cf/P2_TTA_MAP_PANEL_CONTRACT.md",
          "docs/inference_cf/P2_TTA1_INHERITANCE_CONTRACT.md")
TTA0 = ROOT / "results/inference_cf/p2tta0"
OUT = ROOT / "results/inference_cf/p2tta_funnel"
R_MAP = {"P2_TTA0_INVALID": "P2_TTA0_R_STILL_INVALID", "P2_TTA0_OBJECTIVE_SELECTED": "P2_TTA0_R_OBJECTIVE_SELECTED",
         "P2_TTA0_NO_VIABLE_OBJECTIVE": "P2_TTA0_R_NO_VIABLE_OBJECTIVE"}


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def frozen_checks(c: dict) -> dict:
    ch = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in FROZEN}
    ch["historical_sources_at_starting_commit"] = all(tau.blob_sha(c["starting_commit"], p) == h for p, h in c["source_sha256"].items())
    ch["immutable_configs_panels_current"] = all(sha(ROOT / p) == c["source_sha256"][p] for p in c["source_sha256"]
                                                 if p.startswith(("configs/", "docs/", "results/")))
    return ch


# ---- TTA0-R gate ------------------------------------------------------------------------------------------------

def cmd_r_gate(args) -> dict:
    c = cfg()
    R = c["TTA0_R"]
    checks = frozen_checks(c)
    checks["sealed_bytes"] = all(sha(ROOT / p) == h for p, h in R["sealed_sha256"].items())
    checks["baseline20_bytes"] = all(sha(ROOT / b["forced_row"]) == b["forced_row_sha256"] and sha(ROOT / b["auto_row"]) == b["auto_row_sha256"]
                                     for b in R["baseline20"])
    t0 = json.loads((ROOT / "configs/inference_cf/p2_tta0.json").read_text())
    checks["original_decision_exact_copy"] = t0["decision"] == c["original_tta0_decision"]
    checks["only_numerical_tolerance_repaired"] = (R["relative_gradient_tolerance"] == 0.02 == c["audit"]["relative_gradient_tolerance"]
                                                   and R["previous_relative_gradient_tolerance"] == 0.001 and R["absolute_loss_tolerance"] == 1e-5
                                                   and R["absolute_gradient_floor"] == 1e-8 and R["new_gpu_jobs"] == 0 and R["no_new_adaptation"] is True
                                                   and c["common"]["optimization"] == t0["optimization"]
                                                   and c["common"]["trainables"] == t0["trainables"]
                                                   and c["common"]["teacher_forcing"] == t0["teacher_forcing"]
                                                   and R["baseline20"] == t0["reuse"]["rows"])
    checks["no_new_adaptation_outputs"] = sorted(x.name for x in TTA0.iterdir()) == sorted(
        ["prerun_audit.json", "pseudo_sealed.json", "run1", "run1_audit.json", "run1_invalid_record.json", "run1_live_audit_diagnostic.json"])
    # independent recomputation of sealed mechanics with the historical independent auditor (reference-free)
    inv = tau.cmd_invalid(SimpleNamespace(run="results/inference_cf/p2tta0/run1", record="results/inference_cf/p2tta0/run1_invalid_record.json"))
    eng = inv["engineering"]
    for k, v in inv["checks"].items():
        if k not in ("label_agrees", "failed_checks_agree"):
            checks[f"sealed:{k}"] = bool(v)
    for k, v in eng.items():
        if k != "live_objective_gradient_check":
            checks[f"mechanics:{k}"] = bool(v)          # trainable/reset identities, theta0, updates, losses, outputs, counts
    rows0 = json.loads((TTA0 / "run1/rows/00.json").read_text())
    rt = json.loads((TTA0 / "run1/runtime.json").read_text())
    rows = [json.loads((TTA0 / f"run1/rows/{i:02d}.json").read_text()) for i in range(20)]
    checks["finite_gradients"] = all(all(o["finite"]) and all(math.isfinite(x) for x in o["grad_l2"] + o["losses"])
                                     for r in rows for o in r["objectives"].values())
    checks["trainable_identity"] = json.loads((TTA0 / "run1/manifest.json").read_text())["trainables"] == \
        [p["name"] for p in c["common"]["trainables"]["parameters"]] and len(c["common"]["trainables"]["parameters"]) == 194
    checks["reset_identities"] = all(o["start_hash"] == o["end_hash"] == rt["theta0_ln_hash"] and o["reset_ok"] for r in rows
                                     for o in r["objectives"].values()) and rt["reset_final_ok"] and rt["nonln_unchanged"]
    checks["output_hashes"] = all(r["final_masters_npz_sha256"] == "sha256:" + sha(TTA0 / f"run1/rows/{i:02d}_final_masters_fp32.npz")
                                  for i, r in enumerate(rows))
    la, live = rows0["live_audit"], {}
    for k in ("A1", "A2"):
        x = la[k]
        live[k] = {"loss_abs_diff": abs(x["primary_loss"] - x["auditor_loss"]), "grad_rel": x["grad_diff_l2"] / x["auditor_grad_l2"],
                   "loss_ok": abs(x["primary_loss"] - x["auditor_loss"]) <= R["absolute_loss_tolerance"],
                   "grad_ok_repaired": x["grad_diff_l2"] <= max(R["absolute_gradient_floor"], R["relative_gradient_tolerance"] * x["auditor_grad_l2"]),
                   "primary_is_sealed_L0": x["primary_loss"] == rows0["objectives"][k]["losses"][0]}
    checks["live_gradient_agreement_repaired"] = all(v["loss_ok"] and v["grad_ok_repaired"] and v["primary_is_sealed_L0"] for v in live.values())
    checks["runtime_invalid_only_live_audit"] = all("live audit" in x for x in rt.get("invalid", []))
    diag = json.loads((TTA0 / "run1_live_audit_diagnostic.json").read_text())["objectives"]
    checks["float64_formula_equivalence"] = all(diag[k]["relative_l2_diff"]["A64_vs_P64"] == 0.0 for k in ("A1", "A2"))
    checks["fp32_repeat_determinism"] = all(diag[k]["relative_l2_diff"]["P32_repeat_vs_P32"] == 0.0 for k in ("A1", "A2"))
    disc = [diag[k]["relative_l2_diff"][n] for k in ("A1", "A2") for n in ("P32_vs_P64", "A32_vs_P64", "A32_vs_P32")]
    checks["recorded_discrepancies_within_repaired_tolerance"] = max(disc) <= R["relative_gradient_tolerance"]
    src = (ROOT / "experiments/inference_cf_p2tta_funnel_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|episodic_tta|inference_cf_p2tta_funnel\b)", src, re.M) is None
    v = "PASS_TO_P2_TTA0_R_EVALUATION" if all(checks.values()) else "P2_TTA0_R_STILL_INVALID"
    return {"schema": "p2_tta0_r_gate_v1", "verdict": v, "checks": checks, "live": live, "recorded_discrepancies": disc,
            "references_loaded": False, "model_loaded": False, "adaptation_run": False, "git_commit": _git("rev-parse", "HEAD")}


def cmd_r_post(args) -> dict:
    c = cfg()
    gate_rel = "results/inference_cf/p2tta_funnel/tta0_r/gate_audit.json"
    gate = json.loads((ROOT / gate_rel).read_text())
    ana = json.loads((ROOT / args.analysis).read_text())
    base = tau.cmd_post(SimpleNamespace(run="results/inference_cf/p2tta0/run1", analysis=args.analysis,
                                        rel_grad_tol=c["TTA0_R"]["relative_gradient_tolerance"]))
    checks = {f"tta0_post:{k}": bool(v) for k, v in base["checks"].items()}
    checks["gate_passed_and_committed"] = gate["verdict"] == "PASS_TO_P2_TTA0_R_EVALUATION" and committed(gate_rel)
    checks["analysis_after_gate"] = ana.get("gate_sha256") == sha(ROOT / gate_rel)
    checks["sealed_bytes_unchanged"] = all(sha(ROOT / p) == h for p, h in c["TTA0_R"]["sealed_sha256"].items())
    own_r = R_MAP[base["label"]]
    checks["r_label_agrees"] = own_r == ana["r_label"]
    checks["selected_agrees"] = base["selected"] == ana["selected"]
    v = "P2_TTA0_R_AUDIT: PASS" if all(checks.values()) else "P2_TTA0_R_AUDIT: BLOCK"
    return {"schema": "p2_tta0_r_post_audit_v1", "verdict": v, "r_label": own_r, "selected": base["selected"], "labels": base["labels"],
            "checks": checks, "independent": base["independent"], "failures": base["failures"], "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("r-gate").add_argument("--out", required=True)
    p = sub.add_parser("r-post")
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"r-gate": cmd_r_gate, "r-post": cmd_r_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "r_label", "selected", "labels")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
