#!/usr/bin/env python
"""Independent P2-RJ-E post-run audit (CPU). Imports neither the P2-RJ-E / P2-RJ analysis modules
nor the runners.

Recomputes from raw rows and vectors:

* population size (vs the frozen P2-R ledger) and dialogue membership (vs the R2 evaluation units);
* margins/gaps, ‖g‖, ‖g_tan‖, A and rho;
* Q50 and its Bonferroni dialogue-bootstrap interval, and the leverage classes (bf16, fp32);
* the precision cosine and the final diagnosis;
* overlap reproduction against raw P2-RJ run1 rows and vectors.

It also checks provenance, data roles and the no-gradient-steering guarantees.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np


def canon(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()


def fhash(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob_hash(commit: str, rel: str) -> str:
    blob = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True).stdout
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def interval(pairs, reps, seed, alpha):
    groups = defaultdict(list)
    for g, v in pairs:
        groups[g].append(v)
    names = sorted(groups)
    m = np.array([sum(groups[n]) / len(groups[n]) for n in names])
    draws = m[np.random.default_rng(seed).integers(0, len(names), size=(reps, len(names)))].mean(axis=1)
    return float(m.mean()), [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))]


def tan(g, r):
    rh = r / np.linalg.norm(r)
    return g - rh * (rh @ g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = ROOT / args.run
    man = json.loads((run / "manifest.json").read_text())
    ana = json.loads((ROOT / args.analysis).read_text())
    cfg = json.loads((ROOT / man["config"]).read_text())
    pos = json.loads((ROOT / man["positions"]).read_text())
    checks, notes = {}, {}
    # provenance / no post-hoc change
    checks["manifest_hash"] = canon({k: v for k, v in man.items() if k != "manifest_hash"}) == man["manifest_hash"]
    checks["sources_match_manifest_commit"] = all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items())
    notes["sources_changed_since_freeze"] = [p for p, h in man["sources"].items() if fhash(ROOT / p) != h]
    checks["decision_sources_unchanged"] = not notes["sources_changed_since_freeze"]
    checks["config_hash"] = canon(cfg) == man["config_hash"]
    checks["positions_hash"] = canon({k: v for k, v in pos.items() if k != "positions_hash"}) == man["positions_hash"]
    checks["runtime_completed"] = json.loads((run / "runtime.json").read_text()).get("status") == "completed"
    # population: size vs the frozen P2-R ledger, membership/dialogues vs R2 evaluation units, originals
    p2r_pop = json.loads((ROOT / cfg["p2r_reference"]["population"]).read_text())
    checks["population_size_equals_frozen_p2r_ledger"] = len(pos["positions"]) == p2r_pop["ledger"]["EN-confusion|valid"]
    keys = [(p["utterance_id"], p["t"]) for p in pos["positions"]]
    checks["no_duplicates"] = len(set(keys)) == len(keys)
    units = json.loads((ROOT / "results/inference_cf/p0_r2/evaluation_units.json").read_text())
    en_units = {(u["utterance_id"], u["position"]): u["dialogue_id"] for u in units
                if u["stratum"] == "EN-confusion" and u["reason"] is None and u["position"] is not None}
    checks["positions_are_r2_en_confusion_units"] = all(k in en_units for k in keys)
    checks["dialogue_membership"] = all(en_units[(p["utterance_id"], p["t"])] == p["dialogue_id"] for p in pos["positions"])
    panel = {r["utterance_id"] for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    checks["utterances_in_d_dev_select_panel"] = set(pos["utterances"]) <= panel and len(panel) == 300
    rj_pos = json.loads((ROOT / cfg["p2rj_reference"]["positions"]).read_text())
    orig = {(p["utterance_id"], p["t"]) for p in rj_pos["positions"] if p["stratum"] == "EN-confusion"}
    checks["original_60_included_once"] = len(orig) == 60 and orig == {k for k, p in zip(keys, pos["positions"]) if p["original"]}
    # static: no gradient steering, no role access
    for rel in ("experiments/inference_cf_p2rje.py", "experiments/inference_cf_p2rj.py"):
        src = (ROOT / rel).read_text()
        bad = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in {"apply_steering", "hook_edit", "solve_scale", "chord_edit", "scaled_direction"}:
                    bad += [name for a in list(node.args) + [k.value for k in node.keywords]
                            if "grad" in (ast.get_source_segment(src, a) or "")]
        checks[f"no_gradient_into_edits:{Path(rel).name}"] = not bad
        checks[f"no_steering_hook:{Path(rel).name}"] = all(s not in src for s in
                                                          ("DecoderPostCrossAttnInterventionHook", "_edit_hook", "pulse_hook", "cached_decode"))
    new_src = "".join((ROOT / f"experiments/inference_cf_p2rje{s}.py").read_text() for s in ("", "_analyze"))
    forbidden = ("dev-confirm", "dev_confirm", "devconfirm", "d-test", "d_test", "dtest", "router-calib", "router_calib",
                 "seame", "fleurs", "vimed", "ascend", "qwen")
    checks["no_forbidden_roles_in_sources"] = not any(w in new_src.lower() for w in forbidden)
    # raw recomputation
    e = float(cfg["energy"]["e_star"])
    rows = {}
    for mode in ("bf16", "fp32"):
        ok = True
        for i, u in enumerate(pos["utterances"]):
            row = json.loads((run / "rows" / f"{i:03d}_{mode}.json").read_text())
            ok &= row["status"] == "ok" and row["manifest_hash"] == man["manifest_hash"] and \
                fhash(run / "rows" / f"{i:03d}_{mode}.npz") == row["vectors_sha256"]
            with np.load(run / "rows" / f"{i:03d}_{mode}.npz") as z:
                row["_v"] = {k: z[k].astype(np.float64) for k in z.files}
            rows[(mode, u)] = row
        checks[f"rows_ok_{mode}"] = ok
    rj_run = ROOT / cfg["p2rj_reference"]["run"]
    rj_ref = {}
    for i, u in enumerate(rj_pos["utterances"]):
        rr = json.loads((rj_run / "rows" / f"{i:03d}_bf16.json").read_text())
        with np.load(rj_run / "rows" / f"{i:03d}_bf16.npz") as z:
            for t, rec in rr["positions"].items():
                if (u, int(t)) in orig:
                    rj_ref[(u, int(t))] = (rec["values"]["margin"], z[f"t{t}_g_margin"].astype(np.float64), z[f"t{t}_r"].astype(np.float64))
    per = {"bf16": [], "fp32": []}
    zero_probe, missing, overlap_dev = True, 0, []
    prec = []
    for p in pos["positions"]:
        u, t = p["utterance_id"], p["t"]
        gts = {}
        for mode in ("bf16", "fp32"):
            row = rows[(mode, u)]
            rec = row["positions"].get(str(t))
            if rec is None:
                missing += 1; continue
            zero_probe &= bool(rec["identity"]["logits_bitwise_equal"])
            v = row["_v"]
            r, g = v[f"t{t}_r"], v[f"t{t}_g_margin"]
            gt = tan(g, r)
            gts[mode] = gt
            m0 = rec["values"]["margin"]
            A = float(np.linalg.norm(gt)) * e
            x = {"dlg": p["dialogue_id"], "A": A, "gap": -m0, "g": float(np.linalg.norm(g)), "gtan": float(np.linalg.norm(gt)),
                 "rho": math.inf if m0 >= 0 else A / (-m0),
                 "state": mode == "fp32" or (rec["identity"]["argmax"] == p["baseline_token"] and rec["identity"]["site_equals_recorder"]
                                             and abs(m0 - rec["identity"]["margin_best_other"]) <= 1e-3)}
            if mode == "bf16" and p["original"]:
                m_ref, g_ref, r_ref = rj_ref[(u, t)]
                gt_ref = tan(g_ref, r_ref)
                overlap_dev.append(max(abs(m0 - m_ref) / max(1, abs(m_ref)),
                                       abs(np.linalg.norm(gt) - np.linalg.norm(gt_ref)) / np.linalg.norm(gt_ref),
                                       1 - float(g @ g_ref / (np.linalg.norm(g) * np.linalg.norm(g_ref)))))
            per[mode].append(x)
        if len(gts) == 2:
            prec.append(float(gts["bf16"] @ gts["fp32"] / (np.linalg.norm(gts["bf16"]) * np.linalg.norm(gts["fp32"]))))
    checks["completeness_all_positions_both_passes"] = missing == 0 and len(per["bf16"]) == len(per["fp32"]) == len(pos["positions"])
    checks["zero_probe_value_identity_all"] = zero_probe
    checks["overlap_reproduces_p2rj_run1"] = len(overlap_dev) == 60 and max(overlap_dev) <= 1e-6
    notes["overlap_max_deviation"] = max(overlap_dev) if overlap_dev else None
    checks["state_basic_all"] = all(x["state"] for x in per["bf16"])
    reps, seed = cfg["analysis"]["bootstrap"]["replicates"], cfg["analysis"]["bootstrap"]["seed"]
    alpha = cfg["analysis"]["alpha_per_bound"]
    out = {"schema": "p2rje_audit_v1", "manifest_hash": man["manifest_hash"], "checks": checks, "notes": notes}

    def classify(xs):
        q = interval([(x["dlg"], 1.0 if x["rho"] >= 0.5 else 0.0) for x in xs], reps, seed, alpha)
        lev = "WEAK" if q[1][1] < 0.5 else "SUBSTANTIAL" if q[1][0] > 0.5 else "UNRESOLVED"
        return {"Q50": q, "leverage": lev, "n": len(xs), "dialogues": len({x["dlg"] for x in xs}),
                "median_gap": float(np.median([x["gap"] for x in xs])), "median_g": float(np.median([x["g"] for x in xs])),
                "median_gtan": float(np.median([x["gtan"] for x in xs])), "median_A": float(np.median([x["A"] for x in xs])),
                "median_rho": float(np.median([x["rho"] for x in xs])),
                "frac_rho": {th: float(np.mean([x["rho"] >= th for x in xs])) for th in (0.1, 0.5, 1.0)}}
    out["bf16"], out["fp32"] = classify(per["bf16"]), classify(per["fp32"])
    out["precision_median_cos"] = float(np.median(prec))
    prec_ok = out["precision_median_cos"] >= 0.9
    valid = all(ana["validity"].values())
    if not valid:
        diag = "INVALID_RUN_NO_DIAGNOSIS"
    elif out["bf16"]["leverage"] != out["fp32"]["leverage"] or not prec_ok:
        diag = "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED"
    else:
        diag = {"SUBSTANTIAL": "P2_RJ_E_DIRECTION_ISSUE", "WEAK": "P2_RJ_E_DIRECTION_AND_SITE_ISSUE"}.get(
            out["bf16"]["leverage"], "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED")
    agree = {}
    for mode in ("bf16", "fp32"):
        est, ci = out[mode]["Q50"]
        a = ana["Q50"][mode]
        agree[f"{mode}.Q50"] = abs(est - a["estimate"]) <= 1e-9 and all(abs(x - y) <= 1e-9 for x, y in zip(ci, a["ci"]))
        agree[f"{mode}.leverage"] = out[mode]["leverage"] == ana["leverage"][mode]
    agree["precision_cos"] = abs(out["precision_median_cos"] - ana["precision"]["cos"]["median"]) <= 1e-9
    agree["median_A"] = abs(out["bf16"]["median_A"] - ana["descriptive"]["all"]["A"]["median"]) <= 1e-9
    agree["median_rho"] = abs(out["bf16"]["median_rho"] - ana["descriptive"]["all"]["rho"]["median"]) <= 1e-9
    agree["diagnosis"] = diag == ana["diagnosis"]
    out["agreement_with_analysis"] = agree
    out["independent_diagnosis"] = diag
    out["verdict"] = "P2_RJ_E_AUDIT: PASS" if all(checks.values()) and all(agree.values()) else "P2_RJ_E_AUDIT: BLOCK"
    Path(ROOT / args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"verdict": out["verdict"], "checks": checks, "agreement": agree, "diagnosis": diag, "notes": notes,
                      "bf16": {k: out["bf16"][k] for k in ("Q50", "leverage")}}, indent=1, default=float))


if __name__ == "__main__":
    main()
