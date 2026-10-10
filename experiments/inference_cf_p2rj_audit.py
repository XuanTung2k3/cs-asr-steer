#!/usr/bin/env python
"""Independent P2-RJ post-run audit (CPU). Does not import the P2-RJ analysis or runner modules.

It recomputes from the raw saved vectors and the raw P2-R run3 rows (not the positions file's
derived values):

* the margins, gradient and tangent norms, A, rho, cosines, kappa and predicted changes;
* the observed P2-R changes;
* the Bonferroni dialogue-bootstrap intervals of Q50 and K_plus;
* the pooled first-order Pearson r;
* the precision cosine;
* the frozen classification and the diagnosis.

It also checks completeness, provenance, data roles, the zero-probe / no-gradient-steering
guarantees and that no decision source changed after the freeze.
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

ARMS = ("plus_d", "minus_d", "random")


def canon(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()


def fhash(p: Path) -> str:
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


def top(summary, tok):
    for i, lp in zip(summary["top_ids"], summary["top_logp"]):
        if int(i) == int(tok):
            return float(lp)
    return None


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
    changed = [p for p, h in man["sources"].items() if fhash(ROOT / p) != h]
    notes["sources_changed_since_freeze"] = changed
    checks["decision_sources_unchanged"] = not [p for p in changed if p != "INFERENCE_STEERING_IMPLEMENTATION_PLAN.md"]
    checks["config_hash"] = canon(cfg) == man["config_hash"]
    checks["positions_hash"] = canon({k: v for k, v in pos.items() if k != "positions_hash"}) == man["positions_hash"]
    rt = json.loads((run / "runtime.json").read_text())
    checks["runtime_completed"] = rt.get("status") == "completed"
    # data role: exactly the P2-R D-dev-select utterances
    pop = json.loads((ROOT / cfg["p2r_reference"]["population"]).read_text())
    checks["utterances_are_p2r_population"] = pos["utterances"] == pop["utterances"]
    p2r_run = ROOT / cfg["p2r_reference"]["run"]
    p2r_man = json.loads((p2r_run / "manifest.json").read_text())
    checks["p2r_run3_manifest"] = p2r_man["manifest_hash"] == cfg["p2r_reference"]["run_manifest_hash"]
    p2r_rows = {u: json.loads((p2r_run / "rows" / f"{i:03d}.json").read_text()) for i, u in enumerate(pop["utterances"])}
    # gradient never used as steering (static)
    src = (ROOT / "experiments/inference_cf_p2rj.py").read_text()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in {"apply_steering", "hook_edit", "solve_scale", "chord_edit", "scaled_direction"}:
                for a in list(node.args) + [k.value for k in node.keywords]:
                    if "grad" in (ast.get_source_segment(src, a) or ""):
                        bad.append(name)
    checks["no_gradient_into_edit_functions"] = not bad
    checks["no_steering_hook_in_runner"] = "DecoderPostCrossAttnInterventionHook" not in src and "_edit_hook" not in src \
        and "pulse_hook" not in src and "cached_decode" not in src
    # raw recomputation
    e = float(cfg["energy"]["e_star"])
    rows = {}
    for mode in ("bf16", "fp32"):
        for i, u in enumerate(pos["utterances"]):
            p = run / "rows" / f"{i:03d}_{mode}.json"
            row = json.loads(p.read_text())
            with np.load(run / "rows" / f"{i:03d}_{mode}.npz") as z:
                row["_v"] = {k: z[k].astype(np.float64) for k in z.files}
            checks.setdefault(f"rows_ok_{mode}", True)
            checks[f"rows_ok_{mode}"] &= row["status"] == "ok" and row["manifest_hash"] == man["manifest_hash"] \
                and fhash(run / "rows" / f"{i:03d}_{mode}.npz") == row["vectors_sha256"]
            rows[(mode, u)] = row
    per = {"bf16": [], "fp32": []}
    zero_probe, missing = True, 0
    prec = []
    for p in pos["positions"]:
        u, t = p["utterance_id"], p["t"]
        pr = p2r_rows[u]["pulses"][str(t)]
        tgt = [int(x) for x in pr["none"]["top_ids"] if int(x) not in set(p["target_ids"])]
        cstar = tgt[0]
        for mode in ("bf16", "fp32"):
            row = rows[(mode, u)]
            rec = row["positions"].get(str(t))
            if rec is None:
                missing += 1
                continue
            zero_probe &= bool(rec["identity"]["logits_bitwise_equal"])
            v = row["_v"]
            k = f"t{t}_"
            r, g = v[k + "r"], v[k + "g_margin"]
            rh = r / np.linalg.norm(r)
            gt = g - rh * (rh @ g)
            A = np.linalg.norm(gt) * e
            m0 = rec["values"]["margin"]
            x = {"dlg": p["dialogue_id"], "stratum": p["stratum"], "A": A, "m0": m0, "cstar_ok": cstar == p["competitor"]}
            if mode == "bf16":
                x["m0_vs_p2r"] = abs(m0 - pr["none"]["margin_ref_vs_competitor"])
                idt = rec["identity"]
                tie_ok = pr["none"]["argmax"] == p["baseline_token"] or idt["logp_p2r_argmax"] == idt["logp_baseline_token"]
                x["state"] = (idt["argmax"] == p["baseline_token"] and tie_ok and
                              abs(idt["pre_norm_audit_style"] - pr["arms"]["plus_d"]["pre_norm"]) <= 1e-6 * pr["arms"]["plus_d"]["pre_norm"] and
                              abs(rec["identity"]["cos_d_state"] - pr["cos_d_state"]) <= 1e-6 and
                              all(rec["solver"][a]["s"] == pr["arms"][a]["solver"]["s"] for a in ARMS))
            d = v[k + "d"] / np.linalg.norm(v[k + "d"])
            dirs = {"plus_d": d, "minus_d": -d, "random": v[k + "v_random"] / np.linalg.norm(v[k + "v_random"])}
            for a in ARMS:
                dl = v[k + "delta_" + a]
                x[f"kappa_{a}"] = float(gt @ dl) / A
                x[f"cos_{a}"] = float(gt @ dirs[a]) / float(np.linalg.norm(gt))
                x[f"pred_{a}"] = float(g @ dl)
                s = pr["arms"][a]["summary"]
                lc, lc0 = top(s, cstar), top(pr["none"], cstar)
                x[f"obs_{a}"] = None if lc is None or lc0 is None else \
                    (s["logp_ref"] - lc) - (pr["none"]["logp_ref"] - lc0)
            if p["stratum"] == "EN-confusion":
                x["rho"] = math.inf if -m0 <= 0 else A / (-m0)
            per[mode].append(x)
            if mode == "bf16":
                x["_gt"] = gt
            elif per["bf16"] and "_gt" in per["bf16"][-1] and per["bf16"][-1]["state"]:
                gb = per["bf16"][-1]["_gt"]
                prec.append(float(gb @ gt / (np.linalg.norm(gb) * np.linalg.norm(gt))))
    checks["completeness_180_each_pass"] = missing == 0 and len(per["bf16"]) == len(per["fp32"]) == 180
    checks["zero_probe_value_identity_all"] = zero_probe
    checks["competitor_rederived_from_p2r_rows"] = all(x["cstar_ok"] for x in per["bf16"])
    checks["baseline_margin_equals_p2r"] = max(x["m0_vs_p2r"] for x in per["bf16"]) <= 1e-3
    notes["state_identity_all"] = all(x["state"] for x in per["bf16"])
    reps, seed = cfg["analysis"]["bootstrap"]["replicates"], cfg["analysis"]["bootstrap"]["seed"]
    af = 0.05 / 2
    out = {"schema": "p2rj_audit_v1", "manifest_hash": man["manifest_hash"], "checks": checks, "notes": notes}

    def classify(xs):
        en = [x for x in xs if x["stratum"] == "EN-confusion"]
        q = interval([(x["dlg"], 1.0 if x["rho"] >= 0.5 else 0.0) for x in en], reps, seed, af)
        kp = interval([(x["dlg"], x["kappa_plus_d"]) for x in en], reps, seed, af)
        lev = "WEAK" if q[1][1] < 0.5 else "SUBSTANTIAL" if q[1][0] > 0.5 else "UNRESOLVED"
        ali = "MISALIGNED" if kp[1][1] < 0.10 else "ALIGNED" if kp[1][0] >= 0.25 else "PARTIAL"
        lab = {("SUBSTANTIAL", "MISALIGNED"): "P2_RJ_DIRECTION_ISSUE", ("WEAK", "MISALIGNED"): "P2_RJ_DIRECTION_AND_SITE_ISSUE",
               ("WEAK", "PARTIAL"): "P2_RJ_SITE_OR_SENSITIVITY_ISSUE", ("WEAK", "ALIGNED"): "P2_RJ_SITE_OR_SENSITIVITY_ISSUE"}
        return {"Q50": q, "K_plus": kp, "leverage": lev, "alignment": ali, "label": lab.get((lev, ali), "P2_RJ_STILL_AMBIGUOUS"),
                "median_rho": float(np.median([x["rho"] for x in en])), "median_gap": float(np.median([-x["m0"] for x in en])),
                "median_A": float(np.median([x["A"] for x in en]))}
    ok_keys = {i for i, x in enumerate(per["bf16"]) if x["state"]}
    checks["state_mismatch_within_frozen_limit"] = len(per["bf16"]) - len(ok_keys) <= 9
    notes["state_mismatch_positions"] = len(per["bf16"]) - len(ok_keys)
    out["bf16"] = classify([x for i, x in enumerate(per["bf16"]) if i in ok_keys])
    out["fp32"] = classify([x for i, x in enumerate(per["fp32"]) if i in ok_keys])
    pairs = [(x["pred_" + a], x["obs_" + a]) for x in per["bf16"] if x["state"] for a in ARMS if x["obs_" + a] is not None]
    xy = np.array(pairs)
    out["fd_pearson_pooled"] = float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1])
    out["precision_median_cos"] = float(np.median(prec)) if prec else None
    # agreement with the analysis
    agree = {}
    for mode in ("bf16", "fp32"):
        for key in ("Q50", "K_plus"):
            est, ci = out[mode][key]
            a = ana[mode][key]
            agree[f"{mode}.{key}"] = abs(est - a["estimate"]) <= 1e-9 and all(abs(p - q) <= 1e-9 for p, q in zip(ci, a["ci"]))
        agree[f"{mode}.classes"] = (out[mode]["leverage"], out[mode]["alignment"]) == (ana[mode]["leverage"], ana[mode]["alignment"])
    agree["fd_estimate"] = abs(out["fd_pearson_pooled"] - ana["fd"]["pooled_margin"]["estimate"]) <= 1e-9
    agree["precision_cos"] = out["precision_median_cos"] is not None and \
        abs(out["precision_median_cos"] - ana["precision"]["cos"]["median"]) <= 1e-9
    reasons = []
    if out["bf16"]["label"] != out["fp32"]["label"]:
        reasons.append("precision_label_change")
    if not (out["precision_median_cos"] or 0) >= 0.9:
        reasons.append("precision_cos")
    diag = out["bf16"]["label"] if not reasons and ana["fd"]["ok"] and all(ana["validity"].values()) else "P2_RJ_STILL_AMBIGUOUS"
    agree["diagnosis"] = diag == ana["decision"]["diagnosis"]
    out["agreement_with_analysis"] = agree
    out["independent_diagnosis"] = diag
    ok = all(checks.values()) and all(agree.values())
    out["verdict"] = "P2_RJ_AUDIT: PASS" if ok else "P2_RJ_AUDIT: BLOCK"
    Path(ROOT / args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"verdict": out["verdict"], "checks": checks, "agreement": agree, "diagnosis": diag,
                      "bf16": {k: out["bf16"][k] for k in ("leverage", "alignment", "label")}}, indent=1))


if __name__ == "__main__":
    main()
