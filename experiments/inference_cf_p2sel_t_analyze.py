#!/usr/bin/env python
"""P2-SEL-T analysis (CPU). Frozen contract fdace84: T0 token-local features, historical-group
statistics, frozen predicates and precedence, R_TOK selection artifact; T1 family-6 statistics and
label precedence. Historical group labels are analysis-only (never an E_tok input).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_r2 import local_support
from experiments.inference_cf_p2dir_analyze import boot_stat, draw_weights, hi, jsonable, lo, logit_metrics, unpack_bf16
import experiments.inference_cf_p2sel_t as run

GROUPS = ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")
EXPECTED = {"EN_TP": 42, "EN_FN": 18, "ZH_TN": 55, "ZH_FP": 5, "EN_CORRECT": 60}
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")


def span(rs) -> int:
    return len({r["dialogue_id"] for r in rs})


def group_diff(rows_a, rows_b, fa, fb, keys, W) -> dict:
    """Dialogue-equal mean(A) - mean(B) on shared draws; each group's own represented denominator."""
    def per(rows, f):
        acc = {}
        for r in rows:
            acc.setdefault(r["dialogue_id"], []).append(float(f(r)))
        return np.array([np.mean(acc[k]) if k in acc else np.nan for k in keys])
    ma, mb = per(rows_a, fa), per(rows_b, fb)
    ha, hb = ~np.isnan(ma), ~np.isnan(mb)
    point = float(np.mean(ma[ha]) - np.mean(mb[hb])) if ha.any() and hb.any() else None
    da, db = W[:, ha].sum(1), W[:, hb].sum(1)
    ok = (da > 0) & (db > 0)
    d = (W[:, ha] @ ma[ha])[ok] / da[ok] - (W[:, hb] @ mb[hb])[ok] / db[ok]
    return {"point": point, "valid_draws": int(ok.sum()),
            "lower80": float(np.quantile(d, 0.20)) if ok.any() else None,
            "ci90": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))] if ok.any() else None}


# ---- frozen T0 decision ------------------------------------------------------------------------------

def predicates(G: dict, keys, W) -> dict:
    fp, tp, fn = G["ZH_FP"], G["EN_TP"], G["EN_FN"]
    fp_sup = [r for r in fp if r["E_tok"] <= 0.10 and r["E_1s"] - r["E_tok"] >= 0.25]
    tp_ev = [r for r in tp if r["E_tok"] >= 0.20]
    sep = group_diff(tp, fp, lambda r: r["E_tok"], lambda r: r["E_tok"], keys, W)
    fp_c = [r for r in fp if r["C_tokctx"] <= -0.25]
    tp_c = [r for r in tp if r["C_tokctx"] >= 0]
    csep = group_diff(tp, fp, lambda r: r["C_tokctx"], lambda r: r["C_tokctx"], keys, W)
    fn_ev = [r for r in fn if r["E_tok"] >= 0.20]
    p = {"fp_suppression": len(fp_sup) >= 4 and span(fp_sup) >= 3,
         "tp_evidence": len(tp_ev) >= 34 and span(tp_ev) >= 3,
         "separation": sep["point"] is not None and sep["point"] >= 0.25 and sep["lower80"] is not None and sep["lower80"] > 0,
         "fp_contrast": len(fp_c) >= 4 and span(fp_c) >= 3,
         "tp_contrast": len(tp_c) >= 26 and span(tp_c) >= 3,
         "contrast_separation": csep["point"] is not None and csep["point"] >= 0.25 and csep["lower80"] is not None and csep["lower80"] > 0,
         "fn_recall": len(fn_ev) >= 9 and span(fn_ev) >= 3}
    counts = {"fp_suppressed": len(fp_sup), "fp_suppressed_dialogues": span(fp_sup), "tp_evidence": len(tp_ev),
              "tp_evidence_dialogues": span(tp_ev), "fp_contrast": len(fp_c), "fp_contrast_dialogues": span(fp_c),
              "tp_contrast": len(tp_c), "tp_contrast_dialogues": span(tp_c), "fn_recall": len(fn_ev),
              "fn_recall_dialogues": span(fn_ev)}
    return {"predicates": p, "counts": counts, "E_tok_TP_minus_FP": sep, "C_TP_minus_FP": csep,
            "draws_ok": sep["valid_draws"] >= 9900 and csep["valid_draws"] >= 9900}


def decide_t0(valid: bool, P: dict) -> str:
    if not valid or not P["draws_ok"]:
        return "P2_SEL_T_INVALID"
    p = P["predicates"]
    if p["fp_suppression"] and p["tp_evidence"] and p["separation"]:
        return "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED"
    if p["fp_contrast"] and p["tp_contrast"] and p["contrast_separation"]:
        return "P2_SEL_T_CONTEXT_CONTRAST_ONLY"
    if p["fn_recall"] and not p["fp_suppression"]:
        return "P2_SEL_T_RECALL_ONLY"
    return "P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE"


def quant(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    return None if a.size == 0 else {"n": int(a.size), "median": float(np.median(a)), "q25": float(np.quantile(a, .25)),
                                     "q75": float(np.quantile(a, .75))}


def analyze_t0(run_dir: Path) -> dict:
    m = json.loads((run_dir / "manifest.json").read_text())
    if m["stage"] != "t0" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid T0 manifest")
    cfg = json.loads((ROOT / m["config"]).read_text())
    con = json.loads((ROOT / m["construction"]).read_text())
    e0 = {(r["utterance_id"], r["t"]): r for r in json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]}
    null = json.loads((ROOT / run.E0_RUN / "null.json").read_text())
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    rows, fails, cache = [], [], {}
    for p in con["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        if i not in cache:
            r = json.loads((run_dir / f"rows/{i:03d}.json").read_text())
            if r.get("status") != "ok" or r["identity"] != uid or r["manifest_hash"] != m["manifest_hash"]:
                raise ValueError(f"T0 row not ok {uid}")
            if file_hash(run_dir / f"rows/{i:03d}.npz") != r["vectors_sha256"]:
                raise ValueError("vector hash")
            e0raw = json.loads((ROOT / run.E0_RUN / f"rows/{i:03d}.json").read_text())["positions"]
            with np.load(run_dir / f"rows/{i:03d}.npz") as z:
                V = {k: z[k] for k in z.files}
            cache = {i: (r, e0raw, V)}
        r, e0raw, V = cache[i]
        x = r["positions"][str(t)]
        old = e0[(uid, t)]
        raw = e0raw[str(t)]
        mean = V[f"t{t}_mean"]
        bounds = run.candidate_bounds(x["window"][2], x["window"][3])
        masses = {k: run.crop_mass(mean, a, b, x["heard_samples"]) for k, (a, b) in bounds.items()}
        E = {k: local_support(x["probs"][k]["pi_E"], x["probs"][k]["pi_M"], null["pi_E"], null["pi_M"])["E"] for k in run.ORDER}
        E_1s = local_support(raw["long"]["pi_E"], raw["long"]["pi_M"], null["pi_E"], null["pi_M"])["E"]
        feat = run.token_features(E, masses)
        heard = x["heard_samples"]
        chk = {"query": x["query"] == old["query"], "window": x["window"] == old["window"],
               "W_mass": abs(x["attention_mass_W"] - old["attention_mass"]) <= 1e-6,
               "W_mass_from_vector": abs(float(mean[x["window"][0]:x["window"][1]].sum()) - old["attention_mass"]) <= 1e-6,
               "E_1s_reproduces": abs(E_1s - old["E"]) <= 1e-6 and abs(E_1s - old["E_p2r"]) <= 1e-6,
               "center_equals_e0_short": list(bounds["C"]) == raw["short"]["bounds"] == x["bounds"]["C"],
               "center_reused": x["probs"]["C"]["pi_E"] == raw["short"]["pi_E"] and x["probs"]["C"]["pi_M"] == raw["short"]["pi_M"],
               "bounds_recorded": all(list(bounds[k]) == x["bounds"][k] for k in run.ORDER),
               "inside_W_and_heard": all(x["window"][2] <= a < b <= x["window"][3] <= heard for a, b in bounds.values()),
               "masses_reproduce": all(abs(masses[k] - x["masses"][k]) <= 1e-12 for k in run.ORDER),
               "selection_reproduces": feat["j"] == x["j"] and feat["E_tok"] == x["E_tok"],
               "new_lid_provider": all(x["probs"][k].get("source") != "new" or (abs(x["probs"][k]["softmax_sum"] - 1) <= 1e-6
                                       and x["probs"][k]["n_probs"] == 100 and x["probs"][k]["all_finite_nonneg"]) for k in run.ORDER),
               "mean_valid": bool(np.isfinite(mean).all() and (mean >= 0).all() and abs(mean.sum() - 1) <= 1e-9)}
        if not all(chk.values()):
            fails.append((uid, t, [k for k, v in chk.items() if not v]))
        rows.append({"utterance_id": uid, "t": t, "query": x["query"], "dialogue_id": p["dialogue_id"], "stratum": p["stratum"],
                     "group": old["group"], "E_1s": E_1s, "E_L": E["L"], "E_C": E["C"], "E_R": E["R"],
                     "a_L": masses["L"], "a_C": masses["C"], "a_R": masses["R"], **feat, "R_B": old["R_B"],
                     "bounds": {k: list(v) for k, v in bounds.items()}, "timing_error_sec": old.get("timing_error_sec"),
                     "checks": chk})
    G = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    rt = json.loads((run_dir / "runtime.json").read_text())
    validity = {"rows_180": len(rows) == 180, "row_checks": not fails, "groups": {g: len(v) for g, v in G.items()} == EXPECTED,
                "runtime_completed": rt.get("status") == "completed",
                "no_steering_readout_autograd": rt.get("autograd_calls") == 0 and rt.get("readout_calls") == 0 and rt.get("steering_calls") == 0,
                "lid_calls_le_360": rt.get("lid_calls", 10 ** 9) <= cfg["compute"]["T0_native_lid_calls_max"]}
    keys, W = draw_weights([r["dialogue_id"] for r in rows], 10000, 240924)
    P = predicates(G, keys, W)
    label = decide_t0(all(validity.values()), P)
    fields = ("E_1s", "E_L", "E_C", "E_R", "E_tok", "E_ctx", "C_tokctx", "a_L", "a_C", "a_R", "a_tok", "a_score", "timing_error_sec")
    desc = {}
    for g, rs in G.items():
        perd = {}
        for r in rs:
            perd[r["dialogue_id"]] = perd.get(r["dialogue_id"], 0) + 1
        desc[g] = {"n": len(rs), "dialogues": sorted(perd), "per_dialogue": perd, **{f: quant([r[f] for r in rs]) for f in fields},
                   "selected_fraction": {k: (sum(r["j"] == k for r in rs) / len(rs)) if rs else None for k in run.ORDER},
                   "E_tok_positive_rate": (sum(r["E_tok"] > 0 for r in rs) / len(rs)) if rs else None,
                   "E_tok_mean_ci90": boot_stat([(r["dialogue_id"], r["E_tok"]) for r in rs], keys, W, 0.10)}
    return {"schema": "p2_sel_t_t0_analysis_v1", "manifest_hash": m["manifest_hash"], "validity": validity, "row_failures": fails,
            "counts": {g: len(v) for g, v in G.items()}, "statistics": P, "label": label, "descriptive": desc, "runtime": rt,
            "rows": rows}


def selection_artifact(ana: dict, analysis_path: str, audit_path: str) -> dict:
    if ana["label"] != "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED":
        raise ValueError("no repair selected")
    cfg = json.loads((ROOT / run.CONFIG).read_text())
    gates = [{"utterance_id": r["utterance_id"], "t": r["t"], "query": r["query"], "E_old": r["E_1s"], "E_new": r["E_tok"],
              "j": r["j"], "R_B": r["R_B"], "g_old": r["E_1s"] * r["R_B"], "g_new": r["E_tok"] * r["R_B"]} for r in ana["rows"]]
    doc = {"schema": "p2_sel_t_selection_v1", "selected_repair": "R_TOK", "formula": cfg["T0"]["only_repair"],
           "windows": cfg["T0"]["windows"], "attention": cfg["T0"]["attention"], "tie": cfg["T0"]["tie"],
           "t0_label": ana["label"], "t0_analysis": analysis_path, "t0_analysis_sha256": file_hash(ROOT / analysis_path),
           "t0_audit": audit_path, "t0_audit_sha256": file_hash(ROOT / audit_path), "config_hash": digest(cfg),
           "population_sha256": cfg["source_sha256"]["results/inference_cf/p2rj/positions.json"],
           "source_sha256": {p: file_hash(ROOT / p) for p in ("experiments/inference_cf_p2sel_t.py", "src/csasr/inference_cf/core_r2.py")},
           "decision_predicates": ana["statistics"]["predicates"], "steering_outcome_exists": False, "gates": gates}
    doc["gates_hash"] = digest(gates)
    return doc


# ---- T1 ------------------------------------------------------------------------------------------------

def decide_t1(valid: bool, f: dict, ratios: dict, obs: dict, en_reg: dict, th: dict) -> dict:
    if not valid:
        return {"label": "P2_SEL_T_INVALID"}
    safety = {"en_margin_lower": lo(f["en"]) is not None and lo(f["en"]) >= th["correct_margin_simultaneous_lower_min_nat"],
              "zh_margin_lower": lo(f["zh"]) is not None and lo(f["zh"]) >= th["correct_margin_simultaneous_lower_min_nat"],
              "en_corr_observed": obs["en"] is not None and obs["en"] <= th["correct_corruption_observed_max"],
              "zh_corr_observed": obs["zh"] is not None and obs["zh"] <= th["correct_corruption_observed_max"],
              "en_corr_upper": hi(f["corr_en"]) is not None and hi(f["corr_en"]) <= th["correct_corruption_simultaneous_upper_max"],
              "zh_corr_upper": hi(f["corr_zh"]) is not None and hi(f["corr_zh"]) <= th["correct_corruption_simultaneous_upper_max"],
              "en_no_regression_mean": en_reg["mean_new_minus_old"] >= -1e-6,
              "en_zero_new_corruptions": en_reg["new_corruptions"] == 0}
    rest = {"benefit_lower": lo(f["conf"]) is not None and lo(f["conf"]) > th["new_confusion_simultaneous_lower_strictly_above"],
            "harm_ratio": ratios["harm_ratio"] <= th["harm_ratio_max"],
            "paired_zh_point": f["paired_zh"]["estimate"] >= th["paired_ZH_new_minus_old_min_nat"],
            "paired_zh_lower": lo(f["paired_zh"]) is not None and lo(f["paired_zh"]) > th["paired_ZH_simultaneous_lower_strictly_above"]}
    out = {"safety": safety, "benefit_point": ratios["benefit_retention"] >= th["benefit_retention_min"], "other": rest}
    if not all(safety.values()):
        out["label"] = "P2_SEL_T_TOKEN_GATE_STILL_UNSAFE"
    elif not out["benefit_point"]:
        out["label"] = "P2_SEL_T_TOKEN_GATE_TOO_CONSERVATIVE"
    elif not all(rest.values()):
        out["label"] = "P2_SEL_T_NO_MATERIAL_GAIN"
    else:
        out["label"] = "P2_SEL_T_TOKEN_GATE_SUPPORTED"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("t0", "select"))
    ap.add_argument("--run")
    ap.add_argument("--analysis")
    ap.add_argument("--audit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("output exists; never overwrite")
    if args.stage == "t0":
        res = jsonable(analyze_t0(ROOT / args.run))
        atomic_json(out, res)
        print(json.dumps({"label": res["label"], "validity": res["validity"], "counts": res["counts"],
                          "statistics": {k: v for k, v in res["statistics"].items()}}, indent=1))
    else:
        doc = selection_artifact(json.loads((ROOT / args.analysis).read_text()), args.analysis, args.audit)
        atomic_json(out, jsonable(doc))
        print(json.dumps({k: doc[k] for k in ("selected_repair", "formula", "gates_hash")}, indent=1))


if __name__ == "__main__":
    main()
