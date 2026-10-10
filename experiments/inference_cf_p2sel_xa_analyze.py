#!/usr/bin/env python
"""P2-SEL-XA analysis (CPU). Frozen contract 595e174: XA0 reconstruction/J/S/C/F, fixed lambda=0.95
finite-difference validity, historical-group statistics, precedence and F_src selection artifact; frozen
XA1 label rule. Historical groups are analysis-only and never construct or orient S/C/F.
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
import torch

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.readout import objective
from csasr.inference_cf.source_compatibility import compatibility
from experiments.inference_cf_p2dir_analyze import boot_stat, draw_weights, hi, jsonable, lo, unpack_bf16
from experiments.inference_cf_p2sel_t_analyze import group_diff, quant, span
import experiments.inference_cf_p2sel_xa as run

GROUPS = ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")
EXPECTED = {"EN_TP": 42, "EN_FN": 18, "ZH_TN": 55, "ZH_FP": 5, "EN_CORRECT": 60}
RECON_TOL = max(1e-3, 4 * float(torch.finfo(torch.bfloat16).eps))      # 0.03125


def fd_validity(rows: list[dict], cfg: dict) -> dict:
    """Frozen label-free material-subset first-order check."""
    fd = cfg["XA0"]["finite_difference"]
    sub = [r for r in rows if abs(r["P_nominal"]) >= 0.02 and abs(r["P_realized"]) >= 0.02]
    out = {"n": len(sub), "dialogues": span(sub)}
    if not sub:
        out.update(sign_obs_realized=0.0, sign_nominal_realized=0.0, median_ratio=None, rel_rms=None, ok=False)
        return out
    sgn = lambda x: (x > 0) - (x < 0)
    out["sign_obs_realized"] = sum(1 for r in sub if r["observed"] != 0 and sgn(r["observed"]) == sgn(r["P_realized"])) / len(sub)
    out["sign_nominal_realized"] = sum(1 for r in sub if sgn(r["P_nominal"]) == sgn(r["P_realized"])) / len(sub)
    out["median_ratio"] = float(np.median([r["observed"] / r["P_realized"] for r in sub]))
    num = sum((r["observed"] - r["P_realized"]) ** 2 for r in sub)
    den = sum(r["P_realized"] ** 2 for r in sub)
    out["rel_rms"] = math.sqrt(num / den)
    lo_r, hi_r = fd["median_realized_ratio_range"]
    out["ok"] = (len(sub) >= fd["minimum_material_rows"] and out["dialogues"] >= fd["minimum_material_dialogues"]
                 and out["sign_obs_realized"] >= fd["sign_agreement_min"]
                 and out["sign_nominal_realized"] >= fd["nominal_vs_realized_sign_agreement_min"]
                 and lo_r <= out["median_ratio"] <= hi_r and out["rel_rms"] <= fd["realized_relative_RMS_error_max"])
    return out


def decide_xa0(valid: bool, G: dict, keys, W) -> dict:
    fp, tp, fn = G["ZH_FP"], G["EN_TP"], G["EN_FN"]
    fp_s = [r for r in fp if r["F"] <= 0.10]
    tp_s = [r for r in tp if r["F"] >= 0.70]
    fn_s = [r for r in fn if r["F"] >= 0.70]
    d = group_diff(tp, fp, lambda r: r["F"], lambda r: r["F"], keys, W)
    p = {"fp_safety": len(fp_s) >= 4 and span(fp_s) >= 3, "tp_retention": len(tp_s) >= 34 and span(tp_s) >= 3,
         "contrast": d["point"] is not None and d["point"] >= 0.50 and d["lower80"] is not None and d["lower80"] > 0,
         "fn_available": len(fn_s) >= 9 and span(fn_s) >= 3}
    counts = {"fp_suppressed": len(fp_s), "fp_suppressed_dialogues": span(fp_s), "tp_retained": len(tp_s),
              "tp_retained_dialogues": span(tp_s), "fn_available": len(fn_s), "fn_available_dialogues": span(fn_s)}
    if not valid or d["valid_draws"] < 9900:
        lab = "P2_SEL_XA_INVALID"
    elif p["fp_safety"] and p["tp_retention"] and p["contrast"]:
        lab = "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    elif not p["fp_safety"] and p["fn_available"]:
        lab = "P2_SEL_XA_SOURCE_COMPATIBILITY_RECALL_ONLY"
    else:
        lab = "P2_SEL_XA_SOURCE_COMPATIBILITY_NOT_DISCRIMINATIVE"
    return {"label": lab, "predicates": p, "counts": counts, "F_TP_minus_FP": d}


def J_of(z: np.ndarray, t: int, sup, beg, part) -> float:
    return float(objective(torch.from_numpy(np.asarray(z, dtype=np.float32)), t, sup, beg, part)["J"])


def analyze_xa0(run_dir: Path) -> dict:
    m = json.loads((run_dir / "manifest.json").read_text())
    if m["stage"] != "xa0" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid XA0 manifest")
    cfg = json.loads((ROOT / m["config"]).read_text())
    con = json.loads((ROOT / m["construction"]).read_text())
    part, sup, beg = m["partition"], m["suppression"]["suppress"], m["suppression"]["begin"]
    e0 = {(r["utterance_id"], r["t"]): r for r in json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]}
    t0 = {(r["utterance_id"], r["t"]): r for r in json.loads((ROOT / "results/inference_cf/p2sel_t/t0_run1_analysis.json").read_text())["rows"]}
    sealed = json.loads((ROOT / run.P2DIR_RUN / "directions_sealed.json").read_text())
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    rows, fails, cache = [], [], {}
    for p in con["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        if i not in cache:
            r = json.loads((run_dir / f"rows/{i:03d}.json").read_text())
            if r.get("status") != "ok" or r["identity"] != uid or r["manifest_hash"] != m["manifest_hash"]:
                raise ValueError(f"XA0 row not ok {uid}")
            for f, h in ((f"rows/{i:03d}.npz", r["vectors_sha256"]), (f"rows/{i:03d}_logits.npz", r["logits_sha256"])):
                if file_hash(run_dir / f) != h:
                    raise ValueError("hash")
            for rel in (f"rows/{i:03d}.npz", f"rows/{i:03d}.json"):
                if file_hash(ROOT / run.P2DIR_RUN / rel) != sealed["files"][rel]:
                    raise ValueError("P2-DIR seal")
            with np.load(run_dir / f"rows/{i:03d}.npz") as z:
                V = {k: z[k] for k in z.files}
            with np.load(run_dir / f"rows/{i:03d}_logits.npz") as z:
                Lg = {k: unpack_bf16(z[k]) for k in z.files}
            with np.load(ROOT / run.P2DIR_RUN / f"rows/{i:03d}.npz") as z:
                P = {k: z[k] for k in z.files}
            prow = json.loads((ROOT / run.P2DIR_RUN / f"rows/{i:03d}.json").read_text())["positions"]
            cache = {i: (r, V, Lg, P, prow)}
        r, V, Lg, P, prow = cache[i]
        x = r["positions"][str(t)]
        pre = f"t{t}_"
        q, u, r1, r95 = (V[pre + k].astype(np.float64) for k in ("q", "u", "r1", "r095"))
        g = P[pre + "g_readout"]
        cp = compatibility(g, V[pre + "u"])
        J1, J95 = J_of(Lg[pre + "l1"], t, sup, beg, part), J_of(Lg[pre + "l095"], t, sup, beg, part)
        g64 = g.astype(np.float64)
        rel = float(np.max(np.abs(q + u - r1)) / max(float(np.max(np.abs(r1))), 1e-12))
        chk = {"query": x["query"] == prow[str(t)]["query"], "scale_calls": x["scale_calls"] == 1,
               "native_sum": x["native_sum_bitwise_r"], "r_capture": x["r_bitwise_capture"],
               "r_sealed_hb": x["r_bitwise_p2dir_hb"] and np.array_equal(V[pre + "r1"], P[pre + "hb"]),
               "logits_sealed_none": x["logits_bitwise_p2dir_none"], "q_unchanged": x["q_unchanged_in_scratch"],
               "cache": x["cache_object_and_length"], "recon_tol": rel <= RECON_TOL,
               "J_matches_sealed_readout": abs(J1 - prow[str(t)]["directions"]["D2"]["provenance"]["J"]) <= cfg["XA0"]["J_audit_abs_tolerance"],
               "g_present_float32_1280": g.dtype == np.float32 and g.shape == (1280,),
               "u095_is_scaled_u": np.array_equal(V[pre + "u095"], (0.95 * torch.from_numpy(V[pre + "u"]).to(torch.bfloat16))
                                                  .to(torch.bfloat16).float().numpy())}
        if not all(chk.values()):
            fails.append((uid, t, [k for k, v in chk.items() if not v]))
        o = e0[(uid, t)]
        rows.append({"utterance_id": uid, "t": t, "query": x["query"], "dialogue_id": p["dialogue_id"], "stratum": p["stratum"],
                     "group": o["group"], "J": J1, "J095": J95, "S": cp["S"], "C": cp["C"], "F": cp["F"],
                     "g_norm": cp["g_norm"], "u_norm": cp["u_norm"], "observed": J95 - J1, "P_nominal": -0.05 * cp["S"],
                     "P_realized": float(g64 @ (r95 - r1)), "recon_rel_err": rel, "E_1s": o["E"], "R_B": o["R_B"],
                     "E_tok": t0[(uid, t)]["E_tok"], "checks": chk})
    G = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    rt = json.loads((run_dir / "runtime.json").read_text())
    fd = fd_validity(rows, cfg)
    validity = {"rows_180": len(rows) == 180, "row_checks": not fails, "groups": {g: len(v) for g, v in G.items()} == EXPECTED,
                "runtime_completed": rt.get("status") == "completed",
                "no_backward_lid_steering": rt["autograd_calls"] == 0 and rt["readout_calls"] == 0 and rt["lid_calls"] == 0 and rt["steering_calls"] == 0,
                "finite_difference": fd["ok"]}
    keys, W = draw_weights([r["dialogue_id"] for r in rows], 10000, 240924)
    dec = decide_xa0(all(validity.values()), G, keys, W)
    fields = ("S", "C", "F", "u_norm", "g_norm", "J", "E_1s", "E_tok", "R_B", "observed", "P_nominal", "P_realized")
    desc = {g: {"n": len(rs), "dialogues": sorted({r["dialogue_id"] for r in rs}), **{f: quant([r[f] for r in rs]) for f in fields},
                "dialogue_mean_ci90": {f: boot_stat([(r["dialogue_id"], r[f]) for r in rs], keys, W, 0.10) for f in ("S", "C", "F")}}
            for g, rs in G.items()}
    return {"schema": "p2_sel_xa_xa0_analysis_v1", "manifest_hash": m["manifest_hash"], "validity": validity, "row_failures": fails,
            "finite_difference": fd, "decision": dec, "label": dec["label"], "descriptive": desc, "runtime": rt,
            "counts": {g: len(v) for g, v in G.items()}, "rows": rows}


def decide_xa1(valid: bool, f: dict, ratios: dict, obs: dict, en_reg: dict, th: dict) -> str:
    """Frozen XA1 precedence."""
    if not valid:
        return "P2_SEL_XA_INVALID"
    safety = (lo(f["en"]) >= th["correct_margin_simultaneous_lower_min_nat"] and lo(f["zh"]) >= th["correct_margin_simultaneous_lower_min_nat"]
              and obs["en"] <= th["correct_corruption_observed_max"] and obs["zh"] <= th["correct_corruption_observed_max"]
              and hi(f["corr_en"]) <= th["correct_corruption_simultaneous_upper_max"] and hi(f["corr_zh"]) <= th["correct_corruption_simultaneous_upper_max"]
              and en_reg["mean_new_minus_old"] >= -1e-6 and en_reg["new_corruptions"] == 0)
    if not safety:
        return "P2_SEL_XA_GATE_STILL_UNSAFE"
    if ratios["benefit_retention"] < th["benefit_retention_min"]:
        return "P2_SEL_XA_GATE_TOO_CONSERVATIVE"
    if not (lo(f["conf"]) > th["new_confusion_simultaneous_lower_strictly_above"] and ratios["harm_ratio"] <= th["harm_ratio_max"]
            and f["paired_zh"]["estimate"] >= th["paired_ZH_new_minus_old_min_nat"] and lo(f["paired_zh"]) > th["paired_ZH_simultaneous_lower_strictly_above"]):
        return "P2_SEL_XA_NO_MATERIAL_GAIN"
    return "P2_SEL_XA_GATE_SUPPORTED"


def selection_artifact(ana: dict, analysis_path: str, audit_path: str) -> dict:
    if ana["label"] != "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED":
        raise ValueError("no gate selected")
    cfg = json.loads((ROOT / run.CONFIG).read_text())
    gates = [{"utterance_id": r["utterance_id"], "t": r["t"], "query": r["query"], "E_old": r["E_1s"], "R_B": r["R_B"], "F_src": r["F"],
              "g_old": r["E_1s"] * r["R_B"], "g_new": r["E_1s"] * r["R_B"] * r["F"]} for r in ana["rows"]]
    doc = {"schema": "p2_sel_xa_selection_v1", "selected": "F_src", "formula": cfg["XA0"]["F_src"], "gate": cfg["XA1"]["gate"],
           "xa0_label": ana["label"], "xa0_analysis_sha256": file_hash(ROOT / analysis_path), "xa0_audit_sha256": file_hash(ROOT / audit_path),
           "config_hash": digest(cfg), "source_sha256": {p: file_hash(ROOT / p) for p in ("src/csasr/inference_cf/source_compatibility.py",
                                                                                          "experiments/inference_cf_p2sel_xa.py")},
           "predicates": ana["decision"]["predicates"], "steering_outcome_exists": False, "gates": gates}
    doc["gates_hash"] = digest(gates)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("xa0", "select"))
    ap.add_argument("--run")
    ap.add_argument("--analysis")
    ap.add_argument("--audit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("output exists; never overwrite")
    if args.stage == "xa0":
        res = jsonable(analyze_xa0(ROOT / args.run))
        atomic_json(out, res)
        print(json.dumps({"label": res["label"], "validity": res["validity"], "finite_difference": res["finite_difference"],
                          "decision": res["decision"], "counts": res["counts"]}, indent=1))
    else:
        doc = selection_artifact(json.loads((ROOT / args.analysis).read_text()), args.analysis, args.audit)
        atomic_json(out, jsonable(doc))
        print(json.dumps({k: doc[k] for k in ("selected", "formula", "gates_hash")}, indent=1))


if __name__ == "__main__":
    main()
