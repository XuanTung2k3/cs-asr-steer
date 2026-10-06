#!/usr/bin/env python
"""P2-SEL-E analysis (CPU). Frozen revised contract (b693b8d): E0 decomposition, failure groups,
hypotheses H_E1..H_E4, deterministic precedence and single-branch selection; E1 family-6 statistics and
label precedence. Evaluator labels (strata, oracle timing) are used ONLY for diagnosis grouping, never
as an E_new input.
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
from experiments.inference_cf_p2dir_analyze import (boot_stat, draw_weights, hi, jsonable, lo,
                                                    logit_metrics, unpack_bf16)
import experiments.inference_cf_p2sel_e as run

GROUPS = ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")
EXPECTED = {"EN_TP": 42, "EN_FN": 18, "ZH_TN": 55, "ZH_FP": 5, "EN_CORRECT": 60}
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")


def group_of(stratum: str, e: float) -> str:
    if stratum == "EN-confusion":
        return "EN_TP" if e > 0 else "EN_FN"
    if stratum == "ZH-correct":
        return "ZH_FP" if e > 0 else "ZH_TN"
    return "EN_CORRECT"


def span(rows) -> int:
    return len({r["dialogue_id"] for r in rows})


def paired_rate_diff(rows_a, rows_b, fa, fb, keys, W) -> dict:
    """Dialogue-equal rate(A) - rate(B) on shared draws; draws missing either group omitted."""
    def per_dialogue(rows, f):
        acc = {}
        for r in rows:
            acc.setdefault(r["dialogue_id"], []).append(1.0 if f(r) else 0.0)
        m = np.array([np.mean(acc[k]) if k in acc else np.nan for k in keys])
        return m
    ma, mb = per_dialogue(rows_a, fa), per_dialogue(rows_b, fb)
    ha, hb = ~np.isnan(ma), ~np.isnan(mb)
    point = float(np.mean(ma[ha]) - np.mean(mb[hb])) if ha.any() and hb.any() else None
    da, db = W[:, ha].sum(1), W[:, hb].sum(1)
    ok = (da > 0) & (db > 0)
    diffs = (W[:, ha] @ ma[ha])[ok] / da[ok] - (W[:, hb] @ mb[hb])[ok] / db[ok]
    return {"point": point, "valid_draws": int(ok.sum()), "lower80": float(np.quantile(diffs, 0.20)) if ok.any() else None,
            "ci90": [float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))] if ok.any() else None}


# ---- frozen E0 decision ------------------------------------------------------------------------------

def hypotheses(G: dict, keys, W, cfg: dict) -> dict:
    """H_E1..H_E4 predicates exactly as frozen. G: group -> rows (with E0 fields)."""
    fp, tp, fn = G["ZH_FP"], G["EN_TP"], G["EN_FN"]
    out = {}
    # H_E1
    fp_static = [r for r in fp if r["E_short"] <= 0.10 and r["E"] - r["E_short"] >= 0.25]
    tp_short = [r for r in tp if r["E_short"] >= 0.20]
    static = len(fp_static) >= 4 and len(tp_short) >= 34 and span(fp_static) >= 3 and span(tp_short) >= 3
    fn_oracle = [r for r in fn if r.get("E_oracle") is not None and r["E_oracle"] >= 0.20 and r["E_oracle"] - r["E"] >= 0.20]
    oracle_fn = len(fn_oracle) >= 9 and span(fn_oracle) >= 3
    fn_short = [r for r in fn if r["E_short"] >= 0.20 and r["E"] == 0]
    short_only = len(fn_short) >= 9 and span(fn_short) >= 3
    out["H_E1"] = {"static_scale_pattern": static, "oracle_FN_pattern": oracle_fn, "short_only_FN_pattern": short_only,
                   "counts": {"fp_static": len(fp_static), "fp_static_dialogues": span(fp_static), "tp_short": len(tp_short),
                              "tp_short_dialogues": span(tp_short), "fn_oracle": len(fn_oracle),
                              "fn_oracle_dialogues": span(fn_oracle), "fn_short": len(fn_short),
                              "fn_short_dialogues": span(fn_short)},
                   "pass": static or oracle_fn or short_only}
    # H_E2
    pers = lambda r: max(r["E_lag1"], r["E_lag2"])
    fp_iso = [r for r in fp if r["E"] >= 0.50 and pers(r) <= 0.10]
    tp_per = [r for r in tp if pers(r) >= 0.25]
    d2 = paired_rate_diff(tp, fp, lambda r: pers(r) >= 0.25, lambda r: pers(r) >= 0.25, keys, W)
    h2 = (len(fp_iso) >= 4 and span(fp_iso) >= 3 and len(tp_per) >= 26 and span(tp_per) >= 3
          and d2["valid_draws"] >= 9900 and d2["lower80"] is not None and d2["lower80"] > 0)
    out["H_E2"] = {"pass": h2, "counts": {"fp_isolated": len(fp_iso), "fp_isolated_dialogues": span(fp_iso),
                                          "tp_persistent": len(tp_per), "tp_persistent_dialogues": span(tp_per)},
                   "bootstrap_TP_minus_FP_persistent": d2}
    # H_E3
    c = cfg["E0"]["hypotheses"]["H_E3"]
    q_low = c["q_low"]
    pc = c["pass_criteria"]
    low = lambda r: r["Q"] <= q_low
    fp_low = [r for r in fp if low(r)]
    tp_low = [r for r in tp if low(r)]
    d3 = paired_rate_diff(fp, tp, low, low, keys, W)
    h3 = (len(fp_low) >= pc["ZH_FP_low_Q_min_count"] and len(tp_low) <= pc["EN_TP_low_Q_max_count"]
          and d3["point"] is not None and d3["point"] >= pc["dialogue_equal_prevalence_difference_min"]
          and span(fp_low) >= pc["low_Q_ZH_FP_min_distinct_dialogues"]
          and d3["valid_draws"] >= pc["minimum_valid_draws"] and d3["lower80"] is not None
          and d3["lower80"] > pc["one_sided_dialogue_bootstrap_lower_bound_min"])
    out["H_E3"] = {"pass": h3, "q_low": q_low, "counts": {"fp_lowQ": len(fp_low), "fp_lowQ_dialogues": span(fp_low),
                                                          "tp_lowQ": len(tp_low)},
                   "bootstrap_FP_minus_TP_lowQ": d3}
    # H_E4
    fp_null = [r for r in fp if r["ell_local"] <= 0 and r["A"] > 0 and r["null_share"] is not None and r["null_share"] >= 0.80]
    med_tp = float(np.median([r["ell_local"] for r in tp])) if tp else None
    h4 = len(fp_null) >= 4 and span(fp_null) >= 3 and med_tp is not None and med_tp > 0
    out["H_E4"] = {"pass": h4, "counts": {"fp_null": len(fp_null), "fp_null_dialogues": span(fp_null)},
                   "median_ell_local_EN_TP": med_tp}
    return out


def decide_e0(valid: bool, H: dict) -> dict:
    """Frozen precedence: INVALID; H_E1 (localizer stop | R2); R1; R3; R4; AMBIGUOUS. Exactly one outcome."""
    if not valid:
        return {"label": "P2_SEL_E_INVALID", "selected": None}
    if H["H_E1"]["pass"]:
        if H["H_E1"]["oracle_FN_pattern"] or H["H_E1"]["short_only_FN_pattern"]:
            return {"label": "P2_SEL_E_LOCALIZER_PRIMARY", "selected": None}
        return {"label": "P2_SEL_E_REPAIR_SELECTED", "selected": "R2"}
    for h, br in (("H_E2", "R1"), ("H_E3", "R3"), ("H_E4", "R4")):
        if H[h]["pass"]:
            return {"label": "P2_SEL_E_REPAIR_SELECTED", "selected": br}
    return {"label": "P2_SEL_E_DIAGNOSIS_AMBIGUOUS", "selected": None}


def quant(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    return None if a.size == 0 else {"n": int(a.size), "median": float(np.median(a)), "q25": float(np.quantile(a, .25)),
                                     "q75": float(np.quantile(a, .75))}


# ---- E0 ------------------------------------------------------------------------------------------------

def e0_rows(run_dir: Path) -> tuple[list[dict], dict, dict]:
    m = json.loads((run_dir / "manifest.json").read_text())
    if m["stage"] != "e0" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid E0 manifest")
    cfg = json.loads((ROOT / m["config"]).read_text())
    pos = json.loads((ROOT / cfg["E0"]["population"]).read_text())
    con = json.loads((ROOT / m["construction"]).read_text())
    null = json.loads((run_dir / "null.json").read_text())
    pop = json.loads((ROOT / "results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(pop["utterances"])}
    sel = {(r["utterance_id"], r["t"]): r for r in
           json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    rows, integ = [], []
    cache = {}
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        if uid not in cache:
            row = json.loads((run_dir / f"rows/{cidx[uid]:03d}.json").read_text())
            if row.get("status") != "ok" or row["identity"] != uid or row["manifest_hash"] != m["manifest_hash"]:
                raise ValueError(f"E0 row not ok: {uid}")
            steps = json.loads((ROOT / f"results/inference_cf/p2r/run3/rows/{pidx[uid]:03d}.json").read_text())["current"]["steps"]
            cache[uid] = (row, {s["query"]: s for s in steps})
        row, st = cache[uid]
        e = row["positions"][str(t)]
        q = p["p2r"]["query"]
        s = st[q]
        lg = run.decompose(e["long"]["pi_E"], e["long"]["pi_M"], null["pi_E"], null["pi_M"])
        sh = run.decompose(e["short"]["pi_E"], e["short"]["pi_M"], null["pi_E"], null["pi_M"])
        lag1, lag2 = run.lag_values({qq: ss["E"] for qq, ss in st.items()}, q)
        e_sel = sel[(uid, t)]["gate"]["E"]
        ac = s.get("acoustic") if p["stratum"] == "EN-confusion" else None
        r = {"utterance_id": uid, "t": t, "query": q, "dialogue_id": p["dialogue_id"], "stratum": p["stratum"],
             "pi_local_E": e["long"]["pi_E"], "pi_local_M": e["long"]["pi_M"], "pi_null_E": null["pi_E"],
             "pi_null_M": null["pi_M"], "ell_local": lg["ell_local"], "ell_null": lg["ell_null"], "A": lg["A"],
             "E": lg["E"], "Q": lg["Q"], "null_share": lg["null_share"], "E_short": sh["E"], "Q_short": sh["Q"],
             "ell_local_short": sh["ell_local"], "E_p2r": s["E"], "E_p2sel": e_sel, "R_B": s["R_B"],
             "window": e["window"], "window_p2r": s["window"], "attention_mass": e["attention_mass"],
             "max_attention_frame": e["max_attention_frame"], "center_sample": e["center_sample"],
             "short_bounds": e["short"]["bounds"], "E_lag1": lag1, "E_lag2": lag2,
             "E_oracle": None if not ac or ac.get("status") != "ok" else ac.get("E_oracle"),
             "timing_error_sec": None if not ac or ac.get("status") != "ok" else ac.get("timing_error_sec")}
        r["group"] = group_of(p["stratum"], e_sel)
        chk = {"query": e["query"] == q, "window": e["window"] == s["window"],
               "E_reproduces_p2r": s["E"] is not None and abs(lg["E"] - s["E"]) <= 1e-6,
               "E_matches_p2sel": abs(e_sel - s["E"]) <= 1e-12,
               "softmax_sums": all(abs(x["softmax_sum"] - 1) <= 1e-6 and x["n_probs"] == 100 and x["all_finite_nonneg"]
                                   for x in (e["long"], e["short"])),
               "Q_range": 0 <= lg["Q"] <= 1 + 1e-12 and 0 <= sh["Q"] <= 1 + 1e-12,
               "short_bounds": e["short"]["bounds"] == list(run.short_bounds(e["window"][2], e["window"][3], e["heard_samples"]))}
        r["checks"] = chk
        if not all(chk.values()):
            integ.append((uid, t, [k for k, v in chk.items() if not v]))
        rows.append(r)
    null_ok = abs(null["softmax_sum"] - 1) <= 1e-6 and null["n_probs"] == 100
    return rows, {"row_failures": integ, "null_ok": null_ok, "n": len(rows)}, {"manifest": m, "cfg": cfg}


def analyze_e0(run_dir: Path) -> dict:
    rows, integ, meta = e0_rows(run_dir)
    m, cfg = meta["manifest"], meta["cfg"]
    pi = m["provider_integrity"]
    I = cfg["E0"]["integrity"]
    G = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    counts = {g: len(v) for g, v in G.items()}
    rt = json.loads((run_dir / "runtime.json").read_text())
    validity = {"rows_180": integ["n"] == 180, "row_integrity": not integ["row_failures"], "null_provider": integ["null_ok"],
                "group_counts": counts == EXPECTED,
                "provider_identity": (pi["generation_config_sha256"] == I["generation_config_sha256"]
                                      and pi["mapping_sha256"] == I["canonical_mapping_sha256"] and pi["n"] == 100
                                      and pi["en_zh"] == I["en_zh_token_ids"]),
                "runtime_completed": rt.get("status") == "completed", "no_steering": rt.get("autograd_calls") == 0}
    keys, W = draw_weights([r["dialogue_id"] for r in rows], 10000, 240924)
    H = hypotheses(G, keys, W, cfg)
    decision = decide_e0(all(validity.values()), H)
    desc = {}
    fields = ("E", "E_short", "Q", "Q_short", "ell_local", "ell_null", "A", "null_share", "attention_mass",
              "E_lag1", "E_lag2", "E_oracle", "timing_error_sec", "R_B")
    for g, rs in G.items():
        per_d = {}
        for r in rs:
            per_d[r["dialogue_id"]] = per_d.get(r["dialogue_id"], 0) + 1
        desc[g] = {"n": len(rs), "dialogues": sorted(per_d), "per_dialogue": per_d,
                   **{f: quant([r[f] for r in rs]) for f in fields},
                   "rates_ci90": {name: boot_stat([(r["dialogue_id"], 1.0 if f(r) else 0.0) for r in rs], keys, W, 0.10)
                                  for name, f in (("lowQ", lambda r: r["Q"] <= cfg["E0"]["hypotheses"]["H_E3"]["q_low"]),
                                                  ("persistent", lambda r: max(r["E_lag1"], r["E_lag2"]) >= 0.25),
                                                  ("E_short_ge_0.2", lambda r: r["E_short"] >= 0.20),
                                                  ("E_short_le_0.1", lambda r: r["E_short"] <= 0.10))}}
    by_stratum = {s: {f: quant([r[f] for r in rows if r["stratum"] == s]) for f in fields} for s in STRATA}
    return {"schema": "p2_sel_e_e0_analysis_v1", "manifest_hash": m["manifest_hash"], "validity": validity,
            "integrity_failures": integ["row_failures"], "counts": counts, "hypotheses": H, "decision": decision,
            "descriptive_groups": desc, "descriptive_strata": by_stratum, "runtime": rt,
            "rows": rows}


def selection_artifact(ana: dict, analysis_path: str) -> dict:
    """Immutable pre-E1 record: diagnosis, branch, formula, constants, per-key reference-free E_new gates."""
    cfg = json.loads((ROOT / run.CONFIG).read_text())
    sel = ana["decision"]["selected"]
    if sel is None:
        raise ValueError("no repair selected")
    gates = []
    for r in ana["rows"]:
        en = run.e_new(sel, r)
        gates.append({"utterance_id": r["utterance_id"], "t": r["t"], "query": r["query"], "E_old": r["E"],
                      "E_new": en, "R_B": r["R_B"], "g_old": r["E"] * r["R_B"], "g_new": en * r["R_B"],
                      "inputs": {k: r[k] for k in run.E_NEW_INPUTS[sel]}})
    formula_key = {"R1": "R1_temporal", "R2": "R2_multiscale", "R3": "R3_confidence", "R4": "R4_null_shrink"}[sel]
    doc = {"schema": "p2_sel_e_selection_v1", "diagnosis": ana["decision"]["label"], "selected_repair": sel,
           "formula": cfg["E0"]["repair_formulas"][formula_key], "formula_inputs": list(run.E_NEW_INPUTS[sel]),
           "constants": {"eps": run.EPS, "alpha": 2, "layer": 16, "q_low": cfg["E0"]["hypotheses"]["H_E3"]["q_low"]},
           "e1_authorized_branches_in_config": cfg["E1"]["requires"][0],
           "e0_analysis": analysis_path, "e0_analysis_sha256": file_hash(ROOT / analysis_path),
           "population_sha256": cfg["E0"]["population_sha256"], "config_hash": digest(cfg),
           "source_sha256": {p: file_hash(ROOT / p) for p in ("experiments/inference_cf_p2sel_e.py",
                                                            "experiments/inference_cf_p2sel_e_analyze.py",
                                                            "src/csasr/inference_cf/core_r2.py")},
           "steering_outcome_exists": False, "gates": gates}
    doc["gates_hash"] = digest(gates)
    return doc


# ---- E1 ------------------------------------------------------------------------------------------------

def decide_e1(valid: bool, f: dict, ratios: dict, obs: dict, th: dict) -> dict:
    """Frozen E1 precedence. f: family-6 intervals (simultaneous 90%)."""
    if not valid:
        return {"label": "P2_SEL_E_INVALID"}
    safety = {"en_margin_lower": lo(f["en"]) is not None and lo(f["en"]) >= th["correct_margin_simultaneous_lower_min_nat"],
              "zh_margin_lower": lo(f["zh"]) is not None and lo(f["zh"]) >= th["correct_margin_simultaneous_lower_min_nat"],
              "en_corr_observed": obs["en"] is not None and obs["en"] <= th["correct_corruption_observed_max"],
              "zh_corr_observed": obs["zh"] is not None and obs["zh"] <= th["correct_corruption_observed_max"],
              "en_corr_upper": hi(f["corr_en"]) is not None and hi(f["corr_en"]) <= th["correct_corruption_simultaneous_upper_max"],
              "zh_corr_upper": hi(f["corr_zh"]) is not None and hi(f["corr_zh"]) <= th["correct_corruption_simultaneous_upper_max"]}
    benefit_point = ratios["benefit_retention"] >= th["benefit_retention_min"]
    rest = {"benefit_lower": lo(f["conf"]) is not None and lo(f["conf"]) > th["new_confusion_simultaneous_lower_strictly_above"],
            "harm_ratio": ratios["harm_ratio"] <= th["harm_ratio_max"],
            "paired_zh_point": f["paired_zh"]["estimate"] is not None and f["paired_zh"]["estimate"] >= th["paired_ZH_new_minus_old_min_nat"],
            "paired_zh_lower": lo(f["paired_zh"]) is not None and lo(f["paired_zh"]) > th["paired_ZH_simultaneous_lower_strictly_above"]}
    out = {"safety": safety, "benefit_point": benefit_point, "other": rest}
    if not all(safety.values()):
        out["label"] = "P2_SEL_E_REPAIR_STILL_UNSAFE"
    elif not benefit_point:
        out["label"] = "P2_SEL_E_REPAIR_TOO_CONSERVATIVE"
    elif not all(rest.values()):
        out["label"] = "P2_SEL_E_NO_MATERIAL_GAIN"
    else:
        out["label"] = "P2_SEL_E_REPAIR_SUPPORTED"
    return out


def analyze_e1(run_dir: Path) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    m = json.loads((run_dir / "manifest.json").read_text())
    if m["stage"] != "e1" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid E1 manifest")
    cfg = json.loads((ROOT / m["config"]).read_text())
    th = cfg["E1"]["thresholds"]
    pos = json.loads((ROOT / cfg["E0"]["population"]).read_text())
    con = json.loads((ROOT / m["construction"]).read_text())
    sel = json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_selection.json").read_text())
    gates = {(g["utterance_id"], g["t"]): g for g in sel["gates"]}
    model = Path(m["model"]["dir"])
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    p2sel = {(r["utterance_id"], r["t"]): r for r in
             json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    e0 = {(r["utterance_id"], r["t"]): r for r in
          json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]}
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    recs, bad = [], []
    cache = {}
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        if i not in cache:
            row = json.loads((run_dir / f"rows/{i:03d}.json").read_text())
            if row.get("status") != "ok" or row["manifest_hash"] != m["manifest_hash"] or row["identity"] != uid:
                raise ValueError("E1 row not ok")
            if file_hash(run_dir / f"rows/{i:03d}_logits.npz") != row["logits_sha256"]:
                raise ValueError("logits hash")
            srow = json.loads((ROOT / f"results/inference_cf/p2sel/s1_run1/rows/{i:03d}_a.json").read_text())
            if file_hash(ROOT / f"results/inference_cf/p2sel/s1_run1/rows/{i:03d}_a_logits.npz") != srow["logits_sha256"]:
                raise ValueError("P2-SEL logits hash")
            L = {}
            with np.load(run_dir / f"rows/{i:03d}_logits.npz") as z:
                L.update({k: unpack_bf16(z[k]) for k in z.files})
            with np.load(ROOT / f"results/inference_cf/p2sel/s1_run1/rows/{i:03d}_a_logits.npz") as z:
                L.update({k: unpack_bf16(z[k]) for k in z.files if k.endswith("_C2")})
            with np.load(ROOT / f"results/inference_cf/p2dir/exp1_run1/rows/{i:03d}_logits.npz") as z:
                L.update({k: unpack_bf16(z[k]) for k in z.files if k.endswith("_none")})
            cache = {i: (row, srow, L)}
        row, srow, L = cache[i]
        rr = row["positions"][str(t)]
        pre = f"t{t}_"
        Y, c = [int(v) for v in p["target_ids"]], int(p["competitor"])
        mn = logit_metrics(L[pre + "none"], t, sup, beg, part, Y, c)
        mo = logit_metrics(L[pre + "C2"], t, sup, beg, part, Y, c)
        mw = logit_metrics(L[pre + "Cnew"], t, sup, beg, part, Y, c)
        g = gates[(uid, t)]
        sa = srow["positions"][str(t)]
        chk = {"query": rr["query_matches_gate"], "hb": rr["hb_bitwise_p2dir"], "none": rr["none_logits_bitwise_p2dir"],
               "restore": rr["restore_bitwise"], "zero_noedit": rr["arm"]["zero_gain_bitwise_none"] in (None, True),
               "old_reuse_gate": abs(sa["gate"]["g"] - g["g_old"]) <= 1e-12 and sa["arms"]["C2"]["zero_gain_bitwise_none"] in (None, True)
               and sa["identity"]["none_logits_bitwise_p2dir"] and sa["restore_bitwise"],
               "old_reuse_p2sel_analysis": abs(p2sel[(uid, t)]["arms"]["C2"]["d_m"] - (mo["m"] - mn["m"])) <= 1e-9,
               "e_new_recomputed": abs(run.e_new(sel["selected_repair"], e0[(uid, t)]) - g["E_new"]) <= 1e-12}
        if not all(chk.values()):
            bad.append((uid, t, [k for k, v in chk.items() if not v]))
        recs.append({"utterance_id": uid, "t": t, "dialogue_id": p["dialogue_id"], "stratum": p["stratum"],
                     "group": e0[(uid, t)]["group"], "E_old": g["E_old"], "E_new": g["E_new"], "g_new": g["g_new"],
                     "dm_old": mo["m"] - mn["m"], "dm_new": mw["m"] - mn["m"], "ok_old": mo["top1_in_ref"],
                     "ok_new": mw["top1_in_ref"], "edit_new": rr["arm"]["edit_norm"],
                     "edit_old": sa["arms"]["C2"]["edit_norm"], "checks": chk})
    keys, W = draw_weights([p["dialogue_id"] for p in pos["positions"]], 10000, 240924)
    af = 0.10 / cfg["E1"]["bootstrap"]["family_size"]
    S = {s: [r for r in recs if r["stratum"] == s] for s in STRATA}
    v = lambda rs, fn: [(r["dialogue_id"], fn(r)) for r in rs]
    f = {"conf": boot_stat(v(S["EN-confusion"], lambda r: r["dm_new"]), keys, W, af),
         "en": boot_stat(v(S["EN-correct"], lambda r: r["dm_new"]), keys, W, af),
         "zh": boot_stat(v(S["ZH-correct"], lambda r: r["dm_new"]), keys, W, af),
         "paired_zh": boot_stat(v(S["ZH-correct"], lambda r: r["dm_new"] - r["dm_old"]), keys, W, af),
         "corr_en": boot_stat(v(S["EN-correct"], lambda r: 0.0 if r["ok_new"] else 1.0), keys, W, af),
         "corr_zh": boot_stat(v(S["ZH-correct"], lambda r: 0.0 if r["ok_new"] else 1.0), keys, W, af)}
    old = {"conf": boot_stat(v(S["EN-confusion"], lambda r: r["dm_old"]), keys, W, 0.10),
           "zh": boot_stat(v(S["ZH-correct"], lambda r: r["dm_old"]), keys, W, 0.10)}
    denom_ok = (old["conf"]["estimate"] is not None and math.isfinite(old["conf"]["estimate"]) and old["conf"]["estimate"] != 0
                and abs(old["conf"]["estimate"] - th["old_confusion_delta_m_nat"]) <= 5e-4
                and old["zh"]["estimate"] not in (None, 0) and abs(old["zh"]["estimate"] - th["old_ZH_delta_m_nat"]) <= 5e-4)
    ratios = {"benefit_retention": f["conf"]["estimate"] / old["conf"]["estimate"] if denom_ok else float("nan"),
              "harm_ratio": abs(f["zh"]["estimate"]) / abs(old["zh"]["estimate"]) if denom_ok else float("nan"),
              "old_conf": old["conf"]["estimate"], "old_zh": old["zh"]["estimate"]}
    obs = {"en": f["corr_en"]["estimate"], "zh": f["corr_zh"]["estimate"]}
    rt = json.loads((run_dir / "runtime.json").read_text())
    validity = {"rows_180": len(recs) == 180, "row_checks": not bad, "denominators": denom_ok,
                "valid_draws": all(x["valid_draws"] >= 9900 for x in f.values()), "runtime_completed": rt.get("status") == "completed"}
    decision = decide_e1(all(validity.values()), f, ratios, obs, th)
    desc = {}
    for s in STRATA:
        rs = S[s]
        desc[s] = {"E_old_pos_rate": float(np.mean([r["E_old"] > 0 for r in rs])), "E_new_pos_rate": float(np.mean([r["E_new"] > 0 for r in rs])),
                   "edit_sq_new": quant([r["edit_new"] ** 2 for r in rs]), "edit_sq_old": quant([r["edit_old"] ** 2 for r in rs]),
                   "mean_edit_sq_new": float(np.mean([r["edit_new"] ** 2 for r in rs])),
                   "mean_edit_sq_old": float(np.mean([r["edit_old"] ** 2 for r in rs])),
                   "dm_old": boot_stat(v(rs, lambda r: r["dm_old"]), keys, W, 0.10),
                   "dm_new": boot_stat(v(rs, lambda r: r["dm_new"]), keys, W, 0.10),
                   "noncorrect_old": sum(not r["ok_old"] for r in rs), "noncorrect_new": sum(not r["ok_new"] for r in rs)}
    hist = {g: [{k: r[k] for k in ("utterance_id", "t", "E_old", "E_new", "dm_old", "dm_new", "ok_old", "ok_new")}
                for r in recs if r["group"] == g] for g in ("ZH_FP", "EN_FN")}
    return {"schema": "p2_sel_e_e1_analysis_v1", "manifest_hash": m["manifest_hash"], "selected_repair": sel["selected_repair"],
            "validity": validity, "row_failures": bad, "family": f, "family_alpha": af, "old": old, "ratios": ratios,
            "observed_corruption": obs, "decision": decision, "descriptive": desc, "historical_subsets": hist,
            "runtime": rt, "per_position": recs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("e0", "e1", "select"))
    ap.add_argument("--run")
    ap.add_argument("--analysis")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("output exists; never overwrite")
    if args.stage == "e0":
        res = jsonable(analyze_e0(ROOT / args.run))
        atomic_json(out, res)
        print(json.dumps({"decision": res["decision"], "validity": res["validity"], "counts": res["counts"],
                          "hypotheses": {k: {kk: vv for kk, vv in v.items()} for k, v in res["hypotheses"].items()}}, indent=1))
    elif args.stage == "select":
        ana = json.loads((ROOT / args.analysis).read_text())
        doc = selection_artifact(ana, args.analysis)
        atomic_json(out, jsonable(doc))
        print(json.dumps({k: doc[k] for k in ("diagnosis", "selected_repair", "formula", "gates_hash")}, indent=1))
    else:
        res = jsonable(analyze_e1(ROOT / args.run))
        atomic_json(out, res)
        print(json.dumps({"decision": res["decision"], "validity": res["validity"], "ratios": res["ratios"],
                          "family": {k: [x["estimate"], x["ci"]] for k, x in res["family"].items()},
                          "observed": res["observed_corruption"]}, indent=1))


if __name__ == "__main__":
    main()
