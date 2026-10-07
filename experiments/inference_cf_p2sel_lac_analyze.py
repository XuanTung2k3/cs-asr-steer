#!/usr/bin/env python
"""P2-SEL-LAC analysis (CPU). Frozen contract be460ce: LAC0 lexical counterfactual scalars from sealed
unmasked and masked raw logits, descriptive candidate probabilities/ranks, historical-group statistics,
frozen precedence and F_lex selection artifact; frozen LAC1 label rule. Groups are analysis-only.
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
from csasr.inference_cf.lexical_compatibility import lexical, select_candidates
from experiments.inference_cf_p2dir_analyze import boot_stat, draw_weights, hi, jsonable, lo, unpack_bf16
from experiments.inference_cf_p2sel_t_analyze import group_diff, quant, span
import experiments.inference_cf_p2sel_lac as run

GROUPS = ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")
EXPECTED = {"EN_TP": 42, "EN_FN": 18, "ZH_TN": 55, "ZH_FP": 5, "EN_CORRECT": 60}
S_STRONG = math.log(17 / 3)
S_WEAK = math.log(11 / 9)


def prob_rank(z, tok: int, t: int, sup, beg) -> tuple[float, int, int]:
    """Descriptive generate-suppressed probability / rank (ties: lower ID first) and top1 (lowest ID)."""
    x = np.asarray(z, dtype=np.float64).copy()
    if sup:
        x[sup] = -np.inf
    if t == 0 and beg:
        x[beg] = -np.inf
    lz = np.max(x) + math.log(np.sum(np.exp(x - np.max(x))))
    v = float(np.asarray(z, dtype=np.float64)[tok])                 # sealed candidates are never suppressed
    rank = 1 + int(np.sum(x > v)) + int(np.sum(x[:tok] == v))
    return math.exp(v - lz), rank, int(np.argmax(x))


def decide_lac0(valid: bool, G: dict, keys, W) -> dict:
    fp, tp, fn = G["ZH_FP"], G["EN_TP"], G["EN_FN"]
    tps = [r for r in tp if r["S_lex"] >= S_STRONG]
    fps = [r for r in fp if r["S_lex"] <= S_WEAK]
    fns = [r for r in fn if r["S_lex"] >= S_STRONG]
    d = group_diff(tp, fp, lambda r: r["S_lex"], lambda r: r["S_lex"], keys, W)
    p = {"tp_strong": len(tps) >= 34 and span(tps) >= 3, "fp_weak": len(fps) >= 4 and span(fps) >= 3,
         "contrast": d["point"] is not None and d["point"] >= 0.50 and d["lower80"] is not None and d["lower80"] > 0,
         "fn_strong": len(fns) >= 9 and span(fns) >= 3}
    counts = {"tp_strong": len(tps), "tp_strong_dialogues": span(tps), "fp_weak": len(fps), "fp_weak_dialogues": span(fps),
              "fn_strong": len(fns), "fn_strong_dialogues": span(fns)}
    if not valid or d["valid_draws"] < 9900:
        lab = "P2_SEL_LAC_INVALID"
    elif p["tp_strong"] and p["fp_weak"] and p["contrast"]:
        lab = "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    elif p["fn_strong"]:
        lab = "P2_SEL_LAC_RECALL_ONLY"
    else:
        lab = "P2_SEL_LAC_NOT_DISCRIMINATIVE"
    return {"label": lab, "predicates": p, "counts": counts, "S_TP_minus_FP": d}


def analyze_lac0(run_dir: Path) -> dict:
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    m = json.loads((run_dir / "manifest.json").read_text())
    if m["stage"] != "lac0" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid LAC0 manifest")
    seal = json.loads((ROOT / run.SEAL).read_text())
    if seal["seal_hash"] != m["seal_hash"] or digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"]:
        raise ValueError("seal")
    con = json.loads((ROOT / m["construction"]).read_text())
    part = tokenizer_partition(WhisperProcessor.from_pretrained(Path(m["model"]["dir"]), local_files_only=True).tokenizer)
    tok = WhisperProcessor.from_pretrained(Path(m["model"]["dir"]), local_files_only=True).tokenizer
    sup, beg = m["suppression"]["suppress"], m["suppression"]["begin"]
    e0 = {(r["utterance_id"], r["t"]): r for r in json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]}
    t0 = {(r["utterance_id"], r["t"]): r for r in json.loads((ROOT / "results/inference_cf/p2sel_t/t0_run1_analysis.json").read_text())["rows"]}
    sr = {(r["utterance_id"], r["t"]): r for r in seal["rows"]}
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    rows, fails, cache = [], [], {}
    for p in con["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        if i not in cache:
            r = json.loads((run_dir / f"rows/{i:03d}.json").read_text())
            if r.get("status") != "ok" or r["identity"] != uid or r["manifest_hash"] != m["manifest_hash"]:
                raise ValueError(f"LAC0 row not ok {uid}")
            if file_hash(run_dir / f"rows/{i:03d}_logits.npz") != r["logits_sha256"]:
                raise ValueError("masked logits hash")
            with np.load(run_dir / f"rows/{i:03d}_logits.npz") as z:
                ML = {k: unpack_bf16(z[k]) for k in z.files}
            with np.load(ROOT / run.P2DIR_RUN / f"rows/{i:03d}_logits.npz") as z:
                UL = {k: unpack_bf16(z[k]) for k in z.files if k.endswith("_none")}
            cache = {i: (r, ML, UL)}
        r, ML, UL = cache[i]
        x = r["positions"][str(t)]
        s = sr[(uid, t)]
        z = UL[f"t{t}_none"]
        zm = z if x.get("noop") else ML[f"t{t}_masked"]
        cE, cM = select_candidates(z, part["embedded_ids"], part["matrix_ids"])
        lx = lexical(z, zm, s["c_E"], s["c_M"])
        tp = t0[(uid, t)]
        chk = {"candidates_reproduce_seal": (cE, cM) == (s["c_E"], s["c_M"]),
               "unmasked_vector_hash": run.arr_sha(z.astype(np.float32)) == s["unmasked_logit_vector_sha256"],
               "W_star_equals_T": x["W_star"] == s["W_star"] == tp["bounds"][tp["j"]] and s["j"] == tp["j"],
               "W_inside_W1s_heard": s["W_1s"][2] <= x["W_star"][0] < x["W_star"][1] <= s["W_1s"][3] <= x["heard_samples"],
               "length": x["W_star"][1] - x["W_star"][0] == min(8000, s["W_1s"][3] - s["W_1s"][2]),
               "outside_bytes": x["outside_left_equal"] and x["outside_right_equal"], "inside_zero": x["inside_positive_zero"],
               "change_if_energy": (x["local_energy"] == 0) or x["changed"], "shape_finite": x["shape_dtype_ok"] and x["finite"],
               "candidate_not_suppressed": s["c_E"] not in sup and s["c_M"] not in sup and s["c_E"] not in beg and s["c_M"] not in beg}
        if not x.get("noop"):
            chk.update(fed=x["fed_equals_input_ids"], query=x["query_matches"], logits_finite=x["logits_finite"] and bool(np.isfinite(zm).all()),
                       vocab=x["vocab"] == len(z))
        if not all(chk.values()):
            fails.append((uid, t, [k for k, v in chk.items() if not v]))
        pE0, rE0, top0 = prob_rank(z, s["c_E"], t, sup, beg)
        pM0, rM0, _ = prob_rank(z, s["c_M"], t, sup, beg)
        pE1, rE1, top1 = prob_rank(zm, s["c_E"], t, sup, beg)
        pM1, rM1, _ = prob_rank(zm, s["c_M"], t, sup, beg)
        o = e0[(uid, t)]
        rows.append({"utterance_id": uid, "t": t, "query": s["query"], "dialogue_id": p["dialogue_id"], "stratum": p["stratum"],
                     "group": o["group"], "c_E": s["c_E"], "c_M": s["c_M"], "text_E": tok.convert_ids_to_tokens(s["c_E"]),
                     "text_M": tok.convert_ids_to_tokens(s["c_M"]), **lx, "noop": bool(x.get("noop")),
                     "removed_energy_fraction": x["removed_energy_fraction"], "branch_origin": x.get("branch_origin"),
                     "p_E_unmasked": pE0, "rank_E_unmasked": rE0, "p_M_unmasked": pM0, "rank_M_unmasked": rM0,
                     "p_E_masked": pE1, "rank_E_masked": rE1, "p_M_masked": pM1, "rank_M_masked": rM1,
                     "top1_unmasked": top0, "top1_masked": top1, "E_1s": o["E"], "R_B": o["R_B"], "E_tok": tp["E_tok"], "checks": chk})
    G = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    rt = json.loads((run_dir / "runtime.json").read_text())
    c = rt["counters"]
    validity = {"rows_180": len(rows) == 180, "row_checks": not fails, "groups": {g: len(v) for g, v in G.items()} == EXPECTED,
                "runtime_completed": rt.get("status") == "completed",
                "forward_only": c["autograd_calls"] == 0 and c["readout_calls"] == 0 and c["lid_calls"] == 0 and c["steering_calls"] == 0,
                "eval_cap": c["counterfactual_evaluations"] <= 180,
                "suppression_disjoint": not (set(m["suppression"]["suppress"]) | set(m["suppression"]["begin"])) & (set(part["embedded_ids"]) | set(part["matrix_ids"]))}
    keys, W = draw_weights([r["dialogue_id"] for r in rows], 10000, 240924)
    dec = decide_lac0(all(validity.values()), G, keys, W)
    fields = ("S_lex", "M0", "Mmask", "Delta_E", "Delta_M", "F_lex", "removed_energy_fraction", "p_E_unmasked", "p_M_unmasked",
              "p_E_masked", "p_M_masked", "E_1s", "E_tok", "R_B")
    desc = {g: {"n": len(rs), "dialogues": sorted({r["dialogue_id"] for r in rs}), **{f: quant([r[f] for r in rs]) for f in fields},
                "dialogue_mean_ci90": {f: boot_stat([(r["dialogue_id"], r[f]) for r in rs], keys, W, 0.10)
                                       for f in ("S_lex", "Delta_E", "Delta_M", "F_lex")},
                "noop": sum(r["noop"] for r in rs)} for g, rs in G.items()}
    return {"schema": "p2_sel_lac_lac0_analysis_v1", "manifest_hash": m["manifest_hash"], "validity": validity, "row_failures": fails,
            "decision": dec, "label": dec["label"], "descriptive": desc, "runtime": rt,
            "thresholds": {"S_strong": S_STRONG, "S_weak": S_WEAK}, "counts": {g: len(v) for g, v in G.items()}, "rows": rows}


def decide_lac1(valid: bool, f: dict, ratios: dict, obs: dict, en_reg: dict, th: dict) -> str:
    """Frozen LAC1 precedence (same structure and budgets as XA1/T1)."""
    if not valid:
        return "P2_SEL_LAC_INVALID"
    safety = (lo(f["en"]) >= -0.25 and lo(f["zh"]) >= -0.25 and obs["en"] <= 0.05 and obs["zh"] <= 0.05
              and hi(f["corr_en"]) <= 0.10 and hi(f["corr_zh"]) <= 0.10 and en_reg["mean_new_minus_old"] >= -1e-6
              and en_reg["new_corruptions"] == 0)
    if not safety:
        return "P2_SEL_LAC_GATE_STILL_UNSAFE"
    if ratios["benefit_retention"] < 0.70:
        return "P2_SEL_LAC_GATE_TOO_CONSERVATIVE"
    if not (lo(f["conf"]) > 0 and ratios["harm_ratio"] <= 0.50 and f["paired_zh"]["estimate"] >= 0.10 and lo(f["paired_zh"]) > 0):
        return "P2_SEL_LAC_NO_MATERIAL_GAIN"
    return "P2_SEL_LAC_GATE_SUPPORTED"


def selection_artifact(ana: dict, analysis_path: str, audit_path: str) -> dict:
    if ana["label"] != "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED":
        raise ValueError("no repair selected")
    cfg = json.loads((ROOT / run.CONFIG).read_text())
    gates = [{"utterance_id": r["utterance_id"], "t": r["t"], "query": r["query"], "E_old": r["E_1s"], "R_B": r["R_B"],
              "F_lex": r["F_lex"], "g_old": r["E_1s"] * r["R_B"], "g_new": r["E_1s"] * r["R_B"] * r["F_lex"]} for r in ana["rows"]]
    doc = {"schema": "p2_sel_lac_selection_v1", "selected": "F_lex", "formula": cfg["LAC0"]["only_factor"], "gate": "E_1s*R_B*F_lex*D2",
           "lac0_label": ana["label"], "lac0_analysis_sha256": file_hash(ROOT / analysis_path), "lac0_audit_sha256": file_hash(ROOT / audit_path),
           "seal_sha256": file_hash(ROOT / run.SEAL), "config_hash": digest(cfg), "steering_outcome_exists": False, "gates": gates}
    doc["gates_hash"] = digest(gates)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lac0", "select"))
    ap.add_argument("--run")
    ap.add_argument("--analysis")
    ap.add_argument("--audit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("output exists; never overwrite")
    if args.stage == "lac0":
        res = jsonable(analyze_lac0(ROOT / args.run))
        atomic_json(out, res)
        print(json.dumps({"label": res["label"], "validity": res["validity"], "decision": res["decision"], "counts": res["counts"]}, indent=1))
    else:
        doc = selection_artifact(json.loads((ROOT / args.analysis).read_text()), args.analysis, args.audit)
        atomic_json(out, jsonable(doc))
        print(json.dumps({k: doc[k] for k in ("selected", "formula", "gates_hash")}, indent=1))


if __name__ == "__main__":
    main()
