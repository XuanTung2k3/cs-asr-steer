#!/usr/bin/env python
"""S1 evaluator (post-seal only; spec sections 7-10). Isolated from the runner: never imported by experiments/acoustic_s1.py.

Preconditions (checked before any reference file is opened): the immutable output seal is committed AND contained in
origin/cs-asr-steer-inf, every sealed file is unchanged, and S1_AUDIT: PASS (PRIMARY) is committed. Only then the exposed
P2-RJ acceptable first-token sets / strata (results/inference_cf/p2rj/positions.json) and the historical P2-R structural
ledger are read. Hits, ranks, accessibility and paired-accessible membership are evaluator conditioning only; nothing is
returned to runtime. Gate H, Gate D (point / coverage / protection + family-8 adjusted MRR inference) and the exact
terminal precedence follow configs/inference_cf/s1_acoustic_evidence.json without any post-hoc choice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

CONFIG = "configs/inference_cf/s1_acoustic_evidence.json"
BASE = "results/inference_cf/s1"
RUN = f"{BASE}/run1"
PLAN = f"{BASE}/plan_sealed.json"
OUTPUT_SEAL = f"{BASE}/output_seal.json"
PRIMARY = f"{BASE}/primary_analysis.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
ANALYSIS = f"{BASE}/analysis.json"
POSITIONS = "results/inference_cf/p2rj/positions.json"
P2R_POPULATION = "results/inference_cf/p2r/population.json"
LAC0_ANALYSIS = "results/inference_cf/p2sel_lac/lac0_run1_analysis.json"
LAC0_FINAL_AUDIT = "results/inference_cf/p2sel_lac/final_audit.json"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
EOS = 50257
SCORERS = ("M_ORIGINAL", "E_ORIGINAL", "AUTO_ORIGINAL", "EN_REGION", "OFF_TARGET", "LAC_UNION", "SHUFFLED_SUPPORT", "NO_CONTRAST")
CONTROLS = ("M_ORIGINAL", "OFF_TARGET", "LAC_UNION", "SHUFFLED_SUPPORT")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def committed(rel: str) -> bool:
    import hashlib
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def on_remote(rel: str) -> bool:
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, "origin/cs-asr-steer-inf"], cwd=ROOT).returncode == 0


def draw_weights(dialogues, reps: int, seed: int):
    """20 fixed dialogue IDs (sorted) sampled with replacement: W[r, d] = multiplicity of dialogue d in draw r."""
    keys = sorted(set(dialogues))
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(reps, len(keys)))
    W = np.zeros((reps, len(keys)), dtype=np.float64)
    for d in range(len(keys)):
        W[:, d] = (idx == d).sum(axis=1)
    return keys, W


def macro(values: list[tuple[str, float]]) -> tuple[float | None, dict]:
    by = defaultdict(list)
    for d, v in values:
        by[d].append(float(v))
    dm = {d: float(np.mean(v)) for d, v in by.items()}
    return (float(np.mean(list(dm.values()))) if dm else None), dm


def boot(values: list[tuple[str, float]], keys, W, q_lo: float, q_hi: float) -> dict:
    """Per draw: average per-row paired effects within each represented eligible dialogue, then average those dialogue
    means weighted by sampled multiplicity; draws with no eligible dialogue are undefined (never imputed)."""
    est, dm = macro(values)
    m = np.array([dm.get(k, np.nan) for k in keys])
    has = ~np.isnan(m)
    num = (W[:, has] * m[has]).sum(axis=1)
    den = W[:, has].sum(axis=1)
    ok = den > 0
    stats = num[ok] / den[ok]
    out = {"estimate": est, "rows": len(values), "dialogues": int(has.sum()), "finite_draws": int(ok.sum()), "lower": None, "upper": None}
    if stats.size:
        out["lower"], out["upper"] = float(np.quantile(stats, q_lo)), float(np.quantile(stats, q_hi))
    return out


def decide(integrity_ok: bool, h_pass: bool, d_point_pass: bool, d_inference_pass: bool) -> str:
    """Exact frozen precedence (spec section 10). BLOCKED_CANDIDATE_SEMANTICS is a pre-run stop and never post-run."""
    if not integrity_ok:
        return "S1_INVALID"
    if not h_pass:
        return "S1_CANDIDATE_HEADROOM_INSUFFICIENT"
    if not d_point_pass:
        return "S1_ACOUSTIC_DISCRIMINATION_INSUFFICIENT"
    if not d_inference_pass:
        return "S1_PARTIAL_FEASIBILITY"
    return "S1_READY_FOR_S2"


def load_sealed():
    seal = json.loads((ROOT / OUTPUT_SEAL).read_text())
    if not (committed(OUTPUT_SEAL) and on_remote(OUTPUT_SEAL)):
        raise SystemExit("output seal not committed and pushed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != "S1_AUDIT: PASS (PRIMARY)":
        raise SystemExit("PRIMARY audit PASS required before any reference access")
    if digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"]:
        raise SystemExit("output seal hash")
    for p, h in seal["files"].items():
        if file_hash(ROOT / p) != h:
            raise SystemExit(f"sealed file changed: {p}")
    return seal


def evaluate() -> dict:
    seal = load_sealed()
    cfg = json.loads((ROOT / CONFIG).read_text())
    plan = json.loads((ROOT / PLAN).read_text())
    prim = json.loads((ROOT / PRIMARY).read_text())
    run = ROOT / RUN
    cand = [json.loads((run / "candidates" / f"{i:03d}.json").read_text()) for i in range(80)]
    aco = [json.loads((run / "acoustic" / f"{i:03d}.json").read_text()) for i in range(80)]
    cq = {q["j"]: (d, q) for d in cand for q in d["queries"]}
    aq = {q["j"]: q for d in aco for q in d["queries"]}
    # ---- references opened only now ----
    pos = json.loads((ROOT / POSITIONS).read_text())["positions"]
    Q = plan["runtime"]["runtime_queries"]
    if [(p["utterance_id"], int(p["t"]), p["dialogue_id"]) for p in pos] != [(q["utterance_id"], q["t"], q["dialogue_id"]) for q in Q]:
        raise SystemExit("positions order differs from the frozen runtime order")
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    latin = set(tokenizer_partition(WhisperProcessor.from_pretrained(cfg["model"]["dir"], local_files_only=True).tokenizer)["embedded_ids"])
    Y = [set(int(x) for x in p["target_ids"] if int(x) < EOS) for p in pos]          # EOS / control never a lexical hit
    S = [p["stratum"] for p in pos]
    D = [p["dialogue_id"] for p in pos]
    integrity = {"seal_files_verified": True, "primary_audit_pass": True,
                 "candidate_outputs_complete_180": prim["completeness"]["candidate_queries"] == 180 and prim["completeness"]["all_candidate_outputs_valid"]
                 and prim["completeness"]["M_bitwise_hist_none_180"] == 180 and prim["completeness"]["same_prefix_180"] == 180,
                 "acoustic_queries_180": len(aq) == 180,
                 "lac_exact_reuse_180": all(aq[j]["lac"]["exact_reuse_ok"] for j in range(180)),
                 "finite_scores": all(aq[j]["K"][k]["finite"] for j in range(180) for k in ("5", "20")),
                 "targets_disjoint_from_eos": all(EOS not in y for y in Y), "strata_60": all(S.count(s) == 60 for s in STRATA)}
    rows = []
    for j in range(180):
        d, c = cq[j]
        a = aq[j]
        r = {"j": j, "dialogue_id": D[j], "stratum": S[j], "paired": a["paired_available"], "lac_ok": a["lac"]["exact_reuse_ok"],
             "target_status": a["target_status"], "off_target_status": a["off_target_status"], "K": {}}
        for k in ("5", "20"):
            un = c["unions"][k]
            ids = un["ids"]
            top = {b: c["branches"][b]["top"][k] for b in ("M", "E", "AUTO")}
            rk = a["K"][k]["rankings"]
            hit = {s: rk[s][0] in Y[j] for s in SCORERS}
            rank = {}
            for s in SCORERS:
                pos_ = [i + 1 for i, tkn in enumerate(rk[s]) if tkn in Y[j]]
                rank[s] = min(pos_) if pos_ else None
            r["K"][k] = {"union_size": len(ids), "access": {b: any(t in Y[j] for t in top[b]) for b in top} | {"UNION": any(t in Y[j] for t in ids)},
                         "hit": hit, "rank": rank, "rr": {s: (1.0 / rank[s] if rank[s] else 0.0) for s in SCORERS},
                         "top1": {s: rk[s][0] for s in SCORERS},
                         "latin_in_union": sum(t in latin for t in ids),
                         "diag_target_only_hit": (a["K"][k]["diagnostic_rankings"].get("EN_REGION_TARGET_ONLY") or [None])[0] in Y[j]}
            r["K"][k]["P"] = (S[j] == "EN-confusion" and r["K"][k]["access"]["UNION"] and r["paired"] and r["lac_ok"])
        # reference first-token rank in each branch's full processed distribution (descending, lower-ID ties)
        with np.load(run / "candidates" / f"{d['index']:03d}.npz") as z:
            fr = {}
            for b in ("M", "E", "AUTO"):
                key = f"q{j:03d}_{d['auto_alias'] or 'AUTO'}_logp" if b == "AUTO" else f"q{j:03d}_{b}_logp"
                lp = z[key]
                best = None
                for y in Y[j]:
                    if np.isfinite(lp[y]):
                        rr = int(np.sum(lp > lp[y]) + np.sum((lp == lp[y]) & (np.arange(lp.size) < y))) + 1
                        best = rr if best is None else min(best, rr)
                fr[b] = best
        r["full_rank"] = fr
        rows.append(r)
    conf = [r for r in rows if r["stratum"] == "EN-confusion"]
    keys, W = draw_weights(D, cfg["bootstrap"]["draws"], cfg["bootstrap"]["seed"])
    qlo, qhi = cfg["bootstrap"]["lower_quantile"], cfg["bootstrap"]["upper_quantile"]

    def dl(rs):
        return len({r["dialogue_id"] for r in rs})
    # ---- S1-A headroom ----
    A = {}
    for k in ("5", "20"):
        acc = {b: [r for r in conf if r["K"][k]["access"][b]] for b in ("M", "E", "AUTO", "UNION")}
        incE = [r for r in conf if r["K"][k]["access"]["E"] and not r["K"][k]["access"]["M"]]
        incA = [r for r in conf if r["K"][k]["access"]["AUTO"] and not (r["K"][k]["access"]["M"] or r["K"][k]["access"]["E"])]
        incU = [r for r in conf if r["K"][k]["access"]["UNION"] and not r["K"][k]["access"]["M"]]
        A[k] = {"recall_pooled_of_60": {b: len(v) for b, v in acc.items()}, "recall_dialogues": {b: dl(v) for b, v in acc.items()},
                "recall_dialogue_macro": {b: macro([(r["dialogue_id"], float(r["K"][k]["access"][b])) for r in conf])[0] for b in acc},
                "recall_pointwise95": {b: boot([(r["dialogue_id"], float(r["K"][k]["access"][b])) for r in conf], keys, W, .025, .975) for b in acc},
                "incremental_E_beyond_M": len(incE), "incremental_E_dialogues": dl(incE),
                "incremental_AUTO_beyond_M_E": len(incA), "incremental_AUTO_dialogues": dl(incA),
                "union_minus_M": len(incU), "union_minus_M_dialogues": dl(incU),
                "union_size_confusion": {"mean": float(np.mean([r["K"][k]["union_size"] for r in conf])),
                                         "max": int(max(r["K"][k]["union_size"] for r in conf))},
                "accessible_by_dialogue": {dd: sum(1 for r in acc["UNION"] if r["dialogue_id"] == dd) for dd in keys}}
        A[k]["correct_state"] = {s: {"union_access": sum(r["K"][k]["access"]["UNION"] for r in rows if r["stratum"] == s),
                                     "rows_with_latin_in_union": sum(r["K"][k]["latin_in_union"] > 0 for r in rows if r["stratum"] == s),
                                     "mean_latin_in_union": float(np.mean([r["K"][k]["latin_in_union"] for r in rows if r["stratum"] == s]))}
                                 for s in ("EN-correct", "ZH-correct")}
    A["full_reference_rank_confusion"] = {b: {"median": float(np.median([r["full_rank"][b] for r in conf if r["full_rank"][b]])),
                                              "le5": sum(1 for r in conf if r["full_rank"][b] and r["full_rank"][b] <= 5),
                                              "le20": sum(1 for r in conf if r["full_rank"][b] and r["full_rank"][b] <= 20),
                                              "le100": sum(1 for r in conf if r["full_rank"][b] and r["full_rank"][b] <= 100),
                                              "no_legal_target": sum(1 for r in conf if r["full_rank"][b] is None)} for b in ("M", "E", "AUTO")}
    H = cfg["gate_H"]
    hU, hI = A["20"]["recall_pooled_of_60"]["UNION"], A["20"]["union_minus_M"]
    gH = {"complete_valid_original_180": integrity["candidate_outputs_complete_180"],
          "union20_hits": hU >= H["union20_reference_hits_min"], "union20_hit_dialogues": A["20"]["recall_dialogues"]["UNION"] >= H["union20_hit_dialogues_min"],
          "incremental_vs_M20": hI >= H["incremental_vs_M20_min"], "incremental_dialogues": A["20"]["union_minus_M_dialogues"] >= H["incremental_dialogues_min"]}
    gH["PASS"] = all(gH.values())
    # ---- S1-B discrimination ----
    B = {}
    for k in ("5", "20"):
        P = [r for r in conf if r["K"][k]["P"]]
        acc_u = [r for r in conf if r["K"][k]["access"]["UNION"]]
        ent = {"P_rows": len(P), "P_dialogues": dl(P), "union_accessible": len(acc_u),
               "P_fraction_of_accessible": (len(P) / len(acc_u)) if acc_u else None,
               "accessible_unpaired_reasons": dict(sorted(defaultdict(int, {}).items())),
               "per_scorer": {}, "vs_controls": {}}
        reasons = defaultdict(int)
        for r in acc_u:
            if not r["K"][k]["P"]:
                reasons[f"{r['target_status']}|{r['off_target_status']}|lac_ok={r['lac_ok']}"] += 1
        ent["accessible_unpaired_reasons"] = dict(reasons)
        for s in SCORERS:
            ent["per_scorer"][s] = {"hits_P": sum(r["K"][k]["hit"][s] for r in P), "mrr_P_macro": macro([(r["dialogue_id"], r["K"][k]["rr"][s]) for r in P])[0],
                                    "mrr_P_pooled": float(np.mean([r["K"][k]["rr"][s] for r in P])) if P else None,
                                    "hits_unconditional_60": sum(r["K"][k]["hit"][s] for r in conf),
                                    "mrr_unconditional_60_pooled": float(np.mean([r["K"][k]["rr"][s] for r in conf])),
                                    "mean_rank_P": float(np.mean([r["K"][k]["rank"][s] for r in P])) if P else None}
        ent["target_only_diagnostic_hits_accessible"] = sum(r["K"][k]["diag_target_only_hit"] for r in acc_u)
        for c in CONTROLS:
            new = [r for r in P if r["K"][k]["hit"]["EN_REGION"] and not r["K"][k]["hit"][c]]
            lost = [r for r in P if r["K"][k]["hit"][c] and not r["K"][k]["hit"]["EN_REGION"]]
            hv = [(r["dialogue_id"], float(r["K"][k]["hit"]["EN_REGION"]) - float(r["K"][k]["hit"][c])) for r in P]
            mv = [(r["dialogue_id"], r["K"][k]["rr"]["EN_REGION"] - r["K"][k]["rr"][c]) for r in P]
            ent["vs_controls"][c] = {"new_correct": len(new), "new_incorrect": len(lost), "net_hits": len(new) - len(lost),
                                     "gross_new_dialogues": dl(new), "hit1_macro_gain": macro(hv)[0], "mrr_macro_gain": macro(mv)[0],
                                     "hit1_pointwise95": boot(hv, keys, W, .025, .975), "mrr_pointwise95": boot(mv, keys, W, .025, .975),
                                     "mrr_adjusted": boot(mv, keys, W, qlo, qhi)}
        B[k] = ent
    # correct-state protection (K20 primary rankings, fallback rows included)
    prot = {}
    for s in ("EN-correct", "ZH-correct"):
        rs = [r for r in rows if r["stratum"] == s]
        prot[s] = {"M_hits": sum(r["K"]["20"]["hit"]["M_ORIGINAL"] for r in rs), "EN_REGION_hits": sum(r["K"]["20"]["hit"]["EN_REGION"] for r in rs),
                   "new_corruptions": sum(r["K"]["20"]["hit"]["M_ORIGINAL"] and not r["K"]["20"]["hit"]["EN_REGION"] for r in rs),
                   "new_corrections": sum(r["K"]["20"]["hit"]["EN_REGION"] and not r["K"]["20"]["hit"]["M_ORIGINAL"] for r in rs),
                   "top1_changed": sum(r["K"]["20"]["top1"]["EN_REGION"] != r["K"]["20"]["top1"]["M_ORIGINAL"] for r in rs),
                   "eos_promotions": sum(r["K"]["20"]["top1"]["EN_REGION"] == EOS != r["K"]["20"]["top1"]["M_ORIGINAL"] for r in rs),
                   "per_scorer_corruptions": {sc: sum(r["K"]["20"]["hit"]["M_ORIGINAL"] and not r["K"]["20"]["hit"][sc] for r in rs) for sc in SCORERS}}
    zh = [r for r in rows if r["stratum"] == "ZH-correct"]
    prot["ZH-correct"]["new_wrong_latin_promotions"] = sum(
        r["K"]["20"]["top1"]["EN_REGION"] != r["K"]["20"]["top1"]["M_ORIGINAL"] and not r["K"]["20"]["hit"]["EN_REGION"]
        and r["K"]["20"]["top1"]["EN_REGION"] in latin for r in zh)
    prot["confusion_eos_promotions"] = sum(r["K"]["20"]["top1"]["EN_REGION"] == EOS != r["K"]["20"]["top1"]["M_ORIGINAL"] for r in conf)
    G = cfg["gate_D"]
    b20 = B["20"]
    vs = b20["vs_controls"]
    gD = {"P20_rows": b20["P_rows"] >= G["paired_accessible_rows_min"], "P20_dialogues": b20["P_dialogues"] >= G["paired_dialogues_min"],
          "P20_fraction": b20["P_fraction_of_accessible"] is not None and b20["P_fraction_of_accessible"] >= G["paired_fraction_of_accessible_min"],
          "net_hits_vs_M": vs["M_ORIGINAL"]["net_hits"] >= G["net_hit1_gain_vs_M_min"],
          "correction_dialogues": vs["M_ORIGINAL"]["gross_new_dialogues"] >= G["correction_dialogues_min"],
          "hit1_macro_vs_M": vs["M_ORIGINAL"]["hit1_macro_gain"] is not None and vs["M_ORIGINAL"]["hit1_macro_gain"] >= G["dialogue_macro_hit1_gain_vs_M_min"]}
    for c in ("OFF_TARGET", "LAC_UNION", "SHUFFLED_SUPPORT"):
        gD[f"net_hits_vs_{c}"] = vs[c]["net_hits"] >= G["net_hit1_gain_vs_each_control_min"]
    for c, th in G["mrr_macro_gain_min"].items():
        gD[f"mrr_macro_vs_{c}"] = vs[c]["mrr_macro_gain"] is not None and vs[c]["mrr_macro_gain"] >= th
    gD["EN_correct_corruptions"] = prot["EN-correct"]["new_corruptions"] <= G["new_EN_correct_corruptions_max"]
    gD["ZH_correct_corruptions"] = prot["ZH-correct"]["new_corruptions"] <= G["new_ZH_correct_corruptions_max"]
    gD["total_corruptions"] = prot["EN-correct"]["new_corruptions"] + prot["ZH-correct"]["new_corruptions"] <= G["total_correct_state_corruptions_max"]
    gD["ZH_latin_promotions"] = prot["ZH-correct"]["new_wrong_latin_promotions"] <= G["new_wrong_English_promotions_ZH_max"]
    gD["POINT_COVERAGE_PROTECTION_PASS"] = all(gD.values())
    adj = {c: vs[c]["mrr_adjusted"] for c in CONTROLS}
    gI = {f"adjusted_lower_vs_{c}": (adj[c]["finite_draws"] >= cfg["bootstrap"]["minimum_finite_draws"] and adj[c]["lower"] is not None
                                     and adj[c]["lower"] > G["adjusted_lower_mrr_vs_all_four_controls_gt"]) for c in CONTROLS}
    gI["PASS"] = all(gI.values())
    label = decide(all(integrity.values()), gH["PASS"], gD["POINT_COVERAGE_PROTECTION_PASS"], gI["PASS"])
    # ---- descriptive historical context (unchanged, never pooled) ----
    hist = {}
    pop = json.loads((ROOT / P2R_POPULATION).read_text())
    hist["p2r_population_ledger"] = pop.get("ledger")
    for name, p in (("lac0_analysis", LAC0_ANALYSIS), ("lac0_final_audit", LAC0_FINAL_AUDIT)):
        try:
            h = json.loads((ROOT / p).read_text())
            hist[name] = {k: h[k] for k in ("label", "verdict", "terminal_label", "decision") if k in h}
        except FileNotFoundError:
            hist[name] = None
    doc = {"schema": "s1_acoustic_evidence_v1_analysis", "output_seal_hash": seal["seal_hash"], "primary_audit_sha256": file_hash(ROOT / PRIMARY_AUDIT),
           "positions_sha256": file_hash(ROOT / POSITIONS), "integrity": integrity, "S1_A": A, "gate_H": gH, "S1_B": B, "protection": prot,
           "gate_D": gD, "gate_D_inference": gI, "label": label,
           "label_note": "S1_READY_FOR_S2 additionally requires S1_AUDIT: PASS (FULL)" if label == "S1_READY_FOR_S2" else None,
           "historical_context": hist,
           "rows": [{k: v for k, v in r.items()} for r in rows],
           "bootstrap": {"seed": cfg["bootstrap"]["seed"], "draws": cfg["bootstrap"]["draws"], "dialogues": keys, "lower_quantile": qlo, "upper_quantile": qhi,
                         "family": [f"K{k}:EN_REGION-{c}" for k in ("5", "20") for c in CONTROLS]},
           "created_unix": time.time()}
    doc["analysis_hash"] = digest(doc)
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("evaluate",))
    ap.parse_args()
    if (ROOT / ANALYSIS).exists():
        raise FileExistsError("analysis exists; never overwrite")
    doc = evaluate()
    atomic_json(ROOT / ANALYSIS, doc)
    print(json.dumps({"label": doc["label"], "gate_H": doc["gate_H"], "gate_D": doc["gate_D"], "gate_D_inference": doc["gate_D_inference"]}))


if __name__ == "__main__":
    main()
