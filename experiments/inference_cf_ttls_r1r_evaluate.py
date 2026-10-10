#!/usr/bin/env python
"""TTLS-R1R post-seal evaluation (frozen contract configs/inference_cf/ttls_r1r.json). Written before the run.

References are opened only after the committed R1R output seal (and the R1 seal) are verified. Free-decoding transcripts
only. Uses the unchanged TTLS-R1 evaluator primitives (corpus metrics, POI transitions, new-ZH, outside harm, events,
dialogue bootstrap) and adds the frozen lexical split (genuine wrong-language / same-language vs word-boundary vs
deletion), the effect decomposition (repair / optimizing z / preservation / step-0 + global-mask opportunity) and the
frozen R1R label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

import experiments.inference_cf_ttls_r1_evaluate as E
from csasr.inference_cf.core import atomic_json, digest

CONFIG = "configs/inference_cf/ttls_r1r.json"
BASE = "results/inference_cf/ttls_r1r"
SEAL = f"{BASE}/output_seal.json"
R1_RUN = "results/inference_cf/ttls_r1/run1"
ZERO_HIST = "results/inference_cf/ttls_r1_independent_audit/zero_census"
TTLS = ("T1", "T2", "T4", "T6")
MATCH = {"T1": "B2", "T2": "T2A", "T4": "T3", "T6": "T5"}
REUSED = ("T2A", "T3", "T5", "CD", "ACSUB")
SYSTEMS = ("B0", "B1", "B2", "Z0", "Z0H") + TTLS + REUSED + tuple(f"{a}_R1" for a in TTLS)
WRONG_LANG = ("wrong_language_substitution", "phonetic_transliteration_or_script")
DELETION = ("deletion", "boundary_error", "insertion_near_poi")


def lexical_kind(e: dict) -> str:
    """Frozen lexical split of one POI correction (config evaluation.lexical_classes)."""
    cat = e["B0_category"]
    if cat in DELETION:
        return "deletion_or_boundary"
    if cat in WRONG_LANG:
        return "genuine_wrong_language"
    comp = lambda x: "".join((x or "").lower().split())
    p, b = comp(e["poi"]), comp(e["B0"])
    if p and p in b:                                   # spacing-only or concatenation (the correction implies surfaces differ)
        return "word_boundary_repair"
    return "genuine_same_language"


def lexical_summary(events: list) -> dict:
    corr = [e for e in events if e["type"] == "correction"]
    kinds = Counter(lexical_kind(e) for e in corr)
    gen = [e for e in corr if lexical_kind(e).startswith("genuine")]
    return {"corrections": len(corr), "by_kind": dict(kinds), "genuine_lexical": len(gen),
            "genuine_lexical_dialogues": len({e["dialogue_id"] for e in gen}),
            "genuine_lexical_eos_recovery_rows": sum(e["eos_recovery_row"] for e in gen),
            "genuine_events": [{k: e[k] for k in ("utterance_id", "dialogue_id", "poi", "B0", "system", "B0_category")} | {"kind": lexical_kind(e)}
                               for e in gen]}


# ---- load (reference-free) -------------------------------------------------------------------------------------------

def load(run_rel: str) -> dict:
    run = ROOT / run_rel
    man = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / "results/inference_cf/ttls_r1/plan_sealed.json").read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows, integ = [], []
    for i, u in enumerate(man["ids"]):
        for lst, sub in ((rows, "rows"), (integ, "integrity")):
            p = run / sub / f"{i:03d}.json"
            r = json.loads(p.read_text()) if p.exists() else {"status": "missing", "identity": u}
            if r.get("status") == "ok" and (r["identity"] != u or r["manifest_hash"] != man["manifest_hash"]):
                raise ValueError(f"{sub} {i}")
            lst.append(r)
    r1 = E.load(R1_RUN)
    zh = [json.loads((ROOT / ZERO_HIST / f"{i:03d}.json").read_text()) for i in range(100)]
    if [z["id"] for z in zh] != man["ids"]:
        raise ValueError("audit zero census order")
    return {"manifest": man, "plan": plan, "rows": rows, "integrity": integ, "r1": r1, "zero_hist": zh,
            "runtime": json.loads((run / "runtime.json").read_text())}


def integrity(d: dict) -> dict:
    rows, A, rt = d["rows"], d["integrity"], d["runtime"]
    c = {"phaseA_100_ok": len(A) == 100 and all(r.get("status") == "ok" for r in A),
         "phaseB_100_ok": len(rows) == 100 and all(r.get("status") == "ok" for r in rows),
         "runtime_completed": rt.get("status") == "completed", "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("params_unchanged"))
         and bool(rt.get("model_grads_none"))}
    if not (c["phaseA_100_ok"] and c["phaseB_100_ok"]):
        return c
    c["B0_reproduced"] = all(r["B0"]["tokens_equal_S0"] and r["B0"]["terminated_equal_S0"] for r in A)
    c["candidates_equal_R1"] = all(all(v for k, v in r["candidates_equal_R1"].items() if k not in ("records", "p0_top")) for r in A)
    c["zero_decode_identity_ALL_100"] = sum(r["zero"]["ALL"]["decode_equals_B0"] for r in A) == 100
    c["zero_decode_identity_AC"] = all(r["zero"]["AC"]["decode_equals_B0"] for r in A if "AC" in r["zero"])
    c["zero_teacher_forced_bitwise"] = all(z["probe"]["logits_bitwise_equal"] and z["probe"]["ffn_input_bitwise_equal"] for r in A for z in r["zero"].values())
    c["zero_kl"] = all(z["kl0_stable"] <= 1e-12 for r in A for z in r["zero"].values())
    act = [r[a] for r in rows for a in TTLS if r[a]["status"] == "ok"]
    c["arm_gates"] = all(all(x["gates"].values()) for x in act)
    c["two_steps"] = all(len(x["log"]["losses"]) == 3 and len(x["log"]["grad_l2"]) == 2 and all(x["log"]["finite"]) for x in act)
    c["post_row_clean"] = all(r["post_row_clean_equals_B0"] for r in rows)
    c["abstain_is_B0"] = all(r[a]["equal_B0"] for r in rows for a in TTLS if r[a].get("noop"))
    return c


def systems(d: dict):
    H, Tk, Te = E.systems(d["r1"])                    # B0, B1, B2 from the plan; R1 arms / CD / ACSUB from sealed R1 rows
    for a in TTLS:
        H[f"{a}_R1"], Tk[f"{a}_R1"], Te[f"{a}_R1"] = H.pop(a), Tk.pop(a), Te.pop(a)
        H[a] = [r[a]["text"] for r in d["rows"]]
        Tk[a] = [r[a]["tokens"] for r in d["rows"]]
        Te[a] = [r[a]["terminated"] for r in d["rows"]]
    H["Z0"] = [r["zero"]["ALL"]["text"] for r in d["integrity"]]
    Tk["Z0"] = [r["zero"]["ALL"]["tokens"] for r in d["integrity"]]
    Te["Z0"] = [r["zero"]["ALL"]["terminated"] for r in d["integrity"]]
    H["Z0H"] = [z["zero"]["text"] for z in d["zero_hist"]]
    Tk["Z0H"] = [z["zero"]["tokens"] for z in d["zero_hist"]]
    Te["Z0H"] = [z["zero"]["terminated"] for z in d["zero_hist"]]
    return {k: H[k] for k in SYSTEMS}, {k: Tk[k] for k in SYSTEMS}, {k: Te[k] for k in SYSTEMS}


def decide(valid: bool, cmp_: dict, lex: dict) -> dict:
    if not valid:
        return {"label": "TTLS_R1R_INVALID", "promising_arms": []}
    prom = []
    for X in TTLS:
        Y = MATCH[X]
        a = lex[X]["genuine_lexical"] >= 8 and lex[X]["genuine_lexical_dialogues"] >= 4
        c = lex[X]["genuine_lexical"] >= lex[Y]["genuine_lexical"] and cmp_[X]["zh"]["new_zh_errors"] <= cmp_[Y]["zh"]["new_zh_errors"]
        if a and cmp_[X]["safe"] and c:
            prom.append(X)
    return {"label": "TTLS_R1R_VALID_AND_PROMISING" if prom else "TTLS_R1R_VALID_BUT_INSUFFICIENT", "promising_arms": prom}


def describe(d: dict) -> dict:
    rows, A = d["rows"], d["integrity"]
    zero = {"rows": 100, "Z0_ALL_decode_equals_B0": sum(r["zero"]["ALL"]["decode_equals_B0"] for r in A),
            "Z0_AC_rows": sum("AC" in r["zero"] for r in A), "Z0_AC_decode_equals_B0": sum(r["zero"]["AC"]["decode_equals_B0"] for r in A if "AC" in r["zero"]),
            "repaired_max_abs_logit_diff": max(z["probe"]["max_abs_logit_diff"] for r in A for z in r["zero"].values()),
            "repaired_logits_bitwise_rows": sum(r["zero"]["ALL"]["probe"]["logits_bitwise_equal"] for r in A),
            "repaired_ffn_bitwise_rows": sum(r["zero"]["ALL"]["probe"]["ffn_input_bitwise_equal"] for r in A),
            "historical_max_abs_logit_diff": max(r["zero"]["ALL"]["historical_kernel_probe"]["max_abs_logit_diff"] for r in A),
            "historical_logits_bitwise_rows": sum(r["zero"]["ALL"]["historical_kernel_probe"]["logits_bitwise_equal"] for r in A),
            "historical_ffn_chord_max": max(r["zero"]["ALL"]["historical_kernel_probe"]["edited_chord_max"] for r in A),
            "kl0_max": max(z["kl0_stable"] for r in A for z in r["zero"].values()),
            "historical_zero_census_token_changes": sum(z["zero"]["tokens"] != p["y_B"] for z, p in zip(d["zero_hist"], d["plan"]["rows"])),
            "phaseA_sec": d["runtime"].get("phaseA", {}).get("elapsed_sec")}
    opt = {}
    for a in TTLS:
        act = [r[a] for r in rows if r[a]["status"] == "ok"]
        old = [r[a] for r in d["r1"]["rows"] if r[a]["status"] == "ok"]
        if not act:
            opt[a] = {"active_rows": 0}
            continue
        Ls = np.array([x["log"]["losses"] for x in act])
        parts = {k: np.array([[p.get(k, np.nan) for p in x["log"]["parts"]] for x in act]) for k in act[0]["log"]["parts"][0]}
        opt[a] = {"active_rows": len(act), "trainable_scalars": act[0]["log"]["trainable_scalars"],
                  "loss_mean": Ls.mean(0).tolist(), "rows_loss_decreased": int((Ls[:, 2] < Ls[:, 0]).sum()),
                  "parts_mean": {k: np.nanmean(v, 0).tolist() for k, v in parts.items()},
                  "initial_P_max": float(np.nanmax(parts["P"][:, 0])) if "P" in parts else None,
                  "initial_P_max_R1": max(x["log"]["parts"][0]["P"] for x in old) if "P" in parts else None,
                  "initial_P_nonzero_rows_R1": sum(x["log"]["parts"][0]["P"] != 0 for x in old) if "P" in parts else None,
                  "grad_l2_mean": np.mean([x["log"]["grad_l2"] for x in act], 0).tolist(),
                  "grad_l2_mean_R1": np.mean([x["log"]["grad_l2"] for x in old], 0).tolist(),
                  "grad0_min": float(min(x["log"]["grad_l2"][0] for x in act)),
                  "z_norm_final_median": float(np.median([x["log"]["z_norm"][-1] for x in act])),
                  "z_norm_final_median_R1": float(np.median([x["log"]["z_norm"][-1] for x in old])),
                  "projected_rows": int(sum(any(x["log"]["projected"]) for x in act)),
                  "z_cos_R1R_vs_R1_median": float(np.median([np.dot(x["z_effective"], y["z_effective"]) /
                                                            (np.linalg.norm(x["z_effective"]) * np.linalg.norm(y["z_effective"]) + 1e-30)
                                                            for x, y in zip(act, old)])),
                  "ffn_chord_max": float(max(x["probe"]["edited_chord_max"] for x in act)),
                  "norm_rel_err_max": float(max(x["probe"]["edited_norm_rel_err_max"] for x in act)),
                  "outside_chord_max": float(max(x["probe"]["outside_chord_max"] for x in act)),
                  "site_rel_mean": float(np.mean([x["displacement"]["site_rel_mean"] for x in act])),
                  "kl_mean_all": float(np.mean([x["displacement"]["kl_mean_all"] for x in act])),
                  "adapt_sec_total": float(sum(x["log"]["adapt_sec"] for x in act)), "decode_sec_total": float(sum(x["decode_sec"] for x in act)),
                  "adapt_sec_total_R1": float(sum(x["log"]["adapt_sec"] for x in old)),
                  "peak_alloc_max_GB": max((x["peak"] or {}).get("alloc", 0) for x in act) / 1e9}
        if a in ("T4", "T6"):
            opt[a]["target_rank_step"] = np.mean([[p["target_rank"] for p in x["log"]["parts"]] for x in act], 0).tolist()
            opt[a]["margin_vs_b_mean"] = np.mean([[p["margin_vs_b"] for p in x["log"]["parts"]] for x in act], 0).tolist()
    rt = d["runtime"]
    compute = {"job_id": rt["job_id"], "gpu": rt["gpu"], "elapsed_sec": rt["elapsed_sec"], "setup_sec": rt["setup_sec"],
               "phaseA_sec": rt.get("phaseA", {}).get("elapsed_sec"), "phaseB_sec": rt.get("phaseB", {}).get("elapsed_sec"),
               "counters": rt["counters"], "row_elapsed_sec_total": float(sum(r["elapsed_sec"] for r in rows))}
    return {"zero_integrity": zero, "optimization": opt, "compute": compute}


def evaluate(run_rel: str) -> dict:
    c = json.loads((ROOT / CONFIG).read_text())
    c1 = json.loads((ROOT / E.CONFIG).read_text())
    for s_rel in (SEAL, E.SEAL):
        if not E.committed(s_rel):
            raise PermissionError(f"{s_rel} missing/uncommitted: references stay closed")
        seal = json.loads((ROOT / s_rel).read_text())
        if any(E.sha(ROOT / p) != h for p, h in seal["files"].items()):
            raise ValueError(f"sealed outputs changed: {s_rel}")
    seal = json.loads((ROOT / SEAL).read_text())
    d = load(run_rel)
    integ = integrity(d)
    if not (integ["phaseA_100_ok"] and integ["phaseB_100_ok"]):
        return {"schema": "ttls_r1r_evaluation_v1", "integrity": integ, "valid": False, "label": "TTLS_R1R_INVALID"}
    H, Tk, Te = systems(d)
    desc = describe(d)
    from experiments.inference_cf_p2_evaluate import load_references
    from csasr.evaluation.canonical import corpus_metrics
    refs = load_references()
    ids = d["manifest"]["ids"]
    R = [refs[u]["reference"] for u in ids]
    dialogues = [refs[u]["dialogue_id"] for u in ids]
    P = {p["utterance_id"]: p for p in d["plan"]["rows"]}
    integ["dialogues"] = dialogues == [P[u]["dialogue_id"] for u in ids]
    M = {k: {**corpus_metrics(R, H[k]), "caps": sum(t == "cap" for t in Te[k])} for k in SYSTEMS}
    integ["finite_metrics"] = all(M[k][x] is not None and math.isfinite(M[k][x]) for k in M for x in ("pier", "mer", "en_wer", "zh_cer"))
    valid = all(integ.values())
    genuine = set(c1["evaluation"]["genuine_substitution_categories"])
    cmp_ = {X: E.compare(R, H, Tk, Te, X, dialogues, ids, genuine, c1) for X in SYSTEMS if X != "B0"}
    lex = {X: lexical_summary(v["transitions"]["events"]) for X, v in cmp_.items()}
    label = decide(valid, cmp_, lex)
    r1_rule = E.decide(valid, cmp_, c1)                           # descriptive: the TTLS-R1 frozen rule on corrected arms
    vsB1 = {X: {k: M[X][k] - M["B1"][k] for k in ("pier", "mer", "en_wer", "zh_cer")} for X in SYSTEMS}
    pairs = [(X, "B0") for X in TTLS + ("Z0H",)] + [(X, "B1") for X in TTLS] + [(X, MATCH[X]) for X in TTLS] \
        + [(X, f"{X}_R1") for X in TTLS] + [("T2", "T1"), ("T6", "T4"), ("T4", "ACSUB"), ("T6", "ACSUB"), ("T1", "Z0H")]
    boot, C = E.bootstrap(R, H, dialogues, pairs)
    # corrected vs original, token / normalized-transcript level
    from csasr.evaluation.normalization import normalize_text
    vs_r1 = {}
    for X in TTLS:
        ch = [j for j in range(100) if Tk[X][j] != Tk[f"{X}_R1"][j]]
        vs_r1[X] = {"token_changed_rows": len(ch), "normalized_changed_rows": sum(normalize_text(H[X][j]) != normalize_text(H[f"{X}_R1"][j]) for j in ch),
                    "rows": [ids[j] for j in ch],
                    "mixed_error_change": int(sum(C[X][j][2] - C[f"{X}_R1"][j][2] for j in ch)),
                    "metric_delta": {k: M[X][k] - M[f"{X}_R1"][k] for k in ("pier", "mer", "en_wer", "zh_cer")}}
    vs_acsub = {X: {"token_identical_rows": sum(Tk[X][j] == Tk["ACSUB"][j] for j in range(100)),
                    "normalized_identical_rows": sum(normalize_text(H[X][j]) == normalize_text(H["ACSUB"][j]) for j in range(100))} for X in ("T4", "T6")}
    # step-0 / global-mask opportunity: rows whose first content token differs from B0 are unreachable for TTLS
    step0 = {}
    for X in ("B1", "B2", "T2A"):
        rows0 = [j for j in range(100) if Tk[X][j][:1] != Tk["B0"][j][:1]]
        ev = [e for e in cmp_[X]["transitions"]["events"] if e["type"] == "correction"]
        on0 = [e for e in ev if ids.index(e["utterance_id"]) in rows0]
        step0[X] = {"rows_step0_differs": len(rows0), "corrections": len(ev), "corrections_on_step0_rows": len(on0),
                    "genuine_lexical_on_step0_rows": sum(lexical_kind(e).startswith("genuine") for e in on0),
                    "genuine_lexical_total": lex[X]["genuine_lexical"],
                    "new_zh_on_step0_rows": int(sum(max(0, C[X][j][4] - C["B0"][j][4]) for j in rows0))}
    groups = {"D12": [j for j, u in enumerate(ids) if P[u]["group"] == "D"], "A88": [j for j, u in enumerate(ids) if P[u]["group"] == "A"],
              "A3_panel24": [j for j, u in enumerate(ids) if P[u]["a3_group"] is not None],
              "AC_accepted": [j for j, r in enumerate(d["rows"]) if r["candidates"]["t_star"] is not None]}
    sub = {g: {"n": len(ix), "metrics": E.subgroup_metrics(R, H, ix)} for g, ix in groups.items()}
    harm = {}
    for X in SYSTEMS:
        if X == "B0":
            continue
        ch = [j for j in range(100) if Tk[X][j] != Tk["B0"][j]]
        dm = [C[X][j][2] - C["B0"][j][2] for j in ch]
        harm[X] = {"changed_rows": len(ch), "rows_mixed_errors_worse": int(sum(x > 0 for x in dm)), "rows_mixed_errors_better": int(sum(x < 0 for x in dm)),
                   "net_mixed_error_change": int(sum(dm))}
    changed = []
    for j, u in enumerate(ids):
        diff = [X for X in SYSTEMS if X != "B0" and Tk[X][j] != Tk["B0"][j]]
        if any(X in TTLS or X in ("Z0", "Z0H") or X.endswith("_R1") for X in diff):
            changed.append({"utterance_id": u, "dialogue_id": dialogues[j], "group": P[u]["group"], "t_star": d["rows"][j]["candidates"]["t_star"],
                            "reference": R[j], "texts": {X: H[X][j] for X in SYSTEMS}, "differs_from_B0": diff,
                            "counts": {X: C[X][j].tolist() for X in SYSTEMS}})
    strip = lambda r: {k: v for k, v in r.items() if k != "events"}
    return {"schema": "ttls_r1r_evaluation_v1", "manifest_hash": d["manifest"]["manifest_hash"], "output_seal_hash": seal["seal_hash"],
            "integrity": integ, "valid": valid, **label, "r1_rule_descriptive": r1_rule, "metrics": M, "vs_B1": vsB1,
            "vs_B0": {X: {**v, "transitions": strip(v["transitions"])} for X, v in cmp_.items()}, "lexical": lex,
            "transition_events": {X: v["transitions"]["events"] for X, v in cmp_.items()},
            "corrected_vs_original": vs_r1, "ttls_vs_acsub": vs_acsub, "step0_opportunity": step0,
            "bootstrap_descriptive": boot, "subgroups": sub, "active_edit_harm": harm, "descriptive": desc,
            "changed_utterances": changed, "label_rule": c["label_rule"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("evaluation exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable(evaluate(args.run))
    atomic_json(out, res)
    keys = ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "substitutions", "deletions", "insertions", "caps")
    print(json.dumps({"label": res["label"], "promising": res.get("promising_arms"), "r1_rule": res.get("r1_rule_descriptive"),
                      "integrity": res["integrity"], "metrics": {k: {x: v[x] for x in keys} for k, v in res.get("metrics", {}).items()},
                      "vs_B0": {X: {"corr": v["transitions"]["corrections"], "corrupt": v["transitions"]["corruptions"],
                                    "lex": res["lexical"][X]["by_kind"], "genuine_lexical": res["lexical"][X]["genuine_lexical"],
                                    "eosrow": v["transitions"]["eos_recovery_row_corrections"], "newZH": v["zh"]["new_zh_errors"],
                                    "zhrep": v["zh"]["zh_repairs"], "safe": v["safe"], "changed": v["events"]["changed_rows"],
                                    "eos_rec_rows": v["events"]["eos_recovery_rows"], "prem": v["events"]["premature_eos_rows"]}
                              for X, v in res.get("vs_B0", {}).items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
