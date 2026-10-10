#!/usr/bin/env python
"""P2-PATH3 analysis (frozen contract configs/inference_cf/p2_path3.json).

primary   (reference-free) validity over all 100 rows: replay barrier (12 exact vs PATH2), DEV24_OTHER A4 identity,
          A4 episode integrity, controller ownership/no-trigger/A4-winner identity, compute counts; descriptive
          trigger/winner/change/margin summaries for FULL100/DEV24/PATH2_12/NOVEL76, distances, caps, severe flags.
secondary (references behind the committed seal and committed primary-phase audit PASS) canonical B0/AUTO/A2/A4/G1
          metrics on FULL100 and partitions, ABSOLUTE_SAFETY (TTA1 rules), aggregate rescue, POI benefit/retention,
          NOVEL76 breadth/concentration, LODO, paired dialogue-block bootstrap, terminal label (frozen precedence).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.episodic_tta import severe_truncation
from csasr.inference_cf.path_decode import edit_distance, stream

CONFIG = "configs/inference_cf/p2_path3.json"
BASE = "results/inference_cf/p2path3"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
PARTS = ("FULL100", "DEV24", "PATH2_12", "NOVEL76")
SYSTEMS = ("B0_FORCED", "B0_AUTO", "A2", "A4", "G1")
LABELS = ("P2_PATH3_INVALID", "P2_PATH3_SEQUENCE_DAMAGE", "P2_PATH3_NO_USEFUL_GAIN", "P2_PATH3_OVERCONSERVATIVE",
          "P2_PATH3_SIGNAL_CONCENTRATED", "P2_PATH3_SUPPORTED")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- frozen outcome rules (pure) -------------------------------------------------------------------------------

def safety(p: dict, c: dict) -> dict:
    """ABSOLUTE_SAFETY (TTA1 thresholds, G1 vs B0_FORCED). Rates may be None (empty baseline-correct denominator):
    vacuous/not-assessable, never substituted. cap_delta = G1 cap count - B0 cap count."""
    S = c["safety"]
    G = S["float_guard"]
    le = lambda v, t: v is None or v <= t + G
    ge = lambda v, t: v is None or v >= t - G
    crit = {"MER": p["mer_increase"] <= S["max_MER_increase"] + G, "ZH_CER": p["zh_cer_increase"] <= S["max_ZH_CER_increase"] + G,
            "matrix_ZH_retention": ge(p["zh_retention"], S["min_matrix_ZH_retention"]),
            "embedded_EN_retention": ge(p["en_retention"], S["min_embedded_EN_retention"]),
            "outside_harm": le(p["outside_harm_rate"], S["max_outside_POI_harm_rate"]),
            "POI_corruption": le(p["poi_corruption_rate"], S["max_POI_corruption_rate"]),
            "caps": p["cap_delta"] <= S["max_additional_caps"], "severe_truncation": p["new_severe_truncations"] <= S["max_new_severe_truncations"]}
    return {**crit, "pass": all(crit.values()),
            "not_assessable": [k for k, kk in (("matrix_ZH_retention", "zh_retention"), ("embedded_EN_retention", "en_retention"),
                                               ("outside_harm", "outside_harm_rate"), ("POI_corruption", "poi_corruption_rate")) if p[kk] is None]}


def rescue(z_b0: int, z_a4: int, z_g1: int, c: dict) -> dict:
    X = max(z_a4 - z_b0, 0)
    R = z_a4 - z_g1
    req = max(1, math.ceil(c["aggregate_rescue"]["fraction"] * X)) if X > 0 else 1
    return {"Z_B0": z_b0, "Z_A4": z_a4, "Z_G1": z_g1, "X_ZH": X, "R_ZH": R, "required_R_ZH": req, "pass": R >= req}


def benefit(poi_b0: int, poi_a4: int, poi_g1: int, mer_g1: float, mer_a4: float, rescue_pass: bool, c: dict) -> dict:
    B, G = c["benefit"], c["safety"]["float_guard"]
    I_A4, I_G1 = poi_b0 - poi_a4, poi_b0 - poi_g1
    ret = I_G1 / I_A4 if I_A4 > 0 else None
    ret_pass = I_G1 > 0 and (I_A4 <= 0 or ret >= B["retention_min"] - G)
    dm = mer_g1 - mer_a4
    mer_ok = dm <= B["relative_MER_increase_max"] + G
    return {"I_A4": I_A4, "I_G1": I_G1, "retention": ret, "MER_G1_minus_A4": dm, "MER_relative_ok": mer_ok,
            "BENEFIT_RETENTION_PASS": bool(ret_pass), "USEFUL_GAIN_PASS": bool(rescue_pass and I_G1 > 0 and mer_ok)}


def breadth(r: list, dialogues: list, c: dict) -> dict:
    """NOVEL76 only. r_i = ZH_A4(i) - ZH_G1(i); positive rescue for concentration, net rescue for benefit."""
    Bc = c["breadth"]
    p = [max(x, 0) for x in r]
    h = [max(-x, 0) for x in r]
    Rp, Rm, Rn = sum(p), sum(h), sum(r)
    Pd = {}
    for x, d in zip(p, dialogues):
        Pd[d] = Pd.get(d, 0) + x
    n_utt = sum(x >= 1 for x in r)
    n_dlg = sum(v > 0 for v in Pd.values())
    C_utt = max(p) / Rp if Rp > 0 else 1.0
    C_dlg = max(Pd.values()) / Rp if Rp > 0 else 1.0
    crit = {"R_net": Rn >= Bc["R_net_76_min"], "rescue_utterances": n_utt >= Bc["rescue_utterances_min"],
            "rescue_dialogues": n_dlg >= Bc["rescue_dialogues_min"], "C_utt": C_utt <= Bc["C_utt_max"] + 1e-12, "C_dlg": C_dlg <= Bc["C_dlg_max"] + 1e-12}
    return {"R_plus": Rp, "R_minus": Rm, "R_net_76": Rn, "N_rescue_utt": n_utt, "N_rescue_dialogue": n_dlg, "C_utt": C_utt, "C_dlg": C_dlg,
            "P_d": Pd, "criteria": crit, "BREADTH_PASS": all(crit.values())}


def lodo(r: list, dialogues: list) -> dict:
    keys = sorted(set(dialogues))
    vals = {d: sum(x for x, dd in zip(r, dialogues) if dd != d) for d in keys}
    v = list(vals.values())
    pos = sum(x > 0 for x in v)
    need = math.ceil(0.9 * len(keys))
    return {"dialogues": len(keys), "values": vals, "min": min(v), "median": float(np.median(v)), "count_positive": pos,
            "proportion_positive": pos / len(keys), "required": need, "LODO_ROBUST": pos >= need}


def decide(valid: bool, safety_pass: bool, useful_gain: bool, retention_pass: bool, breadth_pass: bool) -> str:
    if not valid:
        return "P2_PATH3_INVALID"
    if not safety_pass:
        return "P2_PATH3_SEQUENCE_DAMAGE"
    if not useful_gain:
        return "P2_PATH3_NO_USEFUL_GAIN"
    if not retention_pass:
        return "P2_PATH3_OVERCONSERVATIVE"
    if not breadth_pass:
        return "P2_PATH3_SIGNAL_CONCENTRATED"
    return "P2_PATH3_SUPPORTED"


def quantiles(v: list):
    if not v:
        return None
    return {"min": min(v), "median": float(np.median(v)), "max": max(v), "q": [float(np.quantile(v, q)) for q in (0, .25, .5, .75, 1)]}


def count_delta_ci(ca: np.ndarray, cb: np.ndarray, col: int, idxs) -> dict:
    vals = [float(ca[i, col].sum() - cb[i, col].sum()) for i in idxs]
    return {"delta": float(ca[:, col].sum() - cb[:, col].sum()), "valid_draws": len(vals),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]}


# ---- load / primary ------------------------------------------------------------------------------------------------

def load(run_dir: Path) -> dict:
    man = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows = [json.loads((run_dir / f"rows/{i:03d}.json").read_text()) if (run_dir / f"rows/{i:03d}.json").exists() else {"status": "missing"}
            for i in range(len(plan["rows"]))]
    return {"manifest": man, "plan": plan, "rows": rows, "runtime": json.loads((run_dir / "runtime.json").read_text())}


def summarize(per: list, ids: list) -> dict:
    S = set(ids)
    es = [e for e in per if e["utterance_id"] in S]
    trig = [e for e in es if e["trigger"]]
    return {"n": len(es), "dialogues": len({e["dialogue_id"] for e in es}), "A4_noop": sum(e["A4_noop"] for e in es),
            "A4_changed_vs_B0": sum(e["A4_changed_vs_B0"] for e in es), "triggers": len(trig), "no_trigger": len(es) - len(trig),
            "theta0_winners": sum(e["winner"] == "theta0" for e in trig), "A4_winners": sum(e["winner"] == "A4" for e in trig),
            "G1_changed_vs_A4": sum(e["G1_changed_vs_A4"] for e in es), "trigger_dialogues": len({e["dialogue_id"] for e in trig}),
            "theta0_winner_dialogues": len({e["dialogue_id"] for e in trig if e["winner"] == "theta0"}),
            "margin": quantiles([e["margin"] for e in trig]), "trigger_k": [e["k"] for e in trig],
            "distances_G1": {t: sum(e[f"d_{t}"] for e in es) for t in ("A4", "B0", "AUTO")},
            "caps": {"A4": sum(e["A4_terminated"] == "cap" for e in es), "G1": sum(e["G1_terminated"] == "cap" for e in es)},
            "severe_vs_B0": {"A4": sum(e["A4_severe_vs_B0"] for e in es), "G1": sum(e["G1_severe_vs_B0"] for e in es)}}


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    rows, plan, rt = d["rows"], d["plan"], d["runtime"]
    integ = {"all_100_ok": len(rows) == 100 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
             "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
             and bool(rt.get("model_grads_none")), "barrier_pass": bool((rt.get("barrier") or {}).get("pass")) and (rt.get("barrier") or {}).get("exact") == 12}
    res = {"schema": "p2_path3_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "references_used": False, "integrity": integ}
    if not integ["all_100_ok"]:
        res["valid"] = False
        res["rows_ok"] = sum(r.get("status") == "ok" for r in rows)
        return res
    ok = {k: True for k in ("identity", "replay12_exact", "dev24_other_identity", "a4_episodes", "ownership", "no_trigger_identity",
                            "a4_winner_identity", "g1_forced_winner", "live_audit", "execution_order")}
    per = []
    order = plan["execution_order"]
    for i, (r, x) in enumerate(zip(rows, plan["rows"])):
        a, au = x["analysis"], x["audit"]
        u = a["utterance_id"]
        ok["identity"] &= r["identity"] == u and r["canonical_index"] == i and r["partition"] == au["partition"] and r["dialogue_id"] == a["dialogue_id"]
        ok["execution_order"] &= order[r["execution_index"]] == u
        part = au["partition"]
        if part == "PATH2_12":
            ok["replay12_exact"] &= (r["replay_exact"] and not r["replay_mismatches"] and r["theta0_equals_B0"] and r["lang_equals_sealed"]
                                     and r["effective_equals_checkpoint_bf16"] and r["G0_equals_sealed_A4"] and r["reset_ok_phase1"] and r["reset_ok_phase2"]
                                     and r["A4_text_equals_sealed"])
        elif part == "DEV24_OTHER":
            ok["dev24_other_identity"] &= r["lang_equals_sealed"] and r["effective_equals_checkpoint_bf16"] and r["G0_equals_sealed_A4"] \
                and r["A4_text_equals_sealed"] and r["reset_ok"]
        else:
            ok["a4_episodes"] &= r["reset_ok"] and r["A4_start_hash"] == rt["theta0_ln_hash"] == r["reset_hash_after"]
        lg = r["A4_log"]
        ok["a4_episodes"] &= lg["steps"] == 2 and lg["loss_evaluations"] == 3 and all(lg["finite"]) and \
            all(math.isfinite(v) for v in lg["losses"] + lg["grad_l2"])
        if lg["noop"]:
            ok["a4_episodes"] &= lg["effective_changed_scalars"] == 0 and r["A4_state_hash"] == rt["theta0_ln_hash"]
        det = r["detector"]
        ok["ownership"] &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        trig = det["trigger"]
        dec = r["decision"]
        G0, G1 = r["G0"], r["G1"]
        e = {"utterance_id": u, "dialogue_id": a["dialogue_id"], "partition": part, "A4_noop": lg["noop"], "trigger": trig, "k": None,
             "winner": None, "margin": None, "S_cons": None}
        if trig:
            ok["ownership"] &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"]
                                   for v in r["score_paths"].values()) and G1["state_locked"] and G1["trace"]["prefix_ok"] and G1["trace"]["positions_ok"]
            w = dec["b0"] if dec["winner"] == "theta0" else dec["b4"]
            ok["g1_forced_winner"] &= G1["forced"] == [w["tokens"][0]] and [f["token"] for f in G1["trace"]["forced"]] == G1["forced"] \
                and G1["decision_hash"] == dec["decision_hash"] and dec["b0"]["H_eff"] >= 1 and dec["b4"]["H_eff"] >= 1
            if dec["winner"] == "A4":
                ok["a4_winner_identity"] &= G1["tokens"] == G0["tokens"] and G1["terminated"] == G0["terminated"]
            e.update(k=dec["k"], winner=dec["winner"], margin=dec["margin"], S_cons=dec["S_cons"], b0=dec["b0"]["tokens"], b4=dec["b4"]["tokens"],
                     H_eff=[dec["b0"]["H_eff"], dec["b4"]["H_eff"]], S={kk: v["score"] for kk, v in dec["scores"].items()}, strict=dec["strict"])
        else:
            ok["no_trigger_identity"] &= dec is None and G1["tokens"] == G0["tokens"] and G1["terminated"] == G0["terminated"]
        B0, AU = au["B0"], a["AUTO"]
        sB, sA, s4, s1 = stream(B0["tokens"], B0["terminated"]), stream(AU["tokens"], AU["terminated"]), stream(G0["tokens"], G0["terminated"]), \
            stream(G1["tokens"], G1["terminated"])
        e.update(G1_changed_vs_A4=s1 != s4, A4_changed_vs_B0=s4 != sB, d_A4=edit_distance(s1, s4), d_B0=edit_distance(s1, sB), d_AUTO=edit_distance(s1, sA),
                 A4_d_B0=edit_distance(s4, sB), A4_length=len(G0["tokens"]), G1_length=len(G1["tokens"]), B0_length=len(B0["tokens"]),
                 A4_terminated=G0["terminated"], G1_terminated=G1["terminated"], B0_terminated=B0["terminated"],
                 A4_severe_vs_B0=severe_truncation(len(B0["tokens"]), len(G0["tokens"]), G0["terminated"]),
                 G1_severe_vs_B0=severe_truncation(len(B0["tokens"]), len(G1["tokens"]), G1["terminated"]),
                 G1_severe_vs_A4=severe_truncation(len(G0["tokens"]), len(G1["tokens"]), G1["terminated"]), native_lang=r["auto_condition"]["lang_id"])
        per.append(e)
    first = rows[plan["ids"].index(order[0])]
    la = first.get("live_audit") or {}
    ok["live_audit"] = bool(la.get("pass")) and la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    n_trig = sum(e["trigger"] for e in per)
    B = c["compute"]
    ok["compute"] = (cnt["A4"]["optimizer_steps"] == B["optimizer_steps"] == 200
                     and cnt["A4"]["backwards"] + cnt["audit"]["backwards"] <= B["backwards_max_including_first_live_audit"]
                     and cnt["triggers"] == n_trig <= B["G1_guard_events_max"] and cnt["rollouts"] == 2 * n_trig and cnt["score_paths"] == 4 * n_trig
                     and cnt["G1_decodes"] == n_trig and cnt["G2_decodes"] == 0 and cnt["G0_decodes"] == 100 and cnt["theta0_decodes"] == 12
                     and cnt["detector_runs"] == 100 and cnt["encoder_passes"] == 100 and cnt["detect_language_calls"] == 100
                     and cnt["rows_entered_after_barrier"] == 88)
    ok["wall_clock"] = rt["elapsed_sec"] <= 60 * B["hard_minutes"]
    integ.update(ok)
    valid = all(integ.values())
    P = plan["partitions"]
    res.update(valid=valid, per_row=per, summaries={k: summarize(per, P[k]) for k in PARTS},
               runtime={k: rt.get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "replay_phase1_sec", "replay_phase2_sec", "remaining_sec",
                                               "peak_alloc", "peak_reserved", "counters", "gpu", "barrier")})
    return res


# ---- secondary -----------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import draws, outside, ratio_delta, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("PATH3 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed PATH3 outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != "P2_PATH3_AUDIT: PASS":
        raise PermissionError("primary-phase audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    refs = load_references()
    plan = d["plan"]
    ids = plan["ids"]
    X = {x["analysis"]["utterance_id"]: x for x in plan["rows"]}
    rowmap = {r["identity"]: r for r in d["rows"]}
    R = {u: refs[u]["reference"] for u in ids}
    D = {u: refs[u]["dialogue_id"] for u in ids}
    integ = {"primary_valid": bool(prim["valid"]), "dialogues_match": all(D[u] == X[u]["analysis"]["dialogue_id"] for u in ids) and len(set(D.values())) == 20}
    H, T, L = {}, {}, {}
    for u in ids:
        a, au, r = X[u]["analysis"], X[u]["audit"], rowmap[u]
        H[u] = {"B0_FORCED": a["y_B_text"], "B0_AUTO": a["AUTO"]["text"], "A2": a["A2"]["text"], "A4": r["G0"]["text"], "G1": r["G1"]["text"]}
        T[u] = {"B0_FORCED": au["B0"]["terminated"], "B0_AUTO": a["AUTO"]["terminated"], "A2": a["A2"]["terminated"], "A4": r["G0"]["terminated"],
                "G1": r["G1"]["terminated"]}
        L[u] = {"B0_FORCED": len(au["B0"]["tokens"]), "B0_AUTO": len(a["AUTO"]["tokens"]), "A2": len(a["A2"]["tokens"]), "A4": len(r["G0"]["tokens"]),
                "G1": len(r["G1"]["tokens"])}
    CNT = {u: {s: utt_counts(R[u], H[u][s]) for s in SYSTEMS} for u in ids}

    def agg(us):
        Rs = [R[u] for u in us]
        Bs = [H[u]["B0_FORCED"] for u in us]
        out = {}
        for s in SYSTEMS:
            Hs = [H[u][s] for u in us]
            cm = corpus_metrics(Rs, Hs)
            e = {"zh": sum(CNT[u][s][4] for u in us), "poi": sum(CNT[u][s][0] for u in us), "mixed": sum(CNT[u][s][2] for u in us),
                 "en": sum(CNT[u][s][6] for u in us), "num_poi_errors": cm["num_poi_errors"], "num_poi": cm["num_poi"], "num_zh_ref": cm["num_zh_ref"],
                 "num_en_ref": cm["num_en_ref"], "pier": cm["pier"], "mer": cm["mer"], "en_wer": cm["en_wer"], "zh_cer": cm["zh_cer"],
                 "substitutions": cm["substitutions"], "deletions": cm["deletions"], "insertions": cm["insertions"],
                 "caps": sum(T[u][s] == "cap" for u in us), "severe_vs_B0": sum(severe_truncation(L[u]["B0_FORCED"], L[u][s], T[u][s]) for u in us)}
            if s != "B0_FORCED":
                cc = correction_corruption(Rs, Bs, Hs)
                outs = [outside(r_, b, h) for r_, b, h in zip(Rs, Bs, Hs)]
                ob = sum(o["baseline_correct_outside"] for o in outs)
                e.update(corrections_vs_B0=cc["corrections"], corruptions_vs_B0=cc["corruptions"], poi_corruption_rate=cc["corruption_rate"],
                         zh_retention_vs_B0=matrix_zh_retention(Rs, Bs, Hs)["rate"], en_retention_vs_B0=embedded_en_retention(Rs, Bs, Hs)["rate"],
                         outside_harm_vs_B0=(sum(o["outside_harm"] for o in outs) / ob) if ob else None, outside_harm_count=sum(o["outside_harm"] for o in outs),
                         outside_baseline_correct=ob)
            out[s] = e
        return out
    P = plan["partitions"]
    M = {k: agg(P[k]) for k in PARTS}
    F = M["FULL100"]
    integ["populations_nonempty"] = all(F["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(F[s][x] is not None and math.isfinite(F[s][x]) for s in SYSTEMS for x in ("pier", "mer", "en_wer", "zh_cer"))
    integ["poi_count_consistent"] = all(F[s]["poi"] == F[s]["num_poi_errors"] for s in SYSTEMS)
    g, b = F["G1"], F["B0_FORCED"]
    point = {"mer_increase": g["mer"] - b["mer"], "zh_cer_increase": g["zh_cer"] - b["zh_cer"], "zh_retention": g["zh_retention_vs_B0"],
             "en_retention": g["en_retention_vs_B0"], "outside_harm_rate": g["outside_harm_vs_B0"], "poi_corruption_rate": g["poi_corruption_rate"],
             "cap_delta": g["caps"] - b["caps"], "added_caps_rowwise": sum(T[u]["G1"] == "cap" and T[u]["B0_FORCED"] != "cap" for u in ids),
             "new_severe_truncations": g["severe_vs_B0"]}
    saf = safety(point, c)
    resc = rescue(F["B0_FORCED"]["zh"], F["A4"]["zh"], F["G1"]["zh"], c)
    ben = benefit(F["B0_FORCED"]["poi"], F["A4"]["poi"], F["G1"]["poi"], F["G1"]["mer"], F["A4"]["mer"], resc["pass"], c)
    n76 = P["NOVEL76"]
    r76 = [CNT[u]["A4"][4] - CNT[u]["G1"][4] for u in n76]
    d76 = [D[u] for u in n76]
    br = breadth(r76, d76, c)
    lo = lodo(r76, d76)
    # concentration context on every partition (descriptive)
    conc = {}
    for k in PARTS:
        us = P[k]
        rr = {u: CNT[u]["A4"][4] - CNT[u]["G1"][4] for u in us}
        pp = {u: max(v, 0) for u, v in rr.items()}
        Rp = sum(pp.values())
        pd_ = {}
        for u in us:
            pd_[D[u]] = pd_.get(D[u], 0) + pp[u]
        conc[k] = {"R_plus": Rp, "R_minus": sum(max(-v, 0) for v in rr.values()), "R_net": sum(rr.values()),
                   "max_row": max(pp, key=pp.get) if Rp else None, "max_row_share": (max(pp.values()) / Rp) if Rp else None,
                   "max_dialogue": max(pd_, key=pd_.get) if Rp else None, "max_dialogue_share": (max(pd_.values()) / Rp) if Rp else None,
                   "rescue_rows": sum(v >= 1 for v in rr.values()), "harm_rows": sum(v <= -1 for v in rr.values())}
    full_pp = {u: max(CNT[u]["A4"][4] - CNT[u]["G1"][4], 0) for u in ids}
    Rp100 = sum(full_pp.values())
    known_u, known_d = "ZH-CN_U0023_S0_664", D["ZH-CN_U0023_S0_664"]
    conc["FULL100_sources"] = {
        "R_plus": Rp100, "from_PATH2_12": sum(full_pp[u] for u in P["PATH2_12"]),
        "from_DEV24_OTHER": sum(full_pp[u] for u in P["DEV24"] if u not in set(P["PATH2_12"])), "from_NOVEL76": sum(full_pp[u] for u in n76),
        "share_PATH2_12": sum(full_pp[u] for u in P["PATH2_12"]) / Rp100 if Rp100 else None,
        "share_NOVEL76": sum(full_pp[u] for u in n76) / Rp100 if Rp100 else None,
        "U0023_S0_664_rescue": full_pp[known_u], "U0023_S0_664_share": full_pp[known_u] / Rp100 if Rp100 else None,
        "known_dialogue": known_d, "known_dialogue_share": sum(full_pp[u] for u in ids if D[u] == known_d) / Rp100 if Rp100 else None}
    # bootstrap (descriptive): paired dialogue blocks over FULL100 in canonical order, shared draws
    Darr = [D[u] for u in ids]
    idxs = draws(Darr, reps=c["bootstrap"]["reps"], seed=c["bootstrap"]["seed"])
    CA = {s: np.array([CNT[u][s] for u in ids], dtype=float) for s in SYSTEMS}
    boot = {}
    for base in ("A4", "B0_FORCED"):
        boot[f"G1_minus_{base}"] = {n: ratio_delta(CA["G1"], CA[base], a_, q_, idxs) for n, (a_, q_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5)}.items()}
        boot[f"G1_minus_{base}"]["poi_errors"] = count_delta_ci(CA["G1"], CA[base], 0, idxs)
    valid = all(integ.values())
    label = decide(valid, saf["pass"], ben["USEFUL_GAIN_PASS"], ben["BENEFIT_RETENTION_PASS"], br["BREADTH_PASS"])
    per = []
    for u in ids:
        e = {"utterance_id": u, "dialogue_id": D[u], "partition": X[u]["audit"]["partition"], "counts": CNT[u], "terminated": T[u], "length": L[u],
             "r_ZH_A4_minus_G1": CNT[u]["A4"][4] - CNT[u]["G1"][4], "dPOI_G1_minus_A4": CNT[u]["G1"][0] - CNT[u]["A4"][0],
             "dMixed_G1_minus_A4": CNT[u]["G1"][2] - CNT[u]["A4"][2], "dZH_G1_minus_B0": CNT[u]["G1"][4] - CNT[u]["B0_FORCED"][4]}
        per.append(e)
    top = sorted([e for e in per if e["r_ZH_A4_minus_G1"] > 0], key=lambda e: -e["r_ZH_A4_minus_G1"])
    harms = [e for e in per if e["r_ZH_A4_minus_G1"] < 0 or e["dPOI_G1_minus_A4"] > 0 or e["dMixed_G1_minus_A4"] > 0]
    auto_delta = {k: F["G1"][k] - F["B0_AUTO"][k] for k in ("pier", "mer", "en_wer", "zh_cer", "zh", "poi", "mixed")}
    return {"schema": "p2_path3_secondary_v1", "output_seal_hash": seal["seal_hash"], "integrity": integ, "valid": valid, "metrics": M,
            "safety_point": point, "safety": saf, "ABSOLUTE_SAFETY_PASS": saf["pass"], "aggregate_rescue": resc, "AGGREGATE_RESCUE_PASS": resc["pass"],
            "benefit": ben, "breadth": br, "BREADTH_PASS": br["BREADTH_PASS"], "lodo": lo, "concentration": conc, "bootstrap": boot,
            "G1_minus_AUTO": auto_delta, "label": label, "top_rescues": top[:10], "new_harms": harms, "per_row": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("primary", "secondary"):
        p = sub.add_parser(n)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable((primary if args.cmd == "primary" else secondary)(args.run))
    atomic_json(out, res)
    if args.cmd == "primary":
        print(json.dumps({k: res.get(k) for k in ("valid", "integrity")}, indent=1))
        print(json.dumps({k: {kk: v[kk] for kk in ("n", "triggers", "no_trigger", "theta0_winners", "A4_winners", "G1_changed_vs_A4", "trigger_dialogues",
                                                    "theta0_winner_dialogues", "margin", "A4_noop")} for k, v in res.get("summaries", {}).items()}, indent=1))
    else:
        print(json.dumps({k: res[k] for k in ("label", "valid", "integrity", "safety", "aggregate_rescue", "benefit")}, indent=1))
        print(json.dumps({k: v for k, v in res["breadth"].items() if k != "P_d"}, indent=1))
        print(json.dumps({s: {k: v[k] for k in ("zh", "poi", "mixed", "pier", "mer", "en_wer", "zh_cer", "caps")} for s, v in res["metrics"]["FULL100"].items()}, indent=1))


if __name__ == "__main__":
    main()
