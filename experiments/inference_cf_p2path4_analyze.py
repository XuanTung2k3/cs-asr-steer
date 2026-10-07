#!/usr/bin/env python
"""P2-PATH4-R1 analysis (frozen contract configs/inference_cf/p2_path4.json, revision PATH4_R1_EOS_BOUNDARY_V1).

primary   (reference-free) validity over all 100 rows: A2 reconstruction barrier (theta0 = B0, fp32 masters / bf16 effective
          == archive, free A2 == sealed A2), dispatch (CONTENT_G1 / EOS_BOUNDARY / NO_TRIGGER) and trigger relation
          (trigger set == A2_DELTA), controller ownership/identities, direct-original comparator, compute counts;
          descriptive trigger/winner/change/margin summaries and EOS_BOUNDARY diagnostics (orientation, winners,
          termination, lengths, length deltas vs A2 and B0).
secondary (references behind the committed seal and committed primary-phase audit PASS) canonical B0/AUTO/A2/G metrics
          (+ sealed PATH3 A4+G1 display only), historical aggregate reproduction, ABSOLUTE_SAFETY (TTA1 rules), aggregate
          rescue, benefit retention (+ PIER/MER/mixed guards), FULL100 breadth with NOVEL76 conditions, LODO, paired
          dialogue-block bootstrap, boundary-row deltas, terminal label (frozen precedence).
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

CONFIG = "configs/inference_cf/p2_path4.json"
BASE = "results/inference_cf/p2path4"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
PARTS = ("FULL100", "DEV24", "PATH2_12", "NOVEL76", "A2_DELTA", "A2_SAME", "A2_DELTA_OUTSIDE_DEV24")
SUMMARY_PARTS = ("FULL100", "DEV24", "NOVEL76", "A2_DELTA_OUTSIDE_DEV24", "A2_DELTA", "PATH2_12")
SYSTEMS = ("B0_FORCED", "B0_AUTO", "A2", "G")
LABELS = ("P2_PATH4_INVALID", "P2_PATH4_SEQUENCE_DAMAGE", "P2_PATH4_SAFE_NO_ADDED_VALUE", "P2_PATH4_OVERCONSERVATIVE",
          "P2_PATH4_SIGNAL_CONCENTRATED", "P2_PATH4_GUARD_TRANSFER_SUPPORTED")


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
    """ABSOLUTE_SAFETY (TTA1 thresholds, G vs B0_FORCED). Rates may be None (empty baseline-correct denominator):
    vacuous/not-assessable, never substituted. added_caps = TTA1 row-wise (G cap and B0 not cap)."""
    S = c["safety"]
    G = S["float_guard"]
    le = lambda v, t: v is None or v <= t + G
    ge = lambda v, t: v is None or v >= t - G
    crit = {"MER": p["mer_increase"] <= S["max_MER_increase"] + G, "ZH_CER": p["zh_cer_increase"] <= S["max_ZH_CER_increase"] + G,
            "matrix_ZH_retention": ge(p["zh_retention"], S["min_matrix_ZH_retention"]),
            "embedded_EN_retention": ge(p["en_retention"], S["min_embedded_EN_retention"]),
            "outside_harm": le(p["outside_harm_rate"], S["max_outside_POI_harm_rate"]),
            "POI_corruption": le(p["poi_corruption_rate"], S["max_POI_corruption_rate"]),
            "caps": p["added_caps"] <= S["max_additional_caps"], "severe_truncation": p["new_severe_truncations"] <= S["max_new_severe_truncations"]}
    return {**crit, "pass": all(crit.values()),
            "not_assessable": [k for k, kk in (("matrix_ZH_retention", "zh_retention"), ("embedded_EN_retention", "en_retention"),
                                               ("outside_harm", "outside_harm_rate"), ("POI_corruption", "poi_corruption_rate")) if p[kk] is None]}


def rescue(z_b0: int, z_a2: int, z_g: int) -> dict:
    X = max(z_a2 - z_b0, 0)
    R = z_a2 - z_g
    req = max(1, math.ceil(0.50 * X)) if X > 0 else 1
    return {"Z_B0": z_b0, "Z_A2": z_a2, "Z_G": z_g, "X_ZH": X, "R_ZH": R, "required_R_ZH": req, "pass": R >= req}


def benefit(poi_b0: int, poi_a2: int, poi_g: int, pier_g: float, pier_a2: float, mer_g: float, mer_a2: float, mixed_g: int, mixed_a2: int,
            c: dict) -> dict:
    B, G = c["benefit"], c["safety"]["float_guard"]
    I_A2, I_G = poi_b0 - poi_a2, poi_b0 - poi_g
    ret = I_G / I_A2 if I_A2 > 0 else None
    crit = {"I_G_positive": I_G > 0, "retention": ret is not None and ret >= B["retention_min"] - G,
            "PIER": pier_g - pier_a2 <= B["relative_PIER_increase_max"] + G, "MER": mer_g - mer_a2 <= B["relative_MER_increase_max"] + G,
            "mixed": mixed_g <= mixed_a2}
    return {"I_A2": I_A2, "I_G": I_G, "retention": ret, "PIER_G_minus_A2": pier_g - pier_a2, "MER_G_minus_A2": mer_g - mer_a2,
            "mixed_G": mixed_g, "mixed_A2": mixed_a2, "criteria": crit, "BENEFIT_RETENTION_PASS": all(crit.values())}


def breadth(r: dict, dialogue: dict, novel: set, c: dict) -> dict:
    """FULL100 r_i = ZH_A2(i) - ZH_G(i); positive rescue for concentration; NOVEL76 net and positive-row conditions."""
    Bc = c["breadth"]
    p = {u: max(x, 0) for u, x in r.items()}
    h = {u: max(-x, 0) for u, x in r.items()}
    Rp, Rm, Rn = sum(p.values()), sum(h.values()), sum(r.values())
    Pd = {}
    for u, x in p.items():
        Pd[dialogue[u]] = Pd.get(dialogue[u], 0) + x
    n_utt = sum(x >= 1 for x in r.values())
    n_dlg = sum(v > 0 for v in Pd.values())
    C_utt = max(p.values()) / Rp if Rp > 0 else 1.0
    C_dlg = max(Pd.values()) / Rp if Rp > 0 else 1.0
    Rn_out = sum(x for u, x in r.items() if u in novel)
    n_out = sum(1 for u, x in r.items() if u in novel and x >= 1)
    crit = {"rescue_utterances": n_utt >= Bc["rescue_utterances_min"], "rescue_dialogues": n_dlg >= Bc["rescue_dialogues_min"],
            "C_utt": C_utt <= Bc["C_utt_max"] + 1e-12, "C_dlg": C_dlg <= Bc["C_dlg_max"] + 1e-12,
            "R_net_NOVEL76": Rn_out >= Bc["R_net_outside_DEV24_min"], "rescue_utterances_NOVEL76": n_out >= Bc["N_rescue_outside_DEV24_min"]}
    return {"R_plus": Rp, "R_minus": Rm, "R_net": Rn, "N_rescue_utt": n_utt, "N_rescue_dlg": n_dlg, "C_utt": C_utt, "C_dlg": C_dlg,
            "R_net_NOVEL76": Rn_out, "N_rescue_NOVEL76": n_out, "P_d": Pd, "criteria": crit, "BREADTH_PASS": all(crit.values())}


def lodo(r: dict, dialogue: dict, remove: list) -> dict:
    vals = {d: sum(x for u, x in r.items() if dialogue[u] != d) for d in remove}
    v = list(vals.values())
    if not v:
        return {"dialogues": 0, "values": {}, "min": None, "median": None, "max": None, "count_positive": 0, "proportion_positive": None}
    pos = sum(x > 0 for x in v)
    return {"dialogues": len(v), "values": vals, "min": min(v), "median": float(np.median(v)), "max": max(v), "count_positive": pos,
            "proportion_positive": pos / len(v)}


def decide(valid: bool, safety_pass: bool, rescue_pass: bool, retention_pass: bool, breadth_pass: bool) -> str:
    if not valid:
        return "P2_PATH4_INVALID"
    if not safety_pass:
        return "P2_PATH4_SEQUENCE_DAMAGE"
    if not rescue_pass:
        return "P2_PATH4_SAFE_NO_ADDED_VALUE"
    if not retention_pass:
        return "P2_PATH4_OVERCONSERVATIVE"
    if not breadth_pass:
        return "P2_PATH4_SIGNAL_CONCENTRATED"
    return "P2_PATH4_GUARD_TRANSFER_SUPPORTED"


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
    return {"n": len(es), "dialogues": len({e["dialogue_id"] for e in es}), "triggers": len(trig), "no_trigger": len(es) - len(trig),
            "content_G1": sum(e["mode"] == "CONTENT_G1" for e in es), "EOS_boundary": sum(e["mode"] == "EOS_BOUNDARY" for e in es),
            "theta0_winners": sum(e["winner"] == "theta0" for e in trig), "A2_winners": sum(e["winner"] == "A2" for e in trig),
            "G_changed_vs_A2": sum(e["G_changed_vs_A2"] for e in es), "trigger_dialogues": len({e["dialogue_id"] for e in trig}),
            "theta0_winner_dialogues": len({e["dialogue_id"] for e in trig if e["winner"] == "theta0"}),
            "changed_dialogues": len({e["dialogue_id"] for e in es if e["G_changed_vs_A2"]}),
            "margin": quantiles([e["margin"] for e in trig]),
            "distances_G": {t: sum(e[f"d_{t}"] for e in es) for t in ("A2", "B0", "AUTO")},
            "caps": {"A2": sum(e["A2_terminated"] == "cap" for e in es), "G": sum(e["G_terminated"] == "cap" for e in es)},
            "severe_vs_B0": {"A2": sum(e["A2_severe_vs_B0"] for e in es), "G": sum(e["G_severe_vs_B0"] for e in es)}}


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    rows, plan, rt = d["rows"], d["plan"], d["runtime"]
    integ = {"all_100_ok": len(rows) == 100 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
             "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
             and bool(rt.get("model_grads_none")), "barrier_pass": bool((rt.get("barrier") or {}).get("pass"))}
    res = {"schema": "p2_path4_primary_v1", "contract_revision": c["contract_revision"], "manifest_hash": d["manifest"]["manifest_hash"],
           "references_used": False, "integrity": integ}
    if not integ["all_100_ok"]:
        res["valid"] = False
        res["rows_ok"] = sum(r.get("status") == "ok" for r in rows)
        return res
    ok = {k: True for k in ("identity", "reconstruction100", "a2_episodes", "ownership", "dispatch_types", "trigger_relation", "trigger_set_equals_A2_DELTA",
                            "no_trigger_identity", "a2_winner_identity", "g_forced_winner", "boundary_semantics", "comparator", "live_audit")}
    per, bnd = [], []
    P = plan["partitions"]
    delta = set(P["A2_DELTA"])
    for i, (r, x) in enumerate(zip(rows, plan["rows"])):
        a, au = x["analysis"], x["audit"]
        u = a["utterance_id"]
        ok["identity"] &= r["identity"] == u and r["canonical_index"] == i and r["dialogue_id"] == a["dialogue_id"]
        ok["reconstruction100"] &= (r["theta0_equals_B0"] and r["A2_equals_sealed"] and r["masters_equal_checkpoint_fp32"]
                                    and r["effective_equals_checkpoint_bf16"] and r["reset_ok_phase1"] and r["reset_ok_phase2"])
        lg = r["A2_log"]
        ok["a2_episodes"] &= lg["steps"] == 2 and lg["loss_evaluations"] == 3 and all(lg["finite"]) and \
            all(math.isfinite(v) for v in lg["losses"] + lg["grad_l2"]) and lg["start_hash"] == lg["end_hash"] == rt["theta0_ln_hash"] == r["reset_hash_after"]
        det, dec, G, A = r["detector"], r["decision"], r["G1"], r["A2_free"]
        ok["ownership"] &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        trig, mode = det["trigger"], r["mode"]
        ok["trigger_relation"] &= r["trigger_relation_ok"]
        ok["trigger_set_equals_A2_DELTA"] &= trig == (u in delta)
        e = {"utterance_id": u, "dialogue_id": a["dialogue_id"], "mode": mode, "trigger": trig, "k": None, "winner": None, "margin": None}
        if trig:
            c0, c2 = det["argmax_theta0"], det["argmax_A4"]
            exp_mode = "EOS_BOUNDARY" if (c0 == 50257) != (c2 == 50257) else "CONTENT_G1"
            ok["dispatch_types"] &= mode == exp_mode
            ok["comparator"] &= r["comparator"]["ok"]
            if mode == "CONTENT_G1":
                win = "theta0" if dec["winner"] == "theta0" else "A2"
                w = dec["b0"] if win == "theta0" else dec["b4"]
                ok["ownership"] &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"]
                                       for v in r["score_paths"].values())
                ok["g_forced_winner"] &= G["forced"] == [w["tokens"][0]] and G["decision_hash"] == dec["decision_hash"] and G["state_locked"]
                e.update(H_eff=[dec["b0"]["H_eff"], dec["b4"]["H_eff"]], S_cons=dec["S_cons"])
            else:
                win = dec["winner"]
                ok["boundary_semantics"] &= (dec["H"] == 1 and dec["k"] == det["k"] and dec["prefix"] == det["prefix"] and (dec["c0"], dec["c2"]) == (c0, c2)
                                             and dec["selected_token"] == (c0 if win == "theta0" else c2) and G["forced"] == [dec["selected_token"]]
                                             and G["state_locked"] and all(p["state_locked"] for p in dec["paths"].values())
                                             and r["comparator"].get("original_refused_EOS_first") is True)
                if dec["selected_token"] == 50257:
                    ok["boundary_semantics"] &= G["tokens"] == det["prefix"] and G["terminated"] == "eos"
                orient = "theta0_EOS" if c0 == 50257 else "A2_EOS"
                bnd.append({"utterance_id": u, "dialogue_id": a["dialogue_id"], "orientation": orient, "k": dec["k"], "winner": win,
                            "winner_action": "EOS" if dec["selected_token"] == 50257 else "content", "S_theta0": dec["S_theta0"], "S_A2": dec["S_A2"],
                            "S_cons": dec["S_cons"], "margin": dec["margin"], "G_terminated": G["terminated"], "G_length": len(G["tokens"]),
                            "A2_length": len(A["tokens"]), "B0_length": len(au["B0"]["tokens"]),
                            "len_G_minus_A2": len(G["tokens"]) - len(A["tokens"]), "len_G_minus_B0": len(G["tokens"]) - len(au["B0"]["tokens"])})
            if win == "A2":
                ok["a2_winner_identity"] &= G["tokens"] == A["tokens"] and G["terminated"] == A["terminated"]
            e.update(k=det["k"], winner=win, margin=dec["margin"])
        else:
            ok["no_trigger_identity"] &= mode == "NO_TRIGGER" and dec is None and G["tokens"] == A["tokens"] and G["terminated"] == A["terminated"]
            ok["comparator"] &= r["comparator"]["ok"]
        B0, AU = au["B0"], a["AUTO"]
        sB, sAU, sA, sG = stream(B0["tokens"], B0["terminated"]), stream(AU["tokens"], AU["terminated"]), stream(A["tokens"], A["terminated"]), \
            stream(G["tokens"], G["terminated"])
        e.update(G_changed_vs_A2=sG != sA, d_A2=edit_distance(sG, sA), d_B0=edit_distance(sG, sB), d_AUTO=edit_distance(sG, sAU),
                 A2_length=len(A["tokens"]), G_length=len(G["tokens"]), B0_length=len(B0["tokens"]), A2_terminated=A["terminated"],
                 G_terminated=G["terminated"], A2_severe_vs_B0=severe_truncation(len(B0["tokens"]), len(A["tokens"]), A["terminated"]),
                 G_severe_vs_B0=severe_truncation(len(B0["tokens"]), len(G["tokens"]), G["terminated"]))
        per.append(e)
    la = rows[0].get("live_audit") or {}
    ok["live_audit"] = bool(la.get("pass")) and la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    n_trig = sum(e["trigger"] for e in per)
    n_c = sum(e["mode"] == "CONTENT_G1" for e in per)
    n_b = sum(e["mode"] == "EOS_BOUNDARY" for e in per)
    B = c["compute"]
    ok["compute"] = (cnt["A2"]["optimizer_steps"] == B["optimizer_steps"] == 200
                     and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] <= B["backwards_max_including_independent_live_check"]
                     and cnt["triggers"] == n_trig == n_c + n_b and cnt["logical_guard_events"] == 100 and cnt["detector_runs"] == 100
                     and cnt["detector_replays"] == n_c and cnt["rollouts"] == 2 * n_c and cnt["score_paths"] == 4 * n_c and cnt["G1_decodes"] == n_c
                     and cnt["boundary_score_paths"] == 2 * n_b and cnt["boundary_executions"] == n_b and cnt["G2_decodes"] == 0 and cnt["A4_calls"] == 0
                     and cnt["theta0_decodes"] == 100 and cnt["encoder_passes"] == 100)
    ok["wall_clock"] = rt["elapsed_sec"] <= 60 * B["hard_minutes"]
    integ.update(ok)
    valid = all(integ.values())
    res.update(valid=valid, per_row=per, summaries={k: summarize(per, P[k]) for k in SUMMARY_PARTS},
               EOS_boundary={"count": n_b, "orientation": {o: sum(b["orientation"] == o for b in bnd) for o in ("theta0_EOS", "A2_EOS")},
                             "winner_action": {w: sum(b["winner_action"] == w for b in bnd) for w in ("EOS", "content")},
                             "winner_model": {w: sum(b["winner"] == w for b in bnd) for w in ("theta0", "A2")}, "rows": bnd},
               runtime={k: rt.get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "phase1_sec", "phase2_sec", "peak_alloc", "peak_reserved",
                                               "counters", "gpu", "barrier")})
    return res


# ---- secondary -----------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import draws, outside, ratio_delta, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("PATH4 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed PATH4 outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != "P2_PATH4_AUDIT: PASS":
        raise PermissionError("primary-phase audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    refs = load_references()
    plan = d["plan"]
    ids = plan["ids"]
    X = {x["analysis"]["utterance_id"]: x for x in plan["rows"]}
    rowmap = {r["identity"]: r for r in d["rows"]}
    p3 = {json.loads(f.read_text())["identity"]: json.loads(f.read_text()) for f in sorted((ROOT / "results/inference_cf/p2path3/run1/rows").glob("*.json"))}
    R = {u: refs[u]["reference"] for u in ids}
    D = {u: refs[u]["dialogue_id"] for u in ids}
    integ = {"primary_valid": bool(prim["valid"]), "dialogues_match": all(D[u] == X[u]["analysis"]["dialogue_id"] for u in ids) and len(set(D.values())) == 20}
    H, T, L = {}, {}, {}
    for u in ids:
        a, au, r = X[u]["analysis"], X[u]["audit"], rowmap[u]
        H[u] = {"B0_FORCED": a["y_B_text"], "B0_AUTO": a["AUTO"]["text"], "A2": a["A2"]["text"], "G": r["G1"]["text"], "PATH3_A4_G1": p3[u]["G1"]["text"]}
        T[u] = {"B0_FORCED": au["B0"]["terminated"], "B0_AUTO": a["AUTO"]["terminated"], "A2": a["A2"]["terminated"], "G": r["G1"]["terminated"],
                "PATH3_A4_G1": p3[u]["G1"]["terminated"]}
        L[u] = {"B0_FORCED": len(au["B0"]["tokens"]), "B0_AUTO": len(a["AUTO"]["tokens"]), "A2": len(a["A2"]["tokens"]), "G": len(r["G1"]["tokens"]),
                "PATH3_A4_G1": len(p3[u]["G1"]["tokens"])}
    SY = SYSTEMS + ("PATH3_A4_G1",)
    CNT = {u: {s: utt_counts(R[u], H[u][s]) for s in SY} for u in ids}

    def agg(us, systems):
        Rs = [R[u] for u in us]
        Bs = [H[u]["B0_FORCED"] for u in us]
        out = {}
        for s in systems:
            Hs = [H[u][s] for u in us]
            cm = corpus_metrics(Rs, Hs)
            e = {"zh": sum(CNT[u][s][4] for u in us), "poi": sum(CNT[u][s][0] for u in us), "mixed": sum(CNT[u][s][2] for u in us),
                 "num_poi_errors": cm["num_poi_errors"], "num_poi": cm["num_poi"], "num_zh_ref": cm["num_zh_ref"], "num_en_ref": cm["num_en_ref"],
                 "pier": cm["pier"], "mer": cm["mer"], "en_wer": cm["en_wer"], "zh_cer": cm["zh_cer"], "substitutions": cm["substitutions"],
                 "deletions": cm["deletions"], "insertions": cm["insertions"], "caps": sum(T[u][s] == "cap" for u in us),
                 "added_caps_vs_B0": sum(T[u][s] == "cap" and T[u]["B0_FORCED"] != "cap" for u in us),
                 "severe_vs_B0": sum(severe_truncation(L[u]["B0_FORCED"], L[u][s], T[u][s]) for u in us)}
            if s != "B0_FORCED":
                cc = correction_corruption(Rs, Bs, Hs)
                outs = [outside(r_, b, h) for r_, b, h in zip(Rs, Bs, Hs)]
                ob = sum(o["baseline_correct_outside"] for o in outs)
                e.update(corrections_vs_B0=cc["corrections"], corruptions_vs_B0=cc["corruptions"], poi_corruption_rate=cc["corruption_rate"],
                         zh_retention_vs_B0=matrix_zh_retention(Rs, Bs, Hs)["rate"], en_retention_vs_B0=embedded_en_retention(Rs, Bs, Hs)["rate"],
                         outside_harm_vs_B0=(sum(o["outside_harm"] for o in outs) / ob) if ob else None)
            out[s] = e
        return out
    P = plan["partitions"]
    M = {"FULL100": agg(P["FULL100"], SY)}
    M.update({k: agg(P[k], SYSTEMS) for k in PARTS if k != "FULL100"})
    F = M["FULL100"]
    hist = c["historical_aggregate"]
    integ["historical_aggregates_reproduced"] = ([F["B0_FORCED"][k] for k in ("zh", "poi", "mixed")] == [hist["B0"]["ZH_errors"], hist["B0"]["POI_errors"], hist["B0"]["mixed_errors"]]
                                                 and [F["A2"][k] for k in ("zh", "poi", "mixed")] == [hist["A2"]["ZH_errors"], hist["A2"]["POI_errors"], hist["A2"]["mixed_errors"]]
                                                 and F["B0_FORCED"]["num_poi"] == hist["POI_denominator"])
    integ["populations_nonempty"] = all(F["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(F[s][x] is not None and math.isfinite(F[s][x]) for s in SY for x in ("pier", "mer", "en_wer", "zh_cer"))
    integ["poi_count_consistent"] = all(F[s]["poi"] == F[s]["num_poi_errors"] for s in SY)
    g, b = F["G"], F["B0_FORCED"]
    point = {"mer_increase": g["mer"] - b["mer"], "zh_cer_increase": g["zh_cer"] - b["zh_cer"], "zh_retention": g["zh_retention_vs_B0"],
             "en_retention": g["en_retention_vs_B0"], "outside_harm_rate": g["outside_harm_vs_B0"], "poi_corruption_rate": g["poi_corruption_rate"],
             "added_caps": g["added_caps_vs_B0"], "cap_count_delta": g["caps"] - b["caps"], "new_severe_truncations": g["severe_vs_B0"]}
    saf = safety(point, c)
    resc = rescue(F["B0_FORCED"]["zh"], F["A2"]["zh"], F["G"]["zh"])
    ben = benefit(F["B0_FORCED"]["poi"], F["A2"]["poi"], F["G"]["poi"], F["G"]["pier"], F["A2"]["pier"], F["G"]["mer"], F["A2"]["mer"],
                  F["G"]["mixed"], F["A2"]["mixed"], c)
    r = {u: CNT[u]["A2"][4] - CNT[u]["G"][4] for u in ids}
    br = breadth(r, D, set(P["NOVEL76"]), c)
    changed = sorted({D[u] for u in ids if rowmap[u]["G1"]["tokens"] != X[u]["analysis"]["A2"]["tokens"]
                      or rowmap[u]["G1"]["terminated"] != X[u]["analysis"]["A2"]["terminated"]})
    lo = {"all20": lodo(r, D, sorted(set(D.values()))), "changed": lodo(r, D, changed), "changed_dialogues": changed}
    conc = {}
    for k in PARTS:
        us = P[k]
        pp = {u: max(r[u], 0) for u in us}
        Rp = sum(pp.values())
        pd_ = {}
        for u in us:
            pd_[D[u]] = pd_.get(D[u], 0) + pp[u]
        conc[k] = {"R_plus": Rp, "R_minus": sum(max(-r[u], 0) for u in us), "R_net": sum(r[u] for u in us),
                   "max_row": max(pp, key=pp.get) if Rp else None, "max_row_share": (max(pp.values()) / Rp) if Rp else None,
                   "max_dialogue": max(pd_, key=pd_.get) if Rp else None, "max_dialogue_share": (max(pd_.values()) / Rp) if Rp else None,
                   "rescue_rows": sum(r[u] >= 1 for u in us), "harm_rows": sum(r[u] <= -1 for u in us)}
    Darr = [D[u] for u in ids]
    idxs = draws(Darr, reps=c["bootstrap"]["reps"], seed=c["bootstrap"]["seed"])
    CA = {s: np.array([CNT[u][s] for u in ids], dtype=float) for s in SYSTEMS}
    boot = {}
    for base in ("A2", "B0_FORCED"):
        boot[f"G_minus_{base}"] = {n: ratio_delta(CA["G"], CA[base], a_, q_, idxs) for n, (a_, q_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5)}.items()}
        boot[f"G_minus_{base}"]["poi_errors"] = count_delta_ci(CA["G"], CA[base], 0, idxs)
    valid = all(integ.values())
    label = decide(valid, saf["pass"], resc["pass"], ben["BENEFIT_RETENTION_PASS"], br["BREADTH_PASS"])
    pm = {e["utterance_id"]: e for e in prim["per_row"]}
    per = []
    for u in ids:
        per.append({"utterance_id": u, "dialogue_id": D[u], "partitions": X[u]["audit"]["partitions"], "mode": pm[u]["mode"], "winner": pm[u]["winner"],
                    "counts": {s: CNT[u][s] for s in SY}, "terminated": T[u], "length": L[u], "r_ZH_A2_minus_G": r[u],
                    "dPOI_G_minus_A2": CNT[u]["G"][0] - CNT[u]["A2"][0], "dMixed_G_minus_A2": CNT[u]["G"][2] - CNT[u]["A2"][2],
                    "dZH_G_minus_B0": CNT[u]["G"][4] - CNT[u]["B0_FORCED"][4]})
    t0w = [e for e in per if e["winner"] == "theta0"]
    theta0_desc = {"rows": len(t0w), "improving": {k: sum(e[f"d{k}_G_minus_A2"] < 0 for e in t0w) for k in ("POI", "Mixed")},
                   "worsening": {k: sum(e[f"d{k}_G_minus_A2"] > 0 for e in t0w) for k in ("POI", "Mixed")},
                   "ZH_improving": sum(e["r_ZH_A2_minus_G"] > 0 for e in t0w), "ZH_worsening": sum(e["r_ZH_A2_minus_G"] < 0 for e in t0w)}
    bset = {b_["utterance_id"]: b_ for b_ in prim["EOS_boundary"]["rows"]}
    boundary_post = [{"utterance_id": u, "winner_action": bset[u]["winner_action"], "dPOI_G_minus_A2": CNT[u]["G"][0] - CNT[u]["A2"][0],
                      "dZH_G_minus_A2": CNT[u]["G"][4] - CNT[u]["A2"][4], "dMixed_G_minus_A2": CNT[u]["G"][2] - CNT[u]["A2"][2],
                      "new_severe_truncation_vs_B0": severe_truncation(L[u]["B0_FORCED"], L[u]["G"], T[u]["G"])} for u in bset]
    top = sorted([e for e in per if e["r_ZH_A2_minus_G"] > 0], key=lambda e: -e["r_ZH_A2_minus_G"])
    harms = [e for e in per if e["r_ZH_A2_minus_G"] < 0 or e["dPOI_G_minus_A2"] > 0 or e["dMixed_G_minus_A2"] > 0]
    auto_delta = {k: F["G"][k] - F["B0_AUTO"][k] for k in ("pier", "mer", "en_wer", "zh_cer", "zh", "poi", "mixed")}
    return {"schema": "p2_path4_secondary_v1", "contract_revision": c["contract_revision"], "output_seal_hash": seal["seal_hash"], "integrity": integ,
            "valid": valid, "metrics": M, "safety_point": point, "safety": saf, "ABSOLUTE_SAFETY_PASS": saf["pass"], "aggregate_rescue": resc,
            "AGGREGATE_RESCUE_PASS": resc["pass"], "benefit": ben, "BENEFIT_RETENTION_PASS": ben["BENEFIT_RETENTION_PASS"], "breadth": br,
            "BREADTH_PASS": br["BREADTH_PASS"], "lodo": lo, "concentration": conc, "bootstrap": boot, "G_minus_AUTO": auto_delta,
            "theta0_winner_descriptive": theta0_desc, "boundary_post_seal": boundary_post, "label": label, "top_rescues": top[:10],
            "new_harms": harms, "per_row": per}


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
        print(json.dumps({k: {kk: v[kk] for kk in ("n", "triggers", "content_G1", "EOS_boundary", "theta0_winners", "A2_winners", "G_changed_vs_A2",
                                                    "trigger_dialogues", "margin")} for k, v in res.get("summaries", {}).items()}, indent=1))
        print(json.dumps(res.get("EOS_boundary"), indent=1))
    else:
        print(json.dumps({k: res[k] for k in ("label", "valid", "integrity", "safety", "aggregate_rescue", "benefit")}, indent=1))
        print(json.dumps({k: v for k, v in res["breadth"].items() if k != "P_d"}, indent=1))
        print(json.dumps({s: {k: v[k] for k in ("zh", "poi", "mixed", "pier", "mer", "en_wer", "zh_cer", "caps")} for s, v in res["metrics"]["FULL100"].items()}, indent=1))


if __name__ == "__main__":
    main()
