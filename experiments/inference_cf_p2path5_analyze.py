#!/usr/bin/env python
"""P2-PATH5 analysis (frozen contract configs/inference_cf/p2_path5.json).

primary   (reference-free) validity over all 300 rows (FIXED100 barrier, A2 episodes, G1A dispatch/abstention identities,
          ownership, compute), NEW200 opportunity counts and the frozen opportunity gate, per-row opportunity classes.
terminal  (reference-free) if the gate fires: P2_PATH5_OPPORTUNITY_SPARSE record (requires committed seal + primary audit PASS);
          NEW200 references are never opened.
secondary (references; committed seal + committed primary audit PASS + open gate) NEW200 B0/AUTO/A2/G1A metrics, absolute
          safety, strong A2 retention, rescue/breadth, frozen precedence, LODO, bootstrap, opportunity-outcome map,
          FULL300 secondary aggregate, all rescue/harm rows.
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

CONFIG = "configs/inference_cf/p2_path5.json"
BASE = "results/inference_cf/p2path5"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
SYSTEMS = ("B0_FORCED", "B0_AUTO", "A2", "G1A")
LABELS = ("P2_PATH5_INVALID", "P2_PATH5_OPPORTUNITY_SPARSE", "P2_PATH5_SEQUENCE_DAMAGE", "P2_PATH5_SAFE_NO_ADDED_VALUE",
          "P2_PATH5_OVERCONSERVATIVE", "P2_PATH5_SIGNAL_CONCENTRATED", "P2_PATH5_G1A_SUPPORTED")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- frozen outcome rules (pure) -------------------------------------------------------------------------------

def opportunity_gate(n_changed: int, n_changed_dlg: int, c: dict) -> dict:
    g = c["opportunity_gate"]
    open_ = n_changed >= g["N_G1A_CHANGED_min"] and n_changed_dlg >= g["N_CHANGED_DLG_min"]
    return {"N_G1A_CHANGED": n_changed, "N_CHANGED_DLG": n_changed_dlg, "open": open_, "result": "OPEN_REFERENCES" if open_ else "OPPORTUNITY_SPARSE"}


def safety(p: dict, c: dict) -> dict:
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


def retention(poi_b0: int, poi_a2: int, poi_g: int, pier_g: float, pier_a2: float, mer_g: float, mer_a2: float, c: dict) -> dict:
    Rc = c["retention"]
    G = Rc["float_guard"]
    I_A2, I_G = poi_b0 - poi_a2, poi_b0 - poi_g
    ret = I_G / I_A2 if I_A2 > 0 else None
    strong = (ret >= Rc["strong_min"] - G) if I_A2 > 0 else (poi_g <= poi_a2)
    crit = {"strong_retention": bool(strong), "PIER": pier_g - pier_a2 <= Rc["PIER_G_minus_A2_max"] + G, "MER": mer_g - mer_a2 <= Rc["MER_G_minus_A2_max"] + G}
    return {"I_A2": I_A2, "I_G": I_G, "retention": ret, "old_0.75_rule_descriptive": (ret >= 0.75 - G) if ret is not None else None,
            "PIER_G_minus_A2": pier_g - pier_a2, "MER_G_minus_A2": mer_g - mer_a2, "criteria": crit, "pass": all(crit.values())}


def breadth(r: dict, dialogue: dict, c: dict) -> dict:
    Bc = c["breadth"]
    p = {u: max(x, 0) for u, x in r.items()}
    Rp, Rm, Rn = sum(p.values()), sum(max(-x, 0) for x in r.values()), sum(r.values())
    Pd = {}
    for u, x in p.items():
        Pd[dialogue[u]] = Pd.get(dialogue[u], 0) + x
    n_utt = sum(x >= 1 for x in r.values())
    n_dlg = sum(v > 0 for v in Pd.values())
    C_utt = max(p.values()) / Rp if Rp > 0 else Bc["C_if_R_plus_0"]
    C_dlg = max(Pd.values()) / Rp if Rp > 0 else Bc["C_if_R_plus_0"]
    crit = {"rescue_utterances": n_utt >= Bc["rescue_utterances_min"], "rescue_dialogues": n_dlg >= Bc["rescue_dialogues_min"],
            "C_utt": C_utt <= Bc["C_utt_max"] + 1e-12, "C_dlg": C_dlg <= Bc["C_dlg_max"] + 1e-12}
    return {"R_plus": Rp, "R_minus": Rm, "R_net": Rn, "N_rescue_utt": n_utt, "N_rescue_dlg": n_dlg, "C_utt": C_utt, "C_dlg": C_dlg, "P_d": Pd,
            "criteria": crit, "pass": all(crit.values())}


def lodo(r: dict, dialogue: dict) -> dict:
    keys = sorted(set(dialogue[u] for u in r))
    vals = {d: sum(x for u, x in r.items() if dialogue[u] != d) for d in keys}
    v = list(vals.values())
    pos = sum(x > 0 for x in v)
    return {"dialogues": len(v), "values": vals, "min": min(v), "median": float(np.median(v)), "max": max(v), "count_positive": pos,
            "proportion_positive": pos / len(v)}


def decide(valid: bool, gate_open: bool, safety_pass: bool, r_net: int, retention_pass: bool, breadth_pass: bool) -> str:
    if not valid:
        return "P2_PATH5_INVALID"
    if not gate_open:
        return "P2_PATH5_OPPORTUNITY_SPARSE"
    if not safety_pass:
        return "P2_PATH5_SEQUENCE_DAMAGE"
    if r_net <= 0:
        return "P2_PATH5_SAFE_NO_ADDED_VALUE"
    if not retention_pass:
        return "P2_PATH5_OVERCONSERVATIVE"
    if not breadth_pass:
        return "P2_PATH5_SIGNAL_CONCENTRATED"
    return "P2_PATH5_G1A_SUPPORTED"


def opp_class(mode: str, winner, changed: bool) -> str:
    if mode == "NO_TRIGGER":
        return "NO_TRIGGER"
    if mode == "EOS_ABSTAIN":
        return "EOS_ABSTAIN"
    if winner != "theta0":
        return "CONTENT_A2_WIN"
    return "CONTENT_THETA0_WIN_CHANGED" if changed else "CONTENT_THETA0_WIN_NO_CHANGE"


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
    trig = [e for e in es if e["mode"] != "NO_TRIGGER"]
    ch = [e for e in es if e["G1A_changed"]]
    return {"n": len(es), "N_DISAGREE": len(trig), "N_CONTENT_ELIGIBLE": sum(e["mode"] == "CONTENT_G1" for e in es),
            "N_EOS_ABSTAIN": sum(e["mode"] == "EOS_ABSTAIN" for e in es), "N_NO_TRIGGER": sum(e["mode"] == "NO_TRIGGER" for e in es),
            "abstain_orientation": {o: sum(e.get("orientation") == o for e in es) for o in ("theta0_EOS", "A2_EOS")},
            "theta0_winners": sum(e["winner"] == "theta0" for e in es), "A2_winners": sum(e["mode"] == "CONTENT_G1" and e["winner"] != "theta0" for e in es),
            "N_G1A_CHANGED": len(ch), "N_CHANGED_DLG": len({e["dialogue_id"] for e in ch}), "changed_rows": [e["utterance_id"] for e in ch],
            "disagreement_dialogues": len({e["dialogue_id"] for e in trig}),
            "classes": {k: sum(e["class"] == k for e in es) for k in ("NO_TRIGGER", "EOS_ABSTAIN", "CONTENT_A2_WIN", "CONTENT_THETA0_WIN_NO_CHANGE",
                                                                      "CONTENT_THETA0_WIN_CHANGED")},
            "content_margins": sorted(e["margin"] for e in es if e["mode"] == "CONTENT_G1")}


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    rows, plan, rt = d["rows"], d["plan"], d["runtime"]
    n = len(plan["rows"])
    integ = {"all_rows_ok": len(rows) == n == 300 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
             "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
             and bool(rt.get("model_grads_none")), "fixed100_barrier": bool((rt.get("barrier") or {}).get("pass")) and (rt.get("barrier") or {}).get("exact") == 100}
    res = {"schema": "p2_path5_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "references_used": False, "integrity": integ}
    if not integ["all_rows_ok"]:
        res.update(valid=False, rows_ok=sum(r.get("status") == "ok" for r in rows), opportunity_gate=None)
        return res
    ok = {k: True for k in ("identity", "fixed100_reconstruction", "a2_episodes", "ownership", "dispatch_types", "trigger_relation",
                            "abstain_identity", "no_trigger_identity", "a2_winner_identity", "g_forced_winner", "live_audit")}
    per = []
    for i, (r, x) in enumerate(zip(rows, plan["rows"])):
        a, au = x["analysis"], x["audit"]
        u = a["utterance_id"]
        ok["identity"] &= r["identity"] == u and r["full_index"] == i and r["partition"] == a["partition"] and r["dialogue_id"] == a["dialogue_id"]
        if a["partition"] == "FIXED100":
            ok["fixed100_reconstruction"] &= (r["theta0_equals_B0"] and r["A2_equals_sealed"] and r["masters_equal_checkpoint_fp32"]
                                              and r["effective_equals_checkpoint_bf16"] and r["barrier_exact"] and r["barrier_mismatches"] == [])
        lg = r["A2_log"]
        ok["a2_episodes"] &= (lg["steps"] == 2 and lg["loss_evaluations"] == 3 and all(lg["finite"]) and all(math.isfinite(v) for v in lg["losses"] + lg["grad_l2"])
                              and lg["start_hash"] == lg["end_hash"] == rt["theta0_ln_hash"] == r["reset_hash_after"] and r["reset_ok_episode"]
                              and r["reset_ok_controller"])
        det, dec, G, A = r["detector"], r["decision"], r["G1"], r["A2_free"]
        ok["ownership"] &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        ok["trigger_relation"] &= r["trigger_relation_ok"]
        mode = r["mode"]
        e = {"utterance_id": u, "dialogue_id": a["dialogue_id"], "partition": a["partition"], "mode": mode, "k": det.get("k"), "winner": None,
             "margin": None, "G1A_changed": r["G1A_changed_vs_A2"]}
        if det["trigger"]:
            c0, c2 = det["argmax_theta0"], det["argmax_A4"]
            is_b = (c0 == 50257) != (c2 == 50257)
            ok["dispatch_types"] &= mode == ("EOS_ABSTAIN" if is_b else "CONTENT_G1")
            if mode == "EOS_ABSTAIN":
                ok["abstain_identity"] &= dec is None and G["tokens"] == A["tokens"] and G["terminated"] == A["terminated"] and not r["G1A_changed_vs_A2"]
                e["orientation"] = r["abstain"]["orientation"]
            else:
                w = dec["b0"] if dec["winner"] == "theta0" else dec["b4"]
                ok["ownership"] &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"]
                                       for v in r["score_paths"].values()) and G["state_locked"]
                ok["g_forced_winner"] &= G["forced"] == [w["tokens"][0]] and G["decision_hash"] == dec["decision_hash"] and G["trace"]["prefix_ok"]
                if dec["winner"] != "theta0":
                    ok["a2_winner_identity"] &= G["tokens"] == A["tokens"] and G["terminated"] == A["terminated"]
                e.update(winner="theta0" if dec["winner"] == "theta0" else "A2", margin=dec["margin"])
        else:
            ok["no_trigger_identity"] &= mode == "NO_TRIGGER" and dec is None and G["tokens"] == A["tokens"] and G["terminated"] == A["terminated"]
        e["class"] = opp_class(mode, e["winner"], r["G1A_changed_vs_A2"])
        th, AU = r["theta0"], a["AUTO"]
        sB, sA, sG, sU = stream(th["tokens"], th["terminated"]), stream(A["tokens"], A["terminated"]), stream(G["tokens"], G["terminated"]), \
            stream(AU["tokens"], AU["terminated"])
        e.update(d_G_A2=edit_distance(sG, sA), d_G_B0=edit_distance(sG, sB), d_G_AUTO=edit_distance(sG, sU), d_A2_B0=edit_distance(sA, sB),
                 B0_length=len(th["tokens"]), A2_length=len(A["tokens"]), G_length=len(G["tokens"]),
                 terminated={"B0": th["terminated"], "A2": A["terminated"], "G1A": G["terminated"]},
                 A2_severe_vs_B0=severe_truncation(len(th["tokens"]), len(A["tokens"]), A["terminated"]),
                 G_severe_vs_B0=severe_truncation(len(th["tokens"]), len(G["tokens"]), G["terminated"]))
        per.append(e)
    la = rows[plan["ids"].index(plan["execution_order"][0])].get("live_audit") or {}
    ok["live_audit"] = bool(la.get("pass")) and la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    nc = sum(e["mode"] == "CONTENT_G1" for e in per)
    ne = sum(e["mode"] == "EOS_ABSTAIN" for e in per)
    B = c["compute"]
    ok["compute"] = (cnt["A2"]["optimizer_steps"] == B["optimizer_steps"] == 600
                     and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] <= B["backwards_max_including_live_check"]
                     and cnt["logical_guard_events"] == cnt["detector_runs"] == 300 and cnt["triggers"] == nc + ne and cnt["content_events"] == nc
                     and cnt["eos_abstentions"] == ne and cnt["detector_replays"] == nc and cnt["rollouts"] == 2 * nc and cnt["score_paths"] == 4 * nc
                     and cnt["G1_decodes"] == nc and cnt["boundary_scores"] == 0 and cnt["G2_decodes"] == 0 and cnt["A4_calls"] == 0
                     and cnt["theta0_decodes"] == 300 and cnt["encoder_passes"] == 300 and cnt["rows_entered_after_barrier"] == 200)
    ok["wall_clock"] = rt["elapsed_sec"] <= 60 * B["hard_minutes"]
    integ.update(ok)
    valid = all(integ.values())
    P = plan["partitions"]
    summ = {k: summarize(per, P[k]) for k in ("NEW200", "FIXED100", "FULL300")}
    gate = opportunity_gate(summ["NEW200"]["N_G1A_CHANGED"], summ["NEW200"]["N_CHANGED_DLG"], c)
    res.update(valid=valid, opportunity_gate=gate, summaries=summ, per_row=per,
               runtime={k: rt.get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "fixed_phase1_sec", "fixed_phase2_sec", "new_sec", "peak_alloc",
                                               "peak_reserved", "counters", "gpu", "barrier")})
    return res


def terminal_sparse(run_rel: str) -> dict:
    """Reference-free terminal record when the gate fires (no evaluator import)."""
    if not committed(SEAL):
        raise PermissionError("seal must be committed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed outputs changed")
    if not committed(PRIMARY_AUDIT):
        raise PermissionError("primary audit must be committed")
    pa = json.loads((ROOT / PRIMARY_AUDIT).read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    valid = bool(prim["valid"]) and pa["verdict"] == "P2_PATH5_AUDIT: PASS"
    gate = prim["opportunity_gate"]
    if valid and gate["open"]:
        raise ValueError("opportunity gate is open; terminal_sparse does not apply")
    label = decide(valid, bool(gate and gate["open"]), False, 0, False, False)
    return {"schema": "p2_path5_terminal_v1", "label": label, "references_opened": False, "opportunity_gate": gate,
            "output_seal_hash": seal["seal_hash"], "primary_audit_verdict": pa["verdict"]}


# ---- secondary -----------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import draws, outside, ratio_delta, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("PATH5 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed PATH5 outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != "P2_PATH5_AUDIT: PASS":
        raise PermissionError("primary-phase audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    if not (prim["valid"] and prim["opportunity_gate"]["open"]):
        raise PermissionError("opportunity gate closed or primary invalid: NEW200 references must not be opened")
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    refs = load_references()
    plan = d["plan"]
    P = plan["partitions"]
    X = {x["analysis"]["utterance_id"]: x for x in plan["rows"]}
    rowmap = {r["identity"]: r for r in d["rows"]}
    ids = plan["ids"]
    R = {u: refs[u]["reference"] for u in ids}
    D = {u: refs[u]["dialogue_id"] for u in ids}
    integ = {"primary_valid": True, "dialogues_match": all(D[u] == X[u]["analysis"]["dialogue_id"] for u in ids)}
    H, T, L = {}, {}, {}
    for u in ids:
        r, a = rowmap[u], X[u]["analysis"]
        H[u] = {"B0_FORCED": r["theta0"]["text"], "B0_AUTO": a["AUTO"]["text"], "A2": r["A2_free"]["text"], "G1A": r["G1"]["text"]}
        T[u] = {"B0_FORCED": r["theta0"]["terminated"], "B0_AUTO": a["AUTO"]["terminated"], "A2": r["A2_free"]["terminated"], "G1A": r["G1"]["terminated"]}
        L[u] = {"B0_FORCED": len(r["theta0"]["tokens"]), "B0_AUTO": len(a["AUTO"]["tokens"]), "A2": len(r["A2_free"]["tokens"]), "G1A": len(r["G1"]["tokens"])}
    CNT = {u: {s: utt_counts(R[u], H[u][s]) for s in SYSTEMS} for u in ids}

    def agg(us):
        Rs, Bs = [R[u] for u in us], [H[u]["B0_FORCED"] for u in us]
        out = {}
        for s in SYSTEMS:
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
                outs = [outside(a_, b, h) for a_, b, h in zip(Rs, Bs, Hs)]
                ob = sum(o["baseline_correct_outside"] for o in outs)
                e.update(corrections_vs_B0=cc["corrections"], corruptions_vs_B0=cc["corruptions"], poi_corruption_rate=cc["corruption_rate"],
                         zh_retention_vs_B0=matrix_zh_retention(Rs, Bs, Hs)["rate"], en_retention_vs_B0=embedded_en_retention(Rs, Bs, Hs)["rate"],
                         outside_harm_vs_B0=(sum(o["outside_harm"] for o in outs) / ob) if ob else None)
            out[s] = e
        return out
    M = {k: agg(P[k]) for k in ("NEW200", "FULL300", "FIXED100")}
    N = M["NEW200"]
    integ["populations_nonempty"] = all(N["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(N[s][x] is not None and math.isfinite(N[s][x]) for s in SYSTEMS for x in ("pier", "mer", "en_wer", "zh_cer"))
    integ["poi_count_consistent"] = all(N[s]["poi"] == N[s]["num_poi_errors"] for s in SYSTEMS)
    g, b = N["G1A"], N["B0_FORCED"]
    point = {"mer_increase": g["mer"] - b["mer"], "zh_cer_increase": g["zh_cer"] - b["zh_cer"], "zh_retention": g["zh_retention_vs_B0"],
             "en_retention": g["en_retention_vs_B0"], "outside_harm_rate": g["outside_harm_vs_B0"], "poi_corruption_rate": g["poi_corruption_rate"],
             "added_caps": g["added_caps_vs_B0"], "cap_count_delta": g["caps"] - b["caps"], "new_severe_truncations": g["severe_vs_B0"]}
    saf = safety(point, c)
    ret = retention(N["B0_FORCED"]["poi"], N["A2"]["poi"], N["G1A"]["poi"], N["G1A"]["pier"], N["A2"]["pier"], N["G1A"]["mer"], N["A2"]["mer"], c)
    new = P["NEW200"]
    r = {u: CNT[u]["A2"][4] - CNT[u]["G1A"][4] for u in new}
    br = breadth(r, D, c)
    valid = all(integ.values())
    label = decide(valid, True, saf["pass"], br["R_net"], ret["pass"], br["pass"])
    lo = lodo(r, D)
    idxs = draws([D[u] for u in new], reps=c["descriptive_only"]["bootstrap"]["reps"], seed=c["descriptive_only"]["bootstrap"]["seed"])
    CA = {s: np.array([CNT[u][s] for u in new], dtype=float) for s in SYSTEMS}
    boot = {}
    for base in ("A2", "B0_FORCED"):
        boot[f"G1A_minus_{base}"] = {nm: ratio_delta(CA["G1A"], CA[base], a_, q_, idxs) for nm, (a_, q_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5)}.items()}
        boot[f"G1A_minus_{base}"]["poi_errors"] = count_delta_ci(CA["G1A"], CA[base], 0, idxs)
    pm = {e["utterance_id"]: e for e in prim["per_row"]}
    per = [{"utterance_id": u, "dialogue_id": D[u], "class": pm[u]["class"], "mode": pm[u]["mode"], "winner": pm[u]["winner"], "r_ZH": r[u],
            "dPOI_G_minus_A2": CNT[u]["G1A"][0] - CNT[u]["A2"][0], "dMixed_G_minus_A2": CNT[u]["G1A"][2] - CNT[u]["A2"][2],
            "dZH_A2_minus_B0": CNT[u]["A2"][4] - CNT[u]["B0_FORCED"][4], "dPOI_A2_minus_B0": CNT[u]["A2"][0] - CNT[u]["B0_FORCED"][0],
            "counts": CNT[u], "terminated": T[u], "length": L[u]} for u in new]
    omap = {}
    for k in ("NO_TRIGGER", "EOS_ABSTAIN", "CONTENT_A2_WIN", "CONTENT_THETA0_WIN_NO_CHANGE", "CONTENT_THETA0_WIN_CHANGED"):
        es = [e for e in per if e["class"] == k]
        omap[k] = {"rows": len(es), "dZH_G_minus_A2": -sum(e["r_ZH"] for e in es), "dPOI_G_minus_A2": sum(e["dPOI_G_minus_A2"] for e in es),
                   "dMixed_G_minus_A2": sum(e["dMixed_G_minus_A2"] for e in es), "dZH_A2_minus_B0": sum(e["dZH_A2_minus_B0"] for e in es),
                   "dPOI_A2_minus_B0": sum(e["dPOI_A2_minus_B0"] for e in es)}
    rescue_rows = [{k: e[k] for k in ("utterance_id", "dialogue_id", "class", "r_ZH", "dPOI_G_minus_A2", "dMixed_G_minus_A2")} for e in per if e["r_ZH"] > 0]
    harm_rows = [{k: e[k] for k in ("utterance_id", "dialogue_id", "class", "r_ZH", "dPOI_G_minus_A2", "dMixed_G_minus_A2")} for e in per
                 if e["r_ZH"] < 0 or e["dPOI_G_minus_A2"] > 0 or e["dMixed_G_minus_A2"] > 0]
    auto_delta = {k: N["G1A"][k] - N["B0_AUTO"][k] for k in ("pier", "mer", "en_wer", "zh_cer", "zh", "poi", "mixed")}
    return {"schema": "p2_path5_secondary_v1", "output_seal_hash": seal["seal_hash"], "integrity": integ, "valid": valid, "metrics": M,
            "safety_point": point, "safety": saf, "ABSOLUTE_SAFETY_PASS": saf["pass"], "retention": ret, "breadth": br, "label": label,
            "lodo": lo, "bootstrap": boot, "opportunity_outcome_map": omap, "rescue_rows": rescue_rows, "harm_rows": harm_rows,
            "G1A_minus_AUTO": auto_delta, "per_row_NEW200": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("primary", "secondary", "terminal"):
        p = sub.add_parser(nm)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    fn = {"primary": primary, "secondary": secondary, "terminal": terminal_sparse}[args.cmd]
    res = jsonable(fn(args.run))
    atomic_json(out, res)
    if args.cmd == "primary":
        print(json.dumps({k: res.get(k) for k in ("valid", "integrity", "opportunity_gate")}, indent=1))
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "content_margins"} for k, v in res.get("summaries", {}).items()}, indent=1))
    elif args.cmd == "terminal":
        print(json.dumps(res, indent=1))
    else:
        print(json.dumps({k: res[k] for k in ("label", "valid", "integrity", "safety", "retention")}, indent=1))
        print(json.dumps({k: v for k, v in res["breadth"].items() if k != "P_d"}, indent=1))
        print(json.dumps({s: {k: v[k] for k in ("zh", "poi", "mixed", "pier", "mer", "en_wer", "zh_cer", "caps")} for s, v in res["metrics"]["NEW200"].items()}, indent=1))


if __name__ == "__main__":
    main()
