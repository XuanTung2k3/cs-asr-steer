#!/usr/bin/env python
"""TTLS-R1 post-seal evaluation (frozen contract configs/inference_cf/ttls_r1.json). Written before the run.

References are opened only after the committed output seal is verified. Free-decoding transcripts only; canonical
csasr.evaluation metrics (PIER, MER, EN-WER, ZH-CER, POI transitions, retention) and the P2-SEQ outside-POI / bootstrap
helpers. The exploratory label follows the frozen rule exactly; the comparison with AUTO (B1) is descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

CONFIG = "configs/inference_cf/ttls_r1.json"
BASE = "results/inference_cf/ttls_r1"
SEAL = f"{BASE}/output_seal.json"
ARMS = ("T1", "T2", "T2A", "T3", "T4", "T5", "T6")
SYSTEMS = ("B0", "B1", "B2") + ARMS + ("CD", "ACSUB")
TTLS_ARMS = ("T1", "T2", "T4", "T6")
MATCH = {"T1": "B2", "T2": "T2A", "T4": "T3", "T6": "T5"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


# ---- load (reference-free) -------------------------------------------------------------------------------------------

def load(run_rel: str) -> dict:
    run = ROOT / run_rel
    man = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows = []
    for i, u in enumerate(man["ids"]):
        p = run / "rows" / f"{i:03d}.json"
        r = json.loads(p.read_text()) if p.exists() else {"status": "missing", "identity": u}
        if r.get("status") == "ok" and (r["identity"] != u or r["manifest_hash"] != man["manifest_hash"]):
            raise ValueError(f"row {i}")
        rows.append(r)
    return {"manifest": man, "plan": plan, "rows": rows, "runtime": json.loads((run / "runtime.json").read_text())}


def integrity(d: dict) -> dict:
    rows, rt = d["rows"], d["runtime"]
    c = {"all_100_ok": len(rows) == 100 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
         "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
         and bool(rt.get("model_grads_none"))}
    if not c["all_100_ok"]:
        return c
    c["B0_reproduced"] = all(r["B0"]["tokens_equal_S0"] and r["B0"]["terminated_equal_S0"] for r in rows)
    c["B1_replayed"] = all(r["B1"]["replay_text_equal"] for r in rows)
    c["B2_reproduced"] = all(r["B2"]["reproduces_historical_A2"] for r in rows)
    v = rows[-1]["validated"]
    c["clean_identity"] = bool(v["clean_identity"] and v["clean_identity"]["empty_mask_decode_equals_B0"])
    c["grad_flow"] = all(v[k] is None or v[k]["pass"] for k in ("ttls_grad_flow_CE", "ttls_grad_flow_AC", "ln_grad_flow_AC")) \
        and v["ttls_grad_flow_CE"] is not None
    c["arm_resets"] = all(r[a]["status"] == "abstain" or r[a]["reset_ok"] for r in rows for a in ARMS)
    c["two_steps"] = all(r[a]["status"] == "abstain" or (len(r[a]["log"]["losses"]) == 3 and len(r[a]["log"]["grad_l2"]) == 2
                                                         and all(r[a]["log"]["finite"])) for r in rows for a in ARMS)
    c["ttls_budget"] = all(r[a]["status"] == "abstain" or r[a]["log"]["master_delta_l2"] <= 1.1260757575454359 * (1 + 1e-6)
                           for r in rows for a in ("T1", "T2", "T4", "T6"))
    c["abstain_is_B0"] = all(r[a]["equal_B0"] for r in rows for a in ("T3", "T4", "T5", "T6", "ACSUB") if r[a].get("noop"))
    return c


def systems(d: dict) -> tuple[dict, dict, dict]:
    P = {p["utterance_id"]: p for p in d["plan"]["rows"]}
    ids = d["manifest"]["ids"]
    H, Tk, Te = {}, {}, {}
    H["B0"] = [P[u]["y_B_text"] for u in ids]
    Tk["B0"] = [P[u]["y_B"] for u in ids]
    Te["B0"] = [P[u]["y_B_terminated"] for u in ids]
    H["B1"] = [P[u]["y_A_text"] for u in ids]
    Tk["B1"] = [P[u]["y_A"] for u in ids]
    Te["B1"] = [P[u]["y_A_terminated"] for u in ids]
    H["B2"] = [P[u]["A2"]["text"] for u in ids]
    Tk["B2"] = [P[u]["A2"]["tokens"] for u in ids]
    Te["B2"] = [P[u]["A2"]["terminated"] for u in ids]
    for s in ARMS + ("CD", "ACSUB"):
        H[s] = [r[s]["text"] for r in d["rows"]]
        Tk[s] = [r[s]["tokens"] for r in d["rows"]]
        Te[s] = [r[s]["terminated"] for r in d["rows"]]
    return H, Tk, Te


# ---- reference-based accounting (behind the seal) -------------------------------------------------------------------

def poi_transitions(R, B, M, eos_rows, dialogues, ids, genuine) -> dict:
    from csasr.evaluation.normalization import normalize_text
    from csasr.evaluation.pier import evaluate_pois, reference_pois
    out = {"corrections": 0, "corruptions": 0, "genuine_substitution_corrections": 0, "deletion_type_corrections": 0,
           "eos_recovery_row_corrections": 0, "baseline_incorrect": 0, "baseline_correct": 0, "corruption_categories": Counter(),
           "correction_baseline_categories": Counter(), "genuine_dialogues": Counter(), "events": []}
    for j, (ref, b, m) in enumerate(zip(R, B, M)):
        if b == m:
            idxs = [i for i, _ in reference_pois(normalize_text(ref))]
            for x in evaluate_pois(ref, b, idxs):
                out["baseline_correct" if x.correct else "baseline_incorrect"] += 1
            continue
        idxs = [i for i, _ in reference_pois(normalize_text(ref))]
        br = {x.poi_index: x for x in evaluate_pois(ref, b, idxs)}
        mr = {x.poi_index: x for x in evaluate_pois(ref, m, idxs)}
        for i in idxs:
            if i not in br or i not in mr:
                continue
            out["baseline_correct" if br[i].correct else "baseline_incorrect"] += 1
            if not br[i].correct and mr[i].correct:
                out["corrections"] += 1
                cat = br[i].category
                out["correction_baseline_categories"][cat] += 1
                kind = "genuine_substitution" if cat in genuine else "deletion_type"
                if eos_rows[j]:
                    out["eos_recovery_row_corrections"] += 1
                if kind == "genuine_substitution":
                    out["genuine_substitution_corrections"] += 1
                    out["genuine_dialogues"][dialogues[j]] += 1
                else:
                    out["deletion_type_corrections"] += 1
                out["events"].append({"utterance_id": ids[j], "dialogue_id": dialogues[j], "type": "correction", "kind": kind,
                                      "poi": br[i].surface, "B0": br[i].hyp_surface, "B0_category": cat, "system": mr[i].hyp_surface,
                                      "eos_recovery_row": bool(eos_rows[j])})
            elif br[i].correct and not mr[i].correct:
                out["corruptions"] += 1
                out["corruption_categories"][mr[i].category] += 1
                out["events"].append({"utterance_id": ids[j], "dialogue_id": dialogues[j], "type": "corruption", "poi": br[i].surface,
                                      "system": mr[i].hyp_surface, "system_category": mr[i].category})
    out["net"] = out["corrections"] - out["corruptions"]
    out["genuine_dialogue_count"] = len(out["genuine_dialogues"])
    out["corruption_rate"] = out["corruptions"] / out["baseline_correct"] if out["baseline_correct"] else None
    for k in ("corruption_categories", "correction_baseline_categories", "genuine_dialogues"):
        out[k] = dict(out[k])
    return out


def zh_changes(R, B, M) -> dict:
    from csasr.evaluation.mer import ZH
    from csasr.evaluation.normalization import normalize_text, segment_units, tag_unit
    from csasr.evaluation.pier import unit_status
    new = rep = base_ok = 0
    for ref, b, m in zip(R, B, M):
        units = segment_units(normalize_text(ref))
        zh = [i for i, u in enumerate(units) if tag_unit(u) == ZH]
        sb = unit_status(ref, b)
        sm = sb if b == m else unit_status(ref, m)
        for i in zh:
            ok_b, ok_m = bool(sb.get(i, (False,))[0]), bool(sm.get(i, (False,))[0])
            base_ok += ok_b
            new += ok_b and not ok_m
            rep += (not ok_b) and ok_m
    return {"new_zh_errors": int(new), "zh_repairs": int(rep), "baseline_correct_zh_units": int(base_ok),
            "zh_unit_retention_all": (1 - new / base_ok) if base_ok else None}


def row_events(Tk, Te, X) -> dict:
    from csasr.inference_cf.episodic_tta import severe_truncation
    b, bt, m, mt = Tk["B0"], Te["B0"], Tk[X], Te[X]
    pre = lambda a, c: len(a) < len(c) and list(c[:len(a)]) == list(a)
    ch = [x != y for x, y in zip(b, m)]
    eos_rec = [bt[j] == "eos" and pre(b[j], m[j]) for j in range(len(b))]
    prem = [mt[j] == "eos" and pre(m[j], b[j]) for j in range(len(b))]
    return {"changed_rows": int(sum(ch)), "eos_recovery_rows": int(sum(eos_rec)), "premature_eos_rows": int(sum(prem)),
            "new_severe_truncations": int(sum(severe_truncation(len(b[j]), len(m[j]), mt[j]) and not severe_truncation(len(b[j]), len(b[j]), bt[j])
                                              for j in range(len(b)))),
            "added_caps": int(sum(mt[j] == "cap" and bt[j] != "cap" for j in range(len(b)))),
            "length_delta_total": int(sum(len(m[j]) - len(b[j]) for j in range(len(b)))), "_eos_rec": eos_rec, "_changed": ch}


def compare(R, H, Tk, Te, X, dialogues, ids, genuine, cfgd) -> dict:
    from csasr.evaluation.canonical import corpus_metrics
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import outside
    ev = row_events(Tk, Te, X)
    mB, mX = corpus_metrics(R, H["B0"]), corpus_metrics(R, H[X])
    tr = poi_transitions(R, H["B0"], H[X], ev["_eos_rec"], dialogues, ids, genuine)
    zr, er = matrix_zh_retention(R, H["B0"], H[X]), embedded_en_retention(R, H["B0"], H[X])
    outs = [outside(r, b, s) if b != s else None for r, b, s in zip(R, H["B0"], H[X])]
    base_out = [outside(r, b, b) for r, b in zip(R, H["B0"])] if any(o is None for o in outs) else None
    outs = [o if o is not None else base_out[j] for j, o in enumerate(outs)]
    osum = {k: sum(o[k] for o in outs) for k in outs[0]}
    ohr = osum["outside_harm"] / osum["baseline_correct_outside"] if osum["baseline_correct_outside"] else None
    zh = zh_changes(R, H["B0"], H[X])
    S = cfgd["evaluation"]["safety_bounds_vs_B0"]
    G = S["float_guard"]
    p = {"mer_increase": mX["mer"] - mB["mer"], "zh_cer_increase": mX["zh_cer"] - mB["zh_cer"], "en_wer_increase": mX["en_wer"] - mB["en_wer"],
         "pier_gain": mB["pier"] - mX["pier"], "zh_retention": zr["rate"], "en_retention": er["rate"], "outside_harm_rate": ohr}
    safety = {"MER": p["mer_increase"] <= S["max_MER_increase"] + G, "ZH_CER": p["zh_cer_increase"] <= S["max_ZH_CER_increase"] + G,
              "matrix_ZH_retention": zr["rate"] is None or zr["rate"] >= S["min_matrix_ZH_retention"] - G,
              "outside_harm": ohr is None or ohr <= S["max_outside_POI_harm_rate"] + G,
              "POI_corruption": tr["corruption_rate"] is None or tr["corruption_rate"] <= S["max_POI_corruption_rate"] + G,
              "caps": ev["added_caps"] <= S["max_additional_caps"], "severe_truncation": ev["new_severe_truncations"] <= S["max_new_severe_truncations"]}
    ev = {k: v for k, v in ev.items() if not k.startswith("_")}
    return {"points": p, "transitions": tr, "zh": zh, "retention": {"matrix_zh": zr, "embedded_en": er}, "outside": osum,
            "events": ev, "safety": safety, "safe": all(safety.values())}


def bootstrap(R, H, dialogues, pairs) -> dict:
    from experiments.inference_cf_p2seq_analyze import draws, ratio_delta, utt_counts
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in H}
    idxs = draws(dialogues)
    nm = {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}
    return {f"{a}_minus_{b}": {n: ratio_delta(C[a], C[b], x, y, idxs) for n, (x, y) in nm.items()} for a, b in pairs}, C


def subgroup_metrics(R, H, idx) -> dict:
    from csasr.evaluation.canonical import corpus_metrics
    out = {}
    for k in H:
        m = corpus_metrics([R[j] for j in idx], [H[k][j] for j in idx])
        out[k] = {x: m[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "num_poi", "substitutions", "deletions", "insertions")}
    return out


def decide(valid: bool, cmp_: dict, cfgd: dict) -> dict:
    if not valid:
        return {"label": "TTLS_R1_INVALID", "promising_arms": [], "mixed_arms": []}
    prom, mixed = [], []
    for X in TTLS_ARMS:
        x, y = cmp_[X], cmp_[MATCH[X]]
        a = x["transitions"]["genuine_substitution_corrections"] >= 8 and x["transitions"]["genuine_dialogue_count"] >= 4
        c = (x["transitions"]["genuine_substitution_corrections"] >= y["transitions"]["genuine_substitution_corrections"]
             and x["zh"]["new_zh_errors"] <= y["zh"]["new_zh_errors"])
        if a and x["safe"] and c:
            prom.append(X)
        if x["transitions"]["genuine_substitution_corrections"] >= 3 and x["transitions"]["net"] > 0:
            mixed.append(X)
    label = "TTLS_R1_PROMISING" if prom else ("TTLS_R1_MIXED" if mixed else "TTLS_R1_NOT_SUPPORTED")
    return {"label": label, "promising_arms": prom, "mixed_arms": mixed}


def describe(d: dict, H, Tk) -> dict:
    """Reference-free optimization, candidate and compute descriptives."""
    from transformers import WhisperTokenizer
    from csasr.inference_cf.core_r2 import tokenizer_partition
    tok = WhisperTokenizer.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    part = tokenizer_partition(tok)
    Mset = set(part["matrix_ids"])
    rows = d["rows"]
    cand = {"rows": 100, "rows_with_structural_steps": 0, "rows_with_M": 0, "rows_accepted": 0, "proposals": 0, "accepted_steps": 0,
            "structural_steps": 0, "accepted_by_group": Counter(), "M_by_group": Counter(), "b_class_accepted": Counter(),
            "e_space_led_accepted": 0, "support_accepted": [], "support_rejected": [], "E_cB_rejected_neg": 0, "E_cE_rejected_neg": 0,
            "accepted": []}
    for r in rows:
        c = r["candidates"]
        st = [x for x in c["records"] if x["structural"]]
        cand["structural_steps"] += len(st)
        cand["rows_with_structural_steps"] += bool(st)
        cand["rows_with_M"] += bool(c["M"])
        cand["M_by_group"][r["group"]] += bool(c["M"])
        cand["proposals"] += len(c["M"])
        cand["accepted_steps"] += len(c["accepted"])
        for x in st:
            if x.get("proposal"):
                (cand["support_accepted"] if x["accepted"] else cand["support_rejected"]).append(x["support"])
                if not x["accepted"]:
                    cand["E_cB_rejected_neg"] += x["evidence"]["cB"]["E"] < 0
                    cand["E_cE_rejected_neg"] += x["evidence"]["cE"]["E"] < 0
        if c["t_star"] is not None:
            cand["rows_accepted"] += 1
            cand["accepted_by_group"][r["group"]] += 1
            x = next(x for x in c["records"] if x["t"] == c["t_star"])
            cand["b_class_accepted"]["HAN" if x["b"] in Mset else "OTHER"] += 1
            e_s = tok.decode([x["e"]])
            cand["e_space_led_accepted"] += e_s.startswith(" ")
            cand["accepted"].append({"utterance_id": r["identity"], "dialogue_id": r["dialogue_id"], "group": r["group"], "t": c["t_star"],
                                     "b": tok.decode([x["b"]]), "c": e_s, "E_cB": x["evidence"]["cB"]["E"], "E_cE": x["evidence"]["cE"]["E"],
                                     "n_accepted_steps": len(c["accepted"]), "n_M": len(c["M"]), "T": c["T"]})
    q = lambda v: {"n": len(v), "median": float(np.median(v)) if v else None, "p10": float(np.quantile(v, .1)) if v else None,
                   "p90": float(np.quantile(v, .9)) if v else None}
    cand["support_accepted"], cand["support_rejected"] = q(cand["support_accepted"]), q(cand["support_rejected"])
    cand["abstain_rate_rows"] = 1 - cand["rows_accepted"] / 100
    cand["acceptance_rate_proposals"] = cand["accepted_steps"] / cand["proposals"] if cand["proposals"] else None
    for k in ("accepted_by_group", "M_by_group", "b_class_accepted"):
        cand[k] = dict(cand[k])
    opt = {}
    for a in ARMS:
        act = [r[a] for r in rows if r[a]["status"] == "ok"]
        if not act:
            opt[a] = {"active_rows": 0}
            continue
        L = np.array([x["log"]["losses"] for x in act])
        parts = {k: np.array([[p.get(k, np.nan) for p in x["log"]["parts"]] for x in act]) for k in act[0]["log"]["parts"][0]}
        disp = lambda k: [x["displacement"][k] for x in act if x["displacement"].get(k) is not None]
        opt[a] = {"active_rows": len(act), "trainable_scalars": act[0]["log"]["trainable_scalars"],
                  "loss_mean": L.mean(0).tolist(), "loss_median": np.median(L, 0).tolist(),
                  "rows_loss_decreased": int((L[:, 2] < L[:, 0]).sum()),
                  "parts_mean": {k: np.nanmean(v, 0).tolist() for k, v in parts.items()},
                  "grad_l2_mean": np.mean([x["log"]["grad_l2"] for x in act], 0).tolist(),
                  "master_delta_l2_mean": float(np.mean([x["log"]["master_delta_l2"] for x in act])),
                  "z_norm_final_median": float(np.median([x["log"]["z_norm"][-1] for x in act])) if act[0]["log"]["z_norm"] else None,
                  "projected_rows": int(sum(any(x["log"]["projected"]) for x in act)),
                  "site_rel_mean": float(np.mean(disp("site_rel_mean"))), "site_rel_edited_mean": float(np.mean(disp("site_rel_edited_mean"))) if disp("site_rel_edited_mean") else None,
                  "kl_mean_all": float(np.mean(disp("kl_mean_all"))), "kl_mean_stable": float(np.mean(disp("kl_mean_stable"))) if disp("kl_mean_stable") else None,
                  "adapt_sec_total": float(sum(x["log"]["adapt_sec"] for x in act)), "decode_sec_total": float(sum(x["decode_sec"] for x in act)),
                  "peak_alloc_max_GB": max((x["peak"] or {}).get("alloc", 0) for x in act) / 1e9}
        if a in ("T3", "T4", "T5", "T6"):
            opt[a]["target_rank_step"] = np.mean([[p["target_rank"] for p in x["log"]["parts"]] for x in act], 0).tolist()
            opt[a]["target_rank1_final_rows"] = int(sum(x["log"]["parts"][-1]["target_rank"] == 0 for x in act))
            opt[a]["margin_vs_b_mean"] = np.mean([[p["margin_vs_b"] for p in x["log"]["parts"]] for x in act], 0).tolist()
    b2 = [r["B2"]["displacement"] for r in rows]
    opt["B2"] = {"trainable_scalars": 248320, "site_rel_mean": float(np.mean([x["site_rel_mean"] for x in b2])),
                 "kl_mean_all": float(np.mean([x["kl_mean_all"] for x in b2])),
                 "kl_mean_stable": float(np.mean([x["kl_mean_stable"] for x in b2 if x["kl_mean_stable"] is not None]))}
    cnt = d["runtime"]["counters"]
    compute = {"elapsed_sec": d["runtime"]["elapsed_sec"], "setup_sec": d["runtime"]["setup_sec"], "job_id": d["runtime"]["job_id"],
               "gpu": d["runtime"]["gpu"], "counters": cnt,
               "row_elapsed_sec_total": float(sum(r["elapsed_sec"] for r in rows)),
               "baseline_decode_sec_total": float(sum(r["baseline_decode_sec"] for r in rows)),
               "branch_sec_total": float(sum(r["branch_sec"] for r in rows)),
               "CD_decode_sec_total": float(sum(r["CD"]["decode_sec"] for r in rows)),
               "B2_decode_sec_total": float(sum(r["B2"]["decode_sec"] for r in rows)),
               "CD_edits_total": int(sum(len(r["CD"]["edits"]) for r in rows))}
    return {"candidates": cand, "optimization": opt, "compute": compute}


def evaluate(run_rel: str) -> dict:
    c = json.loads((ROOT / CONFIG).read_text())
    if not committed(SEAL):
        raise PermissionError("TTLS-R1 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed TTLS-R1 outputs changed")
    d = load(run_rel)
    integ = integrity(d)
    if not integ["all_100_ok"]:
        return {"schema": "ttls_r1_evaluation_v1", "integrity": integ, "valid": False, "label": "TTLS_R1_INVALID"}
    H, Tk, Te = systems(d)
    desc = describe(d, H, Tk)
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
    genuine = set(c["evaluation"]["genuine_substitution_categories"])
    cmp_ = {X: compare(R, H, Tk, Te, X, dialogues, ids, genuine, c) for X in SYSTEMS if X != "B0"}
    vsB1 = {X: {k: M[X][k] - M["B1"][k] for k in ("pier", "mer", "en_wer", "zh_cer")} for X in SYSTEMS}
    pairs = [(X, "B0") for X in SYSTEMS if X != "B0"] + [(X, "B1") for X in SYSTEMS if X != "B1"] + [(X, MATCH[X]) for X in TTLS_ARMS] \
        + [("T2", "T1"), ("T2A", "B2"), ("T5", "T3"), ("T6", "T4"), ("T3", "ACSUB"), ("T4", "ACSUB")]
    boot, C = bootstrap(R, H, dialogues, pairs)
    groups = {"D12": [j for j, u in enumerate(ids) if P[u]["group"] == "D"], "A88": [j for j, u in enumerate(ids) if P[u]["group"] == "A"],
              "A3_panel24": [j for j, u in enumerate(ids) if P[u]["a3_group"] is not None],
              "AC_accepted": [j for j, r in enumerate(d["rows"]) if r["candidates"]["t_star"] is not None]}
    sub = {g: {"n": len(ix), "metrics": subgroup_metrics(R, H, ix)} for g, ix in groups.items()}
    for g, ix in groups.items():
        if not ix:
            continue
        Rg = [R[j] for j in ix]
        Hg = {k: [H[k][j] for j in ix] for k in H}
        Tg = {k: [Tk[k][j] for j in ix] for k in Tk}
        Eg = {k: [Te[k][j] for j in ix] for k in Te}
        sub[g]["vs_B0"] = {X: (lambda r: {"corrections": r["transitions"]["corrections"], "corruptions": r["transitions"]["corruptions"],
                                          "genuine_substitution_corrections": r["transitions"]["genuine_substitution_corrections"],
                                          "new_zh_errors": r["zh"]["new_zh_errors"], "zh_repairs": r["zh"]["zh_repairs"],
                                          "changed_rows": r["events"]["changed_rows"], "eos_recovery_rows": r["events"]["eos_recovery_rows"]})(
            compare(Rg, Hg, Tg, Eg, X, [dialogues[j] for j in ix], [ids[j] for j in ix], genuine, c)) for X in SYSTEMS if X != "B0"}
    # candidate correctness (post-seal): is the accepted candidate word an English word of the reference?
    from csasr.evaluation.normalization import normalize_text
    from csasr.evaluation.pier import reference_pois
    for a in desc["candidates"]["accepted"]:
        j = ids.index(a["utterance_id"])
        pois = [s.lower() for _, s in reference_pois(normalize_text(R[j]))]
        w = a["c"].strip().lower()
        a["candidate_prefix_of_reference_english_word"] = bool(w) and any(p.startswith(w) for p in pois)
        a["reference_has_english"] = bool(pois)
        a["ACSUB_corrections"] = sum(1 for e in cmp_["ACSUB"]["transitions"]["events"] if e["utterance_id"] == a["utterance_id"] and e["type"] == "correction")
        a["ACSUB_corruptions"] = sum(1 for e in cmp_["ACSUB"]["transitions"]["events"] if e["utterance_id"] == a["utterance_id"] and e["type"] == "corruption")
    acc = desc["candidates"]["accepted"]
    desc["candidates"]["precision"] = {"accepted_rows": len(acc), "candidate_prefix_of_reference_english_word": sum(a["candidate_prefix_of_reference_english_word"] for a in acc),
                                       "rows_reference_has_english": sum(a["reference_has_english"] for a in acc)}
    # active-edit harm: among rows where the system output differs from B0, mixed-error change
    harm = {}
    for X in SYSTEMS:
        if X == "B0":
            continue
        ch = [j for j in range(100) if Tk[X][j] != Tk["B0"][j]]
        dm = [C[X][j][2] - C["B0"][j][2] for j in ch]
        harm[X] = {"changed_rows": len(ch), "rows_mixed_errors_worse": int(sum(x > 0 for x in dm)), "rows_mixed_errors_better": int(sum(x < 0 for x in dm)),
                   "harm_rate": (sum(x > 0 for x in dm) / len(ch)) if ch else None, "net_mixed_error_change": int(sum(dm))}
    # outside-region preservation (reference-free): token prefix before the first editable step unchanged
    pref = {}
    for X in ("T1", "T2", "T4", "T6"):
        k = []
        for j, r in enumerate(d["rows"]):
            if r[X]["status"] != "ok":
                continue
            t0 = 1 if X in ("T1", "T2") else r["candidates"]["t_star"]
            k.append(Tk[X][j][:t0] == Tk["B0"][j][:t0])
        pref[X] = {"active_rows": len(k), "prefix_before_region_preserved": int(sum(k))}
    changed = []
    for j, u in enumerate(ids):
        diff = [X for X in SYSTEMS if X != "B0" and Tk[X][j] != Tk["B0"][j]]
        if any(X in ARMS or X in ("CD", "ACSUB") for X in diff):
            changed.append({"utterance_id": u, "dialogue_id": dialogues[j], "group": P[u]["group"], "a3": P[u]["a3_group"] is not None,
                            "t_star": d["rows"][j]["candidates"]["t_star"], "reference": R[j],
                            "texts": {X: H[X][j] for X in SYSTEMS}, "differs_from_B0": diff,
                            "counts": {X: C[X][j].tolist() for X in SYSTEMS}})
    label = decide(valid, cmp_, c)
    strip = lambda r: {k: v for k, v in r.items() if k != "events"}
    return {"schema": "ttls_r1_evaluation_v1", "manifest_hash": d["manifest"]["manifest_hash"], "output_seal_hash": seal["seal_hash"],
            "integrity": integ, "valid": valid, **label, "metrics": M, "vs_B1": vsB1,
            "vs_B0": {X: {**v, "transitions": strip(v["transitions"])} for X, v in cmp_.items()},
            "transition_events": {X: v["transitions"]["events"] for X, v in cmp_.items()},
            "bootstrap_descriptive": boot, "subgroups": sub, "active_edit_harm": harm, "prefix_preservation": pref,
            "descriptive": desc, "changed_utterances": changed,
            "label_rule": c["label_rule"]}


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
    print(json.dumps({"label": res["label"], "promising": res.get("promising_arms"), "mixed": res.get("mixed_arms"), "integrity": res["integrity"],
                      "metrics": {k: {x: v[x] for x in keys} for k, v in res.get("metrics", {}).items()},
                      "vs_B0": {X: {"corr": v["transitions"]["corrections"], "corrupt": v["transitions"]["corruptions"],
                                    "genuine": v["transitions"]["genuine_substitution_corrections"], "gdlg": v["transitions"]["genuine_dialogue_count"],
                                    "eosrow": v["transitions"]["eos_recovery_row_corrections"], "newZH": v["zh"]["new_zh_errors"],
                                    "zhrep": v["zh"]["zh_repairs"], "safe": v["safe"], "changed": v["events"]["changed_rows"],
                                    "eos_rec_rows": v["events"]["eos_recovery_rows"], "prem": v["events"]["premature_eos_rows"]}
                              for X, v in res.get("vs_B0", {}).items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
