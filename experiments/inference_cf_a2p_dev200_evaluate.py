#!/usr/bin/env python
"""A2P-DEV200 post-seal evaluation (frozen contract configs/inference_cf/a2p_dev200.json). Written before the run.

References open only after the committed A2P-DEV200 output seal (and the reused P2-PATH5 / TTLS-R1 seals) verify.
Free-decoding transcripts only. Canonical metrics through the unchanged TTLS-R1 evaluator primitives; frozen lexical
split from TTLS-R1R; breadth / concentration / LODO / dialogue bootstrap and the frozen A2P-DEV200 decision rule.
Primary population NEW200; FIXED100 and FULL300 are secondary and never enter the rule.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

import experiments.inference_cf_ttls_r1_evaluate as E
from experiments.inference_cf_ttls_r1r_evaluate import lexical_kind
from csasr.inference_cf.core import atomic_json, digest

CONFIG = "configs/inference_cf/a2p_dev200.json"
BASE = "results/inference_cf/a2p_dev200"
SEAL = f"{BASE}/output_seal.json"
P5_SEAL = "results/inference_cf/p2path5/output_seal.json"
SYSTEMS = ("B0", "B1", "B2", "B3")
GENUINE = ("genuine_wrong_language", "genuine_same_language")


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
    reg = []
    for s in plan["regression"]:
        p = run / "regression" / f"{s['r1_index']:03d}.json"
        reg.append(json.loads(p.read_text()) if p.exists() else {"status": "missing"})
    return {"manifest": man, "plan": plan, "rows": rows, "regression": reg, "runtime": json.loads((run / "runtime.json").read_text())}


def integrity(d: dict) -> dict:
    rows, rt = d["rows"], d["runtime"]
    c = {"all_200_ok": len(rows) == 200 and all(r.get("status") == "ok" for r in rows),
         "regression_ok": all(r.get("status") == "ok" for r in d["regression"]) and len(d["regression"]) == len(d["plan"]["regression"]),
         "runtime_completed": rt.get("status") == "completed",
         "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged")) and bool(rt.get("model_grads_none"))}
    if not (c["all_200_ok"] and c["regression_ok"]):
        return c
    c["row_gates"] = all(all(r["gates"].values()) for r in rows + d["regression"])
    c["B0_reproduced"] = all(r["B0"]["tokens_equal"] and r["B0"]["terminated_equal"] for r in rows)
    c["B1_replayed_text"] = all(r["B1"]["replay_text_equal"] for r in rows)
    c["B2_step0_identity"] = all(r["gates"]["step0_CE_equal_PATH5_A2"] for r in rows)
    c["regression_T2A_exact"] = all(r["gates"]["T2A_outputs_equal_R1"] and r["gates"]["T2A_log_equal_R1"] for r in d["regression"])
    c["regression_A2_exact"] = all(r["gates"]["A2_masters_equal_archive"] and r["gates"]["A2_decode_equal_sealed"] for r in d["regression"])
    return c


def systems(d: dict):
    P = d["plan"]["rows"]
    H = {"B0": [p["y_B_text"] for p in P], "B1": [p["AUTO"]["text"] for p in P], "B2": [p["A2"]["text"] for p in P],
         "B3": [r["B3"]["text"] for r in d["rows"]]}
    Tk = {"B0": [p["y_B"] for p in P], "B1": [p["AUTO"]["tokens"] for p in P], "B2": [p["A2"]["tokens"] for p in P],
          "B3": [r["B3"]["tokens"] for r in d["rows"]]}
    Te = {"B0": [p["y_B_terminated"] for p in P], "B1": [p["AUTO"]["terminated"] for p in P], "B2": [p["A2"]["terminated"] for p in P],
          "B3": [r["B3"]["terminated"] for r in d["rows"]]}
    return H, Tk, Te


def fixed100_systems():
    """Secondary: FIXED100 from the sealed TTLS-R1 plan / rows (B0 = S0, B1 = AUTO, B2 = historical A2, B3 = T2A)."""
    seal = json.loads((ROOT / E.SEAL).read_text())
    if any(E.sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("TTLS-R1 sealed outputs changed")
    plan = json.loads((ROOT / E.BASE / "plan_sealed.json").read_text())
    rows = [json.loads((ROOT / E.BASE / f"run1/rows/{i:03d}.json").read_text()) for i in range(100)]
    P = plan["rows"]
    H = {"B0": [p["y_B_text"] for p in P], "B1": [p["y_A_text"] for p in P], "B2": [p["A2"]["text"] for p in P], "B3": [r["T2A"]["text"] for r in rows]}
    Tk = {"B0": [p["y_B"] for p in P], "B1": [p["y_A"] for p in P], "B2": [p["A2"]["tokens"] for p in P], "B3": [r["T2A"]["tokens"] for r in rows]}
    Te = {"B0": [p["y_B_terminated"] for p in P], "B1": [p["y_A_terminated"] for p in P], "B2": [p["A2"]["terminated"] for p in P],
          "B3": [r["T2A"]["terminated"] for r in rows]}
    return plan["ids"], [p["dialogue_id"] for p in P], H, Tk, Te


# ---- reference-based accounting -------------------------------------------------------------------------------------

def poi_events(R, Hb, Hs, ids, dialogues) -> list:
    """Per-POI transitions with reference indices (TTLS-R1 poi_transitions semantics) + frozen lexical kind."""
    from csasr.evaluation.normalization import normalize_text
    from csasr.evaluation.pier import evaluate_pois, reference_pois
    ev = []
    for j, (ref, b, m) in enumerate(zip(R, Hb, Hs)):
        if b == m:
            continue
        idxs = [i for i, _ in reference_pois(normalize_text(ref))]
        br = {x.poi_index: x for x in evaluate_pois(ref, b, idxs)}
        mr = {x.poi_index: x for x in evaluate_pois(ref, m, idxs)}
        for i in idxs:
            if i not in br or i not in mr or br[i].correct == mr[i].correct:
                continue
            if not br[i].correct:
                e = {"type": "correction", "B0_category": br[i].category, "poi": br[i].surface, "B0": br[i].hyp_surface, "system": mr[i].hyp_surface}
                e["kind"] = lexical_kind(e)
            else:
                e = {"type": "corruption", "poi": br[i].surface, "system": mr[i].hyp_surface, "system_category": mr[i].category,
                     "kind": "corruption_" + ("lexical" if mr[i].category not in ("deletion", "boundary_error", "insertion_near_poi") else "deletion")}
            ev.append({"j": j, "utterance_id": ids[j], "dialogue_id": dialogues[j], "poi_index": i, **e})
    return ev


def group_events(ev: list) -> list:
    """Lexical event = maximal run of consecutive reference POI indices with the same transition type in one utterance."""
    out = []
    by = defaultdict(list)
    for e in ev:
        by[(e["j"], e["type"])].append(e)
    for (j, typ), es in sorted(by.items()):
        es = sorted(es, key=lambda e: e["poi_index"])
        run = [es[0]]
        for e in es[1:]:
            if e["poi_index"] == run[-1]["poi_index"] + 1:
                run.append(e)
            else:
                out.append(run)
                run = [e]
        out.append(run)
    return [{"utterance_id": r[0]["utterance_id"], "dialogue_id": r[0]["dialogue_id"], "type": r[0]["type"], "units": len(r),
             "kinds": dict(Counter(e["kind"] for e in r)), "wrong_language": any(e["kind"] == "genuine_wrong_language" for e in r),
             "genuine": any(e["kind"] in GENUINE for e in r), "pois": " ".join(e["poi"] for e in r),
             "before": " ".join((e.get("B0") or "") for e in r), "after": " ".join((e.get("system") or "") for e in r)} for r in out]


def zh_per_row(R, Hb, Hs) -> list:
    return [E.zh_changes([r], [b], [m]) if b != m else {"new_zh_errors": 0, "zh_repairs": 0} for r, b, m in zip(R, Hb, Hs)]


def term_changed(Tk, Te, X, j) -> bool:
    b, bt, m, mt = Tk["B0"][j], Te["B0"][j], Tk[X][j], Te[X][j]
    pre = lambda a, c: len(a) < len(c) and list(c[:len(a)]) == list(a)
    return mt != bt or (bt == "eos" and pre(b, m)) or (mt == "eos" and pre(m, b))


def shares(vals: dict) -> dict:
    pos = {k: v for k, v in vals.items() if v > 0}
    tot = sum(pos.values())
    top = sorted(pos.items(), key=lambda x: -x[1])
    return {"positive_total": tot, "n_positive": len(pos), "n_negative": sum(v < 0 for v in vals.values()), "n_zero": sum(v == 0 for v in vals.values()),
            "largest": top[0] if top else None, "largest_share": (top[0][1] / tot) if tot else None,
            "top3": top[:3], "top3_share": (sum(v for _, v in top[:3]) / tot) if tot else None}


def wlt(delta: list) -> dict:
    """delta < 0 = fewer errors (improved)."""
    return {"improved": int(sum(x < 0 for x in delta)), "worsened": int(sum(x > 0 for x in delta)), "tied": int(sum(x == 0 for x in delta))}


def analyse(R, H, Tk, Te, ids, dialogues, cfg1) -> dict:
    """All NEW200-style quantities for one population (used for NEW200 primary, FIXED100 / FULL300 secondary)."""
    from csasr.evaluation.canonical import corpus_metrics
    from experiments.inference_cf_p2seq_analyze import draws, utt_counts
    genuine = set(cfg1["evaluation"]["genuine_substitution_categories"])
    n = len(ids)
    M = {k: {**corpus_metrics(R, H[k]), "caps": sum(t == "cap" for t in Te[k])} for k in SYSTEMS}
    cmpB0 = {X: E.compare(R, H, Tk, Te, X, dialogues, ids, genuine, cfg1) for X in ("B1", "B2", "B3")}
    H2, Tk2, Te2 = ({"B0": D["B2"], "B3": D["B3"]} for D in (H, Tk, Te))
    cmpB2 = E.compare(R, H2, Tk2, Te2, "B3", dialogues, ids, genuine, cfg1)
    ev = {X: poi_events(R, H["B0"], H[X], ids, dialogues) for X in ("B1", "B2", "B3")}
    ev["B3_vs_B2"] = poi_events(R, H["B2"], H["B3"], ids, dialogues)
    lex = {}
    for X, es in ev.items():
        corr = [e for e in es if e["type"] == "correction"]
        groups = group_events(es)
        lex[X] = {"corrections": len(corr), "corruptions": sum(e["type"] == "corruption" for e in es),
                  "by_kind": dict(Counter(e["kind"] for e in es)), "genuine_lexical_units": sum(e["kind"] in GENUINE for e in corr),
                  "wrong_language_units": sum(e["kind"] == "genuine_wrong_language" for e in corr),
                  "genuine_utterances": len({e["utterance_id"] for e in corr if e["kind"] in GENUINE}),
                  "genuine_dialogues": len({e["dialogue_id"] for e in corr if e["kind"] in GENUINE}),
                  "correction_events": sum(g["type"] == "correction" for g in groups),
                  "genuine_events": sum(g["type"] == "correction" and g["genuine"] for g in groups),
                  "wrong_language_events": sum(g["type"] == "correction" and g["wrong_language"] for g in groups),
                  "corruption_events": sum(g["type"] == "corruption" for g in groups), "events": groups}
    # retention of B2 corrections by B3 (unit level)
    key = lambda e: (e["j"], e["poi_index"])
    c2 = {key(e): e for e in ev["B2"] if e["type"] == "correction"}
    c3 = {key(e) for e in ev["B3"] if e["type"] == "correction"}
    g2 = {k for k, e in c2.items() if e["kind"] in GENUINE}
    retention = {"B2_corrections": len(c2), "retained": len(set(c2) & c3), "rate": len(set(c2) & c3) / len(c2) if c2 else None,
                 "B2_genuine": len(g2), "genuine_retained": len(g2 & c3), "genuine_rate": len(g2 & c3) / len(g2) if g2 else None,
                 "B3_new_corrections_not_in_B2": len(c3 - set(c2))}
    # per-utterance new-ZH and errors
    z = {X: zh_per_row(R, H["B0"], H[X]) for X in ("B1", "B2", "B3")}
    newzh = {X: [x["new_zh_errors"] for x in z[X]] for X in z}
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in SYSTEMS}
    d_u = {ids[j]: newzh["B2"][j] - newzh["B3"][j] for j in range(n)}
    dlgs = sorted(set(dialogues))
    d_d = {g: sum(d_u[ids[j]] for j in range(n) if dialogues[j] == g) for g in dlgs}
    N2, N3 = sum(newzh["B2"]), sum(newzh["B3"])
    tc = [term_changed(Tk, Te, "B2", j) or term_changed(Tk, Te, "B3", j) for j in range(n)]
    D_excl = sum(d_u[ids[j]] for j in range(n) if not tc[j])
    lodo = {g: N2 - N3 - d_d[g] for g in dlgs}
    mixed = lambda X, Y: [C[X][j][2] - C[Y][j][2] for j in range(n)]
    dsum = lambda v: [sum(v[j] for j in range(n) if dialogues[j] == g) for g in dlgs]
    breadth = {"mixed_B3_vs_B0_utt": wlt(mixed("B3", "B0")), "mixed_B3_vs_B2_utt": wlt(mixed("B3", "B2")), "mixed_B2_vs_B0_utt": wlt(mixed("B2", "B0")),
               "mixed_B3_vs_B0_dlg": wlt(dsum(mixed("B3", "B0"))), "mixed_B3_vs_B2_dlg": wlt(dsum(mixed("B3", "B2"))),
               "mixed_B2_vs_B0_dlg": wlt(dsum(mixed("B2", "B0"))), "mixed_B3_vs_B1_utt": wlt(mixed("B3", "B1")), "mixed_B3_vs_B1_dlg": wlt(dsum(mixed("B3", "B1"))),
               "zh_avoided_utt": shares(d_u), "zh_avoided_dlg": shares(d_d),
               "mixed_gain_B3_vs_B0_utt": shares({ids[j]: -mixed("B3", "B0")[j] for j in range(n)}),
               "mixed_gain_B3_vs_B0_dlg": shares(dict(zip(dlgs, [-x for x in dsum(mixed("B3", "B0"))]))),
               "mixed_gain_B2_vs_B0_utt": shares({ids[j]: -mixed("B2", "B0")[j] for j in range(n)}),
               "newzh_rows": {X: int(sum(x > 0 for x in newzh[X])) for X in newzh},
               "newzh_dialogues": {X: len({dialogues[j] for j in range(n) if newzh[X][j] > 0}) for X in newzh}}
    idxs = draws(dialogues)
    dv = np.array([d_u[ids[j]] for j in range(n)], dtype=float)
    bd = [float(dv[ix].sum()) for ix in idxs]
    pairs = [("B3", "B2"), ("B3", "B0"), ("B3", "B1"), ("B2", "B0"), ("B2", "B1"), ("B1", "B0")]
    boot, _ = E.bootstrap(R, H, dialogues, pairs)
    boot["newzh_avoided_B2_minus_B3"] = {"delta": float(N2 - N3), "ci95": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))],
                                         "valid_draws": len(bd)}
    nzd = lambda X: np.array(newzh[X], dtype=float)
    for X in ("B2", "B3"):
        v = nzd(X)
        bx = [float(v[ix].sum()) for ix in idxs]
        boot[f"newzh_{X}_vs_B0"] = {"delta": float(v.sum()), "ci95": [float(np.quantile(bx, .025)), float(np.quantile(bx, .975))]}
    lodo_metrics = {}
    for g in dlgs:
        keep = [j for j in range(n) if dialogues[j] != g]
        mm = {X: corpus_metrics([R[j] for j in keep], [H[X][j] for j in keep]) for X in ("B2", "B3")}
        lodo_metrics[g] = {"D_newzh": lodo[g], "dMER_B3_B2": mm["B3"]["mer"] - mm["B2"]["mer"], "dPIER_B3_B2": mm["B3"]["pier"] - mm["B2"]["pier"]}
    events_rows = {X: E.row_events(Tk, Te, X) for X in ("B1", "B2", "B3")}
    return {"n": n, "metrics": M, "vs_B0": {X: {**v, "transitions": {k: x for k, x in v["transitions"].items() if k != "events"}} for X, v in cmpB0.items()},
            "B3_vs_B2": {**cmpB2, "transitions": {k: x for k, x in cmpB2["transitions"].items() if k != "events"}},
            "lexical": lex, "retention_B2_by_B3": retention, "newzh": {"N2": N2, "N3": N3, "D": N2 - N3, "N1": sum(newzh["B1"]),
                                                                     "D_non_termination_rows": D_excl, "termination_changed_rows": int(sum(tc)),
                                                                     "per_utterance_d": d_u, "per_dialogue_d": d_d},
            "lodo": {"D_positive_panels": int(sum(v > 0 for v in lodo.values())), "D_min": min(lodo.values()), "D_max": max(lodo.values()),
                     "per_dialogue": lodo_metrics},
            "breadth": breadth, "bootstrap": boot,
            "row_events": {X: {k: v for k, v in e.items() if not k.startswith("_")} for X, e in events_rows.items()},
            "_C": C, "_newzh": newzh}


def decide(valid: bool, a: dict, rule: dict) -> dict:
    if not valid:
        return {"label": "A2P_DEV200_INVALID", "criteria": {}}
    M, nz, ret, br = a["metrics"], a["newzh"], a["retention_B2_by_B3"], a["breadth"]
    N2, D = nz["N2"], nz["D"]
    if ret["B2_genuine"] >= 3:
        r, rk = ret["genuine_rate"], "genuine"
    elif ret["B2_corrections"] >= 3:
        r, rk = ret["rate"], "all"
    else:
        r, rk = None, "not_assessable"
    ev3, ev2 = a["vs_B0"]["B3"]["events"], a["vs_B0"]["B2"]["events"]
    zu = br["zh_avoided_utt"]
    crit = {
        "1_opportunity": N2 >= 10,
        "2_magnitude": D >= 0.5 * N2 and D > 0,
        "3_breadth": zu["n_positive"] >= 5 and br["zh_avoided_dlg"]["n_positive"] >= 4 and zu["n_positive"] >= 2 * zu["n_negative"],
        "4_concentration": (zu["largest_share"] is not None and zu["largest_share"] <= 0.5) and a["lodo"]["D_positive_panels"] >= 18
        and nz["D_non_termination_rows"] >= 5,
        "5_retention": r is None or r >= 0.8,
        "6_safety_vs_B2": M["B3"]["mer"] <= M["B2"]["mer"] + 0.002 + 1e-12 and M["B3"]["pier"] <= M["B2"]["pier"] + 0.01 + 1e-12
        and ev3["new_severe_truncations"] <= ev2["new_severe_truncations"] and ev3["added_caps"] <= ev2["added_caps"],
        "7_vs_AUTO": M["B3"]["mer"] < M["B1"]["mer"] and M["B3"]["zh_cer"] < M["B1"]["zh_cer"]}
    not_sup = {"D_le_0": D <= 0, "retention_lt_0.5": r is not None and r < 0.5, "MER_regression_gt_0.005": M["B3"]["mer"] > M["B2"]["mer"] + 0.005 + 1e-12}
    if any(not_sup.values()):
        label = "A2P_DEV200_NOT_SUPPORTED"
    elif all(crit.values()):
        label = "A2P_DEV200_BROAD_SUPPORT"
    else:
        label = "A2P_DEV200_CONCENTRATED_OR_MIXED"
    return {"label": label, "criteria": crit, "not_supported_triggers": not_sup, "retention_used": rk, "retention_value": r}


def evaluate(run_rel: str) -> dict:
    c = json.loads((ROOT / CONFIG).read_text())
    c1 = json.loads((ROOT / E.CONFIG).read_text())
    for s_rel in (SEAL, P5_SEAL, E.SEAL):
        if not E.committed(s_rel):
            raise PermissionError(f"{s_rel} missing/uncommitted: references stay closed")
        seal = json.loads((ROOT / s_rel).read_text())
        if any(E.sha(ROOT / p) != h.removeprefix("sha256:") for p, h in seal["files"].items()):
            raise ValueError(f"sealed outputs changed: {s_rel}")
    seal = json.loads((ROOT / SEAL).read_text())
    d = load(run_rel)
    integ = integrity(d)
    if not (integ["all_200_ok"] and integ["regression_ok"]):
        return {"schema": "a2p_dev200_evaluation_v1", "integrity": integ, "valid": False, "label": "A2P_DEV200_INVALID"}
    H, Tk, Te = systems(d)
    from experiments.inference_cf_p2_evaluate import load_references
    refs = load_references()
    ids = d["manifest"]["ids"]
    R = [refs[u]["reference"] for u in ids]
    dialogues = [refs[u]["dialogue_id"] for u in ids]
    integ["dialogues"] = dialogues == [p["dialogue_id"] for p in d["plan"]["rows"]]
    a = analyse(R, H, Tk, Te, ids, dialogues, c1)
    integ["finite_metrics"] = all(a["metrics"][k][x] is not None and math.isfinite(a["metrics"][k][x]) for k in SYSTEMS for x in ("pier", "mer", "en_wer", "zh_cer"))
    valid = all(integ.values())
    label = decide(valid, a, c["decision_rule"])
    C, newzh = a.pop("_C"), a.pop("_newzh")
    changed = []
    for j, u in enumerate(ids):
        if Tk["B2"][j] != Tk["B0"][j] or Tk["B3"][j] != Tk["B0"][j]:
            changed.append({"utterance_id": u, "dialogue_id": dialogues[j], "group": d["plan"]["rows"][j]["group"], "reference": R[j],
                            "texts": {X: H[X][j] for X in SYSTEMS}, "terminated": {X: Te[X][j] for X in SYSTEMS}, "lengths": {X: len(Tk[X][j]) for X in SYSTEMS},
                            "counts": {X: C[X][j].tolist() for X in SYSTEMS}, "new_zh": {X: newzh[X][j] for X in newzh},
                            "B3_equals_B2": Tk["B3"][j] == Tk["B2"][j]})
    # secondary: FIXED100 (sealed TTLS-R1) and FULL300 aggregate
    fids, fdlg, fH, fTk, fTe = fixed100_systems()
    fR = [refs[u]["reference"] for u in fids]
    fx = analyse(fR, fH, fTk, fTe, fids, fdlg, c1)
    fx.pop("_C"), fx.pop("_newzh")
    from csasr.evaluation.canonical import corpus_metrics
    full = {X: corpus_metrics(fR + R, fH[X] + H[X]) for X in SYSTEMS}
    rows = d["rows"]
    opt = {"loss_mean": np.mean([r["B3"]["log"]["losses"] for r in rows], 0).tolist(),
           "parts_mean": {k: np.mean([[p[k] for p in r["B3"]["log"]["parts"]] for r in rows], 0).tolist() for k in ("CE", "P")},
           "grad_l2_mean": np.mean([r["B3"]["log"]["grad_l2"] for r in rows], 0).tolist(),
           "grad0_rel_diff_vs_PATH5_A2_max": max(r["B3"]["grad0_rel_diff_vs_PATH5_A2"] for r in rows),
           "grad0_rel_diff_vs_PATH5_A2_median": float(np.median([r["B3"]["grad0_rel_diff_vs_PATH5_A2"] for r in rows])),
           "master_delta_l2_mean": float(np.mean([r["B3"]["log"]["master_delta_l2"] for r in rows])),
           "rows_empty_S": sum(len(r["candidates"]["S"]) == 0 for r in rows), "S_size_mean": float(np.mean([len(r["candidates"]["S"]) for r in rows])),
           "M_rows": sum(bool(r["candidates"]["M"]) for r in rows), "kl_mean_stable": float(np.mean([r["B3"]["displacement"]["kl_mean_stable"]
                                                                                                  for r in rows if r["B3"]["displacement"]["kl_mean_stable"] is not None])),
           "AUTO_token_replay_equal": sum(r["B1"]["replay_tokens_equal"] for r in rows)}
    rt = d["runtime"]
    compute = {"job_id": rt["job_id"], "gpu": rt["gpu"], "elapsed_sec": rt["elapsed_sec"], "setup_sec": rt["setup_sec"],
               "regression_sec": rt.get("regression", {}).get("elapsed_sec"), "new200_sec": rt.get("new200", {}).get("elapsed_sec"),
               "counters": rt["counters"], "peak_job_GB": {k: v / 1e9 for k, v in rt["peak_job"].items()},
               "B3_adapt_sec_total": float(sum(r["B3"]["log"]["adapt_sec"] for r in rows)), "B3_decode_sec_total": float(sum(r["B3"]["decode_sec"] for r in rows)),
               "B3_peak_alloc_max_GB": max(r["B3"]["peak"]["alloc"] for r in rows) / 1e9, "row_elapsed_sec_total": float(sum(r["elapsed_sec"] for r in rows))}
    reg = [{"utterance_id": r["identity"], "r1_index": r["index"], "gates": r["gates"], "A2_only": r["A2_only"]} for r in d["regression"]]
    return {"schema": "a2p_dev200_evaluation_v1", "manifest_hash": d["manifest"]["manifest_hash"], "output_seal_hash": seal["seal_hash"],
            "integrity": integ, "valid": valid, **label, "NEW200": a, "changed_utterances": changed, "FIXED100_secondary": fx,
            "FULL300_secondary_metrics": full, "optimization": opt, "compute": compute, "regression_oracle": reg,
            "decision_rule": c["decision_rule"]}


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
    a = res.get("NEW200", {})
    keys = ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "substitutions", "deletions", "insertions", "caps")
    print(json.dumps({"label": res["label"], "criteria": res.get("criteria"), "not_supported": res.get("not_supported_triggers"), "integrity": res["integrity"],
                      "metrics": {k: {x: v[x] for x in keys} for k, v in a.get("metrics", {}).items()}, "newzh": {k: v for k, v in a.get("newzh", {}).items() if not k.startswith("per_")},
                      "retention": a.get("retention_B2_by_B3"), "lex": {X: {k: v for k, v in L.items() if k != "events"} for X, L in a.get("lexical", {}).items()}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
