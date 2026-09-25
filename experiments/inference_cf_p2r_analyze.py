#!/usr/bin/env python
"""P2-R analysis (CPU, evaluator side): validity, D1/D2/D3 contrasts, frozen decision rules.

All decision logic below is frozen with the P2-R spec before any P2-R outcome exists.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest

ARMS = ("plus_d", "minus_d", "random")
PRIMARY = ("D1.L_plus", "D1.L_minus", "D1.L_plus_minus", "D1.Lex_plus_minus",
           "D3.L_plus_random", "D3.L_minus_random", "D3.P_plus_random", "D3.P_minus_random",
           "D2.L_oracle_current", "D2.L_oracle_none", "D2.P_oracle_current")


# ---- statistics -----------------------------------------------------------------------------

def dialogue_bootstrap(values: list[tuple[str, float]], reps: int, seed: int, alpha: float) -> dict:
    """Mean of dialogue means; percentile interval at (alpha/2, 1-alpha/2)."""
    by = defaultdict(list)
    for dlg, v in values:
        if v is not None and math.isfinite(v):
            by[dlg].append(v)
    keys = sorted(by)
    if len(keys) < 2:
        return {"estimate": None, "ci": None, "dialogues": len(keys), "positions": sum(map(len, by.values()))}
    means = np.array([np.mean(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    draws = means[idx].mean(axis=1)
    return {"estimate": float(means.mean()), "ci": [float(np.quantile(draws, alpha / 2)),
                                                     float(np.quantile(draws, 1 - alpha / 2))],
            "dialogues": len(keys), "positions": int(sum(map(len, by.values())))}


def lo(x):
    return x["ci"][0] if x and x.get("ci") else None


def hi(x):
    return x["ci"][1] if x and x.get("ci") else None


def pos(x):
    return lo(x) is not None and lo(x) > 0


def neg(x):
    return hi(x) is not None and hi(x) < 0


def contains0(x):
    return x is not None and x.get("ci") is not None and lo(x) <= 0 <= hi(x)


# ---- record extraction ---------------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict, dict, dict]:
    m = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("manifest hash")
    cfg = json.loads((ROOT / m["config"]).read_text())
    pop = json.loads((ROOT / m["population"]).read_text())
    rows = {}
    for i, uid in enumerate(pop["utterances"]):
        p = run / "rows" / f"{i:03d}.json"
        rows[uid] = json.loads(p.read_text()) if p.exists() else {"status": "missing"}
        if p.exists() and rows[uid]["manifest_hash"] != m["manifest_hash"]:
            raise ValueError("stale row")
    return m, cfg, pop, rows


def energy_ok(arms: dict, e_star: float, tol: float) -> bool:
    es = []
    for a in ARMS:
        x = arms.get(a)
        if x is None or x["solver"].get("status") != "ok" or not x["edit_norm"]:
            return False
        es.append(x["edit_norm"] ** 2)
    t2 = e_star ** 2
    return all(abs(e / t2 - 1) <= tol for e in es) and \
        all(abs(a - b) / t2 <= tol for i, a in enumerate(es) for b in es[i + 1:])


def lex(summary: dict) -> float | None:
    pe = summary.get("P_E_processed")
    return None if not pe or summary.get("logp_ref") is None else summary["logp_ref"] - math.log(pe)


def d1_records(cfg, pop, rows) -> list[dict]:
    e_star, tol = cfg["energy"]["target_edit_norm"], cfg["energy"]["max_pairwise_squared_energy_mismatch"]
    out = []
    for s, lst in pop["d1"].items():
        for c in lst:
            row = rows[c["utterance_id"]]
            rec = row.get("pulses", {}).get(str(c["t"])) if row.get("status") == "ok" else None
            r = {"stratum": s, "dialogue_id": c["dialogue_id"], "utterance_id": c["utterance_id"], "t": c["t"],
                 "span_initial": c.get("span_initial"), "companion": c["companion"],
                 "reference_unit_index": c["reference_unit_index"], "present": rec is not None}
            if rec is None:
                out.append(r); continue
            n = rec["none"]
            r.update(dir=rec["dir"], restore=rec["restore_bitwise"], cos_d_state=rec["cos_d_state"],
                     matched=rec["dir"] == "ok" and energy_ok(rec["arms"], e_star, tol),
                     none_argmax_in_ref=n.get("argmax_in_ref"), none_logp=n.get("logp_ref"),
                     none_rank=n.get("rank_ref"))
            for a in ARMS:
                x = rec["arms"].get(a)
                if x is None:
                    continue
                sm = x["summary"]
                r[a] = {"L": sm["logp_ref"] - n["logp_ref"],
                        "P": math.exp(sm["logp_ref"]) - math.exp(n["logp_ref"]),
                        "Lex": (lex(sm) - lex(n)) if lex(sm) is not None and lex(n) is not None else None,
                        "dPE": sm["P_E"] - n["P_E"], "dPM": sm["P_M"] - n["P_M"],
                        "dPE_proc": sm["P_E_processed"] - n["P_E_processed"],
                        "dMargin": sm["margin_ref_vs_competitor"] - n["margin_ref_vs_competitor"],
                        "rank": sm["rank_ref"], "argmax_in_ref": sm["argmax_in_ref"],
                        "top1_changed": sm["argmax"] != n["argmax"], "kl": sm.get("kl_to_none"),
                        "edit_norm": x["edit_norm"], "consumed_edit_norm": x.get("consumed_edit_norm"),
                        "cos_edit_v": x.get("cos_edit_v"), "relative_sq_energy": x.get("relative_sq_energy"),
                        "solver_matches_hook": x.get("solver_matches_hook"), "solver_status": x["solver"].get("status"),
                        "continuation": x.get("continuation"), "terminated": x.get("terminated")}
            out.append(r)
    return out


def d2_records(pop, rows) -> tuple[list[dict], list[dict], dict]:
    targets, controls, energy = [], [], {"current": 0.0, "oracle": 0.0, "per_utt": {}}
    tkeys = {(c["utterance_id"], c["t"]): c for c in pop["d2_targets"]}
    ckeys = {(c["utterance_id"], c["t"]): c for c in pop["d2_controls"]}
    for uid in pop["utterances"]:
        row = rows[uid]
        if row.get("status") != "ok":
            continue
        cur = {s["t"]: s for s in row["current"]["steps"]}
        ora = {s["t"]: s for s in row["oracle"]["steps"]}
        energy["current"] += row["current"]["total_energy"]
        energy["oracle"] += row["oracle"]["total_energy"]
        energy["per_utt"][uid] = [row["current"]["total_energy"], row["oracle"]["total_energy"]]
        moved_to = {m["to"]: m for m in row["plan"]["moves"]}
        for t, s in cur.items():
            key = (uid, t)
            if key not in tkeys and key not in ckeys or "arm" not in s:
                continue
            o = ora[t]
            base = {"utterance_id": uid, "t": t, "dialogue_id": (tkeys.get(key) or ckeys.get(key))["dialogue_id"],
                    "none_equal_across_passes": s["none"] == o["none"]}
            if key in tkeys:
                n, c_, o_ = s["none"], s["arm"], o["arm"]
                base.update(stratum="EN-confusion", span_initial=tkeys[key].get("span_initial"),
                            L_oracle_current=o_["logp_ref"] - c_["logp_ref"], L_oracle_none=o_["logp_ref"] - n["logp_ref"],
                            L_current_none=c_["logp_ref"] - n["logp_ref"],
                            P_oracle_current=math.exp(o_["logp_ref"]) - math.exp(c_["logp_ref"]),
                            Lex_oracle_current=(lex(o_) - lex(c_)) if lex(o_) is not None and lex(c_) is not None else None,
                            current_correct=c_["argmax_in_ref"], oracle_correct=o_["argmax_in_ref"],
                            none_correct=n["argmax_in_ref"], current_edit=s["edit"], oracle_edit=o["edit"],
                            relocated=t in moved_to, retained=t in row["plan"]["retained"],
                            acoustic=s.get("acoustic"))
                targets.append(base)
            else:
                n, c_, o_ = s["none"], s["arm"], o["arm"]
                base.update(stratum=ckeys[key]["stratum"],
                            dlogp_current=c_["logp_baseline_token"] - n["logp_baseline_token"],
                            dlogp_oracle=o_["logp_baseline_token"] - n["logp_baseline_token"],
                            current_corrupt=c_["argmax"] != n["argmax"], oracle_corrupt=o_["argmax"] != n["argmax"])
                controls.append(base)
    return targets, controls, energy


def transcript_eval(tok, refs, uid, base_tokens, t, cont, unit_index):
    from csasr.evaluation.pier import evaluate_pois
    from csasr.evaluation.retention import matrix_zh_retention
    ref = refs[uid]["reference"]
    base = tok.decode(base_tokens, skip_special_tokens=True)
    hyp = tok.decode(list(base_tokens[:t]) + list(cont), skip_special_tokens=True)
    bp = {p.poi_index: p.correct for p in evaluate_pois(ref, base)}
    hp = {p.poi_index: p.correct for p in evaluate_pois(ref, hyp)}
    zr = matrix_zh_retention([ref], [base], [hyp])
    return {"target_correct": hp.get(unit_index), "baseline_target_correct": bp.get(unit_index),
            "corrections": sum(1 for k, v in bp.items() if not v and hp.get(k)),
            "corruptions": sum(1 for k, v in bp.items() if v and not hp.get(k)),
            "zh_retained": zr["numerator"], "zh_denominator": zr["denominator"], "identical": hyp == base}


# ---- frozen decisions ------------------------------------------------------------------------

def decide(prim: dict, sec: dict, facts: dict) -> dict:
    """Frozen P2-R decision rules (spec section 7)."""
    comp = facts["companion"]
    # D1
    contra_plus = comp["plus_d"]["target_recovered"] < comp["minus_d"]["target_recovered"]
    contra_minus = comp["minus_d"]["target_recovered"] < comp["plus_d"]["target_recovered"]
    if pos(prim["D1.L_plus"]) and pos(prim["D1.L_plus_minus"]) and pos(prim["D1.Lex_plus_minus"]) and not contra_plus:
        d1 = "DIRECTION_POSITIVE_SEMANTICS"
    elif pos(prim["D1.L_minus"]) and neg(prim["D1.L_plus_minus"]) and neg(prim["D1.Lex_plus_minus"]) and not contra_minus:
        d1 = "DIRECTION_SIGN_CONTRADICTION"
    elif any(pos(prim[k]) or neg(prim[k]) for k in ("D1.L_plus", "D1.L_minus", "D1.L_plus_minus", "D1.Lex_plus_minus")):
        d1 = "DIRECTION_UNSTABLE"
    else:
        d1 = "DIRECTION_UNINFORMATIVE"
    # D3
    dp = facts["delta_p"]
    g1 = facts["random_top1_flips"] >= 3 or facts["random_companion_changed"] >= 1
    g2 = not pos(prim["D3.L_plus_random"]) and not pos(prim["D3.L_minus_random"])
    inside = lambda x: x.get("ci") is not None and -dp <= lo(x) and hi(x) <= dp
    g3 = inside(prim["D3.P_plus_random"]) and inside(prim["D3.P_minus_random"]) and \
        contains0(prim["D1.L_plus_minus"]) and contains0(prim["D1.Lex_plus_minus"])
    d3 = "GENERIC_SUPPORTED" if (g1 and g2 and g3) else "GENERIC_NOT_ESTABLISHED"
    # D2
    dp2 = facts["delta_p_d2"]
    comp2 = facts["d2_pulses"]
    if not facts["d2_energy_ok"]:
        d2 = "D2_UNRESOLVED_ENERGY_MISMATCH"
    elif pos(prim["D2.L_oracle_current"]) and pos(prim["D2.L_oracle_none"]) and \
            (sec["D2.Lex_oracle_current"]["estimate"] or 0) > 0 and comp2["oracle_recovered"] >= comp2["current_recovered"]:
        d2 = "D2-A_LOCALIZATION_CONTRIBUTES"
    elif hi(prim["D2.P_oracle_current"]) is not None and hi(prim["D2.P_oracle_current"]) < dp2:
        d2 = "D2-B_NO_MATERIAL_ORACLE_RESCUE"
    else:
        d2 = "D2_UNRESOLVED"
    # profiles
    valid = facts["validity_ok"]
    corr_rate = facts["plus_d_confusion_correction_rate"]
    profiles = []
    script_no_lex = pos(sec["D1.dPE_plus"]) and not pos(prim["D1.L_plus"]) and not pos(prim["D1.Lex_plus_minus"])
    control_resp = pos(prim["D1.L_minus"]) or pos(sec["D1.L_random"])
    if (d1 == "DIRECTION_SIGN_CONTRADICTION" or (script_no_lex and control_resp)) and not d2.startswith("D2-A"):
        profiles.append("DIRECTION_ISSUE")
    if d1 == "DIRECTION_POSITIVE_SEMANTICS" and d2.startswith("D2-A"):
        profiles.append("LOCALIZATION_ISSUE")
    if d3 == "GENERIC_SUPPORTED" and not d2.startswith("D2-A"):
        profiles.append("GENERIC_PERTURBATION")
    if d1 == "DIRECTION_POSITIVE_SEMANTICS" and d2.startswith("D2-B") and corr_rate < 0.10 and facts["edits_nonzero"]:
        profiles.append("SITE_OR_SENSITIVITY_ISSUE")
    if d1 == "DIRECTION_POSITIVE_SEMANTICS" and pos(prim["D3.L_plus_random"]) and d2.startswith("D2-B") and \
            corr_rate >= 0.10 and pos(sec["D2.L_current_none"]) and facts["d2_current_net_correction"] > 0 and \
            not neg(sec["D1.L_plus_EN-correct"]):
        profiles.append("CURRENT_MECHANISM_SUPPORTED")
    primary = profiles[0] if (valid and len(profiles) == 1) else "MECHANISM_STILL_AMBIGUOUS"
    return {"D1": d1, "D2": d2, "D3": d3, "D3_components": {"G1_random_changes": g1, "G2_no_specific_advantage": g2,
                                                            "G3_equivalence": g3},
            "profiles_satisfied": profiles, "validity_ok": valid, "primary_diagnosis": primary}


def main() -> None:
    from transformers import WhisperTokenizer
    import experiments.inference_cf_p2_evaluate as ev

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = ROOT / args.run
    m, cfg, pop, rows = load(run)
    bs = cfg["analysis"]["bootstrap"]
    fam = len(PRIMARY)
    a_sim, a_pw = 0.05 / fam, 0.05
    refs = ev.load_references()
    tok = WhisperTokenizer.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    base_dir = ROOT / cfg["baseline_run"]
    bpanel = json.loads((base_dir / "panel.json").read_text())["rows"]
    bidx = {r["utterance_id"]: i for i, r in enumerate(bpanel)}
    base_rows = {u: json.loads((base_dir / "rows" / f"{bidx[u]:03d}.json").read_text()) for u in pop["utterances"]}
    base_tok = {u: r["systems"][cfg["baseline_system"]]["tokens"] for u, r in base_rows.items()}

    # validity -------------------------------------------------------------------------------
    status = {u: r.get("status") for u, r in rows.items()}
    ok_rows = {u: r for u, r in rows.items() if r.get("status") == "ok"}
    replay_ok = all(s["baseline_argmax_ok"] for r in ok_rows.values() for p in ("current", "oracle")
                    for s in r[p]["steps"]) and all(r[p]["lineage_ok"] for r in ok_rows.values() for p in ("current", "oracle"))
    sched_equal = all([ (a["g"], a["fb"], a["dir"]) for a in r["current"]["steps"]] ==
                      [(b["g"], b["fb"], b["dir"]) for b in r["oracle"]["steps"]] for r in ok_rows.values())
    # current-placement replay vs the saved P2 selected run (identical prefix only)
    p2_match = p2_cmp = 0
    for u, r in ok_rows.items():
        sel = base_rows[u]["systems"].get("L16_a2_ER_id_pos")
        if sel is None:
            continue
        stoks, btoks = sel["tokens"], base_tok[u]
        k = next((i for i, (a, b) in enumerate(zip(stoks, btoks)) if a != b), min(len(stoks), len(btoks)))
        cur = {s["t"]: s for s in r["current"]["steps"]}
        for st in sel["steps"]:
            if st["t"] > k or st["t"] not in cur:
                continue
            p2_cmp += 1
            c = cur[st["t"]]
            p2_match += int(c["g"] == st["g"] and c["edit"] == st["edit"] and abs(c["edit_norm"] - st["edit_norm"]) <= 1e-6)
    d1 = d1_records(cfg, pop, rows)
    restore_ok = all(r.get("restore", True) for r in d1 if r["present"])
    baseline_mismatch = sum(len(r.get("baseline_mismatch", [])) for r in ok_rows.values())
    solver_hook = [r[a]["solver_matches_hook"] for r in d1 if r["present"] for a in ARMS if a in r and r[a]["solver_status"] == "ok"]
    conf = [r for r in d1 if r["stratum"] == "EN-confusion" and r.get("matched")]
    conf_d = {r["dialogue_id"] for r in conf}
    minimum = cfg["population"]["minimum_interpretable"]
    matched_share = {s: (sum(1 for r in d1 if r["stratum"] == s and r.get("matched")) /
                         max(1, sum(1 for r in d1 if r["stratum"] == s))) for s in pop["d1"]}
    validity = {"rows_ok": all(v == "ok" for v in status.values()), "replay_baseline_identity": replay_ok,
                "gate_schedule_identical_across_passes": sched_equal,
                "current_replay_matches_p2_steps": [p2_match, p2_cmp],
                "pulse_restore_bitwise": restore_ok, "pulse_baseline_mismatch_steps": baseline_mismatch,
                "solver_matches_hook_share": (sum(map(bool, solver_hook)) / len(solver_hook)) if solver_hook else None,
                "energy_matched_share": matched_share,
                "confusion_matched_positions": len(conf), "confusion_matched_dialogues": len(conf_d)}
    validity_ok = (validity["rows_ok"] and replay_ok and sched_equal and restore_ok and baseline_mismatch == 0
                   and p2_cmp > 0 and p2_match == p2_cmp
                   and len(conf) >= minimum["positions"] and len(conf_d) >= minimum["dialogues"])

    def boot(vals, alpha):
        return dialogue_bootstrap(vals, bs["replicates"], bs["seed"], alpha)

    def col(stratum, f, sub=None):
        out = []
        for r in d1:
            if r["stratum"] != stratum or not r.get("matched"):
                continue
            if sub is not None and r.get("span_initial") != sub:
                continue
            v = f(r)
            if v is not None:
                out.append((r["dialogue_id"], v))
        return out

    C = "EN-confusion"
    prim = {"D1.L_plus": boot(col(C, lambda r: r["plus_d"]["L"]), a_sim),
            "D1.L_minus": boot(col(C, lambda r: r["minus_d"]["L"]), a_sim),
            "D1.L_plus_minus": boot(col(C, lambda r: r["plus_d"]["L"] - r["minus_d"]["L"]), a_sim),
            "D1.Lex_plus_minus": boot(col(C, lambda r: None if r["plus_d"]["Lex"] is None or r["minus_d"]["Lex"] is None
                                          else r["plus_d"]["Lex"] - r["minus_d"]["Lex"]), a_sim),
            "D3.L_plus_random": boot(col(C, lambda r: r["plus_d"]["L"] - r["random"]["L"]), a_sim),
            "D3.L_minus_random": boot(col(C, lambda r: r["minus_d"]["L"] - r["random"]["L"]), a_sim),
            "D3.P_plus_random": boot(col(C, lambda r: r["plus_d"]["P"] - r["random"]["P"]), a_sim),
            "D3.P_minus_random": boot(col(C, lambda r: r["minus_d"]["P"] - r["random"]["P"]), a_sim)}
    targets, controls, energy = d2_records(pop, rows)
    tv = lambda f: [(t["dialogue_id"], f(t)) for t in targets if f(t) is not None]
    prim["D2.L_oracle_current"] = boot(tv(lambda t: t["L_oracle_current"]), a_sim)
    prim["D2.L_oracle_none"] = boot(tv(lambda t: t["L_oracle_none"]), a_sim)
    prim["D2.P_oracle_current"] = boot(tv(lambda t: t["P_oracle_current"]), a_sim)
    sec = {"D1.L_random": boot(col(C, lambda r: r["random"]["L"]), a_pw),
           "D1.dPE_plus": boot(col(C, lambda r: r["plus_d"]["dPE"]), a_pw),
           "D2.Lex_oracle_current": boot(tv(lambda t: t["Lex_oracle_current"]), a_pw),
           "D2.L_current_none": boot(tv(lambda t: t["L_current_none"]), a_pw)}
    for s in pop["d1"]:
        for a in ARMS:
            for f in ("L", "P", "Lex", "dPE", "dPM", "dMargin", "kl"):
                sec[f"D1.{f}_{a}_{s}"] = boot(col(s, lambda r, a=a, f=f: r[a][f]), a_pw)
        sec[f"D1.L_plus_{s}"] = sec[f"D1.L_plus_d_{s}"]
        sec[f"D1.L_plus_minus_{s}"] = boot(col(s, lambda r: r["plus_d"]["L"] - r["minus_d"]["L"]), a_pw)
    for sub, name in ((True, "span_initial"), (False, "within_span")):
        for a in ARMS:
            sec[f"D1.L_{a}_{name}"] = boot(col(C, lambda r, a=a: r[a]["L"], sub), a_pw)
        sec[f"D1.L_plus_minus_{name}"] = boot(col(C, lambda r: r["plus_d"]["L"] - r["minus_d"]["L"], sub), a_pw)
    for s in ("EN-correct", "ZH-correct"):
        sec[f"D2.dlogp_current_{s}"] = boot([(c["dialogue_id"], c["dlogp_current"]) for c in controls if c["stratum"] == s], a_pw)
        sec[f"D2.dlogp_oracle_{s}"] = boot([(c["dialogue_id"], c["dlogp_oracle"]) for c in controls if c["stratum"] == s], a_pw)

    # counts ---------------------------------------------------------------------------------
    counts = {}
    for s in pop["d1"]:
        rs = [r for r in d1 if r["stratum"] == s and r.get("matched")]
        counts[s] = {"matched": len(rs), "dialogues": len({r["dialogue_id"] for r in rs})}
        for a in ARMS:
            counts[s][a] = {"top1_changed": sum(r[a]["top1_changed"] for r in rs),
                            "argmax_in_ref": sum(bool(r[a]["argmax_in_ref"]) for r in rs),
                            "none_argmax_in_ref": sum(bool(r["none_argmax_in_ref"]) for r in rs),
                            "median_rank": float(np.median([r[a]["rank"] for r in rs])) if rs else None}
    companion = {}
    for a in ARMS:
        agg = {"target_recovered": 0, "corrections": 0, "corruptions": 0, "zh_lost": 0, "changed": 0, "n": 0}
        per_stratum = {}
        for r in d1:
            if not r["present"] or not r["companion"] or a not in r or r[a].get("continuation") is None:
                continue
            e = transcript_eval(tok, refs, r["utterance_id"], base_tok[r["utterance_id"]], r["t"],
                                r[a]["continuation"], r["reference_unit_index"])
            ps = per_stratum.setdefault(r["stratum"], {"target_recovered": 0, "target_lost": 0, "corrections": 0,
                                                       "corruptions": 0, "zh_lost": 0, "changed": 0, "n": 0})
            ps["n"] += 1; ps["changed"] += int(not e["identical"])
            ps["corrections"] += e["corrections"]; ps["corruptions"] += e["corruptions"]
            ps["zh_lost"] += e["zh_denominator"] - e["zh_retained"]
            if r["stratum"] == C:
                ps["target_recovered"] += int(bool(e["target_correct"]))
            else:
                ps["target_lost"] += int(e["baseline_target_correct"] and not e["target_correct"])
        agg["target_recovered"] = per_stratum.get(C, {}).get("target_recovered", 0)
        companion[a] = {**agg, "per_stratum": per_stratum}
    d2_pulses = {"current_recovered": 0, "oracle_recovered": 0, "n": 0, "details": []}
    tkeys = {(c["utterance_id"], c["t"]): c for c in pop["d2_targets"]}
    for u, r in ok_rows.items():
        for mv in r["pulse_companions"]:
            tgt = tkeys[(u, mv["to"])]
            cur = r["pulses"][str(mv["from"])]["arms"][f"pulse_current_{mv['to']}"]
            ora = r["pulses"][str(mv["to"])]["arms"][f"pulse_oracle_{mv['to']}"]
            ec = transcript_eval(tok, refs, u, base_tok[u], mv["from"], cur["continuation"], tgt["reference_unit_index"])
            eo = transcript_eval(tok, refs, u, base_tok[u], mv["to"], ora["continuation"], tgt["reference_unit_index"])
            d2_pulses["n"] += 1
            d2_pulses["current_recovered"] += int(bool(ec["target_correct"]))
            d2_pulses["oracle_recovered"] += int(bool(eo["target_correct"]))
            d2_pulses["details"].append({"utterance_id": u, "move": mv, "current": ec, "oracle": eo,
                                         "energies": [cur["edit_norm"], ora["edit_norm"]]})
    n_conf = len(conf)
    rand_flips = sum(r["random"]["top1_changed"] for r in d1 if r.get("matched"))
    facts = {"delta_p": 1.0 / max(1, n_conf), "delta_p_d2": 1.0 / max(1, len(targets)),
             "companion": companion, "d2_pulses": d2_pulses,
             "random_top1_flips": rand_flips,
             "random_companion_changed": sum(ps["changed"] for ps in companion["random"]["per_stratum"].values()),
             "plus_d_confusion_correction_rate": (counts[C]["plus_d"]["argmax_in_ref"] / n_conf) if n_conf else 0.0,
             "edits_nonzero": all(r[a]["edit_norm"] > 0 for r in d1 if r.get("matched") for a in ARMS),
             "d2_current_net_correction": sum(bool(t["current_correct"]) and not t["none_correct"] for t in targets)
             - sum(c["current_corrupt"] for c in controls),
             "validity_ok": validity_ok,
             "d2_energy_ok": bool(energy["current"]) and abs(energy["oracle"] / energy["current"] - 1)
             <= cfg["d2"]["total_energy_tolerance"]}
    decision = decide(prim, sec, facts)
    acoustic = [t["acoustic"] for t in targets if t.get("acoustic") and t["acoustic"].get("status") == "ok"]
    summary = {"schema": "p2r_analysis_v1", "manifest_hash": m["manifest_hash"], "validity": validity,
               "primary_family": list(PRIMARY), "bonferroni_alpha": a_sim, "primary": prim, "secondary": sec,
               "counts": counts, "facts": {k: v for k, v in facts.items() if k != "d2_pulses"},
               "d2_pulses": d2_pulses,
               "d2": {"targets": len(targets), "relocated": sum(t["relocated"] for t in targets),
                      "retained": sum(t["retained"] for t in targets),
                      "current_edited": sum(t["current_edit"] for t in targets),
                      "oracle_edited": sum(t["oracle_edit"] for t in targets),
                      "current_correct": sum(bool(t["current_correct"]) for t in targets),
                      "oracle_correct": sum(bool(t["oracle_correct"]) for t in targets),
                      "none_correct": sum(bool(t["none_correct"]) for t in targets),
                      "controls_corrupt_current": sum(c["current_corrupt"] for c in controls),
                      "controls_corrupt_oracle": sum(c["oracle_corrupt"] for c in controls),
                      "controls": len(controls), "none_equal_across_passes": all(t["none_equal_across_passes"] for t in targets + controls),
                      "total_energy": {"current": energy["current"], "oracle": energy["oracle"],
                                       "ratio": energy["oracle"] / energy["current"] if energy["current"] else None},
                      "acoustic": {"n": len(acoustic),
                                   "median_timing_error_sec": float(np.median([a["timing_error_sec"] for a in acoustic if a["timing_error_sec"] is not None])) if acoustic else None,
                                   "mean_E_current": float(np.mean([a["E_current"] for a in acoustic if a["E_current"] is not None])) if acoustic else None,
                                   "mean_E_oracle": float(np.mean([a["E_oracle"] for a in acoustic])) if acoustic else None}},
               "decision": decision}
    atomic_json(ROOT / args.out, summary)
    print(json.dumps({"validity_ok": validity_ok, "decision": decision,
                      "primary": {k: [v["estimate"], v["ci"]] for k, v in prim.items()}}, indent=2))


if __name__ == "__main__":
    main()
