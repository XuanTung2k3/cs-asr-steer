#!/usr/bin/env python
"""DIR-SPRINT0 post-seal evaluator (CPU; separate process; never imported by the runner).

Opens the selected-role reference column ONLY after the pushed pulse seal and a pushed
``DIR_SPRINT0_AUDIT_PRIMARY: PASS``, filtering the role parquet to the frozen 240 IDs before the reference column is
materialized. Reference units are mapped by the unchanged SRD2-G0 evaluator mapping (canonical normalize/segment/align,
R2 ``unit_to_token_positions``, P2-R ``target_set``; no CTC/MMS-FA timing). Counts actual first-token corrections and
correct-state corruptions for every arm from the sealed processed top-1 decisions, the severe-EOS proxy, energy validity,
family coverage, the mechanistic gap metric, the paired dialogue bootstrap, every frozen family predicate and the
first-matching provisional terminal label (confirmed only by the independent FULL audit). No arm, comparator, metric,
threshold or position is chosen from outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, file_hash

CONFIG = "configs/inference_cf/dir_sprint0.json"
CONFIG_SHA = "sha256:b264460beedd09ae4b0c005ef5f8178494c49a52aa9ba2bec4bc116c984d27db"
POPULATION = "docs/inference_cf/DIR_SPRINT0_POPULATION.json"
RUN = "results/inference_cf/dir_sprint0/run1"
REMOTE = "origin/cs-asr-steer-inf"
MODEL = "/mnt/data/tungnx/whisper-large-v3"
EOS = 50257
PULSE_ARMS = ("D0", "D1", "D2", "VAC", "RND", "D3", "D4", "D5", "D5SH", "D2G", "D3G", "D4G", "D5G", "D5SHG", "RNDG")
ARMS = PULSE_ARMS + ("D3CD",)
GATED_BASE = {"D2G": "D2", "D3G": "D3", "D4G": "D4", "D5G": "D5", "D5SHG": "D5SH", "RNDG": "RND"}
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
FAMILIES = ("D3", "D4", "D5")
VARIANTS = {"D3": ("D3", "D3G"), "D4": ("D4", "D4G"), "D5": ("D5", "D5G")}


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def pushed(rel: str) -> bool:
    import hashlib
    if git("ls-files", rel) != rel:
        return False
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    if "sha256:" + hashlib.sha256(blob).hexdigest() != file_hash(ROOT / rel):
        return False
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


def ratio(a, b):
    return None if b in (0, None) or a is None else a / b


def from_bf16(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int16)
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


# ---- references (selected 240 only) -----------------------------------------------------------------------------------

def load_references(ids: list[str]) -> tuple[dict, dict]:
    """Filter the role parquet to the frozen IDs BEFORE the reference column is materialized."""
    import pyarrow.dataset as ds
    P = json.loads((ROOT / POPULATION).read_text())
    role = P["source"]
    if file_hash(role) != P["source_sha256"]:
        raise ValueError("role parquet changed")
    table = ds.dataset(role, format="parquet").to_table(columns=["utterance_id", "role", "transcript_raw"],
                                                         filter=ds.field("utterance_id").isin(ids))
    rows = table.to_pylist()
    if sorted(r["utterance_id"] for r in rows) != sorted(ids) or any(r["role"] != "D-dev-select" for r in rows):
        raise ValueError("reference rows differ from the frozen selection")
    access = {"path": role, "sha256": P["source_sha256"], "columns": ["utterance_id", "role", "transcript_raw"],
              "filter": "utterance_id in frozen selected240 (pushdown before materialization)", "rows": len(rows)}
    return {r["utterance_id"]: r["transcript_raw"] for r in rows}, access


# ---- outcomes ------------------------------------------------------------------------------------------------------------

def arm_top1(rec: dict, cap_q: dict, arm: str) -> tuple[int, bool]:
    """(top-1, executed) of one arm at one inventory query; D3CD = c_AP where D3 status is 'candidate'."""
    if arm == "D3CD":
        b0 = rec["arms"]["B0"]["top1"]
        if rec["structural"] and cap_q.get("D3", {}).get("status") == "candidate":
            return int(cap_q["D3"]["c_AP"]), True
        return int(b0), False
    r = rec["arms"][arm]
    return int(r["top1"]), bool(r["executed"])


def outcomes(queries: list[dict], pulses: dict, cap: dict, dlg: dict) -> list[dict]:
    out = []
    for qd in queries:
        uid = qd["utterance_id"]
        p = pulses[uid]
        i = p["inventory"].index(qd["t"])
        rec, cq = p["queries"][i], cap[uid]["queries"][i]
        if rec["t"] != qd["t"] or cq["t"] != qd["t"]:
            raise RuntimeError("query order")
        b0 = int(rec["arms"]["B0"]["top1"])
        Y = set(qd["Y"])
        row = {"utterance_id": uid, "dialogue_id": dlg[uid], "t": qd["t"], "stratum": qd["stratum"], "Y": sorted(Y),
               "structural": rec["structural"], "b0_top1": b0, "b0_in_Y": b0 in Y, "arms": {}}
        for a in ARMS:
            top, ex = arm_top1(rec, cq, a)
            ev = {"top1": top, "executed": ex, "in_Y": top in Y, "correction": False, "corruption": False}
            if qd["stratum"] == "EN-confusion":
                ev["correction"] = (b0 not in Y) and (top in Y)
            else:
                ev["corruption"] = (b0 in Y) and (top not in Y)
            row["arms"][a] = ev
        out.append(row)
    return out


def tally(rows: list[dict], dialogues: list[str]) -> dict:
    T = {d: {"n_C": 0, "n_EN": 0, "n_ZH": 0, **{f"{a}_{k}": 0 for a in ARMS for k in ("C", "H_EN", "H_ZH", "act_ZH")}}
         for d in dialogues}
    for r in rows:
        d = T[r["dialogue_id"]]
        d[{"EN-confusion": "n_C", "EN-correct": "n_EN", "ZH-correct": "n_ZH"}[r["stratum"]]] += 1
        for a in ARMS:
            ev = r["arms"][a]
            if r["stratum"] == "EN-confusion":
                d[f"{a}_C"] += int(ev["correction"])
            elif r["stratum"] == "EN-correct":
                d[f"{a}_H_EN"] += int(ev["corruption"])
            else:
                d[f"{a}_H_ZH"] += int(ev["corruption"])
                d[f"{a}_act_ZH"] += int(ev["executed"])
    return T


# ---- reference-free integrity / energy / coverage ---------------------------------------------------------------------

def integrity(cfg: dict, pulses: dict, cap: dict, cons: dict, blocked: dict) -> dict:
    struct = [(u, x) for u, p in pulses.items() for x in p["queries"] if x["structural"]]
    cells = [(u, x, a, x["arms"][a]) for u, x in struct for a in PULSE_ARMS]
    ex = [c for c in cells if c[3]["executed"]]
    sq, ch, cp = [], [], []
    for _, _, _, r in ex:
        g = r["guards"]
        cons_c = r["consumed_chord_actual"]
        sq += [g["proposed_rel_sq_err"], abs(cons_c * cons_c / r["target"] ** 2 - 1.0)]
        ch += [g["proposed_rel_chord_err"], abs(cons_c / r["target"] - 1.0)]
        cp.append(abs(cons_c / g["proposed_chord"] - 1.0))
    clean = [x["clean"]["logits_bitwise_vs_A"] and x["clean"]["site_bitwise_vs_A"] for p in pulses.values() for x in p["queries"]]
    restore = [x["restore_bitwise"] for _, x in struct if x["restore_bitwise"] is not None]
    scratch2 = [x["d2"]["scratch_logits_bitwise"] and x["d2"]["scratch_site_bitwise"] for _, x in struct]
    scratch3 = [x["d3"]["scratch_logits_bitwise"] and x["d3"]["scratch_site_bitwise"] for _, x in struct if "scratch_logits_bitwise" in x["d3"]]
    critical = {"executed_integrity": all(r["integrity_ok"] for *_, r in ex),
                "clean_bitwise_vs_A": bool(clean) and all(clean), "restore_bitwise": all(restore),
                "D2_scratch_identity": bool(scratch2) and all(scratch2), "D3_scratch_identity": all(scratch3),
                "energy_rel_sq_err_max_ok": (max(sq) if sq else 0.0) <= cfg["dose"]["relative_squared_energy_error_max"],
                "chord_err_max_ok": (max(ch) if ch else 0.0) <= cfg["dose"]["relative_chord_error_max"],
                "consumed_vs_proposed_ok": (max(cp) if cp else 0.0) <= cfg["dose"]["consumed_vs_proposed_relative_error_max"]}
    per_arm = {}
    for a in PULSE_ARMS:
        cs = [(u, x, r) for u, x, aa, r in cells if aa == a]
        req = [r for _, _, r in cs if r["target"] > 0 and r["status"] not in ("invalid_direction", "zero_target")]
        exe = [r for r in req if r["executed"]]
        planned = math.fsum(sorted(r["target"] ** 2 for _, _, r in cs if r["target"] > 0 and r["status"] != "invalid_direction"))
        lost = math.fsum(sorted(r["target"] ** 2 for r in req if not r["executed"]))
        realized = math.fsum(sorted(r["consumed_chord_actual"] ** 2 for r in exe))
        per_arm[a] = {"cells": len(cs), "requests_with_valid_direction": len(req), "executed": len(exe),
                      "invalid_direction": sum(r["status"] == "invalid_direction" for _, _, r in cs),
                      "zero_target": sum(r["status"] == "zero_target" for _, _, r in cs),
                      "matched_fraction": ratio(len(exe), len(req)), "planned_sq": planned, "lost_planned_sq": lost,
                      "lost_fraction": ratio(lost, planned) if planned > 0 else None, "realized_sq": realized,
                      "status_counts": dict(Counter(r["status"] for _, _, r in cs))}
    mf = cfg["dose"]["ungated_nonzero_requests_matched_fraction_min"]
    for a in ("D2", "RND"):
        critical[f"{a}_energy_valid"] = per_arm[a]["matched_fraction"] is not None and per_arm[a]["matched_fraction"] >= mf
    return {"critical": critical, "per_arm": per_arm, "executed_cells": len(ex), "structural_queries": len(struct),
            "max_rel_sq_err": max(sq) if sq else None, "max_chord_err": max(ch) if ch else None,
            "max_consumed_vs_proposed": max(cp) if cp else None}


def energy_ok(cfg: dict, integ: dict, arm: str) -> bool:
    pa = integ["per_arm"][arm]
    if arm in GATED_BASE:
        return pa["lost_fraction"] is not None and pa["lost_fraction"] <= cfg["dose"]["gated_planned_squared_energy_lost_max"]
    return pa["matched_fraction"] is not None and pa["matched_fraction"] >= cfg["dose"]["ungated_nonzero_requests_matched_fraction_min"]


def constructible(fam: str, cons_status: dict, pulse_rec: dict) -> bool:
    if fam == "D3":
        s = pulse_rec["d3"]["status"]
        return s == "ok" or s == "no_edit:baseline_is_candidate"
    if fam == "D4":
        return cons_status["D4"]["status"] == "ok"
    return cons_status["D5"]["status"] == "ok"


def coverage(pulses: dict, cons: dict, rows: list[dict], blocked: dict) -> dict:
    out = {}
    en_keys = {(r["utterance_id"], r["t"]) for r in rows if r["stratum"] == "EN-confusion" and r["structural"]}
    for fam in FAMILIES:
        n = k = n_en = k_en = 0
        for uid, p in pulses.items():
            st = {s["t"]: s for s in cons[uid]["status"]}
            for x in p["queries"]:
                if not x["structural"]:
                    continue
                ok = (not blocked[fam]) and constructible(fam, st[x["t"]], x)
                n += 1
                k += int(ok)
                if (uid, x["t"]) in en_keys:
                    n_en += 1
                    k_en += int(ok)
        out[fam] = {"structural": n, "constructible": k, "fraction_structural": ratio(k, n),
                    "EN_confusion_structural": n_en, "EN_confusion_constructible": k_en, "fraction_EN_confusion": ratio(k_en, n_en)}
    return out


# ---- mechanistic gap -----------------------------------------------------------------------------------------------------

def processed(z: np.ndarray, suppress: list[int]) -> np.ndarray:
    x = np.asarray(z, dtype=np.float32).copy()
    x[suppress] = -np.inf
    return x


def gap(z: np.ndarray, Y: list[int]) -> float:
    m = np.zeros(z.size, dtype=bool)
    m[Y] = True
    return float(np.max(z[~m]).astype(np.float64) - np.max(z[m]).astype(np.float64))


def gaps(rows: list[dict], pulses: dict, run: Path, suppress: list[int]) -> dict:
    """G per arm on mapped structural EN-confusion queries (no-edit cells keep G_B0)."""
    by_uid: dict = {}
    for r in rows:
        if r["stratum"] == "EN-confusion" and r["structural"]:
            by_uid.setdefault(r["utterance_id"], []).append(r)
    out = {}
    for uid, rs in by_uid.items():
        p = pulses[uid]
        with np.load(run / "capture" / f"{uid}.npz") as z:
            cached = z["cached_logits"]
        with np.load(run / "pulses" / p["logits"]["file"]) as z:
            keys, ex = z["exec_keys"], z["exec_logits"]
        exec_of = {(int(i), int(a)): k for k, (i, a) in enumerate(keys.tolist())}
        for r in rs:
            i = p["inventory"].index(r["t"])
            g0 = gap(processed(from_bf16(cached[i]), suppress), r["Y"])
            if not math.isfinite(g0):            # an acceptable token is generation-suppressed: gap undefined, query skipped
                continue
            rec = {"B0": g0}
            for a in PULSE_ARMS:
                k = exec_of.get((i, PULSE_ARMS.index(a)))
                rec[a] = g0 if k is None else gap(processed(from_bf16(ex[k]), suppress), r["Y"])
            if all(math.isfinite(v) for v in rec.values()):
                out[(uid, r["t"])] = rec
    return out


# ---- statistics -----------------------------------------------------------------------------------------------------------

def bootstrap(cfg: dict, T: dict, dialogues: list[str], gmap: dict, dlg: dict, comps: dict, mech: dict) -> dict:
    st = cfg["statistics"]
    rng = np.random.default_rng(st["seed"])
    D = len(dialogues)
    draws = rng.integers(0, D, size=(st["bootstrap_replicates"], D))
    keys = sorted(next(iter(T.values())).keys())
    M = np.array([[T[d][k] for k in keys] for d in dialogues], dtype=np.float64)
    Ssum = M[draws].sum(axis=1)
    col = {k: Ssum[:, i] for i, k in enumerate(keys)}
    U = {a: col[f"{a}_C"] - col[f"{a}_H_ZH"] - col[f"{a}_H_EN"] for a in ARMS}
    lo, hi = st["family_quantiles"]

    def iv(x, a=.025, b=.975):
        return [float(np.quantile(x, a)), float(np.quantile(x, b))]
    fam = {}
    for name, (kind, a, b) in comps.items():
        x = (col[f"{a}_C"] - col[f"{b}_C"]) if kind == "C" else (U[a] - U[b])
        fam[name] = {"pct95": iv(x), "bonferroni": iv(x, lo, hi)}
    # mechanistic: dialogue-macro mean of (Delta_v - Delta_RND(matched)) = G_RNDm - G_v
    mq = cfg["mechanistic"]["quantiles"]
    mech_out = {}
    for v, rnd in mech.items():
        per_d = {d: [] for d in dialogues}
        for (uid, t), g in gmap.items():
            per_d[dlg[uid]].append(g[rnd] - g[v])
        means = np.array([np.mean(per_d[d]) if per_d[d] else np.nan for d in dialogues])
        has = ~np.isnan(means)
        point = float(np.mean(means[has])) if has.any() else None
        sel = means[draws]
        hv = has[draws]
        with np.errstate(invalid="ignore"):
            bs = np.nansum(sel, axis=1) / hv.sum(axis=1)
        okb = hv.sum(axis=1) > 0
        mech_out[v] = {"point": point, "dialogues": int(has.sum()), "valid_draws": int(okb.sum()),
                       "pct95": iv(bs[okb]) if okb.sum() else None, "bonferroni": iv(bs[okb], mq[0], mq[1]) if okb.sum() else None}
    rates = {}
    for a in ARMS:
        for name, num, den in (("ZH_corruption_rate", f"{a}_H_ZH", "n_ZH"), ("EN_corruption_rate", f"{a}_H_EN", "n_EN"),
                               ("correction_rate", f"{a}_C", "n_C"), ("ZH_active_corruption_rate", f"{a}_H_ZH", f"{a}_act_ZH")):
            dv = col[den]
            ok = dv > 0
            rates[f"{name}_{a}"] = {"valid_draws": int(ok.sum()),
                                    "pct95": iv(col[num][ok] / dv[ok]) if ok.sum() >= 9500 else "NOT_ESTIMABLE"}
    return {"replicates": int(st["bootstrap_replicates"]), "seed": st["seed"], "family": fam, "mechanistic": mech_out,
            "counts_pct95": {k: iv(col[k]) for k in keys if not k.startswith("n_")}, "utility_pct95": {a: iv(U[a]) for a in ARMS},
            "rates": rates}


def comparisons(cfg: dict) -> dict:
    comps = {}
    mc = cfg["matched_comparators"]
    for v in ("D3", "D3G", "D4", "D4G", "D5", "D5G"):
        for ref in ("D2", "RND"):
            comps[f"C_{v}-C_{mc[v][ref]}"] = ("C", v, mc[v][ref])
            comps[f"U_{v}-U_{mc[v][ref]}"] = ("U", v, mc[v][ref])
    for v in ("D5", "D5G"):
        comps[f"C_{v}-C_{mc[v]['SH']}"] = ("C", v, mc[v]["SH"])
        comps[f"U_{v}-U_{mc[v]['SH']}"] = ("U", v, mc[v]["SH"])
    for v in ("D3", "D3G"):
        comps[f"U_{v}-U_D3CD"] = ("U", v, "D3CD")
    if len(comps) != cfg["statistics"]["family_size"]:
        raise RuntimeError("comparison family size differs from the freeze")
    return comps


# ---- predicates ------------------------------------------------------------------------------------------------------------

def arm_metrics(T: dict, dialogues: list[str], pulses: dict, cap: dict) -> dict:
    def tot(k):
        return sum(T[d][k] for d in dialogues)
    nZH, nEN, nC = tot("n_ZH"), tot("n_EN"), tot("n_C")
    out = {}
    for a in ARMS:
        C, HZ, HE, act = tot(f"{a}_C"), tot(f"{a}_H_ZH"), tot(f"{a}_H_EN"), tot(f"{a}_act_ZH")
        cd = {d: T[d][f"{a}_C"] for d in dialogues}
        out[a] = {"C": C, "H_ZH": HZ, "H_EN": HE, "U": C - HZ - HE, "active_ZH": act,
                  "active_ZH_dialogues": sum(1 for d in dialogues if T[d][f"{a}_act_ZH"] > 0),
                  "correction_dialogues": sum(1 for v in cd.values() if v > 0),
                  "max_correction_dialogue_share": (max(cd.values()) / C) if C > 0 else None,
                  "ZH_harm_dialogues": sum(1 for d in dialogues if T[d][f"{a}_H_ZH"] > 0),
                  "EN_harm_dialogues": sum(1 for d in dialogues if T[d][f"{a}_H_EN"] > 0),
                  "ZH_rate": ratio(HZ, nZH), "ZH_active_rate": ratio(HZ, act), "EN_rate": ratio(HE, nEN),
                  "correction_rate": ratio(C, nC),
                  "damage_dialogue": any(T[d][f"{a}_H_ZH"] >= 3 and T[d][f"{a}_act_ZH"] > 0
                                         and T[d][f"{a}_H_ZH"] / T[d][f"{a}_act_ZH"] > .20 for d in dialogues)}
    for a in ARMS:
        n = 0
        for uid, p in pulses.items():
            T_ = len(cap[uid]["baseline"]["content_ids"])
            for i, x in enumerate(p["queries"]):
                top, _ = arm_top1(x, cap[uid]["queries"][i], a)
                if top == EOS and x["arms"]["B0"]["top1"] != EOS and T_ - x["t"] >= 10:
                    n += 1
        out[a]["EOS_severe"] = n
    return out


def paired_gain_dialogues(T: dict, dialogues: list[str], a: str, b: str) -> int:
    return sum(1 for d in dialogues if T[d][f"{a}_C"] - T[d][f"{b}_C"] > 0)


def variant_eval(cfg: dict, v: str, fam: str, A: dict, T: dict, dialogues: list[str], integ: dict, opp: dict, boot: dict) -> dict:
    mc = cfg["matched_comparators"][v]
    pw, ad, se, sf = cfg["power"], cfg["advantage"], cfg["safety_estimability"], cfg["safety"]
    a = A[v]
    leaves = {}
    leaves["energy"] = {"energy_valid": energy_ok(cfg, integ, v)}
    leaves["power"] = {"corrections": a["C"] >= pw["corrections_min"],
                       "correction_dialogues": a["correction_dialogues"] >= pw["correction_dialogues_min"],
                       "max_correction_dialogue_share": a["max_correction_dialogue_share"] is not None
                       and a["max_correction_dialogue_share"] <= pw["max_correction_dialogue_share"]}
    adv = {"over_D2": a["C"] - A[mc["D2"]]["C"] >= ad["over_D2_corrections_min"],
           "over_RND": a["C"] - A[mc["RND"]]["C"] >= ad["over_RND_corrections_min"],
           "gain_dialogues_vs_D2": paired_gain_dialogues(T, dialogues, v, mc["D2"]) >= ad["positive_paired_correction_gain_vs_D2_dialogues_min"]}
    if fam == "D5":
        adv["over_shuffle"] = a["C"] - A[mc["SH"]]["C"] >= ad["D5_over_shuffle_corrections_min"]
        adv["gain_dialogues_vs_shuffle"] = paired_gain_dialogues(T, dialogues, v, mc["SH"]) >= ad["D5_positive_paired_gain_vs_shuffle_dialogues_min"]
    if fam == "D3":
        adv["utility_over_D3CD"] = a["U"] - A["D3CD"]["U"] >= ad["D3_utility_minus_D3CD_min"]
    leaves["advantage"] = adv
    leaves["estimability"] = {"active_ZH": a["active_ZH"] >= se["active_ZH_correct_min"],
                              "active_ZH_dialogues": a["active_ZH_dialogues"] >= se["active_ZH_correct_dialogues_min"]}
    leaves["safety"] = {"utility": a["U"] >= sf["utility_min"],
                        "ZH_rate": a["ZH_rate"] is not None and a["ZH_rate"] <= sf["unconditional_ZH_corruption_rate_max"],
                        "ZH_active_rate": a["ZH_active_rate"] is not None and a["ZH_active_rate"] <= sf["active_ZH_corruption_rate_max"],
                        "EN_corruptions": a["H_EN"] <= sf["EN_corruptions_max"],
                        "EN_rate": a["EN_rate"] is not None and a["EN_rate"] <= sf["unconditional_EN_corruption_rate_max"],
                        "EOS": a["EOS_severe"] <= sf["EOS_severe_events_max"], "no_damage_dialogue": not a["damage_dialogue"]}
    mech = boot["mechanistic"].get(v, {})
    leaves["mechanistic"] = {"point": mech.get("point") is not None and mech["point"] >= cfg["mechanistic"]["point_min_nats"],
                             "adjusted_lower": mech.get("bonferroni") is not None and mech["bonferroni"][0] > cfg["mechanistic"]["adjusted_lower_gt"]}
    passes = {k: all(v_.values()) for k, v_ in leaves.items()}
    return {"variant": v, "leaves": leaves, "passes": passes}


def family_status(cfg: dict, fam: str, blocked: bool, cov: dict, variants: dict) -> dict:
    c = cfg["coverage"]
    if blocked:
        return {"validity": "PROVIDER_OR_CONSTRUCTION_BLOCKED", "outcome": "NOT_TESTED", "qualifying_variant": None,
                "mechanistic_effect": False}
    fs, fe = cov["fraction_structural"], cov["fraction_EN_confusion"]
    if fs is None or fe is None or fs < c["constructible_fraction_structural_min"] or fe < c["constructible_fraction_EN_confusion_min"]:
        return {"validity": "INSUFFICIENT_COVERAGE", "outcome": "NOT_TESTED", "qualifying_variant": None, "mechanistic_effect": False}
    order = [VARIANTS[fam][0], VARIANTS[fam][1]]
    pa = {v: variants[v]["passes"] for v in order}
    core = {v: pa[v]["energy"] and pa[v]["power"] and pa[v]["advantage"] for v in order}
    mech = any(pa[v]["energy"] and pa[v]["mechanistic"] for v in order)
    for v in order:
        if core[v] and pa[v]["estimability"] and pa[v]["safety"]:
            return {"validity": "VALID_AND_TESTED", "outcome": "PROMISING_FEASIBILITY", "qualifying_variant": v, "mechanistic_effect": mech}
    for v in order:
        if core[v] and pa[v]["estimability"]:
            return {"validity": "VALID_AND_TESTED", "outcome": "SAFETY_FAILED", "qualifying_variant": v, "mechanistic_effect": mech}
    for v in order:
        if core[v]:
            return {"validity": "VALID_AND_TESTED", "outcome": "INSUFFICIENT_COVERAGE", "qualifying_variant": v,
                    "outcome_reason": "safety_not_estimable", "mechanistic_effect": mech}
    return {"validity": "VALID_AND_TESTED", "outcome": "LEXICAL_POWER_INSUFFICIENT", "qualifying_variant": None, "mechanistic_effect": mech}


def decide(flags: dict, statuses: dict) -> str:
    if flags["critical_failure"] or all(s["validity"] == "PROVIDER_OR_CONSTRUCTION_BLOCKED" for s in statuses.values()):
        return "DIR_SPRINT0_INVALID"
    if not flags["resource_pass"]:
        return "DIR_SPRINT0_COMPUTE_BLOCKED"
    if not flags["opportunity_pass"] or not any(s["validity"] == "VALID_AND_TESTED" for s in statuses.values()):
        return "DIR_SPRINT0_OPPORTUNITY_INSUFFICIENT"
    if any(s["outcome"] == "PROMISING_FEASIBILITY" for s in statuses.values()):
        return "DIR_SPRINT0_STEERING_FEASIBILITY_SIGNAL"
    if any(s["outcome"] == "SAFETY_FAILED" for s in statuses.values()):
        return "DIR_SPRINT0_CAUSAL_POWER_WITH_DAMAGE"
    if any(s["validity"] == "VALID_AND_TESTED" and s["mechanistic_effect"] for s in statuses.values()):
        return "DIR_SPRINT0_MECHANISTIC_EFFECT_ONLY"
    return "DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE"


def common_eligible(rows: list[dict], a: str, b: str) -> dict:
    sel = [r for r in rows if r["arms"][a]["executed"] and r["arms"][b]["executed"]]
    return {"queries": len(sel),
            "EN_confusion": sum(r["stratum"] == "EN-confusion" for r in sel),
            f"C_{a}": sum(r["arms"][a]["correction"] for r in sel), f"C_{b}": sum(r["arms"][b]["correction"] for r in sel),
            f"H_{a}": sum(r["arms"][a]["corruption"] for r in sel), f"H_{b}": sum(r["arms"][b]["corruption"] for r in sel)}


def evaluate(cfg: dict, P: dict, cap: dict, cons: dict, pulses: dict, refs: dict, tok, side: dict, run: Path) -> dict:
    from experiments.inference_cf_srd2_g0_evaluate import map_utterance
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    dur = {r["utterance_id"]: float(r["source_duration_sec"]) for r in P["selected"]}
    dialogues = sorted(set(dlg.values()))
    all_q, ledgers, units_all = [], {}, []
    for uid in [r["utterance_id"] for r in P["selected"]]:
        g = cap[uid]
        qs, led = map_utterance(tok, uid, refs[uid], g["baseline"]["content_ids"], g["baseline"]["terminated"], dur[uid])
        all_q += qs
        units_all += led.pop("units")
        ledgers[uid] = led
    rows = outcomes(all_q, pulses, cap, dlg)
    T = tally(rows, dialogues)
    seal_a, fc, prt, app = side["job_A_seal"], side["forecast"], side["pulse_runtime"], side["apparatus"]
    blocked = {f: bool(seal_a["family_construction"][f]["blocked"]) for f in FAMILIES}
    integ = integrity(cfg, pulses, cap, cons, blocked)
    cov = coverage(pulses, cons, rows, blocked)
    A = arm_metrics(T, dialogues, pulses, cap)
    gen = json.loads((Path(MODEL) / "generation_config.json").read_text())
    gmap = gaps(rows, pulses, run, list(gen["suppress_tokens"]))
    mc = cfg["matched_comparators"]
    mech = {v: mc[v]["RND"] for v in ("D3", "D3G", "D4", "D4G", "D5", "D5G")}
    boot = bootstrap(cfg, T, dialogues, gmap, dlg, comparisons(cfg), mech)

    def tot(k):
        return sum(T[d][k] for d in dialogues)
    lat_total = sum(l_["latin_units"] for l_ in ledgers.values())
    lat_mapped = sum(l_["latin_mapped"] for l_ in ledgers.values())
    o = cfg["opportunities"]
    opp = {"mapped_EN_confusion": tot("n_C"), "mapped_EN_confusion_dialogues": sum(1 for d in dialogues if T[d]["n_C"] > 0),
           "mapped_EN_correct": tot("n_EN"), "mapped_EN_correct_dialogues": sum(1 for d in dialogues if T[d]["n_EN"] > 0),
           "mapped_ZH_correct": tot("n_ZH"), "mapped_ZH_correct_dialogues": sum(1 for d in dialogues if T[d]["n_ZH"] > 0),
           "English_mapping_fraction": ratio(lat_mapped, lat_total)}
    opp_leaves = {"EN_confusion": opp["mapped_EN_confusion"] >= o["mapped_EN_confusion_min"],
                  "EN_confusion_dialogues": opp["mapped_EN_confusion_dialogues"] >= o["mapped_EN_confusion_dialogues_min"],
                  "EN_correct": opp["mapped_EN_correct"] >= o["mapped_EN_correct_min"],
                  "EN_correct_dialogues": opp["mapped_EN_correct_dialogues"] >= o["mapped_EN_correct_dialogues_min"],
                  "ZH_correct": opp["mapped_ZH_correct"] >= o["mapped_ZH_correct_min"],
                  "ZH_correct_dialogues": opp["mapped_ZH_correct_dialogues"] >= o["mapped_ZH_correct_dialogues_min"],
                  "English_mapping": opp["English_mapping_fraction"] is not None and opp["English_mapping_fraction"] >= o["English_unit_mapping_fraction_min"],
                  "job_A_coverage": bool(seal_a["predicates"]["job_A_coverage_pass"])}
    variants = {v: variant_eval(cfg, v, fam, A, T, dialogues, integ, opp, boot) for fam in FAMILIES for v in VARIANTS[fam]}
    statuses = {fam: family_status(cfg, fam, blocked[fam], cov[fam], variants) for fam in FAMILIES}
    critical = {"job_A_critical": all(seal_a["critical"].values()), "job_A_compatibility": bool(seal_a["predicates"]["compatibility_pass"]),
                "job_B_completed": prt.get("status") == "completed", "job_B_weights_unchanged": prt.get("weights_unchanged") is True,
                "job_B_no_parameter_grads": prt.get("parameters_with_grad") == 0 and prt.get("parameters_requiring_grad") == 0,
                "apparatus_old30_ok": app.get("ok") is True,
                "all_rows_ok": all(p.get("status") == "ok" for p in pulses.values()) and len(pulses) == len(P["selected"]),
                **integ["critical"]}
    flags = {"critical_failure": not all(critical.values()), "resource_pass": bool(fc.get("resource_pass")),
             "opportunity_pass": all(opp_leaves.values())}
    label = decide(flags, statuses)
    common = {f"{v}|{mc[v][r]}": common_eligible(rows, v, mc[v][r]) for v in mech for r in mc[v]}
    descr = descriptive(rows, gmap, pulses, cap)
    return {"opportunities": opp, "opportunity_leaves": opp_leaves, "arm_metrics": A, "integrity": integ, "coverage": cov,
            "variants": variants, "family_status": statuses, "critical": critical, "flags": flags,
            "provisional_terminal_label": label, "bootstrap": boot, "per_dialogue": T, "common_eligible": common,
            "descriptive": descr,
            "denominators": {"mapped": {"EN-confusion": tot("n_C"), "EN-correct": tot("n_EN"), "ZH-correct": tot("n_ZH")},
                             "structural_mapped": {s: sum(1 for r in rows if r["stratum"] == s and r["structural"]) for s in STRATA},
                             "latin_units": lat_total, "latin_mapped": lat_mapped,
                             "heard_scope_rows": sum(l_["heard_scope"] for l_ in ledgers.values())},
            "unit_reasons": dict(Counter(f"{u['stratum']}|{u['reason']}" for u in units_all)),
            "rows": rows, "units": units_all, "gaps": {f"{u}|{t}": g for (u, t), g in gmap.items()}}


def descriptive(rows: list[dict], gmap: dict, pulses: dict, cap: dict) -> dict:
    """Secondary: gap closure per arm and stratum top-1 change counts (no gate uses these)."""
    out = {"EN_confusion_gap": {}, "top1_changes": {}}
    for a in PULSE_ARMS:
        d = [g["B0"] - g[a] for g in gmap.values()]
        out["EN_confusion_gap"][a] = {"n": len(d), "median_closed": float(np.median(d)) if d else None,
                                      "mean_closed": float(np.mean(d)) if d else None}
    for a in ARMS:
        out["top1_changes"][a] = {s: sum(1 for r in rows if r["stratum"] == s and r["arms"][a]["top1"] != r["b0_top1"]) for s in STRATA}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUN)
    args = ap.parse_args()
    if args.out.rstrip("/") != RUN:
        raise SystemExit("only the frozen run root is allowed")
    run = ROOT / RUN
    if file_hash(ROOT / CONFIG) != CONFIG_SHA:
        raise SystemExit("config changed")
    cfg = json.loads((ROOT / CONFIG).read_text())
    for rel in ("pulse_seal.json", "audit_PRIMARY.json", "job_A_seal.json", "resource_forecast.json"):
        if not pushed(f"{RUN}/{rel}"):
            raise SystemExit(f"{rel} must be committed and pushed before references are opened")
    if json.loads((run / "audit_PRIMARY.json").read_text()).get("verdict") != "DIR_SPRINT0_AUDIT_PRIMARY: PASS":
        raise SystemExit("DIR_SPRINT0_AUDIT_PRIMARY: PASS required before references")
    if (run / "evaluation.json").exists():
        raise FileExistsError("evaluation exists; never overwrite")
    pseal = json.loads((run / "pulse_seal.json").read_text())
    aseal = json.loads((run / "job_A_seal.json").read_text())
    P = json.loads((ROOT / POPULATION).read_text())
    ids = [r["utterance_id"] for r in P["selected"]]
    cap, cons, pulses = {}, {}, {}
    for uid in ids:
        for name, store, seal in (("capture", cap, aseal), ("construct", cons, aseal), ("pulses", pulses, pseal)):
            rel = f"{name}/{uid}.json"
            if file_hash(run / rel) != seal["files"][rel]:
                raise ValueError(f"sealed row changed {rel}")
            store[uid] = json.loads((run / rel).read_text())
        for rel, seal in ((f"capture/{uid}.npz", aseal), (f"pulses/{pulses[uid]['logits']['file']}", pseal)):
            if file_hash(run / rel) != seal["files"][rel]:
                raise ValueError(f"sealed arrays changed {rel}")
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(MODEL, local_files_only=True)
    refs, access = load_references(ids)
    side = {"job_A_seal": aseal, "forecast": json.loads((run / "resource_forecast.json").read_text()),
            "pulse_runtime": json.loads((run / "pulse_runtime.json").read_text()),
            "apparatus": json.loads((run / "apparatus_old30.json").read_text())}
    res = evaluate(cfg, P, cap, cons, pulses, refs, tok, side, run)
    res["provisional_note"] = "assumes independent FULL audit PASS; confirmed or replaced by audit_FULL"
    units = res.pop("units")
    rows = res.pop("rows")
    atomic_json(run / "evaluation_units.json", {"schema": "dir_sprint0_evaluation_units_v1", "units": units, "rows": rows})
    res.update(schema="dir_sprint0_evaluation_v1", reference_access=access, pulse_seal_sha256=file_hash(run / "pulse_seal.json"),
               audit_PRIMARY_sha256=file_hash(run / "audit_PRIMARY.json"), units_sha256=file_hash(run / "evaluation_units.json"),
               git_head=git("rev-parse", "HEAD"), created_unix=time.time())
    atomic_json(run / "evaluation.json", res)
    print(json.dumps({"label": res["provisional_terminal_label"], "family_status": res["family_status"],
                      "C": {a: res["arm_metrics"][a]["C"] for a in ARMS}, "H_ZH": {a: res["arm_metrics"][a]["H_ZH"] for a in ARMS}},
                     indent=1))


if __name__ == "__main__":
    main()
