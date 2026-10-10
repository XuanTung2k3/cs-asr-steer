#!/usr/bin/env python
"""SRC-CF0-P evaluator (isolated; never imported by the runner). Frozen config configs/inference_cf/src_cf0_pilot.json.

gate-a      ONLY after the pushed direction seal + committed `SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)`: joins historical
            STRATA membership (docs/inference_cf/ST_PROMPT_R1_PANEL.json evaluation_membership; no token sets) to the sealed
            reference-free construction records and applies the frozen per-layer coverage / joint-reachability / specificity
            gates (cf_pilot_contract.construction_layer). Writes the detailed evaluator-only gate_a.json and the MINIMAL global
            PB_authorization.json (config / seal / audit hashes, gate_pass, qualified layer IDs, time, self hash).
evaluate-b  ONLY after the pushed pulse seal + committed `SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)`: opens the historical P2-RJ
            acceptable first-token sets / fixed competitors / strata, computes corrections / corruptions / EOS promotions on the
            fixed reference-free cohorts, paired control advantages, dialogue macro, LODO, dialogue-cluster bootstrap, margin
            diagnostic, the global damage stop, the exact terminal precedence (cf_pilot_contract.terminal) and at most one
            selected configuration. Nothing is ever returned to runtime.
            (`--negative` after a failed Gate P-A: the terminal label from gate_a.json alone; no lexical reference opened.)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf import cf_pilot_contract as C

CONFIG = "configs/inference_cf/src_cf0_pilot.json"
BASE = "results/inference_cf/src_cf0_pilot"
RUN = f"{BASE}/run1"
DSEAL = f"{RUN}/direction_seal.json"
CAUDIT = f"{RUN}/construction_audit.json"
GATE_A = f"{RUN}/gate_a.json"
AUTH = f"{RUN}/PB_authorization.json"
PSEAL = f"{RUN}/pulse_seal.json"
PAUDIT = f"{RUN}/primary_audit.json"
EVAL = f"{RUN}/evaluation.json"
MEMBERSHIP = "docs/inference_cf/ST_PROMPT_R1_PANEL.json"
PANEL = "docs/inference_cf/SRC_CF0_PANEL.json"
POSITIONS = "results/inference_cf/p2rj/positions.json"
REMOTE = "origin/cs-asr-steer-inf"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
LAYERS = (16, 24)
ETAS = (0.15, 0.30)
SIGNS = (1, -1)
EOS = 50257


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def committed(rel: str) -> bool:
    import hashlib
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def on_remote(rel: str) -> bool:
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


def J(rel: str):
    return json.loads((ROOT / rel).read_text())


def arm_id(layer, eta, sign, family="target") -> str:
    base = f"L{int(layer)}_eta{float(eta):.2f}_{'plus' if int(sign) > 0 else 'minus'}"
    return base if family == "target" else f"{family}_{base}"


def verify_seal(rel: str) -> dict:
    seal = J(rel)
    if not (committed(rel) and on_remote(rel)):
        raise SystemExit(f"{rel} not committed and pushed")
    if digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"]:
        raise SystemExit(f"{rel} hash")
    for p, h in seal["files"].items():
        if file_hash(ROOT / p) != h:
            raise SystemExit(f"sealed file changed: {p}")
    return seal


def construction_rows() -> dict:
    out = {}
    for i in range(80):
        d = J(f"{RUN}/construction/{i:03d}.json")
        for q in d["queries"]:
            out[q["j"]] = q
    if sorted(out) != list(range(180)):
        raise SystemExit("construction coverage")
    return out


# ---- Gate P-A --------------------------------------------------------------------------------------------------------

def joint(lay: dict, tol: float) -> bool:
    """P-A joint reachability recomputed from the sealed ledger: target + off + random numerically valid, tangent ratio
    >= 0.25 each, every (family, sign, dose) cell reachable with energy checks, paired squared energy within tol."""
    if lay.get("target", {}).get("status") != "OK" or lay.get("off", {}).get("status") != "OK":
        return False
    if not all(lay[f]["tangent_ratio"] >= 0.25 for f in ("target", "off", "random")):
        return False
    for s in SIGNS:
        for e in ETAS:
            cells = [lay["reach"][f"{f}|{s}|{e:.2f}"] for f in ("target", "off", "random")]
            if not all(c.get("reachable") for c in cells):
                return False
            sq = [c["emulated_edit_norm"] ** 2 for c in cells]
            if not (min(sq) > 0 and max(sq) / min(sq) - 1 <= tol):
                return False
    return True


def gate_a(args) -> None:
    cfg = J(CONFIG)
    seal = verify_seal(DSEAL)
    if not committed(CAUDIT) or J(CAUDIT)["verdict"] != "SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)":
        raise SystemExit("construction audit PASS required before the evaluator-only gate")
    st = J(f"{RUN}/construct_status.json")
    rows = construction_rows()
    # ---- historical strata opened only now (membership labels only; no token sets) ----
    mem = J(MEMBERSHIP)["evaluation_membership"]
    proj = J(f"{RUN}/runtime.json")["projection"]["runtime_queries"]
    if [(m["utterance_id"], m["t"]) for m in mem] != [(q["utterance_id"], q["t"]) for q in proj]:
        raise SystemExit("membership order")
    S = [m["stratum"] for m in mem]
    D = [m["dialogue_id"] for m in mem]
    cov = J(PANEL)["coverage_review_only"]
    cross = all(sorted(j for j in range(180) if S[j] == s and rows[j]["region"]["target_status"] == "OK") == cov[s]["target"]["query_indices"]
                and sorted(j for j in range(180) if S[j] == s and rows[j]["region"]["paired_available"]) == cov[s]["paired"]["query_indices"] for s in STRATA)
    tol = cfg["energy"]["paired_relative_squared_energy_difference_max"]
    integrity = {"construct_completed": st["status"] == "completed", "repeat_bitwise_180": all(rows[j]["repeat_bitwise"] for j in range(180)),
                 "historical_identity_180": all(all(rows[j]["historical_identity"].values()) for j in range(180)),
                 "no_critical_direction": all("critical" not in rows[j]["layers"] for j in range(180)),
                 "strata_60": all(S.count(s) == 60 for s in STRATA), "strata_coverage_cross_check": cross,
                 "seal_source_commit_on_remote": bool(seal["source_commit_on_remote"])}
    layers = {}
    for l in LAYERS:
        L = {j: rows[j]["layers"][str(l)] for j in range(180)}
        tgt = [j for j in range(180) if rows[j]["region"]["target_status"] == "OK"]
        prd = [j for j in range(180) if rows[j]["region"]["paired_available"]]
        vt = [j for j in tgt if L[j]["target"].get("status") == "OK"]
        vp = [j for j in prd if L[j]["target"].get("status") == "OK" and L[j]["off"].get("status") == "OK"]
        jn = [j for j in range(180) if joint(L[j], tol)]
        sid = lambda js, s: [j for j in js if S[j] == s]
        dl = lambda js: len({D[j] for j in js})
        spec = [j for j in sid(vp, "EN-confusion") if L[j]["pair"] and L[j]["pair"]["raw_tangent_ratio"] is not None]
        cos = [L[j]["pair"]["abs_cos"] for j in spec]
        rat = [L[j]["pair"]["raw_tangent_ratio"] for j in spec]
        m = {"valid_targets": len(vt), "valid_off_pairs": len(vp), "EN_confusion_target": len(sid(vt, "EN-confusion")),
             "EN_confusion_target_dialogues": dl(sid(vt, "EN-confusion")), "EN_confusion_joint": len(sid(jn, "EN-confusion")),
             "EN_confusion_joint_dialogues": dl(sid(jn, "EN-confusion")), "EN_correct_joint": len(sid(jn, "EN-correct")),
             "EN_correct_joint_dialogues": dl(sid(jn, "EN-correct")), "specificity_pairs": len(spec), "specificity_dialogues": dl(spec),
             "median_abs_cosine": float(np.median(cos)) if cos else None, "median_tangent_ratio": float(np.median(rat)) if rat else None}
        res = C.construction_layer(m, cfg)
        desc = {s: {"raw_targets": len(sid(tgt, s)), "valid_targets": len(sid(vt, s)), "raw_pairs": len(sid(prd, s)), "valid_pairs": len(sid(vp, s)),
                    "joint": len(sid(jn, s)), "joint_dialogues": dl(sid(jn, s)), "target_dialogues": dl(sid(vt, s))} for s in STRATA}
        reasons = defaultdict(int)
        for j in vp:
            if not joint(L[j], tol):
                lay = L[j]
                if not all(lay[f]["tangent_ratio"] >= 0.25 for f in ("target", "off", "random")):
                    reasons["tangent_ratio_below_0.25"] += 1
                elif not all(lay["reach"][k].get("reachable") for k in lay["reach"]):
                    reasons["unreachable_or_energy"] += 1
                else:
                    reasons["paired_energy"] += 1
        geo = {}
        for f in ("target", "off", "random"):
            vals = [L[j][f] for j in range(180) if L[j].get(f, {}).get("status") == "OK"]
            geo[f] = {k: (None if not vals else {"n": len(vals), "min": float(np.min([v[k] for v in vals])), "median": float(np.median([v[k] for v in vals])),
                                                 "max": float(np.max([v[k] for v in vals]))})
                      for k in (("raw_norm", "direction_norm", "tangent_ratio", "raw_tangent_norm", "cos_v_h") if f != "random" else ("tangent_ratio", "cos_v_h"))}
        pairs_all = [L[j]["pair"] for j in vp if L[j]["pair"]]
        geo["pairs_all_strata"] = {"n": len(pairs_all), "median_abs_cos": float(np.median([p["abs_cos"] for p in pairs_all])) if pairs_all else None,
                                   "median_signed_cos": float(np.median([p["cos"] for p in pairs_all])) if pairs_all else None,
                                   "median_raw_tangent_ratio": float(np.median([p["raw_tangent_ratio"] for p in pairs_all if p["raw_tangent_ratio"] is not None])) if pairs_all else None}
        reach_rates = {}
        for f in ("target", "off", "random"):
            for s in SIGNS:
                for e in ETAS:
                    cells = [L[j]["reach"][f"{f}|{s}|{e:.2f}"] for j in vt]
                    reach_rates[f"{f}|{s}|{e:.2f}"] = {"n": len(cells), "reachable": sum(bool(c.get("reachable")) for c in cells)}
        layers[str(l)] = {"metrics": m, "result": res, "by_stratum": desc, "joint_failure_reasons": dict(reasons), "geometry": geo,
                          "reach_rates_on_valid_targets": reach_rates, "joint_members": jn, "specificity_members": spec,
                          "specificity_cohort_abs_cos": cos, "specificity_cohort_ratio": rat}
    integ = all(integrity.values())
    construction = any(layers[str(l)]["result"]["construction"] for l in LAYERS)
    qualified = [l for l in LAYERS if layers[str(l)]["result"]["qualified"]]
    gate_pass = bool(integ and qualified)
    label = None if gate_pass else C.terminal(cfg, integrity=integ, construction=construction, specificity=bool(qualified), arms=[])
    doc = {"schema": "src_cf0_pilot_gate_a_v1", "evaluator_only": True, "direction_seal_hash": seal["seal_hash"], "integrity": integrity,
           "integrity_ok": integ, "layers": layers, "construction_any_layer": construction, "qualified_layers": qualified, "gate_pass": gate_pass,
           "terminal_label_if_stopped": label, "created_unix": time.time()}
    doc["gate_hash"] = digest(doc)
    for p in (GATE_A, AUTH):
        if (ROOT / p).exists():
            raise FileExistsError(f"{p} exists; never overwrite")
    atomic_json(ROOT / GATE_A, doc)
    auth = {"schema": "SRC_CF0_PILOT_PB_AUTHORIZATION_V1", "config_sha256": file_hash(ROOT / CONFIG), "direction_seal_sha256": file_hash(ROOT / DSEAL),
            "construction_audit_sha256": file_hash(ROOT / CAUDIT), "gate_pass": gate_pass, "qualified_layers": qualified,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    auth["authorization_hash"] = digest(auth)
    if sorted(auth) != sorted(cfg["firewall"]["authorization_keys"]):
        raise SystemExit("authorization schema drift")
    atomic_json(ROOT / AUTH, auth)
    print(json.dumps({"gate_pass": gate_pass, "qualified_layers": qualified, "label_if_stopped": label,
                      "metrics": {l: layers[str(l)]["metrics"] for l in LAYERS}, "results": {l: layers[str(l)]["result"] for l in LAYERS},
                      "integrity": integrity}, indent=1))


# ---- P-B lexical evaluation ------------------------------------------------------------------------------------------

def unpack(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def processed(z: np.ndarray, t: int, sup, beg) -> np.ndarray:
    x = np.asarray(z, dtype=np.float64).copy()
    x[sup] = -np.inf
    if t == 0:
        x[beg] = -np.inf
    return x


def lse(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def metrics(z: np.ndarray, t: int, sup, beg, Y: list[int], comp: int) -> dict:
    """top1 (lower token ID on processed ties), fixed-competitor margin, reference rank / log-prob."""
    zp = processed(z, t, sup, beg)
    top1 = int(np.argmax(zp))
    lz = lse(zp)
    ref = np.asarray(Y, dtype=np.int64)
    lref = lse(zp[ref])
    best = float(np.max(zp[ref]))
    bid = min(int(y) for y in ref if zp[int(y)] == best)
    rank = 1 + int(np.sum(zp > best)) + int(np.sum(zp[:bid] == best))
    return {"top1": top1, "in_ref": top1 in set(Y), "m": float(lref - zp[comp]), "logp_ref": float(lref - lz), "rank_ref": rank}


def draw_weights(dialogues, reps: int, seed: int):
    keys = sorted(set(dialogues))
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(reps, len(keys)))
    W = np.zeros((reps, len(keys)), dtype=np.float64)
    for d in range(len(keys)):
        W[:, d] = (idx == d).sum(axis=1)
    return keys, W


def boot(values: list[tuple[str, float]], keys, W, q_lo: float, q_hi: float, min_draws: int) -> dict:
    by = defaultdict(list)
    for d, v in values:
        by[d].append(float(v))
    dm = {d: float(np.mean(v)) for d, v in by.items()}
    est = float(np.mean(list(dm.values()))) if dm else None
    m = np.array([dm.get(k, np.nan) for k in keys])
    has = ~np.isnan(m)
    num = (W[:, has] * m[has]).sum(axis=1)
    den = W[:, has].sum(axis=1)
    ok = den > 0
    stats = num[ok] / den[ok] if has.any() else np.array([])
    out = {"estimate": est, "rows": len(values), "dialogues": int(has.sum()), "finite_draws": int(ok.sum()) if has.any() else 0,
           "lower": None, "upper": None}
    if stats.size and out["finite_draws"] >= min_draws:
        out["lower"], out["upper"] = float(np.quantile(stats, q_lo)), float(np.quantile(stats, q_hi))
    return out


def finite(o):
    """JSON-safe copy: numpy scalars -> Python, non-finite floats (undefined quantities) -> None."""
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def evaluate_b(args) -> None:
    from transformers import GenerationConfig
    cfg = J(CONFIG)
    pseal = verify_seal(PSEAL)
    if not committed(PAUDIT) or J(PAUDIT)["verdict"] != "SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)":
        raise SystemExit("PRIMARY audit PASS required before any lexical reference access")
    gate, auth = J(GATE_A), J(AUTH)
    st = J(f"{RUN}/pulse_status.json")
    crow = construction_rows()
    prows, logits_files = {}, {}
    for i in range(80):
        d = J(f"{RUN}/pulses/{i:03d}.json")
        for js, r in d["positions"].items():
            prows[int(js)] = r
            logits_files[int(js)] = i
    if sorted(prows) != list(range(180)):
        raise SystemExit("pulse coverage")
    gen = GenerationConfig.from_pretrained(cfg["model"]["dir"], local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    # ---- lexical references opened only now ----
    pos = J(POSITIONS)["positions"]
    Q = J(f"{RUN}/runtime.json")["projection"]["runtime_queries"]
    if [(p["utterance_id"], int(p["t"]), p["dialogue_id"]) for p in pos] != [(q["utterance_id"], q["t"], q["dialogue_id"]) for q in Q]:
        raise SystemExit("positions order")
    mem = J(MEMBERSHIP)["evaluation_membership"]
    Y = [[int(x) for x in p["target_ids"] if int(x) < EOS] for p in pos]
    S = [p["stratum"] for p in pos]
    D = [p["dialogue_id"] for p in pos]
    comp = [int(p["competitor"]) for p in pos]
    integrity = {"pulse_seal_verified": True, "primary_audit_pass": True, "pulse_completed": st["status"] == "completed",
                 "strata_60": all(S.count(s) == 60 for s in STRATA), "strata_match_membership": S == [m["stratum"] for m in mem],
                 "targets_nonempty": all(Y), "authorization_gate_pass": bool(auth["gate_pass"]), "gate_a_pass": bool(gate["gate_pass"])}
    qualified = set(auth["qualified_layers"])
    cache: dict = {}

    def logits(j, key):
        i = logits_files[j]
        if i not in cache:
            cache.clear()
            with np.load(ROOT / f"{RUN}/pulses/{i:03d}_logits.npz") as z:
                cache[i] = {k: z[k] for k in z.files}
        return unpack(cache[i][f"q{j:03d}_{key}"])

    base = {}
    for i in range(80):
        with np.load(ROOT / f"{RUN}/construction/{i:03d}.npz") as z:
            for q in J(f"{RUN}/construction/{i:03d}.json")["queries"]:
                base[q["j"]] = unpack(z[f"q{q['j']:03d}_clean_r1_raw"])
    rows = []
    for j in range(180):
        t = Q[j]["t"]
        nm = metrics(base[j], t, sup, beg, Y[j], comp[j])
        r = {"j": j, "dialogue_id": D[j], "stratum": S[j], "none": nm, "cells": {}}
        if nm["top1"] != prows[j]["none_top1"]:
            integrity["none_top1_consistent"] = False
        for fam in ("target", "off", "random"):
            for l in LAYERS:
                for e in ETAS:
                    for s in SIGNS:
                        aid = arm_id(l, e, s, fam)
                        c = prows[j]["cells"][aid]
                        mm = metrics(logits(j, aid), t, sup, beg, Y[j], comp[j]) if c["steered"] else nm
                        r["cells"][aid] = {"steered": bool(c["steered"]), "valid": bool(c["valid"]), "top1": mm["top1"], "in_ref": mm["in_ref"],
                                           "d_m": mm["m"] - nm["m"], "d_rank": mm["rank_ref"] - nm["rank_ref"], "d_logp_ref": mm["logp_ref"] - nm["logp_ref"],
                                           "actual_eta": c.get("actual_eta"), "realized": c.get("realized_edit_norm")}
        rows.append(r)
    integrity.setdefault("none_top1_consistent", True)
    integrity["clean_correct_strata_consistent"] = all(r["none"]["in_ref"] == (r["stratum"] != "EN-confusion") for r in rows)
    keys, W = draw_weights(D, cfg["statistics"]["bootstrap_draws"], cfg["statistics"]["seed"])
    a16 = .05 / (2 * cfg["statistics"]["primary_family"])
    mlo, mhi = cfg["statistics"]["margin_adjusted_CI_quantiles"]
    mind = cfg["statistics"]["minimum_finite_bootstrap_draws"]
    g = cfg["gate_b"]
    tol = cfg["energy"]["paired_relative_squared_energy_difference_max"]
    arms_out = []
    for l in LAYERS:
        for e in ETAS:
            for s in SIGNS:
                aid, ro, rr = arm_id(l, e, s), arm_id(l, e, s, "off"), arm_id(l, e, s, "random")
                pred = [j for j in range(180) if joint(crow[j]["layers"][str(l)], tol)]
                cohort = [j for j in pred if prows[j]["paired_energy"][aid]["ok"]]
                retention = len(cohort) / len(pred) if pred else 0.0
                conf = [j for j in cohort if S[j] == "EN-confusion"]
                ind = {x: {j: int((not rows[j]["none"]["in_ref"]) and rows[j]["cells"][x]["in_ref"]) for j in conf} for x in (aid, ro, rr)}
                corr = [j for j in conf if ind[aid][j]]
                ctrl = {}
                for name, cid in (("random", rr), ("off_target", ro)):
                    diff = {j: ind[aid][j] - ind[cid][j] for j in conf}
                    byd = defaultdict(int)
                    for j_, v_ in diff.items():
                        byd[D[j_]] += v_
                    rep = sorted({D[j] for j in conf})
                    bs = boot([(D[j], float(diff[j])) for j in conf], keys, W, a16, 1 - a16, mind)
                    ctrl[name] = {"control_arm": cid, "control_corrections": sum(ind[cid].values()), "net_corrections": int(sum(diff.values())),
                                  "positive_dialogues": sum(1 for d in rep if byd[d] > 0), "macro_advantage": bs["estimate"] if bs["estimate"] is not None else float("nan"),
                                  "lodo_net": [int(sum(v_ for j_, v_ in diff.items() if D[j_] != d)) for d in rep], "lodo_dialogues": rep,
                                  "bootstrap_adjusted": bs}
                active = [j for j in range(180) if rows[j]["cells"][aid]["steered"]]
                corrupt = lambda st_: [j for j in active if S[j] == st_ and rows[j]["none"]["in_ref"] and not rows[j]["cells"][aid]["in_ref"]]
                eos = [j for j in active if S[j] != "EN-confusion" and rows[j]["none"]["top1"] != EOS and rows[j]["cells"][aid]["top1"] == EOS]
                allc = lambda st_: [j for j in range(180) if S[j] == st_ and rows[j]["none"]["in_ref"] and not rows[j]["cells"][aid]["in_ref"]]
                mb = boot([(D[j], rows[j]["cells"][aid]["d_m"]) for j in conf], keys, W, mlo, mhi, mind)
                integ_arm = not any(prows[j]["cells"][x]["integrity_failures"] for j in range(180) for x in (aid, ro, rr))
                m = {"id": aid, "layer": l, "eta": e, "sign": s, "qualified_layer": l in qualified,
                     "energy_valid": bool(retention >= g["actual_edit_valid_fraction_of_sealed_prediction_min"] and integ_arm),
                     "sealed_joint_prediction": len(pred), "paired_cohort_all_strata": len(cohort), "retention": retention,
                     "paired_rows": len(conf), "paired_dialogues": len({D[j] for j in conf}), "corrections": len(corr),
                     "corrected_dialogues": len({D[j] for j in corr}), "corrected_positions": corr, "random": ctrl["random"], "off_target": ctrl["off_target"],
                     "EN_corruptions": len(corrupt("EN-correct")), "ZH_corruptions": len(corrupt("ZH-correct")), "correct_EOS_promotions": len(eos),
                     "EN_corruptions_all_rows": len(allc("EN-correct")), "ZH_corruptions_all_rows": len(allc("ZH-correct")),
                     "margin_macro": mb["estimate"] if mb["estimate"] is not None else float("-inf"), "margin_adjusted_lower": mb["lower"],
                     "margin_bootstrap": mb, "active": {st_: sum(1 for j in active if S[j] == st_) for st_ in STRATA},
                     "active_dialogues": {st_: len({D[j] for j in active if S[j] == st_}) for st_ in STRATA},
                     "paired_by_stratum": {st_: sum(1 for j in cohort if S[j] == st_) for st_ in STRATA},
                     "abstentions_by_stratum": {st_: 60 - sum(1 for j in active if S[j] == st_) for st_ in STRATA},
                     "unconditional_corrections_of_60": sum(1 for j in active if S[j] == "EN-confusion" and not rows[j]["none"]["in_ref"] and rows[j]["cells"][aid]["in_ref"]),
                     "realized_eta": {"n": len(active), "min": min([rows[j]["cells"][aid]["actual_eta"] for j in active], default=None),
                                      "max": max([rows[j]["cells"][aid]["actual_eta"] for j in active], default=None)},
                     "mandarin_active_changes": [{"j": j, "dialogue_id": D[j], "none_top1": rows[j]["none"]["top1"], "edited_top1": rows[j]["cells"][aid]["top1"],
                                                  "edited_in_ref": rows[j]["cells"][aid]["in_ref"]} for j in active if S[j] == "ZH-correct"
                                                 and rows[j]["cells"][aid]["top1"] != rows[j]["none"]["top1"]],
                     "EN_correct_active_changes": [{"j": j, "dialogue_id": D[j], "none_top1": rows[j]["none"]["top1"], "edited_top1": rows[j]["cells"][aid]["top1"],
                                                    "edited_in_ref": rows[j]["cells"][aid]["in_ref"]} for j in active if S[j] == "EN-correct"
                                                   and rows[j]["cells"][aid]["top1"] != rows[j]["none"]["top1"]],
                     "mean_d_margin_conf_cohort": float(np.mean([rows[j]["cells"][aid]["d_m"] for j in conf])) if conf else None,
                     "mean_d_rank_conf_cohort": float(np.mean([rows[j]["cells"][aid]["d_rank"] for j in conf])) if conf else None,
                     "integrity_ok": integ_arm}
                m["power_pass"] = bool(C.power_pass(m, g))
                m["observed_safety_pass"] = bool(C.observed_safety_pass(m, g))
                # descriptive: target-only cohort (matching target eligibility) vs random, and each control's own damage
                tonly = [j for j in range(180) if S[j] == "EN-confusion" and rows[j]["cells"][aid]["valid"] and rows[j]["cells"][rr]["valid"]]
                m["descriptive_target_vs_random_on_target_eligible_EN_confusion"] = {
                    "rows": len(tonly), "target_corrections": sum(1 for j in tonly if not rows[j]["none"]["in_ref"] and rows[j]["cells"][aid]["in_ref"]),
                    "random_corrections": sum(1 for j in tonly if not rows[j]["none"]["in_ref"] and rows[j]["cells"][rr]["in_ref"])}
                m["controls_damage"] = {x: {"EN": sum(1 for j in range(180) if S[j] == "EN-correct" and rows[j]["none"]["in_ref"] and not rows[j]["cells"][x]["in_ref"]),
                                            "ZH": sum(1 for j in range(180) if S[j] == "ZH-correct" and rows[j]["none"]["in_ref"] and not rows[j]["cells"][x]["in_ref"]),
                                            "corrections_all": sum(1 for j in range(180) if S[j] == "EN-confusion" and rows[j]["cells"][x]["in_ref"])}
                                        for x in (ro, rr)}
                arms_out.append(m)
    integ = all(integrity.values())
    label = C.terminal(cfg, integrity=integ, construction=True, specificity=True, arms=arms_out)
    eligible = [m for m in arms_out if m["power_pass"] and m["observed_safety_pass"]]
    order = sorted(eligible, key=lambda m: (-m["corrections"], -m["corrected_dialogues"],
                                           -min(m["random"]["net_corrections"], m["off_target"]["net_corrections"]),
                                           m["ZH_corruptions"], m["EN_corruptions"], m["eta"], m["layer"], m["id"]))
    selected = order[0]["id"] if (order and label == "SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED") else None
    d = cfg["adverse_damage"]
    glob = [m["id"] for m in arms_out if m["EN_corruptions_all_rows"] >= d["global_any_primary_EN_correct_corruptions_at_least"]
            or m["ZH_corruptions_all_rows"] >= d["global_any_primary_ZH_correct_corruptions_at_least"]
            or m["correct_EOS_promotions"] >= d["global_any_primary_correct_state_EOS_promotions_at_least"]]
    union = sorted({j for m in arms_out for j in m["corrected_positions"]})
    doc = {"schema": "src_cf0_pilot_evaluation_v1", "pulse_seal_hash": pseal["seal_hash"], "integrity": integrity, "integrity_ok": integ,
           "qualified_layers": sorted(qualified), "arms": arms_out, "global_damage_stop_arms": glob, "label": label, "selected": selected,
           "power_pass": [m["id"] for m in arms_out if m["power_pass"]], "power_and_safety_pass": [m["id"] for m in eligible],
           "oracle_union_NOT_DEPLOYABLE": {"positions": union, "count": len(union), "dialogues": len({D[j] for j in union})},
           "bootstrap": {"dialogues": keys, "draws": int(W.shape[0]), "seed": cfg["statistics"]["seed"], "primary_quantiles": [a16, 1 - a16],
                         "margin_quantiles": [mlo, mhi]},
           "none_by_stratum": {s: {"top1_in_ref": sum(1 for r in rows if r["stratum"] == s and r["none"]["in_ref"]), "n": 60} for s in STRATA},
           "rows": rows, "created_unix": time.time()}
    doc = finite(doc)
    doc["evaluation_hash"] = digest(doc)
    if (ROOT / EVAL).exists():
        raise FileExistsError("evaluation exists; never overwrite")
    atomic_json(ROOT / EVAL, doc)
    print(json.dumps({"label": label, "selected": selected, "integrity": integrity, "global_damage_stop_arms": glob,
                      "arms": [{k: m[k] for k in ("id", "qualified_layer", "energy_valid", "retention", "paired_rows", "paired_dialogues", "corrections",
                                                  "corrected_dialogues", "EN_corruptions", "ZH_corruptions", "correct_EOS_promotions", "margin_macro",
                                                  "margin_adjusted_lower", "power_pass")} | {"net_random": m["random"]["net_corrections"],
                                                                                             "net_off": m["off_target"]["net_corrections"]} for m in arms_out]},
                     indent=1, default=float))


def negative(args) -> None:
    """After a failed Gate P-A: terminal label from the sealed gate result only (no lexical reference is opened)."""
    gate = J(GATE_A)
    if not (committed(GATE_A) and on_remote(GATE_A)) or gate["gate_pass"]:
        raise SystemExit("negative evaluation only after a committed, pushed FAILED Gate P-A")
    doc = {"schema": "src_cf0_pilot_evaluation_v1", "phase_b": "NOT_RUN", "label": gate["terminal_label_if_stopped"], "selected": None,
           "gate_a_hash": gate["gate_hash"], "references_opened": False, "created_unix": time.time()}
    doc["evaluation_hash"] = digest(doc)
    if (ROOT / EVAL).exists():
        raise FileExistsError("evaluation exists; never overwrite")
    atomic_json(ROOT / EVAL, doc)
    print(json.dumps(doc, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate-a")
    sub.add_parser("evaluate-b")
    sub.add_parser("negative")
    args = ap.parse_args()
    {"gate-a": gate_a, "evaluate-b": evaluate_b, "negative": negative}[args.cmd](args)


if __name__ == "__main__":
    main()
