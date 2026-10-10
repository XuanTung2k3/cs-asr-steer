#!/usr/bin/env python
"""P2-DIR analysis (CPU, evaluator side). Frozen spec sections 6-9.

Exp-1: per-position evaluator metrics from the saved full (lossless bf16) logits and site vectors,
engineering / state-identity validity, the frozen family-10 dialogue bootstrap, candidate
qualification and the single-selection rule. The Exp-2 and Exp-3 decision rules are frozen here as
pure functions before any Exp-1 outcome exists.

Definitions (fixed): processed logits z' = generate()-suppressed float logits; fixed-competitor
margin m = logsumexp z'(Y_ref) - z'(c*) with the frozen P2-RJ Y_ref and c*; delta = arm - none at
the same unedited state; corruption at a correct-stratum position = post-edit decode-rule
(first-index) argmax not in Y_ref; correction at a confusion position = post-edit argmax in Y_ref.
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

from csasr.inference_cf.core import atomic_json, digest, file_hash

STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
ARMS = ("D0", "D1", "D2")
CANDIDATES = ("D1", "D2")
EPS_J = 1e-12
MAX_STATE_MISMATCH = 9                   # > 5% of 180 -> run invalid
MIN_VALID_DRAWS = 9900
E1_FAMILY = 10
E2_FAMILY = 5
E3_FAMILY = 10


# ---- logits metrics ------------------------------------------------------------------------------

def _lse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mx = float(np.max(x))
    if not math.isfinite(mx):
        return mx
    return mx + math.log(float(np.sum(np.exp(x - mx))))


def processed(z: np.ndarray, step: int, suppress, begin) -> np.ndarray:
    zp = np.asarray(z, dtype=np.float64).copy()
    if suppress:
        zp[list(suppress)] = -np.inf
    if step == 0 and begin:
        zp[list(begin)] = -np.inf
    return zp


def logit_metrics(z: np.ndarray, step: int, suppress, begin, partition: dict, ref_ids: list[int],
                  competitor: int) -> dict:
    """All evaluator scalars of one next-token distribution (float64 from float32 logits)."""
    zp = processed(z, step, suppress, begin)
    lz = _lse(zp)
    lp = zp - lz
    ref = np.asarray(ref_ids, dtype=np.int64)
    lref = _lse(zp[ref])
    top1 = int(np.argmax(zp))                 # first index on ties (decode rule)
    others = zp.copy()
    others[ref] = -np.inf
    best_other = int(np.argmax(others))
    best_ref_val = float(np.max(zp[ref]))
    best_ref = min(int(x) for x in ref if zp[int(x)] == best_ref_val)
    # full-vocabulary rank of the best reference token, deterministic first-index tie order
    rank = 1 + int(np.sum(zp > best_ref_val)) + int(np.sum(zp[:best_ref] == best_ref_val))
    log_pe = _lse(lp[partition["embedded_ids"]])
    log_pm = _lse(lp[partition["matrix_ids"]])
    leps = math.log(EPS_J)
    J = float(np.logaddexp(log_pe, leps) - np.logaddexp(log_pm, leps))
    return {"m": float(lref - zp[competitor]), "logp_ref": float(lref - lz), "rank_ref": rank,
            "top1": top1, "top1_in_ref": bool(top1 in set(int(x) for x in ref_ids)),
            "best_other": best_other, "margin_best_other": float(lref - zp[best_other]),
            "logp_competitor": float(lp[competitor]), "max_other_logp": float(lp[best_other]),
            "P_E": math.exp(log_pe), "P_M": math.exp(log_pm), "log_PE": log_pe, "log_PM": log_pm, "J": J,
            "raw_logit_competitor": float(z[competitor]), "raw_lse_ref": _lse(np.asarray(z, dtype=np.float64)[ref])}


def tangent(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    rh = r / np.linalg.norm(r)
    return g - rh * float(rh @ g)


def cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else None


# ---- bootstrap ------------------------------------------------------------------------------------

def draw_weights(dialogues, reps: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Shared draws: sorted dialogue ids, rng.integers(0, n, (reps, n)); W[r, j] = multiplicity."""
    keys = sorted(set(dialogues))
    n = len(keys)
    idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    W = np.zeros((reps, n), dtype=np.float64)
    for j in range(n):
        W[:, j] = (idx == j).sum(axis=1)
    return keys, W


def boot_stat(values: list[tuple[str, float]], keys: list[str], W: np.ndarray, alpha: float) -> dict:
    """Mean of dialogue means (positions within dialogue, then dialogues); percentile interval at
    (alpha/2, 1-alpha/2) over the shared draws; draws without any row are omitted and counted."""
    by = defaultdict(list)
    for d, v in values:
        if v is not None and math.isfinite(v):
            by[d].append(float(v))
    m = np.array([np.mean(by[k]) if by.get(k) else np.nan for k in keys])
    has = ~np.isnan(m)
    out = {"dialogues": int(has.sum()), "positions": int(sum(len(v) for v in by.values())),
           "estimate": None, "ci": None, "valid_draws": 0, "omitted_draws": int(W.shape[0])}
    if not has.any():
        return out
    out["estimate"] = float(np.mean(m[has]))
    num = W[:, has] @ m[has]
    den = W[:, has].sum(axis=1)
    ok = den > 0
    draws = num[ok] / den[ok]
    out["valid_draws"], out["omitted_draws"] = int(ok.sum()), int((~ok).sum())
    if ok.any():
        out["ci"] = [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))]
    return out


def lo(x):
    return x["ci"][0] if x and x.get("ci") else None


def hi(x):
    return x["ci"][1] if x and x.get("ci") else None


# ---- frozen decisions -----------------------------------------------------------------------------

def qualify_exp1(fam: dict, observed_corruption: dict, candidate_valid_rate: float, engineering_ok: bool,
                 population_ok: bool, cfg: dict) -> dict:
    """Frozen Exp-1 qualification of ONE candidate (spec section 6). fam keys: conf, paired, en, zh, corr."""
    e = cfg["exp1"]
    c = {"conf_point": fam["conf"]["estimate"] is not None and fam["conf"]["estimate"] >= e["material_margin_nats"],
         "conf_lower": lo(fam["conf"]) is not None and lo(fam["conf"]) > 0,
         "paired_lower": lo(fam["paired"]) is not None and lo(fam["paired"]) > e["paired_D0_lower_bound_nats"],
         "en_margin_lower": lo(fam["en"]) is not None and lo(fam["en"]) >= e["correct_margin_lower_bound_nats"],
         "zh_margin_lower": lo(fam["zh"]) is not None and lo(fam["zh"]) >= e["correct_margin_lower_bound_nats"],
         "corruption_upper": hi(fam["corr"]) is not None and hi(fam["corr"]) <= e["corruption_upper_bound"],
         "corruption_each_stratum": all(v is not None and v <= e["corruption_upper_bound"]
                                        for v in observed_corruption.values()),
         "valid_rate": candidate_valid_rate >= e["candidate_valid_rate_min"],
         "population": bool(population_ok), "engineering": bool(engineering_ok)}
    benefit = c["conf_point"] and c["conf_lower"] and c["paired_lower"]
    safety = c["en_margin_lower"] and c["zh_margin_lower"] and c["corruption_upper"] and c["corruption_each_stratum"]
    return {"checks": c, "benefit": benefit, "safety": safety, "qualifies": all(c.values())}


def select_exp1(engineering_ok: bool, q: dict, tie: dict | None, cfg: dict) -> dict:
    """Frozen selection (spec sections 6, 9). q: {D1: qualify, D2: qualify}; tie: D2-minus-D1 CI."""
    if not engineering_ok:
        return {"label": "P2_DIR_INVALID", "selected": None, "supplementary": []}
    qual = [k for k in CANDIDATES if q[k]["qualifies"]]
    supp = []
    for k in CANDIDATES:
        if not q[k]["qualifies"] and q[k]["benefit"] and not q[k]["safety"]:
            supp.append(f"P2_DIR_CAUSAL_POWER_WITH_DAMAGE:{k}")
    if not qual:
        return {"label": "P2_DIR_NO_NEW_DIRECTION_SUPPORTED", "selected": None, "supplementary": supp}
    if len(qual) == 1:
        sel = qual[0]
    else:
        sel = "D2" if lo(tie) is not None and lo(tie) > 0.25 else "D1"
    return {"label": "P2_DIR_UNIQUE_SELECTED" if sel == "D1" else "P2_DIR_READOUT_SELECTED",
            "selected": sel, "supplementary": supp}


def decide_exp2(fam: dict, exp1_conf_point: float, observed_corruption: dict, direction_valid_rate: float,
                engineering_ok: bool, cfg: dict) -> dict:
    """Frozen Exp-2 rule (spec section 7). fam keys: conf, new_minus_old, en, zh, corr."""
    e = cfg["exp2"]
    if not engineering_ok:
        return {"label": "P2_DIR_INVALID", "pass": False, "checks": {}}
    pt = fam["conf"]["estimate"]
    c = {"conf_point": pt is not None and pt >= e["material_margin_nats"],
         "conf_retained": pt is not None and pt >= e["minimum_retained_exp1_benefit_fraction"] * exp1_conf_point,
         "conf_lower": lo(fam["conf"]) is not None and lo(fam["conf"]) > 0,
         "new_minus_old_lower": lo(fam["new_minus_old"]) is not None and lo(fam["new_minus_old"]) > e["paired_D0_lower_bound_nats"],
         "en_margin_lower": lo(fam["en"]) is not None and lo(fam["en"]) >= e["correct_margin_lower_bound_nats"],
         "zh_margin_lower": lo(fam["zh"]) is not None and lo(fam["zh"]) >= e["correct_margin_lower_bound_nats"],
         "corruption_upper": hi(fam["corr"]) is not None and hi(fam["corr"]) <= e["corruption_upper_bound"],
         "corruption_each_stratum": all(v is not None and v <= e["corruption_upper_bound"] for v in observed_corruption.values()),
         "direction_valid": direction_valid_rate >= cfg["exp1"]["candidate_valid_rate_min"]}
    safety = c["en_margin_lower"] and c["zh_margin_lower"] and c["corruption_upper"] and c["corruption_each_stratum"]
    benefit = c["conf_point"] and c["conf_retained"] and c["conf_lower"] and c["new_minus_old_lower"]
    if not c["direction_valid"]:
        label = "P2_DIR_INVALID"
    elif not safety:
        label = "P2_DIR_CAUSAL_POWER_WITH_DAMAGE"
    elif not benefit:
        label = "P2_DIR_GATE_COUPLING_SUSPECTED"
    else:
        label = "P2_DIR_EXP2_PASS"
    return {"label": label, "pass": label == "P2_DIR_EXP2_PASS", "checks": c, "benefit": benefit, "safety": safety}


def decide_exp3(fam: dict, broad_safety_pointwise: dict, broad_valid: bool, engineering_ok: bool, cfg: dict) -> dict:
    """Frozen Exp-3 rule (spec sections 8-9), precedence order.

    fam (Bonferroni 0.05/10): pier_gain_vs_b0, pier_gain_vs_old (gain = baseline - method),
    damage_mer, damage_zh_cer, damage_en_wer (method - B0), retention_loss_en, retention_loss_zh,
    outside_harm (NEW rate), broad_minus_new_pier_gain (NEW PIER - BROAD PIER), outside_harm_diff
    (BROAD - NEW). broad_safety_pointwise: the same absolute safety quantities for BROAD
    (pointwise 95%)."""
    e = cfg["exp3"]
    if not engineering_ok:
        return {"label": "P2_DIR_INVALID"}

    def safe(f) -> dict:
        return {"mer": hi(f["damage_mer"]) is not None and hi(f["damage_mer"]) <= e["MER_damage_upper_bound"],
                "zh_cer": hi(f["damage_zh_cer"]) is not None and hi(f["damage_zh_cer"]) <= e["ZH_CER_damage_upper_bound"],
                "en_wer": hi(f["damage_en_wer"]) is not None and hi(f["damage_en_wer"]) <= e["EN_WER_damage_upper_bound"],
                "ret_en": hi(f["retention_loss_en"]) is not None and hi(f["retention_loss_en"]) <= e["retention_loss_upper_bound"],
                "ret_zh": hi(f["retention_loss_zh"]) is not None and hi(f["retention_loss_zh"]) <= e["retention_loss_upper_bound"],
                "outside": hi(f["outside_harm"]) is not None and hi(f["outside_harm"]) <= e["outside_target_harm_upper_bound"]}

    def material(x) -> bool:
        return x["estimate"] is not None and x["estimate"] >= e["material_PIER_gain"] and lo(x) is not None and lo(x) > 0

    s_new = safe(fam)
    out = {"new_safety": s_new}
    if not all(s_new.values()):
        out["label"] = "P2_DIR_CAUSAL_POWER_WITH_DAMAGE"
        return out
    if material(fam["pier_gain_vs_b0"]) and material(fam["pier_gain_vs_old"]):
        out["label"] = "P2_DIR_OLD_DIRECTION_PRIMARY_BOTTLENECK_SUPPORTED"
        return out
    s_broad = safe(broad_safety_pointwise)
    out["broad_safety"] = s_broad
    if broad_valid and material(fam["broad_minus_new_pier_gain"]) and all(s_broad.values()) and \
            hi(fam["outside_harm_diff"]) is not None and hi(fam["outside_harm_diff"]) <= e["outside_target_harm_upper_bound"]:
        out["label"] = "P2_DIR_SELECTIVE_GATE_LIMIT_SUSPECTED"
        return out
    out["label"] = "P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED"
    return out


# ---- Exp-1 loading / records ----------------------------------------------------------------------

def unpack_bf16(a: np.ndarray) -> np.ndarray:
    u = np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def load_exp1(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"] or m["stage"] != "exp1":
        raise ValueError("invalid exp1 manifest")
    con = json.loads((ROOT / m["construction"]).read_text())
    pos = json.loads((ROOT / m["positions"]).read_text())
    if pos["positions_hash"] != m["positions_hash"] or con["construction_hash"] != m["construction_hash"]:
        raise ValueError("population hash mismatch")
    sealed = json.loads((run / "directions_sealed.json").read_text()) if (run / "directions_sealed.json").exists() else {}
    rows, evals = {}, {}
    for i, uid in enumerate(con["utterances"]):
        for kind, store in (("", rows), ("_eval", evals)):
            p = run / "rows" / f"{i:03d}{kind}.json"
            if not p.exists():
                store[uid] = {"status": "missing"}
                continue
            row = json.loads(p.read_text())
            if row["manifest_hash"] != m["manifest_hash"] or row["identity"] != uid:
                raise ValueError(f"stale row {p}")
            if row.get("status") == "ok":
                npz = run / "rows" / f"{i:03d}{kind}.npz"
                if file_hash(npz) != row["vectors_sha256"]:
                    raise ValueError(f"vector hash {npz}")
                with np.load(npz) as z:
                    row["_vec"] = {k: z[k] for k in z.files}
                if kind == "":
                    lg = run / "rows" / f"{i:03d}_logits.npz"
                    if file_hash(lg) != row["logits_sha256"]:
                        raise ValueError(f"logits hash {lg}")
                    with np.load(lg) as z:
                        row["_logits"] = {k: unpack_bf16(z[k]) for k in z.files}
            store[uid] = row
    rt = json.loads((run / "runtime.json").read_text()) if (run / "runtime.json").exists() else {}
    rte = json.loads((run / "runtime_eval.json").read_text()) if (run / "runtime_eval.json").exists() else {}
    return {"manifest": m, "construction": con, "positions": pos, "rows": rows, "evals": evals, "sealed": sealed,
            "runtime": rt, "runtime_eval": rte}


def position_records(data: dict, partition: dict, suppress, begin, e_star: float, cfg: dict) -> list[dict]:
    rows, evals = data["rows"], data["evals"]
    tol_e = cfg["energy"]["max_relative_squared_error"]
    out = []
    for p in data["positions"]["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        r = {"utterance_id": uid, "t": t, "dialogue_id": p["dialogue_id"], "stratum": p["stratum"],
             "span_initial": p.get("span_initial"), "present": False}
        row, ev = rows.get(uid, {}), evals.get(uid, {})
        rec = row.get("positions", {}).get(str(t)) if row.get("status") == "ok" else None
        erec = ev.get("positions", {}).get(str(t)) if ev.get("status") == "ok" else None
        if rec is None or erec is None:
            out.append(r)
            continue
        r["present"] = True
        pre = f"t{t}_"
        vec, lg, evv = row["_vec"], row["_logits"], ev["_vec"]
        Y, cstar = [int(x) for x in p["target_ids"]], int(p["competitor"])
        none = logit_metrics(lg[pre + "none"], t, suppress, begin, partition, Y, cstar)
        r["none"] = none
        idt, ref = rec["identity"], p["p2r"]
        lp_none = processed(lg[pre + "none"], t, suppress, begin)
        lp_none = lp_none - _lse(lp_none)
        state = {
            "argmax": idt["argmax"] == p["baseline_token"] == none["top1"] and
                      (ref["argmax"] == p["baseline_token"] or lp_none[ref["argmax"]] == lp_none[p["baseline_token"]]),
            "logp_ref": abs(none["logp_ref"] - ref["logp_ref"]) <= 1e-3,
            "margin": abs(none["m"] - ref["margin"]) <= 1e-3 and abs(none["margin_best_other"] - ref["margin"]) <= 1e-3,
            "competitor": none["best_other"] == cstar or none["logp_competitor"] == none["max_other_logp"],
            "r_norm": abs(idt["pre_norm_audit_style"] - ref["pre_norm"]) <= 1e-6 * ref["pre_norm"],
            "cos_d_state": idt["cos_d_state"] is not None and abs(idt["cos_d_state"] - ref["cos_d_state"]) <= 1e-6,
            "dir": rec["directions"]["D0"]["status"] == "ok" and ref["dir"] == "ok",
            "d0_solver_s": rec["arms"]["D0"]["solver"].get("status") == "ok" and ref["arms"]["plus_d"]["status"] == "ok"
                           and abs(rec["arms"]["D0"]["solver"]["s"] - ref["arms"]["plus_d"]["s"]) <= 1e-9 * abs(ref["arms"]["plus_d"]["s"]),
        }
        eng = {"extracted_hb_bitwise": idt["extracted_hb_bitwise"], "extracted_he_bitwise": idt["extracted_he_bitwise"],
               "d2_logits_bitwise": idt["d2_logits_bitwise"], "d2_site_bitwise": idt["d2_site_bitwise"],
               "restore_bitwise": rec["restore_bitwise"],
               "directions_unchanged": all(rec["directions"][a].get("unchanged_after_pulses") for a in ARMS),
               "eval_logits_bitwise": erec["logits_bitwise"], "eval_site_bitwise": erec["site_bitwise"],
               "eval_extracted_bitwise": erec["extracted_hb_bitwise"]}
        r["state_checks"], r["engineering_checks"] = state, eng
        r["state_ok"], r["engineering_ok"] = all(state.values()), all(eng.values())
        hb = vec[pre + "hb"].astype(np.float64)
        hnorm = float(np.linalg.norm(hb))
        g_m = evv[pre + "g_margin"].astype(np.float64)
        g_tan = tangent(g_m, hb)
        g_tan_n = float(np.linalg.norm(g_tan))
        r["g_margin_tan_norm"] = g_tan_n
        r["A"] = g_tan_n * e_star
        d2v = vec.get(pre + "D2")
        r["arms"] = {}
        for a in ARMS:
            ar = rec["arms"][a]
            dr = rec["directions"][a]
            x = {"direction_status": dr["status"], "direction_reason": dr["reason"], "direction_norm": dr["norm"],
                 "direction_sha256": dr["sha256"], "fold": dr["provenance"].get("fold_dialogue"),
                 "solver_status": ar["solver"].get("status"), "edit_norm": ar["edit_norm"],
                 "steered": ar["steered"], "pulse_runtime_sec": ar["pulse_runtime_sec"]}
            if a == "D2":
                x["readout"] = {k: dr["provenance"].get(k) for k in ("J", "log_PE", "log_PM", "g_norm", "tangent_norm",
                                                                      "g_radial", "unit_error", "radial_dot")}
                x["autograd_calls"] = dr["counters"].get("autograd_calls")
            post = vec.get(pre + "post_" + a)
            valid = (dr["status"] == "ok" and ar["solver"].get("status") == "ok" and ar["steered"]
                     and ar.get("solver_matches_hook", False) and post is not None
                     and abs(ar["edit_norm"] ** 2 / e_star ** 2 - 1.0) <= tol_e)
            x["valid"] = bool(valid)
            if post is not None:
                edit = post.astype(np.float64) - hb
                en = float(np.linalg.norm(edit))
                x["realized_edit_norm_states"] = en
                x["relative_edit"] = en / hnorm
                x["kappa"] = float(g_tan @ edit) / (g_tan_n * e_star) if g_tan_n > 0 else None
            dv = vec.get(pre + a)
            x["cos_gref_tan"] = cos(g_tan, dv)
            x["cos_readout"] = cos(dv, d2v)
            x["cos_d0"] = cos(dv, vec.get(pre + "D0"))
            am = logit_metrics(lg[pre + a], t, suppress, begin, partition, Y, cstar)
            x["post"] = am
            x["d_m"] = am["m"] - none["m"]
            x["d_logp_ref"] = am["logp_ref"] - none["logp_ref"]
            x["d_J"] = am["J"] - none["J"]
            x["d_log_PE"] = am["log_PE"] - none["log_PE"]
            x["d_log_PM"] = am["log_PM"] - none["log_PM"]
            x["d_rank_ref"] = am["rank_ref"] - none["rank_ref"]
            x["correct_after"] = am["top1_in_ref"]
            x["top1_changed"] = am["top1"] != none["top1"]
            r["arms"][a] = x
        ens = [r["arms"][a]["edit_norm"] ** 2 for a in ARMS if r["arms"][a]["valid"]]
        r["pairwise_energy_ok"] = len(ens) == 3 and max(ens) / min(ens) - 1.0 <= cfg["energy"]["max_pairwise_relative_squared_error"]
        r["matched"] = bool(r["state_ok"] and r["pairwise_energy_ok"] and all(r["arms"][a]["valid"] for a in ARMS))
        out.append(r)
    return out


def _vals(recs, fn):
    return [(r["dialogue_id"], fn(r)) for r in recs]


def analyze_exp1(run: Path) -> dict:
    from transformers import WhisperProcessor, GenerationConfig
    from csasr.inference_cf.core_r2 import tokenizer_partition
    data = load_exp1(run)
    m = data["manifest"]
    cfg = json.loads((ROOT / m["config"]).read_text())
    if digest(cfg) != m["config_hash"]:
        raise ValueError("config changed after manifest")
    model = Path(m["model"]["dir"])
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    if partition["hash"] != m["partition_hash"]:
        raise ValueError("partition hash")
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if digest({"suppress": suppress, "begin": begin}) != m["suppression_hash"]:
        raise ValueError("suppression hash")
    e_star = float(cfg["energy"]["e_star"])
    bcfg = cfg["bootstrap"]
    reps, seed = bcfg["replicates"], bcfg["seed"]
    recs = position_records(data, partition, suppress, begin, e_star, cfg)
    N = len(recs)
    present = [r for r in recs if r["present"]]
    mismatch = [(r["utterance_id"], r["t"], [k for k, v in r["state_checks"].items() if not v]) for r in present if not r["state_ok"]]
    eng_fail = [(r["utterance_id"], r["t"], [k for k, v in r["engineering_checks"].items() if not v]) for r in present if not r["engineering_ok"]]
    rows_ok = all(v.get("status") == "ok" for v in data["rows"].values()) and all(v.get("status") == "ok" for v in data["evals"].values())
    valid_rate = {a: sum(1 for r in present if r["state_ok"] and r["arms"][a]["valid"]) / N for a in ARMS}
    matched = [r for r in recs if r.get("matched")]
    by_s = {s: [r for r in matched if r["stratum"] == s] for s in STRATA}
    pop = {s: {"positions": len(v), "dialogues": len({r["dialogue_id"] for r in v})} for s, v in by_s.items()}
    population_ok = all(pop[s]["positions"] >= cfg["exp1"]["matched_min_positions_per_stratum"] and
                        pop[s]["dialogues"] >= cfg["exp1"]["matched_min_dialogues_per_stratum"] for s in STRATA)
    eng = {"rows_complete": rows_ok and len(present) == N == 180,
           "runtime_completed": data["runtime"].get("status") == "completed" and data["runtime_eval"].get("status") == "completed",
           "directions_sealed": data["sealed"].get("status") == "SEALED",
           "engineering_bitwise": not eng_fail,
           "state_identity": len(mismatch) <= MAX_STATE_MISMATCH,
           "d0_valid_rate": valid_rate["D0"] >= cfg["exp1"]["candidate_valid_rate_min"]}
    # shared dialogue draws over the frozen population's dialogues
    keys, W = draw_weights([p["dialogue_id"] for p in data["positions"]["positions"]], reps, seed)
    a_fam = 0.05 / cfg["bootstrap"]["family_size_exp1"]
    correct = [r for r in matched if r["stratum"] != "EN-confusion"]
    fam, observed, q = {}, {}, {}
    for c in CANDIDATES:
        f = {"conf": boot_stat(_vals(by_s["EN-confusion"], lambda r: r["arms"][c]["d_m"]), keys, W, a_fam),
             "paired": boot_stat(_vals(by_s["EN-confusion"], lambda r: r["arms"][c]["d_m"] - r["arms"]["D0"]["d_m"]), keys, W, a_fam),
             "en": boot_stat(_vals(by_s["EN-correct"], lambda r: r["arms"][c]["d_m"]), keys, W, a_fam),
             "zh": boot_stat(_vals(by_s["ZH-correct"], lambda r: r["arms"][c]["d_m"]), keys, W, a_fam),
             "corr": boot_stat(_vals(correct, lambda r: 0.0 if r["arms"][c]["correct_after"] else 1.0), keys, W, a_fam)}
        fam[c] = f
        observed[c] = {s: boot_stat(_vals(by_s[s], lambda r: 0.0 if r["arms"][c]["correct_after"] else 1.0), keys, W, 0.05)
                       for s in ("EN-correct", "ZH-correct")}
        obs_pt = {s: v["estimate"] for s, v in observed[c].items()}
        q[c] = qualify_exp1(f, obs_pt, valid_rate[c], all(eng.values()) and
                            all(x["valid_draws"] >= MIN_VALID_DRAWS for x in f.values()), population_ok, cfg)
    tie = boot_stat(_vals(by_s["EN-confusion"], lambda r: r["arms"]["D2"]["d_m"] - r["arms"]["D1"]["d_m"]), keys, W, 0.05)
    eng["bootstrap_draws"] = all(x["valid_draws"] >= MIN_VALID_DRAWS for f in fam.values() for x in f.values())
    decision = select_exp1(all(eng.values()), q, tie, cfg)
    res = {"schema": "p2dir_exp1_analysis_v1", "manifest_hash": m["manifest_hash"], "config_hash": m["config_hash"],
           "e_star": e_star, "bootstrap": {"replicates": reps, "seed": seed, "dialogues": keys,
                                           "family_alpha": a_fam, "percentiles": [a_fam / 2, 1 - a_fam / 2]},
           "engineering": eng, "population": pop, "population_ok": population_ok, "valid_rate": valid_rate,
           "counts": {"positions": N, "present": len(present), "matched": len(matched), "state_mismatch": mismatch,
                      "engineering_failures": eng_fail},
           "family": fam, "observed_corruption": observed, "qualification": q, "tie_D2_minus_D1": tie,
           "decision": decision}
    res["descriptive"] = describe_exp1(recs, matched, keys, W)
    res["runtime"] = {"exp1": data["runtime"], "eval": data["runtime_eval"]}
    res["per_position"] = [compact(r) for r in recs]
    return res


def describe_exp1(recs, matched, keys, W) -> dict:
    """All arms (D0 included), all strata; pointwise 95%. Valid-only (matched) and ITT tables."""
    out = {}
    stat_fns = {"d_m": lambda r, a: r["arms"][a]["d_m"], "d_logp_ref": lambda r, a: r["arms"][a]["d_logp_ref"],
                "d_J": lambda r, a: r["arms"][a]["d_J"], "d_log_PE": lambda r, a: r["arms"][a]["d_log_PE"],
                "d_log_PM": lambda r, a: r["arms"][a]["d_log_PM"], "d_rank_ref": lambda r, a: r["arms"][a]["d_rank_ref"],
                "correct_after": lambda r, a: 1.0 if r["arms"][a]["correct_after"] else 0.0,
                "top1_changed": lambda r, a: 1.0 if r["arms"][a]["top1_changed"] else 0.0,
                "cos_gref_tan": lambda r, a: r["arms"][a]["cos_gref_tan"], "kappa": lambda r, a: r["arms"][a]["kappa"],
                "cos_readout": lambda r, a: r["arms"][a]["cos_readout"], "cos_d0": lambda r, a: r["arms"][a]["cos_d0"],
                "edit_norm": lambda r, a: r["arms"][a]["edit_norm"], "relative_edit": lambda r, a: r["arms"][a].get("relative_edit"),
                "post_margin_best_other": lambda r, a: r["arms"][a]["post"]["margin_best_other"],
                "post_P_E": lambda r, a: r["arms"][a]["post"]["P_E"], "post_P_M": lambda r, a: r["arms"][a]["post"]["P_M"]}
    for s in STRATA:
        rs = [r for r in matched if r["stratum"] == s]
        out[s] = {"valid_only": {a: {k: boot_stat(_vals(rs, lambda r, f=f, a=a: f(r, a)), keys, W, 0.05)
                                     for k, f in stat_fns.items()} for a in ARMS}}
        itt = [r for r in recs if r["stratum"] == s and r.get("state_ok")]
        out[s]["itt_zero_change"] = {a: {
            "d_m": boot_stat(_vals(itt, lambda r, a=a: r["arms"][a]["d_m"] if r["arms"][a]["valid"] else 0.0), keys, W, 0.05),
            "changed_top1": boot_stat(_vals(itt, lambda r, a=a: (1.0 if r["arms"][a]["top1_changed"] else 0.0)
                                            if r["arms"][a]["valid"] else 0.0), keys, W, 0.05),
            "n": len(itt)} for a in ARMS}
        out[s]["none"] = {"m0": boot_stat(_vals(rs, lambda r: r["none"]["m"]), keys, W, 0.05),
                          "A": boot_stat(_vals(rs, lambda r: r["A"]), keys, W, 0.05),
                          "J0": boot_stat(_vals(rs, lambda r: r["none"]["J"]), keys, W, 0.05)}
        out[s]["counts"] = {a: {"corrections_or_retained": sum(r["arms"][a]["correct_after"] for r in rs),
                                "top1_changed": sum(r["arms"][a]["top1_changed"] for r in rs), "n": len(rs)} for a in ARMS}
    present = [r for r in recs if r["present"]]
    out["direction_status"] = {a: {"ok": sum(r["arms"][a]["direction_status"] == "ok" for r in present),
                                   "reasons": sorted({str(r["arms"][a]["direction_reason"]) for r in present
                                                      if r["arms"][a]["direction_status"] != "ok"}),
                                   "solver_unreachable": sum(r["arms"][a]["solver_status"] == "energy_unreachable" for r in present)}
                               for a in ARMS}
    out["d2_autograd_calls"] = sum(r["arms"]["D2"].get("autograd_calls") or 0 for r in present)
    out["pulse_runtime_sec"] = {a: float(np.mean([r["arms"][a]["pulse_runtime_sec"] for r in present])) for a in ARMS}
    return out


def compact(r: dict) -> dict:
    """Committed per-position evidence (no vectors/logits)."""
    if not r["present"]:
        return r
    keep = {k: r[k] for k in ("utterance_id", "t", "dialogue_id", "stratum", "span_initial", "state_ok",
                              "engineering_ok", "state_checks", "engineering_checks", "matched", "pairwise_energy_ok",
                              "A", "g_margin_tan_norm")}
    keep["none"] = {k: r["none"][k] for k in ("m", "logp_ref", "rank_ref", "top1", "top1_in_ref", "P_E", "P_M", "J",
                                               "margin_best_other")}
    keep["arms"] = {}
    for a, x in r["arms"].items():
        y = {k: v for k, v in x.items() if k != "post"}
        y["post"] = {k: x["post"][k] for k in ("m", "logp_ref", "rank_ref", "top1", "top1_in_ref", "P_E", "P_M", "J",
                                               "margin_best_other")}
        keep["arms"][a] = y
    return keep


def jsonable(obj):
    """Post-hoc serialization only (numpy scalars -> Python); never alters a value or decision."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("exp1",))
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = jsonable(analyze_exp1(ROOT / args.run))
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    atomic_json(out, res)
    print(json.dumps({"decision": res["decision"], "engineering": res["engineering"], "valid_rate": res["valid_rate"],
                      "population": res["population"],
                      "family": {c: {k: [v["estimate"], v["ci"]] for k, v in f.items()} for c, f in res["family"].items()},
                      "qualification": {c: v["checks"] for c, v in res["qualification"].items()}}, indent=1))


if __name__ == "__main__":
    main()
