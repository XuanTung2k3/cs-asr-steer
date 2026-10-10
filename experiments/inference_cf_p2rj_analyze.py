#!/usr/bin/env python
"""P2-RJ analysis (CPU, evaluator side): validity V1-V7, leverage / alignment statistics and the
frozen decision rules of docs/inference_cf/P2_RJ_JACOBIAN_DIAGNOSIS_SPEC.md section 7.

All decision logic below is frozen with the P2-RJ spec before any P2-RJ outcome exists.
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
from experiments.inference_cf_p2r_analyze import dialogue_bootstrap, hi, lo

ARMS = ("plus_d", "minus_d", "random")
TARGETS = ("margin", "logp_ref", "log_PE", "log_PM")
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
PASSES = ("bf16", "fp32")
DECISION_FAMILY = ("Q50", "K_plus")
MAX_STATE_MISMATCH = 9          # > 5% of 180 positions -> run invalid (implementation defect)


# ---- frozen classification ------------------------------------------------------------------

def leverage_class(q50: dict) -> str:
    if hi(q50) is not None and hi(q50) < 0.5:
        return "WEAK"
    if lo(q50) is not None and lo(q50) > 0.5:
        return "SUBSTANTIAL"
    return "UNRESOLVED"


def alignment_class(k_plus: dict) -> str:
    if hi(k_plus) is not None and hi(k_plus) < 0.10:
        return "MISALIGNED"
    if lo(k_plus) is not None and lo(k_plus) >= 0.25:
        return "ALIGNED"
    return "PARTIAL"


def label(leverage: str, alignment: str) -> str:
    if leverage == "SUBSTANTIAL" and alignment == "MISALIGNED":
        return "P2_RJ_DIRECTION_ISSUE"
    if leverage == "WEAK" and alignment == "MISALIGNED":
        return "P2_RJ_DIRECTION_AND_SITE_ISSUE"
    if leverage == "WEAK" and alignment in ("PARTIAL", "ALIGNED"):
        return "P2_RJ_SITE_OR_SENSITIVITY_ISSUE"
    return "P2_RJ_STILL_AMBIGUOUS"


def decide(valid: bool, leverage: str, alignment: str, fp32_label: str | None, fd_ok: bool,
           precision_cos_ok: bool) -> dict:
    """Primary diagnosis: the bf16 label, only when V1-V7 all hold; otherwise STILL_AMBIGUOUS."""
    bf16_label = label(leverage, alignment)
    reasons = []
    if not valid:
        reasons.append("validity_V1_to_V5")
    if not fd_ok:
        reasons.append("V6_first_order_sanity")
    if not precision_cos_ok:
        reasons.append("V7_precision_gradient_cosine")
    if fp32_label != bf16_label:
        reasons.append("V7_precision_label_change")
    return {"bf16_label": bf16_label, "fp32_label": fp32_label, "blocking": reasons,
            "diagnosis": bf16_label if not reasons else "P2_RJ_STILL_AMBIGUOUS"}


# ---- geometry (recomputed from the saved vectors) --------------------------------------------

def tangent(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    rh = r / np.linalg.norm(r)
    return g - rh * float(rh @ g)


def recompute(vec: dict, t: int, e_star: float) -> dict:
    """rho-independent per-position quantities from the float vectors (float64 arithmetic)."""
    k = f"t{t}_"
    r = vec[k + "r"].astype(np.float64)
    g = vec[k + "g_margin"].astype(np.float64)
    gt = tangent(g, r)
    A = float(np.linalg.norm(gt)) * e_star
    out = {"g_norm": float(np.linalg.norm(g)), "g_tan_norm": float(np.linalg.norm(gt)), "A": A, "arms": {}}
    dirs = {}
    if k + "d" in vec:
        d = vec[k + "d"].astype(np.float64)
        d = d / np.linalg.norm(d)
        dirs = {"plus_d": d, "minus_d": -d}
    v = vec[k + "v_random"].astype(np.float64)
    dirs["random"] = v / np.linalg.norm(v)
    for a, u in dirs.items():
        rec = {"cos_gtan_v": float(gt @ u) / float(np.linalg.norm(gt))}
        if k + "delta_" + a in vec:
            dl = vec[k + "delta_" + a].astype(np.float64)
            rec["pred_margin"] = float(g @ dl)
            rec["kappa"] = float(gt @ dl) / A if A > 0 else None
        out["arms"][a] = rec
    return out


# ---- loading / records -----------------------------------------------------------------------

def load(run: Path) -> tuple[dict, dict, dict, dict]:
    m = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("manifest hash")
    cfg = json.loads((ROOT / m["config"]).read_text())
    pos = json.loads((ROOT / m["positions"]).read_text())
    rows = {mode: {} for mode in PASSES}
    for mode in PASSES:
        for i, uid in enumerate(pos["utterances"]):
            p = run / "rows" / f"{i:03d}_{mode}.json"
            if not p.exists():
                rows[mode][uid] = {"status": "missing"}
                continue
            row = json.loads(p.read_text())
            if row["manifest_hash"] != m["manifest_hash"] or row["identity"] != uid:
                raise ValueError(f"stale row {p}")
            npz = run / "rows" / f"{i:03d}_{mode}.npz"
            if row.get("status") == "ok":
                if file_hash(npz) != row["vectors_sha256"]:
                    raise ValueError(f"vector file hash mismatch {npz}")
                with np.load(npz) as z:
                    row["_vec"] = {k: z[k] for k in z.files}
            rows[mode][uid] = row
    return m, cfg, pos, rows


def state_checks(p: dict, rec: dict) -> dict:
    """V3/V4 per position (bf16 pass): identity with the saved P2-R run3 state."""
    idt, ref = rec["identity"], p["p2r"]
    # v1.1 (pre-outcome): the decode-rule argmax (first index) must be the baseline token; the P2-R
    # summary argmax (topk order) may differ only by an exact logit tie with it.
    c = {
        "argmax": idt["argmax"] == p["baseline_token"] and
                  (ref["argmax"] == p["baseline_token"] or idt["logp_p2r_argmax"] == idt["logp_baseline_token"]),
        "logp_ref": abs(idt["logp_ref"] - ref["logp_ref"]) <= 1e-3,
        "margin": abs(rec["values"]["margin"] - ref["margin"]) <= 1e-3
                  and abs(idt["margin_best_other"] - ref["margin"]) <= 1e-3,
        "competitor": idt["competitor_recomputed"] == p["competitor"] or idt["logp_competitor"] == idt["max_other_logp"],
        "r_norm": abs(idt["pre_norm_audit_style"] - ref["pre_norm"]) <= 1e-6 * ref["pre_norm"],
        "cos_d_state": idt["cos_d_state"] is not None and abs(idt["cos_d_state"] - ref["cos_d_state"]) <= 1e-6,
        "dir": idt["dir"] == ref["dir"] == "ok",
        "solver_s": all(rec["solver"].get(a, {}).get("status") == "ok" and
                        abs(rec["solver"][a]["s"] - ref["arms"][a]["s"]) <= 1e-9 * abs(ref["arms"][a]["s"])
                        for a in ARMS),
        "logits_bitwise": bool(idt["logits_bitwise_equal"]),
        "site_equals_recorder": bool(idt["site_equals_recorder"]),
    }
    return c


def finite_nonzero(rec: dict) -> bool:
    ts = rec["stats"]["targets"]
    return all(math.isfinite(ts[k]["g_norm"]) and ts[k]["g_norm"] > 0 for k in TARGETS)


def records(pos: dict, rows: dict, mode: str, e_star: float) -> list[dict]:
    out = []
    for p in pos["positions"]:
        row = rows[mode].get(p["utterance_id"], {})
        rec = row.get("positions", {}).get(str(p["t"])) if row.get("status") == "ok" else None
        r = {"stratum": p["stratum"], "dialogue_id": p["dialogue_id"], "utterance_id": p["utterance_id"],
             "t": p["t"], "span_initial": p.get("span_initial"), "present": rec is not None}
        if rec is None:
            out.append(r); continue
        checks = state_checks(p, rec) if mode == "bf16" else {"logits_bitwise": bool(rec["identity"]["logits_bitwise_equal"])}
        r["checks"] = checks
        r["state_ok"] = all(checks.values())
        r["finite"] = finite_nonzero(rec)
        st = rec["stats"]
        m0 = rec["values"]["margin"]
        r["m0"] = m0
        r["gap"] = -m0
        r["A"] = st["A"]
        if p["stratum"] == "EN-confusion":
            r["rho"] = math.inf if r["gap"] <= 0 else st["A"] / r["gap"]
        else:
            r["surplus_ratio"] = st["A"] / m0 if m0 > 0 else math.inf
        r["g_norm"] = st["targets"]["margin"]["g_norm"]
        r["g_tan_norm"] = st["targets"]["margin"]["g_tan_norm"]
        r["g_rad"] = st["targets"]["margin"]["g_rad"]
        for k in ("logp_ref", "log_PE", "log_PM"):
            r[f"{k}_g_tan_norm"] = st["targets"][k]["g_tan_norm"]
            r[f"{k}_cos_with_margin_tan"] = st["targets"][k]["cos_with_margin_tan"]
        for a in ARMS:
            x = st["arms"].get(a, {})
            r[f"kappa_{a}"] = x.get("kappa")
            r[f"cos_{a}"] = x.get("cos_gtan_v_margin")
            for k in TARGETS:
                r[f"pred_{k}_{a}"] = x.get(f"pred_{k}")
            r[f"cos_{a}_log_PE"] = x.get("cos_gtan_v_log_PE")
            r[f"cos_{a}_log_PM"] = x.get("cos_gtan_v_log_PM")
            r[f"cos_{a}_logp_ref"] = x.get("cos_gtan_v_logp_ref")
            ref = p["p2r"]["arms"][a]
            r[f"obs_margin_{a}"] = ref["obs_d_margin"]
            r[f"obs_logp_ref_{a}"] = ref["obs_d_logp_ref"]
        r["_vec_key"] = (p["utterance_id"], p["t"])
        out.append(r)
    return out


# ---- statistics ------------------------------------------------------------------------------

def boot(recs, key, reps, seed, alpha, fn=None):
    vals = []
    for r in recs:
        v = fn(r) if fn else r.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append((r["dialogue_id"], float(v)))
    return dialogue_bootstrap(vals, reps, seed, alpha)


def pooled_pearson_boot(pairs: list[tuple[str, float, float]], reps: int, seed: int, alpha: float) -> dict:
    """Pearson r over pooled (pred, obs) pairs; dialogue-cluster percentile bootstrap."""
    by = defaultdict(list)
    for dlg, x, y in pairs:
        by[dlg].append((x, y))
    keys = sorted(by)
    def r_of(ks):
        xy = np.array([p for k in ks for p in by[k]])
        if len(xy) < 3 or xy[:, 0].std() == 0 or xy[:, 1].std() == 0:
            return float("nan")
        return float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1])
    est = r_of(keys)
    rng = np.random.default_rng(seed)
    draws = np.array([r_of([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(reps)])
    draws = draws[np.isfinite(draws)]
    xy = np.array([p for k in keys for p in by[k]])
    slope = float((xy[:, 0] @ xy[:, 1]) / (xy[:, 0] @ xy[:, 0])) if len(xy) else None
    return {"estimate": est, "ci": [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))],
            "pairs": len(xy), "dialogues": len(keys), "slope_through_origin": slope}


def quantiles(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.array(xs, dtype=float)
    return {"n": len(a), "median": float(np.median(a)), "q25": float(np.quantile(a, .25)),
            "q75": float(np.quantile(a, .75)), "min": float(a.min()), "max": float(a.max())}


def leverage_alignment(recs: list[dict], reps: int, seed: int, alpha_family: float) -> dict:
    en = [r for r in recs if r["stratum"] == "EN-confusion"]
    q50 = boot(en, None, reps, seed, alpha_family, fn=lambda r: 1.0 if r["rho"] >= 0.5 else 0.0)
    kp = boot(en, "kappa_plus_d", reps, seed, alpha_family)
    return {"Q50": q50, "K_plus": kp, "leverage": leverage_class(q50), "alignment": alignment_class(kp)}


def describe(recs: list[dict], reps: int, seed: int) -> dict:
    out = {}
    for s in STRATA:
        rs = [r for r in recs if r["stratum"] == s]
        d = {"n": len(rs), "dialogues": len({r["dialogue_id"] for r in rs}),
             "m0": quantiles([r["m0"] for r in rs]), "g_norm": quantiles([r["g_norm"] for r in rs]),
             "g_tan_norm": quantiles([r["g_tan_norm"] for r in rs]), "A": quantiles([r["A"] for r in rs]),
             "g_rad_abs": quantiles([abs(r["g_rad"]) for r in rs])}
        if s == "EN-confusion":
            d["gap"] = quantiles([r["gap"] for r in rs])
            d["rho"] = quantiles([r["rho"] for r in rs])
            for th in (0.1, 0.5, 1.0):
                d[f"frac_rho_ge_{th}"] = {
                    "positions": float(np.mean([r["rho"] >= th for r in rs])) if rs else None,
                    "dialogue_weighted": boot(rs, None, reps, seed, 0.05, fn=lambda r, th=th: 1.0 if r["rho"] >= th else 0.0)}
        else:
            d["surplus_ratio"] = quantiles([r["surplus_ratio"] for r in rs])
            d["frac_surplus_ratio_ge_1"] = float(np.mean([r["surplus_ratio"] >= 1 for r in rs])) if rs else None
        for a in ARMS:
            d[a] = {"kappa": boot(rs, f"kappa_{a}", reps, seed, 0.05),
                    "cos_gtan_v": boot(rs, f"cos_{a}", reps, seed, 0.05),
                    "pred_margin": boot(rs, f"pred_margin_{a}", reps, seed, 0.05),
                    "obs_margin_p2r": boot(rs, f"obs_margin_{a}", reps, seed, 0.05),
                    "pred_logp_ref": boot(rs, f"pred_logp_ref_{a}", reps, seed, 0.05),
                    "obs_logp_ref_p2r": boot(rs, f"obs_logp_ref_{a}", reps, seed, 0.05),
                    "pred_log_PE": boot(rs, f"pred_log_PE_{a}", reps, seed, 0.05),
                    "cos_gtan_logPE_v": boot(rs, f"cos_{a}_log_PE", reps, seed, 0.05),
                    "cos_gtan_logPM_v": boot(rs, f"cos_{a}_log_PM", reps, seed, 0.05),
                    "cos_gtan_logpref_v": boot(rs, f"cos_{a}_logp_ref", reps, seed, 0.05),
                    "abs_cos_gtan_v_median": (quantiles([abs(r[f"cos_{a}"]) for r in rs if r.get(f"cos_{a}") is not None]) or {}).get("median")}
        d["K_plus_minus_K_random"] = boot(rs, None, reps, seed, 0.05,
                                          fn=lambda r: None if r["kappa_plus_d"] is None or r["kappa_random"] is None
                                          else r["kappa_plus_d"] - r["kappa_random"])
        for k in ("logp_ref", "log_PE", "log_PM"):
            d[f"cos_margin_tan_vs_{k}_tan"] = boot(rs, f"{k}_cos_with_margin_tan", reps, seed, 0.05)
        out[s] = d
    en = [r for r in recs if r["stratum"] == "EN-confusion"]
    out["EN-confusion_span"] = {
        name: {"n": len(sub), "rho": quantiles([r["rho"] for r in sub]),
               "kappa_plus_d": boot(sub, "kappa_plus_d", reps, seed, 0.05),
               "A": quantiles([r["A"] for r in sub]), "gap": quantiles([r["gap"] for r in sub])}
        for name, sub in (("span_initial", [r for r in en if r["span_initial"]]),
                          ("within_span", [r for r in en if not r["span_initial"]]))}
    return out


def fd_check(recs: list[dict], reps: int, seed: int) -> dict:
    pairs = [(r["dialogue_id"], r[f"pred_margin_{a}"], r[f"obs_margin_{a}"]) for r in recs for a in ARMS
             if r.get(f"pred_margin_{a}") is not None and r.get(f"obs_margin_{a}") is not None]
    out = {"pooled_margin": pooled_pearson_boot(pairs, reps, seed, 0.05)}
    pl = [(r["dialogue_id"], r[f"pred_logp_ref_{a}"], r[f"obs_logp_ref_{a}"]) for r in recs for a in ARMS
          if r.get(f"pred_logp_ref_{a}") is not None]
    out["pooled_logp_ref"] = pooled_pearson_boot(pl, 2000, seed, 0.05)
    for s in STRATA:
        for a in ARMS:
            pp = [(r["dialogue_id"], r[f"pred_margin_{a}"], r[f"obs_margin_{a}"]) for r in recs
                  if r["stratum"] == s and r.get(f"pred_margin_{a}") is not None and r.get(f"obs_margin_{a}") is not None]
            out[f"{s}.{a}"] = pooled_pearson_boot(pp, 2000, seed, 0.05) if len(pp) >= 3 else None
    out["ok"] = out["pooled_margin"]["ci"][0] > 0
    return out


def precision_cos(rows: dict, pos: dict, keys: set) -> dict:
    """V7: cos(g_tan bf16, g_tan fp32) over the valid positions ``keys``."""
    cs = []
    for i, uid in enumerate(pos["utterances"]):
        a, b = rows["bf16"].get(uid, {}), rows["fp32"].get(uid, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        for t in a["positions"]:
            k = f"t{t}_"
            if (uid, int(t)) not in keys or k + "g_margin" not in b["_vec"]:
                continue
            ga = tangent(a["_vec"][k + "g_margin"].astype(np.float64), a["_vec"][k + "r"].astype(np.float64))
            gb = tangent(b["_vec"][k + "g_margin"].astype(np.float64), b["_vec"][k + "r"].astype(np.float64))
            cs.append((uid, int(t), float(ga @ gb / (np.linalg.norm(ga) * np.linalg.norm(gb))),
                       float(np.linalg.norm(gb) / np.linalg.norm(ga))))
    return {"n": len(cs), "cos": quantiles([c[2] for c in cs]), "norm_ratio_fp32_over_bf16": quantiles([c[3] for c in cs]),
            "per_position": cs}


def analyze(run: Path) -> dict:
    m, cfg, pos, rows = load(run)
    an = cfg["analysis"]
    reps, seed = an["bootstrap"]["replicates"], an["bootstrap"]["seed"]
    alpha_family = 0.05 / len(DECISION_FAMILY)
    e_star = float(cfg["energy"]["e_star"])
    runtime = json.loads((run / "runtime.json").read_text()) if (run / "runtime.json").exists() else {}
    res = {"schema": "p2rj_analysis_v1", "manifest_hash": m["manifest_hash"], "runtime": runtime}
    rec = {mode: records(pos, rows, mode, e_star) for mode in PASSES}
    # V2 completeness, V3 state identity, V4 finiteness (bf16 = primary)
    bf = rec["bf16"]
    present = sum(r["present"] for r in bf)
    mismatch = [(r["utterance_id"], r["t"], [k for k, v in r["checks"].items() if not v])
                for r in bf if r["present"] and not r["state_ok"]]
    nonfinite = [(r["utterance_id"], r["t"]) for r in bf if r["present"] and not r["finite"]]
    valid_bf = [r for r in bf if r["present"] and r["state_ok"] and r["finite"]]
    keys = {(r["utterance_id"], r["t"]) for r in valid_bf}
    valid_fp = [r for r in rec["fp32"] if r["present"] and r["finite"] and r["checks"]["logits_bitwise"]
                and (r["utterance_id"], r["t"]) in keys]
    en_valid = [r for r in valid_bf if r["stratum"] == "EN-confusion"]
    rows_ok = all(rows[mode][u].get("status") == "ok" for mode in PASSES for u in pos["utterances"])
    v = {"V2_completeness": rows_ok and present == len(pos["positions"]),
         "V3_state_identity": len(mismatch) <= MAX_STATE_MISMATCH,
         "V4_finite": not nonfinite,
         "V5_min_population": len(en_valid) >= cfg["population"]["minimum_interpretable"]["positions"]
         and len({r["dialogue_id"] for r in en_valid}) >= cfg["population"]["minimum_interpretable"]["dialogues"],
         "runtime_completed": runtime.get("status") == "completed"}
    res["validity"] = v
    res["counts"] = {"positions": len(pos["positions"]), "present_bf16": present, "state_mismatch": mismatch,
                     "nonfinite": nonfinite, "valid_bf16": len(valid_bf), "valid_fp32": len(valid_fp),
                     "valid_en_confusion": len(en_valid),
                     "valid_en_confusion_dialogues": len({r["dialogue_id"] for r in en_valid})}
    # independent-in-module recomputation from vectors must agree with the runner's stats
    diffs = []
    vec_by = {(u, int(t)): rows["bf16"][u]["_vec"] for u in pos["utterances"] if rows["bf16"][u].get("status") == "ok"
              for t in rows["bf16"][u]["positions"]}
    for r in valid_bf:
        rc = recompute(vec_by[(r["utterance_id"], r["t"])], r["t"], e_star)
        diffs.append(abs(rc["A"] - r["A"]) / max(r["A"], 1e-12))
        for a in ARMS:
            if rc["arms"].get(a, {}).get("kappa") is not None and r[f"kappa_{a}"] is not None:
                diffs.append(abs(rc["arms"][a]["kappa"] - r[f"kappa_{a}"]))
    res["vector_recompute_max_diff"] = max(diffs) if diffs else None
    # decision statistics (bf16 primary, fp32 precision replicate)
    res["bf16"] = leverage_alignment(valid_bf, reps, seed, alpha_family)
    res["fp32"] = leverage_alignment(valid_fp, reps, seed, alpha_family)
    res["fp32"]["label"] = label(res["fp32"]["leverage"], res["fp32"]["alignment"])
    res["fd"] = fd_check(valid_bf, reps, seed)
    pc = precision_cos(rows, pos, keys)
    res["precision"] = {k: v for k, v in pc.items() if k != "per_position"}
    res["precision"]["ok"] = pc["cos"] is not None and pc["cos"]["median"] >= 0.9
    valid = all(v.values()) and (res["vector_recompute_max_diff"] or 0) <= 1e-6
    res["decision"] = decide(valid, res["bf16"]["leverage"], res["bf16"]["alignment"], res["fp32"]["label"],
                             res["fd"]["ok"], res["precision"]["ok"])
    res["descriptive"] = {"bf16": describe(valid_bf, reps, seed), "fp32": describe(valid_fp, 2000, seed)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = analyze(ROOT / args.run)
    atomic_json(ROOT / args.out, res)
    print(json.dumps({"diagnosis": res["decision"], "validity": res["validity"],
                      "Q50": res["bf16"]["Q50"], "K_plus": res["bf16"]["K_plus"]}, indent=2))


if __name__ == "__main__":
    main()
