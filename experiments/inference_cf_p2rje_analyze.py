#!/usr/bin/env python
"""P2-RJ-E analysis (CPU): validity, the single frozen leverage statistic Q50 on the complete
eligible EN-confusion population, and the terminal diagnosis (spec section 6).

Frozen before any P2-RJ-E outcome. Reuses the P2-RJ frozen statistics (dialogue bootstrap,
leverage_class, state_checks) unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
import experiments.inference_cf_p2rj_analyze as rja

PASSES = ("bf16", "fp32")
ARMS = ("plus_d", "minus_d", "random")
ALPHA_PER_BOUND = 0.05 / 2          # P2-RJ Bonferroni decision-family multiplicity, unchanged
MISMATCH_FRACTION = 0.05


def final_label(valid: bool, lev_bf16: str, lev_fp32: str, precision_ok: bool) -> str:
    if not valid:
        return "INVALID_RUN_NO_DIAGNOSIS"
    if lev_bf16 != lev_fp32 or not precision_ok:
        return "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED"
    return {"SUBSTANTIAL": "P2_RJ_E_DIRECTION_ISSUE", "WEAK": "P2_RJ_E_DIRECTION_AND_SITE_ISSUE"}.get(
        lev_bf16, "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED")


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
                rows[mode][uid] = {"status": "missing"}; continue
            row = json.loads(p.read_text())
            if row["manifest_hash"] != m["manifest_hash"] or row["identity"] != uid:
                raise ValueError(f"stale row {p}")
            if row.get("status") == "ok":
                npz = run / "rows" / f"{i:03d}_{mode}.npz"
                if file_hash(npz) != row["vectors_sha256"]:
                    raise ValueError(f"vector hash mismatch {npz}")
                with np.load(npz) as z:
                    row["_vec"] = {k: z[k] for k in z.files}
            rows[mode][uid] = row
    return m, cfg, pos, rows


def p2rj_overlap_reference(cfg: dict) -> dict:
    """P2-RJ run1 bf16 stats and margin-gradient vectors at the original EN-confusion positions."""
    run = ROOT / cfg["p2rj_reference"]["run"]
    man = json.loads((run / "manifest.json").read_text())
    if man["manifest_hash"] != cfg["p2rj_reference"]["run_manifest_hash"]:
        raise ValueError("P2-RJ run1 manifest differs from the frozen reference")
    pos = json.loads((ROOT / man["positions"]).read_text())
    out = {}
    for i, uid in enumerate(pos["utterances"]):
        row = json.loads((run / "rows" / f"{i:03d}_bf16.json").read_text())
        with np.load(run / "rows" / f"{i:03d}_bf16.npz") as z:
            for t, rec in row["positions"].items():
                if rec["stratum"] == "EN-confusion":
                    out[(uid, int(t))] = {"A": rec["stats"]["A"], "m0": rec["values"]["margin"],
                                          "g_tan_norm": rec["stats"]["targets"]["margin"]["g_tan_norm"],
                                          "g": z[f"t{t}_g_margin"].astype(np.float64)}
    return out


def checks_for(p: dict, rec: dict, row: dict, mode: str, overlap: dict, vec: dict) -> dict:
    idt = rec["identity"]
    if mode == "fp32":
        return {"logits_bitwise": bool(idt["logits_bitwise_equal"])}
    c = {"argmax": idt["argmax"] == p["baseline_token"],
         "logits_bitwise": bool(idt["logits_bitwise_equal"]),
         "site_equals_recorder": bool(idt["site_equals_recorder"]),
         "competitor": idt["competitor_recomputed"] == rec_comp(row, p) or idt["logp_competitor"] == idt["max_other_logp"],
         "margin_paths": abs(rec["values"]["margin"] - idt["margin_best_other"]) <= 1e-3}
    if p["original"]:
        c["competitor_rule_equals_frozen"] = bool(row["competitor_check"][str(p["t"])]["rule_equals_frozen"])
        pp = dict(p)
        c.update({f"p2r_{k}": v for k, v in rja.state_checks(pp, rec).items()})
        ref = overlap[(p["utterance_id"], p["t"])]
        st = rec["stats"]
        g = vec[f"t{p['t']}_g_margin"].astype(np.float64)
        c["overlap_A"] = abs(st["A"] - ref["A"]) <= 1e-6 * ref["A"]
        c["overlap_gap"] = abs(rec["values"]["margin"] - ref["m0"]) <= 1e-6 * max(1.0, abs(ref["m0"]))
        c["overlap_g_tan_norm"] = abs(st["targets"]["margin"]["g_tan_norm"] - ref["g_tan_norm"]) <= 1e-6 * ref["g_tan_norm"]
        c["overlap_g_cos"] = float(g @ ref["g"] / (np.linalg.norm(g) * np.linalg.norm(ref["g"]))) >= 0.9999
    return c


def rec_comp(row: dict, p: dict) -> int:
    ch = row["competitor_check"][str(p["t"])]
    return ch["frozen"] if ch["frozen"] is not None else ch["competitor"]


def records(pos: dict, rows: dict, mode: str, overlap: dict) -> list[dict]:
    out = []
    for p in pos["positions"]:
        row = rows[mode].get(p["utterance_id"], {})
        rec = row.get("positions", {}).get(str(p["t"])) if row.get("status") == "ok" else None
        r = {"stratum": "EN-confusion", "dialogue_id": p["dialogue_id"], "utterance_id": p["utterance_id"],
             "t": p["t"], "original": p["original"], "span_initial": p.get("span_initial"), "present": rec is not None}
        if rec is None:
            out.append(r); continue
        r["checks"] = checks_for(p, rec, row, mode, overlap, row["_vec"])
        r["state_ok"] = all(r["checks"].values())
        r["finite"] = rja.finite_nonzero(rec)
        st = rec["stats"]
        r["m0"] = rec["values"]["margin"]
        r["gap"] = -r["m0"]
        r["A"] = st["A"]
        r["rho"] = math.inf if r["gap"] <= 0 else st["A"] / r["gap"]
        r["g_norm"] = st["targets"]["margin"]["g_norm"]
        r["g_tan_norm"] = st["targets"]["margin"]["g_tan_norm"]
        for a in ARMS:
            x = st["arms"].get(a, {})
            r[f"kappa_{a}"] = x.get("kappa")
            r[f"cos_{a}"] = x.get("cos_gtan_v_margin")
        out.append(r)
    return out


def describe(recs: list[dict], reps: int, seed: int) -> dict:
    q = rja.quantiles
    d = {"n": len(recs), "dialogues": len({r["dialogue_id"] for r in recs}),
         "gap": q([r["gap"] for r in recs]), "g_norm": q([r["g_norm"] for r in recs]),
         "g_tan_norm": q([r["g_tan_norm"] for r in recs]), "A": q([r["A"] for r in recs]), "rho": q([r["rho"] for r in recs])}
    for th in (0.1, 0.5, 1.0):
        d[f"frac_rho_ge_{th}"] = {"positions": float(np.mean([r["rho"] >= th for r in recs])) if recs else None,
                                  "dialogue_weighted": rja.boot(recs, None, reps, seed, 0.05,
                                                                fn=lambda r, th=th: 1.0 if r["rho"] >= th else 0.0)}
    for a in ARMS:
        d[f"kappa_{a}"] = rja.boot(recs, f"kappa_{a}", reps, seed, 0.05)
        d[f"cos_{a}"] = rja.boot(recs, f"cos_{a}", reps, seed, 0.05)
        d[f"abs_cos_{a}_median"] = (q([abs(r[f"cos_{a}"]) for r in recs if r.get(f"cos_{a}") is not None]) or {}).get("median")
    return d


def analyze(run: Path) -> dict:
    m, cfg, pos, rows = load(run)
    reps, seed = cfg["analysis"]["bootstrap"]["replicates"], cfg["analysis"]["bootstrap"]["seed"]
    overlap = p2rj_overlap_reference(cfg)
    runtime = json.loads((run / "runtime.json").read_text()) if (run / "runtime.json").exists() else {}
    rec = {mode: records(pos, rows, mode, overlap) for mode in PASSES}
    bf = rec["bf16"]
    mismatch = [(r["utterance_id"], r["t"], [k for k, v in r["checks"].items() if not v]) for r in bf if r["present"] and not r["state_ok"]]
    nonfinite = [(r["utterance_id"], r["t"]) for r in bf if r["present"] and not r["finite"]]
    valid_bf = [r for r in bf if r["present"] and r["state_ok"] and r["finite"]]
    keys = {(r["utterance_id"], r["t"]) for r in valid_bf}
    valid_fp = [r for r in rec["fp32"] if r["present"] and r["finite"] and r["checks"]["logits_bitwise"]
                and (r["utterance_id"], r["t"]) in keys]
    n = len(pos["positions"])
    mn = cfg["population"]["minimum_interpretable"]
    validity = {
        "V2_completeness": all(rows[md][u].get("status") == "ok" for md in PASSES for u in pos["utterances"])
        and sum(r["present"] for r in bf) == n and sum(r["present"] for r in rec["fp32"]) == n,
        "V3_state_identity": len(mismatch) <= math.floor(MISMATCH_FRACTION * n),
        "V4_finite": not nonfinite,
        "V5_min_population": len(valid_bf) >= mn["positions"] and len({r["dialogue_id"] for r in valid_bf}) >= mn["dialogues"],
        "runtime_completed": runtime.get("status") == "completed"}
    orig = [r for r in bf if r["original"] and r["present"]]
    overlap_ok = all(all(v for k, v in r["checks"].items() if k.startswith("overlap_")) for r in orig)
    lev = {"bf16": rja.leverage_alignment(valid_bf, reps, seed, ALPHA_PER_BOUND),
           "fp32": rja.leverage_alignment(valid_fp, reps, seed, ALPHA_PER_BOUND)}
    pc = rja.precision_cos(rows, pos, keys)
    precision_ok = pc["cos"] is not None and pc["cos"]["median"] >= 0.9
    valid = all(validity.values())
    label = final_label(valid, lev["bf16"]["leverage"], lev["fp32"]["leverage"], precision_ok)
    res = {"schema": "p2rje_analysis_v1", "manifest_hash": m["manifest_hash"], "runtime": runtime,
           "validity": validity, "overlap_reproduction_all_original": overlap_ok,
           "counts": {"positions": n, "original": sum(p["original"] for p in pos["positions"]),
                      "new": sum(not p["original"] for p in pos["positions"]), "valid_bf16": len(valid_bf),
                      "valid_fp32": len(valid_fp), "dialogues": len({r["dialogue_id"] for r in valid_bf}),
                      "state_mismatch": mismatch, "nonfinite": nonfinite,
                      "competitor_ties": sum(1 for u in pos["utterances"] if rows["bf16"][u].get("status") == "ok"
                                             for c in rows["bf16"][u]["competitor_check"].values() if c["tied_non_reference"] > 0)},
           "Q50": {md: lev[md]["Q50"] for md in PASSES},
           "leverage": {md: lev[md]["leverage"] for md in PASSES},
           "K_plus_descriptive": {md: lev[md]["K_plus"] for md in PASSES},
           "precision": {k: v for k, v in pc.items() if k != "per_position"} | {"ok": precision_ok},
           "diagnosis": label,
           "descriptive": {"all": describe(valid_bf, reps, seed),
                           "original_60": describe([r for r in valid_bf if r["original"]], reps, seed),
                           "new": describe([r for r in valid_bf if not r["original"]], reps, seed),
                           "fp32_all": describe(valid_fp, 2000, seed)},
           "Q50_original_subset_this_run": rja.leverage_alignment([r for r in valid_bf if r["original"]], reps, seed, ALPHA_PER_BOUND)["Q50"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = analyze(ROOT / args.run)
    atomic_json(ROOT / args.out, res)
    print(json.dumps({k: res[k] for k in ("diagnosis", "validity", "overlap_reproduction_all_original", "Q50", "leverage", "precision")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
