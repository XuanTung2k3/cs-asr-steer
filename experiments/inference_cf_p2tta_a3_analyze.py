#!/usr/bin/env python
"""P2-TTA-A3 primary analysis (frozen contract configs/inference_cf/p2_tta_a3.json). Mechanistic diagnostics are
reference-free; canonical ASR metrics open references only behind the committed output seal. Frozen precedence:
INVALID -> GAP_NOT_CLOSED -> SEQUENCE_LEVERAGE_LIMIT -> TEACHER_UNSAFE -> CONDITIONING_ONLY -> PROMISING
(PROMISING additionally requires the independent post-audit PASS, which the auditor confirms)."""
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

CONFIG = "configs/inference_cf/p2_tta_a3.json"
BASE = "results/inference_cf/p2tta_a3"
SEAL = f"{BASE}/output_seal.json"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- frozen predicates (pure) ----------------------------------------------------------------------------------

def gap_closed(d0: list[float], d2: list[float], c: dict) -> dict:
    M = c["mechanism"]
    n = len(d0)
    rel = [(a - b) / max(a, M["relative_eps"]) for a, b in zip(d0, d2)]
    dec = sum(b < a for a, b in zip(d0, d2))
    need = math.ceil(M["gap_rows_decreased_fraction_min"] * n)
    med = float(np.median(rel)) if rel else None
    return {"n_D": n, "rows_decreased": dec, "rows_decreased_required": need, "median_relative_reduction": med, "relative": rel,
            "pass": bool(n > 0 and dec >= need and med is not None and med >= M["gap_median_relative_reduction_min"] - 1e-12)}


def movement(d_BA: list[int], d_XA: list[int], c: dict) -> dict:
    M = c["mechanism"]
    closer = sum(x < b for b, x in zip(d_BA, d_XA))
    farther = sum(x > b for b, x in zip(d_BA, d_XA))
    R = 1 - sum(d_XA) / sum(d_BA) if sum(d_BA) else None
    ok = (closer >= M["movement_closer_min"] and farther <= M["movement_farther_max"]) or \
        (R is not None and R >= M["movement_pooled_distance_reduction_min"] - 1e-12)
    return {"closer": closer, "same": len(d_BA) - closer - farther, "farther": farther, "R_dist": R, "pass": bool(ok)}


def safety(p: dict, c: dict) -> dict:
    S = c["safety"]
    G = S["float_guard"]
    le = lambda v, t: v is None or v <= t + G
    ge = lambda v, t: v is None or v >= t - G
    return {"MER": p["mer_increase"] <= S["max_MER_increase"] + G, "ZH_CER": p["zh_cer_increase"] <= S["max_ZH_CER_increase"] + G,
            "matrix_ZH_retention": ge(p["zh_retention"], S["min_matrix_ZH_retention"]),
            "embedded_EN_retention": ge(p["en_retention"], S["min_embedded_EN_retention"]),
            "outside_harm": le(p["outside_harm_rate"], S["max_outside_POI_harm_rate"]),
            "POI_corruption": le(p["poi_corruption_rate"], S["max_POI_corruption_rate"]),
            "caps": p["added_caps"] <= S["max_additional_caps"], "severe_truncation": p["new_severe_truncations"] <= S["max_new_severe_truncations"]}


def vs_a2(poi_a3: int, poi_a2: int, mer_a3: float, mer_a2: float, c: dict) -> dict:
    tol = c["A2_comparison"]["max_MER_worse_than_A2"]
    prom = poi_a3 <= poi_a2 and mer_a3 - mer_a2 <= tol + 1e-12
    return {"poi_diff": poi_a3 - poi_a2, "mer_diff": mer_a3 - mer_a2, "promising_condition": bool(prom),
            "no_advantage": bool(poi_a3 > poi_a2 or (poi_a3 == poi_a2 and mer_a3 - mer_a2 > tol))}


def decide(valid: bool, gap: dict, mov: dict, safe: dict | None, a2: dict | None) -> str:
    if not valid:
        return "P2_TTA_A3_INVALID"
    if not gap["pass"]:
        return "P2_TTA_A3_GAP_NOT_CLOSED"
    if not mov["pass"]:
        return "P2_TTA_A3_SEQUENCE_LEVERAGE_LIMIT"
    if not all(safe.values()):
        return "P2_TTA_A3_TEACHER_UNSAFE"
    if not a2["promising_condition"]:
        return "P2_TTA_A3_CONDITIONING_ONLY"
    return "P2_TTA_A3_PROMISING"


# ---- load / integrity / mechanics (reference-free) ----------------------------------------------------------------

def load(run_dir: Path) -> dict:
    man = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows = []
    for i, u in enumerate(man["ids"]):
        p = run_dir / f"rows/{i:02d}.json"
        r = json.loads(p.read_text()) if p.exists() else {"status": "missing", "identity": u}
        if r.get("status") == "ok" and ("sha256:" + sha(run_dir / f"rows/{i:02d}_final_masters_fp32.npz") != r["final_masters_npz_sha256"]
                                        or r["identity"] != u):
            raise ValueError(f"row {i}")
        rows.append(r)
    return {"manifest": man, "plan": plan, "rows": rows, "runtime": json.loads((run_dir / "runtime.json").read_text())}


def integrity(d: dict) -> dict:
    rows, rt, plan = d["rows"], d["runtime"], d["plan"]
    c = {"all_24_ok": len(rows) == 24 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
         "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
         and bool(rt.get("model_grads_none"))}
    if not c["all_24_ok"]:
        return c
    a3 = [r["A3"] for r in rows]
    c["theta0_forced_equals_S0"] = all(r["theta0_forced"]["tokens_equal_S0"] and r["theta0_forced"]["terminated_equal_S0"] for r in rows)
    c["auto_replay_equals_historical"] = all(r["auto_condition"]["replay_text_equal"] for r in rows)
    la = rows[0].get("live_audit")
    c["live_audit"] = bool(la) and la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    c["two_steps_three_losses"] = all(o["steps"] == 2 and o["loss_evaluations"] == 3 and len(o["grad_l2"]) == 2 for o in a3)
    c["finite"] = all(all(o["finite"]) and all(math.isfinite(x) for x in o["losses"] + o["d_cond"] + o["grad_l2"]) for o in a3)
    c["resets"] = all(o["reset_ok"] and o["end_hash"] == rt["theta0_ln_hash"] and o["other_versions_unchanged"] for o in a3)
    c["noop_identity"] = all(o["equal_B0_FORCED"] and o["master_delta_l2"] == 0 for o in a3 if o["noop"])
    c["loss_equals_d_cond"] = all(o["losses"] == o["d_cond"] for o in a3 if not o["noop"])
    cnt = rt["counters"]
    c["compute_counts"] = (cnt["A3"].get("optimizer_steps") == 48 and cnt["A3"].get("backwards", 0) + cnt["audit"]["backwards"] <= 49
                           and cnt["A3"].get("backwards") == 48 and cnt["final_decodes"] == 24 and cnt["encoder_passes"] == 24)
    c["groups_match_panel"] = [r["group"] for r in rows] == [p["group"] for p in plan["rows"]]
    return c


def mechanics(d: dict) -> dict:
    c = cfg()
    rows = d["rows"]
    D = [r for r in rows if r["group"] == "D"]
    A = [r for r in rows if r["group"] == "A"]

    def tw(rs, k):                          # token-weighted D_cond over valid positions
        n = sum(r["A3"]["valid"] for r in rs)
        return sum(r["A3"]["d_cond"][k] * r["A3"]["valid"] for r in rs) / n if n else None
    gap = gap_closed([r["A3"]["d_cond"][0] for r in D], [r["A3"]["d_cond"][2] for r in D], c)
    mov = movement([r["A3"]["d_BA"] for r in D], [r["A3"]["d_A3A"] for r in D], c)
    mov_a2 = movement([r["A3"]["d_BA"] for r in D], [r["A3"]["d_A2A"] for r in D], c)
    ent = lambda rs, th: (lambda v: float(np.mean(v)) if v else None)([x for r in rs for x, ok in zip(r["A3"]["diag"][th]["entropy"],
                                                                                                     _mask(d, r)) if ok])
    grp = lambda rs: {"n": len(rs), "D_cond_tw": [tw(rs, k) for k in range(3)], "D_cond_mean": [float(np.mean([r["A3"]["d_cond"][k] for r in rs])) for k in range(3)],
                      "identical_condition": sum(r["A3"]["identical_condition"] for r in rs),
                      "langs": sorted({r["auto_condition"]["lang_token"] for r in rs}),
                      "changed_vs_B0": sum(not r["A3"]["equal_B0_FORCED"] for r in rs),
                      "entropy_mean": {"theta0": ent(rs, "theta0"), "theta2": ent(rs, "theta2")},
                      "master_delta_l2_mean": float(np.mean([r["A3"]["master_delta_l2"] for r in rs])),
                      "grad_l2_mean": [float(np.mean([r["A3"]["grad_l2"][j] for r in rs])) for j in range(2)],
                      "length_mean": float(np.mean([r["A3"]["length"] for r in rs]))}
    return {"gap": gap, "movement_A3": mov, "movement_A2": mov_a2, "D": grp(D), "A": grp(A),
            "distance_D": {"B0": sum(r["A3"]["d_BA"] for r in D), "A2": sum(r["A3"]["d_A2A"] for r in D), "A3": sum(r["A3"]["d_A3A"] for r in D)},
            "per_row": [{"utterance_id": r["identity"], "group": r["group"], "lang": r["auto_condition"]["lang_token"],
                         "d_cond": r["A3"]["d_cond"], "rel": r["A3"]["relative_gap_reduction"], "d_BA": r["A3"]["d_BA"], "d_A2A": r["A3"]["d_A2A"],
                         "d_A3A": r["A3"]["d_A3A"], "movement": r["A3"]["movement"], "changed": not r["A3"]["equal_B0_FORCED"],
                         "length": r["A3"]["length"], "terminated": r["A3"]["terminated"]} for r in rows]}


def _mask(d, r):
    return next(p["y_A_valid_mask"] for p in d["plan"]["rows"] if p["utterance_id"] == r["identity"])


# ---- evaluation (behind the committed output seal) -----------------------------------------------------------------

def evaluate(run_rel: str) -> dict:
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import draws, outside, ratio_delta, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("A3 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed A3 outputs changed")
    d = load(ROOT / run_rel)
    integ = integrity(d)
    mech = mechanics(d) if integ["all_24_ok"] else None
    from experiments.inference_cf_p2_evaluate import load_references
    plan, rows = d["plan"], d["rows"]
    ids = plan["ids"]
    P = {p["utterance_id"]: p for p in plan["rows"]}
    refs = load_references()
    R = [refs[u]["reference"] for u in ids]
    Dg = [refs[u]["dialogue_id"] for u in ids]
    integ["dialogues"] = Dg == [P[u]["dialogue_id"] for u in ids]
    H = {"B0_FORCED": [P[u]["y_B_text"] for u in ids], "B0_AUTO": [P[u]["y_A_text"] for u in ids], "A2": [P[u]["A2"]["text"] for u in ids]}
    term = {"B0_FORCED": [P[u]["y_B_terminated"] for u in ids], "B0_AUTO": [P[u]["y_A_terminated"] for u in ids],
            "A2": [P[u]["A2"]["terminated"] for u in ids]}
    if integ["all_24_ok"]:
        H["A3"] = [r["A3"]["text"] for r in rows]
        term["A3"] = [r["A3"]["terminated"] for r in rows]
    M = {k: {**corpus_metrics(R, h), "caps": sum(t == "cap" for t in term[k])} for k, h in H.items()}
    integ["populations_nonempty"] = all(M["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(M[k][x] is not None and math.isfinite(M[k][x]) for k in M for x in ("pier", "mer", "en_wer", "zh_cer"))
    valid = all(integ.values())
    res = {"schema": "p2_tta_a3_evaluation_v1", "manifest_hash": d["manifest"]["manifest_hash"], "output_seal_hash": seal["seal_hash"],
           "integrity": integ, "valid": valid, "metrics": M, "mechanics": mech}
    if "A3" not in H or mech is None:
        res["label"] = "P2_TTA_A3_INVALID"
        return res
    def comp(X):
        cc = correction_corruption(R, H["B0_FORCED"], H[X])
        zr, er = matrix_zh_retention(R, H["B0_FORCED"], H[X]), embedded_en_retention(R, H["B0_FORCED"], H[X])
        outs = [outside(r, b, s) for r, b, s in zip(R, H["B0_FORCED"], H[X])]
        osum = {x: sum(o[x] for o in outs) for x in outs[0]}
        ohr = osum["outside_harm"] / osum["baseline_correct_outside"] if osum["baseline_correct_outside"] else None
        return {"mer_increase": M[X]["mer"] - M["B0_FORCED"]["mer"], "zh_cer_increase": M[X]["zh_cer"] - M["B0_FORCED"]["zh_cer"],
                "en_wer_increase": M[X]["en_wer"] - M["B0_FORCED"]["en_wer"], "pier_gain": M["B0_FORCED"]["pier"] - M[X]["pier"],
                "net_poi_error_reduction": M["B0_FORCED"]["num_poi_errors"] - M[X]["num_poi_errors"], "corrections": cc["corrections"],
                "corruptions": cc["corruptions"], "zh_retention": zr["rate"], "en_retention": er["rate"], "poi_corruption_rate": cc["corruption_rate"],
                "outside_harm_rate": ohr, "outside": osum,
                "added_caps": sum(a == "cap" and b != "cap" for a, b in zip(term[X], term["B0_FORCED"]))}
    pA3, pA2 = comp("A3"), comp("A2")
    pA3["new_severe_truncations"] = sum(r["A3"]["severe_truncation"] for r in rows)
    pA2["new_severe_truncations"] = sum(len(P[u]["y_B"]) >= 10 and P[u]["A2"]["terminated"] == "eos"
                                        and P[u]["A2"]["length"] <= math.floor(0.5 * len(P[u]["y_B"])) for u in ids)
    safe = safety(pA3, c)
    a2 = vs_a2(M["A3"]["num_poi_errors"], M["A2"]["num_poi_errors"], M["A3"]["mer"], M["A2"]["mer"], c)
    label = decide(valid, mech["gap"], mech["movement_A3"], safe, a2)
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in H}
    idxs = draws(Dg)
    nm = {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}
    boot = {f"A3_minus_{b}": {n: ratio_delta(C["A3"], C[b], x, y, idxs) for n, (x, y) in nm.items()} for b in ("B0_FORCED", "A2", "B0_AUTO")}
    grp_counts = {g: {k: {"poi_errors": int(sum(C[k][j][0] for j, u in enumerate(ids) if P[u]["group"] == g)),
                          "mixed_errors": int(sum(C[k][j][2] for j, u in enumerate(ids) if P[u]["group"] == g))} for k in H} for g in ("D", "A")}
    res.update(points={"A3": pA3, "A2": pA2}, safety=safe, vs_A2=a2, label=label, bootstrap_descriptive=boot, group_counts=grp_counts,
               per_utterance=[{"utterance_id": u, "group": P[u]["group"], **{k: H[k][j] for k in H}, "counts": {k: C[k][j].tolist() for k in C}}
                              for j, u in enumerate(ids)])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable(evaluate(args.run))
    atomic_json(out, res)
    m = res.get("mechanics") or {}
    print(json.dumps({"label": res["label"], "integrity": res["integrity"], "gap": {k: v for k, v in m.get("gap", {}).items() if k != "relative"},
                      "movement_A3": m.get("movement_A3"), "movement_A2": m.get("movement_A2"), "distance_D": m.get("distance_D"),
                      "D": m.get("D"), "A": m.get("A"), "safety": res.get("safety"), "vs_A2": res.get("vs_A2"),
                      "metrics": {k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
