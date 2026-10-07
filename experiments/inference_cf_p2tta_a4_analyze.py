#!/usr/bin/env python
"""P2-TTA-A4 primary analysis (frozen contract configs/inference_cf/p2_tta_a4.json). Mechanistic diagnostics are
reference-free; canonical ASR metrics open references only behind the committed output seal. Frozen precedence:
INVALID -> ENGLISH_TRANSFER_WEAK -> SHARED_PARAMETER_INTERFERENCE -> SEQUENCE_SAFETY_NOT_RESCUED -> SAFE_BUT_NO_GAIN
-> PROMISING (PROMISING additionally requires the independent post-audit PASS)."""
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

CONFIG = "configs/inference_cf/p2_tta_a4.json"
BASE = "results/inference_cf/p2tta_a4"
SEAL = f"{BASE}/output_seal.json"
SYSTEMS = ("B0_FORCED", "B0_AUTO", "A2", "A3", "A4")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- frozen predicates (pure) -------------------------------------------------------------------------------------

def en_transfer(rows: list[dict], c: dict) -> dict:
    """rows: eligible D rows (n_E > 0) with d_E [3] and R_E."""
    n = len(rows)
    dec = sum(r["d_E"][2] < r["d_E"][0] for r in rows)
    need = math.ceil(0.75 * n)
    if n == c["token_counts_pre_outcome"]["eligible_D_rows"] and need != c["mechanism"]["en_rows_decreased_min"]:
        raise ValueError("frozen EN row requirement mismatch")
    med = float(np.median([r["R_E"] for r in rows])) if rows else None
    return {"n_eligible": n, "rows_decreased": dec, "required": need, "median_R_E": med,
            "pass": bool(n > 0 and dec >= need and med is not None and med >= c["mechanism"]["en_median_R_E_min"] - 1e-12)}


def anchor(rows: list[dict], c: dict) -> dict:
    eps = c["mechanism"]["eps"]
    nE = sum(r["n_E"] for r in rows)
    nNE = sum(r["n_nonE"] for r in rows if r["d_anchor"][2] is not None)
    G = sum(r["n_E"] * (r["d_E"][0] - r["d_E"][2]) for r in rows) / nE if nE else None
    DA2 = sum(r["n_nonE"] * r["d_anchor"][2] for r in rows if r["d_anchor"][2] is not None) / nNE if nNE else 0.0
    DA0 = sum(r["n_nonE"] * r["d_anchor"][0] for r in rows if r["d_anchor"][0] is not None) / nNE if nNE else 0.0
    nM = sum(r["n_M"] for r in rows if r["d_M"][0] is not None)
    DM = [sum(r["n_M"] * r["d_M"][k] for r in rows if r["d_M"][k] is not None) / nM if nM else None for k in range(3)]
    rho = DA2 / max(G, eps) if G is not None else None
    rowwise = [r["d_anchor"][2] / max(r["d_E"][0] - r["d_E"][2], eps) for r in rows if r["d_anchor"][2] is not None]
    ok = rho is not None and math.isfinite(rho) and math.isfinite(DA2) and rho <= c["mechanism"]["rho_anchor_max"] + 1e-12
    return {"G_E_tw": G, "D_E_tw": [sum(r["n_E"] * r["d_E"][k] for r in rows) / nE if nE else None for k in range(3)],
            "D_ANCHOR0_tw": DA0, "D_ANCHOR2_tw": DA2, "D_M_tw": DM, "rho_anchor": rho,
            "rho_rowwise_median": float(np.median(rowwise)) if rowwise else None, "pass": bool(ok)}


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


def partial_rescue(p: dict, c: dict) -> dict:
    R = c["partial_rescue"]
    a, S, w = R["A3_reference"], c["safety"], R["materially_worse_abs"]
    frac = {"zh_cer": (a["zh_cer_increase"] - p["zh_cer_increase"]) / (a["zh_cer_increase"] - S["max_ZH_CER_increase"]),
            "retention": (p["zh_retention"] - a["zh_retention"]) / (S["min_matrix_ZH_retention"] - a["zh_retention"]),
            "outside": (a["outside_harm_rate"] - p["outside_harm_rate"]) / (a["outside_harm_rate"] - S["max_outside_POI_harm_rate"])}
    worse = {"zh_cer": p["zh_cer_increase"] - a["zh_cer_increase"] > w, "retention": a["zh_retention"] - p["zh_retention"] > w,
             "outside": p["outside_harm_rate"] - a["outside_harm_rate"] > w}
    half = {k: v >= R["half"] - 1e-12 for k, v in frac.items()}
    return {"fraction_closed": frac, "half_rescued": half, "materially_worse_than_A3": worse,
            "partial": bool(sum(half.values()) >= R["min_measures"] and not any(worse.values()))}


def benefit(poi_a4: int, poi_a2: int, mer_a4: float, mer_a2: float, severe_a4: int, c: dict) -> dict:
    B = c["benefit_vs_A2"]
    return {"poi_diff": poi_a4 - poi_a2, "mer_diff": mer_a4 - mer_a2,
            "pass": bool(poi_a4 <= poi_a2 + B["poi_slack"] and mer_a4 - mer_a2 <= B["mer_max_worse"] + 1e-12 and severe_a4 == 0)}


def decide(valid: bool, en: dict, anc: dict, full_rescue: bool | None, ben: dict | None) -> str:
    if not valid:
        return "P2_TTA_A4_INVALID"
    if not en["pass"]:
        return "P2_TTA_A4_ENGLISH_TRANSFER_WEAK"
    if not anc["pass"]:
        return "P2_TTA_A4_SHARED_PARAMETER_INTERFERENCE"
    if not full_rescue:
        return "P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED"
    if not ben["pass"]:
        return "P2_TTA_A4_SAFE_BUT_NO_GAIN"
    return "P2_TTA_A4_PROMISING"


# ---- load / integrity / mechanics (reference-free) ---------------------------------------------------------------

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
    a4 = [r["A4"] for r in rows]
    tol = cfg()["mechanism"]["theta0_anchor_abs_tol"]
    c["theta0_forced_equals_S0"] = all(r["theta0_forced"]["tokens_equal_S0"] and r["theta0_forced"]["terminated_equal_S0"] for r in rows)
    c["lang_equals_A3"] = all(r["auto_condition"]["lang_equals_A3_sealed"] for r in rows)
    la = rows[0].get("live_audit")
    c["live_audit"] = bool(la) and la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    c["two_steps_three_losses"] = all(o["steps"] == 2 and o["loss_evaluations"] == 3 and len(o["grad_l2"]) == 2 for o in a4)
    c["finite"] = all(all(o["finite"]) and all(math.isfinite(x) for x in o["losses"] + o["grad_l2"]) for o in a4)
    c["resets"] = all(o["reset_ok"] and o["end_hash"] == rt["theta0_ln_hash"] and o["other_versions_unchanged"] for o in a4)
    c["noop_identity"] = all(o["equal_B0_FORCED"] and o["master_delta_l2"] == 0 for o in a4 if o["noop"])
    c["theta0_anchor_zero"] = all(o["d_anchor"][0] is None or abs(o["d_anchor"][0]) <= tol for o in a4)
    cnt = rt["counters"]
    c["compute_counts"] = (cnt["A4"].get("optimizer_steps") == 48 and cnt["A4"].get("backwards") == 48 and cnt["audit"]["backwards"] == 1
                           and cnt["final_decodes"] == 24 and cnt["encoder_passes"] == 24)
    c["groups_match_plan"] = [r["group"] for r in rows] == [p["group"] for p in plan["rows"]]
    c["class_counts_match_plan"] = all((o["n_E"], o["n_M"], o["n_O"]) == (p["counts_valid"]["E"], p["counts_valid"]["M"], p["counts_valid"]["O"])
                                       for o, p in zip(a4, plan["rows"]))
    return c


def mechanics(d: dict) -> dict:
    c = cfg()
    rows = d["rows"]
    D = [r for r in rows if r["group"] == "D"]
    elig = [r["A4"] for r in D if r["A4"]["n_E"] > 0]
    en, anc = en_transfer(elig, c), anchor(elig, c)
    P = {p["utterance_id"]: p for p in d["plan"]["rows"]}
    def grp(rs):
        return {"n": len(rs), "noop": sum(r["A4"]["noop"] for r in rs), "langs": sorted({r["auto_condition"]["lang_token"] for r in rs}),
                "n_E": sum(r["A4"]["n_E"] for r in rs), "n_M": sum(r["A4"]["n_M"] for r in rs), "n_O": sum(r["A4"]["n_O"] for r in rs),
                "changed_vs_B0": sum(not r["A4"]["equal_B0_FORCED"] for r in rs), "changed_vs_A2": sum(not r["A4"]["equal_A2"] for r in rs),
                "changed_vs_A3": sum(not r["A4"]["equal_A3"] for r in rs),
                "master_delta_l2_mean": float(np.mean([r["A4"]["master_delta_l2"] for r in rs])),
                "length_mean": float(np.mean([r["A4"]["length"] for r in rs])),
                "eos": sum(r["A4"]["terminated"] == "eos" for r in rs), "caps": sum(r["A4"]["terminated"] == "cap" for r in rs),
                "severe": sum(r["A4"]["severe_truncation"] for r in rs)}
    dist = {"B0": sum(r["A4"]["d_BA"] for r in D), "A2": sum(P[r["identity"]]["d_A2A"] for r in D),
            "A3": sum(_lev(P[r["identity"]]["A3"]["tokens"], P[r["identity"]]["y_A"]) for r in D), "A4": sum(r["A4"]["d_A4_AUTO"] for r in D)}
    mv = {k: sum(r["A4"]["movement"] == k for r in D) for k in ("CLOSER", "SAME", "FARTHER")}
    return {"en_transfer": en, "anchor": anc, "D": grp(D), "A": grp([r for r in rows if r["group"] == "A"]), "distance_to_AUTO_D": dist,
            "movement_D": mv,
            "per_row": [{"utterance_id": r["identity"], "group": r["group"], "lang": r["auto_condition"]["lang_token"], "noop": r["A4"]["noop"],
                         "n": [r["A4"]["n_E"], r["A4"]["n_M"], r["A4"]["n_O"]], "d_E": r["A4"]["d_E"], "d_M": r["A4"]["d_M"],
                         "d_anchor": r["A4"]["d_anchor"], "R_E": r["A4"]["R_E"], "loss": r["A4"]["losses"], "grad_l2": r["A4"]["grad_l2"],
                         "master_delta_l2": r["A4"]["master_delta_l2"], "d_AUTO": r["A4"]["d_A4_AUTO"], "d_B0": r["A4"]["d_A4_B0"],
                         "d_A2": r["A4"]["d_A4_A2"], "d_A3": r["A4"]["d_A4_A3"], "movement": r["A4"]["movement"], "length": r["A4"]["length"],
                         "terminated": r["A4"]["terminated"]} for r in rows]}


def _lev(a, b):
    from csasr.inference_cf.episodic_tta import levenshtein
    return levenshtein(a, b)


# ---- evaluation (behind the committed output seal) -----------------------------------------------------------------

def evaluate(run_rel: str) -> dict:
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import draws, outside, ratio_delta, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("A4 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed A4 outputs changed")
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
    H = {"B0_FORCED": [P[u]["y_B_text"] for u in ids], "B0_AUTO": [P[u]["y_A_text"] for u in ids], "A2": [P[u]["A2"]["text"] for u in ids],
         "A3": [P[u]["A3"]["text"] for u in ids]}
    term = {"B0_FORCED": [P[u]["y_B_terminated"] for u in ids], "B0_AUTO": [P[u]["y_A_terminated"] for u in ids],
            "A2": [P[u]["A2"]["terminated"] for u in ids], "A3": [P[u]["A3"]["terminated"] for u in ids]}
    lens = {"A2": [P[u]["A2"]["length"] for u in ids], "A3": [P[u]["A3"]["length"] for u in ids]}
    if integ["all_24_ok"]:
        H["A4"] = [r["A4"]["text"] for r in rows]
        term["A4"] = [r["A4"]["terminated"] for r in rows]
        lens["A4"] = [r["A4"]["length"] for r in rows]
    M = {k: {**corpus_metrics(R, h), "caps": sum(t == "cap" for t in term[k])} for k, h in H.items()}
    integ["populations_nonempty"] = all(M["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(M[k][x] is not None and math.isfinite(M[k][x]) for k in M for x in ("pier", "mer", "en_wer", "zh_cer"))
    valid = all(integ.values())
    res = {"schema": "p2_tta_a4_evaluation_v1", "manifest_hash": d["manifest"]["manifest_hash"], "output_seal_hash": seal["seal_hash"],
           "integrity": integ, "valid": valid, "metrics": M, "mechanics": mech}
    if "A4" not in H or mech is None:
        res["label"] = "P2_TTA_A4_INVALID"
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
                "added_caps": sum(a == "cap" and b != "cap" for a, b in zip(term[X], term["B0_FORCED"])),
                "new_severe_truncations": sum(len(P[u]["y_B"]) >= 10 and t == "eos" and n <= math.floor(0.5 * len(P[u]["y_B"]))
                                              for u, t, n in zip(ids, term[X], lens[X]))}
    pts = {X: comp(X) for X in ("A2", "A3", "A4")}
    safe = safety(pts["A4"], c)
    full = all(safe.values())
    part = partial_rescue(pts["A4"], c)
    ben = benefit(M["A4"]["num_poi_errors"], M["A2"]["num_poi_errors"], M["A4"]["mer"], M["A2"]["mer"], pts["A4"]["new_severe_truncations"], c)
    label = decide(valid, mech["en_transfer"], mech["anchor"], full, ben)
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in H}
    idxs = draws(Dg)
    nm = {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}
    boot = {f"A4_minus_{b}": {n: ratio_delta(C["A4"], C[b], x, y, idxs) for n, (x, y) in nm.items()} for b in ("B0_FORCED", "A2", "A3", "B0_AUTO")}
    grp_counts = {g: {k: {"poi_errors": int(sum(C[k][j][0] for j, u in enumerate(ids) if P[u]["group"] == g)),
                          "mixed_errors": int(sum(C[k][j][2] for j, u in enumerate(ids) if P[u]["group"] == g))} for k in H} for g in ("D", "A")}
    res.update(points=pts, safety=safe, full_matrix_rescue=full, partial_matrix_rescue=part, benefit_vs_A2=ben, label=label,
               bootstrap_descriptive=boot, group_counts=grp_counts,
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
    print(json.dumps({"label": res["label"], "integrity": res["integrity"], "en_transfer": m.get("en_transfer"), "anchor": m.get("anchor"),
                      "D": m.get("D"), "A": m.get("A"), "distance_to_AUTO_D": m.get("distance_to_AUTO_D"), "movement_D": m.get("movement_D"),
                      "safety": res.get("safety"), "partial": res.get("partial_matrix_rescue"), "benefit": res.get("benefit_vs_A2"),
                      "metrics": {k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
