#!/usr/bin/env python
"""P2-TTA-FUNNEL primary evaluation (frozen contract ff75e1a). Evaluator side; references are opened only behind
each stage's committed independent gate.

r-evaluate: TTA0-R. Requires the committed PASS_TO_P2_TTA0_R_EVALUATION gate and re-verified sealed bytes, then runs
the ORIGINAL TTA0 evaluator/decision (`inference_cf_p2tta0_analyze.analyze`, thresholds = original decision section)
with only the live complete-gradient tolerance re-adjudicated at 0.02. Stage label mapped to the R vocabulary.
"""
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

from csasr.inference_cf.core import atomic_json

CONFIG = "configs/inference_cf/p2_tta_funnel.json"
R_DIR = "results/inference_cf/p2tta_funnel/tta0_r"
R_GATE = f"{R_DIR}/gate_audit.json"
R_MAP = {"P2_TTA0_INVALID": "P2_TTA0_R_STILL_INVALID", "P2_TTA0_OBJECTIVE_SELECTED": "P2_TTA0_R_OBJECTIVE_SELECTED",
         "P2_TTA0_NO_VIABLE_OBJECTIVE": "P2_TTA0_R_NO_VIABLE_OBJECTIVE"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gate_ok(rel: str, verdict: str) -> bool:
    if not (ROOT / rel).exists():
        return False
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel) and json.loads((ROOT / rel).read_text())["verdict"] == verdict


def r_evaluate() -> dict:
    c = json.loads((ROOT / CONFIG).read_text())
    if not gate_ok(R_GATE, "PASS_TO_P2_TTA0_R_EVALUATION"):
        raise PermissionError("TTA0-R gate missing/uncommitted/not PASS: references stay closed")
    if any(sha(ROOT / p) != h for p, h in c["TTA0_R"]["sealed_sha256"].items()):
        raise ValueError("sealed TTA0 bytes changed")
    import experiments.inference_cf_p2tta0_analyze as an
    if an.TH != c["original_tta0_decision"]["thresholds"]:
        raise ValueError("original TTA0 thresholds differ from master copy")
    res = an.analyze(ROOT / "results/inference_cf/p2tta0/run1", rel_grad_tol=c["TTA0_R"]["relative_gradient_tolerance"])
    seal = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta0/pseudo_sealed.json").read_text())["rows"]}
    rows = [json.loads((ROOT / f"results/inference_cf/p2tta0/run1/rows/{i:02d}.json").read_text()) for i in range(20)]
    differing = [u for u in res["per_utterance"] if seal[u["utterance_id"]]["y_B"] != seal[u["utterance_id"]]["y_A"]]
    res["auto_forced_differing_cases"] = [
        {"utterance_id": u["utterance_id"], **{k: {"changed_vs_B0": not r["objectives"][k]["vs_B0_FORCED"]["equal"],
                                                   "lev_to_AUTO": r["objectives"][k]["vs_B0_AUTO"]["levenshtein"]} for k in ("A1", "A2")},
         "lev_B0_to_AUTO": None} for u in differing for r in rows if r["identity"] == u["utterance_id"]]
    for d in res["auto_forced_differing_cases"]:
        from csasr.inference_cf.episodic_tta import levenshtein
        s = seal[d["utterance_id"]]
        d["lev_B0_to_AUTO"] = levenshtein(s["y_B"], s["y_A"])
    res["r_label"] = R_MAP[res["label"]]
    res["gate_sha256"] = sha(ROOT / R_GATE)
    res["repaired_relative_gradient_tolerance"] = c["TTA0_R"]["relative_gradient_tolerance"]
    return res


# ---- TTA1 -----------------------------------------------------------------------------------------------------

T1 = "results/inference_cf/p2tta_funnel/tta1"


def tta1_resolve(run_dir: Path) -> dict:
    """Load sealed plan/manifest/runtime/new rows and the reused ancestor episodes; no reference access."""
    plan = json.loads((ROOT / T1 / "plan_sealed.json").read_text())
    man = json.loads((run_dir / "manifest.json").read_text())
    rt = json.loads((run_dir / "runtime.json").read_text())
    from csasr.inference_cf.core import digest
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"] or man["plan_hash"] != plan["plan_hash"]:
        raise ValueError("manifest/plan")
    new = {}
    for i, u in enumerate(man["new_ids"]):
        p = run_dir / f"rows/{i:02d}.json"
        new[u] = json.loads(p.read_text()) if p.exists() else {"status": "missing"}
    out = {}
    for r in plan["rows"]:
        u = r["utterance_id"]
        if r["origin"] == "new":
            row = new[u]
            if row.get("status") == "ok" and sha(run_dir / f"rows/{man['new_ids'].index(u):02d}_final_masters_fp32.npz") != row["final_masters_npz_sha256"].split(":")[-1]:
                raise ValueError(f"masters {u}")
            out[u] = {"origin": "new", "row": row, "obj": row.get("objectives", {}).get("A2")}
        else:
            a = r["ancestor"]
            if sha(ROOT / a["row"]) != a["row_sha256"] or sha(ROOT / a["masters_npz"]) != a["masters_npz_sha256"]:
                raise ValueError(f"ancestor {u}")
            row = json.loads((ROOT / a["row"]).read_text())
            out[u] = {"origin": "reuse_TTA0_R_A2", "row": row, "obj": row["objectives"]["A2"]}
    return {"plan": plan, "manifest": man, "runtime": rt, "rows": out}


def tta1_integrity(d: dict) -> dict:
    plan, man, rt, rows = d["plan"], d["manifest"], d["runtime"], d["rows"]
    c = {"all_100_resolved": len(rows) == 100 and all(v["obj"] is not None and (v["origin"] != "new" or v["row"].get("status") == "ok")
                                                       for v in rows.values()) and list(rows) == plan["ids"],
         "runtime_completed": rt.get("status") == "completed", "no_runtime_invalid": not rt.get("invalid"),
         "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged")) and bool(rt.get("model_grads_none"))}
    if not c["all_100_resolved"]:
        return c
    newr = [v for v in rows.values() if v["origin"] == "new"]
    c["theta0_matches_S0_new"] = all(v["row"]["theta0_decode"]["tokens_equal_S0"] and v["row"]["theta0_decode"]["terminated_equal_S0"] for v in newr)
    la = rows[man["new_ids"][0]]["row"].get("live_audit", {}).get("A2")
    c["live_audit_pass"] = bool(la) and la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    objs = [v["obj"] for v in rows.values()]
    c["two_steps_three_losses"] = all(o["steps"] == 2 and o["loss_evaluations"] == 3 and len(o["grad_l2"]) == 2 for o in objs)
    c["finite"] = all(all(o["finite"]) and all(math.isfinite(x) for x in o["losses"] + o["grad_l2"]) for o in objs)
    c["reset_each_episode"] = all(o["reset_ok"] and o["start_hash"] == o["end_hash"] == rt["theta0_ln_hash"] and o["other_versions_unchanged"]
                                  for o in objs)            # reused ancestors share the identical theta0 LN bytes
    cnt = rt.get("counters", {})
    P = plan["planned"]
    c["compute_counts"] = (cnt.get("A2", {}).get("optimizer_steps") == P["optimizer_steps"]
                           and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] == P["backwards"]
                           and cnt["A2"]["teacher_forwards"] + cnt["common_theta0_forwards"] + cnt["audit"]["forwards"] == P["teacher_forwards"]
                           and cnt["A2"]["final_decodes"] == P["final_decodes"] and cnt["theta0_integrity_decodes"] == P["theta0_integrity_decodes"])
    return c


def tta1_decide(valid: bool, p: dict) -> dict:
    TH = json.loads((ROOT / CONFIG).read_text())["TTA1"]["thresholds"]
    G = TH["float_guard"]
    le = lambda v, t: v is None or v <= t + G
    ge = lambda v, t: v is None or v >= t - G
    safety = {"MER": p["mer_increase"] <= TH["max_MER_increase"] + G, "ZH_CER": p["zh_cer_increase"] <= TH["max_ZH_CER_increase"] + G,
              "matrix_ZH_retention": ge(p["zh_retention"], TH["min_matrix_ZH_retention"]),
              "embedded_EN_retention": ge(p["en_retention"], TH["min_embedded_EN_retention"]),
              "outside_harm": le(p["outside_harm_rate"], TH["max_outside_POI_harm_rate"]),
              "POI_corruption": le(p["poi_corruption_rate"], TH["max_POI_corruption_rate"]),
              "caps": p["added_caps"] <= TH["max_additional_caps"], "severe_truncation": p["new_severe_truncations"] <= TH["max_new_severe_truncations"]}
    benefit = {"net_POI": p["net_poi_error_reduction"] >= TH["min_net_POI_error_reduction"],
               "PIER": p["pier_gain"] >= TH["min_absolute_PIER_reduction"] - G}
    if not valid:
        lab = "P2_TTA1_INVALID"
    elif not all(safety.values()):
        lab = "P2_TTA1_SEQUENCE_DAMAGE"
    elif all(benefit.values()):
        lab = "P2_TTA1_SUPPORTED"
    else:
        lab = "P2_TTA1_NO_USEFUL_GAIN"
    return {"label": lab, "safety": safety, "benefit": benefit}


def tta1_evaluate(run_rel: str) -> dict:
    import numpy as np
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption, error_metric_gains
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from csasr.inference_cf.episodic_tta import levenshtein
    from experiments.inference_cf_p2seq_analyze import draws, outside, rate_ci, ratio_delta, utt_counts
    run_dir = ROOT / run_rel
    seal_rel = f"{T1}/output_seal.json"
    if not gate_ok(f"{T1}/prerun_audit.json", "PASS_TO_P2_TTA1") or not (ROOT / seal_rel).exists() or not committed_file(seal_rel):
        raise PermissionError("TTA1 prerun gate / output seal missing or uncommitted: references stay closed")
    oseal = json.loads((ROOT / seal_rel).read_text())
    if any(sha(ROOT / p) != h for p, h in oseal["files"].items()):
        raise ValueError("sealed TTA1 outputs changed")
    d = tta1_resolve(run_dir)
    integ = tta1_integrity(d)
    from experiments.inference_cf_p2_evaluate import load_references
    plan, rows = d["plan"], d["rows"]
    ids = plan["ids"]
    P = {r["utterance_id"]: r for r in plan["rows"]}
    refs = load_references()
    R = [refs[u]["reference"] for u in ids]
    D = [refs[u]["dialogue_id"] for u in ids]
    integ["dialogues"] = D == [P[u]["dialogue_id"] for u in ids] and len(set(D)) == 20
    H = {"B0_FORCED": [P[u]["y_B_text"] for u in ids], "B0_AUTO": [P[u]["y_A_text"] for u in ids]}
    term = {"B0_FORCED": [P[u]["y_B_terminated"] for u in ids], "B0_AUTO": [P[u]["y_A_terminated"] for u in ids]}
    lens = {"B0_FORCED": [len(P[u]["y_B"]) for u in ids], "B0_AUTO": [len(P[u]["y_A"]) for u in ids]}
    if integ["all_100_resolved"]:
        H["SELECTED_TTA"] = [rows[u]["obj"]["text"] for u in ids]
        term["SELECTED_TTA"] = [rows[u]["obj"]["terminated"] for u in ids]
        lens["SELECTED_TTA"] = [rows[u]["obj"]["length"] for u in ids]
    M = {k: {**corpus_metrics(R, h), "caps": sum(t == "cap" for t in term[k]), "mean_length": float(np.mean(lens[k])),
             "total_length": int(sum(lens[k]))} for k, h in H.items()}
    integ["populations_nonempty"] = all(M["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(M[k][x] is not None and math.isfinite(M[k][x]) for k in M for x in ("pier", "mer", "en_wer", "zh_cer"))
    valid = all(integ.values())
    res = {"schema": "p2_tta1_evaluation_v1", "manifest_hash": d["manifest"]["manifest_hash"], "output_seal_hash": oseal["seal_hash"],
           "integrity": integ, "valid": valid, "metrics": M}
    if "SELECTED_TTA" not in H:
        res.update(label="P2_TTA1_INVALID", decision={"label": "P2_TTA1_INVALID"})
        return res
    S = "SELECTED_TTA"
    cc = correction_corruption(R, H["B0_FORCED"], H[S])
    zr, er = matrix_zh_retention(R, H["B0_FORCED"], H[S]), embedded_en_retention(R, H["B0_FORCED"], H[S])
    outs = [outside(r, b, s) for r, b, s in zip(R, H["B0_FORCED"], H[S])]
    osum = {x: sum(o[x] for o in outs) for x in outs[0]}
    ohr = osum["outside_harm"] / osum["baseline_correct_outside"] if osum["baseline_correct_outside"] else None
    sev = [rows[u]["obj"]["severe_truncation"] for u in ids]
    point = {"mer_increase": M[S]["mer"] - M["B0_FORCED"]["mer"], "zh_cer_increase": M[S]["zh_cer"] - M["B0_FORCED"]["zh_cer"],
             "en_wer_increase": M[S]["en_wer"] - M["B0_FORCED"]["en_wer"], "pier_gain": M["B0_FORCED"]["pier"] - M[S]["pier"],
             "net_poi_error_reduction": M["B0_FORCED"]["num_poi_errors"] - M[S]["num_poi_errors"], "zh_retention": zr["rate"],
             "en_retention": er["rate"], "poi_corruption_rate": cc["corruption_rate"], "outside_harm_rate": ohr,
             "added_caps": sum(a == "cap" and b != "cap" for a, b in zip(term[S], term["B0_FORCED"])),
             "cap_count_change": M[S]["caps"] - M["B0_FORCED"]["caps"], "new_severe_truncations": int(sum(sev))}
    dec = tta1_decide(valid, point)
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in H}
    idxs = draws(D)
    nm = {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}
    boot = {f"{S}_minus_{b}": {n: ratio_delta(C[S], C[b], a, q, idxs) for n, (a, q) in nm.items()} for b in ("B0_FORCED", "B0_AUTO")}
    rC = np.zeros((len(ids), 6))
    for j, (r, b, s) in enumerate(zip(R, H["B0_FORCED"], H[S])):
        z1, e1, c1 = matrix_zh_retention([r], [b], [s]), embedded_en_retention([r], [b], [s]), correction_corruption([r], [b], [s])
        rC[j] = [z1["numerator"], z1["denominator"], e1["numerator"], e1["denominator"], c1["corruptions"], c1["num_baseline_correct_poi"]]
    oC = np.array([[o["outside_harm"], o["baseline_correct_outside"]] for o in outs], dtype=float)
    rates = {"zh_retention": rate_ci(rC, 0, 1, idxs), "en_retention": rate_ci(rC, 2, 3, idxs), "poi_corruption": rate_ci(rC, 4, 5, idxs),
             "outside_harm": rate_ci(oC, 0, 1, idxs)}
    objs = [rows[u]["obj"] for u in ids]
    grp = {u: P[u]["group"] for u in ids}
    dm = C[S][:, 2] - C["B0_FORCED"][:, 2]
    lev0 = {u: levenshtein(P[u]["y_B"], P[u]["y_A"]) for u in ids}
    def grpstat(g):
        us = [u for u in ids if grp[u] == g]
        return {"n": len(us), "changed_vs_B0": sum(not rows[u]["obj"]["vs_B0_FORCED"]["equal"] for u in us),
                "closer_to_AUTO": sum(rows[u]["obj"]["vs_B0_AUTO"]["levenshtein"] < lev0[u] for u in us),
                "farther_from_AUTO": sum(rows[u]["obj"]["vs_B0_AUTO"]["levenshtein"] > lev0[u] for u in us),
                "lev_to_AUTO_sum": sum(rows[u]["obj"]["vs_B0_AUTO"]["levenshtein"] for u in us), "lev_B0_to_AUTO_sum": sum(lev0[u] for u in us)}
    rt = d["runtime"]
    desc = {"nll_loss_mean": [float(np.mean([o["losses"][j] for o in objs])) for j in range(3)],
            "nll_decreased": sum(o["losses"][2] < o["losses"][0] for o in objs), "no_valid_content": sum(o["no_valid_content"] for o in objs),
            "grad_l2_mean": [float(np.mean([o["grad_l2"][j] for o in objs])) for j in range(2)],
            "master_delta_l2_mean": float(np.mean([o["master_delta_l2"] for o in objs])),
            "master_delta_rel_mean": float(np.mean([o["master_delta_rel"] for o in objs])),
            "effective_delta_l2_mean": float(np.mean([o["effective_delta_l2"] for o in objs])),
            "outputs_changed_fraction": sum(not o["vs_B0_FORCED"]["equal"] for o in objs) / len(objs),
            "equal_to_AUTO": sum(o["vs_B0_AUTO"]["equal"] for o in objs), "equal_B0_to_AUTO": sum(lev0[u] == 0 for u in ids),
            "group_D": grpstat("D"), "group_A": grpstat("A"),
            "utterances_improved_mixed": int((dm < 0).sum()), "utterances_degraded_mixed": int((dm > 0).sum()),
            "origins": {"reuse_TTA0_R_A2": sum(rows[u]["origin"] != "new" for u in ids), "new": sum(rows[u]["origin"] == "new" for u in ids)},
            "runtime": {k: rt.get(k) for k in ("elapsed_sec", "setup_sec", "encoder_sec", "theta0_decode_sec", "A2", "gpu", "node", "job_id")},
            "counters": rt.get("counters")}
    res.update(point=point, decision=dec, label=dec["label"], transitions=cc, retention={"matrix_zh": zr, "embedded_en": er},
               outside={**osum, "harm_rate": ohr}, gains_vs_B0_FORCED=error_metric_gains(M["B0_FORCED"], M[S]),
               gains_vs_B0_AUTO=error_metric_gains(M["B0_AUTO"], M[S]), bootstrap=boot, rates=rates, descriptive=desc,
               thresholds=json.loads((ROOT / CONFIG).read_text())["TTA1"]["thresholds"],
               per_utterance=[{"utterance_id": u, "dialogue_id": dd, "group": grp[u], "origin": rows[u]["origin"], **{k: H[k][j] for k in H},
                               "terminated": {k: term[k][j] for k in term}, "length": {k: lens[k][j] for k in lens},
                               "counts": {k: C[k][j].tolist() for k in C}} for j, (u, dd) in enumerate(zip(ids, D))])
    return res


def committed_file(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("r-evaluate").add_argument("--out", required=True)
    q = sub.add_parser("tta1-evaluate")
    q.add_argument("--run", required=True)
    q.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    if args.cmd == "tta1-evaluate":
        res = jsonable(tta1_evaluate(args.run))
        atomic_json(out, res)
        print(json.dumps({k: res.get(k) for k in ("label", "integrity", "point", "decision")}, indent=1))
        print(json.dumps({k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()}, indent=1))
        return
    res = jsonable(r_evaluate())
    atomic_json(out, res)
    print(json.dumps({"r_label": res["r_label"], "selected": res["selected"], "labels": res["labels"], "integrity": res["integrity"],
                      "metrics": {k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()},
                      "points": {k: v["point"] for k, v in res["objectives"].items()},
                      "safety": {k: v["safety"] for k, v in res["objectives"].items()},
                      "useful": {k: v["useful"] for k, v in res["objectives"].items()},
                      "cb": res["objectives"].get("A1", {}).get("confirmation_bias"),
                      "differing": res["auto_forced_differing_cases"]}, indent=1))


if __name__ == "__main__":
    main()
