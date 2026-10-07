#!/usr/bin/env python
"""P2-TTA0 canonical evaluation, frozen per-objective labels and objective selection (CPU, evaluator side).
Frozen contract eb0da2a. References are read only here, after all adaptation outputs are sealed in run1.

Systems: B0_FORCED (sealed P2-SEQ S0 = theta0 teacher y_B), B0_AUTO (sealed historical AUTO = y_A), A1 GREEDY-EM,
A2 AUTO-CONSISTENCY (final forced-ZH decodes). Canonical metrics via csasr.evaluation primitives; lexical
outside-POI harm and per-utterance counts via the P2-SEQ evaluator helpers; shared paired dialogue bootstrap
(2000 draws, seed 240924, descriptive only).
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

from csasr.evaluation.canonical import corpus_metrics, correction_corruption, error_metric_gains
from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
from csasr.inference_cf.core import atomic_json, digest, file_hash
from experiments.inference_cf_p2seq_analyze import draws, outside, rate_ci, ratio_delta, utt_counts
import experiments.inference_cf_p2tta0 as run

G = 1e-12
CFG = json.loads((ROOT / run.CONFIG).read_text())
TH = CFG["decision"]["thresholds"]
LABELS = {"A1": ("TTA0_EM_VIABLE", "TTA0_EM_NOT_VIABLE", "TTA0_EM_CONFIRMATION_BIAS"),
          "A2": ("TTA0_AC_VIABLE", "TTA0_AC_NOT_VIABLE", None)}


def safety_predicates(p: dict) -> dict:
    le = lambda v, t: v is None or v <= t + G
    ge = lambda v, t: v is None or v >= t - G
    return {"MER": p["mer_increase"] <= TH["max_MER_increase"] + G,
            "ZH_CER": p["zh_cer_increase"] <= TH["max_ZH_CER_increase"] + G,
            "matrix_ZH_retention": ge(p["zh_retention"], TH["min_matrix_ZH_retention"]),
            "embedded_EN_retention": ge(p["en_retention"], TH["min_embedded_EN_retention"]),
            "outside_harm": le(p["outside_harm_rate"], TH["max_outside_POI_harm_rate"]),
            "POI_corruption": le(p["poi_corruption_rate"], TH["max_POI_corruption_rate"]),
            "caps": p["added_caps"] <= TH["max_additional_caps"],
            "severe_truncation": p["new_severe_truncations"] <= TH["max_new_severe_truncations"]}


def useful_signal(p: dict) -> dict:
    alt = p["improved_utterances"] >= TH["alternative_min_MER_improved_utterances"] and \
        p["improved_utterances"] > p["degraded_utterances"]
    return {"net_POI": p["net_poi_error_reduction"] >= TH["min_net_POI_error_reduction"], "utterance_majority": alt,
            "useful": p["net_poi_error_reduction"] >= TH["min_net_POI_error_reduction"] or alt}


def confirmation_bias(h0: float | None, h2: float | None, safety: dict) -> dict:
    ok = h0 is not None and h2 is not None and math.isfinite(h0) and math.isfinite(h2) and h0 > 0
    rel = (h0 - h2) / h0 if ok else None
    dec = bool(ok and rel >= TH["EM_entropy_relative_reduction_min"] - G)
    return {"theta0": h0, "theta2": h2, "relative_reduction": rel, "denominator_ok": ok, "entropy_decreased_1pct": dec,
            "any_safety_fail": not all(safety.values()), "flag": dec and not all(safety.values())}


def label_objective(kind: str, valid: bool, safety: dict, useful: dict, cb: dict | None) -> str:
    v, nv, cbl = LABELS[kind]
    if not valid:
        return "P2_TTA0_INVALID"
    if kind == "A1" and cb is not None and cb["flag"]:
        return cbl
    return v if useful["useful"] and all(safety.values()) else nv


def select(labels: dict, metrics: dict, mean_update: dict, stage_valid: bool) -> dict:
    viable = [k for k in ("A1", "A2") if labels[k] in ("TTA0_EM_VIABLE", "TTA0_AC_VIABLE")]
    keys_log = []
    if not stage_valid:
        return {"stage_label": "P2_TTA0_INVALID", "selected": None, "viable": viable, "tie_keys": keys_log}
    if not viable:
        return {"stage_label": "P2_TTA0_NO_VIABLE_OBJECTIVE", "selected": None, "viable": viable, "tie_keys": keys_log}
    if len(viable) == 1:
        return {"stage_label": "P2_TTA0_OBJECTIVE_SELECTED", "selected": viable[0], "viable": viable, "tie_keys": keys_log}
    for key in ("mer", "pier", "zh_cer", "update"):
        a = mean_update["A1"] if key == "update" else metrics["A1"][key]
        b = mean_update["A2"] if key == "update" else metrics["A2"][key]
        keys_log.append({"key": key, "A1": a, "A2": b})
        if abs(a - b) > G:
            return {"stage_label": "P2_TTA0_OBJECTIVE_SELECTED", "selected": "A1" if a < b else "A2", "viable": viable, "tie_keys": keys_log}
    return {"stage_label": "P2_TTA0_OBJECTIVE_SELECTED", "selected": "A2", "viable": viable, "tie_keys": keys_log, "exact_tie": True}


def load(run_dir: Path) -> dict:
    m = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("manifest")
    seal = json.loads((ROOT / run.SEAL).read_text())
    if seal["seal_hash"] != m["seal_hash"] or digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"]:
        raise ValueError("seal")
    rows = []
    for i, u in enumerate(m["ids"]):
        p = run_dir / f"rows/{i:02d}.json"
        r = json.loads(p.read_text()) if p.exists() else {"status": "missing", "identity": u}
        if r.get("status") == "ok" and (r["identity"] != u or r["manifest_hash"] != m["manifest_hash"]
                                        or file_hash(run_dir / f"rows/{i:02d}_final_masters_fp32.npz") != r["final_masters_npz_sha256"]):
            raise ValueError(f"row {i}")
        rows.append(r)
    return {"manifest": m, "seal": seal, "rows": rows, "runtime": json.loads((run_dir / "runtime.json").read_text())}


def integrity(data: dict) -> dict:
    rows, rt, seal = data["rows"], data["runtime"], data["seal"]
    ok = all(r.get("status") == "ok" and set(r["objectives"]) == {"A1", "A2"} for r in rows) and len(rows) == 20
    c = {"all_20x2_ok": ok, "runtime_completed": rt.get("status") == "completed", "no_runtime_invalid": not rt.get("invalid")}
    if not ok:
        return c
    th0 = rt["theta0_ln_hash"]
    c["theta0_matches_S0"] = all(r["theta0_decode"]["tokens_equal_S0"] and r["theta0_decode"]["terminated_equal_S0"] for r in rows)
    c["live_audit_pass"] = all(rows[0].get("live_audit", {}).get(k, {}).get("pass") is True for k in ("A1", "A2"))
    objs = [r["objectives"][k] for r in rows for k in ("A1", "A2")]
    c["two_steps_three_losses"] = all(o["steps"] == 2 and o["loss_evaluations"] == 3 and len(o["losses"]) == 3 and len(o["grad_l2"]) == 2
                                      for o in objs)
    c["finite"] = all(all(math.isfinite(x) for x in o["losses"] + o["grad_l2"]) and all(o["finite"])
                      and math.isfinite(o["master_delta_l2"]) and math.isfinite(o["effective_delta_l2"]) for o in objs)
    c["reset_each_objective"] = all(o["reset_ok"] and o["start_hash"] == th0 and o["end_hash"] == th0 and o["other_versions_unchanged"]
                                    for o in objs)
    c["reset_final"] = bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged")) and bool(rt.get("model_grads_none"))
    cnt = rt["counters"]
    c["compute_counts"] = (cnt["A1"]["backwards"] == 40 and cnt["A2"]["backwards"] == 40 and cnt["A1"]["optimizer_steps"] == 40
                           and cnt["A2"]["optimizer_steps"] == 40 and cnt["A1"]["teacher_forwards"] == 60 and cnt["A2"]["teacher_forwards"] == 80
                           and cnt["A1"]["final_decodes"] == 20 and cnt["A2"]["final_decodes"] == 20 and cnt["theta0_integrity_decodes"] == 20
                           and cnt["encoder_passes"] == 20 and cnt["audit"] == {"forwards": 2, "backwards": 2})
    c["teachers_from_seal"] = all(r["identity"] == s["utterance_id"] for r, s in zip(rows, seal["rows"]))
    return c


def analyze(run_dir: Path) -> dict:
    from experiments.inference_cf_p2_evaluate import load_references
    data = load(run_dir)
    ids = data["manifest"]["ids"]
    seal = {r["utterance_id"]: r for r in data["seal"]["rows"]}
    integ = integrity(data)
    valid_runs = all(integ.values())
    refs = load_references()
    R = [refs[u]["reference"] for u in ids]
    D = [refs[u]["dialogue_id"] for u in ids]
    integ["dialogues"] = D == [seal[u]["dialogue_id"] for u in ids] and len(set(D)) == 20
    rows = data["rows"]
    H = {"B0_FORCED": [seal[u]["y_B_text"] for u in ids], "B0_AUTO": [seal[u]["y_A_text"] for u in ids]}
    term = {"B0_FORCED": [seal[u]["y_B_terminated"] for u in ids], "B0_AUTO": [seal[u]["y_A_terminated"] for u in ids]}
    lens = {"B0_FORCED": [len(seal[u]["y_B"]) for u in ids], "B0_AUTO": [len(seal[u]["y_A"]) for u in ids]}
    if integ["all_20x2_ok"]:
        for k in ("A1", "A2"):
            H[k] = [r["objectives"][k]["text"] for r in rows]
            term[k] = [r["objectives"][k]["terminated"] for r in rows]
            lens[k] = [r["objectives"][k]["length"] for r in rows]
    M = {k: {**corpus_metrics(R, h), "caps": sum(t == "cap" for t in term[k]), "mean_length": float(np.mean(lens[k])),
             "eos_terminated": sum(t == "eos" for t in term[k])} for k, h in H.items()}
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in H}
    integ["populations_nonempty"] = all(M["B0_FORCED"][x] > 0 for x in ("num_poi", "num_zh_ref", "num_en_ref"))
    integ["finite_metrics"] = all(M[k][x] is not None and math.isfinite(M[k][x]) for k in M for x in ("pier", "mer", "en_wer", "zh_cer"))
    stage_valid = all(integ.values())
    per_obj, labels, mean_update = {}, {}, {}
    idxs = draws(D)
    for k in ("A1", "A2"):
        if k not in H:
            labels[k] = "P2_TTA0_INVALID"
            continue
        cc = correction_corruption(R, H["B0_FORCED"], H[k])
        zr, er = matrix_zh_retention(R, H["B0_FORCED"], H[k]), embedded_en_retention(R, H["B0_FORCED"], H[k])
        outs = [outside(r, b, s) for r, b, s in zip(R, H["B0_FORCED"], H[k])]
        osum = {x: sum(o[x] for o in outs) for x in outs[0]}
        ohr = osum["outside_harm"] / osum["baseline_correct_outside"] if osum["baseline_correct_outside"] else None
        dm = C[k][:, 2] - C["B0_FORCED"][:, 2]
        sev = [r["objectives"][k]["severe_truncation"] for r in rows]
        point = {"mer_increase": M[k]["mer"] - M["B0_FORCED"]["mer"], "zh_cer_increase": M[k]["zh_cer"] - M["B0_FORCED"]["zh_cer"],
                 "en_wer_increase": M[k]["en_wer"] - M["B0_FORCED"]["en_wer"], "pier_gain": M["B0_FORCED"]["pier"] - M[k]["pier"],
                 "net_poi_error_reduction": M["B0_FORCED"]["num_poi_errors"] - M[k]["num_poi_errors"],
                 "zh_retention": zr["rate"], "en_retention": er["rate"], "poi_corruption_rate": cc["corruption_rate"],
                 "outside_harm_rate": ohr,
                 "added_caps": sum(a == "cap" and b != "cap" for a, b in zip(term[k], term["B0_FORCED"])),
                 "cap_count_change": M[k]["caps"] - M["B0_FORCED"]["caps"],
                 "new_severe_truncations": sum(sev), "improved_utterances": int((dm < 0).sum()), "degraded_utterances": int((dm > 0).sum()),
                 "tied_utterances": int((dm == 0).sum())}
        safety, useful = safety_predicates(point), useful_signal(point)
        objs = [r["objectives"][k] for r in rows]
        mean_update[k] = float(np.mean([o["master_delta_l2"] for o in objs]))
        cb = None
        h = {}
        for th in ("theta0", "theta2"):
            src = "A1"                          # theta0 common y_B summary from A1 L0 (shared); theta2 per objective
            items = [r["objectives"][src if th == "theta0" else k]["common_y_B"].get(th) for r in rows]
            n = sum(x["valid"] for x in items)
            h[th] = sum(x["entropy_sum"] for x in items) / n if n else None
        if k == "A1":
            cb = confirmation_bias(h["theta0"], h["theta2"], safety)
        labels[k] = label_objective(k, stage_valid, safety, useful, cb)

        def cmean(th, key, src):
            v = [x for r in rows for x in r["objectives"][src]["common_y_B"][th][key]]
            return float(np.mean(v)) if v else None
        boot = {f"{k}_minus_{b}": {n: ratio_delta(C[k], C[b], a, d, idxs) for n, (a, d) in
                                    {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}.items()}
                for b in ("B0_FORCED", "B0_AUTO")}
        rC = np.zeros((len(ids), 6))
        for j, (r, b, s) in enumerate(zip(R, H["B0_FORCED"], H[k])):
            z1, e1, c1 = matrix_zh_retention([r], [b], [s]), embedded_en_retention([r], [b], [s]), correction_corruption([r], [b], [s])
            rC[j] = [z1["numerator"], z1["denominator"], e1["numerator"], e1["denominator"], c1["corruptions"], c1["num_baseline_correct_poi"]]
        oC = np.array([[o["outside_harm"], o["baseline_correct_outside"]] for o in outs], dtype=float)
        rates = {"zh_retention": rate_ci(rC, 0, 1, idxs), "en_retention": rate_ci(rC, 2, 3, idxs),
                 "poi_corruption": rate_ci(rC, 4, 5, idxs), "outside_harm": rate_ci(oC, 0, 1, idxs)}
        lev_auto_b0 = [seal[u]["y_B"] != seal[u]["y_A"] for u in ids]
        per_obj[k] = {"point": point, "safety": safety, "useful": useful, "confirmation_bias": cb, "label": labels[k],
                      "transitions": cc, "retention": {"matrix_zh": zr, "embedded_en": er}, "outside": {**osum, "harm_rate": ohr},
                      "gains_vs_B0_FORCED": error_metric_gains(M["B0_FORCED"], M[k]), "gains_vs_B0_AUTO": error_metric_gains(M["B0_AUTO"], M[k]),
                      "bootstrap": boot, "rates": rates,
                      "descriptive": {
                          "loss_mean": [float(np.mean([o["losses"][j] for o in objs])) for j in range(3)],
                          "loss_decreased_utterances": sum(o["losses"][2] < o["losses"][0] for o in objs),
                          "loss_increased_utterances": sum(o["losses"][2] > o["losses"][0] for o in objs),
                          "grad_l2_mean": [float(np.mean([o["grad_l2"][j] for o in objs])) for j in range(2)],
                          "master_delta_l2_mean": mean_update[k], "master_delta_rel_mean": float(np.mean([o["master_delta_rel"] for o in objs])),
                          "effective_delta_l2_mean": float(np.mean([o["effective_delta_l2"] for o in objs])),
                          "effective_changed_scalars_mean": float(np.mean([o["effective_changed_scalars"] for o in objs])),
                          "no_valid_content": sum(o["no_valid_content"] for o in objs),
                          "transcripts_changed_vs_B0_FORCED": sum(not o["vs_B0_FORCED"]["equal"] for o in objs),
                          "equal_to_B0_AUTO": sum(o["vs_B0_AUTO"]["equal"] for o in objs),
                          "lev_B0_to_AUTO_sum": sum(_lev(seal[u]["y_B"], seal[u]["y_A"]) for u in ids),
                          "lev_to_AUTO_sum": sum(o["vs_B0_AUTO"]["levenshtein"] for o in objs),
                          "lev_to_B0_sum": sum(o["vs_B0_FORCED"]["levenshtein"] for o in objs),
                          "auto_differs_from_B0_utterances": sum(lev_auto_b0),
                          "closer_to_AUTO_utterances": sum(o["vs_B0_AUTO"]["levenshtein"] < _lev(seal[u]["y_B"], seal[u]["y_A"])
                                                           for o, u in zip(objs, ids)),
                          "farther_from_AUTO_utterances": sum(o["vs_B0_AUTO"]["levenshtein"] > _lev(seal[u]["y_B"], seal[u]["y_A"])
                                                              for o, u in zip(objs, ids)),
                          "common_y_B_entropy": h,
                          "common_y_B_mean_P_M": {"theta0": cmean("theta0", "P_M", "A1"), "theta2": cmean("theta2", "P_M", k)},
                          "common_y_B_mean_P_E": {"theta0": cmean("theta0", "P_E", "A1"), "theta2": cmean("theta2", "P_E", k)},
                          "common_y_B_mean_EOS": {"theta0": cmean("theta0", "eos_prob", "A1"), "theta2": cmean("theta2", "eos_prob", k)},
                          "final_query_EOS_mean": {"theta0": float(np.mean([r["objectives"]["A1"]["common_y_B"]["theta0"]["eos_prob"][-1] for r in rows])),
                                                   "theta2": float(np.mean([r["objectives"][k]["common_y_B"]["theta2"]["eos_prob"][-1] for r in rows]))},
                          "adapt_sec": data["runtime"]["systems"][k]["adapt_sec"], "decode_sec": data["runtime"]["systems"][k]["decode_sec"],
                          "peak_alloc": data["runtime"]["systems"][k]["peak_alloc"], "peak_reserved": data["runtime"]["systems"][k]["peak_reserved"]}}
    sel = select(labels, M, mean_update, stage_valid)
    return {"schema": "p2_tta0_analysis_v1", "manifest_hash": data["manifest"]["manifest_hash"], "integrity": integ,
            "stage_valid": stage_valid, "metrics": M, "objectives": per_obj, "labels": labels, "selection": sel,
            "label": sel["stage_label"], "selected": sel["selected"], "thresholds": TH, "runtime": data["runtime"],
            "per_utterance": [{"utterance_id": u, "dialogue_id": d, **{k: H[k][j] for k in H}, "terminated": {k: term[k][j] for k in term},
                               "length": {k: lens[k][j] for k in lens}, "counts": {k: C[k][j].tolist() for k in C}}
                              for j, (u, d) in enumerate(zip(ids, D))]}


def _lev(a, b):
    from csasr.inference_cf.episodic_tta import levenshtein
    return levenshtein(a, b)


def invalid_record(run_dir: Path) -> dict:
    """Reference-free terminal record for a technically INVALID run (frozen precedence: INVALID supersedes every
    label). Loads NO reference and computes NO canonical metric, so sealed adapted outputs stay unevaluated.
    Reference-free adaptation mechanics (losses, entropy on the common y_B path, update norms, lengths/EOS,
    token changes vs the sealed baselines) are summarized descriptively."""
    data = load(run_dir)
    integ = integrity(data)
    if all(integ.values()):
        raise ValueError("run is technically valid; use the full evaluator")
    rows, seal = data["rows"], {r["utterance_id"]: r for r in data["seal"]["rows"]}
    ids = data["manifest"]["ids"]
    mech = {}
    if integ["all_20x2_ok"]:
        for k in ("A1", "A2"):
            objs = [r["objectives"][k] for r in rows]
            h = {}
            for th, src in (("theta0", "A1"), ("theta2", k)):
                items = [r["objectives"][src]["common_y_B"][th] for r in rows]
                n = sum(x["valid"] for x in items)
                h[th] = sum(x["entropy_sum"] for x in items) / n if n else None
            mech[k] = {"loss_mean": [float(np.mean([o["losses"][j] for o in objs])) for j in range(3)],
                       "loss_decreased_utterances": sum(o["losses"][2] < o["losses"][0] for o in objs),
                       "grad_l2_mean": [float(np.mean([o["grad_l2"][j] for o in objs])) for j in range(2)],
                       "master_delta_l2_mean": float(np.mean([o["master_delta_l2"] for o in objs])),
                       "master_delta_rel_mean": float(np.mean([o["master_delta_rel"] for o in objs])),
                       "effective_delta_l2_mean": float(np.mean([o["effective_delta_l2"] for o in objs])),
                       "effective_changed_scalars_mean": float(np.mean([o["effective_changed_scalars"] for o in objs])),
                       "common_y_B_token_weighted_entropy": h,
                       "common_y_B_entropy_relative_reduction": (h["theta0"] - h["theta2"]) / h["theta0"] if h["theta0"] else None,
                       "transcripts_changed_vs_B0_FORCED": sum(not o["vs_B0_FORCED"]["equal"] for o in objs),
                       "equal_to_B0_AUTO": sum(o["vs_B0_AUTO"]["equal"] for o in objs),
                       "lev_to_AUTO_sum": sum(o["vs_B0_AUTO"]["levenshtein"] for o in objs),
                       "lev_B0_to_AUTO_sum": sum(_lev(seal[u]["y_B"], seal[u]["y_A"]) for u in ids),
                       "closer_to_AUTO_utterances": sum(o["vs_B0_AUTO"]["levenshtein"] < _lev(seal[u]["y_B"], seal[u]["y_A"]) for o, u in zip(objs, ids)),
                       "farther_from_AUTO_utterances": sum(o["vs_B0_AUTO"]["levenshtein"] > _lev(seal[u]["y_B"], seal[u]["y_A"]) for o, u in zip(objs, ids)),
                       "mean_length": float(np.mean([o["length"] for o in objs])),
                       "mean_length_B0_FORCED": float(np.mean([len(seal[u]["y_B"]) for u in ids])),
                       "eos_terminated": sum(o["terminated"] == "eos" for o in objs), "caps": sum(o["terminated"] == "cap" for o in objs),
                       "new_severe_truncations": sum(o["severe_truncation"] for o in objs),
                       "adapt_sec": data["runtime"]["systems"][k]["adapt_sec"], "decode_sec": data["runtime"]["systems"][k]["decode_sec"],
                       "peak_alloc": data["runtime"]["systems"][k]["peak_alloc"], "peak_reserved": data["runtime"]["systems"][k]["peak_reserved"]}
    return {"schema": "p2_tta0_invalid_record_v1", "manifest_hash": data["manifest"]["manifest_hash"], "integrity": integ,
            "failed_checks": [k for k, v in integ.items() if not v], "runtime_invalid": data["runtime"].get("invalid"),
            "live_audit": rows[0].get("live_audit") if rows and rows[0].get("status") == "ok" else None,
            "stage_valid": False, "labels": {"A1": "P2_TTA0_INVALID", "A2": "P2_TTA0_INVALID"}, "label": "P2_TTA0_INVALID",
            "selected": None, "references_loaded": False, "canonical_metrics_computed": False, "mechanics_reference_free": mech,
            "runtime": data["runtime"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--invalid-record", action="store_true", help="reference-free terminal record for a technically INVALID run")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    if args.invalid_record:
        res = jsonable(invalid_record(ROOT / args.run))
        atomic_json(out, res)
        print(json.dumps({k: res[k] for k in ("label", "failed_checks", "runtime_invalid", "live_audit", "mechanics_reference_free")}, indent=1))
        return
    res = jsonable(analyze(ROOT / args.run))
    atomic_json(out, res)
    print(json.dumps({"label": res["label"], "selected": res["selected"], "labels": res["labels"], "integrity": res["integrity"],
                      "metrics": {k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()},
                      "points": {k: v["point"] for k, v in res["objectives"].items()},
                      "safety": {k: v["safety"] for k, v in res["objectives"].items()},
                      "useful": {k: v["useful"] for k, v in res["objectives"].items()},
                      "cb": res["objectives"].get("A1", {}).get("confirmation_bias")}, indent=1))


if __name__ == "__main__":
    main()
