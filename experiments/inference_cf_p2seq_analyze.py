#!/usr/bin/env python
"""P2-SEQ canonical evaluation and frozen decision (CPU, evaluator side). Frozen contract 660a619.

Systems: S0 matched forced-ZH (alpha 0, new driver), S1 ordinary AUTO (reused or computed), S2 STEER.
Metrics through ``csasr.evaluation.canonical`` / pier / mer / retention primitives; lexical outside-POI
edits and correctness-flip harm from reference-anchored align_tokens projections and unit_status;
count-sum paired dialogue bootstrap (2000 draws, seed 240924, pointwise 95%, descriptive only);
frozen four-label precedence. References are read only here.
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
from csasr.evaluation.mer import INS, align_tokens, corpus_mer
from csasr.evaluation.normalization import normalize_text, segment_units
from csasr.evaluation.pier import pier, reference_pois, unit_status
from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
from csasr.inference_cf.core import atomic_json, digest, file_hash
import experiments.inference_cf_p2seq as run

G = 1e-12
TH = {"min_net_POI_error_reduction": 5, "min_PIER_gain_abs": 0.005, "max_MER_increase_abs": 0.005,
      "max_ZH_CER_increase_abs": 0.005, "min_matrix_ZH_retention": 0.99, "min_embedded_EN_retention": 0.95,
      "max_POI_corruption_rate": 0.05, "max_lexical_outside_harm_rate": 0.01, "max_cap_increase_count": 1}


# ---- per-utterance counts / outside-POI ---------------------------------------------------------------

def utt_counts(ref: str, hyp: str) -> list[int]:
    p, m = pier([ref], [hyp]), corpus_mer([ref], [hyp])
    zh = round(m["zh_cer"] * m["num_zh_ref"]) if m["num_zh_ref"] else 0
    en = round(m["en_wer"] * m["num_en_ref"]) if m["num_en_ref"] else 0
    return [p["num_poi_errors"], p["num_poi"], m["substitutions"] + m["deletions"] + m["insertions"], m["num_ref_tokens"],
            zh, m["num_zh_ref"], en, m["num_en_ref"]]


def projection(ref_tokens, hyp_tokens) -> dict:
    """Reference-anchored projection: ref index -> (aligned hyp surface or None, inserted tokens after it);
    leading insertions keyed -1."""
    ops = align_tokens(ref_tokens, hyp_tokens)
    proj = {i: [None, []] for i in range(-1, len(ref_tokens))}
    last = -1
    for o in ops:
        if o.op == INS:
            proj[last][1].append(hyp_tokens[o.hyp_idx])
        else:
            proj[o.ref_idx][0] = hyp_tokens[o.hyp_idx] if o.hyp_idx is not None else None
            last = o.ref_idx
    return proj


def outside(ref: str, base: str, meth: str) -> dict:
    norm = normalize_text(ref)
    units = segment_units(norm)
    rt = [u.surface for u in units]
    pois = {i for i, _ in reference_pois(norm)}
    out_idx = [i for i in range(len(rt)) if i not in pois]
    pb = projection(rt, [u.surface for u in segment_units(normalize_text(base))])
    pm = projection(rt, [u.surface for u in segment_units(normalize_text(meth))])
    sb, sm = unit_status(ref, base), unit_status(ref, meth)
    edits = sum(1 for i in out_idx if pb[i] != pm[i])
    base_ok = [i for i in out_idx if sb.get(i, (False,))[0]]
    harm = sum(1 for i in base_ok if not sm.get(i, (False,))[0])
    return {"outside_units": len(out_idx), "outside_edits": edits, "leading_insertion_change": int(pb[-1] != pm[-1]),
            "baseline_correct_outside": len(base_ok), "outside_harm": harm}


# ---- bootstrap ----------------------------------------------------------------------------------------

def draws(dialogues: list[str], reps=2000, seed=240924):
    keys = sorted(set(dialogues))
    groups = {k: [i for i, d in enumerate(dialogues) if d == k] for k in keys}
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(reps)]


def ratio_delta(ca: np.ndarray, cb: np.ndarray, num: int, den: int, idxs) -> dict:
    """Point and pointwise-95% interval of rate(a) - rate(b) from count sums; empty-denominator draws omitted."""
    def rate(c, idx):
        s = c[idx].sum(axis=0)
        return s[num] / s[den] if s[den] else None
    all_ = np.arange(len(ca))
    pa, pb = rate(ca, all_), rate(cb, all_)
    vals = []
    for idx in idxs:
        a, b = rate(ca, idx), rate(cb, idx)
        if a is not None and b is not None:
            vals.append(a - b)
    return {"delta": None if pa is None or pb is None else pa - pb, "valid_draws": len(vals),
            "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None}


def rate_ci(c: np.ndarray, num: int, den: int, idxs) -> dict:
    s = c.sum(axis=0)
    vals = [c[i].sum(axis=0)[num] / c[i].sum(axis=0)[den] for i in idxs if c[i].sum(axis=0)[den]]
    return {"rate": (s[num] / s[den]) if s[den] else None, "numerator": int(s[num]), "denominator": int(s[den]),
            "valid_draws": len(vals), "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else None}


# ---- frozen decision -----------------------------------------------------------------------------------

def decide(valid: bool, s: dict) -> dict:
    """s: point quantities for STEER vs matched forced."""
    safety = {"MER": s["mer_increase"] <= TH["max_MER_increase_abs"] + G,
              "ZH_CER": s["zh_cer_increase"] <= TH["max_ZH_CER_increase_abs"] + G,
              "matrix_ZH_retention": s["zh_retention"] is None or s["zh_retention"] >= TH["min_matrix_ZH_retention"] - G,
              "embedded_EN_retention": s["en_retention"] is None or s["en_retention"] >= TH["min_embedded_EN_retention"] - G,
              "POI_corruption": s["poi_corruption_rate"] is None or s["poi_corruption_rate"] <= TH["max_POI_corruption_rate"] + G,
              "outside_harm": s["outside_harm_rate"] is None or s["outside_harm_rate"] <= TH["max_lexical_outside_harm_rate"] + G,
              "caps": s["cap_increase"] <= TH["max_cap_increase_count"]}
    benefit = {"net_POI": s["net_poi_error_reduction"] >= TH["min_net_POI_error_reduction"],
               "PIER_gain": s["pier_gain"] >= TH["min_PIER_gain_abs"] - G}
    if not valid:
        lab = "P2_SEQ_INVALID"
    elif not all(safety.values()):
        lab = "P2_SEQ_SEQUENCE_DAMAGE"
    elif all(benefit.values()):
        lab = "P2_SEQ_PROMISING"
    else:
        lab = "P2_SEQ_NO_USEFUL_GAIN"
    return {"label": lab, "safety": safety, "benefit": benefit}


# ---- load / analyze ------------------------------------------------------------------------------------

def load(run_dir: Path) -> dict:
    m = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("manifest")
    reuse = json.loads((ROOT / run.REUSE).read_text())
    if reuse["reuse_hash"] != m["reuse_hash"]:
        raise ValueError("reuse seal")
    rows = {}
    for i, u in enumerate(m["ids"]):
        p = run_dir / f"rows/{i:03d}.json"
        r = json.loads(p.read_text()) if p.exists() else {"status": "missing", "systems": {}}
        if r.get("status") == "ok":
            if r["identity"] != u or r["manifest_hash"] != m["manifest_hash"] or file_hash(run_dir / f"rows/{i:03d}.npz") != r["vectors_sha256"]:
                raise ValueError(f"row {i}")
        rows[u] = r
    auto = {}
    if reuse["decision_AUTO"] == "REUSE_AUTO_ALL_100":
        for x in reuse["rows"]:
            if run.sha_file(ROOT / x["row"]) != x["row_sha256"]:
                raise ValueError("reused AUTO row changed")
            a = json.loads((ROOT / x["row"]).read_text())["systems"]["B0_AUTO"]
            auto[x["utterance_id"]] = {"text": a["text"], "tokens": a["tokens"], "terminated": a["terminated"], "source": x["row"]}
    else:
        auto = {u: rows[u]["systems"].get("S1") for u in m["ids"]}
    return {"manifest": m, "reuse": reuse, "rows": rows, "auto": auto,
            "runtime": json.loads((run_dir / "runtime.json").read_text())}


def integrity(data: dict) -> dict:
    ids = data["manifest"]["ids"]
    rows = data["rows"]
    out = {"all_100_ok": all(rows[u].get("status") == "ok" and {"S0", "S2"} <= set(rows[u]["systems"]) for u in ids)
           and all(data["auto"].get(u) for u in ids)}
    s0_bitwise = s2_pre_edit = d2_ident = lineage = zero_gate = True
    unattributed = []
    for u in ids:
        if rows[u].get("status") != "ok":
            continue
        s0, s2 = rows[u]["systems"]["S0"], rows[u]["systems"]["S2"]
        s0_bitwise &= all(st["S_bitwise_B"] and st["next"] == st["unsteered_next"] and not st["d2_called"] for st in s0["steps"])
        fe = s2["first_edit_t"]
        s2_pre_edit &= all(st["S_bitwise_B"] for st in s2["steps"] if fe is None or st["t"] < fe)
        d2_ident &= all(st["d2_logits_bitwise_B"] and st["d2_site_bitwise_B"] for st in s2["steps"] if st["d2_called"])
        lineage &= s0["lineage_ok"] and s2["lineage_ok"] and s0["distinct_caches"] and s2["distinct_caches"]
        zero_gate &= all(st.get("noedit_site_identity", True) for x in (s0, s2) for st in x["steps"] if not st["edited"])
        if s2["tokens"] != s0["tokens"] or s2["terminated"] != s0["terminated"]:
            k = next((t for t, (a, b) in enumerate(zip(s2["tokens"], s0["tokens"])) if a != b), min(len(s2["tokens"]), len(s0["tokens"])))
            if fe is None or fe > k:
                unattributed.append({"utterance_id": u, "first_divergence": k, "first_edit": fe})
    out.update(s0_bitwise_every_step=s0_bitwise, s2_bitwise_until_first_edit=s2_pre_edit, d2_scratch_identity=d2_ident,
               cache_lineage=lineage, zero_gate_identity=zero_gate, divergences_attributed=not unattributed,
               runtime_completed=data["runtime"].get("status") == "completed",
               s0_no_d2=data["runtime"]["counters"]["S0"]["d2_calls"] == 0 and data["runtime"]["counters"]["S0"]["autograd_calls"] == 0)
    return {"checks": out, "unattributed": unattributed}


def analyze(run_dir: Path) -> dict:
    from experiments.inference_cf_p2_evaluate import load_references
    data = load(run_dir)
    ids = data["manifest"]["ids"]
    refs = load_references()
    R = [refs[u]["reference"] for u in ids]
    dlg = [refs[u]["dialogue_id"] for u in ids]
    panel_dlg = {r["utterance_id"]: r["dialogue_id"] for r in json.loads((ROOT / run.PANEL).read_text())["rows"]}
    integ = integrity(data)
    integ["checks"]["dialogues_match_panel"] = all(panel_dlg[u] == d for u, d in zip(ids, dlg)) and len(set(dlg)) == 20
    rows = data["rows"]
    H = {"S0": [rows[u]["systems"]["S0"]["text"] for u in ids], "S2": [rows[u]["systems"]["S2"]["text"] for u in ids],
         "S1": [data["auto"][u]["text"] for u in ids]}
    caps = {"S0": sum(rows[u]["systems"]["S0"]["terminated"] == "cap" for u in ids),
            "S2": sum(rows[u]["systems"]["S2"]["terminated"] == "cap" for u in ids),
            "S1": sum(data["auto"][u]["terminated"] == "cap" for u in ids)}
    M = {k: {**corpus_metrics(R, h), "caps": caps[k]} for k, h in H.items()}
    for k in M:
        integ["checks"][f"finite_{k}"] = all(M[k][x] is not None for x in ("pier", "mer", "en_wer", "zh_cer"))
    cc = correction_corruption(R, H["S0"], H["S2"])
    integ["checks"]["poi_transition_identity"] = cc["net_corrections"] == M["S0"]["num_poi_errors"] - M["S2"]["num_poi_errors"]
    zr, er = matrix_zh_retention(R, H["S0"], H["S2"]), embedded_en_retention(R, H["S0"], H["S2"])
    outs = [outside(r, b, s) for r, b, s in zip(R, H["S0"], H["S2"])]
    osum = {k: sum(o[k] for o in outs) for k in outs[0]}
    oh_rate = osum["outside_harm"] / osum["baseline_correct_outside"] if osum["baseline_correct_outside"] else None
    point = {"mer_increase": M["S2"]["mer"] - M["S0"]["mer"], "zh_cer_increase": M["S2"]["zh_cer"] - M["S0"]["zh_cer"],
             "en_wer_increase": M["S2"]["en_wer"] - M["S0"]["en_wer"], "pier_gain": M["S0"]["pier"] - M["S2"]["pier"],
             "net_poi_error_reduction": M["S0"]["num_poi_errors"] - M["S2"]["num_poi_errors"],
             "zh_retention": zr["rate"], "en_retention": er["rate"], "poi_corruption_rate": cc["corruption_rate"],
             "outside_harm_rate": oh_rate, "cap_increase": caps["S2"] - caps["S0"]}
    dec = decide(all(integ["checks"].values()), point)
    # bootstrap (descriptive)
    C = {k: np.array([utt_counts(r, h) for r, h in zip(R, H[k])], dtype=float) for k in H}
    idxs = draws(dlg)
    names = {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}
    boot = {f"S2_minus_{c}": {n: ratio_delta(C["S2"], C[c], a, b, idxs) for n, (a, b) in names.items()} for c in ("S0", "S1")}
    primary_ok = all(boot["S2_minus_S0"][n]["valid_draws"] >= 1980 for n in names)
    rC = np.array([[0, 0, 0, 0, 0, 0] for _ in ids], dtype=float)
    for j, (r, b, s) in enumerate(zip(R, H["S0"], H["S2"])):
        z1, e1, c1 = matrix_zh_retention([r], [b], [s]), embedded_en_retention([r], [b], [s]), correction_corruption([r], [b], [s])
        rC[j] = [z1["numerator"], z1["denominator"], e1["numerator"], e1["denominator"], c1["corruptions"], c1["num_baseline_correct_poi"]]
    oC = np.array([[o["outside_harm"], o["baseline_correct_outside"]] for o in outs], dtype=float)
    rates = {"zh_retention": rate_ci(rC, 0, 1, idxs), "en_retention": rate_ci(rC, 2, 3, idxs),
             "poi_corruption": rate_ci(rC, 4, 5, idxs), "outside_harm": rate_ci(oC, 0, 1, idxs)}
    # steering behavior
    st2 = [st for u in ids for st in rows[u]["systems"]["S2"]["steps"]]
    elig = [s for s in st2 if s["query"] >= 4 and s["fb"] is None]
    edited = [s for s in st2 if s["edited"]]
    behavior = {"utterances_edited": sum(any(s["edited"] for s in rows[u]["systems"]["S2"]["steps"]) for u in ids),
                "decoded_steps": len(st2), "eligible_steps": len(elig), "positive_gate": sum(s["g"] > 0 for s in elig),
                "d2_calls": sum(s["d2_called"] for s in st2), "valid_direction": sum(s["dir"] == "ok" for s in st2 if s["d2_called"]),
                "actual_edits": len(edited), "realized_energy": float(sum(s["edit_norm"] ** 2 for s in edited)),
                "relative_energy": float(sum(s["edit_norm"] ** 2 for s in edited) / max(sum(s["pre_norm"] ** 2 for s in edited), 1e-12))
                if edited else 0.0, "mean_gate_all": float(np.mean([s["g"] for s in st2])), "max_gate_all": float(max(s["g"] for s in st2)),
                "mean_gate_eligible": float(np.mean([s["g"] for s in elig])) if elig else None,
                "transcripts_changed_vs_S0": sum(rows[u]["systems"]["S2"]["tokens"] != rows[u]["systems"]["S0"]["tokens"] for u in ids)}
    if not primary_ok:
        dec = {**dec, "label": "P2_SEQ_INVALID", "reason": "bootstrap_valid_draws"}
    return {"schema": "p2_seq_analysis_v1", "manifest_hash": data["manifest"]["manifest_hash"], "integrity": integ,
            "metrics": M, "gains_S2_vs_S0": error_metric_gains(M["S0"], M["S2"]), "gains_S2_vs_S1": error_metric_gains(M["S1"], M["S2"]),
            "transitions": cc, "retention": {"matrix_zh": zr, "embedded_en": er}, "outside": {**osum, "harm_rate": oh_rate},
            "point": point, "thresholds": TH, "decision": dec, "label": dec["label"], "bootstrap": boot, "rates": rates,
            "behavior": behavior, "runtime": data["runtime"], "auto_source": data["reuse"]["decision_AUTO"],
            "per_utterance": [{"utterance_id": u, "dialogue_id": d, "S0": H["S0"][j], "S1": H["S1"][j], "S2": H["S2"][j],
                               "counts": {k: C[k][j].tolist() for k in C}, "outside": outs[j]} for j, (u, d) in enumerate(zip(ids, dlg))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable(analyze(ROOT / args.run))
    atomic_json(out, res)
    print(json.dumps({"label": res["label"], "integrity": res["integrity"]["checks"], "point": res["point"],
                      "metrics": {k: {x: v[x] for x in ("pier", "mer", "en_wer", "zh_cer", "num_poi_errors", "caps")} for k, v in res["metrics"].items()},
                      "decision": res["decision"], "behavior": res["behavior"]}, indent=1))


if __name__ == "__main__":
    main()
