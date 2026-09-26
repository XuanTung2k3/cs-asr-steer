#!/usr/bin/env python
"""CPU-only P2 evaluation: metrics vs the matched alpha=0 baseline (v1.1), frozen validity/selection,
References are read here (evaluator side) and never by the runner."""
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

from csasr.evaluation.mer import corpus_mer
from csasr.evaluation.pier import evaluate_pois, pier
from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
from csasr.inference_cf.core import atomic_json, digest

CONFIG = ROOT / "configs/inference_cf/p2_compact_development.json"
SELECTION = ROOT / "results/inference_cf/p2_selection.json"
DOSE_RANK = {"id": 0, "sqrt": 1}


def load_references() -> dict:
    from experiments.inference_cf_p0_r2_prepare import build_panels
    _, ev = build_panels()
    r2m = json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())
    if digest(ev) != r2m["evaluation_panel_hash"]:
        raise ValueError("evaluation panel differs from the frozen R2 evaluation panel")
    return {r["utterance_id"]: r for r in ev["rows"]}


def load_run(out: Path) -> tuple[dict, dict, dict]:
    m = json.loads((out / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError(f"invalid manifest {out}")
    panel = json.loads((out / "panel.json").read_text())
    shards = {}
    for i, item in enumerate(panel["rows"]):
        p = out / "rows" / f"{i:03d}.json"
        shards[item["utterance_id"]] = json.loads(p.read_text()) if p.exists() else {"status": "missing", "systems": {}}
        if p.exists() and shards[item["utterance_id"]]["manifest_hash"] != m["manifest_hash"]:
            raise ValueError("stale shard")
    rt = json.loads((out / "runtime.json").read_text()) if (out / "runtime.json").exists() else {}
    return m, shards, rt


def hyps(shards: dict, ids: list[str], system: str) -> list[str] | None:
    out = []
    for u in ids:
        s = shards[u]
        if s.get("status") != "ok" or system not in s["systems"]:
            return None
        out.append(s["systems"][system]["text"])
    return out


def flips(refs: list[str], base: list[str], meth: list[str]) -> dict:
    corr = corrupt = persist = 0
    for r, b, m in zip(refs, base, meth):
        bb = {p.poi_index: p.correct for p in evaluate_pois(r, b)}
        mm = {p.poi_index: p.correct for p in evaluate_pois(r, m)}
        for k, v in bb.items():
            if not v and mm.get(k):
                corr += 1
            elif v and not mm.get(k):
                corrupt += 1
            elif not v and not mm.get(k):
                persist += 1
    return {"corrections": corr, "corruptions": corrupt, "persistent_errors": persist}


def edit_stats(shards: dict, ids: list[str], system: str) -> dict:
    steps = [s for u in ids for s in shards[u]["systems"][system].get("steps", [])]
    edited = [s for s in steps if s["edit"]]
    elig = [s for s in steps if s["fb"] is None and s["t"] >= 1]   # query >= 4 <=> t >= 1
    changed = [s for s in edited if s["next"] != s["unsteered_next"]]
    return {"steps": len(steps), "gate_positive": sum(s["g"] > 0 for s in steps),
            "edited": len(edited), "edited_token_changed": len(changed),
            "edited_no_token_change": len(edited) - len(changed),
            "structural_fallback": sum(s["fb"] is not None for s in steps),
            "direction_invalid": sum(s["dir"] != "ok" for s in steps),
            "eligible_steps": len(elig),
            "realized_energy": float(sum(s["edit_norm"] ** 2 for s in edited)),
            "mean_relative_edit": (float(np.mean([s["edit_norm"] / s["pre_norm"] for s in edited]))
                                   if edited else 0.0),
            "nominal_dose_sum": float(sum(s["dose"] for s in steps))}


def metrics(refs, ids, shards, system, base_hyps=None) -> dict:
    h = hyps(shards, ids, system)
    if h is None:
        return {"complete": False}
    rr = [refs[u]["reference"] for u in ids]
    p, m = pier(rr, h), corpus_mer(rr, h)
    out = {"complete": True, "pier": p["pier"], "pier_errors": p["num_poi_errors"], "num_poi": p["num_poi"],
           "per_category": p["per_category"], "mer": m["mer"], "en_wer": m["en_wer"], "zh_cer": m["zh_cer"],
           "mer_counts": {k: m[k] for k in ("substitutions", "deletions", "insertions", "num_ref_tokens")},
           "caps": sum(shards[u]["systems"][system]["terminated"] == "cap" for u in ids)}
    if base_hyps is not None:
        zr, er = matrix_zh_retention(rr, base_hyps, h), embedded_en_retention(rr, base_hyps, h)
        out.update(flips(rr, base_hyps, h))
        out["matrix_zh_retention"] = zr["numerator"] / zr["denominator"] if zr["denominator"] else None
        out["matrix_damage_units"] = zr["denominator"] - zr["numerator"]
        out["en_retention"] = er["numerator"] / er["denominator"] if er["denominator"] else None
    if system not in ("B0", "B1", "B0_AUTO") and not system.startswith("B0M_"):
        out["edits"] = edit_stats(shards, ids, system)
    return out


FLOAT_GUARD = 1e-12   # float-representation guard only; thresholds are the frozen values


def validity(m: dict, b0: dict, v: dict) -> dict:
    checks = {"V1_decode_health": m["complete"] and m["caps"] <= b0["caps"] + v["max_cap_increase"],
              "V2_zh_cer": m["zh_cer"] - b0["zh_cer"] <= v["max_zh_cer_increase_abs"] + FLOAT_GUARD,
              "V3_mer": m["mer"] - b0["mer"] <= v["max_mer_increase_abs"] + FLOAT_GUARD,
              "V4_retention": (m["matrix_zh_retention"] or 0) >= v["min_matrix_zh_retention"] - FLOAT_GUARD,
              "V5_efficacy": b0["pier"] - m["pier"] > v["min_pier_gain_exclusive"]}
    return {"checks": checks, "valid": all(checks.values())}


def select(cands: list[dict]) -> dict | None:
    """Frozen tie rule: higher dPIER, lower ZH-CER increase, lower energy, simpler dose, lower alpha."""
    valid = [c for c in cands if c["validity"]["valid"]]
    if not valid:
        return None
    return sorted(valid, key=lambda c: (-c["delta"]["pier_errors"], c["delta_zh_cer_increase"],
                                        c["metrics"]["edits"]["realized_energy"],
                                        DOSE_RANK[c["config"]["dose"]], c["config"]["alpha"]))[0]


def alpha_c(m: dict) -> float:
    e = m["edits"]
    return math.sqrt(e["realized_energy"] / e["eligible_steps"]) if e["eligible_steps"] else 0.0


def per_utt_counts(ref: str, hyp: str) -> tuple:
    p, m = pier([ref], [hyp]), corpus_mer([ref], [hyp])
    zh_err = (m["zh_cer"] * m["num_zh_ref"]) if m["num_zh_ref"] else 0.0
    en_err = (m["en_wer"] * m["num_en_ref"]) if m["num_en_ref"] else 0.0
    return (p["num_poi_errors"], p["num_poi"], m["substitutions"] + m["deletions"] + m["insertions"],
            m["num_ref_tokens"], zh_err, m["num_zh_ref"], en_err, m["num_en_ref"])


def paired_bootstrap(refs, ids, a_hyps, b_hyps, reps: int, seed: int) -> dict:
    """Dialogue-cluster paired bootstrap of (B - A) for PIER, MER, ZH-CER, EN-WER (positive = A better)."""
    ca = np.array([per_utt_counts(refs[u]["reference"], h) for u, h in zip(ids, a_hyps)], dtype=float)
    cb = np.array([per_utt_counts(refs[u]["reference"], h) for u, h in zip(ids, b_hyps)], dtype=float)
    dlg = [refs[u]["dialogue_id"] for u in ids]
    groups = defaultdict(list)
    for i, d in enumerate(dlg):
        groups[d].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)

    def rates(c, idx):
        s = c[idx].sum(axis=0)
        return np.array([s[0] / s[1], s[2] / s[3], s[4] / s[5], s[6] / s[7]])
    point = rates(cb, np.arange(len(ids))) - rates(ca, np.arange(len(ids)))
    draws = []
    for _ in range(reps):
        idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        draws.append(rates(cb, idx) - rates(ca, idx))
    draws = np.array(draws)
    names = ("pier", "mer", "zh_cer", "en_wer")
    return {n: {"delta": float(point[k]), "ci95": [float(np.quantile(draws[:, k], .025)),
                                                     float(np.quantile(draws[:, k], .975))]}
            for k, n in enumerate(names)}


def divergence_attribution(shards: dict, ids: list[str], name: str, base: str) -> dict:
    """Every method-vs-matched-baseline divergence must follow an applied edit (v1.1 validity)."""
    diverged, unattributed = 0, []
    for u in ids:
        x, b = shards[u]["systems"][name], shards[u]["systems"][base]
        if x["tokens"] == b["tokens"]:
            continue
        diverged += 1
        k = next((t for t, (p, q) in enumerate(zip(x["tokens"], b["tokens"])) if p != q),
                 min(len(x["tokens"]), len(b["tokens"])))
        if not any(st["edit"] and st["t"] <= k for st in x.get("steps", [])):
            unattributed.append({"utterance_id": u, "first_divergence": k})
    return {"diverged_utterances": diverged, "unattributed": unattributed}


def evaluate_configs(refs, ids, runs: dict, cfg) -> tuple[list[dict], dict]:
    """Each configuration against the matched alpha=0 baseline of its own run and layer."""
    out, baselines = [], {}
    for d, (m, shards, _) in runs.items():
        for layer in m["matched_baseline_layers"]:
            bname = f"B0M_L{layer}"
            bh = hyps(shards, ids, bname)
            baselines[f"{d}:{bname}"] = {
                "metrics": metrics(refs, ids, shards, bname),
                "zero_dose_bitwise": all(shards[u]["systems"][bname]["zero_dose_bitwise"] for u in ids
                                         if shards[u].get("status") == "ok"),
                "lineage_ok": all(shards[u]["systems"][bname]["lineage_ok"] for u in ids
                                  if shards[u].get("status") == "ok"),
                "hyps": bh}
        for c in m["configs"]:
            bkey = f"{d}:B0M_L{c['layer']}"
            b0 = baselines[bkey]["metrics"]
            m_ = metrics(refs, ids, shards, c["name"], baselines[bkey]["hyps"])
            if not m_["complete"] or not b0["complete"]:
                out.append({"config": c, "run": d, "baseline": bkey, "metrics": m_, "delta": {},
                            "validity": {"valid": False, "checks": {"complete": False}},
                            "delta_zh_cer_increase": None, "attribution": None})
                continue
            out.append({"config": c, "run": d, "baseline": bkey, "metrics": m_,
                        "validity": validity(m_, b0, cfg["validity"]),
                        "delta": {"pier": b0["pier"] - m_["pier"], "pier_errors": b0["pier_errors"] - m_["pier_errors"],
                                  "mer": b0["mer"] - m_["mer"], "en_wer": b0["en_wer"] - m_["en_wer"],
                                  "zh_cer": b0["zh_cer"] - m_["zh_cer"]},
                        "delta_zh_cer_increase": m_["zh_cer"] - b0["zh_cer"],
                        "attribution": divergence_attribution(shards, ids, c["name"], f"B0M_L{c['layer']}")})
    return out, baselines


def experiment_validity(evals, baselines, runs) -> dict:
    fails = {d: [u for u, sh in s.items() if sh.get("status") != "ok"] for d, (_, s, _) in runs.items()}
    unattr = {e["config"]["name"]: e["attribution"]["unattributed"] for e in evals
              if e.get("attribution") and e["attribution"]["unattributed"]}
    checks = {"no_failures": not any(fails.values()),
              "matched_baselines_zero_dose_bitwise": all(b["zero_dose_bitwise"] for b in baselines.values()),
              "matched_baselines_lineage": all(b["lineage_ok"] for b in baselines.values()),
              "all_divergences_edit_attributed": not unattr,
              "all_configs_complete": all(e["metrics"].get("complete") for e in evals)}
    return {"valid": all(checks.values()), "checks": checks, "failures": fails, "unattributed": unattr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "final", "A_diag", "B"])
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--baseline-dir", required=True, help="stage-A dir holding B1/B0_AUTO/cached-greedy diagnostics")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    refs = load_references()
    _, bshards, _ = load_run(ROOT / args.baseline_dir)
    ids = [r["utterance_id"] for r in json.loads((ROOT / args.baseline_dir / "panel.json").read_text())["rows"]]
    runs = {d: load_run(ROOT / d) for d in args.dirs}
    evals, baselines = evaluate_configs(refs, ids, runs, cfg)
    reference_hyps = next(iter(baselines.values()))["hyps"]
    diag = {name: metrics(refs, ids, bshards, name, reference_hyps) for name in ("B0", "B1", "B0_AUTO")}
    ev_valid = experiment_validity(evals, baselines, runs)
    matched_identical = len({tuple(b["hyps"] or []) for b in baselines.values()}) == 1
    summary = {"schema": "p2_evaluation_v1_1", "stage": args.stage,
               "matched_baselines": {k: {kk: vv for kk, vv in v.items() if kk != "hyps"} for k, v in baselines.items()},
               "matched_baselines_identical_across_runs": matched_identical,
               "diagnostic_systems": diag,
               "experiment_validity": ev_valid,
               "runtime": {d: rt for d, (_, _, rt) in runs.items()},
               "manifests": {d: m["manifest_hash"] for d, (m, _, _) in runs.items()},
               "configs": evals}
    reps, seed = cfg["bootstrap"]["replicates"], cfg["bootstrap"]["seed"]
    auto_hyps = hyps(bshards, ids, "B0_AUTO")

    def intervals(sel):
        d = sel["run"]
        shards = runs[d][1]
        sel_hyps = hyps(shards, ids, sel["config"]["name"])
        return {"vs_matched_B0": paired_bootstrap(refs, ids, sel_hyps, baselines[sel["baseline"]]["hyps"], reps, seed),
                "vs_B0_AUTO": paired_bootstrap(refs, ids, sel_hyps, auto_hyps, reps, seed)}

    if not ev_valid["valid"]:
        summary["verdict"] = "P2_BLOCKED_INVALID_EXPERIMENT"
        atomic_json(ROOT / args.out, summary)
        print(json.dumps({"stage": args.stage, "verdict": summary["verdict"],
                          "checks": ev_valid["checks"]}, indent=2))
        return
    if args.stage == "A":
        sel = select(evals)
        if sel is not None:
            selection = {"p2_a_verdict": "P2_A_VALID_CONFIG_EXISTS", "selected": sel["config"],
                         "alpha_c": alpha_c(sel["metrics"]), "alpha_c_formula": cfg["stages"]["B"]["alpha_c"]}
            summary["selected_intervals"] = intervals(sel)
        else:
            pos = [e for e in evals if e["metrics"].get("complete") and e["delta"].get("pier_errors", 0) > 0]
            best = sorted(pos, key=lambda c: (-c["delta"]["pier_errors"], c["delta_zh_cer_increase"],
                                              c["metrics"]["edits"]["realized_energy"], c["config"]["alpha"]))
            diagc = best[0]["config"] if best else {"layer": 24, "alpha": 1.0, "gate": "ER", "dose": "id",
                                                    "direction_sign": 1.0}
            selection = {"p2_a_verdict": "P2_A_NO_VALID_CONFIG", "diagnostic_config": diagc}
        summary["selection"] = selection
        atomic_json(ROOT / args.out, summary)
        atomic_json(SELECTION, {**selection, "source_summary": args.out})
    elif args.stage == "final":
        sel = select([e for e in evals if e["config"]["gate"] == "ER" and e["config"]["direction_sign"] > 0])
        summary["final_selection"] = sel["config"] if sel else None
        if sel is not None:
            summary["selected_intervals"] = intervals(sel)
        atomic_json(ROOT / args.out, summary)
    else:
        atomic_json(ROOT / args.out, summary)
    print(json.dumps({"stage": args.stage, "selection": summary.get("selection") or summary.get("final_selection"),
                      "experiment_valid": ev_valid["valid"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
