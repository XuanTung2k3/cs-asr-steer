#!/usr/bin/env python
"""P2-1 full report and terminal verdict (CPU). Rules: docs/reft/P2_1_DECISION_RULES.md (fixed before submission).

Reads the frozen P2-1 runs (p2_1/R2, p2_1/L1), their selections, the evaluation and reload-verification records, scores
every epoch (0-3) and regime (beam-5 primary, greedy secondary) against the pinned canonical P2 baseline B0, and writes
p2_1/report.json. dev150 is exposed exploratory development data (also used for checkpoint selection).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, fillers, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import git_state, now, sha  # noqa: E402

METHODS = ("R2", "L1")
RULES = {"cs_gain": -0.005, "retention_min": 0.99, "zh_cer_tol": 0.002, "loop_tol": 1, "min_dialogues": 4,
         "max_share": 0.40, "min_gross_corrections": 10}


# ====================================================================== decision rule (unit-tested)

def preservation_ok(a: dict) -> bool:
    return (a["mandarin_retention"] >= RULES["retention_min"] and a["zh_cer"] <= a["zh_cer_b0"] + RULES["zh_cer_tol"]
            and a["loops"] <= a["loops_b0"] + RULES["loop_tol"])


def useful(a: dict) -> bool:
    return (a["d_cs_mer"] <= RULES["cs_gain"] and a["d_cs_mer_ci"][1] < 0 and a["d_mer"] <= 0 and a["net"] > 0
            and a["dialogues_with_corrections"] >= RULES["min_dialogues"]
            and (a["max_share"] if a["max_share"] is not None else 1.0) <= RULES["max_share"] and preservation_ok(a))


def correction_but_damage(a: dict) -> bool:
    evidence = a["corrections"] >= RULES["min_gross_corrections"] or a["d_cs_mer"] <= RULES["cs_gain"]
    return evidence and (not preservation_ok(a) or a["d_mer"] > 0)


def uncertain_gain(a: dict) -> bool:
    return a["d_cs_mer"] <= RULES["cs_gain"] and a["d_cs_mer_ci"][1] >= 0 and preservation_ok(a)


def decide(integrity_ok: bool, arms: dict) -> str:
    if not integrity_ok:
        return "P2_1_ENGINEERING_BLOCKED"
    if useful(arms["R2"]):
        return "P2_1_REPRESENTATION_FEASIBLE"
    if useful(arms["L1"]):
        return "P2_1_WEIGHT_PEFT_ONLY"
    if correction_but_damage(arms["R2"]) or correction_but_damage(arms["L1"]):
        return "P2_1_CORRECTION_PRESERVATION_FAILURE"
    if uncertain_gain(arms["R2"]) or uncertain_gain(arms["L1"]):
        return "P2_1_INCONCLUSIVE"
    return "P2_1_NO_FREE_DECODING_GAIN"


# ====================================================================== report

def per_dialogue(base, sysr):
    d = collections.defaultdict(lambda: {"b0_errors": 0, "sys_errors": 0, "ref_units": 0, "corr": 0, "corrupt": 0})
    for b, s in zip(base, sysr):
        x = d[b["dialogue"]]
        x["b0_errors"] += b["counts"][2]
        x["sys_errors"] += s["counts"][2]
        x["ref_units"] += b["counts"][3]
        x["corr"] += sum(1 for p, q in zip(b["sub_en_ok"], s["sub_en_ok"]) if not p and q)
        x["corrupt"] += sum(1 for p, q in zip(b["sub_en_ok"], s["sub_en_ok"]) if p and not q)
    rows = {k: {**v, "delta_errors": v["sys_errors"] - v["b0_errors"]} for k, v in sorted(d.items())}
    vals = [v["delta_errors"] for v in rows.values()]
    return {"improved": sum(v < 0 for v in vals), "worsened": sum(v > 0 for v in vals), "tied": sum(v == 0 for v in vals),
            "largest_improvement_share": (min(vals) / sum(v for v in vals if v < 0)) if any(v < 0 for v in vals) else None,
            "rows_anonymised": [{"delta_errors": v["delta_errors"], "corr": v["corr"], "corrupt": v["corrupt"]}
                                for v in sorted(rows.values(), key=lambda v: v["delta_errors"])]}


def run():
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    C = p21()
    BS = C["evaluation"]["bootstrap"]
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    LU = C["evaluation"]["loop_units"]
    pb = C["frozen_baseline"]
    integ = {"pinned_baseline_hashes": all(sha(art(pb[r])) == pb[f"{r}_sha256"] for r in ("beam5", "greedy"))}
    base = {r: score_system(meta, refs, {x["id"]: x for x in read_jsonl(art(pb[r]))}, FILL, LU) for r in ("beam5", "greedy")}
    rep = {"B0": {r: metrics(base[r]) for r in base}, "arms": {}}
    ver = read_json(art("p2_1/verify.json")) if art("p2_1/verify.json").exists() else None
    integ["verify_present_and_ok"] = bool(ver and ver["all_ok"])
    integ["evaluation_present"] = art("p2_1/evaluation.json").exists()
    arms, scored_sel = {}, {}
    for m in METHODS:
        R = art(f"p2_1/{m}")
        man = read_json(R / "manifest.json") if (R / "manifest.json").exists() else None
        sel = read_json(R / "selection.json") if (R / "selection.json").exists() else None
        integ[f"{m}_manifest"] = man is not None
        integ[f"{m}_selection"] = sel is not None
        if man is None or sel is None:
            continue
        integ[f"{m}_weights_unchanged"] = man["weights_unchanged"]
        integ[f"{m}_zero_init_identity"] = man["history"][0]["zero_init_identity_beam5"] == len(meta)
        ep_sel = sel["selected"]["epoch"]
        epochs = {}
        for ep in range(0, C["training"]["epochs"] + 1):
            epochs[ep] = {}
            for r in ("beam5", "greedy"):
                sc = score_system(meta, refs, {x["id"]: x for x in read_jsonl(R / f"epoch{ep}" / f"dev_{r}.jsonl")}, FILL, LU)
                mm = metrics(sc)
                pp = paired(base[r], sc, BS["reps"], BS["seed"])
                epochs[ep][r] = {"metrics": mm, "vs_B0": pp, "dialogues": per_dialogue(base[r], sc)}
                if ep == ep_sel and r == "beam5":
                    scored_sel[m] = sc
        s = epochs[ep_sel]["beam5"]
        b0 = rep["B0"]["beam5"]
        arms[m] = {"d_mer": s["vs_B0"]["delta_mer"]["point"], "d_mer_ci": s["vs_B0"]["delta_mer"]["ci95"],
                   "d_cs_mer": s["vs_B0"]["delta_cs_mer"]["point"], "d_cs_mer_ci": s["vs_B0"]["delta_cs_mer"]["ci95"],
                   "net": s["vs_B0"]["net"], "corrections": s["vs_B0"]["substantive_en_corrections"],
                   "corruptions": s["vs_B0"]["substantive_en_corruptions"],
                   "dialogues_with_corrections": s["vs_B0"]["dialogues_with_corrections"],
                   "max_share": s["vs_B0"]["max_utterance_share_of_net"], "mandarin_retention": s["vs_B0"]["mandarin_retention"],
                   "zh_cer": s["metrics"]["overall"]["zh_cer"], "zh_cer_b0": b0["overall"]["zh_cer"],
                   "loops": s["metrics"]["overall"]["loop_utterances"], "loops_b0": b0["overall"]["loop_utterances"]}
        rep["arms"][m] = {"selected_epoch": ep_sel, "selection": sel, "epochs": epochs, "history": man["history"],
                          "efficiency": man["efficiency"], "decision_inputs": arms[m],
                          "useful": useful(arms[m]), "preservation_ok": preservation_ok(arms[m]),
                          "correction_but_damage": correction_but_damage(arms[m]), "uncertain_gain": uncertain_gain(arms[m])}
    if set(scored_sel) == set(METHODS):
        rep["R2_minus_L1_selected_beam5"] = paired(scored_sel["L1"], scored_sel["R2"], BS["reps"], BS["seed"])
    integrity_ok = all(integ.values()) and set(arms) == set(METHODS)
    verdict = decide(integrity_ok, arms) if integrity_ok else "P2_1_ENGINEERING_BLOCKED"
    out = {"schema": "reft_p2_1_report_v1", "utc": now(), "git": git_state(), "verdict": verdict, "integrity": integ,
           "rules": RULES, "rules_doc_sha256": sha(ROOT / "docs/reft/P2_1_DECISION_RULES.md"), "verify": ver, **rep,
           "note": "dev150 is exposed exploratory development data used for checkpoint selection; no untouched-test claim"}
    write_json(art("p2_1/report.json"), out)
    print(json.dumps({"verdict": verdict, "integrity": integ, "decision_inputs": arms}, indent=1, default=str))


if __name__ == "__main__":
    run()
