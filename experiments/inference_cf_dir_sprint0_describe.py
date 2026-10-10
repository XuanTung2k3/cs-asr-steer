#!/usr/bin/env python
"""DIR-SPRINT0 post-terminal descriptive tables (CPU; non-decisional; written after the FULL audit).

Reads the sealed Job A / construct / Job B arrays, the evaluation and its local unit ledger. Nothing here changes a
count, predicate, family status or the terminal label; it only describes geometry and composition for the report:

* cosine of every steering direction with the D2 readout tangent at the same query (by evaluator stratum);
* D3 candidate composition (script class of c_AP, plausible-set size, null-floor hits) and its correction anatomy;
* D4 task-language entanglement (translate-branch top-1 script and English mass at the same query);
* D5 evidence statuses, concept evidence and tone-bearing share by stratum, prototype diagnostics;
* every corrected / severe-EOS query (exploratory; the counts are tiny).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, file_hash

RUN = ROOT / "results/inference_cf/dir_sprint0/run1"
POPULATION = ROOT / "docs/inference_cf/DIR_SPRINT0_POPULATION.json"
FAMS = ("D0", "D1", "D3", "D4", "D5", "D5SH", "VAC", "RND")


def unbf16(a):
    a = np.asarray(a, dtype=np.int16)
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if not x.size:
        return None
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.quantile(x, .1)),
            "p90": float(np.quantile(x, .9)), "mean_abs": float(np.abs(x).mean())}


def main() -> None:
    term = json.loads((RUN / "terminal.json").read_text())
    if term["audits"]["FULL"] != "DIR_SPRINT0_AUDIT_FULL: PASS":
        raise SystemExit("post-terminal only")
    ev = json.loads((RUN / "evaluation.json").read_text())
    units = json.loads((RUN / "evaluation_units.json").read_text())
    if file_hash(RUN / "evaluation_units.json") != ev["units_sha256"]:
        raise ValueError("unit ledger changed")
    rows = units["rows"]
    stratum = {(r["utterance_id"], r["t"]): r for r in rows}
    P = json.loads(POPULATION.read_text())
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    from transformers import WhisperTokenizer
    from csasr.inference_cf.core_r2 import tokenizer_partition
    tok = WhisperTokenizer.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    part = tokenizer_partition(tok)
    E, M = set(part["embedded_ids"]), set(part["matrix_ids"])

    def cls(i):
        return "EOS" if i == 50257 else "LATIN" if i in E else "HAN" if i in M else "OTHER"
    geo = defaultdict(lambda: defaultdict(list))
    d3 = {"c_class": Counter(), "c_class_by_stratum": defaultdict(Counter), "plausible": [], "A_c": [], "floor_hit": 0,
          "status_by_stratum": defaultdict(Counter)}
    tl = {"top1_class": Counter(), "P_E_TL": [], "P_E_B0": [], "top1_class_by_stratum": defaultdict(Counter), "raw_norm": []}
    d5 = {"status": Counter(), "q_by_stratum": defaultdict(list), "tone_by_stratum": defaultdict(list)}
    pev = json.loads((RUN / "construct/prospective_evidence.json").read_text())["rows"]
    for uid in [r["utterance_id"] for r in P["selected"]]:
        cap = json.loads((RUN / "capture" / f"{uid}.json").read_text())
        con = json.loads((RUN / "construct" / f"{uid}.json").read_text())
        pul = json.loads((RUN / "pulses" / f"{uid}.json").read_text())
        with np.load(RUN / "construct" / con["arrays"]["file"]) as z:
            dirs = {k: z[k] for k in z.files}
        with np.load(RUN / "pulses" / pul["arrays"]["file"]) as z:
            d2 = z["d2_direction"]
            d3d = z["d3_direction"]
        st = {s["t"]: s for s in con["status"]}
        for k, t in enumerate(con["structural_ts"]):
            q = cap["queries"][cap["inventory"].index(t)]
            row = stratum.get((uid, t))
            s = row["stratum"] if row else "unmapped"
            v2 = d2[k].astype(np.float64)
            if not np.isnan(v2).any():
                for f in FAMS:
                    v = d3d[k] if f == "D3" else dirs[f][k]
                    if not np.isnan(v).any():
                        c = float(v.astype(np.float64) @ v2 / (np.linalg.norm(v) * np.linalg.norm(v2)))
                        geo[f][s].append(c)
                        geo[f]["ALL"].append(c)
            dd = q["D3"]
            d3["status_by_stratum"][s][dd["status"]] += 1
            if dd["status"] in ("candidate", "baseline_is_candidate"):
                d3["plausible"].append(dd["plausible_count"])
            if dd["status"] == "candidate":
                cc = cls(dd["c_AP"])
                d3["c_class"][cc] += 1
                d3["c_class_by_stratum"][s][cc] += 1
                d3["A_c"].append(dd["A_c"])
                d3["floor_hit"] += int(dd["null_floor_hit_c"])
            tq = q["branches"]["TL"]
            if tq.get("top1") is not None:
                tl["top1_class"][cls(tq["top1"])] += 1
                tl["top1_class_by_stratum"][s][cls(tq["top1"])] += 1
                tl["P_E_TL"].append(tq.get("P_E"))
                tl["P_E_B0"].append(q["gate"]["baseline"]["P_E"] if q["gate"].get("baseline") else None)
            tl["raw_norm"].append(st[t]["D4"]["raw_norm"])
            e = pev[uid][str(t)]
            d5["status"][e["status"]] += 1
            if e["status"] == "ok":
                d5["q_by_stratum"][s].append(e["q"])
                d5["tone_by_stratum"][s].append(e.get("tone_bearing_share"))
    out = {"schema": "dir_sprint0_descriptive_v1", "non_decisional": True, "terminal_label": term["terminal_label"],
           "cos_with_D2": {f: {s: stats(v) for s, v in by.items()} for f, by in geo.items()},
           "D3": {"c_class": dict(d3["c_class"]), "c_class_by_stratum": {s: dict(c) for s, c in d3["c_class_by_stratum"].items()},
                  "status_by_stratum": {s: dict(c) for s, c in d3["status_by_stratum"].items()},
                  "plausible_count": stats(d3["plausible"]), "A_c": stats(d3["A_c"]), "null_floor_hits_at_candidate": d3["floor_hit"]},
           "D4_task_entanglement": {"translate_top1_class": dict(tl["top1_class"]),
                                    "translate_top1_class_by_stratum": {s: dict(c) for s, c in tl["top1_class_by_stratum"].items()},
                                    "P_E_translate": stats(tl["P_E_TL"]), "P_E_transcribe_B0": stats(tl["P_E_B0"]),
                                    "raw_norm_TR_minus_TL": stats(tl["raw_norm"])},
           "D5": {"evidence_status": dict(d5["status"]),
                  "q_mean_by_stratum": {s: np.asarray(v).mean(axis=0).tolist() for s, v in d5["q_by_stratum"].items()},
                  "tone_bearing_share_by_stratum": {s: stats(v) for s, v in d5["tone_by_stratum"].items()},
                  "concepts": ["NAS", "STOP", "FRIC", "LAB", "COR", "DOR"]}}
    corr, eos = [], []
    for r in rows:
        for a, x in r["arms"].items():
            if x["correction"]:
                corr.append({"arm": a, "utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "t": r["t"],
                             "b0_top1": r["b0_top1"], "b0_text": tok.decode([r["b0_top1"]]), "arm_top1": x["top1"],
                             "arm_text": tok.decode([x["top1"]])})
    for uid in [r["utterance_id"] for r in P["selected"]]:
        cap = json.loads((RUN / "capture" / f"{uid}.json").read_text())
        pul = json.loads((RUN / "pulses" / f"{uid}.json").read_text())
        T = len(cap["baseline"]["content_ids"])
        for x in pul["queries"]:
            for a in pul["arm_order"]:
                if x["arms"][a]["top1"] == 50257 and x["arms"]["B0"]["top1"] != 50257 and T - x["t"] >= 10:
                    eos.append({"arm": a, "utterance_id": uid, "dialogue_id": dlg[uid], "t": x["t"], "remaining": T - x["t"]})
    out["corrections"] = corr
    out["severe_EOS_events"] = eos
    out["severe_EOS_by_arm"] = dict(Counter(e["arm"] for e in eos))
    out["per_dialogue_corrections"] = {a: {d: v[f"{a}_C"] for d, v in ev["per_dialogue"].items() if v[f"{a}_C"]} for a in ev["arm_metrics"]}
    out["per_dialogue_ZH_corruptions"] = {a: {d: v[f"{a}_H_ZH"] for d, v in ev["per_dialogue"].items() if v[f"{a}_H_ZH"]} for a in ev["arm_metrics"]}
    out["common_eligible"] = ev["common_eligible"]
    atomic_json(RUN / "descriptive.json", out)
    print(json.dumps({"cos_with_D2_ALL": {f: (v.get("ALL") or {}).get("mean") for f, v in out["cos_with_D2"].items()},
                      "cos_with_D2_ENconf": {f: (v.get("EN-confusion") or {}).get("mean") for f, v in out["cos_with_D2"].items()},
                      "D3_c_class": out["D3"]["c_class"], "D3_c_class_by_stratum": out["D3"]["c_class_by_stratum"],
                      "TL_top1": out["D4_task_entanglement"]["translate_top1_class"],
                      "corrections": [(c["arm"], c["dialogue_id"], c["b0_text"], c["arm_text"]) for c in corr],
                      "EOS": out["severe_EOS_by_arm"]}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
