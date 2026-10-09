#!/usr/bin/env python
"""SRD2-G0 descriptive tables for the final report (CPU, post-terminal, descriptive only).

Reads the sealed gate/pulse rows, the committed evaluation and the FULL audit; writes
``descriptive.json``. Nothing here enters a predicate, label, threshold or selection: the terminal label is
fixed by ``audit_FULL.json``/``terminal.json``. No reference text is read (only the evaluator's per-query
stratum rows, which carry no surfaces).
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

RUN = ROOT / "results/inference_cf/srd2_g0/run1"
ARMS = ("B1", "B2", "B3")
EOS = 50257


def qs(xs):
    a = np.asarray(xs, dtype=np.float64)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()), "p25": float(np.quantile(a, .25)),
            "median": float(np.quantile(a, .5)), "p75": float(np.quantile(a, .75)), "p95": float(np.quantile(a, .95)),
            "max": float(a.max())}


def main() -> None:
    term = json.loads((RUN / "terminal.json").read_text())
    ev = json.loads((RUN / "evaluation.json").read_text())
    full = json.loads((RUN / "audit_FULL.json").read_text())
    seal = json.loads((RUN / "pulse_seal.json").read_text())
    gseal = json.loads((RUN / "gate_seal.json").read_text())
    pop = json.loads((ROOT / "docs/inference_cf/SRD2_G0_POPULATION.json").read_text())["selected"]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in pop}
    gate, pulse = {}, {}
    for uid in dlg:
        for name, store, s in (("gate", gate, gseal), ("pulses", pulse, seal)):
            rel = f"{name}/{uid}.json"
            if file_hash(RUN / rel) != s["files"][rel]:
                raise ValueError(f"sealed row changed {rel}")
            store[uid] = json.loads((RUN / rel).read_text())
    inv = [(u, q) for u, g in gate.items() for q in g["queries"]]
    struct = [(u, q) for u, q in inv if q["structural"]["eligible"]]
    out = {"terminal": term, "inventory_queries": len(inv), "structural_queries": len(struct),
           "rows_terminated": dict(Counter(g["baseline"]["terminated"] for g in gate.values())),
           "content_tokens": qs([g["baseline"]["T"] for g in gate.values()]),
           "structural_ineligibility_reasons": dict(Counter(r for _, q in inv for r in q["structural"]["reasons"])),
           "gate_fallbacks_all_queries": dict(Counter(str(q["gate"]["fallback_reason"]) for _, q in inv)),
           "gate_fallbacks_structural": dict(Counter(str(q["gate"]["fallback_reason"]) for _, q in struct)),
           "lid_calls": sum(g["lid"]["calls"] for g in gate.values()),
           "lid_cache_hits": sum(g["lid"]["cache_hits"] for g in gate.values())}
    E = [q["gate"]["E"] for _, q in struct if q["gate"]["E"] is not None]
    Rb = [q["gate"]["R_B"] for _, q in struct if q["gate"]["R_B"] is not None]
    g = [q["g"] for _, q in struct]
    out["gate_distribution_structural"] = {"E": qs(E), "E_positive_fraction": float(np.mean(np.asarray(E) > 0)),
                                           "R_B": qs(Rb), "R_B_positive_fraction": float(np.mean(np.asarray(Rb) > 0)),
                                           "g": qs(g), "g_positive": int(sum(x > 0 for x in g)),
                                           "g_positive_fraction": float(np.mean(np.asarray(g) > 0)),
                                           "g_nonzero": qs([x for x in g if x > 0])}
    per_d = defaultdict(list)
    for u, q in struct:
        per_d[dlg[u]].append(q["g"])
    out["gate_per_dialogue"] = {d: {"structural": len(v), "g_positive": int(sum(x > 0 for x in v)),
                                    "mean_g": float(np.mean(v)), "planned_sq_B2": float(sum((1.1260757575454359 * x) ** 2 for x in v))}
                                for d, v in sorted(per_d.items())}
    # ---- gate exposure by evaluator stratum (descriptive selectivity of the detector itself) ----------------------
    gq = {(u, q["t"]): q for u, gg in gate.items() for q in gg["queries"]}
    pq = {(u, q["t"]): q for u, pp in pulse.items() for q in pp["queries"]}
    by_s = defaultdict(list)
    for r in ev["rows"]:
        by_s[r["stratum"]].append((r["utterance_id"], r["t"]))
    strat = {}
    for s, keys in sorted(by_s.items()):
        st = [k for k in keys if gq[k]["structural"]["eligible"]]
        gv = [gq[k]["g"] for k in st]
        strat[s] = {"mapped": len(keys), "structural": len(st), "g_positive": int(sum(x > 0 for x in gv)),
                    "g_positive_fraction": float(np.mean(np.asarray(gv) > 0)) if gv else None, "g": qs(gv),
                    "E_positive_fraction": float(np.mean([(gq[k]["gate"]["E"] or 0) > 0 for k in st])) if st else None,
                    "executed": {a: int(sum(pq[k]["arms"][a]["executed"] for k in st)) for a in ARMS},
                    "executed_fraction_of_mapped": {a: (sum(pq[k]["arms"][a]["executed"] for k in st) / len(keys)) if keys else None for a in ARMS},
                    "top1_changed": {a: int(sum(pq[k]["arms"][a]["top1"] != pq[k]["arms"]["B0"]["top1"] for k in keys)) for a in ARMS}}
    out["stratum_exposure"] = strat
    # ---- events -----------------------------------------------------------------------------------------
    rows = ev["rows"]
    out["events"] = {a: {"corrections": [{"utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "t": r["t"],
                                         "g": gq[(r["utterance_id"], r["t"])]["g"]} for r in rows if r["arms"][a]["correction"]],
                        "ZH_corruptions_by_dialogue": dict(Counter(r["dialogue_id"] for r in rows if r["stratum"] == "ZH-correct" and r["arms"][a]["corruption"])),
                        "ZH_corruption_g": qs([gq[(r["utterance_id"], r["t"])]["g"] for r in rows if r["stratum"] == "ZH-correct" and r["arms"][a]["corruption"]]),
                        "EN_corruptions": [{"utterance_id": r["utterance_id"], "t": r["t"]} for r in rows if r["stratum"] == "EN-correct" and r["arms"][a]["corruption"]],
                        "active_ZH_corruption_rate": (sum(r["arms"][a]["corruption"] for r in rows if r["stratum"] == "ZH-correct") /
                                                      max(1, sum(r["arms"][a]["executed"] for r in rows if r["stratum"] == "ZH-correct")))}
                     for a in ARMS}
    # ---- EOS risk proxy (reference-free; all inventory queries) ----------------------------------------------------
    eos = {}
    for a in ARMS:
        ev_ = []
        for u, pp in pulse.items():
            T = len(gate[u]["baseline"]["content_ids"])
            for q in pp["queries"]:
                if q["arms"][a]["top1"] == EOS and q["arms"]["B0"]["top1"] != EOS:
                    ev_.append({"utterance_id": u, "dialogue_id": dlg[u], "t": q["t"], "remaining_content_tokens": T - q["t"],
                                "g": gq[(u, q["t"])]["g"], "target": q["arms"][a]["target"]})
        eos[a] = {"new_EOS_any": len(ev_), "new_EOS_remaining_ge10": [e for e in ev_ if e["remaining_content_tokens"] >= 10],
                  "remaining_tokens": qs([e["remaining_content_tokens"] for e in ev_])}
    out["eos_proxy"] = eos
    # ---- dose ------------------------------------------------------------------------------------------
    dose = {}
    for a in ARMS:
        cells = [q["arms"][a] for _, pp in pulse.items() for q in pp["queries"] if q["structural"]]
        dose[a] = {"status": dict(Counter(c["status"] for c in cells)),
                   "planned_target": qs([c["target"] for c in cells if c["target"] > 0]),
                   "realized_consumed_chord": qs([c["consumed_chord_actual"] for c in cells if c["executed"]]),
                   "unattainable_targets": qs([c["target"] for c in cells if c["status"] in ("solver_target_unattainable", "native_consumption_unattainable")])}
    out["dose"] = dose
    out["top1_changes_all_queries"] = {a: int(sum(q["arms"][a]["top1"] != q["arms"]["B0"]["top1"] for pp in pulse.values() for q in pp["queries"]))
                                       for a in ARMS}
    out["d2_status"] = dict(Counter(q["d2"]["status"] for pp in pulse.values() for q in pp["queries"] if q["structural"]))
    out["timing"] = {"d2_sec": qs([q["d2"]["runtime_sec"] for pp in pulse.values() for q in pp["queries"] if q["structural"]]),
                     "pulse_sec": qs([q["arms"][a]["pulse_sec"] for pp in pulse.values() for q in pp["queries"] if q["structural"]
                                      for a in ARMS if q["arms"][a]["executed"]])}
    out["bootstrap"] = ev["bootstrap"]
    out["full_audit"] = {"verdict": full["verdict"], "label": full["terminal_label"]}
    out["note"] = "descriptive, post-terminal; no element of this file enters any predicate, label or selection"
    atomic_json(RUN / "descriptive.json", out)
    print(json.dumps({k: out[k] for k in ("stratum_exposure", "eos_proxy", "top1_changes_all_queries")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
