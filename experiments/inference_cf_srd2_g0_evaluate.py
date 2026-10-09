#!/usr/bin/env python
"""SRD2-G0 post-seal evaluator (CPU; separate process; never imported by the runner).

Opens the selected-role reference column ONLY after the pushed pulse seal and a pushed
``SRD2_G0_AUDIT_PRIMARY: PASS``, filtering the role parquet to the frozen 400 IDs before the
reference column is materialized. Maps reference units with the canonical normalize/segment/align
and R2 ``unit_to_token_positions`` / ``evaluator_labels`` (no CTC/MMS-FA timing), builds the
historical P2-R English ``target_set`` and the singleton Mandarin baseline target, counts actual
first-token corrections/corruptions per arm from the sealed processed top-1 decisions, computes every
frozen opportunity/power/control/dose/acceptance/severe-damage predicate, the paired dialogue
bootstrap, and the first-matching provisional terminal label (confirmed only by the independent FULL
audit). No arm, comparator, metric, threshold or position is chosen from outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

CONFIG = "configs/inference_cf/srd2_g0.json"
CONFIG_SHA = "sha256:218b3be997e8fafc102f7ec866b0a82b9925bb180e671f714c651bddee392eb9"
POPULATION = "docs/inference_cf/SRD2_G0_POPULATION.json"
RUN = "results/inference_cf/srd2_g0/run1"
REMOTE = "origin/cs-asr-steer-inf"
MODEL = "/mnt/data/tungnx/whisper-large-v3"
EOS = 50257
ARMS = ("B1", "B2", "B3")
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
PROVIDER = ("localizer_fail", "local_support_fail", "baseline_provider_fail", "nonfinite_signal")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def pushed(rel: str) -> bool:
    import hashlib
    if git("ls-files", rel) != rel:
        return False
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    if "sha256:" + hashlib.sha256(blob).hexdigest() != file_hash(ROOT / rel):
        return False
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


# ---- references (selected 400 only) -----------------------------------------------------------------------

def load_references(ids: list[str]) -> tuple[dict, dict]:
    """Filter the role parquet to the frozen IDs BEFORE the reference column is materialized."""
    import pyarrow.dataset as ds
    P = json.loads((ROOT / POPULATION).read_text())
    role = P["source"]
    if file_hash(role) != P["source_sha256"]:
        raise ValueError("role parquet changed")
    dataset = ds.dataset(role, format="parquet")
    table = dataset.to_table(columns=["utterance_id", "role", "transcript_raw"], filter=ds.field("utterance_id").isin(ids))
    rows = table.to_pylist()
    if sorted(r["utterance_id"] for r in rows) != sorted(ids) or any(r["role"] != "D-dev-select" for r in rows):
        raise ValueError("reference rows differ from the frozen selection")
    access = {"path": role, "sha256": P["source_sha256"], "columns": ["utterance_id", "role", "transcript_raw"],
              "filter": "utterance_id in frozen selected400 (pushdown before materialization)", "rows": len(rows)}
    return {r["utterance_id"]: r["transcript_raw"] for r in rows}, access


# ---- mapping --------------------------------------------------------------------------------------------

def latin_mapping(ref: str, hyp: str, positions: list, eos_slot) -> tuple[int, int]:
    """English-unit mapping coverage (pre-registered reading of the frozen denominator): EVERY normalized
    Latin reference unit; mapped iff its canonical alignment op yields a query -- MATCH/SUB -> first generated
    token of the aligned hypothesis unit (R2 unit_to_token_positions), DEL -> the query of the next aligned
    hypothesis unit, else the EOS slot -- with no missing offset or capped gap. Before target-set/stratum filters."""
    from csasr.data.normalize import normalize_text, segment_units
    from csasr.evaluation.mer import DEL, MATCH, SUB, align_tokens
    ru = segment_units(normalize_text(ref))
    hu = segment_units(normalize_text(hyp))
    ops = align_tokens([u.surface for u in ru], [u.surface for u in hu])
    total = mapped = 0
    for i, op in enumerate(ops):
        if op.ref_idx is None or ru[op.ref_idx].kind != "latin":
            continue
        total += 1
        pos = None
        if op.op in (MATCH, SUB) and op.hyp_idx is not None:
            pos = positions[op.hyp_idx]
        elif op.op == DEL:
            nxt = next((o.hyp_idx for o in ops[i + 1:] if o.ref_idx is not None and o.hyp_idx is not None), None)
            pos = positions[nxt] if nxt is not None else eos_slot
        mapped += int(pos is not None)
    return total, mapped


def map_utterance(tok, uid: str, ref: str, content: list[int], terminated: str, duration: float) -> tuple[list[dict], dict]:
    """Reference units -> unique (UID,t) queries with stratum and acceptable first-token set Y."""
    from csasr.data.normalize import normalize_text, segment_units
    from experiments.inference_cf_p0_r2_evaluate import evaluator_labels, unit_to_token_positions
    from experiments.inference_cf_p2r_population import raw_latin_surfaces, target_set
    hyp = tok.decode(content, skip_special_tokens=True)
    positions = unit_to_token_positions(tok, content, hyp)
    eos_slot = len(content) if terminated == "eos" else None
    units, _ = evaluator_labels(ref, hyp, positions, eos_slot=eos_slot, heard_sec=min(30.0, duration), ctc_by_index={})
    ref_units = segment_units(normalize_text(ref))
    raw = raw_latin_surfaces(ref, ref_units)
    heard_scope = duration > 30.0
    latin_total, latin_mapped = latin_mapping(ref, hyp, positions, eos_slot)
    if heard_scope:
        latin_mapped = 0
    for u in units:
        if heard_scope:
            u["position"], u["reason"] = None, "unalignable:heard_scope"
    # multiple English units sharing one query -> excluded from primary contrasts
    en_at = Counter(u["position"] for u in units if u["language"] == "EN" and u["position"] is not None and u["reason"] is None)
    for u in units:
        if u["language"] == "EN" and u["position"] is not None and u["reason"] is None and en_at[u["position"]] > 1:
            u["candidate_position"], u["position"], u["reason"] = u["position"], None, "unalignable:multiple_english_units"
    queries = {}
    for u in units:
        s, t = u["stratum"], u["position"]
        if u["reason"] is not None or t is None or s not in STRATA:
            continue
        if s == "ZH-correct":
            if t >= len(content):
                u["reason"] = "unalignable:zh_position_beyond_content"
                continue
            if t in queries:                      # same-category Mandarin deduplicated to one query
                u["deduplicated_into"] = t
                continue
            queries[t] = {"utterance_id": uid, "t": t, "stratum": s, "Y": [int(content[t])], "units": [u["reference_unit_index"]]}
            continue
        Y, why = target_set(tok, content[:t], u["surface"], raw.get(u["reference_unit_index"]))
        if not Y:
            u["reason"] = f"unalignable:target_set:{why}"
            continue
        b0 = int(content[t]) if t < len(content) else EOS
        if s == "EN-confusion" and b0 in Y:
            u["reason"] = "unalignable:target_set_baseline_in_set"
            continue
        if s == "EN-correct" and b0 not in Y:
            u["reason"] = "unalignable:target_set_baseline_not_in_set"
            continue
        if t in queries:
            raise RuntimeError("English query collision escaped the multiple-unit rule")
        queries[t] = {"utterance_id": uid, "t": t, "stratum": s, "Y": sorted(int(y) for y in Y), "units": [u["reference_unit_index"]]}
    for u in units:
        u["utterance_id"] = uid
    return list(queries.values()), {"latin_units": latin_total, "latin_mapped": latin_mapped, "heard_scope": heard_scope,
                                    "units": units, "hypothesis_token_count": len(content)}


# ---- predicates -----------------------------------------------------------------------------------------

def predicate(rule: dict, metrics: dict) -> bool:
    import operator
    ops = {"==": operator.eq, "<=": operator.le, ">=": operator.ge, ">": operator.gt}
    if "all" in rule:
        return all(predicate(x, metrics) for x in rule["all"])
    if "any" in rule:
        return any(predicate(x, metrics) for x in rule["any"])
    v = metrics.get(rule["metric"])
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return False
    return bool(ops[rule["op"]](v, rule["value"]))


def leaf_report(rule: dict, metrics: dict) -> list[dict]:
    if "all" in rule or "any" in rule:
        return [x for c in rule.get("all", rule.get("any")) for x in leaf_report(c, metrics)]
    v = metrics.get(rule["metric"])
    status = "NOT_ESTIMABLE" if v is None or (isinstance(v, float) and not math.isfinite(v)) else \
        ("PASS" if predicate(rule, metrics) else "FAIL")
    return [{"metric": rule["metric"], "op": rule["op"], "threshold": rule["value"], "value": v, "status": status}]


def condition(rule: dict, gates: dict, flags: dict) -> bool:
    if "all" in rule:
        return all(condition(x, gates, flags) for x in rule["all"])
    if "any" in rule:
        return any(condition(x, gates, flags) for x in rule["any"])
    if "flag" in rule:
        return bool(flags[rule["flag"]])
    return gates[rule["gate"]] is rule["is"]


def decide(cfg: dict, gates: dict, flags: dict) -> str:
    return next(r["label"] for r in cfg["terminal_rules"] if condition(r["when"], gates, flags))


def ratio(a, b):
    return None if b in (0, None) or a is None else a / b


# ---- reference-free integrity from the sealed pulse matrix -------------------------------------------------

def integrity(cfg: dict, pulses: dict, gate: dict, perm: dict) -> dict:
    e = float(cfg["dose"]["e_star"])
    ex = [(u, x, a, x["arms"][a]) for u, p in pulses.items() for x in p["queries"] if x["structural"] for a in ARMS
          if x["arms"][a]["executed"]]
    sq, ch, cp = [], [], []
    for _, x, a, r in ex:
        g = r["guards"]
        cons = r["consumed_chord_actual"]
        t2 = r["target"] ** 2
        sq += [g["proposed_rel_sq_err"], abs(cons * cons / t2 - 1.0)]
        ch += [g["proposed_rel_chord_err"], abs(cons / r["target"] - 1.0)]
        cp.append(abs(cons / g["proposed_chord"] - 1.0))
    struct = [(u, x) for u, p in pulses.items() for x in p["queries"] if x["structural"]]
    restore = [x["restore_bitwise"] for _, x in struct if x["restore_bitwise"] is not None]
    clean = [x["clean"]["logits_bitwise_vs_A"] and x["clean"]["site_bitwise_vs_A"]
             for p in pulses.values() for x in p["queries"]]
    scratch = [x["d2"]["scratch_logits_bitwise"] and x["d2"]["scratch_site_bitwise"] for _, x in struct]
    b1 = [x["arms"]["B1"] for _, x in struct]
    metrics = {"executed_relative_squared_energy_error_max": max(sq) if sq else None,
               "executed_relative_chord_error_max": max(ch) if ch else None,
               "consumed_proposed_relative_error_max": max(cp) if cp else None,
               "solver_hook_identity_pass": bool(ex) and all(r["integrity_ok"] for *_, r in ex),
               "zero_no_edit_restore_bitwise": bool(clean) and all(restore) and all(clean) and all(scratch),
               "B1_matched_fraction": (sum(r["executed"] for r in b1) / len(b1)) if b1 else None}
    planned = {a: [] for a in ("B2", "B3")}
    realized = {a: [] for a in ("B2", "B3")}
    lost = {a: [] for a in ("B2", "B3")}
    for _, x in struct:
        for a in ("B2", "B3"):
            r = x["arms"][a]
            planned[a].append(r["target"])
            if r["executed"]:
                realized[a].append(r["consumed_chord_actual"] ** 2)
            elif r["target"] > 0:
                lost[a].append(r["target"] ** 2)
    from csasr.inference_cf.core import digest as _d

    def bits(v):
        return np.float64(v).view(np.uint64).item().to_bytes(8, "big").hex()
    tot = {a: math.fsum(sorted(t * t for t in planned[a])) for a in planned}
    real = {a: math.fsum(sorted(realized[a])) for a in realized}
    dose = {"planned_B2_B3_multisets_equal": _d(sorted(bits(v) for v in planned["B2"])) == _d(sorted(bits(v) for v in planned["B3"]))
            and tot["B2"] == tot["B3"],
            "energy_ratio_B2_B3": ratio(real["B2"], real["B3"]) if real["B3"] > 0 else None,
            "lost_planned_energy_B2": ratio(math.fsum(sorted(lost["B2"])), tot["B2"]) if tot["B2"] > 0 else None,
            "lost_planned_energy_B3": ratio(math.fsum(sorted(lost["B3"])), tot["B3"]) if tot["B3"] > 0 else None}
    status = Counter((a, x["arms"][a]["status"]) for _, x in struct for a in ARMS)
    return {"energy_execution": metrics, "dose_comparison": dose, "planned_sq": tot, "realized_sq": real,
            "arm_status_counts": {f"{a}|{s}": n for (a, s), n in sorted(status.items())},
            "executed_cells": len(ex), "structural_queries": len(struct), "e_star": e}


# ---- outcomes -----------------------------------------------------------------------------------------

def outcomes(queries: list[dict], pulses: dict, dlg: dict) -> list[dict]:
    """Per mapped query and arm: top-1 decisions and correction/corruption events."""
    out = []
    for qd in queries:
        p = pulses[qd["utterance_id"]]
        rec = next(x for x in p["queries"] if x["t"] == qd["t"])
        b0 = rec["arms"]["B0"]["top1"]
        Y = set(qd["Y"])
        row = {"utterance_id": qd["utterance_id"], "dialogue_id": dlg[qd["utterance_id"]], "t": qd["t"],
               "stratum": qd["stratum"], "structural": rec["structural"], "b0_top1": b0, "b0_in_Y": b0 in Y, "arms": {}}
        for a in ARMS:
            r = rec["arms"][a]
            top = r["top1"]
            ev = {"top1": top, "executed": bool(r["executed"]), "status": r["status"], "in_Y": top in Y,
                  "correction": False, "corruption": False}
            if qd["stratum"] == "EN-confusion":
                ev["correction"] = (b0 not in Y) and (top in Y)
            else:
                ev["corruption"] = (b0 in Y) and (top not in Y)
            row["arms"][a] = ev
        out.append(row)
    return out


def tally(rows: list[dict], dialogues: list[str]) -> dict:
    """Per-arm, per-dialogue integer tallies (the unit of the paired cluster bootstrap)."""
    T = {d: {"n_C": 0, "n_EN": 0, "n_ZH": 0, **{f"{a}_{k}": 0 for a in ARMS for k in ("C", "H_EN", "H_ZH", "act_ZH")}}
         for d in dialogues}
    for r in rows:
        d = T[r["dialogue_id"]]
        key = {"EN-confusion": "n_C", "EN-correct": "n_EN", "ZH-correct": "n_ZH"}[r["stratum"]]
        d[key] += 1
        for a in ARMS:
            ev = r["arms"][a]
            if r["stratum"] == "EN-confusion":
                d[f"{a}_C"] += int(ev["correction"])
            elif r["stratum"] == "EN-correct":
                d[f"{a}_H_EN"] += int(ev["corruption"])
            else:
                d[f"{a}_H_ZH"] += int(ev["corruption"])
                d[f"{a}_act_ZH"] += int(ev["executed"])
    return T


def bootstrap(T: dict, dialogues: list[str], cfg: dict) -> dict:
    st = cfg["statistics"]
    rng = np.random.default_rng(st["seed"])
    D = len(dialogues)
    draws = rng.integers(0, D, size=(st["bootstrap_replicates"], D))
    keys = sorted(next(iter(T.values())).keys())
    M = np.array([[T[d][k] for k in keys] for d in dialogues], dtype=np.float64)
    S = M[draws].sum(axis=1)                       # (R, K) resampled totals
    col = {k: S[:, i] for i, k in enumerate(keys)}
    U = {a: col[f"{a}_C"] - col[f"{a}_H_ZH"] - col[f"{a}_H_EN"] for a in ARMS}
    fam = {"C_B2-C_B3": col["B2_C"] - col["B3_C"], "U_B2-U_B3": U["B2"] - U["B3"],
           "H_ZH_B3-H_ZH_B2": col["B3_H_ZH"] - col["B2_H_ZH"], "H_ZH_B1-H_ZH_B2": col["B1_H_ZH"] - col["B2_H_ZH"]}
    lo, hi = st["family_quantiles"]

    def iv(x, a=.025, b=.975):
        return [float(np.quantile(x, a)), float(np.quantile(x, b))]
    out = {"replicates": int(st["bootstrap_replicates"]), "seed": st["seed"], "draw_method": "numpy default_rng(seed).integers(0, 20, size=(10000, 20)) over sorted dialogues",
           "family": {k: {"pct95": iv(v), "bonferroni_98.75": iv(v, lo, hi)} for k, v in fam.items()},
           "counts_pct95": {f"{k}": iv(col[k]) for k in keys if k[:2] in ("B1", "B2", "B3")},
           "utility_pct95": {a: iv(U[a]) for a in ARMS}}
    rates = {}
    for a in ARMS:
        for name, num, den in (("ZH_corruption_rate", f"{a}_H_ZH", "n_ZH"), ("EN_corruption_rate", f"{a}_H_EN", "n_EN"),
                               ("correction_rate", f"{a}_C", "n_C"), ("ZH_active_corruption_rate", f"{a}_H_ZH", f"{a}_act_ZH")):
            dv = col[den]
            ok = dv > 0
            n_ok = int(ok.sum())
            rates[f"{name}_{a}"] = {"valid_draws": n_ok,
                                    "pct95": iv(col[num][ok] / dv[ok]) if n_ok >= 9500 else "NOT_ESTIMABLE"}
    out["rates"] = rates
    return out


def evaluate(cfg: dict, pulses: dict, gate: dict, perm: dict, refs: dict, P: dict, tok, side: dict) -> dict:
    """Pure core: ``side`` carries the sealed gate seal, resource forecast, pulse runtime and apparatus records."""
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    dur = {r["utterance_id"]: float(r["source_duration_sec"]) for r in P["selected"]}
    dialogues = sorted(set(dlg.values()))
    all_q, ledgers, units_all = [], {}, []
    for uid in [r["utterance_id"] for r in P["selected"]]:
        g = gate[uid]
        qs, led = map_utterance(tok, uid, refs[uid], g["baseline"]["content_ids"], g["baseline"]["terminated"], dur[uid])
        all_q += qs
        units_all += led.pop("units")
        ledgers[uid] = led
    rows = outcomes(all_q, pulses, dlg)
    T = tally(rows, dialogues)

    def tot(k):
        return sum(T[d][k] for d in dialogues)

    def dcount(pred):
        return sum(1 for d in dialogues if pred(T[d]))
    C = {a: tot(f"{a}_C") for a in ARMS}
    HZ = {a: tot(f"{a}_H_ZH") for a in ARMS}
    HE = {a: tot(f"{a}_H_EN") for a in ARMS}
    U = {a: C[a] - HZ[a] - HE[a] for a in ARMS}
    nC, nEN, nZH = tot("n_C"), tot("n_EN"), tot("n_ZH")
    act = {a: tot(f"{a}_act_ZH") for a in ARMS}
    gain_c = {d: T[d]["B2_C"] - T[d]["B3_C"] for d in dialogues}
    gain_u = {d: (T[d]["B2_C"] - T[d]["B2_H_ZH"] - T[d]["B2_H_EN"]) - (T[d]["B3_C"] - T[d]["B3_H_ZH"] - T[d]["B3_H_EN"]) for d in dialogues}
    pos_u = [v for v in gain_u.values() if v > 0]
    lat_total = sum(l["latin_units"] for l in ledgers.values())
    lat_mapped = sum(l["latin_mapped"] for l in ledgers.values())
    integ = integrity(cfg, pulses, gate, perm)
    # reference-free single-query early-stop proxy (all inventory queries of all 400 rows)
    eos_proxy = {a: sum(1 for p in pulses.values() for x in p["queries"] if x["arms"][a]["top1"] == EOS
                        and x["arms"]["B0"]["top1"] != EOS and len(gate[p["utterance_id"]]["baseline"]["content_ids"]) - x["t"] >= 10)
                 for a in ARMS}
    dmg_dialogue = any(T[d]["B2_H_ZH"] >= 3 and T[d]["B2_act_ZH"] > 0 and T[d]["B2_H_ZH"] / T[d]["B2_act_ZH"] > .20
                       for d in dialogues)
    seal_a, fc, prt, app = side["gate_seal"], side["forecast"], side["pulse_runtime"], side["apparatus"]
    critical = {"job_A_critical": all(seal_a["critical"].values()), "job_B_completed": prt.get("status") == "completed",
                "job_B_weights_unchanged": prt.get("weights_unchanged") is True,
                "job_B_no_parameter_grads": prt.get("parameters_with_grad") == 0 and prt.get("parameters_requiring_grad") == 0,
                "apparatus_old30_ok": app.get("ok") is True,
                "all_rows_ok": all(p.get("status") == "ok" for p in pulses.values()) and len(pulses) == len(P["selected"])}
    m = {"mapped_EN_confusion": nC, "mapped_EN_confusion_dialogues": dcount(lambda x: x["n_C"] > 0),
         "mapped_EN_correct": nEN, "mapped_EN_correct_dialogues": dcount(lambda x: x["n_EN"] > 0),
         "mapped_ZH_correct": nZH, "mapped_ZH_correct_dialogues": dcount(lambda x: x["n_ZH"] > 0),
         "active_B2_ZH_correct": act["B2"], "active_B2_ZH_dialogues": dcount(lambda x: x["B2_act_ZH"] > 0),
         "English_mapping_fraction": ratio(lat_mapped, lat_total),
         "C_B2": C["B2"], "corrected_dialogues_B2": dcount(lambda x: x["B2_C"] > 0),
         "C_B1": C["B1"], "H_ZH_B1": HZ["B1"], "ZH_harmed_dialogues_B1": dcount(lambda x: x["B1_H_ZH"] > 0),
         "uninformative_query_fraction": ratio(perm["uninformative_queries"], perm["structural_queries"]),
         "correction_retention_B2_B1": ratio(C["B2"], C["B1"]), "ZH_harm_ratio_B2_B1": ratio(HZ["B2"], HZ["B1"]),
         "H_ZH_B2": HZ["B2"], "ZH_corruption_rate_B2": ratio(HZ["B2"], nZH),
         "ZH_active_corruption_rate_B2": ratio(HZ["B2"], act["B2"]), "H_EN_B2": HE["B2"],
         "EN_corruption_rate_B2": ratio(HE["B2"], nEN), "C_B2_minus_B3": C["B2"] - C["B3"], "U_B2_minus_B3": U["B2"] - U["B3"],
         "H_ZH_B2_minus_B3": HZ["B2"] - HZ["B3"],
         "positive_correction_gain_dialogues": sum(1 for v in gain_c.values() if v > 0),
         "positive_utility_gain_dialogues": len(pos_u),
         "max_positive_utility_dialogue_share": (max(pos_u) / sum(pos_u)) if pos_u else 1.0, "U_B2": U["B2"],
         "any_dialogue_ZH_damage_events_ge3_AND_active_rate_gt020": dmg_dialogue, "new_EOS_proxy_events_B2": eos_proxy["B2"],
         **integ["energy_execution"], **integ["dose_comparison"], **seal_a["metrics"], **fc["metrics"]}
    gates = {name: predicate(rule, m) for name, rule in cfg["gate_predicates"].items()}
    integer_forms = {"retention_integer_form": C["B2"] >= math.ceil(.6 * C["B1"]) if C["B1"] else None,
                     "harm_integer_form": HZ["B2"] <= math.floor(.5 * HZ["B1"]) if HZ["B1"] else None}
    return {"metrics": m, "gates": gates, "integer_forms": integer_forms, "critical": critical,
            "leaves": {name: leaf_report(rule, m) for name, rule in cfg["gate_predicates"].items()},
            "counts": {"C": C, "H_ZH": HZ, "H_EN": HE, "U": U, "active_ZH": act,
                       "correction_dialogues": {a: dcount(lambda x, a=a: x[f"{a}_C"] > 0) for a in ARMS},
                       "ZH_harm_dialogues": {a: dcount(lambda x, a=a: x[f"{a}_H_ZH"] > 0) for a in ARMS},
                       "EN_harm_dialogues": {a: dcount(lambda x, a=a: x[f"{a}_H_EN"] > 0) for a in ARMS},
                       "harm_dialogues": {a: dcount(lambda x, a=a: x[f"{a}_H_ZH"] + x[f"{a}_H_EN"] > 0) for a in ARMS},
                       "EOS_proxy": eos_proxy},
           "per_dialogue": T, "paired_gain": {"correction": gain_c, "utility": gain_u},
           "denominators": {"mapped": {"EN-confusion": nC, "EN-correct": nEN, "ZH-correct": nZH},
                            "structural_mapped": {s: sum(1 for r in rows if r["stratum"] == s and r["structural"]) for s in STRATA},
                            "latin_units": lat_total, "latin_mapped": lat_mapped,
                            "heard_scope_rows": sum(l["heard_scope"] for l in ledgers.values())},
           "unit_reasons": dict(Counter(f"{u['stratum']}|{u['reason']}" for u in units_all)),
           "integrity": integ, "bootstrap": bootstrap(T, dialogues, cfg), "rows": rows, "units": units_all}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUN)
    args = ap.parse_args()
    if args.out.rstrip("/") != RUN:
        raise SystemExit("only the frozen run root is allowed")
    run = ROOT / RUN
    if file_hash(ROOT / CONFIG) != CONFIG_SHA:
        raise SystemExit("config changed")
    cfg = json.loads((ROOT / CONFIG).read_text())
    for rel in ("pulse_seal.json", "audit_PRIMARY.json", "gate_seal.json", "permutation.json", "resource_forecast.json"):
        if not pushed(f"{RUN}/{rel}"):
            raise SystemExit(f"{rel} must be committed and pushed before references are opened")
    prim = json.loads((run / "audit_PRIMARY.json").read_text())
    if prim.get("verdict") != "SRD2_G0_AUDIT_PRIMARY: PASS":
        raise SystemExit("SRD2_G0_AUDIT_PRIMARY: PASS required before references")
    if (run / "evaluation.json").exists():
        raise FileExistsError("evaluation exists; never overwrite")
    pseal = json.loads((run / "pulse_seal.json").read_text())
    gseal = json.loads((run / "gate_seal.json").read_text())
    P = json.loads((ROOT / POPULATION).read_text())
    ids = [r["utterance_id"] for r in P["selected"]]
    gate, pulses = {}, {}
    for uid in ids:
        for name, store, seal in (("gate", gate, gseal), ("pulses", pulses, pseal)):
            rel = f"{name}/{uid}.json"
            if file_hash(run / rel) != seal["files"][rel]:
                raise ValueError(f"sealed row changed {rel}")
            store[uid] = json.loads((run / rel).read_text())
    perm = json.loads((run / "permutation.json").read_text())
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(MODEL, local_files_only=True)
    refs, access = load_references(ids)
    side = {"gate_seal": gseal, "forecast": json.loads((run / "resource_forecast.json").read_text()),
            "pulse_runtime": json.loads((run / "pulse_runtime.json").read_text()),
            "apparatus": json.loads((run / "apparatus_old30.json").read_text())}
    res = evaluate(cfg, pulses, gate, perm, refs, P, tok, side)
    flags = {"critical_failure": not all(res["critical"].values()), "independent_FULL_audit_PASS": True}
    res["provisional_terminal_label"] = decide(cfg, res["gates"], flags)
    res["provisional_note"] = "assumes independent FULL audit PASS and no critical failure; confirmed or replaced by audit_FULL"
    units = res.pop("units")
    atomic_json(run / "evaluation_units.json", {"schema": "srd2_g0_evaluation_units_v1", "units": units})
    res.update(schema="srd2_g0_evaluation_v1", reference_access=access, pulse_seal_sha256=file_hash(run / "pulse_seal.json"),
               audit_PRIMARY_sha256=file_hash(run / "audit_PRIMARY.json"), units_sha256=file_hash(run / "evaluation_units.json"),
               git_head=git("rev-parse", "HEAD"), created_unix=time.time())
    atomic_json(run / "evaluation.json", res)
    print(json.dumps({"label": res["provisional_terminal_label"], "gates": res["gates"], "counts": res["counts"]}, indent=1))


if __name__ == "__main__":
    main()
