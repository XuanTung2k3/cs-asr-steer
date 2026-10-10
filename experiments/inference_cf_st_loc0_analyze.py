#!/usr/bin/env python
"""ST-LOC0 analysis (frozen configs/inference_cf/st_loc0.json).

primary   (reference-free) matrix completeness (180 x 21 cells), historical barrier, zero-dose/restore/cache integrity,
          matched-energy validity and pairwise energy, no-edit reasons, per-arm valid counts, eligibility
          (>=57 valid and >=12 valid dialogues in EACH construction stratum), V1/V2 provenance, compute.
secondary (evaluator; only after the committed seal + committed primary audit PASS) fixed historical margin
          m = logsumexp(z[Y_ref]) - z[c_fixed] via inference_cf_p2dir_analyze.logit_metrics; per arm/stratum deltas,
          top-1 corrections/corruptions/changes, rank and script-mass change; dialogue-macro estimand with the shared
          dialogue-cluster bootstrap (B=10000, seed 240924; Bonferroni family 16 for primary EN-confusion margins);
          controls; historical D0/D1/D2 evaluator reproduction; evaluator-only oracle envelope; frozen label + winner.
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

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

CONFIG = "configs/inference_cf/st_loc0.json"
PANEL = "docs/inference_cf/ST_LOC0_PANEL.json"
BASE = "results/inference_cf/st_loc0"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
POSITIONS = "results/inference_cf/p2rj/positions.json"
HIST_ANALYSIS = "results/inference_cf/p2dir/exp1_run1_analysis.json"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
BARRIER_ARMS = ("v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "v_unq_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "D2_L16_POST_CROSS_ATTENTION_PRE_FFN")
HIST_NAME = {BARRIER_ARMS[0]: "D0", BARRIER_ARMS[1]: "D1", BARRIER_ARMS[2]: "D2"}
LABELS = ("ST_LOC0_INVALID", "ST_LOC0_NO_FIXED_DIRECTION_LEVER", "ST_LOC0_LOCAL_EFFECT_ONLY", "ST_LOC0_SITE_FEASIBLE")


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- frozen rules (pure) --------------------------------------------------------------------------------------------

def eligible(valid_by_stratum: dict, valid_dialogues_by_stratum: dict, c: dict) -> bool:
    e = c["cell_eligibility"]
    return all(valid_by_stratum.get(s, 0) >= e["minimum_valid_per_stratum"] and valid_dialogues_by_stratum.get(s, 0) >= e["minimum_valid_dialogues_per_stratum"]
               for s in STRATA)


def qualifies(arm: dict, c: dict) -> bool:
    d = c["decision"]
    return bool(arm["eligible"] and arm["integrity_pass"] and arm["adj_lower"] is not None and arm["adj_lower"] > d["adjusted_lower_bound_strictly_above"]
                and arm["macro"] is not None and arm["macro"] >= d["minimum_EN_confusion_dialogue_macro_delta_margin_nats"]
                and arm["corrections"] >= d["minimum_top1_reference_corrections"] and arm["correction_dialogues"] >= d["minimum_correction_dialogues"])


def decide(valid: bool, arms: list[dict], c: dict) -> dict:
    if not valid:
        return {"label": "ST_LOC0_INVALID", "winner": None, "qualifying": []}
    pos = [a for a in arms if a["eligible"] and a["adj_lower"] is not None and a["adj_lower"] > 0]
    if not pos:
        return {"label": "ST_LOC0_NO_FIXED_DIRECTION_LEVER", "winner": None, "qualifying": []}
    q = [a for a in arms if qualifies(a, c)]
    if not q:
        return {"label": "ST_LOC0_LOCAL_EFFECT_ONLY", "winner": None, "qualifying": [], "adjusted_positive": [a["id"] for a in pos]}
    q.sort(key=lambda a: (-a["corrections"], -a["correction_dialogues"], -a["macro"], a["correct_corruptions"], a["id"]))
    return {"label": "ST_LOC0_SITE_FEASIBLE", "winner": q[0]["id"], "qualifying": [a["id"] for a in q], "adjusted_positive": [a["id"] for a in pos],
            "requires_independent_post_audit_PASS": True}


# ---- loading --------------------------------------------------------------------------------------------------------

def load(run_rel: str) -> dict:
    run = ROOT / run_rel
    man = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / PLAN).read_text())
    rt = json.loads((run / "runtime.json").read_text())
    cells, missing = {}, []
    for name in ("barrier", "pulses"):
        for i in range(80):
            p = run / name / f"{i:03d}.json"
            if not p.exists():
                missing.append(f"{name}/{i:03d}")
                continue
            row = json.loads(p.read_text())
            if row["manifest_hash"] != man["manifest_hash"] or file_hash(run / name / f"{i:03d}_logits.npz") != row["logits_sha256"]:
                raise ValueError(f"row hash {name}/{i}")
            for t, pr in row["positions"].items():
                cells.setdefault(pr["q"], {"q": pr["q"], "t": pr["t"], "rows": {}})["rows"][name] = {**pr, "_file": f"{name}/{i:03d}"}
    return {"run": run, "manifest": man, "plan": plan, "runtime": rt, "cells": cells, "missing": missing}


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(run_rel)
    plan, rt, cells = d["plan"], d["runtime"], d["cells"]
    P = json.loads((ROOT / PANEL).read_text())
    con = P["construction_positions"]
    arm_ids = [a["id"] for a in plan["arms"]]
    integ = {"runtime_completed": rt.get("status") == "completed", "no_runtime_failures": not rt.get("n_failures"), "rows_present": not d["missing"],
             "barrier_pass": bool((rt.get("barrier") or {}).get("pass")), "model_grads_none": bool(rt.get("model_grads_none")),
             "phaseA_identity": bool((rt.get("phaseA_identity") or {}).get("all_ok")), "calibration_GO": bool(rt.get("go"))}
    complete, energy_bad, restore_bad, zero_bad, clean_bad, integ_fail = 0, [], [], [], [], []
    per_arm = {a: {"valid": {s: 0 for s in STRATA}, "valid_dlg": {s: set() for s in STRATA}, "no_edit": {}, "rel_energy": [], "rel_mag": []} for a in arm_ids}
    for j in range(180):
        cl = cells.get(j)
        if cl is None or set(cl["rows"]) != {"barrier", "pulses"}:
            continue
        s, dlg = con[j]["stratum"], con[j]["dialogue_id"]
        allcells = {**cl["rows"]["barrier"]["cells"], **cl["rows"]["pulses"]["cells"]}
        if set(allcells) == set(arm_ids):
            complete += 1
        for nm in ("barrier", "pulses"):
            r = cl["rows"][nm]
            if not (r["restore_bitwise"] and r["cache_prefix_unchanged"]):
                restore_bad.append(j)
            if not r["clean_replay_bitwise"]:
                clean_bad.append(j)
        if not all(cl["rows"]["barrier"].get("zero_dose", {}).values()) or len(cl["rows"]["barrier"].get("zero_dose", {})) != 4:
            zero_bad.append(j)
        ens = []
        for a, rec in allcells.items():
            if rec["integrity_failures"]:
                integ_fail.append((j, a, rec["integrity_failures"]))
            if rec["valid"]:
                per_arm[a]["valid"][s] += 1
                per_arm[a]["valid_dlg"][s].add(dlg)
                per_arm[a]["rel_energy"].append(rec["relative_sq_energy"])
                per_arm[a]["rel_mag"].append(rec["relative_magnitude"])
                ens.append(rec["edit_norm"] ** 2)
            else:
                per_arm[a]["no_edit"][rec["no_edit_reason"]] = per_arm[a]["no_edit"].get(rec["no_edit_reason"], 0) + 1
        if ens and max(ens) / min(ens) - 1.0 > c["energy"]["max_pairwise_relative_squared_error"]:
            energy_bad.append(j)
    integ.update(matrix_complete=complete == 180, pairwise_energy=not energy_bad, restore_cache=not restore_bad, zero_dose=not zero_bad,
                 clean_replay=not clean_bad, cell_integrity=not integ_fail)
    valid = all(integ.values())
    arms_out = {}
    for a in arm_ids:
        x = per_arm[a]
        vd = {s: len(x["valid_dlg"][s]) for s in STRATA}
        re = x["rel_energy"]
        arms_out[a] = {"valid_by_stratum": x["valid"], "valid_dialogues_by_stratum": vd, "eligible": eligible(x["valid"], vd, c), "no_edit_reasons": x["no_edit"],
                       "relative_sq_energy": {"min": min(re), "median": float(np.median(re)), "max": max(re)} if re else None,
                       "relative_magnitude_median": float(np.median(x["rel_mag"])) if x["rel_mag"] else None}
    cal = json.loads((d["run"] / "calibration" / "calibration_seal.json").read_text())
    fold_status = cal["fold_status"]
    return {"schema": "st_loc0_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "references_used": False, "valid": valid, "integrity": integ,
            "failures": {"energy": energy_bad, "restore": restore_bad, "zero": zero_bad, "clean": clean_bad, "cells": integ_fail[:50]},
            "arms": arms_out, "calibration": {"seal_hash": cal["seal_hash"], "fold_status": fold_status, "V1_invalid": cal["V1_invalid"],
                                              "fold_invalid": {sk: sum(v != "ok" for v in st.values()) for sk, st in fold_status.items()}},
            "runtime": {k: rt.get(k) for k in ("job_id", "elapsed_sec", "phaseA_sec", "phaseB_sec", "wait_sec", "barrier_sec", "pulses_sec", "peak_alloc",
                                               "peak_reserved", "counters", "gpu")}}


# ---- secondary ------------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from experiments.inference_cf_p2dir_analyze import boot_stat, draw_weights, logit_metrics, unpack_bf16
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("ST-LOC0 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(file_hash(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != "ST_LOC0_AUDIT: PASS":
        raise PermissionError("primary-phase audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(run_rel)
    run, plan = d["run"], d["plan"]
    P = json.loads((ROOT / PANEL).read_text())
    con = P["construction_positions"]
    pos = json.loads((ROOT / POSITIONS).read_text())["positions"]          # evaluator references: opened only now
    if [(p["utterance_id"], int(p["t"])) for p in pos] != [(p["utterance_id"], int(p["t"])) for p in con]:
        raise ValueError("evaluator positions order")
    model = Path(d["manifest"]["model"]["dir"])
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    with np.load(run / "calibration" / "baseline_logits.npz") as z:
        base = {int(k[1:]): unpack_bf16(z[k]) for k in z.files}
    lg_cache: dict = {}

    def logits(file, key):
        if file not in lg_cache:
            with np.load(run / f"{file}_logits.npz") as z:
                lg_cache[file] = {k: z[k] for k in z.files}
        return unpack_bf16(lg_cache[file][key])
    arm_ids = [a["id"] for a in plan["arms"]]
    rows = []
    for j, p in enumerate(pos):
        t, Y, cs = int(p["t"]), [int(x) for x in p["target_ids"]], int(p["competitor"])
        nm = logit_metrics(base[j], t, sup, beg, partition, Y, cs)
        cl = d["cells"][j]
        r = {"q": j, "utterance_id": p["utterance_id"], "dialogue_id": p["dialogue_id"], "stratum": con[j]["stratum"], "none": nm, "arms": {}}
        for nmx in ("barrier", "pulses"):
            for a, rec in cl["rows"][nmx]["cells"].items():
                am = logit_metrics(logits(cl["rows"][nmx]["_file"], f"q{j:03d}_{a}"), t, sup, beg, partition, Y, cs)
                r["arms"][a] = {"valid": rec["valid"], "d_m": am["m"] - nm["m"], "top1": am["top1"], "in_ref": am["top1_in_ref"],
                                "base_in_ref": nm["top1_in_ref"], "top1_changed": am["top1"] != nm["top1"], "d_rank": am["rank_ref"] - nm["rank_ref"],
                                "d_P_E": am["P_E"] - nm["P_E"], "d_P_M": am["P_M"] - nm["P_M"], "d_logp_ref": am["logp_ref"] - nm["logp_ref"]}
        rows.append(r)
    # historical evaluator reproduction (D0/D1/D2 at L16 DG-02)
    hist = json.loads((ROOT / HIST_ANALYSIS).read_text())["per_position"]
    hmax = 0.0
    hflip = 0
    for r, h in zip(rows, hist):
        for a, hn in HIST_NAME.items():
            hmax = max(hmax, abs(r["arms"][a]["d_m"] - h["arms"][hn]["d_m"]))
            hflip += int(r["arms"][a]["top1"] != h["arms"][hn]["post"]["top1"])
    integ = {"primary_valid": bool(prim["valid"]), "historical_evaluator_reproduced": hmax <= c["historical_reproduction"]["evaluator_scalar_abs"] and hflip == 0}
    keys, W = draw_weights([r["dialogue_id"] for r in rows], c["bootstrap"]["replicates"], c["bootstrap"]["seed"])
    a_fam = c["bootstrap"]["family_alpha"] / c["bootstrap"]["family_size"]
    by_s = {s: [r for r in rows if r["stratum"] == s] for s in STRATA}
    arms = []
    primary_ids = [a["id"] for a in c["primary_arms"]]
    for a in arm_ids:
        conf = by_s["EN-confusion"]
        corr = [r for r in conf if (not r["arms"][a]["base_in_ref"]) and r["arms"][a]["in_ref"]]
        st = {}
        for s in STRATA:
            rs = by_s[s]
            alpha = a_fam if (a in primary_ids and s == "EN-confusion") else 0.05
            st[s] = {"macro": boot_stat([(r["dialogue_id"], r["arms"][a]["d_m"]) for r in rs], keys, W, alpha),
                     "position_mean_d_m": float(np.mean([r["arms"][a]["d_m"] for r in rs])),
                     "top1_changes": sum(r["arms"][a]["top1_changed"] for r in rs),
                     "corrections": sum((not r["arms"][a]["base_in_ref"]) and r["arms"][a]["in_ref"] for r in rs),
                     "corruptions": sum(r["arms"][a]["base_in_ref"] and not r["arms"][a]["in_ref"] for r in rs),
                     "mean_d_rank": float(np.mean([r["arms"][a]["d_rank"] for r in rs])), "rank_improved": sum(r["arms"][a]["d_rank"] < 0 for r in rs),
                     "mean_d_P_E": float(np.mean([r["arms"][a]["d_P_E"] for r in rs])), "mean_d_P_M": float(np.mean([r["arms"][a]["d_P_M"] for r in rs])),
                     "invalid_cells": sum(not r["arms"][a]["valid"] for r in rs)}
            st[s]["corruption_rate"] = st[s]["corruptions"] / len(rs)
        pa = prim["arms"][a]
        mc = st["EN-confusion"]["macro"]
        arms.append({"id": a, "primary": a in primary_ids, "eligible": pa["eligible"], "integrity_pass": bool(prim["valid"]),
                     "macro": mc["estimate"], "adj_lower": mc["ci"][0] if mc["ci"] else None, "adj_upper": mc["ci"][1] if mc["ci"] else None,
                     "corrections": len(corr), "correction_dialogues": len({r["dialogue_id"] for r in corr}), "corrected_positions": [r["q"] for r in corr],
                     "correct_corruptions": st["EN-correct"]["corruptions"] + st["ZH-correct"]["corruptions"], "strata": st,
                     "valid_by_stratum": pa["valid_by_stratum"]})
    # paired vs same-site random (pointwise 95%, descriptive)
    am = {a["id"]: a for a in arms}
    plan_arms = {a["id"]: a for a in plan["arms"]}
    rnd = {(x["layer"], x["site"]): x["id"] for x in plan["arms"] if x["family"] == "random"}
    for a in primary_ids:
        ra = rnd[(plan_arms[a]["layer"], plan_arms[a]["site"])]
        am[a]["vs_random"] = boot_stat([(r["dialogue_id"], r["arms"][a]["d_m"] - r["arms"][ra]["d_m"]) for r in by_s["EN-confusion"]], keys, W, 0.05)
        am[a]["vs_D2"] = boot_stat([(r["dialogue_id"], r["arms"][a]["d_m"] - r["arms"][BARRIER_ARMS[2]]["d_m"]) for r in by_s["EN-confusion"]], keys, W, 0.05)
        twin = a.replace("v_prompt_", "v_unq_") if a.startswith("v_prompt_") else None
        if twin:
            am[a]["V1_minus_V2_same_site_sign"] = boot_stat([(r["dialogue_id"], r["arms"][a]["d_m"] - r["arms"][twin]["d_m"]) for r in by_s["EN-confusion"]], keys, W, 0.05)
    valid = all(integ.values())
    dec = decide(valid, [am[a] for a in primary_ids], c)
    # oracle envelope: evaluator-only, primary 16 arms only, reference-dependent, non-deployable
    conf = by_s["EN-confusion"]
    fam = lambda a: plan_arms[a]["family"]
    def corr_set(ids):
        return {r["q"] for r in conf if not r["none"]["top1_in_ref"] and any(r["arms"][a]["in_ref"] for a in ids)}
    v1 = corr_set([a for a in primary_ids if fam(a) == "v_prompt"])
    v2 = corr_set([a for a in primary_ids if fam(a) == "v_unq"])
    un = v1 | v2
    dl = {r["q"]: r["dialogue_id"] for r in rows}
    damaged = {r["q"] for r in rows if r["stratum"] != "EN-confusion" and r["none"]["top1_in_ref"] and any(not r["arms"][a]["in_ref"] for a in primary_ids)}
    best = max((am[a]["corrections"] for a in primary_ids), default=0)
    per_utt, per_dlg = {}, {}
    for q in un:
        u = rows[q]["utterance_id"]
        per_utt[u] = per_utt.get(u, 0) + 1
        per_dlg[dl[q]] = per_dlg.get(dl[q], 0) + 1
    oracle = {"label": "REFERENCE-DEPENDENT UPPER BOUND; NOT DEPLOYABLE", "baseline_incorrect_confusions": sum(not r["none"]["top1_in_ref"] for r in conf),
              "V1_correctable": len(v1), "V2_correctable": len(v2), "union_correctable": len(un), "union_fraction": len(un) / len(conf),
              "union_dialogues": len({dl[q] for q in un}), "contributing_arms": {a: am[a]["corrections"] for a in primary_ids if am[a]["corrections"]},
              "correct_state_positions_damaged_by_some_arm": len(damaged), "best_single_arm_corrections": best,
              "max_utterance_share": (max(per_utt.values()) / len(un)) if un else None, "max_dialogue_share": (max(per_dlg.values()) / len(un)) if un else None}
    controls = {"NONE": "baseline (delta 0 by definition)", "random": {x: am[x] for x in rnd.values()}, "D2": am[BARRIER_ARMS[2]]}
    return {"schema": "st_loc0_secondary_v1", "output_seal_hash": seal["seal_hash"], "integrity": integ, "valid": valid,
            "historical_evaluator_max_abs": hmax, "historical_top1_mismatch": hflip,
            "bootstrap": {"replicates": c["bootstrap"]["replicates"], "seed": c["bootstrap"]["seed"], "dialogues": keys, "family_alpha": a_fam,
                          "adjusted_quantiles": [a_fam / 2, 1 - a_fam / 2]},
            "arms": arms, "controls": controls, "oracle": oracle, "decision": dec, "label": dec["label"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("primary", "secondary"):
        p = sub.add_parser(nm)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable((primary if args.cmd == "primary" else secondary)(args.run))
    atomic_json(out, res)
    if args.cmd == "primary":
        print(json.dumps({k: res[k] for k in ("valid", "integrity", "calibration")}, indent=1))
    else:
        print(json.dumps({"label": res["label"], "decision": res["decision"], "integrity": res["integrity"], "oracle": res["oracle"]}, indent=1))
        for a in res["arms"]:
            print(a["id"], a["eligible"], round(a["macro"], 4), a["adj_lower"], a["corrections"], a["correction_dialogues"], a["correct_corruptions"])


if __name__ == "__main__":
    main()
