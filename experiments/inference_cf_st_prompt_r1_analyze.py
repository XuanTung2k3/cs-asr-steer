#!/usr/bin/env python
"""ST-PROMPT-R1-A analysis (frozen spec 'R1-A matrix, barriers and measures' / 'uncertainty and deterministic decision').

primary   reference-free: complete 180 x 36 cells, integrity, barrier, geometry gate, valid / joint-valid counts per arm and
          stratum (offline membership labels only), realized-eta distributions, eligibility.
secondary ONLY after the committed A output seal + committed `ST_PROMPT_R1_A_AUDIT: PASS (PRIMARY)`: frozen P2-RJ reference
          sets / fixed competitors -> margin, corrections / corruptions, rank, target log-prob, EN/ZH mass, KL(edit||NONE);
          shared dialogue bootstrap (B=10000, seed 240924), Bonferroni family 48; frozen gates, label, selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, file_hash

CONFIG = "configs/inference_cf/st_prompt_r1.json"
PANEL = "docs/inference_cf/ST_PROMPT_R1_PANEL.json"
BASE = "results/inference_cf/st_prompt_r1"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal_A.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit_A.json"
POSITIONS = "results/inference_cf/p2rj/positions.json"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    return tracked and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def random_of(arm: dict) -> str:
    return f"random_L{arm['layer']:02d}_eta{arm['eta']:.2f}"


# ---- frozen decision (pure) -----------------------------------------------------------------------------------------

def benefit(a: dict, c: dict) -> bool:
    g = c["R1_A"]
    return bool(a["eligible"] and a["corrections"] >= g["minimum_corrections"] and a["correction_dialogues"] >= g["minimum_correction_dialogues"]
                and a["macro"] is not None and a["macro"] >= g["minimum_macro_margin_nats"]
                and a["lower_none"] is not None and a["lower_none"] > g["adjusted_lower_strictly_above"]
                and a["random_point"] is not None and a["random_point"] > g["minimum_random_contrast_nats"]
                and a["lower_random"] is not None and a["lower_random"] > g["adjusted_lower_strictly_above"])


def safe(a: dict, c: dict) -> bool:
    g = c["R1_A"]
    return a["EN_corruptions"] + a["ZH_corruptions"] <= g["maximum_correct_corruptions"] and a["ZH_corruptions"] <= g["maximum_ZH_corruptions"]


def decide(valid: bool, arms: list[dict], c: dict) -> dict:
    if not valid:
        return {"label": "ST_PROMPT_R1_A_INVALID", "selected": [], "full_pass": []}
    full = [a for a in arms if benefit(a, c) and safe(a, c)]
    if full:
        order = sorted(full, key=lambda a: (-a["corrections"], -a["correction_dialogues"], a["ZH_corruptions"], a["EN_corruptions"],
                                            -a["lower_none"], a["eta"], a["id"]))
        return {"label": "ST_PROMPT_R1_A_CAUSAL_PROMISE", "full_pass": [a["id"] for a in full],
                "selected": [a["id"] for a in order[:c["R1_A"]["max_selected"]]]}
    if any(benefit(a, c) for a in arms):
        return {"label": "ST_PROMPT_R1_A_POWER_WITH_DAMAGE", "selected": [], "full_pass": [], "benefit_only": [a["id"] for a in arms if benefit(a, c)]}
    both = [a["id"] for a in arms if a["eligible"] and a["lower_none"] is not None and a["lower_none"] > 0 and a["lower_random"] is not None and a["lower_random"] > 0]
    if both:
        return {"label": "ST_PROMPT_R1_A_MARGIN_ONLY", "selected": [], "full_pass": [], "both_lower_positive": both}
    return {"label": "ST_PROMPT_R1_A_NO_CORRECTION_POWER", "selected": [], "full_pass": []}


# ---- loading ---------------------------------------------------------------------------------------------------------

def load(run: Path) -> dict:
    man = json.loads((run / "manifest.json").read_text())
    cells, missing, barrier = {}, [], {}
    for i in range(80):
        for nm in ("barrier", "pulses"):
            p = run / nm / f"{i:03d}.json"
            if not p.exists():
                missing.append(f"{nm}/{i:03d}")
                continue
            r = json.loads(p.read_text())
            if r["manifest_hash"] != man["manifest_hash"] or (r["logits_sha256"] and file_hash(run / nm / f"{i:03d}_logits.npz") != r["logits_sha256"]):
                raise ValueError(f"row hash {nm}/{i}")
            for t, pr in r["positions"].items():
                (barrier if nm == "barrier" else cells)[pr["q"]] = {**pr, "_file": f"{nm}/{i:03d}"}
    return {"manifest": man, "cells": cells, "barrier": barrier, "missing": missing}


def primary(run_rel: str) -> dict:
    c = cfg()
    run = ROOT / run_rel
    d = load(run)
    plan = json.loads((ROOT / PLAN).read_text())
    mem = json.loads((ROOT / PANEL).read_text())["evaluation_membership"]
    rt = json.loads((run / "runtime.json").read_text())
    cap = json.loads((run / "capture" / "capture.json").read_text())
    gate = json.loads((run / "geometry_gate.json").read_text())
    arm_ids = [a["id"] for a in plan["arms"]]
    arms = {a["id"]: a for a in plan["arms"]}
    integ = {"capture_identity": bool(cap["identity_ok"]), "geometry_gate_pass": gate["verdict"] == "GEOMETRY_COVERAGE_PASS",
             "runtime_completed": rt.get("status") == "completed" and not rt.get("n_failures"), "rows_present": not d["missing"],
             "barrier_pass": bool((rt.get("barrier") or {}).get("pass")), "frozen_model": bool(rt.get("model_grads_none")) and not rt.get("requires_grad_any") and bool(rt.get("weights_unchanged_sample"))}
    complete = sum(1 for j in range(180) if j in d["cells"] and set(d["cells"][j]["cells"]) == set(arm_ids))
    integ["matrix_complete_180x36"] = complete == 180
    integ["barrier_rows_180"] = len(d["barrier"]) == 180 and all(all(b["zero_dose"].values()) and len(b["zero_dose"]) == 4 and b["restore_bitwise"]
                                                                and all(all(v for v in x["barrier"].values() if v is not None) for x in b["cells"].values())
                                                                for b in d["barrier"].values())
    fails = []
    for j, pr in d["cells"].items():
        if not (pr["clean_replay_bitwise"] and pr["restore_bitwise"] and all(pr["pairwise"].values())):
            fails.append((j, "lineage/pairwise"))
        for a, x in pr["cells"].items():
            if x["integrity_failures"]:
                fails.append((j, a, x["integrity_failures"]))
    integ["cell_integrity"] = not fails
    valid = {a: {s: 0 for s in STRATA} for a in arm_ids}
    vdl = {a: {s: set() for s in STRATA} for a in arm_ids}
    joint = {a: {s: 0 for s in STRATA} for a in arm_ids if arms[a]["family"] == "prompt"}
    jdl = {a: {s: set() for s in STRATA} for a in joint}
    eta_act = defaultdict(list)
    reasons = {a: defaultdict(int) for a in arm_ids}
    for j, pr in d["cells"].items():
        s, dl = mem[j]["stratum"], mem[j]["dialogue_id"]
        for a, x in pr["cells"].items():
            if x["valid"]:
                valid[a][s] += 1
                vdl[a][s].add(dl)
                eta_act[a].append(x["actual_eta"])
            else:
                reasons[a][x.get("no_edit_reason")] += 1
        for a in joint:
            if pr["cells"][a]["valid"] and pr["cells"][random_of(arms[a])]["valid"]:
                joint[a][s] += 1
                jdl[a][s].add(dl)
    e = c["eligibility"]
    ok = lambda n, dd: all(n[s] >= e["minimum_valid_per_stratum"] and len(dd[s]) >= e["minimum_valid_dialogues_per_stratum"] for s in STRATA)
    out_arms = {}
    for a in arm_ids:
        ea = eta_act[a]
        out_arms[a] = {"valid_by_stratum": valid[a], "valid_dialogues_by_stratum": {s: len(vdl[a][s]) for s in STRATA}, "eligible_valid": ok(valid[a], vdl[a]),
                       "no_edit_reasons": dict(reasons[a]),
                       "actual_eta": {"min": min(ea), "median": float(np.median(ea)), "max": max(ea)} if ea else None}
        if a in joint:
            out_arms[a]["joint_valid_by_stratum"] = joint[a]
            out_arms[a]["joint_valid_dialogues_by_stratum"] = {s: len(jdl[a][s]) for s in STRATA}
            out_arms[a]["eligible"] = ok(valid[a], vdl[a]) and ok(joint[a], jdl[a])
    return {"schema": "st_prompt_r1_A_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "references_used": False, "valid": all(integ.values()),
            "integrity": integ, "failures": [str(x) for x in fails[:50]], "arms": out_arms,
            "runtime": {k: rt.get(k) for k in ("job_id", "elapsed_sec", "peak_alloc", "peak_reserved", "counters", "barrier", "pulses_sec")},
            "capture": {k: cap.get(k) for k in ("elapsed_sec", "counters", "peak_alloc", "job_id")}}


# ---- secondary -------------------------------------------------------------------------------------------------------

def kl_edit_none(z_edit, z_none, t, sup, beg) -> float:
    from experiments.inference_cf_p2dir_analyze import processed
    a = processed(z_edit, t, sup, beg)
    b = processed(z_none, t, sup, beg)
    la = a - np.logaddexp.reduce(a[np.isfinite(a)])
    lb = b - np.logaddexp.reduce(b[np.isfinite(b)])
    m = np.isfinite(la)
    p = np.exp(la[m])
    return float(np.sum(p * (la[m] - lb[m])))


def secondary(run_rel: str) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from experiments.inference_cf_p2dir_analyze import boot_stat, draw_weights, logit_metrics, unpack_bf16
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("A output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(file_hash(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != c["audit"]["post_A"] + " (PRIMARY)":
        raise PermissionError("primary audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis_A.json").read_text())
    run = ROOT / run_rel
    d = load(run)
    plan = json.loads((ROOT / PLAN).read_text())
    mem = json.loads((ROOT / PANEL).read_text())["evaluation_membership"]
    pos = json.loads((ROOT / POSITIONS).read_text())["positions"]          # evaluator references opened only now
    if [(p["utterance_id"], int(p["t"])) for p in pos] != [(p["utterance_id"], int(p["t"])) for p in mem]:
        raise ValueError("positions order")
    model = c["conditions"]["model"]["dir"]
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    with np.load(run / "capture" / "none_logits.npz") as z:
        none = {int(k[1:]): unpack_bf16(z[k]) for k in z.files}
    arms = {a["id"]: a for a in plan["arms"]}
    cache: dict = {}
    rows = []
    for j, p in enumerate(pos):
        t, Y, cc = int(p["t"]), [int(x) for x in p["target_ids"]], int(p["competitor"])
        nm = logit_metrics(none[j], t, sup, beg, part, Y, cc)
        pr = d["cells"][j]
        if pr["_file"] not in cache:
            cache.clear()
            with np.load(run / f"{pr['_file']}_logits.npz") as z:
                cache[pr["_file"]] = {k: z[k] for k in z.files}
        lg = cache[pr["_file"]]
        r = {"q": j, "dialogue_id": p["dialogue_id"], "stratum": mem[j]["stratum"], "none": nm, "arms": {}}
        for a in arms:
            z_ = unpack_bf16(lg[f"q{j:03d}_{a}"])
            am = logit_metrics(z_, t, sup, beg, part, Y, cc)
            x = pr["cells"][a]
            r["arms"][a] = {"valid": x["valid"], "d_m": am["m"] - nm["m"], "top1": am["top1"], "in_ref": am["top1_in_ref"], "changed": am["top1"] != nm["top1"],
                            "d_rank": am["rank_ref"] - nm["rank_ref"], "d_logp_ref": am["logp_ref"] - nm["logp_ref"], "d_P_E": am["P_E"] - nm["P_E"],
                            "d_P_M": am["P_M"] - nm["P_M"], "kl": kl_edit_none(z_, none[j], t, sup, beg)}
        rows.append(r)
    keys, W = draw_weights([r["dialogue_id"] for r in rows], c["bootstrap"]["replicates"], c["bootstrap"]["seed"])
    adj = c["bootstrap"]["alpha"] / c["bootstrap"]["joint_family_size"]
    by = {s: [r for r in rows if r["stratum"] == s] for s in STRATA}
    out = []
    for a, arm in arms.items():
        st = {}
        for s in STRATA:
            rs = by[s]
            al = adj if (arm["family"] == "prompt" and s == "EN-confusion") else 0.05
            st[s] = {"macro": boot_stat([(r["dialogue_id"], r["arms"][a]["d_m"]) for r in rs], keys, W, al),
                     "position_mean_d_m": float(np.mean([r["arms"][a]["d_m"] for r in rs])),
                     "corrections": sum((not r["none"]["top1_in_ref"]) and r["arms"][a]["in_ref"] for r in rs),
                     "corruptions": sum(r["none"]["top1_in_ref"] and not r["arms"][a]["in_ref"] for r in rs),
                     "top1_changes": sum(r["arms"][a]["changed"] for r in rs), "rank_improved": sum(r["arms"][a]["d_rank"] < 0 for r in rs),
                     "mean_d_rank": float(np.mean([r["arms"][a]["d_rank"] for r in rs])), "mean_d_logp_ref": float(np.mean([r["arms"][a]["d_logp_ref"] for r in rs])),
                     "mean_d_P_E": float(np.mean([r["arms"][a]["d_P_E"] for r in rs])), "mean_d_P_M": float(np.mean([r["arms"][a]["d_P_M"] for r in rs])),
                     "mean_kl": float(np.mean([r["arms"][a]["kl"] for r in rs])), "invalid": sum(not r["arms"][a]["valid"] for r in rs)}
        conf = by["EN-confusion"]
        corr = [r for r in conf if (not r["none"]["top1_in_ref"]) and r["arms"][a]["in_ref"]]
        rec = {"id": a, "family": arm["family"], "layer": arm["layer"], "eta": arm["eta"], "sign": arm["sign"], "strata": st,
               "macro": st["EN-confusion"]["macro"]["estimate"], "lower_none": (st["EN-confusion"]["macro"]["ci"] or [None])[0],
               "upper_none": (st["EN-confusion"]["macro"]["ci"] or [None, None])[1], "usable_draws_none": st["EN-confusion"]["macro"]["valid_draws"],
               "corrections": len(corr), "correction_dialogues": len({r["dialogue_id"] for r in corr}), "corrected_positions": [r["q"] for r in corr],
               "EN_corruptions": st["EN-correct"]["corruptions"], "ZH_corruptions": st["ZH-correct"]["corruptions"],
               "eligible": bool(prim["arms"][a].get("eligible", prim["arms"][a]["eligible_valid"]))}
        if arm["family"] == "prompt":
            ra = random_of(arm)
            pairs = [(r["dialogue_id"], (r["arms"][a]["d_m"] - r["arms"][ra]["d_m"]) if (r["arms"][a]["valid"] and r["arms"][ra]["valid"]) else 0.0) for r in conf]
            vr = boot_stat(pairs, keys, W, adj)
            rec.update(random_point=vr["estimate"], lower_random=(vr["ci"] or [None])[0], upper_random=(vr["ci"] or [None, None])[1],
                       usable_draws_random=vr["valid_draws"], random_arm=ra)
            draws_ok = rec["usable_draws_none"] >= c["bootstrap"]["minimum_valid_draws"] and vr["valid_draws"] >= c["bootstrap"]["minimum_valid_draws"]
            if not draws_ok:
                rec["lower_none"] = rec["lower_random"] = None
            rec["draws_ok"] = draws_ok
        out.append(rec)
    primary_arms = [x for x in out if x["family"] == "prompt"]
    valid = bool(prim["valid"])
    dec = decide(valid, primary_arms, c)
    union = sorted({q for x in primary_arms for q in x["corrected_positions"]})
    oracle = {"label": "REFERENCE-DEPENDENT UPPER BOUND; NOT DEPLOYABLE", "union_corrected_positions": union, "union_count": len(union),
              "union_dialogues": len({rows[q]["dialogue_id"] for q in union}),
              "random_union": sorted({q for x in out if x["family"] == "random" for q in x["corrected_positions"]})}
    for x in primary_arms:
        x["benefit_pass"], x["safety_pass"] = benefit(x, c), safe(x, c)
    return {"schema": "st_prompt_r1_A_secondary_v1", "output_seal_hash": seal["seal_hash"], "valid": valid, "arms": out, "oracle_union": oracle,
            "decision": dec, "label": dec["label"], "bootstrap": {"dialogues": keys, "replicates": c["bootstrap"]["replicates"], "seed": c["bootstrap"]["seed"],
                                                                  "adjusted_quantiles": [adj / 2, 1 - adj / 2]}}


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
        print(json.dumps({"valid": res["valid"], "integrity": res["integrity"]}, indent=1))
    else:
        print(json.dumps({"label": res["label"], "decision": res["decision"], "union": res["oracle_union"]["union_count"]}, indent=1))
        for a in res["arms"]:
            print(a["id"], a["eligible"], round(a["macro"], 3), a["lower_none"], a.get("random_point"), a.get("lower_random"), a["corrections"], a["correction_dialogues"], a["EN_corruptions"], a["ZH_corruptions"])


if __name__ == "__main__":
    main()
