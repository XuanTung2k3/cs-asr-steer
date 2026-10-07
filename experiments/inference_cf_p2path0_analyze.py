#!/usr/bin/env python
"""P2-PATH0 analysis (frozen contract configs/inference_cf/p2_path0.json).

primary   (reference-free) suffix-only causal endpoints sliced at the frozen release index, DIRECT_ONLY handling,
          RESCUE_1/3 and INDUCE_1/3 PASS, frozen label precedence. Written BEFORE the output seal.
secondary (references, only behind the committed output seal) canonical per-row/aggregate errors and the frozen
          descriptive ASR_HARM_ALIGNMENT flag; never changes the primary label.
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

from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.path_decode import edit_distance, stream

CONFIG = "configs/inference_cf/p2_path0.json"
BASE = "results/inference_cf/p2path0"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
G = 1e-12


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- pure endpoint / criterion functions -------------------------------------------------------------------------

def row_endpoint(role: str, release: int, B0: list, AUTO: list, FREE: list, ARM: list) -> dict:
    sl = lambda x: list(x)[release:]
    if role == "RESCUE":
        d0, d1 = edit_distance(sl(FREE), sl(B0)), edit_distance(sl(ARM), sl(B0))
        out = {"d0": d0, "d_target": d1}
    else:
        d0, dA, dB = edit_distance(sl(B0), sl(AUTO)), edit_distance(sl(ARM), sl(AUTO)), edit_distance(sl(ARM), sl(B0))
        out = {"d0": d0, "d_target": dA, "dB": dB}
    out["eligible"] = out["d0"] > 0
    out["R"] = 1 - out["d_target"] / out["d0"] if out["eligible"] else None
    return out


def criterion(role: str, rows: list[dict], c: dict) -> dict:
    """rows: [{'dialogue_id', 'd0', 'd_target', 'dB'?, 'eligible', 'R'}] for one arm/L."""
    S = c["suffix_decision"]
    el = [r for r in rows if r["eligible"]]
    n = len(el)
    thr = S["row_distance_reduction_min"]
    succ = [r for r in el if r["R"] >= thr - G and (role == "RESCUE" or r["dB"] > 0)]
    need = (n // 2 + 1) if role == "RESCUE" else max(3, math.ceil(n / 2))
    pooled = 1 - sum(r["d_target"] for r in el) / sum(r["d0"] for r in el) if n else None
    agg_min = S[role]["aggregate_distance_reduction_min"]
    dlg = len({r["dialogue_id"] for r in succ})
    ok = (n >= S[role]["minimum_eligible"] and len(succ) >= need and pooled is not None and pooled >= agg_min - G
          and dlg >= S[role]["minimum_success_dialogues"])
    return {"n_eligible": n, "direct_only": len(rows) - n, "successes": len(succ), "required": need, "success_dialogues": dlg,
            "pooled_reduction": pooled, "sum_d0": sum(r["d0"] for r in el), "sum_d_target": sum(r["d_target"] for r in el),
            "rows_toward_target": sum(r["d_target"] < r["d0"] for r in el),
            "rows_leaving_B0": sum(r.get("dB", 0) > 0 for r in el) if role == "INDUCE" else None, "pass": bool(ok)}


def decide(valid: bool, p: dict) -> str:
    if not valid:
        return "P2_PATH0_INVALID"
    if p["RESCUE_1_PASS"] and p["INDUCE_1_PASS"]:
        return "P2_PATH0_SINGLE_TOKEN_CAUSAL"
    if p["RESCUE_3_PASS"] and p["INDUCE_3_PASS"]:
        return "P2_PATH0_SHORT_PREFIX_CAUSAL"
    if any(p.values()):
        return "P2_PATH0_ASYMMETRIC"
    return "P2_PATH0_NO_LOCAL_BRANCH_CAUSALITY"


# ---- primary (reference-free) ----------------------------------------------------------------------------------

def load(run_dir: Path) -> dict:
    man = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows = [json.loads((run_dir / f"rows/{i:02d}.json").read_text()) if (run_dir / f"rows/{i:02d}.json").exists() else {"status": "missing"}
            for i in range(len(plan["rows"]))]
    return {"manifest": man, "plan": plan, "rows": rows, "runtime": json.loads((run_dir / "runtime.json").read_text())}


def integrity(d: dict) -> dict:
    rows, rt, plan = d["rows"], d["runtime"], d["plan"]
    c = {"all_12_ok": len(rows) == 12 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
         "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
         and bool(rt.get("model_grads_none"))}
    if not c["all_12_ok"]:
        return c
    c["theta0_equals_B0"] = all(r["theta0_equals_B0"] for r in rows)
    c["lang_equals_sealed"] = all(r["lang_equals_sealed"] for r in rows)
    c["effective_equals_checkpoint"] = all(r["effective_equals_checkpoint_bf16"] for r in rows)
    c["FREE_equals_A4"] = all(r["FREE_equals_sealed_A4"] for r in rows)
    c["SELF_equals_FREE"] = all(r["SELF_equals_FREE"] for r in rows)
    c["resets"] = all(r["reset_ok_phase1"] and r["reset_ok_phase2"] for r in rows)
    c["clamp_traces"] = all(r[a]["trace"]["prefix_ok"] and r[a]["trace"]["suppression_ok"] and r[a]["trace"]["positions_ok"]
                            and [f["token"] for f in r[a]["trace"]["forced"]] == (p["self_clamp"] if a == "SELF" else p["clamps"][a[1]]["tokens"])
                            and r[a]["trace"]["release_index"] == p["site"] + len(r[a]["trace"]["forced"])
                            for r, p in zip(rows, plan["rows"]) for a in ("SELF", "L1", "L3"))
    c["live_audit"] = bool(rows[0].get("live_audit", {}).get("pass"))
    cnt = rt["counters"]
    B = cfg()["compute"]
    c["compute"] = (cnt["A4"]["optimizer_steps"] == B["optimizer_steps"] and cnt["A4"]["backwards"] + cnt["audit"]["backwards"] <= B["backwards_max"]
                    and cnt["adapted_decodes"] == B["adapted_decodes"] and cnt["theta0_decodes"] == B["theta0_integrity_decodes"]
                    and cnt["A4"]["teacher_forwards"] + cnt["A4"]["student_forwards"] + cnt["audit"]["forwards"] <= B["A4_teacher_student_forwards_max_including_live_audit"])
    return c


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    integ = integrity(d)
    res = {"schema": "p2_path0_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "integrity": integ, "references_used": False}
    if not integ["all_12_ok"]:
        res.update(label="P2_PATH0_INVALID", passes=None)
        return res
    per, arms = [], {("RESCUE", "1"): [], ("RESCUE", "3"): [], ("INDUCE", "1"): [], ("INDUCE", "3"): []}
    for r, p in zip(d["rows"], d["plan"]["rows"]):
        B0, AU = stream(p["y_B"], p["y_B_terminated"]), stream(p["y_A"], p["y_A_terminated"])
        FR = stream(r["FREE"]["tokens"], r["FREE"]["terminated"])
        e = {"utterance_id": p["utterance_id"], "dialogue_id": p["dialogue_id"], "role": p["role"], "site": p["site"],
             "branch_diagnostics": r["branch_diagnostics"], "arms": {}}
        for L in ("1", "3"):
            ARM = stream(r[f"L{L}"]["tokens"], r[f"L{L}"]["terminated"])
            ep = row_endpoint(p["role"], p["clamps"][L]["release_index"], B0, AU, FR, ARM)
            ep["sealed_baseline_matches"] = ep["d0"] == p["clamps"][L]["suffix_baseline_distance"]
            ep.update(dialogue_id=p["dialogue_id"], output_equals_FREE=ARM == FR, effective_length=p["clamps"][L]["effective_length"])
            e["arms"][L] = ep
            arms[(p["role"], L)].append(ep)
        per.append(e)
    integ["baselines_match_sealed"] = all(ep["sealed_baseline_matches"] for e in per for ep in e["arms"].values())
    crit = {f"{role}_{L}": criterion(role, rs, c) for (role, L), rs in arms.items()}
    passes = {f"{k}_PASS": v["pass"] for k, v in crit.items()}
    valid = all(integ.values())
    res.update(criteria=crit, passes=passes, label=decide(valid, passes), per_row=per, valid=valid,
               runtime={k: d["runtime"].get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "phase1_sec", "phase2_sec", "peak_alloc",
                                                         "peak_reserved", "counters", "gpu", "node")})
    return res


# ---- secondary (references behind the committed seal) --------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from transformers import WhisperProcessor
    from csasr.evaluation.canonical import corpus_metrics
    from experiments.inference_cf_p2seq_analyze import utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("PATH0 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed PATH0 outputs changed")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    a4plan = json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())
    txt = {r["utterance_id"]: r for r in a4plan["rows"]}
    refs = load_references()
    dec = lambda t: tok.decode(t, skip_special_tokens=True)
    rows = []
    for r, p in zip(d["rows"], d["plan"]["rows"]):
        u = p["utterance_id"]
        H = {"B0": txt[u]["y_B_text"], "AUTO": txt[u]["y_A_text"], "FREE": dec(r["FREE"]["tokens"]), "L1": dec(r["L1"]["tokens"]),
             "L3": dec(r["L3"]["tokens"])}
        rows.append({"utterance_id": u, "dialogue_id": p["dialogue_id"], "role": p["role"], "ref": refs[u]["reference"], "H": H,
                     "counts": {k: utt_counts(refs[u]["reference"], h) for k, h in H.items()},
                     "length": {"FREE": len(r["FREE"]["tokens"]), "L1": len(r["L1"]["tokens"]), "L3": len(r["L3"]["tokens"])},
                     "terminated": {"FREE": r["FREE"]["terminated"], "L1": r["L1"]["terminated"], "L3": r["L3"]["terminated"]},
                     "eligible": {L: next(e for e in prim["per_row"] if e["utterance_id"] == u)["arms"][L]["eligible"] for L in ("1", "3")}})
    keys = ("poi_errors", "num_poi", "mixed_errors", "ref_units", "zh_errors", "zh_ref", "en_errors", "en_ref")

    def agg(rs, sys_):
        tot = {k: sum(x["counts"][sys_][i] for x in rs) for i, k in enumerate(keys)}
        cm = corpus_metrics([x["ref"] for x in rs], [x["H"][sys_] for x in rs])
        return {**tot, "substitutions": cm["substitutions"], "deletions": cm["deletions"], "insertions": cm["insertions"],
                "mer": cm["mer"], "zh_cer": cm["zh_cer"], "pier": cm["pier"], "en_wer": cm["en_wer"]}
    out = {}
    for role, systems in (("RESCUE", ("B0", "FREE", "L1", "L3")), ("INDUCE", ("B0", "FREE", "AUTO", "L1", "L3"))):
        rs = [x for x in rows if x["role"] == role]
        out[role] = {s_: agg(rs, s_) for s_ in systems}
    flags = {}
    for L in ("1", "3"):
        rr = [x for x in rows if x["role"] == "RESCUE" and x["eligible"][L]]
        zr = sum(x["counts"]["FREE"][4] - x["counts"][f"L{L}"][4] for x in rr)
        rows_red = sum(x["counts"][f"L{L}"][4] < x["counts"]["FREE"][4] for x in rr)
        poi_up = sum(x["counts"][f"L{L}"][0] for x in rr) > sum(x["counts"]["FREE"][0] for x in rr)
        flags[f"rescue_{L}"] = {"zh_reduction": zr, "rows_reducing_zh": rows_red, "poi_increase": poi_up,
                                "aligned": bool(prim["passes"][f"RESCUE_{L}_PASS"] and zr >= 2 and rows_red >= 2 and not poi_up)}
        ui = [x for x in rows if x["role"] == "INDUCE" and x["eligible"][L]]
        zi = sum(x["counts"][f"L{L}"][4] - x["counts"]["FREE"][4] for x in ui)
        rows_inc = sum(x["counts"][f"L{L}"][4] > x["counts"]["FREE"][4] for x in ui)
        flags[f"induce_{L}"] = {"zh_increase": zi, "rows_increasing_zh": rows_inc,
                                "aligned": bool(prim["passes"][f"INDUCE_{L}_PASS"] and zi >= 2 and rows_inc >= 2)}
    resc = any(flags[f"rescue_{L}"]["aligned"] for L in ("1", "3"))
    ind = any(flags[f"induce_{L}"]["aligned"] for L in ("1", "3"))
    lab = "BIDIRECTIONAL" if resc and ind else "RESCUE_ONLY" if resc else "INDUCTION_ONLY" if ind else "NONE"
    return {"schema": "p2_path0_secondary_v1", "output_seal_hash": seal["seal_hash"], "primary_label": prim["label"], "aggregate": out,
            "ASR_HARM_ALIGNMENT": lab, "alignment_detail": flags,
            "per_row": [{k: v for k, v in x.items() if k != "ref"} for x in rows]}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("primary", "secondary"):
        p = sub.add_parser(n)
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
        print(json.dumps({k: res.get(k) for k in ("label", "integrity", "passes", "criteria")}, indent=1))
    else:
        print(json.dumps({k: res[k] for k in ("primary_label", "ASR_HARM_ALIGNMENT", "alignment_detail", "aggregate")}, indent=1))


if __name__ == "__main__":
    main()
