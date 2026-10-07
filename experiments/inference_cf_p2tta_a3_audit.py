#!/usr/bin/env python
"""Independent P2-TTA-A3 auditor. Does NOT import inference_cf_p2tta_a3_analyze, csasr.inference_cf.soft_auto_tta or
csasr.inference_cf.episodic_tta; reuses the independent TTA0 auditor helpers and locked canonical primitives only.

prerun -> PASS_TO_P2_TTA_A3 | BLOCK_BEFORE_P2_TTA_A3
live   -> ``live_kl_check`` (runner, first row, theta0, no optimizer update): independent full-vocabulary masked forward
          KL(q_AUTO || p_FORCED) via F.kl_div(log_target=True) over -inf-masked logits, one backward to fresh fp32 masters.
post   -> P2_TTA_A3_AUDIT: PASS | BLOCK
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

import experiments.inference_cf_p2tta0_audit as tau

FREEZE = "bd41bc5"
CONFIG = "configs/inference_cf/p2_tta_a3.json"
PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
BASE = ROOT / "results/inference_cf/p2tta_a3"
PLAN_REL = "results/inference_cf/p2tta_a3/plan_sealed.json"


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- live first-row check --------------------------------------------------------------------------------------

def live_kl_check(model, names, encoded, cA, cB, y, *, suppress, begin, tokenizer) -> dict:
    import torch
    import torch.nn.functional as F
    params = dict(model.named_parameters())
    masters = [torch.nn.Parameter(params[n].detach().float().clone()) for n in names]
    dev = masters[0].device
    valid = torch.tensor(tau.own_valid(y, suppress, begin, tokenizer), dtype=torch.bool, device=dev)
    T = len(y)

    def masked(logits):
        q = logits[0].float()[3:3 + T]
        ban = torch.zeros_like(q, dtype=torch.bool)
        if suppress:
            ban[:, list(suppress)] = True
        if begin and T:
            ban[0, list(begin)] = True
        return q.masked_fill(ban, float("-inf")), ban
    with torch.no_grad():
        tl, ban = masked(model(encoder_outputs=encoded, decoder_input_ids=torch.tensor([list(cA) + list(y)], device=dev), use_cache=False).logits)
        logq = torch.log_softmax(tl, -1)
    with torch.enable_grad():
        out = torch.func.functional_call(model, {n: m.to(torch.bfloat16) for n, m in zip(names, masters)}, args=(),
                                         kwargs={"encoder_outputs": encoded, "decoder_input_ids": torch.tensor([list(cB) + list(y)], device=dev),
                                                 "use_cache": False}, strict=False, tie_weights=True)
        sl, _ = masked(out.logits)
        logp = torch.log_softmax(sl, -1)
        keep = ~ban
        kl_rows = F.kl_div(torch.where(keep, logp, torch.zeros_like(logp)), torch.where(keep, logq, torch.zeros_like(logq)),
                           reduction="none", log_target=True)
        kl_rows = torch.where(keep, kl_rows, torch.zeros_like(kl_rows)).sum(-1)
        loss = kl_rows[valid].mean() if int(valid.sum()) and list(cA) != list(cB) else sum((m * 0).sum() for m in masters)
        loss.backward()
    g = torch.cat([(m.grad if m.grad is not None else torch.zeros_like(m)).detach().flatten() for m in masters])
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("live audit contaminated model gradients")
    return {"auditor_loss": float(loss), "valid": int(valid.sum()), "_grad": g}


# ---- independent panel replay -----------------------------------------------------------------------------------------

def own_panel(c: dict) -> list[tuple]:
    f = json.loads((ROOT / "configs/inference_cf/p2_tta_funnel.json").read_text())
    parent = json.loads((ROOT / c["panel"]["parent"]).read_text())["rows"]
    base = {b["utterance_id"]: b for b in f["baseline100"]}
    txt = {}
    for r in parent:
        b = base[r["utterance_id"]]
        txt[r["utterance_id"]] = (json.loads((ROOT / b["forced_path"]).read_text())["systems"]["S0"]["text"],
                                  json.loads((ROOT / b["auto_path"]).read_text())["systems"]["B0_AUTO"]["text"])
    dorder = list(dict.fromkeys(r["dialogue_id"] for r in parent))
    isD = {u: a != b for u, (a, b) in txt.items()}
    Drows = [r for r in parent if isD[r["utterance_id"]]]
    assert len(Drows) <= 16
    chosen = [(r["utterance_id"], r["dialogue_id"], "D") for r in Drows]
    n = {d: sum(1 for x in chosen if x[1] == d) for d in dorder}
    Aq = {d: [r["utterance_id"] for r in parent if r["dialogue_id"] == d and not isD[r["utterance_id"]]] for d in dorder}
    for d in dorder:
        if n[d] == 0 and Aq[d] and len(chosen) < 24:
            chosen.append((Aq[d].pop(0), d, "A"))
            n[d] += 1
    reps = [d for d in dorder if any(x[1] == d and x[2] == "D" for x in chosen)]
    for pool, cap in ((reps, 2), (dorder, 2), (dorder, 10 ** 9)):
        grew = True
        while len(chosen) < 24 and grew:
            grew = False
            for d in pool:
                if len(chosen) < 24 and Aq[d] and n[d] < cap:
                    chosen.append((Aq[d].pop(0), d, "A"))
                    n[d] += 1
                    grew = True
    return chosen


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {}
    for rel in ("docs/inference_cf/P2_TTA_A3_SPEC.md", "docs/inference_cf/P2_TTA_A3_CLAUDE_DESIGN.md", CONFIG, PANEL):
        checks[f"frozen:{rel}"] = sha(ROOT / rel) == tau.blob_sha(FREEZE, rel)
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    panel = json.loads((ROOT / PANEL).read_text())
    mine = own_panel(c)
    checks["panel_replayed_independently"] = [(r["utterance_id"], r["dialogue_id"], r["group"]) for r in panel["rows"]] == mine \
        and sha(ROOT / PANEL) == c["panel"]["byte_sha256"] and [x[0] for x in mine] == c["panel"]["ids"] \
        and sum(x[2] == "D" for x in mine) == 12 and len({x[1] for x in mine}) == 20
    f = json.loads((ROOT / "configs/inference_cf/p2_tta_funnel.json").read_text())
    t0 = json.loads((ROOT / "configs/inference_cf/p2_tta0.json").read_text())
    sel = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta0_r/selected_objective.json").read_text())
    checks["actuator_inherited"] = (c["trainables"] == f["common"]["trainables"] == t0["trainables"] and c["optimization"] == f["common"]["optimization"]
                                    and c["optimization"]["lr"] == 1e-3 and c["optimization"]["steps"] == 2 and c["optimization"]["weight_decay"] == 0
                                    and sel["optimizer"] == {k: c["optimization"][k] for k in sel["optimizer"]}
                                    and len(c["trainables"]["parameters"]) == 194)
    keys = ["max_MER_increase", "max_ZH_CER_increase", "min_matrix_ZH_retention", "min_embedded_EN_retention", "max_outside_POI_harm_rate",
            "max_POI_corruption_rate", "max_additional_caps", "max_new_severe_truncations", "float_guard"]
    checks["safety_copied_exactly"] = all(c["safety"][k] == f["TTA1"]["thresholds"][k] == t0["decision"]["thresholds"][k] for k in keys)
    M = c["mechanism"]
    checks["mechanism_thresholds"] = (M["gap_rows_decreased_fraction_min"] == 0.75 and M["gap_median_relative_reduction_min"] == 0.25
                                      and M["movement_closer_min"] == 4 and M["movement_farther_max"] == 1
                                      and M["movement_pooled_distance_reduction_min"] == 0.15 and c["A2_comparison"]["max_MER_worse_than_A2"] == 0.01)
    checks["kl_orientation_and_teacher"] = "forward KL(q_AUTO || p_FORCED)" == c["A3"]["kl_direction"] and c["A3"]["student_condition"] == [50258, 50260, 50360, 50364]
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False and plan["ids"] == c["panel"]["ids"]
    from transformers import GenerationConfig, WhisperProcessor
    proc = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    gen = GenerationConfig.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    t1plan = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())
    t1post = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/post_audit.json").read_text())
    ok = t1post["verdict"] == "P2_TTA1_AUDIT: PASS"
    for r in plan["rows"]:
        pr = next(p for p in panel["rows"] if p["utterance_id"] == r["utterance_id"])
        fr, ar = json.loads((ROOT / pr["forced_path"]).read_text()), json.loads((ROOT / pr["auto_path"]).read_text())
        t1r = next(x for x in t1plan["rows"] if x["utterance_id"] == r["utterance_id"])
        a2rel = (f"results/inference_cf/p2tta_funnel/tta1/run1/rows/{t1plan['new_ids'].index(r['utterance_id']):02d}.json"
                 if t1r["origin"] == "new" else t1r["ancestor"]["row"])
        a2 = json.loads((ROOT / a2rel).read_text())["objectives"]["A2"]
        ok &= (r["y_B"] == fr["systems"]["S0"]["tokens"] and r["y_A"] == ar["systems"]["B0_AUTO"]["tokens"]
               and r["y_A_valid_mask"] == tau.own_valid(r["y_A"], sup, beg, proc.tokenizer) and r["A2"]["row"] == a2rel
               and r["A2"]["tokens"] == a2["tokens"] and r["A2"]["text"] == a2["text"] and r["d_BA"] == tau.own_lev(r["y_B"], r["y_A"])
               and r["d_A2A"] == tau.own_lev(a2["tokens"], r["y_A"]))
    checks["teachers_reuse_masks_distances"] = bool(ok)
    src = (ROOT / "experiments/inference_cf_p2tta_a3.py").read_text()
    code = "\n".join(ast.get_source_segment(src, n) or "" for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("cmd_run", "cmd_prepare"))
    checks["runner_reference_steering_free"] = not any(t in code for t in ("load_references", "inference_cf_p2_evaluate", "corpus_metrics",
                                                                          "ReadoutDirection", "native_lid", "_edit_hook", "seq_decode", "num_beams=",
                                                                          "do_sample=True", "clip_grad", "lr_scheduler", "GradScaler", "autocast"))
    checks["final_decode_forced"] = "dec = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)" in src
    mod = (ROOT / "src/csasr/inference_cf/soft_auto_tta.py").read_text()
    checks["optimizer_inherited_in_code"] = "torch.optim.AdamW(list(masters.values()), **OPTIM)" in mod and "from csasr.inference_cf.episodic_tta import" in mod
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2tta_a3_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|soft_auto_tta|episodic_tta)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2tta_a3.py").exists()
    v = "PASS_TO_P2_TTA_A3" if all(checks.values()) else "BLOCK_BEFORE_P2_TTA_A3"
    return {"schema": "p2_tta_a3_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def cmd_post(args) -> dict:
    import torch
    from transformers import WhisperProcessor
    from experiments.inference_cf_p2_evaluate import load_references
    c = cfg()
    S = c["safety"]
    G = S["float_guard"]
    run = ROOT / args.run
    ana = json.loads((ROOT / args.analysis).read_text())
    m = json.loads((run / "manifest.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    seal = json.loads((BASE / "output_seal.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "seal_committed_unchanged": committed("results/inference_cf/p2tta_a3/output_seal.json") and all(sha(ROOT / p) == h for p, h in seal["files"].items())
              and seal["manifest_hash"] == m["manifest_hash"],
              "analysis_bound": ana["manifest_hash"] == m["manifest_hash"] and ana["output_seal_hash"] == seal["seal_hash"],
              "panel": own_panel(c) == [(r["utterance_id"], r["dialogue_id"], r["group"]) for r in json.loads((ROOT / PANEL).read_text())["rows"]]
              and m["ids"] == c["panel"]["ids"] == plan["ids"],
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"]
              and rt["nonln_hash_start"] == rt["nonln_hash_end"] and rt["model_grads_none"]}
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    sup, beg = plan["suppression"]["suppress"], plan["suppression"]["begin"]
    with np.load(run / "theta0_ln_fp32.npz") as z:
        th0 = [z[f"p{i:03d}"] for i in range(194)]
    checks["theta0_hash"] = tau.bf16_hash(th0) == rt["theta0_ln_hash"]
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(24)]
    P = {r["utterance_id"]: r for r in plan["rows"]}
    bad = {"rows": [], "dcond": [], "update": [], "reset": [], "output": [], "distance": []}
    own = []
    for i, (r, u) in enumerate(zip(rows, m["ids"])):
        p = P[u]
        o = r["A3"]
        if not (r["status"] == "ok" and r["identity"] == u and r["theta0_forced"]["tokens_equal_S0"] and r["auto_condition"]["replay_text_equal"]
                and r["auto_condition"]["cA"] == [50258, r["auto_condition"]["lang_id"], 50360, 50364]
                and r["auto_condition"]["identical_to_cB"] == (r["auto_condition"]["cA"] == [50258, 50260, 50360, 50364])):
            bad["rows"].append(u)
        mk = tau.own_valid(p["y_A"], sup, beg, tok)
        noop = r["auto_condition"]["identical_to_cB"] or sum(mk) == 0
        dc = []
        for k3 in range(3):
            v = [x for x, ok in zip(o["positions"][k3]["kl"], mk) if ok]
            dc.append(float(np.mean(np.array(v, dtype=np.float64))) if v else 0.0)
        if any(abs(a - b) > 1e-5 for a, b in zip(dc, o["d_cond"])) or o["noop"] != noop or (not noop and o["losses"] != o["d_cond"]) \
                or o["steps"] != 2 or len(o["grad_l2"]) != 2 or not all(o["finite"]):
            bad["dcond"].append(u)
        with np.load(run / f"rows/{i:02d}_final_masters_fp32.npz") as z:
            mast = [z[f"A3_p{j:03d}"] for j in range(194)]
        ml2 = math.sqrt(sum(float(((ma.astype(np.float32) - t0).astype(np.float64) ** 2).sum()) for ma, t0 in zip(mast, th0)))
        eff = [torch.from_numpy(ma).to(torch.bfloat16).float().numpy() - t0 for ma, t0 in zip(mast, th0)]
        if abs(ml2 - o["master_delta_l2"]) > 1e-9 * max(1, ml2) or int(sum(int((e != 0).sum()) for e in eff)) != o["effective_changed_scalars"] \
                or (noop and ml2 != 0):
            bad["update"].append(u)
        if not (o["reset_ok"] and o["end_hash"] == rt["theta0_ln_hash"] and o["other_versions_unchanged"]):
            bad["reset"].append(u)
        if o["length"] != len(o["tokens"]) or (o["terminated"] == "cap") != (len(o["tokens"]) == 200) or \
                o["text"] != tok.decode(o["tokens"], skip_special_tokens=True) or o["equal_B0_FORCED"] != (o["tokens"] == p["y_B"]) or \
                (noop and o["tokens"] != p["y_B"]):
            bad["output"].append(u)
        dBA, dA3A = tau.own_lev(p["y_B"], p["y_A"]), tau.own_lev(o["tokens"], p["y_A"])
        dA2A = tau.own_lev(p["A2"]["tokens"], p["y_A"])
        if (dBA, dA3A, dA2A) != (o["d_BA"], o["d_A3A"], o["d_A2A"]):
            bad["distance"].append(u)
        own.append({"u": u, "g": p["group"], "dc": dc, "dBA": dBA, "dA3A": dA3A, "dA2A": dA2A})
    for k, v in bad.items():
        checks[f"{k}_recomputed"] = not v
    la = rows[0]["live_audit"]
    checks["live_audit"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    checks["compute"] = cnt["A3"]["optimizer_steps"] == 48 and cnt["A3"]["backwards"] == 48 and cnt["audit"]["backwards"] == 1 and cnt["final_decodes"] == 24
    D = [x for x in own if x["g"] == "D"]
    M = c["mechanism"]
    rel = [(x["dc"][0] - x["dc"][2]) / max(x["dc"][0], 1e-12) for x in D]
    dec = sum(x["dc"][2] < x["dc"][0] for x in D)
    gap = dec >= math.ceil(M["gap_rows_decreased_fraction_min"] * len(D)) and float(np.median(rel)) >= M["gap_median_relative_reduction_min"] - 1e-12
    closer = sum(x["dA3A"] < x["dBA"] for x in D)
    farther = sum(x["dA3A"] > x["dBA"] for x in D)
    Rd = 1 - sum(x["dA3A"] for x in D) / sum(x["dBA"] for x in D)
    mov = (closer >= M["movement_closer_min"] and farther <= M["movement_farther_max"]) or Rd >= M["movement_pooled_distance_reduction_min"] - 1e-12
    ag = ana["mechanics"]
    checks["mechanics_agree"] = (ag["gap"]["pass"] == gap and ag["gap"]["rows_decreased"] == dec and abs(ag["gap"]["median_relative_reduction"] - float(np.median(rel))) <= 1e-6
                                 and ag["movement_A3"]["pass"] == mov and (ag["movement_A3"]["closer"], ag["movement_A3"]["farther"]) == (closer, farther)
                                 and abs(ag["movement_A3"]["R_dist"] - Rd) <= 1e-12)
    refs = load_references()
    ids = m["ids"]
    R = [refs[u]["reference"] for u in ids]
    H = {"B0_FORCED": [P[u]["y_B_text"] for u in ids], "B0_AUTO": [P[u]["y_A_text"] for u in ids], "A2": [P[u]["A2"]["text"] for u in ids],
         "A3": [r["A3"]["text"] for r in rows]}
    C = {k: np.array([tau.counts(a, b) for a, b in zip(R, H[k])], dtype=float) for k in H}
    rate = lambda c_, n, d: c_[:, n].sum() / c_[:, d].sum()
    mine = {k: {"pier": rate(C[k], 0, 1), "mer": rate(C[k], 2, 3), "zh_cer": rate(C[k], 4, 5), "en_wer": rate(C[k], 6, 7),
                "poi_errors": int(C[k][:, 0].sum())} for k in C}
    checks["metrics_agree"] = max(abs(mine[k][n] - ana["metrics"][k][n]) for k in mine for n in ("pier", "mer", "zh_cer", "en_wer")) <= 1e-12 and \
        all(mine[k]["poi_errors"] == ana["metrics"][k]["num_poi_errors"] for k in mine)
    tr = tau.transitions(R, H["B0_FORCED"], H["A3"])
    tb = [P[u]["y_B_terminated"] for u in ids]
    ta = [r["A3"]["terminated"] for r in rows]
    added = sum(a == "cap" and b != "cap" for a, b in zip(ta, tb))
    sev = sum(len(P[u]["y_B"]) >= 10 and r["A3"]["terminated"] == "eos" and len(r["A3"]["tokens"]) <= math.floor(0.5 * len(P[u]["y_B"])) for r, u in zip(rows, ids))
    safe = {"MER": mine["A3"]["mer"] - mine["B0_FORCED"]["mer"] <= S["max_MER_increase"] + G,
            "ZH_CER": mine["A3"]["zh_cer"] - mine["B0_FORCED"]["zh_cer"] <= S["max_ZH_CER_increase"] + G,
            "ZH_ret": tr["zr"] is None or tr["zr"] >= S["min_matrix_ZH_retention"] - G, "EN_ret": tr["er"] is None or tr["er"] >= S["min_embedded_EN_retention"] - G,
            "outside": tr["ohr"] is None or tr["ohr"] <= S["max_outside_POI_harm_rate"] + G,
            "corruption": tr["corr_rate"] is None or tr["corr_rate"] <= S["max_POI_corruption_rate"] + G,
            "caps": added <= S["max_additional_caps"], "severe": sev <= S["max_new_severe_truncations"]}
    pa3 = ana["points"]["A3"]
    checks["safety_points_agree"] = (tr["corrections"], tr["corruptions"]) == (pa3["corrections"], pa3["corruptions"]) and tr["zr"] == pa3["zh_retention"] \
        and tr["er"] == pa3["en_retention"] and tr["ohr"] == pa3["outside_harm_rate"] and added == pa3["added_caps"] and sev == pa3["new_severe_truncations"]
    prom = mine["A3"]["poi_errors"] <= mine["A2"]["poi_errors"] and mine["A3"]["mer"] - mine["A2"]["mer"] <= c["A2_comparison"]["max_MER_worse_than_A2"] + 1e-12
    valid_pre = all(checks.values())
    lab = ("P2_TTA_A3_INVALID" if not valid_pre else "P2_TTA_A3_GAP_NOT_CLOSED" if not gap else "P2_TTA_A3_SEQUENCE_LEVERAGE_LIMIT" if not mov
           else "P2_TTA_A3_TEACHER_UNSAFE" if not all(safe.values()) else "P2_TTA_A3_CONDITIONING_ONLY" if not prom else "P2_TTA_A3_PROMISING")
    checks["label_agrees"] = lab == ana["label"]
    v = "P2_TTA_A3_AUDIT: PASS" if all(checks.values()) else "P2_TTA_A3_AUDIT: BLOCK"
    return {"schema": "p2_tta_a3_post_audit_v1", "verdict": v, "label": lab, "checks": checks,
            "independent": {"metrics": mine, "gap": {"rows_decreased": dec, "median_rel": float(np.median(rel)), "pass": gap},
                            "movement": {"closer": closer, "farther": farther, "R_dist": Rd, "pass": mov}, "safety": safe, "promising_A2": prom},
            "failures": {k: v_[:10] for k, v_ in bad.items()}, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--run", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "post": cmd_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
