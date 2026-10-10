#!/usr/bin/env python
"""Independent P2-TTA-FUNNEL auditor (frozen contract ff75e1a: P2_TTA_FUNNEL_SPEC.md, P2_TTA_FUNNEL_CODEX_DESIGN.md,
configs/inference_cf/p2_tta_funnel.json). Does NOT import any primary analysis/decision/objective code
(``*_analyze``, ``csasr.inference_cf.episodic_tta``, the funnel runner); reuses the independent TTA0 auditor and
locked canonical primitives only.

r-gate -> PASS_TO_P2_TTA0_R_EVALUATION | P2_TTA0_R_STILL_INVALID   (no reference access, no model, no adaptation)
r-post -> P2_TTA0_R_AUDIT: PASS | BLOCK
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import experiments.inference_cf_p2tta0_audit as tau

FREEZE = "ff75e1a6b524415d8071749554b66c09f8dd2ad2"
CONFIG = "configs/inference_cf/p2_tta_funnel.json"
FROZEN = ("docs/inference_cf/P2_TTA_FUNNEL_SPEC.md", "docs/inference_cf/P2_TTA_FUNNEL_CODEX_DESIGN.md", CONFIG,
          "docs/inference_cf/P2_TTA_MAP_PANEL24.json", "docs/inference_cf/P2_TTA_MAP_PANEL_CONTRACT.md",
          "docs/inference_cf/P2_TTA1_INHERITANCE_CONTRACT.md")
TTA0 = ROOT / "results/inference_cf/p2tta0"
OUT = ROOT / "results/inference_cf/p2tta_funnel"
R_MAP = {"P2_TTA0_INVALID": "P2_TTA0_R_STILL_INVALID", "P2_TTA0_OBJECTIVE_SELECTED": "P2_TTA0_R_OBJECTIVE_SELECTED",
         "P2_TTA0_NO_VIABLE_OBJECTIVE": "P2_TTA0_R_NO_VIABLE_OBJECTIVE"}


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def frozen_checks(c: dict) -> dict:
    ch = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in FROZEN}
    ch["historical_sources_at_starting_commit"] = all(tau.blob_sha(c["starting_commit"], p) == h for p, h in c["source_sha256"].items())
    ch["immutable_configs_panels_current"] = all(sha(ROOT / p) == c["source_sha256"][p] for p in c["source_sha256"]
                                                 if p.startswith(("configs/", "docs/", "results/")))
    return ch


# ---- TTA0-R gate ------------------------------------------------------------------------------------------------

def cmd_r_gate(args) -> dict:
    c = cfg()
    R = c["TTA0_R"]
    checks = frozen_checks(c)
    checks["sealed_bytes"] = all(sha(ROOT / p) == h for p, h in R["sealed_sha256"].items())
    checks["baseline20_bytes"] = all(sha(ROOT / b["forced_row"]) == b["forced_row_sha256"] and sha(ROOT / b["auto_row"]) == b["auto_row_sha256"]
                                     for b in R["baseline20"])
    t0 = json.loads((ROOT / "configs/inference_cf/p2_tta0.json").read_text())
    checks["original_decision_exact_copy"] = t0["decision"] == c["original_tta0_decision"]
    checks["only_numerical_tolerance_repaired"] = (R["relative_gradient_tolerance"] == 0.02 == c["audit"]["relative_gradient_tolerance"]
                                                   and R["previous_relative_gradient_tolerance"] == 0.001 and R["absolute_loss_tolerance"] == 1e-5
                                                   and R["absolute_gradient_floor"] == 1e-8 and R["new_gpu_jobs"] == 0 and R["no_new_adaptation"] is True
                                                   and c["common"]["optimization"] == t0["optimization"]
                                                   and c["common"]["trainables"] == t0["trainables"]
                                                   and c["common"]["teacher_forcing"] == t0["teacher_forcing"]
                                                   and R["baseline20"] == t0["reuse"]["rows"])
    checks["no_new_adaptation_outputs"] = sorted(x.name for x in TTA0.iterdir()) == sorted(
        ["prerun_audit.json", "pseudo_sealed.json", "run1", "run1_audit.json", "run1_invalid_record.json", "run1_live_audit_diagnostic.json"])
    # independent recomputation of sealed mechanics with the historical independent auditor (reference-free)
    inv = tau.cmd_invalid(SimpleNamespace(run="results/inference_cf/p2tta0/run1", record="results/inference_cf/p2tta0/run1_invalid_record.json"))
    eng = inv["engineering"]
    for k, v in inv["checks"].items():
        if k not in ("label_agrees", "failed_checks_agree", "runtime_sources_unchanged"):
            checks[f"sealed:{k}"] = bool(v)
    # runtime code = manifest sources other than post-run analysis/audit extensions and tests (attempt 1 wrongly
    # required the post-run-extended test file to match; tests are neither sealed inputs nor executed in run1)
    man = json.loads((TTA0 / "run1/manifest.json").read_text())
    runtime = [p for p in man["sources"] if not p.endswith(("_analyze.py", "_audit.py")) and not p.startswith("tests/")]
    checks["sealed:runtime_code_unchanged"] = all("sha256:" + sha(ROOT / p) == man["sources"][p] for p in runtime) and \
        "experiments/inference_cf_p2tta0.py" in runtime and "src/csasr/inference_cf/episodic_tta.py" in runtime
    for k, v in eng.items():
        if k != "live_objective_gradient_check":
            checks[f"mechanics:{k}"] = bool(v)          # trainable/reset identities, theta0, updates, losses, outputs, counts
    rows0 = json.loads((TTA0 / "run1/rows/00.json").read_text())
    rt = json.loads((TTA0 / "run1/runtime.json").read_text())
    rows = [json.loads((TTA0 / f"run1/rows/{i:02d}.json").read_text()) for i in range(20)]
    checks["finite_gradients"] = all(all(o["finite"]) and all(math.isfinite(x) for x in o["grad_l2"] + o["losses"])
                                     for r in rows for o in r["objectives"].values())
    checks["trainable_identity"] = json.loads((TTA0 / "run1/manifest.json").read_text())["trainables"] == \
        [p["name"] for p in c["common"]["trainables"]["parameters"]] and len(c["common"]["trainables"]["parameters"]) == 194
    checks["reset_identities"] = all(o["start_hash"] == o["end_hash"] == rt["theta0_ln_hash"] and o["reset_ok"] for r in rows
                                     for o in r["objectives"].values()) and rt["reset_final_ok"] and rt["nonln_unchanged"]
    checks["output_hashes"] = all(r["final_masters_npz_sha256"] == "sha256:" + sha(TTA0 / f"run1/rows/{i:02d}_final_masters_fp32.npz")
                                  for i, r in enumerate(rows))
    la, live = rows0["live_audit"], {}
    for k in ("A1", "A2"):
        x = la[k]
        live[k] = {"loss_abs_diff": abs(x["primary_loss"] - x["auditor_loss"]), "grad_rel": x["grad_diff_l2"] / x["auditor_grad_l2"],
                   "loss_ok": abs(x["primary_loss"] - x["auditor_loss"]) <= R["absolute_loss_tolerance"],
                   "grad_ok_repaired": x["grad_diff_l2"] <= max(R["absolute_gradient_floor"], R["relative_gradient_tolerance"] * x["auditor_grad_l2"]),
                   "primary_is_sealed_L0": x["primary_loss"] == rows0["objectives"][k]["losses"][0]}
    checks["live_gradient_agreement_repaired"] = all(v["loss_ok"] and v["grad_ok_repaired"] and v["primary_is_sealed_L0"] for v in live.values())
    checks["runtime_invalid_only_live_audit"] = all("live audit" in x for x in rt.get("invalid", []))
    diag = json.loads((TTA0 / "run1_live_audit_diagnostic.json").read_text())["objectives"]
    checks["float64_formula_equivalence"] = all(diag[k]["relative_l2_diff"]["A64_vs_P64"] == 0.0 for k in ("A1", "A2"))
    checks["fp32_repeat_determinism"] = all(diag[k]["relative_l2_diff"]["P32_repeat_vs_P32"] == 0.0 for k in ("A1", "A2"))
    disc = [diag[k]["relative_l2_diff"][n] for k in ("A1", "A2") for n in ("P32_vs_P64", "A32_vs_P64", "A32_vs_P32")]
    checks["recorded_discrepancies_within_repaired_tolerance"] = max(disc) <= R["relative_gradient_tolerance"]
    src = (ROOT / "experiments/inference_cf_p2tta_funnel_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|episodic_tta|inference_cf_p2tta_funnel\b)", src, re.M) is None
    v = "PASS_TO_P2_TTA0_R_EVALUATION" if all(checks.values()) else "P2_TTA0_R_STILL_INVALID"
    return {"schema": "p2_tta0_r_gate_v1", "verdict": v, "checks": checks, "live": live, "recorded_discrepancies": disc,
            "references_loaded": False, "model_loaded": False, "adaptation_run": False, "git_commit": _git("rev-parse", "HEAD")}


def cmd_r_post(args) -> dict:
    c = cfg()
    gate_rel = "results/inference_cf/p2tta_funnel/tta0_r/gate_audit.json"
    gate = json.loads((ROOT / gate_rel).read_text())
    ana = json.loads((ROOT / args.analysis).read_text())
    base = tau.cmd_post(SimpleNamespace(run="results/inference_cf/p2tta0/run1", analysis=args.analysis,
                                        rel_grad_tol=c["TTA0_R"]["relative_gradient_tolerance"]))
    checks = {f"tta0_post:{k}": bool(v) for k, v in base["checks"].items()}
    checks["gate_passed_and_committed"] = gate["verdict"] == "PASS_TO_P2_TTA0_R_EVALUATION" and committed(gate_rel)
    checks["analysis_after_gate"] = ana.get("gate_sha256") == sha(ROOT / gate_rel)
    checks["sealed_bytes_unchanged"] = all(sha(ROOT / p) == h for p, h in c["TTA0_R"]["sealed_sha256"].items())
    own_r = R_MAP[base["label"]]
    checks["r_label_agrees"] = own_r == ana["r_label"]
    checks["selected_agrees"] = base["selected"] == ana["selected"]
    v = "P2_TTA0_R_AUDIT: PASS" if all(checks.values()) else "P2_TTA0_R_AUDIT: BLOCK"
    return {"schema": "p2_tta0_r_post_audit_v1", "verdict": v, "r_label": own_r, "selected": base["selected"], "labels": base["labels"],
            "checks": checks, "independent": base["independent"], "failures": base["failures"], "git_commit": _git("rev-parse", "HEAD")}


# ---- TTA1 -------------------------------------------------------------------------------------------------------

T1 = ROOT / "results/inference_cf/p2tta_funnel/tta1"
PLAN_REL = "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json"
SEL_REL = "results/inference_cf/p2tta_funnel/tta0_r/selected_objective.json"


def cmd_tta1_pre(args) -> dict:
    import ast
    from transformers import GenerationConfig, WhisperProcessor
    c = cfg()
    checks = frozen_checks(c)
    plan = json.loads((ROOT / PLAN_REL).read_text())
    sel = json.loads((ROOT / SEL_REL).read_text())
    raud = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta0_r/post_audit.json").read_text())
    checks["selection_committed"] = committed(SEL_REL) and committed("results/inference_cf/p2tta_funnel/tta0_r/post_audit.json")
    checks["predecessor_selection_independent"] = (raud["verdict"] == "P2_TTA0_R_AUDIT: PASS" and raud["r_label"] == "P2_TTA0_R_OBJECTIVE_SELECTED"
                                                   and raud["selected"] == "A2" and raud["labels"]["A2"] == "TTA0_AC_VIABLE"
                                                   and raud["labels"]["A1"] != "TTA0_EM_VIABLE" and sel["objective"]["id"] == "A2"
                                                   and sel["independent_audit"]["sha256"] == sha(ROOT / "results/inference_cf/p2tta_funnel/tta0_r/post_audit.json"))
    t0 = json.loads((ROOT / "configs/inference_cf/p2_tta0.json").read_text())
    checks["inheritance_exact"] = (sel["optimizer"] == {k: c["common"]["optimization"][k] for k in sel["optimizer"]}
                                   and [p["name"] for p in sel["trainables"]["parameters"]] == [p["name"] for p in c["common"]["trainables"]["parameters"]]
                                   and len(sel["trainables"]["parameters"]) == 194 and sel["objective"]["loss"] == t0["objectives"]["A2"]["loss"]
                                   and sel["teacher_forcing"] == c["common"]["teacher_forcing"] and sel["decode"] == c["common"]["decode"]
                                   and plan["objective"] == "A2" and plan["selection_sha256"] == sha(ROOT / SEL_REL))
    checks["plan_committed_and_hashed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["outcomes_computed"] is False and plan["references_used"] is False
    panel = json.loads((ROOT / c["TTA1"]["panel"]).read_text())
    ids = [r["utterance_id"] for r in panel["rows"]]
    checks["panel100"] = sha(ROOT / c["TTA1"]["panel"]) == c["TTA1"]["panel_sha256"] and plan["ids"] == ids and len(set(ids)) == 100
    base = {b["utterance_id"]: b for b in c["baseline100"]}
    proc = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    gen = GenerationConfig.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    ok = True
    t0seal = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta0/pseudo_sealed.json").read_text())["rows"]}
    for r in plan["rows"]:
        u = r["utterance_id"]
        b = base[u]
        fr, ar = json.loads((ROOT / b["forced_path"]).read_text()), json.loads((ROOT / b["auto_path"]).read_text())
        ok &= (sha(ROOT / b["forced_path"]) == b["forced_sha256"] == r["forced_row_sha256"] and sha(ROOT / b["auto_path"]) == b["auto_sha256"]
               and r["y_B"] == fr["systems"]["S0"]["tokens"] and r["y_A"] == ar["systems"]["B0_AUTO"]["tokens"]
               and r["y_B_valid_mask"] == tau.own_valid(r["y_B"], sup, beg, proc.tokenizer)
               and r["y_A_valid_mask"] == tau.own_valid(r["y_A"], sup, beg, proc.tokenizer) and r["group"] == b["group"])
        if r["origin"] != "new":
            a = r["ancestor"]
            ok &= (u in t0seal and c["TTA0_R"]["sealed_sha256"][a["row"]] == a["row_sha256"] == sha(ROOT / a["row"])
                   and c["TTA0_R"]["sealed_sha256"][a["masters_npz"]] == a["masters_npz_sha256"])
    checks["baselines_teachers_masks_ancestors"] = bool(ok)
    checks["reuse_all_compatible_no_filtering"] = sorted(plan["reuse_ids"]) == sorted(t0seal) and len(plan["new_ids"]) == 80 and \
        set(plan["reuse_ids"]) | set(plan["new_ids"]) == set(ids)
    P = plan["planned"]
    checks["budget"] = P["optimizer_steps"] == 160 <= c["compute"]["TTA1_optimizer_steps_max"] and P["backwards"] == 161
    rsrc = (ROOT / "experiments/inference_cf_p2tta_funnel.py").read_text()
    code = "\n".join(ast.get_source_segment(rsrc, n) or "" for n in ast.parse(rsrc).body if isinstance(n, ast.FunctionDef) and n.name.startswith("cmd_tta1"))
    checks["runner_reference_steering_free"] = not any(f in code for f in ("load_references", "inference_cf_p2_evaluate", "corpus_metrics", "ReadoutDirection",
                                                                          "native_lid", "_edit_hook", "seq_decode", "do_sample=True", "num_beams=",
                                                                          "clip_grad", "lr_scheduler", "GradScaler", "autocast", '"A1"', '"A3"'))
    checks["no_tta1_outcome"] = not (T1 / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2tta_funnel_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|episodic_tta|inference_cf_p2tta_funnel\b)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2tta_funnel.py").exists()
    v = "PASS_TO_P2_TTA1" if all(checks.values()) else "BLOCK_BEFORE_P2_TTA1"
    return {"schema": "p2_tta1_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def cmd_tta1_post(args) -> dict:
    import numpy as np
    import torch
    from transformers import WhisperProcessor
    from experiments.inference_cf_p2_evaluate import load_references
    c = cfg()
    TH = c["TTA1"]["thresholds"]
    G = TH["float_guard"]
    run = ROOT / args.run
    ana = json.loads((ROOT / args.analysis).read_text())
    m = json.loads((run / "manifest.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    oseal = json.loads((T1 / "output_seal.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "output_seal_committed_unchanged": committed("results/inference_cf/p2tta_funnel/tta1/output_seal.json")
              and all(sha(ROOT / p) == h for p, h in oseal["files"].items()) and oseal["manifest_hash"] == m["manifest_hash"],
              "analysis_bound": ana["manifest_hash"] == m["manifest_hash"] and ana["output_seal_hash"] == oseal["seal_hash"],
              "plan": m["plan_hash"] == plan["plan_hash"],
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"]
              and rt["nonln_hash_start"] == rt["nonln_hash_end"] and rt["model_grads_none"]}
    proc = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    tok = proc.tokenizer
    sup, beg = plan["suppression"]["suppress"], plan["suppression"]["begin"]
    with np.load(run / "theta0_ln_fp32.npz") as z:
        th0 = [z[f"p{i:03d}"] for i in range(194)]
    checks["theta0_hash"] = tau.bf16_hash(th0) == rt["theta0_ln_hash"]
    P = {r["utterance_id"]: r for r in plan["rows"]}
    ids = plan["ids"]
    objs, bad = {}, {"loss": [], "update": [], "reset": [], "output": [], "theta0": []}
    for u in ids:
        r = P[u]
        if r["origin"] == "new":
            i = m["new_ids"].index(u)
            row = json.loads((run / f"rows/{i:02d}.json").read_text())
            npz = run / f"rows/{i:02d}_final_masters_fp32.npz"
            if not (row["status"] == "ok" and row["identity"] == u and row["theta0_decode"]["tokens_equal_S0"] and row["theta0_decode"]["terminated_equal_S0"]):
                bad["theta0"].append(u)
        else:
            a = r["ancestor"]
            row = json.loads((ROOT / a["row"]).read_text())
            npz = ROOT / a["masters_npz"]
            if sha(ROOT / a["row"]) != c["TTA0_R"]["sealed_sha256"][a["row"]]:
                bad["output"].append((u, "ancestor"))
        o = row["objectives"]["A2"]
        objs[u] = o
        mk = tau.own_valid(r["y_A"], sup, beg, tok)
        if o["steps"] != 2 or len(o["losses"]) != 3 or not all(o["finite"]) or o["valid"] != sum(mk):
            bad["loss"].append((u, "schedule"))
        for k3 in range(3):
            v = [-t for t, ok in zip(o["positions"][k3]["target_logprob"], mk) if ok]
            L = float(np.mean(np.array(v, dtype=np.float64))) if v else 0.0
            if abs(L - o["losses"][k3]) > 1e-5:
                bad["loss"].append((u, k3))
        with np.load(npz) as z:
            mast = [z[f"A2_p{j:03d}"] for j in range(194)]
        ml2 = math.sqrt(sum(float(((ma.astype(np.float32) - t0).astype(np.float64) ** 2).sum()) for ma, t0 in zip(mast, th0)))
        eff = [torch.from_numpy(ma).to(torch.bfloat16).float().numpy() - t0 for ma, t0 in zip(mast, th0)]
        if abs(ml2 - o["master_delta_l2"]) > 1e-9 * max(1, ml2) or int(sum(int((e != 0).sum()) for e in eff)) != o["effective_changed_scalars"]:
            bad["update"].append(u)
        if not (o["reset_ok"] and o["start_hash"] == o["end_hash"] == rt["theta0_ln_hash"] and o["other_versions_unchanged"]):
            bad["reset"].append(u)
        if o["length"] != len(o["tokens"]) or (o["terminated"] == "cap") != (len(o["tokens"]) == 200) or \
                o["text"] != tok.decode(o["tokens"], skip_special_tokens=True) or o["vs_B0_FORCED"]["levenshtein"] != tau.own_lev(o["tokens"], r["y_B"]) or \
                o["vs_B0_AUTO"]["levenshtein"] != tau.own_lev(o["tokens"], r["y_A"]):
            bad["output"].append(u)
    for k, v in bad.items():
        checks[f"{k}_recomputed"] = not v
    la = json.loads((run / "rows/00.json").read_text())["live_audit"]["A2"]
    checks["live_audit_2pct"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt, PL = rt["counters"], plan["planned"]
    checks["compute_counts"] = (cnt["A2"]["optimizer_steps"] == PL["optimizer_steps"] and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] == PL["backwards"]
                                and cnt["A2"]["teacher_forwards"] + cnt["common_theta0_forwards"] + cnt["audit"]["forwards"] == PL["teacher_forwards"]
                                and cnt["A2"]["final_decodes"] == PL["final_decodes"] and cnt["theta0_integrity_decodes"] == PL["theta0_integrity_decodes"])
    refs = load_references()
    R = [refs[u]["reference"] for u in ids]
    D = [refs[u]["dialogue_id"] for u in ids]
    H = {"B0_FORCED": [P[u]["y_B_text"] for u in ids], "B0_AUTO": [P[u]["y_A_text"] for u in ids], "SELECTED_TTA": [objs[u]["text"] for u in ids]}
    C = {k: np.array([tau.counts(a, b) for a, b in zip(R, H[k])], dtype=float) for k in H}
    rate = lambda c_, n, d: c_[:, n].sum() / c_[:, d].sum()
    mine = {k: {"pier": rate(C[k], 0, 1), "mer": rate(C[k], 2, 3), "zh_cer": rate(C[k], 4, 5), "en_wer": rate(C[k], 6, 7),
                "poi_errors": int(C[k][:, 0].sum())} for k in C}
    checks["metrics_agree"] = max(abs(mine[k][n] - ana["metrics"][k][n]) for k in mine for n in ("pier", "mer", "zh_cer", "en_wer")) <= 1e-12 and \
        all(mine[k]["poi_errors"] == ana["metrics"][k]["num_poi_errors"] for k in mine)
    tr = tau.transitions(R, H["B0_FORCED"], H["SELECTED_TTA"])
    term_b = [P[u]["y_B_terminated"] for u in ids]
    term_s = [objs[u]["terminated"] for u in ids]
    added_caps = sum(a == "cap" and b != "cap" for a, b in zip(term_s, term_b))
    sev = sum(len(P[u]["y_B"]) >= 10 and objs[u]["terminated"] == "eos" and len(objs[u]["tokens"]) <= math.floor(0.5 * len(P[u]["y_B"])) for u in ids)
    S = "SELECTED_TTA"
    safety = {"MER": mine[S]["mer"] - mine["B0_FORCED"]["mer"] <= TH["max_MER_increase"] + G,
              "ZH_CER": mine[S]["zh_cer"] - mine["B0_FORCED"]["zh_cer"] <= TH["max_ZH_CER_increase"] + G,
              "ZH_ret": tr["zr"] is None or tr["zr"] >= TH["min_matrix_ZH_retention"] - G,
              "EN_ret": tr["er"] is None or tr["er"] >= TH["min_embedded_EN_retention"] - G,
              "outside": tr["ohr"] is None or tr["ohr"] <= TH["max_outside_POI_harm_rate"] + G,
              "corruption": tr["corr_rate"] is None or tr["corr_rate"] <= TH["max_POI_corruption_rate"] + G,
              "caps": added_caps <= TH["max_additional_caps"], "severe": sev <= TH["max_new_severe_truncations"]}
    net = mine["B0_FORCED"]["poi_errors"] - mine[S]["poi_errors"]
    benefit = net >= TH["min_net_POI_error_reduction"] and mine["B0_FORCED"]["pier"] - mine[S]["pier"] >= TH["min_absolute_PIER_reduction"] - G
    ap = ana["point"]
    checks["points_agree"] = (tr["corrections"], tr["corruptions"]) == (ana["transitions"]["corrections"], ana["transitions"]["corruptions"]) \
        and tr["zr"] == ap["zh_retention"] and tr["er"] == ap["en_retention"] and tr["ohr"] == ap["outside_harm_rate"] \
        and added_caps == ap["added_caps"] and sev == ap["new_severe_truncations"]
    keys = sorted(set(D))
    grp = {k: [i for i, d in enumerate(D) if d == k] for k in keys}
    rng = np.random.default_rng(240924)
    drs = [sum((grp[keys[j]] for j in rng.integers(0, 20, 20)), []) for _ in range(2000)]
    bdiff = []
    for n, (a, b) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}.items():
        for base_ in ("B0_FORCED", "B0_AUTO"):
            vals = [C[S][ix, a].sum() / C[S][ix, b].sum() - C[base_][ix, a].sum() / C[base_][ix, b].sum()
                    for ix in drs if C[S][ix, b].sum() and C[base_][ix, b].sum()]
            ref_ = ana["bootstrap"][f"{S}_minus_{base_}"][n]
            bdiff += [abs(np.quantile(vals, .025) - ref_["ci95"][0]), abs(np.quantile(vals, .975) - ref_["ci95"][1])]
    checks["bootstrap_agrees"] = max(bdiff) <= 1e-9
    valid = all(checks.values())
    lab = "P2_TTA1_INVALID" if not valid else ("P2_TTA1_SEQUENCE_DAMAGE" if not all(safety.values()) else
                                               ("P2_TTA1_SUPPORTED" if benefit else "P2_TTA1_NO_USEFUL_GAIN"))
    checks["label_agrees"] = lab == ana["label"]
    v = "P2_TTA1_AUDIT: PASS" if all(checks.values()) else "P2_TTA1_AUDIT: BLOCK"
    return {"schema": "p2_tta1_post_audit_v1", "verdict": v, "label": lab, "checks": checks,
            "independent": {"metrics": mine, "safety": safety, "net_poi": net, "benefit": benefit, "added_caps": added_caps, "severe": sev},
            "failures": {k: v_[:10] for k, v_ in bad.items()}, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("r-gate").add_argument("--out", required=True)
    p = sub.add_parser("r-post")
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    sub.add_parser("tta1-pre").add_argument("--out", required=True)
    q = sub.add_parser("tta1-post")
    q.add_argument("--run", required=True)
    q.add_argument("--analysis", required=True)
    q.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"r-gate": cmd_r_gate, "r-post": cmd_r_post, "tta1-pre": cmd_tta1_pre, "tta1-post": cmd_tta1_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "r_label", "label", "selected", "labels")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
