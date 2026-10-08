#!/usr/bin/env python
"""Independent P2-A2-MECH0 auditor. Does NOT import the MECH0 runner/analysis or consensus_guard / branch_adjudication / path_decode /
eos_boundary. Own strata/controls from sealed token arrays, own family partition, own norm recomputation from the sealed LN snapshot
archives, own margins/retention/AUTO alignment/severe truncation, canonical counts (TTA0 auditor ``counts`` + canonical corpus_mer
deletions), own F fractions/decomposition/LODO/bootstrap/mechanism label/readiness.

prerun -> PASS_TO_P2_A2_MECH0 | BLOCK_BEFORE_P2_A2_MECH0
post   -> P2_A2_MECH0_AUDIT: PASS | BLOCK  (--phase primary before references; --phase full after secondary)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
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

FREEZE = "9d89640"
CONFIG = "configs/inference_cf/p2_a2_mech0.json"
PANEL = "docs/inference_cf/P2_A2_MECH0_PANEL.json"
FROZEN = ("docs/inference_cf/P2_A2_MECH0_SPEC.md", "docs/inference_cf/P2_A2_MECH0_CODEX_DESIGN.md", PANEL, CONFIG)
BASE = ROOT / "results/inference_cf/p2a2_mech0"
PLAN_REL = "results/inference_cf/p2a2_mech0/plan_sealed.json"
RUNNER = "experiments/inference_cf_p2a2_mech0.py"
ANALYZE = "experiments/inference_cf_p2a2_mech0_analyze.py"
EOS = 50257
G = 1e-12


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def st(t, term):
    return list(t) + ([EOS] if term == "eos" else [])


def severe(base_len, new_len, term):
    return base_len >= 10 and term == "eos" and new_len <= math.floor(0.5 * base_len)


def own_strata():
    Pn = json.loads((ROOT / PANEL).read_text())
    ids = [r["utterance_id"] for r in Pn["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in Pn["rows"]}
    fixed = [r["utterance_id"] for r in json.loads((ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json").read_text())["rows"]]
    B, A, U = {}, {}, {}
    for r in Pn["rows"]:
        p5 = json.loads((ROOT / r["PATH5_output"]["path"]).read_text())
        B[r["utterance_id"]], A[r["utterance_id"]] = p5["theta0"], p5["A2_free"]
        U[r["utterance_id"]] = json.loads((ROOT / r["B0_AUTO"]["path"]).read_text())["systems"]["B0_AUTO"]
    out = {k: [] for k in ("A2_SAME", "A2_DELTA", "EOS_RECOVERY", "EOS_REGRESSION", "CONTENT_DIVERGENCE", "OTHER_INVALID", "AUTO_SAME", "AUTO_DELTA",
                           "AUTO_SAME__A2_SAME", "AUTO_SAME__A2_DELTA", "AUTO_DELTA__A2_SAME", "AUTO_DELTA__A2_DELTA")}
    kmap = {}
    for u in ids:
        sb, sa = st(B[u]["tokens"], B[u]["terminated"]), st(A[u]["tokens"], A[u]["terminated"])
        a2k = "A2_SAME" if sb == sa else "A2_DELTA"
        out[a2k].append(u)
        if sb != sa:
            j = next((i for i in range(min(len(sb), len(sa))) if sb[i] != sa[i]), None)
            kind = "OTHER_INVALID" if j is None else ("EOS_RECOVERY" if sb[j] == EOS and sa[j] != EOS else "EOS_REGRESSION" if sa[j] == EOS and sb[j] != EOS
                                                      else "CONTENT_DIVERGENCE")
            out[kind].append(u)
            kmap[u] = (j, sb[j] if j is not None else None, sa[j] if j is not None else None, kind)
        ak = "AUTO_SAME" if st(U[u]["tokens"], U[u]["terminated"]) == sb else "AUTO_DELTA"
        out[ak].append(u)
        out[f"{ak}__{a2k}"].append(u)
    ctrl = []
    for dd in sorted(set(dlg.values())):
        ctrl += [u for u in ids if dlg[u] == dd and u in set(out["A2_SAME"])][:2]
    out["CONTROL40"] = [u for u in ids if u in set(ctrl)]
    out["MECH_CHANGED"] = list(out["A2_DELTA"])
    out["MECH_PANEL"] = [u for u in ids if u in set(ctrl) | set(out["A2_DELTA"])]
    out["FULL300"], out["FIXED100"], out["NEW200"] = ids, fixed, [u for u in ids if u not in set(fixed)]
    return out, dlg, kmap, B, A, U


def own_mech(valid, eos, non, F_ZH, F_DEL, med_dM2, frac):
    if not valid:
        return "P2_A2_MECH0_INVALID"
    both = F_ZH is not None and F_DEL is not None
    if both and eos["rows"] >= 3 and eos["dlg"] >= 3 and F_ZH >= 0.5 and F_DEL >= 0.5 and med_dM2 is not None and med_dM2 > 0 and frac is not None and frac >= 0.75:
        return "P2_A2_MECH0_TERMINATION_DOMINANT"
    if both and F_ZH < 0.25 and F_DEL < 0.25 and (non["zh"] > 0 or non["mx"] > 0) and non["prow"] >= 3 and non["pdlg"] >= 3:
        return "P2_A2_MECH0_NONTERMINATION_DOMINANT"
    if both:
        mat = lambda g: (g["zh"] >= 5 or g["mx"] >= 5) and g["prow"] >= 2 and g["pdlg"] >= 2
        pos = lambda g: g["zh"] > 0 or g["mx"] > 0
        if (mat(eos) and mat(non)) or ((0.25 <= F_ZH < 0.5 or 0.25 <= F_DEL < 0.5) and pos(eos) and pos(non)):
            return "P2_A2_MECH0_MIXED"
    return "P2_A2_MECH0_NO_CLEAR_MECHANISM"


def _fn_source(rel: str, name: str) -> str:
    src = (ROOT / rel).read_text()
    return next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in FROZEN}
    checks["freeze_pushed"] = subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, "origin/cs-asr-steer-inf"], cwd=ROOT).returncode == 0
    checks["panel_hash"] = sha(ROOT / PANEL) == c["panel_byte_sha256"]
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    checks["A2_code_unmodified"] = all(sha(ROOT / p) == h for p, h in c["instrumentation_provenance"]["original_source_baseline_sha256"].items())
    S, dlg, kmap, B, A, U = own_strata()
    checks["strata_independent"] = (all(fp({"partition": k, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in v]}) == c["partition_sha256"][k]
                                        for k, v in S.items()) and {k: len(v) for k, v in S.items()} == c["counts"]
                                    and len(S["CONTROL40"]) == 40 and len({dlg[u] for u in S["CONTROL40"]}) == 20)
    names = [p["name"] for p in c["A2_inherited"]["trainables"]["parameters"]]
    own = {"SELF": [n for n in names if "self_attn_layer_norm" in n], "CROSS": [n for n in names if "encoder_attn_layer_norm" in n],
           "POST": [n for n in names if "final_layer_norm" in n or n.startswith("model.decoder.layer_norm")]}
    numel = {p["name"]: p["numel"] for p in c["A2_inherited"]["trainables"]["parameters"]}
    checks["family_partition"] = (all(own[f] == c["families"][f] for f in own) and sum(len(v) for v in own.values()) == 194
                                  and set().union(*map(set, own.values())) == set(names) and [sum(numel[n] for n in own[f]) for f in own] == [81920, 81920, 84480])
    plan = json.loads((ROOT / PLAN_REL).read_text())
    p5 = json.loads((ROOT / "results/inference_cf/p2path5/plan_sealed.json").read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False and all(plan["partitions"][k] == v for k, v in S.items())
    checks["runtime_inputs_equal_PATH5"] = [x["runtime"] for x in plan["rows"]] == [x["runtime"] for x in p5["rows"]] and \
        all(set(x["runtime"]) == {"utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask"} for x in plan["rows"])
    rsrc = (ROOT / RUNNER).read_text()
    run_code = _fn_source(RUNNER, "cmd_run")
    obs_node = next(n for n in ast.parse(rsrc).body if isinstance(n, ast.FunctionDef) and n.name == "step_observer")
    body = [b for b in obs_node.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))]
    obs = "\n".join(ast.unparse(b) for b in body)                    # executable code only (docstring excluded)
    checks["observer_pure"] = ("register_optimizer_step_post_hook" in obs and "h.remove()" in obs and "detach()" in obs and "no_grad" in obs
                               and not any(t in obs for t in ("backward", "model(", "zero_grad", ".step(", "random", "manual_seed", "add_", "copy_(")))
    checks["exact_A2_path"] = ('t0run.run_objective(b, g, "A2"' in run_code and "with step_observer() as snaps" in run_code and "len(snaps) != 2" in run_code
                               and "step2_snapshot_equals_final_masters" in run_code and "live_objective_check" in run_code)
    checks["barrier_before_phase2"] = run_code.index("FULL300 A2 reconstruction barrier failed") < run_code.index("Phase 2: MECH_PANEL84")
    dsrc = _fn_source(RUNNER, "drop_family")
    checks["drop_no_reoptimization"] = "theta0_bf16[n] if n in fs else final_bf16[n]" in dsrc and "optim" not in dsrc and "run_objective" not in run_code[run_code.index("Phase 2: MECH_PANEL84"):]
    checks["no_forbidden_paths"] = not any(t in rsrc for t in ("adapt_a4", "detect_language", "score_branch", "g2_forced", "lockstep_detect", "online_g1", "load_references",
                                                                "generate("))
    gsrc = _fn_source(RUNNER, "geometry_path")
    checks["geometry_teacher_forced"] = "mism.append(t)" in gsrc and "new = [int(prefix[t])]" in gsrc and "processed_log_probs" in gsrc and "state_locked" in gsrc
    fsrc = (ROOT / ANALYZE).read_text()
    checks["reference_barrier"] = ("load_references" not in fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")] and "committed(SEAL)" in fsrc
                                   and "committed(PRIMARY_AUDIT)" in fsrc)
    asrc = (ROOT / "experiments/inference_cf_p2a2_mech0_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2a2_mech0|consensus_guard|branch_adjudication|path_decode|eos_boundary)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2a2_mech0.py").exists()
    e = {"rows": 12, "dlg": 10, "zh": 0, "mx": 0, "prow": 0, "pdlg": 0}
    n0 = {"zh": 10, "mx": 10, "prow": 5, "pdlg": 5}
    tt = [own_mech(True, e, n0, 0.5, 0.5, 0.1, 0.75) == "P2_A2_MECH0_TERMINATION_DOMINANT", own_mech(True, e, n0, 0.5, 0.49, 0.1, 0.75) != "P2_A2_MECH0_TERMINATION_DOMINANT",
          own_mech(True, e, n0, 0.6, 0.6, 0.0, 0.9) != "P2_A2_MECH0_TERMINATION_DOMINANT", own_mech(True, e, n0, 0.6, 0.6, 0.1, 0.74) != "P2_A2_MECH0_TERMINATION_DOMINANT",
          own_mech(True, {"rows": 2, "dlg": 2, "zh": 0, "mx": 0, "prow": 0, "pdlg": 0}, n0, 0.9, 0.9, 1, 1) != "P2_A2_MECH0_TERMINATION_DOMINANT",
          own_mech(True, {"rows": 3, "dlg": 3, "zh": 0, "mx": 0, "prow": 0, "pdlg": 0}, n0, 0.2, 0.2, 1, 1) == "P2_A2_MECH0_NONTERMINATION_DOMINANT",
          own_mech(True, {"rows": 3, "dlg": 3, "zh": 0, "mx": 0, "prow": 0, "pdlg": 0}, n0, 0.25, 0.2, 1, 1) != "P2_A2_MECH0_NONTERMINATION_DOMINANT",
          own_mech(True, {"rows": 3, "dlg": 3, "zh": 1, "mx": 0, "prow": 1, "pdlg": 1}, {"zh": 1, "mx": 0, "prow": 1, "pdlg": 1}, 0.3, 0.6, 1, 1) == "P2_A2_MECH0_MIXED",
          own_mech(True, {"rows": 3, "dlg": 3, "zh": 5, "mx": 0, "prow": 2, "pdlg": 2}, {"zh": 0, "mx": 5, "prow": 2, "pdlg": 2}, 0.9, 0.1, -1, 0) == "P2_A2_MECH0_MIXED",
          own_mech(True, e, n0, None, 0.6, 1, 1) == "P2_A2_MECH0_NO_CLEAR_MECHANISM", own_mech(False, e, n0, 0.9, 0.9, 1, 1) == "P2_A2_MECH0_INVALID"]
    checks["own_label_truth_table"] = all(tt)
    B_ = c["compute"]
    checks["compute_plan"] = (B_["jobs_max"] == 1 and B_["updates_exact"] == 600 and B_["backward_max_including_live_audit"] == 601 and B_["total_free_decodes"] == 936
                              and B_["geometry_paths"] == 132 and B_["ablation_current_query_paths"] == 132 and B_["hard_minutes"] == 180)
    sb = (ROOT / "slurm/inference_cf_p2a2_mech0.sbatch").read_text()
    checks["sbatch"] = "--time=03:00:00" in sb and "PASS_TO_P2_A2_MECH0" in sb and "inference_cf_p2a2_mech0.py run" in sb
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    v = "PASS_TO_P2_A2_MECH0" if all(checks.values()) else "BLOCK_BEFORE_P2_A2_MECH0"
    return {"schema": "p2_a2_mech0_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD"),
            "instrumentation_note": "STEP snapshots via global torch optimizer step post-hook; episodic_tta.py and inference_cf_p2tta0.py unmodified (baseline hashes)"}


def own_norms(th, S, names, fam):
    sq = {n: float(((S[n].astype(np.float64) - th[n].astype(np.float64)) ** 2).sum()) for n in names}
    fams = {f: math.sqrt(sum(v for n, v in sq.items() if fam[n] == f)) for f in ("SELF", "CROSS", "POST")}
    return math.sqrt(sum(sq.values())), fams


def cmd_post(args) -> dict:
    c = cfg()
    run = BASE / "run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    S, dlg, kmap, B, A, U = own_strata()
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]
              and rt["barrier"]["pass"] and rt["elapsed_sec"] <= 3 * 3600,
              "seal_committed_unchanged": committed("results/inference_cf/p2a2_mech0/output_seal.json")
              and all(sha(ROOT / p) == h for p, h in json.loads((BASE / "output_seal.json").read_text())["files"].items())}
    files = sorted((run / "rows").glob("*.json"))
    checks["exactly_300_rows"] = [f.name for f in files] == [f"{i:03d}.json" for i in range(300)]
    R = {}
    for f in files:
        r = json.loads(f.read_text())
        R[r["identity"]] = r
    names = [p["name"] for p in c["A2_inherited"]["trainables"]["parameters"]]
    fam = {n: f for f in ("SELF", "CROSS", "POST") for n in c["families"][f]}
    with np.load(run / "states" / "theta0_ln_fp32.npz") as z:
        th = {n: z[f"p{j:03d}"] for j, n in enumerate(names)}
    Pn = {r["utterance_id"]: r for r in json.loads((ROOT / PANEL).read_text())["rows"]}
    bad = []
    for i, u in enumerate(S["FULL300"]):
        r = R[u]
        okr = r["status"] == "ok" and r["canonical_index"] == i
        okr &= r["theta0"] == {k: B[u][k] for k in ("tokens", "terminated", "text")} and r["A2_final"] == {k: A[u][k] for k in ("tokens", "terminated", "text")}
        okr &= r["step2_state_hash"] == Pn[u]["A2_sealed"]["effective_state_sha256"] and r["step2_snapshot_equals_final_masters"] and r["reset_ok"]
        okr &= r["A2_log"]["steps"] == 2 and len(r["A2_log"]["losses"]) == 3 and all(r["A2_log"]["finite"])
        with np.load(run / "states" / f"{i:03d}_step1_masters_fp32.npz") as z:
            S1 = {n: z[f"A2_p{j:03d}"] for j, n in enumerate(names)}
        arch = Pn[u]["historical_final_master_archive"]
        with np.load(ROOT / arch["path"] if arch else run / "states" / f"{i:03d}_step2_masters_fp32.npz") as z:
            S2 = {n: z[f"A2_p{j:03d}"] for j, n in enumerate(names)}
        for sname, Sx in (("step1", S1), ("step2", S2)):
            tot, fams = own_norms(th, Sx, names, fam)
            nr = r["norms"][f"{sname}_master"]
            okr &= abs(tot - nr["total"]) <= 1e-9 * max(1.0, tot) and all(abs(fams[f] - nr["family"][f]) <= 1e-9 * max(1.0, fams[f]) for f in fams)
        okr &= abs(math.sqrt(sum(float(((S2[n].astype(np.float64) - S1[n].astype(np.float64)) ** 2).sum()) for n in names)) - r["norms"]["step1_to_step2_master_l2"]) <= 1e-9
        if not okr:
            bad.append(u)
    checks["reconstruction_states_norms_recomputed"] = not bad
    ch, eosR, cont = S["MECH_CHANGED"], S["EOS_RECOVERY"], S["CONTENT_DIVERGENCE"]
    gbad = []
    dm2_eos = []
    for u in ch:
        r = R[u]
        k, c0, c2, kind = kmap[u]
        okr = all(r["geometry"][s]["k"] == k and r["geometry"][s]["fed_ok"] and r["geometry"][s]["state_locked"] for s in ("STEP0", "STEP1", "STEP2"))
        okr &= r["geometry"]["STEP0"]["state_hash"] == rt["theta0_ln_hash"] and r["geometry"]["STEP2"]["state_hash"] == r["step2_state_hash"]
        for s in ("STEP0", "STEP1", "STEP2") :
            gm = r["geometry"][s]
            okr &= gm["cands"]["B0"]["token"] == c0 and gm["cands"]["A2"]["token"] == c2 and gm["cands"]["B0"]["logp"] is not None and gm["cands"]["A2"]["logp"] is not None
            exp_act = "A2" if gm["argmax"] == c2 else ("B0" if gm["argmax"] == c0 else "third")
            okr &= gm["action"] == exp_act
            cn = gm["continuation"]
            okr &= cn["tokens"] == (A[u]["tokens"][k:k + 3] if c2 != EOS else []) and (cn["C_H"] is None if not cn["logprobs"] else abs(cn["C_H"] - sum(cn["logprobs"]) / len(cn["logprobs"])) <= 1e-12)
        for dname in ("DROP_SELF", "DROP_CROSS", "DROP_POST"):
            dq = r["drop_query"][dname]
            okr &= dq["action"] == ("A2" if dq["argmax"] == c2 else ("B0" if dq["argmax"] == c0 else "third"))
        aa = r["auto_alignment"]
        us = st(U[u]["tokens"], U[u]["terminated"])
        auto_a = us[k] if k < len(us) else None
        okr &= aa["AUTO_action"] == auto_a and aa["AUTO_prefix_equals_common"] == (us[:k] == st(B[u]["tokens"], B[u]["terminated"])[:k])
        if kind == "EOS_RECOVERY":
            mm = [r["geometry"][s]["cands"]["A2"]["logp"] - r["geometry"][s]["cands"]["EOS"]["logp"] for s in ("STEP0", "STEP1", "STEP2")]
            dm2_eos.append((mm[2] - mm[0], dlg[u]))
        if not okr:
            gbad.append(u)
    checks["geometry_margins_actions_recomputed"] = not gbad
    ms = prim["M_stop"]["summary"]
    own_med = float(np.median([x for x, _ in dm2_eos])) if dm2_eos else None
    own_frac = (sum(x > 0 for x, _ in dm2_eos) / len(dm2_eos)) if dm2_eos else None
    checks["M_stop_summary_agrees"] = (own_med is None and ms["median"]["dM2"] is None) or (abs(own_med - ms["median"]["dM2"]) <= 1e-12 and own_frac == ms["dM2_positive_rate"])
    ret = lambda acts: (sum(a == "A2" for a in acts) / len(acts)) if acts else None
    rbad = []
    for dname in ("DROP_SELF", "DROP_CROSS", "DROP_POST"):
        ro, re_ = ret([R[u]["drop_query"][dname]["action"] for u in ch]), ret([R[u]["drop_query"][dname]["action"] for u in eosR])
        ab = prim["ablations"][dname]
        sens = (ro is not None and 1 - ro >= 0.25 - G) or (re_ is not None and 1 - re_ >= 0.25 - G)
        if not (ab["RET_overall"] == ro and ab["RET_EOS"] == re_ and ab["sensitive"] == sens):
            rbad.append(dname)
    checks["retention_sensitivity_recomputed"] = not rbad
    missed = [u for u in S["FULL300"] if severe(len(B[u]["tokens"]), len(A[u]["tokens"]), A[u]["terminated"])]
    checks["severe_agrees"] = missed == prim["FULL_A2_severe_truncations_vs_B0"]
    cnt = rt["counters"]
    checks["counts"] = (cnt["A2"]["optimizer_steps"] == 600 and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] <= 601 and cnt["observer_snapshots"] == 600
                        and cnt["theta0_decodes"] + cnt["A2_final_decodes"] + cnt["step1_decodes"] + cnt["drop_decodes"] == 936
                        and cnt["geometry_paths"] == 132 and cnt["drop_query_paths"] == 132)
    la = R[S["FULL300"][0]]["live_audit"]
    checks["live_audit_2pct"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    checks["primary_valid_agrees"] = prim["valid"] == all(checks.values())
    out = {"missed_severe": missed, "dM2_median": own_med, "dM2_pos_rate": own_frac}
    lab = ready = None
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        from csasr.evaluation.mer import corpus_mer
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        refs = load_references()
        ids = S["FULL300"]
        ref = {u: refs[u]["reference"] for u in ids}
        checks["dialogues"] = all(refs[u]["dialogue_id"] == dlg[u] for u in ids)
        Cn = {u: {s: tau.counts(ref[u], t) + [corpus_mer([ref[u]], [t])["deletions"]] for s, t in (("B0", B[u]["text"]), ("A2", A[u]["text"]))} for u in ids}
        tot = {s: [sum(Cn[u][s][j] for u in ids) for j in (4, 0, 2)] for s in ("B0", "A2")}
        checks["aggregates"] = tot["B0"] == [3309, 1074, 4456] and tot["A2"] == [3037, 1012, 4126] and \
            [sec["systems_FULL300"][s][k] for s in ("B0", "A2") for k in ("zh", "poi", "mixed")] == tot["B0"] + tot["A2"]
        dz = {u: Cn[u]["B0"][4] - Cn[u]["A2"][4] for u in ids}
        dmx = {u: Cn[u]["B0"][2] - Cn[u]["A2"][2] for u in ids}
        dd = {u: Cn[u]["B0"][8] - Cn[u]["A2"][8] for u in ids}
        pz = {u: max(v, 0) for u, v in dz.items()}
        pdl = {u: max(v, 0) for u, v in dd.items()}
        FZ = (sum(pz[u] for u in eosR) / sum(pz.values())) if sum(pz.values()) else None
        FD = (sum(pdl[u] for u in eosR) / sum(pdl.values())) if sum(pdl.values()) else None
        checks["fractions_agree"] = (FZ == sec["F_ZH_EOS"] or abs(FZ - sec["F_ZH_EOS"]) <= 1e-12) and (FD == sec["F_DEL_EOS"] or abs(FD - sec["F_DEL_EOS"]) <= 1e-12)
        gs = lambda us: {"rows": len(us), "dlg": len({dlg[u] for u in us}), "zh": sum(dz[u] for u in us), "mx": sum(dmx[u] for u in us),
                         "prow": sum(1 for u in us if dz[u] > 0 or dmx[u] > 0), "pdlg": len({dlg[u] for u in us if dz[u] > 0 or dmx[u] > 0})}
        eg, ng = gs(eosR), gs(cont + S["EOS_REGRESSION"])
        checks["groups_agree"] = (eg["zh"] == sec["EOS_group"]["net_ZH"] and eg["mx"] == sec["EOS_group"]["net_mixed"] and ng["zh"] == sec["nonEOS_group"]["net_ZH"]
                                  and ng["mx"] == sec["nonEOS_group"]["net_mixed"] and ng["prow"] == sec["nonEOS_group"]["positive_rows"]
                                  and sum(dd[u] for u in ids) == sec["groups"]["FULL300"]["net_dD"])
        keys = sorted(set(dlg.values()))
        lv = [sum(dz[u] for u in ids if dlg[u] != k) for k in keys]
        checks["lodo_agrees"] = min(lv) == sec["lodo"]["dZH"]["min"] and sum(x > 0 for x in lv) == sec["lodo"]["dZH"]["count_positive"]
        grp = {k: [j for j, u in enumerate(ids) if dlg[u] == k] for k in keys}
        rng = np.random.default_rng(240924)
        drs = [sum((grp[keys[j]] for j in rng.integers(0, 20, 20)), []) for _ in range(2000)]
        arr = np.array([pz[u] for u in ids], dtype=float)
        msk = np.array([u in set(eosR) for u in ids])
        vals = [float((arr[ix] * msk[ix]).sum() / arr[ix].sum()) for ix in drs if arr[ix].sum()]
        bz = sec["bootstrap"]["F_ZH_EOS"]["ci95"]
        CA = {s: np.array([Cn[u][s] for u in ids], dtype=float) for s in ("B0", "A2")}
        pv = [CA["A2"][ix, 0].sum() / CA["A2"][ix, 1].sum() - CA["B0"][ix, 0].sum() / CA["B0"][ix, 1].sum() for ix in drs]
        bp = sec["bootstrap"]["pier"]["ci95"]
        checks["bootstrap_agrees"] = max(abs(np.quantile(vals, .025) - bz[0]), abs(np.quantile(vals, .975) - bz[1]),
                                         abs(np.quantile(pv, .025) - bp[0]), abs(np.quantile(pv, .975) - bp[1])) <= 1e-9
        valid = bool(prim["valid"]) and all(v for k, v in checks.items() if k not in ("label_agrees", "readiness_agrees"))
        lab = own_mech(valid, eg, ng, FZ, FD, own_med, own_frac)
        checks["label_agrees"] = lab == sec["label"]
        ready = "YES" if (valid and not missed) else "NO"
        checks["readiness_agrees"] = ready == sec["A2_CONFIRMATION_READY_provisional"]
        out.update(F_ZH_EOS=FZ, F_DEL_EOS=FD, EOS_group=eg, nonEOS_group=ng, aggregates=tot)
    v = "P2_A2_MECH0_AUDIT: PASS" if all(checks.values()) else "P2_A2_MECH0_AUDIT: BLOCK"
    return {"schema": "p2_a2_mech0_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab, "A2_CONFIRMATION_READY": ready if v.endswith("PASS") else "NO",
            "checks": checks, "failures": {"rows": bad, "geometry": gbad, "retention": rbad}, "independent": out, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--phase", choices=("primary", "full"), required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "post": cmd_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "A2_CONFIRMATION_READY")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
