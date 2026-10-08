#!/usr/bin/env python
"""ST-PROMPT-R1-A independent auditor. Does NOT import the R1 runner, provider (prompt_r1), geometry gate or analysis /
decision / bootstrap code. Reuses only the historical independent P2-DIR auditor primitives (hashing, bf16 unpack,
`metrics`) and the tokenizer partition / suppression config readers.

prerun   PASS_TO_ST_PROMPT_R1_A
primary  ST_PROMPT_R1_A_AUDIT: PASS (PRIMARY)  -- reference-free, before evaluator access
full     ST_PROMPT_R1_A_AUDIT: PASS (FULL)     -- independent metrics, bootstrap, gates, label, selection, R1-B authorization
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
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

from experiments.inference_cf_p2dir_audit import ahash, bf16, canon, fhash, git_blob_hash, metrics

FREEZE = "e965a7f"
CONFIG = "configs/inference_cf/st_prompt_r1.json"
PANEL = "docs/inference_cf/ST_PROMPT_R1_PANEL.json"
FROZEN = (CONFIG, PANEL, "docs/inference_cf/ST_PROMPT_R1_SPEC.md", "docs/inference_cf/ST_PROMPT_R1_CODEX_DESIGN.md", "docs/inference_cf/ST_PROMPT_R1_FIREWALL.md",
          "docs/inference_cf/ST_PROMPT_R1_REACHABILITY.json", "tests/test_st_prompt_r1_contract.py", "experiments/inference_cf_st_prompt_r1_preflight.py")
BASE = "results/inference_cf/st_prompt_r1"
LOC0 = "results/inference_cf/st_loc0/run1"
LAYERS = (3, 8, 16, 24)
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
RUNTIME_MODULES = ("experiments/inference_cf_st_prompt_r1.py", "src/csasr/inference_cf/prompt_r1.py")
FORBIDDEN = ("target_ids", "competitor", "positions.json", "p2rj", "evaluation_membership", "stratum", "logit_metrics", "transcript",
             "reference_tokens", "d-dev-confirm", "d-test", "p3_", "oracle")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def names(rel: str) -> set:
    tree = ast.parse((ROOT / rel).read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = set()
    for node in ast.walk(tree):
        if id(node) in docs:
            continue
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.name for a in node.names} | ({node.module} if getattr(node, "module", None) else set())
    return out


DECLARED = {"no evaluator targets/competitors/strata in runner; evaluator only after pushed seal + primary audit PASS", "references_used",
            "reference_access_boundary"}


def hits(rel: str) -> list:
    return sorted({str(n)[:80] for n in names(rel) if str(n) not in DECLARED for f in FORBIDDEN if f in str(n).lower()})


def rand_vec(seed_key: str) -> tuple[int, np.ndarray]:
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    v = np.random.Generator(np.random.PCG64(seed)).standard_normal(1280)
    return seed, (v / np.linalg.norm(v)).astype(np.float32)


def v_prompt(he, hm):
    d = he.astype(np.float64) - hm.astype(np.float64)
    n = float(np.linalg.norm(d))
    if not math.isfinite(n) or n < 1e-4:
        return None
    return (d / (n + 1e-6)).astype(np.float32)


def solver_unit(v32: np.ndarray) -> np.ndarray:
    v = v32.astype(np.float32).astype(np.float64)
    return v / np.linalg.norm(v)


# ===================== prerun =====================

def cmd_prerun(args) -> dict:
    checks, notes = {}, {}
    c = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    for rel in FROZEN:
        checks[f"unchanged_since_freeze:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"source:{rel}"] = fhash(ROOT / rel) == h
    checks["panel_identity"] = canon({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == c["panel_identity_hash"]
    old = json.loads((ROOT / "docs/inference_cf/ST_LOC0_PANEL.json").read_text())
    checks["panel_equals_st_loc0_population"] = P["runtime_queries"] == old["runtime_queries"] and P["utterances"] == old["utterances"] \
        and P["evaluation_membership"] == old["construction_positions"]
    cnt = defaultdict(int)
    for x in P["evaluation_membership"]:
        cnt[x["stratum"]] += 1
    checks["strata_60x3_80_20"] = dict(cnt) == {s: 60 for s in STRATA} and len(P["utterances"]) == 80 and len({q["dialogue_id"] for q in P["runtime_queries"]}) == 20
    checks["runtime_queries_label_free"] = all(set(q) <= {"utterance_id", "dialogue_id", "t", "absolute_query", "content_prefix_sha256",
                                                          "forced_zh_query_input_sha256", "forced_en_query_input_sha256"} for q in P["runtime_queries"])
    checks["model_files"] = all(fhash(Path(c["conditions"]["model"]["dir"]) / n) == h for n, h in c["conditions"]["model"]["files"].items())
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    checks["plan_hash"] = canon({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"] and all(plan["checks"].values())
    checks["plan_runtime_projection_only"] = set(plan["runtime"]) == {"runtime_queries", "utterances"} and plan["runtime"]["runtime_queries"] == P["runtime_queries"] \
        and not any(f in json.dumps(plan) for f in ("stratum", "EN-confusion", "target_ids", "competitor", "evaluation_membership"))
    want = [(L, e, s) for L in LAYERS for e in (0.15, 0.3, 0.45) for s in (1, -1)]
    checks["arms_24_primary"] = sorted((a["layer"], a["eta"], a["sign"]) for a in plan["arms"] if a["family"] == "prompt") == sorted(want)
    checks["arms_12_random"] = sorted((a["layer"], a["eta"]) for a in plan["arms"] if a["family"] == "random") == sorted({(L, e) for L, e, _ in want})
    rnd_ok = True
    for r in c["controls"]["random"]:
        seed, v = rand_vec(r["seed_key"])
        rnd_ok &= seed == r["seed"] and "sha256:" + hashlib.sha256(v.tobytes()).hexdigest() == r["array_bytes_sha256"]
    checks["random_vectors_independent"] = bool(rnd_ok)
    checks["dose_ladder_frozen"] = c["energy"]["eta"] == [0.15, 0.3, 0.45] and c["energy"]["norm_preserve"] and not c["energy"]["depth_rescale"]
    checks["site"] = c["site"] == "POST_CROSS_ATTENTION_PRE_FFN" and c["layers"] == [3, 8, 16, 24]
    # historical comparators sealed by ST-LOC0
    seal = json.loads((ROOT / "results/inference_cf/st_loc0/output_seal.json").read_text())
    comp = [f"{LOC0}/calibration/states_L16_CROSS.npz", f"{LOC0}/calibration/baseline_logits.npz", f"{LOC0}/calibration/v1_records.json", f"{LOC0}/calibration/d2_vectors.npz"]
    checks["st_loc0_comparators_sealed"] = all(seal["files"].get(p) == fhash(ROOT / p) for p in comp)
    # D0 historical vectors reconstructed independently from sealed states
    with np.load(ROOT / LOC0 / "calibration/states_L16_CROSS.npz") as z:
        hb, he = z["H_B"], z["H_E"]
    v1 = json.loads((ROOT / LOC0 / "calibration/v1_records.json").read_text())
    checks["D0_vectors_independent"] = all(ahash(v_prompt(he[j], hb[j])) == v1[f"q{j:03d}_L16_CROSS"]["sha256"] for j in range(180))
    # firewall (static)
    for rel in RUNTIME_MODULES:
        notes[f"forbidden:{rel}"] = hits(rel)
        checks[f"runtime_reference_free:{rel}"] = not notes[f"forbidden:{rel}"]
    geo = names("experiments/inference_cf_st_prompt_r1_geometry.py")
    checks["geometry_gate_no_targets"] = not ({"target_ids", "competitor", "positions.json", "p2rj"} & geo) and not any("p2rj" in str(n) for n in geo)
    an = (ROOT / "experiments/inference_cf_st_prompt_r1_analyze.py").read_text()
    prim = an[an.index("def primary("):an.index("def kl_edit_none(")]
    checks["primary_analysis_reference_free"] = not any(x in prim for x in ("POSITIONS", "target_ids", "competitor", "logit_metrics"))
    sec = an[an.index("def secondary("):]
    checks["secondary_gated"] = sec.index("committed(SEAL)") < sec.index("committed(PRIMARY_AUDIT)") < sec.index("(ROOT / POSITIONS)")
    run_src = (ROOT / "experiments/inference_cf_st_prompt_r1.py").read_text()
    checks["barrier_before_new_pulses"] = run_src.index('run_pass("barrier", True)') < run_src.index('run_pass("pulses", False)') \
        and "historical barrier failed before new pulses" in run_src
    checks["geometry_gate_before_pulses"] = "GEOMETRY_COVERAGE_PASS" in run_src[run_src.index("def cmd_pulses("):run_src.index("def run_pass(")]
    checks["no_training_or_backward"] = not any(x in run_src for x in (".backward(", "optim.", "requires_grad_(True)"))
    checks["no_R1_B_runner"] = not (ROOT / "experiments/inference_cf_st_prompt_r1_schedule.py").exists() and "R1_B" not in run_src \
        and "def decide(" in an and "ST_PROMPT_R1_A_CAUSAL_PROMISE" in an
    checks["bootstrap_family_48"] = c["bootstrap"]["joint_family_size"] == 48 and abs(c["bootstrap"]["adjusted_quantiles"][0] - 0.05 / 96) < 1e-18
    sb = (ROOT / "slurm/inference_cf_st_prompt_r1.sbatch").read_text()
    tm = re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", sb)
    checks["sbatch"] = tm is not None and int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + int(tm.group(3)) <= c["compute"]["hard_ceiling_seconds_per_job"] \
        and c["audit"]["pre_A"] in sb and sb.count("inference_cf_st_prompt_r1.py pulses") == 1 and "geometry" in sb
    # compute: ST-LOC0 measured 17496 forwards + 4500 pulses in 385 s; R1-A ~ capture 2x replays + 180 x (43 pulses + clean/restore)
    loc0 = json.loads((ROOT / LOC0 / "runtime.json").read_text())
    per_fwd = loc0["elapsed_sec"] / loc0["counters"]["forwards"]
    fwd = 2 * 13000 + 2 * 13000 + 180 * (4 + 3 + 36 + 2)
    proj = 3 * fwd * per_fwd + 3 * 120 + 1800
    notes["compute_projection_sec"] = {"per_forward_hist": per_fwd, "forwards_est": fwd, "projected": proj}
    checks["compute_le_3h"] = proj <= c["compute"]["hard_ceiling_seconds_per_job"] - 900
    for t in ("tests/test_st_prompt_r1_contract.py", "tests/test_st_prompt_r1_impl.py"):
        checks[f"test_present:{t}"] = (ROOT / t).exists()
    me = (ROOT / "experiments/inference_cf_st_prompt_r1_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(prompt_r1\b|inference_cf_st_prompt_r1\b|st_prompt_r1_analyze|st_prompt_r1_geometry|p2dir_analyze)", me, re.M) is None
    checks["no_existing_run"] = not (ROOT / BASE / "runA" / "manifest.json").exists()
    v = c["audit"]["pre_A"] if all(checks.values()) else "BLOCK_BEFORE_ST_PROMPT_R1_A"
    return {"schema": "st_prompt_r1_prerun_A_v1", "verdict": v, "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


# ===================== primary =====================

def load_rows(run):
    man = json.loads((run / "manifest.json").read_text())
    rows = {}
    for nm in ("barrier", "pulses"):
        for i in range(80):
            p = run / nm / f"{i:03d}.json"
            if not p.exists():
                return man, None
            r = json.loads(p.read_text())
            lg, ar = {}, {}
            if r["logits_sha256"]:
                with np.load(run / nm / f"{i:03d}_logits.npz") as z:
                    lg = {k: z[k] for k in z.files}
                with np.load(run / nm / f"{i:03d}_arrays.npz") as z:
                    ar = {k: z[k] for k in z.files}
            rows[(nm, i)] = (r, lg, ar)
    return man, rows


def cmd_primary(args) -> dict:
    checks, notes = {}, {}
    c = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    mem = P["evaluation_membership"]
    run = ROOT / args.run
    man, rows = load_rows(run)
    checks["manifest_hash"] = canon({k: v for k, v in man.items() if k != "manifest_hash"}) == man["manifest_hash"]
    checks["manifest_sources_at_commit"] = all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items())
    rt = json.loads((run / "runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and rt["n_failures"] == 0
    checks["frozen_model"] = rt["model_grads_none"] and not rt["requires_grad_any"] and rt["weights_unchanged_sample"]
    checks["no_autograd_LID"] = rt["counters"]["autograd_calls"] == 0 and rt["counters"]["LID_calls"] == 0
    checks["rows_present"] = rows is not None
    if rows is None:
        return {"verdict": "ST_PROMPT_R1_A_AUDIT: FAIL (PRIMARY)", "checks": checks, "notes": notes}
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    arms = {a["id"]: a for a in plan["arms"]}
    # capture identity, independent v_prompt, independent geometry gate recount
    cap = run / "capture"
    S = {}
    for l in LAYERS:
        with np.load(cap / f"states_L{l:02d}.npz") as z:
            S[l] = (z["H_M"], z["H_E"])
    with np.load(ROOT / LOC0 / "calibration/states_L16_CROSS.npz") as z:
        checks["L16_states_bitwise_vs_st_loc0"] = bool(np.array_equal(S[16][0], z["H_B"]) and np.array_equal(S[16][1], z["H_E"]))
    with np.load(cap / "none_logits.npz") as z:
        none = {k: z[k] for k in z.files}
    with np.load(ROOT / LOC0 / "calibration/baseline_logits.npz") as z:
        checks["NONE_logits_bitwise_vs_st_loc0"] = all(np.array_equal(none[k], z[k]) for k in z.files) and len(none) == 180
    with np.load(cap / "v_prompt.npz") as z:
        V = {k: z[k] for k in z.files}
    vok = True
    for j in range(180):
        for l in LAYERS:
            mine = v_prompt(S[l][1][j], S[l][0][j])
            k = f"q{j:03d}_L{l}"
            vok &= (mine is None and k not in V) or (mine is not None and k in V and np.array_equal(mine, V[k]))
    checks["v_prompt_independent_all_layers"] = bool(vok)
    gate = json.loads((run / "geometry_gate.json").read_text())
    ledger = json.loads((cap / "geometry_ledger.json").read_text())
    gok = True
    for aid in arms:
        for s in STRATA:
            js = [j for j in range(180) if mem[j]["stratum"] == s and ledger[f"q{j:03d}|{aid}"]["reachable"]]
            gok &= len(js) >= 54 and len({mem[j]["dialogue_id"] for j in js}) >= 12
    checks["geometry_gate_independent"] = (gate["verdict"] == "GEOMETRY_COVERAGE_PASS") == bool(gok) and bool(gok)
    rv = {r["id"]: rand_vec(r["seed_key"])[1] for r in c["controls"]["random"]}
    # barrier: bitwise vs sealed ST-LOC0 logits / consumed (independent comparison of stored arrays)
    bmap = {"hist_prompt_L16_plus": ("barrier", "v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus"),
            "hist_prompt_L16_minus": ("pulses", "v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_minus"),
            "hist_D2_L16": ("barrier", "D2_L16_POST_CROSS_ATTENTION_PRE_FFN")}
    bar_ok, zero_ok, n_bar = True, True, 0
    for i in range(80):
        r, lg, ar = rows[("barrier", i)]
        hist = {}
        for d in ("barrier", "pulses"):
            with np.load(ROOT / LOC0 / d / f"{i:03d}_logits.npz") as z:
                hist[d] = {k: z[k] for k in z.files}
            with np.load(ROOT / LOC0 / d / f"{i:03d}_arrays.npz") as z:
                hist[d + "_ar"] = {k: z[k] for k in z.files}
        for t, pr in r["positions"].items():
            j = pr["q"]
            zero_ok &= len(pr["zero_dose"]) == 4 and all(pr["zero_dose"].values()) and pr["restore_bitwise"] and pr["clean_replay_bitwise"]
            for bid, (hd, haid) in bmap.items():
                n_bar += 1
                bar_ok &= np.array_equal(lg[f"q{j:03d}_{bid}"], hist[hd][f"q{j:03d}_{haid}"])
                k = f"q{j:03d}_{haid}_consumed"
                if k in hist[hd + "_ar"]:
                    bar_ok &= np.array_equal(ar[f"q{j:03d}_{bid}_consumed"], hist[hd + "_ar"][k])
    checks["barrier_bitwise_independent"] = bool(bar_ok) and n_bar == 540
    checks["zero_restore_clean"] = bool(zero_ok)
    # main cells: completeness, independent energy/direction/no-edit, pairwise, eligibility
    e = c["energy"]
    bad = defaultdict(list)
    valid = defaultdict(lambda: defaultdict(int))
    vdl = defaultdict(lambda: defaultdict(set))
    joint = defaultdict(lambda: defaultdict(int))
    jdl = defaultdict(lambda: defaultdict(set))
    ncell = 0
    for i in range(80):
        r, lg, ar = rows[("pulses", i)]
        for t, pr in r["positions"].items():
            j = pr["q"]
            s, dl = mem[j]["stratum"], mem[j]["dialogue_id"]
            if set(pr["cells"]) != set(arms):
                bad["arm_set"].append(j)
            if not (pr["clean_replay_bitwise"] and pr["restore_bitwise"]):
                bad["lineage"].append(j)
            en = defaultdict(list)
            for aid, x in pr["cells"].items():
                ncell += 1
                a = arms[aid]
                l = a["layer"]
                before = S[l][0][j].astype(np.float64)
                rn = float(np.linalg.norm(before))
                if x["valid"]:
                    cons = ar[f"q{j:03d}_{aid}_consumed"].astype(np.float64)
                    prop = ar[f"q{j:03d}_{aid}_proposed"].astype(np.float64)
                    dc, dp = float(np.linalg.norm(cons - before)), float(np.linalg.norm(prop - before))
                    target = a["eta"] * rn
                    ok = (abs(dc / target - 1) <= e["max_relative_norm_error"] + e["consumed_vs_proposed_relative_norm_tolerance"]
                          and abs(dp / target - 1) <= e["max_relative_norm_error"] and abs(dp ** 2 / target ** 2 - 1) <= e["max_relative_squared_error"]
                          and abs(dc - dp) <= e["consumed_vs_proposed_relative_norm_tolerance"] * dp
                          and abs(float(np.linalg.norm(cons)) - rn) <= e["consumed_vs_proposed_relative_norm_tolerance"] * rn
                          and abs(x["actual_eta"] - x["realized_edit_norm"] / rn) <= 1e-9)
                    vsrc = rv[aid] if a["family"] == "random" else (a["sign"] * V[f"q{j:03d}_L{l}"]).astype(np.float32)
                    # direction + sign: float64 NormPreserve reconstruction with the recorded solver scale
                    ex = before + float(x["solver"]["s"]) * solver_unit(vsrc)
                    ex = ex * rn / (np.linalg.norm(ex) + e["repair_epsilon"])
                    de, dpv = ex - before, prop - before
                    ok &= float(de @ dpv / (np.linalg.norm(de) * np.linalg.norm(dpv))) >= 0.99
                    if not ok:
                        bad["energy_direction"].append((j, aid))
                    en[(l, a["eta"])].append(x["realized_edit_norm"] ** 2)
                    valid[aid][s] += 1
                    vdl[aid][s].add(dl)
                else:
                    if x.get("no_edit_reason") not in c["eligibility"]["no_edit_reasons"] or not np.array_equal(lg[f"q{j:03d}_{aid}"], none[f"q{j:03d}"]):
                        bad["no_edit"].append((j, aid, x.get("no_edit_reason")))
                    if a["family"] == "prompt" and f"q{j:03d}_L{l}" not in V and x.get("no_edit_reason") != "tiny_or_nonfinite_direction":
                        bad["no_edit_reason"].append((j, aid))
                if x["integrity_failures"] or not (x["cache_prefix_unchanged"] and x["cache_positions_ok"]):
                    bad["cell_integrity"].append((j, aid))
            for k, v in en.items():
                if max(v) / min(v) - 1 > e["max_pairwise_relative_squared_error"]:
                    bad["pairwise"].append((j, k))
            for aid, a in arms.items():
                if a["family"] == "prompt":
                    ra = f"random_L{a['layer']:02d}_eta{a['eta']:.2f}"
                    if pr["cells"][aid]["valid"] and pr["cells"][ra]["valid"]:
                        joint[aid][s] += 1
                        jdl[aid][s].add(dl)
    checks["matrix_complete_6480"] = ncell == 6480 and not bad["arm_set"]
    for k in ("lineage", "energy_direction", "no_edit", "no_edit_reason", "cell_integrity", "pairwise"):
        checks[f"independent:{k}"] = not bad[k]
    notes["failures"] = {k: [str(x) for x in v[:15]] for k, v in bad.items() if v}
    elig = {}
    for aid, a in arms.items():
        okv = all(valid[aid][s] >= 54 and len(vdl[aid][s]) >= 12 for s in STRATA)
        if a["family"] == "prompt":
            okv = okv and all(joint[aid][s] >= 54 and len(jdl[aid][s]) >= 12 for s in STRATA)
        elig[aid] = okv
    notes["eligibility"] = elig
    notes["valid_counts"] = {a: dict(v) for a, v in valid.items()}
    prim = json.loads((ROOT / BASE / "primary_analysis_A.json").read_text())
    checks["primary_analysis_counts_agree"] = all(prim["arms"][a]["valid_by_stratum"] == {s: valid[a][s] for s in STRATA}
                                                  and prim["arms"][a].get("eligible", prim["arms"][a]["eligible_valid"]) == elig[a] for a in arms)
    my_valid = all(checks.values())
    checks["primary_valid_agrees"] = prim["valid"] == my_valid
    seal = json.loads((ROOT / BASE / "output_seal_A.json").read_text())
    checks["seal_files"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items()) and seal["manifest_hash"] == man["manifest_hash"]
    checks["seal_committed_pushed"] = git_blob_hash("HEAD", f"{BASE}/output_seal_A.json") == fhash(ROOT / BASE / "output_seal_A.json") and \
        subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", git("rev-parse", "--abbrev-ref", "@{u}")], cwd=ROOT).returncode == 0
    checks["no_reference_outputs_yet"] = not (ROOT / BASE / "secondary_analysis_A.json").exists()
    v = c["audit"]["post_A"] + " (PRIMARY)" if all(checks.values()) else "ST_PROMPT_R1_A_AUDIT: FAIL (PRIMARY)"
    return {"schema": "st_prompt_r1_primary_A_audit_v1", "verdict": v, "stage_valid": my_valid, "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


# ===================== full =====================

def cmd_full(args) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    checks, notes = {}, {}
    c = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    mem = P["evaluation_membership"]
    run = ROOT / args.run
    pa = json.loads((ROOT / BASE / "primary_audit_A.json").read_text())
    checks["primary_audit_committed_pass"] = pa["verdict"] == c["audit"]["post_A"] + " (PRIMARY)" and git_blob_hash("HEAD", f"{BASE}/primary_audit_A.json") == fhash(ROOT / BASE / "primary_audit_A.json")
    sec = json.loads((ROOT / BASE / "secondary_analysis_A.json").read_text())
    seal = json.loads((ROOT / BASE / "output_seal_A.json").read_text())
    checks["seal_unchanged_and_before_eval"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items()) and sec["output_seal_hash"] == seal["seal_hash"]
    man, rows = load_rows(run)
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    arms = {a["id"]: a for a in plan["arms"]}
    model = c["conditions"]["model"]["dir"]
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())["positions"]
    with np.load(run / "capture" / "none_logits.npz") as z:
        none = {k: z[k] for k in z.files}
    R = {}
    for i in range(80):
        r, lg, _ = rows[("pulses", i)]
        for t, pr in r["positions"].items():
            j = pr["q"]
            p = pos[j]
            Y, cc, tt = [int(x) for x in p["target_ids"]], int(p["competitor"]), int(p["t"])
            n = metrics(bf16(none[f"q{j:03d}"]), tt, sup, beg, part, Y, cc)
            R[j] = {"n": n, "a": {aid: metrics(bf16(lg[f"q{j:03d}_{aid}"]), tt, sup, beg, part, Y, cc) for aid in arms}, "valid": {aid: pr["cells"][aid]["valid"] for aid in arms}}
    keys = sorted({q["dialogue_id"] for q in mem})
    idx = np.random.default_rng(c["bootstrap"]["seed"]).integers(0, len(keys), size=(c["bootstrap"]["replicates"], len(keys)))
    W = np.stack([(idx == k).sum(axis=1) for k in range(len(keys))], axis=1).astype(float)
    adj = c["bootstrap"]["alpha"] / c["bootstrap"]["joint_family_size"]

    def boot(pairs, alpha):
        g = defaultdict(list)
        for d, v in pairs:
            g[d].append(v)
        mm = np.array([np.mean(g[k]) if k in g else np.nan for k in keys])
        use = ~np.isnan(mm)
        num, den = W[:, use] @ mm[use], W[:, use].sum(axis=1)
        dr = num[den > 0] / den[den > 0]
        return float(mm[use].mean()), float(np.quantile(dr, alpha / 2)), int((den > 0).sum())
    sa = {a["id"]: a for a in sec["arms"]}
    conf = [j for j in range(180) if mem[j]["stratum"] == "EN-confusion"]
    mine = []
    dmax = cmax = 0.0
    cnt_ok = True
    for aid, a in arms.items():
        dm = {j: R[j]["a"][aid]["m"] - R[j]["n"]["m"] for j in range(180)}
        corr = [j for j in conf if not R[j]["n"]["in_ref"] and R[j]["a"][aid]["in_ref"]]
        enc = sum(R[j]["n"]["in_ref"] and not R[j]["a"][aid]["in_ref"] for j in range(180) if mem[j]["stratum"] == "EN-correct")
        zhc = sum(R[j]["n"]["in_ref"] and not R[j]["a"][aid]["in_ref"] for j in range(180) if mem[j]["stratum"] == "ZH-correct")
        x = sa[aid]
        cnt_ok &= (len(corr), len({mem[j]["dialogue_id"] for j in corr}), enc, zhc) == (x["corrections"], x["correction_dialogues"], x["EN_corruptions"], x["ZH_corruptions"])
        if a["family"] != "prompt":
            continue
        est, lo, nd = boot([(mem[j]["dialogue_id"], dm[j]) for j in conf], adj)
        ra = f"random_L{a['layer']:02d}_eta{a['eta']:.2f}"
        rpairs = [(mem[j]["dialogue_id"], (dm[j] - (R[j]["a"][ra]["m"] - R[j]["n"]["m"])) if (R[j]["valid"][aid] and R[j]["valid"][ra]) else 0.0) for j in conf]
        rest, rlo, rnd = boot(rpairs, adj)
        dmax = max(dmax, abs(est - x["macro"]), abs(rest - x["random_point"]))
        if x["lower_none"] is not None:
            cmax = max(cmax, abs(lo - x["lower_none"]), abs(rlo - x["lower_random"]))
        okd = nd >= 9900 and rnd >= 9900
        elig = pa["notes"]["eligibility"][aid]
        mine.append({"id": aid, "eta": a["eta"], "elig": elig, "corr": len(corr), "cd": len({mem[j]["dialogue_id"] for j in corr}), "en": enc, "zh": zhc,
                     "macro": est, "lo": lo if okd else None, "rp": rest, "rlo": rlo if okd else None})
    checks["counts_agree"] = bool(cnt_ok)
    checks["estimates_agree_1e-9"] = dmax <= 1e-9
    checks["adjusted_bounds_agree_1e-9"] = cmax <= 1e-9
    g = c["R1_A"]
    ben = lambda a: a["elig"] and a["corr"] >= g["minimum_corrections"] and a["cd"] >= g["minimum_correction_dialogues"] and a["macro"] >= g["minimum_macro_margin_nats"] \
        and a["lo"] is not None and a["lo"] > 0 and a["rp"] > 0 and a["rlo"] is not None and a["rlo"] > 0
    saf = lambda a: a["en"] + a["zh"] <= g["maximum_correct_corruptions"] and a["zh"] <= g["maximum_ZH_corruptions"]
    valid = pa["stage_valid"]
    if not valid:
        label, sel = "ST_PROMPT_R1_A_INVALID", []
    elif any(ben(a) and saf(a) for a in mine):
        full = sorted([a for a in mine if ben(a) and saf(a)], key=lambda a: (-a["corr"], -a["cd"], a["zh"], a["en"], -a["lo"], a["eta"], a["id"]))
        label, sel = "ST_PROMPT_R1_A_CAUSAL_PROMISE", [a["id"] for a in full[:2]]
    elif any(ben(a) for a in mine):
        label, sel = "ST_PROMPT_R1_A_POWER_WITH_DAMAGE", []
    elif any(a["elig"] and a["lo"] is not None and a["lo"] > 0 and a["rlo"] is not None and a["rlo"] > 0 for a in mine):
        label, sel = "ST_PROMPT_R1_A_MARGIN_ONLY", []
    else:
        label, sel = "ST_PROMPT_R1_A_NO_CORRECTION_POWER", []
    checks["label_agrees"] = label == sec["label"]
    checks["selection_agrees"] = sel == sec["decision"]["selected"]
    checks["R1_B_authorization_consistent"] = (label == "ST_PROMPT_R1_A_CAUSAL_PROMISE") == bool(sel)
    notes["independent_label"], notes["independent_selection"] = label, sel
    union = sorted({j for a in mine for j in conf if not R[j]["n"]["in_ref"] and R[j]["a"][a["id"]]["in_ref"]})
    checks["oracle_union_agrees"] = union == sec["oracle_union"]["union_corrected_positions"]
    entries = sorted(p.name for p in (ROOT / BASE).iterdir())
    checks["scope_no_B_outputs"] = set(entries) <= {"plan_sealed.json", "prerun_audit_A.json", "runA", "primary_analysis_A.json", "output_seal_A.json",
                                                    "primary_audit_A.json", "secondary_analysis_A.json", "final_audit_A.json"}
    checks["one_job"] = len(list(run.glob("slurm-*.out"))) == 1
    v = c["audit"]["post_A"] + " (FULL)" if all(checks.values()) else "ST_PROMPT_R1_A_AUDIT: FAIL (FULL)"
    return {"schema": "st_prompt_r1_full_A_audit_v1", "verdict": v, "label": label if all(checks.values()) else "ST_PROMPT_R1_A_INVALID",
            "selection": sel, "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    for nm in ("primary", "full"):
        p = sub.add_parser(nm)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "primary": cmd_primary, "full": cmd_full}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit exists; never overwrite")
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({"verdict": res["verdict"], "failed": [k for k, v in res["checks"].items() if not v]}, indent=1))


if __name__ == "__main__":
    main()
