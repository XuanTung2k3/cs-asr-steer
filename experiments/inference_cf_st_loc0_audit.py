#!/usr/bin/env python
"""ST-LOC0 independent auditor (frozen spec section 6/11; config audit block).

Does NOT import the ST-LOC0 runner/analysis decision code, ``unique.fit_fold`` or ``inference_cf_p2dir_analyze``.
Numerical recomputation reuses only the historical INDEPENDENT P2-DIR auditor primitives (``refit``, ``metrics``,
``bf16``, ``lse``, hashing) plus its own energy / eligibility / bootstrap / label / winner / oracle logic.

prerun   PASS_TO_ST_LOC0 before any GPU job: freeze identity, sources, panel + runtime-projection firewall (AST),
         independent fold membership, D0 (180) and D1 (20) historical reconstruction, random-control derivation,
         arm matrix, sbatch limits, compute projection, auditor independence.
primary  ST_LOC0_AUDIT: PASS on reference-free outputs (before references): calibration seal on remote before
         pulses, matrix completeness, barrier, independent energy/no-edit/V1/V2/selection/eligibility checks.
full     after the secondary analysis: independent evaluator metrics, historical evaluator reproduction, shared
         bootstrap, Bonferroni intervals, label precedence, winner, oracle and scope.
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

from experiments.inference_cf_p2dir_audit import ahash, bf16, canon, fhash, git_blob_hash, lse, metrics, refit

FREEZE = "39ff5e9"
CONFIG = "configs/inference_cf/st_loc0.json"
PANEL = "docs/inference_cf/ST_LOC0_PANEL.json"
SPEC = "docs/inference_cf/ST_LOC0_SPEC.md"
DESIGN = "docs/inference_cf/ST_LOC0_CODEX_DESIGN.md"
BASE = "results/inference_cf/st_loc0"
RUNNER = "experiments/inference_cf_st_loc0.py"
CALIB = "experiments/inference_cf_st_loc0_calibrate.py"
SITES_MOD = "src/csasr/inference_cf/loc0_sites.py"
ANALYZE = "experiments/inference_cf_st_loc0_analyze.py"
SBATCH = "slurm/inference_cf_st_loc0.sbatch"
EXTRACT = "results/inference_cf/p2dir/extract_run1/states.npz"
EXP1 = "results/inference_cf/p2dir/exp1_run1"
FOLDS = "results/inference_cf/p2dir/folds_run1/folds.json"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
SK = ("L16_SELF", "L16_CROSS", "L24_SELF", "L24_CROSS")
SITE_OF = {"POST_SELF_ATTENTION": "SELF", "POST_CROSS_ATTENTION_PRE_FFN": "CROSS"}
BARRIER = {"v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus": "D0", "v_unq_L16_POST_CROSS_ATTENTION_PRE_FFN_plus": "D1",
           "D2_L16_POST_CROSS_ATTENTION_PRE_FFN": "D2"}
FORBIDDEN_RUNTIME = ("target_ids", "competitor", "positions.json", "p2rj", "transcript", "Y_ref", "ref_ids", "oracle",
                     "evaluator_targets", "logit_metrics", "stratum", "D-dev-confirm", "d_dev_confirm", "D-test", "d_test",
                     "router_calib", "router-calib", "p3_", "transfer")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sk_of(arm: dict) -> str:
    return f"L{arm['layer']}_{SITE_OF[arm['site']]}"


def names_in(rel: str, func: str | None = None) -> set[str]:
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    if func is not None:
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    docs = {id(b.body[0].value) for b in ast.walk(tree) if isinstance(b, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and b.body and isinstance(b.body[0], ast.Expr) and isinstance(b.body[0].value, ast.Constant)}
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


def forbidden_hits(rel: str, func: str | None = None, extra: tuple = (), allow: tuple = ()) -> list[str]:
    hits = set()
    for n in names_in(rel, func):
        for f in [x for x in FORBIDDEN_RUNTIME + extra if x not in allow]:
            if f.lower() in str(n).lower():
                hits.add(str(n)[:80])
    return sorted(hits)


def independent_arms(cfg: dict) -> list[str]:
    ids = [f"{fam}_L{L}_{s}_{sg}" for fam in ("v_prompt", "v_unq") for L in (16, 24) for s in ("POST_SELF_ATTENTION", "POST_CROSS_ATTENTION_PRE_FFN")
           for sg in ("plus", "minus")]
    return ids + [f"RANDOM_L{L}_{s}" for L in (16, 24) for s in ("POST_SELF_ATTENTION", "POST_CROSS_ATTENTION_PRE_FFN")] + ["D2_L16_POST_CROSS_ATTENTION_PRE_FFN"]


def random_vec(seed_key: str) -> tuple[int, np.ndarray]:
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    v = np.random.Generator(np.random.PCG64(seed)).standard_normal(1280)
    return seed, (v / np.linalg.norm(v)).astype(np.float32)


def d0(he: np.ndarray, hb: np.ndarray):
    dl = he.astype(np.float64) - hb.astype(np.float64)
    n = float(np.linalg.norm(dl))
    if not math.isfinite(n) or n < 1e-4:
        return None
    return (dl / (n + 1e-6)).astype(np.float32)


def members(con: list[dict], d: str) -> tuple[list[int], list[int]]:
    key = lambda i: (con[i]["dialogue_id"], con[i]["utterance_id"], int(con[i]["t"]))
    A = sorted((i for i, p in enumerate(con) if p["dialogue_id"] != d and p["stratum"] == "EN-correct"), key=key)
    B = sorted((i for i, p in enumerate(con) if p["dialogue_id"] != d and p["stratum"] == "ZH-correct"), key=key)
    return A, B


# ===================================================================================================================

def cmd_prerun(args) -> dict:
    checks, notes = {}, {}
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    for rel in (CONFIG, PANEL, SPEC, DESIGN, "tests/test_st_loc0_contract.py"):
        checks[f"unchanged_since_freeze:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in cfg["source_sha256"].items():
        checks[f"source:{rel}"] = fhash(ROOT / rel) == h
    checks["panel_identity"] = canon({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel_identity_hash"]
    con, rq = P["construction_positions"], P["runtime_queries"]
    checks["runtime_order_equals_construction"] = [(q["utterance_id"], q["dialogue_id"], q["t"]) for q in rq] == \
        [(p["utterance_id"], p["dialogue_id"], p["t"]) for p in con]
    cnt = defaultdict(int)
    for p in con:
        cnt[p["stratum"]] += 1
    checks["strata_60_each"] = dict(cnt) == {s: 60 for s in STRATA} and len({p["dialogue_id"] for p in con}) == 20
    checks["runtime_queries_label_free"] = all(set(q) <= {"utterance_id", "dialogue_id", "t", "absolute_query", "content_prefix_sha256",
                                                          "forced_zh_query_input_sha256", "forced_en_query_input_sha256"} for q in rq)
    checks["utterances_label_free"] = not any(f in json.dumps(P["utterances"]) for f in ("stratum", "target_ids", "competitor", "EN-confusion"))
    # independent fold memberships vs historical fold files (counts) and panel
    hist = json.loads((ROOT / FOLDS).read_text())["folds"]
    ext = np.load(ROOT / EXTRACT)
    HB, HE = ext["H_B"], ext["H_E"]
    fold_ok, refit_max, status_ok = True, 0.0, True
    for d in sorted({p["dialogue_id"] for p in con}):
        A, B = members(con, d)
        hf = hist[d]
        fold_ok &= len(A) == hf["n_A"] and len(B) == hf["n_B"] and P["calibration_membership"]["folds"][d]["membership_sha256"] == hf["membership_hash"]
        fold_ok &= not any(con[i]["dialogue_id"] == d for i in A + B)
        r = refit(HB[A], HB[B])
        status_ok &= r["status"] == hf["status"] and (r["status"] != "ok" or r["index"] == hf["selected_index"])
        if r["status"] == "ok":
            refit_max = max(refit_max, float(np.max(np.abs(r["vector"] - np.load(ROOT / hf["vector_path"])))))
    checks["fold_membership_independent"] = bool(fold_ok)
    checks["historical_D1_independent_refit"] = bool(status_ok) and refit_max <= cfg["historical_reproduction"]["D1_vector_max_abs"]
    notes["D1_refit_maxabs"] = refit_max
    # D0 reconstruction from sealed states vs sealed exp1 hashes
    utt = [u["utterance_id"] for u in P["utterances"]]
    d0_ok, n_d0 = True, 0
    for i, uid in enumerate(utt):
        row = json.loads((ROOT / EXP1 / "rows" / f"{i:03d}.json").read_text())
        d0_ok &= row["identity"] == uid if "identity" in row else True
        for j, q in enumerate(rq):
            if q["utterance_id"] != uid:
                continue
            v = d0(HE[j], HB[j])
            h = row["positions"][str(q["t"])]["directions"]["D0"]["sha256"]
            d0_ok &= (v is None and h is None) or (v is not None and ahash(v) == h)
            n_d0 += 1
    checks["historical_D0_independent"] = bool(d0_ok) and n_d0 == 180
    # random controls
    rnd_ok = True
    for c in cfg["controls"]["random"]:
        seed, v = random_vec(c["seed_key"])
        rnd_ok &= seed == int(c["seed"]) and ahash(v) == c["array_sha256"]
    checks["random_controls_independent"] = bool(rnd_ok)
    # plan + arm matrix
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    checks["plan_hash"] = canon({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"]
    checks["plan_checks_all_true"] = all(plan["checks"].values()) and len(plan["checks"]) >= 14
    checks["plan_runtime_projection_only"] = set(plan["runtime"]) == {"runtime_queries", "utterances"} and \
        plan["runtime"]["runtime_queries"] == rq and plan["runtime"]["utterances"] == P["utterances"]
    checks["plan_no_labels"] = not any(f in json.dumps(plan) for f in ("stratum", "EN-confusion", "target_ids", "competitor", "construction_positions"))
    checks["arm_matrix_21"] = [a["id"] for a in plan["arms"]] == independent_arms(cfg) and len(cfg["primary_arms"]) == 16
    checks["cells_3780"] = len(plan["arms"]) * 180 == cfg["compute"]["pulse_cells"] == 3780
    checks["energy_frozen"] = cfg["energy"]["e_star"] == json.loads((ROOT / "configs/inference_cf/p2_dir_direction_identification.json").read_text())["energy"]["e_star"] \
        and cfg["energy"]["norm_preserve"] and not cfg["energy"]["depth_rescale"]
    # firewall (static): runtime modules carry no reference/evaluator/label identifiers and no forbidden data
    for rel in (SITES_MOD,):
        notes[f"forbidden:{rel}"] = forbidden_hits(rel)
        checks[f"reference_free:{rel}"] = not notes[f"forbidden:{rel}"]
    for fn in ("cmd_run", "cell_record", "runtime_projection", "cmd_prepare"):
        notes[f"forbidden:{RUNNER}:{fn}"] = forbidden_hits(RUNNER, fn)
        checks[f"reference_free:{RUNNER}:{fn}"] = not notes[f"forbidden:{RUNNER}:{fn}"]
    calib_names = names_in(CALIB)
    notes[f"forbidden:{CALIB}"] = forbidden_hits(CALIB, allow=("stratum",))
    checks["calibration_builder_reads_only_allowed"] = not ({"runtime_queries", "utterances"} & calib_names) and not notes[f"forbidden:{CALIB}"]
    run_src = (ROOT / RUNNER).read_text()
    checks["runner_calibration_uses_allowlist"] = "calib_in = {k: Pfull[k] for k in calib.ALLOWED}" in run_src and \
        re.search(r"ALLOWED = \(\"construction_positions\", \"calibration_membership\"\)", (ROOT / CALIB).read_text()) is not None
    notes["forbidden:primary_analysis"] = forbidden_hits(ANALYZE, "primary", allow=("stratum",)) + \
        sorted({"POSITIONS", "HIST_ANALYSIS"} & names_in(ANALYZE, "primary"))       # module constants: exact-name match
    checks["primary_analysis_reference_free"] = not notes["forbidden:primary_analysis"]
    sec = (ROOT / ANALYZE).read_text()
    checks["secondary_gated_on_seal_and_primary_audit"] = "committed(SEAL)" in sec and "committed(PRIMARY_AUDIT)" in sec
    i_seal = run_src.index('atomic_json(out / "calibration" / "calibration_seal.json", seal)')
    checks["calibration_seal_GO_before_pulses"] = i_seal < run_src.index("while not go.exists():") < run_src.index("def run_pass(") < run_src.index('run_pass("barrier"')
    checks["barrier_before_new_sites"] = run_src.index('run_pass("barrier"') < run_src.index('run_pass("pulses"') and \
        "historical barrier failed (no new-site pulses run)" in run_src
    sb = (ROOT / SBATCH).read_text()
    tm = re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", sb)
    checks["sbatch_time_le_3h"] = tm is not None and int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + int(tm.group(3)) <= cfg["compute"]["hard_ceiling_seconds"]
    checks["sbatch_requires_pass"] = '"verdict": "PASS_TO_ST_LOC0"' in sb and sb.count("experiments/inference_cf_st_loc0.py run") == 1
    # compute projection from the historical P2-DIR job (extraction + 540 pulses incl. D2 autograd) scaled to 4500 pulse calls
    em = json.loads((ROOT / "results/inference_cf/p2dir/extract_run1/runtime.json").read_text())
    xm = json.loads((ROOT / EXP1 / "runtime.json").read_text())
    ext_s = float(em.get("elapsed_sec") or (em["end_unix"] - em["start_unix"]))
    exp_s = float(xm.get("elapsed_sec") or (xm["end_unix"] - xm["start_unix"]))
    calls = 180 * 21 + 180 * 4 + 180 * 2         # pulses + zero-dose + clean/restore per pass
    proj = 2 * ext_s + exp_s * calls / 540 + 600 + 1800      # x2 extraction (4-site recorder), +fits/hashing, +30 min GO margin
    notes["compute_projection_sec"] = {"historical_extract": ext_s, "historical_exp1": exp_s, "projected": proj}
    checks["compute_projection_le_3h"] = proj <= cfg["compute"]["hard_ceiling_seconds"]
    for t in ("tests/test_st_loc0_contract.py", "tests/test_st_loc0_impl.py"):
        checks[f"test_present:{t}"] = (ROOT / t).exists()
    me = (ROOT / "experiments/inference_cf_st_loc0_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(st_loc0_analyze|p2dir_analyze|inference_cf_st_loc0\b|unique|st_loc0_calibrate)", me, re.M) is None
    checks["no_existing_run_outputs"] = not (ROOT / BASE / "run1" / "manifest.json").exists()
    verdict = "PASS_TO_ST_LOC0" if all(checks.values()) else "BLOCK_BEFORE_ST_LOC0"
    return {"schema": "st_loc0_prerun_audit_v1", "verdict": verdict, "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


# ===================================================================================================================

def load_run(run: Path) -> dict:
    man = json.loads((run / "manifest.json").read_text())
    rows = {}
    for nm in ("barrier", "pulses"):
        for i in range(80):
            p = run / nm / f"{i:03d}.json"
            rows[(nm, i)] = json.loads(p.read_text()) if p.exists() else None
    return {"manifest": man, "rows": rows}


def cmd_primary(args) -> dict:
    checks, notes = {}, {}
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    con = P["construction_positions"]
    run = ROOT / args.run
    d = load_run(run)
    man = d["manifest"]
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    arms = {a["id"]: a for a in plan["arms"]}
    checks["manifest_hash"] = canon({k: v for k, v in man.items() if k != "manifest_hash"}) == man["manifest_hash"]
    checks["manifest_sources_at_commit"] = all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items())
    rt = json.loads((run / "runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and rt["n_failures"] == 0 and rt["model_grads_none"]
    checks["no_autograd_no_LID"] = rt["counters"]["autograd_calls"] == 0 and rt["counters"]["LID_calls"] == 0
    # calibration seal on remote before pulses
    cal = json.loads((run / "calibration" / "calibration_seal.json").read_text())
    go = json.loads((run / "calibration" / "GO.json").read_text())
    checks["calibration_seal_hash"] = canon({k: v for k, v in cal.items() if k != "seal_hash"}) == cal["seal_hash"] == go["calibration_seal_hash"]
    rel_seal = str((run / "calibration" / "calibration_seal.json").relative_to(ROOT))
    checks["calibration_seal_in_GO_remote_commit"] = git_blob_hash(go["remote_commit"], rel_seal) == fhash(run / "calibration" / "calibration_seal.json") and \
        subprocess.run(["git", "merge-base", "--is-ancestor", go["remote_commit"], "HEAD"], cwd=ROOT).returncode == 0
    checks["calibration_files_unchanged"] = all(fhash(run / "calibration" / n) == h for n, h in cal["files"].items())
    checks["calibration_files_in_GO_commit"] = all(git_blob_hash(go["remote_commit"], str((run / "calibration" / n).relative_to(ROOT))) == h for n, h in cal["files"].items())
    # V1 independent (all 4 sites x 180) and V2 independent refits at all 4 sites
    v1rec = json.loads((run / "calibration" / "v1_records.json").read_text())
    S = {}
    v1_ok, n_inv = True, 0
    for sk in SK:
        with np.load(run / "calibration" / f"states_{sk}.npz") as z:
            S[sk] = (z["H_B"], z["H_E"])
        for j in range(180):
            v = d0(S[sk][1][j], S[sk][0][j])
            r = v1rec[f"q{j:03d}_{sk}"]
            v1_ok &= (v is None and r["sha256"] is None) or (v is not None and ahash(v) == r["sha256"])
            n_inv += v is None
    ext = np.load(ROOT / EXTRACT)
    checks["L16_CROSS_states_equal_historical"] = bool(np.array_equal(S["L16_CROSS"][0], ext["H_B"]) and np.array_equal(S["L16_CROSS"][1], ext["H_E"]))
    checks["V1_independent_all_sites"] = bool(v1_ok)
    notes["V1_invalid"] = n_inv
    hist = json.loads((ROOT / FOLDS).read_text())["folds"]
    fv = {}
    v2_ok, v2_max, v2_status = True, 0.0, {}
    for sk in SK:
        summ = json.loads((run / "calibration" / f"folds_{sk}.json").read_text())
        with np.load(run / "calibration" / f"folds_{sk}_vectors.npz") as z:
            vec = {k: z[k] for k in z.files}
        fv[sk] = vec
        for dlg in sorted(hist):
            A, B = members(con, dlg)
            r = refit(S[sk][0][A], S[sk][0][B])
            v2_status[f"{sk}:{dlg}"] = r["status"]
            ok = r["status"] == summ[dlg]["status"] and (r["status"] == "ok") == (dlg in vec)
            if r["status"] == "ok" and dlg in vec:
                ok &= r["index"] == summ[dlg]["selected_index"]
                v2_max = max(v2_max, float(np.max(np.abs(r["vector"] - vec[dlg]))))
            v2_ok &= ok
    with np.load(run / "calibration" / "folds_L16_CROSS_historical_vectors.npz") as z:
        histv = {k: z[k] for k in z.files}
    checks["L16_CROSS_uses_historical_vectors"] = all(ahash(histv[k].astype(np.float32)) == hist[k]["vector_sha256"] for k in hist) and len(histv) == 20
    checks["V2_independent_refit_all_sites"] = bool(v2_ok) and v2_max <= cfg["historical_reproduction"]["D1_vector_max_abs"]
    notes["V2_refit_maxabs"] = v2_max
    notes["V2_invalid_folds"] = {k: v for k, v in v2_status.items() if v != "ok"}
    with np.load(run / "calibration" / "v1_vectors.npz") as z:
        V1 = {k: z[k] for k in z.files}
    with np.load(run / "calibration" / "random_vectors.npz") as z:
        RV = {k: z[k] for k in z.files}
    with np.load(run / "calibration" / "d2_vectors.npz") as z:
        D2 = {k: z[k] for k in z.files}
    with np.load(run / "calibration" / "baseline_logits.npz") as z:
        base = {k: z[k] for k in z.files}

    def expected_dir(a, j):
        fam, sk = arms[a]["family"], sk_of(arms[a])
        if fam == "v_prompt":
            v = V1.get(f"q{j:03d}_{sk}")
        elif fam == "v_unq":
            v = (histv if sk == "L16_CROSS" else fv[sk]).get(con[j]["dialogue_id"])
        elif fam == "random":
            v = RV[a]
        else:
            v = D2[f"q{j:03d}"]
        if v is None:
            return None
        v = arms[a]["sign"] * v.astype(np.float32).astype(np.float64)
        return v / np.linalg.norm(v)
    # matrix, energy, no-edit, selection, barrier, lineage (independent from the analysis)
    e = cfg["energy"]["e_star"]
    tol = cfg["energy"]
    n_cells, bad, energy_pair_bad = 0, defaultdict(list), []
    valid_s = {a: defaultdict(int) for a in arms}
    valid_d = {a: defaultdict(set) for a in arms}
    hb_ok = True
    for i in range(80):
        rb, rp = d["rows"][("barrier", i)], d["rows"][("pulses", i)]
        if rb is None or rp is None:
            bad["missing_row"].append(i)
            continue
        hrow = json.loads((ROOT / EXP1 / "rows" / f"{i:03d}.json").read_text())
        with np.load(ROOT / EXP1 / "rows" / f"{i:03d}_logits.npz") as z:
            hl = {k: z[k] for k in z.files}
        with np.load(ROOT / EXP1 / "rows" / f"{i:03d}.npz") as z:
            hv = {k: z[k] for k in z.files}
        L, A = {}, {}
        for nm, r in (("barrier", rb), ("pulses", rp)):
            if r["manifest_hash"] != man["manifest_hash"] or fhash(run / nm / f"{i:03d}_logits.npz") != r["logits_sha256"] \
                    or fhash(run / nm / f"{i:03d}_arrays.npz") != r["arrays_sha256"]:
                bad["row_hash"].append((nm, i))
            with np.load(run / nm / f"{i:03d}_logits.npz") as z:
                L.update({k: z[k] for k in z.files})
            with np.load(run / nm / f"{i:03d}_arrays.npz") as z:
                A.update({k: z[k] for k in z.files})
        for t, pb in rb["positions"].items():
            pp = rp["positions"][t]
            j = pb["q"]
            s, dlg = con[j]["stratum"], con[j]["dialogue_id"]
            cells = {**pb["cells"], **pp["cells"]}
            if set(cells) != set(arms):
                bad["arm_set"].append(j)
            if not (pb["clean_replay_bitwise"] and pp["clean_replay_bitwise"] and pb["restore_bitwise"] and pp["restore_bitwise"]
                    and pb["cache_prefix_unchanged"] and pp["cache_prefix_unchanged"] and len(pb["zero_dose"]) == 4 and all(pb["zero_dose"].values())):
                bad["lineage"].append(j)
            hb_ok &= np.array_equal(hl[f"t{t}_none"], base[f"q{j:03d}"])
            ens = []
            for a, c in cells.items():
                n_cells += 1
                lg = L[f"q{j:03d}_{a}"]
                if not (c.get("cache_prefix_unchanged_after_arm") and c.get("cache_positions_ok")):
                    bad["cache_after_arm"].append((j, a))
                if a in BARRIER:
                    h = BARRIER[a]
                    if not np.array_equal(lg, hl[f"t{t}_{h}"]):
                        bad["barrier_logits"].append((j, a))
                    if c["steered"] and not np.array_equal(A[f"q{j:03d}_{a}_consumed"], hv[f"t{t}_post_{h}"]):
                        bad["barrier_state"].append((j, a))
                if c["steered"]:
                    before = A[f"q{j:03d}_before_{sk_of(arms[a])}"].astype(np.float64)
                    prop = np.linalg.norm(A[f"q{j:03d}_{a}_proposed"].astype(np.float64) - before)
                    cons = np.linalg.norm(A[f"q{j:03d}_{a}_consumed"].astype(np.float64) - before)
                    ok = abs(prop / e - 1) <= tol["max_relative_norm_error"] and abs(cons / e - 1) <= tol["max_relative_norm_error"] \
                        and abs(cons ** 2 / e ** 2 - 1) <= tol["max_relative_squared_error"] and abs(cons - prop) <= tol["consumed_vs_proposed_relative_norm_tolerance"] * prop
                    vx = expected_dir(a, j)
                    sv = A[f"q{j:03d}_{a}_solver_v"].astype(np.float64)
                    ok &= vx is not None and float(sv @ vx) >= 1 - 1e-9
                    if not ok or not c["valid"]:
                        bad["energy_or_direction"].append((j, a))
                    ens.append(cons ** 2)
                    valid_s[a][s] += 1
                    valid_d[a][s].add(dlg)
                else:
                    if c["valid"] or c["no_edit_reason"] not in cfg["cell_eligibility"]["reasons_for_no_edit"] or not np.array_equal(lg, base[f"q{j:03d}"]):
                        bad["no_edit"].append((j, a, c["no_edit_reason"]))
                    fam = arms[a]["family"]
                    exp_reason = {"v_prompt": "tiny_or_nonfinite_V1_delta", "v_unq": "invalid_new_site_D1_fold"}.get(fam)
                    if expected_dir(a, j) is None and c["no_edit_reason"] != exp_reason:
                        bad["no_edit_reason_mismatch"].append((j, a))
                    if expected_dir(a, j) is not None and c["no_edit_reason"] in ("tiny_or_nonfinite_V1_delta", "invalid_new_site_D1_fold"):
                        bad["no_edit_reason_mismatch"].append((j, a))
            if ens and max(ens) / min(ens) - 1 > tol["max_pairwise_relative_squared_error"]:
                energy_pair_bad.append(j)
    checks["matrix_complete_3780"] = n_cells == 3780 and not bad["missing_row"] and not bad["arm_set"]
    checks["row_hashes"] = not bad["row_hash"]
    checks["historical_baseline_logits_bitwise"] = bool(hb_ok)
    checks["historical_barrier_bitwise"] = not bad["barrier_logits"] and not bad["barrier_state"] and bool(rt["barrier"]["pass"])
    checks["lineage_zero_restore"] = not bad["lineage"] and not bad["cache_after_arm"]
    checks["energy_and_direction_selection"] = not bad["energy_or_direction"]
    checks["pairwise_energy"] = not energy_pair_bad
    checks["no_edit_exact_and_predetermined"] = not bad["no_edit"] and not bad["no_edit_reason_mismatch"]
    notes["failures"] = {k: v[:20] for k, v in bad.items() if v}
    notes["pairwise_energy_bad"] = energy_pair_bad[:20]
    E = cfg["cell_eligibility"]
    elig = {a: all(valid_s[a][s] >= E["minimum_valid_per_stratum"] and len(valid_d[a][s]) >= E["minimum_valid_dialogues_per_stratum"] for s in STRATA) for a in arms}
    notes["eligibility"] = elig
    notes["valid_counts"] = {a: dict(valid_s[a]) for a in arms}
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    checks["primary_analysis_agrees"] = all(prim["arms"][a]["eligible"] == elig[a] and prim["arms"][a]["valid_by_stratum"] == {s: valid_s[a][s] for s in STRATA} for a in arms)
    my_valid = all(v for k, v in checks.items() if k not in ("primary_analysis_agrees", "primary_valid_agrees"))
    checks["primary_valid_agrees"] = prim["valid"] == my_valid
    checks["primary_reference_free"] = prim["references_used"] is False and not (ROOT / BASE / "secondary_analysis.json").exists()
    seal = json.loads((ROOT / BASE / "output_seal.json").read_text())
    checks["output_seal_committed_pushed"] = git_blob_hash("HEAD", f"{BASE}/output_seal.json") == fhash(ROOT / BASE / "output_seal.json") and \
        subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", git("rev-parse", "--abbrev-ref", "@{u}")], cwd=ROOT).returncode == 0
    checks["output_seal_files"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items()) and seal["manifest_hash"] == man["manifest_hash"]
    notes["stage_valid_independent"] = my_valid
    verdict = "ST_LOC0_AUDIT: PASS" if all(checks.values()) else "ST_LOC0_AUDIT: FAIL"
    return {"schema": "st_loc0_primary_audit_v1", "phase": "primary (reference-free)", "verdict": verdict, "stage_valid": my_valid,
            "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


# ===================================================================================================================

def label_logic(arms: list[dict], valid: bool, c: dict) -> tuple[str, str | None]:
    if not valid:
        return "ST_LOC0_INVALID", None
    if not any(a["eligible"] and a["lo"] is not None and a["lo"] > 0 for a in arms):
        return "ST_LOC0_NO_FIXED_DIRECTION_LEVER", None
    dc = c["decision"]
    q = [a for a in arms if a["eligible"] and a["lo"] is not None and a["lo"] > dc["adjusted_lower_bound_strictly_above"] and a["est"] >= dc["minimum_EN_confusion_dialogue_macro_delta_margin_nats"]
         and a["corr"] >= dc["minimum_top1_reference_corrections"] and a["corr_d"] >= dc["minimum_correction_dialogues"]]
    if not q:
        return "ST_LOC0_LOCAL_EFFECT_ONLY", None
    best = sorted(q, key=lambda a: (-a["corr"], -a["corr_d"], -a["est"], a["damage"], a["id"]))[0]
    return "ST_LOC0_SITE_FEASIBLE", best["id"]


def cmd_full(args) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    checks, notes = {}, {}
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    con = P["construction_positions"]
    run = ROOT / args.run
    pa = json.loads((ROOT / BASE / "primary_audit.json").read_text())
    checks["primary_audit_pass_committed"] = pa["verdict"] == "ST_LOC0_AUDIT: PASS" and git_blob_hash("HEAD", f"{BASE}/primary_audit.json") == fhash(ROOT / BASE / "primary_audit.json")
    sec = json.loads((ROOT / BASE / "secondary_analysis.json").read_text())
    seal = json.loads((ROOT / BASE / "output_seal.json").read_text())
    checks["seal_unchanged"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items()) and sec["output_seal_hash"] == seal["seal_hash"]
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    arms = {a["id"]: a for a in plan["arms"]}
    primary_ids = [a["id"] for a in cfg["primary_arms"]]
    man = json.loads((run / "manifest.json").read_text())
    model = Path(man["model"]["dir"])
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())["positions"]
    with np.load(run / "calibration" / "baseline_logits.npz") as z:
        base = {k: z[k] for k in z.files}
    R = {}
    by_q = {}
    for nm in ("barrier", "pulses"):
        for i in range(80):
            r = json.loads((run / nm / f"{i:03d}.json").read_text())
            with np.load(run / nm / f"{i:03d}_logits.npz") as z:
                lg = {k: z[k] for k in z.files}
            for t, p in r["positions"].items():
                for a in p["cells"]:
                    by_q.setdefault(p["q"], {})[a] = lg[f"q{p['q']:03d}_{a}"]
    for j, p in enumerate(pos):
        t, Y, cc = int(p["t"]), [int(x) for x in p["target_ids"]], int(p["competitor"])
        n = metrics(bf16(base[f"q{j:03d}"]), t, sup, beg, part, Y, cc)
        R[j] = {"none": n, "arms": {a: metrics(bf16(z), t, sup, beg, part, Y, cc) for a, z in by_q[j].items()}}
    # evaluator agreement with the secondary analysis + historical exp1 evaluator
    sa = {a["id"]: a for a in sec["arms"]}
    hist = json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1_analysis.json").read_text())["per_position"]
    hmax = 0.0
    for j in range(180):
        for a, h in BARRIER.items():
            hmax = max(hmax, abs((R[j]["arms"][a]["m"] - R[j]["none"]["m"]) - hist[j]["arms"][h]["d_m"]))
    checks["historical_evaluator_reproduced_1e-8"] = hmax <= cfg["historical_reproduction"]["evaluator_scalar_abs"]
    notes["historical_evaluator_max_abs"] = hmax
    # bootstrap (own draws)
    keys = sorted({p["dialogue_id"] for p in pos})
    idx = np.random.default_rng(cfg["bootstrap"]["seed"]).integers(0, len(keys), size=(cfg["bootstrap"]["replicates"], len(keys)))
    W = np.stack([(idx == k).sum(axis=1) for k in range(len(keys))], axis=1).astype(np.float64)
    alpha_adj = cfg["bootstrap"]["family_alpha"] / cfg["bootstrap"]["family_size"]

    def macro(vals):
        g = defaultdict(list)
        for dlg, v in vals:
            g[dlg].append(v)
        m = np.array([np.mean(g[k]) if k in g else np.nan for k in keys])
        use = ~np.isnan(m)
        num, den = W[:, use] @ m[use], W[:, use].sum(axis=1)
        dr = num[den > 0] / den[den > 0]
        return float(m[use].mean()), dr
    conf = [j for j in range(180) if con[j]["stratum"] == "EN-confusion"]
    mine, dm_max, ci_max, cnt_ok = [], 0.0, 0.0, True
    for a in arms:
        dm = {j: R[j]["arms"][a]["m"] - R[j]["none"]["m"] for j in range(180)}
        for s in STRATA:
            rs = [j for j in range(180) if con[j]["stratum"] == s]
            est, dr = macro([(con[j]["dialogue_id"], dm[j]) for j in rs])
            al = alpha_adj if (a in primary_ids and s == "EN-confusion") else 0.05
            lo_, hi_ = float(np.quantile(dr, al / 2)), float(np.quantile(dr, 1 - al / 2))
            st = sa[a]["strata"][s]
            dm_max = max(dm_max, abs(est - st["macro"]["estimate"]))
            ci_max = max(ci_max, abs(lo_ - st["macro"]["ci"][0]), abs(hi_ - st["macro"]["ci"][1]))
            corr_n = sum((not R[j]["none"]["in_ref"]) and R[j]["arms"][a]["in_ref"] for j in rs)
            corrupt_n = sum(R[j]["none"]["in_ref"] and not R[j]["arms"][a]["in_ref"] for j in rs)
            ch = sum(R[j]["arms"][a]["top"] != R[j]["none"]["top"] for j in rs)
            cnt_ok &= corr_n == st["corrections"] and corrupt_n == st["corruptions"] and ch == st["top1_changes"]
            if s == "EN-confusion":
                corr = [j for j in rs if (not R[j]["none"]["in_ref"]) and R[j]["arms"][a]["in_ref"]]
                rec = {"id": a, "est": est, "lo": lo_, "corr": len(corr), "corr_d": len({con[j]["dialogue_id"] for j in corr})}
        damage = sum(R[j]["none"]["in_ref"] and not R[j]["arms"][a]["in_ref"] for j in range(180) if con[j]["stratum"] != "EN-confusion")
        rec.update(damage=damage, eligible=sa[a]["eligible"])
        if a in primary_ids:
            mine.append(rec)
    checks["evaluator_and_macro_agree_1e-8"] = dm_max <= 1e-8
    checks["bootstrap_intervals_agree"] = ci_max <= 1e-9
    checks["counts_agree"] = bool(cnt_ok)
    notes["macro_max_abs"], notes["ci_max_abs"] = dm_max, ci_max
    pr = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    elig_pa = pa["notes"]["eligibility"]
    checks["eligibility_agrees_primary_audit"] = all(sa[a]["eligible"] == elig_pa[a] for a in arms)
    valid = pa["stage_valid"] and pr["valid"] and checks["historical_evaluator_reproduced_1e-8"]
    label, winner = label_logic(mine, valid, cfg)
    checks["label_agrees"] = label == sec["label"]
    checks["winner_agrees"] = winner == sec["decision"].get("winner")
    notes["independent_label"], notes["independent_winner"] = label, winner
    # oracle (primary arms only)
    def fam_set(fam):
        return {j for j in conf if not R[j]["none"]["in_ref"] and any(R[j]["arms"][a]["in_ref"] for a in primary_ids if arms[a]["family"] == fam)}
    v1, v2 = fam_set("v_prompt"), fam_set("v_unq")
    o = sec["oracle"]
    checks["oracle_agrees"] = (len(v1), len(v2), len(v1 | v2)) == (o["V1_correctable"], o["V2_correctable"], o["union_correctable"]) and \
        "NOT DEPLOYABLE" in o["label"] and not any(k.startswith(("RANDOM", "D2")) for k in o["contributing_arms"])
    checks["D2_random_never_selectable"] = all(x not in primary_ids for x in arms if arms[x]["family"] in ("random", "D2")) and \
        (winner is None or arms[winner]["family"] in ("v_prompt", "v_unq"))
    entries = sorted(p.name for p in (ROOT / BASE).iterdir())
    allowed = {"plan_sealed.json", "prerun_audit.json", "run1", "primary_analysis.json", "output_seal.json", "primary_audit.json",
               "secondary_analysis.json", "final_audit.json"}
    checks["scope_no_extra_stages"] = set(entries) <= allowed
    notes["entries"] = entries
    checks["one_job"] = len(list((run).glob("slurm-*.out"))) == 1
    verdict = "ST_LOC0_AUDIT: PASS" if all(checks.values()) else "ST_LOC0_AUDIT: FAIL"
    return {"schema": "st_loc0_full_audit_v1", "phase": "full (post-seal)", "verdict": verdict, "label": label, "winner": winner,
            "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({"verdict": res["verdict"], "failed": [k for k, v in res["checks"].items() if not v]}, indent=1))


if __name__ == "__main__":
    main()
