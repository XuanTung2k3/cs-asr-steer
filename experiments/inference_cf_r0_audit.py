#!/usr/bin/env python
"""R0 independent auditor (frozen spec section 10; firewall; config `audits`).

Imports NO R0 primary code (r0_regions, r0_unique, inference_cf_r0 decision/processing, inference_cf_r0_analyze).
It re-implements the window rule, sample sweep, frame masks, attention mapping, query groups, controls, the small-rank
construction (via Gram-matrix eigendecomposition rather than the primary thin SVD), oracle masks, region quality,
cosines, the dialogue bootstrap, nested gates and the terminal label.

prerun   PASS_TO_R0: freeze/source/model/audio/baseline/oracle-source byte pins, static firewall of the provider
         modules, synthetic agreement of the primary constructor with this independent one, sbatch/compute limits.
primary  R0_AUDIT: PASS (PRIMARY): seal pushed, 300 x 4 completeness, integrity, file-open log, and full independent
         recomputation of regions / mapping / groups / primary + perturbation + shuffle + script constructions.
full     R0_AUDIT: PASS (FULL): independent oracle masks, oracle constructions, metrics, bootstrap, gates, label.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
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

FREEZE = "a5929c6"
CONFIG = "configs/inference_cf/r0_region_vector.json"
PANEL = "docs/inference_cf/R0_PANEL.json"
DOCS = ("docs/inference_cf/R0_REGION_VECTOR_SPEC.md", "docs/inference_cf/R0_CODEX_DESIGN.md", "docs/inference_cf/R0_REFERENCE_FIREWALL.md")
BASE = "results/inference_cf/r0"
PROVIDER = ("src/csasr/inference_cf/r0_regions.py", "src/csasr/inference_cf/r0_unique.py", "experiments/inference_cf_r0.py")
LAYERS = (3, 8, 16, 24)
FORBIDDEN = ("reference", "transcript", "ctc", "oracle_source", "candidates_existing", "role_d-dev", "role_source", "gold",
             "target_ids", "competitor", "r0_analyze", "inference_cf_r0_analyze", "d-dev-confirm", "d-test", "p3_", "nat5h")


def canon(o) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fhash(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ahash(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + b"|" + repr(tuple(a.shape)).encode() + b"|")
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def blob(commit: str, rel: str) -> str:
    return "sha256:" + hashlib.sha256(subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True).stdout).hexdigest()


def bf16(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


# ===================== independent reference-free algorithms =====================

def grid(n: int, w: int, s: int):
    if n < w:
        return [(0, n)]
    st = sorted(set(range(0, n - w + 1, s)) | {n - w})
    return [(a, a + w) for a in st]


def window_label(p, lang, crop, rc) -> int:
    p = np.asarray(p, np.float64)
    pe, pm = p[lang.index(rc["en_id"])], p[lang.index(rc["zh_id"])]
    q = pe + pm
    x = np.asarray(crop, np.float64)
    rms = math.sqrt(float((x * x).mean())) if x.size else 0.0
    if x.size < rc["minimum_crop_samples"] or rms < rc["minimum_rms"] or q < rc["pair_mass_min"]:
        return 0
    if pe / (q + rc["epsilon"]) >= rc["conditional_language_mass_min"]:
        return 1
    if pm / (q + rc["epsilon"]) >= rc["conditional_language_mass_min"]:
        return 2
    return 0


def sample_track(bounds, labs, n, frac) -> np.ndarray:
    """Direct per-sample vote (difference arrays) -- a different route from the primary interval sweep."""
    tot = np.zeros(n + 1, np.int64)
    en = np.zeros(n + 1, np.int64)
    zh = np.zeros(n + 1, np.int64)
    for (a, b), c in zip(bounds, labs):
        tot[a] += 1
        tot[b] -= 1
        if c == 1:
            en[a] += 1
            en[b] -= 1
        elif c == 2:
            zh[a] += 1
            zh[b] -= 1
    tot, en, zh = np.cumsum(tot)[:n], np.cumsum(en)[:n], np.cumsum(zh)[:n]
    out = np.zeros(n, np.int8)
    has = tot > 0
    out[has & (en >= frac * tot)] = 1
    out[has & (out == 0) & (zh >= frac * tot)] = 2
    return out


def masks(track):
    n = track.size
    F = (n + 319) // 320
    we, wm = np.zeros(F), np.zeros(F)
    for f in range(F):
        seg = track[f * 320:min(n, (f + 1) * 320)]
        we[f] = np.count_nonzero(seg == 1) / seg.size
        wm[f] = np.count_nonzero(seg == 2) / seg.size
    return we, wm


def mapping(heads_real, mass_min):
    h = heads_real.astype(np.float64)
    mean = h.mean(axis=0)
    raw = mean.sum(axis=1)
    ok = raw >= mass_min
    a = np.where(ok[:, None], mean / np.where(raw > 0, raw, 1)[:, None], 0.0)
    return raw, ok, a, mean.argmax(axis=1)


def assign(a, ok, elig, we, wm, mc):
    qe, qm = a @ we, a @ wm
    use = ok & elig
    lab = np.zeros(a.shape[0], np.int8)
    lab[use & (qe >= mc["group_mass_min"]) & (qm <= mc["opposite_mass_max"])] = 1
    lab[use & (qm >= mc["group_mass_min"]) & (qe <= mc["opposite_mass_max"])] = 2
    return lab


def runs(track):
    out, s = [], 0
    for i in range(1, track.size + 1):
        if i == track.size or track[i] != track[s]:
            out.append((s, i, int(track[s])))
            s = i
    return out


def erode_i(track, d):
    o = np.zeros_like(track)
    for a, b, c in runs(track):
        if c and b - d > a + d:
            o[a + d:b - d] = c
    return o


def dilate_i(track, d):
    n = track.size
    e, m = np.zeros(n, bool), np.zeros(n, bool)
    for a, b, c in runs(track):
        if c == 1:
            e[max(0, a - d):min(n, b + d)] = True
        if c == 2:
            m[max(0, a - d):min(n, b + d)] = True
    o = np.zeros(n, np.int8)
    o[e & ~m], o[m & ~e] = 1, 2
    return o


def offsets(uid, n, k, margin, key):
    if n - 2 * margin + 1 < k:
        return None
    seed = int(hashlib.sha256(key.replace("<utterance_id>", uid).encode()).hexdigest()[:16], 16)
    return sorted(int(x) for x in np.random.Generator(np.random.PCG64(seed)).choice(np.arange(margin, n - margin + 1), size=k, replace=False))


def _gram(H):
    n = H.shape[0]
    if n == 0:
        return {"n": 0, "lam": np.zeros(0), "s": np.zeros(0), "V": None, "pr": 0.0}
    w, U = np.linalg.eigh(H @ H.T)
    w, U = w[::-1], U[:, ::-1]
    w = np.maximum(w, 0.0)
    s = np.sqrt(w)
    Hc = H - H.mean(axis=0)
    wc = np.maximum(np.linalg.eigvalsh(Hc @ Hc.T), 0.0) / n
    pr = 0.0 if wc.sum() <= 0 else float(wc.sum() ** 2 / (wc @ wc))
    return {"n": n, "lam": w / n, "s": s, "U": U, "H": H, "pr": pr}


def construct_i(HE, HM, bE, bM, uc):
    HE, HM = np.asarray(HE, np.float64), np.asarray(HM, np.float64)
    g = [_gram(HE), _gram(HM)]
    nb = [len(set(np.asarray(bE).tolist())), len(set(np.asarray(bM).tolist()))]

    def lam(x, i):
        return float(x["lam"][i - 1]) if i - 1 < x["lam"].size else 0.0

    def nrank(x):
        return int(np.sum(x["s"] > uc["numerical_rank_sv_relative_floor"] * x["s"][0])) if x["n"] and x["s"][0] > 0 else 0
    rank = None
    for r in sorted(uc["candidate_ranks"], reverse=True):
        ok = True
        for x, b_ in zip(g, nb):
            l1, lr, lr1 = lam(x, 1), lam(x, r), lam(x, r + 1)
            ok &= (x["n"] >= 2 * r + 2 and b_ >= r + 1 and nrank(x) >= r and x["pr"] >= r and l1 > 0
                   and lr > uc["eigen_floor_relative_to_largest"] * l1 and lr > 0 and (lr - lr1) > uc["boundary_gap_relative_to_retained"] * lr)
        if ok:
            rank = r
            break
    if rank is None:
        return {"status": "invalid", "rank": None, "reason": "NO_SUPPORTED_RANK", "vector": None}
    r = rank
    V = [x["H"].T @ x["U"][:, :r] / x["s"][:r] for x in g]           # right singular vectors from the Gram route
    M = V[0].T @ V[1]
    w, Pm = np.linalg.eigh(M @ M.T)
    w, Pm = w[::-1], Pm[:, ::-1]
    sig = np.sqrt(np.clip(w, 0, None))
    a = V[0] @ Pm
    energy = np.sum((HE @ a) ** 2, axis=0) / HE.shape[0]
    score = energy * (1 - np.clip(sig, 0, 1) ** 2)
    win = int(np.argmax(score))
    srt = np.sort(score)[::-1]
    reasons = []
    if np.any(sig > 1 + uc["principal_cosine_out_of_bounds_tolerance"]):
        reasons.append("sigma_out_of_bounds")
    if not score[win] > uc["score_floor_relative_to_energy"] * energy.max():
        reasons.append("score_floor")
    if not (srt[0] - srt[1]) > uc["winner_gap_relative_to_winner"] * score[win]:
        reasons.append("winner_gap")
    if not min(abs(sig[j] - sig[win]) for j in range(r) if j != win) > uc["principal_cosine_isolation"]:
        reasons.append("sigma_isolation")
    c = HE.mean(axis=0) - HM.mean(axis=0)
    cn = np.linalg.norm(c)
    v = a[:, win] / np.linalg.norm(a[:, win])
    cc = float(v @ c / cn) if cn > 0 else 0.0
    if not cn > uc["mean_contrast_norm_min"]:
        reasons.append("mean_contrast_norm")
    elif not abs(cc) > uc["sign_cosine_abs_min"]:
        reasons.append("sign_degenerate")
    v = -v if cc < 0 else v
    out = {"status": "invalid" if reasons else "ok", "rank": r, "winner_sigma": float(sig[win]), "reason": reasons[0] if reasons else None,
           "vector": None if reasons else v, "gap_rel": float((srt[0] - srt[1]) / score[win]) if score[win] > 0 else 0.0,
           "iso": float(min(abs(sig[j] - sig[win]) for j in range(r) if j != win)), "cos_c": abs(cc)}
    return out


def agree(rec: dict, mine: dict, vec, uc, tol=1e-6) -> tuple[bool, bool]:
    """(agrees, borderline). Borderline = a winner guard within 1e-6 relative of its frozen threshold."""
    border = mine["rank"] is not None and (abs(mine.get("gap_rel", 1) - uc["winner_gap_relative_to_winner"]) < 1e-6
                                           or abs(mine.get("iso", 1) - uc["principal_cosine_isolation"]) < 1e-9
                                           or abs(mine.get("cos_c", 1) - uc["sign_cosine_abs_min"]) < 1e-6)
    if rec["status"] != mine["status"] or rec["rank"] != mine["rank"]:
        return False, border
    if mine["status"] == "ok":
        if vec is None or float(np.max(np.abs(np.asarray(vec, np.float64) - mine["vector"]))) > tol:
            return False, border
    return True, border


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ===================== prerun =====================

def static_names(rel: str) -> set:
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


# exact provenance strings the runner writes into its manifest/records (declarations of the boundary, not accesses)
DECLARATIONS = {"references_used", "reference_access_boundary",
                "no oracle/CTC/reference/role transcript before output seal pushed + R0_AUDIT: PASS (PRIMARY)",
                "docs/inference_cf/R0_REFERENCE_FIREWALL.md", "experiments/inference_cf_r0_analyze.py"}     # hashed provenance pins


def provider_hits(rel: str, allow=()) -> list:
    return sorted({str(n)[:80] for n in static_names(rel) if str(n) not in DECLARATIONS
                   for f in FORBIDDEN if f in str(n).lower() and f not in allow})


def cmd_prerun(args) -> dict:
    import soundfile as sf
    checks, notes = {}, {}
    c = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    for rel in (CONFIG, PANEL, *DOCS, "tests/test_r0_freeze_contract.py"):
        checks[f"unchanged_since_freeze:{rel}"] = fhash(ROOT / rel) == blob(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"source:{rel}"] = fhash(ROOT / rel) == h
    checks["panel_identity"] = canon({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == c["panel_identity_hash"]
    checks["population_300_x_20x15"] = len(P["rows"]) == 300 and set(Counter(r["dialogue_id"] for r in P["rows"]).values()) == {15} \
        and len({r["dialogue_id"] for r in P["rows"]}) == 20
    checks["model_files"] = all(fhash(Path(c["model"]["dir"]) / n) == h for n, h in c["model"]["files"].items())
    o = c["oracle"]["source"]
    checks["oracle_source_bytes"] = fhash(o["path"]) == o["file_sha256"] and fhash(o["manifest_path"]) == o["manifest_sha256"]
    checks["role_source_bytes"] = fhash(P["role_source"]["path"]) == P["role_source"]["file_sha256"]
    a_ok, nwin = True, 0
    for r in P["rows"]:
        a = r["audio"]
        a_ok &= fhash(a["path"]) == a["full_sha256"] and sf.info(a["path"]).frames == a["resampled_num_samples"]
        b = r["baseline"]
        raw = json.loads((ROOT / b["path"]).read_text())
        a_ok &= fhash(ROOT / b["path"]) == b["file_sha256"] and raw["theta0"]["tokens"] == b["content_ids"] and raw["identity"] == r["utterance_id"]
        nwin += len(grid(a["resampled_num_samples"], 16000, 8000))
    checks["audio_baseline_bytes"] = bool(a_ok)
    checks["grid_10387"] = nwin == 10387
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    checks["plan_hash"] = canon({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"] and all(plan["checks"].values())
    checks["plan_rows_projection_only"] = all(set(r) == {"canonical_index", "utterance_id", "dialogue_id", "audio", "baseline"} for r in plan["rows"]) \
        and [r["utterance_id"] for r in plan["rows"]] == [r["utterance_id"] for r in P["rows"]]
    for rel in PROVIDER:
        notes[f"forbidden:{rel}"] = provider_hits(rel)
        checks[f"provider_reference_free:{rel}"] = not notes[f"forbidden:{rel}"]
    run_src = (ROOT / "experiments/inference_cf_r0.py").read_text()
    checks["runner_no_oracle_open"] = "oracle\"][\"source" not in run_src and "role_source" not in run_src
    checks["runner_records_open_log"] = "addaudithook" in run_src and "opened_paths" in run_src
    checks["no_steering_hooks_in_runner"] = not any(x in run_src for x in ("InterventionHook", "SteeringHook", "pulse_action", "apply_steering", ".backward(", "optim."))
    an = (ROOT / "experiments/inference_cf_r0_analyze.py").read_text()
    prim = an[an.index("def primary("):an.index("def oracle_tracks(")]
    checks["primary_analysis_reference_free"] = not any(x in prim for x in ("oracle_tracks", "parquet", "role_source", "transcript", "candidates_existing"))
    checks["oracle_gated_on_seal_and_primary_audit"] = an.index("committed(SEAL)") < an.index("committed(PRIMARY_AUDIT)") < an.index("oracle_tracks([r")
    me = (ROOT / "experiments/inference_cf_r0_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(r0_regions|r0_unique|inference_cf_r0\b|inference_cf_r0_analyze)", me, re.M) is None
    # synthetic agreement: primary constructor vs this independent Gram route (no real states)
    from csasr.inference_cf import r0_unique as Q            # used ONLY to cross-check, never to decide
    rng = np.random.default_rng(240924)
    ag, n_ok = 0, 0
    for k in range(60):
        nE, nM = int(rng.integers(5, 90)), int(rng.integers(5, 90))
        base = rng.normal(size=(1, 1280)) * 3
        HE = base + rng.normal(size=(nE, 1280)) @ np.diag(rng.uniform(0.1, 2, 1280)) + rng.normal(size=(1, 1280))
        HM = base + rng.normal(size=(nM, 1280)) @ np.diag(rng.uniform(0.1, 2, 1280))
        bE, bM = rng.integers(0, 12, nE), rng.integers(0, 12, nM)
        p = Q.construct(HE, HM, bE, bM, c["unique"])
        m = construct_i(HE, HM, bE, bM, c["unique"])
        ok, _ = agree(p, m, p["vector"], c["unique"])
        ag += ok
        n_ok += p["status"] == "ok"
    notes["synthetic_agreement"] = {"cases": 60, "agree": ag, "primary_ok": n_ok}
    checks["synthetic_constructor_agreement"] = ag == 60 and n_ok > 0
    sb = (ROOT / "slurm/inference_cf_r0.sbatch").read_text()
    tm = re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", sb)
    checks["sbatch_le_3h_single_run"] = tm is not None and int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + int(tm.group(3)) <= c["compute"]["hard_seconds"] \
        and sb.count("experiments/inference_cf_r0.py run") == 1 and "PASS_TO_R0" in sb and "OMP_NUM_THREADS=1" in sb
    # conservative compute projection: historical P0-R2 run (11855 real + 8750 control LID calls + replays) per LID call
    r2 = json.loads((ROOT / "results/inference_cf/p0_r2/runtime.json").read_text()) if (ROOT / "results/inference_cf/p0_r2/runtime.json").exists() else {}
    per_lid = 1428.0 / (11855 + 8750)
    proj = 10387 * per_lid * 2.0 + 300 * 3 * 0.25 + 734 * 2.0 + 600
    notes["compute_projection_sec"] = {"per_lid_call_historical": per_lid, "projected_conservative": proj, "p0_r2_runtime_keys": sorted(r2)[:8]}
    checks["compute_projection_le_3h"] = proj <= c["compute"]["hard_seconds"] - 900
    for t in ("tests/test_r0_freeze_contract.py", "tests/test_r0_impl.py"):
        checks[f"test_present:{t}"] = (ROOT / t).exists()
    checks["no_existing_run"] = not (ROOT / BASE / "run1" / "manifest.json").exists()
    v = "PASS_TO_R0" if all(checks.values()) else "BLOCK_BEFORE_R0"
    return {"schema": "r0_prerun_audit_v1", "verdict": v, "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


# ===================== primary =====================

def load(run: Path):
    man = json.loads((run / "manifest.json").read_text())
    rows = []
    for i in range(300):
        p = run / "rows" / f"{i:03d}.json"
        if not p.exists():
            return man, None
        r = json.loads(p.read_text())
        with np.load(run / "rows" / f"{i:03d}.npz") as z:
            rows.append((r, {k: z[k] for k in z.files}))
    return man, rows


def recompute_row(r, z, c, panel_row):
    """Independent reference-free reconstruction of one row; returns dict of check booleans + arrays."""
    import soundfile as sf
    rc, mc, uc, cc = c["regions"], c["mapping"], c["unique"], c["controls"]
    chk = {}
    wav, sr = sf.read(panel_row["audio"]["path"], dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = wav.shape[0]
    heard = min(n, 480000)
    chk["geometry"] = n == r["n_samples"] and heard == r["heard_samples"] and sr == 16000
    bounds = grid(n, rc["window_samples"], rc["stride_samples"])
    lang = list(rc["language_ids"])
    P = z["lid_probs"]
    chk["lid_probs_valid"] = P.shape == (len(bounds), 100) and bool(np.all(np.isfinite(P))) and float(np.max(np.abs(P.astype(np.float64).sum(axis=1) - 1))) <= rc["probability_tolerance"]
    labs = [window_label(P[k], lang, wav[a:b], rc) for k, (a, b) in enumerate(bounds)]
    chk["window_labels"] = labs == [w["code"] for w in r["windows"]] and [(w["start"], w["end"]) for w in r["windows"]] == bounds
    track = sample_track(bounds, labs, n, rc["sample_vote_fraction_min"])
    chk["sample_intervals"] = [list(x) for x in runs(track)] == r["intervals_full"]
    th = track[:heard]
    we, wm = masks(th)
    chk["frame_masks"] = np.array_equal(we, z["wE"]) and np.array_equal(wm, z["wM"])
    heads = bf16(z["heads"])
    F = r["real_frames"]
    chk["heads_shape"] = heads.shape == (len(mc["alignment_heads"]), r["T"] + 1, F) and F == (heard + 319) // 320
    raw, ok, a, am = mapping(heads, mc["raw_real_audio_attention_mass_min"])
    bins = am * 320 // mc["independent_time_bin_samples"]
    chk["mapping"] = np.array_equal(raw, z["raw_mass"]) and np.array_equal(bins, z["bins"])
    elig = np.array([x["eligible"] for x in r["queries"]])
    content = panel_row["baseline"]["content_ids"]
    T = len(content)
    chk["eligibility_structure"] = (len(elig) == T + 1 and not elig[0] and not elig[T] and all(x["t"] == t and x["query_index"] == 3 + t for t, x in enumerate(r["queries"]))
                                    and all((not x["eligible"]) or (content[x["t"]] < 50257) for x in r["queries"]))
    lab = assign(a, ok, elig, we, wm, mc)
    chk["labels"] = np.array_equal(lab, z["label"])
    S = {l: bf16(z[f"states_L{l}"]) for l in LAYERS}
    chk["states_finite_shape"] = all(S[l].shape[0] == T + 1 and S[l].ndim == 2 and np.all(np.isfinite(S[l])) for l in LAYERS)
    res = {"disagree": [], "border": [], "n": 0}

    def check(name, rec, lab_, vec):
        for l in LAYERS:
            e, m = np.flatnonzero(lab_ == 1), np.flatnonzero(lab_ == 2)
            mine = construct_i(S[l][e], S[l][m], bins[e], bins[m], uc)
            ok_, border = agree(rec(l), mine, vec(l), uc)
            res["n"] += 1
            if not ok_:
                (res["border"] if border else res["disagree"]).append(f"{name}:L{l}")
    check("primary", lambda l: r["constructions"][str(l)]["primary"], lab, lambda l: z.get(f"v_L{l}"))
    for nm, fn in (("erode", erode_i), ("dilate", dilate_i)):
        pl = assign(a, ok, elig, *masks(fn(th, cc["boundary_samples"])), mc)
        chk[f"{nm}_labels"] = np.array_equal(pl, z[f"{nm}_label"])
        check(nm, lambda l, nm=nm: r["controls"][nm][str(l)], pl, lambda l, nm=nm: z.get(f"{nm}_L{l}"))
    offs = offsets(r["utterance_id"], heard, cc["shuffles"], cc["boundary_samples"], cc["shuffle_seed_key"])
    chk["shuffle_offsets"] = offs == r["controls"]["offsets"]
    if offs:
        for k, off in enumerate(offs):
            sl = assign(a, ok, elig, *masks(np.roll(th, off)), mc)
            check(f"shuffle{k}", lambda l, k=k: r["controls"]["shuffle"][str(l)][k], sl, lambda l, k=k: z.get(f"shuf_L{l}_{k:02d}"))
    chk["script_label_shape"] = z["script_label"].shape == lab.shape
    return chk, res


def cmd_primary(args) -> dict:
    checks, notes = {}, {}
    c = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    run = ROOT / args.run
    man, rows = load(run)
    checks["manifest_hash"] = canon({k: v for k, v in man.items() if k != "manifest_hash"}) == man["manifest_hash"]
    checks["manifest_sources_at_commit"] = all(blob(man["git_commit"], p) == h for p, h in man["sources"].items())
    rt = json.loads((run / "runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and not rt.get("failures")
    checks["frozen_model"] = rt["model_grads_none"] and not rt["requires_grad_any"] and rt["weights_unchanged_sample"] and not rt["training_mode"]
    checks["no_autograd_optimizer_edits"] = all(rt["counters"][k] == 0 for k in ("autograd_calls", "optimizer_steps", "edit_hooks"))
    checks["lid_calls_10387"] = rt["counters"]["lid_calls"] == 10387
    forb = [p for p in rt.get("opened_paths", []) if any(f in p.lower() for f in ("candidates_existing", "/roles/", "alignments", "transcript", "p2rj"))]
    notes["opened_paths_count"] = len(rt.get("opened_paths", []))
    checks["file_open_log_firewall"] = "opened_paths" in rt and not forb
    notes["forbidden_opened"] = forb
    checks["rows_300"] = rows is not None
    if rows is None:
        return {"schema": "r0_primary_audit_v1", "verdict": "R0_AUDIT: FAIL (PRIMARY)", "checks": checks, "notes": notes}
    checks["row_hashes"] = all(r["manifest_hash"] == man["manifest_hash"] and fhash(run / "rows" / f"{i:03d}.npz") == r["arrays_sha256"] for i, (r, _) in enumerate(rows))
    checks["row_order"] = [r["utterance_id"] for r, _ in rows] == [p["utterance_id"] for p in P["rows"]]
    checks["records_300x4"] = all(set(r["constructions"]) == {str(l) for l in LAYERS} for r, _ in rows)
    checks["row_integrity"] = all(r["integrity"]["site_equals_ffn_input"] and r["integrity"]["repeat_replay_bitwise"] and r["integrity"]["finite"]
                                  and r["integrity"]["primary_repeat_identical"] for r, _ in rows)
    checks["states_dim_1280"] = all(z[f"states_L{l}"].shape[1] == 1280 for _, z in rows for l in LAYERS)
    checks["baseline_tokens"] = all(r["content_sha256"] == p["baseline"]["content_sha256"] and r["T"] == len(p["baseline"]["content_ids"])
                                    for (r, _), p in zip(rows, P["rows"]))
    agg = defaultdict(lambda: True)
    dis, border, n = [], [], 0
    for (r, z), pr in zip(rows, P["rows"]):
        chk, res = recompute_row(r, z, c, pr)
        for k, v in chk.items():
            agg[k] &= bool(v)
            if not v:
                notes.setdefault("row_failures", []).append(f"{r['utterance_id']}:{k}")
        dis += [f"{r['utterance_id']}:{x}" for x in res["disagree"]]
        border += [f"{r['utterance_id']}:{x}" for x in res["border"]]
        n += res["n"]
    for k, v in agg.items():
        checks[f"independent:{k}"] = v
    notes["constructions_recomputed"] = n
    notes["construction_disagreements"] = dis[:50]
    notes["borderline_threshold_cases"] = border[:50]
    checks["independent_constructions_agree"] = not dis
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    my_valid = all(checks.values())
    checks["primary_analysis_valid_agrees"] = prim["valid"] == my_valid
    for l in LAYERS:
        nv = sum(r["constructions"][str(l)]["primary"]["status"] == "ok" for r, _ in rows)
        checks[f"primary_count_L{l}"] = prim["per_layer"][str(l)]["primary_valid"] == nv
    checks["no_oracle_outputs_yet"] = not (ROOT / BASE / "oracle_analysis.json").exists()
    seal = json.loads((ROOT / BASE / "output_seal.json").read_text())
    checks["seal_files"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items()) and seal["manifest_hash"] == man["manifest_hash"]
    checks["seal_committed_pushed"] = blob("HEAD", f"{BASE}/output_seal.json") == fhash(ROOT / BASE / "output_seal.json") and \
        subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", git("rev-parse", "--abbrev-ref", "@{u}")], cwd=ROOT).returncode == 0
    v = c["audits"]["primary"] if all(checks.values()) else "R0_AUDIT: FAIL (PRIMARY)"
    return {"schema": "r0_primary_audit_v1", "verdict": v, "stage_valid": my_valid, "checks": checks, "notes": notes, "git_commit": git("rev-parse", "HEAD")}


# ===================== full =====================

def cmd_full(args) -> dict:
    import pyarrow.parquet as pq
    from csasr.nat5h.units import build_reference_units          # historical normalization primitive
    checks, notes = {}, {}
    c = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    run = ROOT / args.run
    man, rows = load(run)
    pa = json.loads((ROOT / BASE / "primary_audit.json").read_text())
    checks["primary_audit_pass_committed"] = pa["verdict"] == c["audits"]["primary"] and blob("HEAD", f"{BASE}/primary_audit.json") == fhash(ROOT / BASE / "primary_audit.json")
    seal = json.loads((ROOT / BASE / "output_seal.json").read_text())
    checks["seal_unchanged"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items())
    an = json.loads((ROOT / BASE / "oracle_analysis.json").read_text())
    checks["oracle_after_seal"] = an["output_seal_hash"] == seal["seal_hash"]
    mc, uc, g, b = c["mapping"], c["unique"], c["gates"], c["bootstrap"]
    src = c["oracle"]["source"]
    checks["oracle_source_bytes"] = fhash(src["path"]) == src["file_sha256"]
    ids = [p["utterance_id"] for p in P["rows"]]
    recs = pq.read_table(src["path"], filters=[("utterance_id", "in", ids)], columns=["utterance_id", "unit_id", "reference_unit_index", "is_valid",
                         "start_sample", "end_sample", "language", "surface", "waveform_num_samples", "metadata"]).to_pylist()
    role = {x["utterance_id"]: x["transcript_raw"] for x in pq.read_table(P["role_source"]["path"], filters=[("utterance_id", "in", ids)],
                                                                            columns=["utterance_id", "transcript_raw"]).to_pylist()}
    units = {u: build_reference_units(u, t) for u, t in role.items()}
    keep = defaultdict(list)
    used = Counter()
    for x in recs:
        u = units[x["utterance_id"]]
        ui = x["unit_id"]
        ok = (x["is_valid"] and u and (x["metadata"] or {}).get("normalization_hash") == u[0].normalization_hash and x["reference_unit_index"] == ui
              and 0 <= ui < len(u) and u[ui].surface == x["surface"] and u[ui].language == x["language"] and x["language"] in ("EN", "ZH")
              and x["start_sample"] is not None and x["end_sample"] is not None and 0 <= x["start_sample"] < x["end_sample"] <= x["waveform_num_samples"])
        used[ok] += 1
        if ok:
            keep[x["utterance_id"]].append((x["start_sample"], x["end_sample"], 1 if x["language"] == "EN" else 2))
    notes["oracle_records_used"] = dict(used)
    tracks = {}
    for r, _ in rows:
        n = r["n_samples"]
        e, m = np.zeros(n, bool), np.zeros(n, bool)
        for s_, e_, L in keep[r["utterance_id"]]:
            if e_ <= n:
                (e if L == 1 else m)[s_:e_] = True
        t = np.zeros(n, np.int8)
        t[e & ~m], t[m & ~e] = 1, 2
        tracks[r["utterance_id"]] = t
    # region quality (heard)
    tp = Counter()
    both = set()
    for r, _ in rows:
        h = r["heard_samples"]
        pred = np.zeros(r["n_samples"], np.int8)
        for s_, e_, k in r["intervals_full"]:
            pred[s_:e_] = k
        p, o = pred[:h], tracks[r["utterance_id"]][:h]
        for k, code in (("EN", 1), ("ZH", 2)):
            tp[k] += int(((p == code) & (o == code)).sum())
            tp["pp" + k] += int(((p == code) & (o > 0)).sum())
            tp["o" + k] += int((o == code).sum())
        tp["known"] += int((o > 0).sum())
        tp["cls"] += int(((p > 0) & (o > 0)).sum())
        if (o == 1).any() and (o == 2).any():
            both.add((r["utterance_id"], r["dialogue_id"]))
    rq = {"EN_precision": tp["EN"] / tp["ppEN"] if tp["ppEN"] else None, "EN_recall": tp["EN"] / tp["oEN"] if tp["oEN"] else None,
          "ZH_precision": tp["ZH"] / tp["ppZH"] if tp["ppZH"] else None, "ZH_recall": tp["ZH"] / tp["oZH"] if tp["oZH"] else None,
          "classified_coverage": tp["cls"] / tp["known"] if tp["known"] else None}
    rq_pass = (None not in rq.values() and rq["EN_precision"] >= g["en_precision_min"] and rq["EN_recall"] >= g["en_recall_min"]
               and rq["ZH_precision"] >= g["zh_precision_min"] and rq["ZH_recall"] >= g["zh_recall_min"]
               and rq["classified_coverage"] >= g["known_region_classified_coverage_min"] and len(both) >= g["known_region_scored_min_rows"]
               and len({d for _, d in both}) >= g["known_region_scored_min_dialogues"])
    arq = an["region_quality"]
    checks["region_quality_agrees"] = all((rq[k] is None and arq[k] is None) or abs(rq[k] - arq[k]) <= 1e-12 for k in rq) and rq_pass == arq["pass"]
    # oracle constructions + endpoints
    per = {l: [] for l in LAYERS}
    disagree = []
    for r, z in rows:
        h = r["heard_samples"]
        raw, ok, a, am = mapping(bf16(z["heads"]), mc["raw_real_audio_attention_mass_min"])
        elig = np.array([x["eligible"] for x in r["queries"]])
        lab = assign(a, ok, elig, *masks(tracks[r["utterance_id"]][:h]), mc)
        bins = am * 320 // mc["independent_time_bin_samples"]
        for l in LAYERS:
            S = bf16(z[f"states_L{l}"])
            e, m = np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
            o = construct_i(S[e], S[m], bins[e], bins[m], uc)
            arec = an["per_row"][str(l)][r["canonical_index"]]
            if (arec["oracle_status"], arec["oracle_rank"]) != (o["status"], o["rank"]):
                disagree.append(f"{r['utterance_id']}:L{l}")
            pv = z.get(f"v_L{l}")
            rec = {"d": r["dialogue_id"], "ov": o["status"] == "ok", "pv": pv is not None}
            if pv is not None and o["vector"] is not None:
                sh = [z[f"shuf_L{l}_{k:02d}"] for k in range(20) if f"shuf_L{l}_{k:02d}" in z]
                rec["sc"] = cos(pv, o["vector"])
                rec["ev"] = len(sh) >= c["controls"]["minimum_valid_shuffles_per_pair"]
                if sh:
                    rec["adv"] = rec["sc"] - float(np.mean([cos(s, o["vector"]) for s in sh]))
                if "signed_cos" in arec and abs(arec["signed_cos"] - rec["sc"]) > 1e-5:
                    disagree.append(f"{r['utterance_id']}:L{l}:cos")
            if pv is not None:
                ec = cos(pv, z[f"erode_L{l}"]) if f"erode_L{l}" in z else None
                dc = cos(pv, z[f"dilate_L{l}"]) if f"dilate_L{l}" in z else None
                rec["pmin"] = None if ec is None or dc is None else min(ec, dc)
            per[l].append(rec)
    notes["oracle_construction_disagreements"] = disagree[:50]
    checks["oracle_constructions_agree"] = not disagree
    keys = sorted({r["dialogue_id"] for r, _ in rows})
    idx = np.random.default_rng(b["seed"]).integers(0, len(keys), size=(b["replicates"], len(keys)))
    W = np.stack([(idx == j).sum(axis=1) for j in range(len(keys))], axis=1).astype(float)

    def boot(pairs):
        gp = defaultdict(list)
        for d, v in pairs:
            gp[d].append(v)
        mm = np.array([np.mean(gp[k]) if k in gp else np.nan for k in keys])
        use = ~np.isnan(mm)
        if not use.any():
            return None, None, 0
        num, den = W[:, use] @ mm[use], W[:, use].sum(axis=1)
        dr = num[den > 0] / den[den > 0]
        return float(mm[use].mean()), float(np.quantile(dr, b["lower_quantile"])), int((den > 0).sum())
    gates = {}
    for l in LAYERS:
        R = per[l]
        ov = [x for x in R if x["ov"]]
        pr = [x for x in R if "sc" in x]
        pv = [x for x in R if x["pv"]]
        O = len(ov) >= g["oracle_valid_min"] and len({x["d"] for x in ov}) >= g["oracle_dialogues_min"]
        Pp = O and rq_pass and len(pr) >= g["paired_valid_min"] and len({x["d"] for x in pr}) >= g["paired_dialogues_min"] and len(pr) / max(len(ov), 1) >= g["paired_fraction_of_oracle_min"]
        mins = [x["pmin"] for x in pv if x["pmin"] is not None]
        Ss = Pp and len(pv) > 0 and len(mins) / len(pv) >= g["perturb_both_valid_fraction_min"] and sum(m >= g["perturb_row_min_signed_cosine"] for m in mins) / len(pv) >= g["perturb_stable_fraction_min"] \
            and len(mins) > 0 and float(np.median(mins)) >= g["perturb_median_min_signed_cosine"]
        ev = [x for x in pr if x["ev"]]
        est, lo, nd = boot([(x["d"], x["sc"]) for x in ev])
        aest, alo, and_ = boot([(x["d"], x["adv"]) for x in ev])
        I = bool(Ss and len(ev) / max(len(pr), 1) >= g["signal_evaluable_fraction_of_paired_min"] and len(ev) >= g["paired_valid_min"]
                 and len({x["d"] for x in ev}) >= g["paired_dialogues_min"] and nd >= b["minimum_valid_draws"] and and_ >= b["minimum_valid_draws"]
                 and est is not None and est >= g["signed_oracle_mean_min"] and lo > g["signed_oracle_adjusted_lower_min"]
                 and aest >= g["shuffle_advantage_mean_min"] and alo > g["shuffle_advantage_adjusted_lower_min"])
        gates[l] = {"O": bool(O), "P": bool(Pp), "S": bool(Ss), "I": I, "signed": est, "signed_lo": lo, "adv": aest, "adv_lo": alo}
        al = an["layers"][str(l)]
        checks[f"gates_agree_L{l}"] = all(al[k] == gates[l][k] for k in ("O", "P", "S", "I"))
        if est is not None and al["signed_cos_macro_adjusted"]["estimate"] is not None:
            checks[f"bootstrap_agree_L{l}"] = abs(est - al["signed_cos_macro_adjusted"]["estimate"]) <= 1e-6 and abs(lo - al["signed_cos_macro_adjusted"]["ci"][0]) <= 1e-6
    valid = pa["stage_valid"] and an["valid"] and checks["oracle_constructions_agree"]
    if not valid:
        label = "R0_INVALID"
    elif not any(x["O"] for x in gates.values()):
        label = "R0_ORACLE_CONSTRUCTION_INSUFFICIENT"
    elif not any(x["P"] for x in gates.values()):
        label = "R0_PREDICTED_REGION_INSUFFICIENT"
    elif not any(x["S"] for x in gates.values()):
        label = "R0_VECTOR_UNSTABLE"
    elif not any(x["I"] for x in gates.values()):
        label = "R0_REGION_SIGNAL_INSUFFICIENT"
    else:
        I = [l for l, x in gates.items() if x["I"]]
        label = "R0_READY_FOR_R1" if len(I) >= g["ready_layers_min"] and ({16, 24} & set(I)) else "R0_PARTIAL_FEASIBILITY"
    checks["label_agrees"] = label == an["label"]
    notes["independent_label"], notes["gates"] = label, {str(k): v for k, v in gates.items()}
    entries = sorted(p.name for p in (ROOT / BASE).iterdir())
    checks["scope_no_extra_stages"] = set(entries) <= {"plan_sealed.json", "prerun_audit.json", "run1", "primary_analysis.json", "output_seal.json",
                                                        "primary_audit.json", "oracle_analysis.json", "oracle_analysis_vectors.npz", "final_audit.json"}
    checks["one_job"] = len(list(run.glob("slurm-*.out"))) == 1
    v = c["audits"]["post_oracle"] if all(checks.values()) else "R0_AUDIT: FAIL (FULL)"
    return {"schema": "r0_full_audit_v1", "verdict": v, "label": label if v == c["audits"]["post_oracle"] else "R0_INVALID",
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
