#!/usr/bin/env python
"""Independent SRD2-G0 auditor (CPU). Stages: ``pre`` -> PASS_TO_SRD2_G0, ``a`` -> SRD2_G0_AUDIT_A,
``primary`` -> SRD2_G0_AUDIT_PRIMARY (no references), ``full`` -> SRD2_G0_AUDIT_FULL + terminal label.

It never imports the SRD2-G0 runner, its mechanics module or the evaluator. Gate windows, raw script
masses, null-adjusted LID support, fallback precedence, structural eligibility, compatibility statistics,
the hash-sorted permutation, dose targets/energies, D2 tangent geometry, top-1 decisions, reference
mapping/target sets, correction/corruption counts, rates, the paired bootstrap, every frozen predicate and
the first-match terminal precedence are recomputed here from raw sealed artifacts. Shared code is limited
to mechanical hashing/JSON helpers, the pinned tokenizer partition and the canonical normalize/align/POI
metric functions.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

CONFIG = "configs/inference_cf/srd2_g0.json"
CONFIG_SHA = "sha256:218b3be997e8fafc102f7ec866b0a82b9925bb180e671f714c651bddee392eb9"
FREEZE = "docs/inference_cf/SRD2_G0_FREEZE.json"
POPULATION = "docs/inference_cf/SRD2_G0_POPULATION.json"
RUN = "results/inference_cf/srd2_g0/run1"
REMOTE = "origin/cs-asr-steer-inf"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
ARCHIVE = "/mnt/data/tungnx/cs-asr-steer/archives/srd2_g0/run1"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
TAG = "SRD2-G0-gate-permutation-v1"
ARMS = ("B1", "B2", "B3")
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
PRIORITY = ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail", "nonfinite_signal")
PROVIDER = PRIORITY[1:]
RUNNER = "experiments/inference_cf_srd2_g0.py"
MECH = "src/csasr/inference_cf/srd2_g0.py"
EVALUATOR = "experiments/inference_cf_srd2_g0_evaluate.py"
TESTS = "tests/test_inference_cf_srd2_g0.py"
SBATCH = "slurm/inference_cf_srd2_g0.sbatch"
FORBIDDEN_OPEN = (".parquet", "transcript", "evaluation_panel", "evaluation_units", "/ctc", "ctc_", "mms", "D-dev-confirm",
                  "D-test", "router-calib", "router_calib", "SRD2_G0_POPULATION.json", "p2r_population", "positions.json",
                  "/roles/", "evaluation.json", "_eval.json", "_eval.npz")
OPS = {"==": operator.eq, "<=": operator.le, ">=": operator.ge, ">": operator.gt}


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def pushed(rel: str) -> bool:
    if git("ls-files", rel) != rel:
        return False
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    if "sha256:" + hashlib.sha256(blob).hexdigest() != file_hash(ROOT / rel):
        return False
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


def cfg_load() -> dict:
    if file_hash(ROOT / CONFIG) != CONFIG_SHA:
        raise SystemExit("config bytes changed")
    return json.loads((ROOT / CONFIG).read_text())


def unbf16(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.int16)
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def fbits(x: float) -> str:
    return np.float64(x).view(np.uint64).item().to_bytes(8, "big").hex()


def quant(xs, p):
    return float(np.quantile(np.asarray(xs, dtype=np.float64), p)) if len(xs) else None


def pred(rule: dict, m: dict) -> bool:
    if "all" in rule:
        return all(pred(x, m) for x in rule["all"])
    if "any" in rule:
        return any(pred(x, m) for x in rule["any"])
    v = m.get(rule["metric"])
    if v is None or isinstance(v, str) or (isinstance(v, float) and not math.isfinite(v)):
        return False
    return bool(OPS[rule["op"]](v, rule["value"]))


def cond(rule: dict, gates: dict, flags: dict) -> bool:
    if "all" in rule:
        return all(cond(x, gates, flags) for x in rule["all"])
    if "any" in rule:
        return any(cond(x, gates, flags) for x in rule["any"])
    if "flag" in rule:
        return bool(flags[rule["flag"]])
    return gates[rule["gate"]] is rule["is"]


def first_label(cfg: dict, gates: dict, flags: dict) -> str:
    for rule in cfg["terminal_rules"]:
        if cond(rule["when"], gates, flags):
            return rule["label"]
    raise RuntimeError("no terminal rule matched")


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


class Tok:
    """Tokenizer-derived facts (partition, suppression, language IDs, byte decoding)."""

    def __init__(self, cfg: dict):
        from transformers import WhisperTokenizer
        from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
        from csasr.inference_cf.core_r2 import tokenizer_partition
        self.tok = WhisperTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        part = tokenizer_partition(self.tok)
        if part["hash"] != cfg["gate"]["partition_hash"]:
            raise SystemExit("partition hash")
        self.E = np.asarray(part["embedded_ids"], dtype=np.int64)
        self.M = np.asarray(part["matrix_ids"], dtype=np.int64)
        gen = json.loads((MODEL / "generation_config.json").read_text())
        self.suppress = list(gen["suppress_tokens"])
        self.begin = list(gen["begin_suppress_tokens"])
        if digest({"suppress": self.suppress, "begin": self.begin}) != cfg["model"]["suppression_hash"]:
            raise SystemExit("suppression hash")
        self.lang = sorted({int(v) for v in gen["lang_to_id"].values()})
        self.en, self.zh = int(gen["lang_to_id"]["<|en|>"]), int(gen["lang_to_id"]["<|zh|>"])
        self.bdec = {v: k for k, v in bytes_to_unicode().items()}
        self.special = set(self.tok.all_special_ids)
        self.eos = EOS

    def argmax(self, z: np.ndarray, t: int) -> int:
        x = np.asarray(z, dtype=np.float64).copy()
        x[self.suppress] = -np.inf
        if t == 0:
            x[self.begin] = -np.inf
        return int(np.argmax(x))

    def utf8_complete(self, ids) -> bool:
        try:
            b = b""
            for i in ids:
                if int(i) in self.special:
                    return False
                b += bytes(self.bdec[c] for c in self.tok.convert_ids_to_tokens(int(i)))
            b.decode("utf-8", errors="strict")
            return True
        except (UnicodeDecodeError, KeyError, ValueError):
            return False

    def masses(self, z: np.ndarray) -> tuple[float, float]:
        x = np.asarray(z, dtype=np.float64)
        m = x.max()
        lse = m + math.log(np.exp(x - m).sum())
        lp = x - lse

        def mass(ids):
            v = lp[ids]
            mm = v.max()
            return float(math.exp(mm + math.log(np.exp(v - mm).sum())))
        return mass(self.E), mass(self.M)


def window(heads_q: np.ndarray, heard: int) -> dict:
    a = np.asarray(heads_q, dtype=np.float64)
    valid = min(1500, (int(heard) + 319) // 320)
    if heard <= 0 or a.ndim != 2 or a.shape[1] < valid:
        raise ValueError("localizer_fail")
    a = a[:, :valid]
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("localizer_fail")
    mean = a.mean(axis=0)
    tot = float(mean.sum())
    if not math.isfinite(tot) or tot <= 0:
        raise ValueError("localizer_fail")
    mean /= tot
    width = min(50, valid)
    masses = np.convolve(mean, np.ones(width, dtype=np.float64), mode="valid")
    start = int(np.argmax(masses))
    h = min(int(heard), 1500 * 320)
    size = min(50 * 320, h)
    s0 = min(start * 320, h - size)
    return {"start_frame": start, "end_frame": start + width, "start_sample": s0, "end_sample": s0 + size}


def support_E(pe, pm, ne, nm) -> float:
    eps = 1e-12
    A = math.log((pe + eps) / (pm + eps)) - math.log((ne + eps) / (nm + eps))
    return max(0.0, math.tanh(A / 2))


def softmax(z):
    x = np.asarray(z, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def heard_mean(h, heard):
    a = np.asarray(h, dtype=np.float64)
    valid = min(1500, (int(heard) + 319) // 320)
    m = a[:, :valid].mean(axis=0)
    return m / m.sum()


def permutation_ref(uid: str, gates: dict) -> list[tuple[int, int, float]]:
    rec = sorted(gates)
    don = sorted(rec, key=lambda t: (hashlib.sha256(f"{TAG}|240924|{uid}|{t}".encode()).hexdigest(), t))
    return [(r, d, gates[d]) for r, d in zip(rec, don)]


def selected() -> tuple[list[dict], dict]:
    P = json.loads((ROOT / POPULATION).read_text())
    return P["selected"], P


# ======================================================================================================
# pre
# ======================================================================================================

def static_scan(rel: str) -> dict:
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docs.add(d)
    code_consts = [c for c in consts if c not in docs]
    return {"imports": sorted(imports), "constants": code_consts}


def cmd_pre(args) -> dict:
    cfg = cfg_load()
    checks, notes = {}, {}
    fr = json.loads((ROOT / FREEZE).read_text())
    checks["freeze_files"] = all(file_hash(ROOT / p) == h for p, h in fr["files"].items())
    sel, P = selected()
    pc = cfg["population"]
    checks["population_hashes"] = (file_hash(ROOT / POPULATION) == pc["file_sha256"]
                                   and digest({k: v for k, v in P.items() if k != "manifest_hash"}) == P["manifest_hash"] == pc["manifest_hash"]
                                   and digest(sel) == pc["selected_hash"] and digest([r["utterance_id"] for r in sel]) == pc["selected_ids_hash"]
                                   and digest(P["roster"]) == pc["roster_hash"])
    checks["selected_400_20x20"] = len(sel) == 400 and set(Counter(r["dialogue_id"] for r in sel).values()) == {20} \
        and len({r["dialogue_id"] for r in sel}) == 20
    r0 = json.loads((ROOT / "docs/inference_cf/R0_PANEL.json").read_text())
    full300 = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("utterance_id", "identity") and isinstance(v, str):
                    full300.add(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(r0)
    checks["FULL300_zero_overlap"] = len(full300) == 300 and not full300 & {r["utterance_id"] for r in sel}
    excluded = {r["utterance_id"] for r in P["roster"] if r["exclusion_reasons"]}
    checks["known_exposure_excluded"] = len(excluded) == 300 and not excluded & {r["utterance_id"] for r in sel}
    panel = json.loads((ROOT / RUN / "runtime_panel.json").read_text())
    own = {"schema": "srd2_g0_runtime_panel_v1",
           "rows": [{"canonical_index": i, "utterance_id": r["utterance_id"], "audio_path": r["audio_path"],
                     "audio_full_sha256": r["audio_full_sha256"]} for i, r in enumerate(sel)],
           "population_selected_ids_hash": digest([r["utterance_id"] for r in sel])}
    own["runtime_hash"] = digest(own)
    checks["runtime_panel_is_exact_four_key_projection"] = panel == own and all(
        set(r) == set(cfg["firewall"]["runtime_input_allowlist"]) for r in panel["rows"])
    m = json.loads((ROOT / RUN / "manifest.json").read_text())
    checks["manifest_self_hash"] = digest({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"]
    checks["manifest_sources_match"] = all(file_hash(ROOT / p) == h for p, h in m["sources"].items())
    checks["manifest_pushed"] = pushed(f"{RUN}/manifest.json") and pushed(f"{RUN}/runtime_panel.json")
    checks["manifest_commit_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", m["git_commit"], "HEAD"], cwd=ROOT).returncode == 0
    changed = git("diff", "--name-only", m["git_commit"], "HEAD").splitlines()
    checks["only_run_dir_changed_since_manifest"] = all(c.startswith(RUN + "/") for c in changed)
    checks["design_freeze_commit_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", m["design_freeze_commit"], "HEAD"], cwd=ROOT).returncode == 0
    checks["historical_pins"] = all(file_hash(ROOT / p) == h for p, h in cfg["source_sha256"].items())
    checks["new_files_pinned"] = all(p in m["sources"] for p in (RUNNER, MECH, EVALUATOR, "experiments/inference_cf_srd2_g0_audit.py", TESTS, SBATCH))
    checks["config_hash_in_manifest"] = m["config_sha256"] == CONFIG_SHA and m["runtime_hash"] == panel["runtime_hash"]
    model_now = {f: file_hash(MODEL / f) for f in cfg["model"]["files"]}
    checks["model_files"] = model_now == cfg["model"]["files"] == m["model"]["files"]
    checks["transformers_forward"] = file_hash(cfg["environment"]["transformers_forward_source"]) == cfg["environment"]["transformers_forward_sha256"]
    T = Tok(cfg)
    checks["partition_suppression_heads"] = (m["partition_hash"] == cfg["gate"]["partition_hash"]
                                             and m["suppression_hash"] == cfg["model"]["suppression_hash"]
                                             and json.loads((MODEL / "generation_config.json").read_text())["alignment_heads"] == cfg["gate"]["alignment_heads"]
                                             and len(T.lang) == 100)
    # ---- static firewall / semantics -------------------------------------------------------------------
    run_s, mech_s = static_scan(RUNNER), static_scan(MECH)
    bad_imports = ("pyarrow", "csasr.evaluation", "csasr.data.normalize", "experiments.inference_cf_p0_r2_evaluate",
                   "experiments.inference_cf_p2r_population", "experiments.inference_cf_srd2_g0_evaluate",
                   "experiments.inference_cf_srd2_g0_audit", "experiments.inference_cf_p2rj")
    checks["runner_imports_no_reference_code"] = not any(i.startswith(b) for i in run_s["imports"] + mech_s["imports"] for b in bad_imports)
    bad_const = (".parquet", "transcript", "/roles/", "evaluation_units", "target_ids", "target_set", "stratum", "ctc", "Y_ref")
    hits = sorted({c for c in run_s["constants"] + mech_s["constants"] for b in bad_const if b in c})
    notes["runner_suspicious_constants"] = hits
    checks["runner_has_no_reference_constants"] = not hits
    msrc = (ROOT / MECH).read_text()
    rsrc = (ROOT / RUNNER).read_text()
    g_src = msrc[msrc.index("def r2_gate"):msrc.index("def float_bits")]
    checks["gate_uses_raw_full_logits"] = ("_processed" not in g_src and "processed_logits" not in g_src
                                           and 'P["conflict_from_logits"](raw_logits' in g_src
                                           and "raw_logits=full_logits[q]" in rsrc and "r2.full_replay(ctx.bundle, enc, list(ctx.prompt), content, attention=True)" in rsrc
                                           and "self.prompt = list(S.CB)" in rsrc and 'cfg["baseline"]["prompt"]' in rsrc)
    checks["d2_is_original_readout_provider"] = "ReadoutDirection(" in rsrc and "DirectionContext(b_cache_pre_step=B.cache" in rsrc
    checks["dose_targets_are_chords"] = ('targets = {"B1": ctx.e_star, "B2": S.target_chord(ctx.e_star, g2), "B3": S.target_chord(ctx.e_star, g3)}' in rsrc
                                         and "self.p2r.solve_scale(r_nat, v, float(target))" in rsrc and 'sol["s"] *' not in rsrc
                                         and "S.CachedSolver(" in rsrc)
    checks["zero_target_bypass"] = 'if target == 0.0:\n            rec["status"] = "zero_target"' in rsrc
    checks["baseline_explicit_argmax"] = "processed_argmax(lg, t, ctx.suppress, ctx.begin)" in rsrc and "topk" not in rsrc
    checks["advance_with_baseline_token_only"] = "S.feed_tokens(content, t, ctx.prompt)" in rsrc
    gpu_fns = ("load_run", "Ctx", "load_model", "encode", "cached_baseline", "capture_utterance", "cmd_capture", "Engine",
               "apparatus_old30", "pulse_utterance", "cmd_pulses")
    tree = ast.parse(rsrc)
    seg = {n.name: ast.get_source_segment(rsrc, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    checks["gpu_phases_never_read_population_or_dialogues"] = all(
        f in seg and "POPULATION" not in seg[f] and "dialogue_of" not in seg[f] and "dialogue_id" not in seg[f] for f in gpu_fns)
    ev = (ROOT / EVALUATOR).read_text()
    checks["evaluator_barrier"] = ('"SRD2_G0_AUDIT_PRIMARY: PASS"' in ev and 'ds.field("utterance_id").isin(ids)' in ev
                                   and ev.index("SRD2_G0_AUDIT_PRIMARY: PASS required") < ev.index("refs, access = load_references(ids)"))
    me = static_scan("experiments/inference_cf_srd2_g0_audit.py")
    checks["auditor_independent_imports"] = not any(i in ("experiments.inference_cf_srd2_g0", "experiments.inference_cf_srd2_g0_evaluate",
                                                          "csasr.inference_cf.srd2_g0") for i in me["imports"])
    sb = (ROOT / SBATCH).read_text()
    checks["slurm_contract"] = ("#SBATCH --partition=mig" in sb and "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1" in sb
                                and "#SBATCH --time=03:00:00" in sb and "capture)" in sb and "pulses)" in sb
                                and "sbatch " not in sb.split("set -euo pipefail", 1)[1] and "--cpus-per-task=4" in sb and "--mem=48G" in sb)
    # ---- statistics / predicate contract -----------------------------------------------------------------
    checks["terminal_precedence"] = [r["label"] for r in cfg["terminal_rules"]] == cfg["terminal_precedence"]
    checks["family_quantiles"] = cfg["statistics"]["family_quantiles"] == [.00625, .99375] and cfg["statistics"]["bootstrap_replicates"] == 10000
    # ---- resources -----------------------------------------------------------------------------------
    c = cfg["compute"]
    checks["pre_job_forecast"] = c["job_B_cost_seconds_per_query"] * c["forecast_conservative_queries"] + c["job_B_fixed_seconds"] <= 9000 \
        and c["job_A_conservative_seconds"] <= 10800
    free = {p: shutil.disk_usage(p).free for p in (str(ROOT), "/mnt/data/tungnx")}
    notes["free_bytes"] = free
    checks["storage_ge_50GB"] = all(v >= 50e9 for v in free.values())
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h"], capture_output=True, text=True)
    notes["squeue"] = q.stdout.strip()
    checks["no_conflicting_jobs"] = q.returncode == 0 and not q.stdout.strip()
    checks["no_prior_outputs"] = not (ROOT / RUN / "gate").exists() and not (ROOT / RUN / "pulses").exists()
    # ---- tests -------------------------------------------------------------------------------------
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}", "CUDA_VISIBLE_DEVICES": ""}
    suites = [TESTS, "tests/test_srd2_g0_freeze.py", "tests/test_inference_cf_p2dir_directions.py",
              "tests/test_inference_cf_p0_r2.py"]
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider", *suites],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    tail = res.stdout.strip().splitlines()[-3:]
    notes["pytest_tail"] = tail
    checks["tests_pass"] = res.returncode == 0
    verdict = "PASS_TO_SRD2_G0" if all(checks.values()) else "BLOCK_BEFORE_SRD2_G0"
    return {"schema": "srd2_g0_audit_pre_v1", "verdict": verdict, "manifest_hash": m["manifest_hash"], "checks": checks,
            "notes": notes, "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


# ======================================================================================================
# common runtime checks
# ======================================================================================================

def opened_paths_check(paths: list[str], panel_rows: list[dict], m: dict, apparatus: bool) -> dict:
    audio = {os.path.abspath(r["audio_path"]) for r in panel_rows}
    repo_allowed = {os.path.abspath(ROOT / p) for p in m["sources"]} | {os.path.abspath(ROOT / RUN / "manifest.json"),
                                                                      os.path.abspath(ROOT / "configs/model/whisper_large_v3.yaml")}
    app_prefixes = tuple(os.path.abspath(ROOT / p) for p in ("results/inference_cf/p2dir/exp1_run1", "results/inference_cf/p2_A_r1_L16",
                                                              "results/inference_cf/p0_r2/inference_panel.json"))
    cats = defaultdict(list)
    for p in paths:
        low = p.lower()
        if any(f.lower() in low for f in FORBIDDEN_OPEN) and not p.startswith(os.path.abspath(ROOT / RUN)):
            cats["forbidden"].append(p)
        elif p in audio:
            cats["audio"].append(p)
        elif p.startswith(str(MODEL)):
            cats["model"].append(p)
        elif p in repo_allowed:
            cats["pinned"].append(p)
        elif p.startswith(os.path.abspath(ROOT / RUN)):
            cats["run"].append(p)
        elif apparatus and p.startswith(app_prefixes):
            cats["apparatus"].append(p)
        elif p.startswith(str(ROOT)):
            cats["repo_other"].append(p)
        elif p.startswith("/mnt/"):
            cats["mnt_other"].append(p)
        else:
            cats["other"].append(p)
    ok = not cats["forbidden"] and not cats["mnt_other"] and not [p for p in cats["repo_other"] if not p.endswith(".py")]
    return {"ok": ok, "counts": {k: len(v) for k, v in cats.items()}, "forbidden": cats["forbidden"],
            "mnt_other": cats["mnt_other"], "repo_other": cats["repo_other"], "other": cats["other"]}


def base_checks(stage: str) -> tuple[dict, dict, list, dict]:
    cfg = cfg_load()
    m = json.loads((ROOT / RUN / "manifest.json").read_text())
    panel = json.loads((ROOT / RUN / "runtime_panel.json").read_text())
    pre = json.loads((ROOT / RUN / "audit_PRE.json").read_text())
    checks = {"manifest_self_hash": digest({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"],
              "pre_audit_pass": pre["verdict"] == "PASS_TO_SRD2_G0" and pre["manifest_hash"] == m["manifest_hash"] and pushed(f"{RUN}/audit_PRE.json"),
              "sources_unchanged": all(file_hash(ROOT / p) == h for p, h in m["sources"].items()),
              "panel_hash": panel["runtime_hash"] == m["runtime_hash"]}
    return cfg, m, panel["rows"], checks


def load_sealed(seal_rel: str, kind: str, rows: list[dict]) -> tuple[dict, dict, dict]:
    seal = json.loads((ROOT / RUN / seal_rel).read_text())
    checks = {"seal_pushed": pushed(f"{RUN}/{seal_rel}"), "seal_status": seal.get("status") == "SEALED",
              "seal_self_hash": digest({k: v for k, v in seal.items() if k != "seal_hash"}) == seal["seal_hash"]}
    bad = [rel for rel, h in seal["files"].items() if file_hash(ROOT / RUN / rel) != h]
    checks["seal_files_match"] = not bad
    arch = seal["archive"]["content_addressed"]
    checks["archive_copies"] = all(Path(p).exists() and file_hash(p) == seal["files"][rel] for rel, p in arch.items())
    checks["archive_complete"] = all(rel in arch for rel in seal["files"] if rel.endswith(".npz"))
    out = {}
    for r in rows:
        rel = f"{kind}/{r['utterance_id']}.json"
        out[r["utterance_id"]] = json.loads((ROOT / RUN / rel).read_text()) if rel in seal["files"] else None
    checks["all_rows_sealed"] = all(v is not None for v in out.values())
    return seal, out, checks


# ======================================================================================================
# audit A
# ======================================================================================================

def recompute_gate_row(T: Tok, g: dict, arr: dict, null: dict) -> tuple[list[dict], dict]:
    EOS = T.eos
    content, term = g["baseline"]["content_ids"], g["baseline"]["terminated"]
    inv = list(range(len(content) + 1)) if term == "eos" else list(range(len(content)))
    errs = Counter()
    if g["inventory"] != inv or [x["t"] for x in g["queries"]] != inv:
        errs["inventory"] += 1
    cl, fl = unbf16(arr["cached_logits"]), unbf16(arr["full_logits"])
    cs, fs = unbf16(arr["cached_site"]), unbf16(arr["full_site"])
    heads = unbf16(arr["heads"])
    lid = {(int(k[0]), int(k[1])): p for k, p in zip(arr["lid_keys"], arr["lid_probs"])}
    li = {v: i for i, v in enumerate(T.lang)}
    heard = int(g["audio"]["samples"])
    out = []
    flags = {"inventory_ok": g["inventory"] == inv and [x["t"] for x in g["queries"]] == inv,
             "query_index_ok": [x["query"] for x in g["queries"]] == [3 + t for t in g["inventory"]]}
    for i, (t, x) in enumerate(zip(inv, g["queries"])):
        exp = content[t] if t < len(content) else EOS
        if x["query"] != 3 + t or x["expected_action"] != exp or T.argmax(cl[i], t) != exp or x["cached_argmax"] != exp:
            errs["query_identity_or_argmax"] += 1
        prefix = content[:t]
        utf8 = T.utf8_complete(prefix)
        if utf8 != x["utf8_complete"]:
            errs["utf8"] += 1
        fin = bool(np.isfinite(cs[i]).all() and np.isfinite(cl[i]).all())
        elig = t >= 1 and utf8 and all(int(p) < EOS for p in prefix) and fin
        if elig != x["structural"]["eligible"]:
            errs["structural"] += 1
        reasons, E, R, win = [], None, None, None
        if not utf8:
            reasons.append("mid_character")
        try:
            win = window(heads[i], heard)
        except ValueError:
            reasons.append("localizer_fail")
        pe = pm = None
        if np.isfinite(fl[i]).all():
            pe, pm = T.masses(fl[i])
            R = max(0.0, pm - pe)
        else:
            reasons.append("nonfinite_signal")
        if not reasons:
            key = (win["start_sample"], win["end_sample"])
            if key not in lid:
                errs["lid_missing"] += 1
                reasons.append("local_support_fail")
            else:
                p = lid[key]
                E = support_E(float(p[li[T.en]]), float(p[li[T.zh]]), null["EN"], null["ZH"])
        fb = next((r for r in PRIORITY if r in reasons), None)
        gv = 0.0 if fb else E * R
        rg = x["gate"]
        if win is not None and rg["window"] is not None:
            if any(win[k] != rg["window"][k] for k in win):
                errs["window"] += 1
        elif (win is None) != (rg["window"] is None):
            errs["window_presence"] += 1
        if fb != rg["fallback_reason"]:
            errs["fallback"] += 1
        if pe is not None and rg["baseline"] is not None and not (abs(pe - rg["baseline"]["P_E"]) <= 1e-6 and abs(pm - rg["baseline"]["P_M"]) <= 1e-6):
            errs["script_mass"] += 1
        if E is not None and (rg["E"] is None or abs(E - rg["E"]) > 1e-9):
            errs["E"] += 1
        if abs(gv - x["g"]) > 1e-6 or fbits(x["g"]) != x["g_bits"]:
            errs["g"] += 1
        rec = {"t": t, "eligible": elig, "g": x["g"], "fallback": fb, "utf8": utf8}
        if elig:
            tv = 0.5 * np.abs(softmax(fl[i]) - softmax(cl[i])).sum()
            pec, pmc = T.masses(cl[i])
            a, b = fs[i].astype(np.float64), cs[i].astype(np.float64)
            nb = np.linalg.norm(b)
            rel = float(np.linalg.norm(a - b) / nb) if nb > 0 else None
            cosv = float(a @ b / (np.linalg.norm(a) * nb)) if nb > 0 and np.linalg.norm(a) > 0 else None
            agree = T.argmax(fl[i], t) == T.argmax(cl[i], t)
            c = x["compat"]
            if not (close(tv, c["TV"]) and abs(abs(pe - pec) - c["PE_abs_diff"]) <= 1e-6 and abs(abs(pm - pmc) - c["PM_abs_diff"]) <= 1e-6
                    and agree == c["argmax_agree"] and close(rel, c["site_rel_L2"]) and close(cosv, c["site_cos"])):
                errs["compat"] += 1
            rec.update(TV=float(tv), dPE=abs(pe - pec), dPM=abs(pm - pmc), agree=agree, rel=rel, cos=cosv)
        out.append(rec)
    probes = []
    for j, p in enumerate(g["probes"]):
        i = inv.index(p["t"])
        tv = 0.5 * np.abs(softmax(unbf16(arr["probe_logits"][j])) - softmax(fl[i])).sum()
        l1 = float(np.abs(heard_mean(heads[i], heard) - heard_mean(unbf16(arr["probe_heads"][j]), heard)).sum())
        if not (close(tv, p["TV"]) and close(l1, p["attention_L1"])):
            errs["probe"] += 1
        probes.append({"TV": float(tv), "L1": l1, "future": p["future_mass_max"]})
    st = [x["t"] for x in g["queries"] if x["structural"]["eligible"]]
    if st:
        want = sorted({st[0], st[(len(st) - 1) // 2], st[-1]})
        if [p["t"] for p in g["probes"]] != want:
            errs["probe_selection"] += 1
    return out, {"errors": dict(errs), "probes": probes, **flags}


def job_a_metrics_independent(cfg: dict, rows: list, recs: dict, row_flags: dict, probes: list, null: dict, dlg: dict,
                              errs: Counter) -> dict:
    """Job-A coverage/compatibility metrics from the independently recomputed rows."""
    ok = {u: v for u, v in recs.items()}
    dialogues = sorted(set(dlg.values()))
    struct = [(u, x) for u, (g, rr) in ok.items() for x in rr if x["eligible"]]
    nz = [(u, x) for u, x in struct if x["g"] > 0]
    cp = [x for u, (g, rr) in ok.items() for x in rr if x["utf8"]]
    fp = [x for x in cp if x["fallback"] not in PROVIDER]
    tv = [x["TV"] for _, x in struct]
    rel = [x["rel"] for _, x in struct]
    met = {"rows_complete": len(ok), "dialogues_complete": sum(1 for d in dialogues if all(u in ok for u in dlg if dlg[u] == d)),
           "inventory_complete_fraction": sum(1 for u in ok if row_flags[u]["inventory_ok"]) / len(rows),
           "structural_queries": len(struct), "structural_dialogues": len({dlg[u] for u, _ in struct}),
           "nonzero_gates": len(nz), "nonzero_gate_dialogues": len({dlg[u] for u, _ in nz}),
           "finite_provider_fraction": len(fp) / len(cp) if cp else None,
           "identity_fraction": sum(1 for u, (g, _) in ok.items() if g["full_replay"]["input_ids_sha256"] == digest(CB + g["baseline"]["content_ids"])
                                    and g["baseline"]["lineage_ok"] and g["full_replay"]["shared_encoder"] is True
                                    and row_flags[u]["query_index_ok"]) / len(rows),
           "cached_baseline_identity_bitwise": bool(ok) and not errs.get("cached_replay_not_bitwise"),
           "raw_TV_median": quant(tv, .5), "raw_TV_p99": quant(tv, .99), "raw_TV_max": max(tv) if tv else None,
           "PE_abs_diff_p99": quant([x["dPE"] for _, x in struct], .99), "PM_abs_diff_p99": quant([x["dPM"] for _, x in struct], .99),
           "processed_argmax_agreement": sum(x["agree"] for _, x in struct) / len(struct) if struct else None,
           "site_rel_L2_p99": quant(rel, .99) if None not in rel else None, "site_rel_L2_max": max(rel) if rel and None not in rel else None,
           "site_cosine_min": min(x["cos"] for _, x in struct) if struct and all(x["cos"] is not None for _, x in struct) else None,
           "causal_probe_TV_max": max(p["TV"] for p in probes) if probes else None,
           "causal_probe_attention_L1_max": max(p["L1"] for p in probes) if probes else None,
           "future_self_attention_mass_max": max([g["full_replay"]["future_mass_max"] for g, _ in ok.values()] + [p["future"] for p in probes]) if ok else None,
           "null_EN_abs_error": abs(null["EN"] - cfg["gate"]["null_EN_historical"]),
           "null_ZH_abs_error": abs(null["ZH"] - cfg["gate"]["null_ZH_historical"])}
    return met


def cmd_a(args) -> dict:
    cfg, m, rows, checks = base_checks("a")
    seal, gate, sc = load_sealed("gate_seal.json", "gate", rows)
    checks.update(sc)
    rt = json.loads((ROOT / RUN / "capture_runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and rt["manifest_hash"] == m["manifest_hash"]
    checks["weights_unchanged"] = rt["weights_unchanged"] is True and rt["weights_digest_start"] == rt["weights_digest_end"]
    checks["no_parameter_grads"] = rt["parameters_with_grad"] == 0 and rt["parameters_requiring_grad"] == 0
    op = opened_paths_check(rt["opened_paths"], rows, m, apparatus=False)
    checks["opened_paths_allowlist"] = op["ok"]
    null = {"EN": rt["null"]["EN"], "ZH": rt["null"]["ZH"]}
    checks["null_pins"] = abs(null["EN"] - cfg["gate"]["null_EN_historical"]) <= 1e-6 and abs(null["ZH"] - cfg["gate"]["null_ZH_historical"]) <= 1e-6
    T = Tok(cfg)
    errs = Counter()
    recs, probes, dlg = {}, [], {r["utterance_id"]: r["dialogue_id"] for r in selected()[0]}
    row_flags = {}
    for r in rows:
        uid = r["utterance_id"]
        g = gate[uid]
        if g is None or g["status"] != "ok":
            errs["row_not_ok"] += 1
            continue
        if g["audio"]["full_sha256"] != r["audio_full_sha256"]:
            errs["audio_hash"] += 1
        with np.load(ROOT / RUN / "gate" / g["arrays"]["file"]) as z:
            arr = {k: z[k] for k in z.files}
        rr, info = recompute_gate_row(T, g, arr, null)
        recs[uid] = (g, rr)
        row_flags[uid] = info
        probes += info["probes"]
        errs.update(info["errors"])
        if not (g["baseline"]["replay_bitwise"] and g["baseline"]["lineage_ok"]):
            errs["cached_replay_not_bitwise"] += 1
    checks["independent_gate_recomputation"] = not errs
    met = job_a_metrics_independent(cfg, rows, recs, row_flags, probes, null, dlg, errs)
    agree = {k: close(v, seal["metrics"].get(k), 1e-9) if not isinstance(v, bool) else v == seal["metrics"].get(k) for k, v in met.items()}
    checks["metrics_agree_with_seal"] = all(agree.values())
    preds = {"job_A_coverage": pred(cfg["gate_predicates"]["job_A_coverage"], met), "compatibility": pred(cfg["gate_predicates"]["compatibility"], met)}
    checks["predicates_agree_with_seal"] = preds == seal["predicates"]
    verdict = "SRD2_G0_AUDIT_A: PASS" if all(checks.values()) else "SRD2_G0_AUDIT_A: BLOCK"
    return {"schema": "srd2_g0_audit_A_v1", "verdict": verdict, "checks": checks, "errors": dict(errs),
            "metrics": met, "metric_agreement": agree, "predicates": preds, "opened_paths": op,
            "gate_seal_sha256": file_hash(ROOT / RUN / "gate_seal.json"), "manifest_hash": m["manifest_hash"],
            "references_used": False, "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


# ======================================================================================================
# audit PRIMARY (no references)
# ======================================================================================================

def primary_recompute(cfg: dict, T: Tok, rows: list[dict], gate: dict, pulses: dict, perm: dict, run: Path | None = None) -> dict:
    run = ROOT / RUN if run is None else Path(run)
    e = float(cfg["dose"]["e_star"])
    dose = cfg["dose"]
    errs = Counter()
    own_perm = {}
    for r in rows:
        uid = r["utterance_id"]
        gates = {x["t"]: float(x["g"]) for x in gate[uid]["queries"] if x["structural"]["eligible"]}
        own_perm[uid] = {rcp: (d, gv) for rcp, d, gv in permutation_ref(uid, gates)}
    pu = {u["utterance_id"]: u for u in perm["utterances"]}
    for uid, mp in own_perm.items():
        pairs = pu[uid]["pairs"]
        if [(p["recipient_t"], p["donor_t"], fbits(p["g_shuffled"])) for p in pairs] != [(rcp, d, fbits(gv)) for rcp, (d, gv) in sorted(mp.items())]:
            errs["permutation"] += 1
    unin = sum(len(mp) for uid, mp in own_perm.items() if mp and all(fbits(gv) == fbits(dict((x["t"], x["g"]) for x in gate[uid]["queries"])[rcp])
                                                                      for rcp, (d, gv) in mp.items()))
    nstruct = sum(len(mp) for mp in own_perm.values())
    if unin != perm["uninformative_queries"] or nstruct != perm["structural_queries"]:
        errs["permutation_flags"] += 1
    sq, ch, cp, b1, planned, real, lost = [], [], [], [], {"B2": [], "B3": []}, {"B2": [], "B3": []}, {"B2": [], "B3": []}
    restore_ok = clean_ok = scratch_ok = integ_ok = True
    executed = autograd = 0
    status = Counter()
    for r in rows:
        uid = r["utterance_id"]
        g, p = gate[uid], pulses[uid]
        if p is None or p["status"] != "ok":
            errs["pulse_row_not_ok"] += 1
            continue
        if p["inventory"] != g["inventory"] or [x["t"] for x in p["queries"]] != g["inventory"]:
            errs["inventory"] += 1
        with np.load(run / "pulses" / p["arrays"]["file"]) as z:
            arr = {k: z[k] for k in z.files}
        with np.load(run / "pulses" / p["logits"]["file"]) as z:
            lg = {k: z[k] for k in z.files}
        with np.load(run / "gate" / g["arrays"]["file"]) as z:
            cl = unbf16(z["cached_logits"])
        if not (np.array_equal(arr["exec_keys"], lg["exec_keys"])):
            errs["exec_keys"] += 1
        ex_logits = unbf16(lg["exec_logits"])
        ex_ffn = unbf16(arr["exec_ffn"]).astype(np.float64)
        si = -1
        k = 0
        for i, x in enumerate(p["queries"]):
            gx = g["queries"][i]
            t = x["t"]
            b0 = T.argmax(cl[i], t)
            clean_ok &= bool(x["clean"]["logits_bitwise_vs_A"] and x["clean"]["site_bitwise_vs_A"])
            if x["arms"]["B0"]["top1"] != b0 or b0 != gx["expected_action"]:
                errs["B0_top1"] += 1
            if x["structural"] != gx["structural"]["eligible"]:
                errs["structural"] += 1
            if not x["structural"]:
                if any(x["arms"][a]["executed"] or x["arms"][a]["top1"] != b0 for a in ARMS):
                    errs["ineligible_not_noop"] += 1
                continue
            si += 1
            d2 = x["d2"]
            scratch_ok &= bool(d2["scratch_logits_bitwise"] and d2["scratch_site_bitwise"])
            autograd += int(d2["counters"].get("autograd_calls", 0))
            rnat = unbf16(arr["native_r"][si]).astype(np.float64)
            dvec = arr["d2_direction"][si]
            if d2["status"] == "ok":
                gr = arr["d2_gradient"][si].astype(np.float64)
                hh = rnat / np.linalg.norm(rnat)
                gp = gr - hh * (hh @ gr)
                dd = (gp / (np.linalg.norm(gp) + 1e-12)).astype(np.float32)
                if not (np.abs(dd.astype(np.float64) - dvec.astype(np.float64)).max() <= 1e-6 and abs(np.linalg.norm(dvec.astype(np.float64)) - 1) <= 2e-6
                        and abs(hh @ dvec.astype(np.float64)) <= 1e-6):
                    errs["d2_geometry"] += 1
                if d2["direction_sha256"] is None:
                    errs["d2_hash"] += 1
            elif not np.isnan(dvec).all():
                errs["invalid_direction_stored"] += 1
            gv = float(gx["g"])
            d_t, gs = own_perm[uid][t]
            tg = {"B1": e, "B2": 0.0 if gv == 0 else e * gv, "B3": 0.0 if gs == 0 else e * gs}
            for a in ARMS:
                ar = x["arms"][a]
                status[(a, ar["status"])] += 1
                if fbits(ar["target"]) != fbits(tg[a]):
                    errs["target"] += 1
                if tg[a] == 0 and ar["status"] != "zero_target":
                    errs["zero_target_status"] += 1
                if d2["status"] != "ok" and tg[a] > 0 and ar["status"] != "invalid_direction":
                    errs["invalid_status"] += 1
                if a in planned:
                    planned[a].append(tg[a])
                if ar["executed"]:
                    executed += 1
                    key = lg["exec_keys"][k]
                    if int(key[0]) != i or int(key[1]) != ARMS.index(a) or ar["logits"] != f"exec:{k}":
                        errs["exec_index"] += 1
                    top = T.argmax(ex_logits[k], t)
                    cons = float(np.linalg.norm(ex_ffn[k] - rnat))
                    gd = ar["guards"]
                    if top != ar["top1"]:
                        errs["exec_top1"] += 1
                    if not close(cons, ar["consumed_chord_actual"], 1e-9) or not close(cons, gd["consumed_chord"], 1e-9):
                        errs["consumed_chord"] += 1
                    t2 = ar["target"] ** 2
                    sq += [abs(gd["proposed_chord"] ** 2 / t2 - 1), abs(cons * cons / t2 - 1)]
                    ch += [abs(gd["proposed_chord"] / ar["target"] - 1), abs(cons / ar["target"] - 1)]
                    cp.append(abs(cons / gd["proposed_chord"] - 1))
                    integ_ok &= bool(ar["integrity_ok"] and ar["solver_calls"] == 1 and ar["ffn_equals_preview_bitwise"]
                                     and ar["solver"]["rel_sq_err"] <= dose["relative_squared_energy_error_max"])
                    if a in real:
                        real[a].append(cons * cons)
                    k += 1
                else:
                    if ar["top1"] != b0:
                        errs["noop_top1"] += 1
                    if a in lost and tg[a] > 0:
                        lost[a].append(tg[a] ** 2)
                if a == "B1":
                    b1.append(bool(ar["executed"]))
            if x["restore_bitwise"] is not None:
                restore_ok &= bool(x["restore_bitwise"])
            elif any(x["arms"][a]["executed"] for a in ARMS):
                errs["restore_missing"] += 1
        if k != len(lg["exec_keys"]):
            errs["exec_count"] += 1
    tot = {a: math.fsum(sorted(v * v for v in planned[a])) for a in planned}
    rs = {a: math.fsum(sorted(real[a])) for a in real}
    energy = {"executed_relative_squared_energy_error_max": max(sq) if sq else None,
              "executed_relative_chord_error_max": max(ch) if ch else None,
              "consumed_proposed_relative_error_max": max(cp) if cp else None,
              "solver_hook_identity_pass": bool(executed) and integ_ok,
              "zero_no_edit_restore_bitwise": restore_ok and clean_ok and scratch_ok,
              "B1_matched_fraction": sum(b1) / len(b1) if b1 else None}
    dosec = {"planned_B2_B3_multisets_equal": sorted(fbits(v) for v in planned["B2"]) == sorted(fbits(v) for v in planned["B3"]) and tot["B2"] == tot["B3"],
             "energy_ratio_B2_B3": rs["B2"] / rs["B3"] if rs["B3"] > 0 else None,
             "lost_planned_energy_B2": math.fsum(sorted(lost["B2"])) / tot["B2"] if tot["B2"] > 0 else None,
             "lost_planned_energy_B3": math.fsum(sorted(lost["B3"])) / tot["B3"] if tot["B3"] > 0 else None}
    return {"errors": dict(errs), "energy_execution": energy, "dose_comparison": dosec, "executed_cells": executed,
            "autograd_calls": autograd, "uninformative_queries": unin, "structural_queries": nstruct,
            "planned_sq": tot, "realized_sq": rs, "arm_status": {f"{a}|{s}": n for (a, s), n in sorted(status.items())}}


def cmd_primary(args) -> dict:
    cfg, m, rows, checks = base_checks("primary")
    gseal, gate, gc = load_sealed("gate_seal.json", "gate", rows)
    pseal, pulses, pc = load_sealed("pulse_seal.json", "pulses", rows)
    checks.update({f"gate_{k}": v for k, v in gc.items()})
    checks.update({f"pulse_{k}": v for k, v in pc.items()})
    a = json.loads((ROOT / RUN / "audit_A.json").read_text())
    checks["audit_A_pass_pushed"] = a["verdict"] == "SRD2_G0_AUDIT_A: PASS" and pushed(f"{RUN}/audit_A.json")
    auth = json.loads((ROOT / RUN / "authorization_B.json").read_text())
    checks["authorization_minimal_and_true"] = list(auth) == cfg["firewall"]["authorization_B_fields"] and auth["authorized"] is True \
        and auth["job_A_seal_hash"] == file_hash(ROOT / RUN / "gate_seal.json") and auth["permutation_hash"] == file_hash(ROOT / RUN / "permutation_seal.json")
    perm = json.loads((ROOT / RUN / "permutation.json").read_text())
    checks["permutation_sealed"] = json.loads((ROOT / RUN / "permutation_seal.json").read_text())["permutation_sha256"] == file_hash(ROOT / RUN / "permutation.json") \
        and pseal["permutation_sha256"] == file_hash(ROOT / RUN / "permutation.json")
    rt = json.loads((ROOT / RUN / "pulse_runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and rt["manifest_hash"] == m["manifest_hash"]
    checks["weights_unchanged"] = rt["weights_unchanged"] is True
    checks["no_parameter_grads"] = rt["parameters_with_grad"] == 0 and rt["parameters_requiring_grad"] == 0
    op = opened_paths_check(rt["opened_paths"], rows, m, apparatus=True)
    checks["opened_paths_allowlist"] = op["ok"]
    app = json.loads((ROOT / RUN / "apparatus_old30.json").read_text())
    hq = cfg["historical_apparatus"]["queries"]
    checks["apparatus_old30"] = (app["ok"] is True and app["n"] == 30 and [(x["utterance_id"], x["t"]) for x in app["positions"]]
                                 == sorted([(x["utterance_id"], x["t"]) for x in hq], key=lambda y: (y[0], y[1]))
                                 and all(x["clean_logits_bitwise"] and x["clean_site_bitwise"] and x["pulse_logits_bitwise"] and x["zero_dose_bitwise"]
                                         and x["d2_max_abs_error"] == 0.0 and x["J_abs_diff"] <= 1e-6 and x["restore_bitwise"] for x in app["positions"])
                                 and not ({x["utterance_id"] for x in hq} & {r["utterance_id"] for r in rows}))
    T = Tok(cfg)
    rec = primary_recompute(cfg, T, rows, gate, pulses, perm)
    checks["independent_recomputation"] = not rec["errors"]
    checks["autograd_count"] = rec["autograd_calls"] == rt["d2_autograd_calls"]
    energy_ok = pred(cfg["gate_predicates"]["energy_execution"], rec["energy_execution"])
    verdict = "SRD2_G0_AUDIT_PRIMARY: PASS" if all(checks.values()) else "SRD2_G0_AUDIT_PRIMARY: BLOCK"
    return {"schema": "srd2_g0_audit_PRIMARY_v1", "verdict": verdict, "checks": checks, "recompute": rec,
            "energy_execution_predicate": energy_ok,
            "dose_comparison_predicate": pred(cfg["gate_predicates"]["dose_comparison"], rec["dose_comparison"]),
            "opened_paths": op, "pulse_seal_sha256": file_hash(ROOT / RUN / "pulse_seal.json"), "manifest_hash": m["manifest_hash"],
            "references_used": False, "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


# ======================================================================================================
# audit FULL (references; independent mapping)
# ======================================================================================================

def own_token_positions(tok, content: list[int], hyp: str) -> list:
    from csasr.data.normalize import normalize_text, segment_units
    final = normalize_text(hyp)
    units = segment_units(final)
    lens, prev = [], 0
    for k in range(1, len(content) + 1):
        pre = normalize_text(tok.decode(content[:k], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        if not final.startswith(pre) or len(pre) < prev:
            return [None] * len(units)
        prev = len(pre)
        lens.append(prev)
    if (lens[-1] if lens else 0) != len(final):
        return [None] * len(units)
    return [next((i for i, L in enumerate(lens) if L > u.char_start), None) for u in units]


def own_target_set(tok, prefix: list[int], word: str, raw) -> list[int]:
    ptext = tok.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out = set()
    for f in sorted({word, word.capitalize(), word.upper()} | ({raw} if raw else set())):
        for sep in ("", " "):
            ids = tok.encode(ptext + sep + f, add_special_tokens=False)
            if ids[:len(prefix)] != list(prefix) or len(ids) <= len(prefix):
                continue
            tid = ids[len(prefix)]
            s = tok.decode([tid], clean_up_tokenization_spaces=False).strip().lower()
            if s and word.lower().startswith(s) and len(s) >= min(2, len(word)):
                out.add(int(tid))
    return sorted(out)


def own_map(tok, ref: str, content: list[int], term: str, dur: float, eos: int = EOS) -> tuple[dict, int, int]:
    from csasr.data.normalize import normalize_text, segment_units
    from csasr.evaluation.mer import DEL, MATCH, SUB, align_tokens
    from csasr.evaluation.pier import LANGUAGE_CONFUSION, evaluate_pois
    hyp = tok.decode(content, skip_special_tokens=True)
    pos = own_token_positions(tok, content, hyp)
    eos_slot = len(content) if term == "eos" else None
    ru = segment_units(normalize_text(ref))
    hu = segment_units(normalize_text(hyp))
    ops = align_tokens([u.surface for u in ru], [u.surface for u in hu])
    pois = {p.poi_index: p for p in evaluate_pois(ref, hyp)}
    raw_words = re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", ref)
    latin = [i for i, u in enumerate(ru) if u.kind == "latin"]
    raw = {}
    if len(raw_words) == len(latin) and all(normalize_text(w) == ru[i].surface for i, w in zip(latin, raw_words)):
        raw = dict(zip(latin, raw_words))
    total = mapped = 0
    units = []
    for oi, op in enumerate(ops):
        if op.ref_idx is None:
            continue
        u = ru[op.ref_idx]
        p_at = None
        if op.op in (MATCH, SUB) and op.hyp_idx is not None:
            p_at = pos[op.hyp_idx]
        elif op.op == DEL:
            nxt = next((o.hyp_idx for o in ops[oi + 1:] if o.ref_idx is not None and o.hyp_idx is not None), None)
            p_at = pos[nxt] if nxt is not None else eos_slot
        if u.kind == "latin":
            total += 1
            mapped += int(p_at is not None)
        cat = None
        if u.kind == "latin" and op.ref_idx in pois:
            pp = pois[op.ref_idx]
            cat = ("EN-correct" if pp.correct else "EN-confusion" if pp.category in LANGUAGE_CONFUSION else
                   "EN-deletion-slot" if pp.category == "deletion" else "EN-same-language-sub" if pp.category == "same_language_substitution"
                   else None)
            if cat is None:
                continue
        elif u.kind == "han" and op.op == MATCH:
            cat = "ZH-correct"
        else:
            continue
        units.append({"idx": op.ref_idx, "surface": u.surface, "cat": cat, "lang": "EN" if u.kind == "latin" else "ZH", "pos": p_at})
    if dur > 30.0:
        return {}, total, 0
    by = defaultdict(set)
    for u in units:
        if u["pos"] is not None:
            by[u["pos"]].add(u["cat"])
    en_count = Counter(u["pos"] for u in units if u["lang"] == "EN" and u["pos"] is not None)
    q = {}
    for u in units:
        t = u["pos"]
        if t is None or len(by[t]) > 1 or u["cat"] not in STRATA:
            continue
        if u["lang"] == "EN" and en_count[t] > 1:
            continue
        if u["cat"] == "ZH-correct":
            if t < len(content) and t not in q:
                q[t] = ("ZH-correct", [int(content[t])])
            continue
        Y = own_target_set(tok, content[:t], u["surface"], raw.get(u["idx"]))
        if not Y:
            continue
        b0 = int(content[t]) if t < len(content) else eos
        if (u["cat"] == "EN-confusion") == (b0 in Y):
            continue
        q[t] = (u["cat"], Y)
    return q, total, mapped


def full_core(cfg: dict, T, sel: list[dict], gate: dict, pulses: dict, refs: dict, rec: dict, a_metrics: dict,
              fc_metrics: dict) -> dict:
    """Independent reference-side recomputation: mapping, events, tallies, every frozen metric and gate."""
    ids = [r["utterance_id"] for r in sel]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in sel}
    dur = {r["utterance_id"]: float(r["source_duration_sec"]) for r in sel}
    dialogues = sorted(set(dlg.values()))
    per = {d: Counter() for d in dialogues}
    lat_t = lat_m = 0
    rows_out = []
    for uid in ids:
        g, p = gate[uid], pulses[uid]
        qmap, lt, lm = own_map(T.tok, refs[uid], g["baseline"]["content_ids"], g["baseline"]["terminated"], dur[uid], T.eos)
        lat_t += lt
        lat_m += lm
        byt = {x["t"]: x for x in p["queries"]}
        for t, (s, Y) in sorted(qmap.items()):
            x = byt[t]
            b0 = x["arms"]["B0"]["top1"]
            d = per[dlg[uid]]
            d[{"EN-confusion": "n_C", "EN-correct": "n_EN", "ZH-correct": "n_ZH"}[s]] += 1
            r_ = {"utterance_id": uid, "t": t, "stratum": s}
            for a in ARMS:
                top = x["arms"][a]["top1"]
                if s == "EN-confusion":
                    d[f"{a}_C"] += int(b0 not in Y and top in Y)
                elif s == "EN-correct":
                    d[f"{a}_H_EN"] += int(b0 in Y and top not in Y)
                else:
                    d[f"{a}_H_ZH"] += int(top not in Y)
                    d[f"{a}_act_ZH"] += int(x["arms"][a]["executed"])
            rows_out.append(r_)

    def S(k):
        return sum(per[d][k] for d in dialogues)

    def nd(f):
        return sum(1 for d in dialogues if f(per[d]))
    C = {a: S(f"{a}_C") for a in ARMS}
    HZ = {a: S(f"{a}_H_ZH") for a in ARMS}
    HE = {a: S(f"{a}_H_EN") for a in ARMS}
    U = {a: C[a] - HZ[a] - HE[a] for a in ARMS}
    nC, nEN, nZH = S("n_C"), S("n_EN"), S("n_ZH")
    actB2 = S("B2_act_ZH")
    gc_ = {d: per[d]["B2_C"] - per[d]["B3_C"] for d in dialogues}
    gu = {d: (per[d]["B2_C"] - per[d]["B2_H_ZH"] - per[d]["B2_H_EN"]) - (per[d]["B3_C"] - per[d]["B3_H_ZH"] - per[d]["B3_H_EN"]) for d in dialogues}
    pos_u = [v for v in gu.values() if v > 0]
    eos_b2 = sum(1 for uid in ids for x in pulses[uid]["queries"] if x["arms"]["B2"]["top1"] == T.eos and x["arms"]["B0"]["top1"] != T.eos
                 and len(gate[uid]["baseline"]["content_ids"]) - x["t"] >= 10)

    def rt(a, b):
        return a / b if b else None
    met = {"mapped_EN_confusion": nC, "mapped_EN_confusion_dialogues": nd(lambda x: x["n_C"] > 0),
           "mapped_EN_correct": nEN, "mapped_EN_correct_dialogues": nd(lambda x: x["n_EN"] > 0),
           "mapped_ZH_correct": nZH, "mapped_ZH_correct_dialogues": nd(lambda x: x["n_ZH"] > 0),
           "active_B2_ZH_correct": actB2, "active_B2_ZH_dialogues": nd(lambda x: x["B2_act_ZH"] > 0),
           "English_mapping_fraction": rt(lat_m, lat_t), "C_B2": C["B2"], "corrected_dialogues_B2": nd(lambda x: x["B2_C"] > 0),
           "C_B1": C["B1"], "H_ZH_B1": HZ["B1"], "ZH_harmed_dialogues_B1": nd(lambda x: x["B1_H_ZH"] > 0),
           "uninformative_query_fraction": rt(rec["uninformative_queries"], rec["structural_queries"]),
           "correction_retention_B2_B1": rt(C["B2"], C["B1"]), "ZH_harm_ratio_B2_B1": rt(HZ["B2"], HZ["B1"]),
           "H_ZH_B2": HZ["B2"], "ZH_corruption_rate_B2": rt(HZ["B2"], nZH), "ZH_active_corruption_rate_B2": rt(HZ["B2"], actB2),
           "H_EN_B2": HE["B2"], "EN_corruption_rate_B2": rt(HE["B2"], nEN), "C_B2_minus_B3": C["B2"] - C["B3"],
           "U_B2_minus_B3": U["B2"] - U["B3"], "H_ZH_B2_minus_B3": HZ["B2"] - HZ["B3"],
           "positive_correction_gain_dialogues": sum(1 for v in gc_.values() if v > 0), "positive_utility_gain_dialogues": len(pos_u),
           "max_positive_utility_dialogue_share": max(pos_u) / sum(pos_u) if pos_u else 1.0, "U_B2": U["B2"],
           "any_dialogue_ZH_damage_events_ge3_AND_active_rate_gt020": any(per[d]["B2_H_ZH"] >= 3 and per[d]["B2_act_ZH"] > 0
                                                                          and per[d]["B2_H_ZH"] / per[d]["B2_act_ZH"] > .2 for d in dialogues),
           "new_EOS_proxy_events_B2": eos_b2, **rec["energy_execution"], **rec["dose_comparison"], **a_metrics, **fc_metrics}
    gates = {k: pred(r, met) for k, r in cfg["gate_predicates"].items()}
    return {"metrics": met, "per": per, "dialogues": dialogues, "C": C, "HZ": HZ, "HE": HE, "U": U, "gates": gates,
            "latin": (lat_t, lat_m)}


def cmd_full(args) -> dict:
    cfg, m, rows, checks = base_checks("full")
    gseal, gate, gc = load_sealed("gate_seal.json", "gate", rows)
    pseal, pulses, pc = load_sealed("pulse_seal.json", "pulses", rows)
    checks.update({f"gate_{k}": v for k, v in gc.items()})
    checks.update({f"pulse_{k}": v for k, v in pc.items()})
    prim = json.loads((ROOT / RUN / "audit_PRIMARY.json").read_text())
    ev = json.loads((ROOT / RUN / "evaluation.json").read_text())
    checks["primary_pass_pushed_before_evaluation"] = prim["verdict"] == "SRD2_G0_AUDIT_PRIMARY: PASS" and pushed(f"{RUN}/audit_PRIMARY.json") \
        and ev["audit_PRIMARY_sha256"] == file_hash(ROOT / RUN / "audit_PRIMARY.json") and ev["pulse_seal_sha256"] == file_hash(ROOT / RUN / "pulse_seal.json")
    checks["evaluation_units_hash"] = ev["units_sha256"] == file_hash(ROOT / RUN / "evaluation_units.json")
    sel, P = selected()
    ids = [r["utterance_id"] for r in sel]
    import pyarrow.dataset as ds
    if file_hash(P["source"]) != P["source_sha256"]:
        raise SystemExit("role parquet changed")
    tbl = ds.dataset(P["source"], format="parquet").to_table(columns=["utterance_id", "role", "transcript_raw"],
                                                            filter=ds.field("utterance_id").isin(ids)).to_pylist()
    refs = {r["utterance_id"]: r["transcript_raw"] for r in tbl}
    checks["reference_scope_exactly_selected400"] = sorted(refs) == sorted(ids) and all(r["role"] == "D-dev-select" for r in tbl) \
        and ev["reference_access"]["rows"] == 400
    T = Tok(cfg)
    perm = json.loads((ROOT / RUN / "permutation.json").read_text())
    rec = primary_recompute(cfg, T, rows, gate, pulses, perm)
    checks["primary_recompute_clean"] = not rec["errors"]
    a_audit = json.loads((ROOT / RUN / "audit_A.json").read_text())
    fc = json.loads((ROOT / RUN / "resource_forecast.json").read_text())
    core = full_core(cfg, T, sel, gate, pulses, refs, rec, a_audit["metrics"], fc["metrics"])
    met, per, dialogues, gates = core["metrics"], core["per"], core["dialogues"], core["gates"]
    C, HZ, HE, U = core["C"], core["HZ"], core["HE"], core["U"]
    # ---- agreement with the primary evaluator ------------------------------------------------------------
    em = ev["metrics"]
    agree = {}
    for k, v in met.items():
        agree[k] = (v == em.get(k)) if isinstance(v, (bool, int)) and not isinstance(v, float) else close(v, em.get(k), 1e-9)
    checks["all_metrics_agree"] = all(agree.values())
    checks["primary_integer_counts_agree"] = (ev["counts"]["C"] == C and ev["counts"]["H_ZH"] == HZ and ev["counts"]["H_EN"] == HE
                                              and ev["counts"]["U"] == U)
    checks["gates_agree"] = gates == ev["gates"]
    checks["per_dialogue_tallies_agree"] = all(int(ev["per_dialogue"][d].get(k, 0)) == int(per[d].get(k, 0))
                                               for d in dialogues for k in set(ev["per_dialogue"][d]) | set(per[d]))
    # ---- bootstrap (same frozen draw definition, independent code) ------------------------------------------
    st = cfg["statistics"]
    rng = np.random.default_rng(st["seed"])
    draws = rng.integers(0, len(dialogues), size=(st["bootstrap_replicates"], len(dialogues)))

    def col(k):
        return np.array([per[d][k] for d in dialogues], dtype=np.float64)[draws].sum(axis=1)
    Ub = {a: col(f"{a}_C") - col(f"{a}_H_ZH") - col(f"{a}_H_EN") for a in ARMS}
    fam = {"C_B2-C_B3": col("B2_C") - col("B3_C"), "U_B2-U_B3": Ub["B2"] - Ub["B3"],
           "H_ZH_B3-H_ZH_B2": col("B3_H_ZH") - col("B2_H_ZH"), "H_ZH_B1-H_ZH_B2": col("B1_H_ZH") - col("B2_H_ZH")}
    lo, hi = st["family_quantiles"]
    boot = {k: {"pct95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                "bonferroni_98.75": [float(np.quantile(v, lo)), float(np.quantile(v, hi))]} for k, v in fam.items()}
    checks["bootstrap_agree"] = all(close(boot[k][w][j], ev["bootstrap"]["family"][k][w][j], 1e-12)
                                    for k in boot for w in ("pct95", "bonferroni_98.75") for j in (0, 1))
    critical = not (all(checks.values()) and rec["errors"] == {} and all(ev["critical"].values()))
    flags = {"critical_failure": critical, "independent_FULL_audit_PASS": not critical}
    label = first_label(cfg, gates, flags)
    checks["terminal_label_agrees"] = label == ev["provisional_terminal_label"] or critical
    if label != ev["provisional_terminal_label"]:
        flags = {"critical_failure": True, "independent_FULL_audit_PASS": False}
        label = first_label(cfg, gates, flags)
    verdict = "SRD2_G0_AUDIT_FULL: PASS" if all(checks.values()) and not critical else "SRD2_G0_AUDIT_FULL: BLOCK"
    leaves = {}
    for name, rule in cfg["gate_predicates"].items():
        def walk(r):
            if "all" in r or "any" in r:
                return [y for c in r.get("all", r.get("any")) for y in walk(c)]
            v = met.get(r["metric"])
            st_ = "NOT_ESTIMABLE" if v is None or (isinstance(v, float) and not math.isfinite(v)) else ("PASS" if pred(r, met) else "FAIL")
            return [{"metric": r["metric"], "op": r["op"], "threshold": r["value"], "value": v, "status": st_}]
        leaves[name] = walk(rule)
    return {"schema": "srd2_g0_audit_FULL_v1", "verdict": verdict, "terminal_label": label, "flags": flags, "checks": checks,
            "metrics": met, "metric_agreement": agree, "gates": gates, "leaves": leaves, "counts": {"C": C, "H_ZH": HZ, "H_EN": HE, "U": U},
            "per_dialogue": {d: dict(per[d]) for d in dialogues}, "bootstrap_family": boot, "recompute": rec,
            "evaluation_sha256": file_hash(ROOT / RUN / "evaluation.json"), "manifest_hash": m["manifest_hash"],
            "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pre", "a", "primary", "full"))
    ap.add_argument("--out", default=RUN)
    args = ap.parse_args()
    if args.out.rstrip("/") != RUN:
        raise SystemExit("only the frozen run root is allowed")
    name = {"pre": "audit_PRE.json", "a": "audit_A.json", "primary": "audit_PRIMARY.json", "full": "audit_FULL.json"}[args.stage]
    dest = ROOT / RUN / name
    if dest.exists():
        raise FileExistsError(f"{name} exists; failed audits are preserved, never overwritten")
    res = {"pre": cmd_pre, "a": cmd_a, "primary": cmd_primary, "full": cmd_full}[args.stage](args)
    atomic_json(dest, res)
    if args.stage == "full":
        atomic_json(ROOT / RUN / "terminal.json", {"schema": "srd2_g0_terminal_v1", "label": res["terminal_label"],
                                                   "audit_FULL": res["verdict"], "flags": res["flags"],
                                                   "audit_FULL_sha256": file_hash(dest), "created_unix": time.time()})
    print(json.dumps({"verdict": res["verdict"], **({"label": res["terminal_label"]} if args.stage == "full" else {}),
                      "failed_checks": [k for k, v in res["checks"].items() if not v]}, indent=1))


if __name__ == "__main__":
    main()
