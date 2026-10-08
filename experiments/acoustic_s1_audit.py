#!/usr/bin/env python
"""S1 independent auditor. Does NOT import the S1 runner (experiments/acoustic_s1.py), the S1 provider
(csasr.inference_cf.s1_evidence), the R0 region helpers or the S1 evaluator / decision code. It re-derives candidates from
the archived raw logits, regions from the archived attention + R0 heard intervals + waveform bytes, masks from waveform
bytes, scores / rankings from the archived masked logits, and (FULL) hits, gates, bootstrap and the terminal label from
raw rows. Generic hashing / bf16 IO and torch.log_softmax are the only shared primitives. The prerun synthetic
agreement check runs the primary provider in a SEPARATE subprocess and only compares its outputs.

prerun      PASS_TO_S1
candidates  S1_AUDIT: PASS (CANDIDATES)   -- after the pushed candidate seal, before any mask
primary     S1_AUDIT: PASS (PRIMARY)      -- after the pushed output seal, before any reference
full        S1_AUDIT: PASS (FULL)         -- independent metrics, bootstrap, gates, terminal label
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

FREEZE = "a89067b55ca89097ca5e0d38f429ed678d986179"
CONFIG = "configs/inference_cf/s1_acoustic_evidence.json"
PANEL = "docs/inference_cf/S1_PANEL.json"
FROZEN = (CONFIG, PANEL, "docs/inference_cf/S1_ACOUSTIC_EVIDENCE_SPEC.md", "docs/inference_cf/S1_CODEX_DESIGN.md",
          "docs/inference_cf/S1_REFERENCE_FIREWALL.md", "tests/test_s1_freeze_contract.py")
BASE = "results/inference_cf/s1"
RUN = f"{BASE}/run1"
PLAN = f"{BASE}/plan_sealed.json"
PRERUN = f"{BASE}/prerun_audit.json"
CAND_SEAL = f"{RUN}/candidates_sealed.json"
CAND_AUDIT = f"{BASE}/candidates_audit.json"
OUTPUT_SEAL = f"{BASE}/output_seal.json"
PRIMARY = f"{BASE}/primary_analysis.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
ANALYSIS = f"{BASE}/analysis.json"
FULL_AUDIT = f"{BASE}/final_audit.json"
RUNNER, PROVIDER, EVALUATOR = "experiments/acoustic_s1.py", "src/csasr/inference_cf/s1_evidence.py", "experiments/acoustic_s1_evaluate.py"
HIST = ("results/inference_cf/st_loc0/run1/calibration/baseline_logits.npz", "results/inference_cf/st_prompt_r1/runA/capture/none_logits.npz")
LAC_SEAL = "results/inference_cf/p2sel_lac/candidates_sealed.json"
CB = [50258, 50260, 50360, 50364]
CE = [50258, 50259, 50360, 50364]
EOS = 50257
FR = 320
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
SCORERS = ("M_ORIGINAL", "E_ORIGINAL", "AUTO_ORIGINAL", "EN_REGION", "OFF_TARGET", "LAC_UNION", "SHUFFLED_SUPPORT", "NO_CONTRAST")
CONTROLS = ("M_ORIGINAL", "OFF_TARGET", "LAC_UNION", "SHUFFLED_SUPPORT")
FORBIDDEN_RUNTIME = ("p2rj", "positions.json", "ST_PROMPT_R1_PANEL", "evaluation", "oracle", "mms", "transcript", "alignment",
                     "_analysis", "/roles/", "candidates_existing", "population.json", "acoustic_s1_evaluate", "acoustic_s1_audit", "final_audit")
FORBIDDEN_SOURCE = ("positions.json", "p2rj", "target_ids", "competitor", "evaluation_membership", "stratum", "ST_PROMPT_R1_PANEL",
                    "oracle_analysis", "script_frame", "script_label", "mms", "population.json", "acoustic_s1_evaluate")


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


def rawsha(a) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def bf16(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def blob(commit: str, rel: str) -> str:
    return "sha256:" + hashlib.sha256(subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True).stdout).hexdigest()


def committed(rel: str) -> bool:
    return git("ls-files", rel) == rel and blob("HEAD", rel) == fhash(ROOT / rel)


def last_commit(rel: str) -> str:
    return git("log", "-n1", "--format=%H", "--", rel)


def ancestor(a: str, b: str) -> bool:
    return bool(a) and bool(b) and subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT).returncode == 0


def on_remote(rel: str) -> bool:
    return ancestor(last_commit(rel), "origin/cs-asr-steer-inf")


def J(rel):
    return json.loads((ROOT / rel).read_text())


def gen_cfg():
    cfg = J(CONFIG)
    g = json.loads((Path(cfg["model"]["dir"]) / "generation_config.json").read_text())
    return cfg, g, list(g["suppress_tokens"]), list(g["begin_suppress_tokens"])


def processed_logp(raw: np.ndarray, t: int, sup, beg):
    import torch
    x = np.asarray(raw, dtype=np.float32).copy()
    x[sup] = -np.inf
    if t == 0:
        x[beg] = -np.inf
    return x, torch.log_softmax(torch.from_numpy(x.copy()), dim=-1).numpy()


def my_topk(x: np.ndarray, k: int) -> list[int]:
    fin = [(-float(x[i]), i) for i in np.flatnonzero(np.isfinite(x))]
    fin.sort()
    return [i for _, i in fin[:k]]


def my_rank(ids, s) -> list[int]:
    return [i for _, i in sorted(zip([-float(v) for v in s], [int(i) for i in ids]))]


def load_wave(path: str) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16000:
        raise ValueError("sample rate")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


class ExactAttention:
    """Independent exact route: Fraction frame masses A_f = sum_h attention, normalized by their heard total; the density
    of frame f is A_f / (T * actual heard samples of f); C(x) is the Fraction integral over samples [0, x)."""

    def __init__(self, heads32: np.ndarray, heard: int):
        h = np.asarray(heads32, dtype=np.float32)
        self.heard, self.F = heard, -(-heard // FR)
        self.A = [sum((Fraction(float(v)) for v in h[:, f]), Fraction(0)) for f in range(self.F)]
        self.T = sum(self.A, Fraction(0))
        self.raw_mass = self.T / h.shape[0]
        self.cum = [Fraction(0)]
        for v in self.A:
            self.cum.append(self.cum[-1] + v)

    def C(self, x: int) -> Fraction:
        f = min(x // FR, self.F - 1)
        dur = min(FR, self.heard - f * FR)
        return self.cum[f] + self.A[f] * Fraction(x - f * FR, dur)

    def v(self, s: int, e: int) -> Fraction:
        return (self.C(e) - self.C(s)) / self.T


def my_target(att: ExactAttention, ivs, x, heard, cfg):
    R = cfg["regions"]
    en = [(max(0, a), min(heard, b)) for a, b, c in ivs if c == 1 and min(heard, b) - max(0, a) >= R["minimum_target_samples"]]
    if not en:
        return {"status": "NO_EN_REGION"}
    if att.raw_mass < Fraction(1, 2):
        return {"status": "LOW_HEARD_MASS"}
    best = None
    for l, r in en:
        L = min(r - l, R["maximum_target_samples"])
        for s in sorted(set(range(l, r - L + 1, FR)) | {r - L}):
            v = att.v(s, s + L)
            if best is None or (-v, s, s + L) < best[0]:
                best = ((-v, s, s + L), s, s + L, v)
    _, s, e, v = best
    seg = x[s:e].astype(np.float64)
    rms = math.sqrt(float(np.mean(seg * seg)))
    st = "LOW_ASSOCIATION" if v < Fraction(1, 10) else ("LOW_ENERGY" if rms < R["target_minimum_rms"] else "OK")
    return {"status": st, "bounds": [s, e], "integral": v, "energy": float((seg * seg).sum())}


def my_off(t, att: ExactAttention, ivs, x, heard):
    if t["status"] != "OK":
        return {"status": "NO_TARGET"}
    a, b = t["bounds"]
    L = b - a
    track = np.zeros(heard, dtype=np.int8)
    for s, e, c in ivs:
        track[max(0, s):min(heard, e)] = c
    cap = min(Fraction(1, 10), t["integral"] / 2)
    best = None
    for s in sorted(set(range(0, heard - L + 1, FR)) | {heard - L}):
        if not (s + L <= a or s >= b):
            continue
        if Fraction(int(np.count_nonzero(track[s:s + L] == 1)), L) > Fraction(1, 10):
            continue
        v = att.v(s, s + L)
        if v > cap:
            continue
        seg = x[s:s + L].astype(np.float64)
        if math.sqrt(float(np.mean(seg * seg))) < 1e-4:
            continue
        ratio = float((seg * seg).sum()) / t["energy"]
        if not (0.5 <= ratio <= 2.0):
            continue
        key = (v, abs(math.log(ratio)), s)
        if best is None or key < best[0]:
            best = (key, s)
    return {"status": "NO_MATCHED_OFFTARGET"} if best is None else {"status": "OK", "bounds": [best[1], best[1] + L], "integral": best[0][0]}


def verdict(stage, checks, notes, ok_label, fail_label):
    return {"schema": f"s1_audit_{stage}_v1", "verdict": ok_label if all(checks.values()) else fail_label,
            "failed": sorted(k for k, v in checks.items() if not v), "checks": checks, "notes": notes, "created_unix": time.time()}


# ---- prerun ---------------------------------------------------------------------------------------------------------

SYNTH = r'''
import json, sys, numpy as np
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1]]
from csasr.inference_cf import s1_evidence as S
cfg = json.load(open(sys.argv[1] + "/configs/inference_cf/s1_acoustic_evidence.json"))
rng = np.random.default_rng(int(sys.argv[2]))
out = []
for case in range(40):
    heard = int(rng.integers(20000, 200000)); F = -(-heard // 320)
    heads = rng.random((10, 1500)) ** 4
    if case % 7 == 0: heads[:, :F] *= 0.01
    x = (rng.standard_normal(heard + int(rng.integers(0, 5000))) * rng.choice([1e-5, 0.05])).astype(np.float32)
    cuts = sorted(set(int(c) for c in rng.integers(1, heard, 6)))
    edges = [0] + cuts + [heard]
    ivs = np.array([[edges[i], edges[i + 1], int(rng.integers(0, 3))] for i in range(len(edges) - 1)], dtype=np.int64)
    att = S.heard_attention(heads.astype(np.float32), heard, 0.5)
    t = S.select_target(att, ivs, x, heard, cfg)
    o = S.select_offtarget(att, ivs, x, heard, t, cfg)
    proc = rng.standard_normal(51866).astype(np.float32); proc[rng.integers(0, 51866, 90)] = -np.inf
    proc[[5, 6, 7]] = proc.max()
    tops = {b: S.topk(proc[::-1].copy() if b == "E" else proc, 20) for b in S.BRANCHES}
    un = S.union(tops, 50257)
    l0 = rng.standard_normal(len(un["ids"])).astype(np.float32); lm = rng.standard_normal(len(un["ids"])).astype(np.float32)
    sc = S.scores(l0, lm)
    out.append({"seed_case": case, "heard": heard,
                "raw_mass": att["raw_mass"], "target": t, "off": o, "tops": tops, "union": un,
                "scores": sc.tolist(), "rank": S.ranking(un["ids"], sc), "shuffle": S.shuffled(np.arange(len(un["ids"]), dtype=float), "U", 3, 20).tolist()})
print(json.dumps(out))
'''


def synthetic_agreement(seed: int = 240924) -> tuple[bool, dict]:
    """Primary provider in a subprocess on synthetic inputs vs this module's independent re-implementation."""
    import os
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}")
    res = subprocess.run([sys.executable, "-c", SYNTH, str(ROOT), str(seed)], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        return False, {"stderr": res.stderr[-2000:]}
    prim = json.loads(res.stdout)
    cfg = J(CONFIG)
    rng = np.random.default_rng(seed)
    bad = []
    for case, p in enumerate(prim):
        heard = int(rng.integers(20000, 200000))
        F = -(-heard // FR)
        heads = rng.random((10, 1500)) ** 4
        if case % 7 == 0:
            heads[:, :F] *= 0.01
        x = (rng.standard_normal(heard + int(rng.integers(0, 5000))) * rng.choice([1e-5, 0.05])).astype(np.float32)
        cuts = sorted(set(int(c) for c in rng.integers(1, heard, 6)))
        edges = [0] + cuts + [heard]
        ivs = [[edges[i], edges[i + 1], int(rng.integers(0, 3))] for i in range(len(edges) - 1)]
        att = ExactAttention(heads.astype(np.float32), heard)
        t = my_target(att, ivs, x, heard, cfg)
        o = my_off(t, att, ivs, x, heard)
        proc = rng.standard_normal(51866).astype(np.float32)
        proc[rng.integers(0, 51866, 90)] = -np.inf
        proc[[5, 6, 7]] = proc.max()
        tops = {b: my_topk(proc[::-1].copy() if b == "E" else proc, 20) for b in ("M", "E", "AUTO")}
        ids = sorted({i for v in tops.values() for i in v})
        l0 = rng.standard_normal(len(ids)).astype(np.float32)
        lm = rng.standard_normal(len(ids)).astype(np.float32)
        sc = l0.astype(np.float64) * 2 - lm.astype(np.float64)
        seed_s = int(hashlib.sha256(b"s1-shuffle-v1|U|3|20").hexdigest()[:16], 16)
        perm = np.random.Generator(np.random.PCG64(seed_s)).permutation(len(ids))
        ok = (abs(p["raw_mass"] - float(att.raw_mass)) < 1e-9 and p["target"]["status"] == t["status"] and p["target"].get("bounds") == t.get("bounds")
              and p["off"]["status"] == o["status"] and p["off"].get("bounds") == o.get("bounds") and p["tops"] == tops and p["union"]["ids"] == ids
              and np.allclose(p["scores"], sc, rtol=0, atol=0) and p["rank"] == my_rank(ids, sc)
              and p["shuffle"] == np.arange(len(ids), dtype=float)[perm].tolist())
        if t.get("integral") is not None and p["target"].get("integral") is not None:
            ok &= Fraction(int(p["target"]["integral_num"]), int(p["target"]["integral_den"])) == t["integral"]
        if not ok:
            bad.append(case)
    statuses = defaultdict(int)
    for p in prim:
        statuses[p["target"]["status"] + "/" + p["off"]["status"]] += 1
    return not bad, {"cases": len(prim), "disagreements": bad, "status_mix": dict(statuses)}


def static_firewall() -> dict:
    run_src = (ROOT / RUNNER).read_text()
    prov = (ROOT / PROVIDER).read_text()
    ev = (ROOT / EVALUATOR).read_text()
    me = Path(__file__).read_text()
    code = lambda s: "\n".join(l for l in s.splitlines() if not l.strip().startswith("#"))
    rs, ps = code(run_src.split('"""', 2)[2]), code(prov.split('"""', 2)[2])
    rs_scan = re.sub(r"SOURCES = FROZEN \+ \(.*?\)\n\n", "", rs, flags=re.S)          # hash-pinned provenance list only
    c = {}
    c["runner_no_reference_identifiers"] = not [f for f in FORBIDDEN_SOURCE if f in rs_scan]
    c["provider_no_reference_identifiers"] = not [f for f in FORBIDDEN_SOURCE if f in ps]
    c["runner_no_steering_or_training"] = not any(x in rs + ps for x in ("InterventionHook", "SteeringHook", "pulse_action", "apply_steering", ".backward(", "optim.", "requires_grad_(True"))
    r0_reads = [rs[m_.end():].splitlines()[1] for m_ in re.finditer(r'np\.load\(ROOT / u\["r0_primary_arrays"\]\) as z:', rs)]
    c["runner_r0_only_heard_track"] = bool(r0_reads) and all(re.findall(r"z\[([^\]]+)\]", l_) == ["R0_KEY"] for l_ in r0_reads) \
        and 'R0_KEY = "track_heard_intervals"' in rs
    c["runner_does_not_import_evaluator_or_auditor"] = (re.search(r"^\s*(from|import)\s+\S*acoustic_s1_(evaluate|audit)", rs, re.M) is None
                                                       and "import_module" not in rs and "__import__" not in rs and "acoustic_s1_evaluate" not in rs_scan)
    c["runner_records_open_log"] = "addaudithook" in rs and "opened_paths" in rs
    seg = lambda name: rs[rs.index(f"def {name}("):][:rs[rs.index(f"def {name}("):].index("\ndef ", 5)]
    aco = seg("acoustic_utterance") + seg("cmd_acoustic")
    c["no_masked_regeneration"] = not any(x in aco for x in ("topk(", "detect_auto(", "candidate_utterance(", "union(", "select_target(", "select_offtarget("))
    c["evaluator_gated_on_seal_and_primary_audit"] = ev.index("def load_sealed") < ev.index("seal = load_sealed()") < ev.index("(ROOT / POSITIONS)")
    c["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(s1_evidence|acoustic_s1\b|acoustic_s1_evaluate|r0_regions)", me, re.M) is None
    return c


def cmd_prerun(args) -> dict:
    cfg, g, sup, beg = gen_cfg()
    checks, notes = {}, {}
    checks["frozen_unchanged_since_freeze"] = all(blob(FREEZE, p) == fhash(ROOT / p) for p in FROZEN)
    P = J(PANEL)
    checks["panel_identity"] = canon({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel_identity_hash"]
    checks["sources_pinned"] = all(fhash(ROOT / p) == h for p, h in cfg["source_sha256"].items())
    checks["model_files"] = all(fhash(Path(cfg["model"]["dir"]) / n) == h for n, h in cfg["model"]["files"].items())
    plan = J(PLAN)
    checks["plan_committed"] = committed(PLAN)
    checks["plan_hash"] = canon({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"]
    checks["plan_checks_all_true"] = all(plan["checks"].values())
    rq = plan["runtime"]["runtime_queries"]
    checks["projection_label_free"] = (rq == P["runtime_queries"] and all(set(u) >= {"utterance_id", "audio_path"} for u in plan["runtime"]["utterances"])
                                       and not any(f in json.dumps(plan["runtime"]) for f in ("stratum", "target_ids", "competitor", "EN-confusion", "p2rj")))
    toks = {u["utterance_id"]: u["baseline_content_tokens"] for u in P["utterances"]}
    checks["projection_prefix_only"] = all(u["content_prefix_tokens"] == toks[u["utterance_id"]][:max(q["t"] for q in rq if q["utterance_id"] == u["utterance_id"])]
                                           for u in plan["runtime"]["utterances"])
    checks["forecast_within_allocation"] = (plan["forecast"]["original_cached_steps_max"] <= cfg["compute"]["original_branch_cached_steps_max"]
                                            and plan["forecast"]["masked_cached_steps_max"] <= cfg["compute"]["masked_cached_steps_max"])
    checks["suppression_matches_installed_generation_config"] = plan["suppression"] == {"suppress": sup, "begin": beg}
    # installed AUTO prompt semantics (stub; no model)
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
    gg = GenerationConfig.from_dict(g)
    gg.language, gg.task, gg.return_timestamps, gg.forced_decoder_ids = None, "transcribe", False, None
    ok = True
    for d in (50259, 50260, 50300, 50358):
        calls = []
        stub = SimpleNamespace(device=torch.device("cpu"), detect_language=lambda **kw: (calls.append(kw), torch.tensor([d]))[1])
        tk = WhisperGenerationMixin._retrieve_init_tokens(stub, None, 1, gg, SimpleNamespace(forced_decoder_ids=None), 3000, {"encoder_outputs": "ENC"})
        ok &= tk.tolist() == [[50258, d, 50360, 50364]] and len(calls) == 1 and calls[0]["encoder_outputs"] == "ENC" and calls[0]["input_features"] is None
    checks["installed_auto_prompt_semantics"] = bool(ok)
    checks["installed_generation_file"] = fhash(cfg["conditions"]["installed_generation_source"]) == cfg["conditions"]["installed_generation_file_sha256"]
    checks["installed_detect_language_lowest_id_argmax"] = ("non_lang_mask[list(generation_config.lang_to_id.values())] = False" in
                                                            Path(cfg["conditions"]["installed_generation_source"]).read_text())
    sf = static_firewall()
    checks.update({f"static:{k}": v for k, v in sf.items()})
    ok, n = synthetic_agreement()
    checks["synthetic_primary_vs_independent"] = ok
    notes["synthetic"] = n
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests/test_s1_impl.py", "tests/test_s1_freeze_contract.py"],
                           cwd=ROOT, capture_output=True, text=True)
    checks["s1_tests_pass"] = tests.returncode == 0
    notes["tests_tail"] = tests.stdout.strip().splitlines()[-1:] if tests.stdout else tests.stderr[-500:]
    return verdict("prerun", checks, notes, "PASS_TO_S1", "FAIL_PRERUN")


# ---- candidates -------------------------------------------------------------------------------------------------------

def runtime_checks(rt: dict, m: dict, stage: str) -> dict:
    forb = [p for p in rt.get("opened_paths", []) if any(f.lower() in p.lower() for f in FORBIDDEN_RUNTIME)]
    return {f"{stage}_runtime_completed": rt["status"] == "completed" and rt["manifest_hash"] == m["manifest_hash"],
            f"{stage}_frozen_model": rt["model_grads_none"] and not rt["requires_grad_any"] and not rt["training_mode"] and rt["weights_unchanged_probe"],
            f"{stage}_no_hooks_left": rt["top_forward_hooks"] == 0,
            f"{stage}_no_autograd_training_steering_lid": all(rt["counters"].get(k, 0) == 0 for k in ("autograd_calls", "optimizer_steps", "steering_hooks", "lid_calls")),
            f"{stage}_file_open_firewall": "opened_paths" in rt and not forb,
            f"{stage}_wall_under_3h": rt["elapsed_sec"] < 10800}, forb


def manifest_ok(stage: str):
    m = J(f"{RUN}/manifest_{stage}.json")
    ok = canon({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"] and all(fhash(ROOT / p) == h for p, h in m["sources"].items())
    return m, ok


def cmd_candidates(args) -> dict:
    cfg, g, sup, beg = gen_cfg()
    checks, notes = {}, {}
    plan = J(PLAN)
    m, checks["manifest_valid_sources_unchanged"] = manifest_ok("candidates")
    checks["manifest_after_prerun_pass"] = J(PRERUN)["verdict"] == "PASS_TO_S1" and ancestor(last_commit(PRERUN), m["git_commit"])
    rt = J(f"{RUN}/candidates_runtime.json")
    rc, forb = runtime_checks(rt, m, "candidates")
    checks.update(rc)
    notes["forbidden_opened"], notes["opened_count"] = forb, len(rt.get("opened_paths", []))
    seal = J(CAND_SEAL)
    checks["seal_hash"] = canon({k: v for k, v in seal.items() if k != "seal_hash"}) == seal["seal_hash"]
    checks["seal_files_unchanged"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items())
    checks["seal_committed_and_on_remote"] = committed(CAND_SEAL) and on_remote(CAND_SEAL)
    checks["seal_covers_all_candidate_files"] = sorted(p for p in seal["files"] if "/candidates/" in p) == sorted(f"{RUN}/candidates/{p.name}" for p in (ROOT / RUN / "candidates").iterdir())
    hist = []
    for p in HIST:
        with np.load(ROOT / p) as z:
            hist.append({k: z[k] for k in z.files})
    lang_ids = sorted(g["lang_to_id"].values())
    bad = defaultdict(list)
    n_alias, steps = 0, {"M": 0, "E": 0, "AUTO": 0}
    statuses = defaultdict(int)
    tie_margin = []
    for i, u in enumerate(plan["runtime"]["utterances"]):
        d = J(f"{RUN}/candidates/{i:03d}.json")
        with np.load(ROOT / RUN / "candidates" / f"{i:03d}.npz") as z:
            arr = {k: z[k] for k in z.files}
        if d["utterance_id"] != u["utterance_id"] or fhash(ROOT / RUN / "candidates" / f"{i:03d}.npz") != d["arrays_file_sha256"]:
            bad["row_identity"].append(i)
        if any(ahash(arr[k]) != v["sha256"] for k, v in d["arrays_index"].items()) or set(arr) != set(d["arrays_index"]):
            bad["array_hashes"].append(i)
        x = load_wave(u["audio_path"])
        heard = min(len(x), 480000)
        if rawsha(x) != d["waveform_sha256"] or heard != d["heard"] or fhash(u["audio_path"]) != u["audio_full_file_sha256"]:
            bad["waveform"].append(i)
        with np.load(ROOT / u["r0_primary_arrays"]) as z:
            ivs = z["track_heard_intervals"].astype(np.int64).tolist()
        if ivs != d["r0_track_heard_intervals"] or fhash(ROOT / u["r0_primary_arrays"]) != u["r0_primary_arrays_sha256"]:
            bad["r0_intervals"].append(i)
        a = d["auto"]
        lv = np.array(a["language_logits"], dtype=np.float32)
        low = lang_ids[int(np.flatnonzero(lv == lv.max())[0])]
        if a["language_ids"] != lang_ids or a["detected"] != low or a["prompt"] != [50258, low, 50360, 50364]:
            bad["auto"].append(i)
        alias = {50260: "M", 50259: "E"}.get(low)
        if d["auto_alias"] != alias:
            bad["auto_alias"].append(i)
        n_alias += alias is not None
        mt = u["max_t"]
        for b in ("M", "E") + (() if alias else ("AUTO",)):
            steps[b] += mt + 1
        prompts = {"M": CB, "E": CE, "AUTO": a["prompt"]}
        for q in d["queries"]:
            j, t = q["j"], q["t"]
            Q = plan["runtime"]["runtime_queries"][j]
            if (Q["utterance_id"], Q["t"]) != (u["utterance_id"], t) or q["query"] != Q["absolute_query"] != 4 + t - 1:
                bad["query_identity"].append(j)
            prefix = u["content_prefix_tokens"][:t]
            if canon(CB + prefix) != Q["forced_zh_query_input_sha256"] or canon(CE + prefix) != Q["forced_en_query_input_sha256"]:
                bad["prefix"].append(j)
            if not all(np.array_equal(arr[f"q{j:03d}_M_raw"], h[f"q{j:03d}"]) for h in hist):
                bad["M_hist_bitwise"].append(j)
            lp = {}
            for b in ("M", "E", "AUTO"):
                src = alias if (b == "AUTO" and alias) else b
                raw = bf16(arr[f"q{j:03d}_{src}_raw"])
                proc, l = processed_logp(raw, t, sup, beg)
                lp[b] = l
                br = q["branches"][b]
                if not np.array_equal(l, arr[f"q{j:03d}_{src}_logp"]):
                    bad["logp_recompute"].append((j, b))
                if br["fed_sha256"] != canon(prompts[b] + prefix) or br["query_index"] != Q["absolute_query"] or not br["fed_is_prompt_plus_prefix"]:
                    bad["branch_lineage"].append((j, b))
                if not np.isfinite(raw).all() or not np.isfinite(l[np.isfinite(proc)]).all() or int(np.isfinite(proc).sum()) != 51866 - len(set(sup)):
                    bad["finite_legal"].append((j, b))
                for k in cfg["candidates"]["budgets"]:
                    if my_topk(proc, k) != br["top"][str(k)]:
                        bad["topk"].append((j, b, k))
                fin = np.sort(proc[np.isfinite(proc)])[::-1]
                tie_margin.append(float(fin[19] - fin[20]))
            for k in map(str, cfg["candidates"]["budgets"]):
                un = q["unions"][k]
                tops = {b: q["branches"][b]["top"][k] for b in ("M", "E", "AUTO")}
                ids = sorted({i_ for v in tops.values() for i_ in v})
                org = [sum(bit for b, bit in (("M", 1), ("E", 2), ("AUTO", 4)) if i_ in tops[b]) for i_ in ids]
                rks = {b: [tops[b].index(i_) + 1 if i_ in tops[b] else None for i_ in ids] for b in tops}
                typ = ["TERMINATE" if i_ == EOS else ("OTHER_ACTION" if i_ > EOS else "LEXICAL") for i_ in ids]
                dg = "sha256:" + hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
                if (un["ids"] != ids or un["origin"] != org or un["ranks"] != rks or un["types"] != typ or un["membership_sha256"] != dg
                        or seal["members"][f"q{j:03d}"][k] != dg or len(ids) > 3 * int(k)):
                    bad["union"].append((j, k))
                if any(abs(un["logp"][b][n] - float(lp[b][c])) > 0 for b in lp for n, c in enumerate(ids)):
                    bad["union_logp"].append((j, k))
            # region: independent per-sample density route
            att = ExactAttention(bf16(arr[f"q{j:03d}_M_heads"]), heard)
            raw_mass = float(att.raw_mass)
            reg = q["region"]
            tgt = my_target(att, ivs, x, heard, cfg)
            off = my_off(tgt, att, ivs, x, heard)
            statuses[tgt["status"] + "/" + off["status"]] += 1
            if (abs(reg["raw_heard_mass"] - raw_mass) > 1e-9 or reg["target"]["status"] != tgt["status"] or reg["target"].get("bounds") != tgt.get("bounds")
                    or reg["off_target"]["status"] != off["status"] or reg["off_target"].get("bounds") != off.get("bounds")
                    or reg["paired_available"] != (tgt["status"] == "OK" and off["status"] == "OK")):
                bad["region"].append(j)
            if tgt.get("integral") is not None and Fraction(int(reg["target"]["integral_num"]), int(reg["target"]["integral_den"])) != tgt["integral"]:
                bad["region_integral"].append(j)
            if off.get("integral") is not None and Fraction(int(reg["off_target"]["integral_num"]), int(reg["target"]["integral_den"])) != off["integral"]:
                bad["region_integral"].append(j)
    checks.update({f"independent:{k}": not bad.get(k) for k in ("row_identity", "array_hashes", "waveform", "r0_intervals", "auto", "auto_alias",
                                                                 "query_identity", "prefix", "M_hist_bitwise", "logp_recompute", "branch_lineage",
                                                                 "finite_legal", "topk", "union", "union_logp", "region", "region_integral")})
    c = rt["counters"]
    checks["counters_encoders_80"] = c["encoder_calls"] == 80 and c["masked_encoder_calls"] == 0
    checks["counters_detection_80"] = c["auto_detection_queries"] == 80 and c["auto_alias"] == n_alias
    checks["counters_steps_exact"] = c["decoder_steps"] == steps and sum(steps.values()) <= cfg["compute"]["original_branch_cached_steps_max"]
    notes["bad"] = {k: v[:20] for k, v in bad.items()}
    notes["region_status_mix"] = dict(statuses)
    notes["min_top20_boundary_gap"] = min(tie_margin)
    notes["auto_alias_count"] = n_alias
    return verdict("candidates", checks, notes, "S1_AUDIT: PASS (CANDIDATES)", "S1_AUDIT: FAIL (CANDIDATES)")


# ---- primary ----------------------------------------------------------------------------------------------------------

def cmd_primary(args) -> dict:
    cfg, g, sup, beg = gen_cfg()
    checks, notes = {}, {}
    plan = J(PLAN)
    m, checks["manifest_valid_sources_unchanged"] = manifest_ok("acoustic")
    mc = J(f"{RUN}/manifest_candidates.json")
    checks["sources_identical_across_jobs"] = {p: h for p, h in m["sources"].items() if p not in (CAND_SEAL, CAND_AUDIT)} == mc["sources"]
    seal_c = J(CAND_SEAL)
    checks["order_candidate_seal_then_audit_then_acoustic"] = (m["candidate_seal_hash"] == seal_c["seal_hash"] and J(CAND_AUDIT)["verdict"] == "S1_AUDIT: PASS (CANDIDATES)"
                                                               and ancestor(last_commit(CAND_SEAL), last_commit(CAND_AUDIT)) and ancestor(last_commit(CAND_AUDIT), m["git_commit"])
                                                               and on_remote(CAND_SEAL))
    rt = J(f"{RUN}/acoustic_runtime.json")
    rc, forb = runtime_checks(rt, m, "acoustic")
    checks.update(rc)
    crt = J(f"{RUN}/candidates_runtime.json")
    allopen = rt.get("opened_paths", []) + crt.get("opened_paths", [])
    checks["no_reference_file_opened_by_any_job"] = not [p for p in allopen if any(f in p for f in ("p2rj", "positions.json", "ST_PROMPT_R1_PANEL", "population.json"))]
    checks["runner_never_opened_evaluator"] = not [p for p in allopen if "acoustic_s1_evaluate" in p]
    checks["acoustic_no_detection_no_candidate_regeneration"] = rt["counters"]["auto_detection_queries"] == 0 and rt["counters"]["candidate_regenerations"] == 0 and rt["counters"]["encoder_calls"] == 0
    notes["forbidden_opened"] = forb
    seal = J(OUTPUT_SEAL)
    checks["output_seal_hash"] = canon({k: v for k, v in seal.items() if k != "seal_hash"}) == seal["seal_hash"]
    checks["output_seal_files_unchanged"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items())
    checks["output_seal_committed_on_remote"] = committed(OUTPUT_SEAL) and on_remote(OUTPUT_SEAL)
    run_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / RUN).rglob("*") if p.is_file())
    checks["output_seal_covers_run"] = all(p in seal["files"] for p in run_files)
    checks["seal_precedes_references"] = not (ROOT / ANALYSIS).exists()
    lseal = {(r["utterance_id"], int(r["t"])): r for r in J(LAC_SEAL)["rows"]}
    bad = defaultdict(list)
    n_groups, steps = 0, 0
    stat = defaultdict(int)
    paired = 0
    for i, u in enumerate(plan["runtime"]["utterances"]):
        cd = J(f"{RUN}/candidates/{i:03d}.json")
        ad = J(f"{RUN}/acoustic/{i:03d}.json")
        with np.load(ROOT / RUN / "candidates" / f"{i:03d}.npz") as z:
            carr = {k: z[k] for k in z.files}
        with np.load(ROOT / RUN / "acoustic" / f"{i:03d}.npz") as z:
            aarr = {k: z[k] for k in z.files}
        if any(ahash(aarr[k]) != v["sha256"] for k, v in ad["arrays_index"].items()) or set(aarr) != set(ad["arrays_index"]):
            bad["array_hashes"].append(i)
        x = load_wave(u["audio_path"])
        heard = min(len(x), 480000)
        if rawsha(x) != ad["waveform_sha256"] or ad["candidate_seal_hash"] != seal_c["seal_hash"] or not ad["zero_change_features_identical"]:
            bad["lineage"].append(i)
        want = defaultdict(list)
        for q in cd["queries"]:
            for role in ("target", "off_target"):
                if q["region"][role]["status"] == "OK":
                    want[tuple(q["region"][role]["bounds"])].append([q["t"], q["j"], role])
        if sorted(tuple(map(int, k.split("_"))) for k in ad["mask_groups"]) != sorted(want):
            bad["mask_groups"].append(i)
        for key, gr in ad["mask_groups"].items():
            a, b = gr["bounds"]
            xm = x.copy()
            xm[a:b] = np.float32(0.0)
            n_groups += 1
            steps += 1 + max(t for t, _, _ in gr["members"])
            if (rawsha(xm) != gr["x_mask_sha256"] or not (0 <= a < b <= heard) or gr["changed_samples"] != int(np.count_nonzero(x[a:b]))
                    or sorted(map(list, gr["members"])) != sorted(want[(a, b)])):
                bad["mask_identity"].append((i, key))
        lrow = json.loads((ROOT / u["lac_row"]).read_text())
        with np.load(ROOT / u["lac_logits"]) as z:
            lac = {k: z[k] for k in z.files}
        aq = {q["j"]: q for q in ad["queries"]}
        for q in cd["queries"]:
            j, t = q["j"], q["t"]
            A = aq[j]
            Q = plan["runtime"]["runtime_queries"][j]
            tg, of = q["region"]["target"], q["region"]["off_target"]
            pr = tg["status"] == "OK" and of["status"] == "OK"
            paired += pr
            stat[A["target_status"] + "/" + A["off_target_status"]] += 1
            if (A["target_status"], A["off_target_status"], A["target_bounds"], A["off_target_bounds"], A["paired_available"]) != (tg["status"], of["status"], tg["bounds"], of["bounds"], pr):
                bad["status_copy"].append(j)
            lr, lp_ = lseal[(u["utterance_id"], t)], lrow["positions"][str(t)]
            lac_ok = (lr["input_ids"] == CB + u["content_prefix_tokens"][:t] and lr["query"] == lp_["query"] == Q["absolute_query"]
                      and rawsha(bf16(carr[f"q{j:03d}_M_raw"])) == lr["unmasked_logit_vector_sha256"] and lp_["x_sha256"] == rawsha(x)
                      and lp_["fed_equals_input_ids"] and lp_["query_matches"] and lp_["inside_positive_zero"] and lp_["outside_left_equal"] and lp_["outside_right_equal"])
            if lac_ok != A["lac"]["exact_reuse_ok"] or not lac_ok:
                bad["lac_reuse"].append(j)
            _, lac_lp = processed_logp(bf16(lac[f"t{t}_masked"]), t, sup, beg)
            ml = {}
            for role in ("target", "off_target"):
                if q["region"][role]["status"] == "OK":
                    _, l = processed_logp(bf16(aarr[f"q{j:03d}_{role}_raw"]), t, sup, beg)
                    if not np.array_equal(l, aarr[f"q{j:03d}_{role}_logp"]):
                        bad["masked_logp"].append((j, role))
                    mr = A["masked"][role]
                    if not (mr["fed_ok"] and mr["query"] == Q["absolute_query"] and mr["raw_finite"]) or mr["group"] != "_".join(map(str, q["region"][role]["bounds"])):
                        bad["masked_lineage"].append((j, role))
                    ml[role] = l
                elif f"q{j:03d}_{role}_raw" in aarr:
                    bad["unexpected_mask"].append((j, role))
            for k in map(str, cfg["candidates"]["budgets"]):
                ids = q["unions"][k]["ids"]
                K = A["K"][k]
                if K["ids"] != ids or seal_c["members"][f"q{j:03d}"][k] != "sha256:" + hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest():
                    bad["union_immutability"].append((j, k))
                l0 = carr[f"q{j:03d}_M_logp"][ids].astype(np.float64)
                src = cd["auto_alias"] or "AUTO"
                sc = {"M_ORIGINAL": l0, "E_ORIGINAL": carr[f"q{j:03d}_E_logp"][ids].astype(np.float64),
                      "AUTO_ORIGINAL": carr[f"q{j:03d}_{src}_logp"][ids].astype(np.float64),
                      "LAC_UNION": 2 * l0 - lac_lp[ids].astype(np.float64)}
                if pr:
                    lt = ml["target"][ids].astype(np.float64)
                    sc["EN_REGION"] = 2 * l0 - lt
                    sc["OFF_TARGET"] = 2 * l0 - ml["off_target"][ids].astype(np.float64)
                    seed = int(hashlib.sha256(f"s1-shuffle-v1|{u['utterance_id']}|{t}|{k}".encode()).hexdigest()[:16], 16)
                    sc["SHUFFLED_SUPPORT"] = l0 + (l0 - lt)[np.random.Generator(np.random.PCG64(seed)).permutation(len(ids))]
                else:
                    sc["EN_REGION"] = sc["OFF_TARGET"] = sc["SHUFFLED_SUPPORT"] = l0
                sc["NO_CONTRAST"] = l0
                for s_ in SCORERS:
                    if not np.array_equal(np.asarray(K["scores"][s_]), sc[s_]) or K["rankings"][s_] != my_rank(ids, sc[s_]) or K["top1"][s_] != my_rank(ids, sc[s_])[0]:
                        bad["scores_rankings"].append((j, k, s_))
                if K["fallback_to_M"] == pr or not all(np.isfinite(v).all() for v in sc.values()):
                    bad["fallback_finite"].append((j, k))
    checks.update({f"independent:{k}": not bad.get(k) for k in ("array_hashes", "lineage", "mask_groups", "mask_identity", "status_copy", "lac_reuse",
                                                                 "masked_logp", "masked_lineage", "unexpected_mask", "union_immutability", "scores_rankings",
                                                                 "fallback_finite")})
    c = rt["counters"]
    checks["counters_masked_encoders"] = c["masked_encoder_calls"] == n_groups <= cfg["compute"]["masked_encoders_max"]
    checks["counters_masked_steps"] = c["masked_decoder_steps"] == steps <= cfg["compute"]["masked_cached_steps_max"]
    checks["counters_lac_reused_180"] = c["lac_reused"] == 180
    prim = J(PRIMARY)
    checks["primary_analysis_counts"] = (prim["output_seal_hash"] == seal["seal_hash"] and prim["paired_available"] == paired
                                         and prim["completeness"]["candidate_queries"] == 180 == prim["completeness"]["acoustic_queries"])
    checks["primary_analysis_reference_free"] = not any(f in json.dumps(prim) for f in ("target_ids", "EN-confusion", "stratum", "hits"))
    notes["bad"] = {k: v[:20] for k, v in bad.items()}
    notes["status_mix"] = dict(stat)
    notes["mask_groups"], notes["masked_steps"], notes["paired"] = n_groups, steps, paired
    return verdict("primary", checks, notes, "S1_AUDIT: PASS (PRIMARY)", "S1_AUDIT: FAIL (PRIMARY)")


# ---- full -------------------------------------------------------------------------------------------------------------

def cmd_full(args) -> dict:
    cfg = J(CONFIG)
    checks, notes = {}, {}
    an = J(ANALYSIS)
    checks["analysis_hash"] = canon({k: v for k, v in an.items() if k != "analysis_hash"}) == an["analysis_hash"]
    checks["evaluation_after_primary_audit_pass"] = (J(PRIMARY_AUDIT)["verdict"] == "S1_AUDIT: PASS (PRIMARY)" and committed(PRIMARY_AUDIT)
                                                     and ancestor(last_commit(OUTPUT_SEAL), last_commit(PRIMARY_AUDIT)) and on_remote(PRIMARY_AUDIT)
                                                     and (not git("ls-files", ANALYSIS) or ancestor(last_commit(PRIMARY_AUDIT), last_commit(ANALYSIS))))
    seal = J(OUTPUT_SEAL)
    checks["sealed_outputs_unchanged"] = all(fhash(ROOT / p) == h for p, h in seal["files"].items()) and an["output_seal_hash"] == seal["seal_hash"]
    pos = J("results/inference_cf/p2rj/positions.json")["positions"]
    plan = J(PLAN)
    Q = plan["runtime"]["runtime_queries"]
    checks["positions_order"] = [(p["utterance_id"], int(p["t"])) for p in pos] == [(q["utterance_id"], q["t"]) for q in Q]
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    latin = set(tokenizer_partition(WhisperProcessor.from_pretrained(cfg["model"]["dir"], local_files_only=True).tokenizer)["embedded_ids"])
    Y = [set(int(x) for x in p["target_ids"]) - {EOS} for p in pos]
    Y = [{y for y in s if y < EOS} for s in Y]
    S = [p["stratum"] for p in pos]
    D = [p["dialogue_id"] for p in pos]
    cand = {}
    aco = {}
    for i in range(80):
        for q in J(f"{RUN}/candidates/{i:03d}.json")["queries"]:
            cand[q["j"]] = q
        for q in J(f"{RUN}/acoustic/{i:03d}.json")["queries"]:
            aco[q["j"]] = q
    R = []
    for j in range(180):
        r = {"d": D[j], "s": S[j], "paired": aco[j]["paired_available"], "lac": aco[j]["lac"]["exact_reuse_ok"], "K": {}}
        for k in ("5", "20"):
            ids = cand[j]["unions"][k]["ids"]
            sc = aco[j]["K"][k]["scores"]
            rk = {s_: my_rank(ids, sc[s_]) for s_ in SCORERS}
            tops = {b: cand[j]["branches"][b]["top"][k] for b in ("M", "E", "AUTO")}
            acc = {b: bool(set(v) & Y[j]) for b, v in tops.items()} | {"UNION": bool(set(ids) & Y[j])}
            rr = {}
            for s_ in SCORERS:
                ps = [n + 1 for n, tkn in enumerate(rk[s_]) if tkn in Y[j]]
                rr[s_] = 1.0 / min(ps) if ps else 0.0
            r["K"][k] = {"acc": acc, "hit": {s_: rk[s_][0] in Y[j] for s_ in SCORERS}, "rr": rr, "top": {s_: rk[s_][0] for s_ in SCORERS}}
            r["K"][k]["P"] = S[j] == "EN-confusion" and acc["UNION"] and r["paired"] and r["lac"]
        R.append(r)
    conf = [r for r in R if r["s"] == "EN-confusion"]
    nd = lambda rs: len({r["d"] for r in rs})
    u20 = [r for r in conf if r["K"]["20"]["acc"]["UNION"]]
    inc = [r for r in conf if r["K"]["20"]["acc"]["UNION"] and not r["K"]["20"]["acc"]["M"]]
    H = cfg["gate_H"]
    prim = J(PRIMARY)
    complete = prim["completeness"]["candidate_queries"] == 180 and prim["completeness"]["all_candidate_outputs_valid"] and prim["completeness"]["M_bitwise_hist_none_180"] == 180
    hpass = complete and len(u20) >= H["union20_reference_hits_min"] and nd(u20) >= H["union20_hit_dialogues_min"] and len(inc) >= H["incremental_vs_M20_min"] and nd(inc) >= H["incremental_dialogues_min"]
    checks["gate_H_matches"] = an["gate_H"]["PASS"] == hpass and an["S1_A"]["20"]["recall_pooled_of_60"]["UNION"] == len(u20) and an["S1_A"]["20"]["union_minus_M"] == len(inc)
    for k in ("5", "20"):
        for b in ("M", "E", "AUTO", "UNION"):
            if an["S1_A"][k]["recall_pooled_of_60"][b] != sum(r["K"][k]["acc"][b] for r in conf):
                checks[f"recall_{k}_{b}"] = False
    # Gate D (independent)
    G = cfg["gate_D"]
    keys = sorted(set(D))
    idx = np.random.default_rng(cfg["bootstrap"]["seed"]).integers(0, len(keys), size=(cfg["bootstrap"]["draws"], len(keys)))
    cnt = np.stack([(idx == n).sum(axis=1) for n in range(len(keys))], axis=1).astype(np.float64)

    def dmac(vals):
        by = defaultdict(list)
        for d, v in vals:
            by[d].append(v)
        return {d: sum(v) / len(v) for d, v in by.items()}

    def bt(vals, lo, hi):
        dm = dmac(vals)
        mv = np.array([dm.get(kk, 0.0) for kk in keys])
        pres = np.array([kk in dm for kk in keys], dtype=np.float64)
        den = cnt @ pres
        num = cnt @ (mv * pres)
        okk = den > 0
        st = num[okk] / den[okk]
        return (float(np.mean(list(dm.values()))) if dm else None), int(okk.sum()), (float(np.quantile(st, lo)) if st.size else None), (float(np.quantile(st, hi)) if st.size else None)
    P = [r for r in conf if r["K"]["20"]["P"]]
    gd = {"P20_rows": len(P) >= G["paired_accessible_rows_min"], "P20_dialogues": nd(P) >= G["paired_dialogues_min"],
          "P20_fraction": bool(u20) and len(P) / len(u20) >= G["paired_fraction_of_accessible_min"]}
    vsd = {}
    for c in CONTROLS:
        new = [r for r in P if r["K"]["20"]["hit"]["EN_REGION"] and not r["K"]["20"]["hit"][c]]
        lost = [r for r in P if r["K"]["20"]["hit"][c] and not r["K"]["20"]["hit"]["EN_REGION"]]
        hm = bt([(r["d"], float(r["K"]["20"]["hit"]["EN_REGION"]) - float(r["K"]["20"]["hit"][c])) for r in P], .025, .975)[0]
        mm, fd, lo, _ = bt([(r["d"], r["K"]["20"]["rr"]["EN_REGION"] - r["K"]["20"]["rr"][c]) for r in P], cfg["bootstrap"]["lower_quantile"], cfg["bootstrap"]["upper_quantile"])
        vsd[c] = {"net": len(new) - len(lost), "new_dialogues": nd(new), "hit_macro": hm, "mrr_macro": mm, "finite": fd, "lower": lo}
        a_ = an["S1_B"]["20"]["vs_controls"][c]
        if a_["net_hits"] != vsd[c]["net"] or a_["mrr_adjusted"]["finite_draws"] != fd or (lo is not None and abs(a_["mrr_adjusted"]["lower"] - lo) > 1e-9) \
                or (mm is not None and abs(a_["mrr_macro_gain"] - mm) > 1e-12):
            checks[f"gate_D_stat_{c}"] = False
    gd["net_vs_M"] = vsd["M_ORIGINAL"]["net"] >= G["net_hit1_gain_vs_M_min"]
    gd["new_dialogues"] = vsd["M_ORIGINAL"]["new_dialogues"] >= G["correction_dialogues_min"]
    gd["hit_macro"] = vsd["M_ORIGINAL"]["hit_macro"] is not None and vsd["M_ORIGINAL"]["hit_macro"] >= G["dialogue_macro_hit1_gain_vs_M_min"]
    for c in ("OFF_TARGET", "LAC_UNION", "SHUFFLED_SUPPORT"):
        gd[f"net_{c}"] = vsd[c]["net"] >= G["net_hit1_gain_vs_each_control_min"]
    for c, th in G["mrr_macro_gain_min"].items():
        gd[f"mrr_{c}"] = vsd[c]["mrr_macro"] is not None and vsd[c]["mrr_macro"] >= th
    corr = {s: sum(r["K"]["20"]["hit"]["M_ORIGINAL"] and not r["K"]["20"]["hit"]["EN_REGION"] for r in R if r["s"] == s) for s in ("EN-correct", "ZH-correct")}
    lat = sum(r["K"]["20"]["top"]["EN_REGION"] != r["K"]["20"]["top"]["M_ORIGINAL"] and not r["K"]["20"]["hit"]["EN_REGION"]
              and r["K"]["20"]["top"]["EN_REGION"] in latin for r in R if r["s"] == "ZH-correct")
    gd["prot"] = (corr["EN-correct"] <= G["new_EN_correct_corruptions_max"] and corr["ZH-correct"] <= G["new_ZH_correct_corruptions_max"]
                  and sum(corr.values()) <= G["total_correct_state_corruptions_max"] and lat <= G["new_wrong_English_promotions_ZH_max"])
    dpass = all(gd.values())
    ipass = all(vsd[c]["finite"] >= cfg["bootstrap"]["minimum_finite_draws"] and vsd[c]["lower"] is not None and vsd[c]["lower"] > 0 for c in CONTROLS)
    checks["gate_D_point_matches"] = an["gate_D"]["POINT_COVERAGE_PROTECTION_PASS"] == dpass
    checks["gate_D_inference_matches"] = an["gate_D_inference"]["PASS"] == ipass
    checks["protection_counts_match"] = (an["protection"]["EN-correct"]["new_corruptions"] == corr["EN-correct"] and an["protection"]["ZH-correct"]["new_corruptions"] == corr["ZH-correct"]
                                         and an["protection"]["ZH-correct"]["new_wrong_latin_promotions"] == lat)
    integ = all(an["integrity"].values())
    label = ("S1_INVALID" if not integ else "S1_CANDIDATE_HEADROOM_INSUFFICIENT" if not hpass else
             "S1_ACOUSTIC_DISCRIMINATION_INSUFFICIENT" if not dpass else "S1_PARTIAL_FEASIBILITY" if not ipass else "S1_READY_FOR_S2")
    checks["label_matches"] = an["label"] == label
    checks["confusion_M_hits_zero"] = sum(r["K"]["20"]["hit"]["M_ORIGINAL"] for r in conf) == 0
    checks["correct_state_M_hits_60"] = all(sum(r["K"]["20"]["hit"]["M_ORIGINAL"] for r in R if r["s"] == s) == 60 for s in ("EN-correct", "ZH-correct"))
    notes.update(independent_label=label, H={"union20": len(u20), "dialogues": nd(u20), "incremental": len(inc), "incremental_dialogues": nd(inc)},
                 P20=len(P), P20_dialogues=nd(P), vs=vsd, corruptions=corr, latin_promotions=lat, gate_D=gd)
    out = verdict("full", checks, notes, "S1_AUDIT: PASS (FULL)", "S1_AUDIT: FAIL (FULL)")
    out["independent_label"] = label
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prerun", "candidates", "primary", "full"))
    a = ap.parse_args()
    out = {"prerun": (cmd_prerun, PRERUN), "candidates": (cmd_candidates, CAND_AUDIT), "primary": (cmd_primary, PRIMARY_AUDIT),
           "full": (cmd_full, FULL_AUDIT)}[a.stage]
    path = ROOT / out[1]
    if path.exists():
        raise FileExistsError(f"{out[1]} exists; failed audits are preserved, write a new numbered attempt")
    doc = out[0](a)
    doc["git_head"] = git("rev-parse", "HEAD")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"verdict": doc["verdict"], "failed": doc["failed"]}, indent=1))


if __name__ == "__main__":
    main()
