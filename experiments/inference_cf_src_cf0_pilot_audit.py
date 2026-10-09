#!/usr/bin/env python
"""SRC-CF0-P independent auditor (pre / construction / primary / full).

Independence: this module never imports the pilot runner, the pilot helper (csasr.inference_cf.src_cf0_pilot), the pilot
evaluator, cf_pilot_contract or s1_evidence. Direction arithmetic, random vectors, geometry, energy checks, joint
reachability, construction gates, cohorts, corrections, LODO, bootstrap, damage stops, terminal precedence and selection are
re-implemented here from the frozen config/spec. Generic hash / git / IO helpers and the S1 auditor's already-independent
exact-Fraction region route (experiments/acoustic_s1_audit.py: ExactAttention / my_target / my_off) are reused. The
primary helper is executed only in a subprocess on SYNTHETIC inputs for an agreement check.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from experiments.acoustic_s1_audit import (ExactAttention, ahash, ancestor, bf16, blob, canon, committed, fhash, git, load_wave,
                                           my_off, my_target, on_remote, rawsha)

FREEZE = "2d235742d5fe59915eb8cc3ba6ad8511f0fa1c9c"
CONFIG = "configs/inference_cf/src_cf0_pilot.json"
CONFIG_SHA = "sha256:f1881050b0718559d05a3ddaa0cb2832953e0fc7553be4c360d952d57f51aa87"
PANEL = "docs/inference_cf/SRC_CF0_PANEL.json"
FROZEN = (CONFIG, PANEL, "docs/inference_cf/SRC_CF0_PILOT_SPEC.md", "docs/inference_cf/SRC_CF0_PILOT_DESIGN.md",
          "docs/inference_cf/SRC_CF0_PILOT_FIREWALL.md", "docs/inference_cf/SRC_CF0_CLAUDE_HANDOFF.md",
          "docs/inference_cf/SRC_CF0_FULL300_SAFETY_PLAN.md", "tests/test_src_cf0_pilot_contract.py",
          "src/csasr/inference_cf/cf_pilot_contract.py")
RUN = "results/inference_cf/src_cf0_pilot/run1"
PROJ, PRERUN, DSEAL, CAUDIT = f"{RUN}/runtime.json", f"{RUN}/prerun_audit.json", f"{RUN}/direction_seal.json", f"{RUN}/construction_audit.json"
GATE_A, AUTH, PSEAL, PAUDIT, EVAL, FAUDIT = (f"{RUN}/gate_a.json", f"{RUN}/PB_authorization.json", f"{RUN}/pulse_seal.json",
                                             f"{RUN}/primary_audit.json", f"{RUN}/evaluation.json", f"{RUN}/full_audit.json")
SMOKE = f"{RUN}/engineering_smoke/cpu_smoke.json"
RUNNER = "experiments/inference_cf_src_cf0_pilot.py"
HELPER = "src/csasr/inference_cf/src_cf0_pilot.py"
EVALUATOR = "experiments/inference_cf_src_cf0_pilot_evaluate.py"
SELF = "experiments/inference_cf_src_cf0_pilot_audit.py"
MEMBERSHIP = "docs/inference_cf/ST_PROMPT_R1_PANEL.json"
POSITIONS = "results/inference_cf/p2rj/positions.json"
S1_CAND, S1_ACO = "results/inference_cf/s1/run1/candidates", "results/inference_cf/s1/run1/acoustic"
LOC0, R1CAP = "results/inference_cf/st_loc0/run1", "results/inference_cf/st_prompt_r1/runA/capture"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
LAYERS, ETAS, SIGNS, FAMS = (16, 24), (0.15, 0.30), (1, -1), ("target", "off", "random")
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
FORBIDDEN_RUNTIME = ("p2rj", "positions.json", "ST_PROMPT_R1_PANEL", "SRC_CF0_PANEL", "gate_a", "evaluation.json", "full_audit",
                     "oracle", "mms", "transcript", "/roles/", "population.json", "src_cf0_pilot_evaluate", "src_cf0_pilot_audit")
FORBIDDEN_SOURCE = ("positions.json", "p2rj", "target_ids", "competitor", "evaluation_membership", "ST_PROMPT_R1_PANEL", "coverage_review",
                    "GATE_A", "gate_a.json", "oracle_analysis", "mms", "population.json", "inference_cf_src_cf0_pilot_evaluate")


def J(rel):
    return json.loads((ROOT / rel).read_text())


def label_id(layer, eta, sign, fam="target") -> str:
    b = f"L{int(layer)}_eta{float(eta):.2f}_{'plus' if int(sign) > 0 else 'minus'}"
    return b if fam == "target" else f"{fam}_{b}"


def verdict(stage, checks, notes, ok_label, fail_label, extra=None):
    d = {"schema": f"src_cf0_pilot_audit_{stage}_v1", "verdict": ok_label if all(checks.values()) else fail_label,
         "failed": sorted(k for k, v in checks.items() if not v), "checks": checks, "notes": notes, "created_unix": time.time()}
    d.update(extra or {})
    return d


def seal_ok(rel) -> tuple[bool, dict]:
    s = J(rel)
    ok = committed(rel) and on_remote(rel) and canon({k: v for k, v in s.items() if k != "seal_hash"}) == s["seal_hash"]
    ok &= all(fhash(ROOT / p) == h for p, h in s["files"].items())
    return bool(ok), s


def manifest_ok(stage: str):
    """Manifest hash valid, every pinned source unchanged (this auditor's own file exempt: documented auditor-rule commits
    only; it is never executed by the runner) and the manifest commit is on the remote."""
    m = J(f"{RUN}/manifest_{stage}.json")
    ok = canon({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"]
    ok &= all(fhash(ROOT / p) == h for p, h in m["sources"].items() if p != SELF)
    ok &= all(fhash(ROOT / p) == h for p, h in m["not_executed_by_runner"].items() if p != SELF)
    ok &= ancestor(m["git_commit"], "origin/cs-asr-steer-inf") and committed(f"{RUN}/manifest_{stage}.json")
    return m, bool(ok)


def runtime_firewall(st: dict, m: dict, stage: str) -> tuple[dict, list]:
    pinned = {str(ROOT / p) for p in m["sources"]}
    forb = [p for p in st.get("opened_paths", []) if p not in pinned and any(f.lower() in p.lower() for f in FORBIDDEN_RUNTIME)]
    forb += [p for p in st.get("opened_paths", []) if "__pycache__" in p and ("src_cf0_pilot_evaluate" in p or "src_cf0_pilot_audit" in p)]
    forb += [p for p in st.get("opened_paths", []) if p.endswith(PANEL) or p.endswith(MEMBERSHIP) or p.endswith(POSITIONS)]
    c = {f"{stage}_completed_same_manifest": st["status"] == "completed" and st["manifest_hash"] == m["manifest_hash"],
         f"{stage}_frozen_model": st["model_grads_none"] and not st["requires_grad_any"] and not st["training_mode"] and st["weights_unchanged"]
         and st["weights_before"] == st["weights_after"],
         f"{stage}_no_hooks_left": st["top_forward_hooks"] == 0,
         f"{stage}_no_autograd_optimizer_lid": all(st["counters"].get(k, 0) == 0 for k in ("autograd_calls", "optimizer_steps", "lid_calls")),
         f"{stage}_file_open_firewall": "opened_paths" in st and not forb, f"{stage}_wall_under_3h": st["elapsed_sec"] < 10800,
         f"{stage}_no_failures": st["n_failures"] == 0}
    return c, forb


# ---- independent numerical primitives ----------------------------------------------------------------------------------

def my_direction(h32: np.ndarray, m32: np.ndarray) -> dict:
    h, m = h32.astype(np.float64), m32.astype(np.float64)
    if h.shape != (1280,) or not (np.isfinite(h).all() and np.isfinite(m).all()) or math.sqrt(float(h @ h)) < 1e-8:
        return {"status": "INVALID"}
    d = h - m
    dn = float(np.linalg.norm(d))
    if dn < 1e-4:
        return {"status": "ABSTAIN_DEGENERATE", "raw_norm": dn}
    v = (d / (dn + 1e-6)).astype(np.float32)
    w = v.astype(np.float64)
    hh = float(h @ h)
    wn = math.sqrt(float(w @ w))
    tp = w - h * (float(h @ w) / hh)
    rp = d - h * (float(h @ d) / hh)
    return {"status": "OK", "raw_norm": dn, "vector": v, "direction_norm": wn, "tangent_ratio": math.sqrt(float(tp @ tp)) / wn,
            "raw_tangent_norm": math.sqrt(float(rp @ rp)), "cos_v_h": float(h @ w) / (wn * math.sqrt(hh))}


def my_random(uid, t, layer) -> np.ndarray:
    seed = int(hashlib.sha256(f"SRC_CF0_P-random-v1|240924|{uid}|{t}|{layer}".encode()).hexdigest()[:16], 16)
    z = np.random.Generator(np.random.PCG64(seed)).standard_normal(1280)
    return (z / np.linalg.norm(z)).astype(np.float32)


def close(a, b, rel=1e-9, ab=1e-12) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(ab, rel * max(abs(float(a)), abs(float(b))))


def my_joint(lay: dict, tol: float) -> bool:
    if lay.get("target", {}).get("status") != "OK" or lay.get("off", {}).get("status") != "OK":
        return False
    for f in FAMS:
        if not lay[f]["tangent_ratio"] >= 0.25:
            return False
    for s in SIGNS:
        for e in ETAS:
            es = []
            for f in FAMS:
                c = lay["reach"][f"{f}|{s}|{e:.2f}"]
                if not (c.get("solver_status") == "ok" and c["rel_sq_err"] <= 0.02 and abs(c["emulated_edit_norm"] / c["target"] - 1) <= 0.02
                        and abs(c["emulated_edit_norm"] ** 2 / c["target"] ** 2 - 1) <= 0.02):
                    return False
                es.append(c["emulated_edit_norm"] ** 2)
            if max(es) / min(es) - 1 > tol:
                return False
    return True


# ---- pre ----------------------------------------------------------------------------------------------------------------

SYNTH = r'''
import json, sys, numpy as np, torch
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1]]
from csasr.inference_cf import src_cf0_pilot as H
import experiments.inference_cf_p2r as p2r
rng = np.random.default_rng(int(sys.argv[2]))
cfg = json.load(open(sys.argv[1] + "/configs/inference_cf/src_cf0_pilot.json"))
ec = H.energy_cfg(cfg)
out = []
for case in range(60):
    h = torch.from_numpy(rng.standard_normal(1280).astype(np.float32) * 0.3).to(torch.bfloat16).float().numpy()
    scale = [1e-7, 1e-3, 0.05, 0.5, 2.0][case % 5]
    m = torch.from_numpy((h + scale * rng.standard_normal(1280)).astype(np.float32)).to(torch.bfloat16).float().numpy()
    if case % 11 == 0: m = h.copy()
    d = H.direction(h, m)
    rec = {"status": d["status"], "raw_norm": d["raw_norm"]}
    if d["vector"] is not None:
        rec.update(vector=d["vector"].tolist(), tangent_ratio=d["tangent_ratio"], raw_tangent_norm=d["raw_tangent_norm"], cos_v_h=d["cos_v_h"],
                   direction_norm=d["direction_norm"])
        hn = torch.from_numpy(h).to(torch.bfloat16)
        rec["reach"] = {f"{s}|{e}": H.reach(hn, d["vector"], e, s, ec, p2r.solve_scale) for s in (1, -1) for e in (0.15, 0.30)}
    out.append(rec)
rv = [H.random_direction("U%d" % k, k, 16 + 8 * (k % 2)).tolist() for k in range(5)]
print(json.dumps({"cases": out, "random": rv}))
'''


def synthetic_agreement(seed: int = 240924) -> tuple[bool, dict]:
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}")
    res = subprocess.run([sys.executable, "-c", SYNTH, str(ROOT), str(seed)], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        return False, {"stderr": res.stderr[-2000:]}
    prim = json.loads(res.stdout)
    import torch
    rng = np.random.default_rng(seed)
    bad, reach_checked = [], 0
    for case, p in enumerate(prim["cases"]):
        h = torch.from_numpy(rng.standard_normal(1280).astype(np.float32) * 0.3).to(torch.bfloat16).float().numpy()
        scale = [1e-7, 1e-3, 0.05, 0.5, 2.0][case % 5]
        m = torch.from_numpy((h + scale * rng.standard_normal(1280)).astype(np.float32)).to(torch.bfloat16).float().numpy()
        if case % 11 == 0:
            m = h.copy()
        d = my_direction(h, m)
        ok = d["status"] == p["status"] and close(d["raw_norm"], p["raw_norm"])
        if d["status"] == "OK":
            ok &= np.array_equal(np.asarray(p["vector"], dtype=np.float32), d["vector"]) and all(close(d[k], p[k]) for k in ("tangent_ratio", "raw_tangent_norm", "cos_v_h", "direction_norm"))
            for k, r in p["reach"].items():
                s, e = k.split("|")
                rn = math.sqrt(float(h.astype(np.float64) @ h.astype(np.float64)))
                ok &= close(r["target"], float(e) * rn) and close(r["r_norm"], rn)
                if r.get("solver_status") == "ok":
                    reach_checked += 1
                    vh = int(s) * d["vector"].astype(np.float64)
                    vh /= np.linalg.norm(vh)
                    x = h.astype(np.float64) + r["s"] * vh
                    x = x / (np.linalg.norm(x) + 1e-6) * rn
                    ok &= abs(np.linalg.norm(x - h) - r["emulated_edit_norm"]) <= 0.01 * rn
                    ok &= r["reachable"] == (r["rel_sq_err"] <= 0.02 and abs(r["emulated_edit_norm"] / r["target"] - 1) <= 0.02)
        if not ok:
            bad.append(case)
    rv_ok = all(np.array_equal(np.asarray(prim["random"][k], dtype=np.float32), my_random("U%d" % k, k, 16 + 8 * (k % 2))) for k in range(5))
    return not bad and rv_ok, {"cases": len(prim["cases"]), "disagreements": bad, "random_vectors_agree": rv_ok, "reach_cells_checked": reach_checked,
                               "statuses": sorted({p["status"] for p in prim["cases"]})}


def segment(src: str, name: str) -> str:
    i = src.index(f"def {name}(") if f"def {name}(" in src else src.index(f"class {name}")
    rest = src[i:]
    nxt = re.search(r"\n(def |class )", rest[5:])
    return rest[:nxt.start() + 5] if nxt else rest


def static_firewall() -> dict:
    rs, hs, ev, me = ((ROOT / p).read_text() for p in (RUNNER, HELPER, EVALUATOR, SELF))
    code = lambda s: "\n".join(l for l in s.splitlines() if not l.strip().startswith("#"))
    gpu = "".join(segment(rs, n) for n in ("load_run", "Engine", "capture_utterance", "directions_for_query", "cmd_construct", "cmd_seal_a",
                                           "check_authorization", "pulse_utterance", "cmd_pulse", "cmd_seal_b", "finish", "encode", "feed"))
    construct = "".join(segment(rs, n) for n in ("capture_utterance", "directions_for_query", "cmd_construct"))
    helper_code = re.sub(r"FORBIDDEN_TEXT = \(.*?\)\n", "", code(hs.split('"""', 2)[2]), flags=re.S)
    c = {"runner_gpu_phases_no_reference_identifiers": not [f for f in FORBIDDEN_SOURCE + ("PANEL", "stratum") if f in code(gpu)],
         "helper_no_reference_identifiers": not [f for f in FORBIDDEN_SOURCE if f in helper_code],
         "construct_phase_no_steering": not any(x in construct for x in ("pulse_action", "relative_action", "InterventionHook", "cross_pulse_hook",
                                                                         "eng.pulse(", "apply_steering")),
         "no_training_or_gradients": not any(x in rs + hs for x in (".backward(", "optim.", "requires_grad_(True", "enable_grad", "set_grad_enabled(True")),
         "runner_does_not_import_evaluator_or_auditor": re.search(r"^\s*(from|import)\s+\S*src_cf0_pilot_(evaluate|audit)", rs, re.M) is None
         and "import_module" not in rs and "__import__" not in rs,
         "runner_records_open_log": "addaudithook" in rs and "opened_paths" in rs,
         "panel_opened_only_by_prepare": all("PANEL" not in segment(rs, n) for n in ("cmd_construct", "cmd_pulse", "load_run", "capture_utterance", "pulse_utterance")),
         "evaluator_gate_a_order": ev.index("seal = verify_seal(DSEAL)") < ev.index('J(CAUDIT)["verdict"]') < ev.index("J(MEMBERSHIP)"),
         "evaluator_b_order": ev.index("pseal = verify_seal(PSEAL)") < ev.index('J(PAUDIT)["verdict"]') < ev.index("J(POSITIONS)"),
         "evaluator_gate_a_no_token_sets": "POSITIONS" not in segment(ev, "gate_a") and "target_ids" not in segment(ev, "gate_a"),
         "auditor_independent": re.search(r"^\s*(from|import)\s+\S*(src_cf0_pilot\b|cf_pilot_contract|s1_evidence|inference_cf_src_cf0_pilot\b|"
                                          r"inference_cf_src_cf0_pilot_evaluate)", me, re.M) is None}
    return c


def cmd_pre(args) -> dict:
    cfg = J(CONFIG)
    ck, notes = {}, {}
    ck["frozen_unchanged_since_freeze"] = all(blob(FREEZE, p) == fhash(ROOT / p) for p in FROZEN)
    ck["config_sha_handoff"] = fhash(ROOT / CONFIG) == CONFIG_SHA
    ck["pins"] = all(fhash(ROOT / p) == h for p, h in {**cfg["pins"], **cfg["historical_inputs"]}.items())
    ck["model_files"] = all(fhash(Path(cfg["model"]["dir"]) / n) == h for n, h in cfg["model"]["files"].items())
    P = J(PANEL)
    ck["panel_identity"] = canon({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel"]["identity_hash"]
    rt = J(PROJ)
    ck["runtime_committed_hash"] = committed(PROJ) and canon({k: v for k, v in rt.items() if k != "runtime_hash"}) == rt["runtime_hash"] \
        and canon(rt["projection"]) == rt["projection_hash"] and all(rt["checks"].values())
    pr = rt["projection"]
    fw = cfg["firewall"]
    tmax = defaultdict(int)
    for q in P["runtime_queries"]:
        tmax[q["utterance_id"]] = max(tmax[q["utterance_id"]], q["t"])
    expect_q = [{k: q[k] for k in fw["runtime_query_keys"]} for q in P["runtime_queries"]]
    expect_u = [{**{k: u[k] for k in fw["runtime_utterance_keys"] if k not in ("content_prefix_tokens", "max_t")},
                 "content_prefix_tokens": u["baseline_content_tokens"][:tmax[u["utterance_id"]]], "max_t": tmax[u["utterance_id"]]} for u in P["utterances"]]
    expect_r = [{k: r[k] for k in fw["region_keys"]} for r in P["sealed_region_inventory"]]
    ck["projection_equals_independent_allowlist"] = pr == {"runtime_queries": expect_q, "utterances": expect_u, "regions": expect_r}
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(pr)
    txt = json.dumps({k: v for k, v in rt.items() if k not in ("checks", "references_used")}).lower()
    ck["projection_no_forbidden_keys_or_text"] = not (keys & set(fw["forbidden"])) and not any(
        w.lower() in txt for w in ("stratum", "EN-confusion", "EN-correct", "ZH-correct", "target_ids", "competitor", "coverage_review", "acceptable"))
    ck["projection_counts"] = (len(pr["runtime_queries"]), len(pr["utterances"]), len({u["dialogue_id"] for u in pr["utterances"]})) == (180, 80, 20)
    # independent region re-derivation (exact Fraction route) from sealed S1 heads + R0 heard intervals + waveform
    rcfg = {"regions": cfg["region_policy"]}
    uidx = {u["utterance_id"]: i for i, u in enumerate(pr["utterances"])}
    pu = {u["utterance_id"]: u for u in P["utterances"]}
    waves = {u["utterance_id"]: load_wave(u["audio_path"]) for u in pr["utterances"]}
    ck["waveform_bytes"] = all(rawsha(waves[u]) == h for u, h in rt["artifacts"]["waveform_sha256"].items())
    bad = []
    for j, (q, r) in enumerate(zip(pr["runtime_queries"], pr["regions"])):
        uid = q["utterance_id"]
        with np.load(ROOT / pu[uid]["r0_primary_arrays"]) as z:
            ivs = [tuple(int(v) for v in row) for row in z["track_heard_intervals"]]
        with np.load(ROOT / f"{S1_CAND}/{uidx[uid]:03d}.npz") as z:
            heads = bf16(z[f"q{j:03d}_M_heads"])
        x = waves[uid]
        heard = min(len(x), 480000)
        att = ExactAttention(heads, heard)
        t_ = my_target(att, ivs, x, heard, rcfg)
        o_ = my_off(t_, att, ivs, x, heard)
        if not (t_["status"] == r["target"]["status"] and t_.get("bounds") == r["target"]["bounds"] and o_["status"] == r["off_target"]["status"]
                and o_.get("bounds") == r["off_target"]["bounds"] and (t_["status"] == o_["status"] == "OK") == r["paired_available"]):
            bad.append(j)
    ck["regions_independently_rederived"] = not bad
    notes["region_disagreements"] = bad
    tg = sum(r["target"]["status"] == "OK" for r in pr["regions"])
    pa = sum(r["paired_available"] for r in pr["regions"])
    ck["regions_73_59"] = (tg, pa) == (73, 59)
    mk = {}
    for j, (q, r) in enumerate(zip(pr["runtime_queries"], pr["regions"])):
        for role in ("target", "off_target"):
            if r[role]["status"] == "OK":
                a, b = r[role]["bounds"]
                y = waves[q["utterance_id"]].copy()
                y[a:b] = 0.0
                mk[f"{q['utterance_id']}|{a}|{b}"] = rawsha(y)
    ck["mask_bytes_independent"] = mk == {k: v["sha256"] for k, v in rt["artifacts"]["masks"].items()} and len(mk) == 124
    steps = sum(1 + u["max_t"] for u in pr["utterances"])
    msteps = sum(1 + max(m_[0] for m_ in v["members"]) for v in rt["artifacts"]["masks"].values())
    ck["construct_bound_exact"] = 2 * (80 + len(mk)) == cfg["compute"]["construction_forward_bound"]["encoder_calls"] and \
        2 * (steps + msteps) == cfg["compute"]["construction_forward_bound"]["cached_decoder_steps"]
    m, mok = manifest_ok("construct")
    ck["manifest_construct"] = mok and m["runtime_hash"] == rt["runtime_hash"] and m["config_sha256"] == CONFIG_SHA and \
        set(cfg["pins"]) <= set(m["sources"]) and PANEL not in m["sources"]
    ck.update({f"static:{k}": v for k, v in static_firewall().items()})
    ok, n = synthetic_agreement()
    ck["synthetic_helper_vs_independent"] = ok
    notes["synthetic"] = n
    sm = J(SMOKE)
    c = sm["cells"]
    ck["engineering_smoke_committed_clean"] = (committed(SMOKE) and sm["synthetic_audio"] and not sm["panel_audio_used"] and not sm["capture_fails"]
                                               and not sm["pulse_fails"] and sm["repeat_bitwise"] and all(c["checks_all"].values())
                                               and not c["integrity_failures"] and c["predicted_vs_actual_mismatch"] == 0
                                               and all(v["none"] and all(v["zero"].values()) and v["restore"] for v in sm["none_zero_restore"].values()))
    notes["smoke_cells"] = c
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/src:{ROOT}")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests/test_src_cf0_pilot_impl.py", "tests/test_src_cf0_pilot_contract.py",
                            "tests/test_src_cf0_design_block.py"], cwd=ROOT, capture_output=True, text=True, env=env)
    ck["pilot_tests_pass"] = tests.returncode == 0
    notes["tests_tail"] = tests.stdout.strip().splitlines()[-1:] if tests.stdout else tests.stderr[-500:]
    return verdict("pre", ck, notes, "PASS_TO_SRC_CF0_PILOT", "FAIL_PRERUN", {"manifest_hash": m["manifest_hash"], "runtime_hash": rt["runtime_hash"]})


# ---- construction -------------------------------------------------------------------------------------------------------

def cmd_construction(args) -> dict:
    cfg = J(CONFIG)
    ck, notes = {}, {}
    sok, seal = seal_ok(DSEAL)
    ck["direction_seal_pushed_unchanged"] = sok and ancestor(seal["source_commit"], "origin/cs-asr-steer-inf")
    m, mok = manifest_ok("construct")
    ck["manifest"] = mok and seal["manifest_hash"] == m["manifest_hash"]
    pre = J(PRERUN)
    ck["prerun_pass_same_manifest"] = pre["verdict"] == "PASS_TO_SRC_CF0_PILOT" and pre["manifest_hash"] == m["manifest_hash"] and committed(PRERUN)
    st = J(f"{RUN}/construct_status.json")
    rc, forb = runtime_firewall(st, m, "construct")
    ck.update(rc)
    notes["forbidden_opened"] = forb
    c = st["counters"]
    ck["counters_exact"] = (c["encoder_calls"] == 160 and c["masked_encoder_calls"] == 248 and c["decoder_steps"] == 13318
                            and c["pulse_forwards"] == 0 and c["steering_hooks"] == 0)
    rt = J(PROJ)
    pr = rt["projection"]
    Q, R = pr["runtime_queries"], pr["regions"]
    with np.load(ROOT / f"{LOC0}/calibration/baseline_logits.npz") as z:
        hn = {k: z[k] for k in z.files}
    hs = {}
    for l in LAYERS:
        with np.load(ROOT / f"{LOC0}/calibration/states_L{l}_CROSS.npz") as z:
            hs[("loc0", l)] = z["H_B"]
        with np.load(ROOT / f"{R1CAP}/states_L{l}.npz") as z:
            hs[("r1", l)] = z["H_M"]
    tol = cfg["energy"]["paired_relative_squared_energy_difference_max"]
    fails = defaultdict(list)
    seen = set()
    for i in range(80):
        d = J(f"{RUN}/construction/{i:03d}.json")
        if fhash(ROOT / f"{RUN}/construction/{i:03d}.npz") != d["arrays_file_sha256"] or d["manifest_hash"] != m["manifest_hash"]:
            fails["row_hash"].append(i)
        with np.load(ROOT / f"{RUN}/construction/{i:03d}.npz") as z:
            A = {k: z[k] for k in z.files}
        if any(ahash(A[k]) != v["sha256"] for k, v in d["arrays_index"].items()) or set(A) != set(d["arrays_index"]):
            fails["array_index"].append(i)
        with np.load(ROOT / f"{S1_CAND}/{i:03d}.npz") as z:
            s1c = {k: z[k] for k in z.files}
        with np.load(ROOT / f"{S1_ACO}/{i:03d}.npz") as z:
            s1a = {k: z[k] for k in z.files}
        # independent repeat identity over every stored sweep-1 / sweep-2 array
        for k, v in A.items():
            if k.startswith("r2__"):
                k1 = k[4:]
                k1 = k1.replace("_raw", "_r1_raw") if k1.endswith("_raw") else k1
                if not np.array_equal(A[k1], v):
                    fails["repeat"].append(k)
        for q in d["queries"]:
            j = q["j"]
            seen.add(j)
            qq = Q[j]
            pre_ = pr["utterances"][i]["content_prefix_tokens"][:qq["t"]]
            if not (qq["utterance_id"] == d["utterance_id"] and q["t"] == qq["t"] and q["absolute_query"] == 4 + qq["t"] - 1):
                fails["identity"].append(j)
            for role, rr in q["inputs"].items():
                if not (rr["fed_sha256"] == canon(CB + pre_) == qq["forced_zh_query_input_sha256"] and rr["query"] == qq["absolute_query"]
                        and rr["raw_finite"] and rr["states_finite"] and all(rr["ffn_input_equals_site"].values())):
                    fails["lineage"].append((j, role))
                if role != "clean" and rr["mask"]["x_mask_sha256"] != rt["artifacts"]["masks"][f"{qq['utterance_id']}|{rr['mask']['bounds'][0]}|{rr['mask']['bounds'][1]}"]["sha256"]:
                    fails["mask"].append((j, role))
            k = f"q{j:03d}"
            hid = [np.array_equal(A[f"{k}_clean_r1_raw"], s1c[f"{k}_M_raw"]), np.array_equal(A[f"{k}_clean_r1_raw"], hn[k]),
                   np.array_equal(A[f"{k}_clean_heads"], s1c[f"{k}_M_heads"])]
            for l in LAYERS:
                hid += [np.array_equal(bf16(A[f"{k}_clean_L{l}"]), hs[("loc0", l)][j]), np.array_equal(bf16(A[f"{k}_clean_L{l}"]), hs[("r1", l)][j])]
            for role in ("target", "off_target"):
                present = f"{k}_{role}_r1_raw" in A
                if present != (R[j][role]["status"] == "OK"):
                    fails["masked_presence"].append((j, role))
                if present:
                    hid.append(np.array_equal(A[f"{k}_{role}_r1_raw"], s1a[f"{k}_{role}_raw"]))
            if not all(hid):
                fails["historical"].append(j)
            for l in LAYERS:
                lay = q["layers"][str(l)]
                h = bf16(A[f"{k}_clean_L{l}"])
                for fam, role in (("target", "target"), ("off", "off_target")):
                    if R[j][role]["status"] != "OK":
                        if not lay[fam]["status"].startswith("NO_") or f"{k}_{fam}_L{l}_v" in A:
                            fails["absent_direction"].append((j, l, fam))
                        continue
                    my = my_direction(h, bf16(A[f"{k}_{role}_L{l}"]))
                    if my["status"] != lay[fam]["status"] or not close(my.get("raw_norm"), lay[fam]["raw_norm"]):
                        fails["direction_status"].append((j, l, fam))
                        continue
                    if my["status"] == "OK":
                        if not (np.array_equal(A[f"{k}_{fam}_L{l}_v"], my["vector"]) and rawsha(my["vector"]) == lay[fam]["vector_sha256"]
                                and all(close(my[x], lay[fam][x]) for x in ("direction_norm", "tangent_ratio", "raw_tangent_norm", "cos_v_h"))):
                            fails["direction_values"].append((j, l, fam))
                rv = my_random(qq["utterance_id"], qq["t"], l)
                if not (np.array_equal(A[f"{k}_random_L{l}_v"], rv) and rawsha(rv) == lay["random"]["vector_sha256"]):
                    fails["random"].append((j, l))
                if lay["pair"] is not None:
                    a_, b_ = A[f"{k}_target_L{l}_v"].astype(np.float64), A[f"{k}_off_L{l}_v"].astype(np.float64)
                    cs = float(a_ @ b_) / (np.linalg.norm(a_) * np.linalg.norm(b_))
                    ratio = lay["target"]["raw_tangent_norm"] / lay["off"]["raw_tangent_norm"] if lay["off"]["raw_tangent_norm"] > 0 else None
                    if not (close(cs, lay["pair"]["cos"]) and (ratio is None) == (lay["pair"]["raw_tangent_ratio"] is None)
                            and (ratio is None or close(ratio, lay["pair"]["raw_tangent_ratio"]))):
                        fails["pair"].append((j, l))
                rn = math.sqrt(float(h.astype(np.float64) @ h.astype(np.float64)))
                for key, cell in lay["reach"].items():
                    fam, s, e = key.split("|")
                    v = A.get(f"{k}_{fam}_L{l}_v") if fam != "random" else (A[f"{k}_random_L{l}_v"] if f"{k}_target_L{l}_v" in A else None)
                    if v is None:
                        if cell.get("reachable") or cell.get("status") != "no_direction":
                            fails["reach_absent"].append((j, l, key))
                        continue
                    if not (close(cell["r_norm"], rn) and close(cell["target"], float(e) * rn)):
                        fails["reach_target"].append((j, l, key))
                    if cell.get("solver_status") == "ok":
                        w = int(s) * v.astype(np.float64)
                        w /= np.linalg.norm(w)
                        if not close(cell["phi"], math.acos(max(-1.0, min(1.0, float(h.astype(np.float64) @ w) / rn))), rel=1e-9, ab=1e-9):
                            fails["reach_phi"].append((j, l, key))
                        x = h.astype(np.float64) + cell["s"] * w
                        x = x / (np.linalg.norm(x) + 1e-6) * rn
                        if abs(np.linalg.norm(x - h) - cell["emulated_edit_norm"]) > 0.01 * rn:
                            fails["reach_emulation_plausibility"].append((j, l, key))
                        rr_ = (cell["rel_sq_err"] <= 0.02 and abs(cell["emulated_edit_norm"] / cell["target"] - 1) <= 0.02
                               and abs(cell["emulated_edit_norm"] ** 2 / cell["target"] ** 2 - 1) <= 0.02)
                        if bool(cell["reachable"]) != rr_:
                            fails["reach_flag"].append((j, l, key))
                    elif cell.get("reachable"):
                        fails["reach_flag"].append((j, l, key))
                if my_joint(lay, tol) != bool(lay["joint_predicted"]):
                    fails["joint"].append((j, l))
    ck["coverage_180x2"] = seen == set(range(180))
    for k in ("row_hash", "array_index", "repeat", "identity", "lineage", "mask", "masked_presence", "historical", "absent_direction",
              "direction_status", "direction_values", "random", "pair", "reach_absent", "reach_target", "reach_phi", "reach_emulation_plausibility",
              "reach_flag", "joint"):
        ck[f"independent:{k}"] = not fails[k]
    notes["failures"] = {k: v[:20] for k, v in fails.items() if v}
    return verdict("construction", ck, notes, "SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)", "SRC_CF0_PILOT_AUDIT: FAIL (CONSTRUCTION)",
                   {"direction_seal_hash": seal["seal_hash"], "manifest_hash": m["manifest_hash"]})


# ---- primary (after the pushed pulse seal; before any lexical reference) --------------------------------------------------

def processed_top1(z32: np.ndarray, t: int, sup, beg) -> int:
    x = z32.astype(np.float64).copy()
    x[sup] = -np.inf
    if t == 0:
        x[beg] = -np.inf
    best = float(np.max(x))
    return int(np.flatnonzero(x == best)[0])


def gen_cfg(cfg):
    g = json.loads((Path(cfg["model"]["dir"]) / "generation_config.json").read_text())
    return list(g["suppress_tokens"]), list(g["begin_suppress_tokens"])


def cmd_primary(args) -> dict:
    cfg = J(CONFIG)
    sup, beg = gen_cfg(cfg)
    ck, notes = {}, {}
    sok, seal = seal_ok(PSEAL)
    ck["pulse_seal_pushed_unchanged"] = sok and ancestor(seal["source_commit"], "origin/cs-asr-steer-inf")
    m, mok = manifest_ok("pulse")
    ck["manifest_pulse"] = mok and seal["manifest_hash"] == m["manifest_hash"]
    auth = J(AUTH)
    keys = cfg["firewall"]["authorization_keys"]
    ck["authorization_minimal_valid"] = (sorted(auth) == sorted(keys) and canon({k: v for k, v in auth.items() if k != "authorization_hash"}) == auth["authorization_hash"]
                                         and auth["gate_pass"] is True and auth["direction_seal_sha256"] == fhash(ROOT / DSEAL)
                                         and auth["construction_audit_sha256"] == fhash(ROOT / CAUDIT) and auth["config_sha256"] == CONFIG_SHA
                                         and auth["authorization_hash"] == m["authorization_hash"] == seal["authorization_hash"]
                                         and set(auth["qualified_layers"]) <= set(LAYERS) and committed(AUTH) and on_remote(AUTH))
    ck["construction_audit_pass"] = J(CAUDIT)["verdict"] == "SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)"
    st = J(f"{RUN}/pulse_status.json")
    rc, forb = runtime_firewall(st, m, "pulse")
    ck.update(rc)
    notes["forbidden_opened"] = forb
    rt = J(PROJ)
    pr = rt["projection"]
    Q = pr["runtime_queries"]
    steps = sum(1 + u["max_t"] for u in pr["utterances"])
    c = st["counters"]
    ck["counters"] = (c["encoder_calls"] == 80 and c["decoder_steps"] == steps and c["zero_dose_forwards"] == 360 and c["apparatus_forwards"] == 540
                      and c["none_forwards"] == 180 and c["matrix_forwards"] <= 4320)
    notes["counters"] = c
    tol = cfg["energy"]["paired_relative_squared_energy_difference_max"]
    fails = defaultdict(list)
    cells_n, steered_n, retention = 0, 0, {}
    pred = {l: set() for l in LAYERS}
    cohort = defaultdict(set)
    for i in range(80):
        cd = J(f"{RUN}/construction/{i:03d}.json")
        with np.load(ROOT / f"{RUN}/construction/{i:03d}.npz") as z:
            A = {k: z[k] for k in z.files}
        d = J(f"{RUN}/pulses/{i:03d}.json")
        ok_files = fhash(ROOT / f"{RUN}/pulses/{i:03d}_logits.npz") == d["logits_file_sha256"] and fhash(ROOT / f"{RUN}/pulses/{i:03d}_arrays.npz") == d["arrays_file_sha256"]
        with np.load(ROOT / f"{RUN}/pulses/{i:03d}_logits.npz") as z:
            LG = {k: z[k] for k in z.files}
        with np.load(ROOT / f"{RUN}/pulses/{i:03d}_arrays.npz") as z:
            AR = {k: z[k] for k in z.files}
        ok_files &= all(ahash(LG[k]) == h for k, h in d["logits_index"].items()) and all(ahash(AR[k]) == h for k, h in d["arrays_index"].items())
        if not ok_files or d["manifest_hash"] != m["manifest_hash"]:
            fails["files"].append(i)
        crec = {q["j"]: q for q in cd["queries"]}
        for js, row in d["positions"].items():
            j = int(js)
            k = f"q{j:03d}"
            t = Q[j]["t"]
            if not (row["none_bitwise_sealed_clean"] and row["none_ffn_equals_site"] and row["restore_fingerprint"] and all(row["zero"].values())
                    and len(row["zero"]) == 2):
                fails["none_zero_restore"].append(j)
            if row["none_top1"] != processed_top1(bf16(A[f"{k}_clean_r1_raw"]), t, sup, beg):
                fails["none_top1"].append(j)
            if set(row["apparatus"]) != {"hist_D0_L16_plus", "hist_D0_L16_minus", "hist_D2_L16"} or not all(
                    all(v for v in b.values() if v is not None) for b in row["apparatus"].values()):
                fails["apparatus"].append(j)
            if len(row["cells"]) != 24:
                fails["matrix"].append(j)
            for l in LAYERS:
                if my_joint(crec[j]["layers"][str(l)], tol):
                    pred[l].add(j)
            for aid, cell in row["cells"].items():
                cells_n += 1
                fam, l, eta, sign = cell["family"], cell["layer"], cell["eta"], cell["sign"]
                if aid != label_id(l, eta, sign, fam):
                    fails["arm_id"].append((j, aid))
                vkey = f"{k}_{'target' if fam == 'random' else fam}_L{l}_v"
                if cell["forward"] != (vkey in A):
                    fails["eligibility"].append((j, aid))
                if cell["integrity_failures"]:
                    fails["integrity"].append((j, aid))
                if not cell["steered"]:
                    if cell.get("output") != "BASELINE_REFERENCE" or (cell["forward"] and cell.get("no_edit_reason") not in (
                            "energy_unreachable", "solver_target_unattainable_within_8_evaluations")) or cell["valid"]:
                        fails["no_edit"].append((j, aid))
                    continue
                steered_n += 1
                before = bf16(A[f"{k}_clean_L{l}"]).astype(np.float64)
                after = bf16(AR[f"{k}_{aid}_consumed"]).astype(np.float64)
                prop = AR[f"{k}_{aid}_proposed"].astype(np.float64)
                rn = math.sqrt(float(before @ before))
                real = float(cell["realized_edit_norm"])
                tgt = eta * rn
                cons = float(np.linalg.norm(after - before))
                pn = float(np.linalg.norm(prop - before))
                mine = {"eta_rel": abs(real / rn / eta - 1) <= 0.02, "sq_energy_rel": abs(real ** 2 / tgt ** 2 - 1) <= 0.02,
                        "solver_matches_hook": abs(cell["solver"]["edit_norm"] - real) <= 1e-6 * max(1.0, real),
                        "consumed_vs_proposed": abs(cons - pn) <= 0.005 * pn}
                if mine != cell["checks"] or cell["valid"] != all(mine.values()) or not close(cell["solver"]["target"], tgt) or not close(cell["consumed_edit_norm"], cons):
                    fails["energy_recompute"].append((j, aid))
                import torch
                v = torch.from_numpy((sign * A[f"{k}_{fam}_L{l}_v"]).astype(np.float32)).double()
                v = v / torch.linalg.vector_norm(v)                     # the hook-side unit direction of the sealed vector
                if cell["solver_v_sha256"] != rawsha(v.numpy()):
                    fails["direction_used"].append((j, aid))
                if processed_top1(bf16(LG[f"{k}_{aid}"]), t, sup, beg) != cell["top1"]:
                    fails["top1"].append((j, aid))
            for l in LAYERS:
                for eta in ETAS:
                    for sign in SIGNS:
                        cs = [row["cells"][label_id(l, eta, sign, f)] for f in FAMS]
                        okp = all(c_["valid"] for c_ in cs)
                        if okp:
                            sq = [c_["realized_edit_norm"] ** 2 for c_ in cs]
                            okp = max(sq) / min(sq) - 1 <= tol
                        if okp != row["paired_energy"][label_id(l, eta, sign)]["ok"]:
                            fails["paired_energy"].append((j, label_id(l, eta, sign)))
                        if okp and j in pred[l]:
                            cohort[label_id(l, eta, sign)].add(j)
    for l in LAYERS:
        for eta in ETAS:
            for sign in SIGNS:
                a = label_id(l, eta, sign)
                retention[a] = {"sealed_joint_prediction": len(pred[l]), "paired_cohort": len(cohort[a]),
                                "retention": len(cohort[a]) / len(pred[l]) if pred[l] else 0.0}
    for k in ("files", "none_zero_restore", "none_top1", "apparatus", "matrix", "arm_id", "eligibility", "integrity", "no_edit", "energy_recompute",
              "direction_used", "top1", "paired_energy"):
        ck[f"independent:{k}"] = not fails[k]
    ck["complete_180x24"] = cells_n == 180 * 24
    notes.update(failures={k: v[:20] for k, v in fails.items() if v}, cells=cells_n, steered=steered_n, retention_reference_free=retention)
    return verdict("primary", ck, notes, "SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)", "SRC_CF0_PILOT_AUDIT: FAIL (PRIMARY)",
                   {"pulse_seal_hash": seal["seal_hash"], "manifest_hash": m["manifest_hash"]})


# ---- full ----------------------------------------------------------------------------------------------------------------

def my_gate_a(cfg) -> dict:
    """Independent Gate P-A from sealed construction records + strata membership."""
    mem = J(MEMBERSHIP)["evaluation_membership"]
    S, D = [x["stratum"] for x in mem], [x["dialogue_id"] for x in mem]
    rows = {}
    for i in range(80):
        for q in J(f"{RUN}/construction/{i:03d}.json")["queries"]:
            rows[q["j"]] = q
    g, nden = cfg["gate_a"]["per_layer"], cfg["gate_a"]["fraction_denominators"]
    tol = cfg["energy"]["paired_relative_squared_energy_difference_max"]
    out = {}
    for l in LAYERS:
        L = {j: rows[j]["layers"][str(l)] for j in range(180)}
        vt = [j for j in range(180) if rows[j]["region"]["target_status"] == "OK" and L[j]["target"].get("status") == "OK"]
        vp = [j for j in range(180) if rows[j]["region"]["paired_available"] and L[j]["target"].get("status") == "OK" and L[j]["off"].get("status") == "OK"]
        jn = [j for j in range(180) if my_joint(L[j], tol)]
        f = lambda js, s: [j for j in js if S[j] == s]
        nd = lambda js: len({D[j] for j in js})
        sp = [j for j in f(vp, "EN-confusion") if L[j]["off"]["raw_tangent_norm"] > 0]
        cos = [abs(float(L[j]["pair"]["cos"])) for j in sp]
        rat = [L[j]["target"]["raw_tangent_norm"] / L[j]["off"]["raw_tangent_norm"] for j in sp]
        cov = (len(vt) >= nden["minimum_valid_target_all_strata"] and len(vp) >= nden["minimum_valid_off_target_all_strata"]
               and len(f(vt, "EN-confusion")) >= g["EN_confusion_target_min"] and nd(f(vt, "EN-confusion")) >= g["EN_confusion_target_dialogues_min"]
               and len(f(jn, "EN-confusion")) >= g["EN_confusion_joint_reachable_min"] and nd(f(jn, "EN-confusion")) >= g["EN_confusion_joint_dialogues_min"]
               and len(f(jn, "EN-correct")) >= g["EN_correct_joint_reachable_min"] and nd(f(jn, "EN-correct")) >= g["EN_correct_joint_dialogues_min"])
        spec = (len(sp) >= g["specificity_pair_min"] and nd(sp) >= g["specificity_dialogues_min"] and bool(cos) and float(np.median(cos)) <= g["median_absolute_target_off_cosine_max"]
                and float(np.median(rat)) >= g["median_raw_tangent_target_off_norm_ratio_min"])
        out[l] = {"valid_targets": len(vt), "valid_off_pairs": len(vp), "EN_confusion_target": len(f(vt, "EN-confusion")),
                  "EN_confusion_target_dialogues": nd(f(vt, "EN-confusion")), "EN_confusion_joint": len(f(jn, "EN-confusion")),
                  "EN_confusion_joint_dialogues": nd(f(jn, "EN-confusion")), "EN_correct_joint": len(f(jn, "EN-correct")),
                  "EN_correct_joint_dialogues": nd(f(jn, "EN-correct")), "specificity_pairs": len(sp), "specificity_dialogues": nd(sp),
                  "median_abs_cosine": float(np.median(cos)) if cos else None, "median_tangent_ratio": float(np.median(rat)) if rat else None,
                  "coverage": bool(cov), "specificity": bool(spec), "joint_members": jn}
    return out


def my_lse(x):
    x = x[np.isfinite(x)]
    mx = float(x.max())
    return mx + math.log(float(np.exp(x - mx).sum()))


def my_boot(vals, keys, idx, lo, hi, mind):
    by = defaultdict(list)
    for d, v in vals:
        by[d].append(v)
    means = {d: sum(v) / len(v) for d, v in by.items()}
    est = sum(means.values()) / len(means) if means else None
    if not means:
        return est, None, None, 0
    dvec = np.array([means.get(k, np.nan) for k in keys])
    stats = []
    for row in idx:
        vals_ = [dvec[i_] for i_ in row if not np.isnan(dvec[i_])]
        if vals_:
            stats.append(sum(vals_) / len(vals_))
    if len(stats) < mind:
        return est, None, None, len(stats)
    return est, float(np.quantile(stats, lo)), float(np.quantile(stats, hi)), len(stats)


def cmd_full(args) -> dict:
    cfg = J(CONFIG)
    ck, notes = {}, {}
    ev = J(EVAL)
    gate = J(GATE_A)
    ck["evaluation_and_gate_committed"] = committed(EVAL) and committed(GATE_A)
    mg = my_gate_a(cfg)
    ck["gate_a_metrics_independent"] = all(
        all((mg[l][k] == gate["layers"][str(l)]["metrics"][k]) if not isinstance(mg[l][k], float) else close(mg[l][k], gate["layers"][str(l)]["metrics"][k])
            for k in ("valid_targets", "valid_off_pairs", "EN_confusion_target", "EN_confusion_target_dialogues", "EN_confusion_joint",
                      "EN_confusion_joint_dialogues", "EN_correct_joint", "EN_correct_joint_dialogues", "specificity_pairs", "specificity_dialogues",
                      "median_abs_cosine", "median_tangent_ratio")) for l in LAYERS)
    my_q = [l for l in LAYERS if mg[l]["coverage"] and mg[l]["specificity"]]
    ck["qualified_layers_independent"] = my_q == gate["qualified_layers"] and all(
        mg[l]["coverage"] == gate["layers"][str(l)]["result"]["construction"] and mg[l]["specificity"] == gate["layers"][str(l)]["result"]["specificity"]
        for l in LAYERS)
    notes["gate_a_independent"] = {l: {k: v for k, v in mg[l].items() if k != "joint_members"} for l in LAYERS}
    if ev.get("phase_b") == "NOT_RUN":
        integ = gate["integrity_ok"]
        my_label = ("SRC_CF0_PILOT_INVALID" if not integ else "SRC_CF0_PILOT_CONSTRUCTION_INSUFFICIENT" if not any(mg[l]["coverage"] for l in LAYERS)
                    else "SRC_CF0_PILOT_DIRECTION_NONSPECIFIC" if not my_q else None)
        ck["label_independent"] = my_label == ev["label"] and my_label is not None
        ck["construction_audit_pass"] = J(CAUDIT)["verdict"] == "SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)"
        return verdict("full", ck, notes, "SRC_CF0_PILOT_AUDIT: PASS (FULL)", "SRC_CF0_PILOT_AUDIT: FAIL (FULL)", {"independent_label": my_label})
    ck["primary_audit_pass"] = J(PAUDIT)["verdict"] == "SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)" and committed(PAUDIT)
    sup, beg = gen_cfg(cfg)
    pos = J(POSITIONS)["positions"]
    Q = J(PROJ)["projection"]["runtime_queries"]
    S, D = [p["stratum"] for p in pos], [p["dialogue_id"] for p in pos]
    Y = [set(int(x) for x in p["target_ids"] if int(x) < EOS) for p in pos]
    comp = [int(p["competitor"]) for p in pos]
    base, prow, LGs = {}, {}, {}
    for i in range(80):
        with np.load(ROOT / f"{RUN}/construction/{i:03d}.npz") as z:
            for q in J(f"{RUN}/construction/{i:03d}.json")["queries"]:
                base[q["j"]] = bf16(z[f"q{q['j']:03d}_clean_r1_raw"])
        d = J(f"{RUN}/pulses/{i:03d}.json")
        with np.load(ROOT / f"{RUN}/pulses/{i:03d}_logits.npz") as z:
            lg = {k: z[k] for k in z.files}
        for js, r in d["positions"].items():
            prow[int(js)] = r
            for aid, c in r["cells"].items():
                if c["steered"]:
                    LGs[(int(js), aid)] = bf16(lg[f"q{int(js):03d}_{aid}"])

    def score(z, j):
        x = z.astype(np.float64).copy()
        x[sup] = -np.inf
        if Q[j]["t"] == 0:
            x[beg] = -np.inf
        top = int(np.flatnonzero(x == x.max())[0])
        return top, top in Y[j], my_lse(x[sorted(Y[j])]) - x[comp[j]]

    none = {j: score(base[j], j) for j in range(180)}
    cell = {}
    for j in range(180):
        for aid, c in prow[j]["cells"].items():
            cell[(j, aid)] = score(LGs[(j, aid)], j) if c["steered"] else none[j]
    keys = sorted(set(D))
    idx = np.random.default_rng(cfg["statistics"]["seed"]).integers(0, len(keys), size=(cfg["statistics"]["bootstrap_draws"], len(keys)))
    a16 = 0.05 / (2 * cfg["statistics"]["primary_family"])
    mlo, mhi = cfg["statistics"]["margin_adjusted_CI_quantiles"]
    mind = cfg["statistics"]["minimum_finite_bootstrap_draws"]
    g, dmg = cfg["gate_b"], cfg["adverse_damage"]
    tol = cfg["energy"]["paired_relative_squared_energy_difference_max"]
    qual = set(J(AUTH)["qualified_layers"])
    mine, diffs = [], []
    evarms = {a["id"]: a for a in ev["arms"]}
    for l in LAYERS:
        pred = set(mg[l]["joint_members"])
        for e in ETAS:
            for s in SIGNS:
                a, ro, rr = label_id(l, e, s), label_id(l, e, s, "off"), label_id(l, e, s, "random")
                coh = []
                for j in sorted(pred):
                    cs = [prow[j]["cells"][x] for x in (a, ro, rr)]
                    if all(c["valid"] for c in cs):
                        sq = [c["realized_edit_norm"] ** 2 for c in cs]
                        if max(sq) / min(sq) - 1 <= tol:
                            coh.append(j)
                conf = [j for j in coh if S[j] == "EN-confusion"]
                corr_of = lambda x: {j: int((not none[j][1]) and cell[(j, x)][1]) for j in conf}
                ti = corr_of(a)
                res = {"id": a, "paired_rows": len(conf), "paired_dialogues": len({D[j] for j in conf}), "corrections": sum(ti.values()),
                       "corrected_dialogues": len({D[j] for j in conf if ti[j]}), "retention": len(coh) / len(pred) if pred else 0.0}
                integ_arm = not any(prow[j]["cells"][x]["integrity_failures"] for j in range(180) for x in (a, ro, rr))
                energy_ok = res["retention"] >= g["actual_edit_valid_fraction_of_sealed_prediction_min"] and integ_arm
                ctl_pass = True
                for name, x in (("random", rr), ("off_target", ro)):
                    ci = corr_of(x)
                    dd = {j: ti[j] - ci[j] for j in conf}
                    rep = sorted({D[j] for j in conf})
                    byd = {d_: sum(v for j, v in dd.items() if D[j] == d_) for d_ in rep}
                    est, lo, hi, nfin = my_boot([(D[j], float(dd[j])) for j in conf], keys, idx, a16, 1 - a16, mind)
                    lodo = [sum(v for j, v in dd.items() if D[j] != d_) for d_ in rep]
                    res[name] = {"net": sum(dd.values()), "pos": sum(1 for d_ in rep if byd[d_] > 0), "macro": est, "lodo": lodo, "lo": lo, "hi": hi}
                    ctl_pass &= (sum(dd.values()) >= g["net_corrections_over_each_control_min"] and res[name]["pos"] >= g["net_positive_dialogues_over_each_control_min"]
                                 and est is not None and est >= g["dialogue_macro_accuracy_advantage_over_each_control_min"] and bool(lodo) and min(lodo) > 0)
                act = [j for j in range(180) if prow[j]["cells"][a]["steered"]]
                res["EN"] = sum(1 for j in act if S[j] == "EN-correct" and none[j][1] and not cell[(j, a)][1])
                res["ZH"] = sum(1 for j in act if S[j] == "ZH-correct" and none[j][1] and not cell[(j, a)][1])
                res["EOS"] = sum(1 for j in act if S[j] != "EN-confusion" and none[j][0] != EOS and cell[(j, a)][0] == EOS)
                me, mlo_, _, _ = my_boot([(D[j], cell[(j, a)][2] - none[j][2]) for j in conf], keys, idx, mlo, mhi, mind)
                res["margin"], res["margin_lower"] = me, mlo_
                res["power"] = bool(l in qual and energy_ok and res["paired_rows"] >= g["paired_EN_confusion_min"]
                                    and res["paired_dialogues"] >= g["paired_EN_confusion_dialogues_min"] and res["corrections"] >= g["paired_corrections_min"]
                                    and res["corrected_dialogues"] >= g["corrected_dialogues_min"] and ctl_pass)
                res["safe"] = (res["EN"] <= g["selected_arm_EN_correct_corruptions_max"] and res["ZH"] <= g["selected_arm_ZH_correct_corruptions_max"]
                               and res["EOS"] <= g["selected_arm_correct_state_EOS_promotions_max"])
                res["glob"] = (res["EN"] >= dmg["global_any_primary_EN_correct_corruptions_at_least"] or res["ZH"] >= dmg["global_any_primary_ZH_correct_corruptions_at_least"]
                               or res["EOS"] >= dmg["global_any_primary_correct_state_EOS_promotions_at_least"])
                res["margin_only"] = bool(l in qual and energy_ok and me is not None and me >= cfg["statistics"]["margin_only_mean_nat_min"]
                                          and mlo_ is not None and mlo_ > 0)
                res.update(layer=l, eta=e)
                mine.append(res)
                E_ = evarms[a]
                same = (E_["paired_rows"] == res["paired_rows"] and E_["paired_dialogues"] == res["paired_dialogues"] and E_["corrections"] == res["corrections"]
                        and E_["corrected_dialogues"] == res["corrected_dialogues"] and E_["EN_corruptions"] == res["EN"] and E_["ZH_corruptions"] == res["ZH"]
                        and E_["correct_EOS_promotions"] == res["EOS"] and close(E_["retention"], res["retention"]) and E_["power_pass"] == res["power"]
                        and E_["observed_safety_pass"] == res["safe"])
                for name in ("random", "off_target"):
                    same &= (E_[name]["net_corrections"] == res[name]["net"] and E_[name]["positive_dialogues"] == res[name]["pos"]
                             and E_[name]["lodo_net"] == res[name]["lodo"] and ((E_[name]["macro_advantage"] is None and res[name]["macro"] is None)
                                                                                 or close(E_[name]["macro_advantage"], res[name]["macro"], rel=1e-9, ab=1e-12))
                             and E_[name]["bootstrap_adjusted"]["lower"] == (None if res[name]["lo"] is None else E_[name]["bootstrap_adjusted"]["lower"])
                             and (res[name]["lo"] is None or close(E_[name]["bootstrap_adjusted"]["lower"], res[name]["lo"], rel=1e-9, ab=1e-12)))
                same &= ((E_["margin_macro"] is None and res["margin"] is None) or close(E_["margin_macro"], res["margin"], rel=1e-9, ab=1e-12))
                same &= ((E_["margin_adjusted_lower"] is None) == (res["margin_lower"] is None)) and (
                    res["margin_lower"] is None or close(E_["margin_adjusted_lower"], res["margin_lower"], rel=1e-9, ab=1e-12))
                if not same:
                    diffs.append(a)
    integ = bool(ev["integrity_ok"]) and all(J(PAUDIT)["checks"].values())
    if not integ:
        label = "SRC_CF0_PILOT_INVALID"
    elif any(r["glob"] for r in mine):
        label = "SRC_CF0_PILOT_OBSERVED_DAMAGE"
    elif any(r["power"] and r["safe"] for r in mine):
        label = "SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED"
    elif any(r["power"] for r in mine):
        label = "SRC_CF0_PILOT_OBSERVED_DAMAGE"
    elif any(r["margin_only"] for r in mine):
        label = "SRC_CF0_PILOT_MARGIN_ONLY"
    else:
        label = "SRC_CF0_PILOT_CAUSAL_INSUFFICIENT"
    elig = sorted([r for r in mine if r["power"] and r["safe"]], key=lambda r: (-r["corrections"], -r["corrected_dialogues"],
                                                                             -min(r["random"]["net"], r["off_target"]["net"]), r["ZH"], r["EN"], r["eta"], r["layer"], r["id"]))
    sel = elig[0]["id"] if (elig and label == "SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED") else None
    ck["arm_metrics_independent"] = not diffs
    ck["label_independent"] = label == ev["label"]
    ck["selection_independent"] = sel == ev["selected"]
    ck["no_oracle_union_selection"] = ev["selected"] is None or ev["selected"] in [r["id"] for r in mine]
    notes.update(arm_diffs=diffs, independent_arms=mine)
    return verdict("full", ck, notes, "SRC_CF0_PILOT_AUDIT: PASS (FULL)", "SRC_CF0_PILOT_AUDIT: FAIL (FULL)", {"independent_label": label, "independent_selected": sel})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pre", "construction", "primary", "full"))
    ap.add_argument("--out", default=None, help="output path (default: the stage's canonical audit file; never overwritten)")
    args = ap.parse_args()
    fn = {"pre": cmd_pre, "construction": cmd_construction, "primary": cmd_primary, "full": cmd_full}[args.stage]
    out = ROOT / (args.out or {"pre": PRERUN, "construction": CAUDIT, "primary": PAUDIT, "full": FAUDIT}[args.stage])
    if out.exists():
        raise FileExistsError(f"{out} exists; preserve failed attempts under a new name")
    res = fn(args)

    def safe(o):
        if isinstance(o, dict):
            return {str(k): safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [safe(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Fraction):
            return str(o)
        return o
    res = safe(res)
    res["audit_hash"] = canon(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({"verdict": res["verdict"], "failed": res["failed"]}, indent=1))


if __name__ == "__main__":
    main()
