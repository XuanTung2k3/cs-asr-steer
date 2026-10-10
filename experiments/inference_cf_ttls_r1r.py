#!/usr/bin/env python
"""TTLS-R1R runner: numerical repair + re-execution of the four TTLS arms. Contract: docs/inference_cf/TTLS_R1R_SPEC.md,
configs/inference_cf/ttls_r1r.json. Reuses the sealed TTLS-R1 plan (same 100 exposed ids, teachers, masks) unchanged.

manifest  (CPU) resolved manifest after the sources are committed.
run       (GPU) ONE job, two phases.
          Phase A (all 100 rows, no optimizer): canonical B0 decode == sealed S0; theta0 branches -> candidates / stable
          set, equal to the sealed R1 rows; zero-vector free decode under the actual masks (ALL; {t*}) == B0; zero-vector
          teacher-forced logits and L16 FFN input bitwise == clean; zero-vector preservation KL. Any failure STOPS the
          job before any TTLS optimization.
          Phase B (all 100 rows): T1 TTLS-CE, T2 TTLS-CE+P, T4 TTLS-AC, T6 TTLS-AC+P on the ratio-first kernel (fresh
          zero z, 2 AdamW steps, projection), repaired-hook free decode, displacement, FFN-consumption / norm / outside-
          mask probe, gradient-at-init and initial-KL gates, post-row clean decode == B0. One JSON per row (atomic).
seal      (CPU) output seal before any reference access.
No reference, evaluator or outcome input anywhere in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "ttls_r1r_v1"
CONFIG = "configs/inference_cf/ttls_r1r.json"
BASE = "results/inference_cf/ttls_r1r"
PLAN = "results/inference_cf/ttls_r1/plan_sealed.json"
R1_ROWS = "results/inference_cf/ttls_r1/run1/rows"
MAX_NEW = 200
ARMS = (("T1", "CE", False), ("T2", "CE", True), ("T4", "AC", False), ("T6", "AC", True))
KL0_MAX = 1e-12
NORM_REL_MAX = 0.02
CAND_KEYS = ("t_star", "c_star", "M", "accepted", "S", "T")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


SOURCES = ("docs/inference_cf/TTLS_R1R_SPEC.md", CONFIG, PLAN, "src/csasr/inference_cf/ttls_r1r.py", "experiments/inference_cf_ttls_r1r.py",
           "experiments/inference_cf_ttls_r1r_evaluate.py", "slurm/inference_cf_ttls_r1r.sbatch", "tests/test_ttls_r1r.py",
           "src/csasr/inference_cf/ttls.py", "experiments/inference_cf_ttls_r1.py", "experiments/inference_cf_ttls_r1_evaluate.py",
           "configs/inference_cf/ttls_r1.json", "results/inference_cf/ttls_r1/output_seal.json",
           "src/csasr/inference_cf/episodic_tta.py", "src/csasr/inference_cf/soft_auto_tta.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py",
           "src/csasr/inference_cf/readout.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "src/csasr/models/whisper.py",
           "experiments/inference_cf_p2tta0.py")


def check_inherited() -> dict:
    c = cfg()["inherits"]
    return {"r1_config": sha_file(ROOT / c["config"]) == c["config_sha256"], "plan": sha_file(ROOT / c["plan"]) == c["plan_sha256"],
            "r1_seal": sha_file(ROOT / c["r1_output_seal"]) == c["r1_output_seal_sha256"]}


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES)
    if dirty:
        raise ValueError("commit TTLS-R1R sources before preparing a manifest:\n" + dirty)
    inh = check_inherited()
    if not all(inh.values()):
        raise ValueError(f"inherited artifacts changed: {inh}")
    plan = json.loads((ROOT / PLAN).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "plan_hash": plan["plan_hash"], "ids": plan["ids"], "arms": [a[0] for a in ARMS],
           "numerics": cfg()["numerics"]["name"], "inherited_checks": inh,
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "seed": 240924, "sources": {p: file_hash(ROOT / p) for p in SOURCES}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---- per-utterance processing (GPU; CPU-testable on the tiny model) -----------------------------------------------

class Ctx:
    def __init__(self, bundle, guard, *, suppress, begin, eos, embedded, cB, cE, layer, encode, encode_null, tau=None):
        from csasr.inference_cf.ttls import TAU
        self.bundle, self.model, self.guard = bundle, bundle.model, guard
        self.suppress, self.begin, self.eos = list(suppress), list(begin), int(eos)
        self.embedded, self.cB, self.cE, self.layer = embedded, list(cB), list(cE), int(layer)
        self.encode = encode
        self.null_inf, self.null_train = encode_null()
        self.tau = TAU if tau is None else float(tau)
        self.counters = {a[0]: {} for a in ARMS}


def _peak_reset():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {"alloc": int(torch.cuda.max_memory_allocated()), "reserved": int(torch.cuda.max_memory_reserved())}
    return None


def _out(d: dict, s: dict) -> dict:
    from csasr.inference_cf.episodic_tta import levenshtein, severe_truncation
    return {"tokens": d["tokens"], "text": d["text"], "terminated": d["terminated"], "length": d["length"],
            "equal_B0": d["tokens"] == s["y_B"] and d["terminated"] == s["y_B_terminated"], "d_to_B0": levenshtein(d["tokens"], s["y_B"]),
            "d_to_AUTO": levenshtein(d["tokens"], s["y_A"]),
            "severe_truncation": severe_truncation(len(s["y_B"]), d["length"], d["terminated"])}


def theta0_branches(ctx: Ctx, s: dict, enc_train) -> dict:
    """Frozen theta0-only quantities on the B0 path (identical to TTLS-R1 process_row)."""
    import torch
    from csasr.inference_cf import ttls as L
    from csasr.inference_cf.episodic_tta import teacher_logits
    y_B, T, model = list(s["y_B"]), len(s["y_B"]), ctx.model
    with torch.no_grad():
        lg = {"cB_clean": teacher_logits(model, enc_train, ctx.cB, y_B, None),
              "cB_null": teacher_logits(model, ctx.null_train, ctx.cB, y_B, None),
              "cE_clean": teacher_logits(model, enc_train, ctx.cE, y_B, None),
              "cE_null": teacher_logits(model, ctx.null_train, ctx.cE, y_B, None)}
    lp = {k: L.processed_logprobs(v, ctx.suppress, ctx.begin) for k, v in lg.items()}
    del lg
    p0_top = [float(np.exp(lp["cB_clean"][t][y_B[t]])) for t in range(T)]
    cand = L.select_candidates(lp, y_B, ctx.embedded, s["utf8_ok"], ctx.eos, tau=ctx.tau)
    S = L.stable_positions(p0_top, s["y_B_valid_mask"], cand["M"])
    base = L.displacement(ctx.bundle, enc_train, y_B, suppress=ctx.suppress, begin=ctx.begin, layer=ctx.layer, prompt=ctx.cB)
    return {"candidates": {"records": cand["records"], "M": cand["M"], "accepted": cand["accepted"], "t_star": cand["t_star"],
                           "c_star": cand["c_star"], "p0_top": p0_top, "S": S, "T": T}, "base": base}


def integrity_row(ctx: Ctx, s: dict, i: int, r1_candidates: dict | None) -> dict:
    """Phase A: zero-vector / no-optimizer controls under the actual masks (repaired kernel) + historical contrast."""
    import torch
    from csasr.inference_cf import ttls as L
    from csasr.inference_cf import ttls_r1r as R
    from csasr.inference_cf.episodic_tta import TTAInvalid, forced_decode
    bundle, guard = ctx.bundle, ctx.guard
    t_u = time.time()
    row = {"identity": s["utterance_id"], "index": i, "dialogue_id": s["dialogue_id"], "group": s["group"], "a3_group": s["a3_group"]}
    enc_inf, enc_train = ctx.encode(s)
    if guard is not None and not guard.verify():
        raise TTAInvalid("theta0 not resident at row start")
    d0 = forced_decode(bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
    row["B0"] = {"tokens_equal_S0": d0["tokens"] == s["y_B"], "terminated_equal_S0": d0["terminated"] == s["y_B_terminated"]}
    br = theta0_branches(ctx, s, enc_train)
    cand, base = br["candidates"], br["base"]
    row["candidates"] = cand
    if r1_candidates is not None:
        row["candidates_equal_R1"] = {k: cand[k] == r1_candidates[k] for k in CAND_KEYS}
        row["candidates_equal_R1"]["records"] = cand["records"] == r1_candidates["records"]
        row["candidates_equal_R1"]["p0_top"] = cand["p0_top"] == r1_candidates["p0_top"]
    dev = next(ctx.model.parameters()).device
    zero = torch.zeros(int(bundle.d_model), device=dev)
    masks = {"ALL": L.ALL}
    if cand["t_star"] is not None:
        masks["AC"] = {cand["t_star"]}
    row["zero"] = {}
    for k, steps in masks.items():
        dz = L.greedy_decode(bundle, enc_inf, ctx.cB, hook_factory=lambda st=steps: R.ttls_hook_r1r(bundle, zero, st, layer=ctx.layer, mode="steer"),
                             max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
        probe = R.site_ffn_probe(bundle, enc_train, s["y_B"], z=zero, steps=steps, layer=ctx.layer, prompt=ctx.cB)
        hist = R.site_ffn_probe(bundle, enc_train, s["y_B"], z=zero, steps=steps, layer=ctx.layer, prompt=ctx.cB, kernel="historical")
        kl0 = R.zero_kl(bundle, enc_train, s["y_B"], base["logq"], cand["S"], steps, ctx.suppress, ctx.begin, layer=ctx.layer, prompt=ctx.cB)
        row["zero"][k] = {**_out(dz, s), "decode_equals_B0": dz["tokens"] == d0["tokens"] and dz["terminated"] == d0["terminated"],
                          "probe": probe, "historical_kernel_probe": hist, "kl0_stable": kl0}
    row["pass"] = bool(row["B0"]["tokens_equal_S0"] and row["B0"]["terminated_equal_S0"]
                       and all(row.get("candidates_equal_R1", {k: True for k in CAND_KEYS})[k] for k in CAND_KEYS)
                       and all(z["decode_equals_B0"] and z["probe"]["logits_bitwise_equal"] and z["probe"]["ffn_input_bitwise_equal"]
                               and z["kl0_stable"] <= KL0_MAX for z in row["zero"].values()))
    row["elapsed_sec"] = time.time() - t_u
    row["status"] = "ok"
    return row


def process_row(ctx: Ctx, s: dict, i: int, phaseA: dict) -> dict:
    """Phase B: the four corrected TTLS arms on one utterance."""
    import torch
    from csasr.inference_cf import ttls as L
    from csasr.inference_cf import ttls_r1r as R
    from csasr.inference_cf.episodic_tta import TTAInvalid, forced_decode
    from csasr.inference_cf.readout import processed_logits
    from csasr.lss.sites import assert_no_site_hooks
    bundle, model, guard = ctx.bundle, ctx.model, ctx.guard
    t_u = time.time()
    row = {"identity": s["utterance_id"], "index": i, "dialogue_id": s["dialogue_id"], "group": s["group"], "a3_group": s["a3_group"]}
    y_B, y_A, T = list(s["y_B"]), list(s["y_A"]), len(s["y_B"])
    enc_inf, enc_train = ctx.encode(s)
    if guard is not None and not guard.verify():
        raise TTAInvalid("theta0 not resident at row start")
    ts = time.time()
    br = theta0_branches(ctx, s, enc_train)
    cand, base = br["candidates"], br["base"]
    if any(cand[k] != phaseA["candidates"][k] for k in CAND_KEYS + ("p0_top",)):
        raise TTAInvalid("theta0 branches not reproducible between phases")
    S, t_star, c_star = cand["S"], cand["t_star"], cand["c_star"]
    row["candidates"] = {k: cand[k] for k in CAND_KEYS}
    row["branch_sec"] = time.time() - ts
    for name, obj, P in ARMS:
        if obj == "AC" and t_star is None:
            row[name] = {"status": "abstain", "noop": True, **_out({"tokens": y_B, "text": s["y_B_text"], "terminated": s["y_B_terminated"],
                                                                     "length": T}, s)}
            continue
        steps = L.ALL if obj == "CE" else {t_star}
        cnt = ctx.counters[name]
        _peak_reset()
        ep = R.EpisodeR1R(bundle, enc_train, steps, layer=ctx.layer)

        def loss_fn(e, obj=obj, P=P):
            parts = {}
            if obj == "CE":
                logits_A, loss = L.ce_terms(e, y_A, s["y_A_valid_mask"], ctx.suppress, ctx.begin, ctx.eos, prompt=ctx.cB)
                parts["CE"] = float(loss.detach())
                logits_B = (logits_A if y_A == y_B else e.forward(ctx.cB, y_B)) if P else None
            else:
                logits_B = e.forward(ctx.cB, y_B)
                loss = L.ac_term(logits_B, t_star, c_star, ctx.suppress, ctx.begin)
                parts["AC"] = float(loss.detach())
                with torch.no_grad():
                    lpb = torch.log_softmax(processed_logits(logits_B[t_star].detach(), t_star, ctx.suppress, ctx.begin), dim=-1)
                    parts["target_rank"] = int((lpb > lpb[int(c_star)]).sum())
                    parts["margin_vs_b"] = float(lpb[int(c_star)] - lpb[int(y_B[t_star])])
            if P:
                p = L.preservation_term(logits_B, base["logq"], S, T, ctx.suppress, ctx.begin)
                parts["P"] = float(p.detach())
                loss = loss + L.LAMBDA_P * p
            return {"loss": loss, "parts": parts}

        res = ep.run(loss_fn, cnt)
        log = res["log"]
        eff = ep.effective()
        g0 = log["grad_l2"][0]
        gates = {"grad0_finite_positive": bool(math.isfinite(g0) and g0 > 0),
                 "initial_P_zero": (not P) or log["parts"][0]["P"] <= KL0_MAX}
        ts = time.time()
        d = L.greedy_decode(bundle, enc_inf, ctx.cB, hook_factory=lambda: R.ttls_hook_r1r(bundle, eff, steps, layer=ctx.layer, mode="steer"),
                            max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
        dec_sec = time.time() - ts
        es = range(1, T + 1) if steps == L.ALL else sorted(steps)
        disp = R.displacement_r1r(bundle, enc_train, y_B, z=eff, steps=steps, base=base, suppress=ctx.suppress, begin=ctx.begin,
                                  S=S, edit_steps=es, layer=ctx.layer, prompt=ctx.cB)
        probe = R.site_ffn_probe(bundle, enc_train, y_B, z=eff, steps=steps, layer=ctx.layer, prompt=ctx.cB)
        nz = bool((eff != 0).any())
        gates["edit_consumed"] = (not nz) or probe["edited_chord_max"] > 0
        gates["norm_preserved"] = probe["edited_norm_rel_err_max"] <= NORM_REL_MAX
        gates["outside_mask_zero"] = probe["outside_chord_max"] == 0.0
        gates["budget"] = log["master_delta_l2"] <= L.E_STAR * (1 + 1e-6)
        cnt["decodes"] = cnt.get("decodes", 0) + 1
        assert_no_site_hooks(bundle)
        gates["reset"] = guard is None or (guard.verify() and guard.current_hash() == guard.theta0_hash)
        row[name] = {"status": "ok", "noop": False, "variable": "TTLS", "objective": obj, "preservation": P, "numerics": R.NUMERICS,
                     "mask": "ALL" if steps == L.ALL else sorted(steps), **_out(d, s), "log": log, "displacement": disp, "probe": probe,
                     "gates": gates, "decode_sec": dec_sec, "peak": _peak(), "z_effective": eff.float().cpu().tolist()}
        if not all(gates.values()):
            raise TTAInvalid(f"{name}: integrity gate failed {gates}")
        del ep, res, eff
    # exact episodic reset: the clean decoder still reproduces B0 after all episodes on this row
    dpost = forced_decode(bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
    row["post_row_clean_equals_B0"] = dpost["tokens"] == y_B and dpost["terminated"] == s["y_B_terminated"]
    if not row["post_row_clean_equals_B0"]:
        raise TTAInvalid("post-row clean decode != B0 (state leaked)")
    assert_no_site_hooks(bundle)
    row["status"] = "ok"
    row["elapsed_sec"] = time.time() - t_u
    return row


# ---- GPU run ------------------------------------------------------------------------------------------------------

def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, tensor_bytes_hash
    from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
    from csasr.inference_cf.ttls import CB, CE_PROMPT, LAYER
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0 as t0run
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", m["git_commit"], "HEAD"], cwd=ROOT).returncode != 0:
        raise ValueError("manifest source commit is not an ancestor of HEAD")
    extra = set(git("diff", "--name-only", m["git_commit"], "HEAD").split()) - {f"{args.out}/manifest.json"}
    if extra:
        raise ValueError(f"files changed after the manifest source commit: {sorted(extra)}")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != m["plan_hash"] or not all(check_inherited().values()):
        raise ValueError("plan / inherited artifacts")
    t_setup = time.time()
    torch.manual_seed(m["seed"])
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    model = bundle.model
    model.eval()
    model.requires_grad_(False)
    names = decoder_ln_names(model)
    guard = LNGuard(model, names)                      # used only to verify theta0 stays resident (no LN arm here)
    tok = bundle.processor.tokenizer
    gen = model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or tok.eos_token_id != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    part = tokenizer_partition(tok)
    if part["hash"] != plan["partition_hash"]:
        raise TTAInvalid("partition")
    all_hash0 = tensor_bytes_hash(list(model.parameters()))

    def encode(s):
        if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
            raise TTAInvalid("audio bytes changed")
        inputs = batch_model_inputs(bundle, [s["audio_path"]])
        with torch.inference_mode():
            h = model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
        h2 = h.clone()
        if h2.is_inference() or h2.requires_grad:
            raise TTAInvalid("encoder clone")
        return BaseModelOutput(last_hidden_state=h), BaseModelOutput(last_hidden_state=h2)

    def encode_null():
        zeros = np.zeros(30 * bundle.sample_rate, dtype=np.float32)
        f = waveform_model_inputs(bundle, zeros)
        with torch.inference_mode():
            h = model.model.encoder(input_features=f["input_features"], attention_mask=f["attention_mask"]).last_hidden_state
        return BaseModelOutput(last_hidden_state=h), BaseModelOutput(last_hidden_state=h.clone())

    ctx = Ctx(bundle, guard, suppress=suppress, begin=begin, eos=tok.eos_token_id, embedded=set(part["embedded_ids"]), cB=CB,
              cE=CE_PROMPT, layer=LAYER, encode=encode, encode_null=encode_null)
    (out / "integrity").mkdir(parents=True, exist_ok=True)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "all_params_hash_start": all_hash0, "manifest_hash": m["manifest_hash"], "numerics": m["numerics"]}
    atomic_json(out / "runtime.json", runtime)
    status, phaseA = "completed", {}
    try:
        # ---- Phase A: zero-vector integrity census on all 100 rows (no optimizer) ----
        tA = time.time()
        fails = []
        for i, s in enumerate(plan["rows"]):
            p = out / "integrity" / f"{i:03d}.json"
            if p.exists() and json.loads(p.read_text()).get("status") == "ok":
                row = json.loads(p.read_text())
            else:
                r1 = json.loads((ROOT / R1_ROWS / f"{i:03d}.json").read_text())
                if r1["identity"] != s["utterance_id"]:
                    raise TTAInvalid("R1 row order")
                row = integrity_row(ctx, dict(s), i, r1["candidates"])
                row["manifest_hash"] = m["manifest_hash"]
                atomic_json(p, row)
            phaseA[i] = row
            if not row["pass"]:
                fails.append(s["utterance_id"])
            hd = row["zero"]["ALL"]["historical_kernel_probe"]["max_abs_logit_diff"]
            print(f"TTLS-R1R A {i + 1}/100 {s['utterance_id']} pass={row['pass']} Z0_ALL={'=' if row['zero']['ALL']['decode_equals_B0'] else 'Δ'} "
                  f"AC={'-' if 'AC' not in row['zero'] else ('=' if row['zero']['AC']['decode_equals_B0'] else 'Δ')} hist_dlogit={hd:.3g} "
                  f"kl0={row['zero']['ALL']['kl0_stable']:.2g} {row['elapsed_sec']:.1f}s", flush=True)
        runtime["phaseA"] = {"rows": len(phaseA), "failures": fails, "pass": not fails, "elapsed_sec": time.time() - tA}
        atomic_json(out / "runtime.json", runtime)
        if fails:
            raise TTAInvalid(f"STOP: zero-edit identity failed on {len(fails)} rows; TTLS efficacy not evaluated")
        # ---- Phase B: the four corrected TTLS arms ----
        tB = time.time()
        for i, s in enumerate(plan["rows"]):
            p = out / "rows" / f"{i:03d}.json"
            if p.exists() and json.loads(p.read_text()).get("status") == "ok":
                continue
            row = process_row(ctx, dict(s), i, phaseA[i])
            row["manifest_hash"] = m["manifest_hash"]
            atomic_json(p, row)
            arms = " ".join(f"{a[0]}={'-' if row[a[0]]['status'] == 'abstain' else ('=' if row[a[0]]['equal_B0'] else 'Δ')}" for a in ARMS)
            print(f"TTLS-R1R B {i + 1}/100 {s['utterance_id']} {s['group']} t*={row['candidates']['t_star']} {arms} {row['elapsed_sec']:.1f}s", flush=True)
            runtime.update(counters=ctx.counters, rows_done=i + 1)
            atomic_json(out / "runtime.json", runtime)
        runtime["phaseB"] = {"elapsed_sec": time.time() - tB}
    except Exception as exc:
        import traceback
        status = "stopped_identity" if "STOP:" in repr(exc) else "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
    runtime["reset_final_ok"] = bool(guard.verify() and guard.current_hash() == guard.theta0_hash)
    runtime["all_params_hash_end"] = tensor_bytes_hash(list(model.parameters()))
    runtime["params_unchanged"] = runtime["all_params_hash_end"] == all_hash0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, counters=ctx.counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit(f"TTLS-R1R run {status}: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file())
    man = json.loads((run / "manifest.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"], "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"])}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--out", required=True)
    sub.add_parser("seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"manifest": cmd_manifest, "run": cmd_run, "seal": cmd_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
