#!/usr/bin/env python
"""TTLS-R1 runner (unified test-time learned steering pilot). Contract: docs/inference_cf/TTLS_R1_SPEC.md,
configs/inference_cf/ttls_r1.json. Exploratory; reuses the frozen fixed-100 panel and the sealed TTA1 teacher plan.

prepare   (CPU) 100-row plan from the sealed TTA1 plan: y_B (P2-SEQ S0), y_A (historical AUTO), A2 outputs + archived
          fp32 masters (TTA1 / TTA0-R), UTF-8 prefix completeness per step, A3-panel membership. No reference.
manifest  (CPU) resolved manifest after the sources are committed.
run       (GPU) per utterance: theta0 forced decode = S0 (B0); historical detect_language AUTO replay = AUTO (B1);
          archived A2 masters decode = historical A2 (B2) + displacement; four frozen teacher-forced branches on the B0
          path (clean/null x zh/en prompt) -> frozen AC candidate rule and stable set; seven adaptation arms
          (T1 TTLS-CE, T2 TTLS-CE+P, T2A A2-CE+P, T3 A2-AC, T4 TTLS-AC, T5 A2-AC+P, T6 TTLS-AC+P), each a fresh
          variable, 2 AdamW steps, exact reset, forced-ZH greedy decode; comparators CD (D3-rule contrastive decoding)
          and ACSUB (accepted candidate substituted, greedy continuation). First rows: clean-identity / gradient-flow /
          reset validation (fail fast). One JSON per row, written atomically (resumable).
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

SCHEMA = "ttls_r1_v1"
CONFIG = "configs/inference_cf/ttls_r1.json"
BASE = "results/inference_cf/ttls_r1"
PLAN = f"{BASE}/plan_sealed.json"
TTA1_PLAN = "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json"
A3_PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
MAX_NEW = 200
ARMS = (("T1", "TTLS", "CE", False), ("T2", "TTLS", "CE", True), ("T2A", "LN", "CE", True),
        ("T3", "LN", "AC", False), ("T4", "TTLS", "AC", False), ("T5", "LN", "AC", True), ("T6", "TTLS", "AC", True))


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- prepare (CPU, reference-free) ---------------------------------------------------------------------------------

def resolve_a2(u: str, t1plan: dict) -> dict:
    """A2 output and archived fp32 masters exactly as the audited TTA1 evaluator resolved them."""
    r = next(x for x in t1plan["rows"] if x["utterance_id"] == u)
    if r["origin"] == "new":
        k = t1plan["new_ids"].index(u)
        rel = f"results/inference_cf/p2tta_funnel/tta1/run1/rows/{k:02d}.json"
        npz = f"results/inference_cf/p2tta_funnel/tta1/run1/rows/{k:02d}_final_masters_fp32.npz"
    else:
        rel, npz = r["ancestor"]["row"], r["ancestor"]["masters_npz"]
    row = json.loads((ROOT / rel).read_text())
    o = row["objectives"]["A2"]
    return {"row": rel, "row_sha256": sha_file(ROOT / rel), "origin": r["origin"], "masters_npz": npz,
            "masters_npz_sha256": sha_file(ROOT / npz), "masters_npz_sha256_recorded": row["final_masters_npz_sha256"],
            "tokens": o["tokens"], "text": o["text"], "terminated": o["terminated"], "length": o["length"]}


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from csasr.inference_cf.core_r2 import prefix_utf8_complete, tokenizer_partition
    from csasr.inference_cf.dir_sprint0 import legal_static
    from csasr.inference_cf.ttls import CB, CE_PROMPT
    c = cfg()
    pins = {p: sha_file(ROOT / p) == h for p, h in c["source_sha256"].items()}
    checks = {"pins": all(pins.values()), "config_committed": committed(CONFIG)}
    t1 = json.loads((ROOT / TTA1_PLAN).read_text())
    panel = json.loads((ROOT / c["population"]["panel"]).read_text())
    a3 = {r["utterance_id"]: r["group"] for r in json.loads((ROOT / A3_PANEL).read_text())["rows"]}
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    tok = proc.tokenizer
    eos = tok.eos_token_id
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    dec = {v: k for k, v in bytes_to_unicode().items()}
    part = tokenizer_partition(tok)
    legal = legal_static(tok, sup, eos)
    D3 = c["comparators"]["CD"]["legal_static"]
    checks["legal_static"] = len(legal) == D3["count"] and digest(legal) == D3["hash"]
    checks["prompts"] = (tok.convert_ids_to_tokens(CB) == c["prompts"]["cB_tokens"] and
                         tok.convert_ids_to_tokens(CE_PROMPT) == c["prompts"]["cE_tokens"])
    checks["partition"] = part["hash"] == c["candidate_rule"]["partition_hash"]
    checks["panel_order"] = [r["utterance_id"] for r in panel["rows"]] == t1["ids"]
    from csasr.inference_cf.exposure_registry import exposed_ids
    exposed = set(exposed_ids(ROOT)["ids"])
    checks["already_exposed"] = set(t1["ids"]) <= exposed
    rows, ok = [], True
    for r in t1["rows"]:
        u = r["utterance_id"]
        a2 = resolve_a2(u, t1)
        fine = (sha_file(ROOT / r["forced_row"]) == r["forced_row_sha256"] and sha_file(ROOT / r["auto_row"]) == r["auto_row_sha256"]
                and a2["masters_npz_sha256"] == a2["masters_npz_sha256_recorded"].removeprefix("sha256:"))
        ok &= bool(fine)
        y_B = r["y_B"]
        rows.append({"utterance_id": u, "dialogue_id": r["dialogue_id"], "group": r["group"], "a3_group": a3.get(u),
                     "audio_path": r["audio_path"], "audio_fingerprint_64k_sizeprefixed": r["audio_fingerprint_64k_sizeprefixed"],
                     "audio_full_sha256": r["audio_full_sha256"], "y_B": y_B, "y_B_terminated": r["y_B_terminated"],
                     "y_B_text": r["y_B_text"], "y_B_valid_mask": r["y_B_valid_mask"], "y_A": r["y_A"],
                     "y_A_terminated": r["y_A_terminated"], "y_A_text": r["y_A_text"], "y_A_valid_mask": r["y_A_valid_mask"],
                     "utf8_ok": [prefix_utf8_complete(tok, y_B[:t], dec) for t in range(len(y_B))], "A2": a2})
    checks["rows"] = bool(ok)
    checks["a3_subset"] = sum(r["a3_group"] is not None for r in rows) == 24
    plan = {"schema": SCHEMA + "_plan", "config_sha256": sha_file(ROOT / CONFIG), "tta1_plan_sha256": sha_file(ROOT / TTA1_PLAN),
            "checks": checks, "pins": pins, "ids": [r["utterance_id"] for r in rows], "suppression": {"suppress": sup, "begin": beg},
            "eos": eos, "partition_hash": part["hash"], "legal_static_hash": digest(legal), "model_files": prep.model_hashes(),
            "environment": prep.environment(), "rows": rows, "outcomes_computed": False, "references_used": False,
            "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps({k: v for k, v in checks.items() if not v}) + json.dumps(pins))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks, "rows": len(rows),
                      "D": sum(r["group"] == "D" for r in rows), "a3": sum(r["a3_group"] is not None for r in rows)}))


SOURCES = ("docs/inference_cf/TTLS_R1_SPEC.md", CONFIG, PLAN, "src/csasr/inference_cf/ttls.py", "experiments/inference_cf_ttls_r1.py",
           "experiments/inference_cf_ttls_r1_evaluate.py", "slurm/inference_cf_ttls_r1.sbatch", "tests/test_ttls_r1.py",
           "src/csasr/inference_cf/episodic_tta.py", "src/csasr/inference_cf/soft_auto_tta.py", "src/csasr/inference_cf/dir_sprint0.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py",
           "src/csasr/inference_cf/readout.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "src/csasr/models/whisper.py",
           "experiments/inference_cf_p2tta0.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES)
    if dirty:
        raise ValueError("commit TTLS-R1 sources before preparing a manifest:\n" + dirty)
    plan = json.loads((ROOT / PLAN).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "plan_hash": plan["plan_hash"], "ids": plan["ids"], "arms": [a[0] for a in ARMS],
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
    """Everything process_row needs; built from the real bundle in ``run`` or from the tiny model in tests."""

    def __init__(self, bundle, guard, *, suppress, begin, eos, embedded, legal_mask, byte_decoder, cB, cE, layer,
                 encode, encode_null, detect_lang, a2_effective, auto_prompt, tau=None):
        self.bundle, self.model, self.guard = bundle, bundle.model, guard
        self.suppress, self.begin, self.eos = list(suppress), list(begin), int(eos)
        self.embedded, self.legal_mask, self.byte_decoder = embedded, legal_mask, byte_decoder
        self.cB, self.cE, self.layer = list(cB), list(cE), int(layer)
        self.encode, self.detect_lang, self.a2_effective, self.auto_prompt = encode, detect_lang, a2_effective, auto_prompt
        self.null_inf, self.null_train = encode_null()
        from csasr.inference_cf.ttls import TAU
        self.tau = TAU if tau is None else float(tau)
        self.counters = {a[0]: {} for a in ARMS}
        self.validated = {"clean_identity": None, "ttls_grad_flow_CE": None, "ttls_grad_flow_AC": None, "ln_grad_flow_AC": None}


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
            "equal_B0": d["tokens"] == s["y_B"], "d_to_B0": levenshtein(d["tokens"], s["y_B"]),
            "d_to_AUTO": levenshtein(d["tokens"], s["y_A"]),
            "severe_truncation": severe_truncation(len(s["y_B"]), d["length"], d["terminated"])}


def process_row(ctx: Ctx, s: dict, i: int) -> dict:
    import torch
    from csasr.inference_cf.episodic_tta import TTAInvalid, forced_decode, teacher_logits
    from csasr.inference_cf import ttls as L
    from csasr.inference_cf.readout import processed_logits
    from csasr.lss.sites import assert_no_site_hooks
    bundle, model, guard = ctx.bundle, ctx.model, ctx.guard
    t_u = time.time()
    row = {"identity": s["utterance_id"], "index": i, "dialogue_id": s["dialogue_id"], "group": s["group"], "a3_group": s["a3_group"]}
    y_B, y_A, T = list(s["y_B"]), list(s["y_A"]), len(s["y_B"])
    enc_inf, enc_train = ctx.encode(s)
    if not guard.verify():
        raise TTAInvalid("theta0 not resident at row start")
    # B0 / B1 reuse identity
    ts = time.time()
    d0 = forced_decode(bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
    lang = ctx.detect_lang(s)
    cA = ctx.auto_prompt(lang)
    rep = forced_decode(bundle, enc_inf, cA, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
    row["B0"] = {"tokens_equal_S0": d0["tokens"] == y_B, "terminated_equal_S0": d0["terminated"] == s["y_B_terminated"]}
    row["B1"] = {"lang_id": lang, "cA": cA, "replay_text_equal": rep["text"] == s["y_A_text"], "replay_tokens_equal": rep["tokens"] == y_A}
    row["baseline_decode_sec"] = time.time() - ts
    # frozen reference-free branches on the B0 path (before any adaptation)
    ts = time.time()
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
    t_star, c_star = cand["t_star"], cand["c_star"]
    base = L.displacement(bundle, enc_train, y_B, suppress=ctx.suppress, begin=ctx.begin, layer=ctx.layer, prompt=ctx.cB)
    row["candidates"] = {"records": cand["records"], "M": cand["M"], "accepted": cand["accepted"], "t_star": t_star, "c_star": c_star,
                         "p0_top": p0_top, "S": S, "T": T}
    row["branch_sec"] = time.time() - ts
    del lp
    # B2: archived historical A2 masters (no re-adaptation)
    eff2 = ctx.a2_effective(s)
    if eff2 is not None:
        ts = time.time()
        guard.materialize(eff2)
        try:
            d2 = forced_decode(bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
            disp2 = L.displacement(bundle, enc_train, y_B, base=base, suppress=ctx.suppress, begin=ctx.begin, S=S,
                                   edit_steps=range(T + 1), layer=ctx.layer, prompt=ctx.cB)
        finally:
            guard.restore()
        if not guard.verify():
            raise TTAInvalid("B2 reset")
        row["B2"] = {**_out(d2, s), "reproduces_historical_A2": d2["tokens"] == s["A2"]["tokens"]
                     and d2["terminated"] == s["A2"]["terminated"], "displacement": disp2, "decode_sec": time.time() - ts}
    # first-row clean identity of the TTLS hook path (empty mask = exact no-op decode; zero edit at all steps)
    if ctx.validated["clean_identity"] is None:
        dev = next(model.parameters()).device
        zero = torch.zeros(int(bundle.d_model), device=dev)
        dI = L.greedy_decode(bundle, enc_inf, ctx.cB, hook_factory=lambda: L.ttls_hook(bundle, zero, set(), layer=ctx.layer, mode="steer"),
                             max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
        with torch.no_grad():
            a = teacher_logits(model, enc_train, ctx.cB, y_B, None)
            with L.ttls_hook(bundle, zero, L.ALL, layer=ctx.layer, mode="steer"):
                b = teacher_logits(model, enc_train, ctx.cB, y_B, None)
        ctx.validated["clean_identity"] = {"row": i, "empty_mask_decode_equals_B0": dI["tokens"] == d0["tokens"],
                                           "zero_edit_all_steps_max_abs_logit_diff": float((a - b).abs().max())}
        if not ctx.validated["clean_identity"]["empty_mask_decode_equals_B0"]:
            raise TTAInvalid("clean identity: empty-mask TTLS decode != B0")
    # adaptation arms
    for name, var, obj, P in ARMS:
        if obj == "AC" and t_star is None:
            row[name] = {"status": "abstain", **_out(d0, s), "noop": True}
            continue
        steps = L.ALL if obj == "CE" else {t_star}
        cnt = ctx.counters[name]
        _peak_reset()
        ep = L.Episode(bundle, guard if var == "LN" else None, enc_train, var, steps=steps, layer=ctx.layer)

        def loss_fn(e, obj=obj, P=P):
            parts = {}
            if obj == "CE":
                logits_A, loss = L.ce_terms(e, y_A, s["y_A_valid_mask"], ctx.suppress, ctx.begin, ctx.eos, prompt=ctx.cB)
                parts["CE"] = float(loss)
                logits_B = (logits_A if y_A == y_B else e.forward(ctx.cB, y_B)) if P else None
            else:
                logits_B = e.forward(ctx.cB, y_B)
                loss = L.ac_term(logits_B, t_star, c_star, ctx.suppress, ctx.begin)
                parts["AC"] = float(loss)
                with torch.no_grad():
                    lpb = torch.log_softmax(processed_logits(logits_B[t_star].detach(), t_star, ctx.suppress, ctx.begin), dim=-1)
                    parts["target_rank"] = int((lpb > lpb[int(c_star)]).sum())
                    parts["margin_vs_b"] = float(lpb[int(c_star)] - lpb[int(y_B[t_star])])
            if P:
                p = L.preservation_term(logits_B, base["logq"], S, T, ctx.suppress, ctx.begin)
                parts["P"] = float(p)
                loss = loss + L.LAMBDA_P * p
            return {"loss": loss, "parts": parts}

        res = ep.run(loss_fn, cnt)
        eff = ep.effective()
        ts = time.time()
        if var == "LN":
            guard.materialize(eff)
            try:
                d = forced_decode(bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
                disp = L.displacement(bundle, enc_train, y_B, base=base, suppress=ctx.suppress, begin=ctx.begin, S=S,
                                      edit_steps=range(T + 1), layer=ctx.layer, prompt=ctx.cB)
            finally:
                guard.restore()
        else:
            d = L.greedy_decode(bundle, enc_inf, ctx.cB, hook_factory=lambda: L.ttls_hook(bundle, eff, steps, layer=ctx.layer, mode="steer"),
                                max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
            es = range(1, T + 1) if steps == L.ALL else sorted(steps)
            disp = L.displacement(bundle, enc_train, y_B, z=eff, steps=steps, base=base, suppress=ctx.suppress, begin=ctx.begin,
                                  S=S, edit_steps=es, layer=ctx.layer, prompt=ctx.cB)
        dec_sec = time.time() - ts
        cnt["decodes"] = cnt.get("decodes", 0) + 1
        reset_ok = guard.verify() and guard.current_hash() == guard.theta0_hash
        assert_no_site_hooks(bundle)
        if not reset_ok:
            raise TTAInvalid(f"{name}: reset failed")
        log = res["log"]
        row[name] = {"status": "ok", "noop": False, "variable": var, "objective": obj, "preservation": P,
                     "mask": "ALL" if steps == L.ALL else sorted(steps), **_out(d, s), "log": log, "displacement": disp,
                     "decode_sec": dec_sec, "peak": _peak(), "reset_ok": reset_ok,
                     "z_effective": eff.float().cpu().tolist() if var == "TTLS" else None}
        key = {("TTLS", "CE"): "ttls_grad_flow_CE", ("TTLS", "AC"): "ttls_grad_flow_AC", ("LN", "AC"): "ln_grad_flow_AC"}.get((var, obj))
        if key and ctx.validated[key] is None and not P:
            g0 = log["grad_l2"][0]
            ctx.validated[key] = {"row": i, "arm": name, "grad_l2_step0": g0, "losses": log["losses"],
                                  "pass": bool(math.isfinite(g0) and g0 > 0)}
            if not ctx.validated[key]["pass"]:
                raise TTAInvalid(f"gradient flow failed: {key}")
        del ep, res, eff
    # no-update comparators
    ts = time.time()
    cd = L.cd_decode(bundle, enc_inf, ctx.null_inf, ctx.cB, ctx.legal_mask, ctx.byte_decoder, max_new_tokens=MAX_NEW,
                     capture_layer=ctx.layer)
    row["CD"] = {**_out(cd, s), "edits": cd["edits"], "decode_sec": time.time() - ts}
    if t_star is None:
        row["ACSUB"] = {"status": "abstain", **_out(d0, s), "noop": True}
    else:
        dsub = L.greedy_decode(bundle, enc_inf, ctx.cB, forced=y_B[:t_star] + [c_star], max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
        row["ACSUB"] = {"status": "ok", "noop": False, **_out(dsub, s)}
    if not guard.verify():
        raise TTAInvalid("row end reset")
    row["validated"] = dict(ctx.validated)
    row["status"] = "ok"
    row["elapsed_sec"] = time.time() - t_u
    return row


# ---- GPU run ------------------------------------------------------------------------------------------------------

def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.dir_sprint0 import legal_static
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, tensor_bytes_hash
    from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
    from csasr.inference_cf.soft_auto_tta import auto_prompt
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
    if extra:                                    # only the committed run manifest may follow the source commit
        raise ValueError(f"files changed after the manifest source commit: {sorted(extra)}")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != m["plan_hash"]:
        raise ValueError("plan")
    t_setup = time.time()
    torch.manual_seed(m["seed"])
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    model = bundle.model
    model.eval()
    model.requires_grad_(False)
    names = decoder_ln_names(model)
    if len(names) != 194:
        raise TTAInvalid("trainable enumeration")
    guard = LNGuard(model, names)
    tok = bundle.processor.tokenizer
    gen = model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or tok.eos_token_id != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    part = tokenizer_partition(tok)
    legal = legal_static(tok, suppress, tok.eos_token_id)
    if part["hash"] != plan["partition_hash"] or digest(legal) != plan["legal_static_hash"]:
        raise TTAInvalid("partition / legal set")
    legal_mask = np.zeros(int(model.config.vocab_size), dtype=bool)
    legal_mask[legal] = True
    names_set = set(names)
    nonln0 = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])

    def encode(s):
        if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
            raise TTAInvalid("audio bytes changed")
        inputs = batch_model_inputs(bundle, [s["audio_path"]])
        s["_features"] = inputs["input_features"]
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

    def detect_lang(s):
        with torch.no_grad():
            return int(model.detect_language(input_features=s.pop("_features"), generation_config=model.generation_config)[0])

    def a2_effective(s):
        z = np.load(ROOT / s["A2"]["masters_npz"])
        return {n: torch.from_numpy(z[f"A2_p{j:03d}"]).to(device=guard.params[n].device, dtype=torch.bfloat16) for j, n in enumerate(names)}

    ctx = Ctx(bundle, guard, suppress=suppress, begin=begin, eos=tok.eos_token_id, embedded=set(part["embedded_ids"]),
              legal_mask=legal_mask, byte_decoder={v: k for k, v in bytes_to_unicode().items()}, cB=CB, cE=CE_PROMPT, layer=LAYER,
              encode=encode, encode_null=encode_null, detect_lang=detect_lang, a2_effective=a2_effective, auto_prompt=auto_prompt)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln0, "manifest_hash": m["manifest_hash"]}
    atomic_json(out / "runtime.json", runtime)
    invalid, status = [], "completed"
    try:
        for i, s in enumerate(plan["rows"]):
            p = out / "rows" / f"{i:03d}.json"
            if p.exists() and json.loads(p.read_text()).get("status") == "ok":
                continue
            row = process_row(ctx, dict(s), i)
            row["manifest_hash"] = m["manifest_hash"]
            for k, ok in (("B0", row["B0"]["tokens_equal_S0"] and row["B0"]["terminated_equal_S0"]), ("B1", row["B1"]["replay_text_equal"]),
                          ("B2", row.get("B2", {}).get("reproduces_historical_A2", False))):
                if not ok:
                    invalid.append(f"{s['utterance_id']}: {k} reuse identity failed")
            atomic_json(p, row)
            arms = " ".join(f"{a[0]}={'=' if row[a[0]]['equal_B0'] else 'Δ'}" for a in ARMS)
            print(f"TTLS-R1 {i + 1}/100 {s['utterance_id']} {s['group']} t*={row['candidates']['t_star']} |M|={len(row['candidates']['M'])} "
                  f"{arms} CD={'=' if row['CD']['equal_B0'] else 'Δ'} {row['elapsed_sec']:.1f}s", flush=True)
            runtime.update(counters=ctx.counters, validated=ctx.validated, invalid=invalid, rows_done=i + 1)
            atomic_json(out / "runtime.json", runtime)
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        try:
            guard.restore()
        except Exception:
            pass
    runtime["reset_final_ok"] = bool(guard.verify() and guard.current_hash() == guard.theta0_hash)
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=ctx.counters, validated=ctx.validated)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("TTLS-R1 run failed: " + runtime["failure"]["reason"])


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
    sub.add_parser("prepare")
    sub.add_parser("manifest").add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--out", required=True)
    sub.add_parser("seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "run": cmd_run, "seal": cmd_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
