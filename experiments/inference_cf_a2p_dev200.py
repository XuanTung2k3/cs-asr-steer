#!/usr/bin/env python
"""A2P-DEV200 runner: A2-CE+P (TTLS-R1 arm T2A, unchanged) on NEW200. Contract: docs/inference_cf/A2P_DEV200_SPEC.md,
configs/inference_cf/a2p_dev200.json. Exposed development, not confirmation.

prepare   (CPU) plan from the frozen P2-PATH5 roster / sealed outputs (NEW200: B0 = live theta0, B1 = AUTO, B2 = live A2,
          teacher y_A) + the FIXED100 regression-oracle subset from the sealed TTLS-R1 plan/rows. No reference.
manifest  (CPU) resolved manifest after the sources are committed.
run       (GPU) ONE job. Phase R (regression oracle, FIXED100 subset): B3 must reproduce sealed TTLS-R1 T2A exactly and an
          A2-only episode must reproduce the archived A2 masters / decode; any mismatch STOPS the job. Phase N (NEW200,
          200 rows): live B0 == P2-PATH5 theta0, AUTO replay == P2-PATH5 AUTO text, theta0 branches -> candidates / S,
          B3 episode (2 AdamW updates on 194 LN tensors), step-0 CE == P2-PATH5 A2 loss, initial P == 0, forced-ZH
          decode, displacement, exact reset. One JSON per row (atomic, resumable).
seal      (CPU) output seal before any NEW200 reference access.
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

SCHEMA = "a2p_dev200_v1"
CONFIG = "configs/inference_cf/a2p_dev200.json"
BASE = "results/inference_cf/a2p_dev200"
PLAN = f"{BASE}/plan_sealed.json"
P5 = "results/inference_cf/p2path5"
R1 = "results/inference_cf/ttls_r1"
MAX_NEW = 200
CAND_KEYS = ("t_star", "c_star", "M", "accepted", "S", "T")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def row_fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def membership_hash(name: str, ids, dlg: dict) -> str:
    """P2-PATH5 convention (experiments/inference_cf_p2path5.membership_hash)."""
    return row_fp({"partition": name, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in ids]})


def regression_indices(r1_rows: list, plan_rows: list) -> list[int]:
    """Frozen oracle rule: every FIXED100 row whose sealed B2 or T2A tokens differ from y_B, plus the first two rows
    (by index) where neither differs."""
    changed = [i for i, (r, p) in enumerate(zip(r1_rows, plan_rows)) if r["B2"]["tokens"] != p["y_B"] or r["T2A"]["tokens"] != p["y_B"]]
    same = [i for i in range(len(plan_rows)) if i not in set(changed)]
    return sorted(changed + same[:2])


# ---- prepare (CPU, reference-free) ---------------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from csasr.inference_cf.core_r2 import prefix_utf8_complete, tokenizer_partition
    from csasr.inference_cf.episodic_tta import special_ids, valid_mask
    from csasr.inference_cf.exposure_registry import exposed_ids
    c = cfg()
    pop = c["population"]
    checks = {"config_committed": committed(CONFIG), "panel_bytes": sha_file(ROOT / pop["panel"]) == pop["panel_sha256"],
              "path5_config_bytes": sha_file(ROOT / pop["path5_config"]) == pop["path5_config_sha256"]}
    panel = json.loads((ROOT / pop["panel"]).read_text())
    p5plan = json.loads((ROOT / P5 / "plan_sealed.json").read_text())
    p5seal = json.loads((ROOT / P5 / "output_seal.json").read_text())
    p5man = json.loads((ROOT / P5 / "run1/manifest.json").read_text())
    checks["path5_seal_files"] = all(sha_file(ROOT / p) == h.removeprefix("sha256:") for p, h in p5seal["files"].items())
    checks["path5_plan_hash"] = p5plan["plan_hash"] == p5man["plan_hash"] == p5seal["plan_hash"]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in panel["rows"]}
    parts = {k: panel["partitions"][k]["ids"] for k in ("FULL300", "FIXED100", "NEW200")}
    checks["membership"] = all(membership_hash(k, parts[k], dlg) == pop["membership_sha256"][k] for k in parts)
    checks["new200_is_full_minus_fixed"] = parts["NEW200"] == [u for u in parts["FULL300"] if u not in set(parts["FIXED100"])]
    checks["new200_shape"] = len(parts["NEW200"]) == 200 and len({dlg[u] for u in parts["NEW200"]}) == 20 \
        and all(sum(dlg[u] == d for u in parts["NEW200"]) == 10 for d in {dlg[u] for u in parts["NEW200"]})
    checks["already_exposed"] = set(parts["FULL300"]) <= set(exposed_ids(ROOT)["ids"])
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    tok = proc.tokenizer
    eos = tok.eos_token_id
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    sp = special_ids(tok)
    dec = {v: k for k, v in bytes_to_unicode().items()}
    part = tokenizer_partition(tok)
    r1plan = json.loads((ROOT / R1 / "plan_sealed.json").read_text())
    checks["suppression"] = {"suppress": sup, "begin": beg} == p5plan["suppression"] == r1plan["suppression"] and eos == p5plan["eos"]
    checks["partition"] = part["hash"] == p5plan["partition_hash"] == r1plan["partition_hash"]
    model_files = prep.model_hashes()
    checks["model_same_as_path5_and_r1"] = model_files == p5man["model"]["files"] == json.loads((ROOT / R1 / "run1/manifest.json").read_text())["model"]["files"]
    # ---- NEW200 rows from the sealed P2-PATH5 plan / outputs ----
    byid = {x["runtime"]["utterance_id"]: x for x in p5plan["rows"]}
    p5rows = {}
    for i in range(300):
        r = json.loads((ROOT / P5 / f"run1/rows/{i:03d}.json").read_text())
        p5rows[r["identity"]] = (f"{P5}/run1/rows/{i:03d}.json", r)
    prow = {r["utterance_id"]: r for r in panel["rows"]}
    rows, ok = [], True
    for u in parts["NEW200"]:
        x, (rel, r) = byid[u], p5rows[u]
        rt, an = x["runtime"], x["analysis"]
        fine = (r["status"] == "ok" and r["partition"] == "NEW200" and an["partition"] == "NEW200" and r["dialogue_id"] == dlg[u]
                and row_fp(rt["y_A"]) == prow[u]["teacher"]["y_A_token_sha256"] and row_fp(rt["y_A_valid_mask"]) == prow[u]["teacher"]["valid_mask_sha256"]
                and rt["y_A_valid_mask"] == valid_mask(rt["y_A"], sup, beg, sp, eos) and bool(r["A2_log"]["reset_ok"])
                and r["A2_log"]["steps"] == 2 and len(r["A2_log"]["losses"]) == 3)
        ok &= bool(fine)
        y_B = r["theta0"]["tokens"]
        rows.append({"utterance_id": u, "dialogue_id": dlg[u], "full_index": prow[u]["full_index"], "group": "D" if rt["y_A"] != y_B else "A",
                     "a3_group": None, "audio_path": rt["audio_path"], "audio_fingerprint_64k_sizeprefixed": rt["audio_fingerprint_64k_sizeprefixed"],
                     "audio_full_sha256": rt["audio_full_sha256"], "y_B": y_B, "y_B_terminated": r["theta0"]["terminated"],
                     "y_B_text": r["theta0"]["text"], "y_B_valid_mask": valid_mask(y_B, sup, beg, sp, eos),
                     "utf8_ok": [prefix_utf8_complete(tok, y_B[:t], dec) for t in range(len(y_B))], "y_A": rt["y_A"],
                     "y_A_valid_mask": rt["y_A_valid_mask"], "y_A_text": tok.decode(rt["y_A"], skip_special_tokens=True), "AUTO": an["AUTO"],
                     "A2": {**r["A2_free"], "loss0": r["A2_log"]["losses"][0], "grad0": r["A2_log"]["grad_l2"][0], "losses": r["A2_log"]["losses"],
                            "state_hash": r["A2_state_hash"]}, "source_row": rel, "source_row_sha256": sha_file(ROOT / rel)})
    checks["new200_rows"] = bool(ok) and len(rows) == 200
    # ---- FIXED100 regression oracle from the sealed TTLS-R1 plan / rows ----
    r1seal = json.loads((ROOT / R1 / "output_seal.json").read_text())
    checks["r1_seal_files"] = all(sha_file(ROOT / p) == h for p, h in r1seal["files"].items())
    r1rows = [json.loads((ROOT / R1 / f"run1/rows/{i:03d}.json").read_text()) for i in range(100)]
    idx = regression_indices(r1rows, r1plan["rows"])
    reg = []
    for i in idx:
        p, r = r1plan["rows"][i], r1rows[i]
        assert r["identity"] == p["utterance_id"]
        reg.append({**{k: p[k] for k in ("utterance_id", "dialogue_id", "group", "a3_group", "audio_path", "audio_fingerprint_64k_sizeprefixed",
                                           "audio_full_sha256", "y_B", "y_B_terminated", "y_B_text", "y_B_valid_mask", "utf8_ok", "y_A",
                                           "y_A_valid_mask", "y_A_text")},
                    "r1_index": i, "A2": p["A2"],
                    "oracle": {"candidates": {k: r["candidates"][k] for k in CAND_KEYS}, "T2A": {k: r["T2A"][k] for k in ("tokens", "terminated", "text")},
                               "T2A_log": {k: r["T2A"]["log"][k] for k in ("losses", "parts", "grad_l2", "master_delta_l2", "effective_delta_l2")}}})
    checks["regression_rows"] = len(reg) >= 3
    plan = {"schema": SCHEMA + "_plan", "config_sha256": sha_file(ROOT / CONFIG), "checks": checks, "suppression": {"suppress": sup, "begin": beg},
            "eos": eos, "partition_hash": part["hash"], "model_files": model_files, "environment": prep.environment(),
            "path5": {"plan_hash": p5plan["plan_hash"], "manifest_hash": p5man["manifest_hash"], "seal_hash": p5seal["seal_hash"]},
            "r1": {"plan_hash": r1plan["plan_hash"], "seal_hash": r1seal["seal_hash"]},
            "ids": parts["NEW200"], "rows": rows, "regression_indices": idx, "regression": reg,
            "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps({k: v for k, v in checks.items() if not v}))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks, "new200": len(rows), "regression": [r["utterance_id"] for r in reg],
                      "D": sum(r["group"] == "D" for r in rows)}))


SOURCES = ("docs/inference_cf/A2P_DEV200_SPEC.md", CONFIG, PLAN, "experiments/inference_cf_a2p_dev200.py",
           "experiments/inference_cf_a2p_dev200_evaluate.py", "slurm/inference_cf_a2p_dev200.sbatch", "tests/test_a2p_dev200.py",
           "src/csasr/inference_cf/ttls.py", "src/csasr/inference_cf/episodic_tta.py", "src/csasr/inference_cf/soft_auto_tta.py",
           "experiments/inference_cf_ttls_r1.py", "experiments/inference_cf_ttls_r1r.py", "experiments/inference_cf_ttls_r1_evaluate.py",
           "experiments/inference_cf_ttls_r1r_evaluate.py", "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/readout.py", "src/csasr/lss/sites.py", "src/csasr/models/whisper.py",
           "experiments/inference_cf_p2tta0.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES)
    if dirty:
        raise ValueError("commit A2P-DEV200 sources before preparing a manifest:\n" + dirty)
    plan = json.loads((ROOT / PLAN).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "plan_hash": plan["plan_hash"], "ids": plan["ids"], "regression_indices": plan["regression_indices"],
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "seed": 240924, "sources": {p: file_hash(ROOT / p) for p in SOURCES}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---- per-utterance processing (GPU; CPU-testable on the tiny model) -----------------------------------------------

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


def b3_loss_fn(ctx, s: dict, base: dict, S: list):
    """TTLS-R1 process_row loss for arm ('T2A', 'LN', 'CE', P=True), unchanged."""
    from csasr.inference_cf import ttls as L
    y_B, y_A, T = list(s["y_B"]), list(s["y_A"]), len(s["y_B"])

    def loss_fn(e):
        parts = {}
        logits_A, loss = L.ce_terms(e, y_A, s["y_A_valid_mask"], ctx.suppress, ctx.begin, ctx.eos, prompt=ctx.cB)
        parts["CE"] = float(loss.detach())
        logits_B = logits_A if y_A == y_B else e.forward(ctx.cB, y_B)
        p = L.preservation_term(logits_B, base["logq"], S, T, ctx.suppress, ctx.begin)
        parts["P"] = float(p.detach())
        return {"loss": loss + L.LAMBDA_P * p, "parts": parts}
    return loss_fn


def a2_loss_fn(ctx, s: dict):
    """A2 alone (P removed) through the same Episode path; regression oracle only."""
    from csasr.inference_cf import ttls as L

    def loss_fn(e):
        _, loss = L.ce_terms(e, list(s["y_A"]), s["y_A_valid_mask"], ctx.suppress, ctx.begin, ctx.eos, prompt=ctx.cB)
        return {"loss": loss, "parts": {"CE": float(loss.detach())}}
    return loss_fn


def _decode_with(ctx, enc_inf, eff):
    from csasr.inference_cf.episodic_tta import forced_decode
    ctx.guard.materialize(eff)
    try:
        return forced_decode(ctx.bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
    finally:
        ctx.guard.restore()


def process_row(ctx, s: dict, i: int, *, regression: bool = False) -> dict:
    import torch
    from csasr.inference_cf import ttls as L
    from csasr.inference_cf.episodic_tta import TTAInvalid, forced_decode, levenshtein, severe_truncation
    from csasr.lss.sites import assert_no_site_hooks
    import experiments.inference_cf_ttls_r1r as RR
    bundle, guard = ctx.bundle, ctx.guard
    t_u = time.time()
    row = {"identity": s["utterance_id"], "index": i, "dialogue_id": s["dialogue_id"], "group": s["group"], "phase": "R" if regression else "N"}
    y_B, T = list(s["y_B"]), len(s["y_B"])
    enc_inf, enc_train = ctx.encode(s)
    if not guard.verify():
        raise TTAInvalid("theta0 not resident at row start")
    ts = time.time()
    d0 = forced_decode(bundle, enc_inf, ctx.cB, max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
    row["B0"] = {"tokens_equal": d0["tokens"] == y_B, "terminated_equal": d0["terminated"] == s["y_B_terminated"]}
    if not regression:
        lang = ctx.detect_lang(s)
        rep = forced_decode(bundle, enc_inf, ctx.auto_prompt(lang), max_new_tokens=MAX_NEW, capture_layer=ctx.layer)
        row["B1"] = {"lang_id": lang, "replay_text_equal": rep["text"] == s["AUTO"]["text"], "replay_tokens_equal": rep["tokens"] == s["AUTO"]["tokens"],
                     "replay_terminated_equal": rep["terminated"] == s["AUTO"]["terminated"]}
    else:
        s.pop("_features", None)
    row["baseline_sec"] = time.time() - ts
    ts = time.time()
    br = RR.theta0_branches(ctx, s, enc_train)
    cand, base = br["candidates"], br["base"]
    S = cand["S"]
    row["candidates"] = {k: cand[k] for k in CAND_KEYS} | {"n_records": len(cand["records"]), "p0_top": cand["p0_top"]}
    row["branch_sec"] = time.time() - ts
    # ---- B3: A2-CE+P (T2A) ----
    _peak_reset()
    cnt = ctx.counters.setdefault("B3", {})
    ep = L.Episode(bundle, guard, enc_train, "LN", layer=ctx.layer)
    res = ep.run(b3_loss_fn(ctx, s, base, S), cnt)
    log = res["log"]
    eff = ep.effective()
    ts = time.time()
    d = _decode_with(ctx, enc_inf, eff)
    guard.materialize(eff)
    try:
        disp = L.displacement(bundle, enc_train, y_B, base=base, suppress=ctx.suppress, begin=ctx.begin, S=S, edit_steps=range(T + 1),
                              layer=ctx.layer, prompt=ctx.cB)
        state_hash = guard.current_hash()
    finally:
        guard.restore()
    cnt["decodes"] = cnt.get("decodes", 0) + 1
    reset_ok = guard.verify() and guard.current_hash() == guard.theta0_hash
    assert_no_site_hooks(bundle)
    row["B3"] = {"tokens": d["tokens"], "text": d["text"], "terminated": d["terminated"], "length": d["length"],
                 "equal_B0": d["tokens"] == y_B and d["terminated"] == s["y_B_terminated"], "d_to_B0": levenshtein(d["tokens"], y_B),
                 "equal_A2": d["tokens"] == s["A2"]["tokens"] and d["terminated"] == s["A2"]["terminated"],
                 "severe_truncation": severe_truncation(T, d["length"], d["terminated"]), "log": log, "displacement": disp,
                 "state_hash": state_hash, "decode_sec": time.time() - ts, "peak": _peak(), "reset_ok": reset_ok}
    gates = {"B0": row["B0"]["tokens_equal"] and row["B0"]["terminated_equal"], "reset": reset_ok,
             "initial_P_zero": log["parts"][0]["P"] == 0.0, "two_updates": len(log["losses"]) == 3 and len(log["grad_l2"]) == 2 and all(log["finite"]),
             "grad0_positive": log["grad_l2"][0] > 0}
    if regression:
        o = s["oracle"]
        gates["candidates_equal_R1"] = all(cand[k] == o["candidates"][k] for k in CAND_KEYS)
        gates["T2A_outputs_equal_R1"] = all(row["B3"][k] == o["T2A"][k] for k in ("tokens", "terminated", "text"))
        gates["T2A_log_equal_R1"] = all(log[k] == o["T2A_log"][k] for k in o["T2A_log"])
        # A2 alone through the same Episode path == archived historical A2 masters and decode
        ep2 = L.Episode(bundle, guard, enc_train, "LN", layer=ctx.layer)
        res2 = ep2.run(a2_loss_fn(ctx, s), ctx.counters.setdefault("regression_A2", {}))
        with np.load(ROOT / s["A2"]["masters_npz"]) as z:
            ck = {n: z[f"A2_p{j:03d}"] for j, n in enumerate(guard.names)}
        gates["A2_masters_equal_archive"] = all(np.array_equal(ep2.params[n].detach().cpu().numpy(), ck[n]) for n in guard.names)
        d2 = _decode_with(ctx, enc_inf, ep2.effective())
        gates["A2_decode_equal_sealed"] = d2["tokens"] == s["A2"]["tokens"] and d2["terminated"] == s["A2"]["terminated"]
        g3, g2 = log["grad_l2"][0], res2["log"]["grad_l2"][0]
        row["A2_only"] = {"losses": res2["log"]["losses"], "grad_l2": res2["log"]["grad_l2"], "step0_CE_equal_B3": res2["log"]["losses"][0] == log["parts"][0]["CE"],
                          "grad0_rel_diff_B3_vs_A2": abs(g3 - g2) / g2 if g2 else None}
        del ep2, res2
    else:
        gates["step0_CE_equal_PATH5_A2"] = log["parts"][0]["CE"] == s["A2"]["loss0"]
        gates["B1_replay_text"] = row["B1"]["replay_text_equal"]
        row["B3"]["grad0_rel_diff_vs_PATH5_A2"] = abs(log["grad_l2"][0] - s["A2"]["grad0"]) / s["A2"]["grad0"] if s["A2"]["grad0"] else None
    gates["reset_end"] = guard.verify() and guard.current_hash() == guard.theta0_hash
    row["gates"] = gates
    if not all(gates.values()):
        raise TTAInvalid(f"{s['utterance_id']}: integrity gate failed {[k for k, v in gates.items() if not v]}")
    del ep, res, eff
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
    from csasr.inference_cf.soft_auto_tta import auto_prompt
    from csasr.inference_cf.ttls import CB, CE_PROMPT, LAYER
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0 as t0run
    import experiments.inference_cf_ttls_r1r as RR
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
    if len(names) != 194 or sum(int(p.numel()) for n, p in model.named_parameters() if n in set(names)) != 248320:
        raise TTAInvalid("trainable enumeration")
    guard = LNGuard(model, names)
    tok = bundle.processor.tokenizer
    gen = model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or tok.eos_token_id != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    part = tokenizer_partition(tok)
    if part["hash"] != plan["partition_hash"]:
        raise TTAInvalid("partition")
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

    ctx = RR.Ctx(bundle, guard, suppress=suppress, begin=begin, eos=tok.eos_token_id, embedded=set(part["embedded_ids"]), cB=CB,
                 cE=CE_PROMPT, layer=LAYER, encode=encode, encode_null=encode_null)
    ctx.detect_lang, ctx.auto_prompt, ctx.counters = detect_lang, auto_prompt, {}
    for sub in ("regression", "rows"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln0, "manifest_hash": m["manifest_hash"]}
    atomic_json(out / "runtime.json", runtime)
    status = "completed"
    torch.cuda.reset_peak_memory_stats()
    try:
        tR = time.time()
        for s in plan["regression"]:
            p = out / "regression" / f"{s['r1_index']:03d}.json"
            if p.exists() and json.loads(p.read_text()).get("status") == "ok":
                continue
            row = process_row(ctx, dict(s), s["r1_index"], regression=True)
            row["manifest_hash"] = m["manifest_hash"]
            atomic_json(p, row)
            print(f"A2P-DEV200 R {s['r1_index']} {s['utterance_id']} T2A==R1 {row['gates']['T2A_outputs_equal_R1']} log==R1 {row['gates']['T2A_log_equal_R1']} "
                  f"A2==archive {row['gates']['A2_masters_equal_archive']} grad0 rel {row['A2_only']['grad0_rel_diff_B3_vs_A2']:.2e} {row['elapsed_sec']:.1f}s", flush=True)
        runtime["regression"] = {"rows": len(plan["regression"]), "pass": True, "elapsed_sec": time.time() - tR}
        atomic_json(out / "runtime.json", runtime)
        tN = time.time()
        for i, s in enumerate(plan["rows"]):
            p = out / "rows" / f"{i:03d}.json"
            if p.exists() and json.loads(p.read_text()).get("status") == "ok":
                continue
            row = process_row(ctx, dict(s), i)
            row["manifest_hash"] = m["manifest_hash"]
            atomic_json(p, row)
            print(f"A2P-DEV200 N {i + 1}/200 {s['utterance_id']} {s['group']} |S|={len(row['candidates']['S'])}/{len(s['y_B'])} "
                  f"B3={'=' if row['B3']['equal_B0'] else 'Δ'}B0 {'=' if row['B3']['equal_A2'] else 'Δ'}A2 AUTO_tok={row['B1']['replay_tokens_equal']} "
                  f"{row['elapsed_sec']:.1f}s", flush=True)
            runtime.update(counters=ctx.counters, rows_done=i + 1)
            atomic_json(out / "runtime.json", runtime)
        runtime["new200"] = {"elapsed_sec": time.time() - tN}
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        try:
            guard.restore()
        except Exception:
            pass
    runtime["peak_job"] = {"alloc": int(torch.cuda.max_memory_allocated()), "reserved": int(torch.cuda.max_memory_reserved())}
    runtime["reset_final_ok"] = bool(guard.verify() and guard.current_hash() == guard.theta0_hash)
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, counters=ctx.counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("A2P-DEV200 run failed: " + runtime["failure"]["reason"])


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
