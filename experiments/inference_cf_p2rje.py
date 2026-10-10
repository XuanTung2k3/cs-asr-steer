#!/usr/bin/env python
"""P2-RJ-E leverage expansion (final diagnostic stage; evaluator-side namespace).

``positions`` (CPU): enumerate ALL structurally valid EN-confusion positions of the D-dev-select
panel with the unchanged P2-R eligibility code (candidates before selection), mark the 60 P2-RJ
positions, freeze the list.
``manifest`` (CPU): freeze config, positions, sources, Git state.
``run`` (GPU): for each utterance, a no-grad pre-pass fixes c* by the frozen P2-R/P2-RJ rule
(first non-Y_ref token of topk(log_softmax(processed), 20)); then the UNCHANGED P2-RJ
``run_utterance`` computes the evaluator-only margin gradient with the zero-valued probe.

EVALUATOR-ONLY. The gradient is never applied as a steering direction; this module builds no
steering hook and runs no edited forward.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "p2rje_leverage_expansion_v1"
CONFIG = "configs/inference_cf/p2_rj_e_leverage_expansion.json"
POSITIONS = "results/inference_cf/p2rje/positions.json"
P2RJ_POSITIONS = "results/inference_cf/p2rj/positions.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
TOPK = 20
SOURCES = ("docs/inference_cf/P2_RJ_E_LEVERAGE_EXPANSION_SPEC.md", "docs/inference_cf/P2_RJ_E_PRE_RUN_AUDIT.md",
           CONFIG, POSITIONS, P2RJ_POSITIONS, "results/inference_cf/p2r/population.json",
           "experiments/inference_cf_p2rje.py", "experiments/inference_cf_p2rje_analyze.py",
           "experiments/inference_cf_p2rj.py", "experiments/inference_cf_p2rj_analyze.py",
           "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2r_population.py",
           "experiments/inference_cf_p2r_analyze.py", "experiments/inference_cf_cached.py",
           "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "slurm/inference_cf_p2rje.sbatch")


# ---- population ------------------------------------------------------------------------------

def eligible_en_confusion() -> tuple[list[dict], dict, str]:
    """All P2-R-eligible EN-confusion candidates (before selection), the P2-R ledger, and the hash
    of the P2-R population rebuilt by the same call (must equal the frozen one)."""
    import experiments.inference_cf_p2r_population as pp
    config = json.loads(pp.CONFIG.read_text())
    cap = {}
    original_select = pp.select

    def capture(cfg, candidates, ledger):
        cap["candidates"], cap["ledger"] = candidates, ledger
        return original_select(cfg, candidates, ledger)
    pp.select = capture
    try:
        pop = pp.build(config)
    finally:
        pp.select = original_select
    rebuilt = digest({k: v for k, v in pop.items()})
    ledger = {k: v for (s, k), v in sorted(cap["ledger"].items()) if s == "EN-confusion"}
    return cap["candidates"]["EN-confusion"], ledger, rebuilt


def build_positions(cands: list[dict], p2rj_positions: list[dict]) -> list[dict]:
    orig = {(p["utterance_id"], p["t"]): p for p in p2rj_positions if p["stratum"] == "EN-confusion"}
    out = []
    for c in sorted(cands, key=lambda c: (c["utterance_id"], c["t"])):
        key = (c["utterance_id"], c["t"])
        o = orig.get(key)
        rec = {"utterance_id": c["utterance_id"], "t": int(c["t"]), "stratum": "EN-confusion",
               "dialogue_id": c["dialogue_id"], "span_initial": c.get("span_initial"),
               "target_ids": [int(x) for x in c["target_ids"]], "baseline_token": int(c["baseline_token"]),
               "original": o is not None, "competitor": None if o is None else int(o["competitor"])}
        if o is not None:
            if o["target_ids"] != rec["target_ids"] or o["baseline_token"] != rec["baseline_token"]:
                raise ValueError(f"original position {key} differs from P2-RJ")
            rec["p2r"] = o["p2r"]
        out.append(rec)
    if len({(p["utterance_id"], p["t"]) for p in out}) != len(out):
        raise ValueError("duplicate positions")
    if sum(p["original"] for p in out) != len(orig):
        raise ValueError("not every original P2-RJ EN-confusion position is eligible")
    return out


def cmd_positions(_args) -> None:
    cfg = json.loads((ROOT / CONFIG).read_text())
    cands, ledger, rebuilt = eligible_en_confusion()
    if rebuilt != cfg["p2r_reference"]["population_hash"]:
        raise ValueError("P2-R eligibility no longer reproduces the frozen P2-R population")
    rj = json.loads((ROOT / P2RJ_POSITIONS).read_text())
    positions = build_positions(cands, rj["positions"])
    doc = {"schema": SCHEMA + "_positions", "config_hash": digest(cfg), "p2r_population_hash": rebuilt,
           "p2rj_positions_hash": rj["positions_hash"], "ledger_en_confusion": ledger,
           "utterances": sorted({p["utterance_id"] for p in positions}), "positions": positions,
           "counts": {"total": len(positions), "original": sum(p["original"] for p in positions),
                      "new": sum(not p["original"] for p in positions),
                      "utterances": len({p["utterance_id"] for p in positions}),
                      "dialogues": len({p["dialogue_id"] for p in positions})}}
    doc["positions_hash"] = digest(doc)
    path = ROOT / POSITIONS
    if path.exists():
        if json.loads(path.read_text())["positions_hash"] != doc["positions_hash"]:
            raise FileExistsError("frozen P2-RJ-E positions differ; never overwrite")
        print("positions unchanged", doc["positions_hash"]); return
    atomic_json(path, doc)
    print(json.dumps({"counts": doc["counts"], "ledger": ledger, "positions_hash": doc["positions_hash"]}))


def cmd_manifest(args) -> None:
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    pos = json.loads((ROOT / POSITIONS).read_text())
    if pos["config_hash"] != digest(cfg) or pos["positions_hash"] != digest({k: v for k, v in pos.items() if k != "positions_hash"}):
        raise ValueError("positions do not match the frozen config")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCES], cwd=ROOT, text=True).strip():
        raise ValueError("commit P2-RJ-E sources before preparing a manifest")
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(MODEL, local_files_only=True).tokenizer)
    manifest = {"schema": SCHEMA, "git_commit": rev, "config": CONFIG, "config_hash": digest(cfg),
                "positions": POSITIONS, "positions_hash": pos["positions_hash"], "partition_hash": partition["hash"],
                "sources": {p: file_hash(ROOT / p) for p in SOURCES}}
    manifest["manifest_hash"] = digest(manifest)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P2-RJ-E manifest exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "git_commit": rev}))


# ---- competitor pre-pass (frozen P2-R summarize ordering) ------------------------------------

def competitor_rule(logits: torch.Tensor, step: int, suppress, begin, target_ids: list[int]) -> dict:
    """c* = first token of topk(log_softmax(processed), 20) not in Y_ref (P2-R summarize order)."""
    import experiments.inference_cf_cached as cached
    proc = cached._processed(logits.detach().float().cpu(), step, suppress, begin)
    lp = torch.log_softmax(proc, dim=-1)
    top = torch.topk(lp, TOPK).indices.tolist()
    ref = set(int(x) for x in target_ids)
    c = next(int(i) for i in top if int(i) not in ref)
    others = lp.clone(); others[list(ref)] = -float("inf")
    ties = int((others == lp[c]).sum()) - 1
    return {"competitor": c, "tied_non_reference": ties}


def competitor_prepass(ctx, *, encoded, tokens: list[int], positions: list[dict]) -> dict[int, dict]:
    """No-grad replay of the unedited B branch (same cached semantics); c* at each position."""
    import experiments.inference_cf_p2rj as rj
    from experiments.inference_cf_p2r import DiagBranch
    B = DiagBranch(ctx.bundle, encoded, ctx.cond["cB"], "B")
    new = list(ctx.cond["cB"])
    jobs = {p["t"]: p for p in positions}
    out = {}
    for t in range(max(jobs) + 1):
        lb, _, _ = rj.nograd_step(B, new, capture_layer=ctx.layer, attention=True)
        if t in jobs:
            out[t] = competitor_rule(lb, t, ctx.suppress, ctx.begin, jobs[t]["target_ids"])
        if t < len(tokens):
            new = [tokens[t]]
    return out


def with_competitors(positions: list[dict], comp: dict[int, dict]) -> tuple[list[dict], dict]:
    """Original positions keep the frozen c* (checked later); new positions take the rule's c*."""
    out, check = [], {}
    for p in positions:
        q = dict(p)
        rule = comp[p["t"]]["competitor"]
        check[str(p["t"])] = {**comp[p["t"]], "frozen": p["competitor"],
                              "rule_equals_frozen": None if p["competitor"] is None else rule == p["competitor"]}
        if q["competitor"] is None:
            q["competitor"] = rule
        out.append(q)
    return out, check


# ---- GPU run -----------------------------------------------------------------------------------

def cmd_run(args) -> None:
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core import validated_row
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import assert_no_site_hooks, num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2rj as rj

    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid P2-RJ-E manifest")
    for path, expected in manifest["sources"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != manifest["git_commit"]:
        raise ValueError("Git commit changed after P2-RJ-E manifest freeze")
    cfg = json.loads((ROOT / manifest["config"]).read_text())
    pos = json.loads((ROOT / manifest["positions"]).read_text())
    if pos["positions_hash"] != manifest["positions_hash"]:
        raise ValueError("positions hash mismatch")
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != manifest["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != int(cfg["site"]["num_forced_prefix"]):
        raise ValueError("forced prefix mismatch")
    bundle.model.requires_grad_(False)       # only the probe delta requires grad
    ctx = rj.Ctx(bundle, cfg, partition, nfp)
    base_dir = ROOT / cfg["baseline_run"]
    base_panel = json.loads((base_dir / "panel.json").read_text())["rows"]
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    base_idx = {r["utterance_id"]: i for i, r in enumerate(base_panel)}
    torch.cuda.reset_peak_memory_stats()
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running", "passes": {}}
    atomic_json(out / "runtime.json", runtime)
    for mode in rj.PASSES:
        if mode == "fp32":
            bundle.model.float()
            if next(bundle.model.parameters()).dtype != torch.float32:
                raise RuntimeError("fp32 cast failed")
        t_pass = time.time()
        for i, uid in enumerate(pos["utterances"]):
            dest = out / "rows" / f"{i:03d}_{mode}.json"
            if validated_row(dest, uid, manifest["manifest_hash"]):
                continue
            plist = [p for p in pos["positions"] if p["utterance_id"] == uid]
            t0 = time.time()
            try:
                bsys = json.loads((base_dir / "rows" / f"{base_idx[uid]:03d}.json").read_text())["systems"][cfg["baseline_system"]]
                inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
                with torch.no_grad():
                    feats = inputs["input_features"].to(next(bundle.model.parameters()).dtype)
                    enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                        input_features=feats, attention_mask=inputs["attention_mask"]).last_hidden_state)
                if mode == "bf16":
                    comp = competitor_prepass(ctx, encoded=enc, tokens=bsys["tokens"], positions=plist)
                else:                                  # c* is fixed by the bf16 pass
                    prev = json.loads((out / "rows" / f"{i:03d}_bf16.json").read_text())
                    comp = {int(t): {"competitor": c["competitor"], "tied_non_reference": c["tied_non_reference"]}
                            for t, c in prev["competitor_check"].items()}
                plist_c, check = with_competitors(plist, comp)
                recs, vecs = rj.run_utterance(ctx, encoded=enc, uid=uid, tokens=bsys["tokens"], positions=plist_c, mode=mode)
                assert_no_site_hooks(bundle)
                np.savez_compressed(out / "rows" / f"{i:03d}_{mode}.npz", **vecs)
                res = {"status": "ok", "positions": recs, "competitor_check": check,
                       "vectors_sha256": file_hash(out / "rows" / f"{i:03d}_{mode}.npz")}
            except Exception as exc:
                import traceback
                res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
            res.update(identity=uid, manifest_hash=manifest["manifest_hash"], mode=mode, elapsed_sec=time.time() - t0)
            atomic_json(dest, res)
            print(f"P2-RJ-E {mode} {i + 1}/{len(pos['utterances'])} {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
        runtime["passes"][mode] = {"elapsed_sec": time.time() - t_pass}
        atomic_json(out / "runtime.json", runtime)
    runtime.update(end_unix=time.time(), status="completed", peak_vram_bytes=torch.cuda.max_memory_allocated())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("positions")
    m = sub.add_parser("manifest"); m.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"positions": cmd_positions, "manifest": cmd_manifest, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
