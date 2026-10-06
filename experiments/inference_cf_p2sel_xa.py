#!/usr/bin/env python
"""P2-SEL-XA runner (separate inference_cf development diagnostic). Frozen contract:
docs/inference_cf/P2_SEL_XA_SPEC.md, P2_SEL_XA_CODEX_DESIGN.md, configs/inference_cf/p2_sel_xa.json (595e174).

``xa0``      GPU, NO steering / backward / LID: replay the unedited forced-ZH B prefix to each frozen query;
             before the clean step clone the pre-step cache (readout.clone_scratch); take the clean step with a
             passive DG-02 recorder to capture q, u_source and r (bitwise vs the sealed P2-DIR h_B and NONE
             logits); then one isolated scratch forward where only this query's encoder_attn output[0] is
             replaced by (0.95*u).to(model dtype); keep r_0.95 and its raw logits, discard the scratch.
             The sealed P2-DIR raw readout gradient g_J is reused (no new autograd).
``manifest`` immutable stage manifest (CPU).
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

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "p2_sel_xa_v1"
CONFIG = "configs/inference_cf/p2_sel_xa.json"
CONSTRUCTION = "results/inference_cf/p2dir/construction_population.json"
P2DIR_RUN = "results/inference_cf/p2dir/exp1_run1"
BASE = "results/inference_cf/p2sel_xa"
LAYER = 16
LAMBDA = 0.95


class SourceScale:
    """Replace ONLY the last (current) query's encoder_attn output[0] at LAYER by (lam*u).to(dtype);
    record q, the scaled u and r' = q + u' in model dtype. q and all other positions/layers unchanged."""

    def __init__(self, bundle, layer: int = LAYER, lam: float = LAMBDA):
        from csasr.lss.sites import assert_dropout_disabled
        self.bundle, self.layer, self.lam = bundle, int(layer), float(lam)
        assert_dropout_disabled(bundle, [self.layer])
        self._h, self._q = [], None
        self.q = self.u = self.r = None
        self.calls = 0

    def _pre(self, _m, args):
        self._q = args[0]

    def _post(self, _m, _i, output):
        from csasr.lss.sites import _first, _rebuild
        u = _first(output)
        u2 = u.clone()
        u2[:, -1] = (self.lam * u[:, -1]).to(u.dtype)
        self.q, self.u = self._q[0, -1].detach().clone(), u2[0, -1].detach().clone()
        self.r = (self._q + u2)[0, -1].detach().clone()
        self.calls += 1
        return _rebuild(output, u2)

    def __enter__(self):
        layer = self.bundle.decoder_layer(self.layer)
        self._h = [layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre),
                   layer.encoder_attn.register_forward_hook(self._post)]
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h, self._q = [], None
        return False


def xa0_utterance(bundle, *, encoded, tokens, ts, cB, ref_vecs, ref_none) -> tuple[dict, dict, dict]:
    import torch
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.readout import cache_fingerprint, clone_scratch
    from csasr.lss.sites import DecoderPostCrossAttnRecorder, assert_no_site_hooks
    B = p2r.DiagBranch(bundle, encoded, cB, "B")
    new = list(cB)
    recs, vecs, logits = {}, {}, {}
    dev = next(bundle.model.parameters()).device
    for t in range(max(ts) + 1):
        L = B.length
        scratch = enc = None
        if t in ts:
            fp = cache_fingerprint(B.cache)
            scratch, enc = clone_scratch(B.cache, encoded)                       # pre-step, isolated
        rec = DecoderPostCrossAttnRecorder(bundle, [LAYER], keep_last_only=True) if t in ts else None
        lb, _, hb = B.step(new, capture_layer=LAYER, attention=True, hook=rec)
        assert_no_site_hooks(bundle)
        if t in ts:
            pre = f"t{t}_"
            qg, ug, rg = rec.q_states[LAYER][0, -1], rec.u_source_states[LAYER][0, -1], rec.states[LAYER][0, -1]
            native_ok = bool(torch.equal((qg.to(torch.bfloat16) + ug.to(torch.bfloat16)).float(), rg))   # model-dtype sum
            q, u, r = qg.cpu(), ug.cpu(), rg.cpu()
            sc = SourceScale(bundle)
            with torch.no_grad(), sc:
                out = bundle.model(encoder_outputs=enc, decoder_input_ids=torch.tensor([list(new)], device=dev),
                                   past_key_values=scratch, use_cache=True,
                                   cache_position=torch.arange(L, L + len(new), device=dev), return_dict=True)
            assert_no_site_hooks(bundle)
            l95 = out.logits[0, -1].float().cpu()
            del out, scratch, enc
            recs[str(t)] = {"t": t, "query": L + len(new) - 1, "scale_calls": sc.calls,
                            "native_sum_bitwise_r": native_ok,
                            "r_bitwise_capture": bool(torch.equal(r, hb)),
                            "r_bitwise_p2dir_hb": bool(np.array_equal(hb.numpy(), ref_vecs[pre + "hb"])),
                            "logits_bitwise_p2dir_none": bool(np.array_equal(p2dir.pack_bf16(lb), ref_none[pre + "none"])),
                            "q_unchanged_in_scratch": bool(torch.equal(sc.q.float().cpu(), q)),
                            "cache_object_and_length": cache_fingerprint(B.cache)[0] == fp[0] and B.length == L + len(new)}
            vecs[pre + "q"], vecs[pre + "u"], vecs[pre + "r1"] = q.numpy(), u.numpy(), r.numpy()
            vecs[pre + "r095"] = sc.r.float().cpu().numpy()
            vecs[pre + "u095"] = sc.u.float().cpu().numpy()
            logits[pre + "l1"], logits[pre + "l095"] = p2dir.pack_bf16(lb), p2dir.pack_bf16(l95)
        if t < len(tokens):
            new = [tokens[t]]
    return recs, vecs, logits


# ---- manifest / job ----------------------------------------------------------------------------------

SOURCES = ("docs/inference_cf/P2_SEL_XA_SPEC.md", "docs/inference_cf/P2_SEL_XA_CODEX_DESIGN.md", CONFIG,
           "docs/inference_cf/P2_SEL_XA_PRE_RUN_AUDIT.md", "results/inference_cf/p2rj/positions.json", CONSTRUCTION,
           "src/csasr/inference_cf/source_compatibility.py", "src/csasr/inference_cf/readout.py", "src/csasr/lss/sites.py",
           "src/csasr/models/hooks.py", "experiments/inference_cf_cached.py", "experiments/inference_cf_p2r.py",
           "experiments/inference_cf_p2dir.py", "experiments/inference_cf_p2sel.py", "experiments/inference_cf_p2sel_xa.py",
           "experiments/inference_cf_p2sel_xa_analyze.py", "experiments/inference_cf_p2sel_xa_audit.py",
           "slurm/inference_cf_p2sel_xa.sbatch", f"{P2DIR_RUN}/directions_sealed.json", f"{P2DIR_RUN}/manifest.json",
           "results/inference_cf/p2sel_e/e0_run1_analysis.json", "results/inference_cf/p2sel_t/t0_run1_analysis.json")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2sel_e as pe
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    extra = [f"{BASE}/prerun_audit.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *extra)
    if dirty:
        raise ValueError("commit P2-SEL-XA sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / extra[0]).read_text())["verdict"] != "PASS_TO_P2_SEL_XA_XA0":
        raise ValueError("pre-XA0 audit did not pass")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    man = {"schema": SCHEMA, "stage": args.stage, "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config": CONFIG, "config_hash": digest(cfg), "construction": CONSTRUCTION, "construction_hash": con["construction_hash"],
           "partition_hash": partition["hash"], "partition": partition,
           "suppression": {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])},
           "provider_integrity": pe.provider_integrity(str(prep.MODEL / "generation_config.json")),
           "precision": {"model": "bfloat16", "site_records": "float32 lossless upcast", "lambda": LAMBDA, "layer": LAYER},
           "role": "D-dev-select", "firewall": prep.firewall(con["utterances"]), "environment": prep.environment(),
           "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + tuple(extra)}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"stage": args.stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2sel as p2sel
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or m["stage"] != "xa0" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("git commit changed")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    _, p2d_vecs, p2d_none = p2sel.load_p2dir_rows(con)                         # sealed-hash verified
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.requires_grad_(False)
    base_dir = ROOT / p2dir.BASELINE_RUN
    bidx = {r["utterance_id"]: i for i, r in enumerate(json.loads((base_dir / "panel.json").read_text())["rows"])}
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / p2dir.AUDIO_PANEL).read_text())["rows"]}
    by_utt: dict = {}
    for p in con["positions"]:
        by_utt.setdefault(p["utterance_id"], []).append(int(p["t"]))
    torch.cuda.reset_peak_memory_stats()
    runtime = {"stage": "xa0", "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running",
               "autograd_calls": 0, "readout_calls": 0, "steering_calls": 0, "lid_calls": 0, "scale_forwards": 0}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "runtime.json", runtime)
    failures = 0
    for i, uid in enumerate(con["utterances"]):
        ts = sorted(by_utt[uid])
        toks = json.loads((base_dir / "rows" / f"{bidx[uid]:03d}.json").read_text())["systems"][p2dir.BASELINE_SYSTEM]["tokens"]
        t0 = time.time()
        try:
            inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                    input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
                recs, vecs, logits = xa0_utterance(bundle, encoded=enc, tokens=toks, ts=ts, cB=p2dir.CB,
                                                   ref_vecs=p2d_vecs[uid], ref_none=p2d_none[uid])
            runtime["scale_forwards"] += len(recs)
            np.savez_compressed(out / "rows" / f"{i:03d}.npz", **vecs)
            np.savez_compressed(out / "rows" / f"{i:03d}_logits.npz", **logits)
            res = {"status": "ok", "positions": recs, "vectors_sha256": file_hash(out / "rows" / f"{i:03d}.npz"),
                   "logits_sha256": file_hash(out / "rows" / f"{i:03d}_logits.npz")}
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=m["manifest_hash"], elapsed_sec=time.time() - t0)
        atomic_json(out / "rows" / f"{i:03d}.json", res)
        print(f"P2-SEL-XA xa0 {i + 1}/{len(con['utterances'])} {uid} {res['status']}", flush=True)
    if any(p.grad is not None for p in bundle.model.parameters()):
        raise RuntimeError("parameter gradient present")
    runtime.update(end_unix=time.time(), status="completed" if failures == 0 else "failed", failures=failures,
                   peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if failures:
        raise SystemExit(f"{failures} failures")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mf = sub.add_parser("manifest")
    mf.add_argument("--stage", required=True, choices=("xa0",))
    mf.add_argument("--out", required=True)
    sub.add_parser("xa0").add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_manifest(args) if args.cmd == "manifest" else run(args)


if __name__ == "__main__":
    main()
