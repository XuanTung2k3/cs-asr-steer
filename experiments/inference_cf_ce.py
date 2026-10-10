#!/usr/bin/env python
"""Pre-P2 cached-equivalence run: cached steering vs the accepted P1 full-replay definition,
compared at every step on the same prefix (spec PRE_P2_CACHED_EQUIVALENCE_SPEC.md)."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from transformers.modeling_outputs import BaseModelOutput

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_p1 import direction, processed_argmax, reference_edit, selected_gate
from csasr.inference_cf.core_r2 import (conflict_from_logits, local_support, max_attention_window,
                                        prefix_utf8_complete, tokenizer_partition)
from csasr.lss.sites import (DecoderPostCrossAttnInterventionHook, DecoderPostCrossAttnRecorder,
                             assert_no_site_hooks, num_forced_prefix_from)
from csasr.models.hooks import apply_steering
from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
from csasr.utils.config import load_config
import experiments.inference_cf_cached as cached
import experiments.inference_cf_p0_r2 as r2

SCHEMA = "pre_p2_cached_equivalence_v1_1"
LAYERS = (16, 24)
ALPHAS = (0.0, 1.0)
INDICES = list(range(0, 300, 30))


@torch.inference_mode()
def replay(bundle, encoded, tokens, layer, *, attention):
    ids = torch.tensor([tokens], device=bundle.device, dtype=torch.long)
    with DecoderPostCrossAttnRecorder(bundle, [layer], keep_last_only=True) as rec:
        out = bundle.model(encoder_outputs=encoded, decoder_input_ids=ids,
                           decoder_attention_mask=torch.ones_like(ids), use_cache=False,
                           output_attentions=attention, return_dict=True)
    heads = None
    if attention:
        heads = torch.stack([out.cross_attentions[l][0, h, -1].float().cpu()
                             for l, h in bundle.model.generation_config.alignment_heads])
    return out.logits[0, -1].float().cpu(), heads, rec.states[layer][0, -1].float().cpu()


@torch.inference_mode()
def steered_replay(bundle, encoded, tokens, edits, alpha, layer, nfp):
    n = len(tokens)
    gain = torch.zeros((1, n))
    dirs = torch.zeros((1, n, int(bundle.d_model)))
    for pos, (g, d) in edits.items():
        gain[0, pos], dirs[0, pos] = g, d
    hook = DecoderPostCrossAttnInterventionHook(
        bundle, layer, None, alpha=alpha, num_forced_prefix=nfp, norm_preserve=True, record=True,
        record_last_only=True,
        action_fn=lambda q, u_source, r, abs_pos: (gain.to(r.device, r.dtype), dirs.to(r.device, r.dtype)))
    ids = torch.tensor([tokens], device=bundle.device, dtype=torch.long)
    with hook:
        out = bundle.model(encoder_outputs=encoded, decoder_input_ids=ids,
                           decoder_attention_mask=torch.ones_like(ids), use_cache=False,
                           output_attentions=True, return_dict=True)
    return out.logits[0, -1].float().cpu(), (hook.records[-1].to_dict() if hook.records else None)


def replay_gate(bundle, prefix, heads, logits_b, waveform, partition, null_probs, language_ids,
                conditions, lid_cache, lid_key, byte_decoder):
    tok = bundle.processor.tokenizer
    en_id, zh_id = conditions["language_token_ids"]
    out = {"fallback_reason": None, "window": None, "E": None, "R_B": None, "g": 0.0}
    reasons = []
    if not prefix_utf8_complete(tok, prefix, byte_decoder):
        reasons.append("mid_character")
    window = None
    try:
        window = max_attention_window(heads.numpy(), len(waveform))
        out["window"] = [window.start_frame, window.end_frame, window.start_sample, window.end_sample]
    except Exception as exc:
        reasons.append(r2._reason(exc, "localizer_fail"))
    baseline = None
    try:
        baseline = conflict_from_logits(logits_b, partition)
        out["R_B"] = baseline["R"]
    except Exception as exc:
        reasons.append(r2._reason(exc, "baseline_provider_fail"))
    if not reasons:
        key = (lid_key, window.start_sample, window.end_sample)
        if key not in lid_cache:
            lid_cache[key] = cached.native_lid(bundle, waveform[key[1]:key[2]], language_ids)
        probs = lid_cache[key]
        support = local_support(probs[en_id], probs[zh_id], null_probs[en_id], null_probs[zh_id])
        out["E"] = support["E"]
        out["g"] = float(selected_gate(support, baseline))
    priority = ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail")
    out["fallback_reason"] = next((r for r in priority if r in reasons), None)
    if out["fallback_reason"] is not None:
        out["g"] = 0.0
    return out


def top2_margin(logits, t, suppress, begin):
    x = cached._processed(logits, t, suppress, begin)
    v = torch.topk(x, 2).values
    return float(v[0] - v[1])


def ce_decode(bundle, *, enc, waveform, uid, layer, alpha, conditions, partition, null_probs,
              language_ids, nfp, suppress, begin, byte_decoder, lid_cache, max_new_tokens=200):
    """One cached decode with a per-step full-replay comparison on the same prefix."""
    comps, replay_edits = [], {}
    probe_time = [0.0]

    def probe(*, t, prefix, step, h_b, h_e, d, logits_b, logits_s):
        p0 = time.time()
        b_in = conditions["cB"] + prefix
        lb, heads, hb_r = replay(bundle, enc, b_in, layer, attention=True)
        _, _, he_r = replay(bundle, enc, conditions["cE"] + prefix, layer, attention=False)
        rg = replay_gate(bundle, prefix, heads, lb, waveform, partition, null_probs,
                         language_ids, conditions, lid_cache, uid, byte_decoder)
        dr = direction(he_r, hb_r)
        query = len(b_in) - 1
        edit_r = alpha * rg["g"] > 0 and dr["d"] is not None and query >= nfp
        if edit_r:
            replay_edits[query] = (rg["g"], dr["d"])
        ls_r, audit_r = steered_replay(bundle, enc, b_in, replay_edits, alpha, layer, nfp)
        rec = {"t": t, "query": query,
               "fb": [step["fallback_reason"], rg["fallback_reason"]],
               "window": [step["window"], rg["window"]],
               "R_B": [(step["baseline"] or {}).get("R"), rg["R_B"]],
               "E": [(step["local_support"] or {}).get("E"), rg["E"]],
               "g": [step["g"], rg["g"]],
               "dir_status": [step["direction_status"], dr["status"]],
               "dir_norm": [step["direction_norm"], dr["norm"]],
               "dir_cos": float(torch.nn.functional.cosine_similarity(
                   (h_e - h_b).double(), (he_r - hb_r).double(), dim=0)),
               "edit": [step["edit_applied"], bool(edit_r)],
               "audit": [step["hook_audit"], audit_r],
               "argmax": [step["steered_next"], processed_argmax(ls_r, t, suppress, begin)],
               "margin": [step["steered_top2_margin"], top2_margin(ls_r, t, suppress, begin)],
               "max_abs_dlogit": float((logits_s - ls_r).abs().max()),
               "s_bitwise_b": step["f3_bitwise_equals_b"]}
        if step["edit_applied"]:
            dev, dt = bundle.device, getattr(bundle, "dtype", h_b.dtype)
            site = h_b.to(device=dev, dtype=dt).view(1, 1, -1)
            rep = apply_steering(site, d.to(dev, dt).view(1, 1, -1), alpha, 1.0,
                                 torch.tensor([[step["g"]]], device=dev, dtype=dt), True)
            rec["replica_edit_norm"] = float((rep - site).float().norm())
            rec["reference_cos"] = reference_edit(h_b, d, alpha, step["g"])["cos_edit_d"]
            # v1.1: each path's *specified* edit (float64, from its own logged inputs) for the
            # cross-path comparison; the realized bf16 norms carry NormPreserve scalar rounding.
            rec["spec_edit_norm_cached"] = reference_edit(h_b, d, alpha, step["g"])["edit_norm"]
        if edit_r:
            dev, dt = bundle.device, getattr(bundle, "dtype", hb_r.dtype)
            site_r = hb_r.to(device=dev, dtype=dt).view(1, 1, -1)
            rep_r = apply_steering(site_r, dr["d"].to(dev, dt).view(1, 1, -1), alpha, 1.0,
                                   torch.tensor([[rg["g"]]], device=dev, dtype=dt), True)
            rec["replay_replica_edit_norm"] = float((rep_r - site_r).float().norm())
            rec["spec_edit_norm_replay"] = reference_edit(hb_r, dr["d"], alpha, rg["g"])["edit_norm"]
        comps.append(rec)
        probe_time[0] += time.time() - p0

    res = cached.cached_decode(bundle, waveform=waveform, encoded=enc, conditions=conditions,
                               partition=partition, null_probs=null_probs,
                               language_ids=language_ids, layer=layer, alpha=alpha,
                               num_forced_prefix=nfp, lid_cache=lid_cache, lid_key=uid,
                               max_new_tokens=max_new_tokens, on_step=probe)
    return res, comps, probe_time[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/inference_cf/pre_p2_ce_r1")
    args = ap.parse_args()
    out = ROOT / args.out
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("tracked files dirty: commit before the equivalence run")
    if (out / "manifest.json").exists():
        raise FileExistsError("equivalence manifest exists; never overwrite a run")
    panel = json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]
    rows = [panel[i] for i in INDICES]
    sources = [ROOT / x for x in ("docs/inference_cf/PRE_P2_CACHED_EQUIVALENCE_SPEC.md",
                                  "experiments/inference_cf_cached.py", "experiments/inference_cf_ce.py",
                                  "experiments/inference_cf_ce_accept.py",
                                  "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py",
                                  "src/csasr/lss/sites.py", "src/csasr/models/hooks.py")]
    manifest = {"schema": SCHEMA, "git_commit": commit, "layers": list(LAYERS), "alphas": list(ALPHAS),
                "indices": INDICES, "utterances": [r["utterance_id"] for r in rows],
                "sources": {str(p): file_hash(p) for p in sources},
                "r2_manifest": json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())["manifest_hash"]}
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(out / "manifest.json", manifest)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    conditions = {"cB": [50258, 50260, 50360, 50364], "cE": [50258, 50259, 50360, 50364],
                  "language_token_ids": [50259, 50260]}
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    null_probs = cached.native_lid(bundle, np.zeros(480000, dtype=np.float32), language_ids)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    r2_rows = {json.loads(p.read_text())["identity"]: p for p in (ROOT / "results/inference_cf/p0_r2/rows").glob("*.json")}
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": t_start, "cached_sec": 0.0,
               "probe_sec": 0.0, "audio_sec": 0.0, "output_tokens": 0}
    for i, item in enumerate(rows):
        uid = item["utterance_id"]
        waveform = load_audio(item["audio_path"], bundle.sample_rate)
        runtime["audio_sec"] += min(len(waveform) / 16000, 30.0)
        inputs = batch_model_inputs(bundle, [item["audio_path"]])
        with torch.inference_mode():
            enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
        lid_cache = {}
        t0 = time.time()
        greedy = cached.cached_greedy(bundle, enc, conditions["cB"])
        runtime["cached_sec"] += time.time() - t0
        r2_generate = json.loads(r2_rows[uid].read_text())["baseline_content_ids"]
        for layer in LAYERS:
            for alpha in ALPHAS:
                t0 = time.time()
                res, comps, ptime = ce_decode(bundle, enc=enc, waveform=waveform, uid=uid, layer=layer,
                                              alpha=alpha, conditions=conditions, partition=partition,
                                              null_probs=null_probs, language_ids=language_ids, nfp=nfp,
                                              suppress=suppress, begin=begin, byte_decoder=byte_decoder,
                                              lid_cache=lid_cache)
                probe_time = [ptime]
                runtime["cached_sec"] += time.time() - t0 - probe_time[0]
                runtime["probe_sec"] += probe_time[0]
                runtime["output_tokens"] += len(res["tokens"])
                assert_no_site_hooks(bundle)
                shard = {"identity": f"{uid}|L{layer}|a{alpha}", "utterance_id": uid, "layer": layer,
                         "alpha": alpha, "manifest_hash": manifest["manifest_hash"],
                         "tokens": res["tokens"], "terminated": res["terminated"],
                         "lineage_ok": res["lineage_ok"], "distinct_caches": res["distinct_caches"],
                         "counters": res["counters"], "cached_greedy_tokens": greedy["tokens"],
                         "r2_generate_tokens": r2_generate,
                         "steps": [{k: s[k] for k in ("t", "query_index", "fallback_reason", "g", "dose",
                                                      "edit_applied", "hook_audit", "steered_next",
                                                      "unsteered_next", "f3_bitwise_equals_b")}
                                   for s in res["steps"]],
                         "comparisons": comps}
                atomic_json(out / "rows" / f"{i:02d}_L{layer}_a{alpha}.json", shard)
                print(f"CE {uid} L{layer} a{alpha} steps={len(res['steps'])}", flush=True)
    assert_no_site_hooks(bundle)
    runtime.update(end_unix=time.time(), elapsed_sec=time.time() - t_start,
                   peak_vram_bytes=torch.cuda.max_memory_allocated(), status="completed")
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
