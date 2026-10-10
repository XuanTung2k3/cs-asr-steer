#!/usr/bin/env python
"""Single physical P0-R1 GPU run: K=3 short-continuation crossed support.

Scientific change vs P0: K=1 argmax candidates -> K=3 greedy continuations scored by
token-average log probability. Site, conditions, panel, permutation, geometry, and the
diagnostic-position state capture are reused unchanged. No intervention; no P1 decoding.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch
from transformers.modeling_outputs import BaseModelOutput
from csasr.inference_cf.core_r1 import (VERSION_R1, K, aligned_input, atomic_json, cache_key,
                                        conditions_identical, digest, generated_content,
                                        condition_tokens, sequence_logprob, support_k3,
                                        truncate_continuation, validated_row)
from csasr.lss.sites import DecoderPostCrossAttnRecorder
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.utils.config import load_config


def encoder(bundle, path):
    inputs = batch_model_inputs(bundle, [path])
    with torch.inference_mode():
        enc = bundle.model.model.encoder(input_features=inputs["input_features"],
                                         attention_mask=inputs["attention_mask"], return_dict=True)
    return inputs, BaseModelOutput(last_hidden_state=enc.last_hidden_state)


def forward_logits(bundle, enc, tokens, capture=False):
    """One full-replay decoder forward (use_cache=False). Returns last-position log-probs,
    optionally the L24 site at the last query, and the full-sequence logits (float, cpu)."""
    ids = torch.tensor([tokens], dtype=torch.long, device=bundle.device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        if capture:
            with DecoderPostCrossAttnRecorder(bundle, [24]) as rec:
                out = bundle.model(encoder_outputs=enc, decoder_input_ids=ids,
                                   decoder_attention_mask=mask, use_cache=False, return_dict=True)
            h = rec.states[24][0, -1].float().cpu().tolist()
        else:
            out = bundle.model(encoder_outputs=enc, decoder_input_ids=ids,
                               decoder_attention_mask=mask, use_cache=False, return_dict=True)
            h = None
        logits = out.logits[0].float().cpu()
    last_logp = torch.log_softmax(logits[-1], dim=-1)
    return last_logp, h, logits


def continue_k3(bundle, enc, prompt, shared, eos, counter, capture=False):
    """Greedy K=3 continuation on the given encoder. Returns (candidate_tokens, state_h).
    Stops early at EOS; the frozen truncation rule then excludes EOS from the candidate."""
    steps, state = [], None
    for step in range(K):
        last_logp, h, _ = forward_logits(bundle, enc, aligned_input(prompt, shared + steps),
                                         capture=(capture and step == 0))
        counter[0] += 1
        if step == 0 and capture:
            state = h
        nxt = int(torch.argmax(last_logp))
        steps.append(nxt)
        if nxt == eos:
            break
    return truncate_continuation(steps, eos, K), state


def score_sequence(bundle, enc, prompt, shared, candidate, counter):
    """Teacher-forced per-step log-probs of `candidate` under one condition/encoder.
    query for candidate[j] is at absolute index len(prompt)+len(shared)+j-1."""
    tokens = aligned_input(prompt, shared + candidate)
    _, _, logits = forward_logits(bundle, enc, tokens)
    counter[0] += 1
    base = len(prompt) + len(shared)
    step_logprobs, query_indices = [], []
    for j in range(len(candidate)):
        q = base + j - 1
        step_logprobs.append(float(torch.log_softmax(logits[q], dim=-1)[candidate[j]]))
        query_indices.append(q)
    align = {"prompt_length": len(prompt), "content_prefix_length": len(shared),
             "candidate_length": len(candidate), "input_length": len(tokens),
             "continuation_query_indices": query_indices,
             "decoder_mask": [1] * len(tokens), "cache_used": False,
             "cache_positions_full_replay": list(range(len(tokens))), "beam_lineage": "greedy:0"}
    return step_logprobs, align


def geometry(h0, hm, he):
    a = torch.tensor(h0, dtype=torch.float64)
    m = torch.tensor(hm, dtype=torch.float64)
    e = torch.tensor(he, dtype=torch.float64)
    delta = e - m
    dn = float(torch.linalg.vector_norm(delta))
    d = delta / (dn + 1e-6)
    rho = float(2 * torch.dot(a - (m + e) / 2, d) / (dn + 1e-6))
    return {"h0_norm": float(torch.linalg.vector_norm(a)),
            "hM_norm": float(torch.linalg.vector_norm(m)),
            "hE_norm": float(torch.linalg.vector_norm(e)),
            "delta_norm": dn, "direction_norm": float(torch.linalg.vector_norm(d)),
            "c0_cM_residual_norm": float(torch.linalg.vector_norm(a - m)),
            "rho": rho, "collapse_factor_diagnostic": max(0., -max(-1., min(1., rho))),
            "finite": bool(torch.isfinite(torch.stack([a, m, e])).all()),
            "zero_delta": dn == 0.}


def diagnostic_state(bundle, enc, prompt, shared, counter):
    _, h, _ = forward_logits(bundle, enc, aligned_input(prompt, shared), capture=True)
    counter[0] += 1
    return h


def cross_scores(bundle, enc, conditions, shared, yE, yM, counter):
    """Score BOTH K=3 candidates under BOTH conditions on one encoder."""
    steps, aligns = {}, {}
    steps["sE_yE"], aligns["sE_yE"] = score_sequence(bundle, enc, conditions["cE"], shared, yE, counter)
    steps["sE_yM"], aligns["sE_yM"] = score_sequence(bundle, enc, conditions["cE"], shared, yM, counter)
    steps["sM_yE"], aligns["sM_yE"] = score_sequence(bundle, enc, conditions["cM"], shared, yE, counter)
    steps["sM_yM"], aligns["sM_yM"] = score_sequence(bundle, enc, conditions["cM"], shared, yM, counter)
    scores = {k: sequence_logprob(v) for k, v in steps.items()}
    return scores, steps, aligns


def inference_row(bundle, *, audio_path, shuffled_audio_path, position_fraction,
                  frozen_baseline_text, audio_sha256, conditions, manifest):
    # This API accepts no reference text, true next token, language label or POI.
    real_inputs, real_enc = encoder(bundle, audio_path)
    _, shuffled_enc = encoder(bundle, shuffled_audio_path)
    with torch.inference_mode():
        baseline_seq = bundle.model.generate(**real_inputs, **manifest["decode"])[0].tolist()
    baseline_text = bundle.processor.batch_decode([baseline_seq], skip_special_tokens=True)[0].strip()
    if baseline_text != frozen_baseline_text:
        return {"status": "skip", "reason": "frozen_baseline_text_mismatch",
                "fresh_baseline_text": baseline_text, "frozen_baseline_text": frozen_baseline_text}
    eos = bundle.processor.tokenizer.eos_token_id
    content = generated_content(baseline_seq, eos)
    if not content:
        return {"status": "skip", "reason": "empty_baseline_content", "baseline_sequence": baseline_seq}
    logical = min(len(content) - 1, max(0, int(math.floor(position_fraction * len(content)))))
    shared = content[:logical]
    real_calls, shuf_calls = [0], [0]
    # diagnostic-position states (G1/G2 continuity): h0 from c0, hM/hE from the K=3 step-0 forwards
    h0 = diagnostic_state(bundle, real_enc, conditions["c0"], shared, real_calls)
    yE, hE = continue_k3(bundle, real_enc, conditions["cE"], shared, eos, real_calls, capture=True)
    yM, hM = continue_k3(bundle, real_enc, conditions["cM"], shared, eos, real_calls, capture=True)
    if not yE or not yM:
        return {"status": "skip", "reason": "empty_continuation",
                "yE": yE, "yM": yM, "shared_content_prefix": shared, "logical_position": logical}
    scores, real_steps, real_aligns = cross_scores(bundle, real_enc, conditions, shared, yE, yM, real_calls)
    shuffled_scores, shuf_steps, _ = cross_scores(bundle, shuffled_enc, conditions, shared, yE, yM, shuf_calls)
    real_support = support_k3(scores, yE, yM)
    shuffled_support = support_k3(shuffled_scores, yE, yM)
    geom = geometry(h0, hM, hE)
    finite = geom["finite"] and all(math.isfinite(x) for x in list(scores.values()) + list(shuffled_scores.values()))
    tok = bundle.processor.tokenizer
    return {"status": "ok" if finite else "skip", "reason": None if finite else "nonfinite",
            "baseline_sequence": baseline_seq, "baseline_text": baseline_text,
            "shared_content_prefix": shared, "logical_position": logical,
            "actual_baseline_next_token_diagnostic": content[logical],
            "conditions": conditions, "states": {"c0": h0, "cM": hM, "cE": hE},
            "geometry": geom, "c0_equals_cM_by_tokens": conditions_identical(conditions),
            "candidates": {"K": K, "yE": yE, "yM": yM,
                           "yE_text": tok.decode(yE), "yM_text": tok.decode(yM),
                           "yE_length": len(yE), "yM_length": len(yM),
                           "yE_has_special": any(t in tok.all_special_ids for t in yE),
                           "yM_has_special": any(t in tok.all_special_ids for t in yM),
                           "full_sequence_collision": list(yE) == list(yM),
                           "longest_common_prefix": real_support["lcp"]},
            "normalization": "token-average log prob over K=3 greedy continuation; EOS excluded by truncation",
            "real_scores": scores, "real_step_logprobs": real_steps, "real_scoring_alignment": real_aligns,
            "real_support": real_support,
            "shuffled_scores": shuffled_scores, "shuffled_step_logprobs": shuf_steps,
            "shuffled_support": shuffled_support,
            "continuation_alignment": {
                c: {"prompt_length": len(conditions[c]), "content_prefix_length": len(shared),
                    "logical_next_content_position": logical, "cache_used": False,
                    "beam_lineage": "greedy:0"} for c in conditions},
            "cache_keys": {c: cache_key(model_revision=manifest["model_revision"],
                                        audio_sha256=audio_sha256, condition=c,
                                        prompt_tokens=conditions[c], content_prefix=shared,
                                        logical_position=logical, layer=24,
                                        site=manifest["site"], beam_lineage="greedy:0",
                                        scorer_version=VERSION_R1, decode_config=manifest["decode"])
                           for c in conditions},
            "model_evaluations": {"encoder": 2, "baseline_generate": 1,
                                  "real_decoder_forwards": real_calls[0],
                                  "shuffled_decoder_forwards": shuf_calls[0]},
            "output_tokens": len(content)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/inference_cf/p0_r1")
    args = p.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema"] == VERSION_R1 and manifest["candidate_k"] == K
    assert digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"]
    panel = json.loads((out / "panel.json").read_text())
    perm = json.loads((out / "audio_permutation.json").read_text())
    assert digest(panel) == manifest["panel_hash"] and digest(perm) == manifest["permutation_hash"]
    rows = {r["identity"]: r for r in panel["rows"]}
    cfg = load_config(ROOT / "configs/model/whisper_large_v3.yaml")
    bundle = load_whisper(cfg)
    assert bundle.revision == manifest["model_revision"] and bundle.tokenizer_revision == manifest["tokenizer_revision"]
    conditions = condition_tokens(bundle.processor)
    assert conditions == manifest["conditions"]["prompt_tokens"]
    runtime = {"job_id": os.getenv("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0),
               "gpu_device_properties": str(torch.cuda.get_device_properties(0)),
               "model": bundle.metadata(), "conditions": conditions, "candidate_k": K,
               "start_unix": time.time()}
    atomic_json(out / "runtime.json", runtime)
    for i, row in enumerate(panel["rows"]):
        identity = row["identity"]
        path = out / "rows" / f"{i:03d}.json"
        if validated_row(path, identity, manifest["manifest_hash"]):
            continue
        started = time.time()
        result = {"identity": identity, "stratum": row["stratum"],
                  "utterance_id": row["utterance_id"], "manifest_hash": manifest["manifest_hash"],
                  "audio_sha256": row["audio_sha256"], "duration_sec": row["duration_sec"],
                  "shuffled_identity": perm[identity],
                  "shuffled_audio_sha256": rows[perm[identity]]["audio_sha256"],
                  "selection_unit_index": row["selection_unit_index"],
                  "selection_surface_evaluator_only": row["selection_surface"],
                  "position_fraction": row["position_fraction"]}
        try:
            result.update(inference_row(bundle, audio_path=row["audio_path"],
                                        shuffled_audio_path=rows[perm[identity]]["audio_path"],
                                        position_fraction=row["position_fraction"],
                                        frozen_baseline_text=row["frozen_baseline_text"],
                                        audio_sha256=row["audio_sha256"],
                                        conditions=conditions, manifest=manifest))
        except Exception as exc:
            result.update(status="failure", reason=repr(exc), traceback=traceback.format_exc())
        result["elapsed_sec"] = time.time() - started
        result["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(path, result)
        print(f"P0R1 {i+1}/{len(panel['rows'])} {identity} {result['status']} {result['elapsed_sec']:.1f}s", flush=True)
    runtime["end_unix"] = time.time()
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    runtime["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
