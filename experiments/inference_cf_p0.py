#!/usr/bin/env python
"""Single physical P0 GPU run: aligned states and K=1 crossed support."""
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
from csasr.inference_cf.core import (VERSION, aligned_input, atomic_json, cache_key, generated_content,
                                      condition_tokens, conditions_identical, digest,
                                      support, validated_row)
from csasr.lss.sites import DecoderPostCrossAttnRecorder
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.utils.config import load_config


def encoder(bundle, path):
    inputs = batch_model_inputs(bundle, [path])
    with torch.inference_mode():
        enc = bundle.model.model.encoder(input_features=inputs["input_features"],
                                         attention_mask=inputs["attention_mask"], return_dict=True)
    return inputs, BaseModelOutput(last_hidden_state=enc.last_hidden_state)


def predict(bundle, enc, prompt, content, capture=False):
    tokens = aligned_input(prompt, content)
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
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return logp, h, {"prompt_length": len(prompt), "content_prefix_length": len(content),
                     "decoder_mask": [1] * len(tokens), "cache_used": False,
                     "cache_positions_full_replay": list(range(len(tokens))),
                     "prediction_query_absolute_index": len(tokens)-1,
                     "logical_next_content_position": len(content), "beam_lineage": "greedy:0"}


def geometry(h0, hm, he):
    a = torch.tensor(h0, dtype=torch.float64)
    m = torch.tensor(hm, dtype=torch.float64)
    e = torch.tensor(he, dtype=torch.float64)
    delta = e-m
    dn = float(torch.linalg.vector_norm(delta))
    d = delta/(dn+1e-6)
    rho = float(2 * torch.dot(a-(m+e)/2, d)/(dn+1e-6))
    return {"h0_norm": float(torch.linalg.vector_norm(a)),
            "hM_norm": float(torch.linalg.vector_norm(m)),
            "hE_norm": float(torch.linalg.vector_norm(e)),
            "delta_norm": dn, "direction_norm": float(torch.linalg.vector_norm(d)),
            "c0_cM_residual_norm": float(torch.linalg.vector_norm(a-m)),
            "rho": rho, "collapse_factor_diagnostic": max(0., -max(-1., min(1., rho))),
            "finite": bool(torch.isfinite(torch.stack([a,m,e])).all()),
            "zero_delta": dn == 0.}


def scores_from_logits(le, lm, ye, ym):
    return {"sE_yE": float(le[ye]), "sE_yM": float(le[ym]),
            "sM_yE": float(lm[ye]), "sM_yM": float(lm[ym])}


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
    p0 = conditions["c0"]
    eos = bundle.processor.tokenizer.eos_token_id
    content = generated_content(baseline_seq, eos)
    if not content:
        return {"status": "skip", "reason": "empty_baseline_content", "baseline_sequence": baseline_seq}
    logical = min(len(content)-1, max(0, int(math.floor(position_fraction * len(content)))))
    shared = content[:logical]
    logits, states, align = {}, {}, {}
    for condition in ("c0", "cM", "cE"):
        logits[condition], states[condition], align[condition] = predict(
            bundle, real_enc, conditions[condition], shared, capture=True)
    ye = int(torch.argmax(logits["cE"]))
    ym = int(torch.argmax(logits["cM"]))
    scores = scores_from_logits(logits["cE"], logits["cM"], ye, ym)
    real_support = support(scores, ye, ym)
    se, _, _ = predict(bundle, shuffled_enc, conditions["cE"], shared)
    sm, _, _ = predict(bundle, shuffled_enc, conditions["cM"], shared)
    shuffled_scores = scores_from_logits(se, sm, ye, ym)
    shuffled_support = support(shuffled_scores, ye, ym)
    geom = geometry(states["c0"], states["cM"], states["cE"])
    finite = geom["finite"] and all(math.isfinite(x) for x in scores.values())
    return {"status": "ok" if finite else "skip", "reason": None if finite else "nonfinite",
            "baseline_sequence": baseline_seq, "baseline_text": baseline_text,
            "shared_content_prefix": shared, "logical_position": logical,
            "actual_baseline_next_token_diagnostic": content[logical],
            "conditions": conditions, "alignment": align, "states": states,
            "geometry": geom, "c0_equals_cM_by_tokens": conditions_identical(conditions),
            "candidates": {"yE": ye, "yM": ym,
                           "yE_text": bundle.processor.tokenizer.decode([ye]),
                           "yM_text": bundle.processor.tokenizer.decode([ym]),
                           "yE_is_eos": ye == eos, "yM_is_eos": ym == eos,
                           "yE_is_special": ye in bundle.processor.tokenizer.all_special_ids,
                           "yM_is_special": ym in bundle.processor.tokenizer.all_special_ids},
            "normalization": "single next-token log_softmax, K=1, no EOS included beyond candidate",
            "real_scores": scores, "real_support": real_support,
            "shuffled_scores": shuffled_scores, "shuffled_support": shuffled_support,
            "cache_keys": {c: cache_key(model_revision=manifest["model_revision"],
                                         audio_sha256=audio_sha256, condition=c,
                                         prompt_tokens=conditions[c], content_prefix=shared,
                                         logical_position=logical, layer=24,
                                         site=manifest["site"], beam_lineage="greedy:0",
                                         scorer_version=VERSION, decode_config=manifest["decode"])
                           for c in conditions},
            "model_evaluations": {"encoder": 2, "baseline_generate": 1,
                                  "real_decoder_full_replay": 3,
                                  "shuffled_decoder_full_replay": 2},
            "output_tokens": len(content)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/inference_cf/p0")
    args = p.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    assert digest({k:v for k,v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"]
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
               "gpu": torch.cuda.get_device_name(0), "gpu_device_properties": str(torch.cuda.get_device_properties(0)),
               "model": bundle.metadata(), "conditions": conditions, "start_unix": time.time()}
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
        result["elapsed_sec"] = time.time()-started
        result["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(path, result)
        print(f"P0 {i+1}/{len(panel['rows'])} {identity} {result['status']} {result['elapsed_sec']:.1f}s", flush=True)
    runtime["end_unix"] = time.time()
    runtime["elapsed_sec"] = runtime["end_unix"]-runtime["start_unix"]
    runtime["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
