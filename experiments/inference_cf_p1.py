#!/usr/bin/env python
"""P1 causal-acceptance decode: R2-selected gate g_old = E*R_B, direction norm(hE - hB),
single DG-02 site (decoder L24 post-cross-attention, pre-FFN), NormPreserve, exact zero-dose
bypass. Not a recognition-optimization experiment; no layer/alpha search.
"""
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

from csasr.inference_cf.core import atomic_json, digest, file_hash, validated_row
from csasr.inference_cf.core_r2 import (conflict_from_logits, local_support, max_attention_window,
                                        prefix_utf8_complete, tokenizer_partition)
from csasr.inference_cf.core_p1 import (LAYER, VERSION_P1, direction, processed_argmax,
                                        reference_edit, selected_gate, step_inputs)
from csasr.lss.sites import (DecoderPostCrossAttnInterventionHook, DecoderPostCrossAttnRecorder,
                             assert_no_site_hooks, num_forced_prefix_from)
from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
from csasr.utils.config import load_config
import experiments.inference_cf_p0_r2 as r2


def native_lid(bundle, waveform: np.ndarray, language_ids: tuple[int, ...]) -> dict[int, float]:
    """Frozen R2 LocalSupport provider (unchanged)."""
    return r2.native_lid(bundle, waveform, language_ids)


@torch.inference_mode()
def replay(bundle, encoded, tokens: list[int], *, attention: bool, capture: bool):
    """Unsteered full causal replay (use_cache=False) of prompt + prefix.

    Returns last-query logits, the current query's alignment-head attention, and the L24 site.
    """
    ids = torch.tensor([tokens], device=bundle.device, dtype=torch.long)
    ctx = DecoderPostCrossAttnRecorder(bundle, [LAYER], keep_last_only=True) if capture else nullcontext()
    with ctx as rec:
        out = bundle.model(encoder_outputs=encoded, decoder_input_ids=ids,
                           decoder_attention_mask=torch.ones_like(ids), use_cache=False,
                           output_attentions=attention, return_dict=True)
    logits = out.logits[0, -1].float().cpu()
    heads = None
    if attention:
        frozen = bundle.model.generation_config.alignment_heads
        if not frozen or out.cross_attentions is None:
            raise ValueError("localizer_fail:missing_alignment_heads")
        heads = torch.stack([out.cross_attentions[layer][0, head, -1].float().cpu()
                             for layer, head in frozen], dim=0)
    h = rec.states[LAYER][0, -1].float().cpu() if capture else None
    return logits, heads, h


@torch.inference_mode()
def steered_replay(bundle, encoded, tokens: list[int], edits: dict[int, tuple[float, torch.Tensor]],
                   alpha: float, num_forced_prefix: int):
    """Steered cB replay: the canonical hook applies every stored edit at its own position.

    Re-applying stored edits at their positions reproduces KV-cached steering exactly, because
    the L24 site at position tau depends only on layers <= 24 at positions <= tau.
    """
    n = len(tokens)
    gain = torch.zeros((1, n), dtype=torch.float32)
    dirs = torch.zeros((1, n, int(bundle.d_model)), dtype=torch.float32)
    for pos, (g, d) in edits.items():
        gain[0, pos] = float(g)
        dirs[0, pos] = d
    def action_fn(q, u_source, r, abs_pos):
        return gain.to(r.device, r.dtype), dirs.to(r.device, r.dtype)
    ids = torch.tensor([tokens], device=bundle.device, dtype=torch.long)
    hook = DecoderPostCrossAttnInterventionHook(
        bundle, LAYER, None, alpha=float(alpha), num_forced_prefix=num_forced_prefix,
        action_fn=action_fn, norm_preserve=True, record=True, record_last_only=True)
    with hook:
        out = bundle.model(encoder_outputs=encoded, decoder_input_ids=ids,
                           decoder_attention_mask=torch.ones_like(ids), use_cache=False,
                           return_dict=True)
    audit = hook.records[-1].to_dict() if hook.records else None
    return out.logits[0, -1].float().cpu(), audit, hook.steered_calls


def decode(bundle, *, waveform: np.ndarray, encoded, conditions: dict, partition: dict,
           null_probs: dict, language_ids: tuple[int, ...], alpha: float,
           max_new_tokens: int, num_forced_prefix: int) -> dict:
    """Reference-free greedy decode with the frozen gate-driven intervention."""
    tok = bundle.processor.tokenizer
    eos = tok.eos_token_id
    gen = bundle.model.generation_config
    suppress = list(gen.suppress_tokens or [])
    begin_suppress = list(gen.begin_suppress_tokens or [])
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    cb, ce = conditions["cB"], conditions["cE"]
    en_id, zh_id = conditions["language_token_ids"]
    tokens: list[int] = []
    edits: dict[int, tuple[float, torch.Tensor]] = {}
    steps, lid_cache, sample_directions = [], {}, []
    counters = {"forwards": 0, "lid_calls": 0, "lid_cache_hits": 0}
    prev_d = None
    terminated = "cap"
    for t in range(max_new_tokens):
        assert_no_site_hooks(bundle)
        b_in, e_in = step_inputs(cb, tokens), step_inputs(ce, tokens)
        query = len(b_in) - 1
        logits_b, heads, h_b = replay(bundle, encoded, b_in, attention=True, capture=True)
        _, _, h_e = replay(bundle, encoded, e_in, attention=False, capture=True)
        counters["forwards"] += 2
        assert_no_site_hooks(bundle)
        step = {"t": t, "query_index": query, "prefix_length": len(tokens),
                "input_is_prompt_plus_prefix": b_in == list(cb) + tokens and e_in == list(ce) + tokens
                                               and b_in[len(cb):] == e_in[len(ce):],
                "fallback_reason": None, "window": None, "local_support": None,
                "baseline": None, "g": 0.0}
        reasons = []
        if not prefix_utf8_complete(tok, tokens, byte_decoder):
            reasons.append("mid_character")
        window = None
        try:
            window = max_attention_window(heads.numpy(), len(waveform))
            step["window"] = window.record()
        except Exception as exc:
            reasons.append(r2._reason(exc, "localizer_fail"))
        try:
            step["baseline"] = conflict_from_logits(logits_b, partition)
        except Exception as exc:
            reasons.append(r2._reason(exc, "baseline_provider_fail"))
        if not reasons:
            try:
                key = (window.start_sample, window.end_sample)
                if key in lid_cache:
                    counters["lid_cache_hits"] += 1
                else:
                    lid_cache[key] = native_lid(bundle, waveform[key[0]:key[1]], language_ids)
                    counters["lid_calls"] += 1
                probs = lid_cache[key]
                step["local_support"] = local_support(probs[en_id], probs[zh_id],
                                                       null_probs[en_id], null_probs[zh_id])
                step["g"] = float(selected_gate(step["local_support"], step["baseline"]))
            except Exception as exc:
                reasons.append(r2._reason(exc, "local_support_fail"))
                step["g"] = 0.0
        priority = ("mid_character", "localizer_fail", "local_support_fail",
                    "baseline_provider_fail", "nonfinite_signal")
        step["fallback_reason"] = next((r for r in priority if r in reasons), None)
        if step["fallback_reason"] is not None:
            step["g"] = 0.0
        dirn = direction(h_e, h_b)
        step["direction_status"] = dirn["status"]
        step["direction_norm"] = dirn["norm"]
        if dirn["d"] is not None:
            step["cos_to_previous_direction"] = (float(torch.dot(dirn["d"], prev_d))
                                                 if prev_d is not None else None)
            prev_d = dirn["d"]
            if len(sample_directions) < 3:
                sample_directions.append(dirn["d"].tolist())
        edit_now = (alpha * step["g"] > 0 and dirn["status"] == "ok"
                    and query >= num_forced_prefix)
        step["edit_applied"] = bool(edit_now)
        step["forced_prefix_blocked"] = bool(alpha * step["g"] > 0 and query < num_forced_prefix)
        if edit_now:
            edits[query] = (step["g"], dirn["d"])
        logits_s, audit, steered_calls = steered_replay(bundle, encoded, b_in, edits, alpha,
                                                        num_forced_prefix)
        counters["forwards"] += 1
        assert_no_site_hooks(bundle)
        step["hook_audit"] = audit
        step["hook_steered_calls"] = steered_calls
        step["history_edits"] = len(edits)
        step["f3_bitwise_equals_f1"] = bool(torch.equal(logits_s, logits_b))
        if edit_now:
            ref = reference_edit(h_b, dirn["d"], alpha, step["g"])
            step["reference_edit_norm"] = ref["edit_norm"]
            step["reference_cos_edit_d"] = ref["cos_edit_d"]
            step["h_b_norm"] = float(torch.linalg.vector_norm(h_b.double()))
        step["unsteered_next"] = processed_argmax(logits_b, t, suppress, begin_suppress)
        step["steered_next"] = processed_argmax(logits_s, t, suppress, begin_suppress)
        steps.append(step)
        if step["steered_next"] == eos:
            terminated = "eos"
            break
        tokens.append(step["steered_next"])
    return {"tokens": tokens, "text": tok.decode(tokens, skip_special_tokens=True),
            "terminated": terminated, "steps": steps, "counters": counters,
            "edited_positions": sorted(edits), "sample_directions": sample_directions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_cf/p1")
    args = parser.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != VERSION_P1 or digest({k: v for k, v in manifest.items()
                                                     if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid P1 manifest")
    panel = json.loads((out / "panel.json").read_text())
    if digest(panel) != manifest["panel_hash"]:
        raise ValueError("panel hash mismatch")
    for path, expected in manifest["sources"].items():
        if file_hash(path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != manifest["git_commit"]:
        raise ValueError("Git commit changed after P1 manifest freeze")
    config = manifest["resolved_config"]
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("P1 requires the frozen CUDA/bfloat16 execution precision")
    if bundle.revision != manifest["model_revision"] or bundle.tokenizer_revision != manifest["tokenizer_revision"]:
        raise ValueError("model/tokenizer revision mismatch")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != manifest["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    conditions = manifest["conditions"]
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != config["num_forced_prefix"]:
        raise ValueError("forced-prefix width changed")
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    null_probs = native_lid(bundle, np.zeros(30 * bundle.sample_rate, dtype=np.float32), language_ids)
    torch.cuda.reset_peak_memory_stats()
    runtime = {"schema": VERSION_P1, "job_id": os.environ.get("SLURM_JOB_ID"),
               "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "model": bundle.metadata(), "null_probs": null_probs,
               "start_unix": time.time(), "status": "running"}
    atomic_json(out / "runtime.json", runtime)
    for index, item in enumerate(panel["rows"]):
        waveform = load_audio(item["audio_path"], bundle.sample_rate)
        inputs = None
        for alpha in config["alphas"]:
            uid = item["utterance_id"]
            identity = f"{uid}|alpha={alpha}"
            dest = out / "rows" / f"{index:03d}_alpha{alpha}.json"
            if validated_row(dest, identity, manifest["manifest_hash"]):
                continue
            started = time.time()
            try:
                if inputs is None:
                    inputs = batch_model_inputs(bundle, [item["audio_path"]])
                    with torch.inference_mode():
                        enc = bundle.model.model.encoder(input_features=inputs["input_features"],
                                                         attention_mask=inputs["attention_mask"],
                                                         return_dict=True)
                    encoded = BaseModelOutput(last_hidden_state=enc.last_hidden_state)
                result = decode(bundle, waveform=waveform, encoded=encoded, conditions=conditions,
                                partition=partition, null_probs=null_probs,
                                language_ids=language_ids, alpha=float(alpha),
                                max_new_tokens=config["decode"]["max_new_tokens"],
                                num_forced_prefix=nfp)
                result.update(status="ok")
            except Exception as exc:
                import traceback
                result = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
            result.update(identity=identity, utterance_id=uid, alpha=float(alpha),
                          manifest_hash=manifest["manifest_hash"],
                          provenance={k: manifest.get(k) for k in ("git_commit", "git_branch",
                                                                    "config_hash", "model_revision",
                                                                    "tokenizer_revision",
                                                                    "partition_hash")},
                          elapsed_sec=time.time() - started)
            atomic_json(dest, result)
            print(f"P1 {index + 1}/{len(panel['rows'])} {identity} {result['status']}", flush=True)
    assert_no_site_hooks(bundle)
    runtime.update(end_unix=time.time(), status="completed",
                   peak_vram_bytes=torch.cuda.max_memory_allocated())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
