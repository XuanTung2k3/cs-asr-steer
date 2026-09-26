#!/usr/bin/env python
"""Unsteered P0-R2 dense gate-feasibility replay. No activation edit exists here."""
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
from transformers.modeling_outputs import BaseModelOutput

from csasr.inference_cf.core import atomic_json, digest, file_hash, generated_content, validated_row
from csasr.inference_cf.core_r2 import (VERSION_R2, gate_values, conflict_from_logits,
                                        local_support, max_attention_window,
                                        prefix_utf8_complete, same_prefix_inputs,
                                        terminated_by_eos, tokenizer_partition,
                                        validate_gate_row)
from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_file


def _features(bundle, waveform: np.ndarray) -> torch.Tensor:
    features = bundle.processor.feature_extractor(
        waveform, sampling_rate=bundle.sample_rate, return_tensors="pt")
    if features.input_features.shape[-1] != 3000:
        raise ValueError("localizer_fail:window_not_padded_to_30s")
    return features.input_features.to(bundle.device, bundle.dtype)


@torch.inference_mode()
def native_lid(bundle, waveform: np.ndarray, language_ids: tuple[int, ...]) -> dict[int, float]:
    """Whisper's language-token readout from the single SOT decoder query."""
    ids = torch.tensor([[bundle.model.generation_config.decoder_start_token_id]],
                       device=bundle.device, dtype=torch.long)
    out = bundle.model(input_features=_features(bundle, waveform), decoder_input_ids=ids,
                       use_cache=False, output_attentions=False, output_hidden_states=False,
                       return_dict=True)
    logits = out.logits[0, -1].float()
    vals = logits[list(language_ids)]
    if not bool(torch.isfinite(vals).all()):
        raise ValueError("nonfinite_signal:lid_logits")
    probs = torch.softmax(vals, dim=0).cpu().tolist()
    return {token_id: float(prob) for token_id, prob in zip(language_ids, probs)}


@torch.inference_mode()
def full_replay(bundle, encoded, prompt: list[int], content: list[int], *, attention: bool):
    """One causal replay; content never comes from Ecf generation."""
    ids = torch.tensor([prompt + content], device=bundle.device, dtype=torch.long)
    out = bundle.model(encoder_outputs=encoded, decoder_input_ids=ids,
                       decoder_attention_mask=torch.ones_like(ids), use_cache=False,
                       output_attentions=attention, output_hidden_states=False, return_dict=True)
    logits = out.logits[0].float().cpu()
    heads = None
    if attention:
        frozen = bundle.model.generation_config.alignment_heads
        if not frozen or out.cross_attentions is None:
            raise ValueError("localizer_fail:missing_alignment_heads")
        heads = torch.stack([out.cross_attentions[layer][0, head].float().cpu()
                             for layer, head in frozen], dim=0)
        if heads.shape[1] != len(prompt) + len(content):
            raise ValueError("localizer_fail:query_shape")
    return logits, heads


def _reason(exc: Exception, default: str) -> str:
    message = str(exc)
    for key in ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail",
                "ecf_provider_fail", "nonfinite_signal"):
        if message.startswith(key):
            return key
    return default


def _zero() -> dict:
    return {"D": 0.0, "g_old": 0.0, "g_cf": 0.0}


def _donor_crop(donor: np.ndarray, start_sample: int, length: int) -> tuple[np.ndarray, int]:
    """Same time coordinate, shifted left only when donor ends sooner."""
    heard = min(len(donor), 30 * 16000)
    if heard <= 0:
        raise ValueError("localizer_fail:empty_donor")
    effective = min(length, heard)
    start = min(start_sample, heard - effective)
    return donor[start:start + effective], start


def inference_utterance(bundle, *, audio_path: str, donor_audio_path: str,
                        audio_sha256: str, conditions: dict, partition: dict,
                        null_probs: dict, language_ids: tuple[int, ...], manifest: dict) -> dict:
    """Inference-only inputs: no reference, label, target time, or future token parameter."""
    waveform = load_audio(audio_path, bundle.sample_rate)
    donor = load_audio(donor_audio_path, bundle.sample_rate)
    inputs = batch_model_inputs(bundle, [audio_path])
    with torch.inference_mode():
        sequence = bundle.model.generate(**inputs, **manifest["decode"])[0].tolist()
        enc = bundle.model.model.encoder(input_features=inputs["input_features"],
                                         attention_mask=inputs["attention_mask"],
                                         return_dict=True)
    encoded = BaseModelOutput(last_hidden_state=enc.last_hidden_state)
    eos = bundle.processor.tokenizer.eos_token_id
    content = generated_content(sequence, eos)
    # generate() strips a final EOS; a below-cap content length still means EOS was emitted.
    has_eos = terminated_by_eos(sequence, content, eos, manifest["decode"]["max_new_tokens"])
    if not content and not has_eos:
        raise ValueError("baseline_provider_fail:empty_generation")
    baseline_text = bundle.processor.tokenizer.decode(content, skip_special_tokens=True)
    b_prompt, e_prompt = conditions["cB"], conditions["cE"]
    if b_prompt == e_prompt or len(b_prompt) != len(e_prompt):
        raise ValueError("conditions not comparable")
    # The replays have the same actual baseline content IDs at every query; only the
    # language token differs. Full replay (use_cache=False) avoids any KV-cache hot swap.
    b_logits, heads = full_replay(bundle, encoded, b_prompt, content, attention=True)
    e_logits, _ = full_replay(bundle, encoded, e_prompt, content, attention=False)
    if b_logits.shape != e_logits.shape or b_logits.shape[0] != len(b_prompt) + len(content):
        raise ValueError("same-prefix replay shape mismatch")
    # Diagnostic-only wrong-language counterfactual (never enters either gate).
    x_prompt = conditions.get("cX")
    x_logits = None
    if x_prompt is not None:
        same_prefix_inputs({"cB": b_prompt, "cE": x_prompt}, [])
        x_logits, _ = full_replay(bundle, encoded, x_prompt, content, attention=False)
        if x_logits.shape != b_logits.shape:
            raise ValueError("wrong-language replay shape mismatch")
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    n = len(content) + int(has_eos)
    lid_cache: dict[tuple, dict] = {}
    lid_calls = {"real": 0, "control": 0, "cache_hits": 0}

    def cached_lid(kind: str, audio: np.ndarray, start: int, end: int) -> dict:
        # Identical crop of identical audio is an identical model input: exact reuse only.
        key = (kind, int(start), int(end))
        if key in lid_cache:
            lid_calls["cache_hits"] += 1
            return lid_cache[key]
        lid_calls[kind] += 1
        lid_cache[key] = native_lid(bundle, audio[start:end], language_ids)
        return lid_cache[key]

    results = []
    for t in range(n):
        prefix = content[:t]
        b_input, e_input = same_prefix_inputs(conditions, prefix)
        query = len(b_input) - 1
        next_token = content[t] if t < len(content) else eos
        row = {"schema": VERSION_R2, "utterance_id": manifest["current_utterance_id"],
               "logical_position": t, "prediction_query_index": query,
               "prefix_ids": prefix, "baseline_input_ids": b_input,
               "ecf_input_ids": e_input, "next_token_id_diagnostic": next_token,
               "replay_argmax_matches_baseline": int(torch.argmax(b_logits[query])) == int(next_token),
               "is_eos_slot": t == len(content), "audio_sha256": audio_sha256,
               "baseline_condition_ids": list(conditions["cB"]),
               "english_condition_ids": list(conditions["cE"]),
               "replay_mode": "full_same_prefix_replay_use_cache_false",
               "same_prefix_validation": (b_input[len(conditions["cB"]):] == e_input[len(conditions["cE"]):] == prefix
                                          and query == len(e_input) - 1),
               "localizer_status": None, "eligibility_status": None,
               "manifest_hash": manifest["manifest_hash"], "baseline": None, "ecf": None,
               "xcf": None, "local_support": None, "window": None, "control": None,
               "control_reason": None,
               "fallback_reason": None, "gate": _zero()}
        # Compute all diagnostics, then select the first reason in frozen precedence.
        # Scope "all" zeroes both gates; scope "cf" (Ecf branch only) zeroes D and g_cf but
        # keeps g_old = E*R_B when every input of the old gate is valid.
        reasons = []
        if not prefix_utf8_complete(bundle.processor.tokenizer, prefix, byte_decoder):
            reasons.append(("mid_character", "all"))
        try:
            window = max_attention_window(heads[:, query].numpy(), len(waveform))
            row["window"] = window.record()
            row["localizer_status"] = "ok"
        except Exception as exc:
            reasons.append((_reason(exc, "localizer_fail"), "all"))
            row["localizer_status"] = str(exc)
        try:
            row["baseline"] = conflict_from_logits(b_logits[query], partition)
        except Exception as exc:
            reasons.append((_reason(exc, "baseline_provider_fail"), "all"))
        try:
            row["ecf"] = conflict_from_logits(e_logits[query], partition)
        except Exception as exc:
            reasons.append(("ecf_provider_fail", "cf"))
            row["ecf_error"] = str(exc)
        if x_logits is not None and row["baseline"] is not None:
            try:
                x_conflict = conflict_from_logits(x_logits[query], partition)
                row["xcf"] = {"language": conditions["wrong_language"], "conflict": x_conflict,
                              "D_X": max(0.0, row["baseline"]["R"] - x_conflict["R"])}
            except Exception as exc:
                row["xcf"] = None
                row["xcf_reason"] = _reason(exc, "xcf_provider_fail")
        en_id, zh_id = conditions["language_token_ids"]
        if not any(scope == "all" for _, scope in reasons):
            try:
                probs = cached_lid("real", waveform, window.start_sample, window.end_sample)
                row["local_support"] = local_support(probs[en_id], probs[zh_id],
                                                      null_probs[en_id], null_probs[zh_id])
            except Exception as exc:
                reasons.append((_reason(exc, "local_support_fail"), "all"))
                row["local_support"] = None
        priority = ("mid_character", "localizer_fail", "local_support_fail",
                    "baseline_provider_fail", "ecf_provider_fail", "nonfinite_signal")
        found = {reason for reason, _ in reasons}
        row["fallback_reason"] = next((reason for reason in priority if reason in found), None)
        if any(scope == "all" for _, scope in reasons):
            row["eligibility_status"] = "ineligible"
            row["gate"] = _zero()
        elif reasons:
            row["eligibility_status"] = "cf_ineligible"
            row["gate"] = {"D": 0.0, "g_old": float(row["local_support"]["E"] * row["baseline"]["R"]),
                           "g_cf": 0.0}
        else:
            try:
                row["gate"] = gate_values(row["local_support"], row["baseline"], row["ecf"])
                row["eligibility_status"] = "eligible"
            except Exception as exc:
                row["fallback_reason"] = _reason(exc, "nonfinite_signal")
                row["eligibility_status"] = "ineligible"
                row["gate"] = _zero()
        if row["local_support"] is not None and row["eligibility_status"] != "ineligible":
            try:
                crop, donor_start = _donor_crop(donor, window.start_sample,
                                                 window.end_sample - window.start_sample)
                control_probs = cached_lid("control", donor, donor_start, donor_start + len(crop))
                control = local_support(control_probs[en_id], control_probs[zh_id],
                                        null_probs[en_id], null_probs[zh_id])
                row["control"] = {"support": control, "donor_start_sample": donor_start,
                                  "donor_end_sample": donor_start + len(crop),
                                  "g_old": control["E"] * row["baseline"]["R"],
                                  "g_cf": control["E"] * row["gate"]["D"]}
            except Exception as exc:
                row["control_reason"] = _reason(exc, "local_support_fail")
        validate_gate_row(row)
        results.append(row)
    return {"identity": manifest["current_utterance_id"], "manifest_hash": manifest["manifest_hash"],
            "provenance": {k: manifest.get(k) for k in ("git_commit", "git_branch", "config_hash",
                                                         "model_revision", "tokenizer_revision",
                                                         "partition_hash", "seed")},
            "status": "ok", "baseline_sequence": sequence, "baseline_content_ids": content,
            "baseline_text": baseline_text, "has_eos": has_eos,
            "eos_in_returned_sequence": eos in sequence,
            "truncated_at_cap": not has_eos and len(content) >= manifest["decode"]["max_new_tokens"],
            "replay_argmax_agreement": (sum(r["replay_argmax_matches_baseline"] for r in results)
                                        / len(results) if results else None),
            "lid_calls": lid_calls, "rows": results, "output_tokens": len(content)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_cf/p0_r2")
    args = parser.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != VERSION_R2 or digest({k: v for k, v in manifest.items()
                                                     if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid manifest")
    panel = json.loads((out / "inference_panel.json").read_text())
    if digest(panel) != manifest["panel_hash"]:
        raise ValueError("panel hash mismatch")
    for path, expected in manifest["sources"].items():
        if file_hash(path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_commit != manifest["git_commit"]:
        raise ValueError("Git commit changed after R2 manifest freeze")
    if digest(manifest["resolved_config"]) != manifest["config_hash"]:
        raise ValueError("resolved config hash mismatch")
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("R2 requires the frozen CUDA/bfloat16 execution precision")
    if bundle.revision != manifest["model_revision"] or bundle.tokenizer_revision != manifest["tokenizer_revision"]:
        raise ValueError("model/tokenizer revision mismatch")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != manifest["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    conditions = manifest["conditions"]
    if conditions["cB"] != conditions["cM"]:
        raise ValueError("B0 is not forced ZH")
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    if len(language_ids) != 100:
        raise ValueError("unexpected language-token vocabulary")
    if [native["<|en|>"], native["<|zh|>"]] != conditions["language_token_ids"]:
        raise ValueError("language ID mismatch")
    if "cX" in conditions:
        expected_x = list(conditions["cB"])
        expected_x[1] = native[f"<|{conditions['wrong_language']}|>"]
        if conditions["cX"] != expected_x or conditions["wrong_language"] in ("en", "zh"):
            raise ValueError("wrong-language diagnostic prompt mismatch")
    null_probs = native_lid(bundle, np.zeros(30 * bundle.sample_rate, dtype=np.float32), language_ids)
    runtime = {"schema": VERSION_R2, "job_id": os.environ.get("SLURM_JOB_ID"),
               "node": socket.gethostname(), "model": bundle.metadata(),
               "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__,
               "null_probs": null_probs, "start_unix": time.time(), "status": "running"}
    atomic_json(out / "runtime.json", runtime)
    rows = panel["rows"]
    by_id = {r["utterance_id"]: r for r in rows}
    for index, item in enumerate(rows):
        uid = item["utterance_id"]
        dest = out / "rows" / f"{index:03d}.json"
        if validated_row(dest, uid, manifest["manifest_hash"]):
            continue
        started = time.time()
        run_manifest = {**manifest, "current_utterance_id": uid}
        try:
            donor = by_id[item["donor_utterance_id"]]
            for member in (item, donor):
                if sha256_file(member["audio_path"], max_bytes=65536) != member["audio_sha256"]:
                    raise ValueError("audio source hash changed after panel freeze")
            result = inference_utterance(bundle, audio_path=item["audio_path"],
                                         donor_audio_path=donor["audio_path"],
                                         audio_sha256=item["audio_sha256"], conditions=conditions,
                                         partition=partition, null_probs=null_probs,
                                         language_ids=language_ids, manifest=run_manifest)
        except Exception as exc:
            result = {"identity": uid, "manifest_hash": manifest["manifest_hash"],
                      "status": "failure", "reason": repr(exc), "rows": []}
        result["elapsed_sec"] = time.time() - started
        atomic_json(dest, result)
        print(f"R2 {index + 1}/{len(rows)} {uid} {result['status']}", flush=True)
    runtime["end_unix"] = time.time()
    runtime["status"] = "completed"
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    runtime["peak_vram_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
