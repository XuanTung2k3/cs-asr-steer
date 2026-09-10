#!/usr/bin/env python
"""Small real-model BASIS-A3 encoder acceptance suite.

This is an acceptance-only program. It reads the frozen SEAME development
panels and Raw encoder artifacts, performs no scientific grid sweep, and writes
only acceptance JSON evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.basis_a3 import (
    DATA_ROOT,
    GEN,
    MODEL_CFG,
    RESULTS,
    _encoder_gain,
    _encoder_span,
    _load_json,
    _panel_frame,
)
from csasr.evaluation import canonical
from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook, EncoderPostSelfAttnRecorder
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.utils.config import load_config


REPO = Path(__file__).resolve().parents[1]
OUT = RESULTS / "acceptance"
MASK_SOURCE = REPO / "experiments/basis_a3.py"


def write_json(name: str, value: object) -> Path:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def seq_and_text(bundle, inputs):
    out = bundle.model.generate(**inputs, **GEN)
    seq = out if isinstance(out, torch.Tensor) else out.sequences
    ids = seq.detach().cpu().tolist()
    text = bundle.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
    return ids, text


def valid_rows(dataset: str, n: int = 4):
    frame = _panel_frame(dataset).sort_values("duration_sec")
    return [r for r in frame.to_dict(orient="records") if r.get("utterance_id")][:n]


def target_rows(dataset: str, n: int = 4):
    rows = []
    for row in _panel_frame(dataset).sort_values("duration_sec").to_dict(orient="records"):
        path = DATA_ROOT / "audio/segments" / str(row["utterance_id"]) / "segments.json"
        if not path.is_file():
            continue
        payload = _load_json(path)
        if any(bool(s.get("target", False)) for s in payload.get("segments", [])):
            rows.append(row)
        if len(rows) >= n:
            break
    return rows


def direction(layer: int) -> torch.Tensor:
    manifest = _load_json(RESULTS / "directions/raw_encoder/manifest.json")
    return torch.from_numpy(np.load(REPO / manifest["layers"][str(layer)]["path"])).float()


def run_encoder(bundle, row, layer: int, rho: float, scope: str, *, record: bool = False):
    inputs = batch_model_inputs(bundle, [row["audio_path"]])
    gain = _encoder_gain(bundle, row["dataset"], row, scope=scope, cs_spans={})
    hook = EncoderPostSelfAttnInterventionHook(
        bundle, layer, direction(layer), alpha=rho, scale=1.0, gain=gain,
        norm_preserve=True, record=record,
    )
    with hook, torch.inference_mode():
        ids, text = seq_and_text(bundle, inputs)
    return ids, text, gain, hook


def rho0_identity(bundle) -> dict:
    observations = []
    for dataset in ("seame_dev_man", "seame_dev_sge"):
        rows = valid_rows(dataset, 3)
        refs, baselines, methods = [], [], []
        for row in rows:
            row = dict(row, dataset=dataset)
            inputs = batch_model_inputs(bundle, [row["audio_path"]])
            with torch.inference_mode():
                base_ids, base_text = seq_and_text(bundle, inputs)
            zero_ids, zero_text, gain, _ = run_encoder(bundle, row, 16, 0.0, "oracle_local")
            refs.append(str(row["reference"])); baselines.append(base_text); methods.append(zero_text)
            observations.append({
                "dataset": dataset,
                "utterance_id": str(row["utterance_id"]),
                "token_ids_identical": base_ids == zero_ids,
                "transcript_identical": base_text == zero_text,
                "baseline_transcript": base_text,
                "rho0_transcript": zero_text,
                "active_mask_frames": int((gain > 0).sum().item()),
            })
        base_metrics = canonical.corpus_metrics(refs, baselines)
        zero_metrics = canonical.corpus_metrics(refs, methods)
        observations.append({
            "dataset": dataset,
            "canonical_metrics_identical": base_metrics == zero_metrics,
            "baseline_metrics": base_metrics,
            "rho0_metrics": zero_metrics,
        })
    passed = all(
        o.get("token_ids_identical", True)
        and o.get("transcript_identical", True)
        and o.get("canonical_metrics_identical", True)
        for o in observations
    )
    return {
        "schema_version": "basis_a3_encoder_rho0_identity_v1",
        "status": "PASS" if passed else "FAIL",
        "datasets": ["seame_dev_man", "seame_dev_sge"],
        "layer": 16,
        "scope": "oracle_local",
        "rho": 0.0,
        "observations": observations,
        "mask_definition": "target-segment union alignment",
        "boundary_tolerance_ms": 80,
    }


def site_states(bundle, row, layer: int, scope: str):
    inputs = batch_model_inputs(bundle, [row["audio_path"]])
    gain = _encoder_gain(bundle, row["dataset"], row, scope=scope, cs_spans={})
    with torch.inference_mode(), EncoderPostSelfAttnRecorder(bundle, [layer]) as rec:
        bundle.model.model.encoder(**inputs)
        before = rec.states[layer].clone()
    hook = EncoderPostSelfAttnInterventionHook(
        bundle, layer, direction(layer), alpha=0.5, scale=1.0, gain=gain,
        norm_preserve=True, record=True,
    )
    with hook, torch.inference_mode(), EncoderPostSelfAttnRecorder(bundle, [layer]) as rec:
        bundle.model.model.encoder(**inputs)
        after = rec.states[layer].clone()
    return before, after, gain, hook


def norm_preserve(bundle) -> dict:
    observations = []
    for dataset in ("seame_dev_man", "seame_dev_sge"):
        for row in target_rows(dataset, 2):
            row = dict(row, dataset=dataset)
            for layer in (0, 16, 31):
                before, after, gain, _ = site_states(bundle, row, layer, "oracle_local")
                active = gain[0] > 0
                before_norm = before[0].norm(dim=-1)[active].double()
                after_norm = after[0].norm(dim=-1)[active].double()
                abs_error = (before_norm - after_norm).abs()
                rel_error = abs_error / before_norm.clamp_min(1e-12)
                observations.append({
                    "dataset": dataset, "utterance_id": str(row["utterance_id"]),
                    "layer": layer, "scope": "oracle_local",
                    "edited_frames": int(active.sum()),
                    "max_abs_error": float(abs_error.max()) if len(abs_error) else 0.0,
                    "max_relative_error": float(rel_error.max()) if len(rel_error) else 0.0,
                })
    max_abs = max(x["max_abs_error"] for x in observations)
    max_rel = max(x["max_relative_error"] for x in observations)
    return {
        "schema_version": "basis_a3_encoder_normpreserve_v1",
        "status": "PASS" if max_rel <= 1e-2 else "FAIL",
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "observations": observations,
        "tensor_dtype": "bfloat16 (Whisper inference path)",
        "tolerance": {"relative": 1e-2, "basis": "src/csasr/models/hook_tests.py::run_hook_tests norm_tol"},
    }


def padding_exclusion(bundle) -> dict:
    observations = []
    for dataset in ("seame_dev_man", "seame_dev_sge"):
        for row in target_rows(dataset, 4):
            row = dict(row, dataset=dataset)
            layer = 16
            before, after, gain, _ = site_states(bundle, row, layer, "oracle_local")
            valid = bundle.valid_frames(row["duration_sec"])
            padded = np.arange(after.shape[1]) >= valid
            diff = (after[0] - before[0]).float().norm(dim=-1).detach().cpu().numpy()
            valid_changed = diff[:valid] > 1e-7
            padded_changed = diff[padded] > 1e-7
            observations.append({
                "dataset": dataset, "utterance_id": str(row["utterance_id"]),
                "duration_sec": float(row["duration_sec"]), "valid_frames": int(valid),
                "total_frames": int(after.shape[1]),
                "valid_edited_frames": int(valid_changed.sum()),
                "padding_changed_frames": int(padded_changed.sum()),
                "padding_max_abs_delta": float(diff[padded].max()) if padded.any() else 0.0,
            })
    passed = all(x["valid_edited_frames"] > 0 and x["padding_changed_frames"] == 0 for x in observations)
    return {
        "schema_version": "basis_a3_encoder_padding_exclusion_v1",
        "status": "PASS" if passed else "FAIL",
        "observations": observations,
        "requirement": "valid frames may change; padded frames have zero gain and remain unchanged",
    }


def segment_metadata(row):
    path = DATA_ROOT / "audio/segments" / str(row["utterance_id"]) / "segments.json"
    payload = _load_json(path)
    silence = float(payload.get("silence_seconds", 0.08))
    cursor = 0.0
    times = {}
    for item in payload.get("segments", []):
        duration = float(item["audio"]["duration"])
        times[int(item["index"])] = (cursor, cursor + duration)
        cursor += duration + silence
    targets = [s for s in payload.get("segments", []) if bool(s.get("target", False))]
    return path, silence, targets, times


def local_mask(bundle) -> dict:
    observations = []
    for dataset in ("seame_dev_man", "seame_dev_sge"):
        for row in target_rows(dataset, 4):
            row = dict(row, dataset=dataset)
            path, silence, targets, times = segment_metadata(row)
            if not targets:
                continue
            span = _encoder_span(dataset, row, bundle)
            if span is None:
                observations.append({"dataset": dataset, "utterance_id": str(row["utterance_id"]), "status": "NO_TARGET_SPAN"})
                continue
            expected = set(range(span[0], span[1]))
            before, after, gain, _ = site_states(bundle, row, 16, "oracle_local")
            delta = (after[0] - before[0]).float().norm(dim=-1)
            actual = set(np.flatnonzero(delta.detach().cpu().numpy() > 1e-7).tolist())
            target_times = [times[int(s["index"])] for s in targets]
            union = [min(x[0] for x in target_times), max(x[1] for x in target_times)]
            observations.append({
                "dataset": dataset, "utterance_id": str(row["utterance_id"]),
                "segment_file": str(path),
                "target_segment_boundaries_seconds": target_times,
                "union_boundaries_seconds": union,
                "silence_seconds": silence,
                "boundary_tolerance_ms": 80,
                "expected_frame_range": [span[0], span[1]],
                "expected_frame_indices": sorted(expected),
                "actual_frame_indices": sorted(actual),
                "expected_edited_frame_count": len(expected),
                "actual_edited_frame_count": len(actual),
                "mismatch_count": len(expected.symmetric_difference(actual)),
            })
    passed = bool(observations) and all(x.get("mismatch_count") == 0 for x in observations)
    return {
        "schema_version": "basis_a3_encoder_local_mask_v1",
        "status": "PASS" if passed else "FAIL",
        "mask_definition": "target-segment union alignment",
        "boundary_tolerance_ms": 80,
        "observations": observations,
    }


def no_gradient(bundle) -> dict:
    params = list(bundle.model.parameters())
    requires_grad = [name for name, p in bundle.model.named_parameters() if p.requires_grad]
    grads_present_before = [name for name, p in bundle.model.named_parameters() if p.grad is not None]
    row = dict(valid_rows("seame_dev_man", 1)[0], dataset="seame_dev_man")
    inputs = batch_model_inputs(bundle, [row["audio_path"]])
    gain = _encoder_gain(bundle, row["dataset"], row, scope="oracle_local", cs_spans={})
    vec = direction(16)
    hook = EncoderPostSelfAttnInterventionHook(bundle, 16, vec, alpha=0.5, scale=1.0, gain=gain, norm_preserve=True)
    with hook, torch.inference_mode():
        grad_disabled = not torch.is_grad_enabled()
        seq, _ = seq_and_text(bundle, inputs)
    grads_present_after = [name for name, p in bundle.model.named_parameters() if p.grad is not None]
    passed = not requires_grad and not grads_present_before and not grads_present_after and grad_disabled and not vec.requires_grad
    return {
        "schema_version": "basis_a3_encoder_no_grad_v1",
        "status": "PASS" if passed else "FAIL",
        "parameter_count": len(params),
        "parameters_requiring_grad": requires_grad,
        "gradients_present_before": grads_present_before,
        "gradients_present_after": grads_present_after,
        "inference_mode_grad_disabled": grad_disabled,
        "optimizer": None,
        "controller": None,
        "direction_requires_grad": bool(vec.requires_grad),
        "generated_token_count": len(seq[0]),
    }


def main() -> int:
    bundle = load_whisper(load_config(MODEL_CFG))
    bundle.model.eval()
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("frozen model precondition failed")
    results = {
        "encoder_rho0_identity.json": rho0_identity(bundle),
        "encoder_normpreserve.json": norm_preserve(bundle),
        "encoder_padding_exclusion.json": padding_exclusion(bundle),
        "encoder_local_mask.json": local_mask(bundle),
        "encoder_no_grad.json": no_gradient(bundle),
    }
    for name, payload in results.items():
        payload["mask_implementation_hash"] = file_hash(MASK_SOURCE)
        write_json(name, payload)
    statuses = {name: payload["status"] for name, payload in results.items()}
    write_json("encoder_acceptance_summary.json", {"status": "PASS" if all(x == "PASS" for x in statuses.values()) else "FAIL", "checks": statuses})
    print(json.dumps(statuses, sort_keys=True))
    return 0 if all(x == "PASS" for x in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
