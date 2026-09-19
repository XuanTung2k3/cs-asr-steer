#!/usr/bin/env python
"""BASIS-A5 unique/shared construction, gates, atlas, and finalization.

This runner is additive to BASIS-A4.  It never writes under ``results/basis_a4``
and never decodes an A4 comparator.  GPU subcommands are intended to be called
from the checked-in Slurm launchers; CPU gates and analysis are safe to run
locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
A4 = REPO / "results/basis_a4"
OUT = REPO / "results/basis_a5_unique_shared"
DATASETS = ("cs_dialogue", "seame_dev_man", "seame_dev_sge")
DIRECTIONS = ("neg_shared", "pos_unique", "unique_minus_shared")
RHO = 0.5
WHISPER_ENC = tuple(range(32))
WHISPER_DEC = tuple(range(32))
QWEN_ENC = tuple(range(24))
QWEN_DEC = tuple(range(28))
# A4's frozen anchor/candidate set used for construction-only stability checks.
# The A5 atlas itself remains full-depth; this constant only bounds the
# pre-GPU rank grid and is clipped to each model's available layers.
A4_STABILITY_ANCHORS = (24, 26, 27, 31)
SITE_WHISPER = {
    "encoder": "encoder_post_self_attn_residual_pre_ffn",
    "decoder": "decoder_post_cross_attn_residual",
}
SITE_QWEN = {
    "encoder": "qwen_audio_encoder_post_self_attn_residual_pre_ffn",
    "decoder": "qwen_text_decoder_post_self_attn_residual_pre_mlp",
}
A5_SPEC = REPO / "docs/current/BASIS_A5_SPEC.md"
A5_PLAN = REPO / "docs/current/BASIS_A5_EXECUTION_PLAN.md"
A5_AMENDMENT = REPO / "docs/current/BASIS_A5_NONBLOCKING_STABILITY_AMENDMENT.md"
A4_PROTOCOL = A4 / "manifests/a4_protocol_freeze.json"
PRESERVED_RANK_GATE = OUT / "manifests/rank_stability.json"
RANK_DIAGNOSTIC = OUT / "manifests/rank_stability_diagnostic.json"


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False, default=str) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _hash_obj(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode()).hexdigest()


def _array_hash(path: Path) -> str:
    return _sha256_file(path)


def _git_state() -> dict:
    try:
        from csasr.utils.logging import git_state
        return git_state(str(REPO))
    except Exception:
        return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   cwd=REPO, text=True).strip()}


def _a4_hash() -> str:
    return _json(A4_PROTOCOL)["protocol_hash"]


def _panel(dataset: str) -> dict:
    return _json(A4 / "panels" / f"{dataset}_300.json")


def _a4_row(model: str, dataset: str, direction: str, side: str, layer: int) -> dict:
    if model == "whisper":
        if direction == "Conditioning":
            path = A4 / "whisper" / "conditioning" / dataset / f"L{layer:02d}" / \
                f"conditioning_decoder_L{layer}_oracle_local_rho0.5.json"
        else:
            path = A4 / "whisper" / "raw" / side / dataset / f"L{layer:02d}" / \
                f"Raw_{side}_L{layer}_oracle_local_rho0.5.json"
    else:
        root = "conditioning" if direction == "Conditioning" else "raw"
        name = f"{direction}_{side}_L{layer}_oracle_local_rho0.5.json"
        path = A4 / "qwen3_asr_1p7b" / root / dataset / f"L{layer:02d}" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return _json(path)


def _a4_direction_path(model: str, side: str, layer: int, kind: str) -> Path:
    if model == "qwen3_asr_1p7b":
        key = "raw_encoder" if side == "encoder" else ("conditioning" if kind == "conditioning" else "raw_decoder")
        return A4 / "qwen3_asr_1p7b" / "directions" / f"{key}_L{layer}.npy"
    if side == "encoder":
        return REPO / "results/basis_a3_raw_cond_scope_depth/directions/raw_encoder" / f"raw_encoder_L{layer}.npy"
    return REPO / "results/basis_frozen_layer_atlas/directions" / \
        f"{'conditioning' if kind == 'conditioning' else 'raw'}_L{layer}.npy"


def _a4_scale(model: str, side: str, layer: int) -> float:
    direction = "Conditioning" if model == "whisper" and side == "decoder" else "Raw"
    row = _a4_row(model, "cs_dialogue", direction, side, layer)
    if "scale" in row and row["scale"] is not None:
        return float(row["scale"])
    if model == "qwen3_asr_1p7b":
        meta = _json(A4 / "qwen3_asr_1p7b/directions/manifest.json")
        key = "raw_encoder" if side == "encoder" else ("conditioning" if direction == "Conditioning" else "raw_decoder")
        return float(meta["layers"][side][str(layer)][key]["mean_site_norm"] if side == "decoder"
                     else meta["layers"][side][str(layer)]["mean_site_norm"])
    raise KeyError(f"A4 scale missing for {model} {side} L{layer}")


def _model_revision(model: str) -> str:
    if model == "whisper":
        return _a4_row("whisper", "cs_dialogue", "Raw", "encoder", 0)["model_revision"]
    return _a4_row("qwen3_asr_1p7b", "cs_dialogue", "Raw", "encoder", 0)["model_revision"]


def _protocol_payload() -> dict:
    a4_final = _json(A4 / "FINAL_MANIFEST.json")
    return {
        "schema_version": "basis_a5_unique_shared_protocol_v1",
        "study": "BASIS-A5",
        "spec_sha256": _sha256_file(A5_SPEC),
        "execution_plan_sha256": _sha256_file(A5_PLAN),
        "protocol_amendment_sha256": _sha256_file(A5_AMENDMENT),
        "rank_stability_policy": "diagnostic_nonblocking",
        "a4_protocol_hash": _a4_hash(),
        "a4_final_manifest_sha256": _sha256_file(A4 / "FINAL_MANIFEST.json"),
        "a4_final_status": a4_final.get("status"),
        "construction_role": "D-construct",
        "evaluation_panels": {d: {"count": _panel(d)["count"], "fingerprint": _panel(d)["fingerprint"]}
                              for d in DATASETS},
        "directions": list(DIRECTIONS),
        "rank": 32,
        "stability_ranks": [16, 32, 64],
        "orthogonality_tolerance": 1e-3,
        "rho": RHO,
        "scope": "oracle_local",
        "norm_preserve": True,
        "depth_rescale": False,
        "whisper": {"encoder_layers": list(WHISPER_ENC), "decoder_layers": list(WHISPER_DEC),
                    "model_revision": _model_revision("whisper"), "sites": SITE_WHISPER},
        "qwen3_asr_1p7b": {"encoder_layers": list(QWEN_ENC), "decoder_layers": list(QWEN_DEC),
                            "model_revision": _model_revision("qwen3_asr_1p7b"), "sites": SITE_QWEN},
        "forbidden_splits": ["D-dev-confirm", "D-test"],
        "decoding": {"do_sample": False, "temperature": 0.0, "num_beams": 1,
                      "condition_on_prev_tokens": False, "max_new_tokens": 200,
                      "whisper_language": "zh", "qwen_force_language": "Chinese", "qwen_context": ""},
        "comparators_read_only": True,
    }


def comparator_audit() -> dict:
    """Audit every A4 comparator needed by the new local-only atlas."""
    expected = []
    for dataset in DATASETS:
        for side, layers in (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC)):
            for layer in layers:
                expected.append(("whisper", dataset, "Raw", side, layer))
        for layer in WHISPER_DEC:
            expected.append(("whisper", dataset, "Conditioning", "decoder", layer))
        for side, layers in (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)):
            for layer in layers:
                expected.append(("qwen3_asr_1p7b", dataset, "Raw", side, layer))
        for layer in QWEN_DEC:
            expected.append(("qwen3_asr_1p7b", dataset, "Conditioning", "decoder", layer))
    rows = []
    failures = []
    for model, dataset, direction, side, layer in expected:
        try:
            x = _a4_row(model, dataset, direction, side, layer)
            required = (x.get("scope") == "oracle_local", float(x.get("rho", -1)) == RHO,
                        x.get("panel_fingerprint") == _panel(dataset)["fingerprint"],
                        x.get("model_revision") == _model_revision(model),
                        bool(x.get("direction_hash")), bool(x.get("site_hash")), bool(x.get("mask_hash")))
            if not all(required):
                failures.append({"model": model, "dataset": dataset, "direction": direction,
                                 "side": side, "layer": layer, "checks": required})
            rows.append({"model": model, "dataset": dataset, "direction": direction, "side": side,
                         "layer": layer, "path": str((_a4_row_path(model, dataset, direction, side, layer)).relative_to(REPO)),
                         "direction_hash": x.get("direction_hash"), "site_hash": x.get("site_hash"),
                         "mask_hash": x.get("mask_hash"), "panel_fingerprint": x.get("panel_fingerprint"),
                         "model_revision": x.get("model_revision")})
        except Exception as exc:
            failures.append({"model": model, "dataset": dataset, "direction": direction,
                             "side": side, "layer": layer, "error": repr(exc)})
    baselines = []
    for model in ("whisper", "qwen3_asr_1p7b"):
        for dataset in DATASETS:
            if model == "whisper":
                path = A4 / "baselines" / f"{dataset}.json"
                if not path.is_file():
                    path = REPO / "results/basis_a3_raw_cond_scope_depth/baselines" / f"{dataset}.json"
            else:
                path = A4 / "qwen3_asr_1p7b/baseline" / f"{dataset}.json"
            if not path.is_file():
                failures.append({"baseline": str(path), "error": "missing"})
            else:
                baselines.append(str(path.relative_to(REPO)))
    return {"schema_version": "basis_a5_comparator_manifest_v1", "status": "PASS" if not failures else "FAIL",
            "a4_protocol_hash": _a4_hash(), "comparators": rows, "baselines": baselines,
            "expected_comparator_cells": len(expected), "failures": failures}


def _a4_row_path(model: str, dataset: str, direction: str, side: str, layer: int) -> Path:
    if model == "whisper":
        if direction == "Conditioning":
            return A4 / "whisper" / "conditioning" / dataset / f"L{layer:02d}" / \
                f"conditioning_decoder_L{layer}_oracle_local_rho0.5.json"
        return A4 / "whisper" / "raw" / side / dataset / f"L{layer:02d}" / \
            f"Raw_{side}_L{layer}_oracle_local_rho0.5.json"
    root = "conditioning" if direction == "Conditioning" else "raw"
    return A4 / "qwen3_asr_1p7b" / root / dataset / f"L{layer:02d}" / \
        f"{direction}_{side}_L{layer}_oracle_local_rho0.5.json"


def freeze() -> int:
    payload = _protocol_payload()
    audit = comparator_audit()
    if audit["status"] != "PASS":
        _write(OUT / "reuse/comparator_manifest.json", audit)
        raise RuntimeError(f"A4 comparator audit failed: {len(audit['failures'])} failures")
    payload["git"] = _git_state()
    payload["protocol_hash"] = _hash_obj(payload)
    payload["status"] = "FROZEN_PRE_RUN"
    preserved = {
        "schema_version": "basis_a5_nonblocking_stability_amendment_v1",
        "status": "ACTIVE",
        "policy": "diagnostic_nonblocking",
        "primary_rank": 32,
        "previous_gate": str(PRESERVED_RANK_GATE.relative_to(REPO)),
        "previous_gate_sha256": (_sha256_file(PRESERVED_RANK_GATE)
                                 if PRESERVED_RANK_GATE.is_file() else None),
        "previous_gate_status": (_json(PRESERVED_RANK_GATE).get("status")
                                 if PRESERVED_RANK_GATE.is_file() else "MISSING"),
        "previous_slurm_job": "53047",
        "previous_commits": ["de3a3e0", "9d31b10"],
        "rank_fixed_before_downstream_outcomes": True,
        "mer_pier_never_select_rank": True,
        "unstable_layers_labeled": True,
        "amendment_sha256": _sha256_file(A5_AMENDMENT),
    }
    _write(OUT / "manifests/nonblocking_stability_amendment.json", preserved)
    _write(OUT / "spec/protocol_payload.json", payload)
    _write(OUT / "manifests/a5_protocol_freeze.json", payload)
    _write(OUT / "reuse/comparator_manifest.json", audit)
    for sub in ("directions", "whisper", "qwen", "tables", "figures", "geometry", "runtime", "manifests"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    return 0


def construction_test() -> int:
    """CPU gate for the math and frozen A4 prerequisite artifacts."""
    from csasr.experiments.basis_a5_unique_shared import construct_unique_shared
    rng = np.random.default_rng(20260919)
    a = rng.normal(size=(128, 16)); b = rng.normal(size=(128, 16))
    a[:, 0] += 3.0; b[:, 1] += 1.0
    out = construct_unique_shared(a.T @ a, len(a), a.sum(0), b.T @ b, len(b), b.sum(0), rank=8)
    checks = {
        "finite": bool(np.isfinite(out.v_unique).all() and np.isfinite(out.v_shared).all()),
        "unit": bool(all(abs(np.linalg.norm(x) - 1) < 2e-5 for x in
                         (out.v_unique, out.v_shared, out.unique_minus_shared))),
        "orthogonal": bool(abs(float(out.v_unique @ out.v_shared)) <= 1e-3),
        "distinct_indices": bool(out.diagnostics["i_unique"] != out.diagnostics["i_shared"]),
    }
    qa = _json(A4 / "acceptance/qwen_acceptance.json")
    checks["a4_qwen_acceptance"] = qa.get("status") == "PASS"
    checks["a4_final_complete"] = _json(A4 / "FINAL_MANIFEST.json").get("status") == "COMPLETE"
    report = {"schema_version": "basis_a5_construction_test_v1", "status": "PASS" if all(checks.values()) else "FAIL",
              "checks": checks, "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"]}
    _write(OUT / "manifests/construction_test.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(report)
    return 0


class _Moments:
    """Streaming uncentered first/second moments for one group."""
    def __init__(self, dim: int):
        self.count = 0
        self.sum = np.zeros(int(dim), dtype=np.float64)
        self.second = np.zeros((int(dim), int(dim)), dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.sum.shape[0]:
            raise ValueError(f"moment update shape {x.shape}, expected (*,{self.sum.shape[0]})")
        if not len(x):
            return
        self.count += int(len(x))
        self.sum += x.sum(axis=0)
        self.second += x.T @ x

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, count=np.asarray(self.count, dtype=np.int64),
                            sum=self.sum, second=self.second)


def _moment(path: Path) -> _Moments:
    z = np.load(path)
    out = _Moments(int(z["sum"].shape[0]))
    out.count = int(z["count"])
    out.sum = np.asarray(z["sum"], dtype=np.float64)
    out.second = np.asarray(z["second"], dtype=np.float64)
    return out


def _construct_manifest(model: str) -> dict:
    return _json(OUT / "directions" / model / "manifest.json")


def _save_directions(model: str, side: str, layer: int, moments_a: _Moments,
                     moments_b: _Moments, *, dim: int) -> dict:
    from csasr.experiments.basis_a5_unique_shared import construct_unique_shared
    out = OUT / "directions" / model / side / f"L{layer:02d}"
    out.mkdir(parents=True, exist_ok=True)
    moments_a.save(out / "moments_A.npz")
    moments_b.save(out / "moments_B.npz")
    built = construct_unique_shared(moments_a.second, moments_a.count, moments_a.sum,
                                    moments_b.second, moments_b.count, moments_b.sum,
                                    rank=32, top_vectors=64)
    dirs = {"v_unique": built.v_unique, "v_shared": built.v_shared,
            "neg_shared": built.neg_shared, "pos_unique": built.pos_unique,
            "unique_minus_shared": built.unique_minus_shared}
    hashes = {}
    for name, vec in dirs.items():
        path = out / f"{name}.npy"
        np.save(path, np.asarray(vec, dtype=np.float32))
        hashes[name] = _array_hash(path)
    raw = None; cond = None
    try:
        raw = np.asarray(np.load(_a4_direction_path(model, side, layer, "raw")), dtype=np.float64)
        raw = raw / np.linalg.norm(raw)
        if side == "decoder":
            cond = np.asarray(np.load(_a4_direction_path(model, side, layer, "conditioning")), dtype=np.float64)
            cond = cond / np.linalg.norm(cond)
    except Exception:
        pass
    diag = dict(built.diagnostics)
    diag.update({
        "schema_version": "basis_a5_unique_shared_direction_v1",
        "model": model, "side": side, "layer": int(layer), "dim": int(dim),
        "construction_role": "D-construct", "rank": 32,
        "site": (SITE_WHISPER if model == "whisper" else SITE_QWEN)[side],
        "model_revision": _model_revision(model), "a4_protocol_hash": _a4_hash(),
        "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
        "git_commit": _git_state().get("commit"),
        "construction_population": "corpus_aggregated_D-construct",
        "construction_count_A": int(moments_a.count), "construction_count_B": int(moments_b.count),
        "direction_hashes": hashes,
        "cos_unique_raw_a4": None if raw is None else float(built.v_unique @ raw),
        "cos_shared_raw_a4": None if raw is None else float(built.v_shared @ raw),
        "cos_unique_conditioning_a4": None if cond is None else float(built.v_unique @ cond),
        "cos_shared_conditioning_a4": None if cond is None else float(built.v_shared @ cond),
        "a4_raw_direction_hash": _a4_row(model, "cs_dialogue", "Raw", side, layer).get("direction_hash"),
        "a4_conditioning_direction_hash": (None if side != "decoder" else
                                             _a4_row(model, "cs_dialogue", "Conditioning", side, layer).get("direction_hash")),
        "a4_site_hash": _a4_row(model, "cs_dialogue", "Raw", side, layer).get("site_hash"),
        "a4_mask_hash": _a4_row(model, "cs_dialogue", "Raw", side, layer).get("mask_hash"),
        "a4_scale": _a4_scale(model, side, layer),
        "projection_energy_interpretation": {
            "A_unique": diag["projection_energy_A_unique"],
            "B_unique": diag["projection_energy_B_unique"],
            "A_shared": diag["projection_energy_A_shared"],
            "B_shared": diag["projection_energy_B_shared"],
        },
    })
    _write(out / "diagnostics.json", diag)
    return diag


def refresh_direction_manifests() -> int:
    """Re-emit provenance from existing corpus moments without model execution."""
    manifests = {}
    for model, sides in (("whisper", (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC))),
                         ("qwen3_asr_1p7b", (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)))):
        layers_meta = {"encoder": {}, "decoder": {}}
        for side, layers in sides:
            for layer in layers:
                root = OUT / "directions" / model / side / f"L{layer:02d}"
                a_path, b_path = root / "moments_A.npz", root / "moments_B.npz"
                if not a_path.is_file() or not b_path.is_file():
                    raise FileNotFoundError(f"missing corpus moments: {root}")
                # The construction artifacts are already frozen.  Refresh
                # only provenance here; recomputing 116 eigensystems would be
                # unnecessary and could create concurrent-write risk at the
                # commit boundary.
                diag_path = root / "diagnostics.json"
                diag = _json(diag_path)
                a, b = _moment(a_path), _moment(b_path)
                diag.update({
                    "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
                    "git_commit": _git_state().get("commit"),
                    "construction_population": "corpus_aggregated_D-construct",
                    "construction_count_A": int(a.count), "construction_count_B": int(b.count),
                })
                _write(diag_path, diag)
                layers_meta[side][str(layer)] = diag
        manifests[model] = {"schema_version": "basis_a5_direction_manifest_v2",
                            "model": model, "source_role": "D-construct",
                            "construction_population": "full eligible baseline-correct corpus samples",
                            "rank": 32, "model_revision": _model_revision(model),
                            "git_commit": _git_state().get("commit"),
                            "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
                            "site": SITE_WHISPER if model == "whisper" else SITE_QWEN,
                            "layers": layers_meta}
        _write(OUT / "directions" / model / "manifest.json", manifests[model])
    _write(OUT / "manifests/direction_refresh.json", {
        "schema_version": "basis_a5_direction_refresh_v1", "status": "PASS",
        "rank": 32, "source_role": "D-construct", "models": list(manifests),
        "qwen_encoder_layers": len(manifests["qwen3_asr_1p7b"]["layers"]["encoder"]),
        "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
    })
    return 0


def _whisper_context():
    from experiments.dg03_build_basis import _load_construct
    from csasr.experiments.v2r3_directions import matrix_runs
    cfg, manifest, construct = _load_construct()
    construct = construct[construct["all_correct"]].copy()
    return cfg, manifest, construct, matrix_runs(
        __import__("csasr.lss.align.conventions", fromlist=["read_development_candidates"])
        .read_development_candidates(Path(cfg["experiment"]["output_root"]) / "alignments/candidates_all.parquet")[0])


def _qwen_context():
    from experiments.dg03_build_basis import _load_construct
    cfg, manifest, construct = _load_construct()
    construct = construct[construct["all_correct"]].copy()
    return cfg, manifest, construct


def _span_token_indices_whisper(bundle, norm, units, spans, ref_ids):
    from csasr.experiments.v2r3_directions import unit_char_spans
    from csasr.data.alignment import token_char_offsets
    char_spans = unit_char_spans(norm, units)
    offsets = token_char_offsets(bundle.processor.tokenizer, ref_ids)
    emb = []
    for span in spans:
        for index in span["unit_indices"]:
            index = int(index)
            if index >= len(char_spans):
                continue
            lo, hi = char_spans[index]
            emb.extend(i for i, (a, b) in enumerate(offsets) if b > lo and a < hi)
    return sorted(set(i for i in emb if i < len(ref_ids)))


def _preceding_indices(embedded: list[int], n_tokens: int | None = None) -> list[int]:
    if not embedded:
        return []
    n = int(n_tokens or len(embedded))
    lo = max(0, min(embedded) - n)
    return list(range(lo, min(embedded)))


def construct_whisper() -> int:
    import torch
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.experiments.v2r3_directions import control_frames, _frame_range
    from csasr.lss.encoder_sites import EncoderPostSelfAttnRecorder
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.generation import teacher_forced_forward
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config

    _stage = OUT / "manifests" / f"construct_whisper_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    _write(_stage, {"schema_version": "basis_a5_stage_manifest_v1", "stage": "construct_whisper",
                    "status": "RUNNING", "git": _git_state(), "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"]})
    _, manifest, construct, matrix = _whisper_context()
    by_uid = {str(u): g for u, g in construct.groupby("utterance_id")}
    rows = manifest[manifest["utterance_id"].astype(str).isin(by_uid)].sort_values("duration_sec")
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    ma_e = {l: _Moments(bundle.d_model) for l in WHISPER_ENC}; mb_e = {l: _Moments(bundle.d_model) for l in WHISPER_ENC}
    ma_d = {l: _Moments(bundle.d_model) for l in WHISPER_DEC}; mb_d = {l: _Moments(bundle.d_model) for l in WHISPER_DEC}
    prefix = build_prefix(bundle.processor, language="zh")
    eot = bundle.processor.tokenizer.eos_token_id
    for number, (_, row0) in enumerate(rows.iterrows(), 1):
        row = row0.to_dict(); uid = str(row["utterance_id"]); spans = by_uid[uid]
        features = batch_model_inputs(bundle, [row["audio_path"]])["input_features"]
        with EncoderPostSelfAttnRecorder(bundle, WHISPER_ENC) as rec, torch.inference_mode():
            bundle.model.model.encoder(features)
        for layer in WHISPER_ENC:
            block = rec.states[layer][0].float().cpu().numpy()
            valid = bundle.valid_frames(float(row["duration_sec"]))
            for _, span in spans.iterrows():
                lo, hi = _frame_range(span["start_sec"], span["end_sec"], bundle.encoder_step_sec, valid)
                control = control_frames(lo, hi, matrix[matrix["utterance_id"].astype(str) == uid],
                                         bundle.encoder_step_sec, valid)
                if hi - lo < 3 or control is None:
                    continue
                clo, chi = control
                ma_e[layer].add(block[lo:hi]); mb_e[layer].add(block[clo:chi])
        norm, units = normalize_and_segment(row["transcript_raw"])
        text_ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
        if not text_ids:
            continue
        seq = list(prefix) + list(text_ids) + [eot]
        with DecoderPostCrossAttnRecorder(bundle, WHISPER_DEC) as rec, torch.inference_mode():
            teacher_forced_forward(bundle, [row["audio_path"]], [seq])
        states = {l: rec.states[l][0].float().cpu().numpy() for l in WHISPER_DEC}
        emb = _span_token_indices_whisper(bundle, norm, units, spans.to_dict("records"), text_ids)
        matrix_idx = _preceding_indices(emb)
        apos = [len(prefix) + i - 1 for i in emb if len(prefix) + i - 1 >= 0]
        bpos = [len(prefix) + i - 1 for i in matrix_idx if len(prefix) + i - 1 >= 0]
        if not apos or not bpos:
            continue
        for layer in WHISPER_DEC:
            ma_d[layer].add(states[layer][apos]); mb_d[layer].add(states[layer][bpos])
        if number % 10 == 0:
            print(f"A5 Whisper construction {number}/{len(rows)}", flush=True)
    model_manifest = {"schema_version": "basis_a5_direction_manifest_v1", "model": "whisper",
                     "source_role": "D-construct", "n_utterances": int(len(rows)),
                     "model_revision": bundle.revision, "site": SITE_WHISPER,
                     "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
                     "layers": {"encoder": {}, "decoder": {}}}
    for side, layers, aa, bb in (("encoder", WHISPER_ENC, ma_e, mb_e), ("decoder", WHISPER_DEC, ma_d, mb_d)):
        for layer in layers:
            model_manifest["layers"][side][str(layer)] = _save_directions("whisper", side, layer, aa[layer], bb[layer], dim=bundle.d_model)
    _write(OUT / "directions/whisper/manifest.json", model_manifest)
    _write(_stage, {"schema_version": "basis_a5_stage_manifest_v1", "stage": "construct_whisper",
                    "status": "COMPLETED", "git": _git_state(), "protocol_hash": model_manifest["protocol_hash"],
                    "model_revision": bundle.revision})
    return 0


def _qwen_audio_frame_count(mel_length: int) -> int:
    leave = int(mel_length) % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (int(mel_length) // 100) * 13


def construct_qwen() -> int:
    import torch
    from csasr.data.normalize import normalize_and_segment
    from csasr.models.qwen3_asr import inputs_for_audio, load_qwen
    from csasr.lss.qwen_sites import QwenAudioSiteRecorder, QwenTextSiteRecorder
    _, manifest, construct = _qwen_context()
    by_uid = {str(u): g for u, g in construct.groupby("utterance_id")}
    rows = manifest[manifest["utterance_id"].astype(str).isin(by_uid)].sort_values("duration_sec")
    bundle = load_qwen(device="cuda:0"); bundle.model.eval()
    ma_e = {l: _Moments(bundle.encoder_dim) for l in QWEN_ENC}; mb_e = {l: _Moments(bundle.encoder_dim) for l in QWEN_ENC}
    ma_d = {l: _Moments(bundle.decoder_dim) for l in QWEN_DEC}; mb_d = {l: _Moments(bundle.decoder_dim) for l in QWEN_DEC}
    from experiments.basis_a4_qwen import _content_plan, _span_token_indices, _content_start
    for number, (_, row0) in enumerate(rows.iterrows(), 1):
        row = row0.to_dict(); uid = str(row["utterance_id"]); spans = by_uid[uid]
        norm, units, ref_ids, _, _ = _content_plan(bundle, row, "Chinese")
        inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese", transcript=norm)
        with torch.inference_mode(), QwenAudioSiteRecorder(bundle, QWEN_ENC) as ae, QwenTextSiteRecorder(bundle, QWEN_DEC) as de:
            bundle.thinker_model(**inp, use_cache=False)
        astates = {l: ae.states[l].numpy() for l in QWEN_ENC}
        zstates = {l: de.states[l].numpy()[0] for l in QWEN_DEC}
        n = int(astates[0].shape[0]); fps = n / max(float(row["duration_sec"]), 1e-6)
        emb = []
        for span in spans.to_dict("records"):
            emb.extend(_span_token_indices(bundle, row, span, ref_ids, norm, units))
        emb = sorted(set(i for i in emb if i < len(ref_ids)))
        midx = _preceding_indices(emb)
        if not emb or not midx:
            continue
        for layer in QWEN_ENC:
            for span in spans.to_dict("records"):
                lo = max(0, int(math.floor(float(span["start_sec"]) * fps)))
                hi = min(n, int(math.ceil(float(span["end_sec"]) * fps)))
                if hi <= lo:
                    continue
                width = hi - lo; clo = max(0, lo - width); chi = lo
                if chi > clo:
                    ma_e[layer].add(astates[layer][lo:hi]); mb_e[layer].add(astates[layer][clo:chi])
        # Teacher-forced input_ids contain expanded audio placeholders and the
        # language/control suffix before the reference.  The frozen A4 helper
        # locates the actual reference subsequence; raw prompt length is not a
        # valid teacher-forced index here.
        content_start = _content_start(inp["input_ids"], ref_ids)
        apos = [content_start + i - 1 for i in emb if i > 0]
        bpos = [content_start + i - 1 for i in midx if i > 0]
        for layer in QWEN_DEC:
            ma_d[layer].add(zstates[layer][apos]); mb_d[layer].add(zstates[layer][bpos])
        if number % 10 == 0:
            print(f"A5 Qwen construction {number}/{len(rows)}", flush=True)
    model_manifest = {"schema_version": "basis_a5_direction_manifest_v1", "model": "qwen3_asr_1p7b",
                     "source_role": "D-construct", "n_utterances": int(len(rows)),
                     "model_revision": bundle.revision, "site": SITE_QWEN,
                     "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
                     "layers": {"encoder": {}, "decoder": {}}}
    for side, layers, aa, bb, dim in (("encoder", QWEN_ENC, ma_e, mb_e, bundle.encoder_dim),
                                      ("decoder", QWEN_DEC, ma_d, mb_d, bundle.decoder_dim)):
        for layer in layers:
            model_manifest["layers"][side][str(layer)] = _save_directions("qwen3_asr_1p7b", side, layer, aa[layer], bb[layer], dim=dim)
    _write(OUT / "directions/qwen3_asr_1p7b/manifest.json", model_manifest)
    return 0


def _direction_audit_one(model: str, side: str, layer: int) -> dict:
    d = _json(OUT / "directions" / model / side / f"L{layer:02d}/diagnostics.json")
    required = ["v_unique", "v_shared", "neg_shared", "pos_unique", "unique_minus_shared"]
    vectors = {name: np.asarray(np.load(OUT / "directions" / model / side / f"L{layer:02d}/{name}.npy"), dtype=np.float64)
               for name in required}
    checks = {
        "finite": all(bool(np.isfinite(v).all()) for v in vectors.values()),
        "unit": all(abs(float(np.linalg.norm(v)) - 1.0) <= 2e-5 for v in vectors.values()),
        "unique_shared_orthogonal": abs(float(vectors["v_unique"] @ vectors["v_shared"])) <= 1e-3,
        "indices_distinct": int(d["i_unique"]) != int(d["i_shared"]),
        "minus_shared_sign": bool(np.allclose(vectors["neg_shared"], -vectors["v_shared"], atol=2e-6)),
        "pos_unique_sign": bool(np.allclose(vectors["pos_unique"], vectors["v_unique"], atol=2e-6)),
        "composite_formula": bool(np.allclose(
            vectors["unique_minus_shared"],
            (vectors["v_unique"] - vectors["v_shared"]) /
            np.linalg.norm(vectors["v_unique"] - vectors["v_shared"]), atol=2e-6)),
        "nondegenerate_support": int(d["count_a"]) >= 8 and int(d["count_b"]) >= 8,
    }
    return {"model": model, "side": side, "layer": int(layer), "checks": checks,
            "diagnostics": d, "status": "PASS" if all(checks.values()) else "FAIL"}


def direction_audit() -> int:
    results = []
    for model, sides in (("whisper", (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC))),
                         ("qwen3_asr_1p7b", (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)))):
        for side, layers in sides:
            for layer in layers:
                results.append(_direction_audit_one(model, side, layer))
    failed = [x for x in results if x["status"] != "PASS"]
    report = {"schema_version": "basis_a5_direction_audit_v1", "status": "PASS" if not failed else "FAIL",
              "rank": 32, "results": results, "failed": failed,
              "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"]}
    _write(OUT / "manifests/direction_audit.json", report)
    if failed:
        raise RuntimeError(f"A5 direction audit failed: {len(failed)} layers")
    return 0


def rank_stability() -> int:
    from csasr.experiments.basis_a5_unique_shared import rank_directions
    rows = []
    for model, sides in (("whisper", (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC))),
                         ("qwen3_asr_1p7b", (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)))):
        for side, layers in sides:
            anchor_layers = tuple(layer for layer in A4_STABILITY_ANCHORS if layer in layers)
            for layer in anchor_layers:
                ddir = OUT / "directions" / model / side / f"L{layer:02d}"
                a = _moment(ddir / "moments_A.npz"); b = _moment(ddir / "moments_B.npz")
                grid = rank_directions(a.second, a.count, a.sum, b.second, b.count, b.sum,
                                       ranks=(16, 32, 64))
                ref = grid["32"]
                rows.append({"model": model, "side": side, "layer": int(layer),
                             "counts": {"A": a.count, "B": b.count},
                             "cos_abs_unique_16_vs_32": abs(float(grid["16"].v_unique @ ref.v_unique)),
                             "cos_abs_unique_64_vs_32": abs(float(grid["64"].v_unique @ ref.v_unique)),
                             "cos_abs_shared_16_vs_32": abs(float(grid["16"].v_shared @ ref.v_shared)),
                             "cos_abs_shared_64_vs_32": abs(float(grid["64"].v_shared @ ref.v_shared))})
    medians = {}
    for model in ("whisper", "qwen3_asr_1p7b"):
        for side in ("encoder", "decoder"):
            group = [x for x in rows if x["model"] == model and x["side"] == side]
            key = f"{model}:{side}"
            if not group:
                # The A4 decoder anchor indices do not intersect the 24-layer
                # Qwen audio encoder.  Record that explicitly; it is not a
                # failed gate and does not authorize a substituted layer.
                medians[key] = {"status": "NOT_APPLICABLE", "n_layers": 0}
            else:
                medians[key] = {
                    k: float(np.median([x[k] for x in group]))
                    for k in ("cos_abs_unique_16_vs_32", "cos_abs_unique_64_vs_32",
                              "cos_abs_shared_16_vs_32", "cos_abs_shared_64_vs_32")}
                medians[key]["n_layers"] = len(group)
    numeric_medians = [v for m in medians.values() for k, v in m.items()
                       if k != "n_layers" and isinstance(v, (int, float))]
    finite_medians = [v for v in numeric_medians if np.isfinite(v)]
    passed = (bool(numeric_medians) and len(finite_medians) == len(numeric_medians)
              and all(v >= 0.90 for v in finite_medians))
    if len(finite_medians) != len(numeric_medians):
        for key, values in medians.items():
            for metric, value in list(values.items()):
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    values[metric] = None
    report = {"schema_version": "basis_a5_rank_stability_v1", "status": "PASS" if passed else "FAIL",
              "primary_rank": 32, "ranks": [16, 32, 64],
              "anchor_policy": "A4 frozen anchor/candidate indices {24,26,27,31}, clipped to model depth",
              "medians": medians, "layers": rows,
              "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"]}
    _write(OUT / "manifests/rank_stability.json", report)
    if not passed:
        raise RuntimeError(f"A5 rank stability gate failed: {medians}")
    return 0


def _stability_status_index() -> dict[tuple[str, str, int], str]:
    """Read-only status view; never rewrites the historical rank gate."""
    index: dict[tuple[str, str, int], str] = {}
    for path in (PRESERVED_RANK_GATE, RANK_DIAGNOSTIC):
        if not path.is_file():
            continue
        for row in _json(path).get("layers", []):
            values = [row.get(k) for k in (
                "cos_abs_unique_16_vs_32", "cos_abs_unique_64_vs_32",
                "cos_abs_shared_16_vs_32", "cos_abs_shared_64_vs_32")]
            if any(v is None for v in values):
                continue
            key = (str(row["model"]), str(row["side"]), int(row["layer"]))
            status = "STABLE" if all(float(v) >= 0.90 for v in values) else "UNSTABLE"
            # The preserved failure is authoritative when both manifests have
            # a row for the same layer.
            if key not in index or path == PRESERVED_RANK_GATE:
                index[key] = status
    return index


def _stability_status(model: str, side: str, layer: int) -> str:
    return _stability_status_index().get((model, side, int(layer)), "NOT_TESTED")


def rank_stability_diagnostic() -> int:
    """Record full-atlas labels without mutating the preserved failed gate."""
    if not PRESERVED_RANK_GATE.is_file():
        raise RuntimeError("preserved rank gate is missing")
    old = _json(PRESERVED_RANK_GATE)
    amendment_path = OUT / "manifests/nonblocking_stability_amendment.json"
    amendment = _json(amendment_path) if amendment_path.is_file() else {}
    if amendment.get("previous_gate_sha256") and _sha256_file(PRESERVED_RANK_GATE) != amendment["previous_gate_sha256"]:
        raise RuntimeError("preserved rank gate hash changed")
    old_rows = {(str(x["model"]), str(x["side"]), int(x["layer"])): x
                for x in old.get("layers", [])}
    rows = []
    for model, sides in (("whisper", (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC))),
                         ("qwen3_asr_1p7b", (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)))):
        for side, layers in sides:
            for layer in layers:
                old_row = old_rows.get((model, side, int(layer)))
                row = {"model": model, "side": side, "layer": int(layer),
                       "status": _stability_status(model, side, layer),
                       "source": "preserved_rank_stability.json" if old_row else "not_tested_by_preserved_gate"}
                if old_row:
                    row.update({k: old_row.get(k) for k in old_row if k.startswith("cos_abs_")})
                rows.append(row)
    report = {
        "schema_version": "basis_a5_rank_stability_diagnostic_v1",
        "status": "DIAGNOSTIC_ONLY", "blocking": False, "primary_rank": 32,
        "rank_fixed_before_downstream_outcomes": True,
        "mer_pier_never_select_rank": True, "preserved_gate_status": old.get("status"),
        "preserved_gate_sha256": _sha256_file(PRESERVED_RANK_GATE),
        "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
        "layers": rows,
        "counts": {s: sum(x["status"] == s for x in rows)
                   for s in ("STABLE", "UNSTABLE", "NOT_TESTED")},
    }
    _write(RANK_DIAGNOSTIC, report)
    return 0


def pre_gpu_gates() -> int:
    """CPU-only gates required before amended atlas submission."""
    checks = {
        "amendment_exists": A5_AMENDMENT.is_file(),
        "preserved_rank_gate_exists": PRESERVED_RANK_GATE.is_file(),
        "qwen_local_repair_pass": _json(A4 / "manifests/qwen_local_repair_completeness.json").get("status") == "PASS",
        "rank_is_32": True, "rho_is_0.5": RHO == 0.5, "scope_is_oracle_local": True,
    }
    if checks["preserved_rank_gate_exists"]:
        checks["preserved_rank_gate_failed_not_passed"] = _json(PRESERVED_RANK_GATE).get("status") == "FAIL"
    else:
        checks["preserved_rank_gate_failed_not_passed"] = False
    missing: list[str] = []
    invalid: list[dict] = []
    expected = {
        "whisper": {"encoder": WHISPER_ENC, "decoder": WHISPER_DEC},
        "qwen3_asr_1p7b": {"encoder": QWEN_ENC, "decoder": QWEN_DEC},
    }
    for model, sides in expected.items():
        manifest_path = OUT / "directions" / model / "manifest.json"
        if not manifest_path.is_file():
            missing.append(str(manifest_path.relative_to(REPO))); continue
        manifest = _json(manifest_path)
        if manifest.get("source_role") != "D-construct":
            invalid.append({"model": model, "reason": "direction manifest source_role"})
        for side, layers in sides.items():
            listed = manifest.get("layers", {}).get(side, {})
            for layer in layers:
                root = OUT / "directions" / model / side / f"L{layer:02d}"
                if str(layer) not in listed or not (root / "diagnostics.json").is_file():
                    missing.append(str(root.relative_to(REPO))); continue
                d = _json(root / "diagnostics.json")
                paths = {name: root / f"{name}.npy" for name in
                         ("v_unique", "v_shared", "neg_shared", "pos_unique", "unique_minus_shared")}
                if any(not p.is_file() for p in paths.values()):
                    missing.append(str(root.relative_to(REPO))); continue
                vectors = {name: np.asarray(np.load(path), dtype=np.float64)
                           for name, path in paths.items()}
                unit_finite = all(np.isfinite(v).all() and abs(np.linalg.norm(v) - 1.0) <= 2e-5
                                  for v in vectors.values())
                composite = (vectors["v_unique"] - vectors["v_shared"])
                valid = (d.get("construction_role") == "D-construct" and d.get("rank") == 32
                         and d.get("count_a", 0) > 0 and d.get("count_b", 0) > 0
                         and unit_finite and int(d.get("i_unique", -1)) != int(d.get("i_shared", -1))
                         and abs(float(vectors["v_unique"] @ vectors["v_shared"])) <= 1e-3
                         and np.allclose(vectors["unique_minus_shared"], composite / np.linalg.norm(composite), atol=2e-6))
                a4 = _a4_row(model, "cs_dialogue", "Raw", side, layer)
                valid = valid and bool(a4.get("site_hash")) and bool(a4.get("mask_hash"))
                if not valid:
                    invalid.append({"model": model, "side": side, "layer": layer,
                                    "reason": "vector/provenance/hash invariant"})
    checks["all_direction_artifacts"] = not missing
    checks["all_vectors_and_provenance_valid"] = not invalid
    report = {"schema_version": "basis_a5_pre_gpu_gates_v1",
              "status": "PASS" if all(checks.values()) else "FAIL", "blocking": True,
              "checks": checks, "missing": missing, "invalid": invalid,
              "qwen_encoder_direction_count": len(list((OUT / "directions/qwen3_asr_1p7b/encoder").glob("L*/diagnostics.json"))),
              "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
              "preserved_rank_gate": str(PRESERVED_RANK_GATE.relative_to(REPO)),
              "amendment": str(A5_AMENDMENT.relative_to(REPO))}
    _write(OUT / "manifests/pre_gpu_gates.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(report)
    return 0


def _a5_direction(model: str, side: str, layer: int, name: str) -> tuple[np.ndarray, str, dict]:
    ddir = OUT / "directions" / model / side / f"L{layer:02d}"
    path = ddir / f"{name}.npy"
    diag = _json(ddir / "diagnostics.json")
    return np.asarray(np.load(path), dtype=np.float32), _array_hash(path), diag


def _result_provenance(model: str, dataset: str, side: str, layer: int,
                       direction: str, dhash: str, diag: dict) -> dict:
    row = _a4_row(model, dataset, "Conditioning" if model == "whisper" and side == "decoder" else "Raw", side, layer)
    return {
        "git_commit": _git_state().get("commit"),
        "model_revision": _model_revision(model),
        "config_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
        "panel_fingerprint": _panel(dataset)["fingerprint"],
        "direction_hash": dhash,
        "mask_hash": row["mask_hash"],
        "site_hash": row["site_hash"],
        "rank": 32,
        "rank_stability_status": _stability_status(model, side, layer),
        "rho": RHO,
        "a4_comparator_direction": row.get("direction_hash"),
        "a4_protocol_hash": _a4_hash(),
        "direction_name": direction,
    }


def _canonical_metrics(dataset: str, base: dict[str, str], texts: dict[str, str], model: str) -> dict:
    from csasr.evaluation import canonical, retention
    refs = [str(x["reference"]) for x in _panel(dataset)["rows"]]
    ids = [str(x["utterance_id"]) for x in _panel(dataset)["rows"]]
    b = [base[x] for x in ids]; m = [texts[x] for x in ids]
    bm = canonical.corpus_metrics(refs, b); mm = canonical.corpus_metrics(refs, m)
    tr = canonical.correction_corruption(refs, b, m)
    out = dict(mm); out.update(canonical.error_metric_gains(bm, mm))
    out.update({"transitions": tr, "poi_corrections": tr["corrections"],
                "poi_corruptions": tr["corruptions"],
                "poi_net_utility": tr["corrections"] - tr["corruptions"],
                "retention": retention.retention_report(refs, b, m),
                "outside_harm": None})
    return out


def _whisper_baseline(dataset: str) -> dict[str, str]:
    paths = [A4 / "baselines" / f"{dataset}.json",
             REPO / "results/basis_a3_raw_cond_scope_depth/baselines" / f"{dataset}.json"]
    for path in paths:
        if path.is_file():
            return _json(path)["texts"]
    raise FileNotFoundError(f"Whisper A4 baseline missing for {dataset}")


def _whisper_decode_cell(bundle, dataset: str, side: str, layer: int,
                         direction: str, vector: np.ndarray, dhash: str,
                         diag: dict) -> dict:
    import torch
    from csasr.experiments.basis_a3 import _cs_eval_spans, _decoder_cached_inputs, _decoder_gate, _encoder_gain
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs
    from experiments.basis_a3 import GEN
    frame = _panel(dataset); base = _whisper_baseline(dataset); texts = {}; records = []
    cs_spans = _cs_eval_spans() if dataset == "cs_dialogue" and side == "encoder" else {}
    t0 = time.monotonic()
    for row in sorted(frame["rows"], key=lambda x: float(x["duration_sec"])):
        uid = str(row["utterance_id"])
        if side == "decoder":
            inp = _decoder_cached_inputs(bundle, uid, row["audio_path"])
            allowed = set(__import__("experiments.basis_a3", fromlist=["_decoder_positions"])._decoder_positions(bundle, row["reference"]))
            hook = DecoderPostCrossAttnInterventionHook(
                bundle, layer, torch.from_numpy(vector), alpha=RHO, scale=float(diag["a4_scale"]),
                num_forced_prefix=num_forced_prefix_from(bundle.processor),
                gate_fn=_decoder_gate(allowed), norm_preserve=True, record=True, record_last_only=False)
        else:
            inp = batch_model_inputs(bundle, [row["audio_path"]])
            gain = _encoder_gain(bundle, dataset, row, scope="oracle_local", cs_spans=cs_spans)
            hook = EncoderPostSelfAttnInterventionHook(bundle, layer, torch.from_numpy(vector),
                alpha=RHO, scale=float(diag["a4_scale"]), gain=gain, norm_preserve=True, record=True)
        with hook, torch.inference_mode():
            out = bundle.model.generate(**inp, **GEN)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        texts[uid] = bundle.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        records.extend(hook.records)
    metrics = _canonical_metrics(dataset, base, texts, "whisper")
    if side == "decoder":
        active = [r for r in records if r.steered]
        edited = len(active)
        energy = float(sum(r.edit_norm for r in active))
        pre = float(sum(r.pre_norm for r in active))
        # AuditRecord.row is the batch row, not a transcript-position index.
        # The denominator is the number of eligible transcript-generation
        # states represented by the audit records.
        total_positions = len(records)
    else:
        edited = int(sum(r.active_frames for r in records)); energy = float(sum(r.edit_norm for r in records))
        pre = float(sum(r.pre_norm * max(r.active_frames, 1) for r in records)); total_positions = sum(r.active_frames for r in records)
    metrics.update({"scope": "oracle_local", "side": side, "edited_positions_or_frames": edited,
                    "total_intervention_energy": energy, "original_hidden_norm": pre / max(total_positions, 1),
                    "mean_perturbation_norm": energy / max(edited, 1),
                    "relative_perturbation": energy / max(pre, 1e-12),
                    "edited_fraction": edited / max(total_positions, 1),
                    "runtime_seconds": time.monotonic() - t0})
    row = _a4_row("whisper", dataset, "Raw", side, layer)
    return {"schema_version": "basis_a5_result_v1", "model": "whisper", "dataset": dataset,
            "direction": direction, "side": side, "layer": int(layer), "scope": "oracle_local", "rho": RHO,
            "site": SITE_WHISPER[side], "site_hash": row["site_hash"], "mask_hash": row["mask_hash"],
            "panel_fingerprint": frame["fingerprint"], "model_revision": bundle.revision,
            "direction_hash": dhash, "rank": 32, "a4_protocol_hash": _a4_hash(),
            "rank_stability_status": _stability_status("whisper", side, layer),
            "config_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
            "provenance": _result_provenance("whisper", dataset, side, layer, direction, dhash, diag),
            "metrics": metrics, "texts": texts, "baseline_texts": base}


def atlas_whisper(dataset: str, side: str, direction: str, layer_start: int = 0,
                  layer_end: int | None = None) -> int:
    import torch
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    sides = ("encoder", "decoder") if side == "both" else (side,)
    directions = DIRECTIONS if direction == "all" else (direction,)
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    selected_by_side = {}
    for current_side in sides:
        layers = WHISPER_ENC if current_side == "encoder" else WHISPER_DEC
        selected = [l for l in layers if l >= int(layer_start) and (layer_end is None or l < int(layer_end))]
        selected_by_side[current_side] = selected
        for current_direction in directions:
            for layer in selected:
                vector, dhash, diag = _a5_direction("whisper", current_side, layer, current_direction)
                path = OUT / "whisper" / current_direction / current_side / dataset / f"L{layer:02d}" / \
                    f"{current_direction}_{current_side}_L{layer}_oracle_local_rho0.5.json"
                if path.is_file():
                    continue
                result = _whisper_decode_cell(bundle, dataset, current_side, layer, current_direction, vector, dhash, diag)
                _write(path, result)
                print(f"A5 Whisper completed {path.relative_to(REPO)}", flush=True)
    _write(OUT / "manifests" / f"atlas_whisper_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
           {"schema_version": "basis_a5_stage_manifest_v1", "status": "COMPLETED", "dataset": dataset,
            "side": side, "direction": direction, "layers": selected_by_side, "git": _git_state(),
            "rank_stability_policy": "diagnostic_nonblocking"})
    return 0


def _qwen_baseline(dataset: str) -> dict[str, str]:
    return _json(A4 / "qwen3_asr_1p7b/baseline" / f"{dataset}.json")["texts"]


def _qwen_local_masks(bundle, dataset: str, row: dict, fps: float, n: int, cs_cache=None):
    from experiments.basis_a4_qwen import (_load_cs_eval_spans, _seame_audio_spans,
                                           _seame_decoder_indices, _audio_mask,
                                           _content_plan, _span_token_indices,
                                           _decoder_local_positions)
    if dataset == "cs_dialogue":
        spans = cs_cache if cs_cache is not None else _load_cs_eval_spans()
        local_spans = spans.get(str(row["utterance_id"]), [])
        norm, units, ref_ids, _, _ = _content_plan(bundle, row, "Chinese")
        idx = []
        for sp in local_spans:
            idx.extend(_span_token_indices(bundle, row, sp, ref_ids, norm, units))
    else:
        local_spans = _seame_audio_spans(row)
        idx = _seame_decoder_indices(bundle, row)
    audio = _audio_mask(bundle, row, scope="oracle_local", spans={str(row["utterance_id"]): local_spans},
                        fps=fps, n=n)
    return audio, idx


def _qwen_decode_cell(bundle, dataset: str, side: str, layer: int,
                      direction: str, vector: np.ndarray, dhash: str, diag: dict) -> dict:
    import torch
    from csasr.models.qwen3_asr import generate_one, inputs_for_audio
    from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
    frame = _panel(dataset); base = _qwen_baseline(dataset); texts = {}; records = []; t0 = time.monotonic()
    fps = float(_json(A4 / "qwen3_asr_1p7b/directions/manifest.json")["measured_audio_fps_median"])
    cs_cache = None
    if dataset == "cs_dialogue":
        from experiments.basis_a4_qwen import _load_cs_eval_spans
        cs_cache = _load_cs_eval_spans()
    for row in sorted(frame["rows"], key=lambda x: float(x["duration_sec"])):
        uid = str(row["utterance_id"])
        if side == "encoder":
            inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
            from experiments.basis_a4_qwen import _qwen_audio_length
            n = _qwen_audio_length(int(inp["feature_attention_mask"].sum()))
            audio_mask, _ = _qwen_local_masks(bundle, dataset, row, fps, n, cs_cache)
            gain = torch.from_numpy(audio_mask).to(bundle.device).view(1, -1)
            hook = QwenSiteInterventionHook(bundle, layer, torch.from_numpy(vector), alpha=RHO,
                site=AUDIO_SITE, scale=float(diag["a4_scale"]), gain=gain, norm_preserve=True, record=True)
        else:
            inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
            prompt_len = int(inp["input_ids"].shape[1])
            _, idx = _qwen_local_masks(bundle, dataset, row, fps, 0, cs_cache)
            from experiments.basis_a4_qwen import _decoder_local_positions
            allowed = _decoder_local_positions(prompt_len, idx)
            excluded = set(getattr(bundle.processor.tokenizer, "all_special_ids", []))
            hook = QwenSiteInterventionHook(bundle, layer, torch.from_numpy(vector), alpha=RHO,
                site=TEXT_SITE, scale=float(diag["a4_scale"]), allowed_positions=allowed,
                excluded_token_ids=excluded, record=True)
        text, _ = generate_one(bundle, row["audio_path"], language="Chinese", hook=hook)
        texts[uid] = text; records.extend(hook.records)
    metrics = _canonical_metrics(dataset, base, texts, "qwen3_asr_1p7b")
    edited = int(sum(r.active_positions for r in records)); energy = float(sum(r.edit_norm_sum for r in records))
    pre = float(sum(r.active_pre_norm_sum for r in records)); eligible = int(sum(r.active_positions for r in records))
    seen = int(sum(r.n_positions for r in records))
    metrics.update({"scope": "oracle_local", "side": side, "edited_positions_or_frames": edited,
                    "total_intervention_energy": energy,
                    "original_hidden_norm": pre / max(eligible, 1),
                    "mean_perturbation_norm": energy / max(edited, 1),
                    "relative_perturbation": energy / max(pre, 1e-12),
                    "edited_fraction": edited / max(seen, 1), "runtime_seconds": time.monotonic() - t0})
    row = _a4_row("qwen3_asr_1p7b", dataset, "Raw", side, layer)
    return {"schema_version": "basis_a5_result_v1", "model": "qwen3_asr_1p7b", "dataset": dataset,
            "direction": direction, "side": side, "layer": int(layer), "scope": "oracle_local", "rho": RHO,
            "site": SITE_QWEN[side], "site_hash": row["site_hash"], "mask_hash": row["mask_hash"],
            "panel_fingerprint": frame["fingerprint"], "model_revision": bundle.revision,
            "direction_hash": dhash, "rank": 32, "a4_protocol_hash": _a4_hash(),
            "rank_stability_status": _stability_status("qwen3_asr_1p7b", side, layer),
            "config_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
            "provenance": _result_provenance("qwen3_asr_1p7b", dataset, side, layer, direction, dhash, diag),
            "metrics": metrics, "texts": texts, "baseline_texts": base}


def atlas_qwen(dataset: str, side: str, direction: str, layer_start: int = 0,
               layer_end: int | None = None) -> int:
    from csasr.models.qwen3_asr import load_qwen
    sides = ("encoder", "decoder") if side == "both" else (side,)
    directions = DIRECTIONS if direction == "all" else (direction,)
    bundle = load_qwen(device="cuda:0"); bundle.model.eval()
    selected_by_side = {}
    for current_side in sides:
        layers = QWEN_ENC if current_side == "encoder" else QWEN_DEC
        selected = [l for l in layers if l >= int(layer_start) and (layer_end is None or l < int(layer_end))]
        selected_by_side[current_side] = selected
        for current_direction in directions:
            for layer in selected:
                vector, dhash, diag = _a5_direction("qwen3_asr_1p7b", current_side, layer, current_direction)
                path = OUT / "qwen" / current_direction / current_side / dataset / f"L{layer:02d}" / \
                    f"{current_direction}_{current_side}_L{layer}_oracle_local_rho0.5.json"
                if path.is_file():
                    continue
                result = _qwen_decode_cell(bundle, dataset, current_side, layer, current_direction, vector, dhash, diag)
                _write(path, result)
                print(f"A5 Qwen completed {path.relative_to(REPO)}", flush=True)
    _write(OUT / "manifests" / f"atlas_qwen_{os.environ.get('SLURM_JOB_ID', 'local')}.json",
           {"schema_version": "basis_a5_stage_manifest_v1", "status": "COMPLETED", "dataset": dataset,
            "side": side, "direction": direction, "layers": selected_by_side, "git": _git_state(),
            "rank_stability_policy": "diagnostic_nonblocking"})
    return 0


def _preflight_whisper() -> dict:
    import torch
    from csasr.experiments.basis_a3 import _decoder_cached_inputs, _decoder_gate, _encoder_gain, GEN
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    rows = _panel("cs_dialogue")["rows"]
    row = sorted(rows, key=lambda x: float(x["duration_sec"]))[0]
    vec, _, diag = _a5_direction("whisper", "decoder", 0, "pos_unique")
    grad_probe = []
    def decode(hook=None):
        inp = _decoder_cached_inputs(bundle, str(row["utterance_id"]), row["audio_path"])
        ctx = hook if hook is not None else _null_context()
        with ctx, torch.inference_mode():
            grad_probe.append(bool(torch.is_grad_enabled()))
            out = bundle.model.generate(**inp, **GEN)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        return bundle.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
    torch.cuda.reset_peak_memory_stats(bundle.model.device if hasattr(bundle.model, "device") else "cuda")
    t0 = time.monotonic(); base = decode(); repeat = decode()
    off = DecoderPostCrossAttnInterventionHook(bundle, 0, torch.from_numpy(vec), alpha=0.0,
        scale=float(diag["a4_scale"]), num_forced_prefix=num_forced_prefix_from(bundle.processor),
        gate_fn=_decoder_gate(set()), norm_preserve=True, record=True, record_last_only=False)
    rho0 = decode(off)
    local = __import__("experiments.basis_a3", fromlist=["_decoder_positions"])._decoder_positions(bundle, row["reference"])
    hooked = DecoderPostCrossAttnInterventionHook(bundle, 0, torch.from_numpy(vec), alpha=RHO,
        scale=float(diag["a4_scale"]), num_forced_prefix=num_forced_prefix_from(bundle.processor),
        gate_fn=_decoder_gate(set(local)), norm_preserve=True, record=True, record_last_only=False)
    steered = decode(hooked)
    edited = sum(1 for r in hooked.records if r.steered)
    enc_vec, _, enc_diag = _a5_direction("whisper", "encoder", 0, "pos_unique")
    from experiments.basis_a3 import _cs_eval_spans
    inp = batch_model_inputs(bundle, [row["audio_path"]]); gain = _encoder_gain(bundle, "cs_dialogue", row, scope="oracle_local", cs_spans=_cs_eval_spans())
    eh = EncoderPostSelfAttnInterventionHook(bundle, 0, torch.from_numpy(enc_vec), alpha=RHO,
        scale=float(enc_diag["a4_scale"]), gain=gain, norm_preserve=True, record=True)
    with eh, torch.inference_mode():
        eo = bundle.model.generate(**inp, **GEN)
    enc_edited = sum(r.active_frames for r in eh.records)
    peak = int(torch.cuda.max_memory_allocated())
    no_gradients = not any(grad_probe)
    return {"model": "whisper", "status": "PASS" if base == repeat == rho0 and edited >= 0 and enc_edited >= 0 and no_gradients else "FAIL",
            "runtime_seconds": time.monotonic() - t0, "peak_vram_gb": peak / 2**30,
            "deterministic": base == repeat, "rho0_identity": base == rho0,
            "hook_off_identity": base == decode(), "decoder_local_positions": len(local),
            "decoder_edited_records": edited, "encoder_edited_frames": int(enc_edited),
            "no_gradients": no_gradients,
            "norm_preserve": bool(hooked.records or eh.records), "directions": ["neg_shared", "pos_unique", "unique_minus_shared"]}


def _preflight_qwen() -> dict:
    import torch
    from csasr.models.qwen3_asr import generate_one, inputs_for_audio, load_qwen
    from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
    from experiments.basis_a4_qwen import _decoder_local_positions, _qwen_audio_length
    bundle = load_qwen(device="cuda:0"); bundle.model.eval()
    row = sorted(_panel("cs_dialogue")["rows"], key=lambda x: float(x["duration_sec"]))[0]
    vec, _, diag = _a5_direction("qwen3_asr_1p7b", "decoder", 0, "pos_unique")
    inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
    grad_probe = []
    def gen(hook=None):
        with torch.inference_mode():
            grad_probe.append(bool(torch.is_grad_enabled()))
            return generate_one(bundle, row["audio_path"], language="Chinese", hook=hook)[0]
    t0 = time.monotonic(); base = gen(); repeat = gen()
    off = QwenSiteInterventionHook(bundle, 0, torch.from_numpy(vec), alpha=0.0, site=TEXT_SITE,
        scale=float(diag["a4_scale"]), allowed_positions=set(), record=True)
    rho0 = gen(off)
    _, idx = _qwen_local_masks(bundle, "cs_dialogue", row,
                               float(_json(A4 / "qwen3_asr_1p7b/directions/manifest.json")["measured_audio_fps_median"]), 0,
                               __import__("experiments.basis_a4_qwen", fromlist=["_load_cs_eval_spans"])._load_cs_eval_spans())
    prompt_len = int(inp["input_ids"].shape[1]); allowed = _decoder_local_positions(prompt_len, idx)
    hook = QwenSiteInterventionHook(bundle, 0, torch.from_numpy(vec), alpha=RHO, site=TEXT_SITE,
        scale=float(diag["a4_scale"]), allowed_positions=allowed,
        excluded_token_ids=set(getattr(bundle.processor.tokenizer, "all_special_ids", [])), record=True)
    steered = gen(hook)
    enc_vec, _, enc_diag = _a5_direction("qwen3_asr_1p7b", "encoder", 0, "pos_unique")
    inp2 = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
    n = _qwen_audio_length(int(inp2["feature_attention_mask"].sum()))
    audio, _ = _qwen_local_masks(bundle, "cs_dialogue", row,
                                 float(_json(A4 / "qwen3_asr_1p7b/directions/manifest.json")["measured_audio_fps_median"]), n,
                                 __import__("experiments.basis_a4_qwen", fromlist=["_load_cs_eval_spans"])._load_cs_eval_spans())
    eh = QwenSiteInterventionHook(bundle, 0, torch.from_numpy(enc_vec), alpha=RHO, site=AUDIO_SITE,
        scale=float(enc_diag["a4_scale"]), gain=torch.from_numpy(audio).view(1, -1), record=True)
    gen(eh)
    peak = int(torch.cuda.max_memory_allocated())
    no_gradients = not any(grad_probe)
    return {"model": "qwen3_asr_1p7b", "status": "PASS" if base == repeat == rho0 and no_gradients else "FAIL",
            "runtime_seconds": time.monotonic() - t0, "peak_vram_gb": peak / 2**30,
            "deterministic": base == repeat, "rho0_identity": base == rho0,
            "hook_off_identity": base == gen(), "decoder_local_positions": len(allowed),
            "decoder_edited_records": sum(r.active_positions for r in hook.records),
            "encoder_edited_frames": sum(r.active_positions for r in eh.records),
            "no_gradients": no_gradients,
            "norm_preserve": bool(hook.records and eh.records), "directions": list(DIRECTIONS)}


class _null_context:
    def __enter__(self): return self
    def __exit__(self, *_exc): return False


def preflight(model: str) -> int:
    report = _preflight_whisper() if model == "whisper" else _preflight_qwen()
    report.update({"schema_version": "basis_a5_gpu_preflight_v1", "git": _git_state(),
                   "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
                   "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown")})
    if report["peak_vram_gb"] > 40.0:
        report["status"] = "FAIL"
    _write(OUT / "runtime" / f"preflight_{model}.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(report)
    return 0


def _a5_files() -> list[Path]:
    return [p for p in (OUT / "whisper").rglob("*.json")] + [p for p in (OUT / "qwen").rglob("*.json")]


def _retention_rate(metrics: dict) -> float | None:
    ret = metrics.get("retention", {})
    for key in ("matrix_zh", "matrix", "embedded_en"):
        value = ret.get(key)
        if isinstance(value, dict) and value.get("rate") is not None:
            return float(value["rate"])
    return None


def _accepted_rows() -> list[dict]:
    rows = []
    for path in _a5_files():
        x = _json(path)
        if x.get("schema_version") != "basis_a5_result_v1":
            continue
        x["_path"] = str(path.relative_to(REPO))
        rows.append(x)
    return rows


def _row_to_table(x: dict) -> dict:
    m = x["metrics"]
    tr = m.get("transitions", {})
    if x["model"] == "whisper":
        comp = _a4_row("whisper", x["dataset"], "Raw", x["side"], x["layer"])
        cond = _a4_row("whisper", x["dataset"], "Conditioning", "decoder", x["layer"]) if x["side"] == "decoder" else None
    else:
        comp = _a4_row("qwen3_asr_1p7b", x["dataset"], "Raw", x["side"], x["layer"])
        cond = _a4_row("qwen3_asr_1p7b", x["dataset"], "Conditioning", "decoder", x["layer"]) if x["side"] == "decoder" else None
    cm = comp.get("metrics", {})
    ccm = cond.get("metrics", {}) if cond else {}
    return {
        "model": x["model"], "dataset": x["dataset"], "side": x["side"], "layer": int(x["layer"]),
        "relative_depth": float(x["layer"] / (31 if x["model"] == "whisper" else (23 if x["side"] == "encoder" else 27))),
        "direction": x["direction"], "scope": x["scope"], "rho": x["rho"],
        "rank_stability_status": x.get("rank_stability_status", x.get("provenance", {}).get("rank_stability_status", "NOT_TESTED")),
        "mer": m.get("mer"), "pier": m.get("pier"), "en_wer": m.get("en_wer"), "matrix_cer": m.get("matrix_cer"),
        "matrix_retention": _retention_rate(m), "poi_corrections": m.get("poi_corrections"),
        "poi_corruptions": m.get("poi_corruptions"), "poi_net_utility": m.get("poi_net_utility"),
        "corrections": m.get("corrections", m.get("poi_corrections")),
        "corruptions": m.get("corruptions", m.get("poi_corruptions")),
        "outside_harm": m.get("outside_harm"), "edited_positions_or_frames": m.get("edited_positions_or_frames"),
        "total_intervention_energy": m.get("total_intervention_energy"),
        "original_hidden_norm": m.get("original_hidden_norm"), "mean_perturbation_norm": m.get("mean_perturbation_norm"),
        "relative_perturbation": m.get("relative_perturbation"), "edited_fraction": m.get("edited_fraction"),
        "raw_local_mer": cm.get("mer"), "raw_local_pier": cm.get("pier"),
        "conditioning_local_mer": ccm.get("mer") if cond else None,
        "conditioning_local_pier": ccm.get("pier") if cond else None,
        "a5_minus_raw_local_mer": (m.get("mer") - cm.get("mer")) if m.get("mer") is not None and cm.get("mer") is not None else None,
        "a5_minus_conditioning_local_mer": (m.get("mer") - ccm.get("mer")) if cond and m.get("mer") is not None and ccm.get("mer") is not None else None,
        "direction_hash": x.get("direction_hash"), "mask_hash": x.get("mask_hash"), "site_hash": x.get("site_hash"),
        "model_revision": x.get("model_revision"), "config_hash": x.get("config_hash"), "_path": x["_path"],
    }


def geometry_and_tables() -> tuple[list[dict], dict]:
    import csv
    rows = _accepted_rows()
    table = [_row_to_table(x) for x in rows]
    _write(OUT / "tables/a5_atlas.json", {"schema_version": "basis_a5_table_v1", "rows": table})
    for name, subset in (("basis_a5_atlas.csv", table),
                         ("whisper_a5.csv", [x for x in table if x["model"] == "whisper"]),
                         ("qwen_a5.csv", [x for x in table if x["model"] == "qwen3_asr_1p7b"])):
        if subset:
            with (OUT / "tables" / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(subset[0].keys()), extrasaction="ignore")
                writer.writeheader(); writer.writerows(subset)
    geometry = []
    direction_geometry = []
    for model, sides in (("whisper", (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC))),
                         ("qwen3_asr_1p7b", (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)))):
        for side, layers in sides:
            for layer in layers:
                for direction in DIRECTIONS:
                    a5, dhash, _ = _a5_direction(model, side, layer, direction)
                    raw = np.asarray(np.load(_a4_direction_path(model, side, layer, "raw")), dtype=np.float64)
                    raw /= np.linalg.norm(raw)
                    vals = [("Raw", raw)]
                    if side == "decoder":
                        cond = np.asarray(np.load(_a4_direction_path(model, side, layer, "conditioning")), dtype=np.float64)
                        cond /= np.linalg.norm(cond); vals.append(("Conditioning", cond))
                    for comparator, vec in vals:
                        cos = float(np.dot(a5, vec) / max(np.linalg.norm(a5) * np.linalg.norm(vec), 1e-12))
                        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                        geometry.append({"model": model, "side": side, "layer": int(layer), "direction": direction,
                                         "comparator": comparator, "cosine": cos, "angle_deg": angle,
                                         "raw_l2": float(np.linalg.norm(a5 - vec)),
                                         "unit_l2_identity": float(np.linalg.norm(a5 - vec)),
                                         "a5_norm": float(np.linalg.norm(a5)), "comparator_norm": float(np.linalg.norm(vec)),
                                         "direction_hash": dhash})
                diag = _json(OUT / "directions" / model / side / f"L{layer:02d}/diagnostics.json")
                direction_geometry.append({
                    "model": model, "side": side, "layer": int(layer),
                    "cos_unique_shared": float(diag["cos_unique_shared"]),
                    "cos_unique_raw": diag.get("cos_unique_raw_a4"),
                    "cos_shared_raw": diag.get("cos_shared_raw_a4"),
                    "cos_unique_conditioning": diag.get("cos_unique_conditioning_a4"),
                    "cos_shared_conditioning": diag.get("cos_shared_conditioning_a4"),
                    "i_unique": int(diag["i_unique"]), "i_shared": int(diag["i_shared"]),
                    "sigma_unique": float(diag["sigma_unique"]), "sigma_shared": float(diag["sigma_shared"]),
                    "energy_A_unique": float(diag["projection_energy_A_unique"]),
                    "energy_B_unique": float(diag["projection_energy_B_unique"]),
                    "energy_A_shared": float(diag["projection_energy_A_shared"]),
                    "energy_B_shared": float(diag["projection_energy_B_shared"]),
                    "count_a": int(diag["count_a"]), "count_b": int(diag["count_b"]),
                    "direction_hash": _a5_direction(model, side, layer, "v_unique")[1],
                })
    _write(OUT / "geometry/a5_vs_a4.json", {"schema_version": "basis_a5_geometry_v1", "rows": geometry})
    with (OUT / "geometry/a5_vs_a4.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(geometry[0].keys())); writer.writeheader(); writer.writerows(geometry)
    _write(OUT / "geometry/direction_geometry.json", {"schema_version": "basis_a5_direction_geometry_v1", "rows": direction_geometry})
    with (OUT / "geometry/direction_geometry.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(direction_geometry[0].keys())); writer.writeheader(); writer.writerows(direction_geometry)
    return table, {"rows": geometry, "direction_rows": direction_geometry}


def _make_figures(table: list[dict], geometry: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    frame = pd.DataFrame(table); made = []
    def line(metric: str, filename: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        for (model, direction, side), g in frame.groupby(["model", "direction", "side"]):
            z = g.groupby("layer", as_index=False)[metric].mean().dropna()
            if len(z): ax.plot(z.layer, z[metric], marker=".", label=f"{model}:{side}:{direction}")
        ax.set_xlabel("layer"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); path = OUT / "figures" / filename; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    line("mer", "mer_vs_depth.png", "MER")
    line("pier", "pier_vs_depth.png", "PIER")
    line("matrix_retention", "matrix_retention_vs_depth.png", "matrix retention")
    line("poi_net_utility", "poi_net_utility_vs_depth.png", "POI net utility")
    for metric, fname, xlabel in (("a5_minus_raw_local_mer", "a5_vs_raw_local.png", "A5 MER − Raw-local MER"),
                                  ("a5_minus_conditioning_local_mer", "a5_vs_conditioning_local.png", "A5 MER − Conditioning-local MER")):
        fig, ax = plt.subplots(figsize=(10, 6))
        z = frame.dropna(subset=[metric])
        for (model, direction), g in z.groupby(["model", "direction"]):
            ax.plot(g.groupby("layer")[metric].mean(), marker=".", label=f"{model}:{direction}")
        ax.axhline(0, color="black", lw=.8); ax.set_xlabel("layer"); ax.set_ylabel(xlabel); ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.tight_layout(); path = OUT / "figures" / fname; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    z = frame.dropna(subset=["poi_corrections", "poi_corruptions"])
    fig, ax = plt.subplots(figsize=(8, 6));
    for direction, g in z.groupby("direction"): ax.scatter(g.poi_corruptions, g.poi_corrections, s=10, label=direction, alpha=.5)
    ax.set_xlabel("POI corruptions"); ax.set_ylabel("POI corrections"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); path = OUT / "figures/correction_damage_frontier.png"; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    gd = pd.DataFrame(geometry["rows"])
    fig, ax = plt.subplots(figsize=(10, 6));
    for (model, side, comparator), g in gd.groupby(["model", "side", "comparator"]):
        z = g.groupby("layer").cosine.mean(); ax.plot(z.index, z, marker=".", label=f"{model}:{side}:{comparator}")
    ax.set_xlabel("layer"); ax.set_ylabel("cosine"); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); path = OUT / "figures/unique_shared_raw_cosine_vs_depth.png"; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    dmeta = []
    for model, sides in (("whisper", (("encoder", WHISPER_ENC), ("decoder", WHISPER_DEC))), ("qwen3_asr_1p7b", (("encoder", QWEN_ENC), ("decoder", QWEN_DEC)))):
        for side, layers in sides:
            for layer in layers:
                d = _json(OUT / "directions" / model / side / f"L{layer:02d}/diagnostics.json")
                dmeta.append({"model": model, "side": side, "layer": layer, "i_unique": d["i_unique"], "i_shared": d["i_shared"],
                              "A_unique": d["projection_energy_A_unique"], "B_unique": d["projection_energy_B_unique"],
                              "A_shared": d["projection_energy_A_shared"], "B_shared": d["projection_energy_B_shared"]})
    dm = pd.DataFrame(dmeta)
    for cols, fname, ylabel in ((["i_unique", "i_shared"], "principal_mode_indices_vs_depth.png", "selected principal index"),
                                (["A_unique", "B_unique", "A_shared", "B_shared"], "projection_energy_diagnostics.png", "projection energy")):
        fig, ax = plt.subplots(figsize=(10, 6))
        for (model, side), g in dm.groupby(["model", "side"]):
            for c in cols: ax.plot(g.layer, g[c], marker=".", label=f"{model}:{side}:{c}")
        ax.set_xlabel("layer"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); path = OUT / "figures" / fname; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, direction, side), g in frame.groupby(["model", "direction", "side"]):
        z = g.groupby("relative_depth").mer.mean(); ax.plot(z.index, z, label=f"{model}:{side}:{direction}")
    ax.set_xlabel("relative depth = layer/(N−1)"); ax.set_ylabel("MER"); ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); path = OUT / "figures/whisper_qwen_relative_depth.png"; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for status, g in frame.groupby("rank_stability_status"):
        z = g.groupby("relative_depth").mer.mean().dropna()
        if len(z):
            ax.plot(z.index, z, marker=".", label=status)
    ax.set_xlabel("relative depth = layer/(N−1)"); ax.set_ylabel("MER")
    ax.set_title("Stable vs unstable vs untested layers (diagnostic labels)")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); path = OUT / "figures/stable_vs_unstable_overlay.png"; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    dg = pd.DataFrame(geometry.get("direction_rows", []))
    if len(dg):
        fig, ax = plt.subplots(figsize=(10, 6))
        for col, label in (("cos_unique_shared", "Unique–Shared"),
                           ("cos_unique_raw", "Unique–Raw"),
                           ("cos_shared_raw", "Shared–Raw")):
            for (model, side), g in dg.groupby(["model", "side"]):
                z = g.sort_values("layer")
                ax.plot(z.layer, z[col], marker=".", label=f"{model}:{side}:{label}")
        ax.axhline(0, color="black", lw=.6); ax.set_xlabel("layer"); ax.set_ylabel("cosine")
        ax.set_title("Unique/Shared/Raw cosine geometry")
        ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); path = OUT / "figures/unique_shared_raw_cosine_vs_depth.png"; fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path.relative_to(REPO)))
    return made


def finalize() -> int:
    import collections
    rows = _accepted_rows()
    expected = 576 + 468
    keys = [tuple(x.get(k) for k in ("model", "dataset", "side", "layer", "direction", "scope", "rho")) for x in rows]
    duplicates = len(keys) - len(set(keys))
    provenance_missing = [x["_path"] for x in rows if not x.get("provenance") or any(not x["provenance"].get(k) for k in ("git_commit", "model_revision", "config_hash", "panel_fingerprint", "direction_hash", "mask_hash", "site_hash", "rank", "rank_stability_status", "rho"))]
    poi_fail = []
    nondeg = []
    for x in rows:
        m = x["metrics"]; tr = m.get("transitions", {})
        if m.get("poi_net_utility") != int(m.get("poi_corrections", 0)) - int(m.get("poi_corruptions", 0)):
            poi_fail.append(x["_path"])
        if int(m.get("edited_positions_or_frames", 0)) <= 0:
            nondeg.append(x["_path"])
    if len(rows) != expected or duplicates or provenance_missing or poi_fail or nondeg:
        raise RuntimeError({"rows": len(rows), "expected": expected, "duplicates": duplicates,
                            "missing_provenance": len(provenance_missing), "poi_fail": len(poi_fail), "nondegenerate_fail": len(nondeg)})
    table, geometry = geometry_and_tables()
    figures = _make_figures(table, geometry)
    by = collections.Counter((x["model"], x["direction"]) for x in rows)
    means = {}
    for model, direction in sorted(by):
        subset = [x for x in table if x["model"] == model and x["direction"] == direction]
        means[f"{model}:{direction}"] = {"mer": float(np.mean([x["mer"] for x in subset])),
                                          "pier": float(np.mean([x["pier"] for x in subset])),
                                          "poi_net_utility": float(np.mean([x["poi_net_utility"] for x in subset])),
                                          "matrix_retention": float(np.nanmean([x["matrix_retention"] for x in subset]))}
    def mean(xs, key):
        vals = [x.get(key) for x in xs if x.get(key) is not None and np.isfinite(x.get(key))]
        return float(np.mean(vals)) if vals else None

    analysis = {"method_names": {"neg_shared": "Minus-Shared", "pos_unique": "Add-Unique",
                                  "unique_minus_shared": "Unique-minus-Shared"}, "by_group": {},
                "stability": {}, "cross_corpus": {}, "cross_model": {}}
    for group in ([(m, s, d) for m in ("whisper", "qwen3_asr_1p7b")
                   for s in ("encoder", "decoder") for d in DIRECTIONS]):
        model, side, direction = group
        subset = [x for x in table if x["model"] == model and x["side"] == side and x["direction"] == direction]
        analysis["by_group"]["%s:%s:%s" % group] = {k: mean(subset, k) for k in
            ("mer", "pier", "en_wer", "matrix_cer", "matrix_retention", "poi_corrections",
             "poi_corruptions", "poi_net_utility", "a5_minus_raw_local_mer",
             "a5_minus_conditioning_local_mer", "total_intervention_energy")}
    for status in ("STABLE", "UNSTABLE", "NOT_TESTED"):
        subset = [x for x in table if x["rank_stability_status"] == status]
        analysis["stability"][status] = {"n": len(subset), "mer": mean(subset, "mer"),
                                          "pier": mean(subset, "pier"),
                                          "matrix_retention": mean(subset, "matrix_retention"),
                                          "poi_net_utility": mean(subset, "poi_net_utility"),
                                          "poi_corrections": mean(subset, "poi_corrections"),
                                          "poi_corruptions": mean(subset, "poi_corruptions"),
                                          "mer_std": (float(np.std([x["mer"] for x in subset]))
                                                      if subset else None)}
    for dataset in DATASETS:
        for direction in DIRECTIONS:
            subset = [x for x in table if x["dataset"] == dataset and x["direction"] == direction]
            analysis["cross_corpus"][f"{dataset}:{direction}"] = {"n": len(subset), "mer": mean(subset, "mer"),
                "pier": mean(subset, "pier"), "poi_net_utility": mean(subset, "poi_net_utility")}
    for direction in DIRECTIONS:
        for side in ("encoder", "decoder"):
            w = [x for x in table if x["model"] == "whisper" and x["side"] == side and x["direction"] == direction]
            q = [x for x in table if x["model"] == "qwen3_asr_1p7b" and x["side"] == side and x["direction"] == direction]
            analysis["cross_model"][f"{side}:{direction}"] = {"whisper_mer": mean(w, "mer"), "qwen_mer": mean(q, "mer"),
                "whisper_pier": mean(w, "pier"), "qwen_pier": mean(q, "pier")}
    def direction_delta(a, b, side=None):
        aa = [x for x in table if x["direction"] == a and (side is None or x["side"] == side)]
        bb = [x for x in table if x["direction"] == b and (side is None or x["side"] == side)]
        return mean(aa, "mer"), mean(bb, "mer"), (mean(aa, "mer") - mean(bb, "mer") if mean(aa, "mer") is not None and mean(bb, "mer") is not None else None)
    au, raw, au_raw = direction_delta("pos_unique", "neg_shared")
    us, _, us_raw = direction_delta("unique_minus_shared", "neg_shared")
    analysis["questions"] = {
        "Q1_plus_unique": {"mean_mer": au, "reference_minus_shared_mer": raw, "descriptive_delta": au_raw},
        "Q2_minus_shared": "Inspect its MER/PIER and matrix retention; a lower MER with lower retention is not a clean benefit.",
        "Q3_us_vs_components": {"us_mean_mer": us, "minus_shared_mean_mer": raw, "descriptive_delta": us_raw},
        "Q4_plus_unique_vs_raw": "Use a5_minus_raw_local_mer for pos_unique; negative is lower MER.",
        "Q5_us_vs_raw": "Use a5_minus_raw_local_mer for unique_minus_shared; negative is lower MER.",
        "Q6_encoder_vs_decoder": "Compare by_group entries with the same method; no side is selected by protocol.",
        "Q7_cross_model": "Compare relative-depth figure and cross_model table; this is descriptive, not a transfer claim.",
        "Q8_cross_corpus": "Compare cross_corpus entries; sign consistency is required before describing transfer.",
        "Q9_rank_instability": "Compare stability entries, including n and mer_std; one favorable unstable layer is insufficient.",
        "Q10_unique_beyond_raw": "Compare pos_unique versus Raw-local and unique_minus_shared versus Raw-local across layers/corpora.",
    }
    report = f"""# BASIS-A5 Unique–Shared Local Steering Atlas

## Status

COMPLETE for the exploratory atlas. The preserved construction-only rank gate remains a recorded `FAIL` and is not relabeled as a pass. Under the explicit amendment, rank stability is diagnostic and nonblocking.

## Direction construction

Directions were constructed from uncentered D-construct moments at the exact A4 sites. Signs were fixed from construction means only. S1 is Minus-Shared, S2 is Add-Unique, and S3 is the Unique-minus-Shared composite; S3 is not a difference-in-means direction.

## Scientific interpretation

The following aggregate metrics are descriptive and do not force a positive conclusion:

```json
{json.dumps(means, indent=2, sort_keys=True)}
```

The accepted table supports direct answers to the A5 questions through the per-cell MER, PIER, retention, correction, corruption, energy, and comparator columns. Any high corrective power paired with damage is reported as such. A5 is Oracle-local and therefore an upper-bound mechanism test, not a deployable localizer.

## Stable versus unstable analysis

The following summaries are descriptive. `UNSTABLE` means the preserved rank diagnostic was below its stability threshold; `NOT_TESTED` means the old anchor diagnostic did not cover that layer. No result was removed because of this label, and no result was used to choose rank.

```json
{json.dumps(analysis, indent=2, sort_keys=True)}
```

Whisper/Qwen comparisons use relative depth only. Hidden coordinates are not compared across models, and rho=.5 is not treated as energy-matched.

## Completeness

Whisper: 576/576. Qwen: 468/468. Total: 1044/1044. Duplicate keys: 0. Empty provenance: 0. Conditioning-Avg: not part of A5.

## Figures and tables

Required figures are under `figures/`; accepted rows are in `tables/basis_a5_atlas.csv`; within-model direction geometry is in `geometry/a5_vs_a4.csv`.

## Data exposure

Vectors use D-construct only. Evaluation uses the frozen A4 D-dev-select, dev-man, and dev-sge panels. D-dev-confirm and D-test were not read or intervened.

## Gate

READY_FOR_INDEPENDENT_AUDIT
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    final_manifest = {"schema_version": "basis_a5_final_manifest_v1", "status": "COMPLETE",
                      "protocol_hash": _json(OUT / "manifests/a5_protocol_freeze.json")["protocol_hash"],
                      "git": _git_state(), "counts": {"whisper": 576, "qwen": 468, "total": len(rows)},
                      "directions": list(DIRECTIONS), "scope": "oracle_local", "rho": RHO,
                      "duplicate_keys": duplicates, "empty_provenance": len(provenance_missing),
                      "metric_arithmetic_failures": len(poi_fail), "nondegenerate_edit_failures": len(nondeg),
                      "rank_stability": "results/basis_a5_unique_shared/manifests/rank_stability.json",
                      "direction_audit": "results/basis_a5_unique_shared/manifests/direction_audit.json",
                      "geometry_rows": len(geometry["rows"]), "direction_geometry_rows": len(geometry["direction_rows"]), "figures": figures,
                      "stability_counts": analysis["stability"],
                      "protocol_amendment": str(A5_AMENDMENT.relative_to(REPO)),
                      "rank_stability_policy": "diagnostic_nonblocking",
                      "a4_comparators_redecoded": False}
    _write(OUT / "FINAL_MANIFEST.json", final_manifest)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("construction-test")
    sub.add_parser("refresh-directions")
    sub.add_parser("direction-audit")
    sub.add_parser("rank-stability")
    sub.add_parser("rank-stability-diagnostic")
    sub.add_parser("pre-gpu-gates")
    c = sub.add_parser("construct"); c.add_argument("--model", choices=("whisper", "qwen3_asr_1p7b"), required=True)
    p = sub.add_parser("preflight"); p.add_argument("--model", choices=("whisper", "qwen3_asr_1p7b"), required=True)
    a = sub.add_parser("atlas")
    a.add_argument("--model", choices=("whisper", "qwen3_asr_1p7b"), required=True)
    a.add_argument("--dataset", choices=DATASETS, required=True)
    a.add_argument("--side", choices=("encoder", "decoder", "both"), required=True)
    a.add_argument("--direction", choices=(*DIRECTIONS, "all"), required=True)
    a.add_argument("--layer-start", type=int, default=0); a.add_argument("--layer-end", type=int)
    sub.add_parser("finalize")
    args = ap.parse_args(argv)
    if args.command == "freeze": return freeze()
    if args.command == "construction-test": return construction_test()
    if args.command == "refresh-directions": return refresh_direction_manifests()
    if args.command == "construct": return construct_whisper() if args.model == "whisper" else construct_qwen()
    if args.command == "direction-audit": return direction_audit()
    if args.command == "rank-stability": return rank_stability()
    if args.command == "rank-stability-diagnostic": return rank_stability_diagnostic()
    if args.command == "pre-gpu-gates": return pre_gpu_gates()
    if args.command == "preflight": return preflight(args.model)
    if args.command == "atlas":
        if args.model == "whisper":
            return atlas_whisper(args.dataset, args.side, args.direction, args.layer_start, args.layer_end)
        return atlas_qwen(args.dataset, args.side, args.direction, args.layer_start, args.layer_end)
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
