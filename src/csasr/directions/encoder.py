"""Encoder language-direction construction (guide sections 20, 21, 23, 25).

d_l = normalize( mean_i [ mu_EN(i,l) - mu_ZH(i,l) ] ) over training utterances,
with every utterance weighted equally and boundary-adjacent frames excluded.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..data.alignment import core_frames
from ..data.language_tags import EN, ZH
from ..models.hooks import ActivationRecorder, assert_no_hooks
from ..models.whisper import batch_features
from ..utils.logging import get_logger
from .accumulators import DirectionAccumulatorSet
from .controls import random_direction, wrong_sign

log = get_logger(__name__)

DIRECTION_FIELDS = [
    "model_id", "model_revision", "module", "layer_index", "hook_location",
    "direction_type", "direction", "direction_norm", "projection_std",
    "centroid_en_projection", "centroid_zh_projection", "midpoint_projection",
    "num_utterances", "num_en_frames_or_tokens", "num_zh_frames_or_tokens",
    "seed", "manifest_hash", "normalization_version",
]

VARIANTS = (
    "within_utterance_correct_only",
    "within_utterance_all_valid",
    "global_centroid_correct_only",
)


def select_frames(units: pd.DataFrame, language: str, num_valid: int,
                  exclude_frames: int, correct_only: bool) -> np.ndarray:
    """Frame indices for one language in one utterance, after all filters."""
    sub = units[(units["language_tag"] == language) & units["is_content"]]
    if correct_only:
        sub = sub[sub["baseline_status"] == "correct"]
    keep = np.zeros(num_valid, dtype=bool)
    for _, r in sub.iterrows():
        if bool(r["is_boundary_adjacent"]):
            s, e = core_frames(int(r["start_frame"]), int(r["end_frame"]), exclude_frames)
        else:
            s, e = int(r["start_frame"]), int(r["end_frame"])
        if e > s:
            keep[max(0, s): min(num_valid, e)] = True
    return np.flatnonzero(keep)


@torch.inference_mode()
def accumulate_directions(bundle, manifest: pd.DataFrame, units: pd.DataFrame,
                          layers: Iterable[int], *, exclude_frames: int,
                          correct_only: bool, min_en_frames: int, min_zh_frames: int,
                          batch_size: int = 8, checkpoint_path: str | Path | None = None,
                          checkpoint_every: int = 200,
                          progress_desc: str = "e2 accumulate") -> DirectionAccumulatorSet:
    """One encoder forward pass per batch; hidden states are discarded at once."""
    layers = list(layers)
    accs = (DirectionAccumulatorSet.load_or_new(checkpoint_path, layers, bundle.d_model)
            if checkpoint_path else DirectionAccumulatorSet(layers, bundle.d_model))

    by_utt = {u: g for u, g in units.groupby("utterance_id")}
    todo = manifest[~manifest["utterance_id"].isin(accs.processed)].reset_index(drop=True)
    todo = todo.sort_values("duration_sec").reset_index(drop=True)
    n_batches = math.ceil(len(todo) / batch_size) if len(todo) else 0
    skipped = {"no_units": 0, "too_few_frames": 0}
    since_ckpt = 0

    for bi in tqdm(range(n_batches), desc=progress_desc, leave=False):
        batch = todo.iloc[bi * batch_size: (bi + 1) * batch_size]
        # decide frame selections before running the model, so useless audio is skipped
        plans = []
        for _, row in batch.iterrows():
            utt = row["utterance_id"]
            u = by_utt.get(utt)
            if u is None or not len(u):
                skipped["no_units"] += 1
                accs.mark(utt)
                continue
            n_valid = bundle.valid_frames(row["duration_sec"])
            en_idx = select_frames(u, EN, n_valid, exclude_frames, correct_only)
            zh_idx = select_frames(u, ZH, n_valid, exclude_frames, correct_only)
            if len(en_idx) < min_en_frames or len(zh_idx) < min_zh_frames:
                skipped["too_few_frames"] += 1
                accs.mark(utt)
                continue
            plans.append((row, en_idx, zh_idx))
        if not plans:
            continue

        paths = [p[0]["audio_path"] for p in plans]
        features = batch_features(bundle, paths)
        with ActivationRecorder(bundle, layers, module="encoder") as rec:
            bundle.model.model.encoder(features)
            states = {l: rec.states[l] for l in layers}
        assert_no_hooks(bundle)

        for l in layers:
            h = states[l]                               # (B, T, D) float32 on device
            acc = accs.acc[l]
            for bi2, (row, en_idx, zh_idx) in enumerate(plans):
                en_t = h[bi2, torch.as_tensor(en_idx, device=h.device)]
                zh_t = h[bi2, torch.as_tensor(zh_idx, device=h.device)]
                both = torch.cat([en_t, zh_t], dim=0)
                gram = (both.T @ both).double().cpu().numpy()
                acc.add_utterance(en_t.double().cpu().numpy(), zh_t.double().cpu().numpy())
                acc.add_frames(both.double().cpu().numpy(), gram=gram)
        del states, features
        for row, _, _ in plans:
            accs.mark(row["utterance_id"])

        since_ckpt += len(plans)
        if checkpoint_path and since_ckpt >= checkpoint_every:
            accs.save(checkpoint_path)
            since_ckpt = 0

    if checkpoint_path:
        accs.save(checkpoint_path)
    log.info("accumulation done: %s | skipped=%s", accs.summary(), skipped)
    accs.skipped = skipped  # type: ignore[attr-defined]
    return accs


def build_direction_record(bundle, layer: int, vector: np.ndarray, acc, *,
                           direction_type: str, seed: int, manifest_hash: str,
                           normalization_version: str, module: str = "encoder") -> dict:
    """Package one direction with all statistics required by section 25."""
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"degenerate direction for layer {layer} ({direction_type})")
    unit = v / norm
    proj_std = acc.projection_std(unit) if acc is not None else float("nan")
    en_proj, zh_proj = acc.centroid_projections(unit) if acc is not None else (float("nan"),) * 2
    hook_loc = (bundle.module_report["encoder_hook_location"] if module == "encoder"
                else bundle.module_report["decoder_hook_location"])
    return {
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "module": module,
        "layer_index": int(layer),
        "hook_location": hook_loc,
        "direction_type": direction_type,
        "direction": torch.from_numpy(unit.astype(np.float32)),
        "direction_norm": 1.0,
        "raw_norm": norm,
        "projection_std": proj_std,
        "centroid_en_projection": en_proj,
        "centroid_zh_projection": zh_proj,
        "midpoint_projection": 0.5 * (en_proj + zh_proj),
        "num_utterances": int(getattr(acc, "num_utterances", 0)),
        "num_en_frames_or_tokens": int(getattr(acc, "en_count", 0)),
        "num_zh_frames_or_tokens": int(getattr(acc, "zh_count", 0)),
        "seed": int(seed),
        "manifest_hash": manifest_hash,
        "normalization_version": normalization_version,
    }


def directions_from_accumulator(bundle, accs: DirectionAccumulatorSet, *,
                                variant: str, seed: int, manifest_hash: str,
                                normalization_version: str) -> dict[int, dict]:
    """Build one direction per layer for a given variant."""
    out: dict[int, dict] = {}
    for layer, acc in accs.acc.items():
        if variant == "global_centroid_correct_only":
            vec = acc.global_delta()
        else:
            vec = acc.mean_delta()
        out[layer] = build_direction_record(
            bundle, layer, vec, acc, direction_type=variant, seed=seed,
            manifest_hash=manifest_hash, normalization_version=normalization_version,
        )
    return out


def control_directions(bundle, primary: Mapping[int, dict], random_seeds: Iterable[int],
                       manifest_hash: str, normalization_version: str) -> dict[str, dict[int, dict]]:
    """Wrong-sign and five matched random controls, sharing the primary's s_l."""
    out: dict[str, dict[int, dict]] = {"wrong_sign": {}}
    for layer, rec in primary.items():
        w = dict(rec)
        w["direction"] = wrong_sign(rec["direction"])
        w["direction_type"] = "wrong_sign"
        w["centroid_en_projection"] = -rec["centroid_en_projection"]
        w["centroid_zh_projection"] = -rec["centroid_zh_projection"]
        w["midpoint_projection"] = -rec["midpoint_projection"]
        out["wrong_sign"][layer] = w
    for s in random_seeds:
        key = f"random_s{s}"
        out[key] = {}
        for layer, rec in primary.items():
            r = dict(rec)
            r["direction"] = random_direction(rec["direction"].numel(), seed=s + layer)
            r["direction_type"] = key
            r["control_seed"] = int(s)
            # matched intervention scale: reuse the primary s_l exactly (section 23)
            r["projection_std"] = rec["projection_std"]
            r["centroid_en_projection"] = float("nan")
            r["centroid_zh_projection"] = float("nan")
            r["midpoint_projection"] = float("nan")
            out[key][layer] = r
    return out


def save_direction(record: dict, root: str | Path) -> Path:
    module = record["module"]
    layer = record["layer_index"]
    path = Path(root) / module / f"layer_{layer}" / f"{record['direction_type']}_seed{record['seed']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    return path


def load_direction(root: str | Path, module: str, layer: int, direction_type: str,
                   seed: int) -> dict:
    path = Path(root) / module / f"layer_{layer}" / f"{direction_type}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"direction artifact missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_direction(record: dict, tol: float = 1e-4) -> None:
    d = record["direction"]
    if not torch.isfinite(d).all():
        raise ValueError(f"non-finite direction: {record['direction_type']} L{record['layer_index']}")
    n = float(d.norm())
    if abs(n - 1.0) > tol:
        raise ValueError(f"direction not unit norm ({n:.6f}): {record['direction_type']}")
