"""Day 3: Site-E and Site-D direction construction on the v2r3 population.

Construction uses `D-construct` only, on baseline-**correct** embedded-English
units at the >=400 ms tier.  The conservative evaluation subset is built from
`D-dev-select` and is reported but never used to fit anything.

Decided inputs, not re-derived here: the provisional aligner path is
`existing_ctc` under the `blank_to_preceding` blank convention.  The
cross-attention path was built, measured, and excluded as a span source
(`docs/V2R3_CROSS_ATTENTION_SPANS_2026-08-15.md`); its artifacts are a
negative-result record and are never read for span derivation.

Every estimator parameter below is frozen before the run and recorded in the
manifest.  This module selects no layer and no strength: it emits one direction
per site per layer per pooling variant, with the diagnostics needed to choose,
and the choice is not made here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.language_tags import EN, ZH
from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..lss.sites import DecoderPostCrossAttnRecorder, assert_no_site_hooks
from ..models.hooks import ActivationRecorder
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging
from .v2r3_headroom_diagnostic import eligible_spans, structurally_eligible

STAGE = "v2r3_directions"

# ---------------------------------------------------------------------------
# FROZEN ESTIMATOR CONFIGURATION
# ---------------------------------------------------------------------------

#: The decided span path.  Not re-derived; see the module docstring.
SPAN_FAMILY = "existing_ctc"
CTC_CONVENTION = "blank_to_preceding"

#: Conservative tier.  The expansion tiers are reported by Session 8 and are not
#: used for construction.
FLOOR_MS = 400.0

#: Pooling variants.  `hann` is the primary; the other two are retained for the
#: Day 6 sensitivity battery.  All normalise to sum 1 and are applied
#: identically to the positive span and its matched control.
POOLING_VARIANTS = ("hann", "central60", "uniform")
PRIMARY_POOLING = "hann"

#: Control frames: the matrix-language frames immediately preceding the span,
#: matched one-for-one in frame count.  Matrix-preceded is an eligibility
#: condition, so a preceding matrix run always exists.
MIN_CONTROL_FRAMES = 3
MIN_SPAN_FRAMES = 3

#: Nuisance vector, standardised before the ridge.  These are span-level
#: quantities a direction must not be explained by.
NUISANCE_FEATURES = ("span_duration_sec", "n_frames", "relative_start",
                     "utterance_duration_sec", "n_units")
RIDGE_ALPHA = 1.0

#: Norm clipping before aggregation.  Individual contrasts are NOT normalised --
#: doing so would discard the magnitude information the weighted mean uses --
#: but a single outlying pair must not dominate a 20-dialogue mean.
NORM_CLIP_QUANTILE = 0.99

#: Prompt subspace rank.  r=1 keeps only Orth[Mean(p_j)].
PROMPT_SUBSPACE_RANK = 1

#: Dialogue bootstrap for cosine stability.  G is the dialogue count, always
#: reported alongside.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260815

FROZEN_CONFIG: dict[str, Any] = {
    "span_family": SPAN_FAMILY,
    "ctc_convention": CTC_CONVENTION,
    "floor_ms": FLOOR_MS,
    "pooling_variants": list(POOLING_VARIANTS),
    "primary_pooling": PRIMARY_POOLING,
    "pooling_normalisation": "weights sum to 1; identical for span and control",
    "control_definition": ("matrix-language frames immediately preceding the "
                           "span, matched one-for-one in frame count"),
    "min_span_frames": MIN_SPAN_FRAMES,
    "min_control_frames": MIN_CONTROL_FRAMES,
    "nuisance_features": list(NUISANCE_FEATURES),
    "ridge_alpha": RIDGE_ALPHA,
    "ridge_fit_split": "D-construct only",
    "ridge_intercept": "retained (residual keeps the mean; only nuisance-predicted variation is removed)",
    "individual_contrasts_normalised": False,
    "norm_clip_quantile": NORM_CLIP_QUANTILE,
    "aggregation": "dialogue-balanced weighted mean, weight 1/(G * n_g)",
    "prompt_subspace_rank": PROMPT_SUBSPACE_RANK,
    "prompt_subspace": "Orth[Mean(p_j), PC_1..r-1(p_j)]; projected out of EACH contrast before aggregation",
    "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                  "unit": "dialogue_id"},
    "selects_layer_or_strength": False,
}


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------

def pool_weights(n_frames: int, kind: str = PRIMARY_POOLING) -> np.ndarray:
    """Pooling weights over `n_frames`, normalised to sum 1."""
    if n_frames <= 0:
        raise ValueError("pooling needs at least one frame")
    if kind == "uniform":
        w = np.ones(n_frames, dtype=float)
    elif kind == "hann":
        # periodic-free symmetric Hann; a single frame degenerates to uniform
        w = np.hanning(n_frames + 2)[1:-1] if n_frames > 1 else np.ones(1)
        if not np.any(w > 0):
            w = np.ones(n_frames, dtype=float)
    elif kind == "central60":
        w = np.zeros(n_frames, dtype=float)
        lo = int(np.floor(n_frames * 0.2))
        hi = int(np.ceil(n_frames * 0.8))
        hi = max(hi, lo + 1)
        w[lo:hi] = 1.0
    else:
        raise ValueError(f"unknown pooling variant {kind!r}")
    total = float(w.sum())
    if total <= 0:
        raise ValueError(f"pooling weights for {kind!r} sum to {total}")
    return w / total


def pooled(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool of a (frames, dim) block."""
    return np.asarray(weights, dtype=float) @ np.asarray(states, dtype=float)


# ---------------------------------------------------------------------------
# subsets
# ---------------------------------------------------------------------------

def _frame_range(start_sec: float, end_sec: float, step: float,
                 valid: int) -> tuple[int, int]:
    lo = max(0, int(np.floor(float(start_sec) / step)))
    hi = min(valid, int(np.ceil(float(end_sec) / step)))
    return lo, max(hi, lo)


def build_spans(candidates: pd.DataFrame, *,
                dialogue_of: dict[str, str] | None = None) -> pd.DataFrame:
    """Eligible embedded-English spans under the decided convention.

    `eligible_spans` reports the eligibility flags but not the span's own time
    edges or its dialogue, both of which the extraction needs, so they are
    attached here from the same convention-adjusted rows the flags came from.
    """
    adjusted = conv.apply_convention(candidates, CTC_CONVENTION, family=SPAN_FAMILY)
    spans = eligible_spans(adjusted, family=SPAN_FAMILY)
    spans = spans[structurally_eligible(spans)].copy()
    spans = spans[spans["span_duration_ms"] >= FLOOR_MS].reset_index(drop=True)

    frame = adjusted[adjusted["aligner_family"].astype(str) == SPAN_FAMILY]
    edges: dict[tuple[str, int], tuple[float, float]] = {}
    for (utterance, index), group in frame.groupby(["utterance_id", "reference_unit_index"]):
        starts = pd.to_numeric(group["start_sec"], errors="coerce")
        ends = pd.to_numeric(group["end_sec"], errors="coerce")
        edges[(str(utterance), int(index))] = (float(starts.min()), float(ends.max()))
    starts, ends = [], []
    for utterance, indices in zip(spans["utterance_id"], spans["unit_indices"]):
        pairs = [edges[(str(utterance), int(i))] for i in indices
                 if (str(utterance), int(i)) in edges]
        starts.append(min(p[0] for p in pairs) if pairs else float("nan"))
        ends.append(max(p[1] for p in pairs) if pairs else float("nan"))
    spans["start_sec"] = starts
    spans["end_sec"] = ends
    spans["dialogue_id"] = [str((dialogue_of or {}).get(str(c), ""))
                            for c in spans["conversation_id"]]
    return spans[spans["start_sec"].notna() & spans["end_sec"].notna()].reset_index(drop=True)


def matrix_runs(candidates: pd.DataFrame) -> pd.DataFrame:
    """Matrix-language (ZH) runs under the decided convention, per utterance."""
    from ..lss.align import annotation_pack as pack_mod

    adjusted = conv.apply_convention(candidates, CTC_CONVENTION, family=SPAN_FAMILY)
    frame = adjusted[adjusted["aligner_family"].astype(str) == SPAN_FAMILY]
    rows = []
    for utterance, group in frame.groupby("utterance_id", sort=False):
        for position, run in enumerate(pack_mod.language_runs(group)):
            if str(run["language"]) != ZH:
                continue
            units = pd.DataFrame(run["rows"])
            starts = pd.to_numeric(units["start_sec"], errors="coerce")
            ends = pd.to_numeric(units["end_sec"], errors="coerce")
            rows.append({"utterance_id": str(utterance), "run_position": position,
                         "start_sec": float(starts.min()), "end_sec": float(ends.max())})
    return pd.DataFrame(rows)


def attach_baseline_status(spans: pd.DataFrame, poi: pd.DataFrame) -> pd.DataFrame:
    """Per span: how many of its units the baseline got right and wrong."""
    keyed = {(str(u), int(i)): bool(c) for u, i, c
             in zip(poi["utterance_id"], poi["reference_unit_index"], poi["correct"])}
    correct, wrong, categories = [], [], []
    for utterance, indices in zip(spans["utterance_id"], spans["unit_indices"]):
        flags = [keyed.get((str(utterance), int(i))) for i in indices]
        correct.append(sum(1 for f in flags if f is True))
        wrong.append(sum(1 for f in flags if f is False))
        categories.append(sum(1 for f in flags if f is None))
    out = spans.copy()
    out["units_correct"] = correct
    out["units_error"] = wrong
    out["units_unmatched"] = categories
    out["all_correct"] = out["units_correct"] == out["n_units"]
    return out


# ---------------------------------------------------------------------------
# estimator: residualisation and dialogue-balanced aggregation
# ---------------------------------------------------------------------------

def nuisance_matrix(spans: pd.DataFrame) -> np.ndarray:
    """Standardised nuisance design.  Columns are frozen in NUISANCE_FEATURES."""
    columns = []
    for name in NUISANCE_FEATURES:
        values = pd.to_numeric(spans[name], errors="coerce").to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=float(np.nanmean(values)) if len(values) else 0.0)
        sd = values.std()
        columns.append((values - values.mean()) / sd if sd > 0 else values * 0.0)
    return np.column_stack(columns) if columns else np.zeros((len(spans), 0))


def residualise(contrasts: np.ndarray, design: np.ndarray, *,
                alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-output ridge, intercept retained.

    Only the nuisance-predicted variation is removed: the fitted intercept stays
    in the residual, so the mean effect the direction is built from survives.
    """
    y = np.asarray(contrasts, dtype=float)
    x = np.asarray(design, dtype=float)
    if x.shape[1] == 0 or len(y) < 3:
        return y, {"fitted": False, "reason": "no nuisance columns or too few pairs",
                   "energy_fraction_removed": 0.0}
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    gram = xc.T @ xc + float(alpha) * np.eye(xc.shape[1])
    beta = np.linalg.solve(gram, xc.T @ yc)          # (features, dim)
    predicted = xc @ beta                             # nuisance-explained part
    residual = y - predicted                          # intercept retained
    total = float((yc ** 2).sum())
    removed = float((predicted ** 2).sum())
    return residual, {"fitted": True, "alpha": float(alpha),
                      "n_pairs": int(len(y)), "n_features": int(x.shape[1]),
                      "energy_fraction_removed": (removed / total) if total > 0 else 0.0}


def clip_norms(contrasts: np.ndarray, quantile: float = NORM_CLIP_QUANTILE
               ) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip contrast norms at a preregistered quantile.  Directions are kept."""
    y = np.asarray(contrasts, dtype=float)
    norms = np.linalg.norm(y, axis=1)
    if not len(norms):
        return y, {"clipped": 0}
    limit = float(np.quantile(norms, quantile))
    scale = np.ones_like(norms)
    over = norms > limit
    scale[over] = limit / np.maximum(norms[over], 1e-12)
    return y * scale[:, None], {"clipped": int(over.sum()), "limit": limit,
                                "quantile": quantile}


def dialogue_balanced_mean(contrasts: np.ndarray,
                           dialogues: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Weighted mean giving each dialogue equal total weight."""
    y = np.asarray(contrasts, dtype=float)
    labels = np.asarray([str(d) for d in dialogues])
    unique = sorted(set(labels))
    if not unique:
        return np.zeros(y.shape[1] if y.ndim > 1 else 0), {"G": 0, "n": 0}
    weights = np.zeros(len(y), dtype=float)
    for dialogue in unique:
        mask = labels == dialogue
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return weights @ y, {"G": len(unique), "n": int(len(y)),
                         "pairs_per_dialogue_min": int(pd.Series(labels).value_counts().min()),
                         "pairs_per_dialogue_max": int(pd.Series(labels).value_counts().max())}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(a @ b / (na * nb))


def bootstrap_cosine(contrasts: np.ndarray, dialogues: Sequence[str], *,
                     draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
                     ) -> dict[str, Any]:
    """Cosine of each dialogue-resampled direction against the full-sample one."""
    y = np.asarray(contrasts, dtype=float)
    labels = np.asarray([str(d) for d in dialogues])
    unique = sorted(set(labels))
    if len(unique) < 2:
        return {"G": len(unique), "draws": 0}
    reference, _ = dialogue_balanced_mean(y, labels)
    index = {d: np.flatnonzero(labels == d) for d in unique}
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(draws):
        picked = rng.choice(len(unique), size=len(unique), replace=True)
        rows = np.concatenate([index[unique[k]] for k in picked])
        drawn_labels = np.concatenate(
            [np.full(len(index[unique[k]]), f"{unique[k]}#{j}") for j, k in enumerate(picked)])
        direction, _ = dialogue_balanced_mean(y[rows], drawn_labels)
        cosines.append(cosine(direction, reference))
    cosines = np.asarray(cosines, dtype=float)
    return {"G": len(unique), "draws": int(draws), "seed": int(seed),
            "cosine_mean": float(np.nanmean(cosines)),
            "cosine_p05": float(np.nanquantile(cosines, 0.05)),
            "cosine_p50": float(np.nanquantile(cosines, 0.50)),
            "cosine_p95": float(np.nanquantile(cosines, 0.95)),
            "cosine_min": float(np.nanmin(cosines))}


def orthonormal_basis(vectors: np.ndarray, rank: int) -> np.ndarray:
    """Orth[Mean(p), PC_1..rank-1(p)] as an (dim, k) orthonormal matrix."""
    p = np.asarray(vectors, dtype=float)
    columns = [p.mean(axis=0)]
    if rank > 1 and len(p) > 1:
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        columns.extend(vt[: rank - 1])
    basis, _ = np.linalg.qr(np.column_stack(columns))
    return basis


def project_out(vectors: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the span of `basis` from each row; report the energy removed."""
    y = np.asarray(vectors, dtype=float)
    if basis is None or basis.size == 0:
        return y, 0.0
    coefficients = y @ basis
    removed = coefficients @ basis.T
    total = float((y ** 2).sum())
    return y - removed, (float((removed ** 2).sum()) / total if total > 0 else 0.0)


def shrinkage_lda(positive: np.ndarray, negative: np.ndarray, basis: np.ndarray | None,
                  *, shrinkage: float = 0.1) -> np.ndarray:
    """LDA direction, estimated INSIDE the complement of `basis`.

    Shrinking and inverting in the full space would reinstate a small variance
    along the removed directions, and the inverse amplifies exactly that leakage
    by its largest factor. The covariance is therefore formed, shrunk, and
    inverted in an orthonormal basis of the complement, then mapped back.
    """
    p = np.asarray(positive, dtype=float)
    n = np.asarray(negative, dtype=float)
    dim = p.shape[1]
    if basis is not None and basis.size:
        full = np.linalg.qr(np.column_stack([basis, np.random.default_rng(0)
                                             .standard_normal((dim, dim - basis.shape[1]))]))[0]
        complement = full[:, basis.shape[1]:]
    else:
        complement = np.eye(dim)
    pc, nc = p @ complement, n @ complement
    pooled_cov = (np.cov(pc, rowvar=False) * (len(pc) - 1)
                  + np.cov(nc, rowvar=False) * (len(nc) - 1)) / max(len(pc) + len(nc) - 2, 1)
    trace = float(np.trace(pooled_cov)) / pooled_cov.shape[0]
    shrunk = (1 - shrinkage) * pooled_cov + shrinkage * trace * np.eye(pooled_cov.shape[0])
    delta = pc.mean(axis=0) - nc.mean(axis=0)
    weights = np.linalg.solve(shrunk, delta)
    return complement @ weights                      # mapped back to the full space


# ---------------------------------------------------------------------------
# Site E: encoder frame contrasts
# ---------------------------------------------------------------------------

def control_frames(span_lo: int, span_hi: int, runs: pd.DataFrame,
                   step: float, valid: int) -> tuple[int, int] | None:
    """Matrix frames immediately preceding the span, matched in count."""
    want = span_hi - span_lo
    before = runs[runs["end_sec"] <= (span_lo + 1) * step]
    if not len(before):
        return None
    run = before.sort_values("end_sec").iloc[-1]
    lo, hi = _frame_range(run["start_sec"], run["end_sec"], step, valid)
    hi = min(hi, span_lo)
    if hi - lo < MIN_CONTROL_FRAMES:
        return None
    start = max(lo, hi - want)
    return start, hi


@torch.inference_mode()
def site_e_contrasts(bundle, manifest: pd.DataFrame, spans: pd.DataFrame,
                     runs: pd.DataFrame, layers: Sequence[int], *,
                     batch_size: int = 4, log=None) -> tuple[pd.DataFrame, dict]:
    """One paired contrast per span per layer per pooling variant."""
    from ..models.generation import batch_model_inputs

    step = float(bundle.encoder_step_sec)
    by_utterance = {u: g for u, g in spans.groupby("utterance_id")}
    runs_by_utterance = {u: g for u, g in runs.groupby("utterance_id")}
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[int, str], list[np.ndarray]] = {}
    skipped = {"no_control": 0, "too_short": 0}

    wanted = manifest[manifest["utterance_id"].isin(by_utterance)].reset_index(drop=True)
    for start in range(0, len(wanted), batch_size):
        batch = wanted.iloc[start:start + batch_size]
        features = batch_model_inputs(bundle, batch["audio_path"].tolist())
        with ActivationRecorder(bundle, list(layers), module="encoder") as recorder:
            bundle.model.model.encoder(features["input_features"])
            states = {int(k): recorder.states[int(k)].float().cpu().numpy()
                      for k in layers}
        for b, (_, row) in enumerate(batch.iterrows()):
            utterance = str(row["utterance_id"])
            valid = bundle.valid_frames(float(row["duration_sec"]))
            for _, span in by_utterance[utterance].iterrows():
                lo, hi = _frame_range(span["start_sec"], span["end_sec"], step, valid)
                if hi - lo < MIN_SPAN_FRAMES:
                    skipped["too_short"] += 1
                    continue
                control = control_frames(lo, hi, runs_by_utterance.get(utterance,
                                                                       pd.DataFrame()),
                                         step, valid)
                if control is None:
                    skipped["no_control"] += 1
                    continue
                clo, chi = control
                for layer in layers:
                    block = states[int(layer)][b]
                    for variant in POOLING_VARIANTS:
                        positive = pooled(block[lo:hi], pool_weights(hi - lo, variant))
                        negative = pooled(block[clo:chi], pool_weights(chi - clo, variant))
                        vectors.setdefault((int(layer), variant), []).append(
                            positive - negative)
                        if layer == layers[0] and variant == PRIMARY_POOLING:
                            rows.append({
                                "utterance_id": utterance,
                                "dialogue_id": str(span["dialogue_id"]),
                                "conversation_id": str(span["conversation_id"]),
                                "span_duration_sec": float(span["span_duration_sec"]),
                                "n_frames": int(hi - lo),
                                "control_frames": int(chi - clo),
                                "relative_start": float(lo / max(valid, 1)),
                                "utterance_duration_sec": float(row["duration_sec"]),
                                "n_units": int(span["n_units"]),
                            })
        del states, features
        if log and (start // batch_size) % 20 == 0:
            log.info("  site E %d/%d utterances", start + len(batch), len(wanted))
    assert_no_site_hooks(bundle)
    return pd.DataFrame(rows), {"vectors": vectors, "skipped": skipped}


# ---------------------------------------------------------------------------
# Site D: decoder switch-onset contrasts
# ---------------------------------------------------------------------------

def unit_char_spans(norm: str, units: Sequence[Any]) -> list[tuple[int, int]]:
    """Character span of each reference unit inside the normalised text."""
    spans, cursor = [], 0
    for unit in units:
        surface = str(getattr(unit, "surface", unit))
        found = norm.find(surface, cursor)
        if found < 0:
            found = cursor
        spans.append((found, found + len(surface)))
        cursor = found + len(surface)
    return spans


def first_token_for_unit(offsets: Sequence[tuple[int, int]],
                         char_start: int) -> int | None:
    """Index of the first BPE token covering `char_start`."""
    for index, (lo, hi) in enumerate(offsets):
        if hi > char_start:
            return index
    return None


@torch.inference_mode()
def control_step_for(onset: int, control_offset: int = 0) -> int:
    """Decoder step of the Mandarin control, always STRICTLY before ``onset``.

    The control is the matrix-language continuation immediately preceding the
    switch onset, so its default is ``onset - 1``.  A positive ``control_offset``
    moves it further back in time; the step is clamped so that no offset -- of
    either sign -- can place the control at or after the onset.  Without that
    clamp an offset of ``-1`` yields ``control == onset`` and the contrast
    ``block[onset] - block[control]`` is identically zero, which silently
    produces a null rather than an error.
    """
    return min(int(onset) - 1, int(onset) - 1 - int(control_offset))


def site_d_contrasts(bundle, manifest: pd.DataFrame, spans: pd.DataFrame,
                     layers: Sequence[int], *, batch_size: int = 4,
                     control_offset: int = 0,
                     log=None) -> tuple[pd.DataFrame, dict]:
    """Switch-onset contrast at the post-cross-attention state.

    ``control_offset`` moves the Mandarin control step further back from the
    onset, so that ``0`` reproduces the frozen Day 3 path exactly. It exists so
    the Day 6 battery can jitter the *control* boundary without duplicating this
    extraction; nothing in the Day 3 pipeline passes it.
    """
    from ..data.alignment import build_prefix, token_char_offsets
    from ..data.normalize import normalize_and_segment
    from ..models.generation import teacher_forced_forward

    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language="zh")
    eot = tokenizer.eos_token_id
    by_utterance = {u: g for u, g in spans.groupby("utterance_id")}
    rows: list[dict[str, Any]] = []
    vectors: dict[int, list[np.ndarray]] = {}
    prompts: dict[int, list[np.ndarray]] = {}
    skipped = {"no_token": 0, "no_control": 0}

    wanted = manifest[manifest["utterance_id"].isin(by_utterance)].reset_index(drop=True)
    for start in range(0, len(wanted), batch_size):
        batch = wanted.iloc[start:start + batch_size]
        sequences, plans = [], []
        for _, row in batch.iterrows():
            norm, units = normalize_and_segment(row["transcript_raw"])
            text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
            sequences.append(list(prefix) + list(text_ids) + [eot])
            offsets = token_char_offsets(tokenizer, text_ids)
            plans.append((row, norm, units, offsets))

        # one forward pass, recorded: teacher_forced_forward already builds the
        # padded decoder ids and the encoder inputs, so running it inside the
        # recorder captures the site without a second pass over the audio
        with DecoderPostCrossAttnRecorder(bundle, list(layers)) as recorder:
            _out, padded, _mask = teacher_forced_forward(
                bundle, batch["audio_path"].tolist(), sequences,
                output_attentions=False)
            states = {int(k): recorder.states[int(k)].float().cpu().numpy()
                      for k in layers}
        del _out

        for b, (row, norm, units, offsets) in enumerate(plans):
            utterance = str(row["utterance_id"])
            char_spans = unit_char_spans(norm, units)
            for layer in layers:
                prompts.setdefault(int(layer), []).append(
                    states[int(layer)][b, : len(prefix)].mean(axis=0))
            for _, span in by_utterance[utterance].iterrows():
                first_unit = int(min(span["unit_indices"]))
                if first_unit >= len(char_spans):
                    skipped["no_token"] += 1
                    continue
                token_index = first_token_for_unit(offsets, char_spans[first_unit][0])
                if token_index is None:
                    skipped["no_token"] += 1
                    continue
                onset = len(prefix) + token_index - 1        # position predicting it
                control = control_step_for(onset, control_offset)  # matrix continuation
                if control < len(prefix) - 1 or onset >= len(sequences[b]):
                    skipped["no_control"] += 1
                    continue
                for layer in layers:
                    block = states[int(layer)][b]
                    vectors.setdefault(int(layer), []).append(block[onset] - block[control])
                    if layer == layers[0]:
                        rows.append({
                            "utterance_id": utterance,
                            "dialogue_id": str(span["dialogue_id"]),
                            "conversation_id": str(span["conversation_id"]),
                            "span_duration_sec": float(span["span_duration_sec"]),
                            "n_frames": int(span["n_units"]),
                            "relative_start": float(token_index / max(len(offsets), 1)),
                            "utterance_duration_sec": float(row["duration_sec"]),
                            "n_units": int(span["n_units"]),
                            "onset_position": int(onset),
                        })
        del states
        if log and (start // batch_size) % 20 == 0:
            log.info("  site D %d/%d utterances", start + len(batch), len(wanted))
    assert_no_site_hooks(bundle)
    return pd.DataFrame(rows), {"vectors": vectors, "prompts": prompts,
                                "skipped": skipped}


# ---------------------------------------------------------------------------
# direction assembly and reporting
# ---------------------------------------------------------------------------

def assemble(contrasts: np.ndarray, meta: pd.DataFrame, *,
             basis: np.ndarray | None = None) -> dict[str, Any]:
    """Raw -> (optional prompt projection) -> ridge -> clip -> balanced mean."""
    raw = np.asarray(contrasts, dtype=float)
    projected, projection_energy = (project_out(raw, basis) if basis is not None
                                    else (raw, 0.0))
    residual, ridge = residualise(projected, nuisance_matrix(meta))
    clipped, clipping = clip_norms(residual)
    direction, weighting = dialogue_balanced_mean(clipped, meta["dialogue_id"])
    raw_direction, _ = dialogue_balanced_mean(raw, meta["dialogue_id"])
    stability = bootstrap_cosine(clipped, meta["dialogue_id"])
    norms = np.linalg.norm(raw, axis=1)
    def _corr(column: str) -> float | None:
        if column not in meta or meta[column].nunique() < 2:
            return None
        values = pd.to_numeric(meta[column], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(values) & np.isfinite(norms)
        return float(np.corrcoef(norms[ok], values[ok])[0, 1]) if ok.sum() > 2 else None
    return {
        "direction": direction,
        "pairs": int(len(raw)),
        "dialogues": weighting["G"],
        "pairs_per_dialogue": [weighting.get("pairs_per_dialogue_min"),
                               weighting.get("pairs_per_dialogue_max")],
        "cos_raw_residualized": cosine(raw_direction, direction),
        "ridge": ridge,
        "prompt_projection_energy_removed": projection_energy,
        "clipping": clipping,
        "bootstrap_cosine": stability,
        "corr_norm_duration": _corr("span_duration_sec"),
        "corr_norm_confidence": _corr("confidence"),
        "direction_norm": float(np.linalg.norm(direction)),
    }


def resolve_output(output: str | Path) -> Path:
    target = Path(output)
    if target.is_symlink():
        raise SystemExit(f"refusing a symlink destination: {target}")
    target = target.resolve()
    if "artifacts_lss" in str(target):
        raise SystemExit(f"refusing to write inside the v1 root: {target}")
    if target.exists() or manifest_mod.manifest_path(target).exists():
        raise SystemExit(f"refusing an existing destination: {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lda-ablation", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke run: cap the number of construction utterances")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target = resolve_output(args.output_dir)
    spec = cfg.get("spec") or {}
    encoder_layers = [int(v) for v in spec.get("encoder_candidate_layers", [15, 23, 27, 31])]
    decoder_layers = [int(v) for v in spec.get("decoder_candidate_layers", [8, 16, 24, 31])]

    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    verdict = manifest_mod.verify(candidate_path, cfg=None, require_identity=False)
    if not verdict["ok"]:
        raise SystemExit(f"candidate table failed byte verification: {verdict}")
    candidates, _ = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)

    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    manifests, pois = [], []
    for role in conv.DEVELOPMENT_ROLES:
        role_path = role_root / f"role_{role}.parquet"
        poi_path = poi_root / f"poi_{role}.parquet"
        for path in (role_path, poi_path):
            check = manifest_mod.verify(path, cfg=None, require_identity=False)
            if not check["ok"]:
                raise SystemExit(f"{path.name} failed byte verification: {check}")
        manifests.append(pd.read_parquet(role_path))
        pois.append(pd.read_parquet(poi_path).rename(
            columns={"poi_index": "reference_unit_index"}))
    manifest = pd.concat(manifests, ignore_index=True)
    poi = pd.concat(pois, ignore_index=True)
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))

    spans = build_spans(candidates, dialogue_of=dialogue_of)
    spans = attach_baseline_status(spans, poi)
    construct = spans[(spans["role"] == "D-construct") & spans["all_correct"]].copy()
    develop = spans[spans["role"] == "D-dev-select"].copy()

    strata = {
        "all_errors": develop[develop["units_error"] > 0],
        "substitution_only": None,
        "wrong_language_substitution_only": None,
    }
    categories = {(str(u), int(i)): str(c) for u, i, c in
                  zip(poi["utterance_id"], poi["reference_unit_index"], poi["category"])}
    def _stratum(predicate) -> pd.DataFrame:
        keep = []
        for _, span in develop.iterrows():
            cats = [categories.get((str(span["utterance_id"]), int(i)))
                    for i in span["unit_indices"]]
            keep.append(any(predicate(c) for c in cats if c))
        return develop[pd.Series(keep, index=develop.index)]
    strata["substitution_only"] = _stratum(lambda c: c not in ("correct", "deletion"))
    strata["wrong_language_substitution_only"] = _stratum(
        lambda c: c == "wrong_language_substitution")

    conservative = {
        "spans": int(len(develop)),
        "dialogues": int(develop["dialogue_id"].nunique()),
        "utterances": int(develop["utterance_id"].nunique()),
        "baseline_error_units": int(develop["units_error"].sum()),
        "strata": {name: {"spans": int(len(frame)),
                          "units": int(frame["units_error"].sum()),
                          "dialogues": int(frame["dialogue_id"].nunique())}
                   for name, frame in strata.items()},
    }
    construction = {
        "spans": int(len(construct)),
        "units": int(construct["units_correct"].sum()),
        "dialogues": int(construct["dialogue_id"].nunique()),
        "utterances": int(construct["utterance_id"].nunique()),
    }
    plan = {"frozen_configuration": FROZEN_CONFIG,
            "encoder_candidate_layers": encoder_layers,
            "decoder_candidate_layers": decoder_layers,
            "conservative_subset_D_dev_select": conservative,
            "construction_set_D_construct": construction,
            "input_identity_format": "legacy (byte-verified; identity never attempted)",
            "output_identity_format": "split (code_config_sha256 + test_sha256)",
            "destination": str(target)}

    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan}, indent=2, default=str))
        return 0

    log = setup_logging(level="INFO")
    bundle = load_whisper(cfg)
    runs = matrix_runs(candidates)
    build = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
    if args.limit:
        build = build.head(int(args.limit))
        construct = construct[construct["utterance_id"].isin(set(build["utterance_id"]))]
        log_limit = True

    log.info("site E over %d spans, %d layers", len(construct), len(encoder_layers))
    e_meta, e_raw = site_e_contrasts(bundle, build, construct, runs, encoder_layers,
                                     batch_size=args.batch_size, log=log)
    log.info("site D over %d spans, %d layers", len(construct), len(decoder_layers))
    d_meta, d_raw = site_d_contrasts(bundle, build, construct, decoder_layers,
                                     batch_size=args.batch_size, log=log)

    results: dict[str, Any] = {"site_e": {}, "site_d": {}}
    saved: dict[str, np.ndarray] = {}
    for (layer, variant), vecs in sorted(e_raw["vectors"].items()):
        entry = assemble(np.vstack(vecs), e_meta)
        saved[f"site_e_layer{layer}_{variant}"] = entry.pop("direction")
        results["site_e"][f"layer{layer}/{variant}"] = entry
    prompt_bases = {}
    for layer, vecs in sorted(d_raw["vectors"].items()):
        prompts = np.vstack(d_raw["prompts"][layer])
        basis = orthonormal_basis(prompts, PROMPT_SUBSPACE_RANK)
        prompt_bases[layer] = basis
        contrasts = np.vstack(vecs)
        entry = assemble(contrasts, d_meta, basis=basis)
        natural, _ = dialogue_balanced_mean(contrasts, d_meta["dialogue_id"])
        entry["cos_v_nat_U_prompt"] = float(
            np.linalg.norm(natural @ basis) / max(np.linalg.norm(natural), 1e-12))
        if args.lda_ablation:
            half = len(contrasts) // 2
            lda = shrinkage_lda(contrasts[:half], -contrasts[half:], basis)
            entry["cos_lda_mean"] = cosine(lda, entry["direction"]
                                           if "direction" in entry else saved.get(
                                               f"site_d_layer{layer}", lda))
        saved[f"site_d_layer{layer}"] = entry.pop("direction")
        results["site_d"][f"layer{layer}"] = entry

    payload = {"frozen_configuration": FROZEN_CONFIG, "plan": plan,
               "site_e": results["site_e"], "site_d": results["site_d"],
               "skipped": {"site_e": e_raw["skipped"], "site_d": d_raw["skipped"]},
               "selects_layer_or_strength": False, "evaluates_no_gate": True}

    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "directions.npz", **saved)
    e_meta.to_parquet(target / "site_e_pairs.parquet", index=False)
    d_meta.to_parquet(target / "site_d_pairs.parquet", index=False)
    (target / "directions_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    published = []
    for name in ("directions.npz", "site_e_pairs.parquet", "site_d_pairs.parquet",
                 "directions_report.json"):
        published.append(manifest_mod.publish(
            target / name, stage=STAGE, cfg=cfg, parents=[parent] if parent else (),
            schema="lss_v2r3_directions_v1",
            extra={"frozen_configuration": FROZEN_CONFIG, "evaluates_no_gate": True}))

    readback = []
    for name in ("directions.npz", "directions_report.json"):
        path = target / name
        side = manifest_mod.load(path)
        fmt, problem = manifest_mod._identity_format(dict(side.get("identity") or {}))
        check = manifest_mod.verify(path, cfg=cfg, require_identity=True)
        readback.append({"artifact": name, "classified": fmt, "problem": problem,
                         "identity_ok": bool(check["ok"]), "verdict": check["verdict"]})
    print(json.dumps({"state": "completed", "output": str(target),
                      "conservative_subset": conservative,
                      "construction_set": construction,
                      "site_e": results["site_e"], "site_d": results["site_d"],
                      "split_branch_readback": readback}, indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
