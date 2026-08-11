"""The frozen span table every later stage consumes.

Loading is hash-checked against the freeze written by the validation stage, so a
downstream stage cannot silently consume spans that were regenerated after Gate A
passed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.config import artifacts_root
from .artifacts import artifact_ref, read_frozen, write_json

SPAN_SCHEMA_VERSION = "lss_spans_v1"
SPANS_FILE = "alignments/consensus_spans_v1.parquet"
REJECTED_FILE = "alignments/consensus_rejected_v1.parquet"
FREEZE_FILE = "freeze/l1b_spans_freeze.json"

PRIMARY_BINS = ("high", "medium")

SPAN_COLUMNS = (
    "schema_version", "utterance_id", "conversation_id", "role", "unit_id",
    "surface", "language", "consensus_start_sample", "consensus_end_sample",
    "start_frame", "end_frame", "safe_start_sample", "safe_end_sample",
    "mask_start_sample", "mask_end_sample", "duration_ms", "n_families",
    "start_disagreement_ms", "end_disagreement_ms", "max_edge_disagreement_ms",
    "confidence_bin", "is_embedded_english", "utterance_contains_code_switch",
    "consensus_tolerance_ms", "consensus_estimator", "spec_freeze_sha256",
)


def spans_to_frames(spans: pd.DataFrame, geometry=None, *,
                    encoder_step_sec: float = 0.02,
                    sample_rate: int = 16000) -> pd.DataFrame:
    """Add encoder-frame columns for every sample-coordinate column."""
    out = spans.copy()
    step = float(encoder_step_sec if geometry is None else geometry.encoder_step_sec)
    rate = float(sample_rate if geometry is None else geometry.sample_rate)
    for prefix, (start_col, end_col) in {
        "": ("consensus_start_sample", "consensus_end_sample"),
        "safe_": ("safe_start_sample", "safe_end_sample"),
        "mask_": ("mask_start_sample", "mask_end_sample"),
    }.items():
        if start_col not in out or end_col not in out:
            continue
        start = np.floor(out[start_col].astype(float) / rate / step).astype(int)
        end = np.ceil(out[end_col].astype(float) / rate / step).astype(int)
        end = np.maximum(end, start + 1)
        out[f"{prefix}start_frame" if prefix else "start_frame"] = start
        out[f"{prefix}end_frame" if prefix else "end_frame"] = end
    return out


def high_confidence_subset(spans: pd.DataFrame,
                           bins: Sequence[str] = PRIMARY_BINS) -> pd.DataFrame:
    """Spans whose edges agree tightly enough for a local claim."""
    if not len(spans) or "confidence_bin" not in spans:
        return spans
    return spans[spans["confidence_bin"].isin(list(bins))].reset_index(drop=True)


def steering_mask(row: Mapping[str, Any], n_frames: int, *,
                  taper_ms: float = 100.0, encoder_step_sec: float = 0.02,
                  sample_rate: int = 16000) -> np.ndarray:
    """Tapered gain over one span's padded mask window."""
    from ..steering.masks import MaskSpec, build_gain, ms_to_frames

    start = int(np.floor(float(row["mask_start_sample"]) / sample_rate / encoder_step_sec))
    end = int(np.ceil(float(row["mask_end_sample"]) / sample_rate / encoder_step_sec))
    spec = MaskSpec(kind="EXACT_TAPER",
                    shoulder_frames=ms_to_frames(taper_ms, encoder_step_sec))
    gain, _ = build_gain(spec, start, max(end, start + 1), int(n_frames), int(n_frames))
    return gain


def freeze_spans(cfg: Mapping[str, Any], spans: pd.DataFrame, rejected: pd.DataFrame,
                 evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Record the hashes downstream stages will verify."""
    root = artifacts_root(cfg)
    payload = {
        "schema_version": SPAN_SCHEMA_VERSION,
        "spans": artifact_ref(root / SPANS_FILE, schema=SPAN_SCHEMA_VERSION,
                              rows=len(spans)),
        "rejected": artifact_ref(root / REJECTED_FILE, rows=len(rejected)),
        "primary_bins": list(PRIMARY_BINS),
        "evidence": dict(evidence),
    }
    write_json(payload, root / FREEZE_FILE)
    return payload


def load_spans(cfg: Mapping[str, Any], *, bins: Sequence[str] = PRIMARY_BINS,
               roles: Sequence[str] | None = None, verify: bool = True) -> pd.DataFrame:
    """Load the frozen spans, hash-checked, restricted to the primary bins."""
    import json

    root = artifacts_root(cfg)
    freeze_path = root / FREEZE_FILE
    path = root / SPANS_FILE
    if verify:
        if not freeze_path.is_file():
            raise FileNotFoundError(
                f"span freeze missing: {freeze_path}. Run l1b_valid first.")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        spans = read_frozen(path, freeze["spans"]["sha256"])
    else:
        spans = pd.read_parquet(path)
    if bins and "confidence_bin" in spans:
        spans = high_confidence_subset(spans, bins)
    if roles and "role" in spans:
        spans = spans[spans["role"].isin(list(roles))]
    return spans.reset_index(drop=True)
