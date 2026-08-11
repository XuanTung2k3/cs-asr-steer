"""NAT5H alignment schema v2 and canonical candidate utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from csasr.utils.hashing import sha256_obj

from .coordinates import EncoderGeometry, sample_span_to_encoder_span, seconds_to_sample, waveform_num_frames
from .units import ReferenceUnit

ALIGNMENT_SCHEMA_VERSION = "nat5h_alignment_v2"

ALIGNER_FAMILY_ORDER = ["existing_ctc", "whisper_dtw", "qwen_forced_aligner"]
ALIGNER_VARIANT_ORDER = [
    "existing_ctc/default",
    "whisper_dtw/zh_median7",
    "qwen_forced_aligner/Chinese",
    "qwen_forced_aligner/English",
]

REQUIRED_CANDIDATE_COLUMNS = [
    "schema_version",
    "utterance_id",
    "conversation_id",
    "split",
    "reference_unit_index",
    "reference_text",
    "reference_language",
    "aligner_family",
    "aligner_variant",
    "start_sec",
    "end_sec",
    "audio_duration_sec",
    "is_valid",
    "failure_code",
    "failure_detail",
    "source_token_count",
    "mapping_method",
    "model_id",
    "model_revision",
    "code_commit",
    "config_hash",
]


@dataclass(frozen=True)
class RunIdentity:
    code_commit: str
    config_hash: str
    model_id: str | None = None
    model_revision: str | None = None

    @classmethod
    def from_cfg(cls, cfg: dict, code_commit: str = "unknown") -> "RunIdentity":
        return cls(
            code_commit=code_commit,
            config_hash=sha256_obj(cfg),
            model_id=(cfg.get("model", {}) or {}).get("hub_id") or (cfg.get("model", {}) or {}).get("id"),
            model_revision=(cfg.get("model", {}) or {}).get("revision"),
        )


def assert_schema_v2(df: pd.DataFrame, *, artifact: str = "alignment artifact") -> None:
    if df.empty:
        return
    missing = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{artifact} is not schema v2; missing columns: {missing}")
    versions = set(df["schema_version"].dropna().astype(str))
    if versions != {ALIGNMENT_SCHEMA_VERSION}:
        raise ValueError(f"{artifact} has unsupported schema versions: {sorted(versions)}")


def artifact_matches_identity(df: pd.DataFrame, identity: RunIdentity) -> bool:
    if df.empty:
        return True
    try:
        assert_schema_v2(df)
    except ValueError:
        return False
    return (
        set(df["config_hash"].astype(str)) == {str(identity.config_hash)}
        and set(df["code_commit"].astype(str)) == {str(identity.code_commit)}
    )


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_CANDIDATE_COLUMNS)


def _audio_duration(row: pd.Series, sample_rate: int) -> tuple[float, int]:
    if "duration_sec" in row and pd.notna(row["duration_sec"]):
        duration = float(row["duration_sec"])
    else:
        duration = float("nan")
    if "audio_path" in row:
        try:
            import soundfile as sf

            info = sf.info(str(row["audio_path"]))
            if int(info.samplerate) == int(sample_rate):
                return int(info.frames) / sample_rate, int(info.frames)
            n = int(round(int(info.frames) * sample_rate / int(info.samplerate)))
            return n / sample_rate, n
        except Exception:
            pass
    if math.isfinite(duration):
        return duration, int(round(duration * sample_rate))
    return float("nan"), 0


def candidate_row(
    *,
    manifest_row: pd.Series,
    unit: ReferenceUnit,
    aligner_family: str,
    aligner_variant: str,
    start_sec: float | None,
    end_sec: float | None,
    geometry: EncoderGeometry,
    identity: RunIdentity,
    source_token_count: int = 1,
    mapping_method: str = "",
    model_id: str | None = None,
    model_revision: str | None = None,
    is_valid: bool | None = None,
    failure_code: str = "",
    failure_detail: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audio_duration_sec, waveform_num_samples = _audio_duration(manifest_row, geometry.sample_rate)
    start = float(start_sec) if start_sec is not None else float("nan")
    end = float(end_sec) if end_sec is not None else float("nan")
    sample_rate = int(geometry.sample_rate)
    if math.isfinite(start):
        start_sample = seconds_to_sample(start, sample_rate)
    else:
        start_sample = -1
    if math.isfinite(end):
        end_sample = seconds_to_sample(end, sample_rate)
    else:
        end_sample = -1
    enc_frames = waveform_num_frames(max(waveform_num_samples, 1), geometry)
    if start_sample >= 0 and end_sample > start_sample:
        start_frame, end_frame = sample_span_to_encoder_span(start_sample, end_sample, geometry, enc_frames)
    else:
        start_frame, end_frame = -1, -1
    if is_valid is None:
        is_valid, failure_code, failure_detail = validate_candidate_times(
            start,
            end,
            audio_duration_sec,
            failure_code=failure_code,
            failure_detail=failure_detail,
        )
    source_aligner = aligner_variant
    row = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "utterance_id": str(unit.utterance_id),
        "conversation_id": manifest_row.get("conversation_id", ""),
        "split": manifest_row.get("internal_split", manifest_row.get("split", "")),
        "reference_unit_index": int(unit.unit_id),
        "reference_text": unit.surface,
        "reference_language": unit.language,
        "aligner_family": aligner_family,
        "aligner_variant": aligner_variant,
        "start_sec": start,
        "end_sec": end,
        "audio_duration_sec": audio_duration_sec,
        "is_valid": bool(is_valid),
        "failure_code": "" if is_valid else (failure_code or "invalid_span"),
        "failure_detail": "" if is_valid else str(failure_detail),
        "source_token_count": int(source_token_count),
        "mapping_method": mapping_method,
        "model_id": model_id or identity.model_id,
        "model_revision": model_revision or identity.model_revision,
        "code_commit": identity.code_commit,
        "config_hash": identity.config_hash,
        # compatibility aliases consumed by N3-N6
        "unit_id": int(unit.unit_id),
        "surface": unit.surface,
        "language": unit.language,
        "source_aligner": source_aligner,
        "sample_rate": sample_rate,
        "waveform_num_samples": int(waveform_num_samples),
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "start_encoder_frame": int(start_frame),
        "end_encoder_frame": int(end_frame),
        "encoder_num_frames": int(enc_frames),
        "coordinate_origin": "short_wav_sample0",
        "encoder_step_sec": geometry.encoder_step_sec,
        "speaker": manifest_row.get("speaker_id", manifest_row.get("speaker", "")),
        "audio_path": manifest_row.get("audio_path", ""),
        "metadata": metadata or {},
    }
    return row


def validate_candidate_times(
    start_sec: float,
    end_sec: float,
    audio_duration_sec: float,
    *,
    epsilon_sec: float = 1e-3,
    failure_code: str = "",
    failure_detail: str = "",
) -> tuple[bool, str, str]:
    if failure_code:
        return False, failure_code, failure_detail
    if not (math.isfinite(start_sec) and math.isfinite(end_sec)):
        return False, "non_finite_timestamp", f"start={start_sec}, end={end_sec}"
    if end_sec <= start_sec:
        return False, "reversed_or_zero_duration", f"start={start_sec}, end={end_sec}"
    if start_sec < -epsilon_sec or end_sec > audio_duration_sec + epsilon_sec:
        return False, "timestamp_out_of_bounds", f"start={start_sec}, end={end_sec}, duration={audio_duration_sec}"
    return True, "", ""


def validate_candidate_sequence(
    candidates: pd.DataFrame,
    *,
    overlap_epsilon_sec: float = 1e-3,
    max_adjacent_overlap_sec: float = 0.04,
) -> pd.DataFrame:
    """Mark per-aligner candidate order/overlap failures without repairing timestamps."""
    if candidates.empty:
        return candidates
    assert_schema_v2(candidates, artifact="candidate table")
    out = candidates.copy()
    out["_orig_order"] = np.arange(len(out))
    for (_, _, variant), g in out.groupby(["utterance_id", "aligner_family", "aligner_variant"], sort=False):
        g = g.sort_values("reference_unit_index")
        prev_start = None
        prev_end = None
        for idx, r in g.iterrows():
            if not bool(out.at[idx, "is_valid"]):
                continue
            s = float(r["start_sec"])
            e = float(r["end_sec"])
            if prev_start is not None and s + overlap_epsilon_sec < prev_start:
                out.at[idx, "is_valid"] = False
                out.at[idx, "failure_code"] = "non_monotonic_span"
                out.at[idx, "failure_detail"] = f"start {s:.3f} < previous start {prev_start:.3f}"
            elif prev_end is not None and s + overlap_epsilon_sec < prev_end:
                overlap = prev_end - s
                out.at[idx, "is_valid"] = False
                out.at[idx, "failure_code"] = "adjacent_overlap"
                out.at[idx, "failure_detail"] = (
                    f"overlap {overlap:.3f}s with previous canonical unit"
                    if overlap > max_adjacent_overlap_sec
                    else f"tiny overlap {overlap:.3f}s with previous canonical unit"
                )
            if bool(out.at[idx, "is_valid"]):
                prev_start = s
                prev_end = e
    return out.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)


def collapse_duplicate_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Guarantee one row per stable unit identity and aligner variant."""
    if candidates.empty:
        return candidates, {"duplicates_collapsed": 0}
    assert_schema_v2(candidates, artifact="candidate table")
    rows = []
    duplicates = 0
    for _, g in candidates.groupby(["utterance_id", "reference_unit_index", "aligner_variant"], sort=False):
        if len(g) == 1:
            rows.append(g.iloc[0].to_dict())
            continue
        duplicates += len(g) - 1
        valid = g[g["is_valid"]]
        use = valid.iloc[0] if len(valid) else g.iloc[0]
        merged = use.to_dict()
        merged["source_token_count"] = int(g["source_token_count"].fillna(0).sum())
        merged["failure_detail"] = (merged.get("failure_detail") or "") + f"; collapsed {len(g)} duplicate candidate rows"
        rows.append(merged)
    return pd.DataFrame(rows).reset_index(drop=True), {"duplicates_collapsed": int(duplicates)}


def collapse_token_rows_to_units(
    raw_tokens: pd.DataFrame,
    reference_units: Iterable[ReferenceUnit],
    *,
    unit_index_col: str = "reference_unit_index",
    token_start_col: str = "start_sec",
    token_end_col: str = "end_sec",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse many mapped raw token/subword rows into one row per reference unit."""
    refs = {int(u.unit_id): u for u in reference_units}
    if raw_tokens.empty:
        return pd.DataFrame(columns=["reference_unit_index", "start_sec", "end_sec", "source_token_count"]), {
            "raw_token_rows": 0,
            "mapped_token_rows": 0,
            "unique_mapped_reference_units": 0,
            "unmapped_tokens": 0,
            "duplicate_mappings_collapsed": 0,
        }
    mapped = raw_tokens[raw_tokens[unit_index_col].notna()].copy()
    mapped[unit_index_col] = mapped[unit_index_col].astype(int)
    mapped = mapped[mapped[unit_index_col].isin(refs)]
    rows = []
    duplicate_count = 0
    for unit_idx, g in mapped.groupby(unit_index_col, sort=True):
        duplicate_count += max(0, len(g) - 1)
        rows.append(
            {
                "reference_unit_index": int(unit_idx),
                "start_sec": float(g[token_start_col].min()),
                "end_sec": float(g[token_end_col].max()),
                "source_token_count": int(len(g)),
            }
        )
    stats = {
        "raw_token_rows": int(len(raw_tokens)),
        "mapped_token_rows": int(len(mapped)),
        "unique_mapped_reference_units": int(len(rows)),
        "unmapped_tokens": int(len(raw_tokens) - len(mapped)),
        "duplicate_mappings_collapsed": int(duplicate_count),
    }
    return pd.DataFrame(rows), stats
