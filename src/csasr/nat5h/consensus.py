"""Mechanical alignment validation and natural multi-aligner consensus."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .coordinates import validate_timestamp_record
from .schema import (
    ALIGNER_FAMILY_ORDER,
    ALIGNER_VARIANT_ORDER,
    ALIGNMENT_SCHEMA_VERSION,
    assert_schema_v2,
)


@dataclass(frozen=True)
class SmokeCriteria:
    min_unit_coverage: float = 0.95
    min_monotonic_valid_units: float = 0.99
    max_zero_duration_rate: float = 0.05
    require_zero_out_of_waveform: bool = True


@dataclass(frozen=True)
class ConsensusCriteria:
    min_aligners: int = 2
    max_start_disagreement_ms: float = 200.0
    max_end_disagreement_ms: float = 200.0
    min_unit_coverage: float = 0.95
    safe_interior_erosion_ms: float = 250.0
    min_safe_interior_ms: float = 200.0
    steering_union_padding_ms: float = 100.0


@dataclass(frozen=True)
class N1ConsensusGate:
    min_consensus_utterances: int = 10
    min_consensus_spans: int = 50
    min_consensus_en_spans: int = 20
    min_consensus_zh_spans: int = 20
    require_ctc_family: bool = True
    min_valid_families: int = 2


def _as_bool_series(values, length: int) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(bool)
    return pd.Series([bool(values)] * length)


def alignment_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-row mechanical validity flags for the common timestamp schema."""
    out = df.copy()
    n = len(out)
    if n == 0:
        for col in (
            "valid_bounds",
            "nonzero_duration",
            "monotonic_in_utterance",
            "out_of_waveform",
            "valid_timestamp",
        ):
            out[col] = pd.Series(dtype=bool)
        return out

    required = {
        "utterance_id",
        "source_aligner",
        "unit_id",
        "language",
        "start_sample",
        "end_sample",
        "waveform_num_samples",
        "start_encoder_frame",
        "end_encoder_frame",
        "encoder_num_frames",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"alignment table missing required columns: {missing}")

    out["valid_bounds"] = (
        (out["start_sample"] >= 0)
        & (out["end_sample"] > out["start_sample"])
        & (out["end_sample"] <= out["waveform_num_samples"])
        & (out["start_encoder_frame"] >= 0)
        & (out["end_encoder_frame"] > out["start_encoder_frame"])
        & (out["end_encoder_frame"] <= out["encoder_num_frames"])
    )
    out["nonzero_duration"] = out["end_sample"] > out["start_sample"]
    out["out_of_waveform"] = (
        (out["start_sample"] < 0) | (out["end_sample"] > out["waveform_num_samples"])
    )
    out = out.sort_values(["source_aligner", "utterance_id", "unit_id"]).reset_index(drop=True)
    prev_end = out.groupby(["source_aligner", "utterance_id"])["end_sample"].shift(1)
    out["monotonic_in_utterance"] = (prev_end.isna()) | (out["start_sample"] >= prev_end)
    out["valid_timestamp"] = (
        out["valid_bounds"] & out["nonzero_duration"] & out["monotonic_in_utterance"]
    )
    return out


def summarize_alignment_smoke(
    alignment_df: pd.DataFrame,
    reference_units_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mechanical validity by aligner for N1 smoke."""
    if alignment_df.empty:
        return pd.DataFrame(
            columns=[
                "source_aligner",
                "reference_units",
                "aligned_units",
                "unit_coverage",
                "monotonic_valid_units",
                "zero_duration_rate",
                "invalid_timestamp_rate",
                "out_of_waveform_rate",
                "median_unit_duration_ms",
                "has_en",
                "has_zh",
            ]
        )

    flags = alignment_validation_flags(alignment_df)
    refs = reference_units_df[reference_units_df["language"].isin(["EN", "ZH"])]
    ref_counts = refs.groupby("utterance_id")["unit_id"].nunique()
    total_ref = int(ref_counts.sum()) if len(ref_counts) else 0
    rows = []
    for aligner, g in flags.groupby("source_aligner", dropna=False):
        aligned = g[["utterance_id", "unit_id"]].drop_duplicates()
        aligned_units = len(aligned)
        valid_count = int(g["valid_timestamp"].sum())
        duration_ms = (g["end_sample"] - g["start_sample"]) / g["sample_rate"] * 1000.0
        rows.append(
            {
                "source_aligner": aligner,
                "reference_units": total_ref,
                "aligned_units": aligned_units,
                "unit_coverage": float(aligned_units / total_ref) if total_ref else 0.0,
                "monotonic_valid_units": float(valid_count / len(g)) if len(g) else 0.0,
                "zero_duration_rate": float((g["end_sample"] <= g["start_sample"]).mean()),
                "invalid_timestamp_rate": float((~g["valid_timestamp"]).mean()),
                "out_of_waveform_rate": float(g["out_of_waveform"].mean()),
                "median_unit_duration_ms": float(np.median(duration_ms)) if len(g) else np.nan,
                "has_en": bool((g["language"] == "EN").any()),
                "has_zh": bool((g["language"] == "ZH").any()),
            }
        )
    return pd.DataFrame(rows)


def summarize_alignment_smoke_v2(
    candidates: pd.DataFrame,
    reference_units_df: pd.DataFrame,
) -> pd.DataFrame:
    """Schema-v2 diagnostic aligner/family stats. Not the scientific gate."""
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "aligner_family",
                "aligner_variant",
                "reference_units",
                "candidate_units",
                "valid_units",
                "unit_coverage",
                "valid_unit_coverage",
                "invalid_timestamp_rate",
                "has_en",
                "has_zh",
                "top_failure_code",
            ]
        )
    assert_schema_v2(candidates, artifact="N1/N2 candidate table")
    refs = reference_units_df[reference_units_df["language"].isin(["EN", "ZH"])]
    expected = int(refs[["utterance_id", "unit_id"]].drop_duplicates().shape[0])
    rows = []
    for (family, variant), g in candidates.groupby(["aligner_family", "aligner_variant"], dropna=False):
        content = g[g["reference_language"].isin(["EN", "ZH"])]
        candidate_units = int(content[["utterance_id", "reference_unit_index"]].drop_duplicates().shape[0])
        valid = content[content["is_valid"]]
        valid_units = int(valid[["utterance_id", "reference_unit_index"]].drop_duplicates().shape[0])
        coverage = candidate_units / expected if expected else 0.0
        valid_coverage = valid_units / expected if expected else 0.0
        if coverage > 1.0 + 1e-9 or valid_coverage > 1.0 + 1e-9:
            raise AssertionError(
                f"schema-v2 coverage exceeded 1 for {family}/{variant}: "
                f"coverage={coverage}, valid={valid_coverage}"
            )
        failures = content.loc[~content["is_valid"], "failure_code"].value_counts()
        rows.append(
            {
                "aligner_family": family,
                "aligner_variant": variant,
                "reference_units": expected,
                "candidate_units": candidate_units,
                "valid_units": valid_units,
                "unit_coverage": float(coverage),
                "valid_unit_coverage": float(valid_coverage),
                "invalid_timestamp_rate": float((~content["is_valid"]).mean()) if len(content) else 0.0,
                "has_en": bool((valid["reference_language"] == "EN").any()),
                "has_zh": bool((valid["reference_language"] == "ZH").any()),
                "top_failure_code": failures.index[0] if len(failures) else "",
            }
        )
    return pd.DataFrame(rows)


def smoke_passes(row: pd.Series | dict, criteria: SmokeCriteria) -> bool:
    r = dict(row)
    if float(r.get("unit_coverage", 0.0)) < criteria.min_unit_coverage:
        return False
    if float(r.get("monotonic_valid_units", 0.0)) < criteria.min_monotonic_valid_units:
        return False
    if float(r.get("zero_duration_rate", 1.0)) > criteria.max_zero_duration_rate:
        return False
    if criteria.require_zero_out_of_waveform and float(r.get("out_of_waveform_rate", 1.0)) != 0.0:
        return False
    return bool(r.get("has_en", False)) and bool(r.get("has_zh", False))


def _order_index(value: str, order: list[str]) -> int:
    try:
        return order.index(str(value))
    except ValueError:
        return len(order) + 100


def _candidate_variant_order(row: pd.Series) -> tuple[int, int, str]:
    return (
        _order_index(str(row["aligner_family"]), ALIGNER_FAMILY_ORDER),
        _order_index(str(row["aligner_variant"]), ALIGNER_VARIANT_ORDER),
        str(row["aligner_variant"]),
    )


def _family_representatives(valid: pd.DataFrame) -> pd.DataFrame:
    """At most one vote per aligner family; Qwen language variants cannot double-vote."""
    reps = []
    for _, g in valid.groupby("aligner_family", sort=False):
        gg = g.copy()
        gg["_ord"] = [_candidate_variant_order(r) for _, r in gg.iterrows()]
        reps.append(gg.sort_values("_ord").iloc[0].drop(labels=["_ord"]).to_dict())
    return pd.DataFrame(reps)


def _agreement_pair(a: pd.Series, b: pd.Series, criteria: ConsensusCriteria) -> dict:
    start_ms = abs(float(a["start_sec"]) - float(b["start_sec"])) * 1000.0
    end_ms = abs(float(a["end_sec"]) - float(b["end_sec"])) * 1000.0
    ok = start_ms <= criteria.max_start_disagreement_ms and end_ms <= criteria.max_end_disagreement_ms
    return {
        "family_a": a["aligner_family"],
        "family_b": b["aligner_family"],
        "variant_a": a["aligner_variant"],
        "variant_b": b["aligner_variant"],
        "start_disagreement_ms": float(start_ms),
        "end_disagreement_ms": float(end_ms),
        "agrees": bool(ok),
    }


def _select_agreeing_set(reps: pd.DataFrame, criteria: ConsensusCriteria) -> tuple[pd.DataFrame, list[dict]]:
    pairs = []
    if len(reps) < 2:
        return reps.iloc[0:0], pairs
    reps = reps.sort_values(["aligner_family", "aligner_variant"]).reset_index(drop=True)
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            pairs.append(_agreement_pair(reps.iloc[i], reps.iloc[j], criteria))
    # Largest mutually agreeing set. With three families this is direct.
    best_indices: list[int] = []
    for size in range(len(reps), 1, -1):
        from itertools import combinations

        for idxs in combinations(range(len(reps)), size):
            ok = True
            for i, j in combinations(idxs, 2):
                p = _agreement_pair(reps.iloc[i], reps.iloc[j], criteria)
                if not p["agrees"]:
                    ok = False
                    break
            if ok:
                best_indices = list(idxs)
                break
        if best_indices:
            break
    if not best_indices:
        return reps.iloc[0:0], pairs
    selected = reps.iloc[best_indices].copy()
    selected["_ord"] = [_candidate_variant_order(r) for _, r in selected.iterrows()]
    selected = selected.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    return selected, pairs


def _reject_row_from_group(
    key: tuple,
    group: pd.DataFrame,
    reference_row: pd.Series | None,
    reason: str,
    detail: str,
    pairwise: list[dict] | None = None,
) -> dict:
    if len(group):
        first = group.iloc[0]
        utt = first["utterance_id"]
        unit_idx = int(first["reference_unit_index"])
        ref_text = first["reference_text"]
        ref_lang = first["reference_language"]
        split = first.get("split", "")
        speaker = first.get("speaker", "")
    else:
        utt, unit_idx = key
        ref_text = reference_row.get("surface", "") if reference_row is not None else ""
        ref_lang = reference_row.get("language", "") if reference_row is not None else ""
        split = ""
        speaker = ""
    return {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "utterance_id": utt,
        "unit_id": unit_idx,
        "reference_unit_index": unit_idx,
        "surface": ref_text,
        "language": ref_lang,
        "reference_text": ref_text,
        "reference_language": ref_lang,
        "split": split,
        "speaker": speaker,
        "accepted": False,
        "rejection_code": reason,
        "rejection_detail": detail,
        "candidate_families": sorted(group["aligner_family"].dropna().unique().tolist()) if len(group) else [],
        "candidate_variants": sorted(group["aligner_variant"].dropna().unique().tolist()) if len(group) else [],
        "valid_candidate_families": sorted(group.loc[group.get("is_valid", False) == True, "aligner_family"].dropna().unique().tolist()) if len(group) and "is_valid" in group else [],
        "pairwise_disagreements": pairwise or [],
    }


def build_unit_consensus_v2(
    candidates: pd.DataFrame,
    criteria: ConsensusCriteria,
    *,
    reference_units_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reference-unit-level consensus from valid candidates in distinct families."""
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    assert_schema_v2(candidates, artifact="candidate table")
    content = candidates[candidates["reference_language"].isin(["EN", "ZH"])].copy()
    groups = {
        (u, int(i)): g
        for (u, i), g in content.groupby(["utterance_id", "reference_unit_index"], sort=False)
    }
    reference_lookup: dict[tuple, pd.Series] = {}
    if reference_units_df is not None and len(reference_units_df):
        refs = reference_units_df[reference_units_df["language"].isin(["EN", "ZH"])]
        reference_lookup = {
            (r["utterance_id"], int(r["unit_id"])): r
            for _, r in refs.iterrows()
        }
        keys = list(reference_lookup)
    else:
        keys = list(groups)

    accepted = []
    rejected = []
    pairwise_rows = []
    for key in keys:
        g = groups.get(key, pd.DataFrame(columns=content.columns))
        valid = g[g["is_valid"]].copy() if len(g) else g
        reps = _family_representatives(valid) if len(valid) else valid
        if len(reps["aligner_family"].unique()) < criteria.min_aligners if len(reps) else True:
            invalid_codes = g.loc[~g["is_valid"], "failure_code"].value_counts().to_dict() if len(g) else {}
            rejected.append(
                _reject_row_from_group(
                    key,
                    g,
                    reference_lookup.get(key),
                    "missing_second_aligner",
                    f"valid families={sorted(reps['aligner_family'].unique().tolist()) if len(reps) else []}; invalid={invalid_codes}",
                )
            )
            continue
        selected, pairs = _select_agreeing_set(reps, criteria)
        for p in pairs:
            pairwise_rows.append({"utterance_id": key[0], "reference_unit_index": key[1], **p})
        if len(selected) < criteria.min_aligners:
            if pairs:
                min_start = min(p["start_disagreement_ms"] for p in pairs)
                min_end = min(p["end_disagreement_ms"] for p in pairs)
                if min_start > criteria.max_start_disagreement_ms:
                    code = "start_disagreement_gt_200ms"
                elif min_end > criteria.max_end_disagreement_ms:
                    code = "end_disagreement_gt_200ms"
                else:
                    code = "insufficient_cross_aligner_agreement"
                detail = f"best_start_ms={min_start:.1f}; best_end_ms={min_end:.1f}"
            else:
                code = "missing_second_aligner"
                detail = "no pairwise candidate comparison available"
            rejected.append(_reject_row_from_group(key, g, reference_lookup.get(key), code, detail, pairs))
            continue
        starts = selected["start_sample"].astype(int).tolist()
        ends = selected["end_sample"].astype(int).tolist()
        sample_rate = int(selected.iloc[0]["sample_rate"])
        consensus_start = int(round(float(np.median(starts))))
        consensus_end = int(round(float(np.median(ends))))
        if consensus_end <= consensus_start:
            rejected.append(_reject_row_from_group(key, g, reference_lookup.get(key), "reversed_or_zero_duration", "median consensus had non-positive duration", pairs))
            continue
        safe = erode_safe_interior(
            consensus_start,
            consensus_end,
            sample_rate,
            criteria.safe_interior_erosion_ms,
            criteria.min_safe_interior_ms,
        )
        if safe is None:
            rejected.append(_reject_row_from_group(key, g, reference_lookup.get(key), "safe_interior_too_short", "250 ms erosion removed usable interior", pairs))
            continue
        mask_start, mask_end = consensus_union_mask(
            starts,
            ends,
            int(selected.iloc[0]["waveform_num_samples"]),
            sample_rate,
            criteria.steering_union_padding_ms,
        )
        first = selected.iloc[0]
        start_dis = (max(starts) - min(starts)) / sample_rate * 1000.0
        end_dis = (max(ends) - min(ends)) / sample_rate * 1000.0
        accepted.append(
            {
                "schema_version": ALIGNMENT_SCHEMA_VERSION,
                "utterance_id": key[0],
                "unit_id": int(key[1]),
                "reference_unit_index": int(key[1]),
                "surface": first["reference_text"],
                "language": first["reference_language"],
                "reference_text": first["reference_text"],
                "reference_language": first["reference_language"],
                "split": first.get("split"),
                "speaker": first.get("speaker"),
                "sample_rate": sample_rate,
                "waveform_num_samples": int(first["waveform_num_samples"]),
                "consensus_start_sample": consensus_start,
                "consensus_end_sample": consensus_end,
                "consensus_start_second": consensus_start / sample_rate,
                "consensus_end_second": consensus_end / sample_rate,
                "safe_start_sample": int(safe[0]),
                "safe_end_sample": int(safe[1]),
                "safe_start_second": safe[0] / sample_rate,
                "safe_end_second": safe[1] / sample_rate,
                "mask_start_sample": int(mask_start),
                "mask_end_sample": int(mask_end),
                "mask_start_second": mask_start / sample_rate,
                "mask_end_second": mask_end / sample_rate,
                "selected_aligner_families": selected["aligner_family"].tolist(),
                "selected_aligner_variants": selected["aligner_variant"].tolist(),
                "candidate_start_sec": selected[["aligner_variant", "start_sec"]].to_dict("records"),
                "candidate_end_sec": selected[["aligner_variant", "end_sec"]].to_dict("records"),
                "pairwise_disagreements": pairs,
                "num_votes": int(len(selected)),
                "start_disagreement_ms": float(start_dis),
                "end_disagreement_ms": float(end_dis),
                "confidence_category": "high" if start_dis <= 100.0 and end_dis <= 100.0 else "medium",
                "confidence_score": 1.0 / (1.0 + (start_dis + end_dis) / 2000.0),
                "coordinate_origin": "short_wav_sample0",
                "pseudo_mask_kind": "consensus-union pseudo-mask",
                "accepted": True,
            }
        )
    return pd.DataFrame(accepted), pd.DataFrame(rejected), pd.DataFrame(pairwise_rows)


def evaluate_n1_consensus_gate(consensus: pd.DataFrame, candidates: pd.DataFrame, gate: N1ConsensusGate) -> tuple[str, dict]:
    """Return (state, evidence) for the revised N1 mechanical gate."""
    valid_families = sorted(candidates.loc[candidates["is_valid"], "aligner_family"].dropna().unique().tolist()) if len(candidates) else []
    evidence = {
        "valid_families": valid_families,
        "num_valid_families": len(valid_families),
        "consensus_utterances": int(consensus["utterance_id"].nunique()) if len(consensus) else 0,
        "consensus_spans": int(len(consensus)),
        "consensus_en_spans": int((consensus["language"] == "EN").sum()) if len(consensus) else 0,
        "consensus_zh_spans": int((consensus["language"] == "ZH").sum()) if len(consensus) else 0,
    }
    failures = []
    if gate.require_ctc_family and "existing_ctc" not in valid_families:
        failures.append("adapter_failure_existing_ctc")
    if len(valid_families) < gate.min_valid_families:
        failures.append("insufficient_valid_families")
    if evidence["consensus_utterances"] < gate.min_consensus_utterances:
        failures.append("insufficient_cross_aligner_agreement_utterances")
    if evidence["consensus_spans"] < gate.min_consensus_spans:
        failures.append("insufficient_cross_aligner_agreement_spans")
    if evidence["consensus_en_spans"] < gate.min_consensus_en_spans:
        failures.append("insufficient_english_coverage")
    if evidence["consensus_zh_spans"] < gate.min_consensus_zh_spans:
        failures.append("insufficient_chinese_coverage")
    evidence["failure_reasons"] = failures
    return ("passed" if not failures else "completed_no_go"), evidence


def pairwise_disagreement(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise start/end disagreement for rows with matching utterance/unit."""
    if alignment_df.empty:
        return pd.DataFrame(
            columns=[
                "utterance_id",
                "unit_id",
                "language",
                "aligner_a",
                "aligner_b",
                "start_disagreement_ms",
                "end_disagreement_ms",
            ]
        )
    flags = alignment_validation_flags(alignment_df)
    valid = flags[flags["valid_timestamp"]].copy()
    rows = []
    for (utt, unit), g in valid.groupby(["utterance_id", "unit_id"], dropna=False):
        if g["source_aligner"].nunique() < 2:
            continue
        for _, a in g.iterrows():
            for _, b in g[g["source_aligner"] > a["source_aligner"]].iterrows():
                sr = float(a["sample_rate"])
                rows.append(
                    {
                        "utterance_id": utt,
                        "unit_id": int(unit),
                        "language": a.get("language"),
                        "aligner_a": a["source_aligner"],
                        "aligner_b": b["source_aligner"],
                        "start_disagreement_ms": abs(
                            int(a["start_sample"]) - int(b["start_sample"])
                        )
                        / sr
                        * 1000.0,
                        "end_disagreement_ms": abs(int(a["end_sample"]) - int(b["end_sample"]))
                        / sr
                        * 1000.0,
                    }
                )
    return pd.DataFrame(rows)


def erode_safe_interior(
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    erosion_ms: float,
    min_safe_ms: float,
) -> tuple[int, int] | None:
    erosion = int(round(sample_rate * erosion_ms / 1000.0))
    safe_start = int(start_sample) + erosion
    safe_end = int(end_sample) - erosion
    min_len = int(round(sample_rate * min_safe_ms / 1000.0))
    if safe_end - safe_start < min_len:
        return None
    return safe_start, safe_end


def consensus_union_mask(
    starts: Iterable[int],
    ends: Iterable[int],
    waveform_num_samples: int,
    sample_rate: int,
    padding_ms: float,
) -> tuple[int, int]:
    pad = int(round(sample_rate * padding_ms / 1000.0))
    mask_start = max(0, min(int(v) for v in starts) - pad)
    mask_end = min(int(waveform_num_samples), max(int(v) for v in ends) + pad)
    if mask_end <= mask_start:
        mask_end = min(int(waveform_num_samples), mask_start + 1)
    return mask_start, mask_end


def build_consensus(
    alignment_df: pd.DataFrame,
    criteria: ConsensusCriteria,
    *,
    valid_aligners: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build high-confidence natural pseudo-alignments from aligner agreement."""
    if alignment_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    flags = alignment_validation_flags(alignment_df)
    valid = flags[flags["valid_timestamp"]].copy()
    if valid_aligners is not None:
        valid = valid[valid["source_aligner"].isin(valid_aligners)].copy()

    agreement_rows = pairwise_disagreement(valid)
    accepted = []
    for keys, g in valid.groupby(["utterance_id", "unit_id"], dropna=False):
        if g["source_aligner"].nunique() < criteria.min_aligners:
            continue
        agree_pair_rows = []
        for a_idx, b_idx in combinations(g.index, 2):
            a = g.loc[a_idx]
            b = g.loc[b_idx]
            sr = int(a["sample_rate"])
            start_dis = abs(int(a["start_sample"]) - int(b["start_sample"])) / sr * 1000.0
            end_dis = abs(int(a["end_sample"]) - int(b["end_sample"])) / sr * 1000.0
            if (
                start_dis <= criteria.max_start_disagreement_ms
                and end_dis <= criteria.max_end_disagreement_ms
            ):
                agree_pair_rows.extend([a_idx, b_idx])
        agree_indices = sorted(set(agree_pair_rows))
        if len(agree_indices) < criteria.min_aligners:
            continue
        gg = g.loc[agree_indices]
        starts = [int(v) for v in gg["start_sample"].tolist()]
        ends = [int(v) for v in gg["end_sample"].tolist()]
        sample_rate = int(gg.iloc[0]["sample_rate"])
        consensus_start = int(round(float(np.median(starts))))
        consensus_end = int(round(float(np.median(ends))))
        if consensus_end <= consensus_start:
            continue
        safe = erode_safe_interior(
            consensus_start,
            consensus_end,
            sample_rate,
            criteria.safe_interior_erosion_ms,
            criteria.min_safe_interior_ms,
        )
        if safe is None:
            continue
        mask_start, mask_end = consensus_union_mask(
            starts,
            ends,
            int(gg.iloc[0]["waveform_num_samples"]),
            sample_rate,
            criteria.steering_union_padding_ms,
        )
        encoder_step = int(round(sample_rate * float(gg.iloc[0].get("encoder_step_sec", 0.02))))
        accepted.append(
            {
                "utterance_id": keys[0],
                "unit_id": int(keys[1]),
                "surface": gg.iloc[0].get("surface"),
                "language": gg.iloc[0].get("language"),
                "split": gg.iloc[0].get("split"),
                "speaker": gg.iloc[0].get("speaker"),
                "sample_rate": sample_rate,
                "waveform_num_samples": int(gg.iloc[0]["waveform_num_samples"]),
                "consensus_start_sample": consensus_start,
                "consensus_end_sample": consensus_end,
                "consensus_start_second": consensus_start / sample_rate,
                "consensus_end_second": consensus_end / sample_rate,
                "safe_start_sample": safe[0],
                "safe_end_sample": safe[1],
                "safe_start_second": safe[0] / sample_rate,
                "safe_end_second": safe[1] / sample_rate,
                "mask_start_sample": mask_start,
                "mask_end_sample": mask_end,
                "mask_start_second": mask_start / sample_rate,
                "mask_end_second": mask_end / sample_rate,
                "agreeing_aligners": sorted(gg["source_aligner"].unique().tolist()),
                "all_candidate_aligners": sorted(g["source_aligner"].unique().tolist()),
                "start_disagreement_ms": (max(starts) - min(starts)) / sample_rate * 1000.0,
                "end_disagreement_ms": (max(ends) - min(ends)) / sample_rate * 1000.0,
                "confidence_score": 1.0
                / (
                    1.0
                    + ((max(starts) - min(starts)) + (max(ends) - min(ends)))
                    / (2.0 * sample_rate)
                ),
                "coordinate_origin": "short_wav_sample0",
                "pseudo_mask_kind": "consensus-union pseudo-mask",
            }
        )
    return pd.DataFrame(accepted), agreement_rows
