"""Deterministic split and POI selection helpers for NAT5H."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from csasr.evaluation.mer import ref_unit_status


def stable_row_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def reject_test_split(df: pd.DataFrame) -> None:
    if "official_split" in df.columns and (df["official_split"].astype(str) == "test").any():
        raise ValueError("NAT5H must not access official test split")
    if "internal_split" in df.columns and (df["internal_split"].astype(str) == "test").any():
        raise ValueError("NAT5H must not access internal test split")


def reject_dev_confirm(df: pd.DataFrame) -> None:
    if "internal_split" in df.columns and (df["internal_split"].astype(str) == "dev_confirm").any():
        raise ValueError("NAT5H fast experiment must not access dev_confirm")


def validate_disjoint_speakers(train: pd.DataFrame, dev: pd.DataFrame) -> None:
    t = set(train.get("speaker_id", pd.Series(dtype=str)).astype(str))
    d = set(dev.get("speaker_id", pd.Series(dtype=str)).astype(str))
    overlap = sorted(t & d)
    if overlap:
        raise ValueError(f"train/dev speaker overlap: {overlap[:10]}")


def speaker_balanced_sample(
    df: pd.DataFrame,
    n: int,
    *,
    seed: int,
    speaker_col: str = "speaker_id",
) -> pd.DataFrame:
    if len(df) <= n:
        sort_cols = [c for c in (speaker_col, "utterance_id") if c in df.columns]
        return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    work = df.copy()
    if speaker_col not in work.columns:
        work[speaker_col] = "unknown"
    work["_stable"] = [
        stable_row_key(seed, r.get(speaker_col, ""), r.get("utterance_id", ""), i)
        for i, r in work.reset_index(drop=True).iterrows()
    ]
    groups = []
    for _, g in work.sort_values("_stable").groupby(speaker_col, sort=True):
        gg = g.copy()
        gg["_rand"] = rng.random(len(gg))
        groups.append(gg.sort_values(["_rand", "_stable"]))
    selected = []
    while len(selected) < n and groups:
        next_groups = []
        for g in groups:
            if len(g) and len(selected) < n:
                selected.append(g.iloc[0])
                rest = g.iloc[1:]
                if len(rest):
                    next_groups.append(rest)
        groups = next_groups
    out = pd.DataFrame(selected).drop(columns=[c for c in ("_stable", "_rand") if c in pd.DataFrame(selected).columns])
    return out.reset_index(drop=True)


def attach_baseline_unit_status(
    consensus: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach baseline correctness/category for each consensus unit."""
    if consensus.empty:
        return consensus.copy()
    ref_by_utt = manifest.set_index("utterance_id")["transcript_raw"].to_dict()
    norm_by_utt = manifest.set_index("utterance_id")["transcript_normalized"].to_dict()
    hyp_by_utt = baseline_predictions.set_index("utterance_id")["hypothesis_normalized"].to_dict()
    rows = []
    status_cache: dict[str, list[str]] = {}
    for _, row in consensus.iterrows():
        utt = str(row["utterance_id"])
        if utt not in status_cache:
            ref_norm = norm_by_utt.get(utt, ref_by_utt.get(utt, ""))
            status_cache[utt] = ref_unit_status(ref_norm, hyp_by_utt.get(utt, ""))
        statuses = status_cache[utt]
        unit_id = int(row["unit_id"])
        cat = statuses[unit_id] if 0 <= unit_id < len(statuses) else "missing"
        rows.append({**row.to_dict(), "baseline_status": cat, "baseline_correct": cat == "correct"})
    return pd.DataFrame(rows)


def select_pois(
    consensus_dev: pd.DataFrame,
    manifest_dev: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    wrong: int,
    correct: int,
    seed: int,
    exclude_utterance_ids: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select at most one English POI per utterance, speaker-balanced."""
    exclude = {str(x) for x in exclude_utterance_ids}
    attached = attach_baseline_unit_status(consensus_dev, manifest_dev, baseline_predictions)
    if attached.empty:
        return attached.copy(), attached.copy()
    attached = attached[(attached["language"] == "EN") & (~attached["utterance_id"].astype(str).isin(exclude))].copy()
    attached["_rank"] = [
        stable_row_key(seed, r["utterance_id"], r["unit_id"], r.get("surface", ""))
        for _, r in attached.iterrows()
    ]
    attached = attached.sort_values("_rank").drop_duplicates("utterance_id", keep="first")
    wrong_df = speaker_balanced_sample(
        attached[~attached["baseline_correct"]].drop(columns=["_rank"], errors="ignore"),
        wrong,
        seed=seed,
        speaker_col="speaker",
    )
    correct_df = speaker_balanced_sample(
        attached[attached["baseline_correct"]].drop(columns=["_rank"], errors="ignore"),
        correct,
        seed=seed + 1,
        speaker_col="speaker",
    )
    return wrong_df.reset_index(drop=True), correct_df.reset_index(drop=True)
