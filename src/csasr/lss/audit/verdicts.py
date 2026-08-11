"""Ingest, unblind, adjudicate and score human verdicts.

Raw verdict files are copied to an immutable, content-hashed location before
anything reads them, and adjudication appends rows rather than editing them, so
the record of what an annotator actually said survives every later decision.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...utils.hashing import sha256_file
from .pack import ANNOTATOR_COLUMNS, VERDICTS


def ingest(raw_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copy every raw verdict file to an immutable path and concatenate them."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    immutable = out_dir / "verdicts_immutable"
    immutable.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []

    for path in sorted(raw_dir.glob("*.csv")) if raw_dir.is_dir() else []:
        digest = sha256_file(path)
        target = immutable / f"{path.stem}.{digest[:16]}.csv"
        if not target.exists():
            shutil.copy2(path, target)
        frame = pd.read_csv(path)
        missing = [c for c in ("audit_item_id", "verdict") if c not in frame.columns]
        if missing:
            files.append({"path": str(path), "sha256": digest, "rows": 0,
                          "error": f"missing columns {missing}"})
            continue
        if "annotator_id" not in frame.columns:
            frame["annotator_id"] = path.stem
        for column in ANNOTATOR_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frames.append(frame[list(ANNOTATOR_COLUMNS)])
        files.append({"path": str(path), "sha256": digest, "rows": int(len(frame)),
                      "immutable_copy": str(target)})

    verdicts = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=list(ANNOTATOR_COLUMNS))
    invalid = verdicts[~verdicts["verdict"].isin(VERDICTS)] if len(verdicts) else verdicts
    return verdicts, {"files": files, "rows": int(len(verdicts)),
                      "annotators": int(verdicts["annotator_id"].nunique())
                      if len(verdicts) else 0,
                      "invalid_verdict_rows": int(len(invalid))}


def unblind(verdicts: pd.DataFrame, key_path: Path) -> pd.DataFrame:
    """Join verdicts to the blinding key. Unknown item ids are rejected."""
    key = pd.DataFrame(json.loads(Path(key_path).read_text(encoding="utf-8")))
    if not len(verdicts):
        return verdicts
    unknown = set(verdicts["audit_item_id"]) - set(key["audit_item_id"])
    if unknown:
        raise ValueError(f"verdicts reference unknown audit items: {sorted(unknown)[:5]}")
    return verdicts.merge(key, on="audit_item_id", how="left")


def boundary_error(unblinded: pd.DataFrame) -> dict[str, Any]:
    """Absolute boundary error of the proposed spans against corrected times."""
    if not len(unblinded):
        return {"n": 0}
    frame = unblinded[unblinded["corrected_start_sec"].notna()
                      & unblinded["corrected_end_sec"].notna()].copy()
    if not len(frame):
        return {"n": 0, "note": "no corrected times supplied"}

    # the annotator saw a possibly displaced boundary, at clip-relative times
    proposed_start = frame["true_start_sec"] - frame["clip_offset_sec"] \
        + frame["applied_offset_ms"].fillna(0.0) / 1000.0
    proposed_end = frame["true_end_sec"] - frame["clip_offset_sec"] \
        + frame["applied_offset_ms"].fillna(0.0) / 1000.0
    frame["start_error_ms"] = (proposed_start - frame["corrected_start_sec"]) * 1000.0
    frame["end_error_ms"] = (proposed_end - frame["corrected_end_sec"]) * 1000.0
    frame["abs_error_ms"] = frame[["start_error_ms", "end_error_ms"]].abs().max(axis=1)

    honest = frame[~frame["is_decoy"].fillna(False)]
    scored = honest if len(honest) else frame

    def stats(subset: pd.DataFrame) -> dict[str, float]:
        if not len(subset):
            return {"n": 0}
        return {
            "n": int(len(subset)),
            "median_abs_error_ms": float(subset["abs_error_ms"].median()),
            "p90_abs_error_ms": float(subset["abs_error_ms"].quantile(0.9)),
            "median_signed_start_ms": float(subset["start_error_ms"].median()),
        }

    per_language = {str(lang): stats(group)
                    for lang, group in scored.groupby("language")}
    en = per_language.get("EN", {}).get("median_abs_error_ms", float("nan"))
    zh = per_language.get("ZH", {}).get("median_abs_error_ms", float("nan"))
    return {
        **stats(scored),
        "by_language": per_language,
        "en_zh_median_diff_ms": abs(float(en) - float(zh))
        if np.isfinite(en) and np.isfinite(zh) else float("nan"),
        "units_with_corrected_times": int(len(frame)),
        "decoys_excluded": int(len(frame) - len(scored)),
    }


def agreement(unblinded: pd.DataFrame) -> dict[str, Any]:
    """Decoy detection, duplicate consistency and inter-annotator agreement."""
    if not len(unblinded):
        return {"n": 0}
    decoys = unblinded[unblinded["is_decoy"].fillna(False)]
    detected = (decoys["verdict"] != "usable").mean() if len(decoys) else float("nan")

    duplicates = unblinded[unblinded["is_duplicate"].fillna(False)]
    consistency = float("nan")
    if len(duplicates):
        pairs = unblinded.merge(
            duplicates[["utterance_id", "unit_id", "annotator_id", "verdict"]],
            on=["utterance_id", "unit_id", "annotator_id"], suffixes=("", "_dup"))
        if len(pairs):
            consistency = float((pairs["verdict"] == pairs["verdict_dup"]).mean())

    alpha = float("nan")
    overlap = (unblinded.groupby(["utterance_id", "unit_id"])["annotator_id"]
               .nunique().rename("n_annotators").reset_index())
    shared = overlap[overlap["n_annotators"] > 1]
    if len(shared):
        joined = unblinded.merge(shared[["utterance_id", "unit_id"]],
                                 on=["utterance_id", "unit_id"])
        observed = []
        for _, group in joined.groupby(["utterance_id", "unit_id"]):
            values = group["verdict"].tolist()
            pairs = [(a, b) for i, a in enumerate(values) for b in values[i + 1:]]
            observed.extend(1.0 if a == b else 0.0 for a, b in pairs)
        if observed:
            po = float(np.mean(observed))
            counts = joined["verdict"].value_counts(normalize=True)
            pe = float((counts ** 2).sum())
            alpha = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {
        "n": int(len(unblinded)),
        "decoys": int(len(decoys)),
        "decoy_detection_rate": float(detected) if np.isfinite(detected) else float("nan"),
        "duplicate_consistency": consistency,
        "overlapping_units": int(len(shared)),
        "inter_annotator_alpha": alpha,
    }


def adjudicate(unblinded: pd.DataFrame, *, max_gap_ms: float = 100.0) -> pd.DataFrame:
    """Flag units needing a third pass. Never mutates the raw rows."""
    if not len(unblinded):
        return unblinded
    rows = []
    for (utt, unit), group in unblinded.groupby(["utterance_id", "unit_id"]):
        verdicts = set(group["verdict"])
        times = group["corrected_start_sec"].dropna()
        spread_ms = float((times.max() - times.min()) * 1000.0) if len(times) > 1 else 0.0
        rows.append({
            "utterance_id": utt, "unit_id": int(unit),
            "n_annotators": int(group["annotator_id"].nunique()),
            "verdicts": "|".join(sorted(verdicts)),
            "verdict_disagreement": bool(len(verdicts) > 1),
            "start_spread_ms": spread_ms,
            "needs_adjudication": bool(len(verdicts) > 1 or spread_ms > max_gap_ms),
        })
    return pd.DataFrame(rows)


def usable_fraction(unblinded: pd.DataFrame) -> float:
    if not len(unblinded):
        return float("nan")
    honest = unblinded[~unblinded["is_decoy"].fillna(False)]
    scored = honest if len(honest) else unblinded
    return float((scored["verdict"] == "usable").mean())
