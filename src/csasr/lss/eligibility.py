"""What counts as an evaluated unit.

P0 measured PIER over *every* English content unit, so on `dev_select` 17,465 of
its 23,726 "English POIs" came from monolingual `<EN>` utterances -- 73% of the
endpoint was not code-switching at all. This module defines the endpoint the
paper actually claims: an **embedded** English unit, meaning an English content
unit in an utterance whose reference also contains Mandarin.

Restricting to embedded English drops dev_select from 23,726 to 6,215 units
(2,745 baseline errors), which still clears every power target in the proposal.

Unit indices come from `csasr.nat5h.units.build_reference_units`, the same
indexing the aligners and the consensus tables use (`reference_unit_index`).
`csasr.evaluation.pier` indexes the output of the same `segment_units` call, so
`poi_index == unit_id`. That invariant is asserted, never assumed.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ..data.language_tags import EN, ZH
from ..evaluation.pier import reference_pois
from ..nat5h.units import build_reference_units

UNIT_COLUMNS = (
    "utterance_id", "unit_id", "surface", "normalized_surface", "language",
    "normalized_char_start", "normalized_char_end", "is_content",
    "utterance_contains_code_switch", "is_embedded_english",
    "n_units_in_utterance", "n_en_units_in_utterance", "unit_position",
)


def utterance_units(utterance_id: str, transcript_raw: str | None) -> pd.DataFrame:
    """Reference units of one utterance, with embedded-English eligibility."""
    units = build_reference_units(utterance_id, transcript_raw)
    languages = [u.language for u in units]
    has_zh = ZH in languages
    has_en = EN in languages
    n_en = sum(1 for lang in languages if lang == EN)
    rows = []
    for u in units:
        content = u.language in (EN, ZH)
        rows.append({
            "utterance_id": u.utterance_id,
            "unit_id": int(u.unit_id),
            "surface": u.surface,
            "normalized_surface": u.normalized_surface,
            "language": u.language,
            "normalized_char_start": int(u.normalized_char_start),
            "normalized_char_end": int(u.normalized_char_end),
            "is_content": bool(content),
            "utterance_contains_code_switch": bool(has_en and has_zh),
            "is_embedded_english": bool(u.language == EN and has_zh),
            "n_units_in_utterance": len(units),
            "n_en_units_in_utterance": int(n_en),
            "unit_position": int(u.unit_id),
        })
    return pd.DataFrame(rows, columns=list(UNIT_COLUMNS))


def unit_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Reference units for every utterance in ``manifest``."""
    frames = [utterance_units(r["utterance_id"], r["transcript_raw"])
              for _, r in manifest.iterrows()]
    if not frames:
        return pd.DataFrame(columns=list(UNIT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def embedded_english_units(manifest: pd.DataFrame) -> pd.DataFrame:
    """The primary endpoint's unit set."""
    units = unit_table(manifest)
    if not len(units):
        return units
    return units[units["is_embedded_english"]].reset_index(drop=True)


def conversation_unit_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation covariates used to balance the data roles.

    Counting embedded-English units here (rather than utterances) is what makes
    the balance criterion meaningful: two conversations can have equal hours and
    very unequal amounts of the thing the paper measures.
    """
    units = unit_table(manifest)
    per_utt = (units.groupby("utterance_id")
               .agg(n_units=("unit_id", "size"),
                    n_en_units=("language", lambda s: int((s == EN).sum())),
                    n_zh_units=("language", lambda s: int((s == ZH).sum())),
                    n_embedded_en_units=("is_embedded_english", "sum"))
               .reset_index())
    merged = manifest.merge(per_utt, on="utterance_id", how="left")
    for col in ("n_units", "n_en_units", "n_zh_units", "n_embedded_en_units"):
        merged[col] = merged[col].fillna(0).astype(int)
    grouped = (merged.groupby("conversation_id")
               .agg(utterances=("utterance_id", "size"),
                    hours=("duration_sec", lambda s: float(s.sum()) / 3600.0),
                    mean_duration_sec=("duration_sec", "mean"),
                    cs_utterances=("contains_code_switch", "sum"),
                    embedded_en_units=("n_embedded_en_units", "sum"),
                    en_units=("n_en_units", "sum"),
                    zh_units=("n_zh_units", "sum"))
               .reset_index())
    grouped["cs_utterances"] = grouped["cs_utterances"].astype(int)
    grouped["cs_rate"] = grouped["cs_utterances"] / grouped["utterances"].clip(lower=1)
    for col in ("gender", "device", "region"):
        if col in manifest.columns:
            mode = (manifest.groupby("conversation_id")[col]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else ""))
            grouped[col] = grouped["conversation_id"].map(mode).fillna("")
    if "topic" in manifest.columns:
        topics = (manifest.groupby("conversation_id")["topic"]
                  .agg(lambda s: sorted({str(v) for v in s if str(v)})))
        grouped["topics"] = grouped["conversation_id"].map(topics)
    return grouped


def poi_index_consistency(manifest: pd.DataFrame, *, sample: int | None = None,
                          seed: int = 0) -> dict:
    """Check `poi_index == unit_id` for English units on real transcripts.

    `evaluation.pier.reference_pois` and `nat5h.units.build_reference_units`
    index independently. Everything that joins a POI table to a span table
    depends on them agreeing, so this is measured rather than trusted.
    """
    df = manifest
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    checked = agree = 0
    mismatches: list[dict] = []
    for _, row in df.iterrows():
        normalized = row.get("transcript_normalized")
        if not isinstance(normalized, str) or not normalized:
            continue
        pois = {idx for idx, _ in reference_pois(normalized)}
        units = utterance_units(row["utterance_id"], row["transcript_raw"])
        english = set(units.loc[units["language"] == EN, "unit_id"].astype(int))
        checked += 1
        if pois == english:
            agree += 1
        elif len(mismatches) < 20:
            mismatches.append({
                "utterance_id": row["utterance_id"],
                "poi_only": sorted(pois - english),
                "unit_only": sorted(english - pois),
            })
    return {
        "utterances_checked": int(checked),
        "utterances_agreeing": int(agree),
        "agreement_rate": (agree / checked) if checked else float("nan"),
        "mismatch_examples": mismatches,
    }


def summarize(units: pd.DataFrame) -> dict:
    """Counts used in reports and gate evidence."""
    if not len(units):
        return {"units": 0, "embedded_english": 0, "en_units": 0, "zh_units": 0,
                "cs_utterances": 0, "utterances": 0}
    return {
        "units": int(len(units)),
        "utterances": int(units["utterance_id"].nunique()),
        "en_units": int((units["language"] == EN).sum()),
        "zh_units": int((units["language"] == ZH).sum()),
        "embedded_english": int(units["is_embedded_english"].sum()),
        "cs_utterances": int(units.loc[units["utterance_contains_code_switch"],
                                       "utterance_id"].nunique()),
    }
