"""Task-specific alignment objects derived from raw normalized units.

Gate A protects span-local analyses.  Its production objects are language-run
outer edges and EN<->ZH switches, not every Mandarin character emitted by a
tokenizer.  This module preserves the raw rows and derives a second, traceable
representation for qualification:

``raw aligner rows -> normalized reference units -> language runs/switch edges``

An invalid internal unit is never rewritten.  It may, however, be internal to a
run whose two outer edges are supported by other positive-duration units.  The
raw invalidity remains in the run's counters and parent-row identifiers.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .autoevidence import INDEPENDENCE_CLASS, NATURAL_SPEECH_NOTE

TARGET_OBJECT_SCHEMA = "lss_alignment_target_objects_v1"
LANGUAGE_RUN_SCHEMA = "lss_alignment_language_runs_v1"
TARGET_LANGUAGES = ("EN", "ZH")


def _physical_valid(row: pd.Series) -> bool:
    """Whether the row itself provides a finite, positive-duration interval."""
    def number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    start = number(row.get("start_sec", float("nan")))
    end = number(row.get("end_sec", float("nan")))
    duration = number(row.get("audio_duration_sec", float("nan")))
    return bool(math.isfinite(start) and math.isfinite(end) and end > start
                and start >= 0.0
                and (not math.isfinite(duration) or end <= duration + 1e-3))


def _edge_supported(row: pd.Series) -> bool:
    """Whether a physical interval may support a run edge.

    Same-language tokenizer overlap is the one raw sequence failure whose
    interval remains meaningful after union. Other raw failures cannot support
    an outer edge merely because stale numeric timestamps happen to be present.
    """
    if not _physical_valid(row):
        return False
    raw_value = row.get("is_valid", True)
    raw_valid = False if pd.isna(raw_value) else bool(raw_value)
    return raw_valid or str(row.get("failure_code", "")) == "adjacent_overlap"


def raw_unit_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-family/language raw validity with independent denominators.

    ``invalid_duration`` is about one interval.  ``nonmonotonic`` is evaluated
    only between otherwise positive-duration intervals.  Consequently one
    zero-duration row cannot count as both failures by construction.
    """
    if candidates is None or not len(candidates):
        return pd.DataFrame()
    frame = candidates.copy()
    frame["_duration_valid"] = frame.apply(_physical_valid, axis=1)
    frame["_raw_valid"] = frame.get(
        "is_valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    frame["_nonmonotonic"] = False
    keys = ["aligner_family", "utterance_id"]
    if "aligner_variant" in frame:
        keys.insert(1, "aligner_variant")
    for _, group in frame.groupby(keys, sort=False):
        ordered = group.sort_values("reference_unit_index")
        eligible = ordered[ordered["_duration_valid"]]
        if len(eligible) < 2:
            continue
        starts = eligible["start_sec"].astype(float)
        bad = starts.diff().fillna(0.0) < 0.0
        frame.loc[eligible.index, "_nonmonotonic"] = bad.to_numpy()

    rows: list[dict[str, Any]] = []
    group_cols = ["aligner_family"]
    if "reference_language" in frame:
        group_cols.append("reference_language")
    for keys_value, group in frame.groupby(group_cols, sort=True, dropna=False):
        values = keys_value if isinstance(keys_value, tuple) else (keys_value,)
        duration_den = int(group["_duration_valid"].sum())
        rows.append({
            "aligner_family": str(values[0]),
            "language": str(values[1]) if len(values) > 1 else "ALL",
            "raw_units": int(len(group)),
            "raw_valid_units": int(group["_raw_valid"].sum()),
            "raw_invalid_units": int((~group["_raw_valid"]).sum()),
            "raw_invalid_rate": float((~group["_raw_valid"]).mean()),
            "invalid_duration_units": int((~group["_duration_valid"]).sum()),
            "invalid_duration_denominator": int(len(group)),
            "invalid_duration_rate": float((~group["_duration_valid"]).mean()),
            "nonmonotonic_units": int(group["_nonmonotonic"].sum()),
            "nonmonotonic_denominator": duration_den,
            "nonmonotonic_rate": (float(group["_nonmonotonic"].sum() / duration_den)
                                  if duration_den else float("nan")),
            "failure_reasons": json.dumps(
                group.loc[~group["_raw_valid"], "failure_code"].fillna("")
                .astype(str).value_counts().to_dict(), sort_keys=True),
        })
    return pd.DataFrame(rows)


def _reference_runs(group: pd.DataFrame) -> list[tuple[int, str, pd.DataFrame]]:
    ordered = group.sort_values("reference_unit_index")
    out: list[tuple[int, str, pd.DataFrame]] = []
    run_id = -1
    last_language: str | None = None
    members: list[int] = []
    for idx, row in ordered.iterrows():
        language = str(row.get("reference_language", ""))
        if language not in TARGET_LANGUAGES:
            continue
        if language != last_language:
            if members:
                out.append((run_id, str(last_language), ordered.loc[members]))
            run_id += 1
            members = []
            last_language = language
        members.append(idx)
    if members:
        out.append((run_id, str(last_language), ordered.loc[members]))
    return out


def aggregate_language_runs(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate same-language units without hiding raw invalidity.

    Adjacent positive-duration intervals from one language are unioned even when
    the raw sequence checker labelled the later interval ``adjacent_overlap``.
    Cross-language intervals are never unioned.  A zero/reversed unit is
    tolerated only strictly inside a run with valid units on both sides; if it
    touches an outer edge, that edge is unavailable.
    """
    if candidates is None or not len(candidates):
        return pd.DataFrame(), {
            "same_language_overlap_merged": 0,
            "same_language_overlap_unresolved": 0,
            "cross_language_overlap_invalid": 0,
            "switch_order_invalid": 0,
        }

    frame = candidates.copy().reset_index(drop=False).rename(columns={"index": "_parent_row"})
    group_cols = ["aligner_family", "aligner_variant", "utterance_id"]
    if "aligner_variant" not in frame:
        frame["aligner_variant"] = frame["aligner_family"].astype(str)
    rows: list[dict[str, Any]] = []
    counters = {
        "same_language_overlap_merged": 0,
        "same_language_overlap_unresolved": 0,
        "cross_language_overlap_invalid": 0,
        "switch_order_invalid": 0,
    }
    for (family, variant, utterance), group in frame.groupby(group_cols, sort=True):
        built: list[dict[str, Any]] = []
        for run_id, language, units in _reference_runs(group):
            units = units.sort_values("reference_unit_index")
            physical = units.apply(_physical_valid, axis=1)
            supported = units.apply(_edge_supported, axis=1)
            starts = units["start_sec"].astype(float)
            ends = units["end_sec"].astype(float)
            overlap_merged = 0
            unresolved = 0
            previous_start = previous_end = None
            for idx, ok in zip(units.index, supported):
                if not bool(ok):
                    continue
                start, end = float(starts.loc[idx]), float(ends.loc[idx])
                if previous_start is not None and start < previous_start:
                    unresolved += 1
                elif previous_end is not None and start < previous_end:
                    overlap_merged += 1
                previous_start, previous_end = start, max(end, previous_end or end)
            counters["same_language_overlap_merged"] += overlap_merged
            counters["same_language_overlap_unresolved"] += unresolved

            first_ok = bool(supported.iloc[0])
            last_ok = bool(supported.iloc[-1])
            valid_positions = np.flatnonzero(supported.to_numpy())
            internal_invalid = 0
            boundary_invalid = 0
            leading_invalid = 0
            trailing_invalid = 0
            if len(valid_positions):
                lo, hi = int(valid_positions[0]), int(valid_positions[-1])
                internal_invalid = int((~physical.iloc[lo:hi + 1]).sum())
                leading_invalid = int((~physical.iloc[:lo]).sum())
                trailing_invalid = int((~physical.iloc[hi + 1:]).sum())
                boundary_invalid = leading_invalid + trailing_invalid
            else:
                boundary_invalid = int((~physical).sum())
                leading_invalid = int((~physical).sum())

            start_sec = float(starts.iloc[0]) if first_ok else float("nan")
            end_sec = float(ends.iloc[-1]) if last_ok else float("nan")
            positive_duration = bool(first_ok and last_ok and end_sec > start_sec)
            within_order = bool(unresolved == 0)
            raw_valid = units.get(
                "is_valid", pd.Series(True, index=units.index)).fillna(False).astype(bool)
            built.append({
                "schema_version": LANGUAGE_RUN_SCHEMA,
                "aligner_family": str(family),
                "aligner_variant": str(variant),
                "utterance_id": str(utterance),
                "run_id": int(run_id),
                "language": language,
                "first_reference_unit_index": int(units["reference_unit_index"].min()),
                "last_reference_unit_index": int(units["reference_unit_index"].max()),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_edge_valid": first_ok,
                "end_edge_valid": last_ok,
                "positive_duration_valid": positive_duration,
                "within_language_order_valid": within_order,
                "incoming_switch_valid": True,
                "outgoing_switch_valid": True,
                "raw_units": int(len(units)),
                "raw_invalid_units": int((~raw_valid).sum()),
                "invalid_duration_units": int((~physical).sum()),
                "internal_invalid_duration_units": internal_invalid,
                "boundary_invalid_duration_units": boundary_invalid,
                "leading_invalid_duration_units": leading_invalid,
                "trailing_invalid_duration_units": trailing_invalid,
                "same_language_overlap_merged": int(overlap_merged),
                "same_language_overlap_unresolved": int(unresolved),
                "parent_candidate_rows": json.dumps(
                    [int(v) for v in units["_parent_row"].tolist()]),
                "parent_reference_unit_indices": json.dumps(
                    [int(v) for v in units["reference_unit_index"].tolist()]),
                "role": str(units["role"].iloc[0]) if "role" in units else "",
            })

        # Separate switch-adjacent failures from utterance-outer failures. Both
        # can invalidate task targets, but only the former explains EN<->ZH
        # switch evidence.
        for position, run in enumerate(built):
            run["switch_adjacent_invalid_duration_units"] = int(
                (run["leading_invalid_duration_units"] if position > 0 else 0)
                + (run["trailing_invalid_duration_units"]
                   if position < len(built) - 1 else 0))
            run["utterance_outer_invalid_duration_units"] = int(
                (run["leading_invalid_duration_units"] if position == 0 else 0)
                + (run["trailing_invalid_duration_units"]
                   if position == len(built) - 1 else 0))

        # A switch is valid only when both outer edges exist and are ordered.
        for left, right in zip(built, built[1:]):
            ordered = bool(left["end_edge_valid"] and right["start_edge_valid"]
                           and math.isfinite(float(left["end_sec"]))
                           and math.isfinite(float(right["start_sec"]))
                           and float(left["end_sec"]) <= float(right["start_sec"]) + 1e-3)
            if not ordered:
                left["outgoing_switch_valid"] = False
                right["incoming_switch_valid"] = False
                counters["switch_order_invalid"] += 1
                if (math.isfinite(float(left["end_sec"]))
                        and math.isfinite(float(right["start_sec"]))
                        and float(left["end_sec"]) > float(right["start_sec"]) + 1e-3):
                    counters["cross_language_overlap_invalid"] += 1
        rows.extend(built)

    runs = pd.DataFrame(rows)
    if len(runs):
        runs["target_valid"] = (
            runs["positive_duration_valid"].astype(bool)
            & runs["within_language_order_valid"].astype(bool)
            & runs["incoming_switch_valid"].astype(bool)
            & runs["outgoing_switch_valid"].astype(bool))
    return runs, counters


def target_objects(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return task target objects, language runs, and aggregation diagnostics."""
    runs, counters = aggregate_language_runs(candidates)
    if not len(runs):
        return pd.DataFrame(), runs, counters
    targets = runs.copy()
    targets["schema_version"] = TARGET_OBJECT_SCHEMA
    targets["target_id"] = targets.apply(
        lambda r: f"{r['utterance_id']}:{int(r['run_id'])}:{r['language']}", axis=1)
    targets["target_kind"] = np.where(
        targets["language"].astype(str) == "EN", "embedded_english_span", "language_run")
    return targets, runs, counters


def reference_target_universe(expected_units: pd.DataFrame) -> pd.DataFrame:
    """Language-run universe implied by frozen reference units, without timing."""
    if expected_units is None or not len(expected_units):
        return pd.DataFrame(columns=["utterance_id", "target_id", "language"])
    frame = expected_units.copy()
    language_col = "language" if "language" in frame else "reference_language"
    index_col = "reference_unit_index" if "reference_unit_index" in frame else "unit_id"
    rows: list[dict[str, Any]] = []
    for utterance, group in frame.groupby("utterance_id", sort=True):
        ordered = group.sort_values(index_col)
        last = None
        run_id = -1
        for _, unit in ordered.iterrows():
            language = str(unit[language_col])
            if language not in TARGET_LANGUAGES:
                continue
            if language != last:
                run_id += 1
                rows.append({"utterance_id": str(utterance), "run_id": run_id,
                             "target_id": f"{utterance}:{run_id}:{language}",
                             "language": language,
                             "role": str(unit.get("role", ""))})
                last = language
    return pd.DataFrame(rows)


def representative_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Choose one variant per family deterministically, never by its metrics."""
    if targets is None or not len(targets):
        return pd.DataFrame()
    keep = []
    for family, group in targets.groupby("aligner_family", sort=True):
        variant = sorted(group["aligner_variant"].astype(str).unique())[0]
        keep.append(group[group["aligner_variant"].astype(str) == variant])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def target_family_validity(targets: pd.DataFrame,
                           expected: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-family qualification metrics over target runs, with explicit counts."""
    selected = representative_targets(targets)
    if not len(selected) or "aligner_family" not in selected.columns:
        return pd.DataFrame()
    expected_n = int(len(expected.drop_duplicates("target_id"))) \
        if expected is not None and len(expected) else 0
    rows: list[dict[str, Any]] = []
    for family, group in selected.groupby("aligner_family", sort=True):
        valid = group["target_valid"].fillna(False).astype(bool)
        produced = int(len(group))
        denominator = expected_n or produced
        order_den = int(group["positive_duration_valid"].fillna(False).sum())
        order_bad = int((group["positive_duration_valid"].fillna(False).astype(bool)
                         & ~group["within_language_order_valid"].fillna(False).astype(bool)).sum())
        rows.append({
            "aligner_family": str(family),
            "independence_class": INDEPENDENCE_CLASS.get(str(family), str(family)),
            "aligner_variant": str(group["aligner_variant"].iloc[0]),
            "candidate_units": produced,
            "valid_units": int(valid.sum()),
            "valid_unit_coverage": float(valid.sum() / denominator) if denominator else float("nan"),
            "invalid_rate": float((~valid).sum() / produced) if produced else float("nan"),
            "nonmonotonic_rate": float(order_bad / order_den) if order_den else float("nan"),
            "target_objects_expected": denominator,
            "target_objects_produced": produced,
            "target_objects_valid": int(valid.sum()),
            "target_invalid_objects": int((~valid).sum()),
            "target_invalid_denominator": produced,
            "target_nonmonotonic_objects": order_bad,
            "target_nonmonotonic_denominator": order_den,
            "raw_invalid_units_retained": int(group["raw_invalid_units"].sum()),
            "internal_invalid_duration_units": int(group["internal_invalid_duration_units"].sum()),
            "same_language_overlap_merged": int(group["same_language_overlap_merged"].sum()),
        })
    return pd.DataFrame(rows)


def paired_target_overlap(targets: pd.DataFrame, family_a: str,
                          family_b: str) -> dict[str, Any]:
    """Eligible same-target overlap for a particular independent family pair."""
    selected = representative_targets(targets)
    a = selected[(selected["aligner_family"].astype(str) == str(family_a))
                 & selected["target_valid"].astype(bool)]
    b = selected[(selected["aligner_family"].astype(str) == str(family_b))
                 & selected["target_valid"].astype(bool)]
    a_ids, b_ids = set(a["target_id"].astype(str)), set(b["target_id"].astype(str))
    paired = a_ids & b_ids
    union = a_ids | b_ids
    return {
        "family_a": str(family_a), "family_b": str(family_b),
        "valid_a": len(a_ids), "valid_b": len(b_ids),
        "paired_target_objects": len(paired),
        "paired_rate": float(len(paired) / len(union)) if union else 0.0,
        "union_target_objects": len(union),
    }


def diagnostic_pair_overlaps(targets: pd.DataFrame) -> list[dict[str, Any]]:
    families = sorted(set(targets.get("aligner_family", pd.Series(dtype=str)).astype(str)))
    return [paired_target_overlap(targets, a, b) for a, b in combinations(families, 2)]


def select_pair(targets: pd.DataFrame, qualifying: Sequence[str],
                priority: Sequence[Sequence[str]], *, min_count: int,
                min_rate: float) -> dict[str, Any]:
    """Select the first preregistered qualifying pair with adequate overlap."""
    qualified = set(map(str, qualifying))
    diagnostics: list[dict[str, Any]] = []
    eligible_pairs = []
    for configured in priority:
        if len(configured) != 2:
            raise ValueError(f"aligner_pair_priority entries must be pairs, got {configured!r}")
        a, b = map(str, configured)
        if INDEPENDENCE_CLASS.get(a, a) == INDEPENDENCE_CLASS.get(b, b):
            raise ValueError(f"configured pair {a},{b} is not independent")
        if a not in qualified or b not in qualified:
            continue
        evidence = paired_target_overlap(targets, a, b)
        evidence["required_count"] = int(min_count)
        evidence["required_rate"] = float(min_rate)
        evidence["sufficient"] = bool(
            evidence["paired_target_objects"] >= int(min_count)
            and evidence["paired_rate"] >= float(min_rate))
        diagnostics.append(evidence)
        eligible_pairs.append([a, b])
        if evidence["sufficient"]:
            return {"available": True, "selected_pair": [a, b],
                    "selection_rule": "first_sufficient_pair_in_configured_priority",
                    "eligible_pairs_in_priority_order": eligible_pairs,
                    "pair_diagnostics": diagnostics}
    return {"available": False, "selected_pair": [],
            "reason": ("fewer_than_two_qualifying_families" if len(qualified) < 2
                       else "insufficient_paired_target_overlap"),
            "selection_rule": "first_sufficient_pair_in_configured_priority",
            "eligible_pairs_in_priority_order": eligible_pairs,
            "pair_diagnostics": diagnostics}


def cross_aligner_target_agreement(targets: pd.DataFrame,
                                   pair: Sequence[str]) -> dict[str, Any]:
    """Natural-speech disagreement on the selected paired target objects."""
    if len(pair) != 2:
        return {"n": 0, "measurement": "cross_aligner_disagreement",
                "reason": "no selected qualifying pair", "note": NATURAL_SPEECH_NOTE}
    selected = representative_targets(targets)
    a, b = map(str, pair)
    keys = ["target_id", "utterance_id", "language"]
    cols = keys + ["start_sec", "end_sec"]
    left = selected[(selected["aligner_family"].astype(str) == a)
                    & selected["target_valid"].astype(bool)][cols]
    right = selected[(selected["aligner_family"].astype(str) == b)
                     & selected["target_valid"].astype(bool)][cols]
    paired = left.merge(right, on=keys, suffixes=("_a", "_b"))
    if not len(paired):
        return {"n": 0, "family_a": a, "family_b": b,
                "measurement": "cross_aligner_disagreement",
                "reason": "no paired eligible target objects", "note": NATURAL_SPEECH_NOTE}
    paired["dstart_ms"] = (paired["start_sec_a"] - paired["start_sec_b"]) * 1000.0
    paired["dend_ms"] = (paired["end_sec_a"] - paired["end_sec_b"]) * 1000.0
    paired["abs_boundary_ms"] = paired[["dstart_ms", "dend_ms"]].abs().max(axis=1)

    def stats(group: pd.DataFrame) -> dict[str, Any]:
        return {"n": int(len(group)),
                "cross_aligner_start_disagreement_ms": float(group["dstart_ms"].abs().median()),
                "cross_aligner_end_disagreement_ms": float(group["dend_ms"].abs().median()),
                "cross_aligner_boundary_disagreement_ms": float(group["abs_boundary_ms"].median()),
                "cross_aligner_boundary_disagreement_p90_ms": float(group["abs_boundary_ms"].quantile(.9)),
                "within_100ms": float((group["abs_boundary_ms"] <= 100.0).mean())}
    by_language = {str(lang): stats(group) for lang, group in paired.groupby("language")}
    en = by_language.get("EN", {}).get("cross_aligner_boundary_disagreement_ms", float("nan"))
    zh = by_language.get("ZH", {}).get("cross_aligner_boundary_disagreement_ms", float("nan"))
    return {**stats(paired), "family_a": a, "family_b": b,
            "by_language": by_language,
            "en_zh_disagreement_diff_ms": (abs(float(en) - float(zh))
                                             if np.isfinite(en) and np.isfinite(zh)
                                             else float("nan")),
            "measurement": "cross_aligner_disagreement",
            "target_object_schema": TARGET_OBJECT_SCHEMA,
            "note": NATURAL_SPEECH_NOTE}
