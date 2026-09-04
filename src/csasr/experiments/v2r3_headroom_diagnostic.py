"""Headroom and whisper_dtw validity on the v2r3 dialogue-atomic population.

Read-only.  It reads the v2r3 candidate table and the v2r3 baseline POI tables,
restricted to the two development roles, and writes counts to a diagnostic root
with `development_only_diagnostic` taint.  It selects no convention, sets no
eligibility floor, evaluates no gate, and publishes no production artifact.

The eligibility rule is applied exactly as stated, with no confidence condition:
an embedded-English lexical unit inside a language run whose aligned span
duration clears the floor, whose preceding run is matrix language, which is not
utterance-final, and whose reference-unit indices are contiguous.

Dialogue counts are the headline throughout.  `dialogue_id` is the independence
unit on this corpus, so a count of units means little without the number of
independent clusters those units came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.align import annotation_pack as pack_mod
from ..lss.align import conventions as conv
from ..utils.config import load_config

STAGE = "v2r3_headroom_diagnostic"
TAINT = conv.TAINT
DEVELOPMENT_ROLES = conv.DEVELOPMENT_ROLES

#: Duration floors, in milliseconds.  These are the tiers named in the runbook;
#: reporting all three is not the same as choosing one.
FLOORS_MS = (400.0, 300.0, 200.0)

#: The family whose spans define "aligned span duration".  `existing_ctc` is the
#: provisional family the annotation pack and Run A already used; it is recorded
#: rather than chosen here, and it is the only family with a 0% invalid rate.
SPAN_FAMILY = "existing_ctc"

ERROR_CATEGORIES = ("deletion", "wrong_language_substitution",
                    "same_language_substitution", "boundary_error",
                    "phonetic_transliteration_or_script", "other")


def eligible_spans(candidates: pd.DataFrame, *, family: str = SPAN_FAMILY
                   ) -> pd.DataFrame:
    """One row per embedded-English language run, with each eligibility flag.

    Flags are kept separate rather than pre-combined so the funnel can be read
    at every stage, and so no floor is baked into the table.
    """
    rows: list[dict[str, Any]] = []
    frame = candidates[candidates["aligner_family"].astype(str) == str(family)]
    for (role, utterance), group in frame.groupby(["role", "utterance_id"], sort=False):
        runs = pack_mod.language_runs(group)
        for position, run in enumerate(runs):
            if str(run["language"]) != "EN":
                continue
            units = pd.DataFrame(run["rows"])
            indices = sorted(int(v) for v in units["reference_unit_index"])
            starts = pd.to_numeric(units["start_sec"], errors="coerce")
            ends = pd.to_numeric(units["end_sec"], errors="coerce")
            duration = float(ends.max() - starts.min()) if len(units) else float("nan")
            previous = runs[position - 1]["language"] if position else None
            first = units.iloc[0]
            rows.append({
                "role": str(role),
                "utterance_id": str(utterance),
                "conversation_id": str(first.get("conversation_id", "")),
                "speaker_id": str(first.get("speaker", first.get("speaker_id", ""))),
                "run_position": int(position),
                "n_units": int(len(units)),
                "unit_indices": indices,
                "span_duration_sec": duration,
                "span_duration_ms": duration * 1000.0,
                "preceded_by_matrix": bool(previous == "ZH"),
                "utterance_final": bool(position == len(runs) - 1),
                "contiguous_indices": bool(
                    len(indices) == (indices[-1] - indices[0] + 1)) if indices else False,
                "finite_span": bool(pd.notna(duration)),
            })
    return pd.DataFrame(rows)


def structurally_eligible(spans: pd.DataFrame) -> pd.Series:
    """Every condition except the duration floor."""
    return (spans["preceded_by_matrix"] & ~spans["utterance_final"]
            & spans["contiguous_indices"] & spans["finite_span"])


def _clusters(frame: pd.DataFrame, dialogue_of: dict[str, str]) -> dict[str, int]:
    dialogues = {dialogue_of.get(c, "") for c in frame.get("conversation_id", [])}
    return {"dialogues": len({d for d in dialogues if d}),
            "speakers": int(frame["conversation_id"].nunique()) if len(frame) else 0}


def headroom(spans: pd.DataFrame, units: pd.DataFrame, *,
             dialogue_of: dict[str, str],
             floors_ms: Sequence[float] = FLOORS_MS) -> dict[str, Any]:
    """Counts per role per floor: spans, units, and the derived error subsets."""
    out: dict[str, Any] = {}
    eligible = spans[structurally_eligible(spans)]
    for role in sorted(spans["role"].unique()):
        per_floor: dict[str, Any] = {}
        role_spans = eligible[eligible["role"] == role]
        role_units = units[units["role"] == role]
        for floor in floors_ms:
            kept = role_spans[role_spans["span_duration_ms"] >= float(floor)]
            keys = {(u, i) for u, idx in zip(kept["utterance_id"], kept["unit_indices"])
                    for i in idx}
            inside = role_units[
                [(u, i) in keys for u, i in zip(role_units["utterance_id"],
                                                role_units["reference_unit_index"])]]
            errors = inside[~inside["correct"].astype(bool)]
            by_category = {}
            for category in ERROR_CATEGORIES:
                sub = errors[errors["category"].astype(str) == category]
                by_category[category] = {"units": int(len(sub)), **_clusters(sub, dialogue_of)}
            substitution = errors[errors["category"].astype(str) != "deletion"]
            wrong_language = errors[errors["category"].astype(str)
                                    == "wrong_language_substitution"]
            correct = inside[inside["correct"].astype(bool)]
            per_floor[f">={int(floor)}ms"] = {
                "structural_spans": int(len(kept)),
                "lexical_units": int(len(inside)),
                "utterances": int(kept["utterance_id"].nunique()),
                **_clusters(kept, dialogue_of),
                "baseline_error_units": int(len(errors)),
                "baseline_correct_units": int(len(correct)),
                "baseline_correct_clusters": _clusters(correct, dialogue_of),
                "by_error_category": by_category,
                "derived_subsets": {
                    "all_baseline_error": {"units": int(len(errors)),
                                           **_clusters(errors, dialogue_of)},
                    "substitution_only_excludes_deletion": {
                        "units": int(len(substitution)),
                        **_clusters(substitution, dialogue_of)},
                    "wrong_language_substitution_only": {
                        "units": int(len(wrong_language)),
                        **_clusters(wrong_language, dialogue_of)},
                },
            }
        out[role] = per_floor
    return out


def funnel(candidates: pd.DataFrame, spans: pd.DataFrame, *,
           family: str = SPAN_FAMILY) -> dict[str, Any]:
    """Sampled utterances -> reference units -> language runs -> eligible spans."""
    frame = candidates[candidates["aligner_family"].astype(str) == str(family)]
    eligible = spans[structurally_eligible(spans)]
    out: dict[str, Any] = {}
    for role in sorted(frame["role"].astype(str).unique()):
        rows = frame[frame["role"].astype(str) == role]
        sampled = rows["utterance_id"].nunique()
        with_units = rows.groupby("utterance_id").size().gt(0).sum()
        runs_per_utterance = {
            u: len(pack_mod.language_runs(g))
            for u, g in rows.groupby("utterance_id", sort=False)}
        two_runs = sum(1 for v in runs_per_utterance.values() if v >= 2)
        role_spans = eligible[eligible["role"] == role]
        out[role] = {
            "sampled_utterances": int(sampled),
            "with_at_least_one_reference_unit": int(with_units),
            "with_at_least_two_language_runs": int(two_runs),
            "with_at_least_one_eligible_embedded_span_any_floor":
                int(role_spans["utterance_id"].nunique()),
        }
    return out


def whisper_invalid_profile(candidates: pd.DataFrame, units: pd.DataFrame,
                            spans: pd.DataFrame, *,
                            family: str = "whisper_dtw") -> dict[str, Any]:
    """What the invalid units are, and what they cost the conservative subset."""
    frame = candidates[candidates["aligner_family"].astype(str) == str(family)].copy()
    frame["invalid"] = ~frame["is_valid"].astype(bool)
    frame["duration_sec"] = (pd.to_numeric(frame["end_sec"], errors="coerce")
                             - pd.to_numeric(frame["start_sec"], errors="coerce"))

    deciles = []
    finite = frame[frame["duration_sec"].notna()].copy()
    if len(finite):
        finite["decile"] = pd.qcut(finite["duration_sec"], 10,
                                   labels=False, duplicates="drop")
        for decile, group in finite.groupby("decile"):
            deciles.append({
                "decile": int(decile) + 1,
                "duration_min_ms": round(float(group["duration_sec"].min() * 1000), 1),
                "duration_max_ms": round(float(group["duration_sec"].max() * 1000), 1),
                "units": int(len(group)),
                "invalid": int(group["invalid"].sum()),
                "invalid_rate": round(float(group["invalid"].mean()), 6),
            })

    by_language = {
        str(language): {"units": int(len(group)), "invalid": int(group["invalid"].sum()),
                        "invalid_rate": round(float(group["invalid"].mean()), 6)}
        for language, group in frame.groupby("reference_language")}

    invalid_keys = {(u, int(i)) for u, i in
                    zip(frame.loc[frame["invalid"], "utterance_id"],
                        frame.loc[frame["invalid"], "reference_unit_index"])}
    poi_hit = units[[(u, int(i)) in invalid_keys for u, i in
                     zip(units["utterance_id"], units["reference_unit_index"])]]
    by_category = (poi_hit["category"].value_counts().to_dict() if len(poi_hit) else {})

    conservative = spans[structurally_eligible(spans)
                         & (spans["span_duration_ms"] >= 400.0)]
    keys = {(u, i) for u, idx in zip(conservative["utterance_id"],
                                     conservative["unit_indices"]) for i in idx}
    lost_units = len(keys & invalid_keys)
    damaged_spans = sum(
        1 for u, idx in zip(conservative["utterance_id"], conservative["unit_indices"])
        if any((u, i) in invalid_keys for i in idx))
    return {
        "family": family,
        "invalid_rate_overall": round(float(frame["invalid"].mean()), 6),
        "failure_codes": {str(k): int(v) for k, v in
                          frame.loc[frame["invalid"], "failure_code"]
                          .value_counts(dropna=False).items()},
        "invalid_condition": (
            "csasr.nat5h.schema.py:257-262 marks a unit invalid with "
            "failure_code='adjacent_overlap' when its start_sec precedes the "
            "previous canonical unit's end_sec by more than overlap_epsilon_sec. "
            "It is a condition between two adjacent units, not a property of one "
            "interval, which is why invalid_duration and nonmonotonic are both "
            "zero: those describe single intervals and ordering of starts."),
        "edge_support_note": (
            "csasr.lss.align.target_objects._edge_supported admits an "
            "adjacent_overlap unit as a run edge, because same-language "
            "tokenizer overlap stays meaningful after union."),
        "by_duration_decile": deciles,
        "by_language_class": by_language,
        "by_poi_error_category": {str(k): int(v) for k, v in by_category.items()},
        "poi_units_affected": int(len(poi_hit)),
        "conservative_subset_400ms": {
            "spans": int(len(conservative)),
            "lexical_units": int(len(keys)),
            "units_lost_to_invalid": int(lost_units),
            "spans_touching_an_invalid_unit": int(damaged_spans),
            "fraction_of_units_lost": round(lost_units / len(keys), 6) if keys else 0.0,
        },
    }


def resolve_output(output: str | Path) -> Path:
    target = Path(output).resolve()
    if "artifacts_lss" in str(target) or "/freeze" in str(target):
        raise SystemExit(f"diagnostic output must not touch production: {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--poi-root", required=True)
    parser.add_argument("--diagnostic-output", required=True)
    parser.add_argument("--span-family", default=SPAN_FAMILY)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target = resolve_output(args.diagnostic_output)
    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    candidates, verdict = conv.read_development_candidates(candidate_path)
    if not verdict["ok"]:
        raise SystemExit(f"cached candidates are not authentic: {verdict}")
    conv.assert_development_only(candidates)

    poi_root = Path(args.poi_root)
    frames = []
    for role in DEVELOPMENT_ROLES:
        path = poi_root / f"poi_{role}.parquet"
        if not path.is_file():
            raise SystemExit(f"POI table missing: {path}")
        frames.append(pd.read_parquet(path))
    units = pd.concat(frames, ignore_index=True)
    units = units.rename(columns={"poi_index": "reference_unit_index"})

    dialogue_of = dict(zip(units["conversation_id"].astype(str),
                           units["dialogue_id"].astype(str)))
    spans = eligible_spans(candidates, family=args.span_family)
    payload = {
        "diagnostic_only": True,
        "taint_reasons": [TAINT],
        "selects_no_convention": True,
        "sets_no_eligibility_floor": True,
        "evaluates_no_gate": True,
        "declares_no_readiness": True,
        "roles": list(DEVELOPMENT_ROLES),
        "span_family": args.span_family,
        "span_family_note": (
            "the provisional family the annotation pack and Run A already used; "
            "recorded here, not chosen here"),
        "eligibility_rule": [
            "embedded-English lexical unit",
            "aligned span duration >= floor",
            "matrix-language preceding context",
            "not utterance-final",
            "contiguous reference-unit indices",
            "no confidence condition",
        ],
        "source_candidate_path": str(candidate_path),
        "source_candidate_manifest": verdict.get("manifest") or {},
        "poi_root": str(poi_root),
        "floors_ms": list(FLOORS_MS),
        "funnel": funnel(candidates, spans, family=args.span_family),
        "headroom": headroom(spans, units, dialogue_of=dialogue_of),
        "whisper_dtw_invalid": whisper_invalid_profile(candidates, units, spans),
        "span_totals": {
            role: {"embedded_english_runs": int((spans["role"] == role).sum()),
                   "structurally_eligible": int(
                       (structurally_eligible(spans) & (spans["role"] == role)).sum())}
            for role in sorted(spans["role"].unique())},
    }

    target.mkdir(parents=True, exist_ok=True)
    spans_path = target / "eligible_spans.parquet"
    spans.assign(unit_indices=spans["unit_indices"].map(json.dumps)).to_parquet(
        spans_path, index=False)
    report = target / "headroom_report.json"
    report.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    for artifact in (spans_path, report):
        manifest_mod.publish(artifact, stage=STAGE, cfg=cfg,
                             parents=[parent] if parent else (),
                             taint_reasons=(TAINT,),
                             schema="lss_v2r3_headroom_diagnostic_v1")
    print(json.dumps({"state": "completed", "output": str(target),
                      "funnel": payload["funnel"],
                      "span_totals": payload["span_totals"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
