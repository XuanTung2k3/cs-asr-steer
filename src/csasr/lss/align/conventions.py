"""Named CTC blank-run boundary conventions as a derived view over candidates.

CTC token spans exclude blank states, so between two adjacent language runs
there is a stretch of audio no unit owns.  A *convention* is a rule for who owns
that blank run.  With the blank run spanning ``[b_start, b_end]`` -- the raw end
of the preceding run and the raw start of the following one -- four are
implemented:

``blank_excluded``      leave the blank unowned (what the production path does)
``blank_to_preceding``  the preceding run's end extends to ``b_end``
``blank_to_following``  the following run's start moves back to ``b_start``
``blank_midpoint``      both edges move to ``(b_start + b_end) / 2``

:func:`convention_boundary` is the whole rule, and it is a pure function of the
blank run: it cannot see a measured disagreement, the other aligner's spans, or
any fitted quantity, because none of them is an argument.  That distinction
matters -- an offset subtraction fitted to reduce error would be a correction,
and this module must not contain one.

The switch universe comes from :mod:`annotation_pack`, so the conventions and
the human annotation pack enumerate the same boundaries.

Nothing here selects a convention.  It reports each one's numbers and stops.

Diagnostic-only: the spans returned are a derived view.  Raw candidate rows are
never mutated, and callers must not write the view into the production
artifacts root.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import annotation_pack as pack_mod
from . import target_objects as target_mod

CONVENTIONS: tuple[str, ...] = (
    "blank_excluded",
    "blank_to_preceding",
    "blank_to_following",
    "blank_midpoint",
)

#: Only these two roles may inform a development decision.  D-dev-confirm,
#: D-test and any gate generation are held out and must never reach this module.
DEVELOPMENT_ROLES: tuple[str, ...] = ("D-construct", "D-dev-select")

CONVENTION_FAMILY = "existing_ctc"
REFERENCE_FAMILY = "whisper_dtw"
TAINT = "development_only_diagnostic"
TRANSITION_SCHEMA = "lss_ctc_switch_transition_inventory_v1"
CONVENTION_DISAGREEMENT_SCHEMA = "lss_ctc_convention_disagreement_v1"

#: The languages that form a target object.  Taken from the production module so
#: a language the aggregation ignores cannot become a run edge here.
TARGET_LANGUAGES: tuple[str, ...] = tuple(target_mod.TARGET_LANGUAGES)

#: Reporting tolerances.  Both fractions are required output; neither is a gate
#: threshold and neither replaces one.
TOLERANCES_MS: tuple[float, ...] = (100.0, 200.0)

# --- how a transition's blank run is classified ---------------------------
#: Exactly one blank run separates the two runs: a convention can act.
SINGLE_BLANK_RUN = "single_blank_run"
#: The runs touch.  All four conventions coincide, so acting is a no-op.
ZERO_LENGTH_BLANK_RUN = "zero_length_blank_run"
#: The runs overlap, so no blank run exists to reassign.
NO_BLANK_RUN = "no_blank_run_units_overlap"
#: One or more runs in a third language sit between them, so the interval holds
#: several blank runs plus audio a third unit owns.
MULTIPLE_BLANK_RUNS = "multiple_blank_runs"
#: The reference-unit indices skip, so units absent from the table own part of
#: the interval.
NON_CONTIGUOUS = "non_contiguous_reference_unit_index"
LEFT_UNUSABLE = "left_unit_cannot_support_an_edge"
RIGHT_UNUSABLE = "right_unit_cannot_support_an_edge"
#: Backstop: an edge that is not a finite number.  The two unusable checks above
#: already reject these, so this is expected to stay at zero.
NON_FINITE = "non_finite_boundary"

CLASSIFICATIONS: tuple[str, ...] = (
    SINGLE_BLANK_RUN, ZERO_LENGTH_BLANK_RUN, NO_BLANK_RUN, MULTIPLE_BLANK_RUNS,
    NON_CONTIGUOUS, LEFT_UNUSABLE, RIGHT_UNUSABLE, NON_FINITE)

#: Only these carry a blank run a convention can reassign.  Every other
#: classification keeps the boundary ``blank_excluded`` already produced, and is
#: counted rather than dropped.
ACTIONABLE: tuple[str, ...] = (SINGLE_BLANK_RUN,)

#: Two consecutive target-language runs of the *same* language, separated only
#: by a third-language run.  The aggregation merges them into one target object,
#: so there is no boundary here at all.
MERGED_BY_AGGREGATION = "same_language_target_runs_merged"
#: A run edge with no adjacent language run: the first target run's start and
#: the last target run's end in each utterance.  No blank run exists to
#: reassign, so no convention can move it.
UTTERANCE_BOUNDARY_EDGE = "transition_at_utterance_boundary"

#: Denominators rather than edge cases; reported beside them so the counts can
#: be reconciled without a second lookup.
TRANSITION_TOTALS: tuple[str, ...] = ("transitions_total", "transitions_actionable",
                                      "target_language_runs")


class SplitDisciplineError(RuntimeError):
    """Raised when held-out data reaches a development-only computation."""


# --------------------------------------------------------------------------
# split discipline
# --------------------------------------------------------------------------

def assert_development_only(frame: pd.DataFrame) -> None:
    """Fail closed if anything outside the permitted development roles is present.

    A missing ``role`` column is an error, not a licence to use every row: the
    whole point is that this computation cannot see D-dev-confirm, D-test or a
    gate generation, and an unlabelled frame cannot demonstrate that.
    """
    if "role" not in frame.columns:
        raise SplitDisciplineError(
            "candidate rows carry no role column, so development-only use "
            "cannot be demonstrated")
    present = {str(role) for role in frame["role"].dropna().unique()}
    disallowed = sorted(present - set(DEVELOPMENT_ROLES))
    if disallowed:
        raise SplitDisciplineError(
            f"held-out roles reached a development-only computation: {disallowed}")


def read_development_candidates(path: str | Path, *,
                                roles: Sequence[str] = DEVELOPMENT_ROLES
                                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Authenticate the cached table by its bytes, then read only ``roles``.

    The role predicate is pushed into the parquet scan, so a held-out row is
    never materialised into a frame this process can reach.  Filtering a
    fully-loaded table would leave the same guarantee resting on a later
    ``.isin`` call, which is exactly the thing that must not be trusted.

    Authentication is the manifest's sha256 over the whole file, which already
    establishes byte-identity with what the producer published.  The manifest's
    ``expected_keys_sha256`` is deliberately *not* rechecked: doing so would
    require reading every row, including the held-out ones, to prove something
    the file hash has already proved.
    """
    from ..manifest import verify                    # local: keeps I/O off import

    target = Path(path)
    verdict = verify(target, cfg=None, require_identity=False)
    verdict["key_check_skipped_reason"] = (
        "rechecking expected_keys_sha256 would require reading held-out rows; "
        "the file sha256 already establishes byte-identity")
    if not verdict["ok"]:
        return pd.DataFrame(), verdict
    frame = pd.read_parquet(
        target, engine="pyarrow", filters=[("role", "in", list(roles))])
    frame = frame.reset_index(drop=True)
    # Fail closed even so: if the predicate were ever dropped or the column
    # renamed, a silent full read must not become a silent full analysis.
    assert_development_only(frame)
    verdict["roles_requested"] = list(roles)
    verdict["rows_read"] = int(len(frame))
    return frame, verdict


def role_scope(development: pd.DataFrame, *,
               declared_source_roles: Sequence[str] = ()) -> dict[str, Any]:
    """What was read, and what the source stage declared but this run did not.

    The withheld list comes from the *configuration* that produced the candidate
    table, not from the table's own role column: naming the held-out roles must
    not require reading their rows.
    """
    declared = [str(role) for role in declared_source_roles]
    return {
        "roles_permitted": list(DEVELOPMENT_ROLES),
        "roles_declared_by_source_stage": sorted(declared),
        "roles_withheld": sorted(set(declared) - set(DEVELOPMENT_ROLES)),
        "withheld_source": "configuration, not the candidate table's role column",
        "rows_read": int(len(development)),
        "rows_read_by_role": {
            str(role): int(count) for role, count
            in development["role"].value_counts().sort_index().items()}
        if len(development) else {},
        # The all-role figure this diagnostic is often compared against comes
        # from csasr.experiments.lss_alignment_dev_diagnostic, which reads every
        # role.  A development-only run reads fewer rows and therefore reports a
        # smaller paired count by construction; the two are not comparable and
        # neither is adjusted to match the other.
        "not_comparable_with": "csasr.experiments.lss_alignment_dev_diagnostic "
                               "(all roles, including held-out)",
    }


# --------------------------------------------------------------------------
# the conventions themselves
# --------------------------------------------------------------------------

def convention_boundary(convention: str, b_start: float,
                        b_end: float) -> tuple[float, float]:
    """New ``(preceding end, following start)`` for one blank run.

    ``b_start`` is the raw end of the preceding run and ``b_end`` the raw start
    of the following one, so ``[b_start, b_end]`` is exactly the blank run.  The
    result depends on nothing else: not on the other aligner, not on a measured
    disagreement, not on any fitted constant.  A reassignment of these frames is
    what each convention is; subtracting a learned offset would not be.
    """
    start, end = float(b_start), float(b_end)
    if convention == "blank_excluded":
        return start, end                       # the blank stays unowned
    if convention == "blank_to_preceding":
        return end, end
    if convention == "blank_to_following":
        return start, start
    if convention == "blank_midpoint":
        middle = (start + end) / 2.0
        return middle, middle
    raise ValueError(f"unknown boundary convention: {convention!r}")


def _number(value: Any) -> float:
    """A float, or NaN.  Mirrors the production physical-validity coercion.

    A missing edge must reach the classifier as NaN so it is *counted* as a
    non-finite boundary; letting the conversion raise would lose the row.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _variants(frame: pd.DataFrame) -> pd.Series:
    """Aligner variant per row, synthesised from the family when absent.

    The aggregation groups by variant, so a reassignment must be keyed by it
    too; otherwise one variant's blank run would move another variant's edge.
    """
    if "aligner_variant" in frame:
        return frame["aligner_variant"].astype(str)
    return frame["aligner_family"].astype(str)


def switch_transitions(candidates: pd.DataFrame, *,
                       family: str = CONVENTION_FAMILY
                       ) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """One row per language-run transition, classified by its blank run.

    Runs come from ``annotation_pack.language_runs`` and directly adjacent run
    pairs from ``annotation_pack.adjacent_run_transitions``; every *actionable*
    transition is one of those directly adjacent pairs, so a convention can only
    move a boundary the annotation pack would also recognise.

    The inventory is deliberately wider than the pack's eligible universe in one
    direction only: it also enumerates target-language run pairs separated by a
    third-language run, which the pack drops silently.  Those appear here as
    ``multiple_blank_runs`` and are never actionable, so they change no boundary
    -- they exist so the denominator is visible instead of implied.

    Every transition is kept, including the ones no convention can act on, so
    the reader can see exactly how much of the audio each convention leaves
    where ``blank_excluded`` put it.

    The second return value is keyed by aligner variant and counts boundaries
    that are *not* transitions: run edges at an utterance boundary, and
    same-language target runs the aggregation merges.  It is per variant because
    the disagreement statistics use one representative variant, and a count
    summed over every variant would not be that denominator.
    """
    columns = ["utterance_id", "aligner_variant", "left_reference_unit_index",
               "right_reference_unit_index", "switch_direction", "left_language",
               "right_language", "b_start_sec", "b_end_sec", "blank_run_ms",
               "intervening_runs", "directly_adjacent_runs", "classification",
               "actionable"]
    edges: dict[str, dict[str, int]] = {}
    units = candidates[candidates["aligner_family"].astype(str) == str(family)]
    if not len(units):
        return pd.DataFrame(columns=columns), edges

    units = units.copy()
    units["_variant"] = _variants(units)
    rows: list[dict[str, Any]] = []
    for (variant, utterance), group in units.groupby(["_variant", "utterance_id"],
                                                     sort=True):
        runs = pack_mod.language_runs(group)
        # The pack's own adjacency, keyed by the position of the left run.
        direct = {t["position"]: t
                  for t in pack_mod.adjacent_run_transitions(runs)}
        positions = [i for i, run in enumerate(runs)
                     if str(run["language"]) in TARGET_LANGUAGES]
        if not positions:
            continue
        tally = edges.setdefault(str(variant), {
            UTTERANCE_BOUNDARY_EDGE: 0, MERGED_BY_AGGREGATION: 0,
            "target_language_runs": 0, "utterances_with_target_runs": 0})
        # Target objects are the maximal same-language stretches of these runs,
        # so the two outer edges of the utterance belong to no transition.
        merged = sum(1 for a, b in zip(positions, positions[1:])
                     if runs[a]["language"] == runs[b]["language"])
        tally[MERGED_BY_AGGREGATION] += merged
        tally["target_language_runs"] += len(positions) - merged
        tally[UTTERANCE_BOUNDARY_EDGE] += 2
        tally["utterances_with_target_runs"] += 1

        for a, b in zip(positions, positions[1:]):
            left_run, right_run = runs[a], runs[b]
            if left_run["language"] == right_run["language"]:
                continue                        # counted as merged above
            adjacency = direct.get(a) if b == a + 1 else None
            if adjacency is not None:
                # Exactly the pack's transition object, units included.
                left, right = adjacency["left_unit"], adjacency["right_unit"]
                direction = adjacency["direction"]
            else:
                left, right = left_run["rows"][-1], right_run["rows"][0]
                direction = f"{left_run['language']}->{right_run['language']}"
            b_start, b_end = _number(left["end_sec"]), _number(right["start_sec"])
            intervening = b - a - 1
            index_step = (int(right["reference_unit_index"])
                          - int(left["reference_unit_index"]))
            if adjacency is None:
                # A third language sits between them: several blank runs, and
                # audio a unit of that language owns.
                classification = MULTIPLE_BLANK_RUNS
            elif index_step != 1:
                # A unit absent from the table owns part of the interval;
                # handing it to a neighbour would attribute a third unit's audio.
                classification = NON_CONTIGUOUS
            elif not target_mod._edge_supported(left):
                classification = LEFT_UNUSABLE
            elif not target_mod._edge_supported(right):
                classification = RIGHT_UNUSABLE
            elif not (math.isfinite(b_start) and math.isfinite(b_end)):
                classification = NON_FINITE
            elif b_end < b_start:
                classification = NO_BLANK_RUN
            elif b_end == b_start:
                classification = ZERO_LENGTH_BLANK_RUN
            else:
                classification = SINGLE_BLANK_RUN
            rows.append({
                "utterance_id": str(utterance),
                "aligner_variant": str(variant),
                "left_reference_unit_index": int(left["reference_unit_index"]),
                "right_reference_unit_index": int(right["reference_unit_index"]),
                "switch_direction": direction,
                "left_language": str(left_run["language"]),
                "right_language": str(right_run["language"]),
                "b_start_sec": b_start,
                "b_end_sec": b_end,
                "blank_run_ms": (b_end - b_start) * 1000.0,
                "intervening_runs": int(intervening),
                "directly_adjacent_runs": adjacency is not None,
                "classification": classification,
                "actionable": classification in ACTIONABLE,
            })
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["schema_version"] = TRANSITION_SCHEMA
    return frame, edges


def representative_variant(targets: pd.DataFrame, *,
                           family: str = CONVENTION_FAMILY) -> str:
    """The one variant the disagreement statistics are computed on.

    Read back from ``representative_targets`` rather than re-derived, so the
    edge-case denominator cannot name a different variant than the metrics use.
    """
    selected = target_mod.representative_targets(targets)
    if not len(selected):
        return ""
    side = selected[selected["aligner_family"].astype(str) == str(family)]
    if not len(side):
        return ""
    return str(side["aligner_variant"].iloc[0])


def edge_case_counts(transitions: pd.DataFrame,
                     edges: Mapping[str, Mapping[str, int]], *,
                     variant: str | None = None) -> dict[str, int]:
    """Every classification and non-transition case, counted, none omitted.

    ``variant`` restricts the counts to the aligner variant the disagreement
    statistics use.  Without it the counts would sum every variant of the
    family and so describe a larger population than the metrics do.

    Classifications with no instances are reported as zero rather than left out,
    because an absent key and a zero count are not the same claim.
    """
    if variant:
        transitions = (transitions[transitions["aligner_variant"].astype(str)
                                   == str(variant)]
                       if len(transitions) else transitions)
        tallies = [edges.get(str(variant), {})]
    else:
        tallies = list(edges.values())
    seen = (transitions["classification"].value_counts().to_dict()
            if len(transitions) else {})
    counts = {name: int(seen.get(name, 0)) for name in CLASSIFICATIONS}

    def total(name: str) -> int:
        return int(sum(int(tally.get(name, 0)) for tally in tallies))

    counts[UTTERANCE_BOUNDARY_EDGE] = total(UTTERANCE_BOUNDARY_EDGE)
    counts[MERGED_BY_AGGREGATION] = total(MERGED_BY_AGGREGATION)
    counts["transitions_total"] = int(len(transitions))
    counts["transitions_actionable"] = int(sum(
        counts[name] for name in ACTIONABLE))
    counts["target_language_runs"] = total("target_language_runs")
    counts["utterances_with_target_runs"] = total("utterances_with_target_runs")
    return counts


def blank_run_summary(transitions: pd.DataFrame, *,
                      variant: str | None = None) -> dict[str, Any]:
    """How much audio sits in the blank runs a convention can act on.

    Restricted to the same variant as the edge-case counts, so every reported
    population in the payload has one denominator.
    """
    if not len(transitions):
        return {"n": 0}
    if variant:
        transitions = transitions[
            transitions["aligner_variant"].astype(str) == str(variant)]
        if not len(transitions):
            return {"n": 0}
    actionable = transitions[transitions["actionable"].astype(bool)]

    def describe(frame: pd.DataFrame) -> dict[str, Any]:
        if not len(frame):
            return {"n": 0}
        blank = frame["blank_run_ms"]
        return {"n": int(len(frame)),
                "median_blank_run_ms": float(blank.median()),
                "p90_blank_run_ms": float(blank.quantile(0.9)),
                "total_blank_run_sec": float(blank.sum() / 1000.0)}

    return {
        "all_actionable": describe(actionable),
        "by_switch_direction": {
            str(direction): describe(group) for direction, group
            in actionable.groupby("switch_direction", sort=True)},
    }


def apply_convention(candidates: pd.DataFrame, convention: str, *,
                     family: str = CONVENTION_FAMILY,
                     transitions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a copy of ``candidates`` with ``family`` edges under ``convention``.

    The input frame is never mutated.  New boundaries come from
    :func:`convention_boundary` applied to the *original* spans, so a run that is
    the left side of one transition and the right side of the next cannot chain
    one reassignment into another.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown boundary convention: {convention!r}")
    frame = candidates.copy()
    frame["boundary_convention"] = convention
    frame["convention_adjusted_start"] = False
    frame["convention_adjusted_end"] = False
    if convention == "blank_excluded":
        # The identity convention: every edge stays where the cached candidate
        # put it.  Labelled anyway, so a downstream reader can never mistake an
        # unadjusted frame for an unlabelled one.
        return frame
    if not len(frame):
        return frame

    if transitions is None:
        transitions, _ = switch_transitions(candidates, family=family)
    actionable = transitions[transitions["actionable"].astype(bool)] \
        if len(transitions) else transitions
    if not len(actionable):
        return frame

    is_family = frame["aligner_family"].astype(str) == str(family)
    key = pd.Series(list(zip(frame["utterance_id"].astype(str),
                             _variants(frame),
                             frame["reference_unit_index"].astype(int))),
                    index=frame.index)

    new_end: dict[tuple[str, str, int], float] = {}
    new_start: dict[tuple[str, str, int], float] = {}
    for row in actionable.itertuples(index=False):
        preceding_end, following_start = convention_boundary(
            convention, row.b_start_sec, row.b_end_sec)
        left = (row.utterance_id, row.aligner_variant,
                int(row.left_reference_unit_index))
        right = (row.utterance_id, row.aligner_variant,
                 int(row.right_reference_unit_index))
        # Which side a convention moves is read off the convention's own output
        # rather than hard-coded here, so convention_boundary stays the only
        # place a rule lives.  The comparison is against the raw span carried in
        # the transition row -- both edges are finite for an actionable
        # transition -- and never against a frame cell that might be missing.
        if float(preceding_end) != float(row.b_start_sec):
            new_end[left] = float(preceding_end)
        if float(following_start) != float(row.b_end_sec):
            new_start[right] = float(following_start)
    if not (new_end or new_start):
        return frame

    def override(column: str, mapping: dict[tuple[str, str, int], float]
                 ) -> tuple[pd.Series, pd.Series]:
        """New edges plus the mask of rows that actually received one.

        The mask comes from the reassignment map, never from comparing the new
        value with the old: an edge that is already NaN compares unequal to
        itself, which would mark an untouched row as adjusted and then feed NaN
        into the sample cast.
        """
        replacement = key.map(mapping)
        adjusted = is_family & replacement.notna()
        return replacement.where(adjusted, frame[column]).astype(float), adjusted

    frame["end_sec"], frame["convention_adjusted_end"] = override("end_sec", new_end)
    frame["start_sec"], frame["convention_adjusted_start"] = override(
        "start_sec", new_start)

    moved = frame["convention_adjusted_start"] | frame["convention_adjusted_end"]
    if "sample_rate" in frame:
        rate = frame["sample_rate"].astype(float)
        for sec_column, sample_column, mask in (
                ("start_sec", "start_sample", frame["convention_adjusted_start"]),
                ("end_sec", "end_sample", frame["convention_adjusted_end"])):
            if sample_column not in frame:
                continue
            # A reassigned edge is finite by construction, so only rows the
            # convention actually moved are recomputed; every other sample value
            # is carried across untouched rather than round-tripped through a
            # cast that would turn a missing second into a bogus integer.
            updated = (frame[sec_column] * rate).round()
            keep = mask & updated.notna()
            frame[sample_column] = frame[sample_column].where(
                ~keep, updated).astype(frame[sample_column].dtype)
    # Encoder frame indices are deliberately left alone rather than recomputed:
    # in the cached tables they already disagree with the second-domain edges for
    # a tenth of the CTC rows, and quietly rewriting them here would hide that.
    frame["encoder_frame_stale"] = moved
    return frame


def convention_targets(candidates: pd.DataFrame, convention: str, *,
                       family: str = CONVENTION_FAMILY,
                       transitions: pd.DataFrame | None = None
                       ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Target objects rebuilt from spans under ``convention``.

    Aggregation is the expensive step, so callers build once per convention and
    read both the own-set and common-set statistics off the same frame.
    """
    adjusted = apply_convention(candidates, convention, family=family,
                                transitions=transitions)
    targets, _, aggregation = target_mod.target_objects(adjusted)
    return targets, aggregation


# --------------------------------------------------------------------------
# disagreement against the reference aligner
# --------------------------------------------------------------------------

def paired_target_ids(targets: pd.DataFrame,
                      pair: Sequence[str]) -> set[tuple[str, str]]:
    """Targets both families produced and both marked valid.

    A set operation, not a statistic: it exists so conventions can be compared
    on one common denominator instead of four moving ones.
    """
    if targets is None or not len(targets) or len(pair) != 2:
        return set()
    selected = target_mod.representative_targets(targets)
    a, b = map(str, pair)
    sides = []
    for family in (a, b):
        side = selected[(selected["aligner_family"].astype(str) == family)
                        & selected["target_valid"].astype(bool)]
        sides.append({(str(t), str(u)) for t, u
                      in zip(side["target_id"], side["utterance_id"])})
    return sides[0] & sides[1]


def paired_boundaries(targets: pd.DataFrame, pair: Sequence[str]) -> pd.DataFrame:
    """The rows the production agreement statistic is computed over.

    Mirrors ``cross_aligner_target_agreement``'s pairing exactly -- one
    representative variant per family, both sides ``target_valid``, merged on
    target/utterance/language, disagreement as the larger of the two absolute
    edge differences -- so the extra 200 ms fraction this diagnostic must report
    is measured on the identical denominator rather than a convenient one.
    :func:`agreement_entry` re-checks the shared statistics against the
    production function and reports the comparison instead of assuming it.
    """
    columns = ["target_id", "utterance_id", "language", "utterance_final",
               "dstart_ms", "dend_ms", "abs_boundary_ms"]
    if targets is None or not len(targets) or len(pair) != 2:
        return pd.DataFrame(columns=columns)
    selected = target_mod.representative_targets(targets)
    a, b = map(str, pair)
    selected = selected.copy()
    selected["utterance_final"] = selected["run_id"].eq(
        selected.groupby("utterance_id")["run_id"].transform("max"))
    keys = ["target_id", "utterance_id", "language"]
    cols = keys + ["start_sec", "end_sec"]
    left = selected[(selected["aligner_family"].astype(str) == a)
                    & selected["target_valid"].astype(bool)][
                        cols + ["utterance_final"]]
    right = selected[(selected["aligner_family"].astype(str) == b)
                     & selected["target_valid"].astype(bool)][cols]
    paired = left.merge(right, on=keys, suffixes=("_a", "_b"))
    if not len(paired):
        return pd.DataFrame(columns=columns)
    paired["dstart_ms"] = (paired["start_sec_a"] - paired["start_sec_b"]) * 1000.0
    paired["dend_ms"] = (paired["end_sec_a"] - paired["end_sec_b"]) * 1000.0
    paired["abs_boundary_ms"] = paired[["dstart_ms", "dend_ms"]].abs().max(axis=1)
    return paired


def _block(paired: pd.DataFrame) -> dict[str, Any]:
    """Disagreement statistics for one set of paired targets."""
    if not len(paired):
        return {"n": 0}
    values = paired["abs_boundary_ms"]
    out: dict[str, Any] = {
        "n": int(len(paired)),
        "median_boundary_disagreement_ms": float(values.median()),
        "p90_boundary_disagreement_ms": float(values.quantile(0.9)),
        "median_start_disagreement_ms": float(paired["dstart_ms"].abs().median()),
        "median_end_disagreement_ms": float(paired["dend_ms"].abs().median()),
    }
    for tolerance in TOLERANCES_MS:
        out[f"within_{int(tolerance)}ms"] = float((values <= tolerance).mean())
    return out


def disagreement_summary(targets: pd.DataFrame, pair: Sequence[str]) -> dict[str, Any]:
    """Overall, per-language, and utterance-final disagreement on one paired set.

    Both languages are cut from the same merged frame, so a per-language median
    is always a subset of the overall one and never a differently-paired set.
    The utterance-final split is another partition of that same paired frame;
    it does not repoint, drop, or repair any boundary.
    """
    paired = paired_boundaries(targets, pair)
    summary: dict[str, Any] = {
        **_block(paired),
        "by_language": {},
        "by_utterance_final": {
            "not_utterance_final": {"n": 0},
            "utterance_final": {"n": 0},
        },
    }
    if not len(paired):
        return summary
    summary["by_language"] = {str(language): _block(group) for language, group
                              in paired.groupby("language", sort=True)}
    summary["by_utterance_final"] = {
        ("utterance_final" if bool(is_final) else "not_utterance_final"):
            _block(group)
        for is_final, group in paired.groupby("utterance_final", sort=True)
    }
    medians = {language: summary["by_language"].get(language, {}).get(
        "median_boundary_disagreement_ms", float("nan"))
        for language in TARGET_LANGUAGES}
    en = float(medians.get("EN", float("nan")))
    zh = float(medians.get("ZH", float("nan")))
    both = math.isfinite(en) and math.isfinite(zh)
    summary["en_median_boundary_disagreement_ms"] = en
    summary["zh_median_boundary_disagreement_ms"] = zh
    # Signed, because the hypothesis under test is directional: ZH is predicted
    # to disagree more than EN.  The absolute difference is kept beside it
    # because that is the form the existing evidence is stated in.
    summary["en_minus_zh_median_ms"] = (en - zh) if both else float("nan")
    summary["abs_en_zh_median_difference_ms"] = abs(en - zh) if both else float("nan")
    return summary


def _reproduces(summary: Mapping[str, Any],
                production: Mapping[str, Any]) -> dict[str, Any]:
    """Does the local pairing reproduce the production statistic exactly?

    Reported, not asserted.  A mismatch means this harness is measuring a
    different quantity than the production path, which would make every other
    number it prints suspect -- so it must be visible, not smoothed over.
    """
    checks = {
        "n": ("n", "n"),
        "median_boundary_disagreement_ms": (
            "median_boundary_disagreement_ms",
            "cross_aligner_boundary_disagreement_ms"),
        "p90_boundary_disagreement_ms": (
            "p90_boundary_disagreement_ms",
            "cross_aligner_boundary_disagreement_p90_ms"),
        "within_100ms": ("within_100ms", "within_100ms"),
    }
    out: dict[str, Any] = {}
    for name, (here, there) in checks.items():
        mine, theirs = summary.get(here), production.get(there)
        if mine is None and theirs is None:
            # Neither side defines the statistic (an empty paired set); that is
            # agreement about its absence, not a mismatch.
            out[name] = True
            continue
        if mine is None or theirs is None:
            out[name] = False
            continue
        mine, theirs = float(mine), float(theirs)
        out[name] = bool(mine == theirs
                         or (math.isnan(mine) and math.isnan(theirs)))
    out["all"] = all(v is True for v in out.values())
    return out


def agreement_entry(targets: pd.DataFrame, aggregation: Mapping[str, Any],
                    convention: str, *,
                    family: str = CONVENTION_FAMILY,
                    reference_family: str = REFERENCE_FAMILY,
                    restrict_to: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Cross-aligner disagreement for one convention's targets.

    The headline statistics come from the production function itself, so median,
    P90 and the within-100 ms rate keep their production definitions rather than
    being respecified here.  The 200 ms fraction and the signed EN-ZH difference
    are computed beside them on the same paired rows.
    """
    pair = [family, reference_family]
    available = paired_target_ids(targets, pair)
    if restrict_to is not None and len(targets):
        keys = pd.Series(list(zip(targets["target_id"].astype(str),
                                  targets["utterance_id"].astype(str))),
                         index=targets.index)
        targets = targets[keys.isin(restrict_to)]
    if targets is None or not len(targets):
        production: dict[str, Any] = {
            "n": 0, "family_a": family, "family_b": reference_family,
            "measurement": "cross_aligner_disagreement",
            "reason": "no target objects under this convention"}
    else:
        production = target_mod.cross_aligner_target_agreement(targets, pair)
    summary = disagreement_summary(targets, pair)
    return {
        "boundary_convention": convention,
        "convention_kind": "blank_run_reassignment",
        "family_under_convention": family,
        "reference_family": reference_family,
        "signed_convention": f"{family} minus {reference_family}",
        "paired_targets_available": int(len(available)),
        "restricted_to_common_set": restrict_to is not None,
        "aggregation": dict(aggregation),
        "agreement": production,
        "disagreement": summary,
        "reproduces_production_statistics": _reproduces(summary, production),
    }


def _same_edge(left: pd.Series, right: pd.Series) -> pd.Series:
    """Equality that treats two missing edges as equal.

    ``NaN != NaN`` is True, so a plain comparison reports an invalid target
    whose boundary was never computed as *moved* -- including under
    ``blank_excluded``, which by definition moves nothing.
    """
    return (left == right) | (left.isna() & right.isna())


def target_edges_moved(built: Mapping[str, tuple[pd.DataFrame, Any]], *,
                       family: str = CONVENTION_FAMILY) -> dict[str, int]:
    """Target objects whose edges differ from ``blank_excluded``, per convention.

    Reassigning blank inside a language run cannot move that run's outer edges,
    so this counts the boundaries each convention actually changed.  Reported as
    a measurement; nothing is asserted from it.

    Both sides are reduced to one representative variant per family first -- the
    same rule the disagreement statistics use -- so the join is one-to-one.
    Merging on target/utterance/language alone would go many-to-many when a
    family has two variants, and could compare one variant against another.
    """
    baseline = built.get("blank_excluded", (pd.DataFrame(), None))[0]
    keys = ["target_id", "utterance_id", "language", "aligner_variant"]
    if not len(baseline):
        return {}

    def one_variant(targets: pd.DataFrame) -> pd.DataFrame:
        selected = target_mod.representative_targets(targets)
        if not len(selected):
            return pd.DataFrame(columns=keys + ["start_sec", "end_sec"])
        side = selected[selected["aligner_family"].astype(str) == str(family)]
        return side[keys + ["start_sec", "end_sec"]]

    left = one_variant(baseline)
    moved: dict[str, int] = {}
    for convention, (targets, _) in built.items():
        if not len(targets) or not len(left):
            moved[convention] = 0
            continue
        merged = left.merge(one_variant(targets), on=keys,
                            suffixes=("_base", "_conv"), validate="one_to_one")
        differs = ~(_same_edge(merged["start_sec_base"], merged["start_sec_conv"])
                    & _same_edge(merged["end_sec_base"], merged["end_sec_conv"]))
        moved[convention] = int(differs.sum())
    return moved


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

DISAGREEMENT_NOTE = ("another aligner's output is not gold; this is "
                     "disagreement between two estimators, not error")


def convention_table(payload: Sequence[Mapping[str, Any]], *,
                     edge_cases: Mapping[str, int],
                     moved: Mapping[str, int] | None = None) -> pd.DataFrame:
    """One row per convention: the required statistics and every edge-case count.

    The edge-case counts are identical across conventions by construction -- a
    transition is classified from raw spans alone -- and are repeated on each row
    so a single row is self-contained.  ``target_edges_moved`` is the column that
    differs, and it is zero for ``blank_excluded`` by definition.
    """
    moved = dict(moved or {})
    rows: list[dict[str, Any]] = []
    for entry in payload:
        summary = dict(entry.get("disagreement") or {})
        by_language = dict(summary.get("by_language") or {})
        by_utterance_final = dict(summary.get("by_utterance_final") or {})
        final = dict(by_utterance_final.get("utterance_final") or {})
        nonfinal = dict(by_utterance_final.get("not_utterance_final") or {})
        convention = str(entry["boundary_convention"])
        row: dict[str, Any] = {
            "boundary_convention": convention,
            "convention_kind": str(entry.get("convention_kind", "")),
            "paired_set": ("common_across_conventions"
                           if entry.get("restricted_to_common_set") else "own"),
            "family_under_convention": str(entry.get("family_under_convention", "")),
            "reference_family": str(entry.get("reference_family", "")),
            "n_paired": int(summary.get("n", 0) or 0),
            "median_boundary_disagreement_ms": summary.get(
                "median_boundary_disagreement_ms", float("nan")),
            "p90_boundary_disagreement_ms": summary.get(
                "p90_boundary_disagreement_ms", float("nan")),
            "median_start_disagreement_ms": summary.get(
                "median_start_disagreement_ms", float("nan")),
            "median_end_disagreement_ms": summary.get(
                "median_end_disagreement_ms", float("nan")),
            "en_median_boundary_disagreement_ms": summary.get(
                "en_median_boundary_disagreement_ms", float("nan")),
            "zh_median_boundary_disagreement_ms": summary.get(
                "zh_median_boundary_disagreement_ms", float("nan")),
            "en_minus_zh_median_ms": summary.get("en_minus_zh_median_ms",
                                                 float("nan")),
            "abs_en_zh_median_difference_ms": summary.get(
                "abs_en_zh_median_difference_ms", float("nan")),
            "n_en": int((by_language.get("EN") or {}).get("n", 0) or 0),
            "n_zh": int((by_language.get("ZH") or {}).get("n", 0) or 0),
            "n_utterance_final": int(final.get("n", 0) or 0),
            "utterance_final_median_boundary_disagreement_ms": final.get(
                "median_boundary_disagreement_ms", float("nan")),
            "utterance_final_p90_boundary_disagreement_ms": final.get(
                "p90_boundary_disagreement_ms", float("nan")),
            "n_not_utterance_final": int(nonfinal.get("n", 0) or 0),
            "not_utterance_final_median_boundary_disagreement_ms": nonfinal.get(
                "median_boundary_disagreement_ms", float("nan")),
            "not_utterance_final_p90_boundary_disagreement_ms": nonfinal.get(
                "p90_boundary_disagreement_ms", float("nan")),
            "target_edges_moved_vs_blank_excluded": int(moved.get(convention, 0)),
            "reproduces_production_statistics": bool(
                (entry.get("reproduces_production_statistics") or {}).get("all", False)),
            "measurement": "cross_aligner_disagreement",
            "note": DISAGREEMENT_NOTE,
        }
        for tolerance in TOLERANCES_MS:
            name = f"within_{int(tolerance)}ms"
            row[name] = summary.get(name, float("nan"))
            row[f"utterance_final_{name}"] = final.get(name, float("nan"))
            row[f"not_utterance_final_{name}"] = nonfinal.get(
                name, float("nan"))
        for name, count in edge_cases.items():
            # Totals stay unprefixed; the named cases are prefixed so a reader
            # can pick the edge-case block out of a wide table at a glance.
            prefix = "" if name in TRANSITION_TOTALS else "edge_case_"
            row[f"{prefix}{name}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)
