"""Automatic Gate-A evidence: what two aligners can establish without humans.

The primary path to Gate A requires no new annotation. This module assembles
the evidence it is allowed to use, and -- just as importantly -- refuses to
dress one kind of evidence up as another.

Three quantities are measured on natural speech:

* **validity**: does each family produce well-formed, monotonic spans;
* **independence**: are at least two *genuinely different* aligners valid, where
  "different" means a different acoustic model and a different alignment
  algorithm, not a second variant of the same one;
* **cross-aligner disagreement**: how far apart two independent aligners place
  the same boundary.

Cross-aligner disagreement is **not** boundary error. Two aligners that share a
bias agree perfectly and are both wrong; on this data Whisper-DTW's own signed
error against synthetic truth was -490 ms while it agreed with itself to 10 ms
across configurations. Every natural-speech metric here is therefore named
`cross_aligner_*_disagreement_ms`, and `assert_no_absolute_error_claims` fails
the build if a criterion derived from natural speech is ever named as an error.

Absolute boundary error requires boundaries that are known rather than
estimated. There are exactly two sources of those:

* synthetic splices, whose boundaries are known by construction -- fitted and
  scored by `csasr.lss.align.synthetic`, and reported by
  `synthetic_calibration_status` below;
* genuine manual annotation, which is the optional manual mode only.

If neither is present the gate blocks with an explicit reason. It never
substitutes agreement for accuracy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .bias import paired_edges

#: aligner family -> the evidence class it belongs to. Two families in the same
#: class are not independent: `whisper_dtw` at two `pred_start` offsets, or two
#: median-filter widths, are the same estimator looking at the same attention.
INDEPENDENCE_CLASS: dict[str, str] = {
    "existing_ctc": "ctc_forced_alignment",          # MMS-FA acoustic CTC + uroman
    "whisper_dtw": "whisper_cross_attention_dtw",    # Whisper attention + DTW
    "qwen_forced_aligner": "qwen_forced_alignment",  # Qwen3 forced aligner
}

#: machine-readable reasons a gate can block on, rather than guess
BLOCKED_MISSING_SYNTHETIC = "blocked_missing_synthetic_calibration"
BLOCKED_INSUFFICIENT_ALIGNERS = "blocked_insufficient_independent_aligners"
BLOCKED_NO_CANDIDATES = "blocked_no_candidate_alignments"
BLOCKED_TAINTED_INPUTS = "blocked_tainted_inputs"
BLOCKED_MISSING_VERDICTS = "blocked_missing_manual_verdicts"
BLOCKED_UNAUTHENTICATED_CANDIDATES = "blocked_unauthenticated_candidate_evidence"
BLOCKED_UNAUTHENTICATED_SYNTHETIC = "blocked_unauthenticated_synthetic_evidence"
BLOCKED_NO_PAIRED_AGREEMENT = "blocked_no_paired_cross_aligner_evidence"
BLOCKED_EMPTY_CONSENSUS = "blocked_empty_consensus"
BLOCKED_MISSING_JITTER = "blocked_missing_jitter_evidence"
BLOCKED_MISSING_LANGUAGE = "blocked_missing_language_subset"
BLOCKED_EXPLORATORY_CANDIDATES = "blocked_exploratory_candidate_source"
#: The selection never ran, or its frozen record is not authentic. Distinct from
#: BLOCKED_MISSING_SYNTHETIC, which used to cover this too: once synthetic scores
#: exist, a run reporting "missing synthetic calibration" while a valid score
#: table sits on disk tells a reader the opposite of what is wrong.
BLOCKED_UNSELECTED_TOLERANCE = "blocked_unselected_operating_tolerance"

#: The selection *ran* on development boundaries and no swept tolerance met the
#: preregistered accuracy. Deliberately not `completed_no_go`: with no selected
#: operating point every downstream number describes a configuration nobody
#: chose, so the gate has not measured the experiment it would be reporting. The
#: measured evidence is in `metrics/l1b_tolerance_selection.parquet` and in the
#: frozen operating point, so nothing is hidden by blocking.
BLOCKED_NO_QUALIFYING_TOLERANCE = "blocked_no_qualifying_operating_tolerance"

#: A family qualifies on natural speech but has no synthetic-gate calibration.
#: Without this, CTC+Qwen could qualify naturally while the accuracy evidence
#: came from CTC+Whisper -- two independence classes on each side, a different
#: pair on each side, and an absolute-error claim about aligners that did not
#: produce the spans.
BLOCKED_UNCALIBRATED_FAMILIES = "blocked_uncalibrated_natural_families"

#: L1a's authenticated synthetic development set is missing, so no configuration
#: can be selected from development-only evidence.
BLOCKED_MISSING_DEVELOPMENT_SET = "blocked_missing_development_set"

#: The gate set could only be built from source audio an earlier generation
#: already exposed. Missing evidence, not a defect: a set whose results have been
#: read cannot confirm a configuration chosen after reading them, so there is
#: nothing here to judge with (`csasr.lss.align.exposure`).
BLOCKED_EXPOSED_GATE_SET = "blocked_exposed_gate_set"

#: substrings no natural-speech criterion may contain. Natural speech supports
#: statements about *disagreement between estimators*, never about error.
FORBIDDEN_NATURAL_NAME_PARTS = ("absolute_boundary_error", "abs_error",
                                "boundary_error", "ground_truth", "true_error")

NATURAL_SPEECH_NOTE = (
    "Measured between two independent automatic aligners on natural speech. "
    "This is estimator disagreement, not boundary error: a shared bias would "
    "make both aligners agree and both be wrong. Absolute error is only "
    "measurable against constructed synthetic boundaries or manual annotation."
)


# --------------------------------------------------------------------------
# validity and independence
# --------------------------------------------------------------------------
def _monotonic_flags(group: pd.DataFrame) -> pd.Series:
    """Per-row: does this span start after the previous unit's span starts."""
    ordered = group.sort_values("reference_unit_index")
    start = ordered["start_sample"].astype(float)
    end = ordered["end_sample"].astype(float)
    ok = (end > start) & (start >= 0)
    ok &= start.diff().fillna(0.0) >= 0
    return ok.reindex(group.index)


def family_validity(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-family coverage, invalid rate and nonmonotonic rate."""
    if candidates is None or not len(candidates):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for family, group in candidates.groupby("aligner_family"):
        valid = group.get("is_valid", pd.Series(True, index=group.index)).astype(bool)
        monotonic = pd.Series(True, index=group.index)
        # Monotonicity is a property of one aligner's output for one utterance.
        # Pooling two variants of the same family interleaves two independently
        # monotonic sequences and reports the result as nonmonotonic.
        variant_key = ["utterance_id"]
        if "aligner_variant" in group.columns:
            variant_key.append("aligner_variant")
        for _, utt in group.groupby(variant_key):
            monotonic.loc[utt.index] = _monotonic_flags(utt)
        rows.append({
            "aligner_family": str(family),
            "independence_class": INDEPENDENCE_CLASS.get(str(family), str(family)),
            "candidate_units": int(len(group)),
            "valid_units": int(valid.sum()),
            "valid_unit_coverage": float(valid.mean()),
            "invalid_rate": float(1.0 - valid.mean()),
            "nonmonotonic_rate": float((~monotonic).mean()),
            "variants": int(group["aligner_variant"].nunique())
            if "aligner_variant" in group else 1,
        })
    return pd.DataFrame(rows).sort_values("aligner_family").reset_index(drop=True)


def independent_valid_families(validity: pd.DataFrame, *,
                               min_coverage: float = 0.95,
                               max_invalid_rate: float = 0.01,
                               max_nonmonotonic_rate: float = 0.01,
                               min_units: int = 1) -> dict[str, Any]:
    """Which families clear the validity floor, and how many are independent.

    Counting is by independence class, so two variants of the same estimator can
    never together satisfy a two-aligner requirement.

    A family must clear *both* floors. The 0.95 coverage floor alone would admit
    5% invalid spans while the proposal's limit is 1%; they are different
    quantities (how much was aligned vs. how much of it is well formed) and both
    are required.
    """
    empty = {"qualifying_families": [], "independence_classes": [],
             "n_independent_valid_families": 0,
             "min_valid_unit_coverage": float(min_coverage),
             "max_invalid_rate": float(max_invalid_rate),
             "max_nonmonotonic_rate": float(max_nonmonotonic_rate),
             "rejected_families": [], "rejection_reasons": {}}
    if validity is None or not len(validity):
        return empty

    reasons: dict[str, list[str]] = {}
    qualifying: list[str] = []
    for row in validity.to_dict(orient="records"):
        family = str(row["aligner_family"])
        why: list[str] = []
        if float(row["valid_unit_coverage"]) < float(min_coverage):
            why.append(f"valid_unit_coverage={row['valid_unit_coverage']:.3f}"
                       f"<{min_coverage}")
        if float(row["invalid_rate"]) > float(max_invalid_rate):
            why.append(f"invalid_rate={row['invalid_rate']:.3f}>{max_invalid_rate}")
        if float(row["nonmonotonic_rate"]) > float(max_nonmonotonic_rate):
            why.append(f"nonmonotonic_rate={row['nonmonotonic_rate']:.3f}"
                       f">{max_nonmonotonic_rate}")
        if int(row["valid_units"]) < int(min_units):
            why.append(f"valid_units={row['valid_units']}<{min_units}")
        if why:
            reasons[family] = why
        else:
            qualifying.append(family)

    ok = validity[validity["aligner_family"].astype(str).isin(qualifying)]
    classes = sorted(set(ok["independence_class"].astype(str)))
    return {
        **empty,
        "qualifying_families": sorted(qualifying),
        "independence_classes": classes,
        "n_independent_valid_families": int(len(classes)),
        "rejected_families": sorted(reasons),
        "rejection_reasons": reasons,
    }


def unit_coverage(candidates: pd.DataFrame,
                  expected_units: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fraction of the *expected* reference units that a family aligned validly.

    The denominator must be the frozen universe of units the stage set out to
    align, not the rows that happen to be in the candidate table. Using the
    candidate rows makes coverage 1.0 by construction whenever an aligner
    silently drops utterances -- exactly the failure the threshold exists to
    catch.
    """
    key = ["utterance_id", "reference_unit_index"]
    if candidates is None or not len(candidates):
        expected = 0 if expected_units is None else int(len(
            expected_units.drop_duplicates([c for c in key
                                            if c in expected_units.columns])))
        return {"reference_units": expected, "covered_units": 0,
                "coverage": 0.0 if expected else float("nan"),
                "denominator": "expected_universe" if expected else "none",
                "missing_units": expected}

    valid = candidates
    if "is_valid" in candidates:
        valid = candidates[candidates["is_valid"].astype(bool)]
    covered = valid.drop_duplicates(key)

    if expected_units is not None and len(expected_units):
        present = [c for c in key if c in expected_units.columns]
        if len(present) == len(key):
            universe = expected_units.drop_duplicates(key)
            total = int(len(universe))
            seen = set(map(tuple, covered[key].astype(str).to_numpy()))
            want = set(map(tuple, universe[key].astype(str).to_numpy()))
            hit = len(want & seen)
            return {
                "reference_units": total,
                "covered_units": int(hit),
                "coverage": float(hit / total) if total else float("nan"),
                "denominator": "expected_universe",
                "missing_units": int(total - hit),
            }

    total = int(len(candidates.drop_duplicates(key)))
    return {
        "reference_units": total,
        "covered_units": int(len(covered)),
        "coverage": float(len(covered) / total) if total else float("nan"),
        # flagged so a reader knows this number cannot detect dropped utterances
        "denominator": "candidate_rows_only",
        "missing_units": 0,
    }


def usable_item_rate(spans: pd.DataFrame, *,
                     primary_bins: Sequence[str] = ("high", "medium")) -> dict[str, Any]:
    """Automatic analogue of the manual "at least 90% of audited units usable".

    An item is usable when it survives into the primary confidence bins with a
    non-degenerate safe interior -- that is, when it is something a steering
    mask could actually be built on.
    """
    if spans is None or not len(spans):
        return {"n": 0, "usable": 0, "usable_rate": float("nan")}
    total = int(len(spans))
    usable = spans
    if "confidence_bin" in usable:
        usable = usable[usable["confidence_bin"].isin(list(primary_bins))]
    if "safe_interior_ms" in usable:
        usable = usable[usable["safe_interior_ms"].astype(float) > 0]
    return {"n": total, "usable": int(len(usable)),
            "usable_rate": float(len(usable) / total) if total else float("nan"),
            "primary_bins": list(primary_bins)}


def language_eligibility(spans: pd.DataFrame,
                         languages: Sequence[str] = ("EN", "ZH"),
                         *, minimum: int = 1) -> dict[str, Any]:
    """Both language subsets must exist before any EN-ZH comparison means anything."""
    counts = {}
    for language in languages:
        if spans is None or not len(spans) or "language" not in spans:
            counts[str(language)] = 0
        else:
            counts[str(language)] = int((spans["language"] == language).sum())
    missing = sorted(k for k, v in counts.items() if v < int(minimum))
    return {"counts": counts, "missing_languages": missing,
            "eligible": not missing, "minimum": int(minimum)}


# --------------------------------------------------------------------------
# cross-aligner agreement on natural speech
# --------------------------------------------------------------------------
def cross_aligner_agreement(candidates: pd.DataFrame, *, family_a: str,
                            family_b: str,
                            sample_rate: int = 16000) -> dict[str, Any]:
    """Disagreement between two independent aligners, overall and per language.

    `boundary` is the worse of the two edges of a unit, which is the quantity a
    steering mask actually depends on.
    """
    pairs = paired_edges(candidates, sample_rate=sample_rate,
                         family_a=family_a, family_b=family_b)
    if not len(pairs):
        return {"n": 0, "family_a": family_a, "family_b": family_b,
                "measurement": "cross_aligner_disagreement",
                "note": NATURAL_SPEECH_NOTE}

    boundary = pairs[["abs_dstart_ms", "abs_dend_ms"]].max(axis=1)
    frame = pairs.assign(abs_boundary_ms=boundary)

    def stats(subset: pd.DataFrame) -> dict[str, float]:
        if not len(subset):
            return {"n": 0}
        return {
            "n": int(len(subset)),
            "cross_aligner_start_disagreement_ms":
                float(subset["abs_dstart_ms"].median()),
            "cross_aligner_end_disagreement_ms":
                float(subset["abs_dend_ms"].median()),
            "cross_aligner_boundary_disagreement_ms":
                float(subset["abs_boundary_ms"].median()),
            "cross_aligner_boundary_disagreement_p90_ms":
                float(subset["abs_boundary_ms"].quantile(0.9)),
            "median_signed_start_ms": float(subset["dstart_ms"].median()),
            "within_100ms": float((subset["abs_boundary_ms"] <= 100).mean()),
        }

    per_language = {str(lang): stats(group) for lang, group in frame.groupby("language")}
    en = per_language.get("EN", {}).get("cross_aligner_boundary_disagreement_ms",
                                        float("nan"))
    zh = per_language.get("ZH", {}).get("cross_aligner_boundary_disagreement_ms",
                                        float("nan"))
    diff = (abs(float(en) - float(zh))
            if np.isfinite(en) and np.isfinite(zh) else float("nan"))
    return {
        **stats(frame),
        "family_a": family_a, "family_b": family_b,
        "by_language": per_language,
        "en_zh_disagreement_diff_ms": diff,
        "measurement": "cross_aligner_disagreement",
        "note": NATURAL_SPEECH_NOTE,
    }


def agreement_for_qualifying_pair(candidates: pd.DataFrame,
                                  qualifying: Sequence[str],
                                  **kwargs: Any) -> dict[str, Any]:
    """Agreement between the first two independent qualifying families."""
    seen: dict[str, str] = {}
    for family in qualifying:
        cls = INDEPENDENCE_CLASS.get(str(family), str(family))
        seen.setdefault(cls, str(family))
    families = list(seen.values())
    if len(families) < 2:
        return {"n": 0, "reason": "fewer than two independent valid families",
                "measurement": "cross_aligner_disagreement",
                "note": NATURAL_SPEECH_NOTE}
    return cross_aligner_agreement(candidates, family_a=families[0],
                                   family_b=families[1], **kwargs)


def family_correspondence(qualifying: Sequence[str],
                          synthetically_scored: Sequence[str], *,
                          min_families: int = 2) -> dict[str, Any]:
    """Do the families that qualify on natural speech have accuracy evidence?

    Gate A used to check two things separately: that two independence classes
    qualified on natural speech, and that two independence classes were scored on
    the synthetic gate set. Nothing required them to be the *same* two. So
    CTC+Qwen could supply the spans and the cross-aligner disagreement while
    CTC+Whisper supplied the absolute error -- a false-pass path in which the
    accuracy claim is about an aligner that produced none of the spans being
    claimed about.

    The corresponding pair is what both statements must be made on: the
    intersection of the two class sets, and it must itself reach `min_families`.
    A synthetic score from a family that was *rejected* on natural speech cannot
    stand in for a qualifying family's missing calibration, because it is not the
    aligner whose boundaries the gate is about.
    """
    natural = {INDEPENDENCE_CLASS.get(str(f), str(f)): str(f) for f in qualifying}
    scored = {INDEPENDENCE_CLASS.get(str(f), str(f)): str(f)
              for f in synthetically_scored}
    corresponding = sorted(set(natural) & set(scored))
    uncalibrated = sorted(set(natural) - set(scored))
    return {
        "natural_qualifying_classes": sorted(natural),
        "synthetically_scored_classes": sorted(scored),
        "corresponding_classes": corresponding,
        "corresponding_families": [natural[c] for c in corresponding],
        "uncalibrated_natural_classes": uncalibrated,
        "uncalibrated_natural_families": [natural[c] for c in uncalibrated],
        # scored but rejected naturally: recorded so a reader can see that the
        # evidence exists and why it cannot be used
        "scored_but_not_qualifying_classes": sorted(set(scored) - set(natural)),
        "n_corresponding": len(corresponding),
        "min_families": int(min_families),
        "sufficient": bool(len(corresponding) >= int(min_families)
                           and not uncalibrated),
    }


# --------------------------------------------------------------------------
# synthetic exact-boundary calibration
# --------------------------------------------------------------------------
#: written by whoever implements synthetic scoring; read here, never invented
SYNTHETIC_SCORES_FILE = "metrics/l1b_synthetic_scores.parquet"

MISSING_SYNTHETIC_DESCRIPTION = (
    "No authenticated synthetic exact-boundary scores were found, so the "
    "automatic gate has no source of absolute boundary error. Scoring is "
    "implemented -- `lss_l1b_valid._score_synthetic_sets` renders the splices "
    "via `synthetic.build_set`, aligns them with `candidates.run_families`, maps "
    "predicted unit edges onto the constructed boundary "
    "(`synthetic.score_rendered_set`) and publishes "
    f"{SYNTHETIC_SCORES_FILE} -- so an absent table means that step did not run "
    "or its output was rejected. Check, in order: was the `synthetic` part "
    "included (it is skipped by `--only` and by `--dry-run`); did the aligner "
    "families produce candidates for the rendered audio "
    "(`diagnostics/l1b_automatic_evidence.json` and the stage log); is the "
    "table manifested and untainted. Cross-aligner disagreement is not a "
    "substitute for accuracy and is never used as one."
)


#: schema every synthetic score table must declare, so an older or hand-made
#: file cannot be read as if it meant the same thing
SYNTHETIC_SCORES_SCHEMA = "lss_synthetic_scores_v1"

#: the columns Gate A reads. `score_family` emits exactly these.
SYNTHETIC_REQUIRED_COLUMNS = (
    "schema_version", "family", "convention", "edge", "purpose",
    "num_boundaries", "within_100ms", "median_abs_error_ms",
    "p90_abs_error_ms", "median_signed_error_ms",
)


def synthetic_calibration_status(artifacts_root: str | Path,
                                 *, min_boundaries: int = 100,
                                 cfg: Mapping[str, Any] | None = None,
                                 require_authentication: bool = True,
                                 expected_gate_generation: int | None = None,
                                 expected_gate_request_sha256: str | None = None,
                                 expected_gate_item_sha256: str | None = None
                                 ) -> dict[str, Any]:
    """Is there scored synthetic ground truth, and what does it say?

    Returns ``available: False`` with an explicit reason when the scores do not
    exist, are not authentic, or do not declare the schema this reader
    understands. It never falls back to any other measurement.

    The expected generation, aligner request, and full item-set fingerprint bind
    the score table to both the model inputs and the construction truth it was
    scored against. The generation number or aligner request alone is
    insufficient: known boundaries could change while audio/transcript inputs
    remain byte-for-byte identical.
    """
    from ..manifest import read_verified

    root = Path(artifacts_root)
    path = root / SYNTHETIC_SCORES_FILE

    def unavailable(reason: str, detail: str) -> dict[str, Any]:
        return {"available": False, "reason": reason, "expected_path": str(path),
                "detail": detail, "missing_implementation": MISSING_SYNTHETIC_DESCRIPTION}

    if not path.is_file():
        return unavailable(BLOCKED_MISSING_SYNTHETIC, "the score table does not exist")

    manifest: Mapping[str, Any] = {}
    if require_authentication:
        scored, verdict = read_verified(path, cfg=cfg, require_identity=cfg is not None)
        if not verdict["ok"]:
            return unavailable(
                BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                f"{verdict['verdict']}: {verdict.get('detail') or ''}".strip(": "))
        manifest = verdict.get("manifest") or {}
        if manifest.get("diagnostic_only"):
            return unavailable(BLOCKED_TAINTED_INPUTS,
                               f"produced by {manifest.get('taint_reasons')}")
    else:
        scored = pd.read_parquet(path)

    if expected_gate_generation is not None:
        recorded = manifest.get("synthetic_gate_generation")
        if recorded != expected_gate_generation:
            return unavailable(
                BLOCKED_EXPOSED_GATE_SET,
                f"the score table was produced from gate generation {recorded!r}, "
                f"and the exposure ledger's unexposed generation is now "
                f"{expected_gate_generation!r}; those items' scores have already "
                "been read")
        if not expected_gate_request_sha256:
            return unavailable(
                BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                f"gate generation {expected_gate_generation!r} has no immutable "
                "alignment-request fingerprint in the exposure ledger")
        recorded_request = manifest.get("synthetic_gate_request_sha256")
        if recorded_request != expected_gate_request_sha256:
            return unavailable(
                BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                f"the score table describes gate request {recorded_request!r}, "
                f"but generation {expected_gate_generation!r} is bound to "
                f"{expected_gate_request_sha256!r}")
        if not expected_gate_item_sha256:
            return unavailable(
                BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                f"gate generation {expected_gate_generation!r} has no immutable "
                "full item-set fingerprint in the exposure ledger")
        recorded_items = manifest.get("synthetic_gate_item_sha256")
        if recorded_items != expected_gate_item_sha256:
            return unavailable(
                BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                f"the score table describes gate items {recorded_items!r}, but "
                f"generation {expected_gate_generation!r} is bound to full "
                f"item set {expected_gate_item_sha256!r}")

    missing = sorted(set(SYNTHETIC_REQUIRED_COLUMNS) - set(scored.columns))
    if missing or not len(scored):
        return unavailable(
            BLOCKED_MISSING_SYNTHETIC,
            f"unusable table (missing columns {missing}, {len(scored)} rows)")
    versions = sorted(set(scored["schema_version"].astype(str)))
    if versions != [SYNTHETIC_SCORES_SCHEMA]:
        return unavailable(BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                           f"schema {versions}, expected {SYNTHETIC_SCORES_SCHEMA}")

    numeric = ["num_boundaries", "within_100ms", "median_abs_error_ms",
               "p90_abs_error_ms", "median_signed_error_ms"]
    if not np.isfinite(scored[numeric].to_numpy(dtype=float)).all():
        return unavailable(BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                           "the table contains non-finite measurements")
    if (scored["within_100ms"] < 0).any() or (scored["within_100ms"] > 1).any():
        return unavailable(BLOCKED_UNAUTHENTICATED_SYNTHETIC,
                           "within_100ms is not a fraction in [0, 1]")

    # Gate A judges the gate set; the dev set exists to choose a configuration
    # and scoring it here would judge a choice on the data that made it.
    gate_rows = scored[scored["purpose"].astype(str) == "gate"]
    if not len(gate_rows):
        return unavailable(BLOCKED_MISSING_SYNTHETIC,
                           "no rows with purpose='gate'; only the development "
                           "set was scored, which cannot judge itself")

    # `num_boundaries` repeats across (family, convention, edge) rows describing
    # the SAME boundaries; summing it inflates the count several-fold.
    per_family = (gate_rows.groupby("family")["num_boundaries"].max()
                  if "family" in gate_rows else pd.Series(dtype=float))
    boundaries = int(per_family.min()) if len(per_family) else 0

    families = sorted(set(gate_rows["family"].astype(str)))
    classes = sorted({INDEPENDENCE_CLASS.get(f, f) for f in families})
    edges = sorted(set(gate_rows["edge"].astype(str)))

    return {
        "available": True,
        "path": str(path),
        "schema_version": SYNTHETIC_SCORES_SCHEMA,
        "families_scored": families,
        "independence_classes_scored": classes,
        "edges_scored": edges,
        "num_boundaries": boundaries,
        "boundary_count_rule": "min over families of that family's boundary count",
        "sufficient_boundaries": bool(boundaries >= int(min_boundaries)),
        # these ARE absolute errors: the boundaries are known by construction
        "absolute_boundary_error": {
            "within_100ms": float(gate_rows["within_100ms"].min()),
            "median_abs_error_ms": float(gate_rows["median_abs_error_ms"].max()),
            "p90_abs_error_ms": float(gate_rows["p90_abs_error_ms"].max()),
            "max_abs_bias_ms": float(gate_rows["median_signed_error_ms"].abs().max()),
            "by_row": gate_rows.to_dict(orient="records"),
        },
        "note": ("Absolute error against boundaries known by construction, "
                 "worst case over families, conventions and edges. This is the "
                 "only automatic source of accuracy evidence."),
    }


# --------------------------------------------------------------------------
# guard
# --------------------------------------------------------------------------
def assert_no_absolute_error_claims(criteria: Sequence[Mapping[str, Any]],
                                    *, source: str = "natural speech") -> None:
    """Fail loudly if a natural-speech criterion is named as an error.

    Naming is the whole mechanism by which a reader learns which claims the
    evidence supports, so it is enforced rather than reviewed.
    """
    offenders = [str(c.get("name", "")) for c in criteria
                 if any(part in str(c.get("name", "")).lower()
                        for part in FORBIDDEN_NATURAL_NAME_PARTS)]
    if offenders:
        raise AssertionError(
            f"criteria derived from {source} are named as absolute error: "
            f"{offenders}. {NATURAL_SPEECH_NOTE}")
