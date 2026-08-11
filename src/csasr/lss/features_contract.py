"""What the test-time selector is allowed to see.

Plan v1 listed a feature "boundary confidence" sourced from consensus edge
disagreement. Consensus comes from forced-aligning the *reference*, which the
automatic system does not have. A selector trained on it would score well in
development and be unbuildable at test time. It is removed here, and replaced by
an inference-legal measure of the same intuition (how sharply the localizer's
own score falls off at the candidate edges).

The guard rejects forbidden **sources and columns**, never rows from a
particular split: a guard that refuses test rows would block the locked run from
computing legitimate features.

Every feature carries three things a reviewer will ask about: whether it exists
at inference, what it does when no first-pass token overlaps the candidate
(deletions have none), and how a missing value is encoded.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

ALLOWLIST_VERSION = "lss_features_v1"

#: encoding for a feature that legitimately has no value for a candidate
MISSING_ENCODING = "NaN plus a <feature>_missing boolean"


@dataclass(frozen=True)
class FeatureContract:
    name: str
    source_artifact: str
    inference_available: bool
    missing_policy: str        # "nan_plus_flag" | "zero" | "forbidden"
    candidate_mapping: str     # behaviour when no first-pass token overlaps
    dtype: str = "float32"


FEATURE_ALLOWLIST: tuple[FeatureContract, ...] = (
    FeatureContract("localizer_score_max", "l6_localizer", True, "forbidden",
                    "always defined: the candidate exists because of this score"),
    FeatureContract("localizer_score_mean", "l6_localizer", True, "forbidden",
                    "always defined"),
    FeatureContract("candidate_duration_ms", "l6_localizer", True, "forbidden",
                    "always defined"),
    FeatureContract("candidate_edge_sharpness", "l6_localizer", True, "nan_plus_flag",
                    "gradient of the localizer score at the candidate edges; "
                    "replaces plan v1's consensus-derived boundary confidence"),
    FeatureContract("encoder_en_margin", "l2b_cache", True, "nan_plus_flag",
                    "projection of pooled span states on the frozen direction; "
                    "acoustic, so defined even when the transcript has no token"),
    FeatureContract("first_pass_token_is_latin", "l2b_cache", True, "nan_plus_flag",
                    "missing when no first-pass token overlaps (deletions)"),
    FeatureContract("token_confidence_min", "l2b_cache", True, "nan_plus_flag",
                    "missing for deletions"),
    FeatureContract("token_entropy_max", "l2b_cache", True, "nan_plus_flag",
                    "missing for deletions"),
    FeatureContract("token_top2_margin_min", "l2b_cache", True, "nan_plus_flag",
                    "missing for deletions"),
    FeatureContract("encoder_decoder_disagreement", "l2b_cache", True, "nan_plus_flag",
                    "encoder margin sign vs decoded token language; missing for deletions"),
    FeatureContract("cross_attention_concentration", "l2b_cache", True, "nan_plus_flag",
                    "attention mass on the candidate frames, from first-pass weights"),
    FeatureContract("candidate_position_in_utterance", "l2b_cache", True, "forbidden",
                    "always defined"),
    FeatureContract("utterance_duration_sec", "manifest", True, "forbidden",
                    "always defined"),
    FeatureContract("term_frequency_proxy", "train_transcripts", True, "nan_plus_flag",
                    "frequency of the first-pass surface in TRAINING transcripts only; "
                    "missing for deletions"),
)

#: column-name patterns that would leak reference or alignment information
FORBIDDEN_SOURCE_PATTERNS: tuple[str, ...] = (
    "reference*", "consensus*", "aligner*", "gold*", "human_*", "poi_*",
    "unit_id", "unit_ids", "surface", "normalized_surface", "transcript*",
    "baseline_correct*", "baseline_category*", "steered_*", "utility*",
    "n_corrected*", "n_corrupted*", "*_disagreement_ms", "confidence_bin",
    "is_embedded_english", "language",
)

#: features removed from plan v1, kept here so the removal is auditable
REMOVED_FEATURES: tuple[tuple[str, str], ...] = (
    ("boundary_confidence", "sourced from consensus edge disagreement, which is "
                            "reference-derived and unavailable at inference"),
)


def feature_names() -> tuple[str, ...]:
    return tuple(f.name for f in FEATURE_ALLOWLIST)


def contract_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "name": f.name, "source_artifact": f.source_artifact,
        "inference_available": f.inference_available,
        "missing_policy": f.missing_policy,
        "candidate_mapping": f.candidate_mapping, "dtype": f.dtype,
    } for f in FEATURE_ALLOWLIST])


def is_forbidden(column: str,
                 patterns: Sequence[str] = FORBIDDEN_SOURCE_PATTERNS) -> bool:
    return any(fnmatch.fnmatch(str(column), p) for p in patterns)


def assert_inference_safe(frame: pd.DataFrame, *,
                          allowlist: Iterable[FeatureContract] = FEATURE_ALLOWLIST,
                          patterns: Sequence[str] = FORBIDDEN_SOURCE_PATTERNS,
                          extra_allowed: Sequence[str] = ()) -> None:
    """Raise if a feature frame carries a column the test-time system cannot have.

    Identity columns (candidate/utterance/conversation ids) are permitted via
    ``extra_allowed`` so that a frame can still be joined and grouped.
    """
    allowed = {f.name for f in allowlist} | set(extra_allowed)
    offenders = [c for c in frame.columns if c not in allowed and is_forbidden(c, patterns)]
    if offenders:
        raise ValueError(
            "selector features must be computable at inference; these columns are "
            f"reference- or alignment-derived: {sorted(offenders)}")
    unlisted = [c for c in frame.columns if c not in allowed and not is_forbidden(c, patterns)]
    if unlisted:
        raise ValueError(
            f"columns are not on the frozen feature allowlist: {sorted(unlisted)}. "
            "Add a FeatureContract (with its missing/candidate-mapping policy) first.")


def allowlist_payload() -> dict:
    """The block written into the spec freeze."""
    return {
        "allowlist_version": ALLOWLIST_VERSION,
        "features": contract_table().to_dict(orient="records"),
        "forbidden_source_patterns": list(FORBIDDEN_SOURCE_PATTERNS),
        "guard": "csasr.lss.features_contract.assert_inference_safe",
        "guard_rejects": "sources and columns, never a data split",
        "missing_value_encoding": MISSING_ENCODING,
        "removed": [{"name": n, "reason": r} for n, r in REMOVED_FEATURES],
    }
