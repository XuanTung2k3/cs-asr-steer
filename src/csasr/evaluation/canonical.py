"""Canonical scoring facade for DG-01 and every later DG stage.

This module is the single authoritative entry point for error metrics and
correctness-transition accounting. It **reuses** the validated primitives in
``csasr.evaluation.mer`` (mixed language-aware Levenshtein alignment),
``csasr.evaluation.pier`` (POI semantics), and ``csasr.lss.outcomes``
(candidate correctness-flip harm, MC §10). It does **not** reimplement any
alignment, and it never uses the legacy whitespace-split blended WER.

Spec: ``docs/current/DG01_METRICS_RESULTS_SPEC.md``. Definitions are locked by
``docs/current/METHOD_CONTRACT.md``.

Sign convention (LOCKED): for every error metric,

    gain(M) = M_baseline - M_method        # positive = improvement

Nothing here emits the ambiguous bare name ``delta_pier``/``delta_MER``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "metrics_v1"

#: error metrics for which a signed improvement (`*_gain`) is defined
ERROR_METRICS = ("mer", "pier", "en_wer", "zh_cer")


def _finite_or_none(value: Any) -> float | None:
    """Return a finite float, or ``None`` for NaN/Inf/None (JSON-safe)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def gain(baseline: float | None, method: float | None) -> float | None:
    """Canonical improvement of an error metric: ``baseline - method``.

    Positive means improvement, zero unchanged, negative degradation. Returns
    ``None`` when either input is missing or non-finite.
    """
    b, m = _finite_or_none(baseline), _finite_or_none(method)
    if b is None or m is None:
        return None
    return b - m


# ---------------------------------------------------------------------------
# corpus error metrics (reuse mer.corpus_mer + pier.pier)
# ---------------------------------------------------------------------------

def corpus_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, Any]:
    """Canonical corpus error metrics for one hypothesis set.

    MER, EN-WER, ZH-CER come from the one mixed language-aware alignment
    (``mer.corpus_mer``: Mandarin per character, English per word, micro-
    averaged). PIER comes from ``pier.pier`` (errors on reference-English POIs
    over the POI count). No metric is rescored independently.
    """
    from .mer import corpus_mer
    from .pier import pier

    m = corpus_mer(references, hypotheses)
    p = pier(references, hypotheses)
    return {
        "mer": _finite_or_none(m["mer"]),
        "en_wer": _finite_or_none(m["en_wer"]),
        "zh_cer": _finite_or_none(m["zh_cer"]),
        "pier": _finite_or_none(p["pier"]),
        "num_ref_units": int(m["num_ref_tokens"]),
        "num_en_ref": int(m["num_en_ref"]),
        "num_zh_ref": int(m["num_zh_ref"]),
        "num_poi": int(p["num_poi"]),
        "num_poi_errors": int(p["num_poi_errors"]),
        "substitutions": int(m["substitutions"]),
        "deletions": int(m["deletions"]),
        "insertions": int(m["insertions"]),
    }


def error_metric_gains(baseline: Mapping[str, Any],
                       method: Mapping[str, Any]) -> dict[str, float | None]:
    """`{mer_gain, pier_gain, en_wer_gain, zh_cer_gain}` = baseline - method."""
    return {f"{name}_gain": gain(baseline.get(name), method.get(name))
            for name in ERROR_METRICS}


# ---------------------------------------------------------------------------
# POI-level correction / corruption (reuse pier.evaluate_pois; PIER-consistent)
# ---------------------------------------------------------------------------

def correction_corruption(references: Sequence[str], baseline: Sequence[str],
                          method: Sequence[str]) -> dict[str, Any]:
    """POI-level correctness transitions with explicit denominators.

    Population = reference POIs scored in **both** transcripts (the same
    normalized POI set the codebase PIER uses, via ``reference_pois`` +
    ``evaluate_pois``). This is the PIER-consistent accounting; it satisfies the
    PIER transition identity and is kept distinct from candidate accounting.

    - correction : baseline incorrect -> method correct
    - corruption : baseline correct   -> method incorrect
    - correction_rate  = corrections / (baseline-incorrect POIs)
    - corruption_rate  = corruptions / (baseline-correct POIs)
    """
    from .normalization import normalize_text
    from .pier import evaluate_pois, reference_pois

    n = corrections = corruptions = base_wrong = base_correct = 0
    for ref, b, m in zip(references, baseline, method):
        idxs = [i for i, _ in reference_pois(normalize_text(ref))]
        br = {x.poi_index: x.correct for x in evaluate_pois(ref, b, idxs)}
        mr = {x.poi_index: x.correct for x in evaluate_pois(ref, m, idxs)}
        for i in idxs:
            if i not in br or i not in mr:
                continue
            n += 1
            if br[i]:
                base_correct += 1
            else:
                base_wrong += 1
            corrections += int(not br[i] and mr[i])
            corruptions += int(br[i] and not mr[i])
    return {
        "level": "poi",
        "num_poi": n,
        "num_baseline_incorrect_poi": base_wrong,
        "num_baseline_correct_poi": base_correct,
        "corrections": corrections,
        "corruptions": corruptions,
        "net_corrections": corrections - corruptions,
        "correction_rate": (corrections / base_wrong) if base_wrong else None,
        "corruption_rate": (corruptions / base_correct) if base_correct else None,
    }


# ---------------------------------------------------------------------------
# candidate-level outside harm (reuse lss.outcomes; MC §10)
# ---------------------------------------------------------------------------

def candidate_outcomes(reference: str, baseline_hyp: str, steered_hyp: str,
                       sets: Any, *, languages: Mapping[int, str] | None = None,
                       eta: float = 1.0, kappa: float = 1.0) -> dict[str, Any]:
    """Canonical candidate correctness-flip accounting for one candidate.

    Reuses ``csasr.lss.outcomes.newly_introduced_errors`` and ``utility``.
    Canonical **outside harm** is ``n_corrupted_outside`` — a correct->incorrect
    flip on a trusted unit *outside* the candidate. A transcript difference that
    is not a correctness flip is never counted here (that distinction is the
    whole point of using this surface rather than the legacy edit fields).
    """
    from ..lss.outcomes import newly_introduced_errors, utility

    out = dict(newly_introduced_errors(reference, baseline_hyp, steered_hyp, sets,
                                       languages=languages))
    out["utility"] = utility(out, eta=eta, kappa=kappa)
    #: canonical name, made obvious in the output
    out["outside_harm"] = int(out["n_corrupted_outside"])
    return out


# ---------------------------------------------------------------------------
# one paired baseline-vs-method report
# ---------------------------------------------------------------------------

def paired_corpus_report(references: Sequence[str], baseline: Sequence[str],
                         method: Sequence[str], *,
                         assert_pier_identity: bool = True) -> dict[str, Any]:
    """Canonical corpus comparison: baseline metrics, method metrics, gains,
    and POI transitions, in one JSON-safe dict.

    ``*_gain`` fields use baseline - method (positive = improvement). The POI
    transition identity is asserted by default (reusing the validated check).
    """
    base_m = corpus_metrics(references, baseline)
    meth_m = corpus_metrics(references, method)
    gains = error_metric_gains(base_m, meth_m)
    trans = correction_corruption(references, baseline, method)

    if assert_pier_identity and trans["num_poi"]:
        from steer_sweep.round1_metrics import assert_pier_identity as _check
        _check(base_m["pier"], meth_m["pier"], {
            "num_poi": trans["num_poi"],
            "baseline_wrong_method_correct": trans["corrections"],
            "baseline_correct_method_wrong": trans["corruptions"],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "scored_utterances": len(references),
        "baseline": base_m,
        "method": meth_m,
        **gains,
        "transitions": trans,
    }
