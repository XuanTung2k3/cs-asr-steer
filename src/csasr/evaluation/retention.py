"""Canonical retention over the three MC §8 populations.

MC §8 distinguishes three retention groups that the legacy single
``zh_retention`` blended together:

1. **matrix-language (Mandarin) retention** — baseline-correct Mandarin units in
   a code-switched utterance (the matrix material an embedded-English
   intervention must not damage);
2. **embedded-English retention** — baseline-correct embedded-English units
   ``C_EN`` (English units inside a code-switched utterance);
3. **monolingual retention** — baseline-correct units in a monolingual utterance
   (whole-utterance Mandarin or English, outside any CS region).

Each is ``kept / baseline_correct`` over its population, reusing the validated
``pier.unit_status`` (per reference-unit correctness) and the versioned language
tagging. Retention is **not** an overall WER/CER, and the three populations are
never merged. An empty population yields ``rate = None`` (not 0).

Spec: ``docs/current/DG01_METRICS_RESULTS_SPEC.md`` §6.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .normalization import EN, ZH, normalize_text, segment_units, tag_unit
from .pier import unit_status


def _reference_is_code_switched(reference_normalized: str) -> bool:
    """True when the reference mixes English and Mandarin content units."""
    tags = {tag_unit(u) for u in segment_units(reference_normalized)}
    return EN in tags and ZH in tags


def _retention_counts(references: Sequence[str], baseline: Sequence[str],
                      steered: Sequence[str],
                      predicate: Callable[[str, bool], bool]) -> dict[str, Any]:
    """Count baseline-correct units that stay correct, over a chosen population.

    ``predicate(unit_language_tag, utterance_is_cs)`` selects the population.
    A unit enters the denominator only if it is correct in the baseline; it
    enters the numerator only if it is also correct under the method.
    """
    kept = total = 0
    for ref, base_hyp, steer_hyp in zip(references, baseline, steered):
        norm = normalize_text(ref)
        units = segment_units(norm)
        is_cs = _reference_is_code_switched(norm)
        before = unit_status(ref, base_hyp)
        after = unit_status(ref, steer_hyp)
        for idx, unit in enumerate(units):
            if not predicate(tag_unit(unit), is_cs):
                continue
            if not bool(before.get(idx, (False, ""))[0]):
                continue  # nothing to retain
            total += 1
            if bool(after.get(idx, (False, ""))[0]):
                kept += 1
    return {
        "numerator": int(kept),
        "denominator": int(total),
        "rate": (kept / total) if total else None,
    }


def matrix_zh_retention(references: Sequence[str], baseline: Sequence[str],
                        steered: Sequence[str]) -> dict[str, Any]:
    """Retention of baseline-correct Mandarin units in code-switched utterances."""
    return _retention_counts(references, baseline, steered,
                             lambda tag, is_cs: is_cs and tag == ZH)


def embedded_en_retention(references: Sequence[str], baseline: Sequence[str],
                          steered: Sequence[str]) -> dict[str, Any]:
    """Retention of baseline-correct embedded-English units ``C_EN``."""
    return _retention_counts(references, baseline, steered,
                             lambda tag, is_cs: is_cs and tag == EN)


def monolingual_retention(references: Sequence[str], baseline: Sequence[str],
                          steered: Sequence[str]) -> dict[str, Any]:
    """Retention of baseline-correct units in monolingual utterances."""
    return _retention_counts(references, baseline, steered,
                             lambda tag, is_cs: not is_cs)


def retention_report(references: Sequence[str], baseline: Sequence[str],
                     steered: Sequence[str]) -> dict[str, Any]:
    """All three MC §8 retention populations as one JSON-safe block."""
    return {
        "matrix_zh": matrix_zh_retention(references, baseline, steered),
        "embedded_en": embedded_en_retention(references, baseline, steered),
        "monolingual": monolingual_retention(references, baseline, steered),
    }
