"""Point-of-Interest Error Rate and POI-level baseline status / error taxonomy.

A POI is an embedded-English content unit in the *reference* of a
code-switched utterance. PIER is the error rate restricted to those units
(Ugan et al., arXiv:2501.09512): it isolates what matters for code-switching
from the dominant matrix-language mass that MER averages over.

The error categories (guide section 8.2) are heuristic surface diagnostics.
They are frozen before any steering result is inspected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from ..data.normalize import is_han
from .mer import DEL, INS, MATCH, SUB, align_tokens
from .normalization import EN, ZH, normalize_text, segment_units, tag_unit

CATEGORIES = (
    "wrong_language_substitution",
    "phonetic_transliteration_or_script",
    "same_language_substitution",
    "deletion",
    "boundary_error",
    "insertion_near_poi",
    "other",
)
LANGUAGE_CONFUSION = ("wrong_language_substitution", "phonetic_transliteration_or_script")


@dataclass
class PoiResult:
    poi_index: int              # index into the reference unit sequence
    surface: str
    correct: bool
    category: str               # "correct" when correct
    hyp_surface: str


def reference_pois(reference_normalized: str) -> list[tuple[int, str]]:
    """(unit index, surface) of every English content unit in the reference."""
    units = segment_units(reference_normalized)
    return [(i, u.surface) for i, u in enumerate(units) if tag_unit(u) == EN]


def _neighbourhood(tokens: Sequence[str], idx: int, radius: int = 2) -> list[str]:
    lo = max(0, idx - radius)
    hi = min(len(tokens), idx + radius + 1)
    return list(tokens[lo:hi])


def _categorize(ref_tokens: Sequence[str], hyp_tokens: Sequence[str], ops,
                poi_idx: int) -> tuple[bool, str, str]:
    """Return (correct, category, hypothesis surface aligned to the POI)."""
    aligned = [o for o in ops if o.ref_idx == poi_idx]
    op = aligned[0] if aligned else None
    ref_tok = ref_tokens[poi_idx]

    if op is not None and op.op == MATCH:
        # a correct token may still have neighbouring insertions; still correct
        return True, "correct", hyp_tokens[op.hyp_idx]

    # tokens inserted immediately around the POI
    near_ins = [o for o in ops if o.op == INS and o.hyp_idx is not None
                and any(a.ref_idx in (poi_idx - 1, poi_idx, poi_idx + 1)
                        for a in ops if a.hyp_idx == o.hyp_idx - 1 or a.hyp_idx == o.hyp_idx + 1)]

    if op is None or op.op == DEL:
        # was the token produced somewhere close instead? -> boundary/ordering error
        if ref_tok in _neighbourhood(hyp_tokens, min(poi_idx, len(hyp_tokens) - 1), radius=3):
            return False, "boundary_error", ""
        if near_ins:
            return False, "insertion_near_poi", " ".join(hyp_tokens[o.hyp_idx] for o in near_ins)
        return False, "deletion", ""

    if op.op == SUB:
        hyp_tok = hyp_tokens[op.hyp_idx]
        if any(is_han(c) for c in hyp_tok):
            # Han substitution for an English word: language confusion. Multi-char
            # Han spans replacing one English word are the transliteration pattern.
            span = [hyp_tokens[o.hyp_idx] for o in ops
                    if o.hyp_idx is not None and o.ref_idx == poi_idx]
            extra = [o for o in ops if o.op == INS and o.hyp_idx is not None
                     and abs(o.hyp_idx - op.hyp_idx) == 1
                     and any(is_han(c) for c in hyp_tokens[o.hyp_idx])]
            if len(span) + len(extra) >= 2:
                return False, "phonetic_transliteration_or_script", "".join(
                    [hyp_tok] + [hyp_tokens[o.hyp_idx] for o in extra])
            return False, "wrong_language_substitution", hyp_tok
        if tag_unit(hyp_tok) == EN:
            return False, "same_language_substitution", hyp_tok
        return False, "other", hyp_tok

    return False, "other", ""


def evaluate_pois(reference: str, hypothesis: str,
                  poi_indices: Sequence[int] | None = None) -> list[PoiResult]:
    """Per-POI correctness and error category for one utterance."""
    ref_units = segment_units(normalize_text(reference))
    hyp_units = segment_units(normalize_text(hypothesis))
    ref_tokens = [u.surface for u in ref_units]
    hyp_tokens = [u.surface for u in hyp_units]
    ops = align_tokens(ref_tokens, hyp_tokens)

    if poi_indices is None:
        poi_indices = [i for i, u in enumerate(ref_units) if tag_unit(u) == EN]

    results = []
    for idx in poi_indices:
        if idx >= len(ref_tokens):
            continue
        correct, category, hyp_surface = _categorize(ref_tokens, hyp_tokens, ops, idx)
        results.append(PoiResult(idx, ref_tokens[idx], correct, category, hyp_surface))
    return results


def pier(references: Sequence[str], hypotheses: Sequence[str],
         poi_indices: Sequence[Sequence[int]] | None = None) -> dict:
    """Corpus PIER: errors on POIs divided by the number of POIs."""
    total, errors = 0, 0
    per_category: dict[str, int] = {c: 0 for c in CATEGORIES}
    for i, (r, h) in enumerate(zip(references, hypotheses)):
        idxs = poi_indices[i] if poi_indices is not None else None
        for res in evaluate_pois(r, h, idxs):
            total += 1
            if not res.correct:
                errors += 1
                per_category[res.category] = per_category.get(res.category, 0) + 1
    return {
        "pier": errors / total if total else float("nan"),
        "num_poi": total,
        "num_poi_errors": errors,
        "per_category": per_category,
    }


def unit_status(reference: str, hypothesis: str) -> dict[int, tuple[bool, str]]:
    """Correctness and category for *every* reference unit, keyed by unit index."""
    ref_units = segment_units(normalize_text(reference))
    hyp_units = segment_units(normalize_text(hypothesis))
    ref_tokens = [u.surface for u in ref_units]
    hyp_tokens = [u.surface for u in hyp_units]
    ops = align_tokens(ref_tokens, hyp_tokens)
    return {
        i: _categorize(ref_tokens, hyp_tokens, ops, i)[:2]
        for i in range(len(ref_tokens))
    }


def annotate_unit_status(units, manifest, predictions):
    """Fill `baseline_status` / `baseline_error_type` on a unit table.

    ``units`` is the E1 unit table, ``manifest`` supplies transcripts and
    ``predictions`` the frozen primary-baseline hypotheses. Units of utterances
    without a prediction keep the empty (unknown) status and are therefore
    excluded by correct-only filtering.
    """
    import pandas as pd  # local import keeps this module import-light

    refs = dict(zip(manifest["utterance_id"], manifest["transcript_raw"]))
    hyps = dict(zip(predictions["utterance_id"], predictions["hypothesis_raw"]))
    status_map: dict[tuple[str, int], tuple[bool, str]] = {}
    for utt, hyp in hyps.items():
        ref = refs.get(utt)
        if ref is None:
            continue
        for idx, (ok, cat) in unit_status(ref, hyp).items():
            status_map[(utt, idx)] = (ok, cat)

    out = units.copy()
    statuses, categories = [], []
    for utt, uid in zip(out["utterance_id"], out["unit_id"]):
        hit = status_map.get((utt, int(uid)))
        if hit is None:
            statuses.append("")
            categories.append("")
        else:
            statuses.append("correct" if hit[0] else "incorrect")
            categories.append("" if hit[0] else hit[1])
    out["baseline_status"] = statuses
    out["baseline_error_type"] = categories
    return out


def poi_table(utterance_ids: Sequence[str], references: Sequence[str],
              hypotheses: Sequence[str]) -> list[dict]:
    """Flat per-POI records for the P0 audit manifests."""
    rows = []
    for utt, ref, hyp in zip(utterance_ids, references, hypotheses):
        for res in evaluate_pois(ref, hyp):
            rows.append({"utterance_id": utt, **asdict(res)})
    return rows
