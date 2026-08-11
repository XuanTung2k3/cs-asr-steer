"""Correction / corruption accounting for steered systems (guide section 42)."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .mer import DEL, INS, MATCH, SUB, align_tokens
from .normalization import normalize_text, segment_units
from .pier import evaluate_pois


def poi_status(reference: str, hypothesis: str, poi_index: int) -> tuple[bool, str]:
    res = evaluate_pois(reference, hypothesis, [poi_index])
    if not res:
        return False, "other"
    return res[0].correct, res[0].category


def outside_region_edits(reference: str, base_hyp: str, steered_hyp: str,
                         poi_index: int, exclude_radius: int = 1) -> dict:
    """Edits between baseline and steered output away from the target POI.

    Both hypotheses are aligned to the reference; reference positions within
    ``exclude_radius`` of the POI are excluded, and the remaining positions are
    compared. Insertions are attributed to the preceding reference position.
    """
    ref_tokens = [u.surface for u in segment_units(normalize_text(reference))]
    lo, hi = poi_index - exclude_radius, poi_index + exclude_radius

    def projection(hyp: str) -> dict[int, str]:
        hyp_tokens = [u.surface for u in segment_units(normalize_text(hyp))]
        ops = align_tokens(ref_tokens, hyp_tokens)
        out: dict[int, str] = {}
        last_ref = -1
        for o in ops:
            if o.ref_idx is not None:
                last_ref = o.ref_idx
                out[o.ref_idx] = hyp_tokens[o.hyp_idx] if o.hyp_idx is not None else ""
            elif o.op == INS:
                key = last_ref
                out[key] = out.get(key, "") + "+" + hyp_tokens[o.hyp_idx]
        return out

    a, b = projection(base_hyp), projection(steered_hyp)
    positions = sorted(set(a) | set(b))
    considered = [p for p in positions if p < lo or p > hi]
    changed = [p for p in considered if a.get(p, "") != b.get(p, "")]
    return {
        "outside_positions": len(considered),
        "outside_changed": len(changed),
        "outside_edit_rate": len(changed) / len(considered) if considered else 0.0,
        "any_change": normalize_text(base_hyp) != normalize_text(steered_hyp),
    }


def correction_harm_table(rows: Sequence[dict]) -> pd.DataFrame:
    """Build the per-POI comparison table.

    Each input row needs: utterance_id, speaker_id, reference, poi_index,
    baseline_hypothesis, steered_hypothesis, baseline_correct, baseline_category.
    """
    out = []
    for r in rows:
        steered_correct, steered_cat = poi_status(r["reference"], r["steered_hypothesis"],
                                                  r["poi_index"])
        edits = outside_region_edits(r["reference"], r["baseline_hypothesis"],
                                     r["steered_hypothesis"], r["poi_index"])
        out.append({
            **{k: r[k] for k in ("utterance_id", "speaker_id", "poi_index") if k in r},
            "baseline_correct": bool(r["baseline_correct"]),
            "baseline_category": r.get("baseline_category", ""),
            "steered_correct": bool(steered_correct),
            "steered_category": steered_cat,
            "corrected": bool(not r["baseline_correct"] and steered_correct),
            "corrupted": bool(r["baseline_correct"] and not steered_correct),
            **edits,
        })
    return pd.DataFrame(out)


EMPTY_SUMMARY = {
    "num_baseline_incorrect": 0, "num_baseline_correct": 0, "num_corrected": 0,
    "num_corrupted": 0, "correction_rate": float("nan"), "corruption_rate": float("nan"),
    "correction_to_corruption_ratio": float("nan"), "net_corrected_pois": 0,
    "raw_corrected_count": 0, "raw_corrupted_count": 0, "net_corrected_count": 0,
    "correction_rate_minus_corruption_rate": float("nan"),
    "zero_corruption": False, "correction_to_corruption_ratio_note": "",
    "outside_edit_rate": float("nan"), "pct_any_transcript_change": float("nan"),
}


def summarize_correction_harm(table: pd.DataFrame) -> dict:
    """CorrectionRate, CorruptionRate, ratio and net corrections."""
    if not len(table):
        return dict(EMPTY_SUMMARY)
    e0 = table[~table["baseline_correct"]]
    c0 = table[table["baseline_correct"]]
    n_corr = int(e0["corrected"].sum())
    n_harm = int(c0["corrupted"].sum())
    correction_rate = n_corr / len(e0) if len(e0) else float("nan")
    corruption_rate = n_harm / len(c0) if len(c0) else float("nan")
    if n_harm > 0:
        ratio = n_corr / n_harm
        ratio_note = "finite ratio over raw corrected/corrupted counts"
    else:
        ratio = float("nan")
        ratio_note = ("undefined because corrupted count is zero; evaluate raw "
                      "corrected/corrupted counts and net error reduction instead")
    return {
        "num_baseline_incorrect": int(len(e0)),
        "num_baseline_correct": int(len(c0)),
        "num_corrected": n_corr,
        "num_corrupted": n_harm,
        "raw_corrected_count": n_corr,
        "raw_corrupted_count": n_harm,
        "correction_rate": correction_rate,
        "corruption_rate": corruption_rate,
        "correction_to_corruption_ratio": ratio,
        "zero_corruption": bool(n_harm == 0),
        "correction_to_corruption_ratio_note": ratio_note,
        "net_corrected_pois": n_corr - n_harm,
        "net_corrected_count": n_corr - n_harm,
        "correction_rate_minus_corruption_rate": (
            float(correction_rate - corruption_rate)
            if pd.notna(correction_rate) and pd.notna(corruption_rate) else float("nan")
        ),
        "outside_edit_rate": float(table["outside_edit_rate"].mean()) if len(table) else float("nan"),
        "pct_any_transcript_change": float(table["any_change"].mean()) if len(table) else float("nan"),
    }


def per_category_correction(table: pd.DataFrame, min_count: int = 5) -> dict:
    e0 = table[~table["baseline_correct"]]
    out = {}
    for cat, grp in e0.groupby("baseline_category"):
        if len(grp) < min_count:
            out[str(cat)] = {"n": int(len(grp)), "correction_rate": None,
                             "note": "sample too small to report"}
        else:
            out[str(cat)] = {"n": int(len(grp)),
                             "correction_rate": float(grp["corrected"].mean())}
    return out
