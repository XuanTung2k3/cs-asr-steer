"""Where the aligners disagree, and whether one of them is systematically wrong.

Measured on the NAT5H candidate table (461 units aligned by both families under
canonical sample coordinates):

    EN: median signed start (DTW - CTC) = -360 ms, median span 470 vs 270 ms
    ZH: median signed start (DTW - CTC) = -180 ms, median span 220 vs 100 ms

and, against synthetic ground truth, Whisper-DTW's own median signed error was
-490 ms. DTW is precisely self-consistent (two configurations agree to 10 ms)
but externally early, and it does not absorb silence. That is the signature of
an indexing convention rather than an acoustic failure, which is what
`pred_start_sweep` tests.

`offset_correction_counterfactual` exists to be *reported and not applied*. A
constant offset would lift 200 ms acceptance from 0.482 to 0.685 (EN 0.167 to
0.438) with no external evidence that it is right; corrections may only be
fitted to ground truth, which happens in the validation stage.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

EDGE_COLUMNS = ("start_sample", "end_sample")


def paired_edges(candidates: pd.DataFrame, *, sample_rate: int = 16000,
                 family_a: str = "whisper_dtw",
                 family_b: str = "existing_ctc") -> pd.DataFrame:
    """One row per reference unit aligned by both families, with signed deltas."""
    valid = candidates
    if "is_valid" in candidates:
        valid = candidates[candidates["is_valid"].astype(bool)]
    keep = valid[valid["aligner_family"].isin([family_a, family_b])]
    if not len(keep):
        return pd.DataFrame()

    index = ["utterance_id", "reference_unit_index", "reference_language"]
    wide = keep.pivot_table(index=index, columns="aligner_family",
                            values=list(EDGE_COLUMNS), aggfunc="first").dropna()
    if not len(wide):
        return pd.DataFrame()

    def col(edge: str, family: str) -> pd.Series:
        return wide[(edge, family)].astype(float)

    ms = 1000.0 / sample_rate
    frame = pd.DataFrame({
        "utterance_id": wide.index.get_level_values("utterance_id"),
        "reference_unit_index": wide.index.get_level_values("reference_unit_index"),
        "language": wide.index.get_level_values("reference_language"),
        "dstart_ms": (col("start_sample", family_a) - col("start_sample", family_b)).to_numpy() * ms,
        "dend_ms": (col("end_sample", family_a) - col("end_sample", family_b)).to_numpy() * ms,
        "duration_a_ms": (col("end_sample", family_a) - col("start_sample", family_a)).to_numpy() * ms,
        "duration_b_ms": (col("end_sample", family_b) - col("start_sample", family_b)).to_numpy() * ms,
        "start_a_sec": col("start_sample", family_a).to_numpy() / sample_rate,
    }).reset_index(drop=True)
    frame["family_a"] = family_a
    frame["family_b"] = family_b
    frame["abs_dstart_ms"] = frame["dstart_ms"].abs()
    frame["abs_dend_ms"] = frame["dend_ms"].abs()
    frame["is_first_unit"] = frame["reference_unit_index"] == \
        frame.groupby("utterance_id")["reference_unit_index"].transform("min")
    return frame


def signed_bias_summary(pairs: pd.DataFrame,
                        by: Sequence[str] = ("language",)) -> pd.DataFrame:
    """Median signed and absolute disagreement, by language and edge."""
    if not len(pairs):
        return pd.DataFrame()
    rows = []
    for keys, group in pairs.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for edge in ("start", "end"):
            signed = group[f"d{edge}_ms"]
            rows.append({
                **dict(zip(by, keys)),
                "edge": edge,
                "n": int(len(group)),
                "median_signed_ms": float(signed.median()),
                "mean_signed_ms": float(signed.mean()),
                "median_abs_ms": float(signed.abs().median()),
                "p90_abs_ms": float(signed.abs().quantile(0.9)),
                "within_50ms": float((signed.abs() <= 50).mean()),
                "within_100ms": float((signed.abs() <= 100).mean()),
                "within_200ms": float((signed.abs() <= 200).mean()),
            })
    return pd.DataFrame(rows)


#: a slope is only believed as evidence of a scale error when the relationship
#: is actually linear; on this data the OLS slope swings from 102 to 19 ms/s
#: depending on the time window while r stays at 0.14 and the intercept stays
#: at -500 ms, which is a constant offset with a fan-shaped residual, not a rate
#: error. Theil-Sen plus a correlation floor is what separates the two.
MIN_R_FOR_SCALE_ERROR = 0.30


def disagreement_regression(pairs: pd.DataFrame, *, edge: str = "start",
                            min_r: float = MIN_R_FOR_SCALE_ERROR) -> dict[str, Any]:
    """Regress signed disagreement on boundary time, robustly.

    A genuine frame-rate or scale error makes disagreement grow with time. A
    constant indexing offset does not. Distinguishing them needs a slope that
    survives outliers *and* a correlation strong enough to believe it.
    """
    if len(pairs) < 3:
        return {"n": int(len(pairs)), "slope_ms_per_s": float("nan"),
                "intercept_ms": float("nan"), "r": float("nan"),
                "scale_error_evidence": False}
    x = pairs["start_a_sec"].to_numpy(dtype=float)
    y = pairs[f"d{edge}_ms"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    try:
        from scipy.stats import theilslopes

        robust_slope, robust_intercept = theilslopes(y, x)[:2]
    except Exception:                                          # pragma: no cover
        robust_slope, robust_intercept = slope, intercept
    return {
        "n": int(len(pairs)), "edge": edge,
        "slope_ms_per_s": float(slope), "intercept_ms": float(intercept), "r": r,
        "theil_slope_ms_per_s": float(robust_slope),
        "theil_intercept_ms": float(robust_intercept),
        "median_signed_ms": float(np.median(y)),
        "min_r_for_scale_error": float(min_r),
        "scale_error_evidence": bool(np.isfinite(r) and abs(r) >= min_r),
        "interpretation": ("a scale error makes disagreement grow with time (large "
                           "robust slope AND |r| >= min_r); a constant indexing "
                           "offset shows a flat slope with a non-zero intercept"),
    }


def pred_start_sweep(bundle, manifest: pd.DataFrame, cfg: dict, truth: pd.DataFrame, *,
                     offsets: Sequence[int] = (-1, 0),
                     language: str = "zh") -> pd.DataFrame:
    """Score each decoder-query convention against synthetic ground truth.

    ``truth`` needs `utterance_id` and `true_boundary_sec` (the instant Mandarin
    stops in a spliced ZH+EN item). For each offset the first ZH->EN switch
    predicted by DTW is compared with that instant.
    """
    from ...data.alignment import align_batch

    acfg = cfg.get("alignment") or {}
    rows: list[dict[str, Any]] = []
    truth_map = dict(zip(truth["utterance_id"], truth["true_boundary_sec"]))
    batch_size = int(acfg.get("batch_size", 4))

    for offset in offsets:
        errors: list[float] = []
        for start in range(0, len(manifest), batch_size):
            batch = manifest.iloc[start:start + batch_size]
            alignments = align_batch(
                bundle, batch, language=language,
                median_filter_width=int(acfg.get("median_filter_width", 7)),
                max_text_tokens=int(acfg.get("max_text_tokens", 220)),
                pred_start_offset=int(offset))
            for alignment in alignments:
                true_sec = truth_map.get(alignment.utterance_id)
                if true_sec is None:
                    continue
                predicted = _first_switch_sec(alignment)
                if predicted is None:
                    continue
                errors.append((predicted - float(true_sec)) * 1000.0)
        error = np.asarray(errors, dtype=float)
        rows.append({
            "pred_start_offset": int(offset),
            "convention": "query_predicting_token" if offset == -1 else "query_at_token",
            "n_pairs": int(len(error)),
            "median_signed_error_ms": float(np.median(error)) if len(error) else float("nan"),
            "mean_signed_error_ms": float(error.mean()) if len(error) else float("nan"),
            "median_abs_error_ms": float(np.median(np.abs(error))) if len(error) else float("nan"),
            "within_100ms": float((np.abs(error) <= 100).mean()) if len(error) else float("nan"),
            "within_200ms": float((np.abs(error) <= 200).mean()) if len(error) else float("nan"),
        })
    return pd.DataFrame(rows)


def _first_switch_sec(alignment) -> float | None:
    """Start of the first English unit after a Mandarin one."""
    previous = None
    for i, tag in enumerate(alignment.tags):
        if tag not in ("EN", "ZH"):
            continue
        if previous == "ZH" and tag == "EN":
            start = float(alignment.start_sec[i])
            return start if np.isfinite(start) else None
        previous = tag
    return None


def offset_correction_counterfactual(pairs: pd.DataFrame,
                                     tolerances_ms: Sequence[float] = (50, 100, 200)
                                     ) -> pd.DataFrame:
    """What a constant offset would do to acceptance. DIAGNOSTIC ONLY.

    Never applied in this stage: an offset fitted to inter-aligner agreement
    optimizes the very statistic it would then be judged by. Corrections are
    fitted to external ground truth in the validation stage.
    """
    if not len(pairs):
        return pd.DataFrame()
    rows = []
    offsets = {"none": 0.0,
               "global_median": float(pairs["dstart_ms"].median())}
    for language in sorted(pairs["language"].unique()):
        offsets[f"median_{language}"] = float(
            pairs.loc[pairs["language"] == language, "dstart_ms"].median())

    for name, offset in offsets.items():
        corrected_start = (pairs["dstart_ms"] - offset).abs()
        corrected_end = (pairs["dend_ms"] - offset).abs()
        for tol in tolerances_ms:
            accepted = (corrected_start <= tol) & (corrected_end <= tol)
            rows.append({
                "correction": name, "offset_ms": offset, "tolerance_ms": float(tol),
                "acceptance_rate": float(accepted.mean()),
                "acceptance_rate_en": float(accepted[pairs["language"] == "EN"].mean())
                if (pairs["language"] == "EN").any() else float("nan"),
                "n": int(len(pairs)),
                "applied": False,
                "note": "diagnostic only; corrections must be fitted to external truth",
            })
    return pd.DataFrame(rows)
