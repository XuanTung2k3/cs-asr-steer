"""Production consensus spans: agreement, confidence bins, safe interiors.

Wraps `csasr.nat5h.consensus.build_unit_consensus_v2` and corrects three things
without editing it.

* **Erosion is derived, not inherited.** `configs/nat5h.yaml` erodes 250 ms per
  edge with a 200 ms minimum interior, which needs spans of at least 700 ms.
  Measured consensus medians here are 225 ms (EN) and 140 ms (ZH), so **zero**
  spans qualify: inheriting those values yields a stage that rejects everything
  with `safe_interior_too_short`. Erosion is a fraction of the span, capped by
  the measured human boundary error.
* **The tolerance is swept, not assumed.** At 200 ms only 16.7% of English units
  reach consensus and three quarters of the English rejections are start-edge
  disagreements, so tightening cannot raise coverage. The operating tolerance is
  chosen by a preregistered rule tied to external accuracy; the confidence bins
  do the tightening downstream.
* **Confidence bins are recomputed.** `nat5h.consensus` hard-codes the `high`
  bin at 100 ms independently of the configured criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...nat5h.consensus import ConsensusCriteria, build_unit_consensus_v2

DEFAULT_BINS = {"high": 50.0, "medium": 100.0, "low": 200.0}


@dataclass(frozen=True)
class ConsensusConfig:
    estimator: str = "both_edges"
    tolerance_ms: float = 200.0
    min_families: int = 2
    bins: Mapping[str, float] = None            # type: ignore[assignment]
    erosion_ms: float | None = None
    erosion_fraction: float = 0.25
    min_safe_interior_ms: float = 60.0
    union_padding_ms: float | None = None

    #: True when `primary_tolerance_ms` was not configured and the widest bin
    #: was used instead. That is a provisional value for sweeps and diagnostics,
    #: never a selection: `select_primary_tolerance` is what selects, and it
    #: needs authenticated external accuracy, which does not exist yet.
    tolerance_selected: bool = False
    selection_rule: str = "unselected: widest confidence bin, provisional"

    #: estimators this module actually implements. Recording a name that changes
    #: nothing would label output as coming from an ablation that never ran.
    SUPPORTED_ESTIMATORS = ("both_edges",)

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ConsensusConfig":
        block = dict((cfg or {}))
        bins = dict(block.get("bins") or DEFAULT_BINS)
        configured = block.get("primary_tolerance_ms")
        estimator = str(block.get("estimator", cls.estimator))
        if estimator not in cls.SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"consensus estimator {estimator!r} is configured but not "
                f"implemented; supported: {list(cls.SUPPORTED_ESTIMATORS)}. "
                "Recording it would label the output as an ablation that never ran.")
        alternates = [a for a in (block.get("alternates") or [])
                      if a not in cls.SUPPORTED_ESTIMATORS]
        if alternates:
            raise ValueError(
                f"consensus alternates {alternates} are configured but not "
                "implemented; remove them or implement them in "
                "`csasr.lss.align.consensus_prod.build`.")
        return cls(
            estimator=estimator,
            tolerance_selected=configured is not None,
            selection_rule=("configured primary_tolerance_ms" if configured is not None
                            else "unselected: widest confidence bin, provisional"),
            tolerance_ms=float(configured
                               or max(bins.values(), default=200.0)),
            min_families=int(block.get("min_families", cls.min_families)),
            bins={k: float(v) for k, v in bins.items()},
            erosion_ms=(None if block.get("erosion_ms") is None
                        else float(block["erosion_ms"])),
            erosion_fraction=float(block.get("erosion_fraction", cls.erosion_fraction)),
            min_safe_interior_ms=float(block.get("min_safe_interior_ms",
                                                 cls.min_safe_interior_ms)),
            union_padding_ms=(None if block.get("union_padding_ms") is None
                              else float(block["union_padding_ms"])),
        )

    def with_measured(self, *, median_error_ms: float,
                      p90_error_ms: float) -> "ConsensusConfig":
        """Set erosion and padding from the measured human boundary error."""
        return ConsensusConfig(
            estimator=self.estimator, tolerance_ms=self.tolerance_ms,
            min_families=self.min_families, bins=self.bins,
            erosion_ms=float(median_error_ms),
            erosion_fraction=self.erosion_fraction,
            min_safe_interior_ms=self.min_safe_interior_ms,
            union_padding_ms=float(p90_error_ms))


def _criteria(config: ConsensusConfig, tolerance_ms: float) -> ConsensusCriteria:
    # erosion is applied by this module, not by nat5h, so its own erosion is
    # disabled here; leaving it on would apply the fatal fixed 250 ms
    return ConsensusCriteria(
        min_aligners=config.min_families,
        max_start_disagreement_ms=tolerance_ms,
        max_end_disagreement_ms=tolerance_ms,
        safe_interior_erosion_ms=0.0,
        min_safe_interior_ms=1.0,
        steering_union_padding_ms=float(config.union_padding_ms or 0.0),
    )


def build(candidates: pd.DataFrame, config: ConsensusConfig, *,
          tolerance_ms: float | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Accepted spans, rejected units and the pairwise agreement table."""
    tolerance = float(tolerance_ms if tolerance_ms is not None else config.tolerance_ms)
    accepted, rejected, pairwise = build_unit_consensus_v2(
        candidates, _criteria(config, tolerance))
    if len(accepted):
        accepted = assign_confidence_bins(accepted, config.bins)
        accepted = add_safe_interior(accepted, config)
        accepted["consensus_tolerance_ms"] = tolerance
        accepted["consensus_estimator"] = config.estimator
    return accepted, rejected, pairwise


def assign_confidence_bins(accepted: pd.DataFrame,
                           bins: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Bin by the worse of the two edge disagreements, using *these* thresholds."""
    bins = dict(bins or DEFAULT_BINS)
    out = accepted.copy()
    worst = np.maximum(out["start_disagreement_ms"].astype(float).fillna(np.inf),
                       out["end_disagreement_ms"].astype(float).fillna(np.inf))
    out["max_edge_disagreement_ms"] = worst
    ordered = sorted(bins.items(), key=lambda kv: kv[1])
    labels = []
    for value in worst:
        label = "out_of_range"
        for name, threshold in ordered:
            if value <= threshold:
                label = name
                break
        labels.append(label)
    out["confidence_bin"] = labels
    return out


def add_safe_interior(accepted: pd.DataFrame, config: ConsensusConfig) -> pd.DataFrame:
    """Erode each span by a *fraction* of its own duration, capped by the error."""
    out = accepted.copy()
    sample_rate = out.get("sample_rate", pd.Series([16000] * len(out))).astype(float)
    start = out["consensus_start_sample"].astype(float)
    end = out["consensus_end_sample"].astype(float)
    duration_ms = (end - start) / sample_rate * 1000.0

    cap_ms = duration_ms * float(config.erosion_fraction)
    erosion_ms = cap_ms if config.erosion_ms is None else np.minimum(
        cap_ms, float(config.erosion_ms))
    erosion_samples = (erosion_ms / 1000.0 * sample_rate).round()

    safe_start = start + erosion_samples
    safe_end = end - erosion_samples
    min_samples = float(config.min_safe_interior_ms) / 1000.0 * sample_rate
    too_short = (safe_end - safe_start) < min_samples
    safe_start = np.where(too_short, start, safe_start)
    safe_end = np.where(too_short, end, safe_end)

    out["safe_start_sample"] = safe_start.astype(int)
    out["safe_end_sample"] = safe_end.astype(int)
    out["safe_interior_ms"] = (safe_end - safe_start) / sample_rate * 1000.0
    out["safe_interior_eroded"] = ~too_short
    out["erosion_ms_applied"] = np.where(too_short, 0.0, erosion_ms)

    padding = float(config.union_padding_ms or 0.0) / 1000.0 * sample_rate
    out["mask_start_sample"] = np.maximum(0, start - padding).astype(int)
    out["mask_end_sample"] = (end + padding).astype(int)
    return out


def tolerance_sweep(candidates: pd.DataFrame, config: ConsensusConfig,
                    tolerances_ms: Sequence[float]) -> pd.DataFrame:
    """Acceptance at each candidate tolerance, overall and for English."""
    rows: list[dict[str, Any]] = []
    total = candidates[candidates["reference_language"].isin(["EN", "ZH"])] \
        .drop_duplicates(["utterance_id", "reference_unit_index"]) \
        if len(candidates) else candidates
    n_en = int((total["reference_language"] == "EN").sum()) if len(total) else 0
    n_zh = int((total["reference_language"] == "ZH").sum()) if len(total) else 0

    for tol in tolerances_ms:
        accepted, rejected, _ = build(candidates, config, tolerance_ms=float(tol))
        language = accepted["language"] if "language" in accepted else pd.Series(dtype=str)
        accepted_en = int((language == "EN").sum()) if len(accepted) else 0
        accepted_zh = int((language == "ZH").sum()) if len(accepted) else 0
        rows.append({
            "tolerance_ms": float(tol),
            "accepted": int(len(accepted)),
            "accepted_en": accepted_en,
            "accepted_zh": accepted_zh,
            "en_retention": (accepted_en / n_en) if n_en else float("nan"),
            "zh_retention": (accepted_zh / n_zh) if n_zh else float("nan"),
            "rejected": int(len(rejected)),
            "high_or_medium": int((accepted["confidence_bin"].isin(["high", "medium"])).sum())
            if len(accepted) else 0,
        })
    return pd.DataFrame(rows)


def select_primary_tolerance(sweep: pd.DataFrame, accuracy: Mapping[float, Mapping[str, float]],
                             *, max_median_ms: float = 100.0,
                             max_p90_ms: float = 200.0) -> dict[str, Any]:
    """Preregistered rule: the largest tolerance that still meets accuracy.

    ``accuracy`` maps a tolerance to the measured human boundary error of the
    spans it accepts. Coverage never selects the tolerance; accuracy does, and
    among the tolerances that qualify the most generous one is taken because
    tightening cannot create agreement that is not there.
    """
    qualifying = []
    for tol in sorted(sweep["tolerance_ms"], reverse=True):
        measured = accuracy.get(float(tol)) or accuracy.get(tol) or {}
        median = float(measured.get("median_abs_error_ms", float("inf")))
        p90 = float(measured.get("p90_abs_error_ms", float("inf")))
        if median <= max_median_ms and p90 <= max_p90_ms:
            qualifying.append(float(tol))
    return {
        "selected_tolerance_ms": qualifying[0] if qualifying else None,
        "qualifying_tolerances_ms": qualifying,
        "rule": ("largest swept tolerance whose accepted spans satisfy human "
                 f"median <= {max_median_ms} ms and p90 <= {max_p90_ms} ms"),
        "selected_by": "external accuracy, never coverage",
    }
