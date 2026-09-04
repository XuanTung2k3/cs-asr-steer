"""Metrics and gates for Track A. Scoring is REUSED, never reimplemented.

  PIER, per-unit status, POI taxonomy   csasr.evaluation.pier
  MER / WER                             csasr.evaluation.mer, v2r3_day5_expansion
  correction / corruption / spillover   v2r3_day5_expansion.outcome_rates
  monolingual retention                 v2r3_gate_b.monolingual_retention
  cluster bootstrap                     csasr.evaluation.bootstrap

Everything is evaluated on ALL 300 utterances of `D-dev-select`, not only the
116 that carry a target: global steering touches every utterance and the harm
denominator has to include the ones it could only damage.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# corpus metrics
# ---------------------------------------------------------------------------

def corpus_metrics(pop, hypotheses: dict[str, str]) -> dict[str, Any]:
    """PIER / MER / WER over the full evaluated population."""
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier
    from csasr.experiments.v2r3_day5_expansion import corpus_word_error_rate
    from . import data as D

    references = D.reference_of(pop)
    ids = [u for u in pop.utterance_ids if u in hypotheses]
    refs = [references[u] for u in ids]
    hyps = [hypotheses[u] for u in ids]
    pier_out = pier(refs, hyps)
    mer_out = corpus_mer(refs, hyps)
    wer_out = corpus_word_error_rate(refs, hyps)
    return {
        "scored_utterances": len(ids),
        "PIER": float(pier_out["pier"]),
        "num_poi": int(pier_out["num_poi"]),
        "num_poi_errors": int(pier_out["num_poi_errors"]),
        "poi_per_category": dict(pier_out["per_category"]),
        "MER": float(mer_out["mer"]) if "mer" in mer_out else float("nan"),
        "WER": float(wer_out["wer"]),
    }


# ---------------------------------------------------------------------------
# per-target outcomes
# ---------------------------------------------------------------------------

def target_outcomes(pop, baseline: dict[str, str],
                    steered: dict[str, str]) -> pd.DataFrame:
    """One row per target unit: corrected / corrupted / spillover."""
    from csasr.experiments.v2r3_day5_expansion import outcome_rates
    from . import data as D

    references = D.reference_of(pop)
    rows: list[dict[str, Any]] = []
    for _, t in pop.targets.iterrows():
        utterance = str(t["utterance_id"])
        if utterance not in baseline or utterance not in steered:
            continue
        reference = references[utterance]
        rates = outcome_rates(reference, baseline[utterance], steered[utterance],
                              int(t["reference_unit_index"]))
        rows.append({
            "utterance_id": utterance,
            "dialogue_id": str(t["dialogue_id"]),
            "reference_unit_index": int(t["reference_unit_index"]),
            "category": str(t["category"]),
            "is_deletion": bool(t["is_deletion"]),
            **rates,
        })
    return pd.DataFrame(rows)


def corruption_and_retention(pop, baseline: dict[str, str],
                             steered: dict[str, str]) -> dict[str, Any]:
    """Corruption over baseline-correct units and ZH retention.

    Both are recomputed from text with the CURRENT source. The stored v2r3
    gate_b fields predate the unit-tagging repair and carry 6,264 / 6,251 / 13
    where the repaired computation gives 5,711 / 5,699 / 12.
    """
    from csasr.experiments.v2r3_gate_b import monolingual_retention
    from csasr.evaluation.pier import unit_status
    from . import data as D

    references = D.reference_of(pop)
    at_risk = corrupted = 0
    zh_total = zh_kept = zh_lost = 0
    for utterance, reference in references.items():
        if utterance not in baseline or utterance not in steered:
            continue
        before = unit_status(reference, baseline[utterance])
        after = unit_status(reference, steered[utterance])
        for index in set(before) | set(after):
            if not bool(before.get(index, (False, ""))[0]):
                continue
            at_risk += 1
            if not bool(after.get(index, (False, ""))[0]):
                corrupted += 1
        keep = monolingual_retention(reference, baseline[utterance], steered[utterance])
        zh_total += keep["monolingual_baseline_correct"]
        zh_kept += keep["monolingual_retained"]
        zh_lost += keep["monolingual_lost"]
    return {
        "units_at_risk": int(at_risk),
        "units_corrupted": int(corrupted),
        "zh_baseline_correct": int(zh_total),
        "zh_retained": int(zh_kept),
        "zh_lost": int(zh_lost),
        "zh_retention": (zh_kept / zh_total) if zh_total else float("nan"),
    }


def poi_decomposition(pop, hypotheses: dict[str, str]) -> dict[str, int]:
    """Deletion / wrong-language substitution / other substitution on targets."""
    from csasr.evaluation.pier import evaluate_pois
    from . import data as D

    references = D.reference_of(pop)
    counts = {"correct": 0, "deletion": 0, "wrong_language_substitution": 0,
              "other_substitution": 0, "other": 0}
    for _, t in pop.targets.iterrows():
        utterance = str(t["utterance_id"])
        if utterance not in hypotheses:
            continue
        res = evaluate_pois(references[utterance], hypotheses[utterance],
                            [int(t["reference_unit_index"])])
        if not res:
            counts["other"] += 1
            continue
        r = res[0]
        if r.correct:
            counts["correct"] += 1
        elif r.category == "deletion":
            counts["deletion"] += 1
        elif r.category in ("wrong_language_substitution",
                            "phonetic_transliteration_or_script"):
            counts["wrong_language_substitution"] += 1
        elif r.category in ("same_language_substitution", "boundary_error",
                            "insertion_near_poi"):
            counts["other_substitution"] += 1
        else:
            counts["other"] += 1
    return counts


# ---------------------------------------------------------------------------
# delta PIER with dialogue-level cluster bootstrap
# ---------------------------------------------------------------------------

def delta_pier(pop, baseline: dict[str, str], steered: dict[str, str], *,
               draws: int = C.BOOTSTRAP_DRAWS,
               seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    """delta PIER = PIER(baseline) - PIER(steered), positive means better.

    The bootstrap resamples DIALOGUES, not POIs. The prior programme's
    positive interval came from two dialogues out of twenty; an item bootstrap
    would have hidden that and a dialogue bootstrap does not.
    """
    from csasr.evaluation.bootstrap import paired_bootstrap
    from csasr.evaluation.pier import evaluate_pois
    from . import data as D

    references = D.reference_of(pop)
    dialogue_of = {str(u): str(d) for u, d in
                   zip(pop.manifest["utterance_id"], pop.manifest["dialogue_id"])}
    base_err: list[float] = []
    steer_err: list[float] = []
    clusters: list[str] = []
    for utterance, reference in references.items():
        if utterance not in baseline or utterance not in steered:
            continue
        b = evaluate_pois(reference, baseline[utterance])
        s = evaluate_pois(reference, steered[utterance])
        by_index = {r.poi_index: r for r in s}
        for r in b:
            base_err.append(0.0 if r.correct else 1.0)
            other = by_index.get(r.poi_index)
            steer_err.append(0.0 if (other is not None and other.correct) else 1.0)
            clusters.append(dialogue_of.get(utterance, utterance))
    if not base_err:
        return {"delta_pier": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_poi": 0, "n_groups": 0}
    # paired_bootstrap returns statistic(b) - statistic(a); we want
    # PIER_baseline - PIER_steered, so pass (steered, baseline)
    out = paired_bootstrap(np.asarray(steer_err), np.asarray(base_err), clusters,
                           n_resamples=draws, seed=seed,
                           alpha=1.0 - C.BOOTSTRAP_CI)
    return {
        "delta_pier": float(out["diff"]),
        "ci_low": float(out["ci_low"]),
        "ci_high": float(out["ci_high"]),
        "p_value_two_sided": float(out.get("p_value_two_sided", float("nan"))),
        "n_poi": int(len(base_err)),
        "n_groups": int(out.get("n_groups", 0)),
        "cluster": "dialogue_id",
        "draws": int(draws),
    }


# ---------------------------------------------------------------------------
# the four gates
# ---------------------------------------------------------------------------

def evaluate_gates(*, delta: dict[str, Any], outcomes: pd.DataFrame,
                   risk: dict[str, Any], health: dict[str, Any]) -> dict[str, Any]:
    """G1-G4 plus the GO / NO-GO / INVALID verdict.

    A run whose decode is unhealthy is INVALID: its PIER is not interpreted
    and it never enters a stage ranking.
    """
    corrections = int(outcomes["corrected"].sum()) if len(outcomes) else 0
    corruptions = int(risk.get("units_corrupted", 0))
    dialogues = (int(outcomes.loc[outcomes["corrected"], "dialogue_id"].nunique())
                 if len(outcomes) else 0)
    ratio = (corrections / corruptions) if corruptions else (
        float("inf") if corrections else 0.0)
    retention = float(risk.get("zh_retention", float("nan")))

    g1 = bool(delta["delta_pier"] > 0 and delta["ci_low"] > 0)
    g2 = bool(ratio >= C.GATES["G2_correction_ratio"]["threshold"])
    g3 = bool(dialogues >= C.GATES["G3_dialogue_coverage"]["threshold"])
    g4 = bool(np.isfinite(retention)
              and retention >= C.GATES["G4_zh_retention"]["threshold"])

    failed = [n for n, ok in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)) if not ok]
    if not health.get("healthy", False):
        verdict = "INVALID"
    elif not failed:
        verdict = "GO"
    else:
        verdict = "NO-GO"
    return {
        "G1_delta_pier": g1,
        "G2_correction_ratio": g2,
        "G3_dialogue_coverage": g3,
        "G4_zh_retention": g4,
        "gates_failed": failed,
        "verdict": verdict,
        "corrections": corrections,
        "corruptions": corruptions,
        "corrections_to_corruptions": float(ratio),
        "dialogues_with_correction": dialogues,
        "zh_retention": retention,
        "zh_retention_denominator": int(risk.get("zh_baseline_correct", 0)),
        "zh_retention_denominator_expected": C.ANCHOR_BASELINE_CORRECT_ZH_UNITS,
    }


# ---------------------------------------------------------------------------
# the fixed between-stage selection rule (3.6)
# ---------------------------------------------------------------------------

def select(rows: Sequence[dict[str, Any]], *, top: int = 1
           ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply SELECTION_RULE. Hard-coded; it cannot change after seeing results.

    1. drop runs with unhealthy decode
    2. drop runs with ZH retention < 0.99
    3. sort by dialogues_with_correction DESC, corrections:corruptions DESC,
       delta PIER DESC
    """
    healthy = [r for r in rows if r.get("verdict") != "INVALID"]
    dropped_unhealthy = [r["cell_id"] for r in rows if r.get("verdict") == "INVALID"]
    floor = C.GATES["G4_zh_retention"]["threshold"]
    eligible, dropped_retention = [], []
    for r in healthy:
        value = float(r.get("zh_retention") or 0.0)
        (eligible if value >= floor else dropped_retention).append(r)
    dropped_retention = [r["cell_id"] for r in dropped_retention]

    def key(r: dict[str, Any]):
        ratio = float(r.get("corrections_to_corruptions", 0.0))
        if not np.isfinite(ratio):
            ratio = float("1e18")             # zero corruption with corrections
        return (int(r.get("dialogues_with_correction", 0)), ratio,
                float(r.get("delta_pier", float("-inf")) or float("-inf")))

    ranked = sorted(eligible, key=key, reverse=True)
    return ranked[:top], {
        "n_input": len(rows),
        "dropped_unhealthy_decode": dropped_unhealthy,
        "dropped_zh_retention_below_0.99": dropped_retention,
        "n_eligible": len(eligible),
        "ranking": [(r["cell_id"], key(r)) for r in ranked],
        "rule": C.SELECTION_RULE,
    }
