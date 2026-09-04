"""Round-1 paired metrics with explicit baseline/protocol accounting."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def pier_transition_counts(references: Sequence[str], baseline: Sequence[str],
                           method: Sequence[str]) -> dict[str, int]:
    from csasr.evaluation.pier import evaluate_pois, reference_pois
    from csasr.evaluation.normalization import normalize_text
    n = c = h = 0
    for ref, b, m in zip(references, baseline, method):
        # reference_pois expects the already-normalized reference.  The
        # existing corpus `pier()` path normalizes inside evaluate_pois(); use
        # the same normalized surface here so the transition denominator and
        # the codebase PIER denominator are identical.
        indices = [i for i, _ in reference_pois(normalize_text(ref))]
        br = {x.poi_index: x.correct for x in evaluate_pois(ref, b, indices)}
        mr = {x.poi_index: x.correct for x in evaluate_pois(ref, m, indices)}
        for i in indices:
            if i not in br or i not in mr:
                continue
            n += 1
            c += int(not br[i] and mr[i])
            h += int(br[i] and not mr[i])
    return {"num_poi": n, "baseline_wrong_method_correct": c,
            "baseline_correct_method_wrong": h}


def assert_pier_identity(baseline_pier: float, method_pier: float,
                         transitions: dict[str, int], tol: float = 1e-10) -> None:
    n = int(transitions["num_poi"])
    if n == 0:
        return
    expected = (transitions["baseline_correct_method_wrong"] -
                transitions["baseline_wrong_method_correct"]) / n
    actual = float(method_pier - baseline_pier)
    if abs(actual - expected) > tol:
        raise AssertionError(f"PIER transition identity failed: {actual} != {expected}")


def paired_metric_report(references: Sequence[str], baseline: Sequence[str],
                         method: Sequence[str], *, beam: int, baseline_protocol: str,
                         method_protocol: str, utterance_ids: Sequence[str] | None = None,
                         dialogue_ids: Sequence[str] | None = None) -> dict[str, Any]:
    if baseline_protocol != method_protocol:
        raise ValueError("paired comparisons require matching decode protocols")
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier
    mer_b = corpus_mer(references, baseline)
    mer_m = corpus_mer(references, method)
    pier_b = pier(references, baseline)
    pier_m = pier(references, method)
    transitions = pier_transition_counts(references, baseline, method)
    assert_pier_identity(pier_b["pier"], pier_m["pier"], transitions)
    # Transcript-level aggregation is intentionally one row per utterance.
    from csasr.evaluation.mer import mer
    wins = ties = losses = changed = 0
    distances = []
    for r, b, m in zip(references, baseline, method):
        eb, em = mer(r, b)["rate"], mer(r, m)["rate"]
        wins += int(em < eb); ties += int(em == eb); losses += int(em > eb)
        distances.append(_edit_distance(_units(b), _units(m)))
        changed += int(_units(b) != _units(m))
    n_utt = max(1, len(references))
    return {
        "beam": int(beam), "baseline_protocol": baseline_protocol,
        "method_protocol": method_protocol, "scored_utterances": len(references),
        "MER": float(mer_m["mer"]), "baseline_MER": float(mer_b["mer"]),
        "delta_MER": float(mer_m["mer"] - mer_b["mer"]),
        "substitutions": int(mer_m["substitutions"]),
        "deletions": int(mer_m["deletions"]), "insertions": int(mer_m["insertions"]),
        "EN_WER": float(mer_m["en_wer"]), "ZH_CER": float(mer_m["zh_cer"]),
        "baseline_PIER": float(pier_b["pier"]), "PIER": float(pier_m["pier"]),
        "delta_PIER": float(pier_m["pier"] - pier_b["pier"]),
        "num_poi": int(transitions["num_poi"]),
        "corrections": int(transitions["baseline_wrong_method_correct"]),
        "corruptions": int(transitions["baseline_correct_method_wrong"]),
        "net_corrections": int(transitions["baseline_wrong_method_correct"] - transitions["baseline_correct_method_wrong"]),
        "outside_edit_rate": float(sum(distances) / n_utt),
        "unique_utterances_changed_pct": float(100.0 * changed / n_utt),
        "mean_baseline_to_method_edit_distance": float(np.mean(distances)) if distances else 0.0,
        "utterance_wins": wins, "utterance_ties": ties, "utterance_losses": losses,
        "pier_identity_residual": float(pier_m["pier"] - pier_b["pier"] -
            (transitions["baseline_correct_method_wrong"] - transitions["baseline_wrong_method_correct"]) / max(1, transitions["num_poi"])),
        "utterance_ids": list(utterance_ids or []), "dialogue_ids": list(dialogue_ids or []),
    }


def _units(text: str) -> list[str]:
    from csasr.data.normalize import segment_units, normalize_text
    return [u.surface for u in segment_units(normalize_text(text))]


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]
