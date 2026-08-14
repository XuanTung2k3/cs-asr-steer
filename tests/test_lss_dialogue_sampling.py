"""Dialogue-stratified L1b sampling semantics."""
from __future__ import annotations

import pandas as pd
import pytest

from csasr.experiments import lss_l1b_valid


def _manifest(counts: dict[str, int], *, false_dialogues=()) -> pd.DataFrame:
    rows = []
    for dialogue in sorted(counts):
        for index in range(counts[dialogue]):
            rows.append({
                "utterance_id": f"{dialogue}_u{index:03d}",
                "dialogue_id": dialogue,
                "contains_code_switch": dialogue not in false_dialogues,
            })
    return pd.DataFrame(rows)


def _design(*, dialogues="all", k=3, seed=17) -> dict:
    return {
        "sample_dialogues": dialogues,
        "utterances_per_dialogue": k,
        "sample_stratify_field": "dialogue_id",
        "sample_seed": seed,
    }


def test_realized_dialogue_count_matches_the_configured_cluster_count():
    frame = _manifest({f"d{i}": 5 for i in range(6)})
    sample, report = lss_l1b_valid.dialogue_stratified_sample(
        frame, _design(dialogues=4, k=2), role="D-construct")

    assert report["available_dialogues"] == 6
    assert report["selected_dialogues"] == 4
    assert sample["dialogue_id"].nunique() == 4
    assert len(sample) == 8


def test_underpopulated_dialogue_is_not_backfilled_from_another_dialogue():
    frame = _manifest({"small": 1, "large": 8})
    sample, report = lss_l1b_valid.dialogue_stratified_sample(
        frame, _design(k=3), role="D-dev-select")

    counts = sample.groupby("dialogue_id").size().to_dict()
    assert counts == {"large": 3, "small": 1}
    assert len(sample) == 4
    assert report["underpopulated_dialogues"] == 1
    assert {r["dialogue_id"]: r["utterance_shortfall"]
            for r in report["per_dialogue"]} == {"large": 0, "small": 2}


def test_seed_is_reproducible_input_order_independent_and_changes_the_draw():
    frame = _manifest({f"d{i}": 8 for i in range(8)})
    first, _ = lss_l1b_valid.dialogue_stratified_sample(
        frame, _design(dialogues=5, k=3, seed=23), role="loc-train")
    reordered = frame.sample(frac=1.0, random_state=999).reset_index(drop=True)
    again, _ = lss_l1b_valid.dialogue_stratified_sample(
        reordered, _design(dialogues=5, k=3, seed=23), role="loc-train")
    other, _ = lss_l1b_valid.dialogue_stratified_sample(
        frame, _design(dialogues=5, k=3, seed=24), role="loc-train")

    assert first["utterance_id"].tolist() == again["utterance_id"].tolist()
    assert first["utterance_id"].tolist() != other["utterance_id"].tolist()


def test_code_switch_filter_is_applied_before_dialogue_selection():
    frame = _manifest({"not-eligible": 20, "eligible": 4},
                      false_dialogues={"not-eligible"})
    sample, report = lss_l1b_valid.dialogue_stratified_sample(
        frame, _design(dialogues="all", k=3), role="util-train")

    assert report["available_dialogues"] == 1
    assert report["eligible_utterances"] == 4
    assert set(sample["dialogue_id"]) == {"eligible"}
    assert len(sample) == 3


def test_fewer_dialogues_than_requested_uses_all_and_reports_shortfall():
    frame = _manifest({"d0": 5, "d1": 5})
    sample, report = lss_l1b_valid.dialogue_stratified_sample(
        frame, _design(dialogues=5, k=2), role="router-calib")

    assert report["requested_dialogues"] == 5
    assert report["selected_dialogues"] == 2
    assert report["dialogue_shortfall"] == 3
    assert sample["dialogue_id"].nunique() == 2
    assert len(sample) == 4


def test_old_total_utterance_config_fails_loudly_even_if_new_fields_exist():
    cfg = {"alignment": {"diagnostics": {
        **_design(), "sample_utterances": 300,
    }}}
    with pytest.raises(ValueError, match="sample_utterances is forbidden"):
        lss_l1b_valid._sampling_config(cfg)


def test_missing_dialogue_id_refuses_a_contiguous_fallback():
    frame = _manifest({"d0": 5}).drop(columns="dialogue_id")
    with pytest.raises(ValueError, match="missing column.*dialogue_id"):
        lss_l1b_valid.dialogue_stratified_sample(
            frame, _design(), role="D-construct")
