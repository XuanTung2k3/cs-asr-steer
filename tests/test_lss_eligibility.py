"""LSS unit test: only *embedded* English counts toward the primary endpoint."""
from __future__ import annotations

import pandas as pd

from csasr.lss.eligibility import (
    conversation_unit_counts,
    embedded_english_units,
    poi_index_consistency,
    unit_table,
    utterance_units,
)

MIXED = "我们 用 machine learning 做 这个"
EN_ONLY = "we use machine learning for this"
ZH_ONLY = "我们 用 这个 做 实验"


def _manifest(rows):
    from csasr.data.normalize import normalize_text

    return pd.DataFrame([{
        "utterance_id": uid,
        "conversation_id": uid.rsplit("_", 1)[0],
        "speaker_id": uid.rsplit("_", 1)[0],
        "transcript_raw": text,
        "transcript_normalized": normalize_text(text),
        "duration_sec": 4.0,
        "contains_code_switch": ("machine" in text and "我" in text),
        "gender": "F", "device": "mic", "region": "north", "topic": "study",
    } for uid, text in rows])


def test_monolingual_english_contributes_no_eligible_unit():
    """P0's endpoint counted these: 73% of dev_select's 'English POIs' were
    monolingual-English utterances, not code-switches."""
    units = utterance_units("u1", EN_ONLY)
    assert (units["language"] == "EN").sum() > 0
    assert units["is_embedded_english"].sum() == 0


def test_code_switched_utterance_yields_its_english_units():
    units = utterance_units("u2", MIXED)
    embedded = units[units["is_embedded_english"]]
    assert set(embedded["surface"]) == {"machine", "learning"}
    assert bool(units["utterance_contains_code_switch"].all())


def test_mandarin_only_yields_nothing():
    units = utterance_units("u3", ZH_ONLY)
    assert units["is_embedded_english"].sum() == 0
    assert (units["language"] == "ZH").sum() > 0


def test_embedded_english_units_filters_a_manifest():
    man = _manifest([("c1_1", MIXED), ("c1_2", EN_ONLY), ("c1_3", ZH_ONLY)])
    eligible = embedded_english_units(man)
    assert set(eligible["utterance_id"]) == {"c1_1"}
    assert len(eligible) == 2


def test_unit_ids_are_dense_and_ordered():
    units = utterance_units("u4", MIXED)
    assert list(units["unit_id"]) == list(range(len(units)))


def test_poi_index_equals_unit_id_on_real_shaped_transcripts():
    """Every join between a POI table and a span table depends on this."""
    man = _manifest([("c1_1", MIXED), ("c1_2", EN_ONLY), ("c1_3", ZH_ONLY),
                     ("c1_4", "他 说 ok 然后 就 走 了"),
                     ("c1_5", "这个 project 的 deadline 是 明天")])
    report = poi_index_consistency(man)
    assert report["utterances_checked"] == 5
    assert report["agreement_rate"] == 1.0, report["mismatch_examples"]


def test_conversation_counts_carry_the_covariates_roles_are_balanced_on():
    man = _manifest([("c1_1", MIXED), ("c1_2", ZH_ONLY), ("c2_1", MIXED)])
    counts = conversation_unit_counts(man)
    assert set(counts["conversation_id"]) == {"c1", "c2"}
    c1 = counts[counts["conversation_id"] == "c1"].iloc[0]
    assert c1["utterances"] == 2
    assert c1["embedded_en_units"] == 2
    assert 0.0 <= c1["cs_rate"] <= 1.0
    for col in ("hours", "mean_duration_sec", "cs_utterances", "gender", "device"):
        assert col in counts.columns


def test_unit_table_is_empty_for_an_empty_manifest():
    empty = unit_table(pd.DataFrame(columns=["utterance_id", "transcript_raw"]))
    assert len(empty) == 0
