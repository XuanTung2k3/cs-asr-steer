"""E1 unit test 1: split speakers must not overlap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.data.manifest import assert_no_test_data, parse_utterance_id
from csasr.data.splits import (
    assign_internal_splits,
    build_direction_subsets,
    split_dev_speakers,
    validate_splits,
)


def _fake_manifest(n_train_spk=8, n_dev_spk=6, per_spk=5):
    rows = []
    for split, n in (("train", n_train_spk), ("dev", n_dev_spk)):
        for s in range(n):
            spk = f"ZH-CN_U{s:04d}_S{0 if split == 'train' else 1}"
            for u in range(per_spk):
                rows.append({
                    "utterance_id": f"{spk}_{u}",
                    "conversation_id": spk.rsplit("_", 1)[0],
                    "speaker_id": spk,
                    "audio_path": f"/tmp/{spk}_{u}.wav",
                    "sample_rate": 16000,
                    "duration_sec": 3.0 + u,
                    "official_split": split,
                    "internal_split": "",
                    "transcript_raw": "我 like 这个 project",
                    "transcript_normalized": "我 like 这个 project",
                    "contains_en": True,
                    "contains_zh": True,
                    "contains_code_switch": True,
                    "audio_sha256": "",
                })
    return pd.DataFrame(rows)


def test_dev_split_is_deterministic_and_disjoint():
    speakers = [f"spk{i:02d}" for i in range(30)]
    a1, b1 = split_dev_speakers(speakers, n_select=20, seed=42)
    a2, b2 = split_dev_speakers(list(reversed(speakers)), n_select=20, seed=42)
    assert a1 == a2 and b1 == b2, "split must not depend on input order"
    assert len(a1) == 20 and len(b1) == 10
    assert not set(a1) & set(b1)


def test_internal_splits_have_no_speaker_leakage():
    df = _fake_manifest()
    work = assign_internal_splits(df, n_select=4, seed=42)
    subsets = build_direction_subsets(work, pilot_size=10, seed=42,
                                      bootstrap_seeds=(42, 43))
    report = validate_splits(work, subsets)
    assert report["overlap_train_dev_select"] == []
    assert report["overlap_train_dev_confirm"] == []
    assert report["overlap_dev_select_dev_confirm"] == []
    assert report["dev_select"]["speakers"] == 4
    assert report["dev_confirm"]["speakers"] == 2


def test_direction_subsets_exclude_dev_speakers():
    df = _fake_manifest()
    work = assign_internal_splits(df, n_select=4, seed=42)
    subsets = build_direction_subsets(work, pilot_size=10, seed=42, bootstrap_seeds=(42,))
    dev_speakers = set(work.loc[work["internal_split"].str.startswith("dev"), "speaker_id"])
    for name, sub in subsets.items():
        assert not set(sub["speaker_id"]) & dev_speakers, name
        assert (sub["official_split"] == "train").all()


def test_test_split_is_mechanically_blocked():
    df = _fake_manifest()
    df.loc[0, "official_split"] = "test"
    with pytest.raises(RuntimeError, match="prohibited"):
        assert_no_test_data(df)


def test_utterance_id_parsing():
    conv, spk = parse_utterance_id("ZH-CN_U0001_S0_42")
    assert conv == "ZH-CN_U0001"
    assert spk == "ZH-CN_U0001_S0"
