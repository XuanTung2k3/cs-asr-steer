"""LSS unit test: the annotation pack ships no machine boundary, ever.

The pack's whole value is that its reference owes nothing to an aligner. Each
test here guards a way a boundary estimate could leak into it, or a way the
sample could stop being the sample it claims to be.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.lss.align import annotation_pack as pk

SR = 16000


def _unit(utt, index, language, surface, start, end, *, role="D-dev-select",
          speaker="spk1", valid=True):
    return {
        "utterance_id": utt, "reference_unit_index": index,
        "reference_language": language, "surface": surface,
        "reference_text": surface, "aligner_family": pk.PROVISIONAL_FAMILY,
        "aligner_variant": f"{pk.PROVISIONAL_FAMILY}/default",
        "start_sec": start, "end_sec": end, "is_valid": valid,
        "audio_duration_sec": 30.0, "audio_path": f"/tmp/{utt}.wav",
        "conversation_id": utt.rsplit("_", 1)[0], "speaker": speaker,
        "role": role, "sample_rate": SR,
    }


def _switch_utterance(utt, *, role="D-dev-select", speaker="spk1",
                      en_start=1.4, en_end=2.0):
    """ZH ... EN ... ZH, so the English run has both an onset and an offset."""
    return [
        _unit(utt, 0, "ZH", "我", 0.0, 1.0, role=role, speaker=speaker),
        _unit(utt, 1, "EN", "okay", en_start, en_end, role=role, speaker=speaker),
        _unit(utt, 2, "ZH", "了", 2.4, 3.0, role=role, speaker=speaker),
    ]


def test_a_textgrid_cannot_carry_a_boundary_or_a_label():
    """The writer takes tier names only -- there is no parameter to fill."""
    text = pk.textgrid_text()
    tiers = pk.parse_textgrid_tiers(text)
    assert [t["name"] for t in tiers] == list(pk.TIER_NAMES)
    for tier in tiers:
        assert tier["intervals"] == 1          # no interior boundary
        assert tier["labels"] == [""]          # no label
    # No time other than the clip's own extent appears anywhere in the file.
    times = {token for token in text.replace("=", " ").split() if token.replace(".", "").isdigit()}
    assert times <= {"0", "1", "2", "3.0"}


def test_the_manifest_never_carries_a_provisional_boundary():
    """The centring estimate lives on an underscored field and must not ship."""
    from csasr.experiments.annotation_pack_export import MANIFEST_COLUMNS

    leaked = [c for c in MANIFEST_COLUMNS
              if "provisional" in c or "center" in c or "centre" in c
              or c.endswith("_boundary_sec")]
    assert leaked == []
    assert "_provisional_center_sec" not in MANIFEST_COLUMNS


def test_annotator_facing_columns_carry_no_timing_at_all():
    """items.csv sits beside the clips, so it must be invertible into nothing."""
    timing = [c for c in pk.ANNOTATOR_COLUMNS
              if "sec" in c or "offset" in c or "start" in c or "end" in c
              or "duration" in c or "stratum" in c]
    assert timing == []


def test_the_universe_only_admits_the_permitted_role():
    frame = pd.DataFrame(
        _switch_utterance("u1")
        + _switch_utterance("u2", role="D-dev-confirm")
        + _switch_utterance("u3", role="D-construct"))
    universe, _ = pk.switch_universe(frame)
    assert set(universe["utterance_id"]) == {"u1"}


def test_a_frame_without_roles_is_refused_rather_than_guessed():
    frame = pd.DataFrame(_switch_utterance("u1")).drop(columns=["role"])
    with pytest.raises(pk.AnnotationPackError):
        pk.switch_universe(frame)


def test_both_switch_directions_carry_the_same_embedded_span():
    """An English run's onset and offset are two items about one span."""
    frame = pd.DataFrame(_switch_utterance("u1", en_start=1.4, en_end=2.0))
    universe, _ = pk.switch_universe(frame)
    assert sorted(universe["switch_direction"]) == ["EN->ZH", "ZH->EN"]
    assert universe["embedded_span_duration_sec"].nunique() == 1
    assert float(universe["embedded_span_duration_sec"].iloc[0]) == pytest.approx(0.6)
    flanking = dict(zip(universe["switch_direction"],
                        zip(universe["word_before_switch"], universe["word_after_switch"])))
    assert flanking["ZH->EN"] == ("我", "okay")
    assert flanking["EN->ZH"] == ("okay", "了")


def test_a_switch_across_a_missing_unit_is_excluded_and_counted():
    frame = pd.DataFrame([
        _unit("u1", 0, "ZH", "我", 0.0, 1.0),
        _unit("u1", 2, "EN", "okay", 1.4, 2.0),      # index 1 absent
    ])
    universe, excluded = pk.switch_universe(frame)
    assert len(universe) == 0
    assert excluded["non_contiguous_reference_unit_index"] == 1


def test_the_clip_is_centred_on_the_provisional_boundary():
    start, offset = pk.clip_bounds(10.0)
    assert start == pytest.approx(8.5)
    assert offset == pytest.approx(1.5)


def test_a_boundary_near_the_file_head_is_padded_not_slid():
    """Sliding the window would move the boundary off centre for those items."""
    samples = np.ones(SR * 5, dtype=np.float32)
    clip, pad_head, pad_tail = pk.extract_clip(samples, SR, -0.5)
    assert len(clip) == int(pk.CLIP_DURATION_SEC * SR)
    assert pad_head == pytest.approx(0.5)
    assert pad_tail == pytest.approx(0.0)
    assert clip[:int(0.5 * SR)].tolist() == [0.0] * int(0.5 * SR)


def test_double_annotation_items_are_a_random_subset_not_the_sorted_tail():
    """Flagging the tail of a sorted stratum skews agreement to some speakers.

    Two speakers alternate through the universe; a tail-flagging sampler hands
    every double-annotation item to whichever speaker sorts last.
    """
    rows: list[dict] = []
    for index in range(40):
        speaker = "spkA" if index % 2 == 0 else "spkB"
        rows += _switch_utterance(f"{speaker}_u{index:03d}", speaker=speaker)
    universe, _ = pk.switch_universe(pd.DataFrame(rows))
    pack = pk.sample_pack(universe, seed=7, n_primary=20, n_double=10)
    doubles = pack[pack["double_annotation"]]
    assert len(doubles) == 10
    assert doubles["speaker_id"].nunique() == 2


def test_the_same_seed_reproduces_the_same_items_and_a_new_seed_does_not():
    rows: list[dict] = []
    for index in range(30):
        rows += _switch_utterance(f"u{index:03d}")
    universe, _ = pk.switch_universe(pd.DataFrame(rows))
    first = pk.sample_pack(universe, seed=11, n_primary=12, n_double=6)
    again = pk.sample_pack(universe, seed=11, n_primary=12, n_double=6)
    other = pk.sample_pack(universe, seed=12, n_primary=12, n_double=6)

    def identity(frame: pd.DataFrame) -> list[tuple]:
        return list(zip(frame["utterance_id"], frame["left_reference_unit_index"],
                        frame["switch_direction"], frame["double_annotation"]))

    assert identity(first) == identity(again)
    assert identity(first) != identity(other)


def test_the_stream_key_survives_a_new_interpreter():
    """hash() is salted per process; the draw must not depend on it."""
    assert pk._stream_key("ZH->EN|short") == pk._stream_key("ZH->EN|short")
    assert pk._stream_key("ZH->EN|short") != pk._stream_key("EN->ZH|short")
    # Pinned so a future refactor cannot silently reshuffle an issued pack.
    assert pk._stream_key("ZH->EN|short") == 3625059265


def test_allocation_never_exceeds_what_a_stratum_holds():
    quota = pk.allocate(30, ["a", "b", "c"], {"a": 4, "b": 100, "c": 100})
    assert quota["a"] == 4
    assert sum(quota.values()) == 30


def test_the_readme_tells_the_annotator_to_estimate_rather_than_skip():
    text = pk.ANNOTATOR_README
    assert "best estimate" in text
    assert "skipped" in text or "skip" in text
    assert "clip" in text and "seconds" in text
    for tier in pk.TIER_NAMES:
        assert tier in text
    # It must not hand the annotator the centring trick as a hint.
    assert "midpoint" in text          # explicitly warned against
    assert "1.5" not in text
