from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from csasr.nat5h.consensus import (
    ConsensusCriteria,
    N1ConsensusGate,
    build_unit_consensus_v2,
    evaluate_n1_consensus_gate,
    summarize_alignment_smoke_v2,
)
from csasr.nat5h.coordinates import EncoderGeometry
from csasr.nat5h.schema import (
    ALIGNMENT_SCHEMA_VERSION,
    RunIdentity,
    artifact_matches_identity,
    assert_schema_v2,
    candidate_row,
    collapse_duplicate_candidates,
    collapse_token_rows_to_units,
    validate_candidate_sequence,
)
from csasr.nat5h.aligners import qwen_items_from_result, qwen_result_structure, _qwen_items_to_unit_spans
from csasr.nat5h.units import build_reference_units


def _manifest_row(utt="u1"):
    return pd.Series(
        {
            "utterance_id": utt,
            "conversation_id": "c1",
            "internal_split": "dev_select",
            "speaker_id": "s1",
            "audio_path": "/nonexistent/toy.wav",
            "duration_sec": 10.0,
        }
    )


def _unit(text="hello", idx=0, utt="u1"):
    return build_reference_units(utt, text)[idx]


def _cand(family, variant, idx=0, start=1.0, end=1.5, utt="u1", lang_text="hello"):
    geom = EncoderGeometry.from_values()
    ident = RunIdentity(code_commit="test", config_hash="cfg", model_id="m", model_revision="r")
    unit = _unit(lang_text, idx=idx, utt=utt)
    return candidate_row(
        manifest_row=_manifest_row(utt),
        unit=unit,
        aligner_family=family,
        aligner_variant=variant,
        start_sec=start,
        end_sec=end,
        geometry=geom,
        identity=ident,
        mapping_method="toy_non_scientific",
    )


def test_schema_v2_reference_unit_keys_are_stable_and_required():
    r = _cand("existing_ctc", "existing_ctc/default")
    df = pd.DataFrame([r])
    assert_schema_v2(df)
    assert (r["utterance_id"], r["reference_unit_index"]) == ("u1", 0)
    assert r["schema_version"] == ALIGNMENT_SCHEMA_VERSION
    with pytest.raises(ValueError):
        assert_schema_v2(pd.DataFrame([{"utterance_id": "old", "unit_id": 0}]))


def test_dtw_subwords_collapse_and_coverage_cannot_exceed_one():
    refs = build_reference_units("u1", "hello world")
    raw = pd.DataFrame(
        [
            {"reference_unit_index": 0, "start_sec": 0.0, "end_sec": 0.2},
            {"reference_unit_index": 0, "start_sec": 0.2, "end_sec": 0.5},
            {"reference_unit_index": None, "start_sec": 0.5, "end_sec": 0.6},
            {"reference_unit_index": 1, "start_sec": 0.7, "end_sec": 1.0},
        ]
    )
    collapsed, stats = collapse_token_rows_to_units(raw, refs)
    assert len(collapsed) == 2
    assert collapsed.loc[collapsed["reference_unit_index"] == 0, "source_token_count"].iloc[0] == 2
    assert stats["duplicate_mappings_collapsed"] == 1
    assert stats["unmapped_tokens"] == 1
    coverage = stats["unique_mapped_reference_units"] / len(refs)
    assert 0 <= coverage <= 1


def test_duplicate_candidate_rows_collapse_to_one_variant_vote():
    a = _cand("whisper_dtw", "whisper_dtw/zh_median7", start=1.0, end=1.3)
    b = _cand("whisper_dtw", "whisper_dtw/zh_median7", start=1.1, end=1.4)
    out, stats = collapse_duplicate_candidates(pd.DataFrame([a, b]))
    assert len(out) == 1
    assert stats["duplicates_collapsed"] == 1


def test_invalid_timestamp_order_and_large_overlap_are_rejected():
    bad = _cand("whisper_dtw", "whisper_dtw/zh_median7", start=2.0, end=1.0)
    assert bad["is_valid"] is False
    assert bad["failure_code"] == "reversed_or_zero_duration"
    a = _cand("whisper_dtw", "whisper_dtw/zh_median7", idx=0, start=1.0, end=2.0, lang_text="hello world")
    b = _cand("whisper_dtw", "whisper_dtw/zh_median7", idx=1, start=1.5, end=2.5, lang_text="hello world")
    seq = validate_candidate_sequence(pd.DataFrame([a, b]))
    assert bool(seq.iloc[1]["is_valid"]) is False
    assert seq.iloc[1]["failure_code"] == "adjacent_overlap"


@dataclass
class FakeQwenItem:
    text: str
    start_time: float
    end_time: float


@dataclass
class FakeQwenResult:
    items: list[FakeQwenItem]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_qwen_nested_output_is_parsed_and_empty_is_explicit():
    nested = [FakeQwenResult([FakeQwenItem("hello", 0.1, 0.4), FakeQwenItem("你", 0.5, 0.7)])]
    diag = qwen_result_structure(nested)
    assert diag["return_outer_type"] == "list"
    assert diag["outer_length"] == 1
    items = qwen_items_from_result(nested)
    assert len(items) == 2
    spans = _qwen_items_to_unit_spans(items)
    assert spans[0]["surface"] == "hello"
    assert qwen_items_from_result([]) == []


def test_qwen_variants_do_not_count_as_two_consensus_votes():
    ctc = _cand("existing_ctc", "existing_ctc/default", start=1.0, end=1.5)
    qzh = _cand("qwen_forced_aligner", "qwen_forced_aligner/Chinese", start=1.05, end=1.55)
    qen = _cand("qwen_forced_aligner", "qwen_forced_aligner/English", start=1.04, end=1.54)
    consensus, rejected, _ = build_unit_consensus_v2(pd.DataFrame([ctc, qzh, qen]), ConsensusCriteria(safe_interior_erosion_ms=0, min_safe_interior_ms=1), reference_units_df=pd.DataFrame([{"utterance_id": "u1", "unit_id": 0, "surface": "hello", "language": "EN"}]))
    assert len(consensus) == 1
    assert consensus.iloc[0]["num_votes"] == 2
    assert consensus.iloc[0]["selected_aligner_families"] == ["existing_ctc", "qwen_forced_aligner"]


def test_ctc_valid_dtw_span_can_pass_when_other_dtw_rows_fail_globally():
    good_ctc = _cand("existing_ctc", "existing_ctc/default", idx=0, start=1.00, end=1.50, lang_text="hello world")
    good_dtw = _cand("whisper_dtw", "whisper_dtw/zh_median7", idx=0, start=1.10, end=1.55, lang_text="hello world")
    bad_dtw = _cand("whisper_dtw", "whisper_dtw/zh_median7", idx=1, start=1.20, end=2.00, lang_text="hello world")
    bad_dtw["is_valid"] = False
    bad_dtw["failure_code"] = "adjacent_overlap"
    refs = pd.DataFrame(
        [
            {"utterance_id": "u1", "unit_id": 0, "surface": "hello", "language": "EN"},
            {"utterance_id": "u1", "unit_id": 1, "surface": "world", "language": "EN"},
        ]
    )
    consensus, rejected, _ = build_unit_consensus_v2(pd.DataFrame([good_ctc, good_dtw, bad_dtw]), ConsensusCriteria(safe_interior_erosion_ms=0, min_safe_interior_ms=1), reference_units_df=refs)
    assert len(consensus) == 1
    assert consensus.iloc[0]["reference_unit_index"] == 0
    assert "missing_second_aligner" in set(rejected["rejection_code"])


def test_200ms_threshold_rejects_and_three_aligner_tie_is_deterministic():
    ctc = _cand("existing_ctc", "existing_ctc/default", start=1.00, end=1.50)
    dtw_far = _cand("whisper_dtw", "whisper_dtw/zh_median7", start=1.30, end=1.55)
    consensus, rejected, _ = build_unit_consensus_v2(pd.DataFrame([ctc, dtw_far]), ConsensusCriteria(safe_interior_erosion_ms=0, min_safe_interior_ms=1), reference_units_df=pd.DataFrame([{"utterance_id": "u1", "unit_id": 0, "surface": "hello", "language": "EN"}]))
    assert len(consensus) == 0
    assert rejected.iloc[0]["rejection_code"] == "start_disagreement_gt_200ms"

    dtw = _cand("whisper_dtw", "whisper_dtw/zh_median7", start=1.05, end=1.55)
    qwen = _cand("qwen_forced_aligner", "qwen_forced_aligner/Chinese", start=1.08, end=1.52)
    consensus, _, _ = build_unit_consensus_v2(pd.DataFrame([qwen, dtw, ctc]), ConsensusCriteria(safe_interior_erosion_ms=0, min_safe_interior_ms=1), reference_units_df=pd.DataFrame([{"utterance_id": "u1", "unit_id": 0, "surface": "hello", "language": "EN"}]))
    assert consensus.iloc[0]["selected_aligner_families"] == ["existing_ctc", "whisper_dtw", "qwen_forced_aligner"]


def test_n1_diagnostics_are_not_the_n2_gate_and_old_schema_cache_rejected():
    ctc = _cand("existing_ctc", "existing_ctc/default", start=1.00, end=1.50)
    dtw = _cand("whisper_dtw", "whisper_dtw/zh_median7", start=1.05, end=1.55)
    refs = pd.DataFrame([{"utterance_id": "u1", "unit_id": 0, "surface": "hello", "language": "EN"}])
    metrics = summarize_alignment_smoke_v2(pd.DataFrame([ctc, dtw]), refs)
    assert metrics["unit_coverage"].max() <= 1.0
    consensus, _, _ = build_unit_consensus_v2(pd.DataFrame([ctc, dtw]), ConsensusCriteria(safe_interior_erosion_ms=0, min_safe_interior_ms=1), reference_units_df=refs)
    state, evidence = evaluate_n1_consensus_gate(consensus, pd.DataFrame([ctc, dtw]), N1ConsensusGate(min_consensus_utterances=1, min_consensus_spans=1, min_consensus_en_spans=1, min_consensus_zh_spans=0))
    assert state == "passed"
    assert not artifact_matches_identity(pd.DataFrame([{"utterance_id": "old"}]), RunIdentity("test", "cfg"))
    assert artifact_matches_identity(pd.DataFrame([ctc]), RunIdentity("test", "cfg"))
