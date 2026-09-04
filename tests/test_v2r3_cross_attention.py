"""Cross-attention pseudo-label spans: the frame-to-span logic and its guards.

The forward pass needs a GPU, but everything that decides *what a span is* is
pure and is tested here: the token classes, the smoothing, the span rule, and
the overlap matching used for the three-way comparison.
"""
from __future__ import annotations

import pandas as pd
import pytest

from csasr.experiments import v2r3_cross_attention_spans as cas


def test_frozen_configuration_is_recorded_and_declares_the_method():
    frozen = cas.FROZEN_CONFIG
    assert frozen["decoder_layer_index"] == -1
    assert frozen["head_set"] == "all_heads_mean"
    assert frozen["smoothing"]["window_frames"] == 9
    assert frozen["min_span"]["frames"] == 5
    # the method claims that must remain true of this module
    assert frozen["uses_dtw"] is False
    assert frozen["requires_monotonic_path"] is False
    assert frozen["parameters_updated"] is False
    assert frozen["teacher_forced"] is True
    # the minimum span must not pre-empt the smallest eligibility tier
    assert frozen["min_span"]["ms"] < 200


def test_module_contains_no_dtw_and_no_parameter_update():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(cas.__file__).read_text(encoding="utf-8"))
    called = {getattr(n.func, "id", getattr(n.func, "attr", ""))
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for banned in ("dtw", "backward", "step", "train", "requires_grad_"):
        assert banned not in called, f"module must not call {banned}"


class _Tokenizer:
    """Minimal stand-in: ids 0-3 special, 10 English, 20 Mandarin, 30 punctuation."""

    all_special_ids = [0, 1, 2, 3]

    def decode(self, ids, skip_special_tokens=False):
        table = {0: "<|startoftranscript|>", 1: "<|zh|>", 2: "<|transcribe|>",
                 3: "<|notimestamps|>", 10: " project", 20: "我", 30: " ,"}
        return "".join(table.get(int(i), "") for i in ids)


def test_special_tokens_are_their_own_class_never_a_language():
    classes = cas.token_classes(_Tokenizer(), [0, 1, 2, 3, 10, 20, 30])
    assert classes[:4] == ["special"] * 4
    assert classes[4] == "EN"
    assert classes[5] == "ZH"
    assert classes[6] == "neutral"          # punctuation, not silently a language
    assert "special" not in cas.LANGUAGE_CLASSES
    assert "neutral" not in cas.LANGUAGE_CLASSES


def test_mode_filter_removes_single_frame_flicker_but_keeps_a_real_switch():
    labels = ["ZH"] * 10 + ["EN"] * 10
    labels[4] = "EN"                                   # one-frame flicker
    smoothed = cas.mode_filter(labels, 9)
    assert smoothed[4] == "ZH"
    assert smoothed[:4] == ["ZH"] * 4
    assert smoothed[-4:] == ["EN"] * 4                 # the real switch survives


def test_spans_require_the_minimum_length_and_ignore_non_language_runs():
    step = 0.02
    labels = (["ZH"] * 10 + ["special"] * 10 + ["EN"] * 3 + ["ZH"] * 10)
    spans = cas.spans_from_labels(labels, step_sec=step)
    languages = [s["language"] for s in spans]
    assert languages == ["ZH", "ZH"]                   # the 3-frame EN run is dropped
    assert all(s["n_frames"] >= cas.MIN_SPAN_FRAMES for s in spans)
    assert spans[0]["start_sec"] == 0.0
    assert spans[0]["end_sec"] == pytest.approx(10 * step)
    # a special stretch never becomes a span and never merges the two ZH runs
    assert spans[0]["end_sec"] < spans[1]["start_sec"]


def test_overlap_matching_is_one_to_one_and_prefers_the_larger_overlap():
    left = pd.DataFrame([{"start_sec": 0.0, "end_sec": 1.0},
                         {"start_sec": 2.0, "end_sec": 3.0}])
    right = pd.DataFrame([{"start_sec": 0.9, "end_sec": 2.9},   # overlaps both
                          {"start_sec": 0.0, "end_sec": 0.95}])
    pairs = cas._match_by_overlap(left, right)
    assert len(pairs) == 2
    assert len({i for i, _ in pairs}) == 2 and len({j for _, j in pairs}) == 2
    assert (0, 1) in pairs        # left[0] takes right[1], the larger overlap


def test_comparison_reports_disagreement_not_error():
    pseudo = pd.DataFrame([{"utterance_id": "u1", "language": "EN",
                            "start_sec": 1.00, "end_sec": 2.00}])
    other = pd.DataFrame([{"utterance_id": "u1", "language": "EN",
                           "start_sec": 1.10, "end_sec": 2.05}])
    result = cas.compare_paths(pseudo, other, "existing_ctc/blank_excluded")
    assert result["n"] == 1
    assert result["median_boundary_ms"] == pytest.approx(100.0)
    assert result["measurement"] == "cross_aligner_disagreement"
    assert "never error against truth" in result["note"]


def test_comparison_skips_utterances_only_one_path_found():
    pseudo = pd.DataFrame([{"utterance_id": "u1", "language": "EN",
                            "start_sec": 0.0, "end_sec": 1.0}])
    other = pd.DataFrame([{"utterance_id": "u2", "language": "EN",
                           "start_sec": 0.0, "end_sec": 1.0}])
    assert cas.compare_paths(pseudo, other, "x")["n"] == 0


def test_destination_guards(tmp_path):
    existing = tmp_path / "gen"
    existing.mkdir()
    with pytest.raises(SystemExit, match="existing destination"):
        cas.resolve_output(existing)
    with pytest.raises(SystemExit, match="v1 root"):
        cas.resolve_output("/mnt/data/x/artifacts_lss/spans")
    assert cas.resolve_output(tmp_path / "fresh") == (tmp_path / "fresh").resolve()
