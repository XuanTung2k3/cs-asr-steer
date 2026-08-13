"""Regression tests for the post-job-38745 Gate-A correctness repairs."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from csasr.experiments import lss_l1b_valid
from csasr.lss.align import autoevidence, synthetic
from csasr.lss.align import target_objects as target


def _row(family, unit, language, start, end, *, valid=True, failure=""):
    return {
        "aligner_family": family, "aligner_variant": f"{family}/default",
        "utterance_id": "u1", "reference_unit_index": unit,
        "reference_language": language, "reference_text": str(unit),
        "start_sec": start, "end_sec": end,
        "start_sample": int(start * 16000) if np.isfinite(start) else -1,
        "end_sample": int(end * 16000) if np.isfinite(end) else -1,
        "audio_duration_sec": 4.0, "is_valid": valid,
        "failure_code": failure, "role": "D-construct",
    }


def _good_family(family="existing_ctc"):
    return pd.DataFrame([
        _row(family, 0, "ZH", 0.0, .4),
        _row(family, 1, "ZH", .4, .8),
        _row(family, 2, "EN", .8, 1.3),
        _row(family, 3, "EN", 1.3, 1.8),
        _row(family, 4, "ZH", 1.8, 2.2),
    ])


def test_zero_duration_is_invalid_but_not_automatically_nonmonotonic():
    frame = _good_family()
    frame.loc[1, ["start_sec", "end_sec", "start_sample", "end_sample"]] = [.4, .4, 6400, 6400]
    frame.loc[1, ["is_valid", "failure_code"]] = [False, "reversed_or_zero_duration"]
    raw = autoevidence.family_validity(frame).iloc[0]
    assert raw["invalid_duration_units"] == 1
    assert raw["nonmonotonic_units"] == 0
    assert raw["nonmonotonic_denominator"] == len(frame) - 1


def test_nonmonotonicity_uses_only_positive_duration_objects():
    frame = _good_family()
    frame.loc[2, ["start_sec", "end_sec", "start_sample", "end_sample"]] = [.1, .3, 1600, 4800]
    raw = autoevidence.family_validity(frame).iloc[0]
    assert raw["invalid_duration_units"] == 0
    assert raw["nonmonotonic_units"] == 1


def test_internal_zero_duration_retains_run_edges_and_raw_failure():
    frame = pd.DataFrame([
        _row("qwen_forced_aligner", 0, "ZH", 0.0, .3),
        _row("qwen_forced_aligner", 1, "ZH", .3, .3,
             valid=False, failure="reversed_or_zero_duration"),
        _row("qwen_forced_aligner", 2, "ZH", .3, .8),
        _row("qwen_forced_aligner", 3, "EN", .8, 1.2),
    ])
    objects, runs, _ = target.target_objects(frame)
    zh = runs[runs["language"] == "ZH"].iloc[0]
    assert bool(zh["target_valid"])
    assert zh["internal_invalid_duration_units"] == 1
    assert zh["raw_invalid_units"] == 1
    assert objects["raw_invalid_units"].sum() == 1


def test_invalid_unit_on_switch_boundary_invalidates_target():
    frame = pd.DataFrame([
        _row("qwen_forced_aligner", 0, "ZH", 0.0, .3),
        _row("qwen_forced_aligner", 1, "ZH", .3, .3,
             valid=False, failure="reversed_or_zero_duration"),
        _row("qwen_forced_aligner", 2, "EN", .3, .8),
    ])
    _, runs, _ = target.target_objects(frame)
    assert not bool(runs[runs["language"] == "ZH"].iloc[0]["target_valid"])
    assert not bool(runs[runs["language"] == "EN"].iloc[0]["target_valid"])
    assert runs["switch_adjacent_invalid_duration_units"].sum() == 1
    assert runs["utterance_outer_invalid_duration_units"].sum() == 0


def test_same_language_overlap_merges_but_cross_language_overlap_does_not():
    same = pd.DataFrame([
        _row("whisper_dtw", 0, "ZH", 0.0, .5),
        _row("whisper_dtw", 1, "ZH", .4, .8,
             valid=False, failure="adjacent_overlap"),
        _row("whisper_dtw", 2, "EN", .8, 1.2),
    ])
    _, runs, counts = target.target_objects(same)
    assert counts["same_language_overlap_merged"] == 1
    assert bool(runs["target_valid"].all())
    assert int(runs["raw_invalid_units"].sum()) == 1

    cross = same.copy()
    cross.loc[2, ["start_sec", "start_sample"]] = [.7, 11200]
    _, runs, counts = target.target_objects(cross)
    assert counts["cross_language_overlap_invalid"] == 1
    assert not bool(runs["target_valid"].all())


def test_blocker_reason_depends_on_qualification_not_raw_overlap():
    targets, _, _ = target.target_objects(pd.concat([
        _good_family("existing_ctc"), _good_family("whisper_dtw")],
        ignore_index=True))
    # Only one family is declared qualifying despite complete raw overlap.
    result = target.select_pair(
        targets, ["existing_ctc"], [["existing_ctc", "whisper_dtw"]],
        min_count=1, min_rate=.9)
    assert result["reason"] == "fewer_than_two_qualifying_families"


def test_two_qualifying_families_with_low_pairing_get_paired_blocker_reason():
    objects, _, _ = target.target_objects(pd.concat([
        _good_family("existing_ctc"), _good_family("whisper_dtw")],
        ignore_index=True))
    result = target.select_pair(
        objects, ["existing_ctc", "whisper_dtw"],
        [["existing_ctc", "whisper_dtw"]], min_count=100, min_rate=.9)
    assert result["reason"] == "insufficient_paired_target_overlap"


def test_pair_selection_is_config_order_not_best_error():
    objects, _, _ = target.target_objects(pd.concat([
        _good_family("existing_ctc"), _good_family("whisper_dtw"),
        _good_family("qwen_forced_aligner")], ignore_index=True))
    result = target.select_pair(
        objects, ["existing_ctc", "whisper_dtw", "qwen_forced_aligner"],
        [["existing_ctc", "whisper_dtw"],
         ["existing_ctc", "qwen_forced_aligner"]], min_count=3, min_rate=.9)
    assert result["selected_pair"] == ["existing_ctc", "whisper_dtw"]
    assert "error" not in result["selection_rule"]


def test_no_natural_pair_cannot_unlock_lexical_calibration(tmp_path):
    status = autoevidence.synthetic_calibration_status(
        tmp_path, selected_pair=[], require_authentication=False)
    assert not status["available"]
    assert status["reason"] == autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS


def test_candidate_pool_must_exceed_usable_boundary_requirement():
    cfg = {"synthetic": {"num_pairs_dev": 100, "num_pairs_gate": 100},
           "gate_a": {"min_synthetic_boundaries": 100,
                      "data_sufficiency_universe": "full_l0_role_manifest"}}
    with pytest.raises(ValueError, match="must exceed"):
        lss_l1b_valid._validate_gate_configuration(cfg)


def test_unreachable_sampled_data_requirement_is_rejected():
    cfg = {"synthetic": {"num_pairs_dev": 150, "num_pairs_gate": 150},
           "alignment": {"diagnostics": {"sample_utterances": 300}},
           "gate_a": {"min_synthetic_boundaries": 100,
                      "data_sufficiency_universe": "audit_sample",
                      "min_construct_bilingual_utterances": 500}}
    with pytest.raises(ValueError, match="300.*500"):
        lss_l1b_valid._validate_gate_configuration(cfg)


def test_authenticated_exactly_required_dev_pool_is_not_reused(monkeypatch, tmp_path):
    """The old authenticated 100-item L1a table must not defeat a new request
    for attrition margin merely because cross-stage config identity is relaxed."""
    from types import SimpleNamespace

    cfg = {
        "experiment": {"output_root": str(tmp_path)},
        "synthetic": {"num_pairs_dev": 150, "num_pairs_gate": 150},
        "gate_a": {"min_synthetic_boundaries": 100,
                   "data_sufficiency_universe": "full_l0_role_manifest"},
        "alignment": {"consensus": {}},
    }
    old = pd.DataFrame({
        "pair_id": [f"dev_{i}" for i in range(100)],
        "reference_kind": ["constructed_exact_lexical"] * 100,
    })
    monkeypatch.setattr(
        "csasr.lss.align.devselect.load_dev_set",
        lambda root: (old, {"available": True, "reason": "ok",
                            "sha256": "old-but-authenticated"}))
    logger = SimpleNamespace(info=lambda *a, **k: None,
                             warning=lambda *a, **k: None,
                             error=lambda *a, **k: None)

    result = lss_l1b_valid._prepare_synthetic_evidence(
        cfg, logger, None, {"taint_reasons": []}, [],
        config=object())

    assert autoevidence.BLOCKED_MISSING_DEVELOPMENT_SET in result["blocked"]
    assert result["gate_generation"] is None
    values = {c["name"]: c["value"] for c in result["criteria"]}
    assert values["synthetic_development_candidate_count"] == 100
    assert values["held_out_gate_generation_allocated"] == 0


def test_full_role_counts_cannot_be_confused_with_audit_sample(monkeypatch):
    manifest = pd.DataFrame({
        "utterance_id": [f"u{i}" for i in range(600)],
        "contains_code_switch": [True] * 550 + [False] * 50})
    units = pd.DataFrame([
        {"utterance_id": f"u{i}", "unit_id": j, "language": lang}
        for i in range(600) for j, lang in enumerate(("ZH", "EN", "ZH"))])
    monkeypatch.setattr(lss_l1b_valid, "load_role", lambda cfg, role: manifest)
    monkeypatch.setattr(lss_l1b_valid, "role_path",
                        lambda cfg, role: f"/frozen/{role}.parquet")
    monkeypatch.setattr(lss_l1b_valid, "unit_table", lambda frame: units)
    cfg = {"alignment": {"diagnostics": {"sample_utterances": 300}},
           "gate_a": {"data_sufficiency_universe": "full_l0_role_manifest",
                      "min_construct_bilingual_utterances": 500}}
    result = lss_l1b_valid._full_role_data_sufficiency(cfg, ["D-construct"])
    assert result["audit_sample_size_per_role"] == 300
    assert result["d_construct_bilingual_count"] == 550
    assert result["per_role"]["D-construct"]["embedded_english_targets"] == 600


def _write_reference_tables(root, bad_family_error=900.0):
    families = ("existing_ctc", "whisper_dtw", "qwen_forced_aligner")
    scores, items = [], []
    for family in families:
        error = bad_family_error if family == "qwen_forced_aligner" else 20.0
        scores.append(synthetic.score_family(
            np.array([error / 1000.0] * 120), np.zeros(120), family=family,
            edge="combined", purpose="gate"))
        for i in range(120):
            items.append({
                "schema_version": autoevidence.SYNTHETIC_ITEMS_SCHEMA,
                "pair_id": f"gate_{i}", "family": family,
                "variant": f"{family}/default", "purpose": "gate",
                "reference_kind": "constructed_exact_lexical", "scorable": True,
                "signed_start_error_ms": error, "signed_end_error_ms": error,
                "absolute_start_error_ms": error, "absolute_end_error_ms": error,
                "absolute_boundary_error_ms": error})
    score_path = root / autoevidence.SYNTHETIC_SCORES_FILE
    item_path = root / autoevidence.SYNTHETIC_ITEMS_FILE
    score_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_parquet(score_path, index=False)
    pd.DataFrame(items).to_parquet(item_path, index=False)


def test_rejected_third_family_cannot_worsen_selected_pair_calibration(tmp_path):
    _write_reference_tables(tmp_path)
    result = autoevidence.synthetic_calibration_status(
        tmp_path, selected_pair=["existing_ctc", "whisper_dtw"],
        min_boundaries=100, require_authentication=False)
    assert result["available"]
    assert result["absolute_boundary_error"]["median_abs_error_ms"] == 20.0
    assert "qwen_forced_aligner" not in result["families_scored"]


def test_ctc_timing_uses_waveform_and_emission_lengths(monkeypatch):
    import csasr.data.ctc_alignment as ctc

    aligner = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {"a": 1})),
        blank_id=0)
    monkeypatch.setattr(ctc, "romanize", lambda text: "a")
    monkeypatch.setattr(ctc, "viterbi_align", lambda logp, ids, blank: [(1, 2)])
    monkeypatch.setattr(ctc, "ctc_log_posteriors",
                        lambda aligner, audio, sr: torch.zeros((10, 2)))
    one = ctc.align_units(aligner, np.zeros(16000, dtype=np.float32), ["x"])[0]
    monkeypatch.setattr(ctc, "ctc_log_posteriors",
                        lambda aligner, audio, sr: torch.zeros((25, 2)))
    two = ctc.align_units(aligner, np.zeros(32000, dtype=np.float32), ["x"])[0]
    assert one["seconds_per_emission"] == pytest.approx(.1)
    assert two["seconds_per_emission"] == pytest.approx(.08)
    assert one["end_sec"] == pytest.approx(.2)  # exclusive end emission
    assert two["end_sec"] == pytest.approx(.16)


def test_whisper_processor_mask_reaches_generate(monkeypatch):
    from csasr.models import generation

    expected_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    monkeypatch.setattr(generation, "batch_model_inputs", lambda bundle, paths: {
        "input_features": torch.ones((2, 4, 3)), "attention_mask": expected_mask})

    class Model:
        config = SimpleNamespace(pad_token_id=0)

        def __init__(self):
            self.model = SimpleNamespace(
                encoder=SimpleNamespace(layers=[]),
                decoder=SimpleNamespace(layers=[]))

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(sequences=torch.tensor([[1, 2], [1, 2]]), scores=())

    model = Model()
    bundle = SimpleNamespace(
        model=model,
        processor=SimpleNamespace(
            batch_decode=lambda seq, skip_special_tokens: ["a", "b"],
            tokenizer=SimpleNamespace(eos_token_id=2)))
    cfg = {"decoding": {"output_scores": True}}
    generation.decode_batch(bundle, ["a.wav", "b.wav"], cfg)
    assert torch.equal(model.kwargs["attention_mask"], expected_mask)
    assert model.kwargs["return_dict_in_generate"] is True
    assert model.kwargs["output_scores"] is True


def test_generation_score_options_are_disabled_together():
    from csasr.models.generation import _generate_kwargs

    kwargs = _generate_kwargs({"decoding": {"output_scores": False}}, None)
    assert kwargs["return_dict_in_generate"] is False
    assert "output_scores" not in kwargs


def test_teacher_forced_whisper_also_receives_the_processor_mask(monkeypatch):
    from csasr.models import generation

    expected_mask = torch.tensor([[1, 1, 0]])
    monkeypatch.setattr(generation, "batch_model_inputs", lambda bundle, paths: {
        "input_features": torch.ones((1, 4, 3)),
        "attention_mask": expected_mask})

    class Model:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return object()

    model = Model()
    bundle = SimpleNamespace(
        model=model, device="cpu",
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    generation.teacher_forced_forward(bundle, ["a.wav"], [[1, 2]])
    assert torch.equal(model.kwargs["attention_mask"], expected_mask)


def test_missing_or_nonfinite_times_cannot_improve_validity():
    frame = _good_family()
    frame["is_valid"] = frame["is_valid"].astype(object)
    frame.loc[1, "is_valid"] = np.nan
    frame.loc[1, "start_sample"] = np.nan
    frame.loc[1, "start_sec"] = np.nan
    raw = autoevidence.family_validity(frame).iloc[0]
    assert raw["valid_units"] == len(frame) - 1
    assert raw["invalid_duration_units"] == 1
    objects, _, _ = target.target_objects(frame)
    assert not bool(objects.iloc[0]["target_valid"])


def test_development_diagnostic_writes_only_tainted_outputs(tmp_path, monkeypatch):
    from csasr.experiments import lss_alignment_dev_diagnostic as diagnostic

    production = tmp_path / "production"
    production.mkdir()
    output = tmp_path / "diagnostic"
    cfg = {"experiment": {"output_root": str(production)}}
    source_manifest = {"path": str(production / "candidates.parquet"),
                       "sha256": "source", "producing_run_id": "l1b/old",
                       "taint_reasons": [], "parent_run_ids": []}
    monkeypatch.setattr(diagnostic, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(
        diagnostic.manifest_mod, "read_verified",
        lambda *a, **k: (pd.DataFrame({"x": [1]}),
                         {"ok": True, "manifest": source_manifest}))
    monkeypatch.setattr(
        diagnostic, "evaluate_cached",
        lambda *a, **k: {"diagnostic_only": True,
                         "production_gate_evaluated": False,
                         "production_spans_frozen": False,
                         "l1c_unlocked": False})
    monkeypatch.setattr(diagnostic, "_seam_diagnostics",
                        lambda root: {"available": False})

    assert diagnostic.main([
        "--diagnostic-output", str(output)]) == 0
    for name in ("alignment_gate_a_development_diagnostic.json",
                 "alignment_gate_a_development_diagnostic.md"):
        manifest = json.loads((output / f"{name}.manifest.json").read_text())
        assert manifest["diagnostic_only"] is True
        assert manifest["taint_reasons"] == ["development_only_diagnostic"]
        assert manifest["held_out_gate_generation_consumed"] is False
    assert not (production / "status/l1b_valid.json").exists()
    assert not (production / "freeze/l1b_spans_freeze.json").exists()

    with pytest.raises(SystemExit, match="outside the production artifacts root"):
        diagnostic.main([
            "--diagnostic-output", str(production / "reports" / "diagnostic")])
