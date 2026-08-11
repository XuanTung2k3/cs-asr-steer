from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from csasr.nat5h.consensus import (
    ConsensusCriteria,
    SmokeCriteria,
    alignment_validation_flags,
    build_consensus,
    consensus_union_mask,
    erode_safe_interior,
    pairwise_disagreement,
    smoke_passes,
    summarize_alignment_smoke,
)
from csasr.nat5h.coordinates import (
    EncoderGeometry,
    make_timestamp_record,
    roundtrip_report,
)
from csasr.nat5h.model_cache import FileLock, ensure_qwen_checkpoint
from csasr.nat5h.selection import (
    reject_test_split,
    select_pois,
    speaker_balanced_sample,
    validate_disjoint_speakers,
)
from csasr.nat5h.statusing import StageTimer, read_status, write_stage_status
from csasr.nat5h.units import build_reference_units, transcript_has_bilingual_content
from csasr.models.hooks import apply_steering


def _ts(utt: str, aligner: str, unit_id: int, start: float, end: float, lang="EN"):
    geom = EncoderGeometry.from_values()
    return make_timestamp_record(
        utterance_id=utt,
        source_aligner=aligner,
        unit_id=unit_id,
        surface="hello" if lang == "EN" else "你",
        language=lang,
        start_second=start,
        end_second=end,
        sample_rate=16000,
        waveform_num_samples=16000 * 10,
        geometry=geom,
        preprocessing="test",
    ).to_dict() | {
        "speaker": "spk1",
        "split": "dev_select",
        "encoder_step_sec": geom.encoder_step_sec,
    }


def test_nat5h_transcript_language_units_are_deterministic():
    units = build_reference_units("u1", "你好 NASA ok 123, B2B 嗯")
    rows = [(u.surface, u.language) for u in units]
    assert ("你", "ZH") in rows
    assert ("好", "ZH") in rows
    assert ("nasa", "EN") in rows
    assert ("ok", "EN") in rows
    assert ("123", "OTHER") in rows
    assert ("b2b", "OTHER") in rows
    assert transcript_has_bilingual_content(units)
    assert build_reference_units("empty", "") == []


def test_nat5h_coordinate_roundtrips_and_short_wav_origin():
    geom = EncoderGeometry.from_values(sample_rate=16000, hop_length=160, conv_stride=2)
    r = roundtrip_report(12345, geom)
    assert r["sample_roundtrip_error"] == 0
    assert r["frame_roundtrip_error_samples"] <= geom.encoder_step_samples
    rec = _ts("utt", "a", 0, 0.5, 0.9)
    assert rec["coordinate_origin"] == "short_wav_sample0"
    assert rec["start_sample"] == 8000
    assert rec["end_sample"] == 14400
    with pytest.raises(ValueError):
        make_timestamp_record(
            utterance_id="bad",
            source_aligner="a",
            unit_id=0,
            surface="x",
            language="EN",
            start_second=1.0,
            end_second=1.0,
            sample_rate=16000,
            waveform_num_samples=16000,
            geometry=geom,
            preprocessing="test",
        )


def test_nat5h_alignment_validation_zero_duration_monotonic_and_smoke():
    good = [_ts("u1", "a", 0, 0.1, 0.4), _ts("u1", "a", 1, 0.4, 0.7, "ZH")]
    bad = dict(good[1])
    bad.update(source_aligner="b", unit_id=0, start_sample=2000, end_sample=2000)
    df = pd.DataFrame(good + [bad])
    flags = alignment_validation_flags(df)
    assert flags["valid_timestamp"].sum() == 2
    refs = pd.DataFrame(
        [
            {"utterance_id": "u1", "unit_id": 0, "language": "EN"},
            {"utterance_id": "u1", "unit_id": 1, "language": "ZH"},
        ]
    )
    smoke = summarize_alignment_smoke(pd.DataFrame(good), refs).iloc[0]
    assert smoke_passes(smoke, SmokeCriteria())


def test_nat5h_pairwise_consensus_safe_interior_and_union_mask():
    rows = [
        _ts("u1", "a", 0, 1.00, 2.00, "EN"),
        _ts("u1", "b", 0, 1.10, 2.05, "EN"),
        _ts("u1", "a", 1, 3.00, 4.00, "ZH"),
        _ts("u1", "b", 1, 3.05, 4.10, "ZH"),
    ]
    df = pd.DataFrame(rows)
    disagree = pairwise_disagreement(df)
    assert float(disagree["start_disagreement_ms"].max()) <= 100.0
    consensus, _ = build_consensus(df, ConsensusCriteria())
    assert len(consensus) == 2
    en = consensus[consensus["language"] == "EN"].iloc[0]
    assert abs(en["consensus_start_second"] - 1.05) <= 0.001
    assert en["pseudo_mask_kind"] == "consensus-union pseudo-mask"
    assert erode_safe_interior(16000, 32000, 16000, 250, 200) == (20000, 28000)
    assert consensus_union_mask([100], [200], waveform_num_samples=250, sample_rate=1000, padding_ms=200) == (0, 250)


def test_nat5h_split_leakage_and_deterministic_sampling():
    train = pd.DataFrame({"speaker_id": ["a"], "official_split": ["train"], "utterance_id": ["u1"]})
    dev = pd.DataFrame({"speaker_id": ["b"], "official_split": ["dev"], "utterance_id": ["u2"]})
    validate_disjoint_speakers(train, dev)
    with pytest.raises(ValueError):
        validate_disjoint_speakers(train, pd.DataFrame({"speaker_id": ["a"], "utterance_id": ["u3"]}))
    with pytest.raises(ValueError):
        reject_test_split(pd.DataFrame({"official_split": ["test"], "utterance_id": ["u4"]}))
    df = pd.DataFrame({"speaker_id": ["a", "a", "b", "b"], "utterance_id": ["u1", "u2", "u3", "u4"]})
    s1 = speaker_balanced_sample(df, 3, seed=7)
    s2 = speaker_balanced_sample(df, 3, seed=7)
    assert s1["utterance_id"].tolist() == s2["utterance_id"].tolist()


def test_nat5h_one_poi_per_utterance_selection():
    manifest = pd.DataFrame(
        {
            "utterance_id": ["u1", "u2"],
            "speaker_id": ["s1", "s2"],
            "transcript_raw": ["你好 hello", "你好 world"],
            "transcript_normalized": ["你 好 hello", "你 好 world"],
        }
    )
    baseline = pd.DataFrame(
        {
            "utterance_id": ["u1", "u2"],
            "hypothesis_normalized": ["你 好", "你 好 world"],
        }
    )
    consensus = pd.DataFrame(
        {
            "utterance_id": ["u1", "u2"],
            "unit_id": [2, 2],
            "language": ["EN", "EN"],
            "speaker": ["s1", "s2"],
            "mask_start_frame": [10, 20],
            "mask_end_frame": [20, 30],
        }
    )
    wrong, correct = select_pois(consensus, manifest, baseline, wrong=1, correct=1, seed=9)
    assert len(wrong) == 1
    assert len(correct) == 1
    assert wrong["utterance_id"].is_unique
    assert correct["utterance_id"].is_unique


def test_nat5h_direction_sign_zero_mask_and_norm_preservation():
    h = torch.randn(2, 5, 4)
    d = torch.randn(4)
    d = d / d.norm()
    zero = torch.zeros(2, 5)
    assert torch.equal(apply_steering(h, d, 1.0, 1.0, zero), h)
    one = torch.ones(2, 5)
    pos = apply_steering(h, d, 1.0, 1.0, one, norm_preserve=False)
    neg = apply_steering(h, -d, 1.0, 1.0, one, norm_preserve=False)
    assert ((pos - h) @ d).mean() > 0
    assert ((neg - h) @ d).mean() < 0
    kept = apply_steering(h, d, 1.0, 1.0, one, norm_preserve=True)
    assert torch.allclose(kept.norm(dim=-1), h.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_nat5h_atomic_status_and_resume_state(tmp_path: Path):
    timer = StageTimer()
    write_stage_status(
        tmp_path,
        {
            "stage": "n0_preflight",
            "state": "passed",
            "started_at": timer.started_at,
            "finished_at": timer.started_at,
            "elapsed_seconds": 0.0,
            "config_hash": "cfg",
            "code_commit": "none",
            "input_manifest_hashes": {},
            "output_paths": [],
            "gate_evidence": {},
            "next_permitted_stage": "n1_align_smoke",
            "failure_or_no_go_reason": None,
            "exploratory": True,
        },
    )
    assert read_status(tmp_path, "n0_preflight")["state"] == "passed"
    assert (tmp_path / "status" / "overview.json").exists()


def test_nat5h_model_cache_marker_and_lock_no_download(tmp_path: Path):
    ckpt = tmp_path / "qwen"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "model.safetensors").write_bytes(b"fake")
    marker = tmp_path / "marker.json"
    res = ensure_qwen_checkpoint(
        model_id="Qwen/Qwen3-ForcedAligner-0.6B",
        local_model_dir=ckpt,
        cache_dir=tmp_path / "cache",
        marker_path=marker,
        allow_download=False,
    )
    assert res["state"] == "present_marker_created"
    assert json.loads(marker.read_text())["downloaded"] is False
    lock_path = tmp_path / "x.lock"
    with FileLock(lock_path, timeout_seconds=1, poll_seconds=0.01):
        assert lock_path.exists()
    assert not lock_path.exists()
