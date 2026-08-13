"""Automatic alignment-quality checks that replace the manual boundary audit."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from csasr.data.alignment_checks import (
    FRAME_SEC,
    _edge_silence_frames,
    frame_rms,
    summarize_boundary_error,
)
from csasr.experiments.e4_oracle import relocated_spans

SR = 16000


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    n = int(seconds * SR)
    return np.sin(2 * np.pi * freq * np.arange(n) / SR).astype("float32")


def test_frame_rms_uses_the_encoder_frame_grid():
    rms = frame_rms(_tone(1.0), SR)
    assert len(rms) == int(1.0 / FRAME_SEC)          # 50 frames of 20 ms
    assert abs(float(rms.mean()) - 0.707) < 0.01     # RMS of a unit sine


def test_frame_rms_handles_audio_shorter_than_one_frame():
    assert len(frame_rms(np.zeros(10, dtype="float32"), SR)) == 0


def test_leading_silence_is_measured_in_frames():
    audio = np.concatenate([np.zeros(int(0.2 * SR), dtype="float32"), _tone(1.0)])
    rms = frame_rms(audio, SR)
    thr = 0.05 * float(np.percentile(rms, 95))
    assert _edge_silence_frames(rms, thr, 0, len(rms)) == 10           # 200 ms
    assert _edge_silence_frames(rms, thr, 0, len(rms), from_end=True) == 0


def test_boundary_error_summary_reports_bias_and_coverage():
    out = summarize_boundary_error(np.array([1.02, 2.03, 3.01]),
                                   np.array([1.00, 2.00, 3.00]))
    assert out["num_boundaries"] == 3
    assert out["mean_signed_error_ms"] > 0            # predictions are late
    assert out["pct_within_50ms"] == 1.0


def test_boundary_error_summary_survives_an_empty_sample():
    out = summarize_boundary_error(np.array([]), np.array([]))
    assert out["num_boundaries"] == 0


def test_audio_seam_summary_never_uses_lexical_error_names():
    out = summarize_boundary_error(
        np.array([1.02, 2.03]), np.array([1.00, 2.00]),
        reference_kind="audio_splice")
    assert out["num_audio_seams"] == 2
    assert out["metric_semantics"] == "audio_seam_relative_not_lexical_accuracy"
    assert out["pct_within_50ms_of_seam"] == 1.0
    assert "median_abs_error_ms" not in out
    assert "mean_signed_error_ms" not in out


# ---------------------------------------------------------------------------
# ENC-RANDLOC-1L span relocation
# ---------------------------------------------------------------------------
BUNDLE = SimpleNamespace(valid_frames=lambda d: int(round(d / FRAME_SEC)))
TARGETS = pd.DataFrame([
    {"utterance_id": "u1", "start_frame": 100, "end_frame": 120, "duration_sec": 10.0},
    {"utterance_id": "u3", "start_frame": 50, "end_frame": 60, "duration_sec": 6.0},
])


def test_relocated_spans_preserve_width_and_avoid_the_target():
    spans, unrelocated = relocated_spans(BUNDLE, TARGETS, seed=42)
    assert unrelocated == 0
    for _, row in TARGETS.iterrows():
        s, e = spans[row["utterance_id"]]
        assert e - s == row["end_frame"] - row["start_frame"]
        assert e <= row["start_frame"] or s >= row["end_frame"]


def test_relocated_spans_are_deterministic_per_seed():
    assert relocated_spans(BUNDLE, TARGETS, seed=42)[0] == \
           relocated_spans(BUNDLE, TARGETS, seed=42)[0]
    assert relocated_spans(BUNDLE, TARGETS, seed=42)[0] != \
           relocated_spans(BUNDLE, TARGETS, seed=43)[0]


def test_span_covering_the_whole_utterance_is_reported_as_unrelocated():
    """No disjoint room: the control degrades to the original span, conservatively."""
    full = pd.DataFrame([{"utterance_id": "u2", "start_frame": 0,
                          "end_frame": 40, "duration_sec": 0.8}])
    spans, unrelocated = relocated_spans(BUNDLE, full, seed=42)
    assert unrelocated == 1
    assert spans["u2"] == (0, 40)


def test_speech_bounds_strips_edge_silence():
    import numpy as np
    from csasr.data.alignment_checks import speech_bounds

    sr = 16000
    audio = np.concatenate([np.zeros(sr, dtype=np.float32),          # 1.0 s lead-in
                            np.ones(sr // 2, dtype=np.float32) * 0.5,
                            np.zeros(sr, dtype=np.float32)])         # 1.0 s lead-out
    start, end = speech_bounds(audio, sr, pad_ms=20.0)
    assert abs(start / sr - 0.98) < 0.03      # 1.0 s minus the 20 ms pad
    assert abs(end / sr - 1.52) < 0.03


def test_rendered_pair_boundary_is_a_speech_transition(tmp_path):
    """The constructed instant must be where speech switches, not the file join.

    Untrimmed clips put ~1.3 s of silence at the splice, which moved the "true"
    boundary ~0.9 s away from anything the aligner could mark (job 31473).
    """
    import numpy as np
    import soundfile as sf
    from csasr.data.alignment_checks import render_synthetic_pair

    sr = 16000
    def clip(path, lead, body, trail):
        sf.write(str(path), np.concatenate([
            np.zeros(int(lead * sr), dtype=np.float32),
            (np.random.default_rng(0).standard_normal(int(body * sr)) * 0.2).astype(np.float32),
            np.zeros(int(trail * sr), dtype=np.float32)]), sr)

    zh, en = tmp_path / "zh.wav", tmp_path / "en.wav"
    clip(zh, 0.5, 1.0, 0.8)
    clip(en, 0.7, 1.0, 0.4)
    pair = {"pair_id": "syn_test", "zh_audio_path": str(zh), "en_audio_path": str(en),
            "gap_sec": 0.0, "sample_rate": sr, "zh_text": "你好", "en_text": "hello"}
    out = render_synthetic_pair(pair, tmp_path / "syn.wav")

    # boundary sits at the end of the ZH body, not 0.8 s later at the file join
    assert abs(out["true_boundary_sec"] - 1.04) < 0.08
    assert out["duration_sec"] < 2.4          # both silences removed


def test_synthetic_pairs_respect_the_encoder_window():
    import pandas as pd
    from csasr.data.alignment_checks import build_synthetic_pairs

    manifest = pd.DataFrame([
        {"utterance_id": "zh_long", "transcript_raw": "你好世界", "audio_path": "/x/a.wav",
         "duration_sec": 40.0},
        {"utterance_id": "zh_ok", "transcript_raw": "你好世界", "audio_path": "/x/b.wav",
         "duration_sec": 5.0},
        {"utterance_id": "en_long", "transcript_raw": "hello world", "audio_path": "/x/c.wav",
         "duration_sec": 40.0},
        {"utterance_id": "en_ok", "transcript_raw": "hello world", "audio_path": "/x/d.wav",
         "duration_sec": 5.0},
    ])
    pairs = build_synthetic_pairs(manifest, n_pairs=10, seed=42, max_duration_sec=28.0)
    assert len(pairs) == 1
    assert pairs[0]["zh_utterance_id"] == "zh_ok"
    assert pairs[0]["en_utterance_id"] == "en_ok"
