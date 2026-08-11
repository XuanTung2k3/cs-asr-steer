"""Canonical short-wav coordinate handling for the nat5h experiment.

The only time origin used here is sample 0 of the exact CS-Dialogue short_wav
waveform passed to Whisper. External VAD/cropping/full-dialogue offsets are not
represented.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EncoderGeometry:
    sample_rate: int
    hop_length: int
    conv_stride: int
    encoder_step_samples: int
    encoder_step_sec: float
    max_encoder_frames: int

    @classmethod
    def from_bundle(cls, bundle) -> "EncoderGeometry":
        fe = bundle.processor.feature_extractor
        sample_rate = int(fe.sampling_rate)
        hop = int(fe.hop_length)
        conv_stride = 2
        step = int(hop * conv_stride)
        return cls(
            sample_rate=sample_rate,
            hop_length=hop,
            conv_stride=conv_stride,
            encoder_step_samples=step,
            encoder_step_sec=step / sample_rate,
            max_encoder_frames=int(bundle.max_encoder_frames),
        )

    @classmethod
    def from_values(cls, sample_rate: int = 16000, hop_length: int = 160,
                    conv_stride: int = 2, max_encoder_frames: int = 1500) -> "EncoderGeometry":
        step = int(hop_length * conv_stride)
        return cls(sample_rate, hop_length, conv_stride, step,
                   step / sample_rate, int(max_encoder_frames))


@dataclass(frozen=True)
class TimestampRecord:
    utterance_id: str
    source_aligner: str
    unit_id: int
    surface: str
    language: str
    start_second: float
    end_second: float
    start_sample: int
    end_sample: int
    sample_rate: int
    waveform_num_samples: int
    start_encoder_frame: int
    end_encoder_frame: int
    encoder_num_frames: int
    coordinate_origin: str
    preprocessing_hash: str
    confidence: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocessing_hash(*, audio_path: str | Path, sample_rate: int,
                       waveform_num_samples: int, policy: str = "short_wav_no_crop_v1") -> str:
    payload = {
        "audio_path": str(audio_path),
        "sample_rate": int(sample_rate),
        "waveform_num_samples": int(waveform_num_samples),
        "policy": policy,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def seconds_to_sample(seconds: float, sample_rate: int) -> int:
    return int(round(float(seconds) * int(sample_rate)))


def sample_to_seconds(sample: int, sample_rate: int) -> float:
    return int(sample) / int(sample_rate)


def sample_to_encoder_frame(sample: int, geometry: EncoderGeometry,
                            encoder_num_frames: int | None = None) -> int:
    limit = geometry.max_encoder_frames if encoder_num_frames is None else int(encoder_num_frames)
    return max(0, min(int(math.floor(int(sample) / geometry.encoder_step_samples)), limit - 1))


def sample_span_to_encoder_span(start_sample: int, end_sample: int,
                                geometry: EncoderGeometry,
                                encoder_num_frames: int) -> tuple[int, int]:
    start = sample_to_encoder_frame(start_sample, geometry, encoder_num_frames)
    end = int(math.ceil(int(end_sample) / geometry.encoder_step_samples))
    end = max(start + 1, min(end, int(encoder_num_frames)))
    return start, end


def encoder_frame_to_sample(frame: int, geometry: EncoderGeometry) -> int:
    return int(frame) * geometry.encoder_step_samples


def waveform_num_frames(waveform_num_samples: int, geometry: EncoderGeometry) -> int:
    n = int(math.ceil(int(waveform_num_samples) / geometry.encoder_step_samples))
    return max(1, min(n, geometry.max_encoder_frames))


def make_timestamp_record(*, utterance_id: str, source_aligner: str, unit_id: int,
                          surface: str, language: str, start_second: float,
                          end_second: float, sample_rate: int,
                          waveform_num_samples: int, geometry: EncoderGeometry,
                          preprocessing: str, confidence: float | None = None,
                          metadata: dict[str, Any] | None = None) -> TimestampRecord:
    start_sample = seconds_to_sample(start_second, sample_rate)
    end_sample = seconds_to_sample(end_second, sample_rate)
    encoder_frames = waveform_num_frames(waveform_num_samples, geometry)
    start_frame, end_frame = sample_span_to_encoder_span(
        start_sample, end_sample, geometry, encoder_frames)
    rec = TimestampRecord(
        utterance_id=str(utterance_id),
        source_aligner=str(source_aligner),
        unit_id=int(unit_id),
        surface=str(surface),
        language=str(language),
        start_second=float(start_second),
        end_second=float(end_second),
        start_sample=int(start_sample),
        end_sample=int(end_sample),
        sample_rate=int(sample_rate),
        waveform_num_samples=int(waveform_num_samples),
        start_encoder_frame=int(start_frame),
        end_encoder_frame=int(end_frame),
        encoder_num_frames=int(encoder_frames),
        coordinate_origin="short_wav_sample0",
        preprocessing_hash=str(preprocessing),
        confidence=confidence,
        metadata=metadata or {},
    )
    validate_timestamp_record(rec)
    return rec


def validate_timestamp_record(rec: TimestampRecord) -> None:
    if rec.coordinate_origin != "short_wav_sample0":
        raise ValueError(f"unsupported coordinate origin: {rec.coordinate_origin}")
    if not (0 <= rec.start_sample < rec.end_sample <= rec.waveform_num_samples):
        raise ValueError(f"invalid sample span for {rec.utterance_id}/{rec.unit_id}: "
                         f"{rec.start_sample}-{rec.end_sample} / {rec.waveform_num_samples}")
    if not (0 <= rec.start_encoder_frame < rec.end_encoder_frame <= rec.encoder_num_frames):
        raise ValueError(f"invalid encoder span for {rec.utterance_id}/{rec.unit_id}: "
                         f"{rec.start_encoder_frame}-{rec.end_encoder_frame} / "
                         f"{rec.encoder_num_frames}")
    if not np.isfinite(rec.start_second) or not np.isfinite(rec.end_second):
        raise ValueError("timestamp seconds must be finite")


def roundtrip_report(sample: int, geometry: EncoderGeometry) -> dict[str, Any]:
    sec = sample_to_seconds(sample, geometry.sample_rate)
    sample2 = seconds_to_sample(sec, geometry.sample_rate)
    frame = sample_to_encoder_frame(sample, geometry)
    approx_sample = encoder_frame_to_sample(frame, geometry)
    return {
        "sample": int(sample),
        "seconds": sec,
        "sample_roundtrip": sample2,
        "encoder_frame": frame,
        "approx_sample_from_frame": approx_sample,
        "sample_roundtrip_error": abs(sample2 - int(sample)),
        "frame_roundtrip_error_samples": abs(approx_sample - int(sample)),
        "encoder_step_samples": geometry.encoder_step_samples,
    }

