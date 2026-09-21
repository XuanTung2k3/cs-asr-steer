"""Canonical ASCEND adapter and frozen transcript eligibility rule."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..data.language_tags import EN, ZH, tag_units
from ..data.normalize import NORMALIZATION_VERSION, normalize_and_segment
from ..utils.hashing import sha256_obj

ASCEND_REPO = "CAiRE/ASCEND"
ELIGIBILITY_RULE_VERSION = "basis-a6-ascend-en-zh-content-v1"
ELIGIBILITY_RULE_HASH = "sha256:" + sha256_obj({
    "version": ELIGIBILITY_RULE_VERSION,
    "normalization": NORMALIZATION_VERSION,
    "tags": "csasr.data.language_tags.tag_unit",
    "eligible": [EN, ZH],
}).strip()


@dataclass(frozen=True)
class CanonicalUtterance:
    utterance_id: str
    audio: Any
    sample_rate: int | None
    transcript_raw: str
    transcript_normalized: str
    units: tuple[dict[str, Any], ...]
    duration_sec: float | None
    speaker_id: str
    session_id: str
    topic: str
    language_metadata: str
    source_split: str

    @property
    def en_units(self) -> int:
        return sum(u["language_tag"] == EN for u in self.units)

    @property
    def zh_units(self) -> int:
        return sum(u["language_tag"] == ZH for u in self.units)

    @property
    def cs_eligible(self) -> bool:
        return self.en_units >= 1 and self.zh_units >= 1


def _value(row: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        try:
            if isinstance(row, dict) and name in row:
                return row[name]
            if hasattr(row, name):
                return getattr(row, name)
        except Exception:
            continue
    return default


def _audio(row: Any) -> tuple[Any, int | None]:
    audio = _value(row, "audio", default=None)
    if isinstance(audio, dict):
        # HF Audio(decode=False) may expose only the basename in audio["path"]
        # while ASCEND's canonical row-level `path` retains the extracted local
        # file. Prefer the row-level path when it is present and usable; retain
        # the array/bytes fallback for in-memory fixtures.
        row_path = _value(row, "path", default=None)
        audio_path = audio.get("path")
        if row_path and Path(str(row_path)).is_file():
            audio_path = row_path
        # On login nodes the HF extracted cache path may not be mounted even
        # though the Arrow row carries the encoded WAV bytes. Preserve those
        # bytes as the canonical fallback; model runners materialize them in a
        # job-local temporary file when a path is required.
        if audio_path and Path(str(audio_path)).is_file():
            return audio_path, audio.get("sampling_rate")
        return audio.get("array", audio.get("bytes")), audio.get("sampling_rate")
    return audio, _value(row, "sampling_rate", "sample_rate", default=None)


def adapt_ascend_row(row: Any, *, split: str) -> CanonicalUtterance:
    if split not in {"train", "validation"}:
        raise ValueError("BASIS-A6 ASCEND adapter refuses the test split")
    raw = str(_value(row, "transcription", "transcript", "text", default=""))
    norm, units = normalize_and_segment(raw)
    tags = tag_units(units)
    unit_rows = tuple({"surface": u.surface, "kind": u.kind, "char_start": u.char_start,
                       "char_end": u.char_end, "language_tag": tag}
                      for u, tag in zip(units, tags))
    audio, sr = _audio(row)
    source_id = str(_value(row, "id", "utterance_id", default=""))
    if not source_id:
        raise ValueError("ASCEND row has no id")
    duration = _value(row, "duration", "duration_sec", default=None)
    # The upstream `id` is not globally unique: a small number of ids recur
    # across train/validation. The split-qualified namespace is therefore the
    # canonical identity required to make the frozen role manifests truly
    # disjoint while retaining the upstream id in the audio path/metadata.
    return CanonicalUtterance(
        utterance_id=f"ascend/{split}/{source_id}", audio=audio, sample_rate=None if sr is None else int(sr),
        transcript_raw=raw, transcript_normalized=norm, units=unit_rows,
        duration_sec=None if duration is None else float(duration),
        speaker_id=str(_value(row, "original_speaker_id", "speaker", "speaker_id", default="")),
        session_id=str(_value(row, "session_id", "session", default="")),
        topic=str(_value(row, "topic", default="")),
        language_metadata=str(_value(row, "language", default="")), source_split=split)


def load_ascend_split(dataset_dir: str | Path, split: str) -> list[CanonicalUtterance]:
    """Load exactly one allowed split from a saved HF DatasetDict.

    The function deliberately has no `test` escape hatch: all downstream A6
    code goes through this adapter, making accidental test use a hard error.
    """
    if split not in {"train", "validation"}:
        raise ValueError("ASCEND test is DO-NOT-READ / DO-NOT-USE")
    from datasets import Audio, load_from_disk
    ds = load_from_disk(str(dataset_dir))
    if split not in ds:
        raise KeyError(f"missing ASCEND split {split!r}")
    # Subset freezing needs transcript/metadata and must not materialize every
    # waveform into Python. The canonical record still carries the reversible
    # local audio path; model runners may decode that path later.
    part = ds[split].cast_column("audio", Audio(sampling_rate=16000, decode=False))
    return [adapt_ascend_row(part[i], split=split) for i in range(len(part))]


def eligibility_record(item: CanonicalUtterance) -> dict[str, Any]:
    return {"utterance_id": item.utterance_id, "source_split": item.source_split,
            "speaker_id": item.speaker_id, "session_id": item.session_id, "topic": item.topic,
            "transcript_normalized": item.transcript_normalized, "en_units": item.en_units,
            "zh_units": item.zh_units, "duration_sec": item.duration_sec,
            "eligible": item.cs_eligible}


def rows_to_records(rows: Iterable[CanonicalUtterance]) -> list[dict[str, Any]]:
    return [eligibility_record(row) for row in rows]
