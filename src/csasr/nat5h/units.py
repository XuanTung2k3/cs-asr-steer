"""Automatic transcript units for the NAT5H exploratory pipeline.

The NAT5H experiment has no human annotation step.  Every transcript unit used
for alignment, direction construction, and POI selection is therefore produced
by this deterministic normalizer/tagger.  The labels are deliberately
conservative: only ZH and EN content can enter scientific stages; OTHER remains
visible in artifacts instead of being silently discarded.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

from csasr.data.language_tags import EN, OTHER, UNKNOWN, ZH, tag_unit
from csasr.data.normalize import NORMALIZATION_VERSION, normalize_and_segment

NAT5H_UNIT_VERSION = f"nat5h_units:{NORMALIZATION_VERSION}:v1"


@dataclass(frozen=True)
class ReferenceUnit:
    utterance_id: str
    unit_id: int
    original_text: str
    normalized_transcript: str
    surface: str
    normalized_surface: str
    language: str
    normalized_char_start: int
    normalized_char_end: int
    original_char_start: int | None
    original_char_end: int | None
    normalization_version: str
    normalization_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalization_hash(original_text: str, normalized_transcript: str) -> str:
    h = hashlib.sha256()
    h.update(NAT5H_UNIT_VERSION.encode("utf-8"))
    h.update(b"\0")
    h.update(str(original_text or "").encode("utf-8"))
    h.update(b"\0")
    h.update(normalized_transcript.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def _language_for_surface(surface: str) -> str:
    tag = tag_unit(surface)
    return OTHER if tag == UNKNOWN else tag


def _nfkc_lower(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower()


def _find_original_span(
    original_text: str,
    surface: str,
    *,
    search_from: int,
) -> tuple[int | None, int | None, int]:
    """Best-effort original-character mapping for reader-facing artifacts.

    Exact comparison is done in normalized space.  This mapping is for audit and
    report traceability, so it is intentionally conservative: if the surface
    cannot be located deterministically in NFKC/lowercased original text, the
    original offsets are left null instead of guessed.
    """
    if not surface:
        return None, None, search_from
    comparable = _nfkc_lower(original_text)
    idx = comparable.find(surface.lower(), max(0, search_from))
    if idx < 0:
        return None, None, search_from
    return idx, idx + len(surface), idx + len(surface)


def build_reference_units(utterance_id: str, transcript: str | None) -> list[ReferenceUnit]:
    original = "" if transcript is None else str(transcript)
    normalized, segmented = normalize_and_segment(original)
    nhash = normalization_hash(original, normalized)
    units: list[ReferenceUnit] = []
    search_from = 0
    for i, unit in enumerate(segmented):
        o_start, o_end, search_from = _find_original_span(
            original, unit.surface, search_from=search_from
        )
        units.append(
            ReferenceUnit(
                utterance_id=str(utterance_id),
                unit_id=i,
                original_text=original,
                normalized_transcript=normalized,
                surface=unit.surface,
                normalized_surface=unit.normalized_surface,
                language=_language_for_surface(unit.surface),
                normalized_char_start=int(unit.char_start),
                normalized_char_end=int(unit.char_end),
                original_char_start=o_start,
                original_char_end=o_end,
                normalization_version=NAT5H_UNIT_VERSION,
                normalization_hash=nhash,
            )
        )
    return units


def units_to_rows(units: Iterable[ReferenceUnit]) -> list[dict]:
    return [u.to_dict() for u in units]


def content_units(units: Iterable[ReferenceUnit]) -> list[ReferenceUnit]:
    return [u for u in units if u.language in {EN, ZH}]


def transcript_has_bilingual_content(units: Iterable[ReferenceUnit]) -> bool:
    langs = {u.language for u in units}
    return EN in langs and ZH in langs
