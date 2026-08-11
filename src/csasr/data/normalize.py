"""Versioned text normalization and unit segmentation (guide section 7.3).

One pipeline is used everywhere: baseline error detection, correctness filtering
for direction construction, MER, PIER and correction/corruption evaluation.

Version history
---------------
v1  Initial: NFKC, CS-Dialogue non-speech markers removed, punctuation removed,
    Latin lowercased, Han kept per character, English kept per word.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

NORMALIZATION_VERSION = "v1"

# CS-Dialogue annotation markers: <FIL/> filler, <SPK/> speaker noise,
# <NON/> non-speech, <NPS/> unintelligible; "**" marks an edited region.
MARKER_RE = re.compile(r"<\s*(FIL|SPK|NON|NPS)\s*/?\s*>", re.IGNORECASE)
STAR_RE = re.compile(r"\*+")
ANY_TAG_RE = re.compile(r"<[^>]{0,16}>")

PUNCT_CHARS = "，。？！；：、,.?!;:\"“”‘’()（）《》【】…—～~/\\|+=＿_[]{}"
PUNCT_RE = re.compile("[" + re.escape(PUNCT_CHARS) + "]")

HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# A Latin "word": letters/digits with internal apostrophes or hyphens.
LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
WS_RE = re.compile(r"\s+")


def is_han(ch: str) -> bool:
    return bool(HAN_RE.match(ch))


def normalize_text(text: str, version: str = NORMALIZATION_VERSION) -> str:
    """Normalize a transcript to the comparison form used by all metrics."""
    if version != NORMALIZATION_VERSION:
        raise ValueError(f"unsupported normalization version {version!r}")
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = MARKER_RE.sub(" ", t)
    t = ANY_TAG_RE.sub(" ", t)
    t = STAR_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    t = t.replace("　", " ")
    t = t.lower()
    # keep only Han, latin word characters, apostrophe, hyphen and space
    t = "".join(
        ch if (is_han(ch) or ch.isalnum() or ch in "'- ") else " " for ch in t
    )
    t = WS_RE.sub(" ", t).strip()
    return t


@dataclass(frozen=True)
class Unit:
    """One reference/hypothesis unit: a Han character or a Latin word."""

    surface: str          # as it appears in the normalized string
    kind: str             # "han" | "latin"
    char_start: int       # offset into the normalized string
    char_end: int

    @property
    def normalized_surface(self) -> str:
        return self.surface


def segment_units(normalized: str) -> list[Unit]:
    """Split a *normalized* string into scoring units.

    Chinese is segmented per character, English per word. This is the mixed
    tokenization used by MER and by the POI/PIER machinery.
    """
    units: list[Unit] = []
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        if ch.isspace():
            i += 1
            continue
        if is_han(ch):
            units.append(Unit(ch, "han", i, i + 1))
            i += 1
            continue
        m = LATIN_WORD_RE.match(normalized, i)
        if m:
            units.append(Unit(m.group(0), "latin", m.start(), m.end()))
            i = m.end()
            continue
        i += 1  # residual symbol: drop
    return units


def tokenize(text: str) -> list[str]:
    """Normalize then return unit surfaces (the MER token sequence)."""
    return [u.surface for u in segment_units(normalize_text(text))]


def normalize_and_segment(text: str) -> tuple[str, list[Unit]]:
    norm = normalize_text(text)
    return norm, segment_units(norm)
