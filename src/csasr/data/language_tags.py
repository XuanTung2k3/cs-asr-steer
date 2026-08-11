"""Reference-unit language tagging: EN / ZH / OTHER / UNKNOWN (guide section 11).

Rules are deliberately conservative: not every Latin token is English. Fillers,
bare numerals and known romanized-Mandarin forms are excluded from direction
construction so that the estimated direction is not contaminated.
"""
from __future__ import annotations

from collections import Counter

from .normalize import Unit, is_han

EN = "EN"
ZH = "ZH"
OTHER = "OTHER"
UNKNOWN = "UNKNOWN"
TAGS = (EN, ZH, OTHER, UNKNOWN)

# Non-lexical fillers / backchannels written in Latin script. Acoustically these
# are not English words, so they are excluded (OTHER) rather than tagged EN.
FILLERS = {
    "uh", "uhh", "uhm", "um", "umm", "er", "err", "erm", "ah", "ahh", "aah",
    "eh", "ehh", "mm", "mmm", "hmm", "hmmm", "huh", "hm", "mhm", "uhhuh",
    "oh", "ooh", "ohh", "aha", "ha", "haha", "hah", "eng", "ne", "ba", "la",
    "ma", "ya", "yo", "wa", "ei", "ao", "o",
}

# Romanized Mandarin and pinyin-like tokens that are Latin script but Mandarin.
ROMANIZED_MANDARIN = {
    "zhongguo", "beijing", "shanghai", "hanyu", "putonghua", "ni", "hao",
    "nihao", "xiexie", "jiayou", "renminbi", "yuan", "kuai", "mao",
    "laoshi", "tongxue", "shifu", "gege", "jiejie", "didi", "meimei",
    "wechat", "weibo", "bilibili", "taobao", "zhihu", "douyin",
}

# Latin tokens that ARE English despite being short/ambiguous.
SHORT_ENGLISH = {"a", "i", "an", "am", "as", "at", "be", "by", "do", "go", "he",
                 "if", "in", "is", "it", "me", "my", "no", "of", "on", "or",
                 "so", "to", "up", "us", "we", "ok", "okay", "yes", "yeah",
                 "yep", "nope", "hi", "hey", "bye", "the", "and", "but"}


def tag_unit(unit: Unit | str) -> str:
    """Assign a language tag to a single normalized unit."""
    surface = unit.surface if isinstance(unit, Unit) else str(unit)
    if not surface:
        return OTHER
    if any(is_han(c) for c in surface):
        return ZH if all(is_han(c) or c in "'-" for c in surface) else UNKNOWN
    core = surface.replace("'", "").replace("-", "")
    if not core:
        return OTHER
    if core.isdigit():
        return OTHER  # bare numerals: language of realization is unknown
    if not core.isascii() or not core.isalnum():
        return UNKNOWN
    low = surface.lower()
    if low in FILLERS:
        return OTHER
    if low in ROMANIZED_MANDARIN:
        return OTHER
    if low in SHORT_ENGLISH:
        return EN
    if len(core) == 1:
        # single stray letter: likely a spelled letter or annotation artefact
        return UNKNOWN
    if any(c.isdigit() for c in core) and any(c.isalpha() for c in core):
        return UNKNOWN  # alphanumeric codes ("3d", "b2b") need audit
    return EN


def tag_units(units: list[Unit]) -> list[str]:
    return [tag_unit(u) for u in units]


def is_content(unit: Unit | str, tag: str | None = None) -> bool:
    """Content units are the ones eligible for direction/POI use."""
    tag = tag if tag is not None else tag_unit(unit)
    return tag in (EN, ZH)


def tag_summary(all_units: list[tuple[str, str]]) -> dict:
    """Counts and the most frequent ambiguous surfaces, for the E1 audit."""
    counts = Counter(tag for _, tag in all_units)
    ambiguous = Counter(
        surface for surface, tag in all_units if tag in (UNKNOWN, OTHER)
    )
    latin_en = Counter(surface for surface, tag in all_units if tag == EN
                       and not any(is_han(c) for c in surface))
    return {
        "tag_counts": dict(counts),
        "top_ambiguous": ambiguous.most_common(50),
        "top_english": latin_en.most_common(50),
        "num_distinct_english": len(latin_en),
    }
