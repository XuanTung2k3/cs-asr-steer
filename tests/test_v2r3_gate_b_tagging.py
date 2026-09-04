"""Language tagging must receive the Unit, not its repr.

`tag_unit` accepts a `Unit` or a surface string.  Passing `str(unit)` hands it
`"Unit(surface='period', kind='latin', ...)"`, which tags as UNKNOWN rather than
EN -- so every embedded-English count silently becomes zero and every English
unit is silently counted as monolingual material.
"""
from __future__ import annotations

from csasr.data.language_tags import EN, tag_unit
from csasr.data.normalize import normalize_and_segment

MIXED = "就是那一个 period 在时间 high school 比较简单"


def test_embedded_english_units_are_tagged() -> None:
    _, units = normalize_and_segment(MIXED)
    english = [u.surface for u in units if tag_unit(u) == EN]
    assert english == ["period", "high", "school"]


def test_the_repr_form_finds_nothing() -> None:
    """The failing fixture: this is what the defect did."""
    _, units = normalize_and_segment(MIXED)
    assert [u.surface for u in units if tag_unit(str(u)) == EN] == []


def test_monolingual_retention_excludes_english() -> None:
    from csasr.experiments.v2r3_gate_b import monolingual_retention
    out = monolingual_retention(MIXED, MIXED, MIXED)
    _, units = normalize_and_segment(MIXED)
    n_zh = sum(1 for u in units if tag_unit(u) != EN)
    assert out["monolingual_baseline_correct"] <= n_zh
    assert out["monolingual_lost"] == 0
