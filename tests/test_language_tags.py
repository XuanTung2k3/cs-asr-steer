"""Normalization and language tagging (guide section 11)."""
from __future__ import annotations

from csasr.data.language_tags import EN, OTHER, UNKNOWN, ZH, is_content, tag_unit
from csasr.data.normalize import normalize_text, segment_units, tokenize


def test_markers_and_punctuation_removed():
    raw = "嗯，我是 <FIL/> Luna ，然后 ** graduate。"
    norm = normalize_text(raw)
    assert "<" not in norm and "*" not in norm and "，" not in norm
    assert "luna" in norm and "graduate" in norm


def test_segmentation_is_char_for_chinese_word_for_english():
    units = segment_units(normalize_text("我 like this project 的"))
    surfaces = [u.surface for u in units]
    assert surfaces == ["我", "like", "this", "project", "的"]
    assert [u.kind for u in units] == ["han", "latin", "latin", "latin", "han"]


def test_offsets_are_within_the_normalized_string():
    norm = normalize_text("所以你是在 high school 的时候")
    for u in segment_units(norm):
        assert norm[u.char_start:u.char_end] == u.surface


def test_tagging_rules():
    assert tag_unit("我") == ZH
    assert tag_unit("project") == EN
    assert tag_unit("i") == EN            # short English word, allow-listed
    assert tag_unit("uh") == OTHER        # non-lexical filler
    assert tag_unit("weibo") == OTHER     # romanized Mandarin
    assert tag_unit("123") == OTHER       # bare numeral
    assert tag_unit("q") == UNKNOWN       # stray single letter
    assert tag_unit("3d") == UNKNOWN      # alphanumeric code needs audit


def test_only_en_and_zh_are_content():
    assert is_content("project") and is_content("我")
    assert not is_content("uh") and not is_content("123")


def test_not_every_latin_token_is_english():
    tags = [tag_unit(t) for t in tokenize("uh weibo project 我")]
    assert tags == [OTHER, OTHER, EN, ZH]


def test_normalization_is_idempotent():
    raw = "呃，我 like <SPK/> Python！"
    once = normalize_text(raw)
    assert normalize_text(once) == once
