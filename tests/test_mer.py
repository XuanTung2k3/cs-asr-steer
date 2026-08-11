"""Mixed error rate, alignment and per-language breakdowns."""
from __future__ import annotations

import math

from csasr.evaluation.mer import align_tokens, corpus_mer, counts_from_ops, error_rate, mer


def test_perfect_match_is_zero():
    r = mer("我喜欢 project", "我喜欢 project")
    assert r["rate"] == 0.0
    assert r["num_ref"] == 4          # 3 Han chars + 1 English word


def test_substitution_deletion_insertion_counts():
    ops = align_tokens(["a", "b", "c"], ["a", "x", "c", "d"])
    c = counts_from_ops(ops)
    assert c["match"] == 2 and c["sub"] == 1 and c["ins"] == 1 and c["del"] == 0


def test_chinese_counted_per_character_english_per_word():
    r = error_rate(["我", "们", "like"], ["我", "们", "likes"])
    assert r["num_ref"] == 3
    assert math.isclose(r["rate"], 1 / 3)


def test_normalization_is_applied_inside_mer():
    assert mer("我 LIKE，Project。", "我 like project")["rate"] == 0.0


def test_corpus_breakdown_separates_languages():
    refs = ["我喜欢 project", "他用 python 写"]
    hyps = ["我喜欢 project", "他用 派森 写"]     # English word replaced by Han
    out = corpus_mer(refs, hyps)
    assert out["num_en_ref"] == 2
    assert out["en_wer"] > 0
    # the substituted Han tokens are counted as ZH insertions, so ZH is affected too
    assert out["mer"] > 0
    assert out["num_zh_ref"] == 6      # 我喜欢 + 他用写


def test_empty_hypothesis_is_all_deletions():
    r = mer("我喜欢", "")
    assert r["rate"] == 1.0 and r["del"] == 3
