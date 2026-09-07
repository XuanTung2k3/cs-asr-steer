"""DG-01 canonical retention: three MC §8 populations kept distinct."""
from __future__ import annotations

from csasr.evaluation import retention


def test_three_populations_are_independent():
    # one code-switched utterance: EN unit "school", matrix ZH units.
    refs = ["我 想 去 school"]
    baseline = ["我 想 去 school"]           # everything correct at baseline
    steered = ["我 想 去 school"]            # everything retained
    rep = retention.retention_report(refs, baseline, steered)
    assert rep["embedded_en"]["denominator"] == 1     # the EN word
    assert rep["embedded_en"]["rate"] == 1.0
    assert rep["matrix_zh"]["denominator"] == 3       # 我 想 去 (school segments to 1 word)
    assert rep["matrix_zh"]["rate"] == 1.0
    # no monolingual utterance present
    assert rep["monolingual"]["denominator"] == 0
    assert rep["monolingual"]["rate"] is None


def test_embedded_en_corruption_drops_rate():
    refs = ["我 想 去 school"]
    baseline = ["我 想 去 school"]           # EN correct at baseline
    steered = ["我 想 去 学 校"]             # EN unit lost
    rep = retention.retention_report(refs, baseline, steered)
    assert rep["embedded_en"]["denominator"] == 1
    assert rep["embedded_en"]["numerator"] == 0
    assert rep["embedded_en"]["rate"] == 0.0
    # matrix Mandarin was not baseline-correct-then-lost here
    assert rep["matrix_zh"]["rate"] == 1.0


def test_monolingual_population_separated_from_matrix():
    # a monolingual Mandarin utterance contributes only to monolingual retention
    refs = ["我 想 去 学 校"]
    baseline = ["我 想 去 学 校"]
    steered = ["我 想 去 学 校"]
    rep = retention.retention_report(refs, baseline, steered)
    assert rep["monolingual"]["denominator"] == 5
    assert rep["monolingual"]["rate"] == 1.0
    assert rep["matrix_zh"]["denominator"] == 0       # not a CS utterance
    assert rep["embedded_en"]["denominator"] == 0


def test_empty_population_rate_is_none_not_zero():
    refs = ["hello world"]        # monolingual English
    rep = retention.retention_report(refs, refs, refs)
    assert rep["matrix_zh"]["rate"] is None
    assert rep["embedded_en"]["rate"] is None
    assert rep["monolingual"]["rate"] == 1.0
