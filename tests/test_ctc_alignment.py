"""Independent CTC forced alignment used to audit the DTW boundaries."""
from __future__ import annotations

import pytest
import torch

from csasr.data.ctc_alignment import CTCUnavailable, load_ctc_aligner, viterbi_align

BLANK = 0


def _posteriors(segments: list[tuple[int, int, int]], n_frames: int,
                vocab: int = 3) -> torch.Tensor:
    """Log-probs where each (start, end, token) segment dominates its frames."""
    lp = torch.full((n_frames, vocab), -10.0)
    for start, end, token in segments:
        lp[start:end, token] = 0.0
    return torch.log_softmax(lp, dim=-1)


def test_two_tokens_align_to_their_dominant_frames():
    lp = _posteriors([(0, 4, 1), (4, 8, 2)], 8)
    spans = viterbi_align(lp, [1, 2], BLANK)
    assert spans == [(0, 4), (4, 8)]


def test_spans_are_monotonic_and_disjoint():
    lp = _posteriors([(0, 3, 1), (3, 5, 2), (5, 9, 1)], 9)
    spans = viterbi_align(lp, [1, 2, 1], BLANK)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end <= next_start


def test_a_repeated_token_is_separated_by_a_blank():
    """CTC collapses adjacent identical labels, so the path must pass a blank."""
    lp = _posteriors([(0, 2, 1), (2, 4, BLANK), (4, 6, 1)], 6)
    spans = viterbi_align(lp, [1, 1], BLANK)
    assert spans[0][1] <= spans[1][0]
    assert spans[0] != spans[1]


def test_more_tokens_than_frames_is_rejected():
    lp = _posteriors([(0, 2, 1)], 2)
    with pytest.raises(ValueError, match="cannot align"):
        viterbi_align(lp, [1, 2, 1, 2], BLANK)


def test_empty_token_sequence_yields_no_spans():
    assert viterbi_align(_posteriors([], 4), [], BLANK) == []


def test_missing_model_degrades_instead_of_crashing():
    """An absent aligner must never fail E1; it is an optional auditor."""
    with pytest.raises(CTCUnavailable):
        load_ctc_aligner({"alignment": {"ctc": {}}})
    with pytest.raises(CTCUnavailable):
        load_ctc_aligner({"alignment": {"ctc": {"model_id": "/nonexistent/aligner"}}})


def test_the_romanizer_is_built_once_not_once_per_unit(monkeypatch):
    """`align_units` romanizes every unit separately; building a romanizer per
    call cost ~3.1 s each, so a 19-unit utterance spent ~59 s on romanization
    alone. That single line made CTC forced alignment project at ~80 GPU-hours
    and failed the L0 budget gate.
    """
    import sys
    import types

    import csasr.data.ctc_alignment as ctc

    constructions = []

    class FakeUroman:
        def __init__(self):
            constructions.append(1)

        def romanize_string(self, text):
            return str(text).upper()

    module = types.ModuleType("uroman")
    module.Uroman = FakeUroman
    monkeypatch.setitem(sys.modules, "uroman", module)

    ctc._uroman.cache_clear()
    try:
        out = [ctc.romanize(u) for u in ("我", "用", "machine", "做")]
        assert out == ["我", "用", "MACHINE", "做"]
        assert len(constructions) == 1, (
            f"romanizer rebuilt {len(constructions)} times for 4 units")
    finally:
        ctc._uroman.cache_clear()


def _viterbi_reference(log_probs, token_ids, blank_id=0):
    """The original scalar implementation, kept as the correctness oracle."""
    import numpy as _np
    import torch as _t

    if not token_ids:
        return []
    T, _ = log_probs.shape
    ext = [blank_id]
    for tok in token_ids:
        ext.extend([int(tok), blank_id])
    S = len(ext)
    neg = float("-inf")
    score = _t.full((S,), neg, dtype=_t.float64)
    score[0] = float(log_probs[0, ext[0]])
    if S > 1:
        score[1] = float(log_probs[0, ext[1]])
    back = _np.zeros((T, S), dtype=_np.int8)
    for t in range(1, T):
        prev = score
        cur = _t.full((S,), neg, dtype=_t.float64)
        for s in range(S):
            best, arg = prev[s], 0
            if s >= 1 and prev[s - 1] > best:
                best, arg = prev[s - 1], 1
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s - 2] and prev[s - 2] > best:
                best, arg = prev[s - 2], 2
            if best > neg:
                cur[s] = best + float(log_probs[t, ext[s]])
                back[t, s] = arg
        score = cur
    s = S - 1 if score[S - 1] >= score[S - 2] else S - 2
    path = _np.zeros(T, dtype=_np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
        s = max(0, s)
    spans = []
    for i in range(len(token_ids)):
        frames = _np.nonzero(path == 2 * i + 1)[0]
        spans.append((int(frames[0]), int(frames[-1]) + 1) if len(frames) else (-1, -1))
    return spans


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vectorised_viterbi_matches_the_scalar_original(seed):
    """Same trellis, same tie-breaks, same spans -- only the Python loop is gone.

    The inner loop indexed torch tensors one scalar at a time and cost ~4.4 s
    on a 550-frame, 301-state trellis. Vectorising over states must not change
    a single boundary, so it is checked against the original implementation.
    """
    from csasr.data.ctc_alignment import viterbi_align

    torch.manual_seed(seed)
    V, T, N = 12, 60, 7
    log_probs = torch.log_softmax(torch.randn(T, V), dim=-1)
    token_ids = [1 + (i % (V - 1)) for i in range(N)]
    assert viterbi_align(log_probs, token_ids) == _viterbi_reference(log_probs, token_ids)


def test_vectorised_viterbi_handles_repeated_tokens():
    """Repeats are the case the blank-skip rule exists for."""
    from csasr.data.ctc_alignment import viterbi_align

    torch.manual_seed(9)
    log_probs = torch.log_softmax(torch.randn(40, 8), dim=-1)
    token_ids = [3, 3, 5, 5, 5, 2]          # adjacent repeats may not skip a blank
    assert viterbi_align(log_probs, token_ids) == _viterbi_reference(log_probs, token_ids)


# --------------------------------------------------------------------------
# Han numerals: the whole of existing_ctc's invalid rate
# --------------------------------------------------------------------------
def test_han_numerals_are_pronounced_before_uroman_sees_them():
    """uroman turns `一` into the digit `1`, and MMS-FA's 31-symbol vocabulary has
    no digits, so `align_units` dropped every character of the romanization and
    the unit came back from `run_existing_ctc` as `mapping_failed`.

    Measured on job 38573: 2,283 of 88,426 CTC candidates invalid, all Mandarin,
    2,010 of them the single character `一`. A 2.58% invalid rate against a 1%
    limit, from one romanization gap.
    """
    from csasr.data.ctc_alignment import CJK_NUMERAL_PINYIN, pronounce_numerals

    # the fourteen surfaces that actually failed on the production sweep
    observed = "一十三二四五零六九百八千七陌"
    for char in observed:
        assert char in CJK_NUMERAL_PINYIN, char
        assert pronounce_numerals(char).strip() == CJK_NUMERAL_PINYIN[char]
        assert pronounce_numerals(char).strip().isalpha(), \
            "the substitution has to be letters; a digit is what broke it"


def test_a_multi_character_numeral_becomes_separate_syllables():
    from csasr.data.ctc_alignment import pronounce_numerals

    # uroman renders this as the digits `123`
    assert pronounce_numerals("一百二十三").split() == ["yi", "bai", "er", "shi",
                                                       "san"]


def test_non_numeral_text_is_left_for_uroman():
    from csasr.data.ctc_alignment import pronounce_numerals

    assert pronounce_numerals("我们") == "我们"
    assert pronounce_numerals("machine learning") == "machine learning"
    assert pronounce_numerals("") == ""


def test_the_pronounced_form_survives_a_letters_only_vocabulary():
    """The property that matters: after vocabulary filtering the unit still has
    tokens, so the acoustic model gets a chance to place a span. It is not
    relabelled valid -- CTC still has to find it."""
    from csasr.data.ctc_alignment import CJK_NUMERAL_PINYIN, pronounce_numerals

    letters_only = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
    for char in CJK_NUMERAL_PINYIN:
        kept = [c for c in pronounce_numerals(char).strip().lower()
                if c in letters_only]
        assert kept, f"{char} still romanizes to nothing this vocabulary has"


def test_every_mapped_reading_is_the_mandarin_one():
    """A wrong pronunciation would align to the wrong acoustics, which is worse
    than no span at all."""
    from csasr.data.ctc_alignment import CJK_NUMERAL_PINYIN

    expected = {"零": "ling", "〇": "ling", "一": "yi", "二": "er", "三": "san",
                "四": "si", "五": "wu", "六": "liu", "七": "qi", "八": "ba",
                "九": "jiu", "十": "shi", "百": "bai", "千": "qian",
                "壹": "yi", "贰": "er", "叁": "san", "肆": "si", "伍": "wu",
                "陆": "liu", "柒": "qi", "捌": "ba", "玖": "jiu", "拾": "shi",
                "佰": "bai", "仟": "qian", "陌": "mo"}
    assert CJK_NUMERAL_PINYIN == expected
