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
