"""The Mandarin control step must never collapse onto the switch onset.

Day 6's `jitter_zh -200 ms` cell reported exactly +0.0000 [0, 0] because the
sign mapping produced `control = onset - 1 - (-1) = onset`.  The contrast
`block[onset] - block[control]` was then identically zero, so the "direction"
was the zero vector and correct/null margins were bit-identical.  That is a
silent null, not a measurement.
"""
from __future__ import annotations

import pytest

from csasr.experiments.v2r3_directions import control_step_for

ONSET = 12
#: The Day 6 grid, in decoder steps, both signs.
OFFSET_GRID = (-3, -2, -1, 0, 1, 2, 3)


@pytest.mark.parametrize("offset", OFFSET_GRID)
def test_control_is_strictly_before_onset(offset: int) -> None:
    assert control_step_for(ONSET, offset) < ONSET


def test_default_is_the_immediately_preceding_step() -> None:
    assert control_step_for(ONSET) == ONSET - 1
    assert control_step_for(ONSET, 0) == ONSET - 1


def test_positive_offset_moves_the_control_earlier() -> None:
    assert control_step_for(ONSET, 1) == ONSET - 2
    assert control_step_for(ONSET, 2) == ONSET - 3


def test_negative_offset_is_clamped_not_collapsed() -> None:
    """The failing fixture: the pre-fix expression put the control ON the onset."""
    collapsed = ONSET - 1 - (-1)          # the defective Day 6 expression
    assert collapsed == ONSET             # documents the defect
    assert control_step_for(ONSET, -1) == ONSET - 1
    assert control_step_for(ONSET, -1) != collapsed


def test_contrast_cannot_be_identically_zero() -> None:
    """A collapsed control makes onset and control the same step."""
    for offset in OFFSET_GRID:
        assert control_step_for(ONSET, offset) != ONSET
