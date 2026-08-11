"""E1 unit tests 3, 4, 7, 8: frame ranges, padding, boundary trim, synthetic case."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.data.alignment import core_frames, dtw, median_filter_1d, token_frame_spans, validate_unit_table


def test_known_one_second_example_maps_exactly(tiny_bundle):
    # 0.02 s per frame: [0.00, 1.00) -> frames [0, 50)
    s, e = tiny_bundle.sec_to_frames(0.0, 1.0, num_valid=50)
    assert (s, e) == (0, 50)
    s, e = tiny_bundle.sec_to_frames(0.10, 0.30, num_valid=50)
    assert (s, e) == (5, 15)


def test_spans_are_half_open_and_non_empty(tiny_bundle):
    s, e = tiny_bundle.sec_to_frames(0.201, 0.209, num_valid=50)
    assert s < e
    # floor(start) / ceil(end)
    assert s == 10 and e == 11


def test_clamped_to_valid_frames_never_touching_padding(tiny_bundle):
    s, e = tiny_bundle.sec_to_frames(0.9, 5.0, num_valid=20)
    assert e <= 20 and s < e
    assert tiny_bundle.valid_frames(0.35) == 18
    assert tiny_bundle.valid_frames(1e6) == tiny_bundle.max_encoder_frames


def test_boundary_trim_removes_exactly_the_requested_frames():
    assert core_frames(10, 30, 5) == (15, 25)
    # trimming everything yields an empty span rather than an inverted one
    s, e = core_frames(10, 14, 5)
    assert s == e == 10


def test_validate_unit_table_rejects_padding_labels(tiny_bundle):
    manifest = pd.DataFrame([{"utterance_id": "u1", "duration_sec": 0.2}])
    good = pd.DataFrame([{"utterance_id": "u1", "unit_id": 0, "start_frame": 0,
                          "end_frame": 10, "language_tag": "ZH"}])
    validate_unit_table(good, manifest, 50, 0.02)

    bad = good.copy()
    bad.loc[0, "end_frame"] = 40          # 0.8 s of audio that does not exist
    with pytest.raises(AssertionError, match="padding"):
        validate_unit_table(bad, manifest, 50, 0.02)


def test_dtw_recovers_a_diagonal_path():
    cost = np.ones((4, 8))
    for i in range(4):
        cost[i, i * 2] = 0.0
        cost[i, i * 2 + 1] = 0.0
    ti, tj = dtw(cost)
    assert ti[0] == 0 and ti[-1] == 3
    assert np.all(np.diff(ti) >= 0) and np.all(np.diff(tj) >= 0)
    for k in range(4):
        frames = tj[ti == k]
        assert set(frames.tolist()) <= {2 * k, 2 * k + 1}


def test_median_filter_shape_and_smoothing():
    x = np.zeros((2, 3, 11))
    x[..., 5] = 10.0
    y = median_filter_1d(x, 7)
    assert y.shape == x.shape
    assert y.max() == 0.0        # an isolated spike is removed

def test_token_spans_are_disjoint_and_use_eos_as_final_boundary():
    ti = np.array([0, 0, 1, 1, 2, 2])
    tj = np.array([0, 1, 1, 2, 2, 3])
    starts, ends = token_frame_spans(ti, tj, n_text=2, n_valid=10)
    assert np.all(starts < ends)
    assert np.all(ends[:-1] <= starts[1:])
    assert ends[-1] == 2

def test_token_spans_reject_more_tokens_than_frames():
    with pytest.raises(ValueError, match="cannot fit"):
        token_frame_spans(np.array([0]), np.array([0]), n_text=3, n_valid=2)


def _units(rows):
    import pandas as pd
    cols = ["utterance_id", "unit_id", "surface", "language_tag",
            "start_sec", "end_sec", "start_frame", "end_frame"]
    return pd.DataFrame(rows, columns=cols)


def test_shared_boundary_frame_is_resolved_by_truncating_the_earlier_unit():
    """floor/ceil rounding makes adjacent spans share one frame (guide section 12)."""
    from csasr.data.alignment import resolve_boundary_overlaps

    units = _units([
        ("u1", 0, "你好", "ZH", 0.00, 2.30, 0, 115),
        ("u1", 1, "common", "EN", 2.28, 2.92, 114, 146),
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    assert info["overlaps_resolved"] == 1 and info["overlaps_unresolved"] == 0
    assert int(fixed.loc[0, "end_frame"]) == 114      # earlier unit truncated
    assert int(fixed.loc[1, "start_frame"]) == 114    # EN onset preserved
    assert fixed.loc[0, "end_sec"] <= 2.30


def test_units_sharing_a_start_frame_delay_the_later_unit_instead():
    from csasr.data.alignment import resolve_boundary_overlaps

    units = _units([
        ("u1", 0, "是", "ZH", 1.00, 1.20, 50, 60),
        ("u1", 1, "yes", "EN", 1.00, 1.60, 50, 80),
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    assert info["overlaps_resolved"] == 1 and info["overlaps_unresolved"] == 0
    assert int(fixed.loc[1, "start_frame"]) == 60
    assert int(fixed.loc[1, "end_frame"]) > int(fixed.loc[1, "start_frame"])


def test_same_language_neighbours_are_left_alone():
    from csasr.data.alignment import resolve_boundary_overlaps

    units = _units([
        ("u1", 0, "你", "ZH", 0.00, 1.00, 0, 50),
        ("u1", 1, "好", "ZH", 0.98, 1.40, 49, 70),
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    assert info["overlaps_resolved"] == 0
    assert int(fixed.loc[0, "end_frame"]) == 50       # untouched


def test_validate_unit_table_reports_instead_of_raising_when_not_strict(tiny_bundle):
    """E1 needs a clean gate failure, not a crash that discards every artifact."""
    from csasr.data.alignment import validate_unit_table
    import pandas as pd

    bad = _units([
        ("u1", 0, "你", "ZH", 0.00, 1.00, 0, 60),
        ("u1", 1, "hi", "EN", 0.40, 1.00, 20, 50),
    ])
    manifest = pd.DataFrame([{"utterance_id": "u1", "duration_sec": 1.4}])
    report = validate_unit_table(bad, manifest, 100, 0.02, strict=False)
    assert report["conflicting_language_overlaps"] == 1


def test_fully_nested_conflict_forfeits_the_language_tag(tiny_bundle):
    """A nested EN span inside a ZH one cannot be separated by moving an edge.

    Neither claim is better evidenced, so the nested unit is excluded from the
    language sets rather than repaired by invention (job 31473 hit exactly one
    such case in 178,033 units).
    """
    from csasr.data.alignment import resolve_boundary_overlaps, validate_unit_table
    import pandas as pd

    units = _units([
        ("u1", 0, "这个", "ZH", 0.00, 1.20, 0, 60),
        ("u1", 1, "the", "EN", 0.00, 0.24, 0, 12),
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    assert info["overlaps_dropped"] == 1
    assert info["overlaps_unresolved"] == 0
    assert fixed.loc[1, "language_tag"] == "UNKNOWN"
    assert int(fixed.loc[0, "end_frame"]) == 60        # enclosing span untouched

    manifest = pd.DataFrame([{"utterance_id": "u1", "duration_sec": 1.4}])
    report = validate_unit_table(fixed, manifest, 100, 0.02, strict=False)
    assert report["conflicting_language_overlaps"] == 0


def test_dropped_unit_does_not_become_a_conflict_for_its_successor(tiny_bundle):
    """Once a nested unit forfeits its tag it must leave the conflict graph."""
    from csasr.data.alignment import resolve_boundary_overlaps, validate_unit_table
    import pandas as pd

    units = _units([
        ("u1", 0, "这个", "ZH", 0.00, 1.20, 0, 60),
        ("u1", 1, "the", "EN", 0.00, 0.24, 0, 12),   # nested -> dropped
        ("u1", 2, "好", "ZH", 0.20, 0.60, 10, 30),   # overlaps the dropped unit
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    assert info["overlaps_dropped"] == 1
    # the dropped unit's span is left exactly as it was; only its tag changed
    assert int(fixed.loc[1, "start_frame"]) == 0 and int(fixed.loc[1, "end_frame"]) == 12
    assert fixed.loc[2, "language_tag"] == "ZH"

    manifest = pd.DataFrame([{"utterance_id": "u1", "duration_sec": 1.4}])
    assert validate_unit_table(fixed, manifest, 100, 0.02,
                               strict=False)["conflicting_language_overlaps"] == 0


def test_identical_conflicting_spans_both_forfeit_their_claim(tiny_bundle):
    """Two words given the same frames: no basis to prefer either (job 31473)."""
    from csasr.data.alignment import resolve_boundary_overlaps, validate_unit_table
    import pandas as pd

    units = _units([
        ("u1", 0, "life", "EN", 8.60, 9.56, 430, 478),
        ("u1", 1, "所", "ZH", 8.60, 9.56, 430, 478),
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    assert info["overlaps_dropped"] == 2
    assert set(fixed["language_tag"]) == {"UNKNOWN"}
    manifest = pd.DataFrame([{"utterance_id": "u1", "duration_sec": 10.6}])
    assert validate_unit_table(fixed, manifest, 1500, 0.02,
                               strict=False)["conflicting_language_overlaps"] == 0


def test_dropping_a_unit_exposes_its_neighbours_to_each_other(tiny_bundle):
    """The real residual conflict in job 31473: a drop makes a new pair adjacent.

    `life` and `所` share 430-478 exactly; once `所` is withdrawn, `life` (EN) and
    the following `以` (ZH, starting at 477) become neighbours and overlap by one
    frame. A sweep over the original pair list never examines that pair.
    """
    from csasr.data.alignment import resolve_boundary_overlaps, validate_unit_table
    import pandas as pd

    units = _units([
        ("u1", 0, "的", "ZH", 8.18, 8.60, 409, 430),
        ("u1", 1, "life", "EN", 8.60, 9.56, 430, 478),
        ("u1", 2, "所", "ZH", 8.60, 9.56, 430, 478),
        ("u1", 3, "以", "ZH", 9.54, 9.82, 477, 491),
    ])
    fixed, info = resolve_boundary_overlaps(units, 0.02)
    manifest = pd.DataFrame([{"utterance_id": "u1", "duration_sec": 10.6}])
    report = validate_unit_table(fixed, manifest, 1500, 0.02, strict=False)
    assert report["conflicting_language_overlaps"] == 0
    assert report["empty_spans"] == 0
    # and the pass must be a fixed point
    _, again = resolve_boundary_overlaps(fixed, 0.02)
    assert again["overlaps_resolved"] == 0 and again["overlaps_dropped"] == 0


def test_overlap_resolution_postcondition_holds_on_adversarial_tables(tiny_bundle):
    """Whatever the aligner emits, no EN frame may also be a ZH frame.

    Random tables pile units onto identical frames far more aggressively than DTW
    does; two earlier single-pass implementations passed the hand-written cases
    and still left conflicts here.
    """
    import numpy as np
    import pandas as pd
    from csasr.data.alignment import resolve_boundary_overlaps, validate_unit_table

    rng = np.random.default_rng(0)
    for _ in range(300):
        rows, frame = [], 0
        for i in range(int(rng.integers(2, 10))):
            start = max(0, frame - int(rng.integers(0, 5)))
            end = start + int(rng.integers(1, 12))
            rows.append(("u", i, "x", str(rng.choice(["EN", "ZH"])),
                         start * 0.02, end * 0.02, start, end))
            frame = end
        units = _units(rows)
        fixed, _ = resolve_boundary_overlaps(units, 0.02)
        manifest = pd.DataFrame([{"utterance_id": "u", "duration_sec": (frame + 50) * 0.02}])
        report = validate_unit_table(fixed, manifest, 1500, 0.02, strict=False)
        assert report["conflicting_language_overlaps"] == 0
        assert report["empty_spans"] == 0
        assert len(fixed) == len(units)          # rows are retagged, never removed
        _, again = resolve_boundary_overlaps(fixed, 0.02)
        assert again["overlaps_resolved"] == 0 and again["overlaps_dropped"] == 0
