"""Mask construction: exactness, tapering, expansion, jitter, clipping."""
from __future__ import annotations

import numpy as np

from csasr.steering.masks import (
    MaskSpec,
    build_gain,
    hard_mask,
    jitter_span,
    derive_jitter_seed,
    ms_to_frames,
    spec_from_kind,
    tapered_mask,
)

N, VALID = 100, 60


def test_ms_to_frames_uses_the_model_step():
    assert ms_to_frames(100, 0.02) == 5
    assert ms_to_frames(50, 0.02) == 2 or ms_to_frames(50, 0.02) == 3


def test_hard_mask_is_exactly_the_span():
    g = hard_mask(10, 20, N, VALID)
    assert g[:10].sum() == 0 and g[20:].sum() == 0
    assert np.all(g[10:20] == 1.0)


def test_tapered_mask_rises_and_falls_around_the_core():
    g = tapered_mask(20, 30, N, VALID, shoulder=5)
    assert np.all(g[20:30] == 1.0)
    left = g[15:20]
    assert np.all(np.diff(left) > 0) and 0 < left[0] < 1
    right = g[30:35]
    assert np.all(np.diff(right) < 0)
    assert g[:15].sum() == 0 and g[35:].sum() == 0


def test_masks_never_touch_padding_frames():
    for kind in ("EXACT_HARD", "EXACT_TAPER", "EXPAND_200", "JITTER_200", "BOUNDARY_ONLY"):
        spec = spec_from_kind(kind, 0.02, seed=242)
        g, meta = build_gain(spec, VALID - 3, VALID - 1, N, VALID)
        assert g[VALID:].sum() == 0, kind
        assert meta["applied_end"] <= VALID


def test_expansion_widens_and_jitter_moves():
    core = (25, 35)
    exact, _ = build_gain(spec_from_kind("EXACT_TAPER", 0.02), *core, N, VALID)
    wide, _ = build_gain(spec_from_kind("EXPAND_100", 0.02), *core, N, VALID)
    assert (wide > 0).sum() > (exact > 0).sum()

    spec = spec_from_kind("JITTER_100", 0.02, seed=242)
    g, meta = build_gain(spec, *core, N, VALID)
    assert "jitter_start_frames" in meta and "jitter_end_frames" in meta
    assert abs(meta["jitter_start_frames"]) <= spec.jitter_frames


def test_jitter_is_deterministic_per_seed_and_varies_across_seeds():
    a, _ = build_gain(spec_from_kind("JITTER_100", 0.02, seed=242), 25, 35, N, VALID)
    b, _ = build_gain(spec_from_kind("JITTER_100", 0.02, seed=242), 25, 35, N, VALID)
    c, _ = build_gain(spec_from_kind("JITTER_100", 0.02, seed=246), 25, 35, N, VALID)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_jitter_preserves_start_before_end():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s, e, ds, de = jitter_span(20, 21, 10, rng, VALID)
        assert s < e


def test_boundary_only_covers_both_switch_points_not_the_middle():
    spec = spec_from_kind("BOUNDARY_ONLY", 0.02, boundary_only_ms=60)
    g, _ = build_gain(spec, 10, 40, N, VALID)
    assert g[10] > 0 and g[39] > 0
    assert g[25] == 0


def test_global_mask_covers_every_valid_frame():
    g, meta = build_gain(MaskSpec(kind="GLOBAL"), 0, 0, N, VALID)
    assert g[:VALID].sum() == VALID and g[VALID:].sum() == 0


def test_whole_word_plus_context_uses_the_context_span():
    spec = spec_from_kind("WHOLE_WORD_PLUS_CONTEXT", 0.02)
    spec.context_span = (15, 45)
    g, meta = build_gain(spec, 25, 35, N, VALID)
    assert meta["applied_start"] == 15 and meta["applied_end"] == 45

def test_jitter_seed_is_stable_but_independent_per_utterance():
    a = derive_jitter_seed(242, "utt-a")
    b = derive_jitter_seed(242, "utt-b")
    assert a == derive_jitter_seed(242, "utt-a")
    assert a != b
    ga, ma = build_gain(spec_from_kind("JITTER_100", 0.02, seed=a), 25, 35, N, VALID)
    gb, mb = build_gain(spec_from_kind("JITTER_100", 0.02, seed=b), 25, 35, N, VALID)
    assert (ma["jitter_start_frames"], ma["jitter_end_frames"]) != (
        mb["jitter_start_frames"], mb["jitter_end_frames"])
    assert not np.array_equal(ga, gb)
