"""Direction accumulation: equal utterance weighting, statistics, resumability."""
from __future__ import annotations

import numpy as np
import torch

from csasr.directions.accumulators import DirectionAccumulatorSet, LayerAccumulator
from csasr.directions.controls import orthogonalized_random, random_direction, wrong_sign
from types import SimpleNamespace
from csasr.directions.encoder import control_directions


def test_every_utterance_gets_equal_weight():
    acc = LayerAccumulator(4)
    # one utterance with 100 EN frames, one with 2: the mean delta must not be
    # dominated by the long utterance
    acc.add_utterance(np.ones((100, 4)) * 1.0, np.zeros((100, 4)))
    acc.add_utterance(np.ones((2, 4)) * 3.0, np.zeros((2, 4)))
    assert np.allclose(acc.mean_delta(), np.full(4, 2.0))


def test_global_centroid_differs_from_within_utterance():
    acc = LayerAccumulator(2)
    acc.add_utterance(np.ones((100, 2)), np.zeros((100, 2)))
    acc.add_utterance(np.full((2, 2), 3.0), np.zeros((2, 2)))
    within = acc.mean_delta()
    glob = acc.global_delta()
    assert not np.allclose(within, glob)


def test_projection_std_matches_the_direct_computation():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(500, 6)) * np.array([1, 2, 3, 4, 5, 6])
    acc = LayerAccumulator(6)
    acc.add_frames(frames)
    d = np.zeros(6)
    d[2] = 1.0
    assert np.isclose(acc.projection_std(d), frames[:, 2].std(), rtol=1e-6)


def test_centroid_projections_and_midpoint():
    acc = LayerAccumulator(3)
    acc.add_utterance(np.tile([2.0, 0, 0], (5, 1)), np.tile([-2.0, 0, 0], (5, 1)))
    d = np.array([1.0, 0, 0])
    en, zh = acc.centroid_projections(d)
    assert np.isclose(en, 2.0) and np.isclose(zh, -2.0)
    assert np.isclose(0.5 * (en + zh), 0.0)


def test_accumulator_is_resumable_and_idempotent(tmp_path):
    accs = DirectionAccumulatorSet([0, 1], dim=3)
    accs.acc[0].add_utterance(np.ones((4, 3)), np.zeros((4, 3)))
    accs.mark("utt-a")
    path = tmp_path / "acc.npz"
    accs.save(path)

    reloaded = DirectionAccumulatorSet.load(path)
    assert reloaded.has("utt-a") and not reloaded.has("utt-b")
    assert np.allclose(reloaded.acc[0].mean_delta(), np.ones(3))
    assert reloaded.acc[0].num_utterances == 1

    # resuming must not double count
    reloaded.save(path)
    again = DirectionAccumulatorSet.load(path)
    assert again.acc[0].num_utterances == 1


def test_random_controls_are_unit_norm_deterministic_and_distinct():
    a = random_direction(64, seed=142)
    b = random_direction(64, seed=142)
    c = random_direction(64, seed=143)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert abs(float(a.norm()) - 1.0) < 1e-6


def test_wrong_sign_is_the_exact_negation():
    d = random_direction(32, seed=42)
    assert torch.allclose(wrong_sign(d), -d)
    assert abs(float(wrong_sign(d).norm()) - 1.0) < 1e-6


def test_orthogonalized_random_removes_the_language_component():
    d = random_direction(32, seed=42)
    r = orthogonalized_random(d, seed=142)
    assert abs(float(r @ d)) < 1e-5

def test_random_control_filename_seed_stays_direction_seed():
    bundle = SimpleNamespace()
    primary = {15: {
        "direction": random_direction(8, seed=42),
        "direction_type": "within_utterance_correct_only",
        "seed": 42,
        "projection_std": 1.5,
        "centroid_en_projection": 1.0,
        "centroid_zh_projection": -1.0,
        "midpoint_projection": 0.0,
    }}
    rec = control_directions(bundle, primary, [142], "manifest", "v1")[
        "random_s142"][15]
    assert rec["seed"] == 42
    assert rec["control_seed"] == 142
    assert rec["direction_type"] == "random_s142"
