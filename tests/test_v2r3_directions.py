"""Day 3 direction estimator: the parts that decide what a direction is.

The GPU extraction is exercised by the run itself; everything that shapes the
estimate -- pooling, residualisation, clipping, dialogue balancing, the prompt
subspace, and the LDA complement trick -- is pure and is pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.experiments import v2r3_directions as D


def test_frozen_configuration_records_the_decided_inputs():
    frozen = D.FROZEN_CONFIG
    assert frozen["span_family"] == "existing_ctc"
    assert frozen["ctc_convention"] == "blank_to_preceding"
    assert frozen["floor_ms"] == 400.0
    assert frozen["ridge_fit_split"] == "D-construct only"
    assert frozen["individual_contrasts_normalised"] is False
    assert frozen["prompt_subspace_rank"] == 1
    assert frozen["selects_layer_or_strength"] is False


@pytest.mark.parametrize("kind", ["hann", "central60", "uniform"])
def test_pooling_weights_sum_to_one(kind):
    for n in (1, 2, 5, 40):
        w = D.pool_weights(n, kind)
        assert len(w) == n
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_hann_is_centre_weighted_and_central60_is_a_window():
    hann = D.pool_weights(21, "hann")
    assert hann[10] > hann[0] and hann[10] > hann[-1]
    central = D.pool_weights(10, "central60")
    assert central[0] == 0.0 and central[-1] == 0.0
    assert (central[2:8] > 0).all()


def test_residualise_retains_the_intercept():
    """The mean effect must survive; only nuisance-predicted variation goes."""
    rng = np.random.default_rng(0)
    signal = np.tile(np.array([1.0, 0.0, 0.0]), (60, 1))
    nuisance = rng.standard_normal(60)
    contrasts = signal + nuisance[:, None] * np.array([0.0, 1.0, 0.0])
    design = nuisance[:, None]
    residual, info = D.residualise(contrasts, design)
    assert info["fitted"] is True
    assert info["energy_fraction_removed"] > 0.5      # the nuisance axis is removed

    # "intercept retained" means the sample mean survives on every axis --
    # only the nuisance-predicted *variation* is subtracted.
    assert residual.mean(axis=0) == pytest.approx(contrasts.mean(axis=0), abs=1e-9)
    assert residual.mean(axis=0)[0] == pytest.approx(1.0, abs=1e-9)
    # and the variation the nuisance explained is gone
    assert residual[:, 1].std() < 0.05 * contrasts[:, 1].std()
    assert residual[:, 0].std() == pytest.approx(contrasts[:, 0].std(), abs=1e-9)


def test_clipping_limits_norms_without_normalising_direction():
    y = np.array([[1.0, 0.0], [100.0, 0.0], [2.0, 0.0]])
    clipped, info = D.clip_norms(y, quantile=0.5)
    assert info["clipped"] >= 1
    assert np.linalg.norm(clipped[1]) < np.linalg.norm(y[1])
    # direction preserved, magnitudes not all equal -> not normalised
    assert clipped[1][0] > 0 and clipped[1][1] == 0
    assert len({round(float(np.linalg.norm(v)), 6) for v in clipped}) > 1


def test_dialogue_balanced_mean_gives_each_dialogue_equal_weight():
    y = np.array([[1.0, 0.0]] * 9 + [[0.0, 1.0]])     # 9 pairs in A, 1 in B
    dialogues = ["A"] * 9 + ["B"]
    direction, info = D.dialogue_balanced_mean(y, dialogues)
    assert info["G"] == 2
    # equal dialogue weight -> the single B pair counts as much as all nine A
    assert direction[0] == pytest.approx(0.5)
    assert direction[1] == pytest.approx(0.5)


def test_prompt_subspace_rank_one_is_the_mean_direction():
    prompts = np.array([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    basis = D.orthonormal_basis(prompts, 1)
    assert basis.shape == (3, 1)
    assert abs(abs(float(basis[0, 0])) - 1.0) < 1e-9
    contrasts = np.array([[5.0, 1.0, 0.0]])
    projected, energy = D.project_out(contrasts, basis)
    assert projected[0][0] == pytest.approx(0.0, abs=1e-9)   # removed
    assert projected[0][1] == pytest.approx(1.0)             # untouched
    assert 0.9 < energy < 1.0


def test_bootstrap_reports_G_and_is_seeded():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((40, 8)) + np.array([2.0] + [0.0] * 7)
    dialogues = [f"d{i % 10}" for i in range(40)]
    a = D.bootstrap_cosine(y, dialogues, draws=50)
    b = D.bootstrap_cosine(y, dialogues, draws=50)
    assert a == b                                   # seeded and reproducible
    assert a["G"] == 10
    assert a["cosine_p05"] <= a["cosine_p50"] <= a["cosine_p95"]


def test_shrinkage_lda_estimates_inside_the_complement():
    """The removed subspace must carry no LDA weight after mapping back."""
    rng = np.random.default_rng(2)
    dim = 6
    basis = np.zeros((dim, 1))
    basis[0, 0] = 1.0                                # remove axis 0
    positive = rng.standard_normal((80, dim)) + np.array([9.0, 1.0, 0, 0, 0, 0])
    negative = rng.standard_normal((80, dim)) - np.array([9.0, 1.0, 0, 0, 0, 0])
    weights = D.shrinkage_lda(positive, negative, basis)
    # axis 0 has by far the largest raw separation, yet must receive no weight
    assert abs(float(weights[0])) < 1e-8
    assert abs(float(weights[1])) > 1e-3


def test_conservative_and_construction_sets_never_share_a_split():
    """Construction is D-construct only; the dev subset is never fitted on."""
    import ast
    import pathlib

    source = pathlib.Path(D.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "D-dev-confirm" not in literals
    assert "D-test" not in literals
    assert D.FROZEN_CONFIG["ridge_fit_split"] == "D-construct only"


def test_attach_baseline_status_counts_correct_and_error_units():
    spans = pd.DataFrame([{"utterance_id": "u1", "unit_indices": [3, 4],
                           "n_units": 2}])
    poi = pd.DataFrame([{"utterance_id": "u1", "reference_unit_index": 3, "correct": True},
                        {"utterance_id": "u1", "reference_unit_index": 4, "correct": False}])
    out = D.attach_baseline_status(spans, poi)
    assert int(out["units_correct"].iloc[0]) == 1
    assert int(out["units_error"].iloc[0]) == 1
    assert bool(out["all_correct"].iloc[0]) is False
