"""Focused CPU tests for BASIS-A's frozen vector and analysis semantics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from csasr.experiments.basis_ablation import (
    basis_geometry, construct_directions, dose, fit_pca_rows, normalize,
    principal_angles_degrees,
)
from csasr.models.hooks import apply_steering

REPO = Path(__file__).resolve().parents[1]


def test_all_frozen_directions_are_unit_and_mixtures_share_coefficients():
    rng = np.random.default_rng(8)
    raw, local, cond = rng.normal(size=(3, 32))
    d = construct_directions(raw, local, cond, a_local=.5, a_cond=.5)
    assert all(np.isclose(np.linalg.norm(d[k]), 1.0) for k in d)
    assert np.allclose(d["raw_cond"], normalize(.5 * d["raw"] + .5 * d["conditioning"]))
    assert np.allclose(d["local_cond"], normalize(.5 * d["local"] + .5 * d["conditioning"]))


def test_rho_scaling_is_direction_independent_and_zero_is_bit_identical():
    rng = np.random.default_rng(9)
    d = construct_directions(*rng.normal(size=(3, 16)))
    updates = [dose(v, 1.7, 4.0) for v in d.values()]
    assert all(np.isclose(np.linalg.norm(x), 1.7 * 4.0) for x in updates)
    h = torch.randn(2, 4, 16)
    out = apply_steering(h, torch.tensor(d["raw"], dtype=torch.float32), 0.0, 4.0, None, True)
    assert torch.equal(out, h)


def test_geometry_is_finite_and_equivalent_synthetic_spans():
    x = np.eye(5, 2)
    y = x @ np.array([[1., 2.], [-2., 1.]])
    angles = principal_angles_degrees(x, y)
    assert np.allclose(angles, [0., 0.], atol=1e-10)
    d = {"raw": normalize(x[:, 0]), "local": normalize(x[:, 0]),
         "conditioning": normalize(np.array([0., 0., 1., 0., 0.])),
         "raw_cond": normalize(x[:, 0] + np.array([0., 0., 1., 0., 0.])),
         "local_cond": normalize(x[:, 0] + np.array([0., 0., 1., 0., 0.]))}
    g = basis_geometry(d, a_local=.5, a_cond=.5)
    assert np.isfinite(np.asarray(g["cosine_matrix"])).all()
    assert g["subspace"]["equivalent_within_1e-10"]


def test_pca_fits_representation_rows_not_five_directions():
    rng = np.random.default_rng(10)
    rows = rng.normal(size=(27, 12))
    pca = fit_pca_rows(rows, n_components=3)
    assert pca["n_rows"] == 27 and pca["n_features"] == 12
    assert pca["components"].shape == (3, 12)
    assert pca["explained_variance_ratio"].shape == (3,)


def test_dtest_selection_guard_is_inherited():
    from steer_sweep.data import LockedSplitError, load_split
    cfg = json.loads("{}")
    cfg["v2_namespace"] = {"role_root": "/does/not/matter"}
    with pytest.raises(LockedSplitError):
        load_split("D-test", cfg)


def test_runner_has_no_training_or_legacy_post_ffn_stack():
    src = (REPO / "experiments/basis_ablation_frozen.py").read_text()
    assert ".backward(" not in src
    assert "requires_grad_(True)" not in src
    assert "DecoderSteeringHook" not in src
    assert "DecoderPostCrossAttnInterventionHook" not in src or "_decode" in src
