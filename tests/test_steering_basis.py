"""Focused DG-03 tests for the canonical v6 steering-basis builder.

Pure CPU assembly logic (no model): the v6 aggregate-then-residualize order,
normalization, orthogonality, degeneracy guards, deterministic serialization,
controls, the diagnostic dose, and the {16,24} layer guard.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from csasr.directions.steering_basis import (
    CONTRACT_LAYERS,
    build_basis,
    conditioning_residualized_local,
    diagnostic_dose,
    language_conditioning,
    raw_from_states,
    raw_language_contrast,
    write_basis_artifact,
)
from csasr.directions.controls import random_direction, wrong_sign

DIM = 8


def _rng(seed=0):
    return np.random.default_rng(seed)


# 1. raw mean-difference formula: balanced mean of (E−M) == μ_E − μ_M (1:1 pairing)
def test_raw_is_mean_difference():
    r = _rng(1)
    E = r.standard_normal((20, DIM))
    M = r.standard_normal((20, DIM))
    v_raw = raw_from_states(E, M)                       # no groups -> plain means
    assert np.allclose(v_raw, E.mean(0) - M.mean(0))
    # and equals the mean of the per-span contrasts
    assert np.allclose(v_raw, raw_language_contrast(E - M))


# 2. aggregate-then-residualize order (NOT per-sample residualize-then-aggregate)
def test_aggregate_then_residualize_order():
    r = _rng(2)
    contrasts = r.standard_normal((30, DIM))
    v_cond = language_conditioning(r.standard_normal((30, DIM)))
    b = build_basis(layer=16, contrasts_EM=contrasts,
                    cond_contrasts=r.standard_normal((10, DIM)) + 0,  # placeholder cond input
                    enforce_contract_layer=True)
    # builder residualizes the AGGREGATE v_raw against v_cond (closed form)
    v_raw = raw_language_contrast(contrasts)
    v_local_v6 = conditioning_residualized_local(v_raw, b["v_cond"])
    assert np.allclose(b["v_local"], v_local_v6)
    # v_raw is the aggregate itself, not any per-sample-normalized aggregate
    assert np.allclose(b["v_raw"], v_raw)
    assert not np.allclose(b["v_raw"], v_raw / np.linalg.norm(v_raw))   # v_raw NOT unit-normalized
    # a per-sample-NORMALIZED-then-averaged variant (nonlinear) differs -> the
    # builder does not normalize per span before aggregating
    vc = b["v_cond"]
    per_sample = contrasts - (contrasts @ vc)[:, None] * vc
    per_sample = per_sample / np.linalg.norm(per_sample, axis=1, keepdims=True)
    nonlinear = per_sample.mean(0)
    nonlinear = nonlinear / np.linalg.norm(nonlinear)
    assert not np.allclose(b["v_local"], nonlinear)


# 3. unit normalization of v_cond and v_local
def test_unit_norms():
    r = _rng(3)
    b = build_basis(layer=24, contrasts_EM=r.standard_normal((40, DIM)),
                    cond_contrasts=r.standard_normal((15, DIM)))
    assert b["metrics"]["v_cond_norm"] == pytest.approx(1.0, abs=1e-9)
    assert b["metrics"]["v_local_norm"] == pytest.approx(1.0, abs=1e-9)


# 4. v_local orthogonal to v_cond
def test_local_orthogonal_to_cond():
    r = _rng(4)
    b = build_basis(layer=16, contrasts_EM=r.standard_normal((50, DIM)),
                    cond_contrasts=r.standard_normal((20, DIM)))
    assert abs(b["metrics"]["cos_local_cond"]) < 1e-9
    assert b["V0"].shape == (DIM, 2)
    assert np.allclose(b["V0"][:, 0], b["v_local"])
    assert np.allclose(b["V0"][:, 1], b["v_cond"])


# 5. degenerate-vector guards
def test_degenerate_guards():
    # v_raw parallel to v_cond -> residual is ~0 -> raises
    v_cond_dir = np.zeros(DIM); v_cond_dir[0] = 1.0
    contrasts = np.tile(v_cond_dir, (5, 1)) * 2.0        # v_raw ∝ v_cond
    cond = np.tile(v_cond_dir, (5, 1))
    with pytest.raises(ValueError):
        build_basis(layer=16, contrasts_EM=contrasts, cond_contrasts=cond)
    # empty population
    with pytest.raises(ValueError):
        raw_language_contrast(np.zeros((0, DIM)))
    # zero conditioning vector cannot normalize
    with pytest.raises(ValueError):
        language_conditioning(np.zeros((4, DIM)))


# 6. deterministic artifact hash/serialization
def test_artifact_is_deterministic(tmp_path):
    r = _rng(6)
    contrasts = r.standard_normal((25, DIM))
    cond = r.standard_normal((12, DIM))
    prov = {"model_id": "openai/whisper-large-v3", "git_commit": "abc123",
            "dataset_role": "D-construct", "dataset_fingerprint": None,
            "construction_config_hash": "cfg", "mean_site_norm": 3.0, "seeds": [0, 1, 2],
            "reused_cached_activations": False}
    b1 = build_basis(layer=16, contrasts_EM=contrasts, cond_contrasts=cond)
    b2 = build_basis(layer=16, contrasts_EM=contrasts, cond_contrasts=cond)
    r1 = write_basis_artifact(tmp_path / "a", b1, prov)
    r2 = write_basis_artifact(tmp_path / "b", b2, prov)
    assert r1["tensor_hashes"] == r2["tensor_hashes"]
    j1 = json.loads((tmp_path / "a" / "steering_basis_v1_L16.json").read_text())
    j2 = json.loads((tmp_path / "b" / "steering_basis_v1_L16.json").read_text())
    assert j1 == j2
    assert j1["direction_names"] == ["raw_language_contrast", "language_conditioning",
                                     "conditioning_residualized_local"]


# 7. sign-reversed control
def test_sign_reversed_control():
    d = random_direction(DIM, seed=0)
    assert torch.allclose(wrong_sign(d), -d)


# 8. matched-norm random control
def test_random_control_matched_norm_and_seeded():
    a0 = random_direction(DIM, seed=0)
    a0b = random_direction(DIM, seed=0)
    a1 = random_direction(DIM, seed=1)
    assert float(a0.norm()) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(a0, a0b)                       # deterministic per seed
    assert not torch.allclose(a0, a1)                    # different seeds differ


# 9. matched-energy / diagnostic dose logic
def test_diagnostic_dose():
    dose = diagnostic_dose(scale=4.0, rho=1.0)
    assert dose["alpha"] == 1.0 and dose["scale"] == 4.0
    assert dose["nominal_update_norm"] == pytest.approx(4.0)
    # controls reuse the same (alpha, scale) -> identical nominal magnitude
    assert diagnostic_dose(4.0)["nominal_update_norm"] == dose["nominal_update_norm"]
    with pytest.raises(ValueError):
        diagnostic_dose(scale=0.0)


# 10. layer guard {16, 24}
def test_layer_guard():
    assert CONTRACT_LAYERS == (16, 24)
    r = _rng(10)
    for k in (16, 24):
        build_basis(layer=k, contrasts_EM=r.standard_normal((10, DIM)),
                    cond_contrasts=r.standard_normal((6, DIM)))
    with pytest.raises(ValueError, match="not in contract layers"):
        build_basis(layer=8, contrasts_EM=r.standard_normal((10, DIM)),
                    cond_contrasts=r.standard_normal((6, DIM)))
