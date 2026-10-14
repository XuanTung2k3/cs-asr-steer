"""CPU tests for P1-E decoupled read/write audit helpers (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.decoupled_audit import (FREEZE_SOURCES, block_matrix, class_weights, first_order, fit_joint,  # noqa: E402
                                  fit_joint_head, head_apply, net_benefit, principal_cosines, top_right_singular,
                                  unscale_linear, verdict)


def test_unscale_matches_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12)) * rng.uniform(0.5, 3, 12) + rng.normal(size=12)
    y = (X[:, 0] - X[:, 3] > 0).astype(int)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X, y)
    w, b = unscale_linear(m[-1].coef_, m[-1].intercept_, m[0].mean_, m[0].scale_)
    assert np.allclose(X @ w[0] + b[0], m.decision_function(X), atol=1e-10)
    Y = rng.normal(size=(200, 4))
    r = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(X, Y)
    W, bb = unscale_linear(r[-1].coef_, r[-1].intercept_, r[0].mean_, r[0].scale_)
    assert W.shape == (4, 12) and np.allclose(X @ W.T + bb, r.predict(X), atol=1e-10)


def test_block_normalization_equal_energy_and_basis():
    rng = np.random.default_rng(1)
    B = block_matrix(rng.normal(size=1280) * 5, rng.normal(size=(4, 1280)) * 0.1)
    assert B.shape == (5, 1280)
    assert np.isclose(np.linalg.norm(B[0]), 1) and np.isclose(np.linalg.norm(B[1:]), 1)
    U, s = top_right_singular(B, 4)
    assert U.shape == (1280, 4) and np.allclose(U.T @ U, np.eye(4), atol=1e-12)
    U2, _ = top_right_singular(B, 4)
    assert np.array_equal(U, U2)  # deterministic, sign-fixed
    assert all(U[np.abs(U[:, j]).argmax(), j] > 0 for j in range(4))
    # U lies in the row space of B
    P = B.T @ np.linalg.pinv(B.T)
    assert np.allclose(P @ U, U, atol=1e-10) and min(principal_cosines(U, U)) > 1 - 1e-12


def test_first_order_and_net_benefit_signs():
    rng = np.random.default_rng(2)
    V = np.linalg.qr(rng.normal(size=(1280, 4)))[0]
    q = rng.normal(size=(6, 1280))
    hn = np.full(6, 10.0)
    assert np.allclose(first_order(hn, q, V, np.zeros((6, 4)), 0.1), 0)  # zero update -> zero change
    aligned = (q @ V) / np.linalg.norm(q @ V, axis=1, keepdims=True)
    g = first_order(hn, q, V, aligned, 0.1)
    assert (g > 0).all() and np.allclose(g, 0.1 * 10 * np.linalg.norm(q @ V, axis=1))
    assert (first_order(hn, q, V, -aligned, 0.1) < 0).all()  # anti-aligned update on a correct row = damage (negative)
    assert net_benefit(np.zeros(6, bool), g) == 0.0
    assert np.isclose(net_benefit(np.ones(6, bool), g), g.mean())


def test_class_weights_balanced():
    y = np.array([1] * 3 + [0] * 9)
    w = class_weights(y)
    assert np.isclose(w[y == 1].sum(), 0.5) and np.isclose(w[y == 0].sum(), 0.5)


def test_joint_head_deterministic_and_regularized():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(120, 4))
    y = (rng.random(120) < 0.3).astype(int)
    T = np.where(y[:, None] == 1, np.tile([1.0, 0, 0, 0], (120, 1)), 0.0)
    A1, b1 = fit_joint_head(Z, T, class_weights(y), 1e-3, 500)
    A2, b2 = fit_joint_head(Z, T, class_weights(y), 1e-3, 500)
    assert np.array_equal(A1, A2) and np.array_equal(b1, b2)
    Ab, _ = fit_joint_head(Z, T, class_weights(y), 1e6, 500)
    assert np.abs(Ab).max() < 1e-4  # strong ridge shrinks A to ~0
    assert np.allclose(head_apply(np.zeros((1, 4)), np.zeros((4, 4)), np.zeros(4)), 0)


def test_fit_joint_is_construction_only_with_oof_thresholds():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(200, 4)) + 3
    y = (Z[:, 0] > 3.5).astype(int)
    T = np.where(y[:, None] == 1, np.tile([0.0, 1.0, 0, 0], (200, 1)), 0.0)
    g = np.repeat(np.arange(10), 20)
    h = fit_joint(Z, T, y, g, [1e-3, 1e-1], 5, 300)
    assert np.allclose(h.mean, Z.mean(0)) and np.allclose(h.scale, Z.std(0))
    assert h.tau5 >= h.tau10 >= 0
    A, b = h.folded()
    assert np.allclose(head_apply(Z, np.array(A), np.array(b)), h(Z), atol=1e-10)  # folding keeps 20 parameters


def test_verdict_table_and_freeze_sources_exist():
    assert verdict(False, True, True, True) == "P1E_INCONCLUSIVE"
    assert verdict(True, True, True, True) == "P1E_DECOUPLED_FEASIBLE_EXPLORATORY"
    assert verdict(True, True, False, True) == "P1E_WHEN_ONLY"
    assert verdict(True, True, True, False) == "P1E_WHEN_ONLY"
    assert verdict(True, False, True, True) == "P1E_HOW_ONLY"
    assert verdict(True, False, False, True) == "P1E_NO_CONVINCING_SIGNAL"
    assert all((ROOT / p).exists() for p in FREEZE_SOURCES)
