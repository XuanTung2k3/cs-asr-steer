"""CPU tests for the P1-D conditional controllability audit helpers (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.conditional_audit import (FEATURE_INPUTS, best_balanced_threshold, build_features, cluster_bootstrap,  # noqa: E402
                                    fit_logistic, fit_ridge, fpr_threshold, nuisance_matrix, position_stratified_auc,
                                    rms_norm, unit, verdict)


def test_rms_norm_and_unit():
    h = np.random.default_rng(0).normal(size=(5, 1280)) * 7
    z = rms_norm(h, 1e-6)
    assert np.allclose(np.sqrt((z ** 2).mean(1)), 1, atol=1e-6)
    assert np.allclose(rms_norm(3 * h, 1e-6), z, atol=1e-6)  # scale invariant
    u = unit(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(u, [[0.6, 0.8], [0, 0]])


def test_fpr_threshold_and_balanced_threshold():
    neg = np.random.default_rng(1).normal(size=1000)
    for f in (0.05, 0.10):
        assert (neg >= fpr_threshold(neg, f)).mean() <= f + 1e-9
    y = np.array([0] * 50 + [1] * 50)
    s = np.r_[np.linspace(0, 1, 50), np.linspace(2, 3, 50)]
    thr = best_balanced_threshold(y, s)
    assert ((s >= thr) == (y == 1)).all()


def test_position_stratified_auc_removes_pure_position_signal():
    rng = np.random.default_rng(2)
    t = rng.integers(0, 30, 2000)
    y = (rng.random(2000) < 0.1 + 0.6 * (t > 10)).astype(int)  # errors later in the sequence
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, t) > 0.7
    r = position_stratified_auc(y, t.astype(float), t, [[0, 0], [1, 2], [3, 5], [6, 10], [11, 20], [21, 10 ** 5]])
    assert abs(r["stratified_auroc"] - 0.5) < 0.1 and r["supported_A"] == y.sum()


def test_cluster_bootstrap_deterministic_and_skips():
    g = np.repeat(np.arange(6), 10)
    a = cluster_bootstrap(g, lambda ii: float(len(ii)), 50, 42)
    b = cluster_bootstrap(g, lambda ii: float(len(ii)), 50, 42)
    assert a == b and a["skipped"] == 0
    c = cluster_bootstrap(g, lambda ii: None, 10, 42)
    assert c["ci95"] is None and c["skipped"] == 10


def test_verdict_table():
    assert verdict(False, "PASS", True) == "P1D_INCONCLUSIVE"
    assert verdict(True, "PASS", True) == "P1D_CONDITIONAL_INFORMATION_SIGNAL"
    assert verdict(True, "PASS", False) == "P1D_ERROR_DETECTION_ONLY"
    assert verdict(True, "FAIL", True) == "P1D_DIRECTION_ONLY"
    assert verdict(True, "CONFOUNDED_UNRESOLVED", True) == "P1D_DIRECTION_ONLY"
    assert verdict(True, "FAIL", False) == "P1D_NO_CONVINCING_SIGNAL"
    assert verdict(True, "CONFOUNDED_UNRESOLVED", False) == "P1D_INCONCLUSIVE"


def test_fit_logistic_is_construction_only_and_thresholds_from_oof():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 4)) + 5
    y = (X[:, 0] + rng.normal(size=400) * 0.5 > 5).astype(int)
    g = np.repeat(np.arange(20), 20)
    m, info = fit_logistic(X, y, g, [0.01, 1.0], 5)
    assert np.allclose(m[0].mean_, X.mean(0))  # scaler statistics from the fitted (construction) rows only
    assert info["cv_oof_auroc"] > 0.8 and info["tau_fpr05"] >= info["tau_fpr10"]


def test_fit_ridge_recovers_direction():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(300, 4))
    M = rng.normal(size=(4, 4))
    Y = unit(Z @ M.T + 0.05 * rng.normal(size=(300, 4)))
    g = np.repeat(np.arange(15), 20)
    m, info = fit_ridge(Z, Y, g, [0.01, 1.0, 100.0], 5)
    assert info["cv_oof_mean_cos"] > 0.8 and (unit(m.predict(Z)) * Y).sum(1).mean() > 0.8


def test_feature_allowlist_and_shapes():
    rng = np.random.default_rng(5)
    V = np.linalg.qr(rng.normal(size=(1280, 4)))[0]
    data = {"h": rng.normal(size=(6, 1280)), "V": V, "R": {s: np.linalg.qr(rng.normal(size=(1280, 4)))[0] for s in (42, 43, 44, 45, 46)},
            "t": np.arange(6), "duration": np.full(6, 3.0), "lid": ["<|zh|>"] * 3 + ["<|en|>"] * 3}
    F = build_features(data)
    assert F["V"].shape == (6, 4) and F["FULL"].shape == (6, 1280) and F["V+NUIS"].shape == (6, 9)
    assert np.allclose(F["V"], rms_norm(data["h"], 1e-6) @ V)
    assert np.allclose(nuisance_matrix([0, 3], [3.0, 3.0], ["<|zh|>", "<|en|>"])[:, 4], [1, 0])
    with pytest.raises(AssertionError):
        build_features(data, inputs=FEATURE_INPUTS + ("q",))
