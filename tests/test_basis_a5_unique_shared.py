import numpy as np

from csasr.experiments.basis_a5_unique_shared import (
    ORTHOGONALITY_TOL,
    construct_unique_shared,
    rank_directions,
)


def _moments(x):
    x = np.asarray(x, dtype=np.float64)
    return x.T @ x, len(x), x.sum(axis=0)


def test_unique_shared_is_unit_orthogonal_and_sign_oriented():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(128, 12))
    b = rng.normal(size=(128, 12))
    # Give the group means a deterministic A-minus-B orientation.
    a[:, 0] += 3.0
    b[:, 1] += 1.0
    sa, na, ma = _moments(a)
    sb, nb, mb = _moments(b)
    out = construct_unique_shared(sa, na, ma, sb, nb, mb, rank=8, top_vectors=8)
    assert np.isfinite(out.v_unique).all()
    assert np.isfinite(out.v_shared).all()
    assert np.isclose(np.linalg.norm(out.v_unique), 1.0)
    assert np.isclose(np.linalg.norm(out.v_shared), 1.0)
    assert np.isclose(np.linalg.norm(out.unique_minus_shared), 1.0)
    assert abs(float(out.v_unique @ out.v_shared)) <= ORTHOGONALITY_TOL
    assert out.diagnostics["i_unique"] != out.diagnostics["i_shared"]
    delta = (ma / na) - (mb / nb)
    assert float(out.v_unique @ delta) >= -1e-10


def test_rank_grid_is_construction_only_and_deterministic():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(96, 16))
    b = rng.normal(size=(96, 16))
    args = (*_moments(a), *_moments(b))
    first = rank_directions(*args, ranks=(4, 8, 12))
    second = rank_directions(*args, ranks=(4, 8, 12))
    for rank in ("4", "8", "12"):
        assert np.array_equal(first[rank].v_unique, second[rank].v_unique)
        assert np.array_equal(first[rank].v_shared, second[rank].v_shared)


def test_zero_group_is_rejected():
    z = np.zeros((4, 4))
    with np.testing.assert_raises(ValueError):
        construct_unique_shared(z, 0, np.zeros(4), z, 1, np.ones(4))
