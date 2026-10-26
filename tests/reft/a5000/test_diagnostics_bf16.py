"""RC7-0C regression: N4 failed with `TypeError: Got unsupported ScalarType BFloat16` at the probe NumPy boundary."""
import numpy as np
import pytest
import torch

from reft.a5000.diagnostics import controlled_readout, fit_probe


def probe_array(x):
    from reft.a5000.diagnostics import probe_array as f  # RC7-0C helper; absent before the repair
    return f(x)


def _states(n=48, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = torch.randn(n, d, generator=g) + 1.5 * y[:, None].float()
    return x, y


def test_root_cause_bf16_tensor_has_no_numpy_dtype():
    with pytest.raises(TypeError, match='BFloat16'):
        torch.zeros(2, dtype=torch.bfloat16).numpy()


def test_probe_array_casts_bf16_exactly_and_preserves_shape():
    x = torch.randn(5, 3).to(torch.bfloat16)
    a = probe_array(x)
    assert a.dtype == np.float32 and a.shape == (5, 3)
    assert np.array_equal(a, x.float().numpy())  # bf16 -> fp32 is exact


def test_probe_array_leaves_integer_tensors_untouched():
    y = torch.tensor([0, 1, -100])
    assert probe_array(y).dtype == np.int64


def test_fit_probe_bf16_matches_its_exact_fp32_cast():
    x, y = _states()
    xb = x.to(torch.bfloat16)
    a, b = fit_probe(xb, y), fit_probe(xb.float(), y)
    assert np.array_equal(a[-1].coef_, b[-1].coef_) and np.array_equal(a[-1].intercept_, b[-1].intercept_)


def test_controlled_readout_bf16_identical_to_fp32():
    tb, ty = _states(seed=1); ta = tb + 0.1; tp = torch.zeros_like(tb) + 0.01 * torch.randn(tb.shape)
    hb, hy = _states(seed=2); ha = hb + 0.1; hp = torch.zeros_like(hb) + 0.01 * torch.randn(hb.shape)
    args = lambda f: (f(tb), f(ta), ty, f(hb), f(ha), hy, f(tp), f(hp), ['d1'], ['d2'])
    bf = controlled_readout(*args(lambda t: t.to(torch.bfloat16)))
    fp = controlled_readout(*args(lambda t: t.to(torch.bfloat16).float()))
    assert bf == fp
    assert all(0 <= bf[k] <= 1 for k in ['base_heldout', 'fixed_probe_on_adapted', 'refitted_probe_on_adapted', 'prefix_only'])


def test_masked_labels_are_excluded():
    x, y = _states(n=40)
    y2 = y.clone(); y2[::5] = -100
    p = fit_probe(x.to(torch.bfloat16), y2)
    assert p[-1].coef_.shape == (1, x.shape[1])


def test_zero_energy_states_are_finite_and_scoreable():
    x = torch.zeros(20, 4, dtype=torch.bfloat16); y = torch.arange(20) % 2
    p = fit_probe(x, y)
    assert np.isfinite(p[-1].coef_).all() and 0 <= p.score(probe_array(x), y.numpy()) <= 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_states_raise_valueerror_for_blocked_path(bad):
    x, y = _states(); x[3, 1] = bad
    with pytest.raises(ValueError, match='non-finite'):
        fit_probe(x.to(torch.bfloat16), y)


def test_empty_or_single_group_still_raises_valueerror():
    with pytest.raises(ValueError, match='both groups'):
        fit_probe(torch.zeros(0, 3, dtype=torch.bfloat16), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError, match='both groups'):
        fit_probe(torch.zeros(4, 3, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.long))


def test_dialogue_overlap_guard_unchanged():
    x, y = _states()
    with pytest.raises(ValueError, match='overlap'):
        controlled_readout(x, x, y, x, x, y, x, x, ['d1'], ['d1'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not visible')
def test_cuda_bf16_states():
    x, y = _states()
    p = fit_probe(x.to('cuda', torch.bfloat16), y.cuda())
    assert np.array_equal(p[-1].coef_, fit_probe(x.to(torch.bfloat16).float(), y)[-1].coef_)
