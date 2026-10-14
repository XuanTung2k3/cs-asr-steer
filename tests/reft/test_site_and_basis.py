"""CPU tests for the P1 site, gradient capture semantics, row construction, basis weighting and controls.
A tiny randomly initialised WhisperForConditionalGeneration (real transformers classes, no weight download)."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.build_basis import orth_random, principal_cosines, stratum_weights  # noqa: E402
from reft.gradient_capture import build_rows  # noqa: E402
from reft.site import Site, n_hooks  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    c = WhisperConfig(vocab_size=64, num_mel_bins=8, encoder_layers=1, decoder_layers=3, encoder_attention_heads=2,
                      decoder_attention_heads=2, d_model=16, encoder_ffn_dim=32, decoder_ffn_dim=32,
                      max_source_positions=12, max_target_positions=32, pad_token_id=1, decoder_start_token_id=2,
                      eos_token_id=1, attn_implementation="eager")
    m = WhisperForConditionalGeneration(c).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    enc = torch.randn(1, 12, 16)
    X = torch.tensor([[2, 5, 6, 7, 9, 11, 13, 4]])
    return m, enc, X


def fwd(m, enc, X, layer=1, delta=None, record=False):
    s = Site(m, layer, delta=delta, record=record)
    with s:
        lg = m(encoder_outputs=(enc.expand(X.shape[0], -1, -1),), decoder_input_ids=X, use_cache=False).logits
    return lg, s


def test_zero_delta_identity_and_site_is_ffn_input(tiny):
    m, enc, X = tiny
    plain = m(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
    lg, s = fwd(m, enc, X, delta=torch.zeros(1, X.shape[1], 16), record=True)
    assert torch.equal(plain, lg)
    assert torch.equal(s.ffn_in, s.q + s.s) and s.calls == 1
    assert n_hooks(m, 1) == 0


def test_site_is_not_block_output(tiny):
    m, enc, X = tiny
    out = {}
    h = m.model.decoder.layers[1].register_forward_hook(lambda mod, a, o: out.__setitem__("block", o[0]))
    try:
        _, s = fwd(m, enc, X, record=True)
    finally:
        h.remove()
    assert not torch.allclose(out["block"], s.ffn_in)


def test_delta_is_causal_and_gradient_matches_finite_difference(tiny):
    import copy
    m, enc, X = tiny
    t, a, b = 4, 17, 23
    base, _ = fwd(m, enc, X)
    d = torch.zeros(1, X.shape[1], 16)
    d[0, t] = torch.randn(16)
    lg, _ = fwd(m, enc, X, delta=d)
    assert torch.equal(lg[0, :t], base[0, :t]) and not torch.allclose(lg[0, t], base[0, t])
    md, encd = copy.deepcopy(m).double(), enc.double()
    delta = torch.zeros(1, X.shape[1], 16, dtype=torch.float64, requires_grad=True)
    lg, s = fwd(md, encd, X, delta=delta, record=True)
    mval = lg[0, t, a] - lg[0, t, b]
    q = torch.autograd.grad(mval, delta, retain_graph=True)[0][0, t]
    q2 = torch.autograd.grad(mval, delta)[0][0, t]
    assert torch.equal(q, q2) and q.shape == (16,) and torch.isfinite(q).all() and q.norm() > 0
    assert all(p.grad is None for p in md.parameters())
    eta = 1e-3 * float(s.ffn_in[0, t].norm())

    def marg(u):
        dd = torch.zeros(1, X.shape[1], 16, dtype=torch.float64)
        dd[0, t] = u
        lg2, _ = fwd(md, encd, X, delta=dd)
        return float(lg2[0, t, a] - lg2[0, t, b])
    qh = q / q.norm()
    fd = (marg(eta * qh) - marg(-eta * qh)) / (2 * eta)
    assert abs(fd - float(q.norm())) / float(q.norm()) < 1e-4
    r = torch.randn(16, dtype=torch.float64)
    r = r / r.norm()
    assert abs((marg(eta * r) - marg(-eta * r)) / (2 * eta) - float(q @ r)) < 1e-4 * float(q.norm())
    assert marg(eta * qh) > marg(torch.zeros(16, dtype=torch.float64)) > marg(-eta * qh)


def test_batched_delta_rows_are_independent(tiny):
    m, enc, X = tiny
    d = torch.zeros(3, X.shape[1], 16)
    d[1, 3] = torch.randn(16)
    d[2, 5] = torch.randn(16)
    lg, _ = fwd(m, enc, X.expand(3, -1), delta=d)
    base, _ = fwd(m, enc, X)
    assert torch.allclose(lg[0], base[0], atol=1e-6)
    single, _ = fwd(m, enc, X, delta=d[2:3])
    assert torch.allclose(lg[2], single[0], atol=1e-5)


class Tok:
    def __init__(self, pieces):
        self.v = {i + 2: p.encode() for i, p in enumerate(pieces)}
        self.i = {p: k for k, p in self.v.items()}

    def __call__(self, i):
        return self.v.get(i, b"")

    def enc(self, *ps):
        return [self.i[p.encode()] for p in ps]


def test_build_rows_sources_and_prefix_rules():
    EOS = 1
    tk = Tok(["我", "想", " go", " went", "回", "家", " home", "去", "，", "嗯"])
    ids = tk.enc("我", "想", " go", " home") + [EOS]
    tgt = {"target_ids": ids, "target_text": "我想 go home"}
    base = tk.enc("我", "想", " went", " home") + [EOS]
    argmax = tk.enc("我", "去", " went", " home") + [EOS]   # TF: pos1 wrong (B), pos2 wrong (= A position)
    runner = tk.enc("回", "想", " go", "家") + [EOS]
    fd, rows = build_rows("u", tgt, base, argmax, runner, tk, EOS, {"嗯"}, 4, 4)
    A = [r for r in rows if r["source"] == "A"]
    B = [r for r in rows if r["source"] == "B"]
    P = [r for r in rows if r["source"] == "P"]
    assert len(A) == 1 and A[0]["t"] == 2 and A[0]["correct"] == ids[2] and A[0]["competitor"] == base[2]
    assert A[0]["category"] == "en_lexical_substitution"
    assert [r["t"] for r in B] == [1] and B[0]["competitor"] == argmax[1] and B[0]["category"] == "zh_lexical_substitution"
    assert {r["t"] for r in P} == {0} and P[0]["competitor"] == runner[0]  # pos 1 is a TF error, never a control
    assert all(r["t"] < fd["k"] for r in P)


def test_stratum_caps():
    strata = ["CS"] * 10 + ["EN_ONLY"] * 8 + ["ZH_ONLY"] * 1
    w0 = torch.ones(len(strata), dtype=torch.float64)
    w, sh = stratum_weights(strata, w0, {"EN_ONLY": 0.2, "ZH_ONLY": 0.2})
    assert abs(sh["EN_ONLY"] - 0.2) < 1e-9 and sh["ZH_ONLY"] <= 0.2 + 1e-9
    assert torch.equal(w[:10], w0[:10]) and torch.equal(w[-1:], w0[-1:])


def test_random_controls_orthonormal_deterministic_and_svd_recovers_planted_subspace():
    R = orth_random(64, 4, 42)
    assert torch.allclose(R.T @ R, torch.eye(4, dtype=torch.float64), atol=1e-12)
    assert torch.equal(R, orth_random(64, 4, 42)) and not torch.equal(R, orth_random(64, 4, 43))
    g = torch.Generator().manual_seed(0)
    basis = orth_random(64, 2, 7)
    Q = torch.randn(300, 2, generator=g, dtype=torch.float64) @ basis.T * 5 + 0.1 * torch.randn(300, 64, generator=g, dtype=torch.float64)
    _, _, Vh = torch.linalg.svd(Q / Q.norm(dim=1, keepdim=True), full_matrices=False)
    assert min(principal_cosines(Vh[:2].T, basis)) > 0.99
