"""TTLS-R1R focused tests (CPU; tiny real-class Whisper in BF16 like the real run): ratio-first zero-edit identity of the
kernel, the hook, teacher-forced logits / FFN input and free decoding under the actual nonempty masks; zero initial KL;
nonzero gradient at z = 0 (and float64 finite differences for nonzero z); nonzero-edit norm preservation, FFN
consumption and outside-mask invariance; fresh episodes; historical kernel untouched; phase A / B row plumbing;
reference firewall; frozen lexical split."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf import ttls as L  # noqa: E402
from csasr.inference_cf import ttls_r1r as R  # noqa: E402
from csasr.inference_cf.episodic_tta import LNGuard, decoder_ln_names, forced_decode, teacher_logits  # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
from test_ttls_r1 import CB, CE, LAYER, _bundle, _encode  # noqa: E402


@pytest.fixture(scope="module")
def tb():
    return _bundle()


def _zero(b):
    return torch.zeros(int(b.d_model))


# ---- kernel -------------------------------------------------------------------------------------------------------

def test_ratio_first_kernel_zero_identity_and_historical_defect():
    from csasr.models.hooks import apply_steering
    torch.manual_seed(0)
    hist_changed = 0
    for scale in (0.1, 1.0, 10.0, 100.0):
        h = (torch.randn(1, 64, 1280) * scale).bfloat16()
        z = torch.zeros(1280)
        assert torch.equal(R.apply_steering_ratio_first(h, z, 1.0, 1.0, None), h)
        hist_changed += int((apply_steering(h, z, 1.0, 1.0, None) != h).any())
    assert hist_changed > 0                                # the historical BF16 divide-then-multiply is not identity


def test_ratio_first_gradient_at_zero_is_projected_and_nonzero():
    torch.manual_seed(1)
    h = torch.randn(1, 5, 32, dtype=torch.float64)
    z = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    w = torch.randn(1, 5, 32, dtype=torch.float64)
    (R.apply_steering_ratio_first(h, z, 1.0, 1.0, None) * w).sum().backward()
    # analytic d/dz sum_t w_t . ((h_t + z) ||h_t|| / ||h_t + z||) at z = 0 = sum_t (I - u_t u_t^T) w_t, u_t = h_t / ||h_t||
    u = h[0] / h[0].norm(dim=-1, keepdim=True)
    expect = (w[0] - (w[0] * u).sum(-1, keepdim=True) * u).sum(0)
    assert float(z.grad.norm()) > 0 and torch.allclose(z.grad, expect, atol=1e-10)


def test_ratio_first_nonzero_edit_norm_and_masking():
    torch.manual_seed(2)
    h = torch.randn(1, 6, 1280).bfloat16()
    d = torch.randn(1280)
    g = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.bfloat16)
    s = R.apply_steering_ratio_first(h, d, 1.0, 1.0, g)
    on, off = g[0] > 0, g[0] == 0
    assert torch.equal(s[0, off], h[0, off])
    assert not torch.equal(s[0, on], h[0, on])
    rel = (s[0, on].float().norm(dim=-1) / h[0, on].float().norm(dim=-1) - 1).abs()
    assert float(rel.max()) <= 0.01


# ---- hook on the tiny model -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [L.ALL, {1}, {2, 3}])
def test_zero_vector_teacher_forced_and_ffn_identity(tb, steps):
    _, enc = _encode(tb)
    y = [20, 21, 22, 23, 24]
    p = R.site_ffn_probe(tb, enc, y, z=_zero(tb), steps=steps, layer=LAYER, prompt=CB)
    assert p["logits_bitwise_equal"] and p["ffn_input_bitwise_equal"] and p["max_abs_logit_diff"] == 0.0
    assert p["edited_queries"] > 0                         # nonempty actual mask
    with torch.no_grad(), R.ttls_hook_r1r(tb, _zero(tb), steps, layer=LAYER, mode="train") as h:
        b = teacher_logits(tb.model, enc, CB, y, None)
    assert h.steered_calls > 0                             # the edit path really runs (no bypass)
    with torch.no_grad():
        assert torch.equal(b, teacher_logits(tb.model, enc, CB, y, None))


def test_historical_kernel_zero_vector_is_not_identity_on_tiny_model(tb):
    _, enc = _encode(tb)
    p = R.site_ffn_probe(tb, enc, [20, 21, 22, 23, 24, 25, 26], z=_zero(tb), steps=L.ALL, layer=LAYER, prompt=CB, kernel="historical")
    assert not p["ffn_input_bitwise_equal"]


def test_zero_vector_free_decode_identity(tb):
    enc, _ = _encode(tb)
    d0 = forced_decode(tb, enc, CB, max_new_tokens=12, capture_layer=LAYER)
    for steps in (L.ALL, {1}, {3}):
        dz = L.greedy_decode(tb, enc, CB, hook_factory=lambda st=steps: R.ttls_hook_r1r(tb, _zero(tb), st, layer=LAYER, mode="steer"),
                             max_new_tokens=12, capture_layer=LAYER)
        assert dz["tokens"] == d0["tokens"] and dz["terminated"] == d0["terminated"]


def test_zero_initial_kl_is_exactly_zero(tb):
    _, enc = _encode(tb)
    y = [20, 21, 22, 23]
    base = L.displacement(tb, enc, y, suppress=[], begin=[], layer=LAYER, prompt=CB)
    for steps in (L.ALL, {2}):
        assert R.zero_kl(tb, enc, y, base["logq"], [0, 1, 2, 3], steps, [], [], layer=LAYER, prompt=CB) == 0.0


def test_nonzero_edit_consumed_norm_preserved_outside_untouched(tb):
    _, enc = _encode(tb)
    torch.manual_seed(3)
    z = torch.randn(16)
    z = z / z.norm() * L.E_STAR
    p = R.site_ffn_probe(tb, enc, [20, 21, 22, 23, 24], z=z, steps={2}, layer=LAYER, prompt=CB)
    assert p["edited_chord_max"] > 0 and p["outside_chord_max"] == 0.0
    assert p["edited_norm_rel_err_max"] <= 0.02
    pa = R.site_ffn_probe(tb, enc, [20, 21, 22, 23, 24], z=z, steps=L.ALL, layer=LAYER, prompt=CB)
    assert pa["edited_queries"] == 5 and pa["outside_chord_max"] == 0.0 and pa["edited_chord_max"] > 0


def test_gradient_matches_finite_differences_float64_ratio_first():
    b64 = _bundle(torch.float64)
    _, enc = _encode(b64)
    y = [20, 21, 22, 23]
    z = torch.nn.Parameter(torch.randn(16, dtype=torch.float64) * 0.3)

    def f(zz):
        with R.ttls_hook_r1r(b64, zz, {1, 2}, layer=LAYER, mode="train"):
            lg = teacher_logits(b64.model, enc, CB, y, None)
        return -torch.log_softmax(lg[2].double(), -1)[50]
    with torch.enable_grad():
        (g,) = torch.autograd.grad(f(z), z)
    u = torch.randn(16, dtype=torch.float64)
    u /= u.norm()
    eps = 1e-3
    with torch.no_grad():
        fd = (f(z + eps * u) - f(z - eps * u)) / (2 * eps)
    assert float(g.norm()) > 0
    assert abs(float(fd) - float(g @ u)) <= 1e-3 * max(abs(float(fd)), 1e-3)


# ---- episodes -------------------------------------------------------------------------------------------------------

def _ce_loss(y, mask):
    def fn(ep):
        _, loss = L.ce_terms(ep, y, mask, [], [], 2, prompt=CB)
        return {"loss": loss, "parts": {"CE": float(loss.detach())}}
    return fn


def test_episode_fresh_nonzero_grad_at_zero_and_reset(tb):
    _, enc = _encode(tb)
    y, mask = [20, 21, 22, 23], [True] * 4
    before = [p.detach().clone() for p in tb.model.parameters()]
    logs = []
    for _ in range(2):
        ep = R.EpisodeR1R(tb, enc, L.ALL, layer=LAYER)
        assert torch.count_nonzero(ep.z) == 0 and not ep.opt.state
        r = ep.run(_ce_loss(y, mask), {})
        logs.append(r["log"])
        assert r["log"]["grad_l2"][0] > 0 and math.isfinite(r["log"]["grad_l2"][0])
        assert float(ep.z.norm()) <= L.E_STAR + 1e-6
    assert logs[0]["losses"] == logs[1]["losses"]
    assert all(torch.equal(a, b) for a, b in zip(before, tb.model.parameters()))
    assert all(p.grad is None and not p.requires_grad for p in tb.model.parameters())


# ---- runner plumbing ------------------------------------------------------------------------------------------------

def test_phase_a_and_b_rows_tiny(tb, monkeypatch):
    import experiments.inference_cf_ttls_r1r as RR
    import csasr.inference_cf.core_r2 as core_r2
    monkeypatch.setattr(core_r2, "prefix_utf8_complete", lambda *a, **k: True)
    monkeypatch.setattr(RR, "MAX_NEW", 8)
    guard = LNGuard(tb.model, decoder_ln_names(tb.model))
    enc_inf, _ = _encode(tb)
    d0 = forced_decode(tb, enc_inf, CB, max_new_tokens=8, capture_layer=LAYER)
    y_B = d0["tokens"]
    y_A = list(y_B[:1]) + [int(y_B[1]) + 1 if int(y_B[1]) + 1 != 2 else 5] + list(y_B[2:])
    emb = set(range(3, 200)) - set(int(t) for t in y_B)
    real = L.select_candidates

    def forced_candidate(*a, **k):
        out = real(*a, **k)
        if out["t_star"] is None:
            out.update(t_star=1, c_star=50, accepted=[1], M=sorted(set(out["M"]) | {1}))
        return out
    monkeypatch.setattr(L, "select_candidates", forced_candidate)
    ctx = RR.Ctx(tb, guard, suppress=[], begin=[], eos=2, embedded=emb, cB=CB, cE=CE, layer=LAYER,
                 encode=lambda s: _encode(tb), encode_null=lambda: _encode(tb, seed=9), tau=-1e9)
    s = {"utterance_id": "u", "dialogue_id": "d", "group": "D", "a3_group": None, "y_B": y_B, "y_B_terminated": d0["terminated"],
         "y_B_text": d0["text"], "y_B_valid_mask": [True] * len(y_B), "y_A": y_A, "y_A_text": "", "y_A_valid_mask": [True] * len(y_A),
         "utf8_ok": [True] * len(y_B)}
    A = RR.integrity_row(ctx, dict(s), 0, None)
    assert A["pass"] and A["zero"]["ALL"]["decode_equals_B0"] and A["zero"]["AC"]["decode_equals_B0"]
    assert A["zero"]["ALL"]["kl0_stable"] == 0.0 and A["zero"]["ALL"]["probe"]["ffn_input_bitwise_equal"]
    row = RR.process_row(ctx, dict(s), 0, A)
    assert row["status"] == "ok" and row["post_row_clean_equals_B0"]
    for name, obj, P in RR.ARMS:
        a = row[name]
        assert a["status"] == "ok" and all(a["gates"].values()) and a["numerics"] == R.NUMERICS
        assert len(a["log"]["losses"]) == 3 and a["log"]["grad_l2"][0] > 0
        if P:
            assert a["log"]["parts"][0]["P"] == 0.0         # exact: the zero start is the clean decoder
    assert guard.verify() and all(p.grad is None and not p.requires_grad for p in tb.model.parameters())
    json.dumps(row)
    json.dumps(A)


# ---- preservation of history / firewall / config --------------------------------------------------------------------

def test_historical_sources_untouched():
    c = json.loads((ROOT / "configs/inference_cf/ttls_r1.json").read_text())
    for p, h in c["source_sha256"].items():                # every TTLS-R1 pinned source (incl. hooks.py, sites.py, ttls.py)
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
    import inspect
    from csasr.models.hooks import apply_steering
    assert "steered = steered / (new_norm + EPS) * orig_norm" in inspect.getsource(apply_steering)
    r = json.loads((ROOT / "configs/inference_cf/ttls_r1r.json").read_text())["inherits"]
    for k, kh in (("config", "config_sha256"), ("plan", "plan_sha256"), ("r1_output_seal", "r1_output_seal_sha256")):
        assert hashlib.sha256((ROOT / r[k]).read_bytes()).hexdigest() == r[kh], k


FORBIDDEN = ("csasr.evaluation", "inference_cf_p2_evaluate", "p0_r2", "load_references", "evaluate")


@pytest.mark.parametrize("rel", ["src/csasr/inference_cf/ttls_r1r.py", "experiments/inference_cf_ttls_r1r.py"])
def test_no_reference_or_evaluator_input(rel):
    tree = ast.parse((ROOT / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not any(f in m for m in mods for f in FORBIDDEN), (rel, mods)
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("load_references", "reference"), rel
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.lower()
            assert "reference" not in v or v == "references_used" or v.startswith("no reference") or "\n" in v, (rel, node.value)


def test_config_matches_modules():
    import experiments.inference_cf_ttls_r1r as RR
    c = json.loads((ROOT / "configs/inference_cf/ttls_r1r.json").read_text())
    assert c["numerics"]["name"] == R.NUMERICS
    assert {k: tuple(v) for k, v in c["rerun_arms"].items()} == {a[0]: ("TTLS", a[1], a[2]) for a in RR.ARMS}
    assert c["integrity_gates"]["zero_initial_KL_max"] == RR.KL0_MAX and c["integrity_gates"]["nonzero_edit_norm_rel_err_max"] == RR.NORM_REL_MAX


def test_lexical_split():
    from experiments.inference_cf_ttls_r1r_evaluate import lexical_kind
    e = lambda cat, poi, b: {"B0_category": cat, "poi": poi, "B0": b}
    assert lexical_kind(e("same_language_substitution", "people", "peoplethey")) == "word_boundary_repair"
    assert lexical_kind(e("same_language_substitution", "relationship", "somerelationship")) == "word_boundary_repair"
    assert lexical_kind(e("same_language_substitution", "cooking", "cookie")) == "genuine_same_language"
    assert lexical_kind(e("wrong_language_substitution", "cooking", "烧烤")) == "genuine_wrong_language"
    assert lexical_kind(e("deletion", "course", "")) == "deletion_or_boundary"
    assert lexical_kind(e("other", "ok", "o k")) == "word_boundary_repair"
