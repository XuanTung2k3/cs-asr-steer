"""P2-DIR direction providers (spec sections 3-5, 10): D0 regression to core_p1, D1 cross-fit
construction and guards, D2 reference-free readout correctness / isolation / firewall."""
from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, PART, VOCAB, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
from csasr.inference_cf import directions as D  # noqa: E402
from csasr.inference_cf import readout as R  # noqa: E402
from csasr.inference_cf import unique as U  # noqa: E402
from csasr.inference_cf.core_p1 import direction as core_direction  # noqa: E402
from csasr.lss.sites import DecoderPostCrossAttnRecorder, assert_no_site_hooks  # noqa: E402

FORBIDDEN = ("target_ids", "competitor", "reference", "transcript", "alignment", "oracle", "evaluator",
             "SiteGradientProbe", "inference_cf_p2rj", "margin_ref", "Y_ref", "future")


# ---- D0 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_d0_interface_equals_core_p1_bitwise(seed):
    g = torch.Generator().manual_seed(seed)
    hb, he = torch.randn(1280, generator=g), torch.randn(1280, generator=g)
    ref = core_direction(he, hb)
    r = D.OldDirection()(D.DirectionContext(h_b=hb, h_e=he))
    assert r.status == "ok" and r.reason is None and r.direction.dtype == torch.float32
    assert torch.equal(r.direction, ref["d"])
    assert r.provenance["delta_norm"] == ref["norm"]
    # epsilon normalization retained (not renormalized to exact unit norm)
    delta = he.double() - hb.double()
    assert torch.equal(r.direction, (delta / (delta.norm() + 1e-6)).float())


def test_d0_fallbacks_identical_and_zero_edit():
    hb = torch.randn(16)
    for he, status in ((hb + 1e-6, "direction_fail:tiny"), (hb.clone(), "direction_fail:tiny"),
                       (torch.full((16,), float("nan")), "direction_fail:nonfinite"),
                       (torch.full((16,), float("inf")), "direction_fail:nonfinite")):
        r = D.OldDirection()(D.DirectionContext(h_b=hb, h_e=he))
        assert r.direction is None and r.status == "invalid" and r.reason == core_direction(he, hb)["status"] == status


# ---- D1 ---------------------------------------------------------------------------------------------

def _synthetic(seed=0, dim=96, n=80, unique_scale=6.0):
    """A: shared subspace + one A-only axis with large energy; B: shared subspace + B-only axes."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    shared, a_only, b_only = Q[:, :31], Q[:, 31], Q[:, 32:64]
    HA = rng.standard_normal((n, 31)) @ shared.T + unique_scale * rng.standard_normal((n, 1)) @ a_only[None] \
        + 2.0 * a_only[None] + 0.01 * rng.standard_normal((n, dim))
    HB = rng.standard_normal((n, 31)) @ shared.T + rng.standard_normal((n, 32)) @ b_only.T + 0.01 * rng.standard_normal((n, dim))
    return HA, HB, a_only


def test_d1_recovers_known_unique_axis_with_sign_and_unit():
    HA, HB, axis = _synthetic()
    r = U.fit_fold(HA, HB)
    assert r["status"] == "ok", r["reason"]
    v = r["vector"].astype(np.float64)
    assert abs(float(v @ axis)) > 0.999
    assert float(v @ (HA.mean(0) - HB.mean(0))) > 0          # sign toward mu_A - mu_B
    assert r["vector"].dtype == np.float32 and abs(np.linalg.norm(v) - 1) <= U.UNIT_TOL
    assert max(r["ortho_errors"].values()) <= U.ORTHO_TOL
    i = r["selected_index"]
    assert np.allclose(r["scores"], r["E_A"] * (1 - r["sigma"] ** 2))
    assert i == int(np.argmax(r["scores"]))
    # E_A recomputed by hand
    CA = HA.T @ HA / HA.shape[0]
    assert r["E_A"][i] == pytest.approx(float(r["a"][:, i] @ CA @ r["a"][:, i]), rel=1e-10)


def test_d1_sign_flip_follows_mean_contrast():
    HA, HB, axis = _synthetic(seed=3)
    r1 = U.fit_fold(HA, HB)
    r2 = U.fit_fold(-HA, -HB)          # every state flipped -> contrast flips -> vector flips
    assert r1["status"] == r2["status"] == "ok"
    assert np.allclose(r1["vector"], -r2["vector"], atol=1e-6)


def test_d1_uses_exactly_top32_and_boundary_guard():
    HA, HB, _ = _synthetic()
    r = U.fit_fold(HA, HB)
    assert r["U_A"].shape[1] == r["U_B"].shape[1] == 32 == U.RANK
    assert r["eig_A"][31] > r["eig_A"][32]
    # tie at the 32/33 boundary -> invalid (no rank change)
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    HA2 = np.sqrt(64) * Q.T                       # isotropic: all eigenvalues equal
    rr = U.fit_fold(HA2, HB[:, :64])
    assert rr["status"] == "invalid" and rr["reason"].startswith("eigen_gap")


def test_d1_count_and_rank_guards_no_fallback():
    HA, HB, _ = _synthetic()
    assert U.fit_fold(HA[:31], HB)["reason"] == "insufficient_count"
    low = np.repeat(HA[:10], 8, axis=0)            # numerical rank 10 < 32
    r = U.fit_fold(low, HB)
    assert r["status"] == "invalid" and r["reason"] == "numerical_rank" and r["vector"] is None
    bad = HA.copy(); bad[0, 0] = np.nan
    assert U.fit_fold(bad, HB)["reason"] == "nonfinite_state"


def test_d1_degenerate_score_and_sign_guards():
    HA, HB, _ = _synthetic()
    # identical groups -> all sigma = 1 -> every score ~0 -> invalid (no arbitrary orientation)
    r = U.fit_fold(HA, HA.copy())
    assert r["status"] == "invalid" and r["reason"] in ("score_floor", "score_tie", "sigma_degenerate")
    # zero mean contrast -> sign indeterminate
    HA0 = np.concatenate([HA, -HA]); HB0 = np.concatenate([HB, -HB])
    r0 = U.fit_fold(HA0, HB0)
    assert r0["status"] == "invalid" and r0["reason"] in ("mean_contrast_zero", "sign_indeterminate")


def test_d1_deterministic_serialization_and_hash(tmp_path):
    HA, HB, _ = _synthetic(seed=5)
    a, b = U.fit_fold(HA, HB), U.fit_fold(HA.copy(), HB.copy())
    assert a["vector_sha256"] == b["vector_sha256"] and np.array_equal(a["vector"], b["vector"])
    p = tmp_path / "v.npy"
    np.save(p, a["vector"], allow_pickle=False)
    back = np.load(p, allow_pickle=False)
    assert U.array_hash(back) == a["vector_sha256"]
    s = U.summary(a)
    assert s["selected_sigma"] == pytest.approx(float(a["sigma"][a["selected_index"]]))
    assert "vector" not in s and set(U.arrays(a)) >= {"U_A", "U_B", "a", "b", "vector"}


def test_d1_fold_membership_excludes_all_of_dialogue():
    pos = [{"utterance_id": f"u{d}_{j}", "dialogue_id": f"D{d}", "t": 1 + j % 3, "stratum": s}
           for d in range(4) for j, s in enumerate(["EN-correct", "ZH-correct", "EN-confusion"] * 3)]
    for k in U.fold_dialogues(pos):
        m = U.fold_members(pos, k)
        ex = {tuple(x) for x in m["excluded"]}
        assert ex == {(p["utterance_id"], p["t"]) for p in pos if p["dialogue_id"] == k}
        assert not ({tuple(x) for x in m["A"]} | {tuple(x) for x in m["B"]}) & ex
        assert all(u.startswith(f"u{k[1:]}_") for u in m["excluded_utterances"])
        # no confusion states in fitting groups
        conf = {(p["utterance_id"], p["t"]) for p in pos if p["stratum"] == "EN-confusion"}
        assert not ({tuple(x) for x in m["A"]} | {tuple(x) for x in m["B"]}) & conf
    with pytest.raises(ValueError):
        U.fold_members([{**pos[0], "target_ids": [1]}], "D0")


def test_d1_provider_returns_sealed_copy_and_rejects_hash_mismatch():
    HA, HB, _ = _synthetic()
    r = U.fit_fold(HA, HB)
    fold = {"status": "ok", "vector_sha256": r["vector_sha256"], "dialogue": "D7"}
    p = D.UniqueDirection(r["vector"], fold)
    out = p(D.DirectionContext())
    assert out.status == "ok" and np.array_equal(out.direction.numpy(), r["vector"])
    out.direction.mul_(0)                                   # caller mutation cannot alter the seal
    assert np.array_equal(p(D.DirectionContext()).direction.numpy(), r["vector"])
    with pytest.raises(ValueError):
        D.UniqueDirection(r["vector"] * 2, fold)
    bad = D.UniqueDirection(None, {"status": "invalid", "reason": "numerical_rank", "dialogue": "D7"})
    o = bad(D.DirectionContext())
    assert o.direction is None and o.status == "invalid" and o.reason == "fold_invalid:numerical_rank"


def test_d1_has_no_global_refit_option():
    names = {n for n, _ in inspect.getmembers(U, inspect.isfunction)}
    assert not any("global" in n for n in names)
    assert "global" not in inspect.signature(U.fit_fold).parameters


# ---- D2 ---------------------------------------------------------------------------------------------

def _prefix_branch(bundle, steps=3):
    enc = encoded()
    br = cached.Branch(bundle, enc, CB, "B")
    new = list(CB)
    toks = [10, 47, 12, 50]
    for t in range(steps):
        br.step(new, capture_layer=16)
        new = [toks[t]]
    return br, enc, new, steps


def _readout(bundle, br, enc, new, t, part=PART):
    return R.readout_direction(bundle, layer=16, cache=br.cache, encoded=enc, new_tokens=new, start=br.length,
                               step=t, suppress=[0, 3], begin=[9], partition=part)


def test_d2_value_identity_cache_isolation_and_no_param_grads():
    bundle = tiny_bundle()
    bundle.model.requires_grad_(True)               # flags must be restored, grads never accumulate
    br, enc, new, t = _prefix_branch(bundle)
    fp = R.cache_fingerprint(br.cache)
    r = _readout(bundle, br, enc, new, t)
    assert all(p.requires_grad for p in bundle.model.parameters())
    assert all(p.grad is None for p in bundle.model.parameters())
    assert R.cache_fingerprint(br.cache) == fp and br.length == fp[1]
    assert_no_site_hooks(bundle)
    logits, _, site = br.step(new, capture_layer=16)
    assert torch.equal(r["logits"], logits) and torch.equal(r["site"], site)
    assert r["status"] == "ok" and r["counters"] == {"autograd_calls": 1, "scratch_forwards": 1}
    d = r["direction"].double()
    assert abs(float(d.norm()) - 1) <= R.UNIT_TOL
    assert abs(float(torch.dot(d, site.double() / site.double().norm()))) <= R.RADIAL_TOL


def test_d2_objective_exact_suppression_and_epsilon():
    torch.manual_seed(2)
    z = torch.randn(VOCAB) * 2
    v = R.objective(z, 5, [0, 3, 44], [45], PART)
    zp = z.clone(); zp[[0, 3, 44]] = -float("inf")
    lp = torch.log_softmax(zp, -1)
    pe = float(lp[PART["embedded_ids"]].exp().sum())
    pm = float(lp[PART["matrix_ids"]].exp().sum())
    assert float(v["J"]) == pytest.approx(math.log(pe + 1e-12) - math.log(pm + 1e-12), abs=1e-5)
    v0 = R.objective(z, 0, [], [45], PART)                  # begin-suppression only at step 0
    zp0 = z.clone(); zp0[45] = -float("inf")
    pe0 = float(torch.log_softmax(zp0, -1)[PART["embedded_ids"]].exp().sum())
    assert float(v0["log_PE"]) == pytest.approx(math.log(pe0), abs=1e-5)
    # epsilon floor: all embedded mass suppressed -> J finite, = log(1e-12) - log(P_M + 1e-12)
    vz = R.objective(z, 5, PART["embedded_ids"], [], PART)
    assert math.isfinite(float(vz["J"])) and float(vz["J"]) == pytest.approx(math.log(1e-12) - math.log(
        float(torch.log_softmax(z.masked_fill(torch.isin(torch.arange(VOCAB), torch.tensor(PART["embedded_ids"])), -float("inf")), -1)[PART["matrix_ids"]].exp().sum()) + 1e-12), abs=1e-4)


def test_d2_gradient_matches_finite_difference():
    bundle = tiny_bundle(seed=4)
    br, enc, new, t = _prefix_branch(bundle)
    r = _readout(bundle, br, enc, new, t)
    g = r["gradient"].double()
    # finite difference of J along random directions via the exact-site hook (float32 tiny model)
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook
    qpos = br.length + len(new) - 1

    def J_at(vec):
        import copy
        cache = copy.deepcopy(br.cache)

        def act(q=None, u_source=None, r=None, abs_pos=None):
            rr, pos = r, abs_pos
            gg = torch.zeros((1, rr.shape[1]), dtype=rr.dtype)
            dd = torch.zeros((1, rr.shape[1], rr.shape[-1]), dtype=rr.dtype)
            hit = (pos == qpos).nonzero().flatten()
            gg[0, hit] = 1.0
            dd[0, hit] = vec.to(rr.dtype)
            return gg, dd
        hook = DecoderPostCrossAttnInterventionHook(bundle, 16, None, alpha=1.0, num_forced_prefix=4, action_fn=act,
                                                    norm_preserve=False)
        with torch.no_grad(), hook:
            out = bundle.model(encoder_outputs=enc, decoder_input_ids=torch.tensor([new]), past_key_values=cache,
                               use_cache=True, cache_position=torch.arange(br.length, br.length + len(new)))
        return float(R.objective(out.logits[0, -1], t, [0, 3], [9], PART)["J"])
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        v = torch.randn(16, generator=gen, dtype=torch.float64)
        v /= v.norm()
        eps = 1e-3
        fd = (J_at((eps * v).float()) - J_at((-eps * v).float())) / (2 * eps)
        assert fd == pytest.approx(float(g @ v), rel=2e-2, abs=1e-4)


def test_d2_tangent_projection_and_noop_guards():
    h = torch.tensor([3.0, 4.0, 0.0])
    g = torch.tensor([1.0, 1.0, 1.0])
    r = R.tangent_unit(g, h)
    d = r["direction"].double()
    assert r["status"] == "ok" and abs(float(d @ h.double())) < 1e-6 and abs(float(d.norm()) - 1) < 2e-6
    gp = g.double() - h.double() / 5 * float(h.double() @ g.double() / 5)
    assert torch.allclose(d, gp / gp.norm(), atol=1e-7)
    assert R.tangent_unit(h * 2, h)["reason"] == "tiny_tangent"            # purely radial gradient
    assert R.tangent_unit(g, torch.zeros(3))["reason"] == "tiny_state"
    assert R.tangent_unit(torch.tensor([float("nan"), 0, 0]), h)["reason"] == "nonfinite_state_or_gradient"
    assert R.tangent_unit(g * 1e-12, h)["reason"] == "tiny_tangent"
    for bad in ("tiny_tangent", "tiny_state"):
        assert R.tangent_unit(h * 2 if bad == "tiny_tangent" else g, h if bad == "tiny_tangent" else torch.zeros(3))["direction"] is None


def test_d2_current_prefix_only_independent_of_future_tokens():
    bundle = tiny_bundle(seed=6)
    br1, enc, new, t = _prefix_branch(bundle, steps=3)
    r1 = _readout(bundle, br1, enc, new, t)
    br2, enc2, new2, _ = _prefix_branch(bundle, steps=3)
    r2 = _readout(bundle, br2, enc2, new2, t)
    br2.step(new2); br2.step([55])                         # later tokens differ; earlier result fixed
    assert torch.equal(r1["direction"], r2["direction"]) and r1["J"] == r2["J"]
    assert "future" not in inspect.signature(R.readout_direction).parameters


def test_d2_exception_cleanup_removes_hooks_and_restores_flags(monkeypatch):
    bundle = tiny_bundle()
    flags = [p.requires_grad for p in bundle.model.parameters()]
    br, enc, new, t = _prefix_branch(bundle)
    fp = R.cache_fingerprint(br.cache)

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(R, "objective", boom)
    with pytest.raises(RuntimeError, match="boom"):
        _readout(bundle, br, enc, new, t)
    assert_no_site_hooks(bundle)
    assert [p.requires_grad for p in bundle.model.parameters()] == flags
    assert all(p.grad is None for p in bundle.model.parameters())
    assert R.cache_fingerprint(br.cache) == fp


def test_d2_under_outer_inference_mode_and_site_is_dg02():
    bundle = tiny_bundle()
    br, enc, new, t = _prefix_branch(bundle)
    with torch.inference_mode():
        r = _readout(bundle, br, enc, new, t)
    assert r["status"] == "ok"
    rec = DecoderPostCrossAttnRecorder(bundle, [16], keep_last_only=True)
    with torch.no_grad(), rec:
        import copy
        bundle.model(encoder_outputs=enc, decoder_input_ids=torch.tensor([new]), past_key_values=copy.deepcopy(br.cache),
                     use_cache=True, cache_position=torch.arange(br.length, br.length + len(new)))
    assert torch.equal(rec.states[16][0, -1], r["site"])           # r = q + u_source (pre-FFN)


def test_d2_provider_record_and_no_fallthrough():
    bundle = tiny_bundle()
    br, enc, new, t = _prefix_branch(bundle)
    p = D.ReadoutDirection(bundle, layer=16, suppress=[0, 3], begin=[9], partition=PART)
    r = p(D.DirectionContext(b_cache_pre_step=br.cache, encoded=enc, new_tokens=new, start=br.length, step=t))
    rec = r.record()
    assert rec["id"] == "D2" and rec["status"] == "ok" and rec["sha256"].startswith("sha256:")
    assert rec["counters"]["autograd_calls"] == 1


# ---- firewall ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("rel", ["src/csasr/inference_cf/directions.py", "src/csasr/inference_cf/readout.py",
                                 "src/csasr/inference_cf/unique.py"])
def test_construction_modules_reference_free(rel):
    tree = ast.parse(Path(rel).read_text())
    idents = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            idents.add(node.id)
        elif isinstance(node, ast.Attribute):
            idents.add(node.attr)
        elif isinstance(node, ast.arg):
            idents.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            idents |= {a.name for a in node.names} | ({node.module} if getattr(node, "module", None) else set())
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            idents.add(node.name)
    idents.discard("__future__")
    bad = {i for i in idents for f in FORBIDDEN if f.lower() in str(i).lower()}
    assert not bad, bad


def test_direction_context_has_no_reference_fields_and_exactly_three_providers():
    fields = set(D.DirectionContext.__dataclass_fields__)
    assert fields == {"h_b", "h_e", "b_cache_pre_step", "encoded", "new_tokens", "start", "step"}
    assert D.IDS == ("D0", "D1", "D2")
    providers = [c for _, c in inspect.getmembers(D, inspect.isclass) if hasattr(c, "id") and c.__module__ == D.__name__]
    assert sorted(c.id for c in providers) == ["D0", "D1", "D2"]
