"""P2-RJ Jacobian diagnosis: exact frozen state, value-identical zero probe, gradient correctness,
margin/tangent/e* definitions, P2-R direction and random provenance, gradient never used as a
steering direction, serialization, manifest inputs, frozen decision rules."""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART, VOCAB, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2r as p2r  # noqa: E402
import experiments.inference_cf_p2rj as rj  # noqa: E402
import experiments.inference_cf_p2rj_analyze as an  # noqa: E402
import experiments.inference_cf_p2rj_prepare as prep  # noqa: E402
from csasr.inference_cf.core import digest, file_hash  # noqa: E402
from csasr.lss.sites import DecoderPostCrossAttnInterventionHook  # noqa: E402

CFG = json.loads(Path("configs/inference_cf/p2_rj_jacobian_diagnosis.json").read_text())
P2R_CFG = json.loads(Path("configs/inference_cf/p2_r_mechanism_diagnosis.json").read_text())
COND = {"cB": CB, "cE": CE, "language_token_ids": [4, 5]}
E_TINY = 0.03


def _baseline(bundle, monkeypatch):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    return cached.cached_decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=encoded(),
                                conditions=COND, partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                                layer=16, alpha=0.0, max_new_tokens=8, num_forced_prefix=4)


def _ctx(bundle):
    cfg = json.loads(json.dumps(CFG))
    cfg["conditions"] = COND
    cfg["energy"]["e_star"] = E_TINY
    return rj.Ctx(bundle, cfg, PART, 4)


def _positions(toks):
    # EN-like positions: target = some embedded ids, competitor = the baseline token
    return [{"t": t, "stratum": "EN-confusion", "target_ids": [44, 45, 46], "competitor": toks[t]} for t in (1, 2)]


# ---- frozen values -------------------------------------------------------------------------

def test_config_frozen_values_and_e_star_reuse():
    assert CFG["site"]["layer"] == 16 == P2R_CFG["site"]["layer"]
    assert CFG["energy"]["e_star"] == P2R_CFG["energy"]["target_edit_norm"]
    assert CFG["energy"]["e_star"] == pytest.approx(math.sqrt(1203.3762345332195 / 949))
    assert CFG["conditions"] == P2R_CFG["conditions"]
    a = CFG["analysis"]
    assert a["bootstrap"] == {**a["bootstrap"], "replicates": 10000, "seed": 240924}
    assert tuple(a["decision_family"]) == an.DECISION_FAMILY
    assert CFG["gradient"]["primary_norm"] == "tangent"
    assert CFG["p2r_reference"]["population_hash"] == json.loads(
        Path("results/inference_cf/p2r/population.json").read_text())["population_hash"]


def test_decision_rules_frozen():
    b = lambda l, h: {"ci": [l, h]}
    assert an.leverage_class(b(0.1, 0.49)) == "WEAK"
    assert an.leverage_class(b(0.51, 0.9)) == "SUBSTANTIAL"
    assert an.leverage_class(b(0.4, 0.6)) == "UNRESOLVED"
    assert an.leverage_class({"ci": None}) == "UNRESOLVED"
    assert an.alignment_class(b(-0.3, 0.099)) == "MISALIGNED"
    assert an.alignment_class(b(0.25, 0.6)) == "ALIGNED"
    assert an.alignment_class(b(0.05, 0.3)) == "PARTIAL"
    assert an.label("SUBSTANTIAL", "MISALIGNED") == "P2_RJ_DIRECTION_ISSUE"
    assert an.label("WEAK", "MISALIGNED") == "P2_RJ_DIRECTION_AND_SITE_ISSUE"
    assert an.label("WEAK", "PARTIAL") == an.label("WEAK", "ALIGNED") == "P2_RJ_SITE_OR_SENSITIVITY_ISSUE"
    for lev, ali in (("SUBSTANTIAL", "ALIGNED"), ("SUBSTANTIAL", "PARTIAL"), ("UNRESOLVED", "MISALIGNED")):
        assert an.label(lev, ali) == "P2_RJ_STILL_AMBIGUOUS"
    ok = an.decide(True, "WEAK", "MISALIGNED", "P2_RJ_DIRECTION_AND_SITE_ISSUE", True, True)
    assert ok["diagnosis"] == "P2_RJ_DIRECTION_AND_SITE_ISSUE" and not ok["blocking"]
    for args in ((False, True, True, "P2_RJ_DIRECTION_AND_SITE_ISSUE"), (True, False, True, "P2_RJ_DIRECTION_AND_SITE_ISSUE"),
                 (True, True, False, "P2_RJ_DIRECTION_AND_SITE_ISSUE"), (True, True, True, "P2_RJ_DIRECTION_ISSUE")):
        valid, fd, prec, fp = args
        assert an.decide(valid, "WEAK", "MISALIGNED", fp, fd, prec)["diagnosis"] == "P2_RJ_STILL_AMBIGUOUS"


# ---- definitions ---------------------------------------------------------------------------

def test_margin_definition_exact_and_equals_p2r_margin_at_baseline():
    torch.manual_seed(0)
    z = torch.randn(VOCAB) * 3
    suppress, tgt = [0, 3, 50], [44, 45, 50]
    comp = int(torch.argmax(torch.where(torch.isin(torch.arange(VOCAB), torch.tensor(tgt + suppress)), -1e9, z)))
    v = rj.evaluator_targets(z, 5, suppress, [], tgt, comp, PART)
    zp = z.clone(); zp[suppress] = -float("inf")
    ref = torch.logsumexp(zp[tgt], 0)
    assert float(v["margin"]) == pytest.approx(float(ref - zp[comp]), abs=1e-6)
    lz = torch.logsumexp(zp, 0)
    assert float(v["logp_ref"]) == pytest.approx(float(ref - lz), abs=1e-6)
    assert float(v["log_PE"]) == pytest.approx(float(torch.logsumexp(zp[PART["embedded_ids"]], 0) - lz), abs=1e-6)
    # equals the P2-R summary margin (best other) when c* is the best non-reference token
    s, _ = p2r.summarize(z, 5, suppress, [], PART, tgt, None)
    assert float(v["margin"]) == pytest.approx(s["margin_ref_vs_competitor"], abs=1e-5)
    # begin-suppression only at step 0
    v0 = rj.evaluator_targets(z, 0, [], [45], tgt, comp, PART)
    assert float(v0["margin"]) == pytest.approx(float(torch.logsumexp(z[[44, 50]], 0) - z[comp]), abs=1e-6)


def test_tangent_projection_and_first_order_bound_exact():
    torch.manual_seed(1)
    r, g = torch.randn(32, dtype=torch.float64), torch.randn(32, dtype=torch.float64)
    gt = rj.tangent(g, r)
    assert abs(float(torch.dot(gt, r))) < 1e-12
    assert torch.allclose(rj.tangent(gt, r), gt)
    rh = r / r.norm()
    assert torch.allclose(gt + rh * torch.dot(rh, g), g)
    d = torch.randn(32, dtype=torch.float64)
    dl, st = rj.chord_edit(r, d, 0.5)
    assert st == "ok" and float(dl.norm()) == pytest.approx(0.5, abs=1e-12)
    assert float((r + dl).norm()) == pytest.approx(float(r.norm()), abs=1e-12)          # NormPreserve
    stats = rj.position_stats(r, {k: g for k in rj.TARGETS}, {"plus_d": d, "minus_d": -d, "random": None},
                              {"plus_d": dl}, 0.5)
    assert stats["A"] == pytest.approx(float(gt.norm()) * 0.5)
    assert stats["arms"]["plus_d"]["pred_margin"] == pytest.approx(float(torch.dot(g, dl)))
    assert stats["arms"]["plus_d"]["kappa"] == pytest.approx(float(torch.dot(gt, dl)) / stats["A"])
    assert stats["arms"]["plus_d"]["cos_gtan_v_margin"] == pytest.approx(-stats["arms"]["minus_d"]["cos_gtan_v_margin"])
    assert abs(stats["arms"]["plus_d"]["kappa"]) <= 1 + 1e-9
    # numpy recomputation used by the analysis agrees
    vec = {"t3_r": r.numpy(), "t3_g_margin": g.numpy(), "t3_d": d.numpy(), "t3_v_random": d.numpy(),
           "t3_delta_plus_d": dl.numpy()}
    rc = an.recompute(vec, 3, 0.5)
    assert rc["A"] == pytest.approx(stats["A"]) and rc["arms"]["plus_d"]["kappa"] == pytest.approx(stats["arms"]["plus_d"]["kappa"])


# ---- probe: exact state, value identity, correct gradient, never an edit -------------------

def test_probe_forward_is_value_identical_and_gradient_matches_finite_difference(monkeypatch):
    b = tiny_bundle()
    b.model.requires_grad_(False)
    toks = _baseline(b, monkeypatch)["tokens"]
    ctx = _ctx(b)
    br = p2r.DiagBranch(b, encoded(), CB, "B")
    rj.nograd_step(br, CB, capture_layer=16, attention=True)
    new = [toks[0]]
    tgt, comp = [44, 45, 46], toks[1]
    gres = rj.grad_step(ctx, br, new, 1, tgt, comp)
    L = br.length
    lb, _, hb = rj.nograd_step(br, new, capture_layer=16, attention=True)
    assert torch.equal(gres["logits"], lb)                           # zero probe: value identical
    assert torch.equal(gres["site"].float().cpu(), hb)               # probe site == recorder site r
    assert gres["query"] == L
    # directional finite difference through a (test-only) additive edit at the same site
    g = gres["grads"]["margin"]
    u = torch.randn(g.numel(), dtype=torch.float64); u /= u.norm()
    eps = 1e-3

    def m_at(sign):
        br.crop(L)
        hook = DecoderPostCrossAttnInterventionHook(b, 16, (sign * eps * u).float(), alpha=1.0,
                                                    num_forced_prefix=4, norm_preserve=False)
        logits, _, _ = _hooked_step(br, new, hook)
        return float(rj.evaluator_targets(logits, 1, ctx.suppress, ctx.begin, tgt, comp, PART)["margin"])
    fd = (m_at(1) - m_at(-1)) / (2 * eps)
    assert fd == pytest.approx(float(g @ u), rel=2e-2, abs=1e-4)
    # the branch cache was never advanced by the gradient step
    br.crop(L)
    lb2, _, _ = rj.nograd_step(br, new, capture_layer=16, attention=True)
    assert torch.equal(lb2, lb)


def _hooked_step(br, new, hook):
    with torch.no_grad():
        return cached.Branch.step.__wrapped__(br, new, attention=True, hook=hook)


def test_probe_refuses_nonzero_delta():
    b = tiny_bundle()
    probe = rj.SiteGradientProbe(b, 16, 5)
    with torch.no_grad():
        probe.delta.add_(1.0)
    with pytest.raises(RuntimeError, match="never edit"):
        probe.__enter__()
    import inspect
    assert set(inspect.signature(rj.SiteGradientProbe).parameters) == {"bundle", "layer", "query"}


def test_gradient_never_applied_as_steering_direction():
    src = Path("experiments/inference_cf_p2rj.py").read_text()
    tree = ast.parse(src)
    edit_calls = {"apply_steering", "hook_edit", "solve_scale", "chord_edit", "scaled_direction", "pulse_hook",
                  "_edit_hook", "DecoderPostCrossAttnInterventionHook", "emulate_edit_norm"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in edit_calls:
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    txt = ast.get_source_segment(src, arg) or ""
                    assert "grad" not in txt and "gt" != txt.strip(), (name, txt)
    # no steering hook / deployable decode is ever constructed by the runner
    assert "DecoderPostCrossAttnInterventionHook" not in src and "_edit_hook" not in src
    assert "cached_decode" not in src and "pulse_hook" not in src
    # the only hooked forward is the zero probe
    assert src.count("hook=") == 0


def test_deployable_modules_byte_identical_to_p2r_freeze():
    man = json.loads(Path("results/inference_cf/p2r/run3/manifest.json").read_text())
    for rel in ("experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_r2.py",
                "src/csasr/inference_cf/core_p1.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py",
                "experiments/inference_cf_p2r.py"):
        key = next(k for k in man["sources"] if k.endswith(rel))
        assert file_hash(Path(rel)) == man["sources"][key], rel
    for mod in ("experiments/inference_cf_cached.py", "experiments/inference_cf_p2.py"):
        assert "p2rj" not in Path(mod).read_text() and "gradient" not in Path(mod).read_text().lower()


# ---- exact P2-R state / direction / random / solver reproduction ---------------------------

def test_run_utterance_reproduces_p2r_pulse_state_exactly(monkeypatch):
    b = tiny_bundle()
    b.model.requires_grad_(False)
    toks = _baseline(b, monkeypatch)["tokens"]
    positions = _positions(toks)
    # P2-R reference: pulse_pass with the frozen arms at energy E_TINY
    pcfg = json.loads(json.dumps(P2R_CFG)); pcfg["conditions"] = COND; pcfg["site"]["max_new_tokens"] = 8
    pctx = p2r.Ctx(b, pcfg, PART, {4: .5, 5: .5}, (4, 5), 4)
    jobs = {p["t"]: {"target_ids": p["target_ids"], "stratum": p["stratum"],
                     "arms": {a: {"dir": a, "energy": E_TINY, "continue": False} for a in p2r.ARMS}} for p in positions}
    with torch.inference_mode():
        ref = p2r.pulse_pass(pctx, encoded=encoded(), uid="u", tokens=toks, jobs=jobs)
    recs, vecs = rj.run_utterance(_ctx(b), encoded=encoded(), uid="u", tokens=toks, positions=positions, mode="bf16")
    for p in positions:
        t = p["t"]
        rec, pr = recs[str(t)], ref[t]
        idt = rec["identity"]
        assert idt["logits_bitwise_equal"] and idt["site_equals_recorder"]
        assert idt["cos_d_state"] == pr["cos_d_state"] and idt["dir_norm"] == pr["dir_norm"]
        assert idt["pre_norm_audit_style"] == pr["arms"]["plus_d"]["pre_norm"]
        assert idt["logp_ref"] == pr["none"]["logp_ref"] and idt["argmax"] == pr["none"]["argmax"]
        for a in rj.ARMS:
            assert rec["solver"][a]["s"] == pr["arms"][a]["solver"]["s"]          # bitwise
            dl = vecs[f"t{t}_delta_{a}"]
            assert float(np.linalg.norm(dl)) == pytest.approx(pr["arms"][a]["edit_norm"], rel=1e-5)
        # sign exact: +d / -d are exact negatives; random is the frozen P2-R vector
        assert float(np.linalg.norm(vecs[f"t{t}_d"])) == pytest.approx(1.0, abs=1e-3)   # core_p1: delta/(norm+eps)
        assert rec["solver"]["plus_d"]["phi"] + rec["solver"]["minus_d"]["phi"] == pytest.approx(math.pi, abs=1e-9)
        st = rec["stats"]["arms"]
        assert st["plus_d"]["cos_gtan_v_margin"] == pytest.approx(-st["minus_d"]["cos_gtan_v_margin"], abs=1e-12)
        v_ref, _ = p2r.random_direction("u", t, torch.from_numpy(vecs[f"t{t}_r"]))
        assert np.array_equal(vecs[f"t{t}_v_random"], v_ref.numpy())
        assert abs(float(np.dot(vecs[f"t{t}_v_random"], vecs[f"t{t}_r"]))) < 1e-6


def test_fp32_mode_uses_analytic_chord_and_serializes(monkeypatch, tmp_path):
    b = tiny_bundle()
    b.model.requires_grad_(False)
    toks = _baseline(b, monkeypatch)["tokens"]
    recs, vecs = rj.run_utterance(_ctx(b), encoded=encoded(), uid="u", tokens=toks, positions=_positions(toks), mode="fp32")
    for t, rec in recs.items():
        for a in rj.ARMS:
            assert rec["solver"][a]["status"] == "ok"
            assert float(np.linalg.norm(vecs[f"t{t}_delta_{a}"])) == pytest.approx(E_TINY, abs=1e-12)
    np.savez_compressed(tmp_path / "x.npz", **vecs)
    with np.load(tmp_path / "x.npz") as z:
        back = {k: z[k] for k in z.files}
    assert set(back) == set(vecs) and all(np.array_equal(back[k], vecs[k]) for k in vecs)
    js = json.loads(json.dumps(recs))
    for t, rec in js.items():
        rc = an.recompute(back, int(t), E_TINY)
        assert rc["A"] == pytest.approx(rec["stats"]["A"], rel=1e-9)
        for a in rj.ARMS:
            assert rc["arms"][a]["kappa"] == pytest.approx(rec["stats"]["arms"][a]["kappa"], rel=1e-9, abs=1e-12)


# ---- frozen inputs -------------------------------------------------------------------------

def test_positions_file_is_the_unchanged_p2r_population():
    pos = json.loads(Path(prep.POSITIONS).read_text())
    assert pos["positions_hash"] == digest({k: v for k, v in pos.items() if k != "positions_hash"})
    assert pos["config_hash"] == digest(CFG)
    pop = json.loads(Path("results/inference_cf/p2r/population.json").read_text())
    assert pos["utterances"] == pop["utterances"]
    want = [(s, c["utterance_id"], c["t"], c["target_ids"]) for s in an.STRATA for c in pop["d1"][s]]
    got = [(p["stratum"], p["utterance_id"], p["t"], p["target_ids"]) for p in pos["positions"]]
    assert got == want and len(got) == 180
    assert len({(p["utterance_id"], p["t"]) for p in pos["positions"]}) == 180     # runner keys rows by (uid, t)
    for p in pos["positions"]:
        assert p["competitor"] not in p["target_ids"]
        if p["stratum"] == "EN-confusion":
            assert p["competitor"] == p["baseline_token"] and p["p2r"]["margin"] < 0
    # rebuilding from the P2-R run reproduces the frozen file exactly
    man = json.loads(Path(prep.P2R_RUN, "manifest.json").read_text())
    rows = {u: json.loads(Path(prep.P2R_RUN, "rows", f"{i:03d}.json").read_text()) for i, u in enumerate(pop["utterances"])}
    assert prep.build_positions(pop, rows) == pos["positions"]
    assert man["manifest_hash"] == pos["p2r_run_manifest_hash"] == CFG["p2r_reference"]["run_manifest_hash"]


def test_reference_information_stays_out_of_deployable_path():
    runner = Path("experiments/inference_cf_p2rj.py").read_text()
    assert "load_references" not in runner and "transcript_raw" not in runner and "evaluation_units" not in runner
    for mod in ("experiments/inference_cf_cached.py", "experiments/inference_cf_p2.py", "src/csasr/lss/sites.py"):
        assert "SiteGradientProbe" not in Path(mod).read_text()


def test_pooled_pearson_bootstrap_and_dialogue_weighting():
    rng = np.random.default_rng(0)
    pairs = [(f"D{i % 6}", float(x), float(2 * x + rng.normal(0, .1))) for i, x in enumerate(rng.normal(size=60))]
    out = an.pooled_pearson_boot(pairs, 500, 1, 0.05)
    assert out["estimate"] > 0.9 and out["ci"][0] > 0.8 and out["slope_through_origin"] == pytest.approx(2, abs=0.1)
    recs = [{"dialogue_id": "A", "stratum": "EN-confusion", "rho": 0.9, "kappa_plus_d": 0.0}] * 3 + \
           [{"dialogue_id": "B", "stratum": "EN-confusion", "rho": 0.1, "kappa_plus_d": 0.2}]
    la = an.leverage_alignment(recs, 200, 0, 0.025)
    assert la["Q50"]["estimate"] == pytest.approx(0.5) and la["K_plus"]["estimate"] == pytest.approx(0.1)


def test_state_check_tie_aware_v1_1():
    p = {"baseline_token": 7, "competitor": 9,
         "p2r": {"argmax": 8, "logp_ref": -1.0, "margin": -3.0, "pre_norm": 7.5, "cos_d_state": -0.1, "dir": "ok",
                 "arms": {a: {"s": 1.1} for a in an.ARMS}}}
    idt = {"argmax": 7, "logp_ref": -1.0, "margin_best_other": -3.0, "competitor_recomputed": 9,
           "pre_norm_audit_style": 7.5, "cos_d_state": -0.1, "dir": "ok", "logits_bitwise_equal": True,
           "site_equals_recorder": True, "logp_baseline_token": -0.7, "logp_p2r_argmax": -0.7,
           "logp_competitor": -2.0, "max_other_logp": -2.0}
    rec = {"identity": idt, "values": {"margin": -3.0}, "solver": {a: {"status": "ok", "s": 1.1} for a in an.ARMS}}
    assert all(an.state_checks(p, rec).values())                  # exact tie: accepted
    idt["logp_p2r_argmax"] = -0.69
    assert not an.state_checks(p, rec)["argmax"]                  # no tie: a real mismatch
    idt["logp_p2r_argmax"], idt["argmax"] = -0.7, 8
    assert not an.state_checks(p, rec)["argmax"]                  # decode rule must give the baseline token
    idt["argmax"], idt["competitor_recomputed"], idt["logp_competitor"] = 7, 10, -2.5
    assert not an.state_checks(p, rec)["competitor"]
