"""DIR-SPRINT0 focused CPU tests: D3 candidate rule / token-margin gradient, D4, D0, random, D5 evidence / prototypes /
direction / shuffle, targets, runtime firewall, evaluator precedence, and primary-vs-auditor cross-implementation
agreement. No CS-Dialogue audio, reference or model checkpoint is touched (tiny random Whisper only)."""
from __future__ import annotations

import ast
import copy
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from test_inference_cf_p1 import CB as TCB, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
from csasr.inference_cf import dir_sprint0 as X  # noqa: E402
from csasr.inference_cf import readout as R  # noqa: E402
from csasr.inference_cf.core import digest  # noqa: E402
from csasr.inference_cf.core_p1 import direction as core_direction  # noqa: E402

AUD = importlib.import_module("experiments.inference_cf_dir_sprint0_audit")
EVA = importlib.import_module("experiments.inference_cf_dir_sprint0_evaluate")


# ---- runtime firewall ---------------------------------------------------------------------------------------------------

def _pop():
    sel = [{"canonical_index": i, "utterance_id": f"U{i}", "audio_path": f"/a/{i}.wav", "audio_full_sha256": "sha256:x",
            "dialogue_id": "D", "source_duration_sec": 1.0} for i in range(3)]
    bank = [{"bank_index": i, "utterance_id": f"B{i}", "audio_path": f"/b/{i}.wav", "audio_full_sha256": "sha256:y",
             "dialogue_id": "D"} for i in range(2)]
    return {"selected": sel, "calibration_bank": {"rows": bank}}


def test_runtime_projection_allowlist_and_rejections():
    p = X.runtime_projection(_pop())
    rows, bank = X.validate_runtime_panel(p)
    assert all(set(r) == set(X.RUNTIME_KEYS) for r in rows) and all(set(r) == set(X.BANK_KEYS) for r in bank)
    bad = copy.deepcopy(p)
    bad["rows"][0]["dialogue_id"] = "D"
    with pytest.raises(ValueError):
        X.validate_runtime_panel(bad)
    bad = copy.deepcopy(p)
    bad["bank_rows"][0]["utterance_id"] = "U0"
    bad["bank_ids_hash"] = digest(sorted(r["utterance_id"] for r in bad["bank_rows"]))
    bad["runtime_hash"] = digest({k: v for k, v in bad.items() if k != "runtime_hash"})
    with pytest.raises(ValueError):
        X.validate_runtime_panel(bad)
    bad = copy.deepcopy(p)
    bad["extra"] = 1
    with pytest.raises(ValueError):
        X.validate_runtime_panel(bad)


# ---- D3 ------------------------------------------------------------------------------------------------------------------

def test_utf8_boundary_rule():
    assert X.utf8_boundary_legal(b" hello")
    assert X.utf8_boundary_legal("中".encode()[:2])            # lead fragment: legal at a boundary
    assert X.utf8_boundary_legal("中".encode())
    assert not X.utf8_boundary_legal("中".encode()[1:])         # starts with a continuation byte
    assert not X.utf8_boundary_legal(b"\xff")                   # invalid byte
    assert not X.utf8_boundary_legal(b"")


def _logits_from_logp(lp: dict, V=20, floor=-60.0):
    z = np.full(V, floor, dtype=np.float32)
    for k, v in lp.items():
        z[k] = v
    return z


def test_d3_candidate_rule():
    legal = np.ones(20, dtype=bool)
    legal[[0, 19]] = False                                     # 19 plays EOS; 0 is not legal
    zb = _logits_from_logp({5: 10.0, 7: 6.0, 8: 1.0, 9: 9.0})   # b=5; 7 inside log(1e-3)=-6.9 window; 8 outside
    zn = _logits_from_logp({5: 10.0, 7: 0.0, 8: -40.0, 9: 9.0})  # null strongly prefers 5 / 9, not 7 / 8
    r = X.d3_candidate(zb, zn, legal, [0], 5)
    assert r["status"] == "candidate" and r["c_AP"] == 7          # 8 has a huge contrast but is implausible
    # baseline is the winner -> exact no-edit
    r2 = X.d3_candidate(zb, zb, legal, [0], 5)
    assert r2["status"] == "baseline_is_candidate" and r2["c_AP"] == 5
    # suppressed / illegal tokens never chosen
    r3 = X.d3_candidate(zb, zn, legal, [7], 5)
    assert r3["c_AP"] != 7
    # EOS as baseline token is allowed; EOS itself never a candidate
    zb2 = _logits_from_logp({19: 10.0, 7: 6.0})
    r4 = X.d3_candidate(zb2, _logits_from_logp({19: 10.0, 7: 0.0}), legal, [0], 19)
    assert r4["c_AP"] == 7 and r4["baseline_in_P"] is False and r4["status"] == "candidate"
    # nothing legal inside the window
    zb3 = _logits_from_logp({19: 10.0, 0: 9.0})
    r5 = X.d3_candidate(zb3, zb3, legal, [], 19)
    assert r5["status"] == "no_plausible_legal_candidate" and r5["c_AP"] is None


def test_d3_null_floor_and_lowest_id_tie():
    legal = np.ones(20, dtype=bool)
    zb = _logits_from_logp({3: 5.0, 4: 5.0, 6: 9.0})
    zn = _logits_from_logp({6: 50.0}, floor=-200.0)            # 3 and 4 both below the 1e-12 null floor
    r = X.d3_candidate(zb, zn, legal, [], 6)
    assert r["c_AP"] == 3 and r["ties_at_max"] == 2 and r["null_floor_hit_c"]
    lb = torch.log_softmax(torch.tensor(zb), -1).double().numpy()
    assert r["S_c"] == pytest.approx(2 * lb[3] - math.log(1e-12))


@pytest.mark.parametrize("seed", range(6))
def test_d3_primary_equals_auditor(seed):
    rng = np.random.default_rng(seed)
    V = 300
    zb = (rng.standard_normal(V) * 3).astype(np.float32)
    zn = (rng.standard_normal(V) * 3).astype(np.float32)
    legal = rng.random(V) > 0.1
    sup = [1, 2, 3]
    b = int(np.argmax(np.where(np.isin(np.arange(V), sup), -np.inf, zb)))
    p = X.d3_candidate(zb, zn, legal, sup, b)
    a = AUD.own_d3(zb, zn, legal, sup, b)
    assert p["status"] == a["status"] and p["c_AP"] == a["c_AP"]


def _prefix_branch(bundle, steps=3):
    enc = encoded()
    br = cached.Branch(bundle, enc, TCB, "B")
    new = list(TCB)
    toks = [10, 47, 12, 50]
    for t in range(steps):
        br.step(new, capture_layer=16)
        new = [toks[t]]
    return br, enc, new, steps


def test_d3_token_margin_gradient_identity_isolation_and_fd():
    bundle = tiny_bundle(seed=3)
    bundle.model.requires_grad_(True)
    br, enc, new, t = _prefix_branch(bundle)
    fp = R.cache_fingerprint(br.cache)
    r = X.token_margin_direction(bundle, layer=16, cache=br.cache, encoded=enc, new_tokens=new, start=br.length, c_token=30, b_token=12)
    assert r["status"] == "ok" and r["counters"] == {"autograd_calls": 1, "scratch_forwards": 1}
    assert all(p.requires_grad for p in bundle.model.parameters()) and all(p.grad is None for p in bundle.model.parameters())
    assert R.cache_fingerprint(br.cache) == fp
    logits, _, site = br.step(new, capture_layer=16)
    assert torch.equal(r["logits"], logits) and torch.equal(r["site"], site)
    assert r["J"] == pytest.approx(float(logits[30] - logits[12]), abs=1e-6)
    d = r["direction"].double()
    assert abs(float(d.norm()) - 1) <= R.UNIT_TOL and abs(float(d @ site.double() / site.double().norm())) <= R.RADIAL_TOL
    # finite difference of z[c] - z[b] along random directions through the exact DG-02 site
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook
    br2, enc2, new2, _ = _prefix_branch(bundle)
    qpos = br2.length + len(new2) - 1
    g = r["gradient"].double()

    def J_at(vec):
        cache = copy.deepcopy(br2.cache)

        def act(q=None, u_source=None, r=None, abs_pos=None):
            gg = torch.zeros((1, r.shape[1]), dtype=r.dtype)
            dd = torch.zeros((1, r.shape[1], r.shape[-1]), dtype=r.dtype)
            hit = (abs_pos == qpos).nonzero().flatten()
            gg[0, hit] = 1.0
            dd[0, hit] = vec.to(r.dtype)
            return gg, dd
        hook = DecoderPostCrossAttnInterventionHook(bundle, 16, None, alpha=1.0, num_forced_prefix=4, action_fn=act, norm_preserve=False)
        with torch.no_grad(), hook:
            out = bundle.model(encoder_outputs=enc2, decoder_input_ids=torch.tensor([new2]), past_key_values=cache,
                               use_cache=True, cache_position=torch.arange(br2.length, br2.length + len(new2)))
        z = out.logits[0, -1].float()
        return float(z[30] - z[12])
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        v = torch.randn(16, generator=gen, dtype=torch.float64)
        v /= v.norm()
        eps = 1e-3
        fd = (J_at((eps * v).float()) - J_at((-eps * v).float())) / (2 * eps)
        assert fd == pytest.approx(float(g @ v), rel=2e-2, abs=1e-4)


# ---- D4 / D0 / random ------------------------------------------------------------------------------------------------

def test_d4_tangent_and_sign():
    rng = np.random.default_rng(1)
    htr, htl = rng.standard_normal(64).astype(np.float32), rng.standard_normal(64).astype(np.float32)
    r = X.d4_direction(htr, htl)
    d = r["direction"].double().numpy()
    hh = htr.astype(np.float64) / np.linalg.norm(htr)
    assert r["status"] == "ok" and abs(d @ hh) <= 1e-6 and abs(np.linalg.norm(d) - 1) <= 2e-6
    gp = (htr.astype(np.float64) - htl) - hh * (hh @ (htr.astype(np.float64) - htl))
    assert np.allclose(d, gp / np.linalg.norm(gp), atol=1e-6)            # transcribe minus translate, never reversed
    own, s = AUD.tangent64(htr.astype(np.float64) - htl, htr)
    assert s == "ok" and np.abs(own - r["direction"].numpy()).max() <= 1e-6
    assert X.d4_direction(htr, htr)["status"] == "invalid"


def test_d0_equals_core_p1_and_random_determinism():
    rng = np.random.default_rng(2)
    he, hb = rng.standard_normal(1280).astype(np.float32), rng.standard_normal(1280).astype(np.float32)
    assert torch.equal(X.d0_direction(he, hb)["d"], core_direction(torch.tensor(he), torch.tensor(hb))["d"])
    assert np.abs(AUD.own_d0(he, hb) - X.d0_direction(he, hb)["d"].numpy()).max() <= 1e-7
    a, b = X.random_direction("U", 3), X.random_direction("U", 3)
    assert np.array_equal(a, b) and abs(np.linalg.norm(a.astype(np.float64)) - 1) < 1e-6
    assert not np.array_equal(a, X.random_direction("U", 4)) and np.array_equal(a, AUD.own_random("U", 3))


# ---- D5 --------------------------------------------------------------------------------------------------------------

def _table():
    """Synthetic 6-symbol table: 0 blank, 1-3 segmental, 4 excluded, 5 special; concept matrix [6, 8]."""
    valid = np.array([False, True, True, True, False, False])
    excluded = np.array([False, False, False, False, True, False])
    C = np.zeros((6, 8))
    C[1, [0, 3]] = 1.0          # nasal labial (m)
    C[2, [1, 4]] = 1.0          # stop coronal (t)
    C[3, [2, 5]] = 1.0          # fricative dorsal (x)
    return valid, excluded, C


def _post(frames):
    lp = np.log(np.asarray(frames, dtype=np.float64) + 1e-30).astype(np.float32)
    return lp


def test_concept_evidence_rules():
    valid, excluded, C = _table()
    silence = [[0.95, 0.01, 0.01, 0.01, 0.01, 0.01]] * 60          # segmental mass 0.03 per frame: never emitting
    w = np.ones(60)
    assert X.concept_evidence(_post(silence), w, valid, excluded, C)["status"] == "low_emitting_weight"
    speech = [[0.95, 0.01, 0.01, 0.01, 0.01, 0.01]] * 60
    speech[10] = [0.01, 0.97, 0.01, 0.0, 0.005, 0.005]          # m spike
    speech[20] = [0.01, 0.0, 0.98, 0.0, 0.005, 0.005]           # t spike
    speech[30] = [0.01, 0.0, 0.98, 0.0, 0.005, 0.005]           # t spike
    ev = X.concept_evidence(_post(speech), w, valid, excluded, C)
    assert ev["status"] == "ok" and ev["emitting_weight"] == 3.0
    p = np.exp(_post(speech).astype(np.float64))
    s = p[:, valid].sum(1)
    we = w * (s >= .5)
    q = (we @ (p * valid) @ C) / (we @ s)
    assert np.allclose(ev["q"], q[:6]) and np.allclose(ev["q_descriptive"], q[6:])
    assert ev["q"][1] == pytest.approx((0.01 + 0.98 + 0.98) / (0.98 * 3), abs=1e-6)     # STOP share incl. the 0.01 in the m spike
    own = AUD.own_evidence(_post(speech), 0, 60 * 320 + 80, valid, excluded, C,
                           {"validity": {"window_frames_min": 20, "emitting_weight_min": 2.0, "excluded_share_max": 0.1}})
    assert own["status"] == "ok" and np.allclose(own["q"], ev["q"])
    # too few frames, nonfinite, excluded share
    assert X.concept_evidence(_post(speech), np.r_[np.ones(10), np.zeros(50)], valid, excluded, C)["status"] == "too_few_window_frames"
    bad = _post(speech)
    bad[3, 0] = np.nan
    assert X.concept_evidence(bad, w, valid, excluded, C)["status"] == "nonfinite_posteriors"
    exc = [list(f) for f in speech]
    for j in (40, 41, 42, 43):
        exc[j] = [0.01, 0.0, 0.0, 0.0, 0.98, 0.01]
    exc[44] = [0.01, 0.0, 0.6, 0.0, 0.38, 0.01]                 # emitting frames (s_j = 0.6) with large excluded mass;
    exc[45] = [0.01, 0.0, 0.6, 0.0, 0.38, 0.01]                 # non-emitting frames 40-43 never count
    assert X.concept_evidence(_post(exc), w, valid, excluded, C)["status"] == "excluded_mass"


def test_window_weights_fractional_overlap():
    w = X.window_weights(0, 16000, 60)
    assert w[0] == 1.0 and w[49] == pytest.approx((16000 - 49 * 320) / 400) and w[50] == 0.0
    assert np.array_equal(w, AUD.own_weights(0, 16000, 60))
    w2 = X.window_weights(100, 900, 5)
    assert w2.tolist() == pytest.approx([300 / 400, 1.0, 260 / 400, 0, 0])


def _bank(seed=0, n=4000, dims=32, dialogues=20):
    rng = np.random.default_rng(seed)
    Q = rng.random((n, 6))
    axis = np.linalg.qr(rng.standard_normal((dims, 6)))[0].T          # concept k raises the state along axis[k]
    H = 5 * rng.standard_normal((n, dims)) / np.sqrt(dims) + 10 * np.ones(dims) / np.sqrt(dims) + Q @ axis
    D = [f"D{i % dialogues:02d}" for i in range(n)]
    return Q, H, D, axis


def test_prototypes_recover_axes_and_match_auditor(monkeypatch):
    Q, H, D, axis = _bank()
    p = X.build_prototypes(Q, H, D)
    assert p["status"] == "ok"
    for k in range(6):
        assert float(p["V"][k] @ axis[k]) / np.linalg.norm(axis[k]) > 0.5
    d5 = {"bank": {"quantiles": [0.2, 0.8], "dialogue_side_min": 5, "dialogues_min": 15, "side_total_min": 500}}
    own = AUD.own_prototypes(Q, H, D, d5)
    assert own["ok"] and np.abs(own["V"] - p["V"]).max() <= 1e-12 and np.abs(own["mu"] - p["mu"]).max() <= 1e-12
    # degenerate concept and insufficient dialogues
    Q2 = Q.copy()
    Q2[:, 2] = 0.0
    assert X.build_prototypes(Q2, H, D)["per_concept"][2]["status"] == "degenerate_quantiles"
    assert X.build_prototypes(Q[:400], H[:400], D[:400])["status"] == "prototype_construction_failed"


def test_d5_direction_and_shuffle():
    Q, H, D, _ = _bank(1)
    p = X.build_prototypes(Q, H, D)
    h = H[0].astype(np.float32)
    r = X.d5_direction(Q[0], p["mu"], p["sd"], p["V"], h)
    z = (Q[0] - p["mu"]) / p["sd"]
    own, _ = AUD.tangent64(z @ p["V"], h)
    assert r["status"] == "ok" and np.abs(own - r["direction"].numpy()).max() <= 1e-6
    assert X.d5_direction(p["mu"], p["mu"], p["sd"], p["V"], h)["reason"] == "null_evidence_contrast"
    sh = X.d5_shuffle("U1", [5, 1, 9, 3])
    assert sh["fixed_points"] == 0 and sorted(sh["donor"]) == sorted(sh["donor"].values()) == [1, 3, 5, 9]
    assert sh["donor"] == AUD.own_shuffle("U1", [1, 3, 5, 9]) == X.d5_shuffle("U1", [9, 3, 1, 5])["donor"]
    one = X.d5_shuffle("U1", [4])
    assert one["singleton"] and one["donor"] == {4: 4}


def test_concept_matrix_order_with_frozen_table():
    import json
    ct = json.loads((ROOT / "results/inference_cf/dir_sprint0/design/d5_concept_table.json").read_text())
    C, classes = X.concept_matrix(ct, 392)
    assert classes == list(X.PRIMARY_CONCEPTS) + list(X.DESCRIPTIVE_CONCEPTS)
    sym = {r["symbol"]: r["id"] for r in ct["rows"]}
    assert C[sym["m"]].tolist()[:6] == [1, 0, 0, 1, 0, 0] and C[sym["k"]].tolist()[:6] == [0, 1, 0, 0, 0, 1]
    assert C[sym["h"]].sum() == 0 and C[sym["ts.h"]][7] == 1.0          # glottal outside primary; aspiration descriptive
    assert C[0].sum() == 0                                              # blank carries nothing


def test_frozen_constants_equal_config():
    import json
    cfg = json.loads((ROOT / "configs/inference_cf/dir_sprint0.json").read_text())
    v = cfg["families"]["D5"]["validity"]
    assert (X.WINDOW_FRAMES_MIN, X.EMITTING_WEIGHT_MIN, X.EXCLUDED_SHARE_MAX) == (v["window_frames_min"], v["emitting_weight_min"], v["excluded_share_max"])
    b = cfg["families"]["D5"]["bank"]
    assert [X.Q_LO, X.Q_HI] == b["quantiles"] and X.DIALOGUES_MIN == b["dialogues_min"] and X.SIDE_TOTAL_MIN == b["side_total_min"]
    assert X.DIALOGUE_SIDE_MIN == b["dialogue_side_min"] and list(X.PRIMARY_CONCEPTS) == cfg["families"]["D5"]["primary_concepts"]
    assert X.D3_LOG_ALPHA == math.log(1e-3) and X.D3_LOG_NULL_FLOOR == math.log(1e-12)
    assert list(X.PULSE_ARMS) == cfg["arm_order"] == list(EVA.PULSE_ARMS) == list(AUD.PULSE_ARMS)
    assert X.GATED_BASE == EVA.GATED_BASE == AUD.GATED_BASE
    assert X.SHUFFLE_TAG == cfg["families"]["D5"]["shuffle"]["tag"] and X.RANDOM_TAG == cfg["families"]["RND"]["tag"]


def test_arm_targets():
    e = 1.1260757575454359
    assert X.arm_target("D3", e, 0.0) == e and X.arm_target("D3G", e, 0.0) == 0.0
    assert X.arm_target("D5G", e, 0.5) == e * 0.5
    with pytest.raises(ValueError):
        X.arm_target("B0", e, 1.0)


# ---- evaluator decision structure (synthetic) --------------------------------------------------------------------------

def _variant(energy=True, power=True, adv=True, est=True, safe=True, mech=False):
    return {"passes": {"energy": energy, "power": power, "advantage": adv, "estimability": est, "safety": safe, "mechanistic": mech}}


def test_family_status_precedence():
    import json
    cfg = json.loads((ROOT / "configs/inference_cf/dir_sprint0.json").read_text())
    cov = {"fraction_structural": 0.9, "fraction_EN_confusion": 0.9}
    v = {"D3": _variant(safe=False), "D3G": _variant()}
    s = EVA.family_status(cfg, "D3", False, cov, v)
    assert s["outcome"] == "PROMISING_FEASIBILITY" and s["qualifying_variant"] == "D3G"
    v = {"D3": _variant(safe=False), "D3G": _variant(power=False)}
    assert EVA.family_status(cfg, "D3", False, cov, v)["outcome"] == "SAFETY_FAILED"
    v = {"D3": _variant(est=False, safe=False), "D3G": _variant(power=False)}
    assert EVA.family_status(cfg, "D3", False, cov, v)["outcome"] == "INSUFFICIENT_COVERAGE"
    v = {"D3": _variant(adv=False, mech=True), "D3G": _variant(energy=False)}
    s = EVA.family_status(cfg, "D3", False, cov, v)
    assert s["outcome"] == "LEXICAL_POWER_INSUFFICIENT" and s["mechanistic_effect"]
    assert EVA.family_status(cfg, "D3", True, cov, v)["validity"] == "PROVIDER_OR_CONSTRUCTION_BLOCKED"
    assert EVA.family_status(cfg, "D3", False, {"fraction_structural": 0.4, "fraction_EN_confusion": 0.9}, v)["validity"] == "INSUFFICIENT_COVERAGE"


def test_terminal_precedence():
    ok = {"critical_failure": False, "resource_pass": True, "opportunity_pass": True}
    st = lambda val, out, m=False: {"validity": val, "outcome": out, "mechanistic_effect": m}  # noqa: E731
    V = "VALID_AND_TESTED"
    base = {"D3": st(V, "LEXICAL_POWER_INSUFFICIENT"), "D4": st(V, "LEXICAL_POWER_INSUFFICIENT"), "D5": st(V, "LEXICAL_POWER_INSUFFICIENT")}
    assert EVA.decide(ok, base) == "DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE"
    assert EVA.decide(ok, {**base, "D4": st(V, "LEXICAL_POWER_INSUFFICIENT", True)}) == "DIR_SPRINT0_MECHANISTIC_EFFECT_ONLY"
    assert EVA.decide(ok, {**base, "D3": st(V, "SAFETY_FAILED")}) == "DIR_SPRINT0_CAUSAL_POWER_WITH_DAMAGE"
    assert EVA.decide(ok, {**base, "D3": st(V, "SAFETY_FAILED"), "D5": st(V, "PROMISING_FEASIBILITY")}) == "DIR_SPRINT0_STEERING_FEASIBILITY_SIGNAL"
    assert EVA.decide({**ok, "opportunity_pass": False}, {**base, "D5": st(V, "PROMISING_FEASIBILITY")}) == "DIR_SPRINT0_OPPORTUNITY_INSUFFICIENT"
    assert EVA.decide({**ok, "resource_pass": False}, base) == "DIR_SPRINT0_COMPUTE_BLOCKED"
    assert EVA.decide({**ok, "critical_failure": True}, base) == "DIR_SPRINT0_INVALID"
    blk = st("PROVIDER_OR_CONSTRUCTION_BLOCKED", "NOT_TESTED")
    assert EVA.decide(ok, {"D3": blk, "D4": blk, "D5": blk}) == "DIR_SPRINT0_INVALID"
    cov = st("INSUFFICIENT_COVERAGE", "NOT_TESTED")
    assert EVA.decide(ok, {"D3": blk, "D4": cov, "D5": cov}) == "DIR_SPRINT0_OPPORTUNITY_INSUFFICIENT"


# ---- static firewall ------------------------------------------------------------------------------------------------------

def _imports(rel):
    tree = ast.parse((ROOT / rel).read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            out.add(n.module or "")
    return out


def test_runner_and_mechanics_never_import_reference_code():
    bad = ("pyarrow", "csasr.evaluation", "csasr.data.normalize", "experiments.inference_cf_p0_r2_evaluate",
           "experiments.inference_cf_p2r_population", "experiments.inference_cf_srd2_g0_evaluate",
           "experiments.inference_cf_dir_sprint0_evaluate", "experiments.inference_cf_dir_sprint0_audit", "phonemizer", "panphon")
    for rel in ("experiments/inference_cf_dir_sprint0.py", "src/csasr/inference_cf/dir_sprint0.py"):
        assert not any(i.startswith(b) for i in _imports(rel) for b in bad), rel


def test_auditor_is_independent():
    imp = _imports("experiments/inference_cf_dir_sprint0_audit.py")
    assert not imp & {"experiments.inference_cf_dir_sprint0", "experiments.inference_cf_dir_sprint0_evaluate",
                      "csasr.inference_cf.dir_sprint0", "csasr.inference_cf.srd2_g0", "experiments.inference_cf_srd2_g0"}


def test_gpu_phases_do_not_read_population_or_dialogues():
    src = (ROOT / "experiments/inference_cf_dir_sprint0.py").read_text()
    tree = ast.parse(src)
    seg = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for f in ("Ctx", "cached_baseline", "branch_replay", "capture_utterance", "capture_bank", "cmd_capture", "cmd_phones",
              "Engine", "pulse_utterance", "cmd_pulses"):
        assert "POPULATION" not in seg[f] and "dialogue_id" not in seg[f] and "dialogue_of" not in seg[f], f


def test_mechanics_has_no_reference_arguments():
    import inspect
    for fn in (X.d3_candidate, X.token_margin_direction, X.d4_direction, X.concept_evidence, X.build_prototypes, X.d5_direction,
               X.d5_shuffle, X.vac_target, X.runtime_projection):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"reference", "target_ids", "Y", "stratum", "transcript", "language", "gold"}
