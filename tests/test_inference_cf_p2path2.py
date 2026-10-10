"""P2-PATH2 focused tests (freeze e1c6ee7): exact panel/C-U/expectations; two independent instances (no LN storage alias,
mutating one leaves the other); online first any-token disagreement with exact shared prefix and one event; live H=3
rollouts with EOS/cap shortening; owner-locked fresh caches (stale owner rejected); G1 one-token commit + A4 release;
G2 theta0 history then fresh A4 replay (equivalent to incremental prompt+history feeding; differs from G1 on a toy
fixture); shared decision hash; frozen cutoffs/material advantage/label truth table; reference firewall; auditor
independence."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB  # noqa: E402
from test_inference_cf_p2dir_protocol import _enc  # noqa: E402
from test_inference_cf_p2tta0 import EOS_T, _bundle  # noqa: E402

import csasr.inference_cf.consensus_guard as cg  # noqa: E402
import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import csasr.inference_cf.path_decode as pd  # noqa: E402
import experiments.inference_cf_p2path2 as run  # noqa: E402
import experiments.inference_cf_p2path2_analyze as an  # noqa: E402
import experiments.inference_cf_p2path2_audit as au  # noqa: E402

C = json.loads(Path(run.CONFIG).read_text())


def test_panel_population_and_expectations():
    P = json.loads(Path(run.PANEL).read_text())
    assert [r["analysis_group"] for r in P["rows"]].count("C") == 5 and len(P["rows"]) == 12 and len({r["dialogue_id"] for r in P["rows"]}) == 9
    exp = [(r["utterance_id"], r["audit_only_expected"]["k"]) for r in P["rows"] if r["analysis_group"] == "C"]
    assert [k for _, k in exp] == [0, 5, 15, 0, 10]
    assert C["controller_success"]["ZH_errors_max"] == 29 and C["controller_success"]["POI_errors_max"] == 105 and C["controller_success"]["mixed_errors_max"] == 143


def _pair(perturb=0.0, seed=0):
    b0, b4 = _bundle(seed), _bundle(seed)
    for b in (b0, b4):
        b.processor.tokenizer.eos_token_id = EOS_T
    names = tta.decoder_ln_names(b4.model)
    g0, g4 = tta.LNGuard(b0.model, names), tta.LNGuard(b4.model, names)
    if perturb:
        torch.manual_seed(7)
        g4.materialize({n: (g4.theta0[n].float() + perturb * torch.randn_like(g4.theta0[n].float())).to(torch.bfloat16) for n in names})
    return b0, b4, g0, g4


def test_two_instances_independent_and_lockstep_no_trigger():
    b0, b4, g0, g4 = _pair()
    p0, p4 = dict(b0.model.named_parameters()), dict(b4.model.named_parameters())
    assert all(p0[n].data_ptr() != p4[n].data_ptr() for n in p0)
    d = cg.lockstep_detect(b0, b4, _enc(), CB, owners={"theta0": g0.current_hash(), "A4": g4.current_hash()},
                           hash_fns={"theta0": g0.current_hash, "A4": g4.current_hash}, max_new=8)
    ref = tta.forced_decode(b0, _enc(), CB, max_new_tokens=8)
    assert not d["trigger"] and d["prefix"] == ref["tokens"] and d["terminated"] == ref["terminated"] and d["fed_identical"] and d["state_locked"]
    g4.materialize({n: g4.theta0[n] + 1 for n in g4.names})
    assert g0.verify() and not g4.verify()                            # mutating one instance leaves the other
    g4.restore()


def _trigger_pair():
    for pert in (0.3, 0.6, 1.0, 2.0):
        b0, b4, g0, g4 = _pair(pert)
        d = cg.lockstep_detect(b0, b4, _enc(), CB, owners={"theta0": g0.current_hash(), "A4": g4.current_hash()},
                               hash_fns={"theta0": g0.current_hash, "A4": g4.current_hash}, max_new=8)
        if d["trigger"] and d["argmax_theta0"] != EOS_T and d["argmax_A4"] != EOS_T:
            return b0, b4, g0, g4, d
    pytest.skip("no usable disagreement on tiny model")


def test_first_disagreement_prefix_rollouts_and_owner_lock():
    b0, b4, g0, g4, d = _trigger_pair()
    k = d["k"]
    assert d["prefix"] == [p for p, q in d["argmax_pairs"][:k]] and all(p == q for p, q in d["argmax_pairs"][:k]) and len(d["argmax_pairs"]) == k + 1
    r0 = cg.rollout(b0, _enc(), CB, d["prefix"], owner_hash=g0.current_hash(), hash_fn=g0.current_hash, max_new=8)
    r4 = cg.rollout(b4, _enc(), CB, d["prefix"], owner_hash=g4.current_hash(), hash_fn=g4.current_hash, max_new=8)
    assert r0["tokens"][0] == d["argmax_theta0"] and r4["tokens"][0] == d["argmax_A4"] and 1 <= r0["H_eff"] <= 3 and r0["prefix_ok"] and r0["state_locked"]
    assert r0["terminated"] in (None, "eos", "cap")
    with pytest.raises(ValueError):
        cg.rollout(b4, _enc(), CB, d["prefix"], owner_hash=g0.current_hash(), hash_fn=g4.current_hash, max_new=8)   # stale/cross-model owner
    short = cg.rollout(b0, _enc(), CB, d["prefix"], owner_hash=g0.current_hash(), hash_fn=g0.current_hash, max_new=k + 1)
    assert short["H_eff"] <= 1 and (short["terminated"] in ("cap", "eos") or short["H_eff"] == 1)        # global budget respected


def test_g1_g2_semantics_replay_equivalence_and_shared_decision():
    b0, b4, g0, g4, d = _trigger_pair()
    k, pre = d["k"], d["prefix"]
    b0r = cg.rollout(b0, _enc(), CB, pre, owner_hash=g0.current_hash(), hash_fn=g0.current_hash, max_new=8)
    b4r = cg.rollout(b4, _enc(), CB, pre, owner_hash=g4.current_hash(), hash_fn=g4.current_hash, max_new=8)
    dec = {"k": k, "prefix": pre, "b0": b0r, "b4": b4r, "scores": {}, "winner": "theta0", "margin": 0.1}
    assert cg.g1_forced(dec) == [b0r["tokens"][0]]
    f2 = cg.g2_forced(dec, EOS_T)
    assert f2[:len(b0r["tokens"])] == b0r["tokens"]
    g1 = pd.clamp_decode(b4, _enc(), CB, site=k, forced=cg.g1_forced(dec), expected_prefix=pre, max_new_tokens=8)
    g2 = pd.clamp_decode(b4, _enc(), CB, site=k, forced=f2, expected_prefix=pre, max_new_tokens=8)
    assert g1["tokens"][k] == b0r["tokens"][0] and g2["tokens"][k:k + len(b0r["tokens"])] == b0r["tokens"]
    assert g1["trace"]["prefix_ok"] and g2["trace"]["prefix_ok"] and g2["trace"]["release_index"] == k + len(f2)
    # history-only interpretation: fresh A4 replay == feeding prompt+history incrementally under A4
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    hist = pre + b0r["tokens"]
    br = cached.Branch(b4, _enc(), CB, "m")
    lg, _, _ = br.step(CB, capture_layer=16, attention=True)
    for t in hist:
        lg, _, _ = br.step([t], capture_layer=16, attention=True)
    if len(g2["tokens"]) > len(hist):
        assert processed_argmax(lg, len(hist), [], []) == g2["tokens"][len(hist)]
    a4win = {**dec, "winner": "A4"}
    assert cg.g2_forced(a4win, EOS_T) is None and cg.g1_forced(a4win) == [b4r["tokens"][0]]
    dec["S_cons"] = {}
    assert cg.decision_hash(dec) == cg.decision_hash(dict(dec))


def test_g1_and_g2_can_differ_on_toy_fixture():
    b0, b4, g0, g4, d = _trigger_pair()
    k, pre = d["k"], d["prefix"]
    g1 = pd.clamp_decode(b4, _enc(), CB, site=k, forced=[d["argmax_theta0"]], expected_prefix=pre, max_new_tokens=8)
    alt = 25 if (len(g1["tokens"]) <= k + 1 or g1["tokens"][k + 1] != 25) else 26
    g2 = pd.clamp_decode(b4, _enc(), CB, site=k, forced=[d["argmax_theta0"], alt, alt], expected_prefix=pre, max_new_tokens=8)
    assert g2["tokens"] != g1["tokens"]


def test_outcome_rules_and_label_truth_table():
    c = C
    P = lambda z, p, m, caps=0, sev=0: an.g_pass({"zh": z, "poi": p, "mixed": m}, caps, sev, c)["pass"]
    assert P(29, 105, 143) and not P(30, 105, 143) and not P(29, 106, 143) and not P(29, 105, 144) and not P(29, 105, 143, caps=1) and not P(29, 105, 143, sev=1)
    g1, g2 = {"zh": 29, "poi": 105, "mixed": 143}, {"zh": 27, "poi": 106, "mixed": 143}
    assert an.raw_material(g1, g2, 0, c) and not an.raw_material(g1, {"zh": 28, "poi": 105, "mixed": 142}, 0, c)
    assert not an.raw_material(g1, {"zh": 27, "poi": 107, "mixed": 143}, 0, c) and not an.raw_material(g1, g2, 1, c)
    assert an.decide(False, True, True, True) == "P2_PATH2_INVALID"
    assert an.decide(True, True, True, True) == "P2_PATH2_MIXED_GUARD_PROMISING"
    assert an.decide(True, True, False, True) == "P2_PATH2_BRANCH_CONTROL_SUFFICIENT"
    assert an.decide(True, True, True, False) == "P2_PATH2_BRANCH_CONTROL_SUFFICIENT"
    assert an.decide(True, False, True, False) == "P2_PATH2_ROLLBACK_NEEDED"
    assert an.decide(True, False, False, True) == "P2_PATH2_LOCAL_GUARD_INSUFFICIENT"
    for args in ((29, 105, 143, 0, 29, 105, 141, 0), (29, 105, 143, 0, 27, 107, 143, 0), (30, 105, 143, 0, 29, 105, 143, 0)):
        lab, g1p, g2p, raw = au.own_label(True, *args, c)
        assert lab == an.decide(True, g1p, g2p, raw)


def test_reference_firewall_and_auditor_independence(monkeypatch):
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2path2/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2path2/run1")
    assert not called
    src = Path("experiments/inference_cf_p2path2_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2path2_analyze|inference_cf_p2path2\b|consensus_guard|branch_adjudication|path_decode)", src, re.M) is None
    rsrc = Path("experiments/inference_cf_p2path2.py").read_text()
    phase2 = rsrc[rsrc.index("# ---------------- Phase 2"):rsrc.index("def cmd_seal")]
    assert not any(t in phase2 for t in ('["analysis"]', '["expected"]', "an_[", "AUTO", "load_references", "audit_only", '["group"]'))
