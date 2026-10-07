"""P2-PATH1 focused tests (freeze 5a159f2): exact U/C sites + fingerprints (primary and independent auditor);
state-owned fresh-cache paths (stale cross-state fixture rejected, state locked for the whole path); factorial semantics
(T0B/T0A/A4B/A4A with forced token once, release at k+1, continuation under the same state); H=3 content scoring at
absolute indices (begin-suppression only at index 0, EOS never scored); 0.5/0.5 consensus and tie->B; inherited
INDUCE_1 reuse; I/Delta/tau materiality and mechanism precedence; CONSENSUS_REJECTS_AUTO; reference firewall."""
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

import csasr.inference_cf.branch_adjudication as ba  # noqa: E402
import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import csasr.inference_cf.path_decode as pd  # noqa: E402
import experiments.inference_cf_p2path1 as run  # noqa: E402
import experiments.inference_cf_p2path1_analyze as an  # noqa: E402
import experiments.inference_cf_p2path1_audit as au  # noqa: E402

C = json.loads(Path(run.CONFIG).read_text())


def test_sites_primary_and_independent():
    S = json.loads(Path(run.SITES).read_text())
    P0 = json.loads(Path(run.PARENT).read_text())
    plan4 = json.loads(Path(run.A4_PLAN).read_text())
    for s, p0 in zip(S["rows"], P0["rows"]):
        a4 = json.loads(Path(p0["sources"]["A4"]["path"]).read_text())["A4"]
        assert run.verify_row(s, p0, plan4["rows"][p0["parent_A3_index"]], a4), s["utterance_id"]
    mine, ok = au.own_sites()
    assert ok and sum(x["U"] for x in mine) == 7 and sum(x["U"] and x["dBA"] > 0 for x in mine) == 6 and all(x["k"] == 0 for x in mine if x["U"])
    assert sum(x["dBA"] for x in mine if x["U"]) == 48


def _b(supp=()):
    b = _bundle(suppress=list(supp))
    b.processor.tokenizer.eos_token_id = EOS_T
    return b


def _states(b):
    names = tta.decoder_ln_names(b.model)
    g = tta.LNGuard(b.model, names)
    eff = {n: (g.theta0[n].float() + 0.05 * torch.randn_like(g.theta0[n].float())).to(torch.bfloat16) for n in names}
    return g, eff


def test_state_owned_paths_and_stale_cache_rejected():
    b = _b()
    g, eff = _states(b)
    own = {"state": "theta0", "state_hash": g.theta0_hash, "row": "r", "condition": "T0B"}
    d = ba.owned_clamp(b, _enc(), CB, site=0, forced=[20], expected_prefix=[], owner=own, state_hash_fn=g.current_hash, max_new_tokens=6)
    assert d["state_locked"] and d["trace"]["release_index"] == 1 and d["tokens"][0] == 20
    with pytest.raises(ValueError):                                   # declared theta0 owner but A4 resident -> rejected
        g.materialize(eff)
        try:
            ba.owned_clamp(b, _enc(), CB, site=0, forced=[20], expected_prefix=[], owner=own, state_hash_fn=g.current_hash, max_new_tokens=6)
        finally:
            g.restore()
    assert g.verify()


def test_factorial_semantics_continuation_under_own_state():
    b = _b()
    g, eff = _states(b)
    t0_free = tta.forced_decode(b, _enc(), CB, max_new_tokens=6)
    cB = t0_free["tokens"][0]
    cA = 21 if cB != 21 else 22
    t0b = ba.owned_clamp(b, _enc(), CB, site=0, forced=[cB], expected_prefix=[], owner={"state_hash": g.theta0_hash}, state_hash_fn=g.current_hash, max_new_tokens=6)
    assert t0b["tokens"] == t0_free["tokens"]                          # T0B == theta0 greedy
    t0a = ba.owned_clamp(b, _enc(), CB, site=0, forced=[cA], expected_prefix=[], owner={"state_hash": g.theta0_hash}, state_hash_fn=g.current_hash, max_new_tokens=6)
    g.materialize(eff)
    try:
        h = g.current_hash()
        a4a = ba.owned_clamp(b, _enc(), CB, site=0, forced=[cA], expected_prefix=[], owner={"state_hash": h}, state_hash_fn=g.current_hash, max_new_tokens=6)
        a4_ref = pd.clamp_decode(b, _enc(), CB, site=0, forced=[cA], max_new_tokens=6)
    finally:
        g.restore()
    assert a4a["tokens"] == a4_ref["tokens"] and a4a["state_locked"] and t0a["tokens"][0] == cA == a4a["tokens"][0]
    assert g.verify()


def test_branch_score_absolute_indices_and_eos_rule():
    b = _b(supp=[33])
    b.model.generation_config.begin_suppress_tokens = [30]
    g, _ = _states(b)
    own = {"state_hash": g.theta0_hash}
    s0 = ba.score_branch(b, _enc(), CB, [], [21, 22, 23], owner=own, state_hash_fn=g.current_hash)
    import experiments.inference_cf_cached as cached
    br = cached.Branch(b, _enc(), CB, "x")
    lg, _, _ = br.step(CB, capture_layer=16, attention=True)
    ref0 = float(ba.processed_log_probs(lg, 0, [33], [30])[21])
    assert s0["logprobs"][0] == pytest.approx(ref0, abs=0) and s0["H_eff"] == 3 and s0["score"] == pytest.approx(sum(s0["logprobs"]) / 3)
    assert s0["state_locked"] and s0["fed_ok"] and s0["positions_ok"]
    with pytest.raises(ValueError):
        ba.score_branch(b, _enc(), CB, [], [30], owner=own, state_hash_fn=g.current_hash)        # begin-suppressed at absolute 0
    pre = tta.forced_decode(b, _enc(), CB, max_new_tokens=2)["tokens"][:1]
    ok = ba.score_branch(b, _enc(), CB, pre, [30], owner=own, state_hash_fn=g.current_hash)       # absolute index 1: allowed
    assert ok["prefix_ok"] and len(ok["logprobs"]) == 1
    with pytest.raises(ValueError):
        ba.score_branch(b, _enc(), CB, [], [21, EOS_T], owner=own, state_hash_fn=g.current_hash)  # EOS never scored


def test_consensus_tie_and_criterion():
    c = ba.consensus({"B": -1.0, "ALT": -2.0}, {"B": -3.0, "ALT": -1.0})
    assert c["S_cons"] == {"B": -2.0, "ALT": -1.5} and c["choice"] == "ALT" and not c["strict_B_win"]
    t = ba.consensus({"B": -1.0, "ALT": -1.0 + 1e-13}, {"B": -1.0, "ALT": -1.0})
    assert t["choice"] == "B" and not t["strict_B_win"]
    rows = lambda ch, sw, dl: [{"choice": a, "strict_B_win": s, "dialogue_id": d} for a, s, d in zip(ch, sw, dl)]
    assert an.consensus_rejects_auto(rows("BBBBBA", [1, 1, 1, 1, 0, 0], "abcdee"), C)["pass"]
    assert not an.consensus_rejects_auto(rows("BBBBAA", [1, 1, 1, 1, 0, 0], "abcdee"), C)["pass"]
    assert not an.consensus_rejects_auto(rows("BBBBBA", [1, 1, 1, 0, 0, 0], "abcdee"), C)["pass"]
    assert not an.consensus_rejects_auto(rows("BBBBBA", [1, 1, 1, 1, 0, 0], "aabbee"), C)["pass"]


def _row(u, dlg, d0, dt0, dbt0, da4, dba4):
    return {"utterance_id": u, "dialogue_id": dlg, "d_BA": d0, "eligible": d0 > 0, "theta0": {"d_A": dt0, "d_B": dbt0, "I": 1 - dt0 / d0},
            "A4": {"d_A": da4, "d_B": dba4, "I": 1 - da4 / d0}, "Delta": (dt0 - da4) / d0, "tau": max(0.25, 1 / d0)}


def test_endpoints_and_mechanism_precedence():
    e = an.endpoints(1, [9, 1, 2, 50257], [8, 3, 4, 50257], [8, 3, 4, 50257], [8, 1, 4, 50257])
    assert e["d_BA"] == 2 and e["theta0"]["I"] == 1.0 and e["A4"]["I"] == 0.5 and e["tau"] == 0.5 and e["Delta"] == -0.5
    same = [_row(f"u{i}", d, 10, 2, 8, 2, 8) for i, d in enumerate("abcdef")]
    assert an.mechanism(same, C, True, True, True)["label"] == "P2_PATH1_PREFIX_DOMINANT"
    assert an.mechanism(same, C, True, False, True)["label"] == "P2_PATH1_INVALID"
    adapted = [_row(f"u{i}", d, 10, 10, 0, 1, 9) for i, d in enumerate("abcdef")]
    assert an.mechanism(adapted, C, False, True, True)["label"] == "P2_PATH1_ADAPTED_STATE_DOMINANT"
    amp = [_row(f"u{i}", d, 10, 4, 6, 0, 10) for i, d in enumerate("abcdef")]          # T0A pass but A4 amplifies everywhere
    assert an.mechanism(amp, C, True, True, True)["label"] == "P2_PATH1_MIXED"
    het = [_row("u0", "a", 10, 2, 8, 2, 8), _row("u1", "b", 10, 2, 8, 2, 8), _row("u2", "c", 10, 10, 0, 1, 9), _row("u3", "d", 10, 10, 0, 1, 9),
           _row("u4", "e", 10, 10, 0, 10, 0), _row("u5", "f", 10, 10, 0, 10, 0)]
    m = an.mechanism(het, C, False, True, True)
    assert m["predicates"]["MIXED_c_heterogeneous"] and m["label"] == "P2_PATH1_MIXED"
    none = [_row(f"u{i}", d, 10, 10, 0, 9, 1) for i, d in enumerate("abcdef")]
    assert an.mechanism(none, C, False, True, True)["label"] == "P2_PATH1_NO_CLEAR_MECHANISM"


def test_reference_firewall_and_auditor_independence(monkeypatch):
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2path1/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2path1/run1")
    assert not called
    src = Path("experiments/inference_cf_p2path1_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2path1_analyze|p2path0_analyze|branch_adjudication|path_decode|inference_cf_p2path[01]\b)", src, re.M) is None
    assert "load_references" not in Path("experiments/inference_cf_p2path1.py").read_text()


def test_auditor_owner_check_handles_path_and_score_records():
    """Primary-audit attempt 1 crashed (KeyError) on factorial path records, which carry owner+state_locked only."""
    assert au.owner_ok({"owner": {"state_hash": "h"}, "state_locked": True}, "h")
    assert not au.owner_ok({"owner": {"state_hash": "h"}, "state_locked": False}, "h")
    assert not au.owner_ok({"owner": {"state_hash": "x"}, "state_locked": True}, "h")
    assert au.owner_ok({"owner": {"state_hash": "h"}, "state_locked": True, "state_hash_before": "h", "state_hash_after": "h"}, "h")
    assert not au.owner_ok({"owner": {"state_hash": "h"}, "state_locked": True, "state_hash_before": "h", "state_hash_after": "z"}, "h")
