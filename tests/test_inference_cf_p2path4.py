"""P2-PATH4-R1 runnable-implementation tests (freeze 490ec20): exact fixed100/seven token-only partitions/hashes and the
audit-only first divergences (14 triggers = A2_DELTA; 4 EOS/content, 10 content/content) recomputed from sealed tokens;
R1 dispatcher wiring (one logical detector; CONTENT_G1 -> original online_g1 bit-identical on a toy two-instance fixture;
EOS_BOUNDARY -> H=1 scorer + one selected action, original G1 never called; NO_TRIGGER -> live prefix); exact
comparator; 100-row reconstruction barrier before any controller; no G2/A4/LID; frozen PATH4 cutoffs (Z_G <= 1111,
POI <= 322 with PIER guard, mixed <= 1464, MER), six breadth conditions, LODO, label precedence (all 32 combos vs the
independent auditor); reference firewall and auditor independence."""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB as CB_T  # noqa: E402
from test_inference_cf_p2dir_protocol import _enc  # noqa: E402
from test_inference_cf_p2path2 import _trigger_pair  # noqa: E402

import experiments.inference_cf_p2path3 as path3  # noqa: E402
import experiments.inference_cf_p2path4 as run  # noqa: E402
import experiments.inference_cf_p2path4_analyze as an  # noqa: E402
import experiments.inference_cf_p2path4_audit as au  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / run.CONFIG).read_text())
P = json.loads((ROOT / run.PANEL).read_text())


def _sealed():
    B, A = {}, {}
    for r in P["rows"]:
        B[r["utterance_id"]] = run.selected(r["B0_FORCED"]["path"], r["B0_FORCED"]["selector"])
        A[r["utterance_id"]] = run.selected(r["A2"]["path"], r["A2"]["selector"])
    return B, A


def test_fixed100_seven_partitions_and_first_divergences():
    par = json.loads((ROOT / run.PARENT).read_text())
    full = [r["utterance_id"] for r in par["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in par["rows"]}
    dev = [r["utterance_id"] for r in json.loads((ROOT / run.A3_PANEL).read_text())["rows"]]
    p12 = [r["utterance_id"] for r in json.loads((ROOT / run.P2_PANEL).read_text())["rows"]]
    B, A = _sealed()
    fds = {u: run.first_divergence(B[u]["tokens"], B[u]["terminated"], A[u]["tokens"], A[u]["terminated"]) for u in full}
    assert all(fds[r["utterance_id"]] == r.get("audit_only_first_divergence") for r in P["rows"])
    delta = {u for u, f in fds.items() if f is not None}
    parts = run.partitions(full, dev, p12, delta)
    assert {k: len(v) for k, v in parts.items()} == C["counts"] and all(run.membership_hash(k, parts[k], dlg) == C["partition_sha256"][k] for k in run.PARTS)
    assert parts["A2_DELTA"] == C["G1"]["expected_trigger_ids"]
    bnd = [u for u in parts["A2_DELTA"] if fds[u]["EOS_first_blocker"]]
    assert bnd == [r["utterance_id"] for r in C["known_pre_run_block"]["rows"]] and len(parts["A2_DELTA"]) - len(bnd) == 10
    assert all(fds[u]["theta0_token"] == run.EOS and fds[u]["theta0_content_branch"] == [] for u in bnd)
    with pytest.raises(ValueError):
        run.partitions(full, dev, p12, delta | {"ZH-CN_NOT_IN_FIXED100"})
    assert P["status"] == "BLOCKED_PRE_RUN_EOS_FIRST" and C["original_blocked_provenance"]["commit"].startswith("29661e9")


def _toy(monkeypatch):
    for mod in (run, path3):
        monkeypatch.setattr(mod, "CB", CB_T)
        monkeypatch.setattr(mod, "MAX_NEW", 8)


def _counters():
    return {"logical_guard_events": 0, "detector_runs": 0, "detector_replays": 0, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0,
            "boundary_score_paths": 0, "boundary_executions": 0}


def test_content_dispatch_bit_identical_to_original_online_g1(monkeypatch):
    _toy(monkeypatch)
    b0, b2, g0, g2, d = _trigger_pair()
    eos = b0.processor.tokenizer.eos_token_id
    if d["argmax_theta0"] == eos or d["argmax_A4"] == eos:
        pytest.skip("toy first disagreement involves EOS")
    cnt = _counters()
    rec = run.online_g1_r1(b0, b2, g0, g2, _enc(), "toy", cnt, eos)
    orig = path3.online_g1(b0, b2, g0, g2, _enc(), "toy", {"detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0}, eos)
    assert rec["mode"] == "CONTENT_G1" and rec["fails"] == [] and run.comparator_mismatch(rec, orig) == []
    assert cnt["logical_guard_events"] == 1 and cnt["detector_runs"] == 1 and cnt["detector_replays"] == 1 and cnt["triggers"] == 1
    assert cnt["rollouts"] == 2 and cnt["score_paths"] == 4 and cnt["G1_decodes"] == 1 and cnt["boundary_executions"] == 0
    tampered = json.loads(json.dumps(orig))
    tampered["decision"]["scores"]["A4_b0"]["logprobs"][0] += 1e-9
    assert "decision" in run.comparator_mismatch(rec, tampered)


def _fake_bundle(vocab=10, eos=9):
    tok = SimpleNamespace(eos_token_id=eos, all_special_ids=[eos])
    model = SimpleNamespace(generation_config=SimpleNamespace(suppress_tokens=[], begin_suppress_tokens=[]), config=SimpleNamespace(vocab_size=vocab))
    return SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=tok))


def test_eos_boundary_dispatch_h1_one_action_no_original_g1(monkeypatch):
    import csasr.inference_cf.consensus_guard as cg
    import csasr.inference_cf.eos_boundary as eb
    det = {"trigger": True, "k": 2, "prefix": [3, 4], "argmax_theta0": 9, "argmax_A4": 5, "argmax_pairs": [[3, 3], [4, 4], [9, 5]],
           "terminated": None, "fed_identical": True, "positions_ok": True, "state_locked": True}
    calls = {"detect": 0, "score": 0, "exec": 0}
    monkeypatch.setattr(cg, "lockstep_detect", lambda *a, **k: (calls.__setitem__("detect", calls["detect"] + 1), dict(det))[1])

    def fake_score(b0, b2, enc, prompt, prefix, c0, c2, *, owners, hash_fns):
        calls["score"] += 1
        return {"contract_revision": eb.REVISION, "mode": "EOS_BOUNDARY", "H": 1, "c0": c0, "c2": c2, "S_theta0": {"B": -0.1, "ALT": -3.0},
                "S_A2": {"B": -0.5, "ALT": -1.0}, "S_cons": {"B": -0.3, "ALT": -2.0}, "margin": 1.7, "winner": "theta0", "selected_token": c0,
                "k": len(prefix), "prefix": list(prefix), "paths": {}}

    def fake_exec(b2, enc, prompt, decision, *, owner_hash, hash_fn, max_new_tokens=200):
        calls["exec"] += 1
        assert decision["H"] == 1 and max_new_tokens == run.MAX_NEW
        return {"tokens": list(decision["prefix"]), "terminated": "eos", "state_locked": True,
                "trace": {"prefix_ok": True, "suppression_ok": True, "positions_ok": True, "forced": [{"token": decision["selected_token"]}]}}
    monkeypatch.setattr(eb, "score_boundary", fake_score)
    monkeypatch.setattr(eb, "execute_boundary", fake_exec)
    monkeypatch.setattr(path3, "online_g1", lambda *a, **k: (_ for _ in ()).throw(AssertionError("original G1 must not run on EOS_BOUNDARY")))
    g = SimpleNamespace(current_hash=lambda: "h")
    cnt = _counters()
    rec = run.online_g1_r1(_fake_bundle(), _fake_bundle(), g, g, None, "toy", cnt, 9)
    assert rec["mode"] == "EOS_BOUNDARY" and rec["fails"] == [] and calls == {"detect": 1, "score": 1, "exec": 1}
    assert rec["G1"]["tokens"] == [3, 4] and rec["G1"]["terminated"] == "eos" and rec["G1"]["forced"] == [9]
    assert rec["decision"]["decision_hash"] == run.boundary_hash(rec["decision"])
    assert cnt["boundary_score_paths"] == 2 and cnt["boundary_executions"] == 1 and cnt["rollouts"] == 0 and cnt["detector_replays"] == 0
    det.update(argmax_theta0=5, argmax_A4=9)                                  # reverse orientation also dispatches to the boundary
    assert run.online_g1_r1(_fake_bundle(), _fake_bundle(), g, g, None, "toy", _counters(), 9)["mode"] == "EOS_BOUNDARY"
    det.update(trigger=False, argmax_theta0=None, argmax_A4=None, prefix=[3, 4, 5], terminated="eos")
    nt = run.online_g1_r1(_fake_bundle(), _fake_bundle(), g, g, None, "toy", _counters(), 9)
    assert nt["mode"] == "NO_TRIGGER" and nt["decision"] is None and nt["G1"] == {"tokens": [3, 4, 5], "terminated": "eos"}
    det.update(trigger=True, argmax_theta0=8, argmax_A4=9, k=2, prefix=[3, 4])
    bad = _fake_bundle()
    bad.model.generation_config.suppress_tokens = [8]                          # suppressed proposal -> invalid, no fallback
    with pytest.raises(ValueError):
        run.online_g1_r1(bad, bad, g, g, None, "toy", _counters(), 9)


def test_runner_structure_barrier_no_g2_no_a4():
    src = (ROOT / "experiments/inference_cf_p2path4.py").read_text()
    code = src[src.index("def cmd_run"):src.index("def cmd_seal")]
    i_bar = code.index("A2 reconstruction barrier failed (G1 phase not entered)")
    assert code.index("Phase 1") < i_bar < code.index("online_g1_r1(") < code.index("path3.online_g1(")
    assert "g2_forced" not in src and "adapt_a4" not in src and "detect_language" not in src and "load_references" not in src
    ctrl = src[src.index("def online_g1_r1"):src.index("def _norm")]
    assert not any(t in ctrl for t in au.AUDIT_TOKENS) and ctrl.count("lockstep_detect(") == 1
    assert run.sha_file(ROOT / "experiments/inference_cf_p2path3.py") == C["source_sha256"]["experiments/inference_cf_p2path3.py"]
    assert all(run.sha_file(ROOT / p) == h for p, h in C["amendment_source_sha256"].items())


def test_frozen_cutoffs_and_benefit_guards():
    assert an.rescue(1107, 1116, 1111)["pass"] and not an.rescue(1107, 1116, 1112)["pass"] and an.rescue(1107, 1116, 1111)["required_R_ZH"] == 5
    assert an.rescue(10, 9, 8)["required_R_ZH"] == 1 and not an.rescue(10, 9, 9)["pass"]
    b = lambda poi, mix=1464, dm=0.0, pier_a2=316 / 696: an.benefit(345, 316, poi, poi / 696, pier_a2, 0.25 + dm, 0.25, mix, 1464, C)["BENEFIT_RETENTION_PASS"]
    assert b(322) and not b(323) and not b(324) and b(316) and not b(345) and not b(346)
    assert not b(320, mix=1465) and b(320, mix=1464) and b(320, dm=0.005) and not b(320, dm=0.0051)
    assert an.benefit(345, 316, 322, 0, 0, 0, 0, 0, 0, C)["retention"] == 23 / 29


def test_breadth_six_conditions_and_lodo():
    nov = {"n1", "n2", "n3", "n4"}
    D = {"n1": "a", "n2": "b", "n3": "c", "n4": "d", "d1": "e", "d2": "f", "d3": "g"}
    ok = an.breadth({"n1": 2, "n2": 1, "n3": 1, "n4": 0, "d1": 0, "d2": 0, "d3": 0}, D, nov, C)
    assert ok["BREADTH_PASS"] and ok["C_utt"] == 0.5 and ok["C_dlg"] == 0.5
    dev_only = an.breadth({"n1": 0, "n2": 0, "n3": 0, "n4": 0, "d1": 2, "d2": 1, "d3": 1}, D, nov, C)
    assert not dev_only["BREADTH_PASS"] and dev_only["criteria"]["R_net_NOVEL76"] is False and dev_only["criteria"]["rescue_utterances_NOVEL76"] is False
    harm = an.breadth({"n1": 1, "n2": 0, "n3": 0, "n4": -1, "d1": 2, "d2": 1, "d3": 0}, D, nov, C)
    assert not harm["criteria"]["R_net_NOVEL76"] and harm["criteria"]["rescue_utterances_NOVEL76"]
    single = an.breadth({"n1": 20, "n2": 0, "n3": 0, "n4": 0, "d1": 0, "d2": 0, "d3": 0}, D, nov, C)
    assert not single["BREADTH_PASS"] and single["C_utt"] == 1.0
    zero = an.breadth({u: 0 for u in D}, D, nov, C)
    assert zero["C_utt"] == zero["C_dlg"] == 1.0 and not zero["BREADTH_PASS"]
    same = an.breadth({"n1": 2, "n2": 0, "n3": 0, "n4": 0, "d1": 0, "d2": 0, "d3": 0, "x1": 2, "x2": 2}, {**D, "x1": "a", "x2": "a"}, nov, C)
    assert same["N_rescue_utt"] == 3 and same["N_rescue_dlg"] == 1 and not same["BREADTH_PASS"]
    lo = an.lodo({"n1": 2, "n2": 1}, D, ["a", "b", "c"])
    assert lo["values"] == {"a": 1, "b": 2, "c": 3} and lo["count_positive"] == 3 and lo["min"] == 1 and lo["max"] == 3
    empty = an.lodo({"n1": 2}, D, [])
    assert empty["dialogues"] == 0 and empty["count_positive"] == 0 and empty["min"] is None and empty["proportion_positive"] is None


def test_safety_rowwise_caps_and_none():
    p = {"mer_increase": 0.0, "zh_cer_increase": 0.0, "zh_retention": None, "en_retention": 1.0, "outside_harm_rate": 0.0,
         "poi_corruption_rate": 0.0, "added_caps": 1, "new_severe_truncations": 0}
    s = an.safety(p, C)
    assert s["pass"] and s["not_assessable"] == ["matrix_ZH_retention"]
    assert not an.safety({**p, "added_caps": 2}, C)["pass"] and not an.safety({**p, "new_severe_truncations": 1}, C)["pass"]
    assert an.safety({**p, "mer_increase": 0.01}, C)["pass"] and not an.safety({**p, "mer_increase": 0.0100001}, C)["pass"]


def test_label_precedence_all_combinations_vs_auditor():
    for v, s, r, t, b in itertools.product((False, True), repeat=5):
        lab = an.decide(v, s, r, t, b)
        assert lab == au.own_label(v, s, r, t, b) and lab in an.LABELS
        exp = ("P2_PATH4_INVALID" if not v else "P2_PATH4_SEQUENCE_DAMAGE" if not s else "P2_PATH4_SAFE_NO_ADDED_VALUE" if not r
               else "P2_PATH4_OVERCONSERVATIVE" if not t else "P2_PATH4_SIGNAL_CONCENTRATED" if not b else "P2_PATH4_GUARD_TRANSFER_SUPPORTED")
        assert lab == exp


def test_reference_firewall_and_auditor_independence(monkeypatch):
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2path4/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2path4/run1")
    assert not called
    asrc = (ROOT / "experiments/inference_cf_p2path4_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2path4_analyze|inference_cf_p2path4\b|consensus_guard|branch_adjudication|path_decode|eos_boundary)",
                     asrc, re.M) is None
    fsrc = (ROOT / "experiments/inference_cf_p2path4_analyze.py").read_text()
    assert "load_references" not in fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
