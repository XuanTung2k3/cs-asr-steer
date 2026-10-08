"""P2-PATH5 focused tests (freeze d1d057d): FULL300/FIXED100/NEW200 relation and hashes; A2 inheritance pins; G1A dispatch
(content -> original online_g1 once with replayed-detection check; theta0-EOS and A2-EOS -> abstain == ordinary A2, no
scorer/rollout/decoder path, controller disabled; both-EOS/no-trigger -> A2; invalid proposal rejected; one event); no
row IDs/audit tokens in the controller; fixed100 barrier comparator on sealed PATH4 rows; opportunity gate, 0.90 retention
(+ I_A2<=0 rule), PIER/MER guards, breadth/concentration, LODO, all 64 label combinations vs the independent auditor;
reference firewall (incl. closed gate) and auditor independence."""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.inference_cf_p2path3 as path3  # noqa: E402
import experiments.inference_cf_p2path5 as run  # noqa: E402
import experiments.inference_cf_p2path5_analyze as an  # noqa: E402
import experiments.inference_cf_p2path5_audit as au  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / run.CONFIG).read_text())
P = json.loads((ROOT / run.PANEL).read_text())


def test_population_relation_and_hashes():
    full = [r["utterance_id"] for r in json.loads((ROOT / run.FULL).read_text())["rows"]]
    fx = json.loads((ROOT / run.FIXED).read_text())
    fixed = [r["utterance_id"] for r in fx["rows"]]
    assert fx["parent_panel"] == run.FULL and fx["parent_panel_sha256"] == run.sha_file(ROOT / run.FULL)
    parts = run.partitions(full, fixed)
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["rows"]}
    assert {k: len(v) for k, v in parts.items()} == C["counts"] == {"FULL300": 300, "FIXED100": 100, "NEW200": 200}
    assert all(run.membership_hash(k, parts[k], dlg) == C["partition_sha256"][k] for k in run.PARTS)
    assert set(parts["NEW200"]) == set(full) - set(fixed) and not set(parts["NEW200"]) & set(fixed)
    assert all(len({dlg[u] for u in v}) == 20 for v in parts.values())
    assert run.row_fp(parts["FIXED100"] + parts["NEW200"]) == C["execution_order_sha256"]
    assert C["fixed100_equivalence"]["PATH3_PATH4_FULL100_membership_sha256"] == run.membership_hash("FULL100", fixed, dlg)
    with pytest.raises(ValueError):
        run.partitions(full, fixed + ["ZH-CN_NOT_IN_FULL300"])
    der = {r["utterance_id"] for r in json.loads((ROOT / run.DERIVED).read_text())["rows"]}
    assert der == set(fixed) and all("fixed100" in r for r in P["rows"] if r["partition"] == "FIXED100")


def test_a2_inheritance_and_pins():
    c4 = json.loads((ROOT / "configs/inference_cf/p2_path4.json").read_text())
    assert C["A2_inherited"] == c4["A2_inherited"] and C["A2_objective"] == c4["A2_objective"] and C["suppression"] == c4["suppression"]
    from csasr.inference_cf.episodic_tta import OPTIM, STEPS
    assert STEPS == 2 and OPTIM["lr"] == 1e-3 and OPTIM["weight_decay"] == 0
    assert all(run.sha_file(ROOT / p) == h for p, h in C["source_sha256"].items())
    assert C["G1A"]["H"] == 3 and C["G1A"]["weights"] == [0.5, 0.5] and C["G1A"]["tie_abs"] == 1e-12 and C["G1A"]["max_decision_events"] == 1


def _bundle(vocab=10, eos=9, suppress=()):
    tok = SimpleNamespace(eos_token_id=eos, all_special_ids=[eos])
    model = SimpleNamespace(generation_config=SimpleNamespace(suppress_tokens=list(suppress), begin_suppress_tokens=[]), config=SimpleNamespace(vocab_size=vocab))
    return SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=tok))


def _counters():
    return {"logical_guard_events": 0, "detector_runs": 0, "detector_replays": 0, "triggers": 0, "content_events": 0, "eos_abstentions": 0,
            "rollouts": 0, "score_paths": 0, "G1_decodes": 0}


@pytest.fixture
def patched(monkeypatch):
    import csasr.inference_cf.consensus_guard as cg
    import csasr.inference_cf.eos_boundary as eb
    state = {"det": None, "orig_calls": 0, "detect_calls": 0}

    def fake_detect(*a, **k):
        state["detect_calls"] += 1
        return json.loads(json.dumps(state["det"]))

    def fake_orig(b0, b2, g0, g2, enc, row, inner, eos):
        state["orig_calls"] += 1
        inner.update(detector_runs=1, triggers=1, rollouts=2, score_paths=4, G1_decodes=1)
        det = json.loads(json.dumps(state["det"]))
        win = state.get("winner", "theta0")
        tok0 = det["argmax_theta0"] if win == "theta0" else det["argmax_A4"]
        toks = det["prefix"] + [tok0] + ([7] if win == "theta0" else [6])
        return {"detector": det, "decision": {"winner": win, "k": det["k"]}, "score_paths": {}, "G1": {"tokens": toks, "terminated": "eos"}, "fails": []}
    monkeypatch.setattr(cg, "lockstep_detect", fake_detect)
    monkeypatch.setattr(path3, "online_g1", fake_orig)
    monkeypatch.setattr(eb, "score_boundary", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no boundary scoring in G1A")))
    monkeypatch.setattr(eb, "execute_boundary", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no boundary execution in G1A")))
    return state


G = SimpleNamespace(current_hash=lambda: "h")


def _det(trigger, c0=None, c2=None, prefix=(3, 4), term=None):
    return {"trigger": trigger, "k": len(prefix) if trigger else None, "prefix": list(prefix), "argmax_theta0": c0, "argmax_A4": c2,
            "argmax_pairs": [], "terminated": term, "fed_identical": True, "positions_ok": True, "state_locked": True}


def test_g1a_content_dispatch_original_g1_once(patched):
    patched["det"] = _det(True, 5, 6)
    a2 = {"tokens": [3, 4, 6, 6], "terminated": "eos"}
    cnt = _counters()
    rec = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", cnt, 9, a2)
    assert rec["mode"] == "CONTENT_G1" and rec["fails"] == [] and patched["orig_calls"] == 1 and patched["detect_calls"] == 1
    assert rec["G1"]["tokens"] == [3, 4, 5, 7] and cnt["content_events"] == 1 and cnt["detector_replays"] == 1 and cnt["eos_abstentions"] == 0
    assert cnt["rollouts"] == 2 and cnt["score_paths"] == 4 and cnt["logical_guard_events"] == 1
    patched["winner"] = "A4"
    rec2 = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", _counters(), 9, {"tokens": [3, 4, 6, 6], "terminated": "eos"})
    assert rec2["fails"] == []                                                # A2 winner equals ordinary A2
    rec3 = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", _counters(), 9, {"tokens": [3, 4, 6, 1], "terminated": "eos"})
    assert "A2 winner output != ordinary A2" in rec3["fails"]


@pytest.mark.parametrize("c0,c2,orient", [(9, 5, "theta0_EOS"), (5, 9, "A2_EOS")])
def test_g1a_eos_boundary_abstains_to_a2(patched, c0, c2, orient):
    patched["det"] = _det(True, c0, c2)
    a2 = {"tokens": [3, 4, 5, 8], "terminated": "eos"} if c2 != 9 else {"tokens": [3, 4], "terminated": "eos"}
    cnt = _counters()
    rec = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", cnt, 9, a2)
    assert rec["mode"] == "EOS_ABSTAIN" and rec["decision"] is None and rec["G1"] == {"tokens": a2["tokens"], "terminated": a2["terminated"]}
    assert rec["abstain"]["orientation"] == orient and patched["orig_calls"] == 0 and patched["detect_calls"] == 1
    assert cnt["eos_abstentions"] == 1 and cnt["rollouts"] == cnt["score_paths"] == cnt["G1_decodes"] == cnt["content_events"] == 0
    assert "score_paths" not in rec and rec["fails"] == []


def test_g1a_no_trigger_both_eos_and_invalid(patched):
    patched["det"] = _det(False, prefix=(3, 4, 5), term="eos")
    rec = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", _counters(), 9, {"tokens": [3, 4, 5], "terminated": "eos"})
    assert rec["mode"] == "NO_TRIGGER" and rec["fails"] == [] and patched["orig_calls"] == 0
    bad = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", _counters(), 9, {"tokens": [3, 4, 5, 1], "terminated": "eos"})
    assert "no-trigger output != ordinary A2" in bad["fails"]
    patched["det"] = _det(True, 8, 6)
    with pytest.raises(ValueError):                                           # suppressed live proposal: invalid, no fallback
        run.online_g1a(_bundle(suppress=[8]), _bundle(suppress=[8]), G, G, None, "r", _counters(), 9, {"tokens": [3, 4, 6], "terminated": "eos"})
    patched["det"] = _det(True, 5, 6)
    orig = path3.online_g1

    def drift(*a, **k):
        r = orig(*a, **k)
        r["detector"]["k"] = 99
        return r
    import experiments.inference_cf_p2path3 as p3
    p3.online_g1 = drift
    try:
        r = run.online_g1a(_bundle(), _bundle(), G, G, None, "r", _counters(), 9, {"tokens": [3, 4, 6, 6], "terminated": "eos"})
        assert "replayed detection differs from the logical detection" in r["fails"]
    finally:
        p3.online_g1 = orig


def test_controller_source_has_no_row_ids_or_audit_inputs():
    src = (ROOT / "experiments/inference_cf_p2path5.py").read_text()
    ctrl = src[src.index("def online_g1a"):src.index("def _norm")]
    assert not any(t in ctrl for t in au.AUDIT_TOKENS) and ctrl.count("lockstep_detect(") == 1 and ctrl.count("path3.online_g1(") == 1
    assert "score_boundary" not in ctrl and "execute_boundary" not in ctrl and not re.search(r"ZH-CN_U\d", src)
    code = src[src.index("def cmd_run"):src.index("def cmd_seal")]
    assert code.index("FIXED100 A2 reconstruction barrier failed") < code.index("FIXED100 G1A barrier failed") < code.index("enumerate(order[nfix:]")
    assert "g2_forced" not in src and "adapt_a4" not in src and "detect_language" not in src and "load_references" not in src


def test_fixed100_barrier_comparator_on_sealed_rows():
    der = {r["utterance_id"]: r for r in json.loads((ROOT / run.DERIVED).read_text())["rows"]}
    for i in (9, 15):                                                         # one content row, one EOS-boundary row
        p4 = json.loads((ROOT / f"results/inference_cf/p2path4/run1/rows/{i:03d}.json").read_text())
        d = der[p4["identity"]]
        row = {"mode": "CONTENT_G1" if p4["mode"] == "CONTENT_G1" else "EOS_ABSTAIN", "detector": p4["detector"], "A2_free": p4["A2_free"],
               "decision": p4["decision"] if p4["mode"] == "CONTENT_G1" else None, "score_paths": p4.get("score_paths"),
               "G1": dict(p4["G1"]) if p4["mode"] == "CONTENT_G1" else {"tokens": p4["A2_free"]["tokens"], "terminated": p4["A2_free"]["terminated"]}}
        assert run.barrier_mismatches(row, p4, d["G1A"]) == []
        bad = json.loads(json.dumps(row))
        bad["G1"]["tokens"] = bad["G1"]["tokens"][:-1]
        assert "G1A != derived target" in run.barrier_mismatches(bad, p4, d["G1A"])
    p4 = json.loads((ROOT / "results/inference_cf/p2path4/run1/rows/009.json").read_text())
    row = {"mode": "CONTENT_G1", "detector": p4["detector"], "decision": json.loads(json.dumps(p4["decision"])), "score_paths": p4["score_paths"],
           "G1": p4["G1"], "A2_free": p4["A2_free"]}
    row["decision"]["margin"] += 1e-15
    assert "decision" in run.barrier_mismatches(row, p4, der[p4["identity"]]["G1A"])


def test_opportunity_gate_retention_breadth_lodo():
    assert an.opportunity_gate(3, 3, C)["open"] and not an.opportunity_gate(2, 2, C)["open"] and not an.opportunity_gate(5, 2, C)["open"]
    assert not an.opportunity_gate(2, 3, C)["open"] and an.opportunity_gate(2, 2, C)["result"] == "OPPORTUNITY_SPARSE"
    R = lambda poi_g, poi_a2=80, dp=0.0, dm=0.0: an.retention(100, poi_a2, poi_g, 0.5 + dp, 0.5, 0.3 + dm, 0.3, C)["pass"]
    assert R(82) and not R(83) and R(80) and R(70)
    assert R(101, poi_a2=101) and not R(102, poi_a2=101) and an.retention(100, 101, 101, 0, 0, 0, 0, C)["retention"] is None
    assert R(80, dp=0.005) and not R(80, dp=0.00501) and R(80, dm=0.005) and not R(80, dm=0.00501)
    assert an.retention(100, 80, 84, 0.5, 0.5, 0.3, 0.3, C)["old_0.75_rule_descriptive"] is True
    D = {f"u{i}": f"d{i % 5}" for i in range(10)}
    ok = an.breadth({"u0": 2, "u1": 1, "u2": 1, "u3": 0}, D, C)
    assert ok["pass"] and ok["C_utt"] == 0.5 and ok["C_dlg"] == 0.5
    assert not an.breadth({"u0": 2, "u5": 2, "u1": 0}, D, C)["pass"]                      # 2 dialogues only
    assert not an.breadth({"u0": 3, "u1": 1, "u2": 1}, D, C)["pass"]                      # C_utt 0.6
    assert an.breadth({"u0": 2, "u5": 1, "u1": 1, "u2": 1}, D, C)["C_dlg"] == 0.6 and an.breadth({"u0": 2, "u5": 1, "u1": 1, "u2": 1}, D, C)["pass"]
    z = an.breadth({"u0": 0, "u1": -1}, D, C)
    assert z["C_utt"] == z["C_dlg"] == 1.0 and z["R_net"] == -1 and not z["pass"]
    lo = an.lodo({"u0": 2, "u1": -1}, D)
    assert lo["dialogues"] == 2 and lo["values"] == {"d0": -1, "d1": 2} and lo["count_positive"] == 1


def test_label_precedence_all_64_vs_auditor():
    for v, g, s, rn, t, b in itertools.product((False, True), repeat=6):
        lab = an.decide(v, g, s, 1 if rn else 0, t, b)
        assert lab == au.own_label(v, g, s, 1 if rn else 0, t, b) and lab in an.LABELS
        exp = ("P2_PATH5_INVALID" if not v else "P2_PATH5_OPPORTUNITY_SPARSE" if not g else "P2_PATH5_SEQUENCE_DAMAGE" if not s
               else "P2_PATH5_SAFE_NO_ADDED_VALUE" if not rn else "P2_PATH5_OVERCONSERVATIVE" if not t
               else "P2_PATH5_SIGNAL_CONCENTRATED" if not b else "P2_PATH5_G1A_SUPPORTED")
        assert lab == exp
    assert an.decide(True, True, True, -3, True, True) == "P2_PATH5_SAFE_NO_ADDED_VALUE"


def test_reference_firewall_and_auditor_independence(monkeypatch):
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2path5/__no_seal__.json")
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2path5/run1")
    with pytest.raises(PermissionError):
        an.terminal_sparse("results/inference_cf/p2path5/run1")
    assert not called
    fsrc = (ROOT / "experiments/inference_cf_p2path5_analyze.py").read_text()
    assert "load_references" not in fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
    assert "opportunity gate closed" in fsrc[fsrc.index("def secondary("):fsrc.index("load_references")]
    asrc = (ROOT / "experiments/inference_cf_p2path5_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2path5_analyze|inference_cf_p2path5\b|consensus_guard|branch_adjudication|path_decode|eos_boundary)",
                     asrc, re.M) is None
