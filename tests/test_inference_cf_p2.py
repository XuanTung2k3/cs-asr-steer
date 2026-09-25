"""P2 compact development: frozen configs, validity/selection rules, alpha_c, bootstrap, runner."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, FRAMES, PART, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2 as p2  # noqa: E402
import experiments.inference_cf_p2_evaluate as ev  # noqa: E402
from csasr.inference_cf.core import atomic_json, digest  # noqa: E402

CFG = json.loads(Path("configs/inference_cf/p2_compact_development.json").read_text())


def test_frozen_stage_configs():
    a16 = p2.configs_for_stage("A", CFG, layer=16)
    assert [(c["layer"], c["alpha"], c["gate"], c["dose"]) for c in a16] == \
        [(16, .5, "ER", "id"), (16, 1.0, "ER", "id"), (16, 2.0, "ER", "id")]
    sel = {"selected": {"layer": 24, "alpha": 1.0}, "alpha_c": 0.3, "diagnostic_config": {"layer": 24, "alpha": 2.0}}
    b = p2.configs_for_stage("B", CFG, selection=sel)
    assert [(c["gate"], c["alpha"], c["direction_sign"]) for c in b] == \
        [("one", 1.0, 1.0), ("one", 0.3, 1.0), ("E", 1.0, 1.0), ("R", 1.0, 1.0), ("cf", 1.0, 1.0), ("ER", 1.0, -1.0)]
    c = p2.configs_for_stage("C", CFG, selection=sel)
    assert [(x["dose"], x["alpha"], x["layer"]) for x in c] == [("sqrt", .5, 24), ("sqrt", 1.0, 24), ("sqrt", 2.0, 24)]
    d = p2.configs_for_stage("A_diag", CFG, selection=sel)
    assert [(x["gate"], x["direction_sign"], x["alpha"]) for x in d] == [("one", 1.0, 2.0), ("ER", -1.0, 2.0)]
    assert len({x["name"] for x in a16 + b + c}) == 12
    assert CFG["layers"] == [16, 24] and CFG["alphas"] == [0.5, 1.0, 2.0] and CFG["dose_maps"] == ["id", "sqrt"]


def _cand(dpier, dzh, energy, dose="id", alpha=1.0, valid=True):
    return {"config": {"dose": dose, "alpha": alpha}, "validity": {"valid": valid},
            "delta": {"pier_errors": dpier}, "delta_zh_cer_increase": dzh,
            "metrics": {"edits": {"realized_energy": energy}}}


def test_validity_boundaries_and_tie_rule():
    v = CFG["validity"]
    b0 = {"caps": 2, "zh_cer": .20, "mer": .26, "pier": .47}
    ok = {"complete": True, "caps": 5, "zh_cer": .205, "mer": .265, "matrix_zh_retention": .99, "pier": .46}
    assert ev.validity(ok, b0, v)["valid"]
    for key, bad in (("caps", 6), ("zh_cer", .2051), ("mer", .2651), ("matrix_zh_retention", .989), ("pier", .47)):
        assert not ev.validity({**ok, key: bad}, b0, v)["valid"], key
    c = [_cand(10, .001, 5.0), _cand(12, .003, 9.0), _cand(12, .001, 9.0, alpha=2.0),
         _cand(12, .001, 9.0, dose="sqrt", alpha=.5), _cand(12, .001, 9.0, alpha=.5), _cand(99, 0, 0, valid=False)]
    s = ev.select(c)
    assert s["delta"]["pier_errors"] == 12 and s["config"] == {"dose": "id", "alpha": .5}
    assert ev.select([_cand(5, 0, 0, valid=False)]) is None


def test_alpha_c_formula():
    assert ev.alpha_c({"edits": {"realized_energy": 4.0, "eligible_steps": 16}}) == pytest.approx(0.5)
    assert ev.alpha_c({"edits": {"realized_energy": 1.0, "eligible_steps": 0}}) == 0.0


def test_paired_bootstrap_identity_is_zero():
    refs = ev.load_references()
    ids = sorted(refs)[:40]
    b0 = json.loads(Path("results/dg04/results/B0.json").read_text())["texts"]
    h = [b0[u] for u in ids]
    r = ev.paired_bootstrap(refs, ids, h, h, reps=50, seed=240924)
    assert all(v["delta"] == 0 and v["ci95"] == [0.0, 0.0] for v in r.values())


def test_run_utterance_tiny_configs(monkeypatch):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    b = tiny_bundle()
    monkeypatch.setattr(p2, "CONDITIONS", {"cB": CB, "cE": [1, 4, 6, 7], "language_token_ids": [4, 5]})
    cfgs = p2.configs_for_stage("A", CFG, layer=24)
    res = p2.run_utterance(b, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=encoded(), inputs=None,
                           configs=cfgs, partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                           nfp=4, uid="u", baselines=False, max_new_tokens=6)
    assert set(res["systems"]) == {c["name"] for c in cfgs}
    for c in cfgs:
        s = res["systems"][c["name"]]
        assert s["lineage_ok"] and s["distinct_caches"] and s["steps"]
        assert all(set(x) >= {"t", "fb", "g", "dose", "edit", "edit_norm", "next", "unsteered_next"} for x in s["steps"])


def test_evaluator_stage_a_plumbing_no_valid_when_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "SELECTION", tmp_path / "selection.json")
    refs = ev.load_references()
    panel = json.loads(Path("results/inference_cf/p0_r2/inference_panel.json").read_text())
    b0 = json.loads(Path("results/dg04/results/B0.json").read_text())["texts"]
    cfgs = p2.configs_for_stage("A", CFG, layer=24)[:1]
    m = {"schema": p2.SCHEMA, "configs": cfgs, "baselines": True}
    m["manifest_hash"] = digest(m)
    d = tmp_path / "A"
    atomic_json(d / "manifest.json", m)
    atomic_json(d / "panel.json", panel)
    atomic_json(d / "runtime.json", {"status": "completed"})
    step = {"t": 1, "fb": None, "g": .5, "dose": .5, "E": .6, "R_B": .8, "dir": "ok", "edit": True,
            "edit_norm": .4, "pre_norm": 9.0, "next": 5, "unsteered_next": 5}
    for i, r in enumerate(panel["rows"]):
        t = b0[r["utterance_id"]]
        sysd = {"B0": {"text": t, "terminated": "eos"}, "B1": {"text": t, "terminated": "eos"},
                "B0_AUTO": {"text": t, "terminated": "eos"},
                cfgs[0]["name"]: {"text": t, "terminated": "eos", "steps": [step]}}
        atomic_json(d / "rows" / f"{i:03d}.json", {"status": "ok", "manifest_hash": m["manifest_hash"], "systems": sysd})
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "A", "--dirs", str(d), "--baseline-dir", str(d),
                                      "--out", str(tmp_path / "summary.json")])
    ev.main()
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["selection"]["p2_a_verdict"] == "P2_A_NO_VALID_CONFIG"      # dPIER = 0 fails V5
    assert s["b0_identical_across_jobs"] and s["baselines"]["B0"]["pier"] == pytest.approx(0.47, abs=0.01)
    cfg_eval = s["configs"][0]
    assert cfg_eval["validity"]["checks"]["V5_efficacy"] is False
    assert cfg_eval["metrics"]["edits"]["edited"] == 300 and cfg_eval["metrics"]["corrections"] == 0
