"""P2-DIR statistics, frozen decision rules (Exp-1/2/3 incl. equality and stop cases), BROAD budget,
and independence/agreement of the CPU auditor (spec sections 6-10)."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.inference_cf_p2dir_analyze as an  # noqa: E402
import experiments.inference_cf_p2dir_audit as au  # noqa: E402
from csasr.inference_cf import unique as U  # noqa: E402
from csasr.inference_cf.broad import BroadBudget, energy_match  # noqa: E402

CFG = json.loads(Path("configs/inference_cf/p2_dir_direction_identification.json").read_text())


def iv(est, l, h, n=10000):
    return {"estimate": est, "ci": [l, h], "valid_draws": n}


# ---- bootstrap ----------------------------------------------------------------------------------

def test_bootstrap_hand_computed_dialogue_weighting():
    keys = ["a", "b", "c"]
    W = np.array([[1, 1, 1], [3, 0, 0], [0, 2, 1], [0, 0, 3]], dtype=float)
    vals = [("a", 1.0), ("a", 3.0), ("b", 10.0), ("c", 4.0)]           # dialogue means 2, 10, 4
    s = an.boot_stat(vals, keys, W, 0.5)
    assert s["estimate"] == pytest.approx((2 + 10 + 4) / 3)
    draws = np.array([16 / 3, 2.0, 24 / 3, 4.0])
    assert s["ci"] == pytest.approx([np.quantile(draws, .25), np.quantile(draws, .75)])
    assert s["valid_draws"] == 4 and s["dialogues"] == 3 and s["positions"] == 4


def test_bootstrap_omits_draws_without_rows_and_counts():
    keys = ["a", "b"]
    W = np.array([[2, 0], [0, 2], [1, 1]], dtype=float)
    s = an.boot_stat([("a", 1.0)], keys, W, 0.1)
    assert s["valid_draws"] == 2 and s["omitted_draws"] == 1 and s["estimate"] == 1.0
    empty = an.boot_stat([], keys, W, 0.1)
    assert empty["estimate"] is None and empty["ci"] is None and empty["valid_draws"] == 0


def test_draw_weights_deterministic_shared_and_sorted():
    k1, W1 = an.draw_weights(["c", "a", "b", "a"], 50, 240924)
    k2, W2 = an.draw_weights(["b", "c", "a"], 50, 240924)
    assert k1 == k2 == ["a", "b", "c"] and np.array_equal(W1, W2)
    idx = np.random.default_rng(240924).integers(0, 3, size=(50, 3))
    assert np.array_equal(W1[:, 1], (idx == 1).sum(axis=1)) and (W1.sum(axis=1) == 3).all()


def test_auditor_bootstrap_and_metrics_agree_with_analysis():
    rng = np.random.default_rng(7)
    keys, W = an.draw_weights([f"d{i}" for i in range(20)], 2000, 240924)
    vals = [(f"d{rng.integers(0, 17)}", float(rng.normal())) for _ in range(60)]
    a = an.boot_stat(vals, keys, W, 0.005)
    est, ci, n = au.ci(vals, keys, W, 0.005)
    assert abs(est - a["estimate"]) <= 1e-12 and np.allclose(ci, a["ci"], atol=1e-12, rtol=0) and n == a["valid_draws"]
    part = {"embedded_ids": list(range(10, 20)), "matrix_ids": list(range(20, 40))}
    for _ in range(5):
        z = rng.normal(size=64).astype(np.float32)
        x = an.logit_metrics(z, 3, [0, 1], [2], part, [11, 12, 30], 5)
        y = au.metrics(z, 3, [0, 1], [2], part, [11, 12, 30], 5)
        assert abs(x["m"] - y["m"]) <= 1e-12 and abs(x["logp_ref"] - y["logp_ref"]) <= 1e-12
        assert abs(x["J"] - y["J"]) <= 1e-12 and x["top1_in_ref"] == y["in_ref"] and x["top1"] == y["top"]


def test_rank_first_index_ties_and_processing():
    part = {"embedded_ids": [4], "matrix_ids": [5]}
    z = np.array([5.0, 1.0, 3.0, 3.0, 0.0, 0.0], dtype=np.float32)
    m = an.logit_metrics(z, 2, [0], [1], part, [3, 4], 2)      # token 0 suppressed; 2 and 3 tie
    assert m["top1"] == 2 and m["rank_ref"] == 2 and not m["top1_in_ref"]
    m0 = an.logit_metrics(z, 0, [], [0], part, [2], 3)            # begin suppression at step 0
    assert m0["top1"] == 2 and m0["top1_in_ref"] and m0["rank_ref"] == 1
    assert m["m"] == pytest.approx(math.log(math.exp(3) + 1) - 3.0)   # lse(z[3], z[4]) - z[2]


# ---- Exp-1 decisions ------------------------------------------------------------------------------

def fam(conf=(1.0, 0.1, 2.0), paired=(0.5, 0.01, 1.0), en=(0.0, -0.2, 0.1), zh=(0.0, -0.2, 0.1), corr=(0.0, 0.0, 0.03)):
    return {k: iv(*v) for k, v in zip(("conf", "paired", "en", "zh", "corr"), (conf, paired, en, zh, corr))}


OBS_OK = {"EN-correct": 0.0, "ZH-correct": 0.05}


def test_exp1_qualification_threshold_equalities():
    q = an.qualify_exp1(fam(conf=(0.5, 1e-9, 1)), OBS_OK, 0.95, True, True, CFG)
    assert q["qualifies"]                                         # point >= .5, corr obs == .05, valid == .95
    assert not an.qualify_exp1(fam(conf=(0.4999, 0.1, 1)), OBS_OK, 1, True, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(conf=(0.9, 0.0, 1)), OBS_OK, 1, True, True, CFG)["qualifies"]      # lower > 0 strict
    assert not an.qualify_exp1(fam(paired=(0.1, 0.0, 1)), OBS_OK, 1, True, True, CFG)["qualifies"]
    assert an.qualify_exp1(fam(en=(0, -0.25, 0)), OBS_OK, 1, True, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(zh=(0, -0.2501, 0)), OBS_OK, 1, True, True, CFG)["qualifies"]
    assert an.qualify_exp1(fam(corr=(0.01, 0, 0.05)), OBS_OK, 1, True, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(corr=(0.01, 0, 0.0501)), OBS_OK, 1, True, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(), {"EN-correct": 0.051, "ZH-correct": 0}, 1, True, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(), OBS_OK, 0.9499, True, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(), OBS_OK, 1, False, True, CFG)["qualifies"]
    assert not an.qualify_exp1(fam(), OBS_OK, 1, True, False, CFG)["qualifies"]


def test_exp1_selection_rules():
    ok = an.qualify_exp1(fam(), OBS_OK, 1, True, True, CFG)
    no = an.qualify_exp1(fam(conf=(0.1, -0.1, 0.3)), OBS_OK, 1, True, True, CFG)
    dmg = an.qualify_exp1(fam(zh=(-2, -3, -1)), OBS_OK, 1, True, True, CFG)
    assert an.select_exp1(True, {"D1": no, "D2": no}, None, CFG) == \
        {"label": "P2_DIR_NO_NEW_DIRECTION_SUPPORTED", "selected": None, "supplementary": []}
    s = an.select_exp1(True, {"D1": no, "D2": dmg}, None, CFG)
    assert s["label"] == "P2_DIR_NO_NEW_DIRECTION_SUPPORTED" and s["supplementary"] == ["P2_DIR_CAUSAL_POWER_WITH_DAMAGE:D2"]
    assert an.select_exp1(True, {"D1": ok, "D2": no}, None, CFG)["label"] == "P2_DIR_UNIQUE_SELECTED"
    assert an.select_exp1(True, {"D1": no, "D2": ok}, None, CFG)["label"] == "P2_DIR_READOUT_SELECTED"
    assert an.select_exp1(True, {"D1": ok, "D2": ok}, iv(1, 0.25, 2), CFG)["selected"] == "D1"     # lower > .25 strict
    assert an.select_exp1(True, {"D1": ok, "D2": ok}, iv(1, 0.2501, 2), CFG)["selected"] == "D2"
    assert an.select_exp1(False, {"D1": ok, "D2": ok}, None, CFG)["label"] == "P2_DIR_INVALID"


# ---- Exp-2 / Exp-3 decisions (frozen before outcomes) ---------------------------------------------

def fam2(conf=(0.3, 0.05, 1), nmo=(0.2, 0.01, 1), en=(0, -0.1, 0.1), zh=(0, -0.1, 0.1), corr=(0, 0, 0.02)):
    return {k: iv(*v) for k, v in zip(("conf", "new_minus_old", "en", "zh", "corr"), (conf, nmo, en, zh, corr))}


def test_exp2_rules():
    assert an.decide_exp2(fam2(), 1.0, OBS_OK, 1, True, CFG)["label"] == "P2_DIR_EXP2_PASS"
    assert an.decide_exp2(fam2(conf=(0.1, 0.01, 1)), 0.5, OBS_OK, 1, True, CFG)["label"] == "P2_DIR_EXP2_PASS"     # .1 and .2*.5
    assert an.decide_exp2(fam2(conf=(0.19, 0.01, 1)), 1.0, OBS_OK, 1, True, CFG)["label"] == "P2_DIR_GATE_COUPLING_SUSPECTED"
    assert an.decide_exp2(fam2(conf=(0.0999, 0.01, 1)), 0.1, OBS_OK, 1, True, CFG)["label"] == "P2_DIR_GATE_COUPLING_SUSPECTED"
    assert an.decide_exp2(fam2(nmo=(0.1, 0.0, 1)), 1.0, OBS_OK, 1, True, CFG)["label"] == "P2_DIR_GATE_COUPLING_SUSPECTED"
    assert an.decide_exp2(fam2(en=(0, -0.3, 0)), 1.0, OBS_OK, 1, True, CFG)["label"] == "P2_DIR_CAUSAL_POWER_WITH_DAMAGE"
    assert an.decide_exp2(fam2(conf=(0.0, -1, 1), corr=(0.1, 0, 0.2)), 1.0, OBS_OK, 1, True, CFG)["label"] == \
        "P2_DIR_CAUSAL_POWER_WITH_DAMAGE"
    assert an.decide_exp2(fam2(), 1.0, OBS_OK, 0.94, True, CFG)["label"] == "P2_DIR_INVALID"
    assert an.decide_exp2(fam2(), 1.0, OBS_OK, 1, False, CFG)["label"] == "P2_DIR_INVALID"


def fam3(**over):
    base = {"pier_gain_vs_b0": (0.01, 0.001, 0.02), "pier_gain_vs_old": (0.01, 0.001, 0.02),
            "damage_mer": (0, -0.001, 0.004), "damage_zh_cer": (0, -0.001, 0.005), "damage_en_wer": (0, -0.01, 0.01),
            "retention_loss_en": (0, 0, 0.01), "retention_loss_zh": (0, 0, 0.005), "outside_harm": (0, 0, 0.005),
            "broad_minus_new_pier_gain": (0, -0.01, 0.01), "outside_harm_diff": (0, -0.001, 0.001)}
    base.update(over)
    return {k: iv(*v) for k, v in base.items()}


def broad_safe(**over):
    f = fam3(**over)
    return {k: f[k] for k in ("damage_mer", "damage_zh_cer", "damage_en_wer", "retention_loss_en", "retention_loss_zh", "outside_harm")}


def test_exp3_precedence():
    L = lambda f, b=None, v=True: an.decide_exp3(f, b or broad_safe(), v, True, CFG)["label"]
    assert L(fam3()) == "P2_DIR_OLD_DIRECTION_PRIMARY_BOTTLENECK_SUPPORTED"
    assert L(fam3(damage_mer=(0, 0, 0.0051))) == "P2_DIR_CAUSAL_POWER_WITH_DAMAGE"
    assert L(fam3(retention_loss_en=(0, 0, 0.0101))) == "P2_DIR_CAUSAL_POWER_WITH_DAMAGE"
    weak = dict(pier_gain_vs_old=(0.004, 0.001, 0.01))
    assert L(fam3(**weak)) == "P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED"
    assert L(fam3(**weak, pier_gain_vs_b0=(0.01, 0.0, 0.02))) == "P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED"
    broad = dict(broad_minus_new_pier_gain=(0.005, 0.0001, 0.02))
    assert L(fam3(**weak, **broad)) == "P2_DIR_SELECTIVE_GATE_LIMIT_SUSPECTED"
    assert L(fam3(**weak, **broad), v=False) == "P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED"
    assert L(fam3(**weak, **broad), b=broad_safe(damage_en_wer=(0, 0, 0.02))) == "P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED"
    assert L(fam3(**weak, **broad, outside_harm_diff=(0, 0, 0.0051))) == "P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED"
    assert an.decide_exp3(fam3(), broad_safe(), True, False, CFG)["label"] == "P2_DIR_INVALID"


# ---- BROAD budget -----------------------------------------------------------------------------------

def test_broad_packets_exhaustion_and_cap():
    b = BroadBudget(4.0, 4)
    assert b.packet == pytest.approx(1.0) and b.next_target() == pytest.approx(1.0)
    for _ in range(3):
        b.consume(1.0)
    b.consume(0.5)                                       # under-realized packet -> remaining budget left
    assert b.remaining == pytest.approx(0.75) and b.next_target() == pytest.approx(math.sqrt(0.75))  # capped
    b.consume(math.sqrt(0.75) * 1.001)                  # rounding overshoot -> clamp at zero
    assert b.remaining == 0.0 and b.exhausted and b.next_target() == 0.0 and b.overshoot > 0
    b.consume(0.0)                                       # failure/unreachable consumes nothing
    assert b.packets == 5


def test_broad_zero_budget_and_early_eos():
    assert BroadBudget(0.0, 5).next_target() == 0.0 and BroadBudget(3.0, 0).next_target() == 0.0
    b = BroadBudget(9.0, 9)
    b.consume(1.0)                                       # EOS after one packet: budget unspent, never redistributed
    r = b.record()
    assert r["relative_mismatch"] == pytest.approx(1 / 9 - 1)


def test_broad_energy_match_validity():
    assert energy_match([(1.0, 1.01), (2.0, 1.99), (0.0, 0.0)])["valid"]
    assert not energy_match([(1.0, 0.9), (1.0, 0.9)])["valid"]                  # aggregate > 2%
    rows = [(1.0, 1.0)] * 19 + [(1.0, 1.1)]
    assert energy_match(rows)["valid"]                                         # 5% individually mismatched
    rows = [(1.0, 1.0)] * 18 + [(1.0, 1.1), (1.0, 0.9)]
    m = energy_match(rows)
    assert not m["valid"] and m["unmatched"] == [18, 19]
    assert not energy_match([(0.0, 0.0)])["valid"]


# ---- auditor independence -----------------------------------------------------------------------------

def test_auditor_does_not_import_analysis():
    src = Path("experiments/inference_cf_p2dir_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*p2dir_analyze", src, re.M) is None
    assert "inference_cf_p2dir_analyze" not in src.replace("Does NOT import inference_cf_p2dir_analyze", "")


def test_auditor_refit_matches_construction():
    from test_inference_cf_p2dir_directions import _synthetic
    HA, HB, _ = _synthetic(seed=11)
    a, b = U.fit_fold(HA, HB), au.refit(HA, HB)
    assert a["status"] == b["status"] == "ok", (a["reason"], b.get("reason"))
    assert a["selected_index"] == b["index"]
    assert np.abs(a["vector"].astype(np.float64) - b["vector"].astype(np.float64)).max() <= 2e-6
    assert au.refit(HA[:20], HB)["status"] == "invalid"


def test_auditor_construction_surface_and_runner_clean():
    assert au.construction_surface_clean()["ok"]
    assert au.runner_phase1_clean()["ok"]


def test_spot_positions_frozen_rule():
    con = json.loads(Path("results/inference_cf/p2dir/construction_population.json").read_text())
    sp = au.spot_positions(con)
    assert len(sp) == 30 and [p["stratum"] for p in sp] == ["EN-confusion"] * 10 + ["EN-correct"] * 10 + ["ZH-correct"] * 10
    import hashlib
    h = [hashlib.sha256(f"P2DIR-audit-v1|{p['utterance_id']}|{p['t']}".encode()).hexdigest() for p in sp[:10]]
    assert h == sorted(h)


def test_analysis_serialization_wrapper_preserves_values():
    from csasr.inference_cf.core import canonical
    x = {"a": np.bool_(True), "b": [np.float64(0.5), np.int64(3)], "c": {"d": np.bool_(False)}, "e": None}
    y = an.jsonable(x)
    assert y == {"a": True, "b": [0.5, 3], "c": {"d": False}, "e": None}
    assert type(y["a"]) is bool and type(y["b"][1]) is int
    canonical(y)
