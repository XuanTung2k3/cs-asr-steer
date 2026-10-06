"""Pre-outcome P2-SEL-E contract checks; this file does not execute E0/E1/E2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from csasr.inference_cf.core_r2 import local_support


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/inference_cf/p2_sel_e.json").read_text())


def test_provider_is_the_frozen_100_language_softmax():
    integrity = CFG["E0"]["integrity"]
    assert integrity["language_softmax_size"] == 100
    assert integrity["en_zh_token_ids"] == [50259, 50260]
    source = (ROOT / "experiments/inference_cf_p0_r2.py").read_text()
    assert "vals = logits[list(language_ids)]" in source
    assert "probs = torch.softmax(vals, dim=0)" in source
    assert "if len(language_ids) != 100:" in source
    assert "tuple(sorted(set(int(x) for x in native.values())))" in source
    p2r_source = (ROOT / "experiments/inference_cf_p2r.py").read_text()
    assert "tuple(sorted(set(int(x) for x in native.values())))" in p2r_source
    cached_source = (ROOT / "experiments/inference_cf_cached.py").read_text()
    assert "return r2.native_lid(bundle, waveform, language_ids)" in cached_source

    generation = Path(integrity["generation_config_path"])
    if not generation.exists():
        pytest.skip("frozen model asset is not mounted in this test environment")
    assert hashlib.sha256(generation.read_bytes()).hexdigest() == integrity["generation_config_sha256"]
    model_cfg = json.loads(generation.read_text())
    lang_to_id = model_cfg["lang_to_id"]
    assert len(lang_to_id) == len(set(lang_to_id.values())) == 100
    assert [lang_to_id["<|en|>"], lang_to_id["<|zh|>"]] == integrity["en_zh_token_ids"]
    payload = json.dumps(sorted((str(k), int(v)) for k, v in lang_to_id.items()),
                         separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == integrity["canonical_mapping_sha256"]


def test_q_is_absolute_pair_mass_and_can_be_below_one():
    # The other 98 language outcomes retain the probability mass omitted from this pair.
    probs = [0.03, 0.07] + [0.90 / 98] * 98
    assert sum(probs) == pytest.approx(1.0)
    support = local_support(probs[0], probs[1], 0.2, 0.8)
    q = support["pair_mass"]
    assert q == pytest.approx(probs[0] + probs[1])
    assert q == pytest.approx(0.10)
    assert 0.0 <= q <= 1.0
    assert local_support(0.0, 0.0, 0.2, 0.8)["pair_mass"] == 0.0
    assert local_support(0.6, 0.4, 0.2, 0.8)["pair_mass"] == pytest.approx(1.0)
    assert CFG["E0"]["integrity"]["full_probability_sum_abs_tolerance"] == 1e-6
    assert CFG["E0"]["probabilities"]["Q_semantics"].find("not constrained to 1") >= 0
    assert CFG["E0"]["integrity"]["Q_near_one_integrity_check"] is False


def test_current_e_reproduction_integrity_remains_frozen():
    assert CFG["E0"]["probabilities"]["E_old_abs_tolerance_vs_p2r"] == 1e-6
    assert CFG["E0"]["integrity"]["recomputed_current_E_abs_tolerance"] == 1e-6
    assert CFG["E0"]["integrity"]["epsilon"] == 1e-12
    assert "480000 samples" in CFG["E0"]["integrity"]["null_provider"]

    # The existing stored current-E rows must be byte-equivalent numerically across parent stages.
    p2r = {}
    for path in (ROOT / "results/inference_cf/p2r/run3/rows").glob("*.json"):
        row = json.loads(path.read_text())
        p2r[row["identity"]] = {int(s["t"]): float(s["E"]) for s in row["current"]["steps"]
                                 if s["E"] is not None}
    p2sel = {}
    for path in (ROOT / "results/inference_cf/p2sel/s1_run1/rows").glob("*_a.json"):
        row = json.loads(path.read_text())
        for t, item in row["positions"].items():
            p2sel[(row["identity"], int(t))] = float(item["gate"]["E"])
    positions = json.loads((ROOT / CFG["E0"]["population"]).read_text())["positions"]
    assert len(positions) == 180
    for position in positions:
        key = (position["utterance_id"], int(position["t"]))
        assert key[1] in p2r[key[0]]
        assert key in p2sel
        assert p2r[key[0]][key[1]] == pytest.approx(p2sel[key], abs=1e-12)


def test_r3_is_exact_e_times_q_and_bounded_by_e():
    r3 = CFG["E0"]["repair_formulas"]["R3_confidence"]
    assert r3 == "E_new_t=E_t*Q_local_t; no threshold, fitted coefficient, or extra hyperparameter"
    for e in (0.0, 0.1, 0.75, 1.0):
        for q in (0.0, 0.01, 0.21689272671937943, 0.9, 1.0):
            e_new = e * q
            assert e_new == e * q
            assert -1e-12 <= e_new <= e + 1e-12 <= 1.0 + 1e-12
    assert CFG["frozen_intervention"]["reference_free_runtime_inputs"] is True
    assert "E_t*Q_local_t" in r3 and all(x not in r3.lower() for x in ("reference", "evaluator", "oracle"))


def test_h3_uses_frozen_global_cutoff_and_requires_enrichment_and_dialogues():
    h3 = CFG["E0"]["hypotheses"]["H_E3"]
    assert h3["q_low"] == 0.21689272671937943
    assert h3["q_low_source"].endswith("frozen before E0")
    c = h3["pass_criteria"]
    assert c["bootstrap_rng"] == "numpy.random.default_rng"
    assert c["bootstrap_draws"] == "20 dialogue IDs sampled with replacement; shared between groups"
    assert c["one_sided_80_lower_percentile"] == 20

    def passes(fp_low: int, tp_low: int, fp_dialogues: int,
               dialogue_equal_difference: float, lower80: float) -> bool:
        return (fp_low >= c["ZH_FP_low_Q_min_count"]
                and tp_low <= c["EN_TP_low_Q_max_count"]
                and dialogue_equal_difference >= c["dialogue_equal_prevalence_difference_min"]
                and fp_dialogues >= c["low_Q_ZH_FP_min_distinct_dialogues"]
                and lower80 > c["one_sided_dialogue_bootstrap_lower_bound_min"])

    assert passes(4, 8, 3, 0.55, 0.05)
    assert not passes(3, 0, 4, 0.8, 0.05)  # fails FP count
    assert not passes(4, 11, 4, 0.55, 0.05)  # TP low-Q rate is too high
    assert not passes(4, 8, 2, 0.55, 0.05)  # insufficient dialogue spread
    assert not passes(4, 8, 3, 0.55, 0.0)  # lower bound must be strictly positive
    assert not passes(4, 8, 3, 0.49, 0.05)  # dialogue-equal enrichment is too small


def test_precedence_activates_h3_between_h2_and_h4_without_reordering_others():
    order = CFG["E0"]["precedence"]
    assert order[0] == "P2_SEL_E_INVALID"
    assert order[1].startswith("H_E1")
    assert order[2] == "R1 if H_E2 passes"
    assert order[3] == "R3 if H_E3 passes"
    assert order[4] == "R4 if H_E4 passes"
    assert order[5] == "P2_SEL_E_DIAGNOSIS_AMBIGUOUS"
    assert CFG["E0"]["hypotheses"]["H_E3"]["pass_criteria"]["ZH_FP_total"] == 5
    assert CFG["E0"]["hypotheses"]["H_E3"]["pass_criteria"]["EN_TP_total"] == 42
