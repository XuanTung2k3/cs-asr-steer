"""Focused CPU tests for the MECH-LANG0 offline analysis (synthetic arrays only; no artifact, model or GPU access)."""
from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mech_lang0", ROOT / "experiments" / "inference_cf_mech_lang0.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_unpack_bf16_exact():
    x = np.array([1.0, -2.5, 3.140625, 0.0], dtype=np.float32)
    bits = (x.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
    assert np.array_equal(M.unpack(bits), x)


def test_cos_and_tangent():
    rng = np.random.default_rng(0)
    h, g = rng.normal(size=64), rng.normal(size=64)
    gt = M.tangent(g, h)
    assert abs(gt @ h) < 1e-10 * np.linalg.norm(g) * np.linalg.norm(h)
    assert np.allclose(gt + h * (h @ g) / (h @ h), g)
    assert M.cos(g, 2 * g) == pytest.approx(1.0)
    assert M.cos(g, -g) == pytest.approx(-1.0)
    assert M.cos(np.zeros(3), np.ones(3)) is None
    assert M.cos(None, g) is None


def test_shape_compatibility_enforced():
    with pytest.raises(ValueError):
        M.cos(np.ones(1280), np.ones(1279))
    with pytest.raises(ValueError):
        M.tangent(np.ones(1280), np.ones(64))


def test_gap_sign_and_tie_semantics():
    z = np.full(10, -5.0)
    z[3], z[7] = 2.0, 1.0                       # strongest non-acceptable token 3 wins
    r = M.lex(z, [7], comp=3)
    assert r["G"] == pytest.approx(1.0) and r["top1"] == 3 and not r["in_ref"] and r["rank_ref"] == 2
    assert r["m"] == pytest.approx(-1.0)
    assert M.tie_consistent(r, [7])
    z2 = z.copy()
    z2[7] = 2.0                                 # exact tie: lower ID (3, non-acceptable) is top-1, G == 0
    r2 = M.lex(z2, [7])
    assert r2["G"] == 0 and r2["top1"] == 3 and not r2["in_ref"] and r2["rank_ref"] == 2
    assert M.tie_consistent(r2, [7])
    r3 = M.lex(z2, [3])                         # tie where the lower ID is acceptable -> correct
    assert r3["G"] == 0 and r3["in_ref"] and r3["rank_ref"] == 1
    z4 = z.copy()
    z4[7] = 4.0
    r4 = M.lex(z4, [7, 8])
    assert r4["G"] < 0 and r4["in_ref"]


def test_processed_suppression():
    z = np.arange(6, dtype=np.float32)
    zp = M.processed(z, 3, [5], [4])
    assert zp[5] == -np.inf and zp[4] == 4.0
    assert M.processed(z, 0, [5], [4])[4] == -np.inf


def test_summarize_missing_and_denominators():
    s = M.summarize([("d1", 1.0), ("d1", 3.0), ("d2", -1.0), ("d3", None), ("d3", float("nan"))])
    assert s["n"] == 3 and s["missing"] == 2 and s["dialogues"] == 2
    assert s["dialogue_macro"] == pytest.approx((2.0 - 1.0) / 2)
    assert s["frac_pos"] == pytest.approx(2 / 3)
    assert M.summarize([("d", None)]) == {"n": 0, "missing": 1}


def test_auc_descriptive():
    assert M.auc([3, 4], [1, 2]) == 1.0
    assert M.auc([1], [1]) == 0.5
    assert M.auc([None], [1]) is None


def test_query_alignment_join():
    Q = [{"utterance_id": "u1", "t": 2}, {"utterance_id": "u1", "t": 5}, {"utterance_id": "u2", "t": 1}]
    out, info = M.join_rows({("u1", 5): {"j": 1, "x": 1}, ("u2", 1): {"j": 2}, ("u1", 2): {"j": 0}}, Q, "x")
    assert sorted(out) == [0, 1, 2] and info["joined"] == 3 and not info["unmatched_or_index_mismatch"]
    _, bad = M.join_rows({("u1", 5): {"j": 0}}, Q, "x")          # stored index disagrees with (utterance, t)
    assert bad["joined"] == 0 and bad["unmatched_or_index_mismatch"] == [["u1", 5]]
    _, bad2 = M.join_rows({("u9", 1): {}}, Q, "x")                 # unknown identity is never joined by position
    assert bad2["unmatched_or_index_mismatch"] == [["u9", 1]]


def _arm(rows):
    return {"family": "f", "study": "s", "arm": "a", "layer": 16, "dose": "d", "sign": "+", "rows": rows}


def test_gap_summary_counts():
    rows = [
        {"j": 0, "stratum": "EN-confusion", "dialogue_id": "A", "G_base": 2.0, "G_post": -1.0, "dG": -3.0, "d_rank": -4, "dm": 1.0,
         "base_in_ref": False, "post_in_ref": True, "top1_changed": True, "post_top1": 1},
        {"j": 1, "stratum": "EN-confusion", "dialogue_id": "B", "G_base": 5.0, "G_post": 0.8, "dG": -4.2, "d_rank": -1, "dm": 1.0,
         "base_in_ref": False, "post_in_ref": False, "top1_changed": True, "post_top1": 2},
        {"j": 2, "stratum": "ZH-correct", "dialogue_id": "B", "G_base": -1.0, "G_post": 0.5, "dG": 1.5, "d_rank": 3, "dm": -1.0,
         "base_in_ref": True, "post_in_ref": False, "top1_changed": True, "post_top1": 3},
        {"j": 3, "stratum": "ZH-correct", "dialogue_id": "C", "G_base": -2.0, "G_post": -0.3, "dG": 1.7, "d_rank": 0, "dm": -1.0,
         "base_in_ref": True, "post_in_ref": True, "top1_changed": False, "post_top1": 4},
    ]
    g = M.gap_summary(_arm(rows))
    e = g["EN-confusion"]
    assert e["n"] == 2 and e["corrections"] == 1 and e["correction_dialogues"] == 1 and e["other_top1_changes"] == 1
    assert e["near_crossing_post_le_1.0"] == 1 and e["near_crossing_post_le_0.5"] == 0 and e["moved_toward_boundary"] == 2
    assert e["fraction_gap_closed"]["n"] == 2
    z = g["ZH-correct"]
    assert z["n"] == 2 and z["corruptions"] == 1 and z["near_corruption_post_ge_minus_0.5"] == 1
    assert g["EN-correct"] == {"n": 0}


def test_features_are_reference_free():
    params = list(inspect.signature(M.features).parameters)
    assert params == ["j", "s1row", "d2prov", "base_zp"]
    src = inspect.getsource(M.features).split('"""')[-1]          # code only (the docstring names what is excluded)
    for forbidden in ("target_ids", "stratum", "Y_ref", "competitor", "positions", "in_ref", "corrupt", "correction"):
        assert forbidden not in src
    s1row = {"region": {"target": {"status": "OK", "integral": 0.5}, "raw_heard_mass": 0.9}, "lid_en_minus_zh": -2.0}
    zp = np.log(np.array([0.7, 0.2, 0.1]))
    f = M.features(0, s1row, {"log_PE": -3.0, "log_PM": -0.1}, zp)
    assert f["F1_region_available"] == 1 and f["F2_query_region_attention"] == 0.5
    assert f["F5_script_logodds_PE_minus_PM"] == pytest.approx(-2.9) and f["F6_top1_prob"] == pytest.approx(0.7)
    f2 = M.features(0, {"region": {"target": {"status": "LOW_HEARD_MASS"}, "raw_heard_mass": 0.3}, "lid_en_minus_zh": 1.0},
                    {"log_PE": -1.0, "log_PM": -1.0}, zp)
    assert f2["F1_region_available"] == 0 and f2["F2_query_region_attention"] is None
    assert set(M.FEATURE_SOURCES) == set(f) and all(v[0] == "inference-available" for v in M.FEATURE_SOURCES.values())


def test_finite_sanitizer():
    o = M.finite({"a": np.float64("nan"), "b": [np.int64(2), np.bool_(True)], "c": math.inf})
    assert o == {"a": None, "b": [2, True], "c": None}


def test_no_model_or_gpu_imports():
    src = (ROOT / "experiments" / "inference_cf_mech_lang0.py").read_text()
    for forbidden in ("import torch", "import transformers", "from transformers", "from_pretrained", ".backward(", "torch.autograd",
                      "sbatch", ".cuda(", "subprocess.run([\"sbatch"):
        assert forbidden not in src
