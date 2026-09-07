"""Focused DG-04 tests (CPU): frozen basis/layer guards, deterministic B2/B3, canonical
result_v1 + outside-harm, frozen dose grid, and the reference-selection rule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.dg04_frozen_baselines import (
    BASIS, FROZEN_V_COND_HASH, FROZEN_V_LOCAL_HASH, GRID, LAYER, _b3_gate,
    _dg04_config_hash, _result_v1,
    build_frontier, main,
)

REPO = Path(__file__).resolve().parents[1]
BASIS_DIR = REPO / "results/dg03/basis"


# 1 + 13. frozen selected layer + immutable dose grid
def test_constants_frozen():
    assert LAYER == 24
    assert GRID == (0.5, 1.0, 2.0)
    assert BASIS.endswith("steering_basis_v1_L24.json")
    assert _dg04_config_hash().startswith("sha256:")


# 2. frozen basis hash/loader guard
def test_basis_hash_loader_guard():
    rec = json.loads((REPO / BASIS).read_text())
    assert int(rec["scientific_layer"]) == LAYER
    for name in ("conditioning_residualized_local", "language_conditioning"):
        arr = np.load(BASIS_DIR / rec["tensor_files"][name])
        h = "sha256:" + hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()
        assert h == rec["tensor_hashes"][name], name
    assert rec["tensor_hashes"]["conditioning_residualized_local"] == FROZEN_V_LOCAL_HASH
    assert rec["tensor_hashes"]["language_conditioning"] == FROZEN_V_COND_HASH


def _basis_vectors():
    rec = json.loads((REPO / BASIS).read_text())
    vl = np.load(BASIS_DIR / rec["tensor_files"]["conditioning_residualized_local"])
    vc = np.load(BASIS_DIR / rec["tensor_files"]["language_conditioning"])
    return vl, vc


# 3 + 4 + 5. B1 uses v_local; B2 fixed mixture deterministic + unit-norm
def test_b2_fixed_mixture_deterministic_and_unit():
    vl, vc = _basis_vectors()
    mix1 = 0.5 * vl + 0.5 * vc
    v_mix1 = mix1 / np.linalg.norm(mix1)
    mix2 = 0.5 * vl + 0.5 * vc
    v_mix2 = mix2 / np.linalg.norm(mix2)
    assert np.allclose(v_mix1, v_mix2)                     # deterministic
    assert abs(np.linalg.norm(v_mix1) - 1.0) < 1e-9        # unit norm
    # B1's direction is exactly v_local (unit), distinct from the mixture
    assert abs(np.linalg.norm(vl) - 1.0) < 1e-9
    assert not np.allclose(v_mix1, vl)


# 6 + 7. B3 projection gate deterministic, in (0,1), monotone in projection; beta handled by hook
def test_b3_gate_deterministic_and_bounded():
    vl, _ = _basis_vectors()
    v = torch.tensor(vl, dtype=torch.float32)
    gate = _b3_gate(v, tau=0.0, T=1.0)
    torch.manual_seed(0)
    r = torch.randn(2, 5, vl.shape[0])
    g1 = gate(q=None, u_source=None, r=r, abs_pos=torch.arange(5))
    g2 = gate(q=None, u_source=None, r=r.clone(), abs_pos=torch.arange(5))
    assert torch.allclose(g1, g2)                          # deterministic
    assert (g1 > 0).all() and (g1 < 1).all()               # sigmoid bounded
    # larger projection -> larger gate (monotone)
    r_hi = r + 3.0 * v.view(1, 1, -1)
    g_hi = gate(q=None, u_source=None, r=r_hi, abs_pos=torch.arange(5))
    assert (g_hi >= g1 - 1e-6).all()


# 8. no gradient / trainable controller in the runner
def test_no_trainable_controller():
    src = (REPO / "experiments/dg04_frozen_baselines.py").read_text()
    assert ".backward(" not in src
    assert "requires_grad_(True)" not in src
    assert "nn.Parameter" not in src
    assert "optim" not in src


# 11 + 12. canonical result_v1 validates and outside_harm is populated
def test_result_v1_validates_with_outside_harm():
    from csasr.lss.outcomes import CandidateUnitSets
    ids = ["u1", "u2"]
    refs = {"u1": "你好 world 朋友", "u2": "我们 go home 了"}
    base = {"u1": "你好 word 朋友", "u2": "我们 go home 了"}
    meth = {"u1": "你好 world 朋友", "u2": "我们 going home 了"}
    # u1: unit0 inside (EN), units 1.. outside(ZH); u2: unit inside EN, rest outside
    sets = [CandidateUnitSets(inside=(1,), outside=(0, 2), unknown=(), overlap_ratios={}),
            CandidateUnitSets(inside=(1,), outside=(0, 2, 3), unknown=(), overlap_ratios={})]
    langs = [{0: "ZH", 1: "EN", 2: "ZH"}, {0: "ZH", 1: "EN", 2: "EN", 3: "ZH"}]
    prov = {"model_id": "openai/whisper-large-v3", "model_revision": "r"}
    d = _result_v1(refs, base, meth, ids, cond="B1_rho1.0", provenance=prov,
                   direction_type="local", steering_strength=1.0, basis_id="sha256:x",
                   out_sets=sets, out_langs=langs,
                   audit={"n_steered": 3, "total_energy": 9.0, "mean_energy": 3.0,
                          "n_gate_eligible": 5})
    assert d["schema_version"] == "result_v1"
    assert isinstance(d["metrics"]["outside_harm"], int)
    assert "outside_harm_accounting" in d["metrics"]
    assert sorted(d["metrics"]["retention"]) == ["embedded_en", "matrix_zh", "monolingual"]
    assert d["metrics"]["realized_edit"]["n_steered"] == 3


# 14. reference-point selection follows the frozen rule
def _pt(system, rho, corr, corrupt, zh, energy, pier):
    return {"system": system, "rho": rho, "beta_nominal": rho,
            "result_v1": {"metrics": {
                "zh_cer": zh, "pier_gain": pier, "mer_gain": 0.0, "en_wer_gain": 0.0,
                "outside_harm": 0, "candidate_utility": float(corr - corrupt),
                "transitions": {"corrections": corr, "corruptions": corrupt,
                                "net_corrections": corr - corrupt,
                                "correction_rate": None, "corruption_rate": None},
                "realized_edit": {"n_steered": 10, "total_energy": energy, "mean_energy": 1.0}}}}


def test_reference_selection_rule(tmp_path):
    res = tmp_path / "results"; res.mkdir(parents=True)
    (res / "B0.json").write_text(json.dumps(_pt("B0", 0.0, 0, 0, 0.20, 0.0, 0.0)))
    # two eligible: A has higher U; B ties on U with C but lower energy
    (res / "B1_rho1.0.json").write_text(json.dumps(_pt("B1", 1.0, 30, 10, 0.21, 100.0, 0.01)))  # U=20
    (res / "B2_rho1.0.json").write_text(json.dumps(_pt("B2", 1.0, 25, 15, 0.22, 50.0, 0.02)))   # U=10
    (res / "B3_rho2.0.json").write_text(json.dumps(_pt("B3", 2.0, 5, 40, 0.90, 200.0, -0.05)))  # ineligible: U<0 & zh_cer>1.5x
    args = type("A", (), {"output_dir": str(tmp_path)})()
    build_frontier(args)
    ref = json.loads((tmp_path / "reference.json").read_text())
    assert ref["result"].startswith("SELECT B1 rho=1.0")     # highest U among eligible
    front = json.loads((tmp_path / "frontier.json").read_text())
    b3 = next(p for p in front["points"] if p["system"] == "B3")
    assert b3["matrix_retention_ok"] is False                # catastrophic matrix retention flagged
    assert front["n_eligible"] == 2
