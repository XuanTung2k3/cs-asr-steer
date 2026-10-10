"""CPU-only DG-07A matrix and parameterization guards."""
from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from csasr.evaluation.result_schema import CanonicalResult, validate
from csasr.steering.controller import FixedBasisAdaptiveController
from csasr.steering.dg07_variants import (
    ExactLayerQvLoRA,
    GateOnlyController,
    GlobalVector,
    lora_parameter_count,
    select_lora_rank,
    trainable_parameter_count,
)
from experiments.dg07_baselines_ablations import (
    BETA,
    EPOCHS,
    GRAD_ACCUM,
    LAYER,
    LR,
    VARIANTS,
    _batch_loss,
    _select,
    validate_config,
)
from csasr.utils.config import load_config


REPO = Path(__file__).resolve().parents[1]


class _ToySelfAttention(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.self_attn = _ToySelfAttention(d)


def test_required_matrix_and_deferred_a5_are_frozen():
    assert VARIANTS == (
        "LB1_SALSA_EXACT_GLOBAL", "LB2_LORA_MATCHED_BUDGET",
        "A1_LOCAL_ONLY", "A2_CONDITIONING_ONLY", "A3_FIXED_MIXTURE_GATE")
    cfg = load_config(REPO / "configs/dg07_baselines_ablations.yaml")
    assert cfg["variants"]["optional"]["A5_REFINED_BASIS"].startswith("DEFERRED")


def test_salsa_global_vector_position_invariant_and_exact_site_declared():
    module = GlobalVector(8)
    r = torch.randn(2, 5, 8)
    gate, direction = module.action(r=r)
    assert gate.shape == (2, 5) and torch.all(gate == 1)
    assert direction.shape == (8,)
    assert torch.equal(direction, module.vector)
    assert "DecoderPostCrossAttnInterventionHook" in (
        REPO / "experiments/dg07_baselines_ablations.py").read_text()


def test_salsa_has_no_oracle_input():
    params = inspect.signature(GlobalVector.action).parameters
    assert set(params) == {"self", "r", "_"}
    source = inspect.getsource(GlobalVector.action)
    assert all(word not in source for word in ("reference", "gold", "oracle", "mask"))


def test_lora_targets_qv_and_keeps_base_frozen():
    layer = _ToyLayer()
    for p in layer.parameters():
        p.requires_grad_(False)
    adapter = ExactLayerQvLoRA(layer, rank=2, alpha=2.0, target_layer=24)
    assert adapter.target_layer == 24
    assert adapter.parameter_count() == 2 * 2 * (8 + 8)
    assert all(not p.requires_grad for p in layer.parameters())
    assert all(p.requires_grad for p in adapter.parameters())
    x = torch.randn(3, 8)
    with adapter:
        _ = layer.self_attn.q_proj(x)
        _ = layer.self_attn.v_proj(x)
    assert not adapter._attached


def test_lora_rank_selection_is_nearest_and_frozen():
    rank, count = select_lora_rank(43_651, in_features=1280, out_features=1280)
    assert rank == 9
    assert count == 46_080
    assert lora_parameter_count(9, in_features=1280, out_features=1280) == count


def test_a1_uses_only_local_direction():
    basis = torch.eye(8, 2)
    module = GateOnlyController(basis[:, 0])
    assert torch.allclose(module.direction, basis[:, 0])
    assert any(name.startswith("output.") for name in dict(module.named_parameters()))
    assert "mixture" not in " ".join(dict(module.named_parameters()))


def test_a2_uses_only_conditioning_direction():
    basis = torch.eye(8, 2)
    module = GateOnlyController(basis[:, 1])
    assert torch.allclose(module.direction, basis[:, 1])
    assert not any("basis" in n or "mixture" in n for n, _ in module.named_parameters())


def test_a3_fixed_mixture_and_gate_only():
    basis = torch.eye(8, 2)
    expected = (basis[:, 0] + basis[:, 1]) / torch.sqrt(torch.tensor(2.0))
    module = GateOnlyController(expected)
    assert torch.allclose(module.direction, expected)
    assert not any("pi" in n or "mixture" in n for n, _ in module.named_parameters())
    assert module(torch.randn(2, 4, 8)).shape == (2, 4)


def test_m_star_basis_is_a_non_trainable_buffer():
    basis = torch.randn(8, 2)
    basis = basis / basis.norm(dim=0, keepdim=True)
    module = FixedBasisAdaptiveController(8, basis, bottleneck=2)
    assert "basis" not in dict(module.named_parameters())
    assert module.basis.requires_grad is False
    assert trainable_parameter_count(module) == sum(p.numel() for p in module.trainable_parameters)


def test_shared_objective_and_budget_config():
    cfg = load_config(REPO / "configs/dg07_baselines_ablations.yaml")
    validate_config(cfg, "A1_LOCAL_ONLY")
    assert cfg["training"]["objective"] == "L_corr + lambda_M L_ret,M"
    assert cfg["training"]["lambda_m"] == 1.0
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["epochs"] == EPOCHS
    assert cfg["training"]["gradient_accumulation"] == GRAD_ACCUM
    assert cfg["training"]["learning_rate"] == LR
    assert cfg["training"]["beta"] == BETA


def test_objective_masks_reuse_dg06_ce_and_matrix_masks_only():
    source = inspect.getsource(_batch_loss)
    assert "collate_dg06_batch" in source
    assert 'include_embedded=False' in source
    assert 'batch["ce_mask"]' in source and 'batch["rm_mask"]' in source
    assert 'batch["re_mask"]' not in source


def test_no_confirm_or_test_roles_in_frozen_config():
    cfg = load_config(REPO / "configs/dg07_baselines_ablations.yaml")
    text = str(cfg["data"])
    assert "D-dev-confirm" in text and "D-test" in text  # explicit forbidden list
    assert cfg["data"]["selection_role"] == "D-dev-select"
    assert cfg["data"]["training_roles"] == ["loc-train", "util-train"]
    assert "D-dev-confirm" not in cfg["data"]["training_roles"]
    assert "D-test" not in cfg["data"]["training_roles"]


def test_canonical_result_schema_is_required():
    result = CanonicalResult(run_id="dg07/test", system_name="dg07_test",
                             metrics={"gate_coverage": {
                                 "value": None, "numerator": None, "denominator": None,
                                 "denominator_id": None, "deferred": True}}).to_dict()
    validate(result)
    assert result["schema_version"] == "result_v1"


def test_trainable_parameter_accounting_is_exact():
    assert trainable_parameter_count(GlobalVector(1280)) == 1280
    assert trainable_parameter_count(GateOnlyController(torch.ones(1280))) == 43_585
    assert lora_parameter_count(9, in_features=1280, out_features=1280, target_modules=2) == 46_080


def test_launcher_is_mig_only_and_not_main():
    launcher = (REPO / "sbatch/cs_asr_dg07_baselines_ablations.sh").read_text()
    assert "#SBATCH --partition=mig" in launcher
    assert "partition=main" not in launcher
    assert "--run" in launcher


def test_no_beam5_or_gpu_submission_in_dg07a():
    cfg = load_config(REPO / "configs/dg07_baselines_ablations.yaml")
    assert cfg["selection"]["decode"] == "greedy_temperature_0_beam_1"
    # The launcher is a prepared command, not a submission side effect in this stage.
    assert "sbatch " not in (REPO / "experiments/dg07_baselines_ablations.py").read_text()


def test_lora_none_energy_is_a_valid_selection_tie_break():
    selected = _select([
        {"utility": 3, "pier_gain": 0.1, "matrix_retention": 0.9,
         "valid_outside_harm": True, "total_energy": None},
        {"utility": 2, "pier_gain": 0.9, "matrix_retention": 0.9,
         "valid_outside_harm": True, "total_energy": None},
    ])
    assert selected["utility"] == 3
