"""Focused DG-06 damage-aware population, KL-retention, and loss-composition tests."""
from __future__ import annotations

import pytest
import torch

from csasr.steering.controller import FixedBasisAdaptiveController
from csasr.steering.dg05_training import CorrectionSetIndex, correction_target_mask
from csasr.steering.dg06_losses import (
    RetentionSetIndex,
    baseline_provenance,
    damage_aware_loss,
    kl_retention_loss,
    load_retention_set,
    retention_position_mask,
    retention_set_payload,
)
from csasr.lss.sites import DecoderPostCrossAttnInterventionHook
from experiments.dg06_damage_aware import (
    _controller_state_hash,
    checkpoint_payload,
    select_checkpoint_dg06,
)


def _basis(d_model: int = 12) -> torch.Tensor:
    basis = torch.zeros(d_model, 2)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return basis


# --- populations / masks -------------------------------------------------------

def test_correction_CE_mask_is_baseline_wrong_embedded_only():
    ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    lid = torch.tensor([[-100, -100, 1, 0], [-100, 1, 0, 1]])
    ce = CorrectionSetIndex({"a": (2,), "b": ()}, ("loc-train", "util-train"), "t")
    mask = correction_target_mask(["a", "b"], ids, lid, ce, prefix_width=2)
    assert mask.tolist() == [[False, False, True, False], [False, False, False, False]]


def test_R_E_mask_selects_embedded_positions():
    ids = torch.tensor([[10, 11, 12, 13]])
    lid = torch.tensor([[-100, -100, 1, 0]])
    r_e = RetentionSetIndex({"a": (2,)}, "EN", ("loc-train", "util-train"), "t")
    mask = retention_position_mask(["a"], ids, lid, r_e, prefix_width=2)
    assert mask.tolist() == [[False, False, True, False]]


def test_R_M_mask_selects_matrix_positions():
    ids = torch.tensor([[10, 11, 12, 13]])
    lid = torch.tensor([[-100, -100, 1, 0]])
    r_m = RetentionSetIndex({"a": (3,)}, "ZH", ("loc-train", "util-train"), "t")
    mask = retention_position_mask(["a"], ids, lid, r_m, prefix_width=2)
    assert mask.tolist() == [[False, False, False, True]]


def test_population_semantics_are_disjoint_and_language_checked():
    ids = torch.tensor([[10, 11, 12, 13]])
    lid = torch.tensor([[-100, -100, 1, 0]])
    r_e = RetentionSetIndex({"a": (2,)}, "EN", ("loc-train", "util-train"), "t")
    r_m = RetentionSetIndex({"a": (3,)}, "ZH", ("loc-train", "util-train"), "t")
    me = retention_position_mask(["a"], ids, lid, r_e, prefix_width=2)
    mm = retention_position_mask(["a"], ids, lid, r_m, prefix_width=2)
    assert not bool((me & mm).any())                       # EN/ZH positions disjoint
    bad = RetentionSetIndex({"a": (3,)}, "EN", ("loc-train", "util-train"), "t")
    with pytest.raises(ValueError, match="not a EN token"):
        retention_position_mask(["a"], ids, lid, bad, prefix_width=2)
    with pytest.raises(ValueError, match="forced-prefix"):
        retention_position_mask(["a"], ids, lid,
                                RetentionSetIndex({"a": (1,)}, "ZH", ("loc-train", "util-train"), "t"),
                                prefix_width=2)


# --- KL retention --------------------------------------------------------------

def test_kl_direction_is_p0_given_ptheta():
    torch.manual_seed(0)
    base = torch.randn(1, 1, 6)
    method = torch.randn(1, 1, 6)
    mask = torch.ones(1, 1, dtype=torch.bool)
    lp0 = torch.log_softmax(base[0, 0], -1)
    lpt = torch.log_softmax(method[0, 0], -1)
    forward = float((lp0.exp() * (lp0 - lpt)).sum())       # KL(p0 || p_theta)
    reverse = float((lpt.exp() * (lpt - lp0)).sum())       # KL(p_theta || p0)
    got = float(kl_retention_loss(method, base, mask))
    assert got == pytest.approx(forward, abs=1e-6)
    assert abs(got - reverse) > 1e-4                       # not the reversed direction


def test_kl_is_zero_for_equal_distributions():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert float(kl_retention_loss(logits.clone(), logits.clone(), mask)) == pytest.approx(0.0, abs=1e-6)


def test_empty_population_is_safe_zero():
    torch.manual_seed(2)
    method = torch.randn(1, 3, 5, requires_grad=True)
    base = torch.randn(1, 3, 5)
    empty = torch.zeros(1, 3, dtype=torch.bool)
    loss = kl_retention_loss(method, base, empty)
    assert float(loss.detach()) == 0.0
    loss.backward()                                        # differentiable zero, no error


# --- loss composition ----------------------------------------------------------

def _loss_inputs(seed: int = 3):
    torch.manual_seed(seed)
    method = torch.randn(1, 4, 7, requires_grad=True)
    base = torch.randn(1, 4, 7)
    labels = torch.tensor([[1, 2, 3, 4]])
    return method, base, labels


def test_D1_composition_is_corr_plus_matrix_only():
    method, base, labels = _loss_inputs()
    ce = torch.tensor([[True, False, False, False]])
    rm = torch.tensor([[False, True, False, False]])
    total, comp = damage_aware_loss(method, base, labels, ce, rm, None,
                                    lambda_m=1.0, lambda_e=1.0)
    assert "l_ret_e" not in comp                            # no embedded term in D1
    expected = comp["l_corr"] + 1.0 * comp["l_ret_m"]
    assert float(total.detach()) == pytest.approx(expected, abs=1e-6)


def test_D2_composition_adds_embedded_retention():
    method, base, labels = _loss_inputs()
    ce = torch.tensor([[True, False, False, False]])
    rm = torch.tensor([[False, True, False, False]])
    re = torch.tensor([[False, False, True, False]])
    total, comp = damage_aware_loss(method, base, labels, ce, rm, re,
                                    lambda_m=1.0, lambda_e=1.0)
    expected = comp["l_corr"] + comp["l_ret_m"] + comp["l_ret_e"]
    assert float(total.detach()) == pytest.approx(expected, abs=1e-6)
    assert comp["n_ret_e"] == 1


def test_no_CE_gradient_outside_correction_set():
    method, base, labels = _loss_inputs()
    ce = torch.tensor([[False, True, False, False]])
    empty = torch.zeros(1, 4, dtype=torch.bool)
    total, _ = damage_aware_loss(method, base, labels, ce, empty, None,
                                 lambda_m=1.0, lambda_e=1.0)
    total.backward()
    grad = method.grad[0]
    assert torch.count_nonzero(grad[1]).item() > 0         # C_E position
    assert torch.count_nonzero(grad[0]).item() == 0
    assert torch.count_nonzero(grad[2]).item() == 0
    assert torch.count_nonzero(grad[3]).item() == 0


def test_no_retention_gradient_outside_population():
    method, base, labels = _loss_inputs()
    empty = torch.zeros(1, 4, dtype=torch.bool)
    rm = torch.tensor([[False, False, True, False]])
    total, comp = damage_aware_loss(method, base, labels, empty, rm, None,
                                    lambda_m=1.0, lambda_e=1.0)
    assert comp["l_corr"] == 0.0                            # no C_E in batch
    total.backward()
    grad = method.grad[0]
    assert torch.count_nonzero(grad[2]).item() > 0         # R_M position
    assert torch.count_nonzero(grad[0]).item() == 0
    assert torch.count_nonzero(grad[1]).item() == 0
    assert torch.count_nonzero(grad[3]).item() == 0


# --- frozen backbone / basis / controller gradient (exact-site, CPU) -----------

def test_frozen_whisper_and_basis_with_controller_gradient(tiny_bundle):
    torch.manual_seed(4)
    controller = FixedBasisAdaptiveController(tiny_bundle.d_model, _basis(tiny_bundle.d_model))
    features = torch.randn(1, 8, 100)
    ids = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        base = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                 use_cache=False).logits.float()[:, :-1]
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, 1, direction=None, alpha=0.5, scale=1.0,
        action_fn=controller.action, num_forced_prefix=0, norm_preserve=True, mode="train")
    with hook:
        out = tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
    method = out.logits.float()[:, :-1]
    labels = ids[:, 1:]
    ce = torch.tensor([[True, False, False]])
    rm = torch.tensor([[False, True, False]])
    total, _ = damage_aware_loss(method, base, labels, ce, rm, None, lambda_m=1.0, lambda_e=1.0)
    total.backward()
    assert all(p.grad is None for p in tiny_bundle.model.parameters())   # frozen Whisper
    assert controller.basis.grad is None                                  # frozen V0
    assert any(p.grad is not None for p in controller.trainable_parameters)  # controller trains


# --- initialization / provenance / serialization -------------------------------

def test_controller_initialization_is_deterministic_for_the_seed():
    torch.manual_seed(42)
    a = FixedBasisAdaptiveController(12, _basis())
    torch.manual_seed(42)
    b = FixedBasisAdaptiveController(12, _basis())
    assert _controller_state_hash(a) == _controller_state_hash(b)
    torch.manual_seed(7)
    c = FixedBasisAdaptiveController(12, _basis())
    assert _controller_state_hash(a) != _controller_state_hash(c)


def test_baseline_distribution_provenance_has_required_fields():
    prov = baseline_provenance(model_id="whisper-large-v3", model_revision="rev",
                               training_role_fingerprint="sha256:abc", config_hash="sha256:def",
                               tokenizer="WhisperTokenizerFast", git_commit="deadbeef")
    for key in ("model_id", "model_revision", "training_role_fingerprint",
                "config_hash", "tokenizer", "git_commit", "kl_direction"):
        assert key in prov
    assert prov["kl_direction"] == "p0 || p_theta"


def test_retention_payload_roundtrips_and_selection_uses_matrix_tiebreak():
    payload = retention_set_payload({"b": [3, 1, 1], "a": [2]}, language="ZH", source="x")
    assert payload["schema_version"] == "dg06_retention_set_v1"
    assert list(payload["baseline_correct_positions"]) == ["a", "b"]
    assert payload["baseline_correct_positions"]["b"] == [1, 3]
    loaded = load_retention_set(payload)
    assert loaded.language == "ZH" and loaded.n_positions == 3
    ckpt = checkpoint_payload(FixedBasisAdaptiveController(12, _basis()), epoch=1, variant="d2",
                              cfg={"stage": "dg06"}, basis_record={"tensor_hashes": {}},
                              components={"raw_l_corr": 1.0})
    assert ckpt["variant"] == "d2" and ckpt["lambda_e"] == 1.0
    records = [
        {"utility": 5, "valid_outside_harm": True, "pier_gain": 0.1,
         "matrix_retention": 0.90, "total_energy": 10},
        {"utility": 5, "valid_outside_harm": True, "pier_gain": 0.1,
         "matrix_retention": 0.95, "total_energy": 99},
    ]
    assert select_checkpoint_dg06(records) == records[1]   # higher matrix retention wins the tie
