"""Focused DG-05A controller, exact-site, and correction-only tests."""
from __future__ import annotations

import pytest
import torch

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, DecoderPostCrossAttnRecorder
from csasr.steering.controller import (
    FixedBasisAdaptiveController,
    apply_controller_action,
    assert_inference_input_names,
)
from csasr.steering.dg05_training import (
    CorrectionSetIndex,
    correction_only_loss,
    build_correction_set,
    correction_target_mask,
    correction_set_payload,
)
from experiments.dg05_adaptive_controller import checkpoint_payload, select_checkpoint


def _controller(d_model: int = 12) -> FixedBasisAdaptiveController:
    basis = torch.zeros(d_model, 2)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return FixedBasisAdaptiveController(d_model, basis)


def test_controller_shapes_bounds_mixture_and_unit_direction():
    torch.manual_seed(1)
    module = _controller()
    site = torch.randn(2, 4, 12)
    gate, pi, direction = module(site)
    assert gate.shape == (2, 4)
    assert pi.shape == (2, 4, 2)
    assert direction.shape == (2, 4, 12)
    assert bool(((gate >= 0) & (gate <= 1)).all())
    assert torch.allclose(pi.sum(-1), torch.ones_like(gate), atol=1e-6)
    assert torch.allclose(direction.norm(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_controller_is_token_dependent_and_basis_is_frozen():
    torch.manual_seed(2)
    module = _controller()
    a = torch.randn(1, 3, 12, requires_grad=True)
    b = a.detach().clone(); b[:, 1, 0] += 4.0
    ga, pia, da = module(a)
    gb, pib, db = module(b)
    assert not torch.allclose(ga, gb)
    assert not torch.allclose(pia, pib)
    assert not torch.allclose(da, db)
    (ga.sum() + pia.sum() + da.sum()).backward()
    assert module.basis.grad is None
    assert any(p.grad is not None for p in module.trainable_parameters)


def test_beta_zero_is_noop_and_norm_preserve_is_used():
    module = _controller()
    site = torch.randn(2, 3, 12)
    assert torch.equal(apply_controller_action(site, module, 0.0), site)
    steered = apply_controller_action(site, module, 0.5)
    assert torch.allclose(site.norm(dim=-1), steered.norm(dim=-1), atol=1e-5)


def test_inference_contract_rejects_forbidden_inputs():
    assert_inference_input_names(["site_state"])
    with pytest.raises(ValueError, match="forbidden"):
        assert_inference_input_names(["site_state", "reference_transcript"])


def test_correction_mask_intersects_baseline_wrong_with_embedded_only():
    ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    lid = torch.tensor([[-100, -100, 1, 0], [-100, 1, 0, 1]])
    index = CorrectionSetIndex({"a": (2,), "b": ()},
                               ("loc-train", "util-train"), "unit-test")
    mask = correction_target_mask(["a", "b"], ids, lid, index, prefix_width=2)
    assert mask.tolist() == [[False, False, True, False],
                             [False, False, False, False]]


def test_correction_loss_only_sees_CE():
    torch.manual_seed(3)
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, False, False]])
    loss = correction_only_loss(logits, labels, mask)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[0, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[0, 2]).item() == 0


def test_correction_payload_is_deterministic():
    payload = correction_set_payload({"b": [3, 1, 1], "a": [2]}, source="baseline.json")
    assert payload["schema_version"] == "dg05_correction_set_v1"
    assert list(payload["baseline_wrong_embedded_positions"]) == ["a", "b"]
    assert payload["baseline_wrong_embedded_positions"]["b"] == [1, 3]


def test_correction_set_constructor_requires_explicit_baseline_semantics():
    payload = build_correction_set([
        {"utterance_id": "u1", "role": "loc-train", "token_position": 4,
         "language": "EN", "baseline_correct": False},
        {"utterance_id": "u1", "role": "loc-train", "token_position": 5,
         "language": "ZH", "baseline_correct": False},
        {"utterance_id": "u2", "role": "util-train", "token_position": 3,
         "language": "EN", "baseline_correct": True},
    ], source="canonical-baseline-alignment")
    assert payload["baseline_wrong_embedded_positions"] == {"u1": [4]}
    with pytest.raises(ValueError, match="baseline_correct"):
        build_correction_set([{"utterance_id": "u", "role": "loc-train",
                               "token_position": 1, "language": "EN"}], source="x")


def test_forward_and_checkpoint_are_deterministic(tmp_path):
    torch.manual_seed(9)
    first = _controller()
    x = torch.randn(1, 2, 12)
    out1 = first(x)
    torch.manual_seed(9)
    second = _controller()
    out2 = second(x)
    assert all(torch.allclose(a, b) for a, b in zip(out1, out2))
    payload = checkpoint_payload(
        first, epoch=1, cfg={"stage": "dg05a"},
        basis_record={"tensor_hashes": {"local": "sha256:x"}}, train_targets=3)
    path = tmp_path / "controller.pt"
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["schema_version"] == "dg05_adaptive_controller_v1"
    assert loaded["architecture"]["bottleneck"] == 32
    assert loaded["train_targets"] == 3


def test_checkpoint_selection_rule_is_utility_then_pier_then_energy():
    records = [
        {"utility": 3, "valid_outside_harm": True, "pier_gain": 0.1, "total_energy": 9},
        {"utility": 4, "valid_outside_harm": True, "pier_gain": -0.1, "total_energy": 99},
        {"utility": 4, "valid_outside_harm": True, "pier_gain": 0.2, "total_energy": 10},
    ]
    assert select_checkpoint(records) == records[2]


def test_dynamic_action_uses_the_exact_site_and_removes_hooks(tiny_bundle):
    torch.manual_seed(4)
    basis = torch.zeros(tiny_bundle.d_model, 2)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    module = FixedBasisAdaptiveController(tiny_bundle.d_model, basis)
    features = torch.randn(2, 8, 100)
    ids = torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]])
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, 1, direction=None, alpha=0.5, scale=1.0,
        action_fn=module.action, num_forced_prefix=0, norm_preserve=True,
        mode="train", record=True)
    with hook, DecoderPostCrossAttnRecorder(tiny_bundle, [1]) as rec:
        out = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                use_cache=False)
    assert out.logits.shape[0] == 2
    assert hook.steered_calls == 1
    assert len(hook.records) == 2
    assert torch.isfinite(rec.states[1]).all()
