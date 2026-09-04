import numpy as np
import pytest
import torch

from steer_sweep.counterfactual_geometry import (
    counterfactual_states, downstream_interaction, joint_pca_fit_transform,
    norm_preserve, orthogonal_axes, synergy, validate_record_schema,
)
from steer_sweep.backends import LazyQwenBackend, MockBackend
from steer_sweep.round1_metrics import assert_pier_identity, pier_transition_counts
from steer_sweep.rounds import (
    assert_disjoint_manifests, deterministic_gate_controls, enumerate_coarse,
    enumerate_detail, mixture_direction,
)
from steer_sweep.teacher_forced_sanity import manual_loss, top1_diagnostics


def test_norm_preserve_zero_and_no_mutation():
    h = torch.tensor([[3.0, 4.0]])
    p = h + torch.tensor([[2.0, 0.0]])
    out = norm_preserve(h, p)
    assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1))
    assert torch.equal(norm_preserve(h, h), h)
    assert torch.equal(h, torch.tensor([[3.0, 4.0]]))


def test_mixture_scale_invariant_and_near_opposite():
    a = mixture_direction(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 2, .5)
    b = mixture_direction(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 4, 1)
    assert torch.allclose(a, b)
    c = mixture_direction(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1e-5]), 1, 1)
    assert torch.isfinite(c).all() and torch.allclose(c.norm(), torch.tensor(1.0))


def test_cells_and_controls_are_deterministic():
    assert len(enumerate_coarse()) == 84
    assert len(enumerate_detail((8, 16, 24))) == 126
    x = deterministic_gate_controls([.1, .9, .2, .7], 2, seed=42, utterance_id="u")
    y = deterministic_gate_controls([.1, .9, .2, .7], 2, seed=42, utterance_id="u")
    assert x == y


def test_split_disjointness_and_duplicate_detection():
    assert assert_disjoint_manifests({"a": [{"utterance_id": "u1", "dialogue_id": "d1"}], "b": [{"utterance_id": "u2", "dialogue_id": "d2"}]})["ok"]
    with pytest.raises(AssertionError):
        assert_disjoint_manifests({"a": [{"utterance_id": "u1", "dialogue_id": "d1"}], "b": [{"utterance_id": "u1", "dialogue_id": "d2"}]})


def test_pier_transition_identity():
    refs = ["我 like tea", "你 like coffee"]
    base = ["我 喜欢 tea", "你 like coffee"]
    method = ["我 like tea", "你 喜欢 coffee"]
    t = pier_transition_counts(refs, base, method)
    assert t["baseline_wrong_method_correct"] == 1
    assert t["baseline_correct_method_wrong"] == 1
    assert_pier_identity(0.5, 0.5, t)


def test_pier_transition_uses_normalized_poi_indices():
    # Whitespace/punctuation normalization must not change the transition
    # denominator relative to the existing corpus PIER implementation.
    refs = ["我，like tea。"]
    base = ["我 like tea"]
    method = ["我 喜欢 tea"]
    t = pier_transition_counts(refs, base, method)
    assert t["num_poi"] == 2 and t["baseline_wrong_method_correct"] == 0
    assert t["baseline_correct_method_wrong"] == 1


def test_geometry_axes_counterfactual_synergy_and_joint_pca():
    h = torch.tensor([1.0, 2.0, 3.0])
    states = counterfactual_states(h, torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0]), 1, 2, .5)
    assert set(states) == {"base_00", "natural_10", "prompt_01", "combined_11"}
    assert all(torch.allclose(v.norm(), h.norm()) for v in states.values())
    axes = orthogonal_axes(torch.tensor([1.0, 0]), torch.tensor([1.0, 1e-9]), 1e-6)
    assert axes.degenerate
    model, z = joint_pca_fit_transform(np.eye(4), 2)
    assert z.shape == (4, 2) and model.components_.shape == (2, 4)
    assert synergy(4, 1, 2) == 1
    assert np.allclose(downstream_interaction([0, 0], [1, 0], [0, 1], [2, 2]), [1, 1])


def test_representation_schema_and_lazy_qwen():
    record = {k: 0 for k in ("utterance_id", "dialogue_id", "layer", "downstream_layer", "token_index", "state", "outcome", "beta", "mixture", "x", "y", "gold_logp", "gold_probability", "gold_rank", "gold_margin", "entropy", "english_mass", "mandarin_mass", "local_probe", "prompt_probe", "input_record_hash")}
    validate_record_schema(record)
    q = LazyQwenBackend()
    assert not q.loaded
    with pytest.raises(RuntimeError): q.load()
    assert MockBackend().load().loaded


def test_teacher_forced_prefix_is_ignored_once():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 9)
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    loss = manual_loss(logits, labels, 4)
    expected = torch.nn.functional.cross_entropy(logits[:, 4:].reshape(-1, 9), labels[:, 4:].reshape(-1))
    assert torch.allclose(loss, expected)
    d = top1_diagnostics(logits, labels, 4)
    assert d["ignored_prefix"] == 4 and d["valid_tokens"] == 2
