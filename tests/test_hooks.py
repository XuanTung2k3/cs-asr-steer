"""The eight hook-correctness requirements of guide section 19."""
from __future__ import annotations

import torch

from csasr.models.hook_tests import run_hook_tests
from csasr.models.hooks import (
    ActivationRecorder,
    DecoderSteeringHook,
    EncoderSteeringHook,
    apply_steering,
    assert_no_hooks,
)


def test_all_eight_hook_checks(tiny_bundle, tiny_features):
    report = run_hook_tests(tiny_bundle, tiny_features, layer=1)
    for name, res in report.items():
        if name == "all_passed":
            continue
        assert res["passed"], f"{name} failed: {res}"
    assert report["all_passed"]


def test_recorder_captures_the_block_output(tiny_bundle, tiny_features):
    with ActivationRecorder(tiny_bundle, [0, 2]) as rec:
        tiny_bundle.model.model.encoder(tiny_features)
    assert set(rec.states) == {0, 2}
    for v in rec.states.values():
        assert v.shape == (2, tiny_bundle.max_encoder_frames, tiny_bundle.d_model)


def test_partial_mask_only_changes_masked_frames(tiny_bundle, tiny_features):
    d = torch.zeros(tiny_bundle.d_model)
    d[0] = 1.0
    gain = torch.zeros(2, tiny_bundle.max_encoder_frames)
    gain[:, 5:9] = 1.0

    with ActivationRecorder(tiny_bundle, [1]) as rec:
        tiny_bundle.model.model.encoder(tiny_features)
        ref = rec.states[1].clone()

    captured = {}
    hook = EncoderSteeringHook(tiny_bundle, 1, d, alpha=5.0, scale=1.0, gain=gain)
    with hook:
        with ActivationRecorder(tiny_bundle, [1]) as rec2:
            tiny_bundle.model.model.encoder(tiny_features)
            captured["steered"] = rec2.states[1]
    # the recorder is registered after the steering hook, so it sees the modified value
    diff = (captured["steered"] - ref).abs().sum(-1)
    assert (diff[:, 5:9] > 0).all()
    assert (diff[:, :5] == 0).all() and (diff[:, 9:] == 0).all()
    assert_no_hooks(tiny_bundle)


def test_norm_preservation_and_direction_of_shift():
    torch.manual_seed(0)
    h = torch.randn(3, 4, 16) * 2.0
    d = torch.zeros(16)
    d[3] = 1.0
    gain = torch.ones(3, 4)

    plain = apply_steering(h, d, alpha=1.0, scale=2.0, gain=gain, norm_preserve=False)
    assert torch.allclose(plain - h, torch.zeros_like(h) + 2.0 * d, atol=1e-6)

    kept = apply_steering(h, d, alpha=1.0, scale=2.0, gain=gain, norm_preserve=True)
    rel = ((kept.norm(dim=-1) - h.norm(dim=-1)).abs() / h.norm(dim=-1)).max()
    assert rel < 1e-3
    # the projection still moves in the direction of d
    assert ((kept - h) @ d).mean() > 0


def test_decoder_hook_skips_the_prefix_and_is_removed(tiny_bundle):
    d = torch.zeros(tiny_bundle.d_model)
    d[1] = 1.0
    hook = DecoderSteeringHook(tiny_bundle, 0, d, alpha=1.0, scale=1.0, num_layers=3)
    enc = tiny_bundle.model.model.encoder(torch.randn(1, 8, 100)).last_hidden_state
    with hook:
        # prefill with several tokens -> not steered
        tiny_bundle.model.model.decoder(
            input_ids=torch.tensor([[1, 2, 3]]), encoder_hidden_states=enc)
        assert hook.calls == 0
        # single-token step -> steered
        tiny_bundle.model.model.decoder(
            input_ids=torch.tensor([[4]]), encoder_hidden_states=enc)
        assert hook.calls == 1
    assert_no_hooks(tiny_bundle)


def test_alpha_zero_is_bit_identical():
    torch.manual_seed(0)
    h = torch.randn(2, 3, 16)
    d = torch.randn(16)
    d = d / d.norm()
    out = apply_steering(h, d, alpha=0.0, scale=3.0, gain=torch.ones(2, 3))
    assert torch.equal(out, h)
