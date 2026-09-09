"""CPU acceptance tests for the BASIS-A3 exact encoder site."""
from __future__ import annotations

import torch


def test_decoder_oracle_gate_has_explicit_batch_and_token_axes():
    from experiments.basis_a3 import _decoder_gate

    gate = _decoder_gate({5, 7})
    r = torch.zeros((1, 4, 8))
    out = gate(abs_pos=torch.tensor([4, 5, 6, 7]), r=r)
    assert out.shape == (1, 4)
    assert torch.equal(out, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))


def test_encoder_site_reconstructs_post_self_attention_before_ffn(tiny_bundle, tiny_features):
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook, EncoderPostSelfAttnRecorder
    from csasr.models.hooks import ActivationRecorder

    layer = 1
    with EncoderPostSelfAttnRecorder(tiny_bundle, [layer]) as rec, ActivationRecorder(
            tiny_bundle, [layer], module="encoder") as block:
        tiny_bundle.model.model.encoder(tiny_features)
        base = rec.states[layer].clone()
        block_out = block.states[layer].clone()
    d = torch.zeros(tiny_bundle.d_model); d[0] = 1.0
    with EncoderPostSelfAttnInterventionHook(tiny_bundle, layer, d, alpha=.25,
                                             norm_preserve=False) as hook, \
            EncoderPostSelfAttnRecorder(tiny_bundle, [layer]) as rec:
        tiny_bundle.model.model.encoder(tiny_features)
        steered = rec.states[layer]
    expected = base.clone(); expected[..., 0] += .25
    assert torch.allclose(steered, expected, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(base, block_out)
    assert hook.steered_calls == 1


def test_encoder_site_zero_gain_is_bit_identical_and_padding_is_excluded(tiny_bundle, tiny_features):
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook, EncoderPostSelfAttnRecorder

    layer = 1; d = torch.zeros(tiny_bundle.d_model); d[0] = 1.0
    # Whisper's two stride-2 convolutions turn the 100-sample fixture into
    # the bundle's 50-frame encoder axis.
    zero = torch.zeros((tiny_features.shape[0], tiny_bundle.max_encoder_frames))
    with EncoderPostSelfAttnRecorder(tiny_bundle, [layer]) as rec:
        tiny_bundle.model.model.encoder(tiny_features)
        base = rec.states[layer].clone()
    with EncoderPostSelfAttnInterventionHook(tiny_bundle, layer, d, alpha=.5, gain=zero) as hook, \
            EncoderPostSelfAttnRecorder(tiny_bundle, [layer]) as rec:
        tiny_bundle.model.model.encoder(tiny_features)
        same = rec.states[layer]
    assert torch.equal(base, same)
    assert hook.steered_calls == 0
