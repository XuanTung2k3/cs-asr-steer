"""LSS unit test: the decoder site is the post-cross-attention residual.

The proposal defines the decoder intervention there; the pre-existing
`DecoderSteeringHook` acts on the whole block's output instead. These tests pin
the difference, so the two can never be silently swapped.
"""
from __future__ import annotations

import copy

import pytest
import torch

from csasr.lss.sites import (
    DECODER_TENSOR,
    DecoderPostCrossAttnRecorder,
    DecoderPostCrossAttnSteeringHook,
    assert_no_site_hooks,
    assert_site_reconstruction,
    site_report,
)
from csasr.models.hooks import ActivationRecorder

LAYER = 1


def _inputs(bundle):
    torch.manual_seed(3)
    features = torch.randn(2, 8, 100)
    decoder_input_ids = torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]])
    return features, decoder_input_ids


def _forward(bundle):
    features, ids = _inputs(bundle)

    def run():
        with torch.inference_mode():
            bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
    return run


def test_recorded_site_matches_a_manual_reimplementation(tiny_bundle):
    """Reconstruct the layer's own arithmetic and compare, tensor for tensor."""
    features, ids = _inputs(tiny_bundle)
    layer = tiny_bundle.decoder_layer(LAYER)

    captured: dict[str, torch.Tensor] = {}

    def pre(_m, args):
        captured["residual"] = args[0].detach().clone()

    handle = layer.encoder_attn_layer_norm.register_forward_pre_hook(pre)
    try:
        with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec, \
                torch.inference_mode():
            out = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                    use_cache=False, output_hidden_states=True)
            site = rec.states[LAYER].clone()
            encoder_states = out.encoder_last_hidden_state
    finally:
        handle.remove()

    with torch.inference_mode():
        residual = captured["residual"]
        normed = layer.encoder_attn_layer_norm(residual)
        attn_out = layer.encoder_attn(hidden_states=normed,
                                      key_value_states=encoder_states)[0]
        manual = residual + attn_out

    assert torch.allclose(site, manual, atol=1e-5), (site - manual).abs().max()


def test_site_is_not_the_block_output(tiny_bundle):
    features, ids = _inputs(tiny_bundle)
    with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as site_rec, \
            ActivationRecorder(tiny_bundle, [LAYER], module="decoder") as block_rec, \
            torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
        site = site_rec.states[LAYER]
        block = block_rec.states[LAYER]
    assert site.shape == block.shape
    assert not torch.allclose(site, block), "site must differ from the block output"


def test_steering_adds_exactly_alpha_scale_direction(tiny_bundle):
    report = assert_site_reconstruction(tiny_bundle, _forward(tiny_bundle), LAYER,
                                        alpha=0.5, scale=2.0)
    assert report["reconstruction_ok"], report
    assert report["site_differs_from_block_output"] is True
    assert report["tensor"] == DECODER_TENSOR


def test_norm_preservation_keeps_the_vector_length(tiny_bundle):
    features, ids = _inputs(tiny_bundle)
    direction = torch.zeros(tiny_bundle.d_model)
    direction[2] = 1.0

    with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
        base = rec.states[LAYER].clone()

    hook = DecoderPostCrossAttnSteeringHook(
        tiny_bundle, LAYER, direction, alpha=1.0, scale=1.0,
        norm_preserve=True, steer_prefill=True)
    with hook, DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec2, \
            torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
        steered = rec2.states[LAYER].clone()

    assert torch.allclose(base.norm(dim=-1), steered.norm(dim=-1), atol=1e-4)
    assert not torch.allclose(base, steered)


def test_prefill_is_not_steered_by_default(tiny_bundle):
    features, ids = _inputs(tiny_bundle)
    direction = torch.zeros(tiny_bundle.d_model)
    direction[0] = 1.0
    hook = DecoderPostCrossAttnSteeringHook(tiny_bundle, LAYER, direction,
                                            alpha=1.0, scale=1.0)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
    assert hook.calls == 1              # one multi-token prefill call
    assert hook.steered_calls == 0      # which was left alone


def test_hooks_are_always_removed_even_on_error(tiny_bundle):
    direction = torch.zeros(tiny_bundle.d_model)
    with pytest.raises(RuntimeError):
        with DecoderPostCrossAttnSteeringHook(tiny_bundle, LAYER, direction, 1.0, 1.0):
            raise RuntimeError("boom")
    assert_no_site_hooks(tiny_bundle)

    with pytest.raises(RuntimeError):
        with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]):
            raise RuntimeError("boom")
    assert_no_site_hooks(tiny_bundle)


def test_gain_batch_mismatch_is_rejected(tiny_bundle):
    features, ids = _inputs(tiny_bundle)
    direction = torch.zeros(tiny_bundle.d_model)
    gain = torch.ones(5, 4)                      # batch is 2, not 5
    hook = DecoderPostCrossAttnSteeringHook(tiny_bundle, LAYER, direction, 1.0, 1.0,
                                            gain=gain, steer_prefill=True)
    with pytest.raises(ValueError, match="batch size"):
        with hook, torch.inference_mode():
            tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                              use_cache=False)
    assert_no_site_hooks(tiny_bundle)


def test_site_report_names_the_rejected_site(tiny_bundle):
    report = site_report(tiny_bundle, layers=[LAYER])
    assert report["decoder"]["tensor"] == DECODER_TENSOR
    assert report["decoder"]["rejected_site"] == "decoder_block_output"
    assert report["decoder"]["depth_rescale"] is False
    assert "encoder_attn_layer_norm" in report["decoder"]["capture"]["residual_in"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_reconstruction_error_is_rounding_not_wiring(tiny_bundle, dtype):
    """The site identity is exact in real arithmetic, so error tracks the dtype.

    The production H100 run failed a flat `rel_err <= 1e-3` at 1.674e-3 in
    bfloat16 -- 0.21 of a single epsilon. Measured here on one fixed set of
    weights the error is 6.7 ULP in float32, 2.9 in float16 and 1.5 in
    bfloat16, i.e. it is a property of the float format and not of the hook.
    An absolute tolerance below the dtype's epsilon is therefore unsatisfiable
    by any correct implementation, which is why the gate checks ULP and the
    block gap instead.
    """
    import dataclasses

    bundle = dataclasses.replace(
        tiny_bundle, model=copy.deepcopy(tiny_bundle.model).to(dtype), dtype=dtype)
    features, ids = _inputs(bundle)

    def run():
        with torch.inference_mode():
            bundle.model(input_features=features.to(dtype),
                         decoder_input_ids=ids, use_cache=False)

    report = assert_site_reconstruction(bundle, run, LAYER)

    assert report["rel_err_ulp"] <= 8.0, report
    assert report["err_vs_block_gap"] <= 0.25, report
    assert report["reconstruction_ok"], report


def test_a_misplaced_hook_fails_the_block_gap_check(tiny_bundle):
    """Steering the block output instead of the site must not pass the gate.

    This is the failure the three replacement criteria exist to catch, and the
    one a loosened absolute tolerance would have let through.
    """
    from csasr.models.hooks import DecoderSteeringHook

    features, ids = _inputs(tiny_bundle)

    def run():
        with torch.inference_mode():
            tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                              use_cache=False)

    with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec, \
            ActivationRecorder(tiny_bundle, [LAYER], module="decoder") as block_rec:
        run()
        base_site = rec.states[LAYER].clone()
        block = block_rec.states[LAYER].clone()

    direction = torch.zeros(tiny_bundle.d_model, dtype=torch.float32)
    direction[0] = 1.0
    hook = DecoderSteeringHook(tiny_bundle, LAYER, direction, alpha=0.5, scale=1.0,
                               num_layers=tiny_bundle.num_decoder_layers,
                               norm_preserve=False, steer_prefill=True)
    with hook, DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec:
        run()
        wrong_site = rec.states[LAYER].clone()

    expected = base_site.clone()
    expected[..., 0] += 0.5
    gap = float((base_site - block).abs().max())
    err = float((wrong_site - expected).abs().max())
    # the site sits upstream of the block output, so steering the block leaves
    # it untouched and the whole requested delta shows up as error
    assert err / gap > 1.0, (err, gap)
    assert err / gap > 4 * 0.25, "must be far outside the gate's rounding bound"
