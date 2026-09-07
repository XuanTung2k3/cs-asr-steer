"""DG-02 exact post-cross-attention / pre-FFN intervention site.

CPU/synthetic matrix (spec §13 + Stage-3 Phase C). Every test runs on the real
``transformers.WhisperDecoderLayer`` class with random weights -- the tiny model
in ``conftest.tiny_bundle`` is a real ``WhisperForConditionalGeneration`` -- so no
pretrained weights are downloaded. Nothing here selects a layer, direction, rank,
beta, or gate (that is DG-03); these pin the *plumbing*.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from csasr.lss.sites import (
    CONTRACT_DECODER_LAYERS,
    AuditRecord,
    DecoderPostCrossAttnInterventionHook,
    DecoderPostCrossAttnRecorder,
    assert_no_site_hooks,
)
from csasr.models.hooks import ActivationRecorder

LAYER = 1


def _inputs():
    torch.manual_seed(3)
    features = torch.randn(2, 8, 100)
    ids = torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]])
    return features, ids


def _unit_direction(bundle, idx: int = 0) -> torch.Tensor:
    d = torch.zeros(bundle.d_model, dtype=torch.float32)
    d[idx] = 1.0
    return d


def _forward(bundle, features, ids):
    with torch.inference_mode():
        return bundle.model(input_features=features, decoder_input_ids=ids,
                            use_cache=False).logits


def _deep_bundle(num_decoder_layers: int):
    """A tiny but *deep* real-class bundle, for the L16/L24 mapping tests."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    from csasr.models.whisper import WhisperBundle, inspect_modules

    cfg = WhisperConfig(
        vocab_size=60, num_mel_bins=8, encoder_layers=2,
        decoder_layers=num_decoder_layers, encoder_attention_heads=2,
        decoder_attention_heads=2, d_model=16, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=50, max_target_positions=32,
        decoder_start_token_id=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        suppress_tokens=[],
    )
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return WhisperBundle(
        model=model, processor=None, config=cfg, device="cpu", dtype=torch.float32,
        model_id="deep-test", revision="test", encoder_step_sec=0.02,
        max_encoder_frames=50, num_encoder_layers=2,
        num_decoder_layers=num_decoder_layers, d_model=16, sample_rate=16000,
        chunk_sec=1.0, module_report=inspect_modules(model),
    )


# --------------------------------------------------------------------------
# 1. r == q + u_source  (recorder forward decomposition)
# --------------------------------------------------------------------------
def test_recorder_r_equals_q_plus_u_source(tiny_bundle):
    features, ids = _inputs()
    with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec, \
            torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    q = rec.q_states[LAYER]
    u = rec.u_source_states[LAYER]
    r = rec.states[LAYER]
    err = float((r - (q + u)).abs().max())
    eps = float(torch.finfo(torch.float32).eps)
    assert err <= 8.0 * eps, err
    # detached analysis mode retains no graph (spec §9 / test 13)
    assert q.grad_fn is None and u.grad_fn is None and r.grad_fn is None


# --------------------------------------------------------------------------
# 2. beta=0 is baseline-identical, bit-for-bit  (blocking invariant, A9)
# --------------------------------------------------------------------------
def test_beta_zero_is_bit_identical(tiny_bundle):
    features, ids = _inputs()
    base = _forward(tiny_bundle, features, ids)
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=0.0,
        num_forced_prefix=0)
    with hook, torch.inference_mode():
        got = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                use_cache=False).logits
    assert torch.equal(base, got)
    assert hook.steered_calls == 0


# --------------------------------------------------------------------------
# 3. disabled (all-zero gate) and removed hook restore the baseline path
# --------------------------------------------------------------------------
def test_zero_gate_and_removed_hook_are_identity(tiny_bundle):
    features, ids = _inputs()
    base = _forward(tiny_bundle, features, ids)

    # A gate that is zero everywhere disables the edit even with alpha != 0.
    zero_gate = lambda **kw: torch.zeros(kw["r"].shape[:2])  # noqa: E731
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=0, gate_fn=zero_gate)
    with hook, torch.inference_mode():
        disabled = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                     use_cache=False).logits
    assert torch.equal(base, disabled)
    assert hook.steered_calls == 0

    # After the context exits the hook is gone: forward matches the baseline.
    assert_no_site_hooks(tiny_bundle)
    assert torch.equal(base, _forward(tiny_bundle, features, ids))


# --------------------------------------------------------------------------
# 4 + 11 + 12. exact site + FFN consumes the repaired r  (post-cross/pre-FFN)
# --------------------------------------------------------------------------
def test_ffn_consumes_repaired_r_at_the_exact_site(tiny_bundle):
    features, ids = _inputs()
    layer = tiny_bundle.decoder_layer(LAYER)
    direction = _unit_direction(tiny_bundle, 2)

    # Baseline q / u_source / r with no intervention.
    with DecoderPostCrossAttnRecorder(tiny_bundle, [LAYER]) as rec, \
            torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    q = rec.q_states[LAYER]
    r = rec.states[LAYER]

    # Repaired site r̃ computed independently from apply_steering (norm-preserve).
    from csasr.models.hooks import apply_steering
    r_tilde = apply_steering(r, direction, alpha=1.5, scale=1.0, gain=None,
                             norm_preserve=True)

    # The layer's own FFN applied to r̃ (what the block output must become).
    with torch.inference_mode():
        h = layer.final_layer_norm(r_tilde)
        h = layer.activation_fn(layer.fc1(h))
        h = layer.fc2(h)
        expected_block = r_tilde + h

    # Now run the model with the intervention hook and capture the *block output*.
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, direction, alpha=1.5, num_forced_prefix=0)
    with hook, ActivationRecorder(tiny_bundle, [LAYER], module="decoder") as block, \
            torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    got_block = block.states[LAYER]

    assert hook.steered_calls == 1
    # The FFN consumed the repaired r̃, not the baseline r.
    assert torch.allclose(got_block, expected_block, atol=1e-5), \
        float((got_block - expected_block).abs().max())
    # And the site genuinely sits upstream of the block output.
    assert not torch.allclose(r, got_block)
    # The edit changed q as well (site was actually steered).
    assert not torch.allclose(q, r_tilde)


# --------------------------------------------------------------------------
# 4b. norm preservation  (‖r̃‖ ≈ ‖r‖ for a non-zero edit; zero edit is exact)
# --------------------------------------------------------------------------
def test_norm_preservation(tiny_bundle):
    features, ids = _inputs()
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=2.0,
        num_forced_prefix=0, norm_preserve=True, record=True,
        record_last_only=False)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    steered = [rec for rec in hook.records if rec.steered]
    assert steered
    for rec in steered:
        assert rec.edit_norm > 0.0
        assert rec.post_norm == pytest.approx(rec.pre_norm, rel=1e-4)

    # norm_preserve off: the norm is free to move.
    hook2 = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=2.0,
        num_forced_prefix=0, norm_preserve=False, record=True,
        record_last_only=False)
    with hook2, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    moved = [r for r in hook2.records if r.steered
             and abs(r.post_norm - r.pre_norm) > 1e-3]
    assert moved


# --------------------------------------------------------------------------
# 5 + 6. forced-prefix positions never edited; eligible positions are
# --------------------------------------------------------------------------
def test_forced_prefix_excluded_eligible_edited(tiny_bundle):
    features, ids = _inputs()
    prefix = 2
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=prefix, record=True, record_last_only=False)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    by_pos: dict[int, list[AuditRecord]] = {}
    for rec in hook.records:
        by_pos.setdefault(rec.abs_pos, []).append(rec)

    for pos in range(prefix):
        for rec in by_pos[pos]:
            assert rec.is_forced_prefix
            assert rec.gate == 0.0
            assert rec.edit_norm == 0.0
            assert not rec.steered
    for pos in range(prefix, 4):
        assert any(rec.edit_norm > 0.0 for rec in by_pos[pos])
        for rec in by_pos[pos]:
            assert not rec.is_forced_prefix


# --------------------------------------------------------------------------
# 7. per-token gate values -> different edits
# --------------------------------------------------------------------------
def test_per_token_gate_gives_per_token_edits(tiny_bundle):
    features, ids = _inputs()
    gain = torch.tensor([[0.0, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0]])
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=0, gain=gain, norm_preserve=False, record=True,
        record_last_only=False)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    edit_by_pos = {rec.abs_pos: rec.edit_norm for rec in hook.records if rec.row == 0}
    assert edit_by_pos[0] == 0.0
    # strictly increasing gate -> strictly increasing edit energy
    assert edit_by_pos[1] < edit_by_pos[2] < edit_by_pos[3]


# --------------------------------------------------------------------------
# 8. per-beam gate: different states -> different gate values (not from index)
# --------------------------------------------------------------------------
def test_state_dependent_gate_is_row_local(tiny_bundle):
    features, _ = _inputs()
    seen: list[torch.Tensor] = []

    def gate_fn(*, q, u_source, r, abs_pos):
        # a gate computed from each row's OWN state, last position
        g = torch.sigmoid(r[:, -1, :].mean(dim=-1))          # (B,)
        seen.append(g.detach().clone())
        return g

    # Two rows with *different* decoder states.
    diff_ids = torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]])
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=0, gate_fn=gate_fn)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=diff_ids,
                          use_cache=False)
    g_diff = seen[-1]
    assert g_diff.shape[0] == 2
    assert float((g_diff[0] - g_diff[1]).abs()) > 1e-6   # states differ -> gates differ

    # Same state in both rows -> identical gates (so the difference above was
    # driven by state, never by beam index).
    seen.clear()
    same_ids = torch.tensor([[1, 5, 7, 9], [1, 5, 7, 9]])
    same_feats = features.clone()
    same_feats[1] = same_feats[0]
    hook2 = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=0, gate_fn=gate_fn)
    with hook2, torch.inference_mode():
        tiny_bundle.model(input_features=same_feats, decoder_input_ids=same_ids,
                          use_cache=False)
    g_same = seen[-1]
    assert float((g_same[0] - g_same[1]).abs()) < 1e-6


# --------------------------------------------------------------------------
# 9 + 10 + "no reset". cached positions advance; agree with full-sequence
# --------------------------------------------------------------------------
def test_cache_positions_advance_and_agree_with_full_sequence(tiny_bundle):
    features, _ = _inputs()
    prefix_ids = torch.tensor([[1, 5, 7, 9]])           # T=4 prefill
    feats1 = features[:1]

    # Record the absolute positions the hook sees, across cached calls.
    positions: list[list[int]] = []

    def record_gate(*, q, u_source, r, abs_pos):
        positions.append([int(x) for x in abs_pos.tolist()])
        return torch.ones(r.shape[:2])

    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=0.0,
        num_forced_prefix=4, gate_fn=record_gate)

    with hook, torch.inference_mode():
        out = tiny_bundle.model(input_features=feats1, decoder_input_ids=prefix_ids,
                                use_cache=True)
        past = out.past_key_values
        nxt = torch.tensor([[11]])
        for _ in range(3):
            out = tiny_bundle.model(input_features=feats1, decoder_input_ids=nxt,
                                    past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = torch.tensor([[13]])

    assert positions[0] == [0, 1, 2, 3]                 # prefill
    assert positions[1] == [4]                          # first cached step
    assert positions[2] == [5]                          # later cached step: no reset
    assert positions[3] == [6]
    flat = [p for step in positions for p in step]
    assert flat == sorted(flat) and len(set(flat)) == len(flat)

    # Full-sequence (use_cache=False) sees the same absolute positions.
    positions.clear()
    full_ids = torch.tensor([[1, 5, 7, 9, 11, 13, 13]])
    hook2 = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=0.0,
        num_forced_prefix=4, gate_fn=record_gate)
    with hook2, torch.inference_mode():
        tiny_bundle.model(input_features=feats1, decoder_input_ids=full_ids,
                          use_cache=False)
    assert positions[0] == [0, 1, 2, 3, 4, 5, 6]


def test_cached_prefix_eligibility_transitions(tiny_bundle):
    """Prefix positions excluded during prefill; first generated step eligible."""
    features, _ = _inputs()
    feats1 = features[:1]
    prefix_ids = torch.tensor([[1, 5, 7, 9]])

    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=4, record=True, record_last_only=False)
    with hook, torch.inference_mode():
        out = tiny_bundle.model(input_features=feats1, decoder_input_ids=prefix_ids,
                                use_cache=True)
        prefill_records = list(hook.records)
        past = out.past_key_values
        tiny_bundle.model(input_features=feats1, decoder_input_ids=torch.tensor([[11]]),
                          past_key_values=past, use_cache=True)
        step_records = hook.records[len(prefill_records):]

    assert all(rec.is_forced_prefix and rec.edit_norm == 0.0
               for rec in prefill_records)                    # all prefill excluded
    assert step_records and step_records[-1].abs_pos == 4
    assert not step_records[-1].is_forced_prefix              # first generated eligible
    assert step_records[-1].edit_norm > 0.0


# --------------------------------------------------------------------------
# 13. detached recording holds no autograd graph
# --------------------------------------------------------------------------
def test_recording_is_detached(tiny_bundle):
    features, ids = _inputs()
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=0, record=True)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    assert hook.records
    for rec in hook.records:                    # plain python floats, no tensors held
        assert isinstance(rec.pre_norm, float)
        assert isinstance(rec.to_dict()["edit_norm"], float)


# --------------------------------------------------------------------------
# 14 + 15. gradient reaches a small trainable gate; backbone stays frozen
# --------------------------------------------------------------------------
def test_gradient_reaches_trainable_gate_backbone_frozen(tiny_bundle):
    features, ids = _inputs()
    scale_param = torch.nn.Parameter(torch.tensor(0.5))

    def trainable_gate(*, q, u_source, r, abs_pos):
        return scale_param * torch.ones(r.shape[:2], dtype=r.dtype)

    direction = _unit_direction(tiny_bundle)
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, direction, alpha=1.0, num_forced_prefix=0,
        gate_fn=trainable_gate, mode="train")
    # NOTE: no inference_mode / no_grad -- the training path must stay differentiable.
    with hook:
        logits = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                   use_cache=False).logits
        loss = logits.float().pow(2).mean()
        loss.backward()

    assert scale_param.grad is not None
    assert float(scale_param.grad.abs()) > 0.0
    # Every backbone parameter stayed frozen and accumulated no gradient.
    for p in tiny_bundle.model.parameters():
        assert p.requires_grad is False
        assert p.grad is None


# --------------------------------------------------------------------------
# 16 + lifecycle. cleanup, double-install refusal, exception safety
# --------------------------------------------------------------------------
def test_lifecycle_install_cleanup_and_no_double_install(tiny_bundle):
    features, ids = _inputs()
    direction = _unit_direction(tiny_bundle)
    layer = tiny_bundle.decoder_layer(LAYER)

    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, direction, alpha=1.0, num_forced_prefix=0)
    with hook:
        # exactly one layer pre-hook + one ln pre-hook + one encoder_attn hook
        assert len(layer._forward_pre_hooks) == 1
        assert len(layer.encoder_attn_layer_norm._forward_pre_hooks) == 1
        assert len(layer.encoder_attn._forward_hooks) == 1
        with pytest.raises(RuntimeError, match="already installed"):
            hook.__enter__()
    assert_no_site_hooks(tiny_bundle)                     # all removed on exit

    # cleanup survives an exception raised inside the context
    hook2 = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, direction, alpha=1.0, num_forced_prefix=0)
    with pytest.raises(RuntimeError, match="boom"):
        with hook2:
            raise RuntimeError("boom")
    assert_no_site_hooks(tiny_bundle)


def test_single_application_per_forward(tiny_bundle):
    features, ids = _inputs()
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, LAYER, _unit_direction(tiny_bundle), alpha=1.0,
        num_forced_prefix=0)
    with hook, torch.inference_mode():
        tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                          use_cache=False)
    assert hook.calls == 1            # one prefill/full-seq call at this layer
    assert hook.steered_calls == 1    # steered exactly once -- no double application


# --------------------------------------------------------------------------
# 17 + 18. L16 / L24 map to decoder.layers[16] / [24]; range + contract guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [16, 24])
def test_contract_layer_maps_to_that_decoder_block(k):
    bundle = _deep_bundle(32)
    assert bundle.decoder_layer(k) is bundle.model.model.decoder.layers[k]
    hook = DecoderPostCrossAttnInterventionHook(
        bundle, k, _unit_direction(bundle), alpha=0.0, num_forced_prefix=0,
        enforce_contract_layer=True)
    with hook:
        target = bundle.model.model.decoder.layers[k]
        assert len(target._forward_pre_hooks) == 1
        # no other decoder block carries a site hook
        for j, layer in enumerate(bundle.model.model.decoder.layers):
            if j != k:
                assert len(layer._forward_pre_hooks) == 0
                assert len(layer.encoder_attn._forward_hooks) == 0
    assert_no_site_hooks(bundle)


def test_layer_guards():
    assert CONTRACT_DECODER_LAYERS == (16, 24)
    bundle = _deep_bundle(32)
    d = _unit_direction(bundle)
    with pytest.raises(ValueError, match="not a contract candidate"):
        DecoderPostCrossAttnInterventionHook(
            bundle, 8, d, alpha=1.0, num_forced_prefix=0,
            enforce_contract_layer=True)
    with pytest.raises(ValueError, match="out of range"):
        DecoderPostCrossAttnInterventionHook(
            bundle, 40, d, alpha=1.0, num_forced_prefix=0)


# --------------------------------------------------------------------------
# real WhisperDecoderLayer class, standalone, two-hook reconstruction (§13)
# --------------------------------------------------------------------------
def test_standalone_whisper_decoder_layer_reconstruction():
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer

    cfg = WhisperConfig(d_model=16, decoder_attention_heads=2, decoder_ffn_dim=32,
                        decoder_layers=1, max_target_positions=32)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    layer = WhisperDecoderLayer(cfg, layer_idx=0).eval()
    for p in layer.parameters():
        p.requires_grad_(False)

    hidden = torch.randn(2, 5, 16)
    encoder_states = torch.randn(2, 7, 16)

    captured: dict[str, torch.Tensor] = {}

    def pre(_m, args):
        captured["q"] = args[0].detach().clone()

    def post(_m, _i, output):
        captured["u"] = output[0].detach().clone()

    h1 = layer.encoder_attn_layer_norm.register_forward_pre_hook(pre)
    h2 = layer.encoder_attn.register_forward_hook(post)
    try:
        with torch.inference_mode():
            out = layer(hidden, encoder_hidden_states=encoder_states)[0]
    finally:
        h1.remove()
        h2.remove()

    site = captured["q"] + captured["u"]
    # site (pre-FFN) is not the block output (post-FFN)
    assert not torch.allclose(site, out)
    # rebuild the block output from the site through the layer's own FFN
    with torch.inference_mode():
        h = layer.final_layer_norm(site)
        h = layer.activation_fn(layer.fc1(h))
        h = layer.fc2(h)
        rebuilt = site + h
    assert torch.allclose(rebuilt, out, atol=1e-5)
