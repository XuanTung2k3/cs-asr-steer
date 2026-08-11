"""Shared fixtures: a tiny random Whisper so hook/frame tests need no GPU."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_bundle():
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    from csasr.models.whisper import WhisperBundle, inspect_modules

    cfg = WhisperConfig(
        vocab_size=200, num_mel_bins=8, encoder_layers=3, decoder_layers=3,
        encoder_attention_heads=2, decoder_attention_heads=2, d_model=16,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=50,
        max_target_positions=32, decoder_start_token_id=1, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, suppress_tokens=[],
    )
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return WhisperBundle(
        model=model, processor=None, config=cfg, device="cpu", dtype=torch.float32,
        model_id="tiny-test", revision="test", encoder_step_sec=0.02,
        max_encoder_frames=50, num_encoder_layers=3, num_decoder_layers=3,
        d_model=16, sample_rate=16000, chunk_sec=1.0,
        module_report=inspect_modules(model),
    )


@pytest.fixture(scope="session")
def tiny_features(tiny_bundle):
    torch.manual_seed(1)
    return torch.randn(2, 8, 100)
