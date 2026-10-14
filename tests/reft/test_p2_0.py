"""CPU tests for P2-0: LoReFT / LoRA adapters, teacher-forcing contract, cached decoding positions, selection rule.
Tiny randomly initialised WhisperForConditionalGeneration (real transformers classes, no weight download)."""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.adapters import LoRAAdapter, LoReFTAdapter, site_hook_count  # noqa: E402
from reft.auto_policy import IGNORE, build_tf_batch, tf_loss  # noqa: E402
from reft.evaluate_supervised import select_epoch  # noqa: E402

D, LAYERS, EOS = 16, 3, 1


@pytest.fixture(scope="module")
def tiny():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    c = WhisperConfig(vocab_size=64, num_mel_bins=8, encoder_layers=1, decoder_layers=LAYERS, encoder_attention_heads=2,
                      decoder_attention_heads=2, d_model=D, encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=12,
                      max_target_positions=40, pad_token_id=EOS, decoder_start_token_id=2, eos_token_id=EOS, attn_implementation="eager")
    m = WhisperForConditionalGeneration(c).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, torch.randn(1, 12, D)


L = LAYERS - 1
PROMPT = [2, 5, 6, 7]


def logits(m, enc, X, adapter=None):
    if adapter is None:
        return m(encoder_outputs=(enc.expand(X.shape[0], -1, -1),), decoder_input_ids=X, use_cache=False).logits
    with adapter.active(m):
        return m(encoder_outputs=(enc.expand(X.shape[0], -1, -1),), decoder_input_ids=X, use_cache=False).logits


def rand_loreft(seed=0, start=3):
    a = LoReFTAdapter(D, 2, L, 42, start_pos=start)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        a.module.learned.weight.add_(torch.randn(a.module.learned.weight.shape, generator=g) * 0.3)
        a.module.learned.bias.add_(torch.randn(2, generator=g))
    return a


def test_loreft_param_count_identity_and_orthonormal(tiny):
    m, enc = tiny
    a = LoReFTAdapter(D, 2, L, 42)
    assert a.n_trainable() == 2 * D * 2 + 2  # 5,122 for d = 1280
    R = a.module.rot.weight
    assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-6)
    X = torch.tensor([PROMPT + [9, 11, 13, 15]])
    assert torch.equal(logits(m, enc, X), logits(m, enc, X, a))  # exact identity at init
    assert site_hook_count(m, L) == 0  # hooks removed


def test_loreft_formula_matches_definition():
    a = LoReFTAdapter(D, 2, L, 42)
    with torch.no_grad():
        a.module.learned.weight.add_(torch.randn(2, D) * 0.5)
        a.module.learned.bias.add_(torch.randn(2))
    h = torch.randn(5, D)
    R, W, b = a.module.rot.weight, a.module.learned.weight, a.module.learned.bias
    ref = (h @ W.T + b - h @ R.T) @ R
    assert torch.allclose(a.module(h), ref, atol=1e-6)


def test_loreft_start_position_and_causality(tiny):
    m, enc = tiny
    X = torch.tensor([PROMPT + [9, 11, 13, 15, 17]])
    base = logits(m, enc, X)
    a = rand_loreft()
    lg = logits(m, enc, X, a)
    assert torch.equal(lg[:, :3], base[:, :3])          # prefix states (sot, lang, task) untouched
    assert not torch.allclose(lg[:, 3], base[:, 3])      # <|notimestamps|> state (predicts content token 0) adapted
    b6 = rand_loreft(start=6)
    l6 = logits(m, enc, X, b6)
    assert torch.equal(l6[:, :6], base[:, :6]) and not torch.allclose(l6[:, 6], base[:, 6])


def test_loreft_cached_incremental_equals_full_forward(tiny):
    """Absolute positions come from cache_position: prefill (4 tokens) + one-token steps == full forward."""
    m, enc = tiny
    a = rand_loreft()
    X = torch.tensor([PROMPT + [9, 11, 13, 15, 17]])
    full = logits(m, enc, X, a)
    with a.active(m), torch.no_grad():
        out = m(encoder_outputs=(enc,), decoder_input_ids=X[:, :4], use_cache=True)
        steps = [out.logits]
        past = out.past_key_values
        for t in range(4, X.shape[1]):
            out = m(encoder_outputs=(enc,), decoder_input_ids=X[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            steps.append(out.logits)
    inc = torch.cat(steps, 1)
    assert torch.allclose(inc, full, atol=1e-5)
    # a hook that ignored cache_position (treating each 1-token step as position 0) would leave steps unadapted:
    base = logits(m, enc, X)
    assert not torch.allclose(inc[:, 4:], base[:, 4:])


def test_loreft_disabled_is_identity(tiny):
    m, enc = tiny
    a = rand_loreft()
    X = torch.tensor([PROMPT + [9, 11]])
    with a.disabled():
        assert torch.equal(logits(m, enc, X, a), logits(m, enc, X))


def test_lora_count_identity_disable_and_restore(tiny):
    m, enc = tiny
    mm = copy.deepcopy(m)
    a = LoRAAdapter(mm, 1, 1.0, L, 42)
    assert a.n_trainable() == 2 * (D + D)  # 5,120 for 1280 x 1280 projections
    X = torch.tensor([PROMPT + [9, 11, 13]])
    assert torch.equal(logits(mm, enc, X), logits(mm, enc, X, a))
    with torch.no_grad():
        for v in a.wrapped.values():
            v.B.add_(torch.randn(v.B.shape) * 0.5)
    lg = logits(mm, enc, X, a)
    assert not torch.allclose(lg, logits(mm, enc, X))
    with a.disabled():
        assert torch.equal(logits(mm, enc, X, a), logits(mm, enc, X))
    att = mm.model.decoder.layers[L].encoder_attn
    assert isinstance(att.q_proj, torch.nn.Linear) and not hasattr(att.q_proj, "A")  # restored after the context


def test_tf_batch_alignment_masking_and_eos():
    X, Y = build_tf_batch([PROMPT, PROMPT], [[9, 11, EOS], [13, EOS]], EOS, EOS)
    assert X.tolist() == [PROMPT + [9, 11], PROMPT + [13, EOS]]
    assert Y.tolist() == [[IGNORE] * 3 + [9, 11, EOS], [IGNORE] * 3 + [13, EOS, IGNORE]]
    assert (Y[:, :3] == IGNORE).all() and int((Y == EOS).sum()) == 2   # EOS supervised exactly once per row
    for bad in ([9, 11], [9, EOS, EOS], [-5, EOS]):
        with pytest.raises(ValueError):
            build_tf_batch([PROMPT], [bad], EOS, EOS)
    with pytest.raises(ValueError):
        build_tf_batch([[2, 5, 6]], [[9, EOS]], EOS, EOS)


def test_tf_loss_ignores_padding_and_prefix():
    torch.manual_seed(1)
    X, Y = build_tf_batch([PROMPT, PROMPT], [[9, 11, EOS], [13, EOS]], EOS, EOS)
    lg = torch.randn(2, X.shape[1], 64)
    loss, n = tf_loss(lg, Y)
    manual = [-F.log_softmax(lg[i, 3 + t], -1)[y] for i, ys in enumerate([[9, 11, EOS], [13, EOS]]) for t, y in enumerate(ys)]
    assert n == 5 and torch.allclose(loss, torch.stack(manual).mean())
    lg2 = lg.clone()
    lg2[1, 5] = 1e4  # padding position of row 1 and prefix positions must not matter
    lg2[:, :3] = -1e4
    assert torch.allclose(tf_loss(lg2, Y)[0], loss)


def test_first_content_token_is_trainable(tiny):
    """CE on content token 0 only (label at position 3) produces LoReFT gradients; with start_pos 4 it cannot."""
    m, enc = tiny
    X, Y = build_tf_batch([PROMPT], [[9, 11, EOS]], EOS, EOS)
    Y0 = torch.full_like(Y, IGNORE)
    Y0[0, 3] = Y[0, 3]
    for start, expect in ((3, True), (4, False)):
        a = rand_loreft(start=start)
        loss, _ = tf_loss(logits(m, enc, X, a), Y0)
        loss.backward()
        g = a.module.learned.weight.grad
        assert (g is not None and float(g.abs().sum()) > 0) == expect


def test_one_step_update_backbone_frozen_and_save_load(tiny):
    m, enc = tiny
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    X, Y = build_tf_batch([PROMPT], [[9, 11, 13, EOS]], EOS, EOS)
    for make in (lambda mm: LoReFTAdapter(D, 2, L, 42), lambda mm: LoRAAdapter(mm, 1, 1.0, L, 42)):
        a = make(m)
        opt = torch.optim.AdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0)
        base = logits(m, enc, X)
        loss, _ = tf_loss(logits(m, enc, X, a), Y)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
        assert all(p.grad is None for p in m.parameters())
        opt.step()
        after = logits(m, enc, X, a)
        assert not torch.equal(after[:, 3:], base[:, 3:])  # effective update
        sd = a.state_dict()
        b2 = make(m)
        b2.load_state_dict(sd)
        assert torch.equal(logits(m, enc, X, b2), after)
    assert all(torch.equal(v, w0[k]) for k, v in m.state_dict().items())


def test_selection_rule():
    recs = [{"epoch": 1, "cs_mer": 0.10, "mer": 0.11, "zh_cer": 0.09}, {"epoch": 2, "cs_mer": 0.10, "mer": 0.11, "zh_cer": 0.08},
            {"epoch": 3, "cs_mer": 0.12, "mer": 0.09, "zh_cer": 0.01}]
    assert select_epoch(recs)["epoch"] == 2
    recs[1]["zh_cer"] = 0.09
    assert select_epoch(recs)["epoch"] == 1


def test_freeze_chain_requires_predecessor_hash():
    from reft.train_supervised import chain_ok
    recs = [{"sources": {}}, {"amendment": 1, "supersedes_sha256": "h0"}, {"amendment": 2, "supersedes_sha256": "h1"}]
    assert chain_ok(recs, ["h0", "h1", "h2"])
    assert not chain_ok(recs, ["hX", "h1", "h2"])            # original freeze rewritten
    assert not chain_ok(recs[:1] + recs[2:], ["h0", "h2"])  # amendment skipped / renumbered
    assert chain_ok(recs[:1], ["h0"])
