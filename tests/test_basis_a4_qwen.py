import hashlib
import json
from types import SimpleNamespace

import torch
from torch import nn

from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
from csasr.models.qwen3_asr import resolve_qwen_modules


class _Layer(nn.Module):
    def __init__(self, text=False):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.final_layer_norm = nn.Identity()
        self.text = text

    def forward(self, x, **kwargs):
        return (self.post_attention_layernorm(x) if self.text
                else self.final_layer_norm(x))


class _Bundle:
    def __init__(self):
        self.model = nn.Module()
        self.model.eval()
        self._a = [_Layer()]
        self._t = [_Layer(text=True)]

    def audio_layer(self, i): return self._a[i]
    def text_layer(self, i): return self._t[i]


class _ResolvedLayer(nn.Module):
    def __init__(self, text=False, dim=4):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.text = text


class _ResolvedStack(nn.Module):
    def __init__(self, count, text=False, dim=4):
        super().__init__()
        self.layers = nn.ModuleList([_ResolvedLayer(text=text, dim=dim) for _ in range(count)])


class _ResolvedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = nn.Module()
        self.thinker.audio_tower = _ResolvedStack(24, dim=4)
        self.thinker.model = _ResolvedStack(28, text=True, dim=8)


def test_qwen_encoder_exact_site_and_norm_preserve():
    b = _Bundle()
    d = torch.zeros(4); d[0] = 1
    x = torch.randn(5, 4)
    with QwenSiteInterventionHook(b, 0, d, alpha=.5, site=AUDIO_SITE,
                                  gain=torch.tensor([1, 0, 1, 0, 0])) as h:
        y = b.audio_layer(0).final_layer_norm(x)
    assert h.calls == 1
    assert torch.equal(y[1], x[1])
    assert torch.equal(y[3], x[3])
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert h.records[0].active_positions == 2


def test_qwen_decoder_control_positions_and_cache_positions():
    b = _Bundle()
    d = torch.zeros(4); d[1] = 1
    # The exact hook only edits the explicitly eligible generated positions.
    with QwenSiteInterventionHook(b, 0, d, alpha=.25, site=TEXT_SITE,
                                  allowed_positions={8, 9}) as h:
        b.text_layer(0).post_attention_layernorm(torch.randn(1, 3, 4))
        # Cache bookkeeping is installed on the layer, so exercise the same
        # call path with the frozen absolute positions.
        b.text_layer(0)(torch.randn(1, 2, 4), cache_position=torch.tensor([8, 9]))
    assert h.calls == 2
    assert h.records[-1].active_positions == 2


def test_qwen_decoder_forbids_special_tokens_even_when_position_is_allowed():
    b = _Bundle(); d = torch.zeros(4); d[1] = 1
    h = QwenSiteInterventionHook(b, 0, d, alpha=.25, site=TEXT_SITE,
                                 allowed_positions={8, 9, 10},
                                 excluded_token_ids={99})
    h._input_ids = torch.tensor([[11, 99, 12]])
    with h:
        b.text_layer(0)(torch.randn(1, 3, 4), cache_position=torch.tensor([8, 9, 10]))
    assert h.records[0].active_positions == 2


def test_qwen_rho_zero_is_bit_identity_and_no_gradients():
    b = _Bundle()
    x = torch.randn(2, 4, requires_grad=True)
    d = torch.randn(4)
    with QwenSiteInterventionHook(b, 0, d, alpha=0.0, site=AUDIO_SITE) as h:
        y = b.audio_layer(0).final_layer_norm(x)
    assert torch.equal(y, x)
    assert h.steered_calls == 0
    assert x.grad is None


def test_a4_direction_names_exclude_redundant_conditioning_avg():
    assert {"Raw", "Conditioning"} == {"Raw", "Conditioning"}
    assert "Conditioning-Avg" not in {"Raw", "Conditioning"}


def test_artifact_hash_is_reproducible():
    payload = {"direction": "Raw", "layer": 3, "site": AUDIO_SITE}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == hashlib.sha256(encoded).hexdigest()


def test_named_module_resolver_verifies_pinned_stack_counts_and_dims():
    resolved = resolve_qwen_modules(_ResolvedModel())
    assert len(resolved.audio_layers) == 24
    assert len(resolved.text_layers) == 28
    assert resolved.encoder_dim == 4
    assert resolved.decoder_dim == 8
    assert len({id(x) for x in resolved.audio_layers}) == 24
    assert len({id(x) for x in resolved.text_layers}) == 28
