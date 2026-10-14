"""P2 adapters for Whisper-large-v3 decoder layer 31 (frozen backbone).

LoReFT (R2): at the post-cross-attention, pre-FFN residual h = q + u of decoder layer L (q: residual entering the
cross-attention block, u: the layer's own encoder_attn output),

    h~ = h + R^T (W h + b - R h),     R in R^{r x d} with orthonormal rows, W in R^{r x d}, b in R^r.

Implemented as u -> u + delta inside the layer's encoder_attn output, so h~ feeds both the FFN branch and the decoder
residual path (exactly the P1 site semantics, reft/site.py). R uses torch's orthogonal parametrization (Householder for
a wide matrix); initialisation W = R, b = 0 makes delta exactly zero. The update is applied only at absolute decoder
positions >= ``start_pos`` (default 3: the <|notimestamps|> state that predicts content token 0, and every content
state); the absolute position comes from the layer's ``cache_position`` argument, so cached beam search sees the same
positions as teacher forcing. Parameters: r*d (R) + r*d (W) + r (b) = 5,122 for r = 2, d = 1280.

LoRA (L1): rank-r update on decoder layer L encoder_attn.q_proj and .v_proj, y = W0 x + b0 + (alpha/r) B A x, with A
Kaiming-uniform (PEFT default bound 1/sqrt(fan_in)), B = 0, no dropout, no adapted bias.
Parameters: r*(1280+1280) per projection = 5,120 for r = 1.

Both adapters keep their parameters in float32 and add a float32 update cast to the backbone dtype. ``disabled()``
turns an adapter off (used for the frozen Auto-LID decision). The LoReFT hook also exposes the last q and u it saw
(``record=True``): the interface a later acoustic-conditioned variant (M1) would read. M1 is not implemented.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoReFTIntervention(nn.Module):
    def __init__(self, d: int, r: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Q, Rr = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))
        R0 = (Q * torch.sign(torch.diagonal(Rr))).T.contiguous().float()  # r x d, orthonormal rows
        self.rot = nn.Linear(d, r, bias=False)
        torch.nn.utils.parametrizations.orthogonal(self.rot)
        self.rot.weight = R0
        self.learned = nn.Linear(d, r, bias=True)
        with torch.no_grad():
            self.learned.weight.copy_(self.rot.weight)
            self.learned.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        R = self.rot.weight
        # (W - R) h + b, written so that W == R gives exactly zero (identity at initialisation)
        return (F.linear(h, self.learned.weight - R) + self.learned.bias) @ R


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, seed: int):
        super().__init__()
        self.base = base
        g = torch.Generator().manual_seed(seed)
        bound = 1.0 / math.sqrt(base.in_features)
        self.A = nn.Parameter((torch.rand(r, base.in_features, generator=g) * 2 - 1) * bound)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + (self.scale * F.linear(F.linear(x.float(), self.A), self.B)).to(y.dtype)


class Adapter:
    """Common interface: ``with adapter.active(model): ...`` installs hooks / module swaps; ``disabled()`` bypasses."""

    method: str

    def parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.module.state_dict().items()}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)


class LoReFTAdapter(Adapter):
    method = "loreft"

    def __init__(self, d: int, r: int, layer: int, seed: int, start_pos: int = 3, device="cpu"):
        self.module = LoReFTIntervention(d, r, seed).to(device)
        self.layer, self.start_pos = layer, start_pos
        self.enabled, self.record = True, False
        self.q = self.u = self.cache_pos = None
        self.calls = 0
        self._handles = []

    # -- hooks
    def _layer_pre(self, mod, args, kwargs):
        cp = kwargs.get("cache_position")
        if cp is None:
            raise RuntimeError("decoder layer called without cache_position; absolute positions unknown")
        self.cache_pos = cp

    def _q_pre(self, mod, args):
        self.q = args[0]

    def _attn(self, mod, args, out):
        if not self.enabled:
            return None
        u = out[0]
        h = self.q + u                                           # the site (input of final_layer_norm and FFN residual)
        T = h.shape[1]
        pos = self.cache_pos[-T:] if self.cache_pos.numel() >= T else self.cache_pos
        mask = (pos >= self.start_pos).to(torch.float32)[None, :, None]
        delta = self.module(h.float()) * mask
        self.calls += 1
        if self.record:
            self.u = u
        return (u + delta.to(u.dtype),) + tuple(out[1:])

    @contextlib.contextmanager
    def active(self, model):
        L = model.model.decoder.layers[self.layer]
        self._handles = [L.register_forward_pre_hook(self._layer_pre, with_kwargs=True),
                         L.encoder_attn_layer_norm.register_forward_pre_hook(self._q_pre),
                         L.encoder_attn.register_forward_hook(self._attn)]
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles, self.q, self.u, self.cache_pos = [], None, None, None

    @contextlib.contextmanager
    def disabled(self):
        prev, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = prev


class LoRAAdapter(Adapter):
    method = "lora"
    TARGETS = ("q_proj", "v_proj")

    def __init__(self, model, r: int, alpha: float, layer: int, seed: int, device="cpu"):
        att = model.model.decoder.layers[layer].encoder_attn
        self.layer = layer
        self.wrapped = {}
        for i, name in enumerate(self.TARGETS):
            base = getattr(att, name)
            assert isinstance(base, nn.Linear), (name, type(base))
            self.wrapped[name] = LoRALinear(base, r, alpha, seed + i).to(device)
        self.module = nn.ModuleDict({k: nn.ModuleDict({"A": nn.ParameterList([v.A]), "B": nn.ParameterList([v.B])})
                                     for k, v in self.wrapped.items()})

    def state_dict(self):
        return {f"{k}.{p}": getattr(v, p).detach().cpu().clone() for k, v in self.wrapped.items() for p in ("A", "B")}

    def load_state_dict(self, sd):
        with torch.no_grad():
            for k, v in self.wrapped.items():
                v.A.copy_(sd[f"{k}.A"])
                v.B.copy_(sd[f"{k}.B"])

    @contextlib.contextmanager
    def active(self, model):
        att = model.model.decoder.layers[self.layer].encoder_attn
        orig = {k: getattr(att, k) for k in self.TARGETS}
        assert all(orig[k] is self.wrapped[k].base for k in self.TARGETS), "LoRA target modules changed"
        for k in self.TARGETS:
            setattr(att, k, self.wrapped[k])
        try:
            yield self
        finally:
            for k in self.TARGETS:
                setattr(att, k, orig[k])

    @contextlib.contextmanager
    def disabled(self):
        prev = {k: v.enabled for k, v in self.wrapped.items()}
        for v in self.wrapped.values():
            v.enabled = False
        try:
            yield
        finally:
            for k, v in self.wrapped.items():
                v.enabled = prev[k]


def make_adapter(method: str, model, cfg: dict, device="cpu") -> Adapter:
    d = int(model.config.d_model)
    if method == "R2":
        a = cfg["methods"]["R2"]
        return LoReFTAdapter(d, a["rank"], a["layer"], a["init_seed"], a["start_pos"], device)
    if method == "L1":
        a = cfg["methods"]["L1"]
        return LoRAAdapter(model, a["rank"], a["alpha"], a["layer"], a["init_seed"], device)
    raise ValueError(method)


def site_hook_count(model, layer: int) -> int:
    L = model.model.decoder.layers[layer]
    mods = (L, L.encoder_attn_layer_norm, L.encoder_attn, L.final_layer_norm)
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in mods)
