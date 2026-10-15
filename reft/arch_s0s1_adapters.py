"""S0/S1 architectures at the R2 site (decoder layer 31, post-cross-attention residual, pre-FFN; docs/reft/ARCH_S0S1_PROTOCOL.md).

R2 (historical, reft/adapters.py, unchanged):  h~ = h + R^T (W h + b - R h),            R, W in R^{2 x d}
R3 (standard LoReFT, rank 3):                h~ = h + R3^T (W3 h + b3 - R3 h),       R3, W3 in R^{3 x d}  -> 7,683 parameters
M1 (cross-attention-conditioned, rank 2):    h~ = h + R^T (W_h h + W_u u + b - R h), R, W_h, W_u in R^{2 x d} -> 7,682

h = q + u, with q the residual entering the cross-attention block (input of encoder_attn_layer_norm) and u the layer's
own encoder_attn output (the actual cross-attention contribution at that site). Both adapters reuse the R2 hook
conventions: the update is added to u inside the encoder_attn output, so h~ feeds the FFN branch and the residual path;
absolute positions from the layer's cache_position, update only at positions >= start_pos (3); adapter computation in
float32, update cast to the backbone dtype; disabled() bypasses (frozen Auto-LID).

R3 is reft.adapters.LoReFTAdapter with r = 3 (no new code path). M1 subclasses the R2 intervention: its R, W_h (= R2's
`learned.weight`) and b (= `learned.bias`) are built by exactly the R2 constructor, so under the same global seed M1's
initial R and orthogonal-parametrization `base` equal R2's, and W_u (`w_u`) starts at 0 without consuming any RNG.
With W_u = 0 the extra branch adds exact zeros, so M1 reproduces R2 (same update, same logits).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reft.adapters import LoReFTAdapter, LoReFTIntervention

D_MODEL, LAYER, START_POS = 1280, 31, 3
EXPECTED = {"R2": 5122, "R3": 7683, "M1": 7682}
R2_KEYS = ("rot.parametrizations.weight.original", "rot.parametrizations.weight.0.base", "learned.weight", "learned.bias")


class CondLoReFTIntervention(LoReFTIntervention):
    """R2 intervention plus a zero-initialized readout W_u of the cross-attention contribution u."""

    def __init__(self, d: int, r: int, seed: int):
        super().__init__(d, r, seed)
        self.w_u = nn.Parameter(torch.zeros(r, d))

    def components(self, h: torch.Tensor, u: torch.Tensor) -> dict:
        """Pre-projection (r-dim) pieces of the edit e = (W_h - R) h + W_u u + b (diagnostics; not used in forward)."""
        R = self.rot.weight
        return {"e_h": F.linear(h, self.learned.weight - R), "e_u": F.linear(u, self.w_u), "b": self.learned.bias,
                "W_h_h": F.linear(h, self.learned.weight), "R_h": F.linear(h, R)}

    def forward(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        R = self.rot.weight
        # identical to LoReFTIntervention.forward plus W_u u: W_h == R, W_u == 0, b == 0 gives exactly zero
        return (F.linear(h, self.learned.weight - R) + self.learned.bias + F.linear(u, self.w_u)) @ R


class CondLoReFTAdapter(LoReFTAdapter):
    """M1. Same hooks as LoReFTAdapter (layer pre-hook for cache_position, q from the encoder_attn_layer_norm pre-hook);
    the encoder_attn forward hook reads the fresh u of this call, consumes q (cleared, so a stale q is never reused) and
    returns u + delta once."""

    method = "cond_loreft"

    def __init__(self, d: int, r: int, layer: int, seed: int, start_pos: int = 3, device="cpu"):  # noqa: D107
        self.module = CondLoReFTIntervention(d, r, seed).to(device)
        self.layer, self.start_pos = layer, start_pos
        self.enabled, self.record = True, False
        self.q = self.u = self.cache_pos = None
        self.calls = 0
        self.records = []
        self._handles = []

    def _attn(self, mod, args, out):
        if not self.enabled:
            return None
        u = out[0]
        q, self.q = self.q, None
        if q is None or q.shape != u.shape:
            raise RuntimeError("cross-attention output without a matching residual q from the same layer call")
        h = q + u                                                # same bf16 construction as R2
        T = h.shape[1]
        pos = self.cache_pos[-T:] if self.cache_pos.numel() >= T else self.cache_pos
        mask = (pos >= self.start_pos).to(torch.float32)[None, :, None]
        delta = self.module(h.float(), u.float()) * mask
        self.calls += 1
        if self.record:
            self.records.append({"q": q.detach(), "u": u.detach(), "pos": pos.detach().clone()})
        return (u + delta.to(u.dtype),) + tuple(out[1:])


def make_arch_adapter(arch: str, model, seed: int, device="cpu"):
    """R2 through the historical factory (P2-1b seed semantics); R3 / M1 with the same layer, start_pos and init seed."""
    d = int(model.config.d_model)
    if arch == "R2":
        from reft.adapters import make_adapter
        from reft.p2_1b_train import seed_config
        from reft.train_supervised import c21
        return make_adapter("R2", model, seed_config(c21(), seed), device)
    if arch == "R3":
        return LoReFTAdapter(d, 3, LAYER, seed, START_POS, device)
    if arch == "M1":
        return CondLoReFTAdapter(d, 2, LAYER, seed, START_POS, device)
    raise ValueError(arch)


def named_trainable(ad) -> dict:
    return {n: p for n, p in ad.module.named_parameters() if p.requires_grad}


def transplant_r2_into_m1(m1: CondLoReFTAdapter, r2_state: dict) -> None:
    """Load an R2 state dict into M1's shared fields (R incl. base, W_h, b) and set W_u = 0. Exactly w_u may be missing."""
    assert set(r2_state) == set(R2_KEYS), sorted(r2_state)
    res = m1.module.load_state_dict(r2_state, strict=False)
    assert list(res.missing_keys) == ["w_u"] and not res.unexpected_keys, res
    with torch.no_grad():
        m1.module.w_u.zero_()


def orth_error(ad) -> float:
    with torch.no_grad():
        R = ad.module.rot.weight
        return float((R @ R.T - torch.eye(R.shape[0], device=R.device, dtype=R.dtype)).abs().max())


def perturb(ad, scale_w: float, scale_b: float, seed: int) -> None:
    """Nonzero adapter for engineering checks: Gaussian noise on W (W_h), b and, for M1, W_u (local generator)."""
    g = torch.Generator().manual_seed(seed)
    m = ad.module
    with torch.no_grad():
        m.learned.weight.add_((torch.randn(m.learned.weight.shape, generator=g) * scale_w).to(m.learned.weight.device))
        m.learned.bias.add_((torch.randn(m.learned.bias.shape, generator=g) * scale_b).to(m.learned.bias.device))
        if hasattr(m, "w_u"):
            m.w_u.add_((torch.randn(m.w_u.shape, generator=g) * scale_w).to(m.w_u.device))


def init_record(ad) -> dict:
    """Hashes of every initial tensor (incl. the orthogonal-parametrization base buffer), per method/seed."""
    import hashlib
    sd = ad.state_dict()
    return {k: "sha256:" + hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in sorted(sd.items())}
