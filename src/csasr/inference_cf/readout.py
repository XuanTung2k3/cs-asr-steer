"""P2-DIR D2 READOUT direction (frozen spec section 5): reference-free readout tangent.

At the unedited forced-ZH B-branch DG-02 L16 site ``h_t`` (post-cross-attention, pre-FFN, the
query predicting content token t):

    J = log(P_E + 1e-12) - log(P_M + 1e-12)

with P_E / P_M the full-vocabulary softmax masses of the frozen ``core_r2.tokenizer_partition``
embedded / matrix id sets, after generate() suppression (``suppress_tokens``; ``begin_suppress``
only at step 0), temperature 1, float32 logits. The gradient ``g = dJ/dh`` is read once with
``torch.autograd.grad`` through a ZERO-valued float32 leaf probe added to ``u_source`` at the
current query only (value-identical forward). Then, in CPU float64,
``d = (g - hhat <hhat, g>) / (||g_perp|| + 1e-12)``, returned as float32.

Inputs are exclusively: the frozen model, the pre-step B KV cache, encoder output, the current
token(s) / cache position, the step index, suppression lists and the vocabulary partition. There
is no reference transcript, reference token set, competitor, alignment, future token or evaluator
quantity anywhere in this module (enforced by tests). The scratch forward never advances or
mutates the persistent B/E/S caches: the pre-step cache is deep-copied into ordinary
(non-inference) detached tensors, used once, and discarded with its graph.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Sequence

import torch

from csasr.lss.sites import _first, _rebuild, assert_dropout_disabled, assert_no_site_hooks

VERSION = "p2dir_readout_v1"
EPS = 1e-12
TINY_STATE = 1e-8
TINY_TANGENT = 1e-8
UNIT_TOL = 2e-6
RADIAL_TOL = 1e-6


# ---- objective -------------------------------------------------------------------------------

def processed_logits(z: torch.Tensor, step: int, suppress: Sequence[int], begin: Sequence[int]) -> torch.Tensor:
    """Differentiable generate() suppression on float32 logits (begin-suppression only at step 0)."""
    z = z.float()
    mask = torch.zeros_like(z, dtype=torch.bool)
    if suppress:
        mask[list(suppress)] = True
    if step == 0 and begin:
        mask[list(begin)] = True
    return z.masked_fill(mask, -float("inf"))


def objective(z: torch.Tensor, step: int, suppress, begin, partition: dict) -> dict[str, torch.Tensor]:
    """J = log(P_E+eps) - log(P_M+eps) with stable logsumexp / logaddexp (exact objective)."""
    zp = processed_logits(z, step, suppress, begin)
    logz = torch.logsumexp(zp, 0)
    e_ids = torch.as_tensor(partition["embedded_ids"], dtype=torch.long, device=zp.device)
    m_ids = torch.as_tensor(partition["matrix_ids"], dtype=torch.long, device=zp.device)
    log_pe = torch.logsumexp(zp[e_ids], 0) - logz
    log_pm = torch.logsumexp(zp[m_ids], 0) - logz
    leps = torch.tensor(math.log(EPS), dtype=zp.dtype, device=zp.device)
    j = torch.logaddexp(log_pe, leps) - torch.logaddexp(log_pm, leps)
    return {"J": j, "log_PE": log_pe, "log_PM": log_pm}


# ---- tangent geometry ------------------------------------------------------------------------

def tangent_unit(g: torch.Tensor, h: torch.Tensor) -> dict:
    """Float64 tangent projection and epsilon normalization with the frozen no-edit guards."""
    g64 = g.detach().double().cpu().reshape(-1)
    h64 = h.detach().double().cpu().reshape(-1)
    out = {"direction": None, "status": "invalid", "reason": None, "g_norm": None, "h_norm": None,
           "tangent_norm": None, "g_radial": None, "unit_error": None, "radial_dot": None}
    if not (bool(torch.isfinite(g64).all()) and bool(torch.isfinite(h64).all())):
        out["reason"] = "nonfinite_state_or_gradient"
        return out
    hn = float(torch.linalg.vector_norm(h64))
    out["h_norm"], out["g_norm"] = hn, float(torch.linalg.vector_norm(g64))
    if not hn > TINY_STATE:
        out["reason"] = "tiny_state"
        return out
    hhat = h64 / hn
    rad = float(torch.dot(hhat, g64))
    gperp = g64 - hhat * rad
    tn = float(torch.linalg.vector_norm(gperp))
    out["g_radial"], out["tangent_norm"] = rad, tn
    if not math.isfinite(tn) or tn < TINY_TANGENT:
        out["reason"] = "tiny_tangent"
        return out
    d32 = (gperp / (tn + EPS)).float()
    d64 = d32.double()
    out["unit_error"] = abs(float(torch.linalg.vector_norm(d64)) - 1.0)
    out["radial_dot"] = abs(float(torch.dot(hhat, d64)))
    if out["unit_error"] > UNIT_TOL or out["radial_dot"] > RADIAL_TOL:
        out["reason"] = "normalization_fail"
        return out
    out["direction"], out["status"] = d32, "ok"
    return out


# ---- scratch cache ---------------------------------------------------------------------------

def _cache_tensors(cache) -> list[torch.Tensor]:
    out = []
    for sub in (cache.self_attention_cache, cache.cross_attention_cache):
        for layer in sub.layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    out.append(t)
    return out


def clone_scratch(cache, encoded):
    """Deep copy of the pre-step cache / encoder output into ordinary detached tensors."""
    with torch.inference_mode(False):
        scratch = copy.deepcopy(cache)
        enc = copy.copy(encoded)
        enc.last_hidden_state = encoded.last_hidden_state.detach().clone()
    src_ptrs = {t.data_ptr() for t in _cache_tensors(cache) if t.numel()}
    for t in _cache_tensors(scratch) + [enc.last_hidden_state]:
        if t.is_inference() or t.requires_grad:
            raise RuntimeError("scratch tensor is an inference tensor or requires grad")
        if t.numel() and t.data_ptr() in src_ptrs:
            raise RuntimeError("scratch cache aliases the persistent cache")
    if scratch.get_seq_length() != cache.get_seq_length():
        raise RuntimeError("scratch cache length mismatch")
    return scratch, enc


def cache_fingerprint(cache) -> tuple:
    """Identity fingerprint (object ids, data pointers, shapes, length) of a persistent cache."""
    ts = _cache_tensors(cache)
    return (id(cache), cache.get_seq_length(), tuple((id(t), t.data_ptr(), tuple(t.shape)) for t in ts))


# ---- zero-valued probe -----------------------------------------------------------------------

class ZeroProbe:
    """Adds a float32 zero leaf (cast to the site dtype) to u_source at ONE absolute query of the
    DG-02 site and records r = q + u_source there. The forward is value-identical to the
    unprobed step; the probe never carries a nonzero value."""

    def __init__(self, bundle, layer: int, query: int):
        self.bundle, self.layer, self.query = bundle, int(layer), int(query)
        dim = bundle.decoder_layer(self.layer).encoder_attn.out_proj.out_features
        dev = next(bundle.model.parameters()).device
        self.delta = torch.zeros(dim, dtype=torch.float32, device=dev, requires_grad=True)
        self.site: torch.Tensor | None = None
        self.calls = 0
        self._q = None
        self._cp = None
        self._handles: list = []

    def _zero(self) -> None:
        if int(torch.count_nonzero(self.delta.detach())) != 0:
            raise RuntimeError("readout probe delta is nonzero")

    def _layer_pre(self, _mod, args, kwargs):
        self._cp = kwargs.get("cache_position", None)
        return None

    def _ln_pre(self, _mod, args):
        self._q = args[0]
        return None

    def _post(self, _mod, _inp, output):
        self._zero()
        u = _first(output)
        T = u.shape[1]
        if self._cp is None or int(self._cp.reshape(-1)[-1]) != self.query:
            raise RuntimeError("readout probe query does not match the step's last cache position")
        self.site = (self._q + u)[0, T - 1].detach().clone()
        self.calls += 1
        mask = torch.zeros((1, T, 1), device=u.device, dtype=u.dtype)
        mask[0, T - 1, 0] = 1
        return _rebuild(output, u + mask * self.delta.to(u.dtype).view(1, 1, -1))

    def __enter__(self):
        self._zero()
        assert_dropout_disabled(self.bundle, [self.layer])
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(layer.register_forward_pre_hook(self._layer_pre, with_kwargs=True))
        self._handles.append(layer.encoder_attn_layer_norm.register_forward_pre_hook(self._ln_pre))
        self._handles.append(layer.encoder_attn.register_forward_hook(self._post))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q = self._cp = None
        return False


# ---- provider ---------------------------------------------------------------------------------

def readout_direction(bundle, *, layer: int, cache, encoded, new_tokens: Sequence[int], start: int,
                      step: int, suppress: Sequence[int], begin: Sequence[int], partition: dict) -> dict:
    """Reference-free D2 direction at the current B query (computed from the PRE-step B cache).

    Returns {direction, status, reason, J, log_PE, log_PM, logits, site, counters, ...}. ``logits``
    and ``site`` are the scratch forward's last-query logits (float32 CPU) and DG-02 site; the
    caller verifies them bitwise against its own clean B step.
    """
    t0 = time.perf_counter()
    params = list(bundle.model.parameters())
    flags = [p.requires_grad for p in params]
    counters = {"autograd_calls": 0, "scratch_forwards": 0}
    if any(p.grad is not None for p in params):
        raise RuntimeError("model parameter has a gradient before readout")
    fp_before = cache_fingerprint(cache)
    if cache.get_seq_length() != int(start):
        raise RuntimeError("readout start does not match the pre-step cache length")
    query = int(start) + len(new_tokens) - 1
    out = {"version": VERSION, "query": query, "direction": None, "status": "invalid", "reason": None}
    scratch = enc = probe = res = None
    try:
        for p in params:
            p.requires_grad_(False)
        with torch.inference_mode(False), torch.enable_grad():
            scratch, enc = clone_scratch(cache, encoded)
            dev = next(bundle.model.parameters()).device
            ids = torch.tensor([list(new_tokens)], device=dev, dtype=torch.long)
            positions = torch.arange(int(start), int(start) + len(new_tokens), device=dev)
            probe = ZeroProbe(bundle, layer, query)
            with probe:
                res = bundle.model(encoder_outputs=enc, decoder_input_ids=ids, past_key_values=scratch,
                                   use_cache=True, cache_position=positions, output_attentions=False,
                                   return_dict=True)
            counters["scratch_forwards"] += 1
            assert_no_site_hooks(bundle)
            z = res.logits[0, -1]
            vals = objective(z, step, suppress, begin, partition)
            out["J"], out["log_PE"], out["log_PM"] = (float(vals[k].detach()) for k in ("J", "log_PE", "log_PM"))
            out["logits"] = z.detach().float().cpu()
            if probe.calls != 1 or probe.site is None:
                raise RuntimeError(f"readout probe fired {probe.calls} times")
            out["site"] = probe.site.float().cpu()
            if not math.isfinite(out["J"]):
                out["reason"] = "nonfinite_objective"
            else:
                (g,) = torch.autograd.grad(vals["J"], probe.delta)
                counters["autograd_calls"] += 1
                out["gradient"] = g.detach().float().cpu()
                geo = tangent_unit(out["gradient"], probe.site)
                out.update({k: v for k, v in geo.items()})
            probe._zero()
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)
        res = scratch = enc = None
        if probe is not None:
            probe.delta = None
        if any(p.grad is not None for p in params):
            raise RuntimeError("model parameter accumulated a gradient during readout")
    if cache_fingerprint(cache) != fp_before:
        raise RuntimeError("persistent cache changed during readout")
    out["counters"] = counters
    out["runtime_sec"] = time.perf_counter() - t0
    return out
