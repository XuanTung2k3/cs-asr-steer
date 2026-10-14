"""Decoder layer-L post-cross-attention, pre-FFN site (ReFT P1).

In ``WhisperDecoderLayer.forward`` (transformers 4.57):

    residual = h_self                        # q: residual entering the cross-attention block
    s = encoder_attn(encoder_attn_layer_norm(q), H_enc)
    h = q + s                                # <- the site: input of final_layer_norm and the FFN residual
    out = h + fc2(act(fc1(final_layer_norm(h))))

``Site`` adds ``delta`` to the cross-attention output, so the site becomes ``h + delta`` on both FFN paths. A zero
``delta`` that requires grad turns the site into a differentiable leaf: dm/d(delta)[b, t] = dm/dh[b, t]. The same
semantics as the ACL-0 SiteHook and the DG-02 exact site (s -> s + u). Never the post-FFN block output.
"""
from __future__ import annotations


class Site:
    def __init__(self, model, layer: int, delta=None, record: bool = False):
        self.layer = model.model.decoder.layers[layer]
        self.delta, self.record = delta, record
        self.q = self.s = self.ffn_in = None
        self.calls = 0
        self.handles = []

    def __enter__(self):
        L = self.layer
        self.handles = [L.encoder_attn_layer_norm.register_forward_pre_hook(self._pre),
                        L.encoder_attn.register_forward_hook(self._attn),
                        L.final_layer_norm.register_forward_pre_hook(self._ffn)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre(self, mod, args):
        if self.record:
            self.q = args[0]

    def _attn(self, mod, args, out):
        self.calls += 1
        s = out[0]
        if self.record:
            self.s = s
        if self.delta is None:
            return None
        return (s + self.delta.to(s.dtype),) + tuple(out[1:])

    def _ffn(self, mod, args):
        if self.record:
            self.ffn_in = args[0]


def n_hooks(model, layer: int) -> int:
    L = model.model.decoder.layers[layer]
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
               for m in (L.encoder_attn_layer_norm, L.encoder_attn, L.final_layer_norm))
