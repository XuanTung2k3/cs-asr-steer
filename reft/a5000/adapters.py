"""A5000 reconstruction: low-rank affine adaptation AFTER Whisper encoder's final layer norm.

The frozen encoder output (including padded frames) is adapted once, before decoder cross-attention.
All-frame policy matches Whisper's full 1500-frame interface; audio masks are passed unchanged.
No claim of bit-exact reproduction of missing H100 encoder adapter source.
"""
import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from reft.adapters import LoReFTIntervention


class EncoderAdapter(nn.Module):
    def __init__(self, method, d=1280, rank=2, seed=42):
        super().__init__()
        self.method, self.d, self.rank = method, d, rank
        if method == 'ereft':
            self.reft = LoReFTIntervention(d, rank, seed)
        elif method == 'lra':
            # Same seeded orthonormal initial output basis as E-ReFT. A=c=0 gives exact identity;
            # A and c have a learning path from step one; B learns once A/c have moved.
            g = torch.Generator().manual_seed(seed)
            q, r = torch.linalg.qr(torch.randn(d, rank, generator=g, dtype=torch.float64))
            basis = (q * torch.sign(r.diagonal())).float()
            self.A = nn.Parameter(torch.zeros(rank, d))
            self.B = nn.Parameter(basis)
            self.c = nn.Parameter(torch.zeros(rank))
        else:
            raise ValueError(method)
        self.calls = 0
        self._active = False

    def forward(self, h, alpha=1.0):
        if h.shape[-1] != self.d:
            raise ValueError('encoder hidden dimension mismatch')
        if alpha == 0:
            return h
        self.calls += 1
        x = h.float()
        delta = self.reft(x) if self.method == 'ereft' else F.linear(F.linear(x, self.A, self.c), self.B)
        return h + (alpha * delta).to(h.dtype)

    @contextlib.contextmanager
    def active(self, model, alpha=1.0):
        """Native HF forward/generate integration. Mutually exclusive with manual cached application.

        Hook the complete encoder output, NOT its last block or pre-layer-norm tensor.
        Encoder outputs supplied by a caller are already computed and do not trigger this hook.
        """
        if self._active:
            raise RuntimeError('duplicate encoder adapter context')
        self._active = True
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (self(output[0], alpha), *output[1:])
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=self(output.last_hidden_state, alpha),
                                   hidden_states=output.hidden_states, attentions=output.attentions)
        handle = model.model.encoder.register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()
            self._active = False

    def count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
