"""Control directions: matched random vectors and the wrong-sign direction."""
from __future__ import annotations

import torch


def random_direction(dim: int, seed: int, device: str = "cpu",
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unit-norm isotropic random direction (matched in scale by reusing s_l)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    r = torch.randn(dim, generator=g, dtype=torch.float32)
    r = r / r.norm()
    return r.to(device=device, dtype=dtype)


def wrong_sign(direction: torch.Tensor) -> torch.Tensor:
    return -direction


def orthogonalized_random(direction: torch.Tensor, seed: int) -> torch.Tensor:
    """Random direction with the language component projected out (diagnostic)."""
    r = random_direction(direction.numel(), seed, device=str(direction.device),
                         dtype=direction.dtype)
    d = direction / direction.norm()
    r = r - (r @ d) * d
    return r / r.norm()
