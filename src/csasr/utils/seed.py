"""Deterministic seeding across Python, NumPy and PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch CPU and PyTorch CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for pure-data unit tests
        pass


def numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def torch_generator(seed: int, device: str = "cpu"):
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
