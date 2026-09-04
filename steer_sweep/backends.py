"""Model backend boundary. Qwen is intentionally lazy and never imported at module load."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class MockBackend:
    model_id: str = "mock"
    loaded: bool = False

    def load(self):
        self.loaded = True
        return self


class LazyQwenBackend:
    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-1.7B", loader: Callable[..., Any] | None = None):
        self.model_id = model_id
        self.loader = loader
        self.model = None

    def load(self, *, allow_initialize: bool = False, **kwargs):
        if not allow_initialize:
            raise RuntimeError("Qwen backend is lazy; explicit allow_initialize is required")
        if self.loader is None:
            raise RuntimeError("Qwen loader is not configured; no weights were downloaded")
        self.model = self.loader(self.model_id, **kwargs)
        return self.model

    @property
    def loaded(self) -> bool:
        return self.model is not None
