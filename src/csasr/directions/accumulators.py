"""Online, resumable accumulators for direction estimation (guide section 22).

No frame-level activation corpus is ever stored. Per layer we keep

*  the running sum of per-utterance (mu_EN - mu_ZH) deltas,
*  global EN/ZH centroid sums,
*  first and second moments of eligible frames (D and DxD, float64),

which is enough to recover every quantity in section 21 -- including the
projection standard deviation s_l = sqrt(d^T Cov d) -- in a single pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LayerAccumulator:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.delta_sum = np.zeros(dim, dtype=np.float64)
        self.num_utterances = 0
        self.en_sum = np.zeros(dim, dtype=np.float64)
        self.zh_sum = np.zeros(dim, dtype=np.float64)
        self.en_count = 0
        self.zh_count = 0
        self.frame_sum = np.zeros(dim, dtype=np.float64)
        self.frame_sq = np.zeros((dim, dim), dtype=np.float64)
        self.frame_count = 0

    def add_utterance(self, en_frames: np.ndarray, zh_frames: np.ndarray) -> None:
        """``*_frames``: (n, D) float64 activations of one utterance."""
        mu_en = en_frames.mean(axis=0)
        mu_zh = zh_frames.mean(axis=0)
        self.delta_sum += mu_en - mu_zh
        self.num_utterances += 1
        self.en_sum += en_frames.sum(axis=0)
        self.zh_sum += zh_frames.sum(axis=0)
        self.en_count += len(en_frames)
        self.zh_count += len(zh_frames)

    def add_frames(self, frames: np.ndarray, gram: np.ndarray | None = None) -> None:
        """Accumulate moments over eligible frames (both languages)."""
        self.frame_sum += frames.sum(axis=0)
        self.frame_sq += gram if gram is not None else frames.T @ frames
        self.frame_count += len(frames)

    # ---- derived quantities ------------------------------------------
    def mean_delta(self) -> np.ndarray:
        if self.num_utterances == 0:
            raise ValueError("no utterances accumulated")
        return self.delta_sum / self.num_utterances

    def global_delta(self) -> np.ndarray:
        if not self.en_count or not self.zh_count:
            raise ValueError("no EN or ZH frames accumulated")
        return self.en_sum / self.en_count - self.zh_sum / self.zh_count

    def covariance(self) -> np.ndarray:
        if self.frame_count < 2:
            raise ValueError("not enough frames for a covariance estimate")
        mu = self.frame_sum / self.frame_count
        return self.frame_sq / self.frame_count - np.outer(mu, mu)

    def projection_std(self, direction: np.ndarray) -> float:
        var = float(direction @ self.covariance() @ direction)
        return float(np.sqrt(max(var, 0.0)))

    def centroid_projections(self, direction: np.ndarray) -> tuple[float, float]:
        en = float(direction @ (self.en_sum / max(self.en_count, 1)))
        zh = float(direction @ (self.zh_sum / max(self.zh_count, 1)))
        return en, zh

    # ---- (de)serialisation -------------------------------------------
    def state(self) -> dict:
        return {
            "dim": self.dim,
            "delta_sum": self.delta_sum,
            "num_utterances": self.num_utterances,
            "en_sum": self.en_sum,
            "zh_sum": self.zh_sum,
            "en_count": self.en_count,
            "zh_count": self.zh_count,
            "frame_sum": self.frame_sum,
            "frame_sq": self.frame_sq,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_state(cls, st) -> "LayerAccumulator":
        acc = cls(int(st["dim"]))
        acc.delta_sum = np.asarray(st["delta_sum"], dtype=np.float64)
        acc.num_utterances = int(st["num_utterances"])
        acc.en_sum = np.asarray(st["en_sum"], dtype=np.float64)
        acc.zh_sum = np.asarray(st["zh_sum"], dtype=np.float64)
        acc.en_count = int(st["en_count"])
        acc.zh_count = int(st["zh_count"])
        acc.frame_sum = np.asarray(st["frame_sum"], dtype=np.float64)
        acc.frame_sq = np.asarray(st["frame_sq"], dtype=np.float64)
        acc.frame_count = int(st["frame_count"])
        return acc


class DirectionAccumulatorSet:
    """Multi-layer accumulator with idempotent, resumable checkpointing."""

    def __init__(self, layers, dim: int, tag: str = "default"):
        self.layers = list(layers)
        self.dim = int(dim)
        self.tag = tag
        self.acc = {l: LayerAccumulator(dim) for l in self.layers}
        self.processed: set[str] = set()

    def has(self, utterance_id: str) -> bool:
        return utterance_id in self.processed

    def mark(self, utterance_id: str) -> None:
        self.processed.add(utterance_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"__layers__": np.array(self.layers), "__dim__": np.array(self.dim)}
        for l, a in self.acc.items():
            for k, v in a.state().items():
                payload[f"L{l}__{k}"] = np.asarray(v)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(path)
        path.with_suffix(".processed.json").write_text(
            json.dumps(sorted(self.processed)), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "DirectionAccumulatorSet":
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        layers = [int(x) for x in z["__layers__"]]
        dim = int(z["__dim__"])
        out = cls(layers, dim)
        for l in layers:
            st = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"L{l}__")}
            st["dim"] = dim
            out.acc[l] = LayerAccumulator.from_state(st)
        proc = path.with_suffix(".processed.json")
        if proc.exists():
            out.processed = set(json.loads(proc.read_text(encoding="utf-8")))
        return out

    @classmethod
    def load_or_new(cls, path: str | Path, layers, dim: int) -> "DirectionAccumulatorSet":
        path = Path(path)
        if path.exists():
            obj = cls.load(path)
            if set(obj.layers) == set(layers) and obj.dim == dim:
                return obj
        return cls(layers, dim)

    def summary(self) -> dict:
        return {
            str(l): {
                "num_utterances": a.num_utterances,
                "en_frames": a.en_count,
                "zh_frames": a.zh_count,
                "eligible_frames": a.frame_count,
            }
            for l, a in self.acc.items()
        }
