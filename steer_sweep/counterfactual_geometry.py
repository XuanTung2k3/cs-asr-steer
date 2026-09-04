"""Counterfactual representation geometry primitives and stable schemas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .rounds import normalize_direction, mixture_direction


def norm_preserve(original, proposed, eps: float = 1e-8):
    """Scale proposed to the original norm; never changes the input tensors."""
    import torch
    h = torch.as_tensor(original)
    p = torch.as_tensor(proposed, device=h.device, dtype=h.dtype)
    pn = p.norm(dim=-1, keepdim=True)
    hn = h.norm(dim=-1, keepdim=True)
    return p * (hn / pn.clamp_min(eps))


def counterfactual_states(h00, v_nat, v_prompt, beta: float, cn: float, cp: float,
                          eps: float = 1e-8):
    import torch
    h = torch.as_tensor(h00)
    n = normalize_direction(v_nat, eps).to(h)
    p = normalize_direction(v_prompt, eps).to(h)
    d = mixture_direction(n, p, cn, cp, eps).to(h)
    return {
        "base_00": h.clone(),
        "natural_10": norm_preserve(h, h + float(beta) * n, eps),
        "prompt_01": norm_preserve(h, h + float(beta) * p, eps),
        "combined_11": norm_preserve(h, h + float(beta) * d, eps),
    }


@dataclass(frozen=True)
class AxisResult:
    e1: Any
    e2: Any | None
    degenerate: bool
    residual_norm: float


def orthogonal_axes(v_nat, v_prompt, tolerance: float = 1e-6) -> AxisResult:
    import torch
    e1 = normalize_direction(v_nat, tolerance)
    p = normalize_direction(v_prompt, tolerance).to(e1)
    residual = p - torch.dot(p, e1) * e1
    rnorm = float(residual.norm().item())
    if rnorm < tolerance:
        return AxisResult(e1=e1, e2=None, degenerate=True, residual_norm=rnorm)
    return AxisResult(e1=e1, e2=residual / rnorm, degenerate=False, residual_norm=rnorm)


def coordinates(hidden, mu, axes: AxisResult):
    import torch
    h = torch.as_tensor(hidden)
    c = h - torch.as_tensor(mu, device=h.device, dtype=h.dtype)
    x = torch.einsum("...d,d->...", c, axes.e1)
    if axes.degenerate:
        return x, None
    return x, torch.einsum("...d,d->...", c, axes.e2)


def synergy(both: float, natural: float, prompt: float) -> float:
    return float(both - natural - prompt)


def intervention_response(readouts: dict[str, dict[str, float]], baseline: str = "base_00"):
    base = readouts[baseline]
    return {intervention: {name: float(values[name] - base[name]) for name in base}
            for intervention, values in readouts.items() if intervention != baseline}


RECORD_FIELDS = (
    "utterance_id", "dialogue_id", "layer", "downstream_layer", "token_index",
    "state", "outcome", "beta", "mixture", "x", "y", "gold_logp",
    "gold_probability", "gold_rank", "gold_margin", "entropy", "english_mass",
    "mandarin_mass", "local_probe", "prompt_probe", "input_record_hash",
)


def validate_record_schema(record: dict[str, Any]) -> None:
    missing = [x for x in RECORD_FIELDS if x not in record]
    if missing:
        raise ValueError(f"representation record missing fields: {missing}")


def joint_pca_fit_transform(records: np.ndarray, n_components: int = 2):
    """Fit exactly once on the union of conditions and return transform."""
    from sklearn.decomposition import PCA
    x = np.asarray(records, dtype=float)
    model = PCA(n_components=n_components, random_state=0)
    z = model.fit_transform(x)
    return model, z


def downstream_interaction(h00, h10, h01, h11):
    return np.asarray(h11) - np.asarray(h10) - np.asarray(h01) + np.asarray(h00)
