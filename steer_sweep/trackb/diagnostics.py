"""Section 4.3 diagnostics: did the optimizer move toward the constructed
directions, or away from them?

Both outcomes are publishable and this code must not bias toward either.

  converged toward span{v_nat, v_prompt}
      the interpretability work is validated, and Track A's failure is a
      MAGNITUDE problem rather than a direction problem;
  diverged
      this explains Track A's failure: the constructed direction was not the
      direction the model needed.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(a @ b / (na * nb))


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> list[float]:
    """Principal angles in degrees between two subspaces given by row bases."""
    qa, _ = np.linalg.qr(np.asarray(basis_a, dtype=np.float64).T)
    qb, _ = np.linalg.qr(np.asarray(basis_b, dtype=np.float64).T)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return [float(np.degrees(np.arccos(s))) for s in singular]


def theta_vector(state: dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten a module state dict into one parameter vector."""
    return np.concatenate([v.detach().cpu().numpy().reshape(-1)
                           for _, v in sorted(state.items())])


def intervention_diagnostics(module, theta_init: dict[str, torch.Tensor], *,
                             v_nat: np.ndarray, v_prompt: np.ndarray) -> dict[str, Any]:
    """cos to each constructed direction, ||theta - theta_init||, and angles."""
    final_state = {k: v.detach().float().cpu() for k, v in module.state_dict().items()}
    theta_f = theta_vector(final_state)
    theta_0 = theta_vector(theta_init)
    drift = float(np.linalg.norm(theta_f - theta_0)) if theta_f.shape == theta_0.shape \
        else float("nan")

    learned = module.subspace().cpu().numpy()          # (rank, d)
    constructed = np.stack([np.asarray(v_nat, dtype=np.float64).reshape(-1),
                            np.asarray(v_prompt, dtype=np.float64).reshape(-1)])
    angles = principal_angles(learned, constructed)

    # cos(theta_final, v) is reported on the leading learned component, which is
    # the object a direction can be compared to; the whole flattened parameter
    # vector has no shared coordinate system with a d-dimensional direction.
    leading = learned[0]
    out = {
        "cos_theta_final_v_nat": _cos(leading, constructed[0]),
        "cos_theta_final_v_prompt": _cos(leading, constructed[1]),
        "cos_v_nat_v_prompt": _cos(constructed[0], constructed[1]),
        "theta_drift_l2": drift,
        "theta_init_norm": float(np.linalg.norm(theta_0)),
        "theta_final_norm": float(np.linalg.norm(theta_f)),
        "principal_angles_deg": angles,
        "min_principal_angle_deg": float(min(angles)) if angles else float("nan"),
        "mean_principal_angle_deg": float(np.mean(angles)) if angles else float("nan"),
        "subspace_rank": int(learned.shape[0]),
        "interpretation_key": {
            "small principal angles": "the learned subspace converged TOWARD "
                                      "span{v_nat, v_prompt}",
            "angles near 90 deg": "the learned subspace is orthogonal to the "
                                  "constructed one -- it diverged",
        },
    }
    return out


def summarise_convergence(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the direction verdict across arms/seeds without editorialising."""
    angles = [d.get("mean_principal_angle_deg") for d in diagnostics
              if d and np.isfinite(d.get("mean_principal_angle_deg", np.nan))]
    if not angles:
        return {"verdict": "UNDETERMINED", "reason": "no principal angles recorded"}
    mean_angle = float(np.mean(angles))
    # a random rank-2 subspace of R^1280 sits at essentially 90 degrees; the
    # reference point is that null, not zero
    if mean_angle < 60.0:
        verdict = "CONVERGED_TOWARD"
    elif mean_angle > 80.0:
        verdict = "DIVERGED"
    else:
        verdict = "INTERMEDIATE"
    return {
        "verdict": verdict,
        "mean_principal_angle_deg": mean_angle,
        "n_runs": len(angles),
        "random_subspace_reference_deg": 90.0,
        "note": ("a random rank-2 subspace of a 1280-dimensional space sits at "
                 "essentially 90 degrees, so 90 is the null, not zero"),
    }


def seed_range(values: Sequence[float]) -> dict[str, float]:
    """Mean and range, the form B1 must be reported in."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not arr.size:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "min": float(arr.min()),
            "max": float(arr.max()), "n": int(arr.size)}


def beats_range(value: float, rng: dict[str, float]) -> str:
    """Is B2 outside the B1 seed range? Lower MER is better."""
    if not np.isfinite(value) or not rng.get("n"):
        return "UNDETERMINED"
    if value < rng["min"]:
        return "BEATS_RANGE"
    if value > rng["max"]:
        return "WORSE_THAN_RANGE"
    return "WITHIN_RANGE"
