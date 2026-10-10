"""ST-PROMPT-R1 reference-free adapter (frozen docs/inference_cf/ST_PROMPT_R1_SPEC.md, configs/inference_cf/st_prompt_r1.json).

* runtime projection (runtime_queries + utterances only; strata / targets / competitors never enter);
* complete arm table (24 primary v_prompt arms, 12 matched layer/eta random controls);
* fixed PCG64 random control vectors (seed + byte-hash verified);
* relative-dose single pulse: target = eta * ||float64(native bf16 r)|| computed from the ACTUAL query site, then the
  unchanged ST-LOC0 ``loc0_sites.pulse_action`` (= P2-R solve_scale / scaled_direction / apply_steering, same frozen
  no-edit rule when the solver cannot reach the target within 8 evaluations);
* per-cell energy validity and the CPU geometry ledger used by the pre-pulse coverage gate.
No evaluator, reference, stratum or outcome is imported or accepted here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

LAYERS = (3, 8, 16, 24)


def runtime_projection(panel: dict) -> dict:
    return {"runtime_queries": panel["runtime_queries"], "utterances": panel["utterances"]}


def arm_table(cfg: dict) -> list[dict]:
    arms = [{**a, "family": "prompt"} for a in cfg["primary_arms"]]
    arms += [{"id": r["id"], "layer": r["layer"], "eta": r["eta"], "sign": 1, "family": "random"} for r in cfg["controls"]["random"]]
    return arms


def random_vectors(cfg: dict) -> dict:
    out = {}
    for r in cfg["controls"]["random"]:
        seed = int(hashlib.sha256(r["seed_key"].encode()).hexdigest()[:16], 16)
        if seed != r["seed"]:
            raise ValueError(f"random seed mismatch {r['id']}")
        v = np.random.default_rng(seed).standard_normal(1280)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        if "sha256:" + hashlib.sha256(v.tobytes()).hexdigest() != r["array_bytes_sha256"]:
            raise ValueError(f"random vector hash mismatch {r['id']}")
        out[r["id"]] = v
    return out


def relative_action(query: int, v_fn, eta: float, info: dict, max_rel_sq: float, min_state_norm: float, solve_scale, scaled_direction):
    """Relative-dose action: eta x the query site's own float64 norm, then the unchanged ST-LOC0 pulse action."""
    from csasr.inference_cf.loc0_sites import pulse_action

    def action_fn(q=None, u_source=None, r=None, abs_pos=None):
        hit = (abs_pos == query).nonzero().flatten()
        target = 0.0
        vf = v_fn
        if hit.numel():
            rn = float(torch.linalg.vector_norm(r[0, int(hit[0])].detach().cpu().double()))
            info["r_norm"] = rn
            target = eta * rn
            info["target"] = target
            if not (np.isfinite(rn) and rn > min_state_norm):
                vf = lambda _r: (None, "energy_unreachable")
        return pulse_action(query, vf, target, info, max_rel_sq, solve_scale, scaled_direction)(q=q, u_source=u_source, r=r, abs_pos=abs_pos)
    return action_fn


def cell_checks(eta: float, info: dict, rec: dict | None, before: torch.Tensor, after: torch.Tensor, ecfg: dict) -> dict:
    """Energy / solver / consumed-residual validity of one steered cell (all float64)."""
    realized = rec["edit_norm"]
    rn = info["r_norm"]
    target = info["target"]
    consumed = float(torch.linalg.vector_norm(after.double() - before.double()))
    proposed = float(torch.linalg.vector_norm(info["_proposed"].double() - before.double()))
    actual_eta = realized / rn
    return {"actual_eta": actual_eta, "consumed_edit_norm": consumed, "proposed_edit_norm": proposed, "realized_edit_norm": realized,
            "checks": {"eta_rel": abs(actual_eta / eta - 1) <= ecfg["max_relative_norm_error"],
                       "sq_energy_rel": abs(realized ** 2 / target ** 2 - 1) <= ecfg["max_relative_squared_error"],
                       "solver_matches_hook": abs(info.get("edit_norm", -1.0) - realized) <= ecfg["solver_matches_hook_abs_scaled_tolerance"] * max(1.0, realized),
                       "consumed_vs_proposed": abs(consumed - proposed) <= ecfg["consumed_vs_proposed_relative_norm_tolerance"] * proposed,
                       "proposed_equals_hook": abs(proposed - realized) <= ecfg["consumed_vs_proposed_relative_norm_tolerance"] * realized,
                       "norm_preserved": abs(float(after.double().norm()) - rn) <= ecfg["consumed_vs_proposed_relative_norm_tolerance"] * rn}}


def geometry_cell(r_native: torch.Tensor, v: np.ndarray | None, eta: float, ecfg: dict, solve_scale) -> dict:
    """CPU emulation of one dose cell for the pre-pulse coverage gate (no logits, no outcome)."""
    if v is None:
        return {"status": "tiny_or_nonfinite_direction", "reachable": False}
    rn = float(r_native.double().norm())
    if not (np.isfinite(rn) and rn > ecfg["minimum_state_norm"]):
        return {"status": "energy_unreachable", "reachable": False}
    sol = solve_scale(r_native, torch.from_numpy(np.asarray(v, dtype=np.float32)), eta * rn, max_eval=ecfg["solver_max_evaluations"])
    ok = sol["status"] == "ok" and sol.get("rel_sq_err", 1.0) <= ecfg["max_relative_squared_error"]
    return {"status": sol["status"] if sol["status"] != "ok" or ok else "solver_target_unattainable_within_8_evaluations",
            "reachable": bool(ok), "rel_sq_err": sol.get("rel_sq_err"), "evals": sol.get("evals"), "phi": sol.get("phi")}
