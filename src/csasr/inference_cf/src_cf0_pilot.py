"""SRC-CF0-P reference-free pilot helpers (configs/inference_cf/src_cf0_pilot.json; docs/inference_cf/SRC_CF0_PILOT_*.md;
freeze 2d23574).

* ``runtime_projection``: the allowlisted runtime view of SRC_CF0_PANEL.json (config.firewall keys only; the offline
  ``coverage_review_only`` strata and every evaluator field are dropped) with sealed S1 target / off-target regions.
* ``direction``: v_AC = float32((float64(h_clean) - float64(h_mask)) / (||.|| + 1e-6)) and its frozen geometry
  (raw norm, serialized norm, tangent ratio, raw tangent norm, direction/residual cosine). Same arithmetic as the frozen
  ``cf_pilot_contract.direction``; nonfinite / wrong shape / tiny clean state are critical INVALID, a raw delta below
  1e-4 abstains.
* ``pair_geometry``: signed / absolute target-off cosine and the target/off RAW tangent-norm ratio (zero off tangent
  abstains, never infinity).
* ``random_direction``: one PCG64 isotropic vector per UID / t / layer (same vector for both signs and doses).
* ``reach``: P-A native-BF16 reachability of one relative dose via the unchanged P2-R ``solve_scale`` (no forward).
* ``pulse_checks``: the five frozen per-cell energy / consumption checks of one actual pulse (+ descriptive extras).
* ``FFNInputProbe``: passive pre-hook on each layer's ``final_layer_norm`` (the tensor the FFN actually consumes).

No reference, stratum, acceptable-token set, competitor, oracle region or outcome is accepted by any function here.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

LAYERS = (16, 24)
ETAS = (0.15, 0.30)
SIGNS = (1, -1)
FAMILIES = ("target", "off", "random")
DIM = 1280
EPS = 1e-6
MIN_RAW = 1e-4
MIN_CLEAN = 1e-8
CB = [50258, 50260, 50360, 50364]
FORBIDDEN_TEXT = ("stratum", "strata", "EN-confusion", "EN-correct", "ZH-correct", "target_ids", "acceptable", "competitor",
                  "coverage_review", "reference", "oracle", "gold", "mms", "ctc_", "per_row_goodness", "evaluation_membership")


def arm_id(layer: int, eta: float, sign: int, family: str = "target") -> str:
    base = f"L{int(layer)}_eta{float(eta):.2f}_{'plus' if int(sign) > 0 else 'minus'}"
    return base if family == "target" else f"{family}_{base}"


def arms() -> list[dict]:
    """The complete 24-cell matrix: 8 primary (target) arms, 8 off-target and 8 random mirrors."""
    return [{"id": arm_id(l, e, s, f), "family": f, "layer": l, "eta": e, "sign": s, "config": arm_id(l, e, s)}
            for f in FAMILIES for l in LAYERS for e in ETAS for s in SIGNS]


# ---- runtime projection -------------------------------------------------------------------------------------------

def _keys(o, out: set) -> set:
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(str(k))
            _keys(v, out)
    elif isinstance(o, list):
        for v in o:
            _keys(v, out)
    return out


def forbidden_hits(obj) -> list[str]:
    """Forbidden key names anywhere in obj, plus forbidden evaluator words anywhere in its canonical text."""
    import json
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return sorted({f for f in FORBIDDEN_TEXT if f.lower() in text.lower()})


def runtime_projection(panel: dict, cfg: dict) -> dict:
    """Allowlisted runtime view. Content tokens are truncated to each utterance's largest query t."""
    fw = cfg["firewall"]
    qk, uk, rk = fw["runtime_query_keys"], fw["runtime_utterance_keys"], fw["region_keys"]
    tmax: dict[str, int] = {}
    queries = []
    for q in panel["runtime_queries"]:
        queries.append({k: q[k] for k in qk})
        tmax[q["utterance_id"]] = max(tmax.get(q["utterance_id"], 0), int(q["t"]))
    utts = []
    for u in panel["utterances"]:
        rec = {}
        for k in uk:
            if k == "content_prefix_tokens":
                rec[k] = [int(x) for x in u["baseline_content_tokens"][:tmax[u["utterance_id"]]]]
            elif k == "max_t":
                rec[k] = tmax[u["utterance_id"]]
            else:
                rec[k] = u[k]
        utts.append(rec)
    inv = panel["sealed_region_inventory"]
    if [(r["j"], r["utterance_id"], r["t"]) for r in inv] != [(j, q["utterance_id"], q["t"]) for j, q in enumerate(queries)]:
        raise ValueError("region inventory order differs from runtime queries")
    regions = [{k: r[k] for k in rk} for r in inv]
    proj = {"runtime_queries": queries, "utterances": utts, "regions": regions}
    if set(_keys(proj["runtime_queries"], set())) != set(qk):
        raise ValueError("runtime query keys")
    hits = forbidden_hits(proj)
    if hits:
        raise ValueError(f"forbidden evaluator content in runtime projection: {hits}")
    return proj


def mask_groups(projection: dict) -> dict:
    """Distinct (utterance, bounds) masks -> sorted [(t, j, role)] members (target / off_target with status OK)."""
    out: dict = {}
    for j, (q, r) in enumerate(zip(projection["runtime_queries"], projection["regions"])):
        for role in ("target", "off_target"):
            if r[role]["status"] == "OK":
                out.setdefault((q["utterance_id"], tuple(int(b) for b in r[role]["bounds"])), []).append((int(q["t"]), j, role))
    return {k: sorted(v) for k, v in out.items()}


# ---- direction and geometry ------------------------------------------------------------------------------------------

def direction(clean, masked) -> dict:
    """Frozen v_AC from two native-BF16 states (given exactly as float32). Raises ValueError('INVALID_...')."""
    h = np.asarray(clean, dtype=np.float64)
    m = np.asarray(masked, dtype=np.float64)
    if h.shape != (DIM,) or m.shape != (DIM,):
        raise ValueError("INVALID_STATE_SHAPE")
    if not (np.isfinite(h).all() and np.isfinite(m).all()):
        raise ValueError("INVALID_NONFINITE_STATE")
    hn = float(np.linalg.norm(h))
    if hn < MIN_CLEAN:
        raise ValueError("INVALID_CLEAN_STATE_NORM")
    delta = h - m
    dn = float(np.linalg.norm(delta))
    if dn < MIN_RAW:
        return {"status": "ABSTAIN_DEGENERATE", "raw_norm": dn, "vector": None}
    v = (delta / (dn + EPS)).astype(np.float32)
    vd = v.astype(np.float64)
    vn = float(np.linalg.norm(vd))
    perp = vd - h * float(np.dot(h, vd)) / (hn * hn)
    rperp = delta - h * float(np.dot(h, delta)) / (hn * hn)
    return {"status": "OK", "raw_norm": dn, "vector": v, "direction_norm": vn, "tangent_ratio": float(np.linalg.norm(perp) / vn),
            "raw_tangent_norm": float(np.linalg.norm(rperp)), "cos_v_h": float(np.dot(vd, h) / (vn * hn)),
            "vector_sha256": "sha256:" + hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}


def unit_geometry(v, h) -> dict:
    vd, hd = np.asarray(v, dtype=np.float64), np.asarray(h, dtype=np.float64)
    vn, hn = float(np.linalg.norm(vd)), float(np.linalg.norm(hd))
    perp = vd - hd * float(np.dot(hd, vd)) / (hn * hn)
    return {"direction_norm": vn, "tangent_ratio": float(np.linalg.norm(perp) / vn), "cos_v_h": float(np.dot(vd, hd) / (vn * hn))}


def pair_geometry(t: dict, o: dict) -> dict:
    """Target vs off-target (both numerically valid). Cosine of the serialized float32 vectors in float64."""
    a, b = t["vector"].astype(np.float64), o["vector"].astype(np.float64)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    ratio = None if not o["raw_tangent_norm"] > 0 else float(t["raw_tangent_norm"] / o["raw_tangent_norm"])
    return {"cos": c, "abs_cos": abs(c), "raw_tangent_ratio": ratio, "ratio_status": "OK" if ratio is not None else "ABSTAIN_ZERO_OFF_TANGENT"}


def random_seed(uid: str, t: int, layer: int) -> int:
    return int(hashlib.sha256(f"SRC_CF0_P-random-v1|240924|{uid}|{int(t)}|{int(layer)}".encode()).hexdigest()[:16], 16)


def random_direction(uid: str, t: int, layer: int) -> np.ndarray:
    v = np.random.Generator(np.random.PCG64(random_seed(uid, t, layer))).standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---- energy ---------------------------------------------------------------------------------------------------------

def energy_cfg(cfg: dict) -> dict:
    """Frozen pilot energy tolerances in the key names of the reused ST-PROMPT-R1 helpers (values from the pilot config)."""
    e = cfg["energy"]
    return {"max_relative_norm_error": e["relative_chord_error_max"], "max_relative_squared_error": e["relative_squared_energy_error_max"],
            "max_pairwise_relative_squared_error": e["paired_relative_squared_energy_difference_max"],
            "consumed_vs_proposed_relative_norm_tolerance": e["consumed_vs_proposed_relative_error_max"],
            "solver_matches_hook_abs_scaled_tolerance": e["solver_vs_hook_absolute_error_factor"],
            "minimum_state_norm": e["minimum_state_norm"], "solver_max_evaluations": e["maximum_evaluations"]}


def reach(h_native, v: np.ndarray | None, eta: float, sign: int, ecfg: dict, solve_scale) -> dict:
    """P-A prediction for one (direction, sign, dose): CPU-float64 chord geometry + exact native-BF16 repair emulation on
    the site device (``h_native``: the native BF16 state tensor). No model forward."""
    import torch
    if v is None:
        return {"status": "no_direction", "reachable": False}
    rn = float(torch.linalg.vector_norm(h_native.detach().cpu().double()))
    if not (math.isfinite(rn) and rn > ecfg["minimum_state_norm"]):
        return {"status": "energy_unreachable", "reachable": False, "r_norm": rn}
    target = float(eta) * rn
    vv = torch.from_numpy((int(sign) * np.asarray(v, dtype=np.float32)).astype(np.float32))
    sol = solve_scale(h_native, vv, target, max_eval=ecfg["solver_max_evaluations"])
    rec = {"r_norm": rn, "target": target, "solver_status": sol["status"], "s": sol.get("s"), "evals": sol.get("evals"),
           "emulated_edit_norm": sol.get("edit_norm"), "rel_sq_err": sol.get("rel_sq_err"), "phi": sol.get("phi")}
    if sol["status"] != "ok":
        rec.update(status=sol["status"], reachable=False)
        return rec
    if not sol["rel_sq_err"] <= ecfg["max_relative_squared_error"]:
        rec.update(status="solver_target_unattainable_within_8_evaluations", reachable=False)
        return rec
    e = float(sol["edit_norm"])
    rec["actual_eta"] = e / rn
    rec["eta_rel_ok"] = abs(e / target - 1.0) <= ecfg["max_relative_norm_error"]
    rec["sq_energy_ok"] = abs(e * e / (target * target) - 1.0) <= ecfg["max_relative_squared_error"]
    rec["reachable"] = bool(rec["eta_rel_ok"] and rec["sq_energy_ok"])
    rec["status"] = "ok" if rec["reachable"] else "energy_check_failed"
    return rec


def paired_energy_ok(energies: list[float], tol: float) -> bool:
    """Max / min squared edit energy within ``tol`` (relative) among paired cells; an empty set is vacuous."""
    sq = [float(x) ** 2 for x in energies]
    return (not sq) or (min(sq) > 0 and max(sq) / min(sq) - 1.0 <= tol)


FROZEN_CHECKS = ("eta_rel", "sq_energy_rel", "solver_matches_hook", "consumed_vs_proposed")
DESCRIPTIVE_CHECKS = ("proposed_equals_hook", "norm_preserved")


def pulse_checks(eta: float, info: dict, rec: dict, before, after, ecfg: dict) -> dict:
    """The reused ST-PROMPT-R1 ``cell_checks`` arithmetic; frozen pilot validity = the four FROZEN_CHECKS (the pairwise
    energy guard is applied across cells). BF16 norm rounding and proposed-vs-hook equality are descriptive only."""
    from csasr.inference_cf.prompt_r1 import cell_checks
    ch = cell_checks(eta, info, rec, before, after, ecfg)
    out = {k: v for k, v in ch.items() if k != "checks"}
    out["checks"] = {k: bool(ch["checks"][k]) for k in FROZEN_CHECKS}
    out["descriptive"] = {k: bool(ch["checks"][k]) for k in DESCRIPTIVE_CHECKS}
    out["norm_rounding_rel"] = abs(float(after.double().norm()) - info["r_norm"]) / info["r_norm"]
    out["valid"] = all(out["checks"].values())
    return out


# ---- passive pre-FFN consumption probe -------------------------------------------------------------------------------

class FFNInputProbe:
    """Records the LAST-position input of each layer's ``final_layer_norm`` (= the residual the FFN consumes)."""

    def __init__(self, bundle, layers):
        self.bundle, self.layers = bundle, [int(x) for x in layers]
        self.states: dict = {}
        self._handles: list = []

    def _hook(self, layer):
        def hook(_mod, args):
            self.states[layer] = args[0][:, -1].detach().float()
            return None
        return hook

    def __enter__(self):
        for l in self.layers:
            self._handles.append(self.bundle.decoder_layer(l).final_layer_norm.register_forward_pre_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def ffn_hooks_clear(bundle) -> bool:
    return all(len(bundle.decoder_layer(l).final_layer_norm._forward_pre_hooks) == 0 for l in range(len(bundle.model.model.decoder.layers)))
