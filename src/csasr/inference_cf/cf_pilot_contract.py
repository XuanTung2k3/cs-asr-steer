"""CPU-only SRC-CF0-P numerical/decision contract, not an inference runner.

Direction construction accepts paired states only. Evaluator predicates accept
aggregate counts only. A future independent auditor must not import these
predicates; it must implement the frozen specification independently.
"""
from __future__ import annotations

import hashlib
import numpy as np


def direction(clean, masked):
    """Historical float64 subtraction, epsilon normalization, float32 storage."""
    h, m = np.asarray(clean, dtype=np.float64), np.asarray(masked, dtype=np.float64)
    if h.shape != (1280,) or m.shape != h.shape:
        raise ValueError("INVALID_STATE_SHAPE")
    if not np.isfinite(h).all() or not np.isfinite(m).all():
        raise ValueError("INVALID_NONFINITE_STATE")
    hn = float(np.linalg.norm(h))
    if hn < 1e-8:
        raise ValueError("INVALID_CLEAN_STATE_NORM")
    delta = h - m
    dn = float(np.linalg.norm(delta))
    if dn < 1e-4:
        return {"status": "ABSTAIN_DEGENERATE", "raw_norm": dn, "vector": None}
    v = (delta / (dn + 1e-6)).astype(np.float32)
    vd = v.astype(np.float64)
    perp = vd - h * float(np.dot(h, vd)) / (hn * hn)
    return {"status": "OK", "raw_norm": dn, "vector": v,
            "direction_norm": float(np.linalg.norm(vd)),
            "tangent_ratio": float(np.linalg.norm(perp) / np.linalg.norm(vd)),
            "raw_tangent_norm": float(np.linalg.norm(delta - h * np.dot(h, delta) / (hn * hn)))}


def random_direction(uid: str, t: int, layer: int):
    key = f"SRC_CF0_P-random-v1|240924|{uid}|{t}|{layer}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    v = np.random.Generator(np.random.PCG64(seed)).standard_normal(1280)
    return (v / np.linalg.norm(v)).astype(np.float32)


def construction_layer(m: dict, config: dict) -> dict:
    """Evaluator-only aggregate layer gates; no per-row labels enter providers."""
    g = config["gate_a"]["per_layer"]
    n = config["gate_a"]["fraction_denominators"]
    thresholds = {
        "valid_targets": n["minimum_valid_target_all_strata"],
        "valid_off_pairs": n["minimum_valid_off_target_all_strata"],
        "EN_confusion_target": g["EN_confusion_target_min"],
        "EN_confusion_target_dialogues": g["EN_confusion_target_dialogues_min"],
        "EN_confusion_joint": g["EN_confusion_joint_reachable_min"],
        "EN_confusion_joint_dialogues": g["EN_confusion_joint_dialogues_min"],
        "EN_correct_joint": g["EN_correct_joint_reachable_min"],
        "EN_correct_joint_dialogues": g["EN_correct_joint_dialogues_min"],
    }
    coverage = all(np.isfinite(m[k]) and m[k] >= v for k, v in thresholds.items())
    cos, ratio = m["median_abs_cosine"], m["median_tangent_ratio"]
    specificity = (m["specificity_pairs"] >= g["specificity_pair_min"] and
                   m["specificity_dialogues"] >= g["specificity_dialogues_min"] and
                   cos is not None and ratio is not None and
                   np.isfinite(cos) and np.isfinite(ratio) and
                   cos <= g["median_absolute_target_off_cosine_max"] and
                   ratio >= g["median_raw_tangent_target_off_norm_ratio_min"])
    return {"construction": bool(coverage), "specificity": bool(specificity),
            "qualified": bool(coverage and specificity)}


def power_pass(m: dict, g: dict) -> bool:
    """Single-arm paired lexical gate; no oracle union or CI significance claim."""
    if not m["qualified_layer"] or not m["energy_valid"]:
        return False
    if not all(np.isfinite(m[k]) for k in
               ("paired_rows", "paired_dialogues", "corrections", "corrected_dialogues")):
        return False
    if (m["paired_rows"] < g["paired_EN_confusion_min"] or
            m["paired_dialogues"] < g["paired_EN_confusion_dialogues_min"] or
            m["corrections"] < g["paired_corrections_min"] or
            m["corrected_dialogues"] < g["corrected_dialogues_min"]):
        return False
    for name in ("random", "off_target"):
        c = m[name]
        if (not all(np.isfinite(c[k]) for k in
                    ("net_corrections", "positive_dialogues", "macro_advantage")) or
                not np.isfinite(c["lodo_net"]).all() or
                c["net_corrections"] < g["net_corrections_over_each_control_min"] or
                c["positive_dialogues"] < g["net_positive_dialogues_over_each_control_min"] or
                c["macro_advantage"] < g["dialogue_macro_accuracy_advantage_over_each_control_min"] or
                not c["lodo_net"] or min(c["lodo_net"]) <= 0):
            return False
    return True


def observed_safety_pass(m: dict, g: dict) -> bool:
    return (m["EN_corruptions"] <= g["selected_arm_EN_correct_corruptions_max"] and
            m["ZH_corruptions"] <= g["selected_arm_ZH_correct_corruptions_max"] and
            m["correct_EOS_promotions"] <= g["selected_arm_correct_state_EOS_promotions_max"])


def terminal(config, *, integrity, construction, specificity, arms):
    """Frozen terminal precedence; labels remain preliminary and safety-limited."""
    prefix = "SRC_CF0_PILOT_"
    if not integrity or (construction and specificity and len(arms) != 8):
        return prefix + "INVALID"
    if not construction:
        return prefix + "CONSTRUCTION_INSUFFICIENT"
    if not specificity:
        return prefix + "DIRECTION_NONSPECIFIC"
    d, g = config["adverse_damage"], config["gate_b"]
    if any(m["EN_corruptions"] >= d["global_any_primary_EN_correct_corruptions_at_least"] or
           m["ZH_corruptions"] >= d["global_any_primary_ZH_correct_corruptions_at_least"] or
           m["correct_EOS_promotions"] >= d["global_any_primary_correct_state_EOS_promotions_at_least"]
           for m in arms):
        return prefix + "OBSERVED_DAMAGE"
    powerful = [m for m in arms if power_pass(m, g)]
    if any(observed_safety_pass(m, g) for m in powerful):
        return prefix + "SIGNAL_SAFETY_UNRESOLVED"
    if powerful:
        return prefix + "OBSERVED_DAMAGE"
    if any(m["qualified_layer"] and m["energy_valid"] and
           m["margin_macro"] >= config["statistics"]["margin_only_mean_nat_min"] and
           m["margin_adjusted_lower"] is not None and m["margin_adjusted_lower"] > 0 for m in arms):
        return prefix + "MARGIN_ONLY"
    return prefix + "CAUSAL_INSUFFICIENT"
