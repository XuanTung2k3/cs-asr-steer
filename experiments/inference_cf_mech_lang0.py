#!/usr/bin/env python
"""MECH-LANG0: offline direction-alignment, lexical-decision-gap and D2-selectivity diagnosis (CPU only, read-only).

Retrospective, exploratory, post-hoc analysis of already-sealed historical artifacts on the exposed D-dev-select 180-query
panel (protocol: docs/inference_cf/MECH_LANG0_PROTOCOL.md). Never loads a model or tokenizer, never runs a forward or an
autograd call, never constructs a vector from outcomes and never selects a direction, dose, sign, layer or threshold.
Evaluator-only references (P2-RJ acceptable sets / competitors / strata, P2-DIR evaluator gradient) are used only for
evaluator-style description. The six D2-selectivity features are built only from reference-free runtime records.

    python experiments/inference_cf_mech_lang0.py [--out results/inference_cf/mech_lang0]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import digest, file_hash

EOS = 50257
LANG_EN, LANG_ZH, TASK_TRANSLATE, TASK_TRANSCRIBE = 50259, 50260, 50359, 50360
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
D = 1280
NEAR = (0.5, 1.0)
CROSS = "POST_CROSS_ATTENTION_PRE_FFN"
MODEL_DIR = Path("/mnt/data/tungnx/whisper-large-v3")

P = {
    "panel_loc0": "docs/inference_cf/ST_LOC0_PANEL.json",
    "panel_r1": "docs/inference_cf/ST_PROMPT_R1_PANEL.json",
    "panel_s1": "docs/inference_cf/S1_PANEL.json",
    "panel_src": "docs/inference_cf/SRC_CF0_PANEL.json",
    "positions": "results/inference_cf/p2rj/positions.json",
    "src_runtime": "results/inference_cf/src_cf0_pilot/run1/runtime.json",
    "p2dir_rows": "results/inference_cf/p2dir/exp1_run1/rows",
    "p2dir_states": "results/inference_cf/p2dir/extract_run1/states.npz",
    "loc0": "results/inference_cf/st_loc0/run1",
    "r1": "results/inference_cf/st_prompt_r1/runA",
    "src": "results/inference_cf/src_cf0_pilot/run1",
    "s1": "results/inference_cf/s1/run1",
    "src_config": "configs/inference_cf/src_cf0_pilot.json",
}

# Historical totals reconstructed independently (from the committed reports; checks only, never inputs).
HISTORICAL = {
    "D2_EN_confusion_corrections": 5, "D2_ZH_correct_corruptions": 8, "D2_EN_correct_corruptions": 1, "D2_EN_confusion_top1_changes": 43,
    "D2_S1_eligible_corrections": 3, "D2_S1_eligible_ZH_corruptions": 3,
    "loc0_v_prompt_corrections": {"L16_plus": 0, "L16_minus": 0, "L24_plus": 0, "L24_minus": 0},
    "loc0_v_prompt_corruptions_EN_ZH": {"L16_plus": (1, 0), "L16_minus": (0, 0), "L24_plus": (2, 1), "L24_minus": (1, 1)},
    "r1_corrections": {"L24_eta0.30_minus": 1, "L24_eta0.45_plus": 1, "L24_eta0.45_minus": 1},  # every other arm 0
    "r1_corruptions_EN_ZH": {"L16_eta0.45_minus": (2, 1), "L24_eta0.30_minus": (1, 2), "L24_eta0.45_minus": (5, 3)},
    "src_corrections_per_arm": 0,
    "src_corruptions_EN_ZH": {"L16_eta0.15_plus": (0, 0), "L16_eta0.15_minus": (0, 0), "L16_eta0.30_plus": (1, 1), "L16_eta0.30_minus": (2, 0),
                              "L24_eta0.15_plus": (1, 1), "L24_eta0.15_minus": (2, 1), "L24_eta0.30_plus": (1, 1), "L24_eta0.30_minus": (3, 2)},
}

CONSUMED: dict[str, str] = {}


def _rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT)) if str(Path(p).resolve()).startswith(str(ROOT)) else str(p)


def J(rel: str):
    path = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
    CONSUMED.setdefault(_rel(path), file_hash(path))
    return json.loads(path.read_text())


def NPZ(rel: str) -> dict:
    path = ROOT / rel
    CONSUMED.setdefault(_rel(path), file_hash(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---- pure numerical helpers (unit-tested) ------------------------------------------------------------------------------

def unpack(a: np.ndarray) -> np.ndarray:
    """bf16 bit pattern stored as int16 -> exact float32."""
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def processed(z: np.ndarray, t: int, sup, beg) -> np.ndarray:
    x = np.asarray(z, dtype=np.float64).copy()
    x[list(sup)] = -np.inf
    if t == 0:
        x[list(beg)] = -np.inf
    return x


def lse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else None


def tangent(g, h) -> np.ndarray:
    """g_tan = g - h (h.g)/(h.h), float64."""
    g, h = np.asarray(g, dtype=np.float64), np.asarray(h, dtype=np.float64)
    if g.shape != h.shape:
        raise ValueError("gradient / state shape mismatch")
    return g - h * (float(h @ g) / float(h @ h))


def lex(zp: np.ndarray, Y, comp: int | None = None) -> dict:
    """Gap G = max_{k not in Y} z_k - max_{k in Y} z_k on processed logits; top-1 = argmax with the lowest ID on ties;
    best-reference rank with the lowest-ID tie order; fixed-competitor margin m = lse z(Y) - z(c*)."""
    Y = sorted({int(y) for y in Y})
    ref = np.asarray(Y, dtype=np.int64)
    mask = np.zeros(zp.shape[0], dtype=bool)
    mask[ref] = True
    best_ref = float(np.max(zp[ref]))
    best_non = float(np.max(np.where(mask, -np.inf, zp)))
    top1 = int(np.argmax(zp))
    bid = min(int(y) for y in Y if zp[int(y)] == best_ref)
    rank = 1 + int(np.sum(zp > best_ref)) + int(np.sum(zp[:bid] == best_ref))
    out = {"G": best_non - best_ref, "top1": top1, "in_ref": bool(mask[top1]), "rank_ref": rank}
    if comp is not None:
        out["m"] = lse(zp[ref]) - float(zp[int(comp)])
    return out


def tie_consistent(r: dict, Y) -> bool:
    """top-1 in Y  <=>  G < 0, or G == 0 and the lowest tied maximal ID is a reference token."""
    if r["G"] < 0:
        return r["in_ref"]
    if r["G"] > 0:
        return not r["in_ref"]
    return r["in_ref"] == (r["top1"] in set(int(y) for y in Y))


def summarize(vals) -> dict:
    """vals: list of (dialogue_id, value or None). Missing values are counted, never imputed."""
    ok = [(d, float(v)) for d, v in vals if v is not None and math.isfinite(float(v))]
    out = {"n": len(ok), "missing": len(vals) - len(ok)}
    if not ok:
        return out
    x = np.array([v for _, v in ok])
    by = defaultdict(list)
    for d, v in ok:
        by[d].append(v)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    out.update({"mean": float(x.mean()), "median": float(np.median(x)), "q1": float(q1), "q3": float(q3), "min": float(x.min()),
                "max": float(x.max()), "frac_pos": float(np.mean(x > 0)), "dialogues": len(by),
                "dialogue_macro": float(np.mean([np.mean(v) for v in by.values()]))})
    return out


def auc(a, b) -> float | None:
    """Descriptive P(A > B) + 0.5 P(A = B) over all pairs; None if either side is empty."""
    a = [float(x) for x in a if x is not None]
    b = [float(x) for x in b if x is not None]
    if not a or not b:
        return None
    s = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return s / (len(a) * len(b))


def finite(o):
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


# ---- identity / joins --------------------------------------------------------------------------------------------------

KEY = ("utterance_id", "t", "dialogue_id", "absolute_query", "content_prefix_sha256")


def master_panel() -> tuple[list[dict], dict]:
    """Master order = ST-LOC0 runtime panel; every other panel / projection / P2-RJ list must agree on its shared keys."""
    base = J(P["panel_loc0"])["runtime_queries"]
    checks = {}
    for name in ("panel_r1", "panel_s1", "panel_src"):
        other = J(P[name])["runtime_queries"]
        shared = [k for k in KEY if k in other[0] and k in base[0]]
        checks[f"{name}_order_equal"] = len(other) == 180 and all(tuple(a[k] for k in shared) == tuple(b[k] for k in shared) for a, b in zip(base, other))
        checks[f"{name}_keys"] = shared
    proj = J(P["src_runtime"])["projection"]["runtime_queries"]
    shared = [k for k in KEY if k in proj[0]]
    checks["src_projection_order_equal"] = all(tuple(a[k] for k in shared) == tuple(b[k] for k in shared) for a, b in zip(base, proj))
    pos = J(P["positions"])["positions"]
    checks["p2rj_positions_order_equal"] = len(pos) == 180 and all(
        (a["utterance_id"], a["t"], a["dialogue_id"]) == (p["utterance_id"], int(p["t"]), p["dialogue_id"]) for a, p in zip(base, pos))
    checks["absolute_query_is_4_plus_t_minus_1"] = all(q["absolute_query"] == 4 + q["t"] - 1 for q in base)
    checks["keys_unique"] = len({(q["utterance_id"], q["t"]) for q in base}) == 180
    return base, checks


def index_map(Q) -> dict:
    return {(q["utterance_id"], int(q["t"])): j for j, q in enumerate(Q)}


def join_rows(rows: dict, Q, label: str) -> tuple[dict, dict]:
    """rows: {(utterance_id, t): payload with optional 'j'}; returns {j: payload} and coverage / consistency counts."""
    imap = index_map(Q)
    out, bad = {}, []
    for k, v in rows.items():
        j = imap.get(k)
        if j is None or (v.get("j") is not None and int(v["j"]) != j):
            bad.append(list(k))
            continue
        out[j] = v
    return out, {"source": label, "joined": len(out), "unmatched_or_index_mismatch": bad}


# ---- loaders -----------------------------------------------------------------------------------------------------------

def load_p2dir(Q):
    rows = {}
    for i in range(80):
        r = J(f"{P['p2dir_rows']}/{i:03d}.json")
        e = J(f"{P['p2dir_rows']}/{i:03d}_eval.json")
        V = NPZ(f"{P['p2dir_rows']}/{i:03d}.npz")
        E = NPZ(f"{P['p2dir_rows']}/{i:03d}_eval.npz")
        if r["identity"] != e["identity"]:
            raise SystemExit("p2dir row/eval identity")
        for ts, rec in r["positions"].items():
            t = int(ts)
            pre = f"t{t}_"
            rows[(r["identity"], t)] = {"hb": V[pre + "hb"], "he": V[pre + "he"], "D0": V[pre + "D0"], "D2": V[pre + "D2"],
                                        "g_readout": V[pre + "g_readout"], "g_margin": E[pre + "g_margin"],
                                        "post_D2": V[pre + "post_D2"], "post_D0": V[pre + "post_D0"],
                                        "prov": rec["directions"]["D2"]["provenance"], "d2_status": rec["directions"]["D2"]["status"],
                                        "eval_values": e["positions"][ts]["values"], "eval_site_bitwise": e["positions"][ts]["site_bitwise"]}
    return join_rows(rows, Q, "p2dir_exp1")


def load_loc0(Q):
    """ST-LOC0 per-utterance barrier + pulse files merged (both carry 'q' = panel index); calibration arrays by panel index."""
    rows = {}
    for i in range(80):
        for part in ("barrier", "pulses"):
            meta = J(f"{P['loc0']}/{part}/{i:03d}.json")
            A = NPZ(f"{P['loc0']}/{part}/{i:03d}_arrays.npz")
            L = NPZ(f"{P['loc0']}/{part}/{i:03d}_logits.npz")
            for ts, rec in meta["positions"].items():
                j = int(rec["q"])
                k = (meta["identity"], int(ts))
                row = rows.setdefault(k, {"j": j, "logits": {}, "consumed": {}, "before": {}})
                if row["j"] != j:
                    raise SystemExit("loc0 q mismatch")
                pre = f"q{j:03d}_"
                for kk, v in L.items():
                    if kk.startswith(pre):
                        row["logits"][kk[len(pre):]] = v
                for kk, v in A.items():
                    if kk.startswith(pre):
                        if kk.endswith("_consumed"):
                            row["consumed"][kk[len(pre):-len("_consumed")]] = v
                        elif "_before_" in kk:
                            row["before"][kk[len(pre) + len("before_"):]] = v
    return join_rows(rows, Q, "st_loc0_barrier_pulses")


def load_r1(Q):
    rows = {}
    for i in range(80):
        meta = J(f"{P['r1']}/pulses/{i:03d}.json")
        A = NPZ(f"{P['r1']}/pulses/{i:03d}_arrays.npz")
        L = NPZ(f"{P['r1']}/pulses/{i:03d}_logits.npz")
        for ts, rec in meta["positions"].items():
            j = int(rec["q"])
            pre = f"q{j:03d}_"
            rows[(meta["identity"], int(ts))] = {
                "j": j, "logits": {k[len(pre):]: v for k, v in L.items() if k.startswith(pre)},
                "consumed": {k[len(pre):-len("_consumed")]: v for k, v in A.items() if k.startswith(pre) and k.endswith("_consumed")}}
    return join_rows(rows, Q, "st_prompt_r1_pulses")


def load_src(Q):
    rows = {}
    for i in range(80):
        c = J(f"{P['src']}/construction/{i:03d}.json")
        C = NPZ(f"{P['src']}/construction/{i:03d}.npz")
        pm = J(f"{P['src']}/pulses/{i:03d}.json")
        A = NPZ(f"{P['src']}/pulses/{i:03d}_arrays.npz")
        L = NPZ(f"{P['src']}/pulses/{i:03d}_logits.npz")
        if c["utterance_id"] != pm["utterance_id"]:
            raise SystemExit("src construction / pulse identity")
        for q in c["queries"]:
            j = int(q["j"])
            pre = f"q{j:03d}_"
            prow = pm["positions"][str(j)]
            if int(prow["t"]) != int(q["t"]) or q["utterance_id"] != c["utterance_id"]:
                raise SystemExit("src query identity")
            rows[(c["utterance_id"], int(q["t"]))] = {
                "j": j, "clean_raw": C[pre + "clean_r1_raw"], "clean_L16": unpack(C[pre + "clean_L16"]), "clean_L24": unpack(C[pre + "clean_L24"]),
                "vac": {l: C.get(f"{pre}target_L{l}_v") for l in (16, 24)}, "rand": {l: C.get(f"{pre}random_L{l}_v") for l in (16, 24)},
                "layers": q["layers"], "region": q["region"], "cells": prow["cells"],
                "logits": {k[len(pre):]: v for k, v in L.items() if k.startswith(pre)},
                "consumed": {k[len(pre):-len("_consumed")]: unpack(v) for k, v in A.items() if k.startswith(pre) and k.endswith("_consumed")}}
    return join_rows(rows, Q, "src_cf0_pilot_construction_pulses")


def load_s1(Q):
    rows = {}
    for i in range(80):
        d = J(f"{P['s1']}/candidates/{i:03d}.json")
        ids = d["auto"]["language_ids"]
        lg = d["auto"]["language_logits"]
        lid = float(lg[ids.index(LANG_EN)]) - float(lg[ids.index(LANG_ZH)])
        for q in d["queries"]:
            rows[(d["utterance_id"], int(q["t"]))] = {"j": int(q["j"]), "region": q["region"], "lid_en_minus_zh": lid,
                                                      "auto_detected": d["auto"]["lowest_id_argmax"]}
    return join_rows(rows, Q, "s1_candidates")


# ---- analysis ----------------------------------------------------------------------------------------------------------

def identity_checks(Q, p2d, loc0cal, r1cap, src, baseline, r1none) -> dict:
    H16 = loc0cal["states_L16_CROSS"]["H_B"]
    H24 = loc0cal["states_L24_CROSS"]["H_B"]
    ch = {
        "p2dir_hb_vs_loc0_H_B_L16": all(np.array_equal(p2d[j]["hb"], H16[j]) for j in range(180)),
        "p2dir_he_vs_loc0_H_E_L16": all(np.array_equal(p2d[j]["he"], loc0cal["states_L16_CROSS"]["H_E"][j]) for j in range(180)),
        "r1_H_M_L16_vs_loc0": bool(np.array_equal(r1cap["states_L16"]["H_M"], H16)),
        "r1_H_M_L24_vs_loc0": bool(np.array_equal(r1cap["states_L24"]["H_M"], H24)),
        "src_clean_L16_vs_loc0": all(np.array_equal(src[j]["clean_L16"], H16[j]) for j in range(180)),
        "src_clean_L24_vs_loc0": all(np.array_equal(src[j]["clean_L24"], H24[j]) for j in range(180)),
        "baseline_logits_r1_none_equal": all(np.array_equal(baseline[j], r1none[j]) for j in range(180)),
        "baseline_logits_src_clean_equal": all(np.array_equal(baseline[j], src[j]["clean_raw"]) for j in range(180)),
        "loc0_states_rows_distinct": len({H16[j].tobytes() for j in range(180)}) == 180,
        "p2dir_eval_site_bitwise_180": all(p2d[j]["eval_site_bitwise"] for j in range(180)),
        "p2dir_D2_status_ok_180": all(p2d[j]["d2_status"] == "ok" for j in range(180)),
    }
    st = NPZ(P["p2dir_states"])
    hb = {H16[j].tobytes(): j for j in range(180)}
    perm = [hb.get(st["H_B"][i].tobytes()) for i in range(st["H_B"].shape[0])]
    ch["p2dir_extract_states_H_B_subset_of_panel"] = all(p is not None for p in perm)
    ch["p2dir_extract_states_order_is_panel_order"] = perm == list(range(180))
    return ch


def direction_checks(Q, p2d, loc0cal, r1cap, src) -> dict:
    v1 = loc0cal["v1_vectors"]
    d2 = loc0cal["d2_vectors"]
    vp = r1cap["v_prompt"]
    return {
        "v_prompt_L16_loc0_vs_r1": all(np.array_equal(v1[f"q{j:03d}_L16_CROSS"], vp[f"q{j:03d}_L16"]) for j in range(180)),
        "v_prompt_L24_loc0_vs_r1": all(np.array_equal(v1[f"q{j:03d}_L24_CROSS"], vp[f"q{j:03d}_L24"]) for j in range(180)),
        "v_prompt_L16_loc0_vs_p2dir_D0": all(np.array_equal(v1[f"q{j:03d}_L16_CROSS"], p2d[j]["D0"]) for j in range(180)),
        "D2_loc0_vs_p2dir": all(np.array_equal(d2[f"q{j:03d}"], p2d[j]["D2"]) for j in range(180)),
        "v_AC_present_iff_target_OK": all(((src[j]["vac"][l] is not None) == (src[j]["layers"][str(l)]["target"].get("status") == "OK"))
                                         for j in range(180) for l in (16, 24)),
        "random_present_180": all(src[j]["rand"][l] is not None for j in range(180) for l in (16, 24)),
        "all_shapes_1280": all(v.shape == (D,) for v in list(v1.values()) + list(d2.values())),
    }


def analysis_a(Q, S, Dg, p2d, loc0cal, src) -> dict:
    v1, d2 = loc0cal["v1_vectors"], loc0cal["d2_vectors"]
    per = []
    for j in range(180):
        h16 = p2d[j]["hb"].astype(np.float64)
        g = p2d[j]["g_margin"].astype(np.float64)
        gt = tangent(g, h16)
        vec = {"v_prompt": v1[f"q{j:03d}_L16_CROSS"], "v_AC": src[j]["vac"][16], "D2": d2[f"q{j:03d}"], "random": src[j]["rand"][16]}
        r = {"j": j, "stratum": S[j], "dialogue_id": Dg[j], "g_norm": float(np.linalg.norm(g)), "g_tan_norm": float(np.linalg.norm(gt)),
             "L16": {}, "L24": {}}
        for a in vec:
            r["L16"][f"cos_{a}_gtan"] = cos(vec[a], gt) if vec[a] is not None else None
        r["L16"]["cos_vprompt_D2"] = cos(vec["v_prompt"], vec["D2"])
        r["L16"]["cos_vAC_D2"] = cos(vec["v_AC"], vec["D2"]) if vec["v_AC"] is not None else None
        r["L16"]["cos_vprompt_vAC"] = cos(vec["v_prompt"], vec["v_AC"]) if vec["v_AC"] is not None else None
        r["L16"]["cos_random_D2"] = cos(vec["random"], vec["D2"])
        r["L16"]["cos_g_readout_gtan"] = cos(tangent(p2d[j]["g_readout"], h16), gt)
        va24 = src[j]["vac"][24]
        r["L24"]["cos_vprompt_vAC"] = cos(v1[f"q{j:03d}_L24_CROSS"], va24) if va24 is not None else None
        r["L24"]["cos_random_vprompt"] = cos(src[j]["rand"][24], v1[f"q{j:03d}_L24_CROSS"])
        r["L24"]["lexical_gradient"] = None  # MISSING: no saved L24 reference gradient anywhere in the archives
        r["v_AC_status"] = {l: src[j]["layers"][str(l)]["target"].get("status") or src[j]["region"]["target_status"] for l in (16, 24)}
        per.append(r)

    def table(layer, key):
        out = {}
        for s in STRATA + ("all",):
            rows = [r for r in per if s == "all" or r["stratum"] == s]
            vals = [(r["dialogue_id"], r[layer][key]) for r in rows]
            sm = summarize(vals)
            sm["abs"] = summarize([(d, abs(v) if v is not None else None) for d, v in vals])
            out[s] = sm
        return out

    keys16 = ["cos_v_prompt_gtan", "cos_v_AC_gtan", "cos_D2_gtan", "cos_random_gtan", "cos_vprompt_D2", "cos_vAC_D2", "cos_vprompt_vAC",
              "cos_random_D2", "cos_g_readout_gtan"]
    res = {"L16": {k: table("L16", k) for k in keys16}, "L24": {k: table("L24", k) for k in ("cos_vprompt_vAC", "cos_random_vprompt")}}
    miss = defaultdict(int)
    for r in per:
        if r["L16"]["cos_v_AC_gtan"] is None:
            miss[f"v_AC_missing:{src[r['j']]['region']['target_status']}"] += 1
    res["coverage"] = {"L16_lexical_gradient": 180, "L16_v_prompt": 180, "L16_D2": 180, "L16_v_AC": sum(r["L16"]["cos_v_AC_gtan"] is not None for r in per),
                       "v_AC_missing_reasons": dict(miss), "L24_lexical_gradient": 0,
                       "L24_missing_reason": "no reference-lexical gradient was ever saved at L24 (P2-RJ / P2-DIR saved L16 DG-02 only); omitted, no surrogate",
                       "L24_D2": 0, "L24_D2_missing_reason": "D2 was only constructed at L16 (P2-DIR)"}
    res["per_dialogue_L16"] = {}
    for key in ("cos_v_prompt_gtan", "cos_v_AC_gtan", "cos_D2_gtan", "cos_random_gtan"):
        for s in STRATA:
            by = defaultdict(list)
            for r in per:
                if r["stratum"] == s and r["L16"][key] is not None:
                    by[r["dialogue_id"]].append(r["L16"][key])
            res["per_dialogue_L16"][f"{key}|{s}"] = {d: {"n": len(v), "mean": float(np.mean(v))} for d, v in sorted(by.items())}
    res["random_reference"] = {"analytic_isotropic_E_abs_cos": math.sqrt(2 / (math.pi * D)), "analytic_sd": 1 / math.sqrt(D),
                               "empirical": "cos_random_gtan / cos_random_D2 rows (SRC-CF0-P per-query PCG64 random, L16)"}
    res["per_query"] = per
    return res


def realized(Q, S, Dg, p2d, loc0, r1, src, base_lex, Y, comp, sup, beg) -> dict:
    """L16 realized-edit alignment kappa and first-order prediction g.edit vs observed delta-m (stored native states only)."""
    specs = [("v_prompt_loc0_plus", "loc0", f"v_prompt_L16_{CROSS}_plus"), ("v_prompt_loc0_minus", "loc0", f"v_prompt_L16_{CROSS}_minus"),
             ("D2_loc0", "loc0", f"D2_L16_{CROSS}"), ("random_loc0", "loc0", f"RANDOM_L16_{CROSS}")]
    for e in ("0.15", "0.30", "0.45"):
        for s in ("plus", "minus"):
            specs.append((f"v_prompt_r1_eta{e}_{s}", "r1", f"prompt_L16_eta{e}_{s}"))
        specs.append((f"random_r1_eta{e}", "r1", f"random_L16_eta{e}"))
    for e in ("0.15", "0.30"):
        for s in ("plus", "minus"):
            specs.append((f"v_AC_src_eta{e}_{s}", "src", f"L16_eta{e}_{s}"))
            specs.append((f"random_src_eta{e}_{s}", "src", f"random_L16_eta{e}_{s}"))
    out = {}
    for name, study, key in specs:
        rows = []
        for j in range(180):
            src_row = {"loc0": loc0, "r1": r1, "src": src}[study][j]
            if key not in src_row["consumed"] or key not in src_row["logits"]:
                continue
            h = p2d[j]["hb"].astype(np.float64)
            before = loc0[j]["before"].get("L16_CROSS") if study == "loc0" else h
            if study == "loc0" and not np.array_equal(before, p2d[j]["hb"]):
                raise SystemExit("loc0 before state")
            edit = src_row["consumed"][key].astype(np.float64) - np.asarray(before, dtype=np.float64)
            g = p2d[j]["g_margin"].astype(np.float64)
            gt = tangent(g, h)
            post = lex(processed(unpack(src_row["logits"][key]), Q[j]["t"], sup, beg), Y[j], comp[j])
            en = float(np.linalg.norm(edit))
            rows.append({"j": j, "stratum": S[j], "dialogue_id": Dg[j], "edit_norm": en, "relative_edit": en / float(np.linalg.norm(h)),
                         "kappa": float(gt @ edit) / (float(np.linalg.norm(gt)) * en) if en > 0 else None,
                         "dm_lin": float(g @ edit), "dm_obs": post["m"] - base_lex[j]["m"], "A": float(np.linalg.norm(gt)) * en})
        summ = {}
        for s in STRATA:
            rs = [r for r in rows if r["stratum"] == s]
            if not rs:
                continue
            obs, lin = np.array([r["dm_obs"] for r in rs]), np.array([r["dm_lin"] for r in rs])
            summ[s] = {"n": len(rs), "kappa": summarize([(r["dialogue_id"], r["kappa"]) for r in rs]),
                       "relative_edit_median": float(np.median([r["relative_edit"] for r in rs])),
                       "dm_obs": summarize([(r["dialogue_id"], r["dm_obs"]) for r in rs]),
                       "dm_lin": summarize([(r["dialogue_id"], r["dm_lin"]) for r in rs]),
                       "corr_obs_lin": float(np.corrcoef(obs, lin)[0, 1]) if len(rs) > 2 and obs.std() > 0 and lin.std() > 0 else None,
                       "A_ceiling_median": float(np.median([r["A"] for r in rs]))}
        out[name] = {"study": study, "key": key, "n": len(rows), "by_stratum": summ}
    return out


def arm_specs():
    """(family, study, arm, layer, dose, sign, logits key, active-cell rule)."""
    sp = []
    for l in (16, 24):
        for s in ("plus", "minus"):
            sp.append(("v_prompt", "ST-LOC0 (e* chord)", f"L{l}_{s}", l, "e*=1.12608", s, f"v_prompt_L{l}_{CROSS}_{s}"))
        sp.append(("random", "ST-LOC0 (e* chord)", f"random_L{l}", l, "e*=1.12608", "+", f"RANDOM_L{l}_{CROSS}"))
    sp.append(("D2", "ST-LOC0/P2-DIR (e* chord)", "D2_L16", 16, "e*=1.12608", "+", f"D2_L16_{CROSS}"))
    for l in (3, 8, 16, 24):
        for e in ("0.15", "0.30", "0.45"):
            for s in ("plus", "minus"):
                sp.append(("v_prompt", "ST-PROMPT-R1-A (relative eta)", f"L{l}_eta{e}_{s}", l, f"eta={e}", s, f"prompt_L{l:02d}_eta{e}_{s}"))
            sp.append(("random", "ST-PROMPT-R1-A (relative eta)", f"random_L{l}_eta{e}", l, f"eta={e}", "+", f"random_L{l:02d}_eta{e}"))
    for l in (16, 24):
        for e in ("0.15", "0.30"):
            for s in ("plus", "minus"):
                sp.append(("v_AC", "SRC-CF0-P (relative eta)", f"L{l}_eta{e}_{s}", l, f"eta={e}", s, f"L{l}_eta{e}_{s}"))
                sp.append(("off_target", "SRC-CF0-P (relative eta)", f"off_L{l}_eta{e}_{s}", l, f"eta={e}", s, f"off_L{l}_eta{e}_{s}"))
                sp.append(("random", "SRC-CF0-P (relative eta)", f"random_L{l}_eta{e}_{s}", l, f"eta={e}", s, f"random_L{l}_eta{e}_{s}"))
    return sp


def gap_rows(Q, S, Dg, Y, comp, sup, beg, base_lex, loc0, r1, src):
    store = {"ST-LOC0 (e* chord)": loc0, "ST-LOC0/P2-DIR (e* chord)": loc0, "ST-PROMPT-R1-A (relative eta)": r1, "SRC-CF0-P (relative eta)": src}
    arms = {}
    for fam, study, arm, l, dose, sign, key in arm_specs():
        rows = []
        for j in range(180):
            row = store[study][j]
            if study.startswith("SRC"):
                cell = row["cells"].get(key)
                if cell is None or not cell.get("steered"):
                    continue
            if key not in row["logits"]:
                if study.startswith("SRC"):
                    raise SystemExit(f"steered SRC cell without logits {key} j={j}")
                continue
            b = base_lex[j]
            p = lex(processed(unpack(row["logits"][key]), Q[j]["t"], sup, beg), Y[j], comp[j])
            if not (tie_consistent(p, Y[j]) and tie_consistent(b, Y[j])):
                raise SystemExit("tie semantics violated")
            rows.append({"j": j, "stratum": S[j], "dialogue_id": Dg[j], "G_base": b["G"], "G_post": p["G"], "dG": p["G"] - b["G"],
                         "d_rank": p["rank_ref"] - b["rank_ref"], "dm": p["m"] - b["m"], "base_in_ref": b["in_ref"], "post_in_ref": p["in_ref"],
                         "top1_changed": p["top1"] != b["top1"], "post_top1": p["top1"]})
        if rows:
            arms[f"{study}|{arm}"] = {"family": fam, "study": study, "arm": arm, "layer": l, "dose": dose, "sign": sign, "rows": rows}
    return arms


def gap_summary(arm: dict) -> dict:
    out = {k: arm[k] for k in ("family", "study", "arm", "layer", "dose", "sign")}
    rows = arm["rows"]
    for s in STRATA:
        rs = [r for r in rows if r["stratum"] == s]
        if not rs:
            out[s] = {"n": 0}
            continue
        o = {"n": len(rs), "dialogues": len({r["dialogue_id"] for r in rs}), "G_base": summarize([(r["dialogue_id"], r["G_base"]) for r in rs]),
             "G_post": summarize([(r["dialogue_id"], r["G_post"]) for r in rs]), "dG": summarize([(r["dialogue_id"], r["dG"]) for r in rs]),
             "dm": summarize([(r["dialogue_id"], r["dm"]) for r in rs]), "d_rank": summarize([(r["dialogue_id"], r["d_rank"]) for r in rs]),
             "rank_improved": sum(r["d_rank"] < 0 for r in rs), "top1_changes": sum(r["top1_changed"] for r in rs)}
        if s == "EN-confusion":
            corr = [r for r in rs if not r["base_in_ref"] and r["post_in_ref"]]
            o["corrections"] = len(corr)
            o["corrected"] = [{"j": r["j"], "dialogue_id": r["dialogue_id"], "G_base": r["G_base"], "G_post": r["G_post"]} for r in corr]
            o["correction_dialogues"] = len({r["dialogue_id"] for r in corr})
            o["other_top1_changes"] = sum(r["top1_changed"] and not (not r["base_in_ref"] and r["post_in_ref"]) for r in rs)
            pos = [r for r in rs if r["G_base"] > 0]
            o["fraction_gap_closed"] = summarize([(r["dialogue_id"], -r["dG"] / r["G_base"]) for r in pos])
            for th in NEAR:
                o[f"near_crossing_post_le_{th}"] = sum(0 < r["G_post"] <= th for r in rs)
                o[f"near_crossing_base_le_{th}"] = sum(0 < r["G_base"] <= th for r in rs)
            o["moved_toward_boundary"] = sum(r["dG"] < 0 for r in rs)
        else:
            cor = [r for r in rs if r["base_in_ref"] and not r["post_in_ref"]]
            o["corruptions"] = len(cor)
            o["corrupted"] = [{"j": r["j"], "dialogue_id": r["dialogue_id"], "G_base": r["G_base"], "G_post": r["G_post"]} for r in cor]
            for th in NEAR:
                o[f"near_corruption_post_ge_minus_{th}"] = sum(-th <= r["G_post"] < 0 for r in rs)
        out[s] = o
    return out


def ceiling(Q, S, Dg, p2d, base_lex) -> dict:
    """First-order ceiling: perfectly aligned tangent edit of norm e moves m by at most ||g_tan|| e; compare with -m_base."""
    out = {}
    for s in STRATA:
        rows = []
        for j in range(180):
            if S[j] != s:
                continue
            h = p2d[j]["hb"].astype(np.float64)
            gt = tangent(p2d[j]["g_margin"], h)
            rows.append((j, float(np.linalg.norm(gt)), float(np.linalg.norm(h)), base_lex[j]["m"], base_lex[j]["G"]))
        o = {"n": len(rows), "g_tan_norm": summarize([(Dg[j], g) for j, g, _, _, _ in rows]),
             "m_base": summarize([(Dg[j], m) for j, _, _, m, _ in rows]), "G_base": summarize([(Dg[j], G) for j, _, _, _, G in rows])}
        if s == "EN-confusion":
            for lab, e_of in (("e*", lambda hn: 1.1260757575454359), ("eta0.15", lambda hn: 0.15 * hn), ("eta0.30", lambda hn: 0.30 * hn),
                              ("eta0.45", lambda hn: 0.45 * hn)):
                A = [(j, g * e_of(hn), -m) for j, g, hn, m, _ in rows]
                o[f"first_order_A_{lab}"] = summarize([(Dg[j], a) for j, a, _ in A])
                o[f"A_ge_deficit_{lab}"] = sum(a >= dfc for _, a, dfc in A)
                o[f"kappa_required_{lab}"] = summarize([(Dg[j], dfc / a) for j, a, dfc in A if a > 0])
        out[s] = o
    return out


def dose_slice(gap_arms: dict, src) -> dict:
    """Descriptive L16 slice at relative energy ~0.15 on the v_AC-active EN-confusion rows (not a controlled comparison)."""
    rows = sorted(j for j in range(180) if src[j]["cells"].get("L16_eta0.15_plus", {}).get("steered"))
    pick = {"v_prompt R1 L16 eta.15 +": "ST-PROMPT-R1-A (relative eta)|L16_eta0.15_plus", "v_prompt R1 L16 eta.15 -": "ST-PROMPT-R1-A (relative eta)|L16_eta0.15_minus",
            "v_AC L16 eta.15 +": "SRC-CF0-P (relative eta)|L16_eta0.15_plus", "v_AC L16 eta.15 -": "SRC-CF0-P (relative eta)|L16_eta0.15_minus",
            "D2 L16 e*": "ST-LOC0/P2-DIR (e* chord)|D2_L16", "random R1 L16 eta.15": "ST-PROMPT-R1-A (relative eta)|random_L16_eta0.15",
            "random SRC L16 eta.15 +": "SRC-CF0-P (relative eta)|random_L16_eta0.15_plus"}
    out = {"rows": rows, "n_rows": len(rows), "note": "same queries; different directions, random draws and dose definitions (e* absolute vs eta relative)"}
    for lab, k in pick.items():
        rs = [r for r in gap_arms[k]["rows"] if r["j"] in set(rows)]
        out[lab] = {}
        for s in STRATA:
            x = [r for r in rs if r["stratum"] == s]
            if not x:
                continue
            o = {"n": len(x), "dG": summarize([(r["dialogue_id"], r["dG"]) for r in x])}
            if s == "EN-confusion":
                o["corrections"] = sum(not r["base_in_ref"] and r["post_in_ref"] for r in x)
            else:
                o["corruptions"] = sum(r["base_in_ref"] and not r["post_in_ref"] for r in x)
            out[lab][s] = o
    return out


def features(j: int, s1row: dict, d2prov: dict, base_zp: np.ndarray) -> dict:
    """The six pre-declared inference-available features. Inputs are reference-free runtime records ONLY (S1 runner
    region / native-LID record, P2-DIR D2 runtime provenance, the NONE processed distribution); no reference, stratum or
    outcome is reachable from here."""
    reg = s1row["region"]
    tgt = reg.get("target") or {}
    p = np.exp(base_zp - lse(base_zp))
    return {"F1_region_available": int(tgt.get("status") == "OK"), "F2_query_region_attention": tgt.get("integral"),
            "F3_raw_heard_mass": reg.get("raw_heard_mass"), "F4_lid_logodds_en_minus_zh": s1row["lid_en_minus_zh"],
            "F5_script_logodds_PE_minus_PM": float(d2prov["log_PE"]) - float(d2prov["log_PM"]), "F6_top1_prob": float(np.max(p))}


FEATURE_SOURCES = {
    "F1_region_available": ("inference-available", "S1 candidates runner region record (R0 predicted EN track + query attention); reference-free"),
    "F2_query_region_attention": ("inference-available", "S1 exact query->crop cross-attention integral; absent for LOW_HEARD_MASS / NO_EN_REGION"),
    "F3_raw_heard_mass": ("inference-available", "S1 alignment-head attention mass on heard frames"),
    "F4_lid_logodds_en_minus_zh": ("inference-available", "native Whisper detect_language logits (one SOT query per utterance)"),
    "F5_script_logodds_PE_minus_PM": ("inference-available", "P2-DIR D2 runtime provenance on the NONE distribution (tokenizer partition)"),
    "F6_top1_prob": ("inference-available", "NONE processed next-token distribution"),
}


def analysis_c(Q, S, Dg, gap_arms, s1, src, p2d, base_zp) -> dict:
    d2 = {r["j"]: r for r in gap_arms["ST-LOC0/P2-DIR (e* chord)|D2_L16"]["rows"]}
    grp = {}
    for j in range(180):
        r = d2[j]
        if S[j] == "EN-confusion":
            grp[j] = "EN_correction" if r["post_in_ref"] else "EN_confusion_uncorrected"
        elif S[j] == "ZH-correct":
            grp[j] = "ZH_corruption" if not r["post_in_ref"] else "ZH_harmless"
        else:
            grp[j] = "EN_correct_corruption" if not r["post_in_ref"] else "EN_correct_harmless"
    # features are computed for every query BEFORE any group label is attached (no reference-based selection)
    F = {j: features(j, s1[j], p2d[j]["prov"], base_zp[j]) for j in range(180)}
    elig_check = all((s1[j]["region"]["target"]["status"] == src[j]["region"]["target_status"]) and
                     (bool(s1[j]["region"]["paired_available"]) == bool(src[j]["region"]["paired_available"])) for j in range(180))
    groups = {}
    for g in ("EN_correction", "ZH_corruption", "EN_correct_corruption", "ZH_harmless", "EN_correct_harmless", "EN_confusion_uncorrected"):
        js = [j for j in range(180) if grp[j] == g]
        o = {"n": len(js), "dialogues": sorted({Dg[j] for j in js}), "S1_target_OK": sum(F[j]["F1_region_available"] for j in js),
             "S1_paired_available": sum(bool(s1[j]["region"]["paired_available"]) for j in js)}
        cnt = defaultdict(int)
        for j in js:
            cnt[s1[j]["region"]["target"]["status"]] += 1
        o["S1_status_counts"] = dict(cnt)
        o["features"] = {}
        for f in FEATURE_SOURCES:
            vals = [F[j][f] for j in js]
            o["features"][f] = summarize([(Dg[j], v) for j, v in zip(js, vals)])
        if g in ("EN_correction", "ZH_corruption", "EN_correct_corruption"):
            o["rows"] = [{"j": j, "utterance_id": Q[j]["utterance_id"], "t": Q[j]["t"], "dialogue_id": Dg[j], "S1_status": s1[j]["region"]["target"]["status"],
                          "paired": bool(s1[j]["region"]["paired_available"]), "G_base": d2[j]["G_base"], "G_post": d2[j]["G_post"], **F[j]} for j in js]
        groups[g] = o
    contrasts = {}
    for a, b in (("EN_correction", "ZH_corruption"), ("ZH_corruption", "ZH_harmless")):
        ja, jb = [j for j in range(180) if grp[j] == a], [j for j in range(180) if grp[j] == b]
        contrasts[f"{a}_vs_{b}"] = {f: {"P(A>B)": auc([F[j][f] for j in ja], [F[j][f] for j in jb]), "nA": len(ja), "nB": len(jb),
                                        "nA_defined": sum(F[j][f] is not None for j in ja), "nB_defined": sum(F[j][f] is not None for j in jb)}
                                    for f in FEATURE_SOURCES}
    zh = [j for j in range(180) if S[j] == "ZH-correct"]
    enc = [j for j in range(180) if S[j] == "EN-correct"]
    conf = [j for j in range(180) if S[j] == "EN-confusion"]
    gate = {}
    for name, rule in (("S1_target_OK", lambda j: F[j]["F1_region_available"] == 1), ("S1_paired", lambda j: bool(s1[j]["region"]["paired_available"]))):
        gate[name] = {"EN_confusion_passing": sum(rule(j) for j in conf), "corrections_retained": sum(rule(j) for j in conf if grp[j] == "EN_correction"),
                      "ZH_correct_passing": sum(rule(j) for j in zh), "ZH_corruptions_retained": sum(rule(j) for j in zh if grp[j] == "ZH_corruption"),
                      "EN_correct_passing": sum(rule(j) for j in enc), "EN_correct_corruptions_retained": sum(rule(j) for j in enc if grp[j] == "EN_correct_corruption")}
        g_ = gate[name]
        g_["ZH_corruption_rate_among_passing"] = g_["ZH_corruptions_retained"] / g_["ZH_correct_passing"] if g_["ZH_correct_passing"] else None
        g_["ZH_corruption_rate_ungated"] = sum(grp[j] == "ZH_corruption" for j in zh) / len(zh)
    zh_dG = summarize([(Dg[j], d2[j]["dG"]) for j in zh])
    return {"groups": groups, "contrasts": contrasts, "retrospective_gate_restriction": gate, "ZH_correct_dG_all60": zh_dG,
            "S1_eligibility_cross_check_S1_vs_SRC_projection": elig_check, "feature_sources": FEATURE_SOURCES,
            "information_classes": {
                "a_inference_available": list(FEATURE_SOURCES),
                "b_evaluator_only_used_to_assess": ["P2-RJ strata", "P2-RJ acceptable first-token sets Y_ref (define correction / corruption)",
                                                    "fixed competitor c*", "P2-DIR evaluator gradient g_margin (Analysis A/B only)"],
                "c_unavailable_at_inference": ["reference transcript / Y_ref", "gold language / error strata", "MMS-FA / CTC oracle timing",
                                               "whether a query is a code-switch confusion"]},
            "per_query_features": {str(j): F[j] for j in range(180)}}


def analysis_h(loc0cal, src, Dg) -> dict:
    """H2 reference-free geometry: per-query v_prompt consistency (no lexical evaluation of any pooled direction)."""
    out = {}
    v1, d2 = loc0cal["v1_vectors"], loc0cal["d2_vectors"]
    sets = {"v_prompt_L16": {j: v1[f"q{j:03d}_L16_CROSS"] for j in range(180)}, "v_prompt_L24": {j: v1[f"q{j:03d}_L24_CROSS"] for j in range(180)},
            "D2_L16": {j: d2[f"q{j:03d}"] for j in range(180)},
            "v_AC_L16": {j: src[j]["vac"][16] for j in range(180) if src[j]["vac"][16] is not None},
            "v_AC_L24": {j: src[j]["vac"][24] for j in range(180) if src[j]["vac"][24] is not None},
            "random_L16": {j: src[j]["rand"][16] for j in range(180)}}
    for name, vs in sets.items():
        js = sorted(vs)
        X = np.stack([np.asarray(vs[j], dtype=np.float64) for j in js])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        R = float(np.linalg.norm(X.mean(axis=0)))
        G = X @ X.T
        iu = np.triu_indices(len(js), 1)
        lodo = []
        for d in sorted({Dg[j] for j in js}):
            inn = [i for i, j in enumerate(js) if Dg[j] == d]
            oth = [i for i, j in enumerate(js) if Dg[j] != d]
            if not oth:
                continue
            m = X[oth].mean(axis=0)
            m /= np.linalg.norm(m)
            lodo += [(d, float(X[i] @ m)) for i in inn]
        out[name] = {"n": len(js), "mean_resultant_length": R, "pairwise_cos_median": float(np.median(G[iu])),
                     "pairwise_cos_q10_q90": [float(np.quantile(G[iu], .1)), float(np.quantile(G[iu], .9))],
                     "lodo_cos_to_pooled": summarize(lodo)}
    return out


def h1_search() -> dict:
    """Search existing archives / configs for any task-token (translate 50359) prompt; record what exists."""
    pat = re.compile(r"\[\s*50258\s*,\s*\d+\s*,\s*50359\b")
    hits, scanned = [], 0
    for base in ("results/inference_cf", "configs/inference_cf", "docs/inference_cf"):
        for p in (ROOT / base).rglob("*.json"):
            scanned += 1
            try:
                if pat.search(p.read_text(errors="ignore")):
                    hits.append(str(p.relative_to(ROOT)))
            except OSError:
                pass
    added = json.loads((MODEL_DIR / "added_tokens.json").read_text())
    return {"translate_id": added.get("<|translate|>"), "transcribe_id": added.get("<|transcribe|>"), "json_files_scanned": scanned,
            "files_with_translate_task_prompt": hits,
            "task_contrast_activations_available": bool(hits),
            "construction_feasibility": "UNKNOWN" if not hits else "INSPECT"}


def historical_checks(gs: dict, rl: dict, c: dict) -> dict:
    H = HISTORICAL
    d2 = gs["ST-LOC0/P2-DIR (e* chord)|D2_L16"]
    ck = {"D2_corrections_5": d2["EN-confusion"]["corrections"] == H["D2_EN_confusion_corrections"],
          "D2_ZH_corruptions_8": d2["ZH-correct"]["corruptions"] == H["D2_ZH_correct_corruptions"],
          "D2_EN_correct_corruptions_1": d2["EN-correct"]["corruptions"] == H["D2_EN_correct_corruptions"],
          "D2_EN_confusion_top1_changes_43": d2["EN-confusion"]["top1_changes"] == H["D2_EN_confusion_top1_changes"],
          "D2_S1_eligible_corrections_3": c["groups"]["EN_correction"]["S1_target_OK"] == H["D2_S1_eligible_corrections"],
          "D2_S1_eligible_ZH_corruptions_3": c["groups"]["ZH_corruption"]["S1_target_OK"] == H["D2_S1_eligible_ZH_corruptions"]}
    for a, n in H["loc0_v_prompt_corrections"].items():
        g = gs[f"ST-LOC0 (e* chord)|{a}"]
        ck[f"loc0_{a}_corrections"] = g["EN-confusion"]["corrections"] == n
        ck[f"loc0_{a}_corruptions"] = (g["EN-correct"]["corruptions"], g["ZH-correct"]["corruptions"]) == H["loc0_v_prompt_corruptions_EN_ZH"][a]
    r1 = {k.split("|")[1]: v for k, v in gs.items() if k.startswith("ST-PROMPT-R1") and v["family"] == "v_prompt"}
    ck["r1_corrections_per_arm"] = all(v["EN-confusion"]["corrections"] == H["r1_corrections"].get(a, 0) for a, v in r1.items()) and len(r1) == 24
    ck["r1_corruptions_selected"] = all((r1[a]["EN-correct"]["corruptions"], r1[a]["ZH-correct"]["corruptions"]) == v for a, v in H["r1_corruptions_EN_ZH"].items())
    src = {k.split("|")[1]: v for k, v in gs.items() if k.startswith("SRC") and v["family"] == "v_AC"}
    ck["src_zero_corrections_8_arms"] = len(src) == 8 and all(v["EN-confusion"]["corrections"] == 0 for v in src.values())
    ck["src_corruptions_active_rows"] = all((src[a]["EN-correct"]["corruptions"], src[a]["ZH-correct"]["corruptions"]) == v for a, v in H["src_corruptions_EN_ZH"].items())
    return ck


def p2dir_reproduction(Q, p2d, arms) -> dict:
    """Per-position reproduction of the sealed P2-DIR exp1 analysis (cos(d, g_ref,tan), kappa with the e* denominator,
    D2 delta-m) from the raw arrays, joined by (utterance_id, t)."""
    an = J("results/inference_cf/p2dir/exp1_run1_analysis.json")["per_position"]
    imap = index_map(Q)
    e_star = 1.1260757575454359
    d2 = {r["j"]: r for r in arms["ST-LOC0/P2-DIR (e* chord)|D2_L16"]["rows"]}
    d0 = {r["j"]: r for r in arms["ST-LOC0 (e* chord)|L16_plus"]["rows"]}
    err = defaultdict(float)
    n = 0
    for rec in an:
        j = imap[(rec["utterance_id"], int(rec["t"]))]
        h = p2d[j]["hb"].astype(np.float64)
        gt = tangent(p2d[j]["g_margin"], h)
        gn = float(np.linalg.norm(gt))
        for a, key in (("D0", "D0"), ("D2", "D2")):
            err[f"cos_gref_tan_{a}"] = max(err[f"cos_gref_tan_{a}"], abs(cos(p2d[j][key], gt) - rec["arms"][a]["cos_gref_tan"]))
            k = float(gt @ (p2d[j][f"post_{a}"].astype(np.float64) - h)) / (gn * e_star)
            err[f"kappa_{a}"] = max(err[f"kappa_{a}"], abs(k - rec["arms"][a]["kappa"]))
        err["d_m_D2"] = max(err["d_m_D2"], abs(d2[j]["dm"] - rec["arms"]["D2"]["d_m"]))
        err["d_m_D0"] = max(err["d_m_D0"], abs(d0[j]["dm"] - rec["arms"]["D0"]["d_m"]))
        err["g_tan_norm"] = max(err["g_tan_norm"], abs(gn - rec["g_margin_tan_norm"]))
        n += 1
    return {"positions": n, "max_abs_error": dict(err), "pass": n == 180 and all(v <= 1e-6 for v in err.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/inference_cf/mech_lang0")
    args = ap.parse_args()
    t0 = time.time()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    Q, panel_checks = master_panel()
    pos = J(P["positions"])["positions"]
    S = [p["stratum"] for p in pos]
    Dg = [q["dialogue_id"] for q in Q]
    Y = [[int(x) for x in p["target_ids"] if int(x) < EOS] for p in pos]
    comp = [int(p["competitor"]) for p in pos]
    gen_path = MODEL_DIR / "generation_config.json"
    cfg = J(P["src_config"])
    gen_hash_ok = file_hash(gen_path) == cfg["model"]["files"]["generation_config.json"]
    CONSUMED[str(gen_path)] = file_hash(gen_path)
    gen = json.loads(gen_path.read_text())
    sup, beg = list(gen["suppress_tokens"] or []), list(gen["begin_suppress_tokens"] or [])
    supp_ok = digest({"suppress": sup, "begin": beg}) == J(P["src_runtime"])["suppression_hash"]

    p2d, j1 = load_p2dir(Q)
    loc0, j2 = load_loc0(Q)
    r1, j3 = load_r1(Q)
    src, j4 = load_src(Q)
    s1, j5 = load_s1(Q)
    joins = [j1, j2, j3, j4, j5]
    if not all(j["joined"] == 180 and not j["unmatched_or_index_mismatch"] for j in joins):
        raise SystemExit(f"join failure {joins}")
    loc0cal = {n: NPZ(f"{P['loc0']}/calibration/{n}.npz") for n in ("states_L16_CROSS", "states_L24_CROSS", "v1_vectors", "d2_vectors", "baseline_logits")}
    r1cap = {n: NPZ(f"{P['r1']}/capture/{n}.npz") for n in ("states_L16", "states_L24", "v_prompt", "none_logits")}
    baseline = {j: loc0cal["baseline_logits"][f"q{j:03d}"] for j in range(180)}
    r1none = {j: r1cap["none_logits"][f"q{j:03d}"] for j in range(180)}
    idc = identity_checks(Q, p2d, loc0cal, r1cap, src, baseline, r1none)
    dc = direction_checks(Q, p2d, loc0cal, r1cap, src)
    required = [k for k in idc if k != "p2dir_extract_states_order_is_panel_order"]
    if not (all(panel_checks[k] for k in panel_checks if not k.endswith("_keys")) and all(idc[k] for k in required) and all(dc.values())
            and gen_hash_ok and supp_ok):
        raise SystemExit(json.dumps(finite({"panel": panel_checks, "identity": idc, "directions": dc, "gen": gen_hash_ok, "supp": supp_ok}), indent=1))

    base_zp = {j: processed(unpack(baseline[j]), Q[j]["t"], sup, beg) for j in range(180)}
    base_lex = {j: lex(base_zp[j], Y[j], comp[j]) for j in range(180)}
    strata_ok = all(base_lex[j]["in_ref"] == (S[j] != "EN-confusion") for j in range(180))

    A = analysis_a(Q, S, Dg, p2d, loc0cal, src)
    RL = realized(Q, S, Dg, p2d, loc0, r1, src, base_lex, Y, comp, sup, beg)
    arms = gap_rows(Q, S, Dg, Y, comp, sup, beg, base_lex, loc0, r1, src)
    GS = {k: gap_summary(v) for k, v in arms.items()}
    CE = ceiling(Q, S, Dg, p2d, base_lex)
    DS = dose_slice(arms, src)
    C = analysis_c(Q, S, Dg, arms, s1, src, p2d, base_zp)
    H = {"H2_geometry": analysis_h(loc0cal, src, Dg), "H1_task_contrast_search": h1_search()}
    HC = historical_checks(GS, RL, C)
    REP = p2dir_reproduction(Q, p2d, arms)
    HC["p2dir_per_position_reproduction"] = REP["pass"]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "mech_lang0_manifest_v1", "head_at_run": head, "protocol": "docs/inference_cf/MECH_LANG0_PROTOCOL.md",
                "protocol_sha256": file_hash(ROOT / "docs/inference_cf/MECH_LANG0_PROTOCOL.md"),
                "script_sha256": file_hash(Path(__file__)), "consumed_artifacts": dict(sorted(CONSUMED.items())), "n_consumed": len(CONSUMED),
                "joins": joins, "panel_checks": panel_checks, "identity_checks": idc, "direction_checks": dc,
                "generation_config_hash_ok": gen_hash_ok, "suppression_hash_ok": supp_ok, "baseline_strata_consistent": strata_ok,
                "model_loaded": False, "forwards": 0, "autograd_calls": 0, "gpu": False, "slurm": False}
    manifest["manifest_hash"] = digest(finite(manifest))
    analysis = {"schema": "mech_lang0_analysis_v1", "status": "EXPLORATORY_POST_HOC", "head_at_run": head, "manifest_hash": manifest["manifest_hash"],
                "historical_checks": HC, "p2dir_reproduction": REP, "historical_checks_all_pass": all(HC.values()), "A_direction_geometry": A, "A_realized_edit_alignment_L16": RL,
                "B_gap_by_arm": GS, "B_first_order_ceiling": CE, "B_dose_aligned_slice_L16": DS, "C_D2_selectivity": C, "D_hypotheses": H,
                "runtime_sec": time.time() - t0}
    (out_dir / "manifest.json").write_text(json.dumps(finite(manifest), indent=1, sort_keys=True) + "\n")
    (out_dir / "analysis.json").write_text(json.dumps(finite(analysis), indent=1, sort_keys=True) + "\n")
    print(json.dumps(finite({"historical_checks": HC, "identity": idc, "directions": dc, "runtime_sec": analysis["runtime_sec"]}), indent=1))


if __name__ == "__main__":
    main()
