#!/usr/bin/env python
"""P2-SEL analysis (CPU, evaluator side). Frozen spec docs/inference_cf/P2_SEL_SPEC.md section 3
(S1) and the pre-committed S2 label rule (section 4).

Reuses the P2-DIR evaluator definitions unchanged (generate-suppressed logits, fixed reference set
and competitor, first-index argmax, dialogue bootstrap draw convention): ``logit_metrics``,
``draw_weights``, ``boot_stat`` from ``inference_cf_p2dir_analyze``. NONE is the bitwise-verified
P2-DIR baseline logits; the ungated-D2 denominators are the audited P2-DIR Exp-1 per-position rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
from experiments.inference_cf_p2dir_analyze import (boot_stat, draw_weights, hi, jsonable, lo,
                                                    logit_metrics, unpack_bf16)

STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
ARMS = ("C1", "C2", "C3")
CORRECT = ("EN-correct", "ZH-correct")
FIXED_UNGATED = {"EN-confusion": 4.499, "ZH-correct": -2.011}


# ---- frozen S1 decision rules -------------------------------------------------------------------------

def standalone(f: dict, arm: str, obs_corr: dict, th: dict) -> dict:
    """Material benefit + correct-state safety for one arm (spec section 3, items 1-3)."""
    c = {"conf_point": f[f"{arm}_conf"]["estimate"] is not None and
                      f[f"{arm}_conf"]["estimate"] >= th["material_en_confusion_delta_m_nats"],
         "conf_lower": lo(f[f"{arm}_conf"]) is not None and lo(f[f"{arm}_conf"]) > th["confusion_ci_lower_strictly_above"]}
    for s, k in (("EN-correct", "en"), ("ZH-correct", "zh")):
        c[f"{k}_margin_lower"] = lo(f[f"{arm}_{k}"]) is not None and lo(f[f"{arm}_{k}"]) >= th["correct_margin_ci_lower_nats_each"]
        c[f"{k}_corr_observed"] = obs_corr[arm][s] is not None and obs_corr[arm][s] <= th["correct_corruption_observed_max_each"]
        c[f"{k}_corr_upper"] = hi(f[f"{arm}_corr_{k}"]) is not None and hi(f[f"{arm}_corr_{k}"]) <= th["correct_corruption_ci_upper_max_each"]
    benefit = c["conf_point"] and c["conf_lower"]
    safety = all(v for k, v in c.items() if not k.startswith("conf_"))
    return {"checks": c, "benefit": benefit, "safety": safety, "pass": benefit and safety}


def decide_s1(valid: bool, f: dict, eq90: dict, obs_corr: dict, th: dict) -> dict:
    """Frozen S1 label precedence."""
    if not valid:
        return {"label": "P2_SEL_INVALID"}
    c2, c3 = standalone(f, "C2", obs_corr, th), standalone(f, "C3", obs_corr, th)
    lo_conf, hi_conf = th["little_vs_broad_equivalence_C2_minus_C3_confusion_ci90_nats"]
    lo_zh, hi_zh = th["little_vs_broad_equivalence_C2_minus_C3_ZH_correct_ci90_nats"]
    equivalent = (eq90["conf"]["ci"] is not None and lo_conf <= eq90["conf"]["ci"][0] and eq90["conf"]["ci"][1] <= hi_conf
                  and eq90["zh"]["ci"] is not None and lo_zh <= eq90["zh"]["ci"][0] and eq90["zh"]["ci"][1] <= hi_zh)
    selective = (lo(f["diff_conf"]) is not None and lo(f["diff_conf"]) >= th["C2_minus_C3_confusion_ci_lower_min_nats"]
                 and lo(f["diff_zh"]) is not None and lo(f["diff_zh"]) >= th["C2_minus_C3_ZH_correct_ci_lower_min_nats"])
    out = {"C2": c2, "C3": c3, "equivalent": equivalent, "selective": selective}
    if c2["pass"] and c3["pass"] and equivalent:
        out["label"] = "P2_SEL_GATE_ADDS_LITTLE_VS_BROAD"
    elif c2["pass"] and selective:
        out["label"] = "P2_SEL_GATE_RESCUES_D2"
    elif c2["safety"] and not c2["benefit"]:
        out["label"] = "P2_SEL_GATE_TOO_CONSERVATIVE"
    else:
        out["label"] = "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    return out


def decide_s2(valid: bool, damage_upper: dict, gain_vs_b0: dict, gain_vs_old: dict, th: dict) -> str:
    """Frozen S2 mini-screen label precedence (pointwise 95%; damage = NEW minus B0)."""
    if not valid:
        return "P2_SEL_MINI_INVALID"
    limits = {"MER": th["MER_damage_upper"], "ZH_CER": th["ZH_CER_damage_upper"], "EN_WER": th["EN_WER_damage_upper"],
              "EN_retention_loss": th["EN_retention_loss_upper"], "ZH_retention_loss": th["ZH_retention_loss_upper"],
              "outside_harm": th["outside_harm_rate_upper"]}
    if any(damage_upper.get(k) is None or damage_upper[k] > v for k, v in limits.items()):
        return "P2_SEL_MINI_DAMAGE_UNRESOLVED"
    m = th["promising_PIER_gain_vs_B0_and_OLD_min"]
    if all(g["estimate"] is not None and g["estimate"] >= m and lo(g) is not None and lo(g) > th["promising_gain_ci_lower_strictly_above"]
           for g in (gain_vs_b0, gain_vs_old)):
        return "P2_SEL_MINI_PROMISING"
    return "P2_SEL_MINI_INCONCLUSIVE"


def quant(xs) -> dict:
    a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "q25": float(np.quantile(a, .25)),
            "q75": float(np.quantile(a, .75)), "frac_pos": float(np.mean(a > 0)), "mean": float(a.mean())}


# ---- loading / rows ------------------------------------------------------------------------------------

def load(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"] or m["stage"] != "s1":
        raise ValueError("invalid S1 manifest")
    con = json.loads((ROOT / m["construction"]).read_text())
    out = {"manifest": m, "construction": con, "rows": {}}
    for i, uid in enumerate(con["utterances"]):
        r = {}
        for part in ("a", "b"):
            p = run / "rows" / f"{i:03d}_{part}.json"
            row = json.loads(p.read_text()) if p.exists() else {"status": "missing"}
            if row.get("status") == "ok":
                if row["manifest_hash"] != m["manifest_hash"] or row["identity"] != uid:
                    raise ValueError(f"stale row {p}")
                if file_hash(run / "rows" / f"{i:03d}_{part}_logits.npz") != row["logits_sha256"]:
                    raise ValueError("logits hash")
                with np.load(run / "rows" / f"{i:03d}_{part}_logits.npz") as z:
                    row["_logits"] = {k: unpack_bf16(z[k]) for k in z.files}
            r[part] = row
        p2 = ROOT / m["p2dir_run"] / f"rows/{i:03d}_logits.npz"
        with np.load(p2) as z:
            r["none"] = {k: unpack_bf16(z[k]) for k in z.files if k.endswith("_none")}
        out["rows"][uid] = r
    out["broad"] = json.loads((run / "broad_target.json").read_text()) if (run / "broad_target.json").exists() else None
    out["runtime"] = json.loads((run / "runtime.json").read_text()) if (run / "runtime.json").exists() else {}
    return out


def records(data: dict, pos: dict, p2dir_ana: dict, partition, suppress, begin) -> list[dict]:
    ungated = {(r["utterance_id"], r["t"]): r["arms"]["D2"]["d_m"] for r in p2dir_ana["per_position"]}
    recs = []
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        rr = data["rows"][uid]
        pre = f"t{t}_"
        x = {"utterance_id": uid, "t": t, "dialogue_id": p["dialogue_id"], "stratum": p["stratum"], "present": False}
        if rr["a"].get("status") != "ok" or rr["b"].get("status") != "ok" or str(t) not in rr["a"]["positions"] \
                or str(t) not in rr["b"]["positions"]:
            recs.append(x)
            continue
        a, b = rr["a"]["positions"][str(t)], rr["b"]["positions"][str(t)]
        Y, c = [int(v) for v in p["target_ids"]], int(p["competitor"])
        none = logit_metrics(rr["none"][pre + "none"], t, suppress, begin, partition, Y, c)
        x.update(present=True, gate=a["gate"], ungated_d2_dm=ungated[(uid, t)], none_m=none["m"])
        idt = a["identity"]
        x["checks"] = {"query": idt["query"] and b["query"] == a["query"], "hb": idt["hb_bitwise_p2dir"] and b["hb_bitwise_p2dir"],
                       "he": idt["he_bitwise_p2dir"], "none_logits": idt["none_logits_bitwise_p2dir"],
                       "d0": idt["d0_bitwise_p2dir"], "R_B": idt["R_B_abs_diff"] <= 1e-6, "g": idt["g_product_abs_diff"] <= 1e-9,
                       "d2_valid": a["directions"]["C2"]["status"] == "ok",
                       "restore": a["restore_bitwise"] and b["restore_bitwise"],
                       "zero_gain_noedit": all(a["arms"][k]["zero_gain_bitwise_none"] in (None, True) for k in GATED_ARMS),
                       "c3_zero_noedit": b["arm"]["zero_bitwise_none"] in (None, True)}
        x["valid"] = all(x["checks"].values())
        arms = {"C1": a["arms"]["C1"], "C2": a["arms"]["C2"], "C3": b["arm"]}
        x["arms"] = {}
        for k, ar in arms.items():
            lg = (rr["a"] if k != "C3" else rr["b"])["_logits"][pre + k]
            mt = logit_metrics(lg, t, suppress, begin, partition, Y, c)
            x["arms"][k] = {"d_m": mt["m"] - none["m"], "correct_after": mt["top1_in_ref"], "top1": mt["top1"],
                            "edit_norm": ar["edit_norm"], "steered": ar["steered"],
                            "dose": ar.get("dose"), "solver_status": ar.get("solver", {}).get("status") if k == "C3" else None}
        recs.append(x)
    return recs


GATED_ARMS = ("C1", "C2")


def _v(rs, fn):
    return [(r["dialogue_id"], fn(r)) for r in rs]


def analyze_s1(run: Path) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    data = load(run)
    m = data["manifest"]
    cfg = json.loads((ROOT / m["config"]).read_text())
    if digest(cfg) != m["config_hash"]:
        raise ValueError("config changed")
    s1 = cfg["S1"]
    th = s1["thresholds"]
    pos = json.loads((ROOT / s1["population"]).read_text())
    p2dir_ana = json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1_analysis.json").read_text())
    model = Path(m["model"]["dir"])
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if partition["hash"] != m["partition_hash"] or digest({"suppress": sup, "begin": beg}) != m["suppression_hash"]:
        raise ValueError("partition/suppression hash")
    recs = records(data, pos, p2dir_ana, partition, sup, beg)
    present = [r for r in recs if r["present"]]
    invalid_rows = [(r["utterance_id"], r["t"], [k for k, v in r.get("checks", {}).items() if not v])
                    for r in recs if not r.get("valid")]
    br = data["broad"]
    q3 = float(sum(r["arms"]["C3"]["edit_norm"] ** 2 for r in present))
    q2 = float(sum(r["arms"]["C2"]["edit_norm"] ** 2 for r in present))
    if q2 > 0:
        energy_rel = q3 / q2 - 1.0
        energy_ok = abs(energy_rel) <= s1["broad_rule"]["aggregate_relative_energy_tolerance"]
    else:
        energy_rel = None
        energy_ok = q3 == 0.0
    keys, W = draw_weights([p["dialogue_id"] for p in pos["positions"]], cfg["S1"]["bootstrap"]["replicates"],
                           cfg["S1"]["bootstrap"]["seed"])
    af = 0.05 / s1["bootstrap"]["decision_family_size"]
    S = {s: [r for r in present if r["stratum"] == s] for s in STRATA}
    corr = lambda arm: (lambda r: 0.0 if r["arms"][arm]["correct_after"] else 1.0)
    f = {}
    for arm in ("C2", "C3"):
        f[f"{arm}_conf"] = boot_stat(_v(S["EN-confusion"], lambda r, a=arm: r["arms"][a]["d_m"]), keys, W, af)
        f[f"{arm}_en"] = boot_stat(_v(S["EN-correct"], lambda r, a=arm: r["arms"][a]["d_m"]), keys, W, af)
        f[f"{arm}_zh"] = boot_stat(_v(S["ZH-correct"], lambda r, a=arm: r["arms"][a]["d_m"]), keys, W, af)
        f[f"{arm}_corr_en"] = boot_stat(_v(S["EN-correct"], corr(arm)), keys, W, af)
        f[f"{arm}_corr_zh"] = boot_stat(_v(S["ZH-correct"], corr(arm)), keys, W, af)
    f["diff_conf"] = boot_stat(_v(S["EN-confusion"], lambda r: r["arms"]["C2"]["d_m"] - r["arms"]["C3"]["d_m"]), keys, W, af)
    f["diff_zh"] = boot_stat(_v(S["ZH-correct"], lambda r: r["arms"]["C2"]["d_m"] - r["arms"]["C3"]["d_m"]), keys, W, af)
    if len(f) != s1["bootstrap"]["decision_family_size"]:
        raise AssertionError("family size")
    eq90 = {"conf": boot_stat(_v(S["EN-confusion"], lambda r: r["arms"]["C2"]["d_m"] - r["arms"]["C3"]["d_m"]), keys, W, 0.10),
            "zh": boot_stat(_v(S["ZH-correct"], lambda r: r["arms"]["C2"]["d_m"] - r["arms"]["C3"]["d_m"]), keys, W, 0.10)}
    obs = {arm: {s: f[f"{arm}_corr_{'en' if s == 'EN-correct' else 'zh'}"]["estimate"] for s in CORRECT} for arm in ("C2", "C3")}
    ung = {s: boot_stat(_v(S[s], lambda r: r["ungated_d2_dm"]), keys, W, 0.05) for s in ("EN-confusion", "ZH-correct")}
    denom_ok = all(ung[s]["estimate"] is not None and math.isfinite(ung[s]["estimate"]) and
                   abs(ung[s]["estimate"] - FIXED_UNGATED[s]) <= 5e-4 for s in ung)
    validity = {"all_rows_each_arm": len(present) == 180 == len(recs) and all(len(r["arms"]) == 3 for r in present),
                "row_checks": not invalid_rows, "runtime_completed": data["runtime"].get("status") == "completed",
                "broad_target_recomputed": br is not None and abs(br["Q2"] - q2) <= 1e-9 * max(1.0, q2) and br["N"] == 180,
                "c3_energy_match": energy_ok,
                "valid_draws": all(x["valid_draws"] >= s1["bootstrap"]["minimum_valid_draws"] for x in list(f.values()) + list(eq90.values())),
                "ungated_denominators": denom_ok,
                "no_autograd": data["runtime"].get("autograd_calls") == 0}
    decision = decide_s1(all(validity.values()), f, eq90, obs, th)
    ratios = {"r_benefit": f["C2_conf"]["estimate"] / ung["EN-confusion"]["estimate"],
              "r_harm": abs(f["C2_zh"]["estimate"]) / abs(ung["ZH-correct"]["estimate"]),
              "denominators_matched": {s: ung[s]["estimate"] for s in ung}, "denominators_fixed": FIXED_UNGATED,
              "C3_r_benefit": f["C3_conf"]["estimate"] / ung["EN-confusion"]["estimate"],
              "C3_r_harm": abs(f["C3_zh"]["estimate"]) / abs(ung["ZH-correct"]["estimate"])}
    desc = {}
    for s in STRATA:
        rs = S[s]
        d = {"n": len(rs), "dialogues": len({r["dialogue_id"] for r in rs}),
             "E": quant([r["gate"]["E"] for r in rs]), "R_B": quant([r["gate"]["R_B"] for r in rs]),
             "g": quant([r["gate"]["g"] for r in rs])}
        for arm in ARMS:
            d[arm] = {"d_m": boot_stat(_v(rs, lambda r, a=arm: r["arms"][a]["d_m"]), keys, W, 0.05),
                      "corruption_or_noncorrect": boot_stat(_v(rs, corr(arm)), keys, W, 0.05),
                      "edit_norm": quant([r["arms"][arm]["edit_norm"] for r in rs]),
                      "edit_sq_energy": quant([r["arms"][arm]["edit_norm"] ** 2 for r in rs]),
                      "dose": quant([r["arms"][arm]["dose"] for r in rs]) if arm != "C3" else None,
                      "top1_in_ref_count": sum(bool(r["arms"][arm]["correct_after"]) for r in rs),
                      "edited_count": sum(bool(r["arms"][arm]["steered"]) for r in rs)}
        d["ungated_D2_d_m"] = boot_stat(_v(rs, lambda r: r["ungated_d2_dm"]), keys, W, 0.05)
        desc[s] = d
    res = {"schema": "p2_sel_s1_analysis_v1", "manifest_hash": m["manifest_hash"], "config_hash": m["config_hash"],
           "validity": validity, "invalid_rows": invalid_rows,
           "energy": {"Q2": q2, "Q3": q3, "relative_mismatch": energy_rel, "e_broad": br["e_broad"] if br else None,
                      "c3_unreachable": sum(r["arms"]["C3"]["solver_status"] == "energy_unreachable" for r in present),
                      "c3_edited": sum(r["arms"]["C3"]["steered"] for r in present)},
           "bootstrap": {"dialogues": keys, "family_alpha": af, "replicates": len(W)},
           "family": f, "equivalence_90": eq90, "observed_corruption": obs, "decision": decision,
           "ratios": ratios, "descriptive": desc, "runtime": data["runtime"],
           "per_position": [{k: v for k, v in r.items()} for r in recs]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("s1",))
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    res = jsonable(analyze_s1(ROOT / args.run))
    atomic_json(out, res)
    print(json.dumps({"decision": {k: res["decision"][k] for k in ("label",)}, "validity": res["validity"],
                      "energy": res["energy"], "ratios": res["ratios"],
                      "family": {k: [v["estimate"], v["ci"]] for k, v in res["family"].items()},
                      "eq90": {k: [v["estimate"], v["ci"]] for k, v in res["equivalence_90"].items()},
                      "C2": res["decision"].get("C2", {}).get("checks"), "C3": res["decision"].get("C3", {}).get("checks"),
                      "selective": res["decision"].get("selective"), "equivalent": res["decision"].get("equivalent")}, indent=1))


if __name__ == "__main__":
    main()
