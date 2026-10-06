#!/usr/bin/env python
"""Independent P2-SEL auditor. Does NOT import the P2-SEL (or P2-DIR) analysis/decision modules.
Reuses only the independent P2-DIR *audit* helpers (canonical hash, file hash, bf16 unpack,
logit metrics, dialogue-bootstrap interval).

``prerun``  CPU, no model inference: frozen contract/panel/population/anchors, P2-DIR D2 seal and
            audit, exact P2-R run3 gate joins, reusable NONE, C3 pooled formula, thresholds, firewall,
            reference-free runner surface, no outcome present -> PASS_TO_P2_SEL_S1 / BLOCK_BEFORE_P2_SEL_S1.
``s1``      CPU post-run: population, gate joins, direction/dose identity (float64 reference edits),
            realized energy and C3 matching, margins/corruption, family + equivalence intervals,
            ratios and label precedence -> ``P2_SEL_AUDIT: PASS (S1)``.
``final``   session contract audit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from experiments.inference_cf_p2dir_audit import bf16, canon, ci, fhash, git_blob_hash, metrics

FREEZE = "b4602b84f1b5ff44bc1f9afd7b9eca0515d57f79"
CONFIG = "configs/inference_cf/p2_sel.json"
FROZEN_DOCS = ("docs/inference_cf/P2_SEL_SPEC.md", "docs/inference_cf/P2_SEL_CODEX_DESIGN.md", CONFIG,
               "docs/inference_cf/P2_SEL_MINI_PANEL.json")
UNCHANGED = ("src/csasr/inference_cf/readout.py", "src/csasr/inference_cf/directions.py", "src/csasr/inference_cf/core_r2.py",
             "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/broad.py", "src/csasr/lss/sites.py",
             "src/csasr/models/hooks.py", "experiments/inference_cf_cached.py", "experiments/inference_cf_p2r.py",
             "experiments/inference_cf_p2dir.py")
P2DIR = "results/inference_cf/p2dir/exp1_run1"
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")


def _git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def panel_rederive(panel: dict) -> list[tuple[str, str]]:
    tag = panel["selection_tag"]
    rows = []
    for d in sorted(panel["eligible_ids_by_dialogue"]):
        ids = sorted(panel["eligible_ids_by_dialogue"][d],
                     key=lambda u: (hashlib.sha256(f"{tag}|{u}".encode()).hexdigest(), u))[:5]
        rows += [(d, u) for u in ids]
    return rows


def gate_join(con: dict) -> dict:
    """Independent (uid, absolute query) join of the P2-R run3 current-gate rows."""
    pop = json.loads((ROOT / "results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(pop["utterances"])}
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    out, bad = {}, []
    for p in con["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        q = json.loads((ROOT / P2DIR / f"rows/{cidx[uid]:03d}.json").read_text())["positions"][str(t)]["query"]
        steps = json.loads((ROOT / f"results/inference_cf/p2r/run3/rows/{pidx[uid]:03d}.json").read_text())["current"]["steps"]
        hit = [s for s in steps if s["query"] == q]
        if len(hit) != 1 or hit[0]["t"] != t or hit[0]["fb"] is not None and hit[0]["g"] != 0.0:
            bad.append((uid, t))
            continue
        out[(uid, t)] = hit[0]
    return {"gates": out, "bad": bad}


def runner_reference_free() -> bool:
    src = (ROOT / "experiments/inference_cf_p2sel.py").read_text()
    tree = ast.parse(src)
    segs = [ast.get_source_segment(src, n) for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in ("s1_pass_a", "s1_pass_b", "join_gates", "broad_target",
                                                            "broad_hook", "gated_hook", "run_s1")]
    return len(segs) == 7 and not any(f in s for s in segs for f in ("target_ids", "competitor", "p2rj/positions",
                                                                       "evaluator", "inference_cf_p2rj"))


# ---- prerun ------------------------------------------------------------------------------------------

def cmd_prerun(args) -> dict:
    cfg = json.loads((ROOT / CONFIG).read_text())
    checks, notes = {}, {}
    for rel in FROZEN_DOCS:
        checks[f"frozen_at_{FREEZE[:7]}:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in cfg["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h.replace("sha256:", "")
    for rel in UNCHANGED:
        checks[f"unchanged_since_freeze:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    # population
    pos = json.loads((ROOT / cfg["S1"]["population"]).read_text())
    checks["population_file_hash"] = fhash(ROOT / cfg["S1"]["population"]) == cfg["S1"]["population_hash"]
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    keys = [(p["utterance_id"], p["t"], p["stratum"], p["dialogue_id"]) for p in pos["positions"]]
    checks["population_180_ordered"] = len(keys) == 180 == len(set(keys)) and \
        keys == [(p["utterance_id"], p["t"], p["stratum"], p["dialogue_id"]) for p in con["positions"]]
    cnt = {s: sum(k[2] == s for k in keys) for s in STRATA}
    checks["strata_60_each"] = cnt == cfg["S1"]["strata_counts"]
    checks["dialogues_20"] = len({k[3] for k in keys}) == 20
    # P2-DIR D2 seal and audit
    sealed = json.loads((ROOT / P2DIR / "directions_sealed.json").read_text())
    checks["p2dir_sealed_files"] = sealed["status"] == "SEALED" and all(fhash(ROOT / P2DIR / r) == h for r, h in sealed["files"].items())
    checks["p2dir_exp1_audit_pass"] = json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1_audit.json").read_text())["verdict"] == "P2_DIR_AUDIT: PASS"
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    d2ok, nonek = 0, 0
    for p in con["positions"]:
        i = cidx[p["utterance_id"]]
        row = json.loads((ROOT / P2DIR / f"rows/{i:03d}.json").read_text())["positions"][str(p["t"])]
        with np.load(ROOT / P2DIR / f"rows/{i:03d}.npz") as z:
            v = z.get(f"t{p['t']}_D2")
            if v is not None and row["directions"]["D2"]["status"] == "ok":
                from csasr.inference_cf.unique import array_hash
                d2ok += array_hash(v) == row["directions"]["D2"]["sha256"]
        with np.load(ROOT / P2DIR / f"rows/{i:03d}_logits.npz") as z:
            nonek += f"t{p['t']}_none" in z.files
    checks["d2_sealed_valid_180"] = d2ok == 180
    checks["none_logits_reusable_180"] = nonek == 180
    # gate joins
    gj = gate_join(con)
    checks["gate_join_exact_180"] = not gj["bad"] and len(gj["gates"]) == 180
    checks["gate_product"] = all(abs(s["E"] * s["R_B"] - s["g"]) <= 1e-12 for s in gj["gates"].values())
    # C3 pooled formula
    import experiments.inference_cf_p2sel as run
    checks["c3_pooled_formula"] = abs(run.broad_target([1.0] * 90 + [0.0] * 90) - math.sqrt(90 / 180)) < 1e-15 and \
        run.broad_target([0.0] * 180) == 0.0 and run.N_ELIGIBLE == cfg["S1"]["broad_rule"]["eligible_query_count"] == 180
    # thresholds frozen exactly as in the prose spec
    th = cfg["S1"]["thresholds"]
    checks["s1_thresholds"] = (th["material_en_confusion_delta_m_nats"] == 0.5 and th["confusion_ci_lower_strictly_above"] == 0.0
                               and th["correct_margin_ci_lower_nats_each"] == -0.25 and th["correct_corruption_observed_max_each"] == 0.05
                               and th["correct_corruption_ci_upper_max_each"] == 0.05 and th["C2_minus_C3_confusion_ci_lower_min_nats"] == -0.25
                               and th["C2_minus_C3_ZH_correct_ci_lower_min_nats"] == 0.1
                               and th["little_vs_broad_equivalence_C2_minus_C3_confusion_ci90_nats"] == [-0.25, 0.25]
                               and th["little_vs_broad_equivalence_C2_minus_C3_ZH_correct_ci90_nats"] == [-0.1, 0.1]
                               and cfg["S1"]["bootstrap"]["decision_family_size"] == 12 and cfg["S1"]["bootstrap"]["replicates"] == 10000
                               and cfg["S1"]["bootstrap"]["seed"] == 240924 and cfg["S1"]["bootstrap"]["minimum_valid_draws"] == 9900)
    checks["s1_label_precedence"] = cfg["S1"]["label_precedence"] == ["P2_SEL_INVALID", "P2_SEL_GATE_ADDS_LITTLE_VS_BROAD",
                                                                       "P2_SEL_GATE_RESCUES_D2", "P2_SEL_GATE_TOO_CONSERVATIVE",
                                                                       "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"]
    fi = cfg["frozen_intervention"]
    checks["frozen_intervention"] = (fi["layer"] == 16 and fi["alpha"] == 2 and fi["gate"] == "E*R_B" and fi["phi"] == "identity"
                                     and fi["norm_preserve"] and not fi["depth_rescale"] and fi["direction_version"] == "p2dir_readout_v1")
    checks["runner_constants"] = run.ALPHA == 2.0 and run.LAYER == 16
    # mini panel
    panel = json.loads((ROOT / cfg["S2"]["panel"]).read_text())
    checks["panel_file_hash"] = fhash(ROOT / cfg["S2"]["panel"]) == "sha256:" + cfg["S2"]["panel_sha256"]
    checks["panel_rederivable"] = panel_rederive(panel) == [(r["dialogue_id"], r["utterance_id"]) for r in panel["rows"]]
    parent = json.loads((ROOT / panel["parent_panel"]).read_text())
    checks["panel_subset_parent_d_dev_select"] = parent["role"] == "D-dev-select" and \
        {r["utterance_id"] for r in panel["rows"]} <= {r["utterance_id"] for r in parent["rows"]} and len(panel["rows"]) == 100 \
        and len({r["dialogue_id"] for r in panel["rows"]}) == 20
    # firewall / outcomes / surface
    inf = json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())
    checks["firewall_d_dev_select"] = inf["role"] == "D-dev-select" and set(con["utterances"]) <= {r["utterance_id"] for r in inf["rows"]}
    base = ROOT / "results/inference_cf/p2sel"
    checks["no_outcome_present"] = not base.exists() or not any(p.name.startswith(("s1", "s2")) for p in base.iterdir())
    checks["runner_reference_free"] = runner_reference_free()
    asrc = (ROOT / "experiments/inference_cf_p2sel_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2sel|p2dir)_analyze", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2sel.py").exists()
    verdict = "PASS_TO_P2_SEL_S1" if all(checks.values()) else "BLOCK_BEFORE_P2_SEL_S1"
    notes["gate_positive_by_stratum_prerun_exposed_p2r"] = {
        s: sum(1 for p in con["positions"] if p["stratum"] == s and gj["gates"][(p["utterance_id"], int(p["t"]))]["g"] > 0)
        for s in STRATA}
    return {"schema": "p2_sel_prerun_audit_v1", "verdict": verdict, "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


# ---- S1 ----------------------------------------------------------------------------------------------

def ref_edit_norm(h: np.ndarray, d: np.ndarray, scale: float) -> float:
    h = h.astype(np.float64)
    til = h + scale * d.astype(np.float64)
    hp = til * (np.linalg.norm(h) / (np.linalg.norm(til) + 1e-6))
    return float(np.linalg.norm(hp - h))


def s1_label(valid, f, eq, obs, th) -> str:
    def stand(a):
        ben = f[a + "_conf"][0] >= 0.5 and f[a + "_conf"][1][0] > 0
        saf = all(f[f"{a}_{k}"][1][0] >= -0.25 for k in ("en", "zh")) and \
            all(obs[a][k] <= 0.05 and f[f"{a}_corr_{k}"][1][1] <= 0.05 for k in ("en", "zh"))
        return ben, saf
    if not valid:
        return "P2_SEL_INVALID"
    b2, s2 = stand("C2")
    b3, s3 = stand("C3")
    eqv = -0.25 <= eq["conf"][1][0] and eq["conf"][1][1] <= 0.25 and -0.1 <= eq["zh"][1][0] and eq["zh"][1][1] <= 0.1
    sel = f["diff_conf"][1][0] >= -0.25 and f["diff_zh"][1][0] >= 0.1
    if b2 and s2 and b3 and s3 and eqv:
        return "P2_SEL_GATE_ADDS_LITTLE_VS_BROAD"
    if b2 and s2 and sel:
        return "P2_SEL_GATE_RESCUES_D2"
    if s2 and not b2:
        return "P2_SEL_GATE_TOO_CONSERVATIVE"
    return "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"


def cmd_s1(args) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    run = ROOT / args.run
    man = json.loads((run / "manifest.json").read_text())
    ana = json.loads((ROOT / args.analysis).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    pos = json.loads((ROOT / cfg["S1"]["population"]).read_text())
    con = json.loads((ROOT / man["construction"]).read_text())
    checks, notes = {}, {}
    checks["manifest_self_hash"] = man["manifest_hash"] == canon({k: v for k, v in man.items() if k != "manifest_hash"})
    checks["config_hash"] = canon(cfg) == man["config_hash"]
    checks["sources_at_manifest_commit"] = all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items())
    checks["analysis_manifest"] = ana["manifest_hash"] == man["manifest_hash"]
    gj = gate_join(con)
    checks["gate_join"] = not gj["bad"] and len(gj["gates"]) == 180
    model = Path(man["model"]["dir"])
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    rows, fails, edit_fail = [], [], []
    cache = {}
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        if i not in cache:
            ra = json.loads((run / f"rows/{i:03d}_a.json").read_text())
            rb = json.loads((run / f"rows/{i:03d}_b.json").read_text())
            Lg = {}
            for part_ in ("a", "b"):
                with np.load(run / f"rows/{i:03d}_{part_}_logits.npz") as z:
                    Lg.update({k: bf16(z[k]) for k in z.files})
            V = {}
            for part_ in ("a", "b"):
                with np.load(run / f"rows/{i:03d}_{part_}.npz") as z:
                    V.update({k: z[k] for k in z.files})
            with np.load(ROOT / P2DIR / f"rows/{i:03d}.npz") as z:
                PV = {k: z[k] for k in z.files}
            with np.load(ROOT / P2DIR / f"rows/{i:03d}_logits.npz") as z:
                PN = {k: bf16(z[k]) for k in z.files if k.endswith("_none")}
            cache = {i: (ra, rb, Lg, V, PV, PN)}
        ra, rb, Lg, V, PV, PN = cache[i]
        pre = f"t{t}_"
        a, b = ra["positions"][str(t)], rb["positions"][str(t)]
        gate = gj["gates"][(uid, t)]
        ok = (abs(a["gate"]["E"] - gate["E"]) == 0 and abs(a["gate"]["R_B"] - gate["R_B"]) == 0 and a["gate"]["g"] == gate["g"]
              and a["identity"]["hb_bitwise_p2dir"] and b["hb_bitwise_p2dir"] and a["identity"]["none_logits_bitwise_p2dir"]
              and a["restore_bitwise"] and b["restore_bitwise"])
        hb, he = PV[pre + "hb"], PV[pre + "he"]
        delta = he.astype(np.float64) - hb.astype(np.float64)
        d0 = delta / (np.linalg.norm(delta) + 1e-6)
        d2 = PV[pre + "D2"]
        g = gate["g"]
        for arm, d in (("C1", d0), ("C2", d2)):
            x = a["arms"][arm]
            exp = ref_edit_norm(hb, d, 2.0 * g) if g > 0 else 0.0
            if (exp == 0) != (x["edit_norm"] == 0) or (exp > 0 and abs(x["edit_norm"] / exp - 1) > 1e-2):
                edit_fail.append((uid, t, arm, exp, x["edit_norm"]))
            if x["edit_norm"] > 0:
                en = float(np.linalg.norm(V[pre + "post_" + arm].astype(np.float64) - hb))
                if abs(en / x["edit_norm"] - 1) > 5e-3:
                    edit_fail.append((uid, t, arm, "state", en, x["edit_norm"]))
            elif not np.array_equal(Lg[pre + arm], PN[pre + "none"]):
                edit_fail.append((uid, t, arm, "zero_not_bitwise"))
        c3 = b["arm"]
        if c3["edit_norm"] > 0:
            en = float(np.linalg.norm(V[pre + "post_C3"].astype(np.float64) - hb))
            if abs(en / c3["edit_norm"] - 1) > 5e-3:
                edit_fail.append((uid, t, "C3", "state", en, c3["edit_norm"]))
        if not ok:
            fails.append((uid, t))
        Y, c = [int(v) for v in p["target_ids"]], int(p["competitor"])
        mn = metrics(PN[pre + "none"], t, sup, beg, part, Y, c)
        r = {"d": p["dialogue_id"], "s": p["stratum"], "uid": uid, "t": t, "e": {}}
        for arm in ("C1", "C2", "C3"):
            mm = metrics(Lg[pre + arm], t, sup, beg, part, Y, c)
            r[arm] = (mm["m"] - mn["m"], mm["in_ref"])
            r["e"][arm] = a["arms"][arm]["edit_norm"] if arm != "C3" else c3["edit_norm"]
        rows.append(r)
    checks["row_identity"] = not fails
    checks["dose_direction_energy_reconstruction"] = not edit_fail
    notes["identity_failures"] = fails[:10]
    notes["edit_failures"] = edit_fail[:10]
    q2 = sum(r["e"]["C2"] ** 2 for r in rows)
    q3 = sum(r["e"]["C3"] ** 2 for r in rows)
    e_broad = math.sqrt(q2 / 180) if q2 > 0 else 0.0
    bt = json.loads((run / "broad_target.json").read_text())
    checks["broad_target"] = abs(bt["e_broad"] - e_broad) <= 1e-12 * max(1, e_broad)
    checks["c3_energy_match"] = (abs(q3 / q2 - 1) <= 0.02) if q2 > 0 else q3 == 0
    notes["energy"] = {"Q2": q2, "Q3": q3, "e_broad": e_broad, "relative": (q3 / q2 - 1) if q2 > 0 else None}
    # bootstrap
    keys = sorted({p["dialogue_id"] for p in pos["positions"]})
    idx = np.random.default_rng(240924).integers(0, len(keys), size=(10000, len(keys)))
    W = np.stack([(idx == j).sum(axis=1) for j in range(len(keys))], axis=1).astype(np.float64)
    af = 0.05 / 12
    S = {s: [r for r in rows if r["s"] == s] for s in STRATA}
    dm = lambda rs, a: [(r["d"], r[a][0]) for r in rs]
    co = lambda rs, a: [(r["d"], 0.0 if r[a][1] else 1.0) for r in rs]
    f = {}
    for a in ("C2", "C3"):
        f[a + "_conf"] = ci(dm(S["EN-confusion"], a), keys, W, af)
        f[a + "_en"] = ci(dm(S["EN-correct"], a), keys, W, af)
        f[a + "_zh"] = ci(dm(S["ZH-correct"], a), keys, W, af)
        f[a + "_corr_en"] = ci(co(S["EN-correct"], a), keys, W, af)
        f[a + "_corr_zh"] = ci(co(S["ZH-correct"], a), keys, W, af)
    f["diff_conf"] = ci([(r["d"], r["C2"][0] - r["C3"][0]) for r in S["EN-confusion"]], keys, W, af)
    f["diff_zh"] = ci([(r["d"], r["C2"][0] - r["C3"][0]) for r in S["ZH-correct"]], keys, W, af)
    eq = {"conf": ci([(r["d"], r["C2"][0] - r["C3"][0]) for r in S["EN-confusion"]], keys, W, 0.10),
          "zh": ci([(r["d"], r["C2"][0] - r["C3"][0]) for r in S["ZH-correct"]], keys, W, 0.10)}
    diffs = []
    for k, (est, iv, nd) in f.items():
        x = ana["family"][k]
        diffs += [abs(est - x["estimate"]), abs(iv[0] - x["ci"][0]), abs(iv[1] - x["ci"][1])]
        checks.setdefault("valid_draws", True)
        checks["valid_draws"] &= nd >= 9900 and nd == x["valid_draws"]
    for k, (est, iv, nd) in eq.items():
        x = ana["equivalence_90"][k]
        diffs += [abs(est - x["estimate"]), abs(iv[0] - x["ci"][0]), abs(iv[1] - x["ci"][1])]
    checks["statistics_agree_1e-8"] = max(diffs) <= 1e-8
    notes["max_stat_diff"] = max(diffs)
    obs = {a: {"en": f[a + "_corr_en"][0], "zh": f[a + "_corr_zh"][0]} for a in ("C2", "C3")}
    # ratios against the audited P2-DIR ungated D2 per-position rows
    p2a = {(r["utterance_id"], r["t"]): r["arms"]["D2"]["d_m"] for r in
           json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1_analysis.json").read_text())["per_position"]}
    ug_c = ci([(r["d"], p2a[(r["uid"], r["t"])]) for r in S["EN-confusion"]], keys, W, .05)[0]
    ug_z = ci([(r["d"], p2a[(r["uid"], r["t"])]) for r in S["ZH-correct"]], keys, W, .05)[0]
    rb_, rh_ = f["C2_conf"][0] / ug_c, abs(f["C2_zh"][0]) / abs(ug_z)
    checks["ratios_agree"] = abs(rb_ - ana["ratios"]["r_benefit"]) <= 1e-8 and abs(rh_ - ana["ratios"]["r_harm"]) <= 1e-8
    checks["ungated_denominators"] = abs(ug_c - 4.499) <= 5e-4 and abs(ug_z + 2.011) <= 5e-4
    rt = json.loads((run / "runtime.json").read_text())
    valid = (checks["row_identity"] and checks["dose_direction_energy_reconstruction"] and checks["c3_energy_match"]
             and checks["gate_join"] and checks["valid_draws"] and checks["ungated_denominators"]
             and rt.get("status") == "completed" and rt.get("autograd_calls") == 0 and len(rows) == 180)
    label = s1_label(valid, f, eq, obs, cfg["S1"]["thresholds"])
    checks["label_agrees"] = label == ana["decision"]["label"]
    verdict = "P2_SEL_AUDIT: PASS" if all(checks.values()) else "P2_SEL_AUDIT: BLOCK"
    return {"schema": "p2_sel_s1_audit_v1", "stage": "S1", "verdict": verdict, "label": label, "checks": checks,
            "notes": notes, "independent_family": f, "independent_eq90": eq, "ratios": {"r_benefit": rb_, "r_harm": rh_}}


# ---- final ---------------------------------------------------------------------------------------------

def cmd_final(args) -> dict:
    cfg = json.loads((ROOT / CONFIG).read_text())
    base = ROOT / "results/inference_cf/p2sel"
    checks, notes = {}, {}
    for rel in FROZEN_DOCS + UNCHANGED:
        checks[f"unchanged_since_freeze:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in cfg["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h.replace("sha256:", "")
    entries = sorted(x.name for x in base.iterdir())
    notes["entries"] = entries
    a1 = json.loads((base / "s1_run1_audit.json").read_text())
    an1 = json.loads((base / "s1_run1_analysis.json").read_text())
    label = an1["decision"]["label"]
    checks["s1_audit_pass"] = a1["verdict"] == "P2_SEL_AUDIT: PASS" and a1["label"] == label
    s2_ran = any(e.startswith("s2") for e in entries)
    checks["s2_only_after_rescue"] = (not s2_ran) or label == "P2_SEL_GATE_RESCUES_D2"
    allowed = {"prerun_audit.json", "s1_run1", "s1_run1_analysis.json", "s1_run1_audit.json", "final_audit.json"}
    if s2_ran:
        allowed |= {"s2_run1", "s2_run1_analysis.json", "s2_run1_audit.json"}
    checks["only_frozen_stages_single_attempt"] = set(entries) <= allowed and \
        len(list((base / "s1_run1").glob("slurm-*.out"))) == 1
    m = json.loads((base / "s1_run1/manifest.json").read_text())
    checks["firewall"] = m["role"] == "D-dev-select" and m["firewall"]["role"] == "D-dev-select"
    checks["frozen_intervention_in_manifest"] = m["frozen"] == cfg["frozen_intervention"]
    rt = json.loads((base / "s1_run1/runtime.json").read_text())
    checks["no_new_direction_no_autograd_s1"] = rt["autograd_calls"] == 0 and rt["readout_calls"] == 0
    checks["no_p3_no_full300"] = not any(("p3" in e.lower()) or ("300" in e) for e in entries)
    checks["runner_reference_free"] = runner_reference_free()
    verdict = "P2_SEL_AUDIT: PASS" if all(checks.values()) else "P2_SEL_AUDIT: BLOCK"
    return {"schema": "p2_sel_final_audit_v1", "verdict": verdict, "s1_label": label, "s2_run": s2_ran,
            "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prerun")
    p.add_argument("--out", required=True)
    s = sub.add_parser("s1")
    s.add_argument("--run", required=True)
    s.add_argument("--analysis", required=True)
    s.add_argument("--out", required=True)
    f = sub.add_parser("final")
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "s1": cmd_s1, "final": cmd_final}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "s1_label")}, indent=1))
    print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
