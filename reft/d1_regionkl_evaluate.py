#!/usr/bin/env python
"""D1 selection, canonical evaluation, pre-registered verdict and bounded training-only representation diagnostics.
docs/reft/D1_REGIONKL_DECISION_RULES.md.

  select     (CPU) unchanged P2-1 checkpoint rule per D1 run -> training/D1_seed{s}/selection.json (write-once)
  evaluate   (CPU) R2-GoldKL (historical, recomputed for parity with P2-1b) and D1 per seed/epoch/regime vs the pinned B0
             with the P2-1b / P2-2 metric code plus the target-window metric; paired D1 - GoldKL (selected and epoch 3)
             with shared dialogue draws -> d1_regionkl/evaluation.json
  mech       (GPU, training utterances only) intervention norms, token KL and edit geometry of GoldKL vs D1 checkpoints
  report     (CPU) deterministic verdict, figures -> d1_regionkl/report.json, manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, fillers, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import git_state, now, sha, slurm  # noqa: E402

SEEDS = (42, 43, 44)
ARMS = ("GoldKL", "D1")
ZH_DENOM = 4950


def cd():
    from reft.d1_regionkl_train import cd as c
    return c()


def out(*p):
    from reft.d1_regionkl_train import out as o
    return o(*p)


# ====================================================================== pure helpers (unit-tested)

def tw_from_events(ev: list[dict], K: int) -> dict:
    """Target-window outcome of one utterance from P2-1D unit events (k order): the first K substantive Mandarin reference
    units of every Mandarin run after an EN->ZH switch (same language-sequence rule as the training region)."""
    from reft.d1_regionkl_train import switchback_windows
    lab = ["EN" if e["substantive_en"] else "ZH" if e["zh_non_filler"] else None for e in ev]
    wins = switchback_windows(lab, K)
    r = {"windows": len(wins), "eligible": 0, "b0_correct": 0, "new": 0, "new_del": 0, "new_sub": 0, "repairs": 0, "sys_correct": 0}
    for w in wins:
        for k in w:
            e = ev[k]
            assert e["k"] == k
            r["eligible"] += 1
            r["b0_correct"] += bool(e["b0_ok"])
            r["sys_correct"] += bool(e["sys_ok"])
            if e["transition"] == "corruption":
                r["new"] += 1
                r["new_del" if e["sys_op"] == "del" else "new_sub"] += 1
            elif e["transition"] == "repair":
                r["repairs"] += 1
    return r


def _mean(x: dict, k: str) -> float:
    return float(np.mean([x[s][k] for s in SEEDS]))


def primary_gain(d1: dict, gk: dict, D: dict) -> dict:
    """A: mean new Mandarin errors lower by >= margin and strictly lower in >= seed_consistency_min matched seeds."""
    dz = _mean(d1, "zh_new") - _mean(gk, "zh_new")
    lower = sum(d1[s]["zh_new"] < gk[s]["zh_new"] for s in SEEDS)
    A1, A2 = dz <= -D["preservation_margin_units"], lower >= D["seed_consistency_min"]
    return {"A1_mean_margin": bool(A1), "A2_seed_consistent": bool(A2), "pass": bool(A1 and A2), "d_zh_new_mean": dz, "seeds_lower": lower,
            "per_seed": {str(s): d1[s]["zh_new"] - gk[s]["zh_new"] for s in SEEDS},
            "rel": dz / _mean(gk, "zh_new") if _mean(gk, "zh_new") else None}


def en_guard(d1: dict, gk: dict, D: dict) -> dict:
    """B: sum_s EN_net(D1) / sum_s EN_net(GoldKL) >= ratio (undefined if the GoldKL sum is <= 0 -> fail)."""
    num, den = sum(d1[s]["en_net"] for s in SEEDS), sum(gk[s]["en_net"] for s in SEEDS)
    ratio = num / den if den > 0 else None
    return {"pass": bool(ratio is not None and ratio >= D["en_net_sum_ratio_min"]), "ratio": ratio, "sum_d1": num, "sum_goldkl": den}


def mer_guard(d1: dict, gk: dict) -> dict:
    """C: mean genuine-CS MER(D1) <= mean CS MER(GoldKL), full precision, no tolerance."""
    d = _mean(d1, "cs_mer") - _mean(gk, "cs_mer")
    return {"pass": bool(d <= 0), "d_cs_mer_mean": d, "per_seed": {str(s): d1[s]["cs_mer"] - gk[s]["cs_mer"] for s in SEEDS}}


def component_gain(d1: dict, gk: dict, key: str, D: dict) -> dict:
    """Secondary preservation component (target-window new errors, long SB dropped units): mean lower by >= component
    margin and strictly lower in >= seed_consistency_min seeds."""
    d = _mean(d1, key) - _mean(gk, key)
    lower = sum(d1[s][key] < gk[s][key] for s in SEEDS)
    return {"pass": bool(d <= -D["component_margin_units"] and lower >= D["seed_consistency_min"]), "d_mean": d, "seeds_lower": lower,
            "per_seed": {str(s): d1[s][key] - gk[s][key] for s in SEEDS}}


def tw_support(d1: dict, gk: dict, D: dict) -> dict:
    """D (descriptive): target-window evidence supports a region-specific explanation iff mean TW new errors are lower and
    lower in >= seed_consistency_min seeds."""
    d = _mean(d1, "tw_new") - _mean(gk, "tw_new")
    lower = sum(d1[s]["tw_new"] < gk[s]["tw_new"] for s in SEEDS)
    return {"supported": bool(d < 0 and lower >= D["seed_consistency_min"]), "d_mean": d, "seeds_lower": lower}


def strong(x: dict, D: dict) -> dict:
    return {str(s): bool(x[s]["zh_new"] <= D["strong_zh_new_max"]) for s in SEEDS}


RECOMMEND = {"D1_INVALID": "HOLD_FOR_ENGINEERING_REPAIR",
             "D1_INCREMENTAL_PRESERVATION_GAIN": "GO_MATCHED_REGION_SPECIFICITY_CONTROL",
             "D1_MIXED_TRADEOFF": "CHARACTERIZE_CORRECTION_PRESERVATION_TRADEOFF",
             "D1_NO_INCREMENTAL_GAIN": "STOP_FIXED_SWITCHBACK_WEIGHTING"}


def decide(integrity_ok: bool, A: dict, B: dict, C: dict, comps: dict) -> dict:
    """INVALID > INCREMENTAL (A, B, C) > MIXED (A but not B-and-C; or not A but a consistent secondary preservation
    component gain) > NO_INCREMENTAL_GAIN."""
    if not integrity_ok:
        v = "D1_INVALID"
    elif A["pass"] and B["pass"] and C["pass"]:
        v = "D1_INCREMENTAL_PRESERVATION_GAIN"
    elif A["pass"] or any(c["pass"] for c in comps.values()):
        v = "D1_MIXED_TRADEOFF"
    else:
        v = "D1_NO_INCREMENTAL_GAIN"
    return {"verdict": v, "recommendation": RECOMMEND[v]}


KEYS = ("mer", "cs_mer", "zh_new", "sb", "en_net", "tw_new")


def dialogue_table(per_utt: list[dict]) -> dict:
    """Per dialogue: [sys err, units, CS sys err, CS units, zh_new, sb, en_net, b0 err, tw_new, tw_b0_correct]."""
    t = collections.defaultdict(lambda: np.zeros(10))
    for u in per_utt:
        cs = u["stratum"] == "CS"
        t[u["dialogue"]] += np.array([u["s_err"], u["units"], u["s_err"] if cs else 0, u["units"] if cs else 0, u["zh_new"], u["sb"],
                                      u["en_corr"] - u["en_corrupt"], u["b_err"], u["tw_new"], u["tw_b0_correct"]], float)
    return t


def tab_stat(tab, dl, idx) -> dict:
    a = sum(tab[dl[i]] for i in idx)
    return {"mer": a[0] / a[1], "cs_mer": a[2] / a[3], "zh_new": a[4], "sb": a[5], "en_net": a[6], "tw_new": a[8]}


def paired_tables(dtabs: dict, dl: list, draws: list) -> dict:
    """Paired D1 - GoldKL per seed (point, dialogue-bootstrap CI over shared draws, worst-dialogue contribution, leave-one-
    dialogue-out range) and the seed mean (pooled over the same draws). Metrics recomputed from resampled dialogue counts."""
    full = list(range(len(dl)))
    pr = {}
    for s in SEEDS:
        x, y = dtabs[("D1", s)], dtabs[("GoldKL", s)]
        pt = {k: tab_stat(x, dl, full)[k] - tab_stat(y, dl, full)[k] for k in KEYS}
        bs = {k: [] for k in KEYS}
        for idx in draws:
            p_, q_ = tab_stat(x, dl, idx), tab_stat(y, dl, idx)
            for k in KEYS:
                bs[k].append(p_[k] - q_[k])
        contrib = {}
        for k, col in (("zh_new", 4), ("sb", 5), ("en_net", 6), ("tw_new", 8)):
            per_d = {d: float(x[d][col] - y[d][col]) for d in dl}
            tot = sum(per_d.values())
            loo = [tot - v for v in per_d.values()]
            contrib[k] = {"max_abs_dialogue_delta": max(abs(v) for v in per_d.values()),
                          "max_abs_share_of_total": (max(abs(v) for v in per_d.values()) / abs(tot)) if tot else None,
                          "leave_one_dialogue_out_range": [min(loo), max(loo)], "dialogues_lower": sum(v < 0 for v in per_d.values()),
                          "dialogues_higher": sum(v > 0 for v in per_d.values())}
        pr[s] = {k: {"point": float(pt[k]), "ci95": [float(np.quantile(bs[k], q)) for q in (0.025, 0.975)]} for k in KEYS}
        pr[s]["dialogue_contribution"] = contrib
    pooled = {k: [] for k in KEYS}
    for idx in draws:
        for k in KEYS:
            pooled[k].append(np.mean([tab_stat(dtabs[("D1", s)], dl, idx)[k] - tab_stat(dtabs[("GoldKL", s)], dl, idx)[k] for s in SEEDS]))
    pr["seed_mean"] = {k: {"point": float(np.mean([pr[s][k]["point"] for s in SEEDS])),
                           "range": [float(min(pr[s][k]["point"] for s in SEEDS)), float(max(pr[s][k]["point"] for s in SEEDS))],
                           "ci95": [float(np.quantile(pooled[k], q)) for q in (0.025, 0.975)]} for k in KEYS}
    return {str(k): v for k, v in pr.items()}


def principal_cosines(R1, R2) -> list[float]:
    """Cosines of the principal angles between the row spaces of two matrices with orthonormal rows."""
    s = np.linalg.svd(np.asarray(R1, float) @ np.asarray(R2, float).T, compute_uv=False)
    return [float(min(1.0, x)) for x in s]


# ====================================================================== runs, selection

def run_dirs() -> dict:
    from reft.d1_regionkl_train import control_runs
    d = {("GoldKL", s): c["dir"] for s, c in control_runs().items()}
    for s in SEEDS:
        d[("D1", s)] = out("training", f"D1_seed{s}")
    return d


def run_select():
    from reft.d1_regionkl_train import check_freeze
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, score_system, select_epoch
    check_freeze()
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    for s in SEEDS:
        rd = out("training", f"D1_seed{s}")
        if (rd / "selection.json").exists():
            raise SystemExit(f"{rd}/selection.json exists; write-once")
        recs, table = [], []
        for ep in (1, 2, 3):
            row = {"epoch": ep}
            for reg in ("beam5", "greedy"):
                m = metrics(score_system(meta, refs, {x["id"]: x for x in read_jsonl(rd / f"epoch{ep}" / f"dev_{reg}.jsonl")}, FILL,
                                         C["evaluation"]["loop_units"]))
                row[reg] = {"mer": m["overall"]["mer"], "cs_mer": m["CS"]["mer"], "zh_cer": m["overall"]["zh_cer"]}
            table.append(row)
            recs.append({"epoch": ep, **{k: row["beam5"][k] for k in ("cs_mer", "mer", "zh_cer")}})
        sel = select_epoch(recs)
        write_json(rd / "selection.json", {"schema": "reft_p2_selection_v1", "utc": now(), "rule": C["selection"]["rule"],
                                           "epochs": table, "selected": sel})
        print(rd.name, sel)


# ====================================================================== evaluate

def run_evaluate():
    from reft.d1_regionkl_train import check_freeze
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    from reft.p2_1b_evaluate import failure, fates
    from reft.p2_1d_audit import utterance_events
    from reft.p2_2_evaluate import span_fates
    fz = check_freeze()
    if out("evaluation.json").exists():
        raise SystemExit("evaluation.json exists; write-once")
    C, D = p21(), cd()
    K = D["region"]["max_units_per_switch"]
    BS = D["bootstrap"]
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    LU = C["evaluation"]["loop_units"]
    pb = C["frozen_baseline"]
    assert sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"]
    base = {r: score_system(meta, refs, {x["id"]: x for x in read_jsonl(art(pb[r]))}, FILL, LU) for r in ("beam5", "greedy")}
    runs, dtabs, eptabs, tw_dlg = {}, {}, {}, {}
    for (arm, s), rd in run_dirs().items():
        sel = read_json(rd / "selection.json")["selected"]["epoch"]
        man = read_json(rd / "manifest.json")
        R = {"selected_epoch": sel, "history": man["history"], "efficiency": man.get("efficiency"), "weights_unchanged": man["weights_unchanged"],
             "zero_init_identity": man["history"][0]["zero_init_identity_beam5"], "regions_sha256": man.get("regions_sha256"), "epochs": {}}
        for ep in (0, 1, 2, 3):
            R["epochs"][ep] = {}
            for reg in ("beam5", "greedy"):
                S = score_system(meta, refs, {x["id"]: x for x in read_jsonl(rd / f"epoch{ep}" / f"dev_{reg}.jsonl")}, FILL, LU)
                m = metrics(S)
                rec = {"metrics": {k: m[k] for k in ("overall", "CS", "EN_ONLY", "ZH_ONLY")}, "vs_B0": paired(base[reg], S, BS["reps"], BS["seed"])}
                if reg == "beam5":
                    f = failure(base["beam5"], S, FILL)
                    rec["failure"] = {k: v for k, v in f.items() if k != "_per_utt"}
                    rec["english_fates"] = fates(base["beam5"], S, FILL)
                    sf = collections.Counter()
                    tw = collections.Counter()
                    per_utt = []
                    for b, s_, u in zip(base["beam5"], S, f["_per_utt"]):
                        ev, _ = utterance_events(b, s_, FILL, None, "beam5", b["id"], b["dialogue"], b["stratum"])
                        sf += span_fates(ev)
                        t_ = tw_from_events(ev, K)
                        tw.update(t_)
                        per_utt.append({**u, "tw_new": t_["new"], "tw_b0_correct": t_["b0_correct"]})
                    rec["switchback_spans"] = dict(sf)
                    rec["target_window"] = dict(tw)
                    tab = dialogue_table(per_utt)
                    rec["dialogues_improved"] = int(sum(v[0] < v[7] for v in tab.values()))
                    rec["dialogues_worsened"] = int(sum(v[0] > v[7] for v in tab.values()))
                    rec["tw_new_per_dialogue_sorted"] = sorted((int(v[8]) for v in tab.values()), reverse=True)
                    eptabs[(arm, s, ep)] = tab
                    if ep == sel:
                        dtabs[(arm, s)] = tab
                R["epochs"][ep][reg] = rec
        runs[f"{arm}_seed{s}"] = R
    E1b = read_json(art("p2_1b/evaluation.json"))["runs"]
    repro = {}
    for s in SEEDS:
        mine, theirs = runs[f"GoldKL_seed{s}"], E1b[f"KL_seed{s}"]
        e = mine["selected_epoch"]
        ok = mine["selected_epoch"] == theirs["selected_epoch"]
        for ep in (1, 2, 3):
            a, t = mine["epochs"][ep]["beam5"], theirs["epochs"][str(ep)]["beam5"]
            ok = ok and all(a["failure"][k] == t["failure"][k] for k in ("zh_new", "sb_units", "worst_dialogue_delta_mixed")) \
                and a["vs_B0"]["net"] == t["vs_B0"]["net"] and a["metrics"]["CS"]["mer"] == t["metrics"]["CS"]["mer"] \
                and a["metrics"]["overall"]["mer"] == t["metrics"]["overall"]["mer"]
        repro[f"GoldKL_{s}"] = {"ok": bool(ok), "selected_epoch": e}
    dl = sorted({m["dialogue_id"] for m in meta})
    rng = np.random.default_rng(BS["seed"])
    draws = [rng.choice(len(dl), len(dl)) for _ in range(BS["reps"])]
    ep3 = {(a, s): eptabs[(a, s, 3)] for a in ARMS for s in SEEDS}
    em = {str(s): {str(ep): {k: tab_stat(eptabs[("D1", s, ep)], dl, range(len(dl)))[k] - tab_stat(eptabs[("GoldKL", s, ep)], dl, range(len(dl)))[k]
                             for k in KEYS} for ep in (1, 2, 3)} for s in SEEDS}
    write_json(out("evaluation.json"), {"schema": "reft_d1_evaluation_v1", "utc": now(), **fz, "git": git_state(), "slurm": slurm(),
                                        "B0": {r: metrics(base[r]) for r in base}, "runs": runs, "goldkl_reproduces_p2_1b": repro,
                                        "paired_selected": paired_tables(dtabs, dl, draws), "paired_epoch3": paired_tables(ep3, dl, draws),
                                        "epoch_matched": em, "n_dialogues": len(dl), "K": K,
                                        "note": "dev150 exposed and used for selection; seeds share the 20 dialogues; descriptive uncertainty"})
    print("evaluation written", repro)


# ====================================================================== mechanism (GPU, training utterances only)

def run_mech():
    """Training-only representation diagnostics at the intervention site (gold prefixes, all 500 training utterances):
    ||delta h||, ||delta h|| / ||h||, token KL(p0 || p_theta) by token region (switch-back window, other Mandarin, English,
    other), GoldKL-vs-D1 per-position edit cosine and norm ratios, principal angles of the R row spaces."""
    import torch
    from reft.arch_s0s1_train import Runtime, manual_delta
    from reft.common import fillers as fl
    from reft.d1_regionkl_train import check_freeze, control_runs, lang_labels, make_r2, utterance_region
    from reft.p2_1b_train import teacher_mask
    from reft.text import tok_bytes_fn, token_unit_map, units
    fz = check_freeze()
    if out("mechanism_private.json").exists():
        raise SystemExit("mechanism_private.json exists; write-once")
    K = cd()["region"]["max_units_per_switch"]
    tg = read_json(art("p1/targets.json"))["targets"]
    rt = Runtime()
    b = rt.b
    tb = tok_bytes_fn(b.processor.tokenizer)
    FILL = fl()
    cls_of = {}
    for r in rt.tm:
        t_text, ids = tg[r["utterance_id"]]["target_text"], r["target_ids"]
        rec = utterance_region(t_text, ids, tb, FILL, K)
        um = token_unit_map(t_text, ids[:-1], tb)
        lab = lang_labels(units(t_text), FILL)
        c = []
        for t in range(len(ids)):
            if t == len(ids) - 1 or um is None:
                c.append("other")
            elif t in set(rec["region_tokens"]):
                c.append("SB")
            elif um[t] and all(lab[k] == "EN" for k in um[t]):
                c.append("EN")
            elif um[t] and all(lab[k] == "ZH" for k in um[t]):
                c.append("ZH_other")
            else:
                c.append("other")
        cls_of[r["utterance_id"]] = c
    ctl = control_runs()
    ck = {}
    for s in SEEDS:
        d1 = out("training", f"D1_seed{s}")
        sel = {"GoldKL": ctl[s]["selected_epoch"], "D1": read_json(d1 / "selection.json")["selected"]["epoch"]}
        for tag in ("selected", "epoch3"):
            for arm, rd in (("GoldKL", ctl[s]["dir"]), ("D1", d1)):
                ep = sel[arm] if tag == "selected" else 3
                ck[(s, tag, arm)] = {"path": rd / f"epoch{ep}" / "adapter.pt", "epoch": ep}
    CL = ("SB", "ZH_other", "EN", "other")
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    pair = collections.defaultdict(lambda: collections.defaultdict(list))
    geom = {}
    t0 = __import__("time").time()
    for s in SEEDS:
        ads = {}
        for key, v in ck.items():
            if key[0] != s:
                continue
            a = make_r2(b, s)
            a.load_state_dict(torch.load(v["path"]))
            a.module.eval()
            for p in a.module.parameters():
                p.requires_grad_(False)
            ads[key] = a
        for tag in ("selected", "epoch3"):
            Rg = ads[(s, tag, "GoldKL")].module.rot.weight.detach().cpu().numpy()
            Rd = ads[(s, tag, "D1")].module.rot.weight.detach().cpu().numpy()
            geom[f"{s}_{tag}"] = {"principal_cosines_D1_vs_GoldKL": principal_cosines(Rd, Rg),
                                  **{f"{arm}_{k}": float(v) for arm in ARMS for k, v in (
                                      ("bias_norm", ads[(s, tag, arm)].module.learned.bias.norm()),
                                      ("W_minus_R_norm", (ads[(s, tag, arm)].module.learned.weight - ads[(s, tag, arm)].module.rot.weight).norm()))},
                                  "epochs": {arm: ck[(s, tag, arm)]["epoch"] for arm in ARMS}}
        for j in range(0, len(rt.tm), 8):
            rows = rt.tm[j:j + 8]
            X, Y, enc = rt.tf_batch(rows)
            q, u, z0 = rt.site_record(enc, X)
            h = (q + u).float()
            M = teacher_mask(z0, Y)
            lp0 = torch.log_softmax(z0.float(), -1)
            dl_, dlt_ = {}, {}
            hn = h.norm(dim=-1).cpu().numpy()
            Mc = M.cpu().numpy()
            for key, a in ads.items():
                dlt = manual_delta(a, q, u)
                with torch.no_grad():
                    zs = rt.logits(a, enc, X)
                    kl = (lp0.exp() * (lp0 - torch.log_softmax(zs.float(), -1))).sum(-1)
                dlt_[key] = dlt
                dl_[key] = (dlt.norm(dim=-1).cpu().numpy(), kl.cpu().numpy())
            cosv = {}
            for tag in ("selected", "epoch3"):
                dg, dd = dlt_[(s, tag, "GoldKL")], dlt_[(s, tag, "D1")]
                cosv[tag] = ((dg * dd).sum(-1) / (dg.norm(dim=-1) * dd.norm(dim=-1) + 1e-30)).cpu().numpy()
            for i, r in enumerate(rows):
                c = cls_of[r["utterance_id"]]
                for t, cl in enumerate(c):
                    pos = 3 + t
                    tc = bool(Mc[i, pos])
                    for key, (dn, kl) in dl_.items():
                        for sub in ("all", "tc") if tc else ("all",):
                            st = stats[(key[1], key[2], s, sub)]
                            st[f"{cl}_dn"].append(float(dn[i, pos]))
                            st[f"{cl}_rel"].append(float(dn[i, pos] / hn[i, pos]))
                            st[f"{cl}_kl"].append(float(kl[i, pos]))
                    for tag in ("selected", "epoch3"):
                        pp = pair[(tag, s)]
                        pp[f"{cl}_cos"].append(float(cosv[tag][i, pos]))
                        pp[f"{cl}_ng"].append(float(dl_[(s, tag, "GoldKL")][0][i, pos]))
                        pp[f"{cl}_nd"].append(float(dl_[(s, tag, "D1")][0][i, pos]))
        del ads
    summ = {}
    for (tag, arm, s, sub), st in stats.items():
        d = summ.setdefault(tag, {}).setdefault(arm, {}).setdefault(str(s), {}).setdefault(sub, {})
        for cl in CL:
            if st[f"{cl}_dn"]:
                d[cl] = {"n": len(st[f"{cl}_dn"]), "dn_mean": float(np.mean(st[f"{cl}_dn"])), "dn_median": float(np.median(st[f"{cl}_dn"])),
                         "dn_p90": float(np.quantile(st[f"{cl}_dn"], 0.9)), "rel_mean": float(np.mean(st[f"{cl}_rel"])),
                         "kl_mean": float(np.mean(st[f"{cl}_kl"])), "kl_p90": float(np.quantile(st[f"{cl}_kl"], 0.9))}
    psum = {}
    for (tag, s), pp in pair.items():
        d = psum.setdefault(tag, {}).setdefault(str(s), {})
        for cl in CL:
            if pp[f"{cl}_cos"]:
                d[cl] = {"n": len(pp[f"{cl}_cos"]), "cos_mean": float(np.mean(pp[f"{cl}_cos"])), "cos_p10": float(np.quantile(pp[f"{cl}_cos"], 0.1)),
                         "norm_ratio_D1_over_GoldKL": float(np.mean(pp[f"{cl}_nd"]) / max(1e-30, np.mean(pp[f"{cl}_ng"])))}
        if "SB" in d and "EN" in d and "ZH_other" in d:
            d["selectivity_SB_vs_EN"] = d["SB"]["norm_ratio_D1_over_GoldKL"] / d["EN"]["norm_ratio_D1_over_GoldKL"]
            d["selectivity_SB_vs_ZH_other"] = d["SB"]["norm_ratio_D1_over_GoldKL"] / d["ZH_other"]["norm_ratio_D1_over_GoldKL"]
    write_json(out("mechanism_private.json"), {"utc": now(), **fz, "slurm": slurm(), "per_checkpoint": summ, "paired": psum, "geometry": geom,
                                               "checkpoints": {f"{k[0]}_{k[1]}_{k[2]}": v["epoch"] for k, v in ck.items()},
                                               "seconds": __import__("time").time() - t0,
                                               "note": "training utterances (seen in training), gold prefixes, teacher forcing; descriptive, not causal; "
                                                       "tc = teacher-correct positions; classes from training references (SB = D1 region tokens)"})
    print("mechanism written")


# ====================================================================== report

def selected_metrics(R_: dict, ep: int) -> dict:
    e = R_["epochs"][str(ep)]["beam5"]
    g = R_["epochs"][str(ep)]["greedy"]
    tw = e["target_window"]
    return {"zh_new": e["failure"]["zh_new"], "sb": e["failure"]["sb_units"], "tw_new": tw["new"], "tw_b0_correct": tw["b0_correct"],
            "tw_eligible": tw["eligible"], "tw_new_del": tw["new_del"], "tw_new_sub": tw["new_sub"], "tw_repairs": tw["repairs"],
            "tw_windows": tw["windows"], "zh_new_outside_tw": e["failure"]["zh_new"] - tw["new"],
            "loops": e["metrics"]["overall"]["loop_utterances"], "nonterm": e["metrics"]["overall"]["not_terminated"],
            "en_net": e["vs_B0"]["net"], "cs_mer": e["metrics"]["CS"]["mer"], "mer": e["metrics"]["overall"]["mer"],
            "en_wer": e["metrics"]["overall"]["en_wer"], "pier": e["metrics"]["overall"]["pier"], "zh_cer": e["metrics"]["overall"]["zh_cer"],
            "retention": e["vs_B0"]["mandarin_retention"], "worst_dlg": e["failure"]["worst_dialogue_delta_mixed"],
            "epoch": ep, "en_corr": e["vs_B0"]["substantive_en_corrections"], "en_corrupt": e["vs_B0"]["substantive_en_corruptions"],
            "zh_repairs": e["vs_B0"]["zh_repairs"], "outside_damage": e["vs_B0"]["outside_target_damage"],
            "zh_del_new": e["failure"]["zh_del_new"], "long_runs": e["failure"]["long_runs"], "near_switch": e["failure"]["near_switch"],
            "far": e["failure"]["far"], "zh_new_no_en_repair": e["failure"]["zh_new_in_no_en_repair_utts"],
            "dialogues_improved": e["dialogues_improved"], "dialogues_worsened": e["dialogues_worsened"], "spans": e["switchback_spans"],
            "en_fates": {k: e["english_fates"][k] for k in ("omission_to_correct", "translation_to_correct", "lexical_to_correct",
                                                             "correct_to_wrong", "wrong_to_other_wrong")},
            "greedy": {"cs_mer": g["metrics"]["CS"]["mer"], "mer": g["metrics"]["overall"]["mer"], "en_net": g["vs_B0"]["net"],
                       "zh_new": g["vs_B0"]["new_zh_errors"]}}


def components(arm: dict, D: dict) -> dict:
    d1, gk = arm["D1"], arm["GoldKL"]
    A, B, C = primary_gain(d1, gk, D), en_guard(d1, gk, D), mer_guard(d1, gk)
    comps = {"tw_new": component_gain(d1, gk, "tw_new", D), "sb": component_gain(d1, gk, "sb", D)}
    return {"A": A, "B": B, "C": C, "components": comps, "D_tw_support": tw_support(d1, gk, D),
            "E_strong": {"D1": strong(d1, D), "GoldKL": strong(gk, D)}}


def run_report():
    from reft.d1_regionkl_train import check_freeze, newest_preflight
    fz = check_freeze()
    if out("report.json").exists():
        raise SystemExit("report.json exists; write-once")
    Cf = cd()
    D = Cf["decision"]
    E = read_json(out("evaluation.json"))
    sel = {n: selected_metrics(R_, R_["selected_epoch"]) for n, R_ in E["runs"].items()}
    e3 = {n: selected_metrics(R_, 3) for n, R_ in E["runs"].items()}
    arm = {a: {s: sel[f"{a}_seed{s}"] for s in SEEDS} for a in ARMS}
    arm3 = {a: {s: e3[f"{a}_seed{s}"] for s in SEEDS} for a in ARMS}
    comp, comp3 = components(arm, D), components(arm3, D)
    ver = read_json(out("verify.json")) if out("verify.json").exists() else None
    pf = newest_preflight()
    pfr = read_json(pf)
    new = {k: v for k, v in E["runs"].items() if k.startswith("D1")}
    integ = {"preflight_ready": pfr["verdict"] == "D1_READY_FOR_TRAINING" and not pfr["dry_run"],
             "verify_ok": bool(ver and ver["all_ok"]),
             "three_runs_three_epochs": len(new) == 3 and all(len(R_["history"]) == 4 for R_ in new.values()),
             "updates_96": all(R_["history"][-1]["updates_total"] == Cf["training"]["updates_per_run"] for R_ in new.values()),
             "zero_init_150": all(R_["zero_init_identity"] == 150 for R_ in E["runs"].values()),
             "weights_unchanged": all(R_["weights_unchanged"] for R_ in E["runs"].values()),
             "parameter_counts": all(R_["efficiency"]["trainable_parameters"] == Cf["method"]["expected_trainable_parameters"] for R_ in new.values()),
             "goldkl_reproduces_p2_1b": all(v["ok"] for v in E["goldkl_reproduces_p2_1b"].values()),
             "paired_mask_tokens_updates": all(all(h["diagnostics"]["paired_with_GoldKL"].values()) for R_ in new.values() for h in R_["history"][1:]),
             "init_base_paired": all(R_["history"][0]["base_equal_GoldKL_seed"] for R_ in new.values()),
             "orthogonality": all(h["diagnostics"]["orth_err_max"] <= Cf["preflight"]["tolerances"]["orth_max_abs"] for R_ in new.values()
                                  for h in R_["history"][1:]),
             "regions_frozen_hash": all(R_["regions_sha256"] == fz["regions_sha256"] for R_ in new.values()),
             "region_weights_active_every_epoch": all(h["region_mask_positions"] > 0 for R_ in new.values() for h in R_["history"][1:]),
             "constant_lr": all(h["diagnostics"]["lr"] == [Cf["training"]["lr"]] for R_ in new.values() for h in R_["history"][1:])}
    dec = decide(all(integ.values()), comp["A"], comp["B"], comp["C"], comp["components"])
    mech = read_json(out("mechanism_private.json")) if out("mechanism_private.json").exists() else None
    rep = {"schema": "reft_d1_report_v1", "utc": now(), **fz, "git": git_state(), "decision": dec, "integrity": integ, "selected": sel,
           "epoch3": e3, "rules": comp, "epoch3_rules_descriptive": comp3, "paired_selected": E["paired_selected"], "paired_epoch3": E["paired_epoch3"],
           "epoch_matched": E["epoch_matched"], "mechanism": {k: mech[k] for k in ("per_checkpoint", "paired", "geometry", "checkpoints")} if mech else None,
           "preflight": pf.name, "rules_doc_sha256": sha(ROOT / "docs/reft/D1_REGIONKL_DECISION_RULES.md")}
    write_json(out("report.json"), rep)
    figures(E, rep)
    write_json(out("manifest.json"), {"utc": now(), **fz, "files": {p.name: sha(p) for p in sorted(out().glob("*.json*")) if p.name != "manifest.json"},
                                      "figures": {p.name: sha(p) for p in sorted(out("figures").glob("*.png"))}})
    print(json.dumps({k: rep[k] for k in ("decision", "integrity", "rules")}, indent=1, default=str))


def figures(E, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out("figures")
    fd.mkdir(parents=True, exist_ok=True)
    col = {"GoldKL": "#1f77b4", "D1": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for a in ARMS:
        for s in SEEDS:
            R_ = E["runs"][f"{a}_seed{s}"]
            for ep in ("1", "2", "3"):
                e = R_["epochs"][ep]["beam5"]
                big = int(ep) == R_["selected_epoch"]
                ax.scatter(e["failure"]["zh_new"], e["vs_B0"]["net"], c=col[a], s=50 if big else 14, marker="o" if big else ".")
                ax.annotate(f"{s}e{ep}", (e["failure"]["zh_new"], e["vs_B0"]["net"]), fontsize=6)
    ax.axvline(49.5, ls="--", c="grey")
    ax.set(xlabel="new Mandarin errors vs B0 (of 4,950)", ylabel="English net corrections vs B0",
           title="F1 correction-preservation (blue GoldKL, red D1; large = selected)")
    fig.tight_layout()
    fig.savefig(fd / "F1_frontier.png", dpi=130)
    plt.close(fig)
    sel = rep["selected"]
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, (k, lab) in zip(axs, (("zh_new", "new Mandarin errors"), ("tw_new", "target-window new errors"), ("sb", "long SB dropped units"),
                                  ("en_net", "English net"), ("cs_mer", "CS MER"))):
        for s in SEEDS:
            ax.plot([0, 1], [sel[f"{a}_seed{s}"][k] for a in ARMS], "o-", label=str(s))
        ax.set_xticks([0, 1], list(ARMS))
        ax.set_title(lab, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("F2 matched seeds, selected checkpoints")
    fig.tight_layout()
    fig.savefig(fd / "F2_matched.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for s in SEEDS:
        hg = E["runs"][f"GoldKL_seed{s}"]["history"][1:]
        hd = E["runs"][f"D1_seed{s}"]["history"][1:]
        ep = [1, 2, 3]
        axs[0].plot(ep, [x["train_ce"] for x in hg], "o-", c=col["GoldKL"], alpha=0.7)
        axs[0].plot(ep, [x["train_ce"] for x in hd], "o-", c=col["D1"], alpha=0.7)
        axs[1].plot(ep, [x["train_kl"] for x in hg], "o-", c=col["GoldKL"], alpha=0.7)
        axs[1].plot(ep, [x["train_kl"] for x in hd], "o--", c=col["D1"], alpha=0.7)
        axs[1].plot(ep, [x["train_regionkl"] for x in hd], "x-", c=col["D1"], alpha=0.7)
        axs[2].plot(ep, [x["grad_norm_mean"] for x in hg], "o-", c=col["GoldKL"], alpha=0.7)
        axs[2].plot(ep, [x["grad_norm_mean"] for x in hd], "o-", c=col["D1"], alpha=0.7)
    for ax, t in zip(axs, ("train CE", "GoldKL (o) / D1 unweighted (o--) and RegionKL (x)", "grad norm (pre-clip mean)")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(fd / "F3_training.png", dpi=130)
    plt.close(fig)
    if rep["mechanism"]:
        ps = rep["mechanism"]["paired"]["selected"]
        fig, ax = plt.subplots(figsize=(7, 4))
        cl = ("SB", "ZH_other", "EN", "other")
        x = np.arange(len(cl))
        for i, s in enumerate(SEEDS):
            ax.bar(x + (i - 1) * 0.25, [ps[str(s)].get(c, {}).get("norm_ratio_D1_over_GoldKL", np.nan) for c in cl], 0.25, label=f"seed {s}")
        ax.axhline(1, c="black", lw=0.6)
        ax.set_xticks(x, cl)
        ax.set_ylabel("mean |Δh| D1 / GoldKL")
        ax.legend(fontsize=7)
        ax.set_title("F4 edit-norm ratio by training-token region (selected; descriptive)", fontsize=9)
        fig.tight_layout()
        fig.savefig(fd / "F4_edit_ratio.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "evaluate", "mech", "report"])
    a = ap.parse_args(argv)
    {"select": run_select, "evaluate": run_evaluate, "mech": run_mech, "report": run_report}[a.cmd]()


if __name__ == "__main__":
    main()
