#!/usr/bin/env python
"""D2 selection, canonical evaluation, pre-registered verdict and bounded training-only diagnostics.
docs/reft/D2_EDIT_LOCALITY_DECISION_RULES.md.

  select     (CPU) unchanged P2-1 checkpoint rule per D2 run -> training/D2_seed{s}/selection.json (write-once)
  evaluate   (CPU) R2-GoldKL (historical, recomputed for parity with P2-1b) and D2 per seed/epoch/regime vs the pinned B0
             with the P2-1b / P2-2 / D1 metric code; paired D2 - GoldKL (selected and epoch 3), shared dialogue draws
             -> d2_edit_locality/evaluation.json
  mech       (GPU, training utterances only) edit energy / norms, token KL and edit geometry of GoldKL vs D2 by token class
  prefix     (GPU, training utterances only) same-prefix adapter-off vs adapter-on next-token comparison on gold, B0-generated
             and own-generated prefixes -> prefix_diagnostic_private.json
  report     (CPU) deterministic verdict, figures -> d2_edit_locality/report.json, manifest.json

The generic pure helpers of reft/d1_regionkl_evaluate.py (unchanged, frozen in the D1 chain) are reused: target-window
metric, A/B/C/component functions, dialogue tables, principal cosines and per-checkpoint metric extraction.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, fillers, read_json, read_jsonl, write_json  # noqa: E402
from reft.d1_regionkl_evaluate import (KEYS, component_gain, dialogue_table, en_guard, mer_guard, primary_gain,  # noqa: E402
                                       principal_cosines, selected_metrics, strong, tab_stat, tw_from_events, tw_support)
from reft.provenance import git_state, now, sha, slurm  # noqa: E402

SEEDS = (42, 43, 44)
ARMS = ("GoldKL", "D2")
NEW = "D2"


def cd():
    from reft.d2_edit_locality_train import cd as c
    return c()


def out(*p):
    from reft.d2_edit_locality_train import out as o
    return o(*p)


# ====================================================================== pure helpers (unit-tested)

RECOMMEND = {"D2_INVALID": "HOLD_FOR_ENGINEERING_REPAIR",
             "D2_INCREMENTAL_PRESERVATION_GAIN": "GO_LOCALITY_SPECIFICITY_CONTROL",
             "D2_MIXED_TRADEOFF": "CHARACTERIZE_CORRECTION_PRESERVATION_TRADEOFF",
             "D2_NO_INCREMENTAL_GAIN": "STOP_GOLD_PREFIX_LOCALITY_OBJECTIVE_EXPANSION"}


def decide(integrity_ok: bool, A: dict, B: dict, C: dict, comps: dict) -> dict:
    """INVALID > INCREMENTAL (A, B, C) > MIXED (A but not B-and-C; or not A but a consistent secondary preservation
    component gain) > NO_INCREMENTAL_GAIN. Representation-norm reductions never enter."""
    if not integrity_ok:
        v = "D2_INVALID"
    elif A["pass"] and B["pass"] and C["pass"]:
        v = "D2_INCREMENTAL_PRESERVATION_GAIN"
    elif A["pass"] or any(c["pass"] for c in comps.values()):
        v = "D2_MIXED_TRADEOFF"
    else:
        v = "D2_NO_INCREMENTAL_GAIN"
    return {"verdict": v, "recommendation": RECOMMEND[v]}


def components(arm: dict, D: dict) -> dict:
    """Pre-registered rule components on {"D2": {seed: metrics}, "GoldKL": {seed: metrics}}."""
    d2, gk = arm[NEW], arm["GoldKL"]
    A, B, C = primary_gain(d2, gk, D), en_guard(d2, gk, D), mer_guard(d2, gk)
    comps = {"tw_new": component_gain(d2, gk, "tw_new", D), "sb": component_gain(d2, gk, "sb", D)}
    return {"A": A, "B": B, "C": C, "components": comps, "D_tw_support": tw_support(d2, gk, D),
            "E_strong": {NEW: strong(d2, D), "GoldKL": strong(gk, D)}}


def paired_tables(dtabs: dict, dl: list, draws: list) -> dict:
    """Paired D2 - GoldKL per seed (point, dialogue-bootstrap CI over shared draws, worst-dialogue contribution, leave-one-
    dialogue-out range) and the seed mean (pooled over the same draws). Metrics recomputed from resampled dialogue counts.
    (reft.d1_regionkl_evaluate.paired_tables with the new arm named D2.)"""
    full = list(range(len(dl)))
    pr = {}
    for s in SEEDS:
        x, y = dtabs[(NEW, s)], dtabs[("GoldKL", s)]
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
            pooled[k].append(np.mean([tab_stat(dtabs[(NEW, s)], dl, idx)[k] - tab_stat(dtabs[("GoldKL", s)], dl, idx)[k] for s in SEEDS]))
    pr["seed_mean"] = {k: {"point": float(np.mean([pr[s][k]["point"] for s in SEEDS])),
                           "range": [float(min(pr[s][k]["point"] for s in SEEDS)), float(max(pr[s][k]["point"] for s in SEEDS))],
                           "ci95": [float(np.quantile(pooled[k], q)) for q in (0.025, 0.975)]} for k in KEYS}
    return {str(k): v for k, v in pr.items()}


def token_script(bts: bytes, is_eos: bool) -> str:
    """Reference-free class of one decoder token: 'eos', 'han' (decodes to text containing a CJK ideograph), 'partial'
    (non-empty bytes that do not decode on their own and are non-ASCII: a fragment of a multi-byte character), 'latin'
    (contains an ASCII letter), or 'other'."""
    if is_eos:
        return "eos"
    try:
        s = bts.decode("utf-8")
    except UnicodeDecodeError:
        return "partial" if any(x >= 0x80 for x in bts) else "other"
    if any("㐀" <= c <= "鿿" or "豈" <= c <= "﫿" for c in s):
        return "han"
    if any(c.isascii() and c.isalpha() for c in s):
        return "latin"
    return "other"


def zhlike(cls: str) -> bool:
    return cls in ("han", "partial")


def position_stats(lp_off, lp_on, cls_off: list[str], cls_on: list[str], start: int = 3, end: int | None = None) -> dict:
    """Same-prefix comparison at decoder positions [start, end): lp_off / lp_on are float32 log-probabilities [T, V] from
    the frozen and the adapted forward on the identical audio and decoder input. cls_* are the reference-free classes of
    each position's top-1 token. Returns per-position KL(p_off || p_on), top-1 change and Han-to-non-Han change, grouped
    by the frozen top-1 class."""
    end = lp_off.shape[0] if end is None else end
    kl = (lp_off.exp() * (lp_off - lp_on)).sum(-1)[start:end].tolist()
    a0 = lp_off.argmax(-1)[start:end].tolist()
    a1 = lp_on.argmax(-1)[start:end].tolist()
    g =collections.defaultdict(lambda: {"n": 0, "kl": 0.0, "top1_change": 0, "zh_to_nonzh": 0, "to_eos": 0})
    for j, p in enumerate(range(start, end)):
        c0 = cls_off[p]
        for key in (c0, "all", "zhlike" if zhlike(c0) else "nonzh"):
            x = g[key]
            x["n"] += 1
            x["kl"] += kl[j]
            x["top1_change"] += a0[j] != a1[j]
            x["zh_to_nonzh"] += zhlike(c0) and not zhlike(cls_on[p])
            x["to_eos"] += cls_on[p] == "eos" and c0 != "eos"
    return {k: dict(v) for k, v in g.items()}


def merge_stats(parts: list[dict]) -> dict:
    t = collections.defaultdict(lambda: collections.Counter())
    for p in parts:
        for k, v in p.items():
            t[k].update(v)
    res = {}
    for k, v in t.items():
        n = v["n"]
        res[k] = {"n": int(n), "kl_mean": v["kl"] / n if n else None, "top1_change_rate": v["top1_change"] / n if n else None,
                  "zh_to_nonzh_rate": v["zh_to_nonzh"] / n if n else None, "to_eos_rate": v["to_eos"] / n if n else None}
    return res


def prefix_cases(uids_with_sb: list[str], n: int) -> list[str]:
    """Frozen prefix-diagnostic population: sha256('d2-prefix:' + id) order, first n."""
    return sorted(uids_with_sb, key=lambda u: hashlib.sha256(("d2-prefix:" + u).encode()).hexdigest())[:n]


# ====================================================================== runs, selection

def run_dirs() -> dict:
    from reft.d2_edit_locality_train import control_runs
    d = {("GoldKL", s): c["dir"] for s, c in control_runs().items()}
    for s in SEEDS:
        d[(NEW, s)] = out("training", f"D2_seed{s}")
    return d


def run_select():
    from reft.d2_edit_locality_train import check_freeze
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, score_system, select_epoch
    check_freeze()
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    for s in SEEDS:
        rd = out("training", f"D2_seed{s}")
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
    from reft.d2_edit_locality_train import check_freeze
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    from reft.p2_1b_evaluate import failure, fates
    from reft.p2_1d_audit import utterance_events
    from reft.p2_2_evaluate import span_fates
    fz = check_freeze()
    if out("evaluation.json").exists():
        raise SystemExit("evaluation.json exists; write-once")
    C, D = p21(), cd()
    K = D["evaluation"]["K"]
    BS = D["bootstrap"]
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    LU = C["evaluation"]["loop_units"]
    pb = C["frozen_baseline"]
    assert sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"]
    base = {r: score_system(meta, refs, {x["id"]: x for x in read_jsonl(art(pb[r]))}, FILL, LU) for r in ("beam5", "greedy")}
    runs, dtabs, eptabs = {}, {}, {}
    for (arm, s), rd in run_dirs().items():
        sel = read_json(rd / "selection.json")["selected"]["epoch"]
        man = read_json(rd / "manifest.json")
        R = {"selected_epoch": sel, "history": man["history"], "efficiency": man.get("efficiency"), "weights_unchanged": man["weights_unchanged"],
             "zero_init_identity": man["history"][0]["zero_init_identity_beam5"], "mask_sha256": man.get("mask_sha256"), "mu": man.get("mu"),
             "epochs": {}}
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
        ok = mine["selected_epoch"] == theirs["selected_epoch"]
        for ep in (1, 2, 3):
            a, t = mine["epochs"][ep]["beam5"], theirs["epochs"][str(ep)]["beam5"]
            ok = ok and all(a["failure"][k] == t["failure"][k] for k in ("zh_new", "sb_units", "worst_dialogue_delta_mixed")) \
                and a["vs_B0"]["net"] == t["vs_B0"]["net"] and a["metrics"]["CS"]["mer"] == t["metrics"]["CS"]["mer"] \
                and a["metrics"]["overall"]["mer"] == t["metrics"]["overall"]["mer"]
        repro[f"GoldKL_{s}"] = {"ok": bool(ok), "selected_epoch": mine["selected_epoch"]}
    dl = sorted({m["dialogue_id"] for m in meta})
    rng = np.random.default_rng(BS["seed"])
    draws = [rng.choice(len(dl), len(dl)) for _ in range(BS["reps"])]
    ep3 = {(a, s): eptabs[(a, s, 3)] for a in ARMS for s in SEEDS}
    em = {str(s): {str(ep): {k: tab_stat(eptabs[(NEW, s, ep)], dl, range(len(dl)))[k] - tab_stat(eptabs[("GoldKL", s, ep)], dl, range(len(dl)))[k]
                             for k in KEYS} for ep in (1, 2, 3)} for s in SEEDS}
    write_json(out("evaluation.json"), {"schema": "reft_d2_evaluation_v1", "utc": now(), **fz, "git": git_state(), "slurm": slurm(),
                                        "B0": {r: metrics(base[r]) for r in base}, "runs": runs, "goldkl_reproduces_p2_1b": repro,
                                        "paired_selected": paired_tables(dtabs, dl, draws), "paired_epoch3": paired_tables(ep3, dl, draws),
                                        "epoch_matched": em, "n_dialogues": len(dl), "K": K,
                                        "note": "dev150 exposed and used for selection; seeds share the 20 dialogues; descriptive uncertainty"})
    print("evaluation written", repro)


# ====================================================================== shared GPU helpers

def token_classes(rt, mask: dict) -> dict:
    """Per training utterance and target index: ZH_SB, ZH_other, EN or other (EOS, ambiguous, unmapped)."""
    cls_of = {}
    for r in rt.tm:
        m = mask[r["utterance_id"]]
        zh, sb, en = set(m["zh_tokens"]), set(m["zh_sb_tokens"]), set(m["en_tokens"])
        cls_of[r["utterance_id"]] = ["ZH_SB" if t in sb else "ZH_other" if t in zh else "EN" if t in en else "other"
                                     for t in range(len(r["target_ids"]))]
    return cls_of


def checkpoints() -> dict:
    from reft.d2_edit_locality_train import control_runs
    ctl = control_runs()
    ck = {}
    for s in SEEDS:
        d2 = out("training", f"D2_seed{s}")
        sel = {"GoldKL": ctl[s]["selected_epoch"], NEW: read_json(d2 / "selection.json")["selected"]["epoch"]}
        for tag in ("selected", "epoch3"):
            for arm, rd in (("GoldKL", ctl[s]["dir"]), (NEW, d2)):
                ep = sel[arm] if tag == "selected" else 3
                ck[(s, tag, arm)] = {"path": rd / f"epoch{ep}" / "adapter.pt", "epoch": ep}
    return ck


def load_frozen_adapter(b, s, path):
    import torch
    from reft.d2_edit_locality_train import make_r2
    a = make_r2(b, s)
    a.load_state_dict(torch.load(path))
    a.module.eval()
    for p in a.module.parameters():
        p.requires_grad_(False)
    return a


# ====================================================================== mechanism (GPU, training utterances only)

def run_mech():
    """Training-only representation diagnostics at the intervention site (gold prefixes, all 500 training utterances):
    ||delta h||, ||delta h||^2 / 1280, ||delta h|| / ||h|| and token KL(p0 || p_theta) by token class (ZH_SB, ZH_other,
    EN, other; all and teacher-correct positions); D2/GoldKL norm ratios and per-position edit cosines by class; principal
    cosines of the R row spaces, ||b|| and ||W - R||."""
    import torch
    from reft.arch_s0s1_train import Runtime, manual_delta
    from reft.d2_edit_locality_train import check_freeze, load_mask
    from reft.p2_1b_train import teacher_mask
    fz = check_freeze()
    if out("mechanism_private.json").exists():
        raise SystemExit("mechanism_private.json exists; write-once")
    mask = load_mask()
    rt = Runtime()
    b = rt.b
    cls_of = token_classes(rt, mask)
    ck = checkpoints()
    CL = ("ZH_SB", "ZH_other", "EN", "other")
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    pair = collections.defaultdict(lambda: collections.defaultdict(list))
    geom = {}
    t0 = time.time()
    for s in SEEDS:
        ads = {key: load_frozen_adapter(b, s, v["path"]) for key, v in ck.items() if key[0] == s}
        for tag in ("selected", "epoch3"):
            Rg = ads[(s, tag, "GoldKL")].module.rot.weight.detach().cpu().numpy()
            Rd = ads[(s, tag, NEW)].module.rot.weight.detach().cpu().numpy()
            geom[f"{s}_{tag}"] = {"principal_cosines_D2_vs_GoldKL": principal_cosines(Rd, Rg),
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
                dg, dd = dlt_[(s, tag, "GoldKL")], dlt_[(s, tag, NEW)]
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
                        for sub in ("all", "tc") if tc else ("all",):
                            pp = pair[(tag, s, sub)]
                            pp[f"{cl}_cos"].append(float(cosv[tag][i, pos]))
                            pp[f"{cl}_ng"].append(float(dl_[(s, tag, "GoldKL")][0][i, pos]))
                            pp[f"{cl}_nd"].append(float(dl_[(s, tag, NEW)][0][i, pos]))
                            pp[f"{cl}_kg"].append(float(dl_[(s, tag, "GoldKL")][1][i, pos]))
                            pp[f"{cl}_kd"].append(float(dl_[(s, tag, NEW)][1][i, pos]))
        del ads
    summ = {}
    for (tag, arm, s, sub), st in stats.items():
        d = summ.setdefault(tag, {}).setdefault(arm, {}).setdefault(str(s), {}).setdefault(sub, {})
        for cl in CL:
            if st[f"{cl}_dn"]:
                dn = np.array(st[f"{cl}_dn"])
                d[cl] = {"n": len(dn), "dn_mean": float(dn.mean()), "dn_median": float(np.median(dn)), "dn_p90": float(np.quantile(dn, 0.9)),
                         "energy_mean": float((dn ** 2).mean() / 1280), "rel_mean": float(np.mean(st[f"{cl}_rel"])),
                         "kl_mean": float(np.mean(st[f"{cl}_kl"])), "kl_p90": float(np.quantile(st[f"{cl}_kl"], 0.9))}
    psum = {}
    for (tag, s, sub), pp in pair.items():
        d = psum.setdefault(tag, {}).setdefault(str(s), {}).setdefault(sub, {})
        for cl in CL:
            if pp[f"{cl}_cos"]:
                ng, nd = np.array(pp[f"{cl}_ng"]), np.array(pp[f"{cl}_nd"])
                d[cl] = {"n": len(ng), "cos_mean": float(np.mean(pp[f"{cl}_cos"])), "cos_p10": float(np.quantile(pp[f"{cl}_cos"], 0.1)),
                         "norm_ratio_D2_over_GoldKL": float(nd.mean() / max(1e-30, ng.mean())),
                         "energy_ratio_D2_over_GoldKL": float((nd ** 2).mean() / max(1e-30, (ng ** 2).mean())),
                         "kl_ratio_D2_over_GoldKL": float(np.mean(pp[f"{cl}_kd"]) / max(1e-30, np.mean(pp[f"{cl}_kg"])))}
        zr = [d[c]["norm_ratio_D2_over_GoldKL"] for c in ("ZH_SB", "ZH_other") if c in d]
        if "EN" in d and zr:
            nz = sum(d[c]["n"] for c in ("ZH_SB", "ZH_other") if c in d)
            zh_ratio = sum(d[c]["norm_ratio_D2_over_GoldKL"] * d[c]["n"] for c in ("ZH_SB", "ZH_other") if c in d) / nz
            d["selectivity_ZH_vs_EN"] = zh_ratio / d["EN"]["norm_ratio_D2_over_GoldKL"]
            d["zh_norm_ratio_weighted"] = zh_ratio
    write_json(out("mechanism_private.json"), {"utc": now(), **fz, "slurm": slurm(), "per_checkpoint": summ, "paired": psum, "geometry": geom,
                                               "checkpoints": {f"{k[0]}_{k[1]}_{k[2]}": v["epoch"] for k, v in ck.items()},
                                               "seconds": time.time() - t0,
                                               "note": "training utterances (seen in training), gold prefixes, teacher forcing; descriptive, not causal; "
                                                       "tc = teacher-correct positions (ZH tc = the D2 edit mask); selectivity = ZH ratio / EN ratio"})
    print("mechanism written")


# ====================================================================== same-prefix diagnostic (GPU, training utterances only)

def run_prefix():
    """For each frozen case (training audio) and each prefix source (gold reference, B0 beam-5 output, the system's own
    beam-5 output), the frozen and the adapted model score the identical decoder input; next-token distributions are
    compared position by position within that row only. Descriptive; never changes mu, selection or the verdict."""
    import torch
    from reft.arch_s0s1_train import Runtime
    from reft.d2_edit_locality_train import check_freeze, load_mask
    from reft.text import tok_bytes_fn
    from reft.train_supervised import encode
    fz = check_freeze()
    if out("prefix_diagnostic_private.json").exists():
        raise SystemExit("prefix_diagnostic_private.json exists; write-once")
    C = cd()
    mask = load_mask()
    rt = Runtime()
    b = rt.b
    tb = tok_bytes_fn(b.processor.tokenizer)
    eos = rt.eos
    byid = {r["utterance_id"]: r for r in rt.tm}
    cases = prefix_cases([u for u, m in mask.items() if m["zh_sb_tokens"]], C["prefix_diagnostic"]["n_cases"])
    ck = checkpoints()
    systems = {f"{arm}_{s}": (s, ck[(s, "selected", arm)]) for s in SEEDS for arm in ARMS}
    ads = {k: load_frozen_adapter(b, s, v["path"]) for k, (s, v) in systems.items()}
    cls_cache = {}

    def cls_tok(i):
        if i not in cls_cache:
            cls_cache[i] = token_script(tb(i), i == eos)
        return cls_cache[i]

    def lp_rows(enc, X, ad):
        with torch.no_grad():
            z = rt.logits(ad, enc, X)
        return torch.log_softmax(z[0].float(), -1)

    t0 = time.time()
    parts = collections.defaultdict(list)
    ref_parts = collections.defaultdict(list)
    cases_out = []
    for uid in cases:
        row = byid[uid]
        _, enc = encode(b, [row["audio_path"]])
        b0, _ = rt.decode(row, 5, None)
        own = {k: rt.decode(row, 5, a)[0] for k, a in ads.items()}
        seqs = {"gold": row["target_ids"], "b0": b0["tokens"]}
        crec = {"case": hashlib.sha256(uid.encode()).hexdigest()[:12], "gold_len": len(row["target_ids"]), "b0_len": len(b0["tokens"]),
                "b0_terminated": b0["tokens"][-1:] == [eos], "b0_equals_gold": b0["tokens"] == row["target_ids"],
                "han_tokens": {"gold": sum(zhlike(cls_tok(i)) for i in row["target_ids"]), "b0": sum(zhlike(cls_tok(i)) for i in b0["tokens"]),
                               **{k: sum(zhlike(cls_tok(i)) for i in v["tokens"]) for k, v in own.items()}},
                "own_equals_b0": {k: v["tokens"] == b0["tokens"] for k, v in own.items()}, "rows": {}}
        for src in ("gold", "b0"):
            tok = seqs[src]
            if len(tok) < 2 or tok[-1] != eos:
                crec["rows"][src] = "ineligible"
                continue
            X = torch.tensor([row["prompt"] + tok[:-1]], device=b.device)
            lp_off = lp_rows(enc, X, None)
            c_off = [cls_tok(i) for i in lp_off.argmax(-1).tolist()]
            for k, a in ads.items():
                lp_on = lp_rows(enc, X, a)
                c_on = [cls_tok(i) for i in lp_on.argmax(-1).tolist()]
                parts[(src, k)].append(position_stats(lp_off, lp_on, c_off, c_on))
                if src == "gold":
                    kl = (lp_off.exp() * (lp_off - lp_on)).sum(-1)
                    m = mask[uid]
                    tc = (lp_off.argmax(-1) == torch.tensor(row["prompt"][1:] + tok, device=b.device)[:X.shape[1]])
                    for name, toks in (("ZH", m["zh_tokens"]), ("ZH_SB", m["zh_sb_tokens"]), ("EN", m["en_tokens"])):
                        sel = [3 + t for t in toks if bool(tc[3 + t])]
                        ref_parts[(k, name)].append((float(kl[sel].sum()) if sel else 0.0, len(sel)))
            crec["rows"][src] = "eligible"
        for k, a in ads.items():
            tok = own[k]["tokens"]
            if len(tok) < 2 or tok[-1] != eos:
                crec["rows"][f"own_{k}"] = "ineligible"
                continue
            X = torch.tensor([row["prompt"] + tok[:-1]], device=b.device)
            lp_off, lp_on = lp_rows(enc, X, None), lp_rows(enc, X, a)
            parts[("own", k)].append(position_stats(lp_off, lp_on, [cls_tok(i) for i in lp_off.argmax(-1).tolist()],
                                                    [cls_tok(i) for i in lp_on.argmax(-1).tolist()]))
            crec["rows"][f"own_{k}"] = "eligible"
        cases_out.append(crec)
    agg = {f"{src}|{k}": merge_stats(v) for (src, k), v in parts.items()}
    ref = {f"{k}|{name}": {"n": sum(n for _, n in v), "kl_mean": (sum(x for x, _ in v) / sum(n for _, n in v)) if sum(n for _, n in v) else None}
           for (k, name), v in ref_parts.items()}
    paired_ = {}
    for s in SEEDS:
        d = {}
        for src in ("gold", "b0", "own"):
            g, n = agg.get(f"{src}|GoldKL_{s}", {}), agg.get(f"{src}|{NEW}_{s}", {})
            for cl in ("zhlike", "nonzh", "all"):
                if cl in g and cl in n and g[cl]["kl_mean"]:
                    d[f"{src}_{cl}"] = {"kl_ratio_D2_over_GoldKL": n[cl]["kl_mean"] / g[cl]["kl_mean"],
                                        "zh_to_nonzh_GoldKL": g[cl]["zh_to_nonzh_rate"], "zh_to_nonzh_D2": n[cl]["zh_to_nonzh_rate"],
                                        "top1_change_GoldKL": g[cl]["top1_change_rate"], "top1_change_D2": n[cl]["top1_change_rate"]}
        if "gold_zhlike" in d and "b0_zhlike" in d:
            d["transfer_b0_over_gold"] = d["b0_zhlike"]["kl_ratio_D2_over_GoldKL"] / d["gold_zhlike"]["kl_ratio_D2_over_GoldKL"]
        if "gold_zhlike" in d and "own_zhlike" in d:
            d["transfer_own_over_gold"] = d["own_zhlike"]["kl_ratio_D2_over_GoldKL"] / d["gold_zhlike"]["kl_ratio_D2_over_GoldKL"]
        paired_[str(s)] = d
    elig = collections.Counter(v for c in cases_out for v in c["rows"].values())
    han = {k: float(np.mean([c["han_tokens"][k] - c["han_tokens"]["b0"] for c in cases_out])) for k in ads}
    write_json(out("prefix_diagnostic_private.json"), {
        "utc": now(), **fz, "slurm": slurm(), "rule": C["prefix_diagnostic"], "n_cases": len(cases), "cases": cases_out,
        "row_eligibility": dict(elig), "aggregate": agg, "reference_classes_gold_tc": ref, "paired": paired_,
        "own_minus_b0_han_tokens_mean": han, "seconds": time.time() - t0,
        "note": "training audio; within each row the audio and decoder input are identical for the frozen and adapted forward; rows are "
                "compared only through aggregates; classes are the frozen top-1 token script; descriptive, not whole-sequence causality"})
    print(json.dumps({"paired": paired_, "eligibility": dict(elig), "han": han}, indent=1))


# ====================================================================== report

def run_report():
    from reft.d2_edit_locality_train import check_freeze, newest_preflight
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
    cal = read_json(out("calibration_private.json"))
    new = {k: v for k, v in E["runs"].items() if k.startswith(NEW)}
    integ = {"preflight_ready": pfr["verdict"] == "D2_READY_FOR_TRAINING" and not pfr["dry_run"],
             "calibration_ok": cal["verdict"] == "D2_CALIBRATION_OK" and not cal["dry_run"],
             "mu_frozen_in_every_run": all(R_["mu"] == fz["mu"] for R_ in new.values()),
             "verify_ok": bool(ver and ver["all_ok"]),
             "three_runs_three_epochs": len(new) == 3 and all(len(R_["history"]) == 4 for R_ in new.values()),
             "updates_96": all(R_["history"][-1]["updates_total"] == Cf["training"]["updates_per_run"] for R_ in new.values()),
             "zero_init_150": all(R_["zero_init_identity"] == 150 for R_ in E["runs"].values()),
             "identity_init": all(R_["history"][0]["identity_init"] for R_ in new.values()),
             "weights_unchanged": all(R_["weights_unchanged"] for R_ in E["runs"].values()),
             "parameter_counts": all(R_["efficiency"]["trainable_parameters"] == Cf["method"]["expected_trainable_parameters"] for R_ in new.values()),
             "goldkl_reproduces_p2_1b": all(v["ok"] for v in E["goldkl_reproduces_p2_1b"].values()),
             "paired_mask_tokens_updates": all(all(h["diagnostics"]["paired_with_GoldKL"].values()) for R_ in new.values() for h in R_["history"][1:]),
             "init_base_paired": all(R_["history"][0]["base_equal_GoldKL_seed"] for R_ in new.values()),
             "orthogonality": all(h["diagnostics"]["orth_err_max"] <= Cf["preflight"]["tolerances"]["orth_max_abs"] for R_ in new.values()
                                  for h in R_["history"][1:]),
             "mask_frozen_hash": all(R_["mask_sha256"] == fz["mask_sha256"] for R_ in new.values()),
             "edit_term_active_every_epoch": all(h["edit_mask_positions"] > 0 and (h["train_mu_edit"] or 0) > 0 for R_ in new.values()
                                                 for h in R_["history"][1:]),
             "constant_lr": all(h["diagnostics"]["lr"] == [Cf["training"]["lr"]] for R_ in new.values() for h in R_["history"][1:])}
    dec = decide(all(integ.values()), comp["A"], comp["B"], comp["C"], comp["components"])
    mech = read_json(out("mechanism_private.json")) if out("mechanism_private.json").exists() else None
    pre = read_json(out("prefix_diagnostic_private.json")) if out("prefix_diagnostic_private.json").exists() else None
    rep = {"schema": "reft_d2_report_v1", "utc": now(), **fz, "git": git_state(), "decision": dec, "integrity": integ, "selected": sel,
           "epoch3": e3, "rules": comp, "epoch3_rules_descriptive": comp3, "paired_selected": E["paired_selected"], "paired_epoch3": E["paired_epoch3"],
           "epoch_matched": E["epoch_matched"],
           "calibration": {k: cal[k] for k in ("mu", "mu_repeat", "median_g_base", "median_g_edit", "checks", "identity", "verdict")}
           | {"per_batch": [{k: r[k] for k in ("g_base", "g_edit", "cos_base_edit", "n_edit", "edit", "base_loss", "mu_edit_over_base_loss")}
                            for r in cal["passes"][0]]},
           "mechanism": {k: mech[k] for k in ("per_checkpoint", "paired", "geometry", "checkpoints")} if mech else None,
           "prefix": {k: pre[k] for k in ("n_cases", "row_eligibility", "aggregate", "reference_classes_gold_tc", "paired",
                                          "own_minus_b0_han_tokens_mean")} if pre else None,
           "preflight": pf.name, "rules_doc_sha256": sha(ROOT / "docs/reft/D2_EDIT_LOCALITY_DECISION_RULES.md")}
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
    col = {"GoldKL": "#1f77b4", NEW: "#2ca02c"}
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
           title="F1 correction-preservation (blue GoldKL, green D2; large = selected)")
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
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    for s in SEEDS:
        hg = E["runs"][f"GoldKL_seed{s}"]["history"][1:]
        hd = E["runs"][f"{NEW}_seed{s}"]["history"][1:]
        ep = [1, 2, 3]
        axs[0].plot(ep, [x["train_ce"] for x in hg], "o-", c=col["GoldKL"], alpha=0.7)
        axs[0].plot(ep, [x["train_ce"] for x in hd], "o-", c=col[NEW], alpha=0.7)
        axs[1].plot(ep, [x["train_kl"] for x in hg], "o-", c=col["GoldKL"], alpha=0.7)
        axs[1].plot(ep, [x["train_kl"] for x in hd], "o-", c=col[NEW], alpha=0.7)
        axs[2].plot(ep, [x["train_edit"] for x in hd], "o-", c=col[NEW], alpha=0.7, label=f"ZH tc {s}")
        axs[2].plot(ep, [x["train_e_en_tc"] for x in hd], "x--", c=col[NEW], alpha=0.7)
        axs[3].plot(ep, [x["grad_norm_mean"] for x in hg], "o-", c=col["GoldKL"], alpha=0.7)
        axs[3].plot(ep, [x["grad_norm_mean"] for x in hd], "o-", c=col[NEW], alpha=0.7)
    for ax, t in zip(axs, ("train CE", "train GoldKL", "D2 edit energy /1280 (o ZH tc = L_edit, x EN tc)", "grad norm (pre-clip mean)")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(fd / "F3_training.png", dpi=130)
    plt.close(fig)
    if rep["mechanism"]:
        ps = rep["mechanism"]["paired"]["selected"]
        fig, ax = plt.subplots(figsize=(7, 4))
        cl = ("ZH_SB", "ZH_other", "EN", "other")
        x = np.arange(len(cl))
        for i, s in enumerate(SEEDS):
            ax.bar(x + (i - 1) * 0.25, [ps[str(s)]["tc" if c != "other" else "all"].get(c, {}).get("norm_ratio_D2_over_GoldKL", np.nan) for c in cl],
                   0.25, label=f"seed {s}")
        ax.axhline(1, c="black", lw=0.6)
        ax.set_xticks(x, cl)
        ax.set_ylabel("mean |Δh| D2 / GoldKL")
        ax.legend(fontsize=7)
        ax.set_title("F4 edit-norm ratio by training-token class (selected; teacher-correct; descriptive)", fontsize=9)
        fig.tight_layout()
        fig.savefig(fd / "F4_edit_ratio.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "evaluate", "mech", "prefix", "report"])
    a = ap.parse_args(argv)
    {"select": run_select, "evaluate": run_evaluate, "mech": run_mech, "prefix": run_prefix, "report": run_report}[a.cmd]()


if __name__ == "__main__":
    main()
