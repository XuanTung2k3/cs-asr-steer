#!/usr/bin/env python
"""P2-1b evaluation and verdict (CPU). Rules: docs/reft/P2_1B_DECISION_RULES.md.

  cases     -- fixed prefix-diagnostic cases from P2-1D data only (written before training)
  select    -- per new run: unchanged P2-1 checkpoint rule -> selection.json (write-once)
  evaluate  -- every run/epoch/regime vs the pinned B0: canonical metrics, P2-1D failure metrics, English fates,
               paired KL-CE dialogue bootstrap with shared draws -> p2_1b/evaluation.json
  report    -- deterministic verdict, figures -> p2_1b/report.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, fillers, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import git_state, now, sha  # noqa: E402

SEEDS = (42, 43, 44)
EN_CATS = ("correct", "translation_like", "omission", "lexical_substitution", "segmentation_artifact", "boundary", "other")
ZH_DENOM = 4950


def cb():
    return yaml.safe_load((ROOT / "configs/reft/p2_1b_frozen.yaml").read_text())


def out(*p):
    return art("p2_1b", *p)


# ====================================================================== pure helpers (unit-tested)

def sb_units_from_events(ev, min_len=5):
    """Long switch-back dropped units: newly wrong Mandarin units deleted by the system inside a maximal run (>= min_len)
    of consecutive system-deleted reference units whose preceding unit is a substantive English unit."""
    from reft.p2_1d_audit import deletion_runs
    ev = sorted(ev, key=lambda e: e["k"])
    runs = deletion_runs([e["sys_op"] == "del" for e in ev])
    tot = long_runs = sb = sb_runs = 0
    for s, t in runs:
        dmg = sum(1 for e in ev[s:t + 1] if e["zh_non_filler"] and e["transition"] == "corruption")
        if not dmg:
            continue
        tot += 1
        if t - s + 1 >= min_len:
            long_runs += 1
            if s > 0 and ev[s - 1]["substantive_en"]:
                sb += dmg
                sb_runs += 1
    return {"damaged_runs": tot, "long_runs": long_runs, "sb_runs": sb_runs, "sb_units": sb}


def preservation_success(ce, kl, G):
    """ce, kl: {seed: {zh_new, sb, loops, nonterm}} at selected checkpoints."""
    P1 = sum(kl[s]["zh_new"] <= G["zh_new_max"] for s in SEEDS) >= 2
    P2 = np.mean([kl[s]["zh_new"] for s in SEEDS]) < np.mean([ce[s]["zh_new"] for s in SEEDS])
    red = [(ce[s]["sb"] - kl[s]["sb"]) / ce[s]["sb"] for s in SEEDS if ce[s]["sb"] > 0]
    P3 = bool(red) and float(np.mean(red)) >= 0.30 and sum(kl[s]["sb"] < ce[s]["sb"] for s in SEEDS) >= 2
    P4 = sum(kl[s]["loops"] for s in SEEDS) <= sum(ce[s]["loops"] for s in SEEDS) + 1 and \
        sum(kl[s]["nonterm"] for s in SEEDS) <= sum(ce[s]["nonterm"] for s in SEEDS) + 1
    zr = 1 - np.mean([kl[s]["zh_new"] for s in SEEDS]) / np.mean([ce[s]["zh_new"] for s in SEEDS]) if np.mean([ce[s]["zh_new"] for s in SEEDS]) else 0.0
    D = bool(P2 and (P3 or zr >= 0.30))
    return {"P1": bool(P1), "P2": bool(P2), "P3": bool(P3), "P4": bool(P4), "P": bool(P1 and P2 and P3 and P4), "D": D,
            "mean_sb_reduction": float(np.mean(red)) if red else None, "mean_zh_new_reduction": float(zr)}


def correction_retained(ce, kl):
    """ce, kl: {seed: {en_net, cs_mer, worst_dlg}}."""
    R1 = sum(kl[s]["en_net"] > 0 for s in SEEDS) >= 2
    mce = float(np.mean([ce[s]["en_net"] for s in SEEDS]))
    ratio = float(np.mean([kl[s]["en_net"] for s in SEEDS])) / mce if mce > 0 else None
    R2 = ratio is not None and ratio >= 0.75
    R3 = float(np.mean([kl[s]["cs_mer"] for s in SEEDS]) - np.mean([ce[s]["cs_mer"] for s in SEEDS])) <= 0.005
    R4 = all(kl[s]["worst_dlg"] <= ce[s]["worst_dlg"] + 10 for s in SEEDS)
    return {"R1": bool(R1), "R2": bool(R2), "R3": bool(R3), "R4": bool(R4), "R": bool(R1 and R2 and R3 and R4), "en_net_ratio": ratio}


def seed_unstable(ce, kl):
    for key in ("zh_new", "sb"):
        d = [kl[s][key] - ce[s][key] for s in SEEDS]
        if min(d) <= -10 and max(d) >= 10:
            return True
    return False


def decide(integrity_ok, P, R, U):
    if not integrity_ok:
        return "P2_1B_ENGINEERING_BLOCKED"
    if P["P"] and R["R"]:
        return "P2_1B_PRESERVATION_RESCUED"
    if U:
        return "P2_1B_SEED_UNSTABLE"
    if P["D"] and not R["R"]:
        return "P2_1B_KL_SUPPRESSES_CORRECTIONS"
    if not P["D"]:
        return "P2_1B_KL_NOT_EFFECTIVE"
    return "P2_1B_INCONCLUSIVE"


# ====================================================================== cases (CPU; before training)

def run_cases():
    from transformers import WhisperTokenizer
    from reft.common import cfg
    from reft.evaluate_supervised import dev_refs
    from reft.p2_1d_audit import deletion_runs
    from reft.text import target_text, tok_bytes_fn, token_unit_map, units
    p = out("prefix_audit/cases_private.json")
    if p.exists():
        raise SystemExit("cases already frozen")
    ev = read_jsonl(art("p2_1d/events_private.jsonl"))
    us = {u["utt"]: u for u in read_jsonl(art("p2_1d/utterance_summary_private.jsonl"))}
    dm = {r["utterance_id"]: r for r in read_json(art("p2_0/dev_manifest.json"))["rows"]}
    refs = dev_refs(sorted(dm))
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb = tok_bytes_fn(tok)
    eos = tok.eos_token_id
    by = collections.defaultdict(list)
    for e in ev:
        by[e["utt"]].append(e)
    for v in by.values():
        v.sort(key=lambda e: e["k"])

    def tokens_for(uid, unit_set, first_unit):
        es = by[uid]
        t_text = target_text(refs[uid])
        if units(t_text) != [e["unit"] for e in es]:
            return None
        ids = tok.encode(t_text, add_special_tokens=False) + [eos]
        um = token_unit_map(t_text, ids[:-1], tb)
        if um is None:
            return None
        sb_tok = next((t for t, x in enumerate(um) if first_unit in x), None)
        span = [t for t, x in enumerate(um) if set(x) & unit_set]
        return None if sb_tok is None else {"target_ids": ids, "sb_token": sb_tok, "span_tokens": span}

    damaged, unm = [], 0
    for uid, es in sorted(by.items()):
        for s, t in deletion_runs([e["sys_op"] == "del" for e in es]):
            dmg = [e["k"] for e in es[s:t + 1] if e["zh_non_filler"] and e["transition"] == "corruption"]
            if not dmg or t - s + 1 < 5 or s == 0 or not es[s - 1]["substantive_en"]:
                continue
            first = next(e["k"] for e in es[s:t + 1] if e["zh_non_filler"])
            tk = tokens_for(uid, set(dmg), first)
            if tk is None:
                unm += 1
                continue
            damaged.append({"kind": "damaged", "id": uid, "dialogue": es[0]["dialogue"], "sb_unit": first, "span_units": dmg,
                            "prompt": dm[uid]["prompt"], **tk})
    safe_pool = []
    for uid, u in sorted(us.items()):
        if u["stratum"] != "CS" or u["zh_new"] > 0:
            continue
        es = by[uid]
        lang = [e for e in es if (e["substantive_en"] or e["zh_non_filler"])]
        cands = [lang[q]["k"] for q in range(1, len(lang)) if lang[q]["zh_non_filler"] and lang[q - 1]["substantive_en"]]
        if not cands:
            continue
        k = min(cands, key=lambda k: hashlib.sha256(f"p2_1b-safe:{uid}:{k}".encode()).hexdigest())
        span = []
        for e in es[k:]:
            if not e["zh_non_filler"] or len(span) >= 5:
                break
            span.append(e["k"])
        safe_pool.append((hashlib.sha256(f"p2_1b-safe:{uid}:{k}".encode()).hexdigest(), uid, k, span))
    safe = []
    for _, uid, k, span in sorted(safe_pool):
        if len(safe) >= len(damaged):
            break
        tk = tokens_for(uid, set(span), k)
        if tk is None:
            continue
        safe.append({"kind": "safe", "id": uid, "dialogue": by[uid][0]["dialogue"], "sb_unit": k, "span_units": span,
                     "prompt": dm[uid]["prompt"], **tk})
    write_json(p, {"schema": "reft_p2_1b_cases_v1", "utc": now(), "source": "P2-1D events (R2-CE seed 42 epoch 2 vs B0) only",
                   "damaged": damaged, "safe": safe, "damaged_unmappable": unm})
    print({"damaged": len(damaged), "safe": len(safe), "unmappable": unm})


# ====================================================================== runs, selection

def run_dirs():
    pre = sorted(out().glob("preflight_*.json"), key=lambda p: p.stat().st_mtime)
    reuse = read_json(pre[-1])["ce42_reuse"]
    d = {}
    for s in SEEDS:
        d[("CE", s)] = art("p2_1/R2") if (s == 42 and reuse) else out("training", f"CE_seed{s}")
        d[("KL", s)] = out("training", f"KL_seed{s}")
    return d, reuse


def run_select():
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, score_system, select_epoch
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    for rd in sorted(out("training").iterdir()):
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

def fates(B, S, FILL):
    from reft.text import en_error_kind
    M = {a: {b: 0 for b in EN_CATS} for a in EN_CATS}
    for b, s in zip(B, S):
        for k, (u, t) in enumerate(zip(b["o"]["ref"], b["o"]["tags"])):
            if t != "EN" or u in FILL:
                continue
            cb_ = "correct" if b["o"]["by"][k][0] == "match" else en_error_kind(b["o"], k)
            cs_ = "correct" if s["o"]["by"][k][0] == "match" else en_error_kind(s["o"], k)
            M[cb_][cs_] += 1
    return {"matrix": M, "omission_to_correct": M["omission"]["correct"], "translation_to_correct": M["translation_like"]["correct"],
            "lexical_to_correct": M["lexical_substitution"]["correct"],
            "correct_to_wrong": sum(v for k, v in M["correct"].items() if k != "correct"),
            "wrong_to_other_wrong": sum(v for a, r in M.items() if a != "correct" for k, v in r.items() if k not in ("correct", a))}


def failure(B, S, FILL):
    from reft.p2_1d_audit import utterance_events
    per_utt, sb, dmg_runs, long_runs = [], 0, 0, 0
    near = [0, 0]
    far = [0, 0]
    for b, s in zip(B, S):
        ev, ins = utterance_events(b, s, FILL, None, "beam5", b["id"], b["dialogue"], b["stratum"])
        r = sb_units_from_events(ev)
        sb += r["sb_units"]
        dmg_runs += r["damaged_runs"]
        long_runs += r["long_runs"]
        zh = [e for e in ev if e["zh_non_filler"] and e["b0_ok"]]
        for e in zh:
            box = near if (e["d_sw"] is not None and e["d_sw"] <= 7) else far
            box[1] += 1
            box[0] += e["transition"] == "corruption"
        en_corr = sum(1 for e in ev if e["substantive_en"] and e["transition"] == "repair")
        en_cor = sum(1 for e in ev if e["substantive_en"] and e["transition"] == "corruption")
        zn = sum(1 for e in zh if e["transition"] == "corruption")
        per_utt.append({"dialogue": b["dialogue"], "stratum": b["stratum"], "zh_new": zn,
                        "zh_del_new": sum(1 for e in zh if e["transition"] == "corruption" and e["sys_op"] == "del"),
                        "sb": r["sb_units"], "en_corr": en_corr, "en_corrupt": en_cor,
                        "b_err": b["counts"][2], "s_err": s["counts"][2], "units": b["counts"][3]})
    d_mixed = collections.Counter()
    for u in per_utt:
        d_mixed[u["dialogue"]] += u["s_err"] - u["b_err"]
    tot_zh = sum(u["zh_new"] for u in per_utt)
    return {"zh_new": tot_zh, "zh_del_new": sum(u["zh_del_new"] for u in per_utt), "damaged_runs": dmg_runs, "long_runs": long_runs,
            "sb_units": sb, "near_switch_rate": near[0] / near[1] if near[1] else None, "far_rate": far[0] / far[1] if far[1] else None,
            "near_switch": near, "far": far,
            "zh_new_in_no_en_repair_utts": sum(u["zh_new"] for u in per_utt if u["en_corr"] == 0),
            "worst_dialogue_delta_mixed": max(d_mixed.values()), "worst_dialogue_share_of_zh_new":
                (max(collections.Counter({d: sum(u["zh_new"] for u in per_utt if u["dialogue"] == d) for d in d_mixed}).values()) / tot_zh) if tot_zh else None,
            "_per_utt": per_utt}


def json_keys(d):
    """Amendment r2: str keys at the top level (seed ints next to "seed_mean" cannot be sorted by write_json)."""
    return {str(k): v for k, v in d.items()}


def dialogue_table(per_utt):
    t = collections.defaultdict(lambda: np.zeros(8))
    for u in per_utt:
        cs = u["stratum"] == "CS"
        t[u["dialogue"]] += np.array([u["s_err"], u["units"], u["s_err"] if cs else 0, u["units"] if cs else 0,
                                      u["zh_new"], u["sb"], u["en_corr"] - u["en_corrupt"], u["b_err"]], float)
    return t


def run_evaluate():
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    C = p21()
    B_ = cb()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    LU = C["evaluation"]["loop_units"]
    BS = B_["bootstrap"]
    pb = C["frozen_baseline"]
    assert sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"]
    base = {r: score_system(meta, refs, {x["id"]: x for x in read_jsonl(art(pb[r]))}, FILL, LU) for r in ("beam5", "greedy")}
    dirs, reuse = run_dirs()
    runs, dtabs = {}, {}
    for (arm, s), rd in dirs.items():
        sel = read_json(rd / "selection.json")["selected"]["epoch"]
        man = read_json(rd / "manifest.json")
        R = {"dir": str(rd), "selected_epoch": sel, "history": man["history"], "efficiency": man.get("efficiency"),
             "weights_unchanged": man["weights_unchanged"], "zero_init_identity": man["history"][0]["zero_init_identity_beam5"], "epochs": {}}
        for ep in (0, 1, 2, 3):
            R["epochs"][ep] = {}
            for reg in ("beam5", "greedy"):
                S = score_system(meta, refs, {x["id"]: x for x in read_jsonl(rd / f"epoch{ep}" / f"dev_{reg}.jsonl")}, FILL, LU)
                m = metrics(S)
                rec = {"metrics": {k: m[k] for k in ("overall", "CS", "EN_ONLY", "ZH_ONLY")},
                       "vs_B0": paired(base[reg], S, BS["reps"], BS["seed"])}
                if reg == "beam5":
                    f = failure(base["beam5"], S, FILL)
                    rec["failure"] = {k: v for k, v in f.items() if k != "_per_utt"}
                    rec["english_fates"] = fates(base["beam5"], S, FILL)
                    if ep == sel:
                        dtabs[(arm, s)] = dialogue_table(f["_per_utt"])
                R["epochs"][ep][reg] = rec
        runs[f"{arm}_seed{s}"] = R
    # paired KL - CE with shared dialogue draws (per seed and seed-mean)
    dl = sorted({m["dialogue_id"] for m in meta})
    rng = np.random.default_rng(BS["seed"])
    draws = [rng.choice(len(dl), len(dl)) for _ in range(BS["reps"])]

    def stat(tab, idx):
        a = sum(tab[dl[i]] for i in idx)
        return {"mer": a[0] / a[1], "cs_mer": a[2] / a[3], "zh_new": a[4], "sb": a[5], "en_net": a[6]}
    keys = ("mer", "cs_mer", "zh_new", "sb", "en_net")
    full = list(range(len(dl)))
    pair = {}
    for s in SEEDS:
        pt = {k: stat(dtabs[("KL", s)], full)[k] - stat(dtabs[("CE", s)], full)[k] for k in keys}
        bs = {k: [] for k in keys}
        for idx in draws:
            a, c_ = stat(dtabs[("KL", s)], idx), stat(dtabs[("CE", s)], idx)
            for k in keys:
                bs[k].append(a[k] - c_[k])
        pair[s] = {k: {"point": pt[k], "ci95": [float(np.quantile(bs[k], q)) for q in (0.025, 0.975)]} for k in keys}
    pooled = {k: [] for k in keys}
    for idx in draws:
        for k in keys:
            pooled[k].append(np.mean([stat(dtabs[("KL", s)], idx)[k] - stat(dtabs[("CE", s)], idx)[k] for s in SEEDS]))
    pair["seed_mean"] = {k: {"point": float(np.mean([pair[s][k]["point"] for s in SEEDS])),
                             "range": [float(min(pair[s][k]["point"] for s in SEEDS)), float(max(pair[s][k]["point"] for s in SEEDS))],
                             "ci95": [float(np.quantile(pooled[k], q)) for q in (0.025, 0.975)]} for k in keys}
    # epoch-matched (secondary)
    em = {s: {ep: {k: runs[f"KL_seed{s}"]["epochs"][ep]["beam5"]["failure"][k] - runs[f"CE_seed{s}"]["epochs"][ep]["beam5"]["failure"][k]
                   for k in ("zh_new", "sb_units")} | {"cs_mer": runs[f"KL_seed{s}"]["epochs"][ep]["beam5"]["metrics"]["CS"]["mer"]
                                                     - runs[f"CE_seed{s}"]["epochs"][ep]["beam5"]["metrics"]["CS"]["mer"]}
              for ep in (1, 2, 3)} for s in SEEDS}
    write_json(out("evaluation.json"), {"schema": "reft_p2_1b_evaluation_v1", "utc": now(), "git": git_state(), "ce42_reused_from_p2_1": reuse,
                                        "B0": {r: metrics(base[r]) for r in base}, "runs": runs, "paired_kl_minus_ce": json_keys(pair),
                                        "epoch_matched_kl_minus_ce": em, "note": "dev150 exposed; descriptive uncertainty"})
    print("evaluation written")


# ====================================================================== report

def run_report():
    from reft.report_p2_1 import preservation_ok, useful
    E = read_json(out("evaluation.json"))
    G = {"zh_new_max": int(ZH_DENOM * 0.01)}
    sel = {}
    for name, R in E["runs"].items():
        e = R["epochs"][str(R["selected_epoch"])]["beam5"]
        sel[name] = {"zh_new": e["failure"]["zh_new"], "sb": e["failure"]["sb_units"], "loops": e["metrics"]["overall"]["loop_utterances"],
                     "nonterm": e["metrics"]["overall"]["not_terminated"], "en_net": e["vs_B0"]["net"], "cs_mer": e["metrics"]["CS"]["mer"],
                     "mer": e["metrics"]["overall"]["mer"], "retention": e["vs_B0"]["mandarin_retention"],
                     "worst_dlg": e["failure"]["worst_dialogue_delta_mixed"], "epoch": R["selected_epoch"]}
    ce = {s: sel[f"CE_seed{s}"] for s in SEEDS}
    kl = {s: sel[f"KL_seed{s}"] for s in SEEDS}
    P = preservation_success(ce, kl, G)
    R = correction_retained(ce, kl)
    U = seed_unstable(ce, kl)
    ver = read_json(out("verify.json")) if out("verify.json").exists() else None
    integ = {"verify_ok": bool(ver and ver["all_ok"]), "all_runs_present": len(E["runs"]) == 6,
             "zero_init_150": all(R_["zero_init_identity"] == 150 for R_ in E["runs"].values()),
             "weights_unchanged": all(R_["weights_unchanged"] for R_ in E["runs"].values()),
             "prefix_audit_present": out("prefix_audit/summary.json").exists()}
    verdict = decide(all(integ.values()), P, R, U)
    vs_b0 = {}
    for name, R_ in E["runs"].items():
        e = R_["epochs"][str(R_["selected_epoch"])]["beam5"]
        v = e["vs_B0"]
        a = {"d_mer": v["delta_mer"]["point"], "d_cs_mer": v["delta_cs_mer"]["point"], "d_cs_mer_ci": v["delta_cs_mer"]["ci95"], "net": v["net"],
             "corrections": v["substantive_en_corrections"], "dialogues_with_corrections": v["dialogues_with_corrections"],
             "max_share": v["max_utterance_share_of_net"], "mandarin_retention": v["mandarin_retention"],
             "zh_cer": e["metrics"]["overall"]["zh_cer"], "zh_cer_b0": E["B0"]["beam5"]["overall"]["zh_cer"],
             "loops": e["metrics"]["overall"]["loop_utterances"], "loops_b0": E["B0"]["beam5"]["overall"]["loop_utterances"]}
        vs_b0[name] = {"p2_1_useful": useful(a), "p2_1_preservation_ok": preservation_ok(a)}
    rep = {"schema": "reft_p2_1b_report_v1", "utc": now(), "git": git_state(), "verdict": verdict, "integrity": integ,
           "selected": sel, "P": P, "R": R, "U": U, "paired_kl_minus_ce": E["paired_kl_minus_ce"], "p2_1_rules_vs_B0": vs_b0,
           "prefix_summary": read_json(out("prefix_audit/summary.json")) if out("prefix_audit/summary.json").exists() else None,
           "rules_doc_sha256": sha(ROOT / "docs/reft/P2_1B_DECISION_RULES.md")}
    write_json(out("report.json"), rep)
    figures(E, rep)
    print(json.dumps({"verdict": verdict, "P": P, "R": R, "U": U, "integrity": integ, "selected": sel}, indent=1))


def figures(E, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = out("figures")
    d.mkdir(parents=True, exist_ok=True)
    col = {"CE": "#4c72b0", "KL": "#c44e52"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, R in E["runs"].items():
        arm = name.split("_")[0]
        for ep in ("1", "2", "3"):
            e = R["epochs"][ep]["beam5"]
            ax.scatter(e["failure"]["zh_new"], e["vs_B0"]["net"], color=col[arm], marker="o" if int(ep) == R["selected_epoch"] else ".",
                       s=60 if int(ep) == R["selected_epoch"] else 25)
            ax.annotate(f"{name.replace('_seed', '')}e{ep}", (e["failure"]["zh_new"], e["vs_B0"]["net"]), fontsize=6)
    ax.axvline(49.5, ls="--", color="grey")
    ax.text(50, ax.get_ylim()[0], " retention 0.99 (≤49 new)", fontsize=7, color="grey")
    ax.set(xlabel="new Mandarin errors vs B0 (of 4,950 B0-correct units)", ylabel="net substantive English corrections vs B0",
           title="F1 correction–preservation frontier (blue CE, red KL; large = selected)")
    fig.tight_layout()
    fig.savefig(d / "F1_frontier.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(SEEDS))
    for i, (key, lab) in enumerate((("sb", "long switch-back dropped units"), ("zh_new", "new Mandarin errors"))):
        ax[i].bar(x - 0.2, [rep["selected"][f"CE_seed{s}"][key] for s in SEEDS], 0.4, label="CE", color=col["CE"])
        ax[i].bar(x + 0.2, [rep["selected"][f"KL_seed{s}"][key] for s in SEEDS], 0.4, label="KL", color=col["KL"])
        ax[i].set_xticks(x, [f"seed {s}" for s in SEEDS])
        ax[i].set(title=f"F2 {lab} (selected checkpoints)", ylabel="reference units")
        ax[i].legend()
    fig.tight_layout()
    fig.savefig(d / "F2_switchback.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (key, lab) in enumerate((("cs_mer", "genuine-CS MER"), ("retention", "Mandarin retention"))):
        b0 = E["B0"]["beam5"]["CS"]["mer"] if key == "cs_mer" else 1.0
        ax[i].axhline(b0, color="black", ls=":", label="B0")
        ax[i].bar(x - 0.2, [rep["selected"][f"CE_seed{s}"][key] for s in SEEDS], 0.4, label="CE", color=col["CE"])
        ax[i].bar(x + 0.2, [rep["selected"][f"KL_seed{s}"][key] for s in SEEDS], 0.4, label="KL", color=col["KL"])
        if key == "retention":
            ax[i].axhline(0.99, color="grey", ls="--", label="0.99")
            ax[i].set_ylim(0.97, 1.001)
        ax[i].set_xticks(x, [f"seed {s}" for s in SEEDS])
        ax[i].set(title=f"F3 {lab} (selected, beam-5)")
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "F3_mer_retention.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, R in E["runs"].items():
        h = [x for x in R["history"] if x["epoch"] >= 1]
        arm = name.split("_")[0]
        ax[0].plot([x["epoch"] for x in h], [x["train_ce"] for x in h], marker="o", color=col[arm], alpha=0.7, label=name)
        if arm == "KL":
            ax[0].plot([x["epoch"] for x in h], [x["train_kl"] for x in h], ls="--", marker="x", color=col[arm], alpha=0.7)
        ax[1].plot([x["epoch"] for x in h], [x["grad_norm_mean"] for x in h], marker="o", color=col[arm], alpha=0.7)
    ax[0].set(title="F4 training CE (solid) and KL (dashed, KL arm)", xlabel="epoch", ylabel="nats / token")
    ax[1].set(title="F4 mean clipped-before gradient norm", xlabel="epoch")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(d / "F4_training.png", dpi=130)
    plt.close(fig)
    ps = rep.get("prefix_summary")
    if ps:
        sysn = list(ps["by_system"])
        fig, ax = plt.subplots(figsize=(9, 4))
        xx = np.arange(len(sysn))
        for off, kind, c_ in ((-0.2, "damaged", "#c44e52"), (0.2, "safe", "#55a868")):
            v = [ps["by_system"][s_][kind]["mean_delta_gold_logp_vs_b0"] or 0 for s_ in sysn]
            ax.bar(xx + off, v, 0.4, label=f"{kind} (n={ps['n_cases'][kind]})", color=c_)
        ax.set_xticks(xx, sysn, rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set(title="F5 gold-prefix Δ log p(first Mandarin token after English) vs B0", ylabel="nats")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(d / "F5_prefix.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cases", "select", "evaluate", "report"])
    a = ap.parse_args(argv)
    {"cases": run_cases, "select": run_select, "evaluate": run_evaluate, "report": run_report}[a.cmd]()


if __name__ == "__main__":
    main()
