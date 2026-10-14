#!/usr/bin/env python
"""P2-1D -- correction-damage diagnostic audit of the completed P2-1 dev150 decodes (CPU only, observational).

Protocol: docs/reft/P2_1D_PROTOCOL.md. Reuses the canonical scorer (reft.evaluate_supervised.score_system / paired /
metrics; reft.text.one / alignment / en_error_kind) unchanged. The alternative alignment below is a labelled SENSITIVITY
check only. No GPU, no model, no decoding, no training. Private outputs under <artifact_root>/p2_1d/.
"""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, fillers, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import git_state, now, sha  # noqa: E402

BINS = ((0, 1, "0-1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10 ** 9, ">=8"))
EN_CATS = ("correct", "translation_like", "omission", "lexical_substitution", "segmentation_artifact", "boundary", "other")
OUT = "p2_1d"


# ====================================================================== pure helpers (unit-tested)

def dist_bin(d):
    if d is None:
        return None
    for lo, hi, lab in BINS:
        if lo <= d <= hi:
            return lab
    raise ValueError(d)


def en_distances(tags, units, FILL):
    """Reference-unit distance of every unit to the nearest substantive English unit (None if the utterance has none)."""
    en = [j for j, (t, u) in enumerate(zip(tags, units)) if t == "EN" and u not in FILL]
    return [min(abs(k - j) for j in en) if en else None for k in range(len(tags))]


def switch_distances(tags, units, FILL):
    """Distance to the nearest EN<->ZH switch in the sequence of non-filler EN/ZH units (0 = adjacent to a switch)."""
    lang = [(j, t) for j, (t, u) in enumerate(zip(tags, units)) if t in ("EN", "ZH") and u not in FILL]
    bounds = [(lang[i][0], lang[i + 1][0]) for i in range(len(lang) - 1) if lang[i][1] != lang[i + 1][1]]
    out = []
    for k in range(len(tags)):
        if not bounds:
            out.append(None)
            continue
        out.append(min(0 if k in (a, b) else (a - k if k < a else k - b if k > b else 1) for a, b in bounds))
    return out


def transition(b_ok: bool, s_ok: bool) -> str:
    return {(False, True): "repair", (True, False): "corruption", (True, True): "unchanged_correct",
            (False, False): "unchanged_wrong"}[(b_ok, s_ok)]


def alt_alignment(a, b):
    """SENSITIVITY ONLY: minimum-edit alignment with the opposite tie-break (insertion, deletion, then diagonal)."""
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(a) + 1)
    d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j - 1] + (a[i - 1] != b[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops, i, j = [], len(a), len(b)
    while i or j:
        if j and d[i, j] == d[i, j - 1] + 1:
            ops.append(("ins", None, j - 1))
            j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i -= 1
            j -= 1
    return ops[::-1]


def edit_cost(ops) -> int:
    return sum(o != "match" for o, _, _ in ops)


def ok_from_ops(ops, n):
    by = {i: o for o, i, j in ops if i is not None}
    return [by[k] == "match" for k in range(n)]


def topk_share(values, k):
    v = sorted(values, reverse=True)
    t = sum(v)
    return sum(v[:k]) / t if t else None


def wilson(x, n, z=1.96):
    if n == 0:
        return None
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def script_of_unit(s):
    from reft.text import is_han
    return "han" if any(is_han(c) for c in s) else "latin" if any(c.isascii() and c.isalpha() for c in s) else "other"


def contingency(utts, key_a="en_repair", key_b="zh_damage"):
    cells = {}
    for a in (True, False):
        for b in (True, False):
            rr = [u for u in utts if u[key_a] == a and u[key_b] == b]
            cells[f"{'EN_REPAIR' if a else 'no_EN_REPAIR'}&{'ZH_DAMAGE' if b else 'no_ZH_DAMAGE'}"] = {
                "utterances": len(rr), "dialogues": len({u["dialogue"] for u in rr}),
                "en_corrections": sum(u["en_corr"] for u in rr), "en_corruptions": sum(u["en_corrupt"] for u in rr),
                "zh_new": sum(u["zh_new"] for u in rr), "zh_repairs": sum(u["zh_rep"] for u in rr),
                "net_mixed_error_change": sum(u["d_mixed"] for u in rr)}
    return cells


# ====================================================================== data-dependent audit

def boot(groups, fn, reps=2000, seed=42):
    from reft.conditional_audit import cluster_bootstrap
    return cluster_bootstrap(groups, fn, reps, seed)["ci95"]


def utterance_events(b, s, FILL, epoch, regime, uid, dlg, stratum):
    """One event per canonical reference unit (B0 vs system); insertions summarised separately."""
    from reft.text import en_error_kind
    ob, os_ = b["o"], s["o"]
    assert ob["ref"] == os_["ref"]
    n = len(ob["ref"])
    dEN = en_distances(ob["tags"], ob["ref"], FILL)
    dSW = switch_distances(ob["tags"], ob["ref"], FILL)
    ev = []
    for k in range(n):
        u, tag = ob["ref"][k], ob["tags"][k]
        bo, so = ob["by"][k], os_["by"][k]
        bok, sok = bo[0] == "match", so[0] == "match"
        sub_en = tag == "EN" and u not in FILL
        zh = tag == "ZH" and u not in FILL
        ev.append({"utt": uid, "dialogue": dlg, "stratum": stratum, "k": k, "n_units": n, "unit": u, "tag": tag,
                   "substantive_en": sub_en, "zh_non_filler": zh, "b0_op": bo[0], "sys_op": so[0], "b0_ok": bok, "sys_ok": sok,
                   "transition": transition(bok, sok),
                   "b0_hyp": ob["hyp"][bo[1]] if bo[1] is not None else None, "sys_hyp": os_["hyp"][so[1]] if so[1] is not None else None,
                   "en_cat_b0": ("correct" if bok else en_error_kind(ob, k)) if sub_en else None,
                   "en_cat_sys": ("correct" if sok else en_error_kind(os_, k)) if sub_en else None,
                   "d_en": dEN[k], "d_sw": dSW[k], "rel_pos": k / max(1, n - 1), "epoch": epoch, "regime": regime})
    ins = {sysn: collections.Counter(script_of_unit(o["hyp"][j]) for op, i, j in o["ops"] if op == "ins")
           for sysn, o in (("b0", ob), ("sys", os_))}
    return ev, ins


def run():
    t0 = time.time()
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    from reft.text import alignment, tokenize
    C = p21()
    BS = C["evaluation"]["bootstrap"]
    FILL = fillers()
    LU = C["evaluation"]["loop_units"]
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    pb = C["frozen_baseline"]
    files = {"B0_beam5": art(pb["beam5"]), "B0_greedy": art(pb["greedy"]),
             **{f"R2_e{e}_beam5": art(f"p2_1/R2/epoch{e}/dev_beam5.jsonl") for e in range(4)},
             **{f"R2_e{e}_greedy": art(f"p2_1/R2/epoch{e}/dev_greedy.jsonl") for e in range(4)},
             "L1_e3_beam5": art("p2_1/L1/epoch3/dev_beam5.jsonl"), "L1_e3_greedy": art("p2_1/L1/epoch3/dev_greedy.jsonl")}
    meta_files = {k: art(f"p2_1/{k}") for k in ("report.json", "evaluation.json", "R2/selection.json", "L1/selection.json",
                                                 "R2/manifest.json", "L1/manifest.json", "verify.json")}
    meta_files["dev_manifest"] = art("p2_0/dev_manifest.json")
    hashes = {k: sha(p) for k, p in {**files, **meta_files}.items()}
    rep = read_json(art("p2_1/report.json"))
    sel = {m: read_json(art(f"p2_1/{m}/selection.json"))["selected"]["epoch"] for m in ("R2", "L1")}
    scored = {k: score_system(meta, refs, {x["id"]: x for x in read_jsonl(p)}, FILL, LU) for k, p in files.items()}
    B0, R2 = scored["B0_beam5"], scored[f"R2_e{sel['R2']}_beam5"]

    # ------------------------------------------------------------------ G1 integrity / exact reproduction
    pr = paired(B0, R2, BS["reps"], BS["seed"])
    mb, mr = metrics(B0), metrics(R2)
    want = rep["arms"]["R2"]["epochs"][str(sel["R2"])]["beam5"]
    ev_r = read_json(art("p2_1/evaluation.json"))["results"]["beam5"]["R2"]["vs_baseline"]
    integ = {"pinned_beam5_hash": hashes["B0_beam5"] == pb["beam5_sha256"], "pinned_greedy_hash": hashes["B0_greedy"] == pb["greedy_sha256"],
             "selected_epochs": sel == {"R2": 2, "L1": 3},
             "paired_equals_report": pr == want["vs_B0"], "paired_equals_evaluation": pr == ev_r,
             "metrics_equal_report": mr == want["metrics"] and mb == rep["B0"]["beam5"],
             "n_utterances_150": len(B0) == 150, "n_dialogues_20": len({u["dialogue"] for u in B0}) == 20,
             "n_cs_97": sum(u["stratum"] == "CS" for u in B0) == 97,
             "counts": {"en_corrections": pr["substantive_en_corrections"], "en_corruptions": pr["substantive_en_corruptions"],
                        "zh_new": pr["new_zh_errors"], "zh_repairs": pr["zh_repairs"],
                        "b0_correct_zh": sum(sum(u["zh_ok"]) for u in B0), "outside": [pr["outside_target_damage"], pr["outside_target_denominator"]],
                        "mer": [mb["overall"]["mer"], mr["overall"]["mer"]], "cs_mer": [mb["CS"]["mer"], mr["CS"]["mer"]],
                        "zh_cer": [mb["overall"]["zh_cer"], mr["overall"]["zh_cer"]]}}
    exp = {"en_corrections": 62, "en_corruptions": 25, "zh_new": 73, "zh_repairs": 103, "b0_correct_zh": 4950, "outside": [76, 4971]}
    integ["expected_counts"] = all(integ["counts"][k] == v for k, v in exp.items())

    # ------------------------------------------------------------------ G2 event table (selected epoch, beam-5)
    events, ins_rows, utts = [], [], []
    for b, s, m in zip(B0, R2, meta):
        assert b["id"] == s["id"] == m["utterance_id"]
        ev, ins = utterance_events(b, s, FILL, sel["R2"], "beam5", b["id"], b["dialogue"], b["stratum"])
        events += ev
        ins_rows.append({"utt": b["id"], "b0": dict(ins["b0"]), "sys": dict(ins["sys"])})
        en = [e for e in ev if e["substantive_en"]]
        zh = [e for e in ev if e["zh_non_filler"]]
        en_changed_ww = sum(1 for e in en if e["transition"] == "unchanged_wrong" and (e["b0_hyp"] != e["sys_hyp"] or e["en_cat_b0"] != e["en_cat_sys"]))
        utts.append({"utt": b["id"], "dialogue": b["dialogue"], "stratum": b["stratum"], "n_units": len(ev),
                     "en_corr": sum(e["transition"] == "repair" for e in en), "en_corrupt": sum(e["transition"] == "corruption" for e in en),
                     "en_wrong_to_wrong_changed": en_changed_ww,
                     "zh_new": sum(e["transition"] == "corruption" for e in zh), "zh_rep": sum(e["transition"] == "repair" for e in zh),
                     "ins_b0": sum(ins["b0"].values()), "ins_sys": sum(ins["sys"].values()),
                     "d_ref_wrong": sum(not e["sys_ok"] for e in ev) - sum(not e["b0_ok"] for e in ev),
                     "d_mixed": s["counts"][2] - b["counts"][2], "loop_sys": bool(s["loop"]), "terminated_sys": bool(s["terminated"])})
    for u in utts:
        u["en_repair"], u["zh_damage"] = u["en_corr"] > 0, u["zh_new"] > 0
        u["d_ins"] = u["ins_sys"] - u["ins_b0"]
    ev_counts = {"en_corrections": sum(u["en_corr"] for u in utts), "en_corruptions": sum(u["en_corrupt"] for u in utts),
                 "zh_new": sum(u["zh_new"] for u in utts), "zh_repairs": sum(u["zh_rep"] for u in utts),
                 "b0_correct_zh": sum(1 for e in events if e["zh_non_filler"] and e["b0_ok"]),
                 "outside": [sum(1 for e in events if e["tag"] != "EN" and e["b0_ok"] and not e["sys_ok"]),
                             sum(1 for e in events if e["tag"] != "EN" and e["b0_ok"])]}
    integ["event_table_reconciles"] = ev_counts == {k: integ["counts"][k] for k in ev_counts}
    integ["event_coverage_exact"] = all(sum(1 for e in events if e["utt"] == u["utt"]) == u["n_units"] == len(next(b for b in B0 if b["id"] == u["utt"])["o"]["ref"]) for u in utts) \
        and len({(e["utt"], e["k"]) for e in events}) == len(events)
    integ["mixed_change_decomposes"] = all(u["d_mixed"] == u["d_ref_wrong"] + u["d_ins"] for u in utts)
    integ["ok"] = all(v for k, v in integ.items() if isinstance(v, bool))

    A = {}
    # ------------------------------------------------------------------ D1 Mandarin damage localization
    zh_u = [u["zh_new"] for u in utts if u["zh_new"] > 0]
    per_d = collections.Counter()
    for u in utts:
        per_d[u["dialogue"]] += u["zh_new"]
    zh_ev = [e for e in events if e["zh_non_filler"] and e["b0_ok"]]
    by_strat = {}
    for st in ("CS", "ZH_ONLY", "EN_ONLY", "OTHER"):
        rr = [e for e in zh_ev if e["stratum"] == st]
        x = sum(e["transition"] == "corruption" for e in rr)
        by_strat[st] = {"b0_correct": len(rr), "new_errors": x, "rate": x / len(rr) if rr else None, "wilson95": wilson(x, len(rr))}

    def binned(key):
        out = {}
        for lab in [b[2] for b in BINS] + [None]:
            rr = [e for e in zh_ev if dist_bin(e[key]) == lab]
            x = sum(e["transition"] == "corruption" for e in rr)
            g = [e["dialogue"] for e in rr]
            flag = np.array([e["transition"] == "corruption" for e in rr])
            out[lab or ("no_en" if key == "d_en" else "no_switch")] = {
                "b0_correct": len(rr), "new_errors": x, "rate": x / len(rr) if rr else None, "wilson95": wilson(x, len(rr)),
                "dialogue_boot95": boot(g, lambda ii: float(flag[ii].mean())) if rr else None, "dialogues": len(set(g))}
        return out
    dmg_ops = collections.Counter()
    for e in zh_ev:
        if e["transition"] == "corruption":
            dmg_ops["deletion" if e["sys_op"] == "del" else f"sub_by_{script_of_unit(e['sys_hyp'])}"] += 1
    rep_ops = collections.Counter(e["b0_op"] for e in events if e["zh_non_filler"] and e["transition"] == "repair")
    dd = collections.defaultdict(lambda: {"d_mixed": 0, "d_ref_wrong": 0, "d_ins": 0, "zh_new": 0, "en_corr": 0, "en_corrupt": 0})
    for u in utts:
        for k_ in ("d_mixed", "d_ref_wrong", "d_ins", "zh_new", "en_corr", "en_corrupt"):
            dd[u["dialogue"]][k_] += u[k_]
    dl_sorted = sorted(dd.items(), key=lambda kv: kv[1]["d_mixed"])
    ins_tot = {s_: collections.Counter() for s_ in ("b0", "sys")}
    for r in ins_rows:
        for s_ in ("b0", "sys"):
            ins_tot[s_].update(r[s_])
    A["D1"] = {"total_new_zh": sum(zh_u), "utterances_affected": len(zh_u), "dialogues_affected": sum(1 for v in per_d.values() if v),
               "utterance_top1_top3_top5_share": [topk_share(zh_u, k) for k in (1, 3, 5)],
               "dialogue_top1_top3_share": [topk_share(list(per_d.values()), k) for k in (1, 3)],
               "per_affected_utterance": {"median": float(np.median(zh_u)), "q90": float(np.quantile(zh_u, 0.9)), "max": int(max(zh_u)),
                                          "sorted": sorted(zh_u, reverse=True)},
               "per_dialogue_sorted": sorted(per_d.values(), reverse=True),
               "by_stratum": by_strat, "by_dist_en": binned("d_en"), "by_dist_switch": binned("d_sw"),
               "damage_operation": dict(dmg_ops), "repair_from_b0_operation": dict(rep_ops),
               "insertions_by_script": {k: dict(v) for k, v in ins_tot.items()},
               "dialogue_decomposition_sorted_anonymised": [{"rank": i + 1, **v} for i, (_, v) in enumerate(dl_sorted)],
               "damage_utterances_with_insertion_increase": sum(1 for u in utts if u["zh_new"] and u["d_ins"] > 0)}

    # ------------------------------------------------------------------ D2 co-occurrence
    cs = [u for u in utts if u["stratum"] == "CS"]
    tot_dmg = sum(u["zh_new"] for u in utts)
    dmg_only = [u for u in utts if u["zh_damage"] and not u["en_repair"]]
    g_all = [u["dialogue"] for u in utts]
    zn = np.array([u["zh_new"] for u in utts], float)
    don = np.array([u["zh_damage"] and not u["en_repair"] for u in utts])
    A["D2"] = {"all": contingency(utts), "cs_only": contingency(cs),
               "damage_only_share_of_total_damage": sum(u["zh_new"] for u in dmg_only) / tot_dmg if tot_dmg else None,
               "damage_only_share_boot95": boot(g_all, lambda ii: float((zn[ii] * don[ii]).sum() / zn[ii].sum()) if zn[ii].sum() else None),
               "damage_given_en_repair": sorted([u["zh_new"] for u in utts if u["en_repair"]], reverse=True),
               "damage_given_no_en_repair": sorted([u["zh_new"] for u in utts if not u["en_repair"] and u["zh_new"]], reverse=True),
               "no_repair_but_other_en_change": {"utterances": sum(1 for u in utts if not u["en_repair"] and (u["en_corrupt"] or u["en_wrong_to_wrong_changed"])),
                                                 "en_corruptions": sum(u["en_corrupt"] for u in utts if not u["en_repair"]),
                                                 "en_wrong_to_wrong_changed": sum(u["en_wrong_to_wrong_changed"] for u in utts if not u["en_repair"])},
               "en_net_vs_zh_net": {"pearson_utterance_descriptive": float(np.corrcoef([u["en_corr"] - u["en_corrupt"] for u in utts],
                                                                                        [u["zh_rep"] - u["zh_new"] for u in utts])[0, 1]),
                                    "quadrants": dict(collections.Counter(
                                        f"en{'+' if u['en_corr'] - u['en_corrupt'] > 0 else '-' if u['en_corr'] - u['en_corrupt'] < 0 else '0'}"
                                        f"/zh{'+' if u['zh_rep'] - u['zh_new'] > 0 else '-' if u['zh_rep'] - u['zh_new'] < 0 else '0'}" for u in utts))}}

    # ------------------------------------------------------------------ D3 epoch dynamics
    man = read_json(art("p2_1/R2/manifest.json"))
    epochs = {}
    for e in range(4):
        S = scored[f"R2_e{e}_beam5"]
        m_ = metrics(S)
        p_ = paired(B0, S, BS["reps"], BS["seed"])
        epochs[e] = {"mer": m_["overall"]["mer"], "cs_mer": m_["CS"]["mer"], "en_wer": m_["overall"]["en_wer"], "pier": m_["overall"]["pier"],
                     "zh_cer": m_["overall"]["zh_cer"], "en_corr": p_["substantive_en_corrections"], "en_corrupt": p_["substantive_en_corruptions"],
                     "zh_new": p_["new_zh_errors"], "zh_rep": p_["zh_repairs"], "retention": p_["mandarin_retention"],
                     "improved_utts": p_["improved_utterances"], "worsened_utts": p_["worsened_utterances"],
                     "dialogues_improved": sum(1 for d in {u["dialogue"] for u in S} if sum(x["counts"][2] for x in S if x["dialogue"] == d) < sum(x["counts"][2] for x in B0 if x["dialogue"] == d)),
                     "dialogues_worsened": sum(1 for d in {u["dialogue"] for u in S} if sum(x["counts"][2] for x in S if x["dialogue"] == d) > sum(x["counts"][2] for x in B0 if x["dialogue"] == d)),
                     "loops": m_["overall"]["loop_utterances"], "not_terminated": m_["overall"]["not_terminated"],
                     "train_ce": next((h.get("train_ce") for h in man["history"] if h["epoch"] == e), None),
                     "grad_norm_mean": next((h.get("grad_norm_mean") for h in man["history"] if h["epoch"] == e), None)}
    trans = {}
    for a_, b_ in (("B0", 1), ("B0", 2), ("B0", 3), (1, 2), (2, 3)):
        Sa = B0 if a_ == "B0" else scored[f"R2_e{a_}_beam5"]
        p_ = paired(Sa, scored[f"R2_e{b_}_beam5"], BS["reps"], BS["seed"])
        trans[f"{a_}->{b_}"] = {k: p_[k] for k in ("substantive_en_corrections", "substantive_en_corruptions", "new_zh_errors", "zh_repairs",
                                                    "improved_utterances", "worsened_utterances")} | {"delta_mer": p_["delta_mer"]["point"]}
    S3 = scored["R2_e3_beam5"]
    loop_idx = [i for i, u in enumerate(S3) if u["loop"] or not u["terminated"]]
    tot_ref = sum(u["counts"][3] for u in B0)
    keep = [i for i in range(len(B0)) if i not in loop_idx]
    sens = {"loop_utterances": len(loop_idx),
            "loop_delta_mixed_errors": [S3[i]["counts"][2] - B0[i]["counts"][2] for i in loop_idx],
            "loop_share_of_epoch3_mer_regression": (sum(S3[i]["counts"][2] - B0[i]["counts"][2] for i in loop_idx) / tot_ref)
            / (epochs[3]["mer"] - epochs[0]["mer"]) if epochs[3]["mer"] != epochs[0]["mer"] else None,
            "DESCRIPTIVE_ONLY_mer_excluding_loop": {"B0": sum(B0[i]["counts"][2] for i in keep) / sum(B0[i]["counts"][3] for i in keep),
                                                     "R2_e3": sum(S3[i]["counts"][2] for i in keep) / sum(S3[i]["counts"][3] for i in keep),
                                                     "R2_e2": sum(R2[i]["counts"][2] for i in keep) / sum(R2[i]["counts"][3] for i in keep)},
            "loop_utterance_stratum": [S3[i]["stratum"] for i in loop_idx],
            "loop_utterance_in_e2_new_zh": [sum(1 for x, y in zip(B0[i]["zh_ok"], R2[i]["zh_ok"]) if x and not y) for i in loop_idx]}
    A["D3"] = {"epochs": epochs, "transitions": trans, "epoch3_loop_sensitivity": sens,
               "greedy": {e: {"mer": metrics(scored[f"R2_e{e}_greedy"])["overall"]["mer"], "cs_mer": metrics(scored[f"R2_e{e}_greedy"])["CS"]["mer"]} for e in range(4)}}

    # ------------------------------------------------------------------ D4 English transitions
    from reft.text import en_error_kind

    def en_matrix(Sb, Ss):
        M = {a: {b: 0 for b in EN_CATS} for a in EN_CATS}
        for b, s in zip(Sb, Ss):
            for k, (u, t) in enumerate(zip(b["o"]["ref"], b["o"]["tags"])):
                if t != "EN" or u in FILL:
                    continue
                cb = "correct" if b["o"]["by"][k][0] == "match" else en_error_kind(b["o"], k)
                cs_ = "correct" if s["o"]["by"][k][0] == "match" else en_error_kind(s["o"], k)
                M[cb][cs_] += 1
        return M

    mats = {f"R2_e{e}": en_matrix(B0, scored[f"R2_e{e}_beam5"]) for e in (1, 2, 3)}
    mats["L1_e3"] = en_matrix(B0, scored["L1_e3_beam5"])
    M = mats["R2_e2"]
    fate = {c: {"n_b0": sum(M[c].values()), "to_correct": M[c]["correct"], "same": M[c][c],
                "other_wrong": sum(v for k, v in M[c].items() if k not in ("correct", c))} for c in EN_CATS if c != "correct"}
    A["D4"] = {"matrices": mats, "fate_by_b0_category_e2": fate,
               "corruptions_by_sys_category_e2": {c: M["correct"][c] for c in EN_CATS if c != "correct"},
               "lexical_substitution_b0_to_correct_e2": M["lexical_substitution"]["correct"],
               "translation_like_b0_to_correct_e2": M["translation_like"]["correct"],
               "translation_like_b0_to_other_wrong_e2": {k: v for k, v in M["translation_like"].items() if k not in ("correct", "translation_like") and v}}
    ex = [{k: e[k] for k in ("utt", "unit", "b0_hyp", "sys_hyp", "en_cat_b0", "en_cat_sys")} for e in events
          if e["substantive_en"] and e["en_cat_b0"] == "lexical_substitution" and e["en_cat_sys"] != "lexical_substitution"][:25]

    # ------------------------------------------------------------------ D5 alignment sensitivity
    def ev_sets(Sb, Ss, aligner):
        en_r, en_c, zh_c = set(), set(), set()
        cost_eq = 0
        for b, s in zip(Sb, Ss):
            a = b["o"]["ref"]
            ob = aligner(a, b["o"]["hyp"])
            os_ = aligner(a, s["o"]["hyp"])
            cost_eq += edit_cost(ob) == edit_cost(alignment(a, b["o"]["hyp"])) and edit_cost(os_) == edit_cost(alignment(a, s["o"]["hyp"]))
            okb, oks = ok_from_ops(ob, len(a)), ok_from_ops(os_, len(a))
            for k, (u, t) in enumerate(zip(a, b["o"]["tags"])):
                if t == "EN" and u not in FILL:
                    if not okb[k] and oks[k]:
                        en_r.add((b["id"], k))
                    if okb[k] and not oks[k]:
                        en_c.add((b["id"], k))
                if t == "ZH" and u not in FILL and okb[k] and not oks[k]:
                    zh_c.add((b["id"], k))
        return en_r, en_c, zh_c, cost_eq
    canon = ev_sets(B0, R2, alignment)
    altn = ev_sets(B0, R2, alt_alignment)
    stab = {nm: {"canonical": len(c), "alternative": len(a_), "both": len(c & a_), "stable_fraction": len(c & a_) / len(c) if c else None}
            for nm, c, a_ in zip(("en_repairs", "en_corruptions", "zh_corruptions"), canon[:3], altn[:3])}
    top = sorted(range(len(utts)), key=lambda i: -utts[i]["zh_new"])[:10]

    def nedit(x, y):
        return edit_cost(alignment(x, y)) / max(1, len(x), len(y))
    derail = [{"rank": r + 1, "zh_new": utts[i]["zh_new"], "en_corr": utts[i]["en_corr"], "stratum": utts[i]["stratum"],
               "hyp_len_ratio": len(R2[i]["o"]["hyp"]) / max(1, len(B0[i]["o"]["hyp"])), "ins_b0": utts[i]["ins_b0"], "ins_sys": utts[i]["ins_sys"],
               "loop": utts[i]["loop_sys"], "b0_vs_sys_hyp_norm_edit": nedit(B0[i]["o"]["hyp"], R2[i]["o"]["hyp"]),
               "zh_sub_vs_del": [sum(1 for e in events if e["utt"] == utts[i]["utt"] and e["zh_non_filler"] and e["transition"] == "corruption" and e["sys_op"] == op)
                                 for op in ("sub", "del")]} for r, i in enumerate(top)]
    A["D5"] = {"alternative_alignment_cost_equal_utterances": altn[3], "event_stability": stab, "top10_damaged_derailment": derail,
               "note": "sensitivity only; canonical reft.text.alignment remains the official scorer"}

    # ------------------------------------------------------------------ oracle (ORACLE_ONLY -- NOT DEPLOYABLE)
    choose = [s["counts"][2] < b["counts"][2] for b, s in zip(B0, R2)]
    orc = [s if c else b for b, s, c in zip(B0, R2, choose)]
    mo = metrics(orc)
    A["ORACLE_ONLY_NOT_DEPLOYABLE"] = {"rule": "per utterance, R2 epoch 2 if it has strictly fewer mixed errors than B0 (reference-based)",
                                       "utterances_using_R2": sum(choose), "mer": mo["overall"]["mer"], "cs_mer": mo["CS"]["mer"],
                                       "new_zh_errors": sum(u["zh_new"] for u, c in zip(utts, choose) if c),
                                       "en_corrections": sum(u["en_corr"] for u, c in zip(utts, choose) if c),
                                       "en_corruptions": sum(u["en_corrupt"] for u, c in zip(utts, choose) if c)}

    # ------------------------------------------------------------------ write
    od = art(OUT)
    od.mkdir(parents=True, exist_ok=True)
    write_jsonl(od / "events_private.jsonl", events)
    write_jsonl(od / "utterance_summary_private.jsonl", utts)
    write_jsonl(od / "dialogue_summary_private.jsonl", [{"dialogue": k, **v} for k, v in dd.items()])
    write_json(od / "examples_private.json", {"en_lexical_substitution_changes": ex,
                                              "top_damaged_utterances": [utts[i]["utt"] for i in top], "epoch3_loop_utterances": [S3[i]["id"] for i in loop_idx]})
    write_json(od / "integrity.json", integ)
    write_json(od / "analysis.json", {"schema": "reft_p2_1d_analysis_v1", "utc": now(), **A})
    make_figures(A, od / "figures")
    gates = {"G1_integrity": bool(integ["pinned_beam5_hash"] and integ["pinned_greedy_hash"] and integ["selected_epochs"] and integ["paired_equals_report"]
                                  and integ["paired_equals_evaluation"] and integ["metrics_equal_report"] and integ["expected_counts"]
                                  and integ["n_utterances_150"] and integ["n_dialogues_20"] and integ["n_cs_97"]),
             "G2_events": bool(integ["event_table_reconciles"] and integ["event_coverage_exact"] and integ["mixed_change_decomposes"]),
             "G3_localization": all(k in A["D1"] for k in ("by_stratum", "by_dist_en", "by_dist_switch", "utterance_top1_top3_top5_share")),
             "G4_cooccurrence": "all" in A["D2"] and "cs_only" in A["D2"],
             "G5_dynamics_lexical": bool(A["D3"]["epochs"]) and "fate_by_b0_category_e2" in A["D4"] and "epoch3_loop_sensitivity" in A["D3"]}
    outs = {p.name: sha(p) for p in sorted(od.glob("*.json*"))}
    outs.update({f"figures/{p.name}": sha(p) for p in sorted((od / "figures").glob("*.png"))})
    write_json(od / "manifest.json", {"schema": "reft_p2_1d_manifest_v1", "utc": now(), "git": git_state(), "inputs": hashes,
                                      "outputs": outs, "gates_data": gates, "cpu_seconds": time.time() - t0,
                                      "no_gpu_no_model_no_decoding": True})
    print(json.dumps({"gates_data": gates, "integrity": integ, "cpu_seconds": time.time() - t0}, indent=1, default=str))


def make_figures(A, d: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d.mkdir(parents=True, exist_ok=True)
    D1, D2, D3, D4 = A["D1"], A["D2"], A["D3"], A["D4"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    v = D1["per_affected_utterance"]["sorted"]
    ax[0].bar(range(1, len(v) + 1), v, color="#4c72b0")
    ax[0].set(title=f"F1a new Mandarin errors per affected utterance (n={len(v)})", xlabel="utterance rank (anonymised)", ylabel="new errors")
    w = [x for x in D1["per_dialogue_sorted"]]
    ax[1].bar(range(1, len(w) + 1), w, color="#55a868")
    ax[1].set(title="F1b new Mandarin errors per dialogue (20)", xlabel="dialogue rank (anonymised)", ylabel="new errors")
    fig.tight_layout()
    fig.savefig(d / "F1_damage_concentration.png", dpi=130)
    plt.close(fig)
    cells = D2["all"]
    names = list(cells)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    for off, key, col in ((-0.27, "en_corrections", "#4c72b0"), (0, "zh_new", "#c44e52"), (0.27, "zh_repairs", "#55a868")):
        ax.bar(x + off, [cells[n][key] for n in names], width=0.27, label=key, color=col)
    ax.set_xticks(x, [f"{n}\n(utts={cells[n]['utterances']})" for n in names], fontsize=8)
    ax.set(title="F2 English repair x Mandarin damage (R2 epoch 2 vs B0, dev150)", ylabel="reference units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "F2_repair_damage_contingency.png", dpi=130)
    plt.close(fig)
    ep = D3["epochs"]
    es = [1, 2, 3]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, col in (("en_corr", "#4c72b0"), ("en_corrupt", "#8172b2"), ("zh_new", "#c44e52"), ("zh_rep", "#55a868")):
        ax[0].plot(es, [ep[e][key] for e in es], marker="o", label=key, color=col)
    ax[0].set(title="F3a paired unit changes vs B0", xlabel="R2 epoch", ylabel="units")
    ax[0].legend(fontsize=8)
    for key, col in (("mer", "#333333"), ("cs_mer", "#dd8452"), ("zh_cer", "#c44e52")):
        ax[1].plot([0] + es, [ep[e][key] for e in [0] + es], marker="o", label=key, color=col)
    ax[1].set(title="F3b beam-5 rates (epoch 0 = B0)", xlabel="R2 epoch")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "F3_epoch_dynamics.png", dpi=130)
    plt.close(fig)
    M = D4["matrices"]["R2_e2"]
    arr = np.array([[M[a][b] for b in EN_CATS] for a in EN_CATS])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(np.log1p(arr), cmap="Blues")
    for i in range(len(EN_CATS)):
        for j in range(len(EN_CATS)):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(EN_CATS)), EN_CATS, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(EN_CATS)), EN_CATS, fontsize=8)
    ax.set(xlabel="R2 epoch 2 category", ylabel="B0 category", title="F4 substantive English unit transitions (counts; colour log1p)")
    fig.tight_layout()
    fig.savefig(d / "F4_english_transitions.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a_, key, title in ((ax[0], "by_dist_en", "distance to nearest substantive EN unit"), (ax[1], "by_dist_switch", "distance to nearest EN<->ZH switch")):
        b = A["D1"][key]
        labs = list(b)
        r = [b[l_]["rate"] or 0 for l_ in labs]
        lo = [r[i] - (b[l_]["wilson95"][0] if b[l_]["wilson95"] else 0) for i, l_ in enumerate(labs)]
        hi = [(b[l_]["wilson95"][1] if b[l_]["wilson95"] else 0) - r[i] for i, l_ in enumerate(labs)]
        a_.bar(range(len(labs)), r, yerr=[lo, hi], capsize=3, color="#c44e52")
        a_.set_xticks(range(len(labs)), [f"{l_}\nn={b[l_]['b0_correct']}" for l_ in labs], fontsize=8)
        a_.set(title=f"F5 new-error rate vs {title}", xlabel="reference-unit distance bin (not time)", ylabel="new Mandarin error rate")
    fig.tight_layout()
    fig.savefig(d / "F5_distance_to_switch.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
