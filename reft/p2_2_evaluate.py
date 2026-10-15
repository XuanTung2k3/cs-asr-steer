#!/usr/bin/env python
"""P2-2 selection, evaluation, verdict and exploratory mechanism audit. docs/reft/P2_2_DECISION_RULES.md.

  select        (CPU) unchanged P2-1 checkpoint rule per M2 run -> selection.json (write-once)
  evaluate      (CPU) C0 (CE), C1 (GoldKL) and M2 per seed/epoch/regime vs pinned B0 with the P2-1b metric code;
                paired M2 - GoldKL (primary) and M2 - CE with shared dialogue draws -> p2_2/evaluation.json
  mech-anchors  (CPU, reference-aware) own-prefix anchors of the M2 selected outputs on the ten P2-1c cases
  mech-replay   (GPU, firewalled) P2-1c replay engine: B0 / GoldKL / M2 at B0 prefixes and own prefixes
  report        (CPU) mechanism scoring, deterministic verdict, figures -> p2_2/report.json
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

from reft.common import art, fillers, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import git_state, now, sha, slurm  # noqa: E402

SEEDS = (42, 43, 44)
ZH_DENOM = 4950
ARMS = ("CE", "KL", "M2")


def out(*p):
    return art("p2_2", *p)


def c22():
    from reft.p2_2_train import c22 as c
    return c()


# ====================================================================== pure decision helpers (unit-tested)

def preservation(kl: dict, m2: dict) -> dict:
    """A: M2 vs matched GoldKL. kl, m2: {seed: {zh_new, sb, loops, nonterm}}."""
    A1 = float(np.mean([m2[s]["zh_new"] for s in SEEDS])) < float(np.mean([kl[s]["zh_new"] for s in SEEDS]))
    red = [(kl[s]["sb"] - m2[s]["sb"]) / kl[s]["sb"] for s in SEEDS if kl[s]["sb"] > 0]
    A2 = bool(red) and float(np.mean(red)) >= 0.20
    A3 = sum(m2[s]["sb"] < kl[s]["sb"] for s in SEEDS) >= 2
    A4 = sum(m2[s]["loops"] for s in SEEDS) <= sum(kl[s]["loops"] for s in SEEDS) + 1 and \
        sum(m2[s]["nonterm"] for s in SEEDS) <= sum(kl[s]["nonterm"] for s in SEEDS) + 1
    mk = float(np.mean([kl[s]["zh_new"] for s in SEEDS]))
    zr = 1 - float(np.mean([m2[s]["zh_new"] for s in SEEDS])) / mk if mk else 0.0
    return {"A1": bool(A1), "A2": bool(A2), "A3": bool(A3), "A4": bool(A4), "A": bool(A1 and A2 and A3 and A4),
            "mean_sb_reduction": float(np.mean(red)) if red else None, "mean_zh_new_reduction": zr}


def retention(kl: dict, m2: dict) -> dict:
    """B: correction retention vs matched GoldKL. {seed: {en_net, cs_mer, worst_dlg}}."""
    B1 = sum(m2[s]["en_net"] > 0 for s in SEEDS) >= 2
    mk = float(np.mean([kl[s]["en_net"] for s in SEEDS]))
    ratio = float(np.mean([m2[s]["en_net"] for s in SEEDS])) / mk if mk > 0 else None
    B2 = ratio is not None and ratio >= 0.75
    B3 = float(np.mean([m2[s]["cs_mer"] for s in SEEDS]) - np.mean([kl[s]["cs_mer"] for s in SEEDS])) <= 0.005
    B4 = all(m2[s]["worst_dlg"] <= kl[s]["worst_dlg"] + 10 for s in SEEDS)
    return {"B1": bool(B1), "B2": bool(B2), "B3": bool(B3), "B4": bool(B4), "B": bool(B1 and B2 and B3 and B4), "en_net_ratio": ratio}


def strong(m2: dict) -> bool:
    """C: Mandarin retention >= 0.99 (zh_new <= 49) in >= 2 seeds."""
    return sum(m2[s]["zh_new"] <= int(ZH_DENOM * 0.01) for s in SEEDS) >= 2


def exposure(per_seed: dict, G: dict) -> dict:
    """D: per seed (epochs 2-3): active non-identity OP events with gradient >= min, and valid share of located >= min."""
    ok = {s: v["non_identity_active"] >= G["min_active_non_identity_events"] and (v["valid_share"] or 0) >= G["min_valid_share_of_located"]
          for s, v in per_seed.items()}
    return {"per_seed": ok, "D": sum(ok.values()) >= G["min_seeds"]}


RECOMMEND = {"P2_2_ENGINEERING_BLOCKED": "HOLD_FOR_ENGINEERING_REPAIR",
             "P2_2_ONPOLICY_EXPOSURE_INSUFFICIENT": "GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING",
             "P2_2_ONPOLICY_FEASIBLE": "GO_P2_3_MATCHED_SUPERVISION_CONTROL",
             "P2_2_CORRECTIONS_SUPPRESSED": "GO_P2_3_ONPOLICY_MECHANISM_VALIDATION",
             "P2_2_NO_INCREMENTAL_GAIN": "GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING",
             "P2_2_INCONCLUSIVE": "GO_P2_3_ONPOLICY_MECHANISM_VALIDATION"}


def decide(integrity_ok: bool, A: dict, B: dict, C: bool, D: dict) -> dict:
    """Precedence: BLOCKED > EXPOSURE_INSUFFICIENT > FEASIBLE(+RESCUED) > CORRECTIONS_SUPPRESSED > NO_INCREMENTAL_GAIN >
    INCONCLUSIVE. An apparent metric gain never hides insufficient on-policy exposure."""
    sub = None
    if not integrity_ok:
        v = "P2_2_ENGINEERING_BLOCKED"
    elif not D["D"]:
        v = "P2_2_ONPOLICY_EXPOSURE_INSUFFICIENT"
    elif A["A"] and B["B"]:
        v = "P2_2_ONPOLICY_FEASIBLE"
        sub = "P2_2_PRESERVATION_RESCUED" if C else None
    elif A["A"]:
        v = "P2_2_CORRECTIONS_SUPPRESSED"
    elif A["mean_zh_new_reduction"] < 0.10 and (A["mean_sb_reduction"] is None or A["mean_sb_reduction"] < 0.10):
        v = "P2_2_NO_INCREMENTAL_GAIN"
    else:
        v = "P2_2_INCONCLUSIVE"
    return {"verdict": v, "substatus": sub, "recommendation": RECOMMEND[v]}


def span_fates(ev: list) -> collections.Counter:
    """Reference switch-back spans (maximal runs of non-filler Mandarin units right after a substantive English unit):
    system retention of their B0-correct units -> intact (>= 0.8) / partial / dropped (<= 0.2)."""
    ev = sorted(ev, key=lambda e: e["k"])
    out_ = collections.Counter()
    i = 0
    while i < len(ev):
        if ev[i]["zh_non_filler"] and i > 0 and ev[i - 1]["substantive_en"]:
            j = i
            while j < len(ev) and ev[j]["zh_non_filler"]:
                j += 1
            ok = [e["transition"] != "corruption" for e in ev[i:j] if e["b0_ok"]]
            if ok:
                f = sum(ok) / len(ok)
                out_["intact" if f >= 0.8 else "dropped" if f <= 0.2 else "partial"] += 1
            i = j
        else:
            i += 1
    return out_


# ====================================================================== select / evaluate

def run_dirs() -> dict:
    from reft.p2_2_train import control_paths
    d = {}
    for n, c in control_paths().items():
        d[(n[:2], c["seed"])] = art(c["run"])
    for s in SEEDS:
        d[("M2", s)] = out("training", f"M2_seed{s}")
    return d


def run_select():
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, score_system, select_epoch
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    for s in SEEDS:
        rd = out("training", f"M2_seed{s}")
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


def run_evaluate():
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    from reft.p2_1b_evaluate import dialogue_table, failure, fates, json_keys
    from reft.p2_1d_audit import utterance_events
    from reft.p2_2_train import check_freeze
    fz = check_freeze()
    if out("evaluation.json").exists():
        raise SystemExit("evaluation.json exists; write-once")
    C = p21()
    BS = c22()["bootstrap"]
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    LU = C["evaluation"]["loop_units"]
    pb = C["frozen_baseline"]
    assert sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"]
    base = {r: score_system(meta, refs, {x["id"]: x for x in read_jsonl(art(pb[r]))}, FILL, LU) for r in ("beam5", "greedy")}
    runs, dtabs = {}, {}
    for (arm, s), rd in run_dirs().items():
        sel = read_json(rd / "selection.json")["selected"]["epoch"]
        man = read_json(rd / "manifest.json")
        R = {"dir": str(rd), "selected_epoch": sel, "history": man["history"], "efficiency": man.get("efficiency"),
             "weights_unchanged": man["weights_unchanged"], "zero_init_identity": man["history"][0]["zero_init_identity_beam5"], "epochs": {}}
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
                    for b, s_ in zip(base["beam5"], S):
                        ev, _ = utterance_events(b, s_, FILL, None, "beam5", b["id"], b["dialogue"], b["stratum"])
                        sf += span_fates(ev)
                    rec["switchback_spans"] = dict(sf)
                    if ep == sel:
                        dtabs[(arm, s)] = dialogue_table(f["_per_utt"])
                R["epochs"][ep][reg] = rec
        runs[f"{arm}_seed{s}"] = R
    # controls reproduce the P2-1b evaluation exactly
    E1b = read_json(art("p2_1b/evaluation.json"))["runs"]
    repro = {}
    for (arm, s) in [(a, s) for a in ("CE", "KL") for s in SEEDS]:
        mine = runs[f"{arm}_seed{s}"]
        theirs = E1b[f"{arm}_seed{s}"]
        e = str(mine["selected_epoch"])
        repro[f"{arm}{s}"] = (mine["selected_epoch"] == theirs["selected_epoch"]
                              and mine["epochs"][int(e)]["beam5"]["failure"]["zh_new"] == theirs["epochs"][e]["beam5"]["failure"]["zh_new"]
                              and mine["epochs"][int(e)]["beam5"]["failure"]["sb_units"] == theirs["epochs"][e]["beam5"]["failure"]["sb_units"]
                              and mine["epochs"][int(e)]["beam5"]["vs_B0"]["net"] == theirs["epochs"][e]["beam5"]["vs_B0"]["net"]
                              and mine["epochs"][int(e)]["beam5"]["metrics"]["CS"]["mer"] == theirs["epochs"][e]["beam5"]["metrics"]["CS"]["mer"])
    dl = sorted({m["dialogue_id"] for m in meta})
    rng = np.random.default_rng(BS["seed"])
    draws = [rng.choice(len(dl), len(dl)) for _ in range(BS["reps"])]

    def stat(tab, idx):
        a = sum(tab[dl[i]] for i in idx)
        return {"mer": a[0] / a[1], "cs_mer": a[2] / a[3], "zh_new": a[4], "sb": a[5], "en_net": a[6]}
    keys = ("mer", "cs_mer", "zh_new", "sb", "en_net")
    full = list(range(len(dl)))

    def pairs(a1, a0):
        pr = {}
        for s in SEEDS:
            pt = {k: stat(dtabs[(a1, s)], full)[k] - stat(dtabs[(a0, s)], full)[k] for k in keys}
            bs = {k: [] for k in keys}
            for idx in draws:
                x, y = stat(dtabs[(a1, s)], idx), stat(dtabs[(a0, s)], idx)
                for k in keys:
                    bs[k].append(x[k] - y[k])
            pr[s] = {k: {"point": pt[k], "ci95": [float(np.quantile(bs[k], q)) for q in (0.025, 0.975)]} for k in keys}
        pooled = {k: [] for k in keys}
        for idx in draws:
            for k in keys:
                pooled[k].append(np.mean([stat(dtabs[(a1, s)], idx)[k] - stat(dtabs[(a0, s)], idx)[k] for s in SEEDS]))
        pr["seed_mean"] = {k: {"point": float(np.mean([pr[s][k]["point"] for s in SEEDS])),
                               "range": [float(min(pr[s][k]["point"] for s in SEEDS)), float(max(pr[s][k]["point"] for s in SEEDS))],
                               "ci95": [float(np.quantile(pooled[k], q)) for q in (0.025, 0.975)]} for k in keys}
        return json_keys(pr)
    em = {s: {ep: {k: runs[f"M2_seed{s}"]["epochs"][ep]["beam5"]["failure"][k] - runs[f"KL_seed{s}"]["epochs"][ep]["beam5"]["failure"][k]
                   for k in ("zh_new", "sb_units")} | {"cs_mer": runs[f"M2_seed{s}"]["epochs"][ep]["beam5"]["metrics"]["CS"]["mer"]
                                                     - runs[f"KL_seed{s}"]["epochs"][ep]["beam5"]["metrics"]["CS"]["mer"],
                                                     "en_net": runs[f"M2_seed{s}"]["epochs"][ep]["beam5"]["vs_B0"]["net"]
                                                     - runs[f"KL_seed{s}"]["epochs"][ep]["beam5"]["vs_B0"]["net"]}
              for ep in (1, 2, 3)} for s in SEEDS}
    write_json(out("evaluation.json"), {"schema": "reft_p2_2_evaluation_v1", "utc": now(), **fz, "git": git_state(), "slurm": slurm(),
                                        "B0": {r: metrics(base[r]) for r in base}, "runs": runs, "controls_reproduce_p2_1b": repro,
                                        "paired_m2_minus_kl": pairs("M2", "KL"), "paired_m2_minus_ce": pairs("M2", "CE"),
                                        "epoch_matched_m2_minus_kl": {str(s): v for s, v in em.items()},
                                        "note": "dev150 exposed; seeds share the 20 dialogues; descriptive uncertainty"})
    print("evaluation written", repro)


# ====================================================================== mechanism audit (exploratory; P2-1c cases)

MECH_SYSTEMS = ("B0", "KL42", "KL43", "KL44", "M2_42", "M2_43", "M2_44")


def m2_selected() -> dict:
    res = {}
    for s in SEEDS:
        rd = out("training", f"M2_seed{s}")
        ep = read_json(rd / "selection.json")["selected"]["epoch"]
        res[f"M2_{s}"] = {"adapter": rd / f"epoch{ep}" / "adapter.pt", "decode": rd / f"epoch{ep}" / "dev_beam5.jsonl", "seed": s, "epoch": ep}
    return res


def run_mech_anchors():
    """Own-prefix anchors of M2's selected outputs (P2-1c T3 rule, P2-1c helpers); B0 anchors reused from P2-1c."""
    from reft.evaluate_supervised import dev_refs
    from reft.p2_1c_continuation_audit import anchor_at, boundaries, hyp_after
    from reft.p2_2_train import check_freeze
    from reft.text import one, tok_bytes_fn
    from transformers import WhisperTokenizer
    from reft.common import cfg
    check_freeze()
    if out("mech_anchors_private.json").exists():
        raise SystemExit("mech anchors exist; write-once")
    A = {c["key"]: c for c in read_json(art("p2_1c/anchors_private.json"))["cases"]}
    MI = read_json(art("p2_1c/anchors_inference.json"))
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    refs = dev_refs(sorted(c["uid"] for c in A.values()))
    sel = m2_selected()
    outs = {n: {x["id"]: x for x in read_jsonl(v["decode"])} for n, v in sel.items()}
    priv, inf = {}, {}
    for key, c in A.items():
        uid, sw, reg = c["uid"], c["switch_unit"], c["en_region"]
        b0_units = None
        priv[key], inf[key] = {}, {}
        for n in sel:
            r_ = outs[n][uid]
            sc = [t for t in r_["tokens"] if t != eos]
            ms, full = boundaries([tb(t) for t in sc])
            o = one(refs[uid], r_["text"])
            last = next((r for r in range(sw - 1, -1, -1) if o["by"][r][0] in ("match", "sub")), None)
            if last is None or full != o["hyp"]:
                priv[key][n] = {"k": None, "valid": False}
                inf[key][n] = None
                continue
            a = anchor_at(ms, hyp_after(o["ops"], last))
            common = reg is not None and last >= reg[0]
            priv[key][n] = {**a, "ref_unit": last, "common_location": common, "valid": a["exact"] and common and a["k"] < len(r_["tokens"])}
            inf[key][n] = a["k"]
    write_json(out("mech_anchors_private.json"), {"utc": now(), "cases": priv})
    write_json(out("mech_anchors_inference.json"), {"utc": now(), "own_k": inf,
                                                    "p2_1c_anchors_inference_sha256": sha(art("p2_1c/anchors_inference.json"))})
    print(json.dumps({k: {n: v["valid"] for n, v in d.items()} for k, d in priv.items()}))


def run_mech_replay():
    """GPU, P2-1c firewall and Engine: B0, KL42-44 (P2-1c systems) and M2 selected checkpoints."""
    import torch
    from reft.adapters import make_adapter
    from reft.p2_1b_train import seed_config
    from reft.p2_1c_continuation_audit import Engine, gpu_firewall
    from reft.p2_2_train import check_freeze
    from reft.train_supervised import c21
    fz = check_freeze()
    for f in ("mech_replay_private.jsonl", "mech_continuation_private.jsonl", "mech_candidates_private.jsonl"):
        if out(f).exists():
            raise SystemExit(f"{f} exists; write-once")
    sel = m2_selected()
    MI = read_json(art("p2_1c/anchors_inference.json"))
    own = read_json(out("mech_anchors_inference.json"))["own_k"]
    gpu_firewall()
    E = Engine()
    C1 = c21()
    for n, v in sel.items():
        a = make_adapter("R2", E.b.model, seed_config(C1, v["seed"]), E.b.device)
        a.load_state_dict(torch.load(v["adapter"]))
        a.module.eval()
        E.ads[n] = a
        E.outputs[n] = {x["id"]: x for x in read_jsonl(v["decode"])}
    p21c_own = {c["key"]: c["own_anchors"] for c in MI["cases"]}
    rep, cont, cands = [], [], []
    for c in MI["cases"]:
        key, uid, P0 = c["key"], c["uid"], c["prompt"]
        E.set_case(uid)
        b0t = E.outputs["B0"][uid]["tokens"]
        kpost = c["b0_anchors"]["post_en"]
        conds = []
        for a in ("start", "pre_en", "post_en", "early_zh"):
            k = c["b0_anchors"][a]
            if k is not None:
                conds += [(a, "B0", k, m) for m in MECH_SYSTEMS]
        for s in MECH_SYSTEMS[1:]:
            k = p21c_own[key].get(s) if s.startswith("KL") else own[key].get(s)
            if k is not None:
                conds += [("own", s, k, m) for m in (s, "B0")]
        for a, src, k, m in conds:
            toks = E.outputs[src][uid]["tokens"]
            P = P0 + toks[:k]
            tg = {"src_next": toks[k] if k < len(toks) else None}
            if a == "own":
                tg["b0_switch"] = b0t[kpost] if kpost is not None and kpost < len(b0t) else None
            rep.append({"case": key, "anchor": a, "prefix_source": src, "k": k, "model": m, **E.next_token(m, P, tg)})
            cont.append({"case": key, "anchor": a, "prefix_source": src, "k": k, "model": m, **E.generate(m, P)})
        if kpost is not None:
            P = P0 + b0t[:kpost]
            pool = collections.OrderedDict()
            pool[tuple(b0t[kpost:])] = ["B0_original"]
            for x in cont:
                if x["case"] == key and x["anchor"] == "post_en" and x["prefix_source"] == "B0":
                    for i, bm in enumerate(x["beams"]):
                        pool.setdefault(tuple(bm["suffix"]), []).append(f"{x['model']}_beam{i}")
            for ci, (cd, srcs) in enumerate(pool.items()):
                cands.append({"case": key, "anchor": "post_en", "k": kpost, "cand": ci, "sources": srcs, "tokens": list(cd),
                              "scores": {m: E.score(m, P, list(cd)) for m in MECH_SYSTEMS}})
    write_jsonl(out("mech_replay_private.jsonl"), rep)
    write_jsonl(out("mech_continuation_private.jsonl"), cont)
    write_jsonl(out("mech_candidates_private.jsonl"), cands)
    write_json(out("mech_integrity.json"), {"utc": now(), **fz, "slurm": slurm(), "hooks_clear": E.hooks_clear(),
                                            "generate_calls": E.n_gen, "tf_forwards": E.n_tf,
                                            "outputs_sha256": {f: sha(out(f)) for f in ("mech_replay_private.jsonl", "mech_continuation_private.jsonl",
                                                                                          "mech_candidates_private.jsonl")}})
    print("mech replay done", E.n_gen, E.n_tf)


def mech_analyze() -> dict:
    """CPU, reference-aware scoring of the mechanism replay (P2-1c scope and status rules)."""
    from transformers import WhisperTokenizer
    from reft.common import cfg
    from reft.evaluate_supervised import dev_refs
    from reft.p2_1c_continuation_audit import cc as c1c, seq_preference, status_of
    from reft.text import one
    R = c1c()["rules"]
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    eos = tok.eos_token_id
    A = {c["key"]: c for c in read_json(art("p2_1c/anchors_private.json"))["cases"]}
    own_priv = read_json(out("mech_anchors_private.json"))["cases"]
    refs = dev_refs(sorted(c["uid"] for c in A.values()))
    sel = m2_selected()
    outs = {"B0": {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}}
    from reft.p2_2_train import control_paths
    for n, c in control_paths().items():
        if n.startswith("KL"):
            outs[n] = {x["id"]: x for x in read_jsonl(c["decode"])}
    for n, v in sel.items():
        outs[n] = {x["id"]: x for x in read_jsonl(v["decode"])}
    text = lambda ids: tok.decode([t for t in ids if t != eos], skip_special_tokens=True).strip()  # noqa: E731

    def ret(key, tokens, scope):
        o = one(refs[A[key]["uid"]], text(tokens))
        return float(np.mean([o["by"][r][0] == "match" for r in scope]))

    def scope(key, anchor):
        c = A[key]
        return c["b0_anchors"]["early_zh"]["scope_units"] if anchor == "early_zh" else c["span_units"]
    CONT = read_jsonl(out("mech_continuation_private.jsonl"))
    REP = {(r["case"], r["anchor"], r["prefix_source"], r["model"]): r for r in read_jsonl(out("mech_replay_private.jsonl"))}
    st = {}
    for x in CONT:
        toks = outs[x["prefix_source"]][A[x["case"]]["uid"]]["tokens"][:x["k"]] + x["beams"][0]["suffix"]
        st[(x["case"], x["anchor"], x["prefix_source"], x["model"])] = status_of(ret(x["case"], toks, scope(x["case"], x["anchor"])), R)
    cands = read_jsonl(out("mech_candidates_private.jsonl"))
    for x in cands:
        toks = outs["B0"][A[x["case"]]["uid"]]["tokens"][:x["k"]] + x["tokens"]
        x["status"] = status_of(ret(x["case"], toks, scope(x["case"], "post_en")), R)
    rows = []
    for key, c in A.items():
        for m in MECH_SYSTEMS[1:]:
            orig = status_of(ret(key, outs[m][c["uid"]]["tokens"], c["span_units"]), R)
            own_valid = (c["own_anchors"][m]["valid"] if m.startswith("KL") else own_priv[key][m]["valid"])
            rS, r0 = REP.get((key, "post_en", "B0", m)), REP.get((key, "post_en", "B0", "B0"))
            q, mg = seq_preference([x for x in cands if x["case"] == key], m)
            rows.append({"case": key, "kind": c["kind"], "group": c["group"], "system": m, "arm": "KL" if m.startswith("KL") else "M2",
                         "orig": orig, "b0_prefix": st.get((key, "post_en", "B0", m)), "own_prefix": st.get((key, "own", m, m)) if own_valid else None,
                         "b0_model_at_own_prefix": st.get((key, "own", m, "B0")) if own_valid else None,
                         "own_valid": own_valid, "pre_en": st.get((key, "pre_en", "B0", m)), "early_zh": st.get((key, "early_zh", "B0", m)),
                         "local_delta": (rS["src_next"]["logp"] - r0["src_next"]["logp"]) if rS else None,
                         "local_rank": rS["src_next"]["rank"] if rS else None, "T5_omission_preferred": q, "T5_margin": mg})
    # P2-1c consistency: KL rows must reproduce the P2-1c analysis (same engine, same checkpoints)
    p1c = {(r["case"], r["system"]): r for r in read_json(art("p2_1c/analysis_r1.json"))["rows"]}
    cons = all(p1c[(r["case"], r["system"])]["B_b0_prefix_status"] == r["b0_prefix"] and p1c[(r["case"], r["system"])]["orig_status"] == r["orig"]
               for r in rows if r["arm"] == "KL" and A[r["case"]]["b0_anchors"]["post_en"]["valid"])
    return {"rows": rows, "kl_rows_reproduce_p2_1c": cons}


# ====================================================================== report

def run_report():
    from reft.p2_2_train import check_freeze
    fz = check_freeze()
    if out("report.json").exists():
        raise SystemExit("report.json exists; write-once")
    E = read_json(out("evaluation.json"))
    G = c22()["exposure_gate"]
    sel = {}
    for name, R_ in E["runs"].items():
        e = R_["epochs"][str(R_["selected_epoch"])]["beam5"]
        sel[name] = {"zh_new": e["failure"]["zh_new"], "sb": e["failure"]["sb_units"], "loops": e["metrics"]["overall"]["loop_utterances"],
                     "nonterm": e["metrics"]["overall"]["not_terminated"], "en_net": e["vs_B0"]["net"], "cs_mer": e["metrics"]["CS"]["mer"],
                     "mer": e["metrics"]["overall"]["mer"], "retention": e["vs_B0"]["mandarin_retention"],
                     "worst_dlg": e["failure"]["worst_dialogue_delta_mixed"], "epoch": R_["selected_epoch"],
                     "zh_del_new": e["failure"]["zh_del_new"], "near_switch": e["failure"]["near_switch"], "far": e["failure"]["far"],
                     "spans": e["switchback_spans"], "en_fates": {k: e["english_fates"][k] for k in ("omission_to_correct", "translation_to_correct", "lexical_to_correct", "correct_to_wrong")}}
    kl = {s: sel[f"KL_seed{s}"] for s in SEEDS}
    m2 = {s: sel[f"M2_seed{s}"] for s in SEEDS}
    exp = {}
    for s in SEEDS:
        h = [x for x in E["runs"][f"M2_seed{s}"]["history"] if x["epoch"] in (2, 3)]
        loc = sum(x["rollout"]["located"] for x in h)
        exp[s] = {"non_identity_active": sum(x["op_non_identity_events"] for x in h), "valid_share": (sum(x["rollout"]["active"] for x in h) / loc) if loc else None,
                  "per_epoch": {x["epoch"]: {k: x["rollout"][k] for k in ("located", "active", "T", "R", "U", "active_non_identity", "active_sb", "valid_share_of_located")}
                                | {"op_grad_norm_mean": x["op_grad_norm_mean"]} for x in E["runs"][f"M2_seed{s}"]["history"][1:]}}
    A, B, Cst, D = preservation(kl, m2), retention(kl, m2), strong(m2), exposure(exp, G)
    ver = read_json(out("verify.json")) if out("verify.json").exists() else None
    mech = mech_analyze() if out("mech_continuation_private.jsonl").exists() else None
    integ = {"verify_ok": bool(ver and ver["all_ok"]), "all_runs_present": len(E["runs"]) == 9,
             "zero_init_150": all(R_["zero_init_identity"] == 150 for R_ in E["runs"].values()),
             "weights_unchanged": all(R_["weights_unchanged"] for R_ in E["runs"].values()),
             "controls_reproduce_p2_1b": all(E["controls_reproduce_p2_1b"].values()),
             "mechanism_audit_present": mech is not None,
             "epoch1_rollouts_identical_to_b0": all(E["runs"][f"M2_seed{s}"]["history"][1]["rollout"]["hyps_identical_to_b0"] == 500 for s in SEEDS),
             "no_label_mismatch": all(x["rollout"]["label_mismatches"] == 0 for s in SEEDS for x in E["runs"][f"M2_seed{s}"]["history"][1:])}
    dec = decide(all(integ.values()), A, B, Cst, D)
    rep = {"schema": "reft_p2_2_report_v1", "utc": now(), **fz, "git": git_state(), "decision": dec, "integrity": integ, "selected": sel,
           "A": A, "B": B, "C_strong_retention": Cst, "D": D, "exposure": exp, "paired_m2_minus_kl": E["paired_m2_minus_kl"],
           "paired_m2_minus_ce": E["paired_m2_minus_ce"], "epoch_matched_m2_minus_kl": E["epoch_matched_m2_minus_kl"],
           "mechanism": {"rows": mech["rows"], "kl_rows_reproduce_p2_1c": mech["kl_rows_reproduce_p2_1c"]} if mech else None,
           "rules_doc_sha256": sha(ROOT / "docs/reft/P2_2_DECISION_RULES.md")}
    write_json(out("report.json"), rep)
    figures(E, rep)
    write_json(out("manifest.json"), {"utc": now(), **fz, "files": {p.name: sha(p) for p in sorted(out().glob("*.json*")) if p.name != "manifest.json"},
                                      "figures": {p.name: sha(p) for p in sorted(out("figures").glob("*.png"))}})
    print(json.dumps({k: rep[k] for k in ("decision", "integrity", "A", "B", "C_strong_retention", "D")}, indent=1))


def figures(E, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out("figures")
    fd.mkdir(parents=True, exist_ok=True)
    sel = rep["selected"]
    col = {"CE": "#d62728", "KL": "#1f77b4", "M2": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for arm in ARMS:
        xs = [sel[f"{arm}_seed{s}"]["zh_new"] for s in SEEDS]
        ys = [sel[f"{arm}_seed{s}"]["en_net"] for s in SEEDS]
        ax.scatter(xs, ys, c=col[arm], label=arm, s=40)
        for s, x, y in zip(SEEDS, xs, ys):
            ax.annotate(str(s), (x, y), fontsize=7)
    ax.set_xlabel("new Mandarin errors vs B0 (selected, beam-5)")
    ax.set_ylabel("English net corrections vs B0")
    ax.legend()
    ax.set_title("F1 correction-preservation frontier")
    fig.tight_layout()
    fig.savefig(fd / "F1_frontier.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (k, lab) in zip(axs, (("sb", "long switch-back dropped units"), ("zh_new", "new Mandarin errors"), ("near", "near-switch damage"))):
        for s in SEEDS:
            v = [sel[f"KL_seed{s}"]["near_switch"][0] if k == "near" else sel[f"KL_seed{s}"][k],
                 sel[f"M2_seed{s}"]["near_switch"][0] if k == "near" else sel[f"M2_seed{s}"][k]]
            ax.plot([0, 1], v, "o-", label=str(s))
        ax.set_xticks([0, 1], ["GoldKL", "M2"])
        ax.set_title(lab, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("F2 paired GoldKL vs M2 (selected checkpoints)")
    fig.tight_layout()
    fig.savefig(fd / "F2_switchback.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, T, Rr, U = [], [], [], []
    for s in SEEDS:
        for ep, v in rep["exposure"][s]["per_epoch"].items():
            labels.append(f"s{s} e{ep}")
            T.append(v["T"])
            Rr.append(v["R"])
            U.append(v["U"])
    x = np.arange(len(labels))
    ax.bar(x, T, label="teacher-valid (T)")
    ax.bar(x, Rr, bottom=T, label="reference-valid (R)")
    ax.bar(x, U, bottom=np.array(T) + np.array(Rr), label="excluded (U)")
    ax.set_xticks(x, labels, rotation=45, fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title("F3 on-policy events per epoch (training rollouts)")
    fig.tight_layout()
    fig.savefig(fd / "F3_coverage.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for s in SEEDS:
        h = E["runs"][f"M2_seed{s}"]["history"][1:]
        ep = [x["epoch"] for x in h]
        axs[0].plot(ep, [x["train_ce"] for x in h], "o-", label=f"CE s{s}")
        axs[0].plot(ep, [x["op_loss_batch_mean"] or 0 for x in h], "x--", label=f"OP s{s}")
        axs[1].plot(ep, [x["grad_norm_mean"] for x in h], "o-", label=f"total s{s}")
        axs[1].plot(ep, [x["op_grad_norm_mean"] for x in h], "x--", label=f"OP s{s}")
        axs[2].bar([f"s{s}"], [E["runs"][f"M2_seed{s}"]["efficiency"]["wall_seconds"] / 60], color="grey")
        axs[2].bar([f"s{s}"], [(E["runs"][f"M2_seed{s}"]["efficiency"]["rollout_seconds"] + E["runs"][f"M2_seed{s}"]["efficiency"]["event_build_seconds"]) / 60], color="orange")
    axs[0].set_title("train CE / OP loss", fontsize=9)
    axs[1].set_title("grad norm (total / OP part)", fontsize=9)
    axs[2].set_title("job minutes (orange = rollouts + events)", fontsize=9)
    axs[0].legend(fontsize=6)
    axs[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(fd / "F4_training.png", dpi=130)
    plt.close(fig)
    if rep["mechanism"]:
        rows = rep["mechanism"]["rows"]
        keys = sorted({r["case"] for r in rows})
        sysn = [m for m in MECH_SYSTEMS[1:]]
        val = {"recovered": 2, "partial": 1, "dropped": 0, None: np.nan}
        M, lab = [], []
        for k in keys:
            for m in sysn:
                r = next(x for x in rows if x["case"] == k and x["system"] == m)
                M.append([val[r["orig"]], val[r["b0_prefix"]], val[r["own_prefix"]]])
                lab.append(f"{k} {m}")
        fig, ax = plt.subplots(figsize=(5, 11))
        ax.imshow(np.array(M, float), cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks([0, 1, 2], ["original", "B0 prefix", "own prefix"], fontsize=7)
        ax.set_yticks(range(len(lab)), lab, fontsize=5)
        ax.set_title("F5 span status, 10 fixed P2-1c cases (exploratory)", fontsize=8)
        fig.tight_layout()
        fig.savefig(fd / "F5_prefix_robustness.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "evaluate", "mech-anchors", "mech-replay", "report"])
    a = ap.parse_args(argv)
    {"select": run_select, "evaluate": run_evaluate, "mech-anchors": run_mech_anchors, "mech-replay": run_mech_replay,
     "report": run_report}[a.cmd]()


if __name__ == "__main__":
    main()
