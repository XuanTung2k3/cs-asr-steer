#!/usr/bin/env python
"""S1 selection, canonical evaluation, pre-registered verdict and bounded descriptive mechanism checks.
docs/reft/ARCH_S0S1_DECISION_RULES.md.

  select        (CPU) unchanged P2-1 checkpoint rule per R3/M1 run -> training/<run>/selection.json (write-once)
  evaluate      (CPU) R2-GoldKL (historical, recomputed for parity), R3 and M1 per seed/epoch/regime vs pinned B0 with the
                P2-1b / P2-2 metric code; paired M1 - R3 (primary), M1 - R2, R3 - R2 with shared dialogue draws
  mech-anchors  (CPU, reference-aware) own-prefix anchors of the R3 / M1 selected outputs on the ten P2-1c cases
  mech-gpu      (GPU, firewalled, reference-free) 13.1 contribution audit on own dev150 beam-5 outputs + 13.2 P2-1c replay
  report        (CPU) mechanism scoring, deterministic verdict, figures -> arch_s0s1/report.json, manifest.json
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
ARMS = ("R2", "R3", "M1")
ZH_DENOM = 4950


def ca():
    from reft.arch_s0s1_train import ca as c
    return c()


def out(*p):
    from reft.arch_s0s1_train import out as o
    return o(*p)


# ====================================================================== pure decision helpers (unit-tested)

def _mean(x: dict, k: str) -> float:
    return float(np.mean([x[s][k] for s in SEEDS]))


def gain(new: dict, ref: dict, D: dict) -> dict:
    """Meaningful preservation gain of `new` over `ref` ({seed: {zh_new, sb}}): both means lower by >= margin units and
    lower in >= seed_consistency_min matched seeds, for new Mandarin errors and long switch-back dropped units."""
    m, k = D["preservation_margin_units"], D["seed_consistency_min"]
    dz, db = _mean(new, "zh_new") - _mean(ref, "zh_new"), _mean(new, "sb") - _mean(ref, "sb")
    sz = sum(new[s]["zh_new"] < ref[s]["zh_new"] for s in SEEDS)
    sb = sum(new[s]["sb"] < ref[s]["sb"] for s in SEEDS)
    r = {"zh_mean_margin": dz <= -m, "zh_seed_consistent": sz >= k, "sb_mean_margin": db <= -m, "sb_seed_consistent": sb >= k,
         "d_zh_new_mean": dz, "d_sb_mean": db, "seeds_zh_lower": sz, "seeds_sb_lower": sb,
         "rel_zh_new": dz / _mean(ref, "zh_new") if _mean(ref, "zh_new") else None,
         "rel_sb": db / _mean(ref, "sb") if _mean(ref, "sb") else None}
    r["pass"] = r["zh_mean_margin"] and r["zh_seed_consistent"] and r["sb_mean_margin"] and r["sb_seed_consistent"]
    return r


def loops_guard(new: dict, ref: dict, D: dict) -> bool:
    t = D["loops_nonterm_tolerance"]
    return sum(new[s]["loops"] for s in SEEDS) <= sum(ref[s]["loops"] for s in SEEDS) + t and \
        sum(new[s]["nonterm"] for s in SEEDS) <= sum(ref[s]["nonterm"] for s in SEEDS) + t


def guard(m1: dict, r2: dict, D: dict) -> dict:
    """B: existing-best baseline guard (M1 vs R2-GoldKL)."""
    B1 = _mean(m1, "zh_new") <= _mean(r2, "zh_new")
    B2 = _mean(m1, "sb") <= _mean(r2, "sb")
    B3 = loops_guard(m1, r2, D)
    return {"B1": bool(B1), "B2": bool(B2), "B3": bool(B3), "pass": bool(B1 and B2 and B3)}


def corrections(new: dict, r2: dict, cmp: dict, D: dict) -> dict:
    """C: English correction preservation of `new`; ratio and CS MER vs R2-GoldKL, CS MER and worst dialogue also vs `cmp`."""
    C1 = sum(new[s]["en_net"] > 0 for s in SEEDS) >= D["seed_consistency_min"]
    mr2 = _mean(r2, "en_net")
    ratio = _mean(new, "en_net") / mr2 if mr2 > 0 else None
    C2 = ratio is not None and ratio >= D["en_net_ratio_min"]
    d_r2 = _mean(new, "cs_mer") - _mean(r2, "cs_mer")
    d_cmp = _mean(new, "cs_mer") - _mean(cmp, "cs_mer")
    C3 = d_r2 <= D["cs_mer_tolerance"] and d_cmp <= D["cs_mer_tolerance"]
    C4 = all(new[s]["worst_dlg"] <= cmp[s]["worst_dlg"] + D["worst_dialogue_tolerance"] for s in SEEDS)
    return {"C1": bool(C1), "C2": bool(C2), "C3": bool(C3), "C4": bool(C4), "pass": bool(C1 and C2 and C3 and C4),
            "en_net_ratio_vs_R2": ratio, "d_cs_mer_vs_R2": d_r2, "d_cs_mer_vs_cmp": d_cmp}


def strong(x: dict, D: dict) -> bool:
    """D: Mandarin retention >= 0.99 (zh_new <= 49 of 4950) in >= 2 seeds."""
    return sum(x[s]["zh_new"] <= D["strong_zh_new_max"] for s in SEEDS) >= D["seed_consistency_min"]


def improvement_over_r2(x: dict, r2: dict, D: dict) -> dict:
    """Meaningful, stable improvement over R2-GoldKL: the A-convention gain on both preservation dimensions, without
    losing corrections (C-convention with R2 as the comparison arm) and without loop/non-termination worsening."""
    g, c, lg = gain(x, r2, D), corrections(x, r2, r2, D), loops_guard(x, r2, D)
    return {"gain": g, "corrections": c, "loops_guard": lg, "pass": bool(g["pass"] and c["pass"] and lg)}


RECOMMEND = {"ARCH_S1_ENGINEERING_BLOCKED": "HOLD_FOR_ENGINEERING_REPAIR",
             "ARCH_S1_M1_PROMISING": "GO_M1_MECHANISM_AND_INDEPENDENT_VALIDATION",
             "ARCH_S1_M1_BEATS_R3_ONLY": "NO_NEW_BEST_METHOD",
             "ARCH_S1_R3_CAPACITY_GAIN": "CAPACITY_BOTTLENECK_MORE_PLAUSIBLE",
             "ARCH_S1_NO_ARCHITECTURAL_GAIN": "STOP_GENERIC_LOREFT_ARCHITECTURE_EXPANSION",
             "ARCH_S1_MIXED_OR_INCONCLUSIVE": "HOLD_ARCHITECTURE_CLAIMS"}


def decide(integrity_ok: bool, A: dict, B: dict, C: dict, R3_over_R2: dict, M1_over_R2: dict) -> dict:
    """Strict precedence: BLOCKED > M1_PROMISING (A,B,C) > M1_BEATS_R3_ONLY (A,C, not B) > R3_CAPACITY_GAIN (M1 fails A-and-C,
    R3 improves over R2) > NO_ARCHITECTURAL_GAIN (neither improves over R2) > MIXED_OR_INCONCLUSIVE."""
    if not integrity_ok:
        v = "ARCH_S1_ENGINEERING_BLOCKED"
    elif A["pass"] and B["pass"] and C["pass"]:
        v = "ARCH_S1_M1_PROMISING"
    elif A["pass"] and C["pass"]:
        v = "ARCH_S1_M1_BEATS_R3_ONLY"
    elif R3_over_R2["pass"]:
        v = "ARCH_S1_R3_CAPACITY_GAIN"
    elif not M1_over_R2["pass"]:
        v = "ARCH_S1_NO_ARCHITECTURAL_GAIN"
    else:
        v = "ARCH_S1_MIXED_OR_INCONCLUSIVE"
    return {"verdict": v, "recommendation": RECOMMEND[v]}


def token_lang(bs: bytes) -> str:
    """Reference-free language tag of a token from its bytes: ZH if it contains a Han character, EN if Latin letters."""
    s = bs.decode("utf-8", errors="ignore")
    if any("一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" for ch in s):
        return "ZH"
    if any(ch.isascii() and ch.isalpha() for ch in s):
        return "EN"
    return "OTHER"


def contributions(module, h, u):
    """Per-position pre-projection pieces (rank-r space) and the post-projection update, float32.
    R2/R3: e = (W - R) h + b. M1: e = e_h + e_u + b with e_h = (W_h - R) h, e_u = W_u u. Update = e R (rows orthonormal,
    so |update| = |e|)."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        R = module.rot.weight
        e_h = F.linear(h, module.learned.weight - R)
        e_u = F.linear(u, module.w_u) if hasattr(module, "w_u") else torch.zeros_like(e_h)
        e = e_h + e_u + module.learned.bias
        upd = e @ R
        return {"e_h": e_h, "e_u": e_u, "e": e, "update": upd, "W_h_h": F.linear(h, module.learned.weight), "R_h": F.linear(h, R)}


# ====================================================================== runs, selection

def run_dirs() -> dict:
    from reft.arch_s0s1_train import control_runs
    d = {("R2", s): c["dir"] for s, c in control_runs().items()}
    for a in ("R3", "M1"):
        for s in SEEDS:
            d[(a, s)] = out("training", f"{a}_seed{s}")
    return d


def run_select():
    from reft.arch_s0s1_train import check_freeze
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, score_system, select_epoch
    check_freeze()
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    for a in ("R3", "M1"):
        for s in SEEDS:
            rd = out("training", f"{a}_seed{s}")
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

KEYS = ("mer", "cs_mer", "zh_new", "sb", "en_net", "zh_new_cs", "sb_cs")


def dialogue_table(per_utt):
    """Per dialogue: [sys err, units, CS sys err, CS units, zh_new, sb, en_net, b0 err, CS zh_new, CS sb]."""
    t = collections.defaultdict(lambda: np.zeros(10))
    for u in per_utt:
        cs = u["stratum"] == "CS"
        t[u["dialogue"]] += np.array([u["s_err"], u["units"], u["s_err"] if cs else 0, u["units"] if cs else 0, u["zh_new"], u["sb"],
                                      u["en_corr"] - u["en_corrupt"], u["b_err"], u["zh_new"] if cs else 0, u["sb"] if cs else 0], float)
    return t


def tab_stat(tab, dl, idx):
    a = sum(tab[dl[i]] for i in idx)
    return {"mer": a[0] / a[1], "cs_mer": a[2] / a[3], "zh_new": a[4], "sb": a[5], "en_net": a[6], "zh_new_cs": a[8], "sb_cs": a[9]}


def paired_tables(dtabs: dict, a1: str, a0: str, dl: list, draws: list) -> dict:
    """Paired a1 - a0 per seed (point, dialogue-bootstrap CI with shared draws, worst-dialogue contribution, leave-one-
    dialogue-out range) and the seed mean (pooled over the same draws)."""
    full = list(range(len(dl)))
    pr = {}
    for s in SEEDS:
        x, y = dtabs[(a1, s)], dtabs[(a0, s)]
        pt = {k: tab_stat(x, dl, full)[k] - tab_stat(y, dl, full)[k] for k in KEYS}
        bs = {k: [] for k in KEYS}
        for idx in draws:
            p_, q_ = tab_stat(x, dl, idx), tab_stat(y, dl, idx)
            for k in KEYS:
                bs[k].append(p_[k] - q_[k])
        contrib = {}
        for k, col in (("zh_new", 4), ("sb", 5), ("en_net", 6)):
            per_d = {d: float(x[d][col] - y[d][col]) for d in dl}
            tot = sum(per_d.values())
            loo = [tot - v for v in per_d.values()]
            contrib[k] = {"max_abs_dialogue_delta": max(abs(v) for v in per_d.values()),
                          "max_abs_share_of_total": (max(abs(v) for v in per_d.values()) / abs(tot)) if tot else None,
                          "leave_one_dialogue_out_range": [min(loo), max(loo)]}
        pr[s] = {k: {"point": float(pt[k]), "ci95": [float(np.quantile(bs[k], q)) for q in (0.025, 0.975)]} for k in KEYS}
        pr[s]["dialogue_contribution"] = contrib
    pooled = {k: [] for k in KEYS}
    for idx in draws:
        for k in KEYS:
            pooled[k].append(np.mean([tab_stat(dtabs[(a1, s)], dl, idx)[k] - tab_stat(dtabs[(a0, s)], dl, idx)[k] for s in SEEDS]))
    pr["seed_mean"] = {k: {"point": float(np.mean([pr[s][k]["point"] for s in SEEDS])),
                           "range": [float(min(pr[s][k]["point"] for s in SEEDS)), float(max(pr[s][k]["point"] for s in SEEDS))],
                           "ci95": [float(np.quantile(pooled[k], q)) for q in (0.025, 0.975)]} for k in KEYS}
    return {str(k): v for k, v in pr.items()}


def run_evaluate():
    from reft.arch_s0s1_train import check_freeze
    from reft.evaluate_supervised import dev_refs, dev_rows, metrics, p21, paired, score_system
    from reft.p2_1b_evaluate import failure, fates
    from reft.p2_1d_audit import utterance_events
    from reft.p2_2_evaluate import span_fates
    fz = check_freeze()
    if out("evaluation.json").exists():
        raise SystemExit("evaluation.json exists; write-once")
    C = p21()
    BS = ca()["bootstrap"]
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
             "zero_init_identity": man["history"][0]["zero_init_identity_beam5"], "epochs": {}}
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
                    tab = dialogue_table(f["_per_utt"])
                    rec["dialogues_improved"] = int(sum(v[0] < v[7] for v in tab.values()))
                    rec["dialogues_worsened"] = int(sum(v[0] > v[7] for v in tab.values()))
                    eptabs[(arm, s, ep)] = tab
                    if ep == sel:
                        dtabs[(arm, s)] = tab
                R["epochs"][ep][reg] = rec
        runs[f"{arm}_seed{s}"] = R
    # historical R2-GoldKL recomputed for parity with the P2-1b evaluation
    E1b = read_json(art("p2_1b/evaluation.json"))["runs"]
    repro = {}
    for s in SEEDS:
        mine, theirs = runs[f"R2_seed{s}"], E1b[f"KL_seed{s}"]
        e = str(mine["selected_epoch"])
        repro[f"R2_{s}"] = (mine["selected_epoch"] == theirs["selected_epoch"]
                            and all(mine["epochs"][int(e)]["beam5"]["failure"][k] == theirs["epochs"][e]["beam5"]["failure"][k]
                                    for k in ("zh_new", "sb_units", "worst_dialogue_delta_mixed"))
                            and mine["epochs"][int(e)]["beam5"]["vs_B0"]["net"] == theirs["epochs"][e]["beam5"]["vs_B0"]["net"]
                            and mine["epochs"][int(e)]["beam5"]["metrics"]["CS"]["mer"] == theirs["epochs"][e]["beam5"]["metrics"]["CS"]["mer"]
                            and mine["epochs"][int(e)]["beam5"]["metrics"]["overall"]["mer"] == theirs["epochs"][e]["beam5"]["metrics"]["overall"]["mer"])
    dl = sorted({m["dialogue_id"] for m in meta})
    rng = np.random.default_rng(BS["seed"])
    draws = [rng.choice(len(dl), len(dl)) for _ in range(BS["reps"])]
    em = {}
    for a1, a0 in (("M1", "R3"), ("M1", "R2"), ("R3", "R2")):
        em[f"{a1}-{a0}"] = {str(s): {str(ep): {k: tab_stat(eptabs[(a1, s, ep)], dl, range(len(dl)))[k] - tab_stat(eptabs[(a0, s, ep)], dl, range(len(dl)))[k]
                                               for k in KEYS} for ep in (1, 2, 3)} for s in SEEDS}
    write_json(out("evaluation.json"), {"schema": "reft_arch_s1_evaluation_v1", "utc": now(), **fz, "git": git_state(), "slurm": slurm(),
                                        "B0": {r: metrics(base[r]) for r in base}, "runs": runs, "r2_reproduces_p2_1b": repro,
                                        "paired_m1_minus_r3": paired_tables(dtabs, "M1", "R3", dl, draws),
                                        "paired_m1_minus_r2": paired_tables(dtabs, "M1", "R2", dl, draws),
                                        "paired_r3_minus_r2": paired_tables(dtabs, "R3", "R2", dl, draws),
                                        "epoch_matched": em, "n_dialogues": len(dl),
                                        "note": "dev150 exposed and used for selection; seeds share the 20 dialogues; descriptive uncertainty"})
    print("evaluation written", repro)


# ====================================================================== mechanism (exploratory)

def arch_selected() -> dict:
    res = {}
    for a in ("R3", "M1"):
        for s in SEEDS:
            rd = out("training", f"{a}_seed{s}")
            ep = read_json(rd / "selection.json")["selected"]["epoch"]
            res[f"{a}_{s}"] = {"arch": a, "seed": s, "epoch": ep, "adapter": rd / f"epoch{ep}" / "adapter.pt",
                               "decode": rd / f"epoch{ep}" / "dev_beam5.jsonl"}
    return res


MECH_SYSTEMS = ("B0", "KL42", "KL43", "KL44", "R3_42", "R3_43", "R3_44", "M1_42", "M1_43", "M1_44")


def run_mech_anchors():
    """Own-prefix anchors of R3 / M1 selected outputs (P2-1c T3 rule and helpers, as in P2-2); B0 / KL anchors from P2-1c."""
    from transformers import WhisperTokenizer
    from reft.arch_s0s1_train import check_freeze
    from reft.common import cfg
    from reft.evaluate_supervised import dev_refs
    from reft.p2_1c_continuation_audit import anchor_at, boundaries, hyp_after
    from reft.text import one, tok_bytes_fn
    check_freeze()
    if out("mech_anchors_private.json").exists():
        raise SystemExit("mech anchors exist; write-once")
    A = {c["key"]: c for c in read_json(art("p2_1c/anchors_private.json"))["cases"]}
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    refs = dev_refs(sorted(c["uid"] for c in A.values()))
    sel = arch_selected()
    outs = {n: {x["id"]: x for x in read_jsonl(v["decode"])} for n, v in sel.items()}
    priv, inf = {}, {}
    for key, c in A.items():
        uid, sw, reg = c["uid"], c["switch_unit"], c["en_region"]
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
    write_json(out("mech_anchors_inference.json"), {"utc": now(), "own_k": inf, "p2_1c_anchors_inference_sha256": sha(art("p2_1c/anchors_inference.json"))})
    print(json.dumps({k: {n: v["valid"] for n, v in d.items()} for k, d in priv.items()}))


def run_mech_gpu():
    """GPU, P2-1c firewall (reference-free). 13.1: pre-/post-projection contribution audit of the selected R2 / R3 / M1
    adapters on their own dev150 beam-5 outputs (teacher-forced). 13.2: P2-1c replay engine at B0 and own prefixes."""
    import torch
    from reft.arch_s0s1_adapters import make_arch_adapter
    from reft.arch_s0s1_train import check_freeze
    from reft.p2_1c_continuation_audit import Engine, gpu_firewall
    from reft.site import Site
    from reft.text import tok_bytes_fn
    from reft.train_supervised import encode
    fz = check_freeze()
    for f in ("mech_contrib.json", "mech_replay_private.jsonl", "mech_continuation_private.jsonl", "mech_candidates_private.jsonl"):
        if out(f).exists():
            raise SystemExit(f"{f} exists; write-once")
    sel = arch_selected()
    own = read_json(out("mech_anchors_inference.json"))["own_k"]
    MI = read_json(art("p2_1c/anchors_inference.json"))
    gpu_firewall()
    E = Engine()
    b = E.b
    for n, v in sel.items():
        torch.manual_seed(v["seed"])
        a = make_arch_adapter(v["arch"], b.model, v["seed"], b.device)
        a.load_state_dict(torch.load(v["adapter"]))
        a.module.eval()
        for p in a.module.parameters():
            p.requires_grad_(False)
        E.ads[n] = a
        E.outputs[n] = {x["id"]: x for x in read_jsonl(v["decode"])}
    # ---- 13.1 contribution audit
    tb = tok_bytes_fn(b.processor.tokenizer)
    dm = read_json(art("p2_0/dev_manifest.json"))["rows"]
    systems = {f"R2_{s}": ("KL" + str(s)) for s in SEEDS} | {n: n for n in sel}
    contrib = {}
    for name, eng in systems.items():
        mod = E.ads[eng].module
        acc = collections.defaultdict(list)
        iso = 0.0
        orth = float((mod.rot.weight @ mod.rot.weight.T - torch.eye(mod.rot.weight.shape[0], device=b.device)).abs().max())
        for r in dm:
            toks = E.outputs[eng][r["utterance_id"]]["tokens"]
            X = torch.tensor([r["prompt"] + toks[:-1]], device=b.device)
            _, enc = encode(b, [r["audio_path"]])
            with torch.no_grad(), Site(b.model, 31, record=True) as s_:
                b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False)
            h = (s_.q + s_.s)[0, 3:].float()
            uu = s_.s[0, 3:].float()
            c = contributions(mod, h, uu)
            langs = [token_lang(tb(t)) for t in toks]
            nh = h.norm(dim=-1)
            ne, neh, neu, nup = (c[k].norm(dim=-1) for k in ("e", "e_h", "e_u", "update"))
            iso = max(iso, float((nup - ne).abs().max()))
            cos_hu = torch.nn.functional.cosine_similarity(c["e_h"], c["e_u"], dim=-1) if name.startswith("M1") else torch.zeros_like(ne)
            cos_ue = torch.nn.functional.cosine_similarity(c["e_u"], c["e"], dim=-1) if name.startswith("M1") else torch.zeros_like(ne)
            for i, lg in enumerate(langs):
                for tag in ("ALL", lg):
                    acc[(tag, "e")].append(float(ne[i]))
                    acc[(tag, "e_h")].append(float(neh[i]))
                    acc[(tag, "e_u")].append(float(neu[i]))
                    acc[(tag, "update")].append(float(nup[i]))
                    acc[(tag, "update_over_h")].append(float(nup[i] / nh[i]))
                    acc[(tag, "W_h_h")].append(float(c["W_h_h"][i].norm()))
                    acc[(tag, "R_h")].append(float(c["R_h"][i].norm()))
                    acc[(tag, "u_norm")].append(float(uu[i].norm()))
                    if name.startswith("M1"):
                        acc[(tag, "cos_eh_eu")].append(float(cos_hu[i]))
                        acc[(tag, "cos_eu_e")].append(float(cos_ue[i]))
                        acc[(tag, "eu_share_sq")].append(float(neu[i] ** 2 / (neh[i] ** 2 + neu[i] ** 2 + 1e-30)))
                        acc[(tag, "eu_gt_eh")].append(float(neu[i] > neh[i]))
        summ = {}
        for (tag, k), v in acc.items():
            a = np.array(v)
            summ.setdefault(tag, {})[k] = {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "n": len(a)}
        contrib[name] = {"orth_err": orth, "bias_norm": float(mod.learned.bias.norm()),
                         "w_u_norm": float(mod.w_u.norm()) if hasattr(mod, "w_u") else None,
                         "isometry_max_abs_update_minus_e": iso, "by_lang": summ}
    write_json(out("mech_contrib.json"), {"utc": now(), **fz, "slurm": slurm(), "systems": contrib,
                                          "note": "own dev150 beam-5 outputs, teacher-forced; positions >= 3 tagged by the predicted token's script; descriptive, not causal"})
    # ---- 13.2 P2-1c replay (same conditions as P2-2)
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
    write_json(out("mech_integrity.json"), {"utc": now(), **fz, "slurm": slurm(), "hooks_clear": E.hooks_clear(), "generate_calls": E.n_gen,
                                            "tf_forwards": E.n_tf, "outputs_sha256": {f: sha(out(f)) for f in ("mech_contrib.json", "mech_replay_private.jsonl",
                                                                                                             "mech_continuation_private.jsonl", "mech_candidates_private.jsonl")}})
    print("mech gpu done", E.n_gen, E.n_tf)


def mech_analyze() -> dict:
    """CPU, reference-aware scoring of the replay (P2-1c scope and status rules), as in P2-2."""
    from transformers import WhisperTokenizer
    from reft.arch_s0s1_train import control_runs
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
    outs = {"B0": {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}}
    for s, c in control_runs().items():
        outs[f"KL{s}"] = {x["id"]: x for x in read_jsonl(c["dir"] / f"epoch{c['selected_epoch']}" / "dev_beam5.jsonl")}
    for n, v in arch_selected().items():
        outs[n] = {x["id"]: x for x in read_jsonl(v["decode"])}
    text = lambda ids: tok.decode([t for t in ids if t != eos], skip_special_tokens=True).strip()  # noqa: E731

    def ret(key, tokens, scope):
        o = one(refs[A[key]["uid"]], text(tokens))
        return float(np.mean([o["by"][r][0] == "match" for r in scope]))

    def scope(key, anchor):
        c = A[key]
        return c["b0_anchors"]["early_zh"]["scope_units"] if anchor == "early_zh" else c["span_units"]
    st = {}
    for x in read_jsonl(out("mech_continuation_private.jsonl")):
        toks = outs[x["prefix_source"]][A[x["case"]]["uid"]]["tokens"][:x["k"]] + x["beams"][0]["suffix"]
        st[(x["case"], x["anchor"], x["prefix_source"], x["model"])] = status_of(ret(x["case"], toks, scope(x["case"], x["anchor"])), R)
    REP = {(r["case"], r["anchor"], r["prefix_source"], r["model"]): r for r in read_jsonl(out("mech_replay_private.jsonl"))}
    cands = read_jsonl(out("mech_candidates_private.jsonl"))
    for x in cands:
        toks = outs["B0"][A[x["case"]]["uid"]]["tokens"][:x["k"]] + x["tokens"]
        x["status"] = status_of(ret(x["case"], toks, scope(x["case"], "post_en")), R)
    rows = []
    for key, c in A.items():
        for m in MECH_SYSTEMS[1:]:
            orig = status_of(ret(key, outs[m][c["uid"]]["tokens"], c["span_units"]), R)
            own_valid = c["own_anchors"][m]["valid"] if m.startswith("KL") else own_priv[key][m]["valid"]
            rS, r0 = REP.get((key, "post_en", "B0", m)), REP.get((key, "post_en", "B0", "B0"))
            q, mg = seq_preference([x for x in cands if x["case"] == key], m)
            rows.append({"case": key, "kind": c["kind"], "group": c["group"], "system": m, "arm": m[:2] if not m.startswith("KL") else "R2",
                         "orig": orig, "b0_prefix": st.get((key, "post_en", "B0", m)), "own_prefix": st.get((key, "own", m, m)) if own_valid else None,
                         "b0_model_at_own_prefix": st.get((key, "own", m, "B0")) if own_valid else None, "own_valid": own_valid,
                         "pre_en": st.get((key, "pre_en", "B0", m)), "early_zh": st.get((key, "early_zh", "B0", m)),
                         "local_delta": (rS["src_next"]["logp"] - r0["src_next"]["logp"]) if rS else None,
                         "local_rank": rS["src_next"]["rank"] if rS else None, "T5_omission_preferred": q, "T5_margin": mg})
    p1c = {(r["case"], r["system"]): r for r in read_json(art("p2_1c/analysis_r1.json"))["rows"]}
    cons = all(p1c[(r["case"], r["system"])]["B_b0_prefix_status"] == r["b0_prefix"] and p1c[(r["case"], r["system"])]["orig_status"] == r["orig"]
               for r in rows if r["arm"] == "R2" and A[r["case"]]["b0_anchors"]["post_en"]["valid"])
    return {"rows": rows, "r2_rows_reproduce_p2_1c": cons}


# ====================================================================== report

def selected_metrics(E: dict) -> dict:
    sel = {}
    for name, R_ in E["runs"].items():
        e = R_["epochs"][str(R_["selected_epoch"])]["beam5"]
        g = R_["epochs"][str(R_["selected_epoch"])]["greedy"]
        sel[name] = {"zh_new": e["failure"]["zh_new"], "sb": e["failure"]["sb_units"], "loops": e["metrics"]["overall"]["loop_utterances"],
                     "nonterm": e["metrics"]["overall"]["not_terminated"], "en_net": e["vs_B0"]["net"], "cs_mer": e["metrics"]["CS"]["mer"],
                     "mer": e["metrics"]["overall"]["mer"], "en_wer": e["metrics"]["overall"]["en_wer"], "pier": e["metrics"]["overall"]["pier"],
                     "zh_cer": e["metrics"]["overall"]["zh_cer"], "retention": e["vs_B0"]["mandarin_retention"],
                     "worst_dlg": e["failure"]["worst_dialogue_delta_mixed"], "epoch": R_["selected_epoch"],
                     "en_corr": e["vs_B0"]["substantive_en_corrections"], "en_corrupt": e["vs_B0"]["substantive_en_corruptions"],
                     "zh_repairs": e["vs_B0"]["zh_repairs"], "outside_damage": e["vs_B0"]["outside_target_damage"],
                     "zh_del_new": e["failure"]["zh_del_new"], "long_runs": e["failure"]["long_runs"], "near_switch": e["failure"]["near_switch"],
                     "far": e["failure"]["far"], "zh_new_no_en_repair": e["failure"]["zh_new_in_no_en_repair_utts"],
                     "improved_utts": e["vs_B0"]["improved_utterances"], "worsened_utts": e["vs_B0"]["worsened_utterances"],
                     "dialogues_improved": e["dialogues_improved"], "dialogues_worsened": e["dialogues_worsened"], "spans": e["switchback_spans"],
                     "en_fates": {k: e["english_fates"][k] for k in ("omission_to_correct", "translation_to_correct", "lexical_to_correct",
                                                                      "correct_to_wrong", "wrong_to_other_wrong")},
                     "greedy": {"cs_mer": g["metrics"]["CS"]["mer"], "mer": g["metrics"]["overall"]["mer"], "en_net": g["vs_B0"]["net"],
                                "zh_new": g["vs_B0"]["new_zh_errors"]}}
    return sel


def run_report():
    from reft.arch_s0s1_adapters import EXPECTED
    from reft.arch_s0s1_train import check_freeze
    fz = check_freeze()
    if out("report.json").exists():
        raise SystemExit("report.json exists; write-once")
    D = ca()["decision"]
    E = read_json(out("evaluation.json"))
    sel = selected_metrics(E)
    arm = {a: {s: sel[f"{a}_seed{s}"] for s in SEEDS} for a in ARMS}
    A = gain(arm["M1"], arm["R3"], D)
    B = guard(arm["M1"], arm["R2"], D)
    Cc = corrections(arm["M1"], arm["R2"], arm["R3"], D)
    R3o, M1o = improvement_over_r2(arm["R3"], arm["R2"], D), improvement_over_r2(arm["M1"], arm["R2"], D)
    Ds = {"M1": strong(arm["M1"], D), "R3": strong(arm["R3"], D), "R2": strong(arm["R2"], D)}
    ver = read_json(out("verify.json")) if out("verify.json").exists() else None
    new = {k: v for k, v in E["runs"].items() if not k.startswith("R2")}
    integ = {"verify_ok": bool(ver and ver["all_ok"]), "six_runs_three_epochs": len(new) == 6 and all(len(R_["history"]) == 4 for R_ in new.values()),
             "updates_96": all(R_["history"][-1]["updates_total"] == ca()["training"]["updates_per_run"] for R_ in new.values()),
             "zero_init_150": all(R_["zero_init_identity"] == 150 for R_ in E["runs"].values()),
             "weights_unchanged": all(R_["weights_unchanged"] for R_ in E["runs"].values()),
             "parameter_counts": all(R_["efficiency"]["trainable_parameters"] == EXPECTED[k[:2]] for k, R_ in new.items()),
             "r2_reproduces_p2_1b": all(E["r2_reproduces_p2_1b"].values()),
             "paired_mask_tokens_updates": all(all(h["diagnostics"]["paired_with_R2_KL"].values()) for R_ in new.values() for h in R_["history"][1:]),
             "m1_init_paired_with_r2": all(E["runs"][f"M1_seed{s}"]["history"][0]["m1_base_equal_R2_seed"] for s in SEEDS),
             "orthogonality": all(h["diagnostics"]["orth_err_max"] <= ca()["preflight"]["tolerances"]["orth_max_abs"] for R_ in new.values() for h in R_["history"][1:])}
    dec = decide(all(integ.values()), A, B, Cc, R3o, M1o)
    mech = mech_analyze() if out("mech_continuation_private.jsonl").exists() else None
    contrib = read_json(out("mech_contrib.json")) if out("mech_contrib.json").exists() else None
    rep = {"schema": "reft_arch_s1_report_v1", "utc": now(), **fz, "git": git_state(), "decision": dec, "integrity": integ, "selected": sel,
           "A_m1_vs_r3": A, "B_guard_vs_r2": B, "C_corrections": Cc, "D_strong_retention": Ds, "R3_over_R2": R3o, "M1_over_R2": M1o,
           "paired_m1_minus_r3": E["paired_m1_minus_r3"], "paired_m1_minus_r2": E["paired_m1_minus_r2"], "paired_r3_minus_r2": E["paired_r3_minus_r2"],
           "epoch_matched": E["epoch_matched"], "mechanism_replay": mech, "mechanism_contrib": contrib["systems"] if contrib else None,
           "rules_doc_sha256": sha(ROOT / "docs/reft/ARCH_S0S1_DECISION_RULES.md")}
    write_json(out("report.json"), rep)
    figures(E, rep)
    write_json(out("manifest.json"), {"utc": now(), **fz, "files": {p.name: sha(p) for p in sorted(out().glob("*.json*")) if p.name != "manifest.json"},
                                      "figures": {p.name: sha(p) for p in sorted(out("figures").glob("*.png"))}})
    print(json.dumps({k: rep[k] for k in ("decision", "integrity", "A_m1_vs_r3", "B_guard_vs_r2", "C_corrections", "D_strong_retention",
                                          "R3_over_R2", "M1_over_R2")}, indent=1, default=str))


def figures(E, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out("figures")
    fd.mkdir(parents=True, exist_ok=True)
    sel = rep["selected"]
    col = {"R2": "#1f77b4", "R3": "#ff7f0e", "M1": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for a in ARMS:
        xs = [sel[f"{a}_seed{s}"]["zh_new"] for s in SEEDS]
        ys = [sel[f"{a}_seed{s}"]["en_net"] for s in SEEDS]
        ax.scatter(xs, ys, c=col[a], label=f"{a}-GoldKL", s=40)
        for s, x, y in zip(SEEDS, xs, ys):
            ax.annotate(str(s), (x, y), fontsize=7)
    ax.set_xlabel("new Mandarin errors vs B0 (selected, beam-5)")
    ax.set_ylabel("English net corrections vs B0")
    ax.legend()
    ax.set_title("F1 correction-preservation frontier")
    fig.tight_layout()
    fig.savefig(fd / "F1_frontier.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (k, lab) in zip(axs, (("zh_new", "new Mandarin errors"), ("sb", "long switch-back dropped units"), ("en_net", "English net corrections"),
                                  ("cs_mer", "CS MER"))):
        for s in SEEDS:
            ax.plot([0, 1, 2], [sel[f"{a}_seed{s}"][k] for a in ARMS], "o-", label=str(s))
        ax.set_xticks([0, 1, 2], list(ARMS))
        ax.set_title(lab, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("F2 matched seeds, selected checkpoints (R2 = historical GoldKL)")
    fig.tight_layout()
    fig.savefig(fd / "F2_matched.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for a in ARMS:
        for s in SEEDS:
            h = E["runs"][f"{a}_seed{s}"]["history"][1:]
            ep = [x["epoch"] for x in h]
            axs[0].plot(ep, [x["train_ce"] for x in h], "o-", c=col[a], alpha=0.7, label=f"{a} s{s}" if s == 42 else None)
            axs[1].plot(ep, [x["train_kl"] for x in h], "o-", c=col[a], alpha=0.7)
            axs[2].plot(ep, [x["grad_norm_mean"] for x in h], "o-", c=col[a], alpha=0.7)
    for ax, t in zip(axs, ("train CE (token mean)", "train GoldKL (masked mean)", "grad norm (pre-clip, mean)")):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("epoch")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fd / "F3_training.png", dpi=130)
    plt.close(fig)
    if rep["mechanism_contrib"]:
        mc = rep["mechanism_contrib"]
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [n for n in mc if n.startswith("M1")]
        x = np.arange(len(names))
        for j, (tag, c) in enumerate((("ZH", "#d62728"), ("EN", "#9467bd"))):
            eh = [mc[n]["by_lang"].get(tag, {}).get("e_h", {}).get("mean", np.nan) for n in names]
            eu = [mc[n]["by_lang"].get(tag, {}).get("e_u", {}).get("mean", np.nan) for n in names]
            ax.bar(x + j * 0.4 - 0.3, eh, 0.18, color=c, alpha=0.5, label=f"|e_h| {tag}")
            ax.bar(x + j * 0.4 - 0.1, eu, 0.18, color=c, label=f"|e_u| {tag}")
        ax.set_xticks(x, names)
        ax.legend(fontsize=7)
        ax.set_title("F4 M1 pre-projection edit components (own dev outputs; descriptive)", fontsize=9)
        fig.tight_layout()
        fig.savefig(fd / "F4_m1_contrib.png", dpi=130)
        plt.close(fig)
    if rep["mechanism_replay"]:
        rows = rep["mechanism_replay"]["rows"]
        keys = sorted({r["case"] for r in rows})
        val = {"recovered": 2, "partial": 1, "dropped": 0, None: np.nan}
        M, lab = [], []
        for k in keys:
            for m in MECH_SYSTEMS[1:]:
                r = next(x for x in rows if x["case"] == k and x["system"] == m)
                M.append([val[r["orig"]], val[r["b0_prefix"]], val[r["own_prefix"]]])
                lab.append(f"{k} {m}")
        fig, ax = plt.subplots(figsize=(5, 14))
        ax.imshow(np.array(M, float), cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks([0, 1, 2], ["original", "B0 prefix", "own prefix"], fontsize=7)
        ax.set_yticks(range(len(lab)), lab, fontsize=4)
        ax.set_title("F5 span status, 10 fixed P2-1c cases (exploratory)", fontsize=8)
        fig.tight_layout()
        fig.savefig(fd / "F5_prefix_cases.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "evaluate", "mech-anchors", "mech-gpu", "report"])
    a = ap.parse_args(argv)
    {"select": run_select, "evaluate": run_evaluate, "mech-anchors": run_mech_anchors, "mech-gpu": run_mech_gpu, "report": run_report}[a.cmd]()


if __name__ == "__main__":
    main()
