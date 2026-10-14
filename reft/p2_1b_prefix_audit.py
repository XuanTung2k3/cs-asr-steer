#!/usr/bin/env python
"""P2-1b targeted switch-back prefix diagnostic (GPU, no training, no new data; docs/reft/P2_1B_PROTOCOL.md section 7).

Fixed cases: p2_1b/prefix_audit/cases_private.json (frozen before training; P2-1D data only). Systems: B0 (no adapter),
R2-CE seed 42 epoch 2, and the selected CE / KL checkpoint of every seed. For each case: gold-prefix log p of the first
Mandarin token after the English span (and margins, span mean), the first divergence of the system's beam-5 output from
B0's (next-token log-probs under both models, EOS, script), and whole-candidate teacher-forced rescoring. Descriptive
A/B/C typing per docs/reft/P2_1B_DECISION_RULES.md. Not acoustic causality.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, firewall, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import now, sha, slurm  # noqa: E402


def script(tok_id, tb, eos):
    from reft.text import is_han
    if tok_id == eos:
        return "eos"
    s = tb(tok_id).decode("utf-8", "ignore")
    if any(is_han(c) for c in s) or (tb(tok_id) and all(x >= 0x80 for x in tb(tok_id))):
        return "han"
    if any(c.isascii() and c.isalpha() for c in s):
        return "latin"
    return "other"


def classify(case_kind, g_b0, g_s, div, drop_nats):
    if case_kind != "damaged":
        return None
    if g_s["logp"] <= g_b0["logp"] - drop_nats or (g_b0["argmax_is_gold"] and not g_s["argmax_is_gold"]):
        return "A"
    if div and div["sys_next_script"] in ("eos", "latin") and div["b0_next_script"] == "han":
        return "B"
    return "C"


def main():
    import torch
    import torch.nn.functional as F
    from reft.adapters import make_adapter
    from reft.p2_1b_evaluate import cb, out, run_dirs
    from reft.p2_1b_train import canonical_env_or_exit, check_freeze, seed_config
    from reft.text import tok_bytes_fn
    from reft.train_supervised import GPU_FORBIDDEN, c21, encode, load_backbone
    fz = check_freeze()
    C1, B = c21(), cb()
    cases = read_json(out("prefix_audit/cases_private.json"))
    allc = cases["damaged"] + cases["safe"]
    dirs, reuse = run_dirs()
    systems = {"CE42_e2_P2-1": (art("p2_1/R2/epoch2/adapter.pt"), art("p2_1/R2/epoch2/dev_beam5.jsonl"))}
    for (arm, s), rd in dirs.items():
        ep = read_json(rd / "selection.json")["selected"]["epoch"]
        systems[f"{arm}{s}_e{ep}"] = (rd / f"epoch{ep}" / "adapter.pt", rd / f"epoch{ep}" / "dev_beam5.jsonl")
    firewall(GPU_FORBIDDEN)
    b, env = load_backbone()
    libs = canonical_env_or_exit()
    tb = tok_bytes_fn(b.processor.tokenizer)
    eos = b.processor.tokenizer.eos_token_id
    base = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}
    t0 = time.time()
    enc_cache = {}
    paths = {r["utterance_id"]: r["audio_path"] for r in read_json(art("p2_0/dev_manifest.json"))["rows"]}

    def enc_of(uid):
        if uid not in enc_cache:
            enc_cache[uid] = encode(b, [paths[uid]])[1]
        return enc_cache[uid]

    def logits(ad, uid, X):
        import contextlib
        ctx = ad.active(b.model) if ad is not None else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            return b.model(encoder_outputs=(enc_of(uid),), decoder_input_ids=torch.tensor([X], device=b.device), use_cache=False).logits[0].float()

    def gold(ad, c):
        X = c["prompt"] + c["target_ids"][:-1]
        lp = F.log_softmax(logits(ad, c["id"], X), -1)
        t = 3 + c["sb_token"]
        g = c["target_ids"][c["sb_token"]]
        row = lp[t]
        other = row.clone()
        other[g] = -float("inf")
        span = [float(lp[3 + k, c["target_ids"][k]]) for k in c["span_tokens"]]
        return {"logp": float(row[g]), "margin_best_other": float(row[g] - other.max()), "margin_vs_eos": float(row[g] - row[eos]),
                "argmax_is_gold": int(row.argmax()) == g, "span_mean_logp": float(np.mean(span)) if span else None}

    def seq_score(ad, uid, prompt, toks):
        lp = F.log_softmax(logits(ad, uid, prompt + toks[:-1]), -1)
        s = float(sum(lp[3 + t, y] for t, y in enumerate(toks)))
        return {"sum": s, "per_token": s / max(1, len(toks))}

    ads = {}
    for name, (ap, _) in systems.items():
        a = make_adapter("R2", b.model, seed_config(C1, 42), b.device)
        a.load_state_dict(torch.load(ap))
        ads[name] = a
    outs = {name: {x["id"]: x for x in read_jsonl(dp)} for name, (_, dp) in systems.items()}
    rows = []
    for c in allc:
        g0 = gold(None, c)
        bt = base[c["id"]]["tokens"]
        for name, a in ads.items():
            st = outs[name][c["id"]]["tokens"]
            d = next((i for i in range(min(len(bt), len(st))) if bt[i] != st[i]), None)
            if d is None and len(bt) != len(st):
                d = min(len(bt), len(st))
            div = None
            if d is not None and d < len(bt) and d < len(st):
                P = c["prompt"] + bt[:d]
                l0 = F.log_softmax(logits(None, c["id"], P), -1)[-1]
                ls = F.log_softmax(logits(a, c["id"], P), -1)[-1]
                div = {"pos_frac": d / max(1, len(bt)), "b0_next_script": script(bt[d], tb, eos), "sys_next_script": script(st[d], tb, eos),
                       "b0_model": {"b0_next": float(l0[bt[d]]), "sys_next": float(l0[st[d]]), "eos": float(l0[eos])},
                       "sys_model": {"b0_next": float(ls[bt[d]]), "sys_next": float(ls[st[d]]), "eos": float(ls[eos])}}
            gs = gold(a, c)
            rs = {f"{who}_out_under_{mdl}": seq_score(m_, c["id"], c["prompt"], toks)
                  for who, toks in (("b0", bt), ("sys", st)) for mdl, m_ in (("b0", None), ("sys", a))}
            rows.append({"case": c["id"], "kind": c["kind"], "system": name, "gold_b0": g0, "gold_sys": gs,
                         "delta_gold_logp": gs["logp"] - g0["logp"], "identical_output": d is None, "divergence": div, "rescoring": rs,
                         "type": classify(c["kind"], g0, gs, div, B["prefix_audit"]["type_a_drop_nats"])})
    write_jsonl(out("prefix_audit/results_private.jsonl"), rows)
    summ = {"n_cases": {"damaged": len(cases["damaged"]), "safe": len(cases["safe"]), "damaged_unmappable": cases["damaged_unmappable"]},
            "by_system": {}, "dominance_fraction": B["prefix_audit"]["dominance_fraction"]}
    for name in ads:
        o = {}
        for kind in ("damaged", "safe"):
            rr = [r for r in rows if r["system"] == name and r["kind"] == kind]
            types = [r["type"] for r in rr if r["type"]]
            div = [r["divergence"] for r in rr if r["divergence"]]
            o[kind] = {"n": len(rr), "mean_delta_gold_logp_vs_b0": float(np.mean([r["delta_gold_logp"] for r in rr])) if rr else None,
                       "gold_argmax_lost": sum(r["gold_b0"]["argmax_is_gold"] and not r["gold_sys"]["argmax_is_gold"] for r in rr),
                       "mean_delta_span_logp": float(np.mean([r["gold_sys"]["span_mean_logp"] - r["gold_b0"]["span_mean_logp"] for r in rr
                                                              if r["gold_sys"]["span_mean_logp"] is not None])) if rr else None,
                       "output_identical_to_b0": sum(r["identical_output"] for r in rr),
                       "divergence_sys_next_script": dict(__import__("collections").Counter(x["sys_next_script"] for x in div)),
                       "sys_prefers_own_output_under_sys_model": sum(r["rescoring"]["sys_out_under_sys"]["sum"] > r["rescoring"]["b0_out_under_sys"]["sum"]
                                                                     for r in rr if not r["identical_output"]),
                       "b0_prefers_b0_output": sum(r["rescoring"]["b0_out_under_b0"]["sum"] > r["rescoring"]["sys_out_under_b0"]["sum"]
                                                   for r in rr if not r["identical_output"])}
            if kind == "damaged":
                cnt = {t: types.count(t) for t in "ABC"}
                o[kind]["types"] = cnt
                o[kind]["dominant"] = next((t for t in "ABC" if types and cnt[t] / len(types) >= B["prefix_audit"]["dominance_fraction"]), None)
        summ["by_system"][name] = o
    summ.update({"utc": now(), **fz, "slurm": slurm(), "cuda_libs": libs, "seconds": time.time() - t0,
                 "results_sha256": sha(out("prefix_audit/results_private.jsonl"))})
    write_json(out("prefix_audit/summary.json"), summ)
    print(json.dumps(summ, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
