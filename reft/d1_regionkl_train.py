#!/usr/bin/env python
"""D1 region-weighted GoldKL pilot on the unchanged R2-LoReFT (docs/reft/D1_REGIONKL_PROTOCOL.md,
docs/reft/D1_REGIONKL_DECISION_RULES.md, configs/reft/d1_regionkl_frozen.yaml).

  regions             (CPU) training-only switch-back region metadata -> d1_regionkl/regions_private.jsonl, region_static.json
  freeze              (CPU) write-once d1_regionkl/freeze.json + AUTHORIZATION.json; upstream chains and the region hash
  freeze-amend --doc  (CPU) write-once amendment record chained to the current head (documented repair only)
  preflight           (GPU) gates G1-G7 -> d1_regionkl/preflight_<job>.json; D1_READY_FOR_TRAINING or D1_BLOCKED_<gates>
  train --seed        (GPU) one write-once run d1_regionkl/training/D1_seed{s}
  verify              (GPU) every D1 epoch adapter reloads and re-decodes the first 20 dev utterances as recorded

Only the KL aggregation of P2-1b GoldKL changes: sum_t M_t w_t KL_t / sum_t M_t w_t per micro-batch, w_t = 2 on the
switch-back region, 1 elsewhere. CE, the teacher, the teacher-correct mask, the R2 adapter, the optimizer and the epoch
order are P2-1b's. With every weight = 1 the D1 path is the P2-1b GoldKL path (preflight G3: bitwise replay of the
historical KL_seed42 epoch-1 adapter).

D1_DRY_ROOT=<scratch dir> redirects every output for pre-freeze engineering dry runs (no freeze check; train refuses).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import cublas_consistent, git_state, now, sha, slurm, weights_hash  # noqa: E402

CONF = ROOT / "configs/reft/d1_regionkl_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/D1_REGIONKL_PROTOCOL.md", "docs/reft/D1_REGIONKL_DECISION_RULES.md", "configs/reft/d1_regionkl_frozen.yaml",
                  "reft/d1_regionkl_train.py", "reft/d1_regionkl_evaluate.py", "tests/reft/test_d1_regionkl.py",
                  "slurm/reft_d1_regionkl.sbatch", "slurm/reft_d1_regionkl_cpu.sbatch", "slurm/reft_d1_regionkl_submit.sh"]
REUSED_SOURCES = ["reft/adapters.py", "reft/site.py", "reft/auto_policy.py", "reft/baseline_auto.py", "reft/train_supervised.py",
                  "reft/p2_1b_train.py", "reft/p2_1b_evaluate.py", "reft/evaluate_supervised.py", "reft/p2_1d_audit.py",
                  "reft/p2_2_evaluate.py", "reft/arch_s0s1_train.py", "reft/arch_s0s1_adapters.py", "reft/arch_s0s1_evaluate.py",
                  "reft/text.py", "reft/common.py", "reft/provenance.py", "src/csasr/data/language_tags.py", "src/csasr/data/normalize.py",
                  "configs/reft/p0p1.yaml", "configs/reft/p2_1_frozen.yaml", "configs/reft/p2_1b_frozen.yaml"]
SEEDS = (42, 43, 44)
START = 3
DRY = os.environ.get("D1_DRY_ROOT")
REGION_FILES = ("regions_private.jsonl", "region_static.json")


def cd() -> dict:
    return yaml.safe_load(CONF.read_text())


def out(*p) -> Path:
    return Path(DRY, *p) if DRY else art("d1_regionkl", *p)


# ====================================================================== pure helpers (unit-tested on CPU)

def lang_labels(U: list[str], FILL: set) -> list:
    """Per canonical unit: 'EN' (substantive English), 'ZH' (substantive Mandarin) or None (filler, OTHER, UNKNOWN)."""
    from reft.text import tag_unit
    out_ = []
    for u in U:
        t = tag_unit(u)
        out_.append(t if t in ("EN", "ZH") and u not in FILL else None)
    return out_


def switchback_windows(labels: list, K: int) -> list[list[int]]:
    """One window per EN->ZH switch-back in the language sequence (labelled units only): the first K substantive ZH units
    of the Mandarin run, stopping at the next substantive EN unit. Unlabelled units are skipped (never counted)."""
    seq = [k for k, lab in enumerate(labels) if lab]
    wins = []
    for q in range(1, len(seq)):
        if labels[seq[q]] == "ZH" and labels[seq[q - 1]] == "EN":
            w = []
            for k in seq[q:]:
                if labels[k] != "ZH" or len(w) == K:
                    break
                w.append(k)
            wins.append(w)
    return wins


def classify_tokens(um: list[list[int]], windows: list[list[int]], labels: list) -> list[str]:
    """Per content token: 'region' (all touched units inside one window), 'outside', 'unmapped' (touches no unit), or
    'ambiguous_<reason>' (touches a window unit and a unit outside the windows): language_boundary (an EN unit),
    window_end (a ZH unit beyond the window), nonlexical (filler / OTHER / mixed-script unit), multi_window."""
    W = {k: i for i, w in enumerate(windows) for k in w}
    cls = []
    for us in um:
        s = set(us)
        if not s:
            cls.append("unmapped")
        elif s <= set(W):
            cls.append("region" if len({W[k] for k in s}) == 1 else "ambiguous_multi_window")
        elif s & set(W):
            rest = s - set(W)
            cls.append("ambiguous_language_boundary" if any(labels[k] == "EN" for k in rest)
                       else "ambiguous_window_end" if any(labels[k] == "ZH" for k in rest) else "ambiguous_nonlexical")
        else:
            cls.append("outside")
    return cls


def utterance_region(t_text: str, ids: list[int], tb, FILL: set, K: int) -> dict:
    """Training-reference region record of one utterance. ids = target_ids (content tokens + final EOS)."""
    from reft.text import token_unit_map, units
    content = ids[:-1]
    U = units(t_text)
    labels = lang_labels(U, FILL)
    wins = switchback_windows(labels, K)
    rec = {"n_target": len(ids), "n_units": len(U), "n_sub_en": labels.count("EN"), "n_sub_zh": labels.count("ZH"),
           "n_switchbacks": len(wins), "window_sizes": [len(w) for w in wins], "windows": wins}
    um = token_unit_map(t_text, content, tb)
    bytes_ok = b"".join(tb(i) for i in content) == t_text.encode("utf-8")
    if um is None or not bytes_ok:
        return {**rec, "mappable": False, "region_tokens": [], "ambiguous": [], "touching_tokens": 0, "first_sb_tokens": [],
                "units_without_region_token": sum(len(w) for w in wins), "class_counts": {}}
    cls = classify_tokens(um, wins, labels)
    win_units = {k for w in wins for k in w}
    reg = [t for t, c in enumerate(cls) if c == "region"]
    covered = {k for t in reg for k in um[t]}
    first = []
    for w in wins:
        t0 = next((t for t, us in enumerate(um) if w[0] in us), None)
        first.append({"t": t0, "unambiguous": t0 is not None and cls[t0] == "region"})
    return {**rec, "mappable": True, "region_tokens": reg, "ambiguous": [[t, c] for t, c in enumerate(cls) if c.startswith("ambiguous")],
            "touching_tokens": sum(1 for us in um if set(us) & win_units), "first_sb_tokens": first,
            "units_without_region_token": len(win_units - covered), "class_counts": dict(collections.Counter(cls))}


def weight_tensor(Y, row_ids: list[list[int]], row_regions: list[list[int]], w_region: float, start: int = START):
    """Float32 weights shaped like Y: w_region at decoder position start + t for every region token t (the position whose
    label is target_ids[t]); 1 elsewhere. Asserts the decoder shift and that EOS / unlabelled positions are never weighted."""
    import torch
    W = torch.ones(Y.shape, dtype=torch.float32)
    for i, (ids, toks) in enumerate(zip(row_ids, row_regions)):
        for t in toks:
            if not (0 <= t < len(ids) - 1) or int(Y[i, start + t]) != ids[t]:
                raise ValueError(f"region token {t} does not label decoder position {start + t}")
            W[i, start + t] = w_region
    return W


def region_kl(zt, zs, Y, W):
    """Region-weighted GoldKL: sum_t M_t w_t KL(p0_t || p_theta_t) / sum_t M_t w_t over the teacher-correct mask M (P2-1b
    teacher_mask), full-vocabulary float32 log-softmax, teacher detached. With W == 1 the arithmetic is P2-1b kl_masked's
    (same products, same reduction, division by a Python scalar). Returns (loss, diagnostics); diagnostics are detached."""
    import torch
    import torch.nn.functional as F
    from reft.p2_1b_train import teacher_mask
    m = teacher_mask(zt, Y)
    n = int(m.sum())
    W = W.to(zs.device)
    reg_all = int((W > 1).sum())
    if n == 0:
        return zs.sum() * 0.0, {"n_mask": 0, "den": 0.0, "n_region_mask": 0, "n_region_tokens": reg_all, "kl_unweighted": 0.0,
                                "region_part": 0.0, "other_part": 0.0, "kl_region_mean": None, "kl_other_mean": None, "kl_min": None}
    lp0 = F.log_softmax(zt.detach().float(), -1)[m]
    lps = F.log_softmax(zs.float(), -1)[m]
    kl = (lp0.exp() * (lp0 - lps)).sum(-1)
    w = W[m]
    den = float(w.sum())
    loss = (kl * w).sum() / den
    with torch.no_grad():
        r = w > 1
        nr = int(r.sum())
        kd = kl.detach()
        diag = {"n_mask": n, "den": den, "n_region_mask": nr, "n_region_tokens": reg_all, "kl_unweighted": float(kd.sum() / max(1, n)),
                "region_part": float((kd * w)[r].sum() / den), "other_part": float((kd * w)[~r].sum() / den),
                "kl_region_mean": float(kd[r].mean()) if nr else None, "kl_other_mean": float(kd[~r].mean()) if nr < n else None,
                "kl_min": float(kd.min())}
    return loss, diag


def effective_factors(n: int, k: int, w_region: float = 2.0) -> dict:
    """Per-token KL weight relative to GoldKL (1/n each) when k of n masked positions carry w_region."""
    den = n + (w_region - 1) * k
    return {"region": w_region * n / den, "other": n / den}


def batch_region_counts(order_rows: list[str], regions: dict, mb: int) -> list[int]:
    """Region tokens (before the teacher mask) per micro-batch of an epoch order (utterance ids in order)."""
    return [sum(len(regions.get(u, [])) for u in order_rows[j:j + mb]) for j in range(0, len(order_rows), mb)]


# ====================================================================== region metadata (CPU, training references only)

def load_regions() -> dict:
    return {r["utterance_id"]: r["region_tokens"] for r in read_jsonl(out("regions_private.jsonl"))}


def run_regions():
    from transformers import WhisperTokenizer
    from reft.common import cfg, fillers
    from reft.text import tok_bytes_fn
    from reft.train_supervised import c21
    for f in REGION_FILES:
        if out(f).exists():
            raise SystemExit(f"{f} exists; write-once")
    C, C1 = cd(), c21()
    assert sha(art("p2_0/train_manifest.json")) == C1["data"]["train_manifest_sha256"]
    t0 = time.time()
    tm = read_json(art("p2_0/train_manifest.json"))["rows"]
    tg = read_json(art("p1/targets.json"))["targets"]
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    FILL = fillers()
    K = C["region"]["max_units_per_switch"]
    recs, mismatch = [], []
    for r in tm:
        uid = r["utterance_id"]
        ids = r["target_ids"]
        if ids != tg[uid]["target_ids"] or ids[-1] != eos or eos in ids[:-1]:
            mismatch.append(uid)
        rec = utterance_region(tg[uid]["target_text"], ids, tb, FILL, K)
        recs.append({"utterance_id": uid, "dialogue_id": r["dialogue_id"], **rec})
    t_map = time.time() - t0
    write_jsonl(out("regions_private.jsonl"), recs)
    S = static_summary(recs, tm, C)
    S.update({"targets_mismatch": len(mismatch), "mapping_seconds": t_map, "K": K, "region_weight": C["objective"]["region_weight"],
              "regions_sha256": sha(out("regions_private.jsonl")), "train_manifest_sha256": sha(art("p2_0/train_manifest.json")),
              "targets_sha256": sha(art("p1/targets.json")), "utc": now(), "git": git_state()})
    write_json(out("region_static.json"), S)
    print(json.dumps({k: v for k, v in S.items() if k not in ("per_batch", "git")}, indent=1))


def static_summary(recs: list[dict], tm: list[dict], C: dict) -> dict:
    """Training-only counts that do not need the teacher (aggregate, no identifiers)."""
    mp = [r for r in recs if r["mappable"]]
    cls = collections.Counter()
    for r in mp:
        cls.update(r["class_counts"])
    amb = collections.Counter(c for r in mp for _, c in r["ambiguous"])
    regions = {r["utterance_id"]: r["region_tokens"] for r in recs}
    mb = 8
    per_batch = {}
    for s in SEEDS:
        for ep in (1, 2, 3):
            order = np.random.default_rng(s + ep).permutation(len(tm))
            cnt = batch_region_counts([tm[i]["utterance_id"] for i in order], regions, mb)
            per_batch[f"{s}_{ep}"] = {"min": int(min(cnt)), "median": float(np.median(cnt)), "max": int(max(cnt)),
                                      "zero_batches": int(sum(c == 0 for c in cnt)), "n_batches": len(cnt)}
    fsb = [f for r in mp for f in r["first_sb_tokens"]]
    return {"utterances": len(recs), "mappable_utterances": len(mp), "unmappable_utterances": len(recs) - len(mp),
            "unmappable_fraction": (len(recs) - len(mp)) / len(recs),
            "supervised_positions": sum(r["n_target"] for r in recs), "eos_positions": len(recs),
            "units": sum(r["n_units"] for r in recs), "substantive_en_units": sum(r["n_sub_en"] for r in recs),
            "substantive_zh_units": sum(r["n_sub_zh"] for r in recs), "switchback_opportunities": sum(r["n_switchbacks"] for r in recs),
            "window_units": sum(sum(r["window_sizes"]) for r in recs),
            "windows_shorter_than_K": sum(1 for r in recs for w in r["window_sizes"] if w < C["region"]["max_units_per_switch"]),
            "window_size_hist": dict(collections.Counter(str(w) for r in recs for w in r["window_sizes"])),
            "tokens_touching_window_units": sum(r["touching_tokens"] for r in mp),
            "region_tokens_unambiguous": sum(len(r["region_tokens"]) for r in mp),
            "ambiguous_tokens": sum(amb.values()), "ambiguous_by_reason": dict(amb),
            "window_units_without_unambiguous_token": sum(r["units_without_region_token"] for r in recs),
            "token_classes": dict(cls), "utterances_with_region": sum(1 for r in recs if r["region_tokens"]),
            "dialogues_with_region": len({r["dialogue_id"] for r in recs if r["region_tokens"]}),
            "dialogues_total": len({r["dialogue_id"] for r in recs}),
            "first_switchback_tokens": len(fsb), "first_switchback_tokens_unambiguous": sum(f["unambiguous"] for f in fsb),
            "eos_weighted": 0, "per_batch": per_batch}


# ====================================================================== freeze chain

def control_runs() -> dict:
    res = {}
    for s, run in cd()["control"]["runs"].items():
        rd = art(run)
        man = read_json(rd / "manifest.json")
        res[int(s)] = {"run": run, "dir": rd, "selected_epoch": read_json(rd / "selection.json")["selected"]["epoch"],
                       "adapter_sha": {h["epoch"]: h["adapter_sha256"] for h in man["history"][1:]}, "history": man["history"]}
    return res


def frozen_inputs() -> list[str]:
    L = ["p2_0/train_manifest.json", "p2_0/dev_manifest.json", "p2_0/data_audit.json", "p2_0/frozen_baseline_dev150_beam5.jsonl",
         "p2_0/frozen_baseline_dev150_greedy.jsonl", "p1/targets.json", "p2_1b/evaluation.json", "p2_1b/report.json", "p2_1b/verify.json",
         "p2_2/report.json", "arch_s0s1/report.json", "arch_s0s1/evaluation.json", "arch_s0s1/freeze.json"]
    for c in control_runs().values():
        L += [f"{c['run']}/selection.json", f"{c['run']}/manifest.json"]
        L += [f"{c['run']}/epoch{e}/adapter.pt" for e in (1, 2, 3)]
        L += [f"{c['run']}/epoch{e}/dev_{r}.jsonl" for e in (0, 1, 2, 3) for r in ("beam5", "greedy")]
    return L


def upstream_chains() -> dict:
    """P2-0 -> P2-1b (via P2-2), P2-1c (via S0/S1) and S0/S1 freeze chains, each verified by its own checker."""
    from reft.arch_s0s1_train import check_freeze as s0s1
    return {"arch_s0s1": s0s1()}


def freeze_path(rev: int) -> Path:
    return out("freeze.json") if rev == 0 else out(f"freeze_r{rev}.json")


def chain_head() -> int:
    rev = 0
    while freeze_path(rev + 1).exists():
        rev += 1
        if read_json(freeze_path(rev))["prev_freeze_sha256"] != sha(freeze_path(rev - 1)):
            raise SystemExit(f"D1 freeze chain broken at {freeze_path(rev).name}")
    return rev


def run_freeze():
    if DRY:
        raise SystemExit("no freeze in dry-run mode")
    p = out("freeze.json")
    if p.exists():
        raise SystemExit(f"{p} exists; write-once")
    g = git_state()
    if g["branch"] != "cs-asr-reft" or g["dirty_paths"]:
        raise SystemExit(f"branch/working tree not clean: {g}")
    if not all(out(f).exists() for f in REGION_FILES):
        raise SystemExit("region metadata missing (run `regions` first)")
    st = read_json(out("region_static.json"))
    if st["regions_sha256"] != sha(out("regions_private.jsonl")):
        raise SystemExit("region metadata hash mismatch")
    ctl = {s: {e: sha(c["dir"] / f"epoch{e}" / "adapter.pt") == h for e, h in c["adapter_sha"].items()} for s, c in control_runs().items()}
    if not all(v for x in ctl.values() for v in x.values()):
        raise SystemExit(f"R2-GoldKL control adapter hash mismatch: {ctl}")
    rec = {"schema": "reft_d1_regionkl_freeze_v1", "utc": now(), "git": g, "sources": {s: sha(ROOT / s) for s in FREEZE_SOURCES},
           "reused_sources": {s: sha(ROOT / s) for s in REUSED_SOURCES}, "inputs": {i: sha(art(i)) for i in frozen_inputs()},
           "region_metadata": {f: sha(out(f)) for f in REGION_FILES}, "controls_match_recorded_sha": {str(k): v for k, v in ctl.items()},
           "upstream": upstream_chains()}
    write_json(p, rec)
    write_json(out("AUTHORIZATION.json"), {"utc": now(), "stage": "D1 region-weighted GoldKL pilot (R2-LoReFT)",
                                           "authorized_by": "project owner (2026-10-15 prompt)", "runs": cd()["new_runs"],
                                           "max_new_training_runs": cd()["max_new_training_runs"],
                                           "conditions": "formal training only if the preflight is D1_READY_FOR_TRAINING; no D2, region-weight "
                                                         "or KL sweeps, R3/M1, rank/layer/adapter changes, on-policy, LoRA, E1+D2, P2-3, P3 or "
                                                         "independent test evaluation; historical controls are not retrained"})
    print(p)


def run_freeze_amend(doc: str):
    if DRY:
        raise SystemExit("no freeze in dry-run mode")
    k = chain_head() + 1
    prev = freeze_path(k - 1)
    rec0 = read_json(prev)
    if not (ROOT / doc).exists():
        raise SystemExit(f"{doc} missing")
    inputs = {i: sha(art(i)) for i in rec0["inputs"]}
    if inputs != rec0["inputs"] or {f: sha(out(f)) for f in REGION_FILES} != rec0["region_metadata"]:
        raise SystemExit(f"frozen inputs or region metadata changed since {prev.name}")
    srcs = list(rec0["sources"]) + ([doc] if doc not in rec0["sources"] else [])
    rec = {"schema": "reft_d1_regionkl_freeze_amendment_v1", "amendment": f"r{k}", "utc": now(), "git": git_state(),
           "prev_freeze": prev.name, "prev_freeze_sha256": sha(prev), "amendment_doc": doc,
           "changed_sources": sorted(x for x in srcs if rec0["sources"].get(x) != sha(ROOT / x)),
           "sources": {x: sha(ROOT / x) for x in srcs}, "reused_sources": {x: sha(ROOT / x) for x in rec0["reused_sources"]},
           "inputs": inputs, "region_metadata": rec0["region_metadata"], "controls_match_recorded_sha": rec0["controls_match_recorded_sha"],
           "upstream": upstream_chains(), "outputs_at_amendment": {q.name: sha(q) for q in sorted(out().glob("*.json"))}}
    print(write_json(freeze_path(k), rec))


def check_freeze() -> dict:
    if DRY:
        return {"dry_run": True}
    head = chain_head()
    p = freeze_path(head)
    rec = read_json(p)
    bad = [s for s, h in {**rec["sources"], **rec["reused_sources"]}.items() if sha(ROOT / s) != h]
    bad += [i for i, h in rec["inputs"].items() if sha(art(i)) != h]
    bad += [f for f, h in rec["region_metadata"].items() if sha(out(f)) != h]
    if bad:
        raise SystemExit(f"D1 frozen sources/inputs/region metadata changed: {bad}")
    if upstream_chains() != rec["upstream"]:
        raise SystemExit("an upstream freeze chain changed (P2-0/P2-1b/P2-1c/P2-2/S0S1)")
    return {"d1_freeze_sha256": sha(p), "d1_freeze_record": p.name, "d1_freeze_chain_len": head + 1,
            "regions_sha256": rec["region_metadata"]["regions_private.jsonl"], "arch_s0s1_freeze_sha256": rec["upstream"]["arch_s0s1"]["arch_freeze_sha256"]}


# ====================================================================== training step / epoch (GPU)

def d1_step(b, ad, rows, lam, regions: dict, w_region: float, grad=True):
    """reft.p2_1b_train.kl_step with the KL aggregation replaced by region_kl. regions = {} gives all-one weights."""
    import torch
    from reft.auto_policy import build_tf_batch, tf_loss
    from reft.train_supervised import encode, tf_forward
    eos = b.processor.tokenizer.eos_token_id
    ids = [r["target_ids"] for r in rows]
    X, Y = build_tf_batch([r["prompt"] for r in rows], ids, eos, eos)
    W = weight_tensor(Y, ids, [regions.get(r["utterance_id"], []) for r in rows], w_region)
    _, enc = encode(b, [r["audio_path"] for r in rows])
    X, Y = X.to(b.device), Y.to(b.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        zt = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
    torch.cuda.synchronize()
    t_teacher = time.time() - t0
    zs = tf_forward(b, ad, enc, X, grad=grad)
    ce, n = tf_loss(zs, Y)
    rkl, dg = region_kl(zt, zs, Y, W)
    return ce + lam * rkl, {"ce": float(ce), "rkl": float(rkl), "n_tok": n, "teacher_s": t_teacher, **dg}, (zt, zs, X, Y, W)


def d1_epoch(b, ad, opt, tm, tr, seed, ep, lam, regions, w_region, steps0, log):
    """One epoch of reft.p2_1b_train.train_loop (arm KL) step for step with d1_step, plus read-only diagnostics in `log`."""
    import torch
    from reft.arch_s0s1_adapters import named_trainable, orth_error
    mb, acc = tr["micro_batch"], tr["grad_accum"]
    steps = steps0
    order = np.random.default_rng(seed + ep).permutation(len(tm))
    batches = [[tm[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]
    torch.cuda.synchronize()
    t0 = time.time()
    sums = collections.Counter()
    pending, gn = 0, []
    params = named_trainable(ad)
    for k, rows in enumerate(batches):
        loss, st, _ = d1_step(b, ad, rows, lam, regions, w_region)
        sums["rkl_w"] += st["rkl"] * st["den"]
        sums["kl_u"] += st["kl_unweighted"] * st["n_mask"]
        for key in ("n_mask", "den", "n_region_mask", "n_region_tokens", "teacher_s"):
            sums[key] += st[key]
        sums["region_part_w"] += st["region_part"] * st["den"]
        sums["empty_mask_batches"] += st["n_mask"] == 0
        (loss / acc).backward()
        sums["ce"] += st["ce"] * st["n_tok"]
        sums["tot"] += float(loss) * st["n_tok"]
        sums["n_tok"] += st["n_tok"]
        log["micro"].append({"epoch": ep, "k": k, **{x: st[x] for x in ("ce", "rkl", "kl_unweighted", "n_tok", "n_mask", "den", "n_region_mask",
                                                                         "n_region_tokens", "region_part", "other_part", "kl_region_mean",
                                                                         "kl_other_mean", "kl_min")}})
        pending += 1
        if pending == acc or k == len(batches) - 1:
            with torch.no_grad():
                gpre = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in params.items()}
                before = {n: p.detach().clone() for n, p in params.items()}
            gn.append(float(torch.nn.utils.clip_grad_norm_(ad.parameters(), tr["clip_norm"])))
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
            with torch.no_grad():
                log["steps"].append({"epoch": ep, "step": steps, "grad_norm_total_preclip": gn[-1], "grad_norm_preclip": gpre,
                                     "lr": opt.param_groups[0]["lr"],
                                     "update_norm": {n: float((p.detach() - before[n]).norm()) for n, p in params.items()},
                                     "param_norm": {n: float(p.detach().norm()) for n, p in params.items()}, "orth_err": orth_error(ad)})
    torch.cuda.synchronize()
    return {"epoch": ep, "train_ce": sums["ce"] / sums["n_tok"], "train_total_tokmean": sums["tot"] / sums["n_tok"],
            "train_regionkl": sums["rkl_w"] / sums["den"] if sums["den"] else None,
            "train_kl": sums["kl_u"] / sums["n_mask"] if sums["n_mask"] else None,
            "mask_positions": sums["n_mask"], "supervised_tokens": sums["n_tok"], "mask_fraction": sums["n_mask"] / sums["n_tok"],
            "weight_sum": sums["den"], "region_tokens": sums["n_region_tokens"], "region_mask_positions": sums["n_region_mask"],
            "region_share_of_weighted_kl": sums["region_part_w"] / sums["rkl_w"] if sums["rkl_w"] else None,
            "empty_mask_batches": sums["empty_mask_batches"], "updates_total": steps, "grad_norm_mean": float(np.mean(gn)),
            "grad_norm_max": float(np.max(gn)), "train_seconds": time.time() - t0, "teacher_forward_seconds": sums["teacher_s"]}


def make_r2(b, seed):
    import torch
    from reft.arch_s0s1_adapters import make_arch_adapter
    torch.manual_seed(seed)
    return make_arch_adapter("R2", b.model, seed, b.device)


# ====================================================================== preflight (GPU)

def run_preflight():
    import torch
    from reft.adapters import site_hook_count
    from reft.arch_s0s1_adapters import init_record, named_trainable, orth_error, perturb
    from reft.arch_s0s1_train import Runtime, cached_vs_full, greedy_vs_tf, make_opt, spy_positions
    from reft.auto_policy import frozen_lid
    from reft.common import cfg, fillers
    from reft.p2_1b_train import kl_step, seed_config, teacher_mask
    from reft.text import tok_bytes_fn
    from reft.train_supervised import batch_loss, encode
    from csasr.models.whisper import batch_model_inputs
    fz = check_freeze()
    t_all = time.time()
    C = cd()
    P, T = C["preflight"], C["preflight"]["tolerances"]
    wr, lam, K = C["objective"]["region_weight"], C["objective"]["lambda_kl"], C["region"]["max_units_per_switch"]
    jid = str(slurm()["SLURM_JOB_ID"] or "local")
    pdir = out("preflight_files", jid)
    # training references for the G2 re-derivation are read before the GPU firewall is raised (training data only)
    tg = read_json(art("p1/targets.json"))["targets"]
    reg_recs = read_jsonl(out("regions_private.jsonl"))
    static = read_json(out("region_static.json"))
    regions = {r["utterance_id"]: r["region_tokens"] for r in reg_recs}
    rt = Runtime()
    b = rt.b
    dev = b.device
    torch.cuda.reset_peak_memory_stats()
    res = {"freeze": fz}
    ctl = control_runs()
    pb = rt.C1["frozen_baseline"]

    # ---- G1 provenance and data firewall
    audit = read_json(art("p2_0/data_audit.json"))
    try:
        sq = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        sq = f"unavailable: {e}"
    me = str(slurm()["SLURM_JOB_ID"])
    others = [x for x in sq.splitlines() if x.strip() and not x.startswith(me + " ") and ("reft" in x and "RUNNING" in x and "d1" not in x)]
    g = git_state()
    res["G1"] = {"branch": g["branch"] == "cs-asr-reft", "git_head": g["head"], "git_clean": not g["dirty_paths"],
                 "freeze_verified": DRY is None and "d1_freeze_sha256" in fz,
                 "controls_sha": all(sha(c["dir"] / f"epoch{e}" / "adapter.pt") == h for c in ctl.values() for e, h in c["adapter_sha"].items()),
                 "control_selection_and_evaluation_present": all((c["dir"] / "selection.json").exists() for c in ctl.values())
                 and art("p2_1b/evaluation.json").exists(),
                 "pinned_baseline": sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"],
                 "manifests": sha(art("p2_0/train_manifest.json")) == rt.C1["data"]["train_manifest_sha256"]
                 and sha(art("p2_0/dev_manifest.json")) == rt.C1["data"]["dev_manifest_sha256"],
                 "weights_expected": rt.w0 == rt.expected_w, "canonical_cuda": cublas_consistent(rt.libs),
                 "data_audit_valid_no_locked": bool(audit["valid"] and audit["checks"]["no_locked_dialogues"]),
                 "no_conflicting_running_reft_jobs": not others, "no_training_outputs_yet": not out("training").exists()}
    res["G1_squeue"] = sq

    # ---- G2 region alignment on the real training set
    tb = tok_bytes_fn(b.processor.tokenizer)
    FILL = fillers()
    rederived = {r["utterance_id"]: utterance_region(tg[r["utterance_id"]]["target_text"], r["target_ids"], tb, FILL, K)["region_tokens"]
                 for r in rt.tm}
    bad_tok = []
    for r in reg_recs:
        if not r["region_tokens"]:
            continue
        ids = next(x for x in rt.tm if x["utterance_id"] == r["utterance_id"])["target_ids"]
        for t in r["region_tokens"]:
            s = tb(ids[t]).decode("utf-8", errors="ignore")
            if t >= len(ids) - 1 or ids[t] == rt.eos or any(c.isascii() and c.isalnum() for c in s):
                bad_tok.append(t)
    sh = sorted(rt.tm, key=lambda r: r["utterance_id"])
    shift_rows = [r for r in sh if regions.get(r["utterance_id"])][:8]
    from reft.auto_policy import build_tf_batch
    Xs, Ys = build_tf_batch([r["prompt"] for r in shift_rows], [r["target_ids"] for r in shift_rows], rt.eos, rt.eos)
    Ws = weight_tensor(Ys, [r["target_ids"] for r in shift_rows], [regions[r["utterance_id"]] for r in shift_rows], wr)
    shift_ok = all(int(Ys[i, START + t]) == r["target_ids"][t] and (int(Xs[i, START + t]) == (r["target_ids"][t - 1] if t else r["prompt"][-1]))
                   for i, r in enumerate(shift_rows) for t in regions[r["utterance_id"]])
    res["G2"] = {"regions_hash_frozen": DRY is not None or sha(out("regions_private.jsonl")) == fz.get("regions_sha256"),
                 "static_hash_consistent": static["regions_sha256"] == sha(out("regions_private.jsonl")),
                 "rederived_equal": all(rederived[u] == regions[u] for u in regions) and set(rederived) == set(regions),
                 "region_tokens_lexical_han_only": not bad_tok, "n_bad_region_tokens": len(bad_tok),
                 "decoder_shift_ok": shift_ok, "weights_only_at_region_positions": int((Ws > 1).sum()) == sum(len(regions[r["utterance_id"]]) for r in shift_rows),
                 "eos_never_weighted": all(int(Ws[i, START + len(r["target_ids"]) - 1]) == 1 for i, r in enumerate(shift_rows)),
                 "prompt_positions_unweighted": bool((Ws[:, :START] == 1).all()),
                 "unmappable_fraction": static["unmappable_fraction"],
                 "unmappable_ok": static["unmappable_fraction"] <= C["region"]["integrity_block"]["max_unmappable_fraction"],
                 "targets_match_manifest": static["targets_mismatch"] == 0, "static": {k: v for k, v in static.items() if k not in ("git", "per_batch")}}

    # ---- coverage: teacher-correct region tokens over the whole training set (manifest order, micro-batches of 8)
    cov = collections.Counter()
    per_dlg = collections.Counter()
    per_batch = []
    fsb = [0, 0]
    t_cov = time.time()
    for j in range(0, len(rt.tm), 8):
        rows = rt.tm[j:j + 8]
        Xc, Yc, ec = rt.tf_batch(rows)
        with torch.no_grad():
            zc = b.model(encoder_outputs=(ec,), decoder_input_ids=Xc, use_cache=False).logits
        M = teacher_mask(zc, Yc).cpu()
        cov["supervised"] += int((Yc != -100).sum())
        cov["teacher_correct"] += int(M.sum())
        nb = 0
        for i, r in enumerate(rows):
            rr = next(x for x in reg_recs if x["utterance_id"] == r["utterance_id"])
            n = len(r["target_ids"])
            cov["eos_correct"] += int(M[i, START + n - 1])
            for t in rr["region_tokens"]:
                c_ = int(M[i, START + t])
                cov["region_tc"] += c_
                cov["region_all"] += 1
                per_dlg[r["dialogue_id"]] += c_
                nb += c_
            for t, _ in rr["ambiguous"]:
                cov["ambiguous_tc"] += int(M[i, START + t])
            for f in rr["first_sb_tokens"]:
                if f["t"] is not None and f["unambiguous"]:
                    fsb[0] += int(M[i, START + f["t"]])
                    fsb[1] += 1
        per_batch.append(nb)
    coverage = {"supervised_positions": cov["supervised"], "teacher_correct_positions": cov["teacher_correct"],
                "teacher_correct_fraction": cov["teacher_correct"] / cov["supervised"],
                "region_tokens_unambiguous": cov["region_all"], "region_tokens_teacher_correct": cov["region_tc"],
                "region_coverage": cov["region_tc"] / cov["region_all"] if cov["region_all"] else None,
                "region_share_of_teacher_correct": cov["region_tc"] / cov["teacher_correct"],
                "first_switchback_tokens_teacher_correct": fsb, "first_switchback_coverage": fsb[0] / fsb[1] if fsb[1] else None,
                "ambiguous_tokens_teacher_correct": cov["ambiguous_tc"], "eos_teacher_correct": [cov["eos_correct"], len(rt.tm)],
                "dialogues_with_teacher_correct_region": sum(1 for v in per_dlg.values() if v > 0),
                "dialogues_total": len({r["dialogue_id"] for r in rt.tm}),
                "top3_dialogue_share_of_region_tc": (sum(sorted(per_dlg.values(), reverse=True)[:3]) / cov["region_tc"]) if cov["region_tc"] else None,
                "per_microbatch_region_tc": {"min": int(min(per_batch)), "median": float(np.median(per_batch)), "max": int(max(per_batch)),
                                             "zero_batches": int(sum(x == 0 for x in per_batch)), "n_batches": len(per_batch)},
                "effective_factors_global": effective_factors(cov["teacher_correct"], cov["region_tc"], wr),
                "seconds": time.time() - t_cov,
                "note": "manifest-order micro-batches; training-time masks are recomputed per micro-batch and logged per run"}
    res["coverage"] = coverage

    # ---- G3 GoldKL equivalence (all weights 1)
    tiny = sh[:P["tf_rows"]]
    g3 = {}
    z0a, z0b = make_r2(b, 42), make_r2(b, 42)
    sd0a, sd0b = z0a.state_dict(), z0b.state_dict()
    g3["init_state_identical_incl_base"] = set(sd0a) == set(sd0b) and all(torch.equal(sd0a[k], sd0b[k]) for k in sd0a)
    for tag, (pa, pb_) in (("zero_init", (z0a, z0b)), ("perturbed", (make_r2(b, 42), make_r2(b, 42)))):
        if tag == "perturbed":
            perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
            perturb(pb_, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
        l1, s1, (zt1, zs1, X1, Y1) = kl_step(b, pa, tiny, lam)
        l1.backward()
        l2, s2, (zt2, zs2, X2, Y2, W2) = d1_step(b, pb_, tiny, lam, {}, wr)
        l2.backward()
        ga, gb = named_trainable(pa), named_trainable(pb_)
        g3[tag] = {"loss_bitwise": bool(torch.equal(l1.detach(), l2.detach())), "ce_equal": s1["ce"] == s2["ce"], "kl_equal": s1["kl"] == s2["rkl"],
                   "n_tok_equal": s1["n_tok"] == s2["n_tok"], "mask_equal": s1["n_mask"] == s2["n_mask"],
                   "mask_tensor_equal": bool(torch.equal(teacher_mask(zt1, Y1), teacher_mask(zt2, Y2))),
                   "teacher_logits_bitwise": bool(torch.equal(zt1, zt2)), "student_logits_bitwise": bool(torch.equal(zs1, zs2)),
                   "weights_all_one": bool((W2 == 1).all()), "den_equals_n_mask": s2["den"] == s2["n_mask"],
                   "grads_bitwise": all(torch.equal(ga[k].grad, gb[k].grad) for k in ga),
                   "grad_max_abs_diff": max(float((ga[k].grad - gb[k].grad).abs().max()) for k in ga), "kl": s1["kl"], "n_mask": s1["n_mask"],
                   "ce": s2["ce"]}
        del pa, pb_
    zi = make_r2(b, 42)
    lz, szi, (ztz, zsz, _, Yz, Wz) = d1_step(b, zi, sh[:8], lam, regions, wr)
    g3["zero_init_d1_with_regions"] = {"initial_kl": szi["rkl"], "initial_kl_ok": abs(szi["rkl"]) <= T["initial_kl_max"],
                                       "student_equals_teacher": bool(torch.equal(zsz.detach(), ztz)), "region_tokens": szi["n_region_tokens"]}
    lce, _, _, _, _ = batch_loss(b, make_r2(b, 42), tiny, grad=True)
    g3["ce_value_p2_1_path"] = float(lce)
    g3["ce_equals_p2_1_ce_path"] = float(lce) == g3["zero_init"]["ce"]
    # historical replay: seed-42 epoch 1 with every weight = 1 vs the P2-1b KL_seed42 epoch-1 adapter
    tr = seed_config(rt.C1, 42)["training"]
    a3 = make_r2(b, 42)
    ref1 = torch.load(ctl[42]["dir"] / "epoch1" / "adapter.pt")
    base_key = "rot.parametrizations.weight.0.base"
    base_eq = torch.equal(a3.state_dict()[base_key], ref1[base_key])
    opt = make_opt(a3, rt.C1, tr)
    lg = {"micro": [], "steps": []}
    h = d1_epoch(b, a3, opt, rt.tm, tr, 42, 1, lam, {}, wr, 0, lg)
    mine = a3.state_dict()
    hist1 = ctl[42]["history"][1]
    g3["replay_KL42_epoch1_uniform"] = {"init_base_equal": base_eq,
                                        "bitwise_equal": set(mine) == set(ref1) and all(torch.equal(mine[k], ref1[k]) for k in ref1),
                                        "max_abs_diff": max(float((mine[k] - ref1[k]).abs().max()) for k in ref1),
                                        "train_kl_equal": h["train_regionkl"] == hist1["train_kl"] and h["train_kl"] == hist1["train_kl"],
                                        "train_ce_equal": h["train_ce"] == hist1["train_ce"],
                                        "mask_positions_equal": h["mask_positions"] == hist1["mask_positions"],
                                        "supervised_tokens_equal": h["supervised_tokens"] == hist1["supervised_tokens"],
                                        "updates": h["updates_total"], "updates_equal": h["updates_total"] == hist1["updates_total"],
                                        "weight_sum_equals_mask": h["weight_sum"] == h["mask_positions"], "region_tokens": h["region_tokens"],
                                        "epoch_seconds": h["train_seconds"], "max_orth_err": max(s["orth_err"] for s in lg["steps"])}
    del a3, opt
    res["G3"] = g3

    # ---- G4 weighting correctness (perturbed adapter, training rows with region tokens)
    g4 = {}
    rows4 = [r for r in sh if regions.get(r["utterance_id"])][:P["tf_rows"]]
    pw, pu = make_r2(b, 42), make_r2(b, 42)
    for a_ in (pw, pu):
        perturb(a_, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
    lw, sw, (ztw, zsw, Xw, Yw, Ww) = d1_step(b, pw, rows4, lam, regions, wr)
    lw.backward()
    lu, su, _ = d1_step(b, pu, rows4, lam, {}, wr)
    lu.backward()
    gw, gu = named_trainable(pw), named_trainable(pu)
    with torch.no_grad():
        m = teacher_mask(ztw, Yw)
        wm = Ww.to(dev)[m]
        # (a) the aggregation D1 changes, on the same float32 per-token KL: explicit weighted mean
        lp0f = torch.log_softmax(ztw.float(), -1)[m]
        klf = (lp0f.exp() * (lp0f - torch.log_softmax(zsw.detach().float(), -1)[m])).sum(-1)
        manual32 = float((klf.double() * wm.double()).sum() / wm.double().sum())
        # (b) float64 reference KL (descriptive precision check; same comparison for unweighted GoldKL)
        lp0 = torch.log_softmax(ztw.double(), -1)[m]
        klm = (lp0.exp() * (lp0 - torch.log_softmax(zsw.detach().double(), -1)[m])).sum(-1)
        manual = float((klm * wm.double()).sum() / wm.double().sum())
        manual_u = float(klm.mean())
        excluded = (Yw != -100) & ~m
    nW = int(m.sum())
    kW = int((Ww.to(dev)[m] > 1).sum())
    g4 = {"n_mask": nW, "n_region_mask": kW, "region_tokens_in_batch": sw["n_region_tokens"],
          "region_active": kW > 0, "den_equals_n_plus_k": sw["den"] == nW + (wr - 1) * kW,
          "manual_weighted_kl_same_float32_kl": manual32, "manual_weighted_kl_float64": manual, "region_kl": sw["rkl"],
          "manual_matches": abs(manual32 - sw["rkl"]) <= T["aggregation_rtol"] * max(1e-12, abs(manual32)),
          "float64_rel_diff_weighted": abs(manual - sw["rkl"]) / max(1e-12, abs(manual)),
          "float64_rel_diff_unweighted_goldkl": abs(manual_u - su["kl_unweighted"]) / max(1e-12, abs(manual_u)),
          "float64_close": abs(manual - sw["rkl"]) <= T["float64_kl_rtol"] * max(1e-12, abs(manual)),
          "factors": effective_factors(nW, kW, wr), "region_factor_gt_1": effective_factors(nW, kW, wr)["region"] > 1,
          "teacher_incorrect_excluded": int(excluded.sum()),
          "kl_nonnegative": sw["kl_min"] >= -1e-6, "kl_min": sw["kl_min"],
          "finite": bool(torch.isfinite(lw)) and all(torch.isfinite(p.grad).all() for p in gw.values()),
          "weights_detached": not Ww.requires_grad, "teacher_detached": not ztw.requires_grad,
          "loss_differs_from_goldkl": float(lw) != float(lu), "d_loss": float(lw) - float(lu),
          "grads_differ_from_goldkl": any(not torch.equal(gw[k].grad, gu[k].grad) for k in gw),
          "grad_rel_diff": {k: float((gw[k].grad - gu[k].grad).norm() / gu[k].grad.norm()) for k in gw},
          "kl_region_mean": sw["kl_region_mean"], "kl_other_mean": sw["kl_other_mean"],
          "grads_reach_R_W_b": {k: float(v.grad.norm()) for k, v in gw.items()},
          "backbone_grads_none": all(p.grad is None for p in b.model.parameters())}
    # empty mask: labels all ignored except where the teacher is wrong -> differentiable zero
    Ye = torch.full_like(Yw, -100)
    le, de = region_kl(ztw, zsw, Ye, Ww)
    g4["empty_mask_zero"] = float(le) == 0.0 and de["n_mask"] == 0 and le.requires_grad
    # weights at teacher-incorrect labelled positions have no effect (the mask removes them before weighting)
    W5 = Ww.clone()
    W5[excluded.cpu()] = 5.0
    with torch.no_grad():
        g4["teacher_incorrect_weight_ignored"] = bool(torch.equal(region_kl(ztw, zsw.detach(), Yw, W5)[0], region_kl(ztw, zsw.detach(), Yw, Ww)[0]))
    del pw, pu
    res["G4"] = g4

    # ---- G5 model and decoding integrity
    g5 = {}
    tiny4 = sh[:P["tf_rows"]]
    X, Y, enc = rt.tf_batch(tiny4)
    q, u, zf = rt.site_record(enc, X)
    ad = make_r2(b, 42)
    g5["n_trainable"] = ad.n_trainable()
    g5["init"] = init_record(ad)
    g5["orth_err_init"] = orth_error(ad)
    g5["W_equals_R_b_zero"] = bool(torch.equal(ad.module.learned.weight, ad.module.rot.weight)) and bool((ad.module.learned.bias == 0).all())
    g5["zero_init_tf_logits_bitwise_frozen"] = bool(torch.equal(rt.logits(ad, enc, X), zf))
    pa = make_r2(b, 42)
    perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
    zp = rt.logits(pa, enc, X)
    with pa.disabled():
        zd = rt.logits(pa, enc, X)
    g5["disabled_reproduces_frozen"] = bool(torch.equal(zd, zf))
    g5["prefix_0_2_unchanged"] = bool(torch.equal(zp[:, :START], zf[:, :START]))
    g5["content_positions_changed"] = not bool(torch.equal(zp[:, START:], zf[:, START:]))
    ck = pdir / "perturbed_adapter.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pa.state_dict(), ck)
    rl = make_r2(b, 42)
    rl.load_state_dict(torch.load(ck))
    g5["reload_logits_bitwise"] = bool(torch.equal(rt.logits(rl, enc, X), zp))
    g5["backbone_trainable"] = sum(p.requires_grad for p in b.model.parameters())
    base5 = {x["id"]: x for x in read_jsonl(art(pb["beam5"]))}
    base1 = {x["id"]: x for x in read_jsonl(art(pb["greedy"]))}
    dev20 = rt.dm[:P["dev_rows"]]
    e5 = e1 = 0
    ts = []
    for row in dev20:
        t0 = time.time()
        a5, _ = rt.decode(row, 5, ad)
        ts.append(time.time() - t0)
        a1, _ = rt.decode(row, 1, ad)
        e5 += a5["tokens"] == base5[row["utterance_id"]]["tokens"]
        e1 += a1["tokens"] == base1[row["utterance_id"]]["tokens"]
    g5["zero_init_dev_beam5_equal"], g5["zero_init_dev_greedy_equal"] = [e5, len(dev20)], [e1, len(dev20)]
    g5["zero_init_beam5_s_per_utt"] = float(np.mean(ts))
    lid_eq, changed, tsp = 0, [], []
    for row in dev20:
        inp = batch_model_inputs(b, [row["audio_path"]])
        with pa.active(b.model):
            lid_eq += frozen_lid(b, inp["input_features"], pa)["token"] == row["frozen_lid_token"]
        t0 = time.time()
        d5, _ = rt.decode(row, 5, pa)
        tsp.append(time.time() - t0)
        changed.append(d5["tokens"] != base5[row["utterance_id"]]["tokens"])
    rx = dev20[changed.index(True)] if any(changed) else rt.dm[0]
    ry = next(r for r in dev20 if r is not rx)
    sg, sb_ = spy_positions(rt, pa, rx, 1), spy_positions(rt, pa, rx, 5)
    x1, _ = rt.decode(rx, 5, pa)
    rt.decode(ry, 5, pa)
    x2, _ = rt.decode(rx, 5, pa)
    x3, _ = rt.decode(rx, 5, pa)
    fz_ = rt.decode(rx, 5, None)[0]
    g5.update({"lid_unchanged_with_adapter": [lid_eq, len(dev20)], "perturbed_changed_dev_decodes": [sum(changed), len(dev20)],
               "perturbed_beam5_s_per_utt": float(np.mean(tsp)),
               "greedy_cache_positions": {k: v for k, v in sg.items() if k != "seq"}, "beam5_cache_positions": {k: v for k, v in sb_.items() if k != "seq"},
               "cached_greedy_vs_tf_mismatch": [greedy_vs_tf(rt, pa, row) for row in rt.dm[:P["cache_rows"]]],
               "xyx_no_state_leakage": x1["tokens"] == x2["tokens"], "deterministic_repeat": x2["tokens"] == x3["tokens"],
               "perturbed_changes_decode": x1["tokens"] != fz_["tokens"], "hooks_after": site_hook_count(b.model, 31)})
    del ad, rl

    # ---- G6 resources: CE vs GoldKL vs D1 micro-batch, region weighting overhead
    mb8 = rows4 + [r for r in sh if not regions.get(r["utterance_id"])][:8 - len(rows4)]
    tim = {}
    for arm in ("CE", "GoldKL", "D1"):
        a_ = make_r2(b, 42)
        tt, tc = [], []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            if arm == "CE":
                l_, _, _, _, _ = batch_loss(b, a_, mb8, grad=True)
                tc.append(0.0)
            elif arm == "GoldKL":
                l_, s_, _ = kl_step(b, a_, mb8, lam)
                tc.append(s_["teacher_s"])
            else:
                l_, s_, _ = d1_step(b, a_, mb8, lam, regions, wr)
                tc.append(s_["teacher_s"])
            l_.backward()
            torch.cuda.synchronize()
            tt.append(time.time() - t0)
            for p in a_.parameters():
                p.grad = None
        tim[arm] = {"micro_batch_seconds": tt, "teacher_seconds": tc}
        del a_
    t0 = time.time()
    for _ in range(100):
        weight_tensor(Ys, [r["target_ids"] for r in shift_rows], [regions[r["utterance_id"]] for r in shift_rows], wr)
    w_build = (time.time() - t0) / 100
    mean = {a: float(np.mean(v["micro_batch_seconds"][1:])) for a, v in tim.items()}
    n_mb = -(-len(rt.tm) // rt.C1["training"]["micro_batch"])
    dev_per_utt = g5["zero_init_beam5_s_per_utt"] * 1.65 + 0.15
    proj = 3 * n_mb * mean["D1"] + 4 * len(rt.dm) * dev_per_utt + 300
    g6 = {"timing": tim, "mean_micro_batch_seconds": mean, "d1_over_goldkl": mean["D1"] / mean["GoldKL"] - 1,
          "weight_tensor_build_seconds": w_build, "region_mapping_seconds_cpu": static["mapping_seconds"],
          "projected_job_seconds": proj, "dev_seconds_per_utt_beam5_plus_greedy": dev_per_utt,
          "adapter_file_bytes": ck.stat().st_size, "trainable_parameter_bytes_fp32": 4 * g5["n_trainable"],
          "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}

    # ---- G5 continued: float32 cached-vs-full semantic check (P2-1c amendment r1 tolerance); BF16 recorded descriptively
    fp_rows = rt.dm[:P["fp32_rows"]]
    cvf = {"bf16": [], "fp32": []}
    for row in fp_rows:
        _, e1_ = encode(b, [row["audio_path"]])
        cvf["bf16"].append(cached_vs_full(b, pa, e1_, row["prompt"] + base5[row["utterance_id"]]["tokens"][:-1], T["fp32_argmax_margin"]))
    w_before_cast = weights_hash(b.model)
    b.model.float()  # in-memory only, last step of this job
    for row in fp_rows:
        inp = batch_model_inputs(b, [row["audio_path"]])
        with torch.inference_mode():
            e1_ = b.model.model.encoder(inp["input_features"].float()).last_hidden_state
        cvf["fp32"].append(cached_vs_full(b, pa, e1_, row["prompt"] + base5[row["utterance_id"]]["tokens"][:-1], T["fp32_argmax_margin"]))
    g5["cached_vs_full"] = cvf
    g5["hooks_after_all"] = site_hook_count(b.model, 31)
    res["G5"], res["G6"] = g5, g6
    res["weights_unchanged_before_fp32_cast"] = w_before_cast == rt.w0
    g7 = {"freeze_verified": res["G1"]["freeze_verified"], "authorization_present": out("AUTHORIZATION.json").exists(),
          "git_clean": res["G1"]["git_clean"], "git_head_matches_freeze": DRY is None and read_json(freeze_path(chain_head()))["git"]["head"] == g["head"]}
    res["G7"] = g7

    cg = C["coverage_gate"]
    gates = {
        "G1_provenance": all(v for k, v in res["G1"].items() if k != "git_head"),
        "G2_region_alignment": all(res["G2"][k] for k in ("regions_hash_frozen", "static_hash_consistent", "rederived_equal",
                                                           "region_tokens_lexical_han_only", "decoder_shift_ok", "weights_only_at_region_positions",
                                                           "eos_never_weighted", "prompt_positions_unweighted", "unmappable_ok", "targets_match_manifest"))
        and (coverage["region_coverage"] or 0) >= cg["min_teacher_correct_region_coverage"]
        and coverage["dialogues_with_teacher_correct_region"] >= cg["min_dialogues_with_teacher_correct_region"],
        "G3_goldkl_equivalence": g3["init_state_identical_incl_base"] and g3["ce_equals_p2_1_ce_path"]
        and all(all(g3[t][k] for k in ("loss_bitwise", "ce_equal", "kl_equal", "n_tok_equal", "mask_equal", "mask_tensor_equal",
                                        "teacher_logits_bitwise", "student_logits_bitwise", "weights_all_one", "den_equals_n_mask", "grads_bitwise"))
                for t in ("zero_init", "perturbed"))
        and g3["zero_init_d1_with_regions"]["initial_kl_ok"] and g3["zero_init_d1_with_regions"]["student_equals_teacher"]
        and all(g3["replay_KL42_epoch1_uniform"][k] for k in ("init_base_equal", "bitwise_equal", "train_kl_equal", "train_ce_equal",
                                                               "mask_positions_equal", "supervised_tokens_equal", "updates_equal", "weight_sum_equals_mask")),
        "G4_weighting": all(g4[k] for k in ("region_active", "den_equals_n_plus_k", "manual_matches", "float64_close", "region_factor_gt_1", "kl_nonnegative", "finite",
                                            "weights_detached", "teacher_detached", "loss_differs_from_goldkl", "grads_differ_from_goldkl",
                                            "backbone_grads_none", "empty_mask_zero", "teacher_incorrect_weight_ignored"))
        and all(v > 0 for v in g4["grads_reach_R_W_b"].values()),
        "G5_model_decoding": g5["n_trainable"] == C["method"]["expected_trainable_parameters"] and g5["orth_err_init"] <= T["orth_max_abs"]
        and g5["W_equals_R_b_zero"] and g5["zero_init_tf_logits_bitwise_frozen"] and g5["disabled_reproduces_frozen"] and g5["prefix_0_2_unchanged"]
        and g5["content_positions_changed"] and g5["reload_logits_bitwise"] and g5["backbone_trainable"] == 0
        and g5["zero_init_dev_beam5_equal"][0] == len(dev20) and g5["zero_init_dev_greedy_equal"][0] == len(dev20)
        and g5["lid_unchanged_with_adapter"][0] == len(dev20) and g5["greedy_cache_positions"]["first"] == [0, 1, 2, 3]
        and g5["greedy_cache_positions"]["incremental_ok"] and g5["beam5_cache_positions"]["first"] == [0, 1, 2, 3]
        and g5["beam5_cache_positions"]["incremental_ok"] and all(x <= T["cached_greedy_vs_tf_max_mismatch"] for x in g5["cached_greedy_vs_tf_mismatch"])
        and g5["xyx_no_state_leakage"] and g5["deterministic_repeat"] and g5["perturbed_changes_decode"] and g5["hooks_after"] == 0
        and g5["hooks_after_all"] == 0 and res["weights_unchanged_before_fp32_cast"]
        and all(x["max_abs_logp_observed"] <= T["fp32_cached_vs_full_max_abs_logp"] and x["argmax_agree_where_sure"] and x["positions_ok"] and x["finite"]
                for x in cvf["fp32"]),
        "G6_resources": proj <= P["max_projected_job_seconds"] and g6["peak_reserved_bytes"] < 38e9,
        "G7_protocol_freeze": all(g7.values()),
    }
    verdict = "D1_READY_FOR_TRAINING" if all(gates.values()) else "D1_BLOCKED_" + "_".join(k.split("_")[0] for k, v in gates.items() if not v)
    rec = {"schema": "reft_d1_preflight_v1", "utc": now(), "dry_run": bool(DRY), "git": g, "slurm": slurm(), "env": rt.env, "cuda_libs": rt.libs,
           "results": res, "gates": gates, "verdict": verdict, "wall_seconds": time.time() - t_all}
    write_json(out(f"preflight_{jid}.json"), rec)
    write_json(out(f"region_coverage_{jid}.json"), {"utc": now(), "static": {k: v for k, v in static.items() if k != "git"}, "teacher": coverage})
    print(json.dumps({"verdict": verdict, "gates": gates, "G1": res["G1"], "G2": {k: v for k, v in res["G2"].items() if k != "static"},
                      "coverage": coverage, "G3": g3, "G4": g4, "G5": {k: v for k, v in g5.items() if k != "init"}, "G6": g6, "G7": g7},
                     indent=1, default=str)[:60000])


def newest_preflight():
    ps = sorted(out().glob("preflight_*.json"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
    return ps[-1] if ps else None


# ====================================================================== train (GPU)

def run_train(seed: int):
    import torch
    from reft.arch_s0s1_adapters import init_record, orth_error
    from reft.arch_s0s1_train import Runtime, make_opt
    from reft.auto_policy import frozen_lid
    from reft.p2_1b_train import seed_config
    if DRY:
        raise SystemExit("D1 training never runs in dry-run mode")
    C = cd()
    assert seed in C["seeds"]
    fz = check_freeze()
    if not out("AUTHORIZATION.json").exists():
        raise SystemExit("D1 not authorized")
    pf = newest_preflight()
    if pf is None or read_json(pf)["verdict"] != "D1_READY_FOR_TRAINING" or read_json(pf)["dry_run"]:
        raise SystemExit(f"D1 preflight not READY: {pf}")
    od = out("training", f"D1_seed{seed}")
    if od.exists():
        raise SystemExit(f"{od} exists; runs are write-once")
    regions = load_regions()
    reg_sha = sha(out("regions_private.jsonl"))
    if reg_sha != fz["regions_sha256"]:
        raise SystemExit("region metadata differs from the frozen hash")
    wr, lam = C["objective"]["region_weight"], C["objective"]["lambda_kl"]
    ctl = control_runs()[seed]
    torch.manual_seed(seed)                       # P2-1b order: seed -> backbone -> adapter
    rt = Runtime()
    b = rt.b
    tr = seed_config(rt.C1, seed)["training"]
    from reft.arch_s0s1_adapters import make_arch_adapter
    ad = make_arch_adapter("R2", b.model, seed, b.device)
    if ad.n_trainable() != C["method"]["expected_trainable_parameters"]:
        raise SystemExit(f"R2 has {ad.n_trainable()} trainable parameters")
    init = init_record(ad)
    ref1 = torch.load(ctl["dir"] / "epoch1" / "adapter.pt")
    base_key = "rot.parametrizations.weight.0.base"
    pairing = {"orth_err_init": orth_error(ad), "base_equal_GoldKL_seed": bool(torch.equal(ad.state_dict()[base_key], ref1[base_key]))}
    if not pairing["base_equal_GoldKL_seed"]:
        raise SystemExit("initial parametrization base differs from R2-GoldKL of the same seed (environment drift)")
    opt = make_opt(ad, rt.C1, tr)
    base = {x["id"]: x for x in read_jsonl(art(rt.C1["frozen_baseline"]["beam5"]))}
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()

    def dev_decode(ep):
        res = {"beam5": [], "greedy": []}
        t0 = time.time()
        for r in rt.dm:
            for reg, nb in (("beam5", 5), ("greedy", 1)):
                rec, inp = rt.decode(r, nb, ad)
                res[reg].append({"id": r["utterance_id"], **rec})
            with ad.active(b.model):
                assert frozen_lid(b, inp["input_features"], ad)["token"] == r["frozen_lid_token"], "LID changed under adapter"
        for reg, v in res.items():
            write_jsonl(od / f"epoch{ep}" / f"dev_{reg}.jsonl", v)
        return time.time() - t0, res

    sec, res0 = dev_decode(0)
    ident = sum(x["tokens"] == base[x["id"]]["tokens"] for x in res0["beam5"])
    if ident != len(rt.dm):
        raise SystemExit(f"zero-init adapter does not reproduce the frozen baseline ({ident}/{len(rt.dm)})")
    hist = [{"epoch": 0, "zero_init_identity_beam5": ident, "dev_decode_seconds": sec, "init_hashes": init, **pairing}]
    steps = 0
    for ep in range(1, tr["epochs"] + 1):
        lg = {"micro": [], "steps": []}
        h = d1_epoch(b, ad, opt, rt.tm, tr, seed, ep, lam, regions, wr, steps, lg)
        steps = h["updates_total"]
        write_jsonl(od / "diagnostics" / f"epoch{ep}_steps.jsonl", lg["steps"])
        write_jsonl(od / "diagnostics" / f"epoch{ep}_microbatches.jsonl", lg["micro"])
        (od / f"epoch{ep}").mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), od / f"epoch{ep}" / "adapter.pt")
        s_, _ = dev_decode(ep)
        hk = ctl["history"][ep]
        st = lg["steps"]
        mi = lg["micro"]
        diag = {"orth_err_max": max(x["orth_err"] for x in st),
                "grad_norm_preclip_mean": {n: float(np.mean([x["grad_norm_preclip"][n] for x in st])) for n in st[0]["grad_norm_preclip"]},
                "update_norm_mean": {n: float(np.mean([x["update_norm"][n] for x in st])) for n in st[0]["update_norm"]},
                "param_norm_end": st[-1]["param_norm"], "lr": sorted({x["lr"] for x in st}),
                "region_mask_per_microbatch": {"min": min(x["n_region_mask"] for x in mi), "median": float(np.median([x["n_region_mask"] for x in mi])),
                                               "max": max(x["n_region_mask"] for x in mi), "zero": sum(x["n_region_mask"] == 0 for x in mi)},
                "region_part_mean": float(np.mean([x["region_part"] for x in mi])), "other_part_mean": float(np.mean([x["other_part"] for x in mi])),
                "kl_region_mean": float(np.mean([x["kl_region_mean"] for x in mi if x["kl_region_mean"] is not None])),
                "kl_other_mean": float(np.mean([x["kl_other_mean"] for x in mi if x["kl_other_mean"] is not None])),
                "paired_with_GoldKL": {"mask_positions_equal": h["mask_positions"] == hk["mask_positions"],
                                       "supervised_tokens_equal": h["supervised_tokens"] == hk["supervised_tokens"],
                                       "updates_equal": h["updates_total"] == hk["updates_total"]},
                "goldkl_train_kl_historical": hk["train_kl"]}
        hist.append({**h, "diagnostics": diag, "dev_decode_seconds": s_, "adapter_sha256": sha(od / f"epoch{ep}" / "adapter.pt"),
                     "adapter_file_bytes": (od / f"epoch{ep}" / "adapter.pt").stat().st_size})
        print(json.dumps({k: v for k, v in hist[-1].items() if k != "diagnostics"} | {"orth_err_max": diag["orth_err_max"],
                                                                                      "paired": diag["paired_with_GoldKL"]}), flush=True)
    w1 = weights_hash(b.model)
    lat = {reg: float(np.mean([x["seconds"] for x in read_jsonl(od / f"epoch{tr['epochs']}" / f"dev_{reg}.jsonl")])) for reg in ("beam5", "greedy")}
    write_json(od / "manifest.json", {
        "schema": "reft_d1_run_v1", "utc": now(), **fz, "git": git_state(), "env": rt.env, "slurm": slurm(), "seed": seed,
        "objective": C["objective"], "region": C["region"], "regions_sha256": reg_sha, "training": tr, "lr": rt.C1["methods"]["R2"]["lr"],
        "history": hist, "weights_before": rt.w0, "weights_after": w1, "weights_unchanged": rt.w0 == w1 == rt.expected_w, "cuda_libs": rt.libs,
        "preflight": pf.name, "preflight_sha256": sha(pf),
        "efficiency": {"trainable_parameters": ad.n_trainable(), "adapter_bytes_fp32": 4 * ad.n_trainable(),
                       "adapter_file_bytes": hist[-1]["adapter_file_bytes"], "optimizer_updates": steps,
                       "supervised_tokens": sum(x.get("supervised_tokens", 0) for x in hist),
                       "train_seconds": sum(x.get("train_seconds", 0) for x in hist),
                       "teacher_forward_seconds": sum(x.get("teacher_forward_seconds", 0) for x in hist),
                       "dev_decode_seconds": sum(x.get("dev_decode_seconds", 0) for x in hist), "wall_seconds": time.time() - t_all,
                       "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "mean_decode_seconds": lat}})


# ====================================================================== verify (GPU)

def run_verify():
    import torch
    from reft.arch_s0s1_train import Runtime
    fz = check_freeze()
    if out("verify.json").exists():
        raise SystemExit("verify.json exists; write-once")
    rt = Runtime()
    res = {}
    for s in SEEDS:
        rd = out("training", f"D1_seed{s}")
        man = read_json(rd / "manifest.json")
        res[rd.name] = {}
        for h in man["history"][1:]:
            p = rd / f"epoch{h['epoch']}" / "adapter.pt"
            ad = make_r2(rt.b, s)
            ad.load_state_dict(torch.load(p))
            rec = {x["id"]: x for x in read_jsonl(rd / f"epoch{h['epoch']}" / "dev_beam5.jsonl")}
            eq = sum(rt.decode(r, 5, ad)[0]["tokens"] == rec[r["utterance_id"]]["tokens"] for r in rt.dm[:20])
            res[rd.name][h["epoch"]] = {"sha_ok": sha(p) == h["adapter_sha256"], "reload_equal": [eq, 20],
                                        "n_trainable_ok": ad.n_trainable() == cd()["method"]["expected_trainable_parameters"]}
    w1 = weights_hash(rt.b.model)
    ok = rt.w0 == w1 == rt.expected_w and all(v["sha_ok"] and v["reload_equal"][0] == 20 and v["n_trainable_ok"] for x in res.values() for v in x.values())
    write_json(out("verify.json"), {"schema": "reft_d1_verify_v1", "utc": now(), **fz, "slurm": slurm(), "cuda_libs": rt.libs,
                                    "results": res, "weights_unchanged": rt.w0 == w1, "all_ok": ok,
                                    "historical_R2_GoldKL": "verified in P2-1b (p2_1b/verify.json all_ok)"
                                    if read_json(art("p2_1b/verify.json"))["all_ok"] else "NOT OK"})
    print(json.dumps({"all_ok": ok, "results": res}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["regions", "freeze", "freeze-amend", "preflight", "train", "verify"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--doc")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        run_train(a.seed)
    elif a.cmd == "freeze-amend":
        run_freeze_amend(a.doc)
    else:
        {"regions": run_regions, "freeze": run_freeze, "preflight": run_preflight, "verify": run_verify}[a.cmd]()


if __name__ == "__main__":
    main()
