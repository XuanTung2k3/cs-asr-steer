#!/usr/bin/env python
"""P2-1b paired CE vs KL-preservation training for R2 LoReFT (docs/reft/P2_1B_PROTOCOL.md).

  freeze              -- write-once p2_1b/freeze.json (sources + pinned inputs; independent of the P2-0 freeze chain)
  smoke               -- GPU preflight (G1-G7): provenance, CE-equivalence (seed-42 epoch 1 vs P2-1 bitwise), KL objective
                         checks, training-set mask coverage, decoding parity, timing/VRAM -> p2_1b/preflight.json
  train --arm --seed  -- GPU: one write-once run p2_1b/training/{CE,KL}_seed{s}
  verify              -- GPU: every new run's epoch adapters reload and re-decode the first 20 dev utterances as recorded

The CE arm calls reft.train_supervised.batch_loss unchanged. The KL arm computes the identical CE plus
lambda * KL(p0 || p_theta) on the teacher-correct mask with a detached frozen-teacher forward (adapter absent).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, firewall, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import cublas_consistent, git_state, loaded_cuda_libs, now, sha, slurm, weights_hash  # noqa: E402

CONF = ROOT / "configs/reft/p2_1b_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/P2_1B_PROTOCOL.md", "docs/reft/P2_1B_DECISION_RULES.md", "configs/reft/p2_1b_frozen.yaml",
                  "reft/p2_1b_train.py", "reft/p2_1b_evaluate.py", "reft/p2_1b_prefix_audit.py", "tests/reft/test_p2_1b.py",
                  "slurm/reft_p2_1b.sbatch", "slurm/reft_p2_1b_cpu.sbatch", "slurm/reft_p2_1b_submit.sh"]
FROZEN_INPUTS = ["p2_0/train_manifest.json", "p2_0/dev_manifest.json", "p2_0/frozen_baseline_dev150_beam5.jsonl",
                 "p2_0/frozen_baseline_dev150_greedy.jsonl", "p2_1/R2/manifest.json", "p2_1/R2/selection.json",
                 "p2_1/R2/epoch1/adapter.pt", "p2_1/R2/epoch2/adapter.pt", "p2_1/R2/epoch3/adapter.pt",
                 *[f"p2_1/R2/epoch{e}/dev_{r}.jsonl" for e in range(4) for r in ("beam5", "greedy")],
                 "p2_1/report.json", "p2_1/verify.json", "p2_1d/analysis.json", "p2_1d/events_private.jsonl",
                 "p2_1d/utterance_summary_private.jsonl", "p1/targets.json", "p2_1b/prefix_audit/cases_private.json",
                 "p2_1b/AUTHORIZATION.json"]
ARMS = ("CE", "KL")


def cb() -> dict:
    return yaml.safe_load(CONF.read_text())


def out(*p) -> Path:
    return art("p2_1b", *p)


# ====================================================================== pure helpers (unit-tested on CPU)

def teacher_mask(zt, Y):
    """Training-only teacher-forced correctness: labelled position whose frozen-teacher argmax equals the target."""
    return (Y != -100) & (zt.argmax(-1) == Y.clamp(min=0))


def kl_masked(zt, zs, Y):
    """Mean over masked positions of KL(p0 || p_theta), float32 log-softmax over the full vocabulary. Returns (kl, count)."""
    import torch
    import torch.nn.functional as F
    m = teacher_mask(zt, Y)
    n = int(m.sum())
    if n == 0:
        return zs.sum() * 0.0, 0
    lp0 = F.log_softmax(zt.detach().float(), -1)[m]
    lps = F.log_softmax(zs.float(), -1)[m]
    kl = (lp0.exp() * (lp0 - lps)).sum(-1)
    return kl.sum() / max(1, n), n


def seed_config(C1: dict, seed: int) -> dict:
    C = copy.deepcopy(C1)
    C["methods"]["R2"]["init_seed"] = seed
    C["training"]["seed"] = seed
    return C


# ====================================================================== freeze

def run_freeze():
    p = out("freeze.json")
    rec = {"schema": "reft_p2_1b_freeze_v1", "utc": now(), "git": git_state(),
           "sources": {s: sha(ROOT / s) for s in FREEZE_SOURCES}, "inputs": {i: sha(art(i)) for i in FROZEN_INPUTS},
           "p2_0_chain_head": None}
    from reft.train_supervised import check_freeze as p20
    rec["p2_0_chain_head"] = p20()["freeze_sha256"]
    print(write_json(p, rec))


def check_freeze() -> dict:
    p = out("freeze.json")
    rec = read_json(p)
    bad = [s for s, h in rec["sources"].items() if sha(ROOT / s) != h] + [i for i, h in rec["inputs"].items() if sha(art(i)) != h]
    if bad:
        raise SystemExit(f"P2-1b frozen sources/inputs changed: {bad}")
    from reft.train_supervised import check_freeze as p20
    if p20()["freeze_sha256"] != rec["p2_0_chain_head"]:
        raise SystemExit("P2-0 freeze chain head changed")
    return {"p2_1b_freeze_sha256": sha(p), "p2_0_chain_head": rec["p2_0_chain_head"]}


# ====================================================================== shared GPU pieces

def kl_step(b, ad, rows, lam, grad=True):
    """KL arm: identical CE to train_supervised.batch_loss, plus lambda * masked KL against the detached frozen teacher."""
    import torch
    from reft.auto_policy import build_tf_batch, tf_loss
    from reft.train_supervised import encode, tf_forward
    eos = b.processor.tokenizer.eos_token_id
    X, Y = build_tf_batch([r["prompt"] for r in rows], [r["target_ids"] for r in rows], eos, eos)
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
    kl, m = kl_masked(zt, zs, Y)
    return ce + lam * kl, {"ce": float(ce), "kl": float(kl), "n_tok": n, "n_mask": m, "teacher_s": t_teacher}, (zt, zs, X, Y)


def ce_step(b, ad, rows):
    from reft.train_supervised import batch_loss
    loss, n, _, _, _ = batch_loss(b, ad, rows, grad=True)
    return loss, {"ce": float(loss), "kl": None, "n_tok": n, "n_mask": None, "teacher_s": 0.0}


def train_loop(b, ad, opt, tm, tr, seed, arm, lam, epochs, on_epoch=None):
    import numpy as np
    import torch
    mb, acc = tr["micro_batch"], tr["grad_accum"]
    hist = []
    steps = 0
    for ep in range(1, epochs + 1):
        order = np.random.default_rng(seed + ep).permutation(len(tm))
        batches = [[tm[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]
        torch.cuda.synchronize()
        t0 = time.time()
        sums = {"ce": 0.0, "kl": 0.0, "tot": 0.0, "n_tok": 0, "n_mask": 0, "teacher_s": 0.0, "empty_mask_batches": 0}
        pending, gn = 0, []
        for k, rows in enumerate(batches):
            if arm == "CE":
                loss, st = ce_step(b, ad, rows)
            else:
                loss, st, _ = kl_step(b, ad, rows, lam)
                sums["kl"] += st["kl"] * st["n_mask"]
                sums["n_mask"] += st["n_mask"]
                sums["teacher_s"] += st["teacher_s"]
                sums["empty_mask_batches"] += st["n_mask"] == 0
            (loss / acc).backward()
            sums["ce"] += st["ce"] * st["n_tok"]
            sums["tot"] += float(loss) * st["n_tok"]
            sums["n_tok"] += st["n_tok"]
            pending += 1
            if pending == acc or k == len(batches) - 1:
                gn.append(float(torch.nn.utils.clip_grad_norm_(ad.parameters(), tr["clip_norm"])))
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                pending = 0
        torch.cuda.synchronize()
        h = {"epoch": ep, "train_ce": sums["ce"] / sums["n_tok"], "train_total_tokmean": sums["tot"] / sums["n_tok"],
             "train_kl": (sums["kl"] / sums["n_mask"]) if arm == "KL" and sums["n_mask"] else None,
             "mask_positions": sums["n_mask"] if arm == "KL" else None, "supervised_tokens": sums["n_tok"],
             "mask_fraction": (sums["n_mask"] / sums["n_tok"]) if arm == "KL" else None,
             "empty_mask_batches": sums["empty_mask_batches"] if arm == "KL" else None,
             "updates_total": steps, "grad_norm_mean": float(np.mean(gn)), "grad_norm_max": float(np.max(gn)),
             "train_seconds": time.time() - t0, "teacher_forward_seconds": sums["teacher_s"]}
        hist.append(h)
        if on_epoch:
            on_epoch(ep, h)
    return hist


def canonical_env_or_exit():
    libs = loaded_cuda_libs()
    if not cublas_consistent(libs) or "nvidia/cublas/lib" not in libs["cublasLt"][0]:
        raise SystemExit(f"not the canonical P2 CUDA environment: {libs}")
    return libs


# ====================================================================== smoke (GPU preflight)

def run_smoke():
    import torch
    from reft.adapters import make_adapter, site_hook_count
    from reft.auto_policy import build_tf_batch, decode_frozen_lid, tf_loss
    from reft.common import fillers
    from reft.text import tag_unit, tok_bytes_fn, token_unit_map, units
    from reft.train_supervised import GPU_FORBIDDEN, batch_loss, c21, encode, load_backbone, randomize, suppress_masks, tf_forward
    t_all = time.time()
    C1, B = c21(), cb()
    fz = check_freeze()
    R = {"freeze": fz}
    # ---- G1 provenance
    p1d = read_json(art("p2_1d/manifest.json"))["inputs"]
    cur = {"report.json": art("p2_1/report.json"), "evaluation.json": art("p2_1/evaluation.json"), "R2/selection.json": art("p2_1/R2/selection.json"),
           "R2/manifest.json": art("p2_1/R2/manifest.json"), "verify.json": art("p2_1/verify.json")}
    g1 = {"p2_1_artifacts_unchanged_vs_p2_1d_record": all(sha(p) == p1d[k] for k, p in cur.items()),
          "pinned_baseline": sha(art("p2_0/frozen_baseline_dev150_beam5.jsonl")) == C1["frozen_baseline"]["beam5_sha256"],
          "manifests": sha(art("p2_0/train_manifest.json")) == C1["data"]["train_manifest_sha256"]
          and sha(art("p2_0/dev_manifest.json")) == C1["data"]["dev_manifest_sha256"],
          "no_training_outputs_yet": not out("training").exists(),
          "branch": git_state()["branch"] == "cs-asr-reft"}
    R["G1"] = g1
    tm = read_json(art("p2_0/train_manifest.json"))["rows"]
    dm = read_json(art("p2_0/dev_manifest.json"))["rows"]
    tg = read_json(art("p1/targets.json"))["targets"]
    firewall(GPU_FORBIDDEN)
    b, env = load_backbone()
    w0 = weights_hash(b.model)
    libs = canonical_env_or_exit()
    torch.cuda.reset_peak_memory_stats()
    eos = b.processor.tokenizer.eos_token_id
    tiny = sorted(tm, key=lambda r: r["utterance_id"])[:4]
    lam = B["kl"]["lambda_kl"]
    # ---- G3 KL objective on the real model
    g3 = {}
    ad = make_adapter("R2", b.model, seed_config(C1, 42), b.device)
    loss_kl, st, (zt, zs, X, Y) = kl_step(b, ad, tiny, lam)
    with torch.no_grad():
        _, enc = encode(b, [r["audio_path"] for r in tiny])
        zplain = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
    g3["teacher_equals_frozen_forward"] = bool(torch.equal(zt, zplain))
    g3["zero_init_student_equals_teacher"] = bool(torch.equal(zs.detach(), zt))
    g3["initial_kl"] = st["kl"]
    g3["initial_kl_near_zero"] = abs(st["kl"]) <= 1e-6
    loss_kl.backward()
    g_kl = {n: p.grad.detach().clone() for n, p in ad.module.named_parameters()}
    ad2 = make_adapter("R2", b.model, seed_config(C1, 42), b.device)
    loss_ce, _, _, _, _ = batch_loss(b, ad2, tiny, grad=True)
    loss_ce.backward()
    g_ce = {n: p.grad.detach().clone() for n, p in ad2.module.named_parameters()}
    g3["ce_value_identical_to_p2_1_path"] = st["ce"] == float(loss_ce)
    # at zero-init the KL gradient is mathematically 0 (softmax(z_s) - p0 = 0) but carries ~1e-9 softmax round-off
    g3["init_grads_equal_ce_path"] = all(torch.allclose(g_kl[n], g_ce[n], rtol=1e-4, atol=1e-7) for n in g_ce)
    g3["init_grad_max_abs_diff"] = max(float((g_kl[n] - g_ce[n]).abs().max()) for n in g_ce)
    g3["backbone_grads_none"] = all(p.grad is None for p in b.model.parameters()) and sum(p.requires_grad for p in b.model.parameters()) == 0
    rnd = make_adapter("R2", b.model, seed_config(C1, 42), b.device)
    randomize(rnd, 0.01, 0.5, seed=7)
    lr, sr, _ = kl_step(b, rnd, tiny, lam)
    lr.backward()
    g3["perturbed_kl"] = sr["kl"]
    g3["perturbed_kl_positive"] = sr["kl"] > 0
    g3["kl_grads_reach_R_W_b"] = {n: float(p.grad.norm()) for n, p in rnd.module.named_parameters()}
    g3["kl_grads_nonzero_finite"] = all(v > 0 and v == v and v != float("inf") for v in g3["kl_grads_reach_R_W_b"].values())
    g3["mask_count_tiny"] = st["n_mask"]
    g3["hooks_after"] = site_hook_count(b.model, C1["methods"]["R2"]["layer"])
    g3["label_contract"] = bool(((Y[:, :3] == -100).all()) and int((Y == eos).sum()) == len(tiny))
    R["G3"] = g3
    del ad2, rnd
    # ---- mask coverage over the whole training set (training data only)
    FILL = fillers()
    tb = tok_bytes_fn(b.processor.tokenizer)
    cov = {"eligible": 0, "teacher_correct": 0, "eos": [0, 0], "lang": {}, "switchback": [0, 0], "unmappable_utterances": 0}
    for j in range(0, len(tm), 8):
        rows = tm[j:j + 8]
        Xc, Yc = build_tf_batch([r["prompt"] for r in rows], [r["target_ids"] for r in rows], eos, eos)
        _, e_ = encode(b, [r["audio_path"] for r in rows])
        with torch.no_grad():
            z_ = b.model(encoder_outputs=(e_,), decoder_input_ids=Xc.to(b.device), use_cache=False).logits
        M = ((Yc.to(b.device) != -100) & (z_.argmax(-1) == Yc.to(b.device).clamp(min=0))).cpu()
        cov["eligible"] += int((Yc != -100).sum())
        cov["teacher_correct"] += int(M.sum())
        for i, r in enumerate(rows):
            ids = r["target_ids"]
            n = len(ids)
            cov["eos"][0] += int(M[i, 3 + n - 1])
            cov["eos"][1] += 1
            t_text = tg[r["utterance_id"]]["target_text"]
            um = token_unit_map(t_text, ids[:-1], tb)
            if um is None:
                cov["unmappable_utterances"] += 1
                continue
            U = units(t_text)
            tags = [tag_unit(u) for u in U]
            lang_units = [k for k, (u, tg_) in enumerate(zip(U, tags)) if tg_ in ("EN", "ZH") and u not in FILL]
            sb_units = {lang_units[q] for q in range(1, len(lang_units)) if tags[lang_units[q]] == "ZH" and tags[lang_units[q - 1]] == "EN"}
            for t, us in enumerate(um):
                lab = "unmapped" if not us else ("EN" if {tags[x] for x in us} == {"EN"} else "ZH" if {tags[x] for x in us} == {"ZH"} else "OTHER")
                c = cov["lang"].setdefault(lab, [0, 0])
                c[0] += int(M[i, 3 + t])
                c[1] += 1
                if us and us[0] in sb_units and (t == 0 or not um[t - 1] or um[t - 1][-1] != us[0]):
                    cov["switchback"][0] += int(M[i, 3 + t])
                    cov["switchback"][1] += 1
    cov["selected_fraction"] = cov["teacher_correct"] / cov["eligible"]
    cov["switchback_coverage"] = cov["switchback"][0] / cov["switchback"][1] if cov["switchback"][1] else None
    cov["eos_coverage"] = cov["eos"][0] / cov["eos"][1]
    R["mask_coverage"] = cov
    # ---- G2 CE equivalence: one full epoch of seed-42 CE vs P2-1 epoch 1 (bitwise)
    tr = C1["training"]
    torch.manual_seed(42)
    ad3 = make_adapter("R2", b.model, seed_config(C1, 42), b.device)
    opt = torch.optim.AdamW(ad3.parameters(), lr=C1["methods"]["R2"]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"], weight_decay=tr["weight_decay"])
    h = train_loop(b, ad3, opt, tm, tr, 42, "CE", lam, 1)
    ref = torch.load(art("p2_1/R2/epoch1/adapter.pt"))
    mine = ad3.state_dict()
    R["G2"] = {"bitwise_equal_p2_1_epoch1": all(torch.equal(mine[k], ref[k]) for k in ref) and set(mine) == set(ref),
               "max_abs_diff": max(float((mine[k] - ref[k]).abs().max()) for k in ref),
               "epoch1_train_ce_new": h[0]["train_ce"], "epoch1_train_ce_p2_1": read_json(art("p2_1/R2/manifest.json"))["history"][1]["train_ce"],
               "updates": h[0]["updates_total"], "train_seconds_ce_epoch": h[0]["train_seconds"]}
    del ad3, opt
    # ---- timing: CE vs KL micro-batch (forward + backward), teacher overhead
    mb = sorted(tm, key=lambda r: r["utterance_id"])[:tr["micro_batch"]]
    tim = {}
    for arm in ARMS:
        a_ = make_adapter("R2", b.model, seed_config(C1, 42), b.device)
        ts, tt = [], []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            if arm == "CE":
                l_, s_ = ce_step(b, a_, mb)
            else:
                l_, s_, _ = kl_step(b, a_, mb, lam)
            l_.backward()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
            tt.append(s_["teacher_s"])
            for p in a_.parameters():
                p.grad = None
        tim[arm] = {"micro_batch_seconds": ts, "teacher_seconds": tt}
        del a_
    # ---- G4 decoding: zero-init parity (20 dev) and random-adapter cached greedy vs TF argmax (2 dev)
    base = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}
    inv = {v: k for k, v in b.model.generation_config.lang_to_id.items()}
    z0 = make_adapter("R2", b.model, seed_config(C1, 43), b.device)
    eq, dec_s = 0, []
    for r in dm[:20]:
        lid = {"token": r["frozen_lid_token"], "language": inv[r["frozen_lid_token"]]}
        t0 = time.time()
        rec, _ = decode_frozen_lid(b, r["audio_path"], lid, 5, z0)
        g_, _ = decode_frozen_lid(b, r["audio_path"], lid, 1, z0)
        dec_s.append(time.time() - t0)
        eq += rec["tokens"] == base[r["utterance_id"]]["tokens"]
    rz = make_adapter("R2", b.model, seed_config(C1, 44), b.device)
    randomize(rz, 0.01, 0.5, seed=11)
    masks = suppress_masks(b)
    mism = []
    for r in dm[:2]:
        lid = {"token": r["frozen_lid_token"], "language": inv[r["frozen_lid_token"]]}
        g1_, _ = decode_frozen_lid(b, r["audio_path"], lid, 1, rz)
        _, e1 = encode(b, [r["audio_path"]])
        Xs = torch.tensor([g1_["prompt"] + g1_["tokens"][:-1]], device=b.device)
        lg = tf_forward(b, rz, e1, Xs)[0].float()
        am = [int(lg[3 + t].masked_fill(masks[1] if t == 0 else masks[0], -float("inf")).argmax()) for t in range(len(g1_["tokens"]))]
        mism.append(sum(x != y for x, y in zip(am, g1_["tokens"])))
    R["G4"] = {"zero_init_beam5_equal_baseline": [eq, 20], "random_cached_greedy_vs_tf_mismatch": mism,
               "hooks_after": site_hook_count(b.model, C1["methods"]["R2"]["layer"])}
    w1 = weights_hash(b.model)
    peak_a, peak_r = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    nmb = -(-len(tm) // tr["micro_batch"])
    per_dev = sum(dec_s) / len(dec_s) + 0.15
    proj = {arm: 3 * nmb * (sum(tim[arm]["micro_batch_seconds"][1:]) / 2) + 4 * len(dm) * per_dev + 180 for arm in ARMS}
    R["G5"] = {"timing": tim, "teacher_overhead_fraction": (sum(tim["KL"]["micro_batch_seconds"][1:]) / sum(tim["CE"]["micro_batch_seconds"][1:])) - 1,
               "peak_allocated_bytes": peak_a, "peak_reserved_bytes": peak_r, "projected_job_seconds": proj, "dev_decode_seconds_per_utt": per_dev}
    R["weights_unchanged"] = w0 == w1
    R["cuda_libs"] = libs
    g = {"G1_provenance": all(g1.values()),
         "G2_ce_equivalence_checked": "bitwise_equal_p2_1_epoch1" in R["G2"],
         "G3_kl_objective": all(g3[k] for k in ("teacher_equals_frozen_forward", "zero_init_student_equals_teacher", "initial_kl_near_zero",
                                                 "ce_value_identical_to_p2_1_path", "init_grads_equal_ce_path", "backbone_grads_none",
                                                 "perturbed_kl_positive", "kl_grads_nonzero_finite", "label_contract")) and g3["hooks_after"] == 0,
         "mask_switchback_coverage_ok": (cov["switchback_coverage"] or 0) >= B["mask_coverage_stop"]["min_switchback_token_coverage"],
         "G4_decoding": eq == 20 and all(x <= 1 for x in mism) and R["G4"]["hooks_after"] == 0,
         "G5_resources": all(v < 9000 for v in proj.values()) and peak_r < 38e9,
         "G6_reproducibility": R["weights_unchanged"] and cublas_consistent(libs),
         "G7_protocol": out("prefix_audit/cases_private.json").exists()}
    verdict = "READY_FOR_P2_1B_EXECUTION" if all(g.values()) else "P2_1B_BLOCKED_PREFLIGHT_" + "_".join(k for k, v in g.items() if not v)
    rec = {"schema": "reft_p2_1b_preflight_v1", "utc": now(), "git": git_state(), "slurm": slurm(), "env": env, "results": R,
           "gates": g, "verdict": verdict, "ce42_reuse": R["G2"]["bitwise_equal_p2_1_epoch1"], "wall_seconds": time.time() - t_all}
    write_json(out(f"preflight_{slurm()['SLURM_JOB_ID'] or 'local'}.json"), rec)
    print(json.dumps({"verdict": verdict, "gates": g, "G2": R["G2"], "G3": g3, "mask_coverage": {k: v for k, v in cov.items() if k != "lang"},
                      "mask_lang": cov["lang"], "G4": R["G4"], "G5": {k: v for k, v in R["G5"].items() if k != "timing"}}, indent=1, default=str))


# ====================================================================== train (GPU)

def run_train(arm: str, seed: int):
    import torch
    from reft.adapters import make_adapter
    from reft.auto_policy import decode_frozen_lid, frozen_lid
    from reft.train_supervised import GPU_FORBIDDEN, c21, load_backbone
    C1, B = c21(), cb()
    assert arm in ARMS and seed in B["seeds"]
    fz = check_freeze()
    if not out("AUTHORIZATION.json").exists():
        raise SystemExit("P2-1b not authorized")
    pre = sorted(out().glob("preflight_*.json"), key=lambda p: p.stat().st_mtime)
    assert pre and read_json(pre[-1])["verdict"] == "READY_FOR_P2_1B_EXECUTION", "preflight not passed"
    if arm == "CE" and seed == 42 and read_json(pre[-1])["ce42_reuse"]:
        raise SystemExit("CE seed 42 is reused from P2-1 (equivalence passed); no rerun")
    od = out("training", f"{arm}_seed{seed}")
    if od.exists():
        raise SystemExit(f"{od} exists; runs are write-once")
    tm, dm = read_json(art("p2_0/train_manifest.json"))["rows"], read_json(art("p2_0/dev_manifest.json"))["rows"]
    base = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}
    firewall(GPU_FORBIDDEN)
    Cs = seed_config(C1, seed)
    tr = Cs["training"]
    torch.manual_seed(seed)
    b, env = load_backbone()
    w0 = weights_hash(b.model)
    libs = canonical_env_or_exit()
    ad = make_adapter("R2", b.model, Cs, b.device)
    assert ad.n_trainable() == C1["methods"]["R2"]["expected_trainable_parameters"]
    opt = torch.optim.AdamW(ad.parameters(), lr=C1["methods"]["R2"]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"], weight_decay=tr["weight_decay"])
    inv = {v: k for k, v in b.model.generation_config.lang_to_id.items()}
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    lam = B["kl"]["lambda_kl"] if arm == "KL" else 0.0

    def dev_decode(ep):
        res = {"beam5": [], "greedy": []}
        t0 = time.time()
        for r in dm:
            lid = {"token": r["frozen_lid_token"], "language": inv[r["frozen_lid_token"]]}
            for reg, nb in (("beam5", 5), ("greedy", 1)):
                rec, inp = decode_frozen_lid(b, r["audio_path"], lid, nb, ad)
                res[reg].append({"id": r["utterance_id"], **rec})
            with ad.active(b.model):
                assert frozen_lid(b, inp["input_features"], ad)["token"] == r["frozen_lid_token"], "LID changed under adapter"
        for reg, v in res.items():
            write_jsonl(od / f"epoch{ep}" / f"dev_{reg}.jsonl", v)
        return time.time() - t0, res

    sec, res0 = dev_decode(0)
    ident = sum(x["tokens"] == base[x["id"]]["tokens"] for x in res0["beam5"])
    if ident != len(dm):
        raise SystemExit(f"zero-init adapter does not reproduce the frozen baseline ({ident}/{len(dm)})")
    hist = [{"epoch": 0, "zero_init_identity_beam5": ident, "dev_decode_seconds": sec}]

    def on_epoch(ep, h):
        (od / f"epoch{ep}").mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), od / f"epoch{ep}" / "adapter.pt")
        s_, _ = dev_decode(ep)
        hist.append({**h, "dev_decode_seconds": s_, "adapter_sha256": sha(od / f"epoch{ep}" / "adapter.pt")})
        print(json.dumps(hist[-1]), flush=True)

    train_loop(b, ad, opt, tm, tr, seed, arm, lam, tr["epochs"], on_epoch)
    w1 = weights_hash(b.model)
    import numpy as np
    lat = {reg: float(np.mean([x["seconds"] for x in read_jsonl(od / f"epoch{tr['epochs']}" / f"dev_{reg}.jsonl")])) for reg in ("beam5", "greedy")}
    write_json(od / "manifest.json", {
        "schema": "reft_p2_1b_run_v1", "utc": now(), **fz, "git": git_state(), "env": env, "slurm": slurm(), "arm": arm, "seed": seed,
        "lambda_kl": lam, "method": "R2", "config": Cs["methods"]["R2"], "training": tr, "history": hist,
        "weights_before": w0, "weights_after": w1, "weights_unchanged": w0 == w1, "cuda_libs": libs,
        "efficiency": {"trainable_parameters": ad.n_trainable(), "adapter_bytes_fp32": 4 * ad.n_trainable(),
                       "optimizer_updates": hist[-1]["updates_total"], "supervised_tokens": sum(h.get("supervised_tokens", 0) for h in hist),
                       "train_seconds": sum(h.get("train_seconds", 0) for h in hist), "teacher_forward_seconds": sum(h.get("teacher_forward_seconds", 0) for h in hist),
                       "wall_seconds": time.time() - t_all, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "mean_decode_seconds": lat}})


# ====================================================================== verify (GPU)

def run_verify():
    import torch
    from reft.adapters import make_adapter
    from reft.auto_policy import decode_frozen_lid
    from reft.common import cfg
    from reft.train_supervised import c21, load_backbone
    C1 = c21()
    fz = check_freeze()
    b, env = load_backbone()
    w0 = weights_hash(b.model)
    libs = canonical_env_or_exit()
    dm = read_json(art("p2_0/dev_manifest.json"))["rows"][:20]
    inv = {v: k for k, v in b.model.generation_config.lang_to_id.items()}
    res = {}
    for rd in sorted(out("training").iterdir()):
        man = read_json(rd / "manifest.json")
        res[rd.name] = {}
        for h in man["history"][1:]:
            p = rd / f"epoch{h['epoch']}" / "adapter.pt"
            ad = make_adapter("R2", b.model, seed_config(C1, man["seed"]), b.device)
            ad.load_state_dict(torch.load(p))
            rec = {x["id"]: x for x in read_jsonl(rd / f"epoch{h['epoch']}" / "dev_beam5.jsonl")}
            eq = 0
            for r in dm:
                d, _ = decode_frozen_lid(b, r["audio_path"], {"token": r["frozen_lid_token"], "language": inv[r["frozen_lid_token"]]}, 5, ad)
                eq += d["tokens"] == rec[r["utterance_id"]]["tokens"]
            res[rd.name][h["epoch"]] = {"sha_ok": sha(p) == h["adapter_sha256"], "reload_equal": [eq, len(dm)]}
    w1 = weights_hash(b.model)
    ok = w0 == w1 == cfg()["model"]["expected_weights_hash"] and all(v["sha_ok"] and v["reload_equal"][0] == len(dm) for x in res.values() for v in x.values())
    write_json(out("verify.json"), {"schema": "reft_p2_1b_verify_v1", "utc": now(), **fz, "slurm": slurm(), "cuda_libs": libs,
                                    "results": res, "weights_unchanged": w0 == w1, "all_ok": ok,
                                    "historical_ce42": "verified in P2-1 (p2_1/verify.json all_ok)" if read_json(art("p2_1/verify.json"))["all_ok"] else "NOT OK"})
    print(json.dumps({"all_ok": ok, "results": res}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "smoke", "train", "verify"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        run_train(a.arm, a.seed)
    else:
        {"freeze": run_freeze, "smoke": run_smoke, "verify": run_verify}[a.cmd]()


if __name__ == "__main__":
    main()
