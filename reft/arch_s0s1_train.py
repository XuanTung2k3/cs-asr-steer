#!/usr/bin/env python
"""S0/S1 parameter-matched architecture screening: R3-GoldKL vs M1-GoldKL (R2-GoldKL = historical reference).
docs/reft/ARCH_S0S1_PROTOCOL.md, docs/reft/ARCH_S0S1_DECISION_RULES.md, configs/reft/arch_s0s1_frozen.yaml.

  freeze              (CPU) write-once arch_s0s1/freeze.json + AUTHORIZATION.json; P2-0/P2-1b/P2-1c/P2-2 chains verified
  freeze-amend --doc  (CPU) write-once amendment record chained to the current head (documented repair only)
  preflight           (GPU) S0 gates G1-G7 -> arch_s0s1/preflight_<job>.json; ARCH_S0_READY_FOR_S1 or ARCH_S0_BLOCKED_<gates>
  train --arch --seed (GPU) one write-once S1 run arch_s0s1/training/{R3,M1}_seed{s}
  verify              (GPU) every S1 epoch adapter reloads and re-decodes the first 20 dev utterances as recorded

The objective is reft.p2_1b_train.kl_step unchanged (token-mean CE + 1.0 * KL(p0 || p_theta) on the teacher-correct
mask). goldkl_epoch mirrors reft.p2_1b_train.train_loop (arm KL) step for step and only adds read-only diagnostics; the
preflight proves it reproduces the historical R2 KL_seed42 epoch-1 adapter bitwise.

ARCH_S0S1_DRY_ROOT=<scratch dir> redirects every output for pre-freeze engineering dry runs (no freeze check; train
refuses to run).
"""
from __future__ import annotations

import argparse
import contextlib
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

from reft.common import art, firewall, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import cublas_consistent, git_state, now, sha, slurm, weights_hash  # noqa: E402

CONF = ROOT / "configs/reft/arch_s0s1_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/ARCH_S0S1_PROTOCOL.md", "docs/reft/ARCH_S0S1_DECISION_RULES.md", "configs/reft/arch_s0s1_frozen.yaml",
                  "reft/arch_s0s1_adapters.py", "reft/arch_s0s1_train.py", "reft/arch_s0s1_evaluate.py", "tests/reft/test_arch_s0s1.py",
                  "slurm/reft_arch_s0s1.sbatch", "slurm/reft_arch_s0s1_cpu.sbatch", "slurm/reft_arch_s0s1_submit.sh"]
REUSED_SOURCES = ["reft/adapters.py", "reft/site.py", "reft/auto_policy.py", "reft/baseline_auto.py", "reft/train_supervised.py",
                  "reft/p2_1b_train.py", "reft/p2_1b_evaluate.py", "reft/evaluate_supervised.py", "reft/p2_1d_audit.py",
                  "reft/p2_1c_continuation_audit.py", "reft/p2_2_evaluate.py", "reft/text.py", "reft/common.py", "reft/provenance.py",
                  "configs/reft/p0p1.yaml", "configs/reft/p2_1_frozen.yaml", "configs/reft/p2_1b_frozen.yaml"]
ARCHS = ("R3", "M1")
SEEDS = (42, 43, 44)
DRY = os.environ.get("ARCH_S0S1_DRY_ROOT")


def ca() -> dict:
    return yaml.safe_load(CONF.read_text())


def out(*p) -> Path:
    return Path(DRY, *p) if DRY else art("arch_s0s1", *p)


def control_runs() -> dict:
    """R2-GoldKL historical runs (P2-1b KL_seed s): selected epoch, recorded adapter hashes per epoch."""
    res = {}
    for s, run in ca()["architectures"]["R2"]["runs"].items():
        rd = art(run)
        man = read_json(rd / "manifest.json")
        res[int(s)] = {"run": run, "dir": rd, "selected_epoch": read_json(rd / "selection.json")["selected"]["epoch"],
                       "adapter_sha": {h["epoch"]: h["adapter_sha256"] for h in man["history"][1:]}, "history": man["history"]}
    return res


def frozen_inputs() -> list[str]:
    L = ["p2_0/train_manifest.json", "p2_0/dev_manifest.json", "p2_0/data_audit.json", "p2_0/frozen_baseline_dev150_beam5.jsonl",
         "p2_0/frozen_baseline_dev150_greedy.jsonl", "p1/targets.json", "p2_1b/evaluation.json", "p2_1b/report.json", "p2_1b/verify.json",
         "p2_1c/anchors_private.json", "p2_1c/anchors_inference.json", "p2_1c/analysis_r1.json", "p2_1c/freeze_r1.json",
         "p2_2/report.json", "p2_2/evaluation.json", "p2_2/freeze_r2.json"]
    for c in control_runs().values():
        L += [f"{c['run']}/selection.json", f"{c['run']}/manifest.json"]
        L += [f"{c['run']}/epoch{e}/adapter.pt" for e in (1, 2, 3)]
        L += [f"{c['run']}/epoch{e}/dev_{r}.jsonl" for e in (0, 1, 2, 3) for r in ("beam5", "greedy")]
    return L


def upstream_chains() -> dict:
    """P2-0 -> P2-1b (via P2-2), P2-1c and P2-2 freeze chains, each verified by its own checker."""
    from reft.p2_1c_continuation_audit import check_freeze as p21c
    from reft.p2_2_train import check_freeze as p22
    c22 = p22()
    c1c = p21c()
    return {"p2_2": c22, "p2_1c": {k: v for k, v in c1c.items() if k != "historical_chain_checked_here"}}


# ====================================================================== freeze chain

def freeze_path(rev: int) -> Path:
    return out("freeze.json") if rev == 0 else out(f"freeze_r{rev}.json")


def chain_head() -> int:
    rev = 0
    while freeze_path(rev + 1).exists():
        rev += 1
        if read_json(freeze_path(rev))["prev_freeze_sha256"] != sha(freeze_path(rev - 1)):
            raise SystemExit(f"S0/S1 freeze chain broken at {freeze_path(rev).name}")
    return rev


def run_freeze():
    if DRY:
        raise SystemExit("no freeze in dry-run mode")
    p = out("freeze.json")
    if p.exists():
        raise SystemExit(f"{p} exists; write-once")
    g = git_state()
    if g["branch"] != "cs-asr-reft":
        raise SystemExit(f"wrong branch {g['branch']}")
    ctl = {s: {e: sha(c["dir"] / f"epoch{e}" / "adapter.pt") == h for e, h in c["adapter_sha"].items()} for s, c in control_runs().items()}
    if not all(v for x in ctl.values() for v in x.values()):
        raise SystemExit(f"R2-GoldKL control adapter hash mismatch: {ctl}")
    rec = {"schema": "reft_arch_s0s1_freeze_v1", "utc": now(), "git": g, "sources": {s: sha(ROOT / s) for s in FREEZE_SOURCES},
           "reused_sources": {s: sha(ROOT / s) for s in REUSED_SOURCES}, "inputs": {i: sha(art(i)) for i in frozen_inputs()},
           "controls_match_recorded_sha": {str(k): v for k, v in ctl.items()}, "upstream": upstream_chains()}
    write_json(p, rec)
    write_json(out("AUTHORIZATION.json"), {"utc": now(), "stage": "S0/S1 parameter-matched R3 vs M1 under GoldKL",
                                           "authorized_by": "project owner (2026-10-15 prompt)", "runs": ca()["new_runs"],
                                           "max_new_training_runs": ca()["max_new_training_runs"],
                                           "conditions": "S1 only if S0 is ARCH_S0_READY_FOR_S1; no E1+D2/P2-3/P3/LoRA; no rank, layer, "
                                                         "seed, loss-weight or data sweeps; no extra data evaluations"})
    print(p)


def run_freeze_amend(doc: str):
    """Amendment r<k>: re-hash the sources after a documented repair; earlier records are kept and chained."""
    if DRY:
        raise SystemExit("no freeze in dry-run mode")
    k = chain_head() + 1
    prev = freeze_path(k - 1)
    rec0 = read_json(prev)
    if not (ROOT / doc).exists():
        raise SystemExit(f"{doc} missing")
    inputs = {i: sha(art(i)) for i in rec0["inputs"]}
    if inputs != rec0["inputs"]:
        raise SystemExit(f"frozen inputs changed since {prev.name}")
    srcs = list(rec0["sources"]) + ([doc] if doc not in rec0["sources"] else [])
    rec = {"schema": "reft_arch_s0s1_freeze_amendment_v1", "amendment": f"r{k}", "utc": now(), "git": git_state(),
           "prev_freeze": prev.name, "prev_freeze_sha256": sha(prev), "amendment_doc": doc,
           "changed_sources": sorted(x for x in srcs if rec0["sources"].get(x) != sha(ROOT / x)),
           "sources": {x: sha(ROOT / x) for x in srcs}, "reused_sources": {x: sha(ROOT / x) for x in rec0["reused_sources"]},
           "inputs": inputs, "controls_match_recorded_sha": rec0["controls_match_recorded_sha"], "upstream": upstream_chains(),
           "outputs_at_amendment": {q.name: sha(q) for q in sorted(out().glob("*.json"))}}
    print(write_json(freeze_path(k), rec))


def check_freeze() -> dict:
    if DRY:
        return {"dry_run": True}
    head = chain_head()
    p = freeze_path(head)
    rec = read_json(p)
    bad = [s for s, h in {**rec["sources"], **rec["reused_sources"]}.items() if sha(ROOT / s) != h]
    bad += [i for i, h in rec["inputs"].items() if sha(art(i)) != h]
    if bad:
        raise SystemExit(f"S0/S1 frozen sources/inputs changed: {bad}")
    up = upstream_chains()
    if up != rec["upstream"]:
        raise SystemExit("an upstream freeze chain changed (P2-0/P2-1b/P2-1c/P2-2)")
    return {"arch_freeze_sha256": sha(p), "arch_freeze_record": p.name, "arch_freeze_chain_len": head + 1,
            "p2_2_freeze_sha256": up["p2_2"]["p2_2_freeze_sha256"], "p2_1b_freeze_sha256": up["p2_2"]["p2_1b_freeze_sha256"],
            "p2_0_chain_head": up["p2_2"]["p2_0_chain_head"], "p2_1c_freeze_sha256": up["p2_1c"].get("p2_1c_freeze_sha256")}


# ====================================================================== runtime (GPU)

class Runtime:
    """Frozen backbone in the canonical environment; training/dev manifests (dev references firewalled)."""

    def __init__(self):
        import torch
        from reft.common import cfg
        from reft.p2_1b_train import canonical_env_or_exit
        from reft.train_supervised import GPU_FORBIDDEN, c21, load_backbone
        self.torch, self.C, self.C1 = torch, ca(), c21()
        assert sha(art("p2_0/train_manifest.json")) == self.C1["data"]["train_manifest_sha256"]
        assert sha(art("p2_0/dev_manifest.json")) == self.C1["data"]["dev_manifest_sha256"]
        self.tm = read_json(art("p2_0/train_manifest.json"))["rows"]
        self.dm = read_json(art("p2_0/dev_manifest.json"))["rows"]
        firewall(GPU_FORBIDDEN)
        self.b, self.env = load_backbone()
        self.libs = canonical_env_or_exit()
        self.w0 = weights_hash(self.b.model)
        self.expected_w = cfg()["model"]["expected_weights_hash"]
        self.inv = {v: k for k, v in self.b.model.generation_config.lang_to_id.items()}
        self.eos = self.b.processor.tokenizer.eos_token_id

    def lid(self, row):
        return {"token": row["frozen_lid_token"], "language": self.inv[row["frozen_lid_token"]]}

    def decode(self, row, beams, ad):
        from reft.auto_policy import decode_frozen_lid
        return decode_frozen_lid(self.b, row["audio_path"], self.lid(row), beams, ad)

    def tf_batch(self, rows):
        from reft.auto_policy import build_tf_batch
        from reft.train_supervised import encode
        X, Y = build_tf_batch([r["prompt"] for r in rows], [r["target_ids"] for r in rows], self.eos, self.eos)
        _, enc = encode(self.b, [r["audio_path"] for r in rows])
        return X.to(self.b.device), Y.to(self.b.device), enc

    def logits(self, ad, enc, X, grad=False):
        from reft.train_supervised import tf_forward
        return tf_forward(self.b, ad, enc, X, grad=grad)

    def site_record(self, enc, X):
        """Frozen forward with the P1 Site recorder: q, u (= encoder_attn output) and the frozen logits."""
        torch = self.torch
        from reft.site import Site
        with torch.no_grad(), Site(self.b.model, 31, record=True) as s:
            z = self.b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
        return s.q, s.s, z

    def site_delta_logits(self, enc, X, delta):
        torch = self.torch
        from reft.site import Site
        with torch.no_grad(), Site(self.b.model, 31, delta=delta):
            return self.b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits


def manual_delta(ad, q, u, start_pos=3):
    """Update recomputed outside the hook from recorded q, u (R2/R3: module(h); M1: module(h, u)), masked at pos >= start_pos."""
    import torch
    h = q + u
    pos = torch.arange(h.shape[1], device=h.device)
    mask = (pos >= start_pos).to(torch.float32)[None, :, None]
    with torch.no_grad():
        d = ad.module(h.float(), u.float()) if ad.method == "cond_loreft" else ad.module(h.float())
    return d * mask


# ====================================================================== training loop (GoldKL, P2-1b semantics)

def goldkl_epoch(b, ad, opt, tm, tr, seed, ep, lam, steps0, log):
    """One epoch of reft.p2_1b_train.train_loop (arm KL), step for step, plus read-only per-step diagnostics in `log`."""
    import torch
    from reft.arch_s0s1_adapters import named_trainable, orth_error
    from reft.p2_1b_train import kl_step
    mb, acc = tr["micro_batch"], tr["grad_accum"]
    steps = steps0
    order = np.random.default_rng(seed + ep).permutation(len(tm))
    batches = [[tm[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]
    torch.cuda.synchronize()
    t0 = time.time()
    sums = {"ce": 0.0, "kl": 0.0, "tot": 0.0, "n_tok": 0, "n_mask": 0, "teacher_s": 0.0, "empty_mask_batches": 0}
    pending, gn = 0, []
    params = named_trainable(ad)
    for k, rows in enumerate(batches):
        loss, st, _ = kl_step(b, ad, rows, lam)
        sums["kl"] += st["kl"] * st["n_mask"]
        sums["n_mask"] += st["n_mask"]
        sums["teacher_s"] += st["teacher_s"]
        sums["empty_mask_batches"] += st["n_mask"] == 0
        (loss / acc).backward()
        sums["ce"] += st["ce"] * st["n_tok"]
        sums["tot"] += float(loss) * st["n_tok"]
        sums["n_tok"] += st["n_tok"]
        log["micro"].append({"epoch": ep, "k": k, "ce": st["ce"], "kl": st["kl"], "n_tok": st["n_tok"], "n_mask": st["n_mask"]})
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
                rec = {"epoch": ep, "step": steps, "grad_norm_total_preclip": gn[-1], "grad_norm_preclip": gpre,
                       "update_norm": {n: float((p.detach() - before[n]).norm()) for n, p in params.items()},
                       "param_norm": {n: float(p.detach().norm()) for n, p in params.items()}, "orth_err": orth_error(ad)}
            log["steps"].append(rec)
    torch.cuda.synchronize()
    return {"epoch": ep, "train_ce": sums["ce"] / sums["n_tok"], "train_total_tokmean": sums["tot"] / sums["n_tok"],
            "train_kl": (sums["kl"] / sums["n_mask"]) if sums["n_mask"] else None, "mask_positions": sums["n_mask"],
            "supervised_tokens": sums["n_tok"], "mask_fraction": sums["n_mask"] / sums["n_tok"],
            "empty_mask_batches": sums["empty_mask_batches"], "updates_total": steps, "grad_norm_mean": float(np.mean(gn)),
            "grad_norm_max": float(np.max(gn)), "train_seconds": time.time() - t0, "teacher_forward_seconds": sums["teacher_s"]}


def make_opt(ad, C1, tr):
    import torch
    return torch.optim.AdamW(ad.parameters(), lr=C1["methods"]["R2"]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"],
                             weight_decay=tr["weight_decay"])


# ====================================================================== preflight helpers (GPU)

def cached_vs_full(b, ad, enc, X, margin):
    """Incremental KV-cached forward vs one full forward on the same tokens (P2-1c T1 semantics), positions spied."""
    import torch
    seen = []
    L = b.model.model.decoder.layers[31]
    hk = L.register_forward_pre_hook(lambda m, a, k: seen.append(k["cache_position"].tolist()), with_kwargs=True)
    ctx = ad.active(b.model) if ad is not None else contextlib.nullcontext()
    dev = b.device
    try:
        with torch.inference_mode(), ctx:
            full = torch.log_softmax(b.model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([X], device=dev),
                                             use_cache=False).logits[0].float(), -1)
            full_pos = seen[-1]
            o = b.model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([X[:4]], device=dev), use_cache=True)
            rows, steps = [torch.log_softmax(o.logits[0].float(), -1)], [seen[-1]]
            pkv = o.past_key_values
            for t in range(4, len(X)):
                o = b.model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([[X[t]]], device=dev), past_key_values=pkv,
                            use_cache=True, cache_position=torch.tensor([t], device=dev))
                pkv = o.past_key_values
                rows.append(torch.log_softmax(o.logits[0].float(), -1))
                steps.append(seen[-1])
            inc = torch.cat(rows, 0)
    finally:
        hk.remove()
    obs = torch.tensor(X[1:], device=full.device)
    d = (full[:-1].gather(1, obs[:, None]) - inc[:-1].gather(1, obs[:, None])).abs().max().item()
    top2 = full.topk(2, -1).values
    sure = (top2[:, 0] - top2[:, 1]) >= margin
    agree = bool((full.argmax(-1) == inc.argmax(-1))[sure].all())
    pos_ok = full_pos == list(range(len(X))) and steps[0] == [0, 1, 2, 3] and all(s == [t] for s, t in zip(steps[1:], range(4, len(X))))
    return {"max_abs_logp_observed": d, "argmax_agree_where_sure": agree, "n_positions": len(X), "positions_ok": pos_ok,
            "finite": bool(torch.isfinite(full).all() and torch.isfinite(inc).all())}


def spy_positions(rt, ad, row, beams):
    seen = []
    orig = ad._layer_pre

    def spy(mod, args, kwargs):
        seen.append(kwargs["cache_position"].tolist())
        return orig(mod, args, kwargs)
    ad._layer_pre = spy
    try:
        rec, _ = rt.decode(row, beams, ad)
    finally:
        ad._layer_pre = orig
    return {"first": seen[0], "steps": len(seen), "incremental_ok": all(seen[i] == [seen[0][-1] + i] for i in range(1, len(seen))),
            "n_tokens": len(rec["tokens"]), "seq": seen}


def greedy_vs_tf(rt, ad, row):
    """Cached greedy decode vs teacher-forced argmax on the decoded tokens (canonical suppression masks)."""
    import torch
    from reft.train_supervised import encode, suppress_masks
    masks = suppress_masks(rt.b)
    g1, _ = rt.decode(row, 1, ad)
    _, e1 = encode(rt.b, [row["audio_path"]])
    Xs = torch.tensor([g1["prompt"] + g1["tokens"][:-1]], device=rt.b.device)
    lg = rt.logits(ad, e1, Xs)[0].float()
    am = [int(lg[3 + t].masked_fill(masks[1] if t == 0 else masks[0], -float("inf")).argmax()) for t in range(len(g1["tokens"]))]
    return sum(x != y for x, y in zip(am, g1["tokens"]))


def beam_vs_tf(rt, ad, row):
    """BF16 beam-5 transition scores vs teacher-forced log-probs of the returned sequence (descriptive)."""
    import torch
    import torch.nn.functional as F
    from csasr.models.whisper import batch_model_inputs
    from reft.common import cfg
    from reft.train_supervised import encode
    code = rt.lid(row)["language"].strip("<|>")
    inp = batch_model_inputs(rt.b, [row["audio_path"]])
    ctx = ad.active(rt.b.model) if ad is not None else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        o = rt.b.model.generate(**inp, task="transcribe", language=code, do_sample=False, temperature=0.0,
                                max_new_tokens=cfg()["decode"]["max_new_tokens"],
                                condition_on_prev_tokens=False, num_beams=5, return_dict_in_generate=True, output_scores=True)
    seq = o.sequences[0].tolist()
    toks = seq[4:]
    if rt.eos in toks:
        toks = toks[:toks.index(rt.eos) + 1]
    vals = []
    for t, y in enumerate(toks):
        v = o.scores[t]
        v = v[0] if v.shape[0] == 1 else v[int(o.beam_indices[0, t])]
        vals.append(float(v[y]))
    _, e1 = encode(rt.b, [row["audio_path"]])
    lp = F.log_softmax(rt.logits(ad, e1, torch.tensor([seq[:4] + toks[:-1]], device=rt.b.device))[0].float(), -1)[3:]
    tf_ = lp.gather(-1, torch.tensor(toks, device=rt.b.device)[:, None])[:, 0].cpu()
    d = (torch.tensor(vals) - tf_).abs()
    return {"n": len(toks), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}


# ====================================================================== preflight (GPU, S0 gates)

def run_preflight():
    import torch
    from reft.adapters import site_hook_count
    from reft.arch_s0s1_adapters import (EXPECTED, R2_KEYS, init_record, make_arch_adapter, named_trainable, orth_error, perturb,
                                         transplant_r2_into_m1)
    from reft.auto_policy import frozen_lid
    from reft.p2_1b_train import kl_step, seed_config, teacher_mask
    from reft.train_supervised import batch_loss
    from csasr.models.whisper import batch_model_inputs
    fz = check_freeze()
    t_all = time.time()
    rt = Runtime()
    b, C, P = rt.b, rt.C, rt.C["preflight"]
    T = P["tolerances"]
    dev = b.device
    torch.cuda.reset_peak_memory_stats()
    res = {"freeze": fz}
    jid = str(slurm()["SLURM_JOB_ID"] or "local")
    pdir = out("preflight_files", jid)

    def mk(arch, seed=42):
        torch.manual_seed(seed)
        return make_arch_adapter(arch, b.model, seed, dev)

    # ---- G1 provenance
    ctl = control_runs()
    audit = read_json(art("p2_0/data_audit.json"))
    pb = rt.C1["frozen_baseline"]
    try:
        sq = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        sq = f"unavailable: {e}"
    g = git_state()
    res["G1"] = {"branch": g["branch"] == "cs-asr-reft", "git_head": g["head"], "git_dirty_paths": g["dirty_paths"],
                 "freeze_verified": DRY is None and "arch_freeze_sha256" in fz,
                 "controls_sha": all(sha(c["dir"] / f"epoch{e}" / "adapter.pt") == h for c in ctl.values() for e, h in c["adapter_sha"].items()),
                 "pinned_baseline": sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"],
                 "weights_expected": rt.w0 == rt.expected_w, "canonical_cuda": cublas_consistent(rt.libs),
                 "data_audit_valid_no_locked": bool(audit["valid"] and audit["checks"]["no_locked_dialogues"]),
                 "no_training_outputs_yet": not out("training").exists()}
    res["G1_squeue"] = sq
    tiny = sorted(rt.tm, key=lambda r: r["utterance_id"])[:P["tf_rows"]]
    X, Y, enc = rt.tf_batch(tiny)
    q, u, z0 = rt.site_record(enc, X)
    dev20 = rt.dm[:P["dev_rows"]]
    base5 = {x["id"]: x for x in read_jsonl(art(pb["beam5"]))}
    base1 = {x["id"]: x for x in read_jsonl(art(pb["greedy"]))}

    # ---- G2 parameters, initialization, identity
    g2 = {}
    lat = {}
    for arch in ("R2", "R3", "M1"):
        ad = mk(arch)
        nt = named_trainable(ad)
        m = ad.module
        r = {"n_trainable": ad.n_trainable(), "expected": EXPECTED[arch], "shapes": {n: list(p.shape) for n, p in nt.items()},
             "trainable_names": sorted(nt), "orth_err": orth_error(ad), "init_hashes": init_record(ad),
             "W_equals_R": bool(torch.equal(m.learned.weight, m.rot.weight)), "b_zero": bool((m.learned.bias == 0).all()),
             "W_u_zero": bool((m.w_u == 0).all()) if arch == "M1" else None,
             "backbone_trainable": sum(p.requires_grad for p in b.model.parameters())}
        r["module_output_zero_at_init"] = bool((manual_delta(ad, q, u) == 0).all())
        r["tf_logits_bitwise_frozen"] = bool(torch.equal(rt.logits(ad, enc, X), z0))
        pa = mk(arch)
        perturb(pa, **{k: P["perturb"][k] for k in ("scale_w", "scale_b")}, seed=P["perturb"]["seed"])
        zp = rt.logits(pa, enc, X)
        with pa.disabled():
            zd = rt.logits(pa, enc, X)
        r["disabled_reproduces_frozen"] = bool(torch.equal(zd, z0))
        r["start_pos_prefix_0_2_unchanged"] = bool(torch.equal(zp[:, :3], z0[:, :3]))
        r["content_positions_changed"] = not bool(torch.equal(zp[:, 3:], z0[:, 3:]))
        if arch != "R2":
            e5 = e1 = 0
            ts = []
            for row in dev20:
                t0 = time.time()
                a5, _ = rt.decode(row, 5, ad)
                ts.append(time.time() - t0)
                a1, _ = rt.decode(row, 1, ad)
                e5 += a5["tokens"] == base5[row["utterance_id"]]["tokens"]
                e1 += a1["tokens"] == base1[row["utterance_id"]]["tokens"]
            r["zero_init_dev_beam5_equal"] = [e5, len(dev20)]
            r["zero_init_dev_greedy_equal"] = [e1, len(dev20)]
            lat[f"{arch}_zero_beam5_s_per_utt"] = float(np.mean(ts))
        r["hooks_after"] = site_hook_count(b.model, 31)
        g2[arch] = r
        del ad, pa
    r2i, m1i = g2["R2"]["init_hashes"], g2["M1"]["init_hashes"]
    g2["M1_shared_init_equals_R2_seed42"] = all(r2i[k] == m1i[k] for k in R2_KEYS) and set(m1i) == set(R2_KEYS) | {"w_u"}
    res["G2"] = g2

    # ---- G3 M1 / R2 equivalence (W_u = 0, transplanted shared parameters)
    g3 = {}
    r2z, m1z = mk("R2"), mk("M1")
    g3["zero_init_shared_bitwise"] = all(torch.equal(r2z.state_dict()[k], m1z.state_dict()[k]) for k in R2_KEYS)
    g3["zero_init_logits_bitwise"] = bool(torch.equal(rt.logits(r2z, enc, X), rt.logits(m1z, enc, X)))
    r2p = mk("R2")
    perturb(r2p, P["perturb"]["scale_w"], P["perturb"]["scale_b"], seed=11)
    sel42 = ctl[42]["selected_epoch"]
    trained = torch.load(ctl[42]["dir"] / f"epoch{sel42}" / "adapter.pt")
    r2t = mk("R2")
    r2t.load_state_dict(trained)
    for tag, r2a in (("perturbed", r2p), ("trained_KL42_selected", r2t)):
        m1a = mk("M1")
        transplant_r2_into_m1(m1a, {k: v.to(dev) for k, v in r2a.state_dict().items()})
        names_r2, names_m1 = set(named_trainable(r2a)), set(named_trainable(m1a))
        d_r2, d_m1 = manual_delta(r2a, q, u), manual_delta(m1a, q, u)
        z_r2, z_m1 = rt.logits(r2a, enc, X), rt.logits(m1a, enc, X)
        g3[tag] = {"param_names_m1_minus_r2": sorted(names_m1 - names_r2), "param_names_r2_minus_m1": sorted(names_r2 - names_m1),
                   "shared_params_bitwise": all(torch.equal(r2a.state_dict()[k].to(dev), m1a.state_dict()[k].to(dev)) for k in R2_KEYS),
                   "update_bitwise": bool(torch.equal(d_r2, d_m1)), "update_max_abs": float((d_r2 - d_m1).abs().max()),
                   "update_nonzero": bool(d_r2.abs().max() > 0),
                   "logits_bitwise": bool(torch.equal(z_r2, z_m1)), "logits_max_abs": float((z_r2.float() - z_m1.float()).abs().max())}
        sp_r2, sp_m1 = spy_positions(rt, r2a, rt.dm[0], 1), spy_positions(rt, m1a, rt.dm[0], 1)
        g3[tag]["cache_positions_equal"] = sp_r2["seq"] == sp_m1["seq"] and sp_r2["first"] == [0, 1, 2, 3] and sp_r2["incremental_ok"]
        if tag == "trained_KL42_selected":
            rec5 = {x["id"]: x for x in read_jsonl(ctl[42]["dir"] / f"epoch{sel42}" / "dev_beam5.jsonl")}
            rec1 = {x["id"]: x for x in read_jsonl(ctl[42]["dir"] / f"epoch{sel42}" / "dev_greedy.jsonl")}
            e5 = sum(rt.decode(row, 5, m1a)[0]["tokens"] == rec5[row["utterance_id"]]["tokens"] for row in dev20)
            e1 = sum(rt.decode(row, 1, m1a)[0]["tokens"] == rec1[row["utterance_id"]]["tokens"] for row in dev20)
            g3[tag]["m1_decode_equals_recorded_r2_beam5"] = [e5, len(dev20)]
            g3[tag]["m1_decode_equals_recorded_r2_greedy"] = [e1, len(dev20)]
            g3[tag]["selected_epoch"] = sel42
        del m1a
    g3["hooks_after"] = site_hook_count(b.model, 31)
    res["G3"] = g3
    del r2z, m1z, r2p

    # ---- G4 gradients, single application, smoke training
    g4 = {}
    mb8 = sorted(rt.tm, key=lambda r: r["utterance_id"])[:rt.C1["training"]["micro_batch"]]
    for arch in ("R3", "M1"):
        r = {}
        ad = mk(arch)
        expected_names = {"rot.parametrizations.weight.original", "learned.weight", "learned.bias"} | ({"w_u"} if arch == "M1" else set())
        r["trainable_names_exact"] = set(named_trainable(ad)) == expected_names
        loss, st, _ = kl_step(b, ad, tiny, 1.0)
        loss.backward()
        r["init_loss_finite"] = bool(torch.isfinite(loss))
        r["init_grad_norms"] = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in named_trainable(ad).items()}
        need = ["learned.weight", "learned.bias"] + (["w_u"] if arch == "M1" else [])
        r["init_grads_reach_W_b" + ("_Wu" if arch == "M1" else "")] = all((r["init_grad_norms"][n] or 0) > 0 for n in need)
        r["backbone_grads_none"] = all(p.grad is None for p in b.model.parameters())
        pa = mk(arch)
        perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
        lp, sp_, _ = kl_step(b, pa, tiny, 1.0)
        lp.backward()
        r["perturbed_kl"] = sp_["kl"]
        r["perturbed_kl_positive"] = sp_["kl"] > 0
        r["perturbed_grad_norms"] = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in named_trainable(pa).items()}
        r["perturbed_grads_all_nonzero_finite"] = all(v is not None and v > 0 and np.isfinite(v) for v in r["perturbed_grad_norms"].values())
        # single application at the exact site: hooked logits == frozen forward with the manually recomputed update added
        for tag, (Xa, ea, qa, ua) in (("batch_a", (X[:2], enc[:2], *rt.site_record(enc[:2], X[:2])[:2])),
                                      ("batch_b", (X[2:], enc[2:], *rt.site_record(enc[2:], X[2:])[:2]))):
            c0 = pa.calls
            zh = rt.logits(pa, ea, Xa)
            r[f"single_application_{tag}"] = bool(torch.equal(zh, rt.site_delta_logits(ea, Xa, manual_delta(pa, qa, ua))))
            r[f"hook_calls_{tag}"] = pa.calls - c0
        # bounded smoke training on one training micro-batch (frozen optimizer settings)
        sm = mk(arch)
        opt = make_opt(sm, rt.C1, rt.C1["training"])
        start = {n: p.detach().clone() for n, p in named_trainable(sm).items()}
        losses = []
        for _ in range(P["smoke_steps"]):
            l_, s_, _ = kl_step(b, sm, mb8, 1.0)
            l_.backward()
            gl = {n: float(p.grad.norm()) for n, p in named_trainable(sm).items()}
            torch.nn.utils.clip_grad_norm_(sm.parameters(), rt.C1["training"]["clip_norm"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(l_))
        lf, sf, _ = kl_step(b, sm, mb8, 1.0, grad=False)
        r["smoke_losses"] = losses + [float(lf)]
        r["smoke_loss_reduced"] = float(lf) < losses[0]
        r["smoke_last_grad_norms"] = gl
        r["smoke_R_grad_nonzero_after_identity"] = gl["rot.parametrizations.weight.original"] > 0
        r["smoke_param_moved"] = {n: float((p.detach() - start[n]).norm()) for n, p in named_trainable(sm).items()}
        r["smoke_all_params_moved"] = all(v > 0 for v in r["smoke_param_moved"].values())
        r["smoke_orth_err"] = orth_error(sm)
        r["smoke_final_kl_positive"] = sf["kl"] > 0
        r["hooks_after"] = site_hook_count(b.model, 31)
        g4[arch] = r
        del ad, pa, sm, opt
    res["G4"] = g4

    # ---- G5 decoding semantics (perturbed adapters, BF16) + LID
    g5 = {}
    for arch in ("R3", "M1"):
        pa = mk(arch)
        perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
        lid_eq = 0
        ts, changed = [], []
        for row in dev20:
            inp = batch_model_inputs(b, [row["audio_path"]])
            with pa.active(b.model):
                lid_eq += frozen_lid(b, inp["input_features"], pa)["token"] == row["frozen_lid_token"]
            t0 = time.time()
            d5, _ = rt.decode(row, 5, pa)
            ts.append(time.time() - t0)
            changed.append(d5["tokens"] != base5[row["utterance_id"]]["tokens"])
        lat[f"{arch}_perturbed_beam5_s_per_utt"] = float(np.mean(ts))
        # leakage / determinism on an utterance whose decode the perturbed adapter actually changes
        rx = dev20[changed.index(True)] if any(changed) else rt.dm[0]
        ry = next(r for r in dev20 if r is not rx)
        sg, sb_ = spy_positions(rt, pa, rx, 1), spy_positions(rt, pa, rx, 5)
        x1, _ = rt.decode(rx, 5, pa)
        rt.decode(ry, 5, pa)
        x2, _ = rt.decode(rx, 5, pa)
        x3, _ = rt.decode(rx, 5, pa)
        fz_ = rt.decode(rx, 5, None)[0]
        g5[arch] = {"lid_unchanged_with_adapter": [lid_eq, len(dev20)], "perturbed_changed_dev_decodes": [sum(changed), len(dev20)],
                    "greedy_cache_positions": {k: v for k, v in sg.items() if k != "seq"},
                    "beam5_cache_positions": {k: v for k, v in sb_.items() if k != "seq"},
                    "cached_greedy_vs_tf_mismatch": [greedy_vs_tf(rt, pa, row) for row in rt.dm[:P["cache_rows"]]],
                    "beam5_vs_tf_bf16": [beam_vs_tf(rt, pa, row) for row in rt.dm[:P["cache_rows"]]],
                    "xyx_no_state_leakage": x1["tokens"] == x2["tokens"], "deterministic_repeat": x2["tokens"] == x3["tokens"],
                    "perturbed_changes_decode": x1["tokens"] != fz_["tokens"], "hooks_after": site_hook_count(b.model, 31)}
        del pa

    # ---- G6 GoldKL parity at zero init + historical regression (shared training path, R2 seed 42, epoch 1)
    g6 = {}
    vals = {}
    for arch in ("R2", "R3", "M1"):
        ad = mk(arch)
        loss, st, (zt, zs, Xk, Yk) = kl_step(b, ad, tiny, 1.0)
        with torch.no_grad():
            zplain = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
        loss.backward()
        gk = {n: p.grad.detach().clone() for n, p in named_trainable(ad).items()}
        ad2 = mk(arch)
        lce, _, _, _, _ = batch_loss(b, ad2, tiny, grad=True)
        lce.backward()
        gc = {n: p.grad.detach().clone() for n, p in named_trainable(ad2).items()}
        vals[arch] = {"ce": st["ce"], "kl": st["kl"], "n_mask": st["n_mask"], "mask": teacher_mask(zt, Yk).cpu()}
        g6[arch] = {"teacher_equals_frozen_forward": bool(torch.equal(zt, zplain)), "student_equals_teacher": bool(torch.equal(zs.detach(), zt)),
                    "initial_kl": st["kl"], "initial_kl_ok": abs(st["kl"]) <= T["initial_kl_max"], "n_mask": st["n_mask"],
                    "ce_equals_ce_path": st["ce"] == float(lce),
                    "init_grads_close_ce_path": all(torch.allclose(gk[n], gc[n], rtol=T["init_grad_rtol"], atol=T["init_grad_atol"]) for n in gc),
                    "init_grad_max_abs_diff": max(float((gk[n] - gc[n]).abs().max()) for n in gc),
                    "label_contract": bool((Yk[:, :3] == -100).all() and int((Yk == rt.eos).sum()) == len(tiny))}
        del ad, ad2
    g6["same_mask_all_archs"] = all(torch.equal(vals["R2"]["mask"], vals[a]["mask"]) for a in ("R3", "M1"))
    g6["same_initial_ce_all_archs"] = vals["R2"]["ce"] == vals["R3"]["ce"] == vals["M1"]["ce"]
    g6["same_initial_kl_all_archs"] = vals["R2"]["kl"] == vals["R3"]["kl"] == vals["M1"]["kl"]
    tr = seed_config(rt.C1, 42)["training"]
    a3 = mk("R2")
    ref1 = torch.load(ctl[42]["dir"] / "epoch1" / "adapter.pt")
    base_key = "rot.parametrizations.weight.0.base"
    base_eq = torch.equal(a3.state_dict()[base_key], ref1[base_key])
    opt = make_opt(a3, rt.C1, tr)
    lg = {"micro": [], "steps": []}
    h = goldkl_epoch(b, a3, opt, rt.tm, tr, 42, 1, C["objective"]["lambda_gold_kl"], 0, lg)
    mine = a3.state_dict()
    hist1 = ctl[42]["history"][1]
    g6["regression_R2_KL42_epoch1"] = {"init_base_equal": base_eq, "bitwise_equal": set(mine) == set(ref1) and all(torch.equal(mine[k], ref1[k]) for k in ref1),
                                       "max_abs_diff": max(float((mine[k] - ref1[k]).abs().max()) for k in ref1),
                                       "train_kl_equal": h["train_kl"] == hist1["train_kl"], "train_ce_equal": h["train_ce"] == hist1["train_ce"],
                                       "mask_positions_equal": h["mask_positions"] == hist1["mask_positions"],
                                       "supervised_tokens_equal": h["supervised_tokens"] == hist1["supervised_tokens"],
                                       "updates": h["updates_total"], "epoch_seconds": h["train_seconds"],
                                       "steps_logged": len(lg["steps"]), "max_orth_err": max(s["orth_err"] for s in lg["steps"])}
    del a3, opt
    res["G6"] = g6

    # ---- G7 resources
    tim = {}
    for arch in ("R2", "R3", "M1"):
        ad = mk(arch)
        ts, tt = [], []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            l_, s_, _ = kl_step(b, ad, mb8, 1.0)
            l_.backward()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
            tt.append(s_["teacher_s"])
            for p in ad.parameters():
                p.grad = None
        ck = pdir / f"smoke_{arch}_adapter.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), ck)
        tim[arch] = {"micro_batch_seconds": ts, "teacher_seconds": tt, "adapter_file_bytes": ck.stat().st_size,
                     "trainable_parameter_bytes_fp32": 4 * ad.n_trainable(), "param_plus_adam_state_bytes": 3 * 4 * ad.n_trainable()}
        del ad
    dev_per_utt = None
    t0 = time.time()
    z0a = mk("R3")
    for row in dev20[:5]:
        rt.decode(row, 5, z0a)
        rt.decode(row, 1, z0a)
    dev_per_utt = (time.time() - t0) / 5 + 0.15
    n_mb = -(-len(rt.tm) // tr["micro_batch"])
    proj = {a: 3 * n_mb * float(np.mean(tim[a]["micro_batch_seconds"][1:])) + 4 * len(rt.dm) * dev_per_utt + 300 for a in tim}
    g7 = {"timing": tim, "dev_seconds_per_utt_beam5_plus_greedy": dev_per_utt, "projected_job_seconds": proj, "decode_latency": lat,
          "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
          "authorization_present": out("AUTHORIZATION.json").exists()}

    # ---- G5 continued: float32 cached-vs-full semantic check (P2-1c amendment r1), BF16 recorded descriptively first
    from reft.train_supervised import encode
    fp_rows = rt.dm[:P["fp32_rows"]]
    cvf = {"bf16": {}, "fp32": {}}
    pads = {}
    for arch in ("R3", "M1"):
        pa = mk(arch)
        perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
        pads[arch] = pa
    for row in fp_rows:
        _, e1 = encode(b, [row["audio_path"]])
        Xs = row["prompt"] + base5[row["utterance_id"]]["tokens"][:-1]
        for arch, pa in pads.items():
            cvf["bf16"].setdefault(arch, []).append(cached_vs_full(b, pa, e1, Xs, T["fp32_argmax_margin"]))
    w_before_cast = weights_hash(b.model)
    b.model.float()  # in-memory only, last step of this job
    for row in fp_rows:
        inp = batch_model_inputs(b, [row["audio_path"]])
        with torch.inference_mode():
            e1 = b.model.model.encoder(inp["input_features"].float()).last_hidden_state
        Xs = row["prompt"] + base5[row["utterance_id"]]["tokens"][:-1]
        for arch, pa in pads.items():
            cvf["fp32"].setdefault(arch, []).append(cached_vs_full(b, pa, e1, Xs, T["fp32_argmax_margin"]))
    g5["cached_vs_full"] = cvf
    g5["hooks_after_all"] = site_hook_count(b.model, 31)
    res["G5"] = g5
    res["G7"] = g7
    res["weights_unchanged_before_fp32_cast"] = w_before_cast == rt.w0

    # ---- gates
    gates = {
        "G1_provenance": all(v for k, v in res["G1"].items() if k not in ("git_head", "git_dirty_paths")),
        "G2_parameters_identity": all(g2[a]["n_trainable"] == g2[a]["expected"] and g2[a]["orth_err"] <= T["orth_max_abs"] and g2[a]["W_equals_R"]
                                      and g2[a]["b_zero"] and g2[a]["module_output_zero_at_init"] and g2[a]["tf_logits_bitwise_frozen"]
                                      and g2[a]["disabled_reproduces_frozen"] and g2[a]["start_pos_prefix_0_2_unchanged"]
                                      and g2[a]["content_positions_changed"] and g2[a]["backbone_trainable"] == 0 and g2[a]["hooks_after"] == 0
                                      for a in ("R2", "R3", "M1"))
        and g2["M1"]["W_u_zero"] and g2["M1_shared_init_equals_R2_seed42"]
        and all(g2[a]["zero_init_dev_beam5_equal"][0] == len(dev20) and g2[a]["zero_init_dev_greedy_equal"][0] == len(dev20) for a in ARCHS),
        "G3_m1_r2_equivalence": g3["zero_init_shared_bitwise"] and g3["zero_init_logits_bitwise"] and g3["hooks_after"] == 0
        and all(g3[t]["param_names_m1_minus_r2"] == ["w_u"] and not g3[t]["param_names_r2_minus_m1"] and g3[t]["shared_params_bitwise"]
                and g3[t]["update_bitwise"] and g3[t]["update_nonzero"] and g3[t]["logits_bitwise"] and g3[t]["cache_positions_equal"]
                for t in ("perturbed", "trained_KL42_selected"))
        and g3["trained_KL42_selected"]["m1_decode_equals_recorded_r2_beam5"][0] == len(dev20)
        and g3["trained_KL42_selected"]["m1_decode_equals_recorded_r2_greedy"][0] == len(dev20),
        "G4_gradients": all(g4[a]["trainable_names_exact"] and g4[a]["init_loss_finite"] and g4[a]["backbone_grads_none"]
                            and g4[a]["init_grads_reach_W_b" + ("_Wu" if a == "M1" else "")] and g4[a]["perturbed_kl_positive"]
                            and g4[a]["perturbed_grads_all_nonzero_finite"] and g4[a]["single_application_batch_a"]
                            and g4[a]["single_application_batch_b"] and g4[a]["hook_calls_batch_a"] == 1 and g4[a]["hook_calls_batch_b"] == 1
                            and g4[a]["smoke_loss_reduced"] and g4[a]["smoke_R_grad_nonzero_after_identity"] and g4[a]["smoke_all_params_moved"]
                            and g4[a]["smoke_orth_err"] <= T["orth_max_abs"] and g4[a]["hooks_after"] == 0 for a in ARCHS),
        "G5_decoding": all(g5[a]["lid_unchanged_with_adapter"][0] == len(dev20) and g5[a]["greedy_cache_positions"]["first"] == [0, 1, 2, 3]
                           and g5[a]["greedy_cache_positions"]["incremental_ok"] and g5[a]["beam5_cache_positions"]["first"] == [0, 1, 2, 3]
                           and g5[a]["beam5_cache_positions"]["incremental_ok"]
                           and all(x <= T["cached_greedy_vs_tf_max_mismatch"] for x in g5[a]["cached_greedy_vs_tf_mismatch"])
                           and g5[a]["xyx_no_state_leakage"] and g5[a]["deterministic_repeat"] and g5[a]["perturbed_changes_decode"]
                           and g5[a]["hooks_after"] == 0
                           and all(x["max_abs_logp_observed"] <= T["fp32_cached_vs_full_max_abs_logp"] and x["argmax_agree_where_sure"]
                                   and x["positions_ok"] and x["finite"] for x in cvf["fp32"][a]) for a in ARCHS)
        and g5["hooks_after_all"] == 0,
        "G6_goldkl_parity": all(g6[a]["teacher_equals_frozen_forward"] and g6[a]["student_equals_teacher"] and g6[a]["initial_kl_ok"]
                                and g6[a]["ce_equals_ce_path"] and g6[a]["init_grads_close_ce_path"] and g6[a]["label_contract"]
                                for a in ("R2", "R3", "M1"))
        and g6["same_mask_all_archs"] and g6["same_initial_ce_all_archs"] and g6["same_initial_kl_all_archs"]
        and g6["regression_R2_KL42_epoch1"]["init_base_equal"] and g6["regression_R2_KL42_epoch1"]["bitwise_equal"],
        "G7_resources_freeze": all(v <= P["max_projected_job_seconds"] for v in proj.values()) and g7["peak_reserved_bytes"] < 38e9
        and g7["authorization_present"] and res["weights_unchanged_before_fp32_cast"] and res["G1"]["freeze_verified"],
    }
    verdict = "ARCH_S0_READY_FOR_S1" if all(gates.values()) else "ARCH_S0_BLOCKED_" + "_".join(k.split("_")[0] for k, v in gates.items() if not v)
    rec = {"schema": "reft_arch_s0_preflight_v1", "utc": now(), "dry_run": bool(DRY), "git": git_state(), "slurm": slurm(), "env": rt.env,
           "cuda_libs": rt.libs, "results": res, "gates": gates, "verdict": verdict, "wall_seconds": time.time() - t_all}
    write_json(out(f"preflight_{jid}.json"), rec)
    print(json.dumps({"verdict": verdict, "gates": gates, "G2": {a: {k: v for k, v in g2[a].items() if k not in ("init_hashes", "shapes")}
                                                                 if isinstance(g2[a], dict) else g2[a] for a in g2},
                      "G3": g3, "G4": g4, "G5": g5, "G6": {k: v for k, v in g6.items()}, "G7": g7}, indent=1, default=str)[:60000])


def newest_preflight():
    ps = sorted(out().glob("preflight_*.json"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
    return ps[-1] if ps else None


# ====================================================================== train (GPU)

def run_train(arch: str, seed: int):
    import torch
    from reft.arch_s0s1_adapters import EXPECTED, R2_KEYS, init_record, make_arch_adapter, orth_error
    from reft.auto_policy import frozen_lid
    from reft.p2_1b_train import seed_config
    if DRY:
        raise SystemExit("S1 training never runs in dry-run mode")
    C = ca()
    assert arch in ARCHS and seed in C["seeds"]
    fz = check_freeze()
    if not out("AUTHORIZATION.json").exists():
        raise SystemExit("S0/S1 not authorized")
    pf = newest_preflight()
    if pf is None or read_json(pf)["verdict"] != "ARCH_S0_READY_FOR_S1" or read_json(pf)["dry_run"]:
        raise SystemExit(f"S0 preflight not READY: {pf}")
    od = out("training", f"{arch}_seed{seed}")
    if od.exists():
        raise SystemExit(f"{od} exists; runs are write-once")
    ctl = control_runs()[seed]
    torch.manual_seed(seed)                       # P2-1b order: seed -> backbone -> adapter
    rt = Runtime()
    b = rt.b
    tr = seed_config(rt.C1, seed)["training"]
    ad = make_arch_adapter(arch, b.model, seed, b.device)
    if ad.n_trainable() != EXPECTED[arch]:
        raise SystemExit(f"{arch} has {ad.n_trainable()} trainable parameters, expected {EXPECTED[arch]}")
    init = init_record(ad)
    ref1 = torch.load(ctl["dir"] / "epoch1" / "adapter.pt")
    base_key = "rot.parametrizations.weight.0.base"
    pairing = {"orth_err_init": orth_error(ad)}
    if arch == "M1":
        # by construction M1's R (incl. base), W_h and b at init are R2 seed s's initial tensors; base never changes in training
        pairing["m1_base_equal_R2_seed"] = bool(torch.equal(ad.state_dict()[base_key], ref1[base_key]))
        if not pairing["m1_base_equal_R2_seed"]:
            raise SystemExit("M1 initial parametrization base differs from R2-GoldKL of the same seed (environment drift)")
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
        h = goldkl_epoch(b, ad, opt, rt.tm, tr, seed, ep, C["objective"]["lambda_gold_kl"], steps, lg)
        steps = h["updates_total"]
        write_jsonl(od / "diagnostics" / f"epoch{ep}_steps.jsonl", lg["steps"])
        write_jsonl(od / "diagnostics" / f"epoch{ep}_microbatches.jsonl", lg["micro"])
        (od / f"epoch{ep}").mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), od / f"epoch{ep}" / "adapter.pt")
        s_, _ = dev_decode(ep)
        hk = ctl["history"][ep]
        st = lg["steps"]
        diag = {"orth_err_max": max(x["orth_err"] for x in st),
                "grad_norm_preclip_mean": {n: float(np.mean([x["grad_norm_preclip"][n] for x in st])) for n in st[0]["grad_norm_preclip"]},
                "update_norm_mean": {n: float(np.mean([x["update_norm"][n] for x in st])) for n in st[0]["update_norm"]},
                "param_norm_end": st[-1]["param_norm"],
                "paired_with_R2_KL": {"mask_positions_equal": h["mask_positions"] == hk["mask_positions"],
                                      "supervised_tokens_equal": h["supervised_tokens"] == hk["supervised_tokens"],
                                      "updates_equal": h["updates_total"] == hk["updates_total"]}}
        hist.append({**h, "diagnostics": diag, "dev_decode_seconds": s_, "adapter_sha256": sha(od / f"epoch{ep}" / "adapter.pt"),
                     "adapter_file_bytes": (od / f"epoch{ep}" / "adapter.pt").stat().st_size})
        print(json.dumps({k: v for k, v in hist[-1].items() if k != "diagnostics"} | {"orth_err_max": diag["orth_err_max"],
                                                                                      "paired": diag["paired_with_R2_KL"]}), flush=True)
    w1 = weights_hash(b.model)
    lat = {reg: float(np.mean([x["seconds"] for x in read_jsonl(od / f"epoch{tr['epochs']}" / f"dev_{reg}.jsonl")])) for reg in ("beam5", "greedy")}
    write_json(od / "manifest.json", {
        "schema": "reft_arch_s1_run_v1", "utc": now(), **fz, "git": git_state(), "env": rt.env, "slurm": slurm(), "arch": arch, "seed": seed,
        "lambda_gold_kl": C["objective"]["lambda_gold_kl"], "architecture": C["architectures"][arch], "training": tr,
        "lr": rt.C1["methods"]["R2"]["lr"], "history": hist, "weights_before": rt.w0, "weights_after": w1,
        "weights_unchanged": rt.w0 == w1 == rt.expected_w, "cuda_libs": rt.libs, "preflight": pf.name, "preflight_sha256": sha(pf),
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
    from reft.arch_s0s1_adapters import EXPECTED, make_arch_adapter
    fz = check_freeze()
    if out("verify.json").exists():
        raise SystemExit("verify.json exists; write-once")
    rt = Runtime()
    res = {}
    for arch in ARCHS:
        for s in SEEDS:
            rd = out("training", f"{arch}_seed{s}")
            man = read_json(rd / "manifest.json")
            res[rd.name] = {}
            for h in man["history"][1:]:
                p = rd / f"epoch{h['epoch']}" / "adapter.pt"
                torch.manual_seed(s)
                ad = make_arch_adapter(arch, rt.b.model, s, rt.b.device)
                ad.load_state_dict(torch.load(p))
                rec = {x["id"]: x for x in read_jsonl(rd / f"epoch{h['epoch']}" / "dev_beam5.jsonl")}
                eq = sum(rt.decode(r, 5, ad)[0]["tokens"] == rec[r["utterance_id"]]["tokens"] for r in rt.dm[:20])
                res[rd.name][h["epoch"]] = {"sha_ok": sha(p) == h["adapter_sha256"], "reload_equal": [eq, 20],
                                            "n_trainable_ok": ad.n_trainable() == EXPECTED[arch]}
    w1 = weights_hash(rt.b.model)
    ok = rt.w0 == w1 == rt.expected_w and all(v["sha_ok"] and v["reload_equal"][0] == 20 and v["n_trainable_ok"]
                                              for x in res.values() for v in x.values())
    write_json(out("verify.json"), {"schema": "reft_arch_s1_verify_v1", "utc": now(), **fz, "slurm": slurm(), "cuda_libs": rt.libs,
                                    "results": res, "weights_unchanged": rt.w0 == w1, "all_ok": ok,
                                    "historical_R2_GoldKL": "verified in P2-1b (p2_1b/verify.json all_ok)"
                                    if read_json(art("p2_1b/verify.json"))["all_ok"] else "NOT OK"})
    print(json.dumps({"all_ok": ok, "results": res}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "freeze-amend", "preflight", "train", "verify"])
    ap.add_argument("--arch", choices=list(ARCHS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--doc")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        run_train(a.arch, a.seed)
    elif a.cmd == "freeze-amend":
        run_freeze_amend(a.doc)
    else:
        {"freeze": run_freeze, "preflight": run_preflight, "verify": run_verify}[a.cmd]()


if __name__ == "__main__":
    main()
