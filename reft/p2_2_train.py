#!/usr/bin/env python
"""P2-2 on-policy continuation preservation: freeze, P2-2A preflight (training-only coverage + engineering gates),
M2 training, reload verification. docs/reft/P2_2_PROTOCOL.md, docs/reft/P2_2_DECISION_RULES.md.

M2 loss per micro-batch: P2-1b ``kl_step`` (gold CE + 1.0 * gold-prefix KL on the teacher-correct mask, unchanged) plus
lambda_op * L_OP over the on-policy events of the same utterances (reft/p2_2_alignment.py). Events come from beam-5
rollouts of the adapter frozen at the start of each epoch, on training audio only; teacher targets are frozen B0 replayed
from exactly the generated prefix (P2-1c replay semantics: GenerationMixin.generate, canonical suppression, begin
suppression pinned at absolute position 4). No gradient through generation or event selection.

Commands: freeze | preflight (GPU) | train --seed S (GPU) | verify (GPU)
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, firewall, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import git_state, now, sha, slurm, weights_hash  # noqa: E402

CONF = ROOT / "configs/reft/p2_2_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/P2_2_PROTOCOL.md", "docs/reft/P2_2_DECISION_RULES.md", "configs/reft/p2_2_frozen.yaml",
                  "reft/p2_2_alignment.py", "reft/p2_2_train.py", "reft/p2_2_evaluate.py", "tests/reft/test_p2_2.py",
                  "slurm/reft_p2_2.sbatch", "slurm/reft_p2_2_cpu.sbatch", "slurm/reft_p2_2_submit.sh"]
SEEDS = (42, 43, 44)


def c22() -> dict:
    return yaml.safe_load(CONF.read_text())


def out(*p) -> Path:
    return art("p2_2", *p)


def control_paths() -> dict:
    """Selected checkpoints of the reused controls (C0 = CE, C1 = GoldKL), with their recorded adapter hashes."""
    res = {}
    for grp in ("C0", "C1"):
        for name, run in c22()["controls"][grp].items():
            rd = art(run)
            ep = read_json(rd / "selection.json")["selected"]["epoch"]
            h = next(h for h in read_json(rd / "manifest.json")["history"] if h["epoch"] == ep)
            res[name] = {"run": run, "epoch": ep, "seed": int(name[2:]), "adapter": rd / f"epoch{ep}" / "adapter.pt",
                         "decode": rd / f"epoch{ep}" / "dev_beam5.jsonl", "expected_sha": h["adapter_sha256"]}
    return res


def frozen_inputs() -> list[str]:
    L = ["p2_0/train_manifest.json", "p2_0/dev_manifest.json", "p2_0/frozen_baseline_dev150_beam5.jsonl",
         "p2_0/frozen_baseline_dev150_greedy.jsonl", "p1/targets.json", "p2_1b/evaluation.json", "p2_1b/report.json",
         "p2_1b/verify.json", "p2_1/verify.json", "p2_1c/analysis_r1.json", "p2_1c/anchors_private.json",
         "p2_1c/anchors_inference.json", "p2_1c/freeze_r1.json"]
    for c in control_paths().values():
        L += [f"{c['run']}/selection.json", f"{c['run']}/manifest.json"]
        L += [f"{c['run']}/epoch{e}/adapter.pt" for e in (1, 2, 3)]
        L += [f"{c['run']}/epoch{e}/dev_{r}.jsonl" for e in (0, 1, 2, 3) for r in ("beam5", "greedy")]
    return L


def run_freeze():
    p = out("freeze.json")
    if p.exists():
        raise SystemExit(f"{p} exists; write-once")
    from reft.p2_1b_train import check_freeze as p21b
    from reft.p2_1c_continuation_audit import check_freeze as p21c
    g = git_state()
    if g["branch"] != "cs-asr-reft":
        raise SystemExit(f"wrong branch {g['branch']}")
    ctl = {n: sha(c["adapter"]) == c["expected_sha"] for n, c in control_paths().items()}
    if not all(ctl.values()):
        raise SystemExit(f"control adapter hash mismatch: {ctl}")
    rec = {"schema": "reft_p2_2_freeze_v1", "utc": now(), "git": g, "sources": {s: sha(ROOT / s) for s in FREEZE_SOURCES},
           "inputs": {i: sha(art(i)) for i in frozen_inputs()}, "controls_match_recorded_sha": ctl,
           "p2_1b_chain": p21b(), "p2_1c_chain": {k: v for k, v in p21c().items() if k != "historical_chain_checked_here"}}
    write_json(p, rec)
    write_json(out("AUTHORIZATION.json"), {"utc": now(), "stage": "P2-2 (P2-2A + P2-2B)", "authorized_by": "project owner (2026-10-15 prompt)",
                                           "runs": [f"M2_seed{s}" for s in SEEDS], "max_new_training_runs": 3,
                                           "conditions": "P2-2B only if every P2-2A gate passes; no P2-3/P3/M1/E1/LoRA; no sweeps"})
    print(p)


def check_freeze() -> dict:
    p = out("freeze.json")
    rec = read_json(p)
    bad = [s for s, h in rec["sources"].items() if sha(ROOT / s) != h] + [i for i, h in rec["inputs"].items() if sha(art(i)) != h]
    if bad:
        raise SystemExit(f"P2-2 frozen sources/inputs changed: {bad}")
    from reft.p2_1b_train import check_freeze as p21b
    if p21b() != rec["p2_1b_chain"]:
        raise SystemExit("P2-1b / P2-0 chain changed")
    return {"p2_2_freeze_sha256": sha(p), "p2_1b_freeze_sha256": rec["p2_1b_chain"]["p2_1b_freeze_sha256"],
            "p2_0_chain_head": rec["p2_1b_chain"]["p2_0_chain_head"], "p2_1c_freeze_sha256": rec["p2_1c_chain"]["p2_1c_freeze_sha256"]}


# ====================================================================== runtime (GPU)

class RT:
    """Backbone, tokenizer, replay settings, training references (training data only; dev references firewalled)."""

    def __init__(self):
        import torch
        from reft.p2_1b_train import canonical_env_or_exit
        from reft.text import tok_bytes_fn
        from reft.common import cfg, fillers
        from reft.train_supervised import GPU_FORBIDDEN, c21, load_backbone
        self.torch, self.C, self.C1 = torch, c22(), c21()
        self.tm = read_json(art("p2_0/train_manifest.json"))["rows"]
        self.dm = read_json(art("p2_0/dev_manifest.json"))["rows"]
        self.tg = read_json(art("p1/targets.json"))["targets"]
        firewall(GPU_FORBIDDEN)
        self.b, self.env = load_backbone()
        self.libs = canonical_env_or_exit()
        self.w0 = weights_hash(self.b.model)
        self.expected_w = cfg()["model"]["expected_weights_hash"]
        gj = json.loads((Path(cfg()["model"]["path"]) / "generation_config.json").read_text())
        gen = self.b.model.generation_config
        assert list(gen.suppress_tokens) == gj["suppress_tokens"] and list(gen.begin_suppress_tokens) == gj["begin_suppress_tokens"]
        self.supp, self.begin_supp = list(gj["suppress_tokens"]), list(gj["begin_suppress_tokens"])
        self.tok = self.b.processor.tokenizer
        self.eos, self.pad, self.bos, self.sot = self.tok.eos_token_id, gen.pad_token_id, gen.bos_token_id, gen.decoder_start_token_id
        self.tb = tok_bytes_fn(self.tok)
        self.inv = {v: k for k, v in gen.lang_to_id.items()}
        self.FILL = fillers()
        self.n_teacher = self.n_teacher_steps = 0

    def encode_text(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def lid(self, row):
        return {"token": row["frozen_lid_token"], "language": self.inv[row["frozen_lid_token"]]}

    def rollout(self, rows, ad, beams=5):
        """Canonical frozen-LID decode (beam-5) of training audio with adapter `ad` (None = frozen B0)."""
        from reft.auto_policy import decode_frozen_lid
        res = []
        for r in rows:
            rec, _ = decode_frozen_lid(self.b, r["audio_path"], self.lid(r), beams, ad)
            assert rec["prompt"] == r["prompt"], "rollout prompt != frozen-LID prompt"
            res.append({"id": r["utterance_id"], **rec})
        return res

    def replay(self, inp, P, max_new):
        """Frozen B0 beam-5 continuation from the forced decoder prefix P (P2-1c replay semantics). Top beam suffix."""
        torch = self.torch
        from transformers import GenerationConfig
        from transformers.generation.logits_process import (LogitsProcessorList, SuppressTokensAtBeginLogitsProcessor,
                                                            SuppressTokensLogitsProcessor)
        from transformers.generation.utils import GenerationMixin
        R = self.C["replay"]
        dev = self.b.device
        gc = GenerationConfig(num_beams=R["num_beams"], num_return_sequences=1, do_sample=False, max_length=len(P) + max_new,
                              length_penalty=R["length_penalty"], early_stopping=R["early_stopping"], eos_token_id=self.eos,
                              pad_token_id=self.pad, bos_token_id=self.bos, decoder_start_token_id=self.sot,
                              return_dict_in_generate=True, output_scores=True, use_cache=True)
        procs = LogitsProcessorList([SuppressTokensAtBeginLogitsProcessor(self.begin_supp, begin_index=R["begin_suppress_index"], device=dev),
                                     SuppressTokensLogitsProcessor(self.supp, device=dev)])
        with torch.inference_mode():
            o = GenerationMixin.generate(self.b.model, **inp, decoder_input_ids=torch.tensor([P], device=dev),
                                         generation_config=gc, logits_processor=procs)
        s = o.sequences[0].tolist()
        assert s[:len(P)] == P, "forced prefix not preserved"
        suf = s[len(P):]
        if self.eos in suf:
            suf = suf[:suf.index(self.eos) + 1]
        self.n_teacher += 1
        self.n_teacher_steps += len(s) - len(P)
        return suf

    # ---------------------------------------------------------------- events
    def hyp_view(self, uid, rec):
        from reft.p2_1c_continuation_audit import boundaries
        from reft.text import one
        content = [t for t in rec["tokens"] if t != self.eos]
        ms, full = boundaries([self.tb(t) for t in content])
        o = one(self.tg[uid]["target_text"], rec["text"])
        return {"content": content, "ms": ms, "units": full, "o": o, "ok": full == o["hyp"]}

    def build_events(self, rows, hyps: dict, b0hyps: dict, seed: int, epoch: int) -> tuple[list, dict]:
        """On-policy events for every training utterance (<= 1 per category), teacher replays from B0. Returns
        (events, stats). Every opportunity and every chosen event (including U) is recorded."""
        from csasr.models.whisper import batch_model_inputs
        from reft.p2_2_alignment import (CATS, locate, location_label_consistent, opportunities, order_key, prefix_class,
                                         reference_target, teacher_target)
        E = self.C["events"]
        events, opp = [], collections.Counter()
        t0 = time.time()
        for row in rows:
            uid = row["utterance_id"]
            hv = self.hyp_view(uid, hyps[uid])
            o = hv["o"]
            ops = opportunities(o["ref"], o["tags"], self.FILL, E)
            for c in CATS:
                opp[c] += len(ops[c])
            if not any(ops.values()):
                continue
            bv = self.hyp_view(uid, b0hyps[uid])
            inp = None
            for c in CATS:
                if not ops[c]:
                    continue
                cand = sorted(ops[c], key=lambda r: order_key(E["order_salt"], seed, epoch, uid, c, r))
                ev = {"uid": uid, "dialogue": row["dialogue_id"], "cat": c, "n_candidates": len(cand)}
                loc, why = None, collections.Counter()
                if hv["ok"]:
                    for r in cand:
                        loc = locate(o, hv["ms"], r, E["context_units"])
                        if loc["ok"]:
                            ev["r"] = r
                            break
                        why[loc["reason"]] += 1
                ev["locate_failures"] = dict(why)
                if loc is None or not loc["ok"]:
                    events.append({**ev, "status": "U", "reason": "no_located_candidate" if hv["ok"] else "hyp_units_mismatch",
                                   "located": False})
                    continue
                r, k, h = ev["r"], loc["k"], loc["h"]
                prefix = hv["content"][:k]
                pbytes = b"".join(self.tb(t) for t in prefix)
                punits = hv["units"][:h]
                bloc = locate(bv["o"], bv["ms"], r, E["context_units"]) if bv["ok"] else None
                pc = prefix_class(prefix, punits, bloc, bv["content"], bv["units"])
                if inp is None:
                    inp = batch_model_inputs(self.b, [row["audio_path"]])
                cont = self.replay(inp, row["prompt"] + prefix, E["teacher_max_new_tokens"])
                T = teacher_target(o["ref"], r, pbytes, punits, cont, self.tb, self.eos, self.tg[uid]["target_text"], E)
                ev.update({"located": True, "k": k, "h": h, "prefix": prefix, "prefix_class": pc,
                           "teacher_rel_retention": T.get("teacher_rel_retention"), "teacher_reason": T.get("reason")})
                if T["ok"]:
                    ev.update({"status": "T", "targets": T["targets"], "scope": T["scope"]})
                else:
                    Rr = reference_target(self.tg[uid]["target_text"], o["ref"], r, pbytes, punits, self.encode_text, self.tb, E)
                    if Rr["ok"]:
                        ev.update({"status": "R", "targets": Rr["targets"], "scope": Rr["scope"]})
                    else:
                        ev.update({"status": "U", "reason": f"{T['reason']}+{Rr['reason']}"})
                if ev["status"] in ("T", "R"):
                    tbytes = b"".join(self.tb(t) for t in ev["targets"])
                    full_scope = ev["scope"][1] - ev["scope"][0]
                    trunc = len(ev["targets"]) == E["max_tokens_per_event"]
                    ev["label_consistent"] = location_label_consistent(o["ref"], pbytes, punits, tbytes, r, full_scope, trunc)
                    ev["n_tok"] = len(ev["targets"])
                events.append(ev)
        st = event_stats(events, opp)
        st["seconds"] = time.time() - t0
        return events, st


def event_stats(events: list, opp: dict) -> dict:
    act = [e for e in events if e["status"] in ("T", "R")]
    loc = [e for e in events if e.get("located")]
    by = collections.Counter((e["cat"], e["status"]) for e in events)
    reasons = collections.Counter(e.get("reason") or "" for e in events if e["status"] == "U")
    teach_fail = collections.Counter(e.get("teacher_reason") for e in loc if e.get("teacher_reason"))
    pcs = collections.Counter(e["prefix_class"] for e in act)
    sb_loc = [e for e in loc if e["cat"] == "SB"]
    return {"opportunities": dict(opp), "events_chosen": len(events), "located": len(loc), "active": len(act),
            "T": sum(e["status"] == "T" for e in events), "R": sum(e["status"] == "R" for e in events),
            "U": sum(e["status"] == "U" for e in events), "by_cat_status": {f"{c}/{s}": n for (c, s), n in sorted(by.items())},
            "u_reasons": dict(reasons), "teacher_invalid_reasons": dict(teach_fail),
            "active_prefix_class": dict(pcs), "active_non_identity": sum(e["prefix_class"] != "identical_tokens" for e in act),
            "active_utterances": len({e["uid"] for e in act}), "active_dialogues": len({e["dialogue"] for e in act}),
            "active_sb": sum(e["cat"] == "SB" for e in act),
            "valid_share_of_located": len(act) / len(loc) if loc else None,
            "sb_share_of_opportunities": sum(e["cat"] == "SB" for e in act) / opp["SB"] if opp.get("SB") else None,
            "b0_drops_under_own_prefix_sb": sum(1 for e in sb_loc if (e.get("teacher_rel_retention") is not None and e["teacher_rel_retention"] <= 0.2)),
            "sb_located": len(sb_loc),
            "label_mismatches": sum(1 for e in act if not e.get("label_consistent", False)),
            "tokens_per_event": dict(collections.Counter(e["n_tok"] for e in act)),
            "prefix_len_quartiles": [float(x) for x in np.quantile([e["k"] for e in act], [0.25, 0.5, 0.75])] if act else None}


def coverage_gate(st: dict, G: dict) -> dict:
    g = {"valid_events": st["active"] >= G["min_valid_events"], "utterances": st["active_utterances"] >= G["min_utterances"],
         "dialogues": st["active_dialogues"] >= G["min_dialogues"], "sb_events": st["active_sb"] >= G["min_valid_sb_events"],
         "non_b0_prefix_events": st["active_non_identity"] >= G["min_valid_non_b0_prefix_events"],
         "label_mismatches": st["label_mismatches"] <= G["max_location_label_mismatches"]}
    return {**g, "pass": all(g.values())}


# ====================================================================== on-policy loss

def op_step(R: RT, ad, rows, evs, grad=True):
    """L_OP for the active events of one micro-batch. Per event: token-mean KL(p0 || p_theta) (T) or token-mean CE (R) at
    the positions predicting its targets after [prompt + generated prefix]; L_OP = mean over events."""
    import torch
    from reft.train_supervised import encode, tf_forward
    if not evs:
        return None, {"n_T": 0, "n_R": 0, "n_tok": 0}
    _, enc = encode(R.b, [r["audio_path"] for r in rows])
    ridx = {r["utterance_id"]: i for i, r in enumerate(rows)}
    pr = {r["utterance_id"]: r["prompt"] for r in rows}
    seqs, plen = [], []
    for e in evs:
        P = pr[e["uid"]] + e["prefix"]
        seqs.append(P + e["targets"][:-1])
        plen.append(len(P))
    T = max(len(s) for s in seqs)
    X = torch.full((len(seqs), T), R.eos, dtype=torch.long)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor(s)
    X = X.to(R.b.device)
    enc_e = enc[torch.tensor([ridx[e["uid"]] for e in evs], device=enc.device)]
    with torch.no_grad():
        zt = R.b.model(encoder_outputs=(enc_e,), decoder_input_ids=X, use_cache=False).logits
    zs = tf_forward(R.b, ad, enc_e, X, grad=grad)
    from reft.p2_2_alignment import op_event_losses
    kinds = [e["status"] for e in evs]
    losses, L = op_event_losses(zs, zt, plen, [e["targets"] for e in evs], kinds)
    tv = [float(l_.detach()) for l_, k in zip(losses, kinds) if k == "T"]
    rv = [float(l_.detach()) for l_, k in zip(losses, kinds) if k == "R"]
    return L, {"n_T": len(tv), "n_R": len(rv), "T_sum": sum(tv), "R_sum": sum(rv), "n_tok": sum(len(e["targets"]) for e in evs),
               "per_event": [float(x.detach()) for x in losses]}


def m2_epoch(R: RT, ad, opt, tm, tr, seed, ep, ev_by_uid, lam_kl, lam_op, steps0=0):
    """One M2 epoch: P2-1b batch order and accumulation (partial final step), kl_step unchanged, plus lambda_op * L_OP."""
    import torch
    from reft.p2_1b_train import kl_step
    mb, acc = tr["micro_batch"], tr["grad_accum"]
    order = np.random.default_rng(seed + ep).permutation(len(tm))
    batches = [[tm[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]
    torch.cuda.synchronize()
    t0 = time.time()
    s = collections.Counter()
    pending, gn, gop, steps, op_t = 0, [], [], steps0, 0.0
    for k, rows in enumerate(batches):
        loss, st, _ = kl_step(R.b, ad, rows, lam_kl)
        (loss / acc).backward()
        s["ce"] += st["ce"] * st["n_tok"]
        s["kl"] += st["kl"] * st["n_mask"]
        s["n_mask"] += st["n_mask"]
        s["n_tok"] += st["n_tok"]
        s["teacher_s"] += st["teacher_s"]
        evs = [e for r in rows for e in ev_by_uid.get(r["utterance_id"], []) if e["status"] in ("T", "R")]
        if evs and lam_op > 0:
            t1 = time.time()
            before = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in ad.parameters()]
            lop, ost = op_step(R, ad, rows, evs)
            (lam_op * lop / acc).backward()
            gop.append(float(torch.sqrt(sum(((p.grad - b_) ** 2).sum() for p, b_ in zip(ad.parameters(), before)))))
            torch.cuda.synchronize()
            op_t += time.time() - t1
            s["op_events"] += len(evs)
            s["op_T"] += ost["n_T"]
            s["op_R"] += ost["n_R"]
            s["op_T_sum"] += ost["T_sum"]
            s["op_R_sum"] += ost["R_sum"]
            s["op_tok"] += ost["n_tok"]
            s["op_loss_sum"] += float(lop)
            s["op_batches"] += 1
            s["op_non_identity_events"] += sum(e["prefix_class"] != "identical_tokens" for e in evs)
        pending += 1
        if pending == acc or k == len(batches) - 1:
            gn.append(float(torch.nn.utils.clip_grad_norm_(ad.parameters(), tr["clip_norm"])))
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
    torch.cuda.synchronize()
    return {"epoch": ep, "train_ce": s["ce"] / s["n_tok"], "train_goldkl": s["kl"] / s["n_mask"] if s["n_mask"] else None,
            "mask_positions": s["n_mask"], "supervised_tokens": s["n_tok"], "updates_total": steps,
            "op_events": s["op_events"], "op_T": s["op_T"], "op_R": s["op_R"], "op_tokens": s["op_tok"],
            "op_T_kl_mean": s["op_T_sum"] / s["op_T"] if s["op_T"] else None, "op_R_ce_mean": s["op_R_sum"] / s["op_R"] if s["op_R"] else None,
            "op_loss_batch_mean": s["op_loss_sum"] / s["op_batches"] if s["op_batches"] else None,
            "op_non_identity_events": s["op_non_identity_events"],
            "grad_norm_mean": float(np.mean(gn)), "grad_norm_max": float(np.max(gn)),
            "op_grad_norm_mean": float(np.mean(gop)) if gop else 0.0, "op_grad_norm_max": float(np.max(gop)) if gop else 0.0,
            "train_seconds": time.time() - t0, "op_seconds": op_t, "teacher_forward_seconds": s["teacher_s"]}


# ====================================================================== preflight (GPU)

def run_preflight():
    import torch
    from reft.adapters import make_adapter, site_hook_count
    from reft.auto_policy import decode_frozen_lid
    from reft.p2_1b_train import seed_config
    from reft.train_supervised import randomize
    fz = check_freeze()
    t_all = time.time()
    R = RT()
    C = R.C
    torch.cuda.reset_peak_memory_stats()
    res = {"freeze": fz}
    ctl = control_paths()
    res["G1"] = {"branch": git_state()["branch"] == "cs-asr-reft", "controls_sha": all(sha(c["adapter"]) == c["expected_sha"] for c in ctl.values()),
                 "weights_expected": R.w0 == R.expected_w, "no_training_outputs_yet": not out("training").exists(),
                 "pinned_baseline": sha(art("p2_0/frozen_baseline_dev150_beam5.jsonl")) == R.C1["frozen_baseline"]["beam5_sha256"]}
    # ---- P2-2A rollouts on training audio: B0, CE42, KL42 (existing checkpoints; no training)
    ads = {"B0": None}
    for n in ("CE42", "KL42"):
        a = make_adapter("R2", R.b.model, seed_config(R.C1, 42), R.b.device)
        a.load_state_dict(torch.load(ctl[n]["adapter"]))
        ads[n] = a
    rolls, tro = {}, {}
    for n, a in ads.items():
        p = out("preflight_rollouts", f"{n}.jsonl")
        t0 = time.time()
        rolls[n] = {x["id"]: x for x in R.rollout(R.tm, a)}
        tro[n] = time.time() - t0
        write_jsonl(p, list(rolls[n].values()))
    cov, evs_all = {}, {}
    for n in ("B0", "CE42", "KL42"):
        evs, st = R.build_events(R.tm, rolls[n], rolls["B0"], 42, 0)
        evs_all[n] = evs
        write_jsonl(out("preflight_rollouts", f"{n}_events.jsonl"), evs)
        cov[n] = {**st, "rollout_seconds": tro[n], "gate": coverage_gate(st, C["coverage_gate"]) if n != "B0" else None}
    coverage_ok = all(cov[n]["gate"]["pass"] for n in ("CE42", "KL42"))
    # ---- zero-update checks: identity adapter reproduces B0 (training rollouts and dev beam-5/greedy)
    z = make_adapter("R2", R.b.model, seed_config(R.C1, 42), R.b.device)
    tr20 = R.rollout(R.tm[:20], z)
    base5 = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}
    base1 = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_greedy.jsonl"))}
    eq5 = eq1 = 0
    t_dev = time.time()
    for r in R.dm[:20]:
        a5, _ = decode_frozen_lid(R.b, r["audio_path"], R.lid(r), 5, z)
        a1, _ = decode_frozen_lid(R.b, r["audio_path"], R.lid(r), 1, z)
        eq5 += a5["tokens"] == base5[r["utterance_id"]]["tokens"]
        eq1 += a1["tokens"] == base1[r["utterance_id"]]["tokens"]
    dev_per_utt = (time.time() - t_dev) / 20   # beam-5 + greedy per dev utterance
    # replay engine parity: empty content prefix, canonical budget, frozen B0 and CE42 = recorded rollout
    from csasr.models.whisper import batch_model_inputs
    par = 0
    for r in R.tm[:10]:
        inp = batch_model_inputs(R.b, [r["audio_path"]])
        par += R.replay(inp, r["prompt"], 200) == rolls["B0"][r["utterance_id"]]["tokens"]
    res["G5"] = {"identity_train_rollout_equal_b0": [sum(x["tokens"] == rolls["B0"][x["id"]]["tokens"] for x in tr20), 20],
                 "identity_dev_beam5_equal": [eq5, 20], "identity_dev_greedy_equal": [eq1, 20], "replay_parity_b0": [par, 10],
                 "hooks_after": site_hook_count(R.b.model, 31)}
    # ---- G2-G4 loss smoke on real events (KL42 rollout events of the first micro-batch that has T and R events)
    act = [e for e in evs_all["KL42"] if e["status"] in ("T", "R")]
    uids = []
    for e in act:
        if e["uid"] not in uids:
            uids.append(e["uid"])
        if len(uids) == 8:
            break
    rows = [r for r in R.tm if r["utterance_id"] in uids]
    evs = [e for e in act if e["uid"] in uids]
    g = {"n_events": len(evs), "n_T": sum(e["status"] == "T" for e in evs), "n_R": sum(e["status"] == "R" for e in evs)}
    z.module.zero_grad(set_to_none=True)
    L0, s0 = op_step(R, z, rows, evs)
    tl = [v for v, e in zip(s0["per_event"], evs) if e["status"] == "T"]
    g["identity_T_kl_max"] = max(tl) if tl else None
    g["identity_T_kl_zero"] = (max(tl) == 0.0) if tl else True
    g["R_ce_finite_positive"] = all(v > 0 and np.isfinite(v) for v, e in zip(s0["per_event"], evs) if e["status"] == "R")
    L0.backward()
    g["grads_finite"] = all(bool(torch.isfinite(p.grad).all()) for p in z.parameters() if p.grad is not None)
    rz = make_adapter("R2", R.b.model, seed_config(R.C1, 42), R.b.device)
    randomize(rz, 0.01, 0.5, seed=7)
    L1, s1 = op_step(R, rz, rows, evs)
    L1.backward()
    g["perturbed_T_kl_positive"] = all(v > 0 for v, e in zip(s1["per_event"], evs) if e["status"] == "T")
    g["grads_reach_R_W_b"] = {n_: float(p.grad.norm()) for n_, p in rz.module.named_parameters()}
    g["grads_nonzero_finite"] = all(v > 0 and np.isfinite(v) for v in g["grads_reach_R_W_b"].values())
    g["backbone_grads_none"] = all(p.grad is None for p in R.b.model.parameters())
    # padding / batch-shape independence: each event alone vs batched (BF16 tolerance)
    single = []
    for e in evs[:6]:
        r_ = [r for r in rows if r["utterance_id"] == e["uid"]]
        l_, _ = op_step(R, rz, r_, [e], grad=False)
        single.append(float(l_))
    g["batched_vs_single"] = [[a, b_, e["status"]] for a, b_, e in zip(s1["per_event"][:6], single, evs[:6])]
    g["batched_vs_single_ok"] = all(abs(a - b_) <= 0.02 + 0.05 * abs(b_) for a, b_ in zip(s1["per_event"][:6], single))  # BF16
    g["hooks_after"] = site_hook_count(R.b.model, 31)
    res["G4"] = g
    del rz
    # ---- GoldKL reuse: one epoch of the M2 loop with lambda_op = 0 reproduces P2-1b KL_seed42 epoch 1 bitwise
    tr = seed_config(R.C1, 42)["training"]
    torch.manual_seed(42)
    a3 = make_adapter("R2", R.b.model, seed_config(R.C1, 42), R.b.device)
    ref1 = torch.load(art("p2_1b/training/KL_seed42/epoch1/adapter.pt"))
    base_eq = torch.equal(a3.state_dict()["rot.parametrizations.weight.0.base"], ref1["rot.parametrizations.weight.0.base"])
    opt = torch.optim.AdamW(a3.parameters(), lr=R.C1["methods"]["R2"]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"], weight_decay=tr["weight_decay"])
    h = m2_epoch(R, a3, opt, R.tm, tr, 42, 1, {}, C["lambda_gold_kl"], 0.0)
    mine = a3.state_dict()
    res["G4_goldkl_reuse"] = {"init_base_equal_KL42": base_eq, "bitwise_equal_KL42_epoch1": set(mine) == set(ref1) and all(torch.equal(mine[k], ref1[k]) for k in ref1),
                              "max_abs_diff": max(float((mine[k] - ref1[k]).abs().max()) for k in ref1), "epoch_seconds_lambda0": h["train_seconds"]}
    # ---- timing of one M2 micro-batch (gold + OP) and projections
    torch.cuda.synchronize()
    tt = []
    for _ in range(3):
        t0 = time.time()
        from reft.p2_1b_train import kl_step
        l_, _, _ = kl_step(R.b, a3, rows, 1.0)
        l_.backward()
        lo, _ = op_step(R, a3, rows, evs)
        lo.backward()
        torch.cuda.synchronize()
        tt.append(time.time() - t0)
        for p in a3.parameters():
            p.grad = None
    t_events = max(cov[n]["seconds"] for n in ("CE42", "KL42"))       # alignment + teacher replays, 500 utterances
    roll = max(tro.values())
    n_mb = -(-len(R.tm) // tr["micro_batch"])
    epoch_m2 = max(h["train_seconds"], n_mb * float(np.mean(tt[1:])))   # conservative: every micro-batch carries OP events
    proj = 3 * (roll + t_events + epoch_m2) + 4 * len(R.dm) * dev_per_utt + 300
    res["G6"] = {"rollout_seconds_500": tro, "event_build_seconds": {n: cov[n]["seconds"] for n in cov}, "teacher_replays": R.n_teacher,
                 "m2_microbatch_seconds": tt, "gold_epoch_seconds": h["train_seconds"], "m2_epoch_seconds_projected": epoch_m2,
                 "dev_seconds_per_utt": dev_per_utt, "projected_run_seconds": proj,
                 "projected_total_slice_hours": 3 * proj / 3600, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                 "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    w1 = weights_hash(R.b.model)
    res["weights_unchanged"] = R.w0 == w1
    gates = {"G1": all(res["G1"].values()),
             "G2_G3_coverage": coverage_ok,
             "G4": g["identity_T_kl_zero"] and g["R_ce_finite_positive"] and g["grads_finite"] and g["perturbed_T_kl_positive"]
             and g["grads_nonzero_finite"] and g["backbone_grads_none"] and g["batched_vs_single_ok"] and g["hooks_after"] == 0
             and g["n_T"] > 0 and g["n_R"] > 0 and res["G4_goldkl_reuse"]["bitwise_equal_KL42_epoch1"] and base_eq,
             "G5": res["G5"]["identity_train_rollout_equal_b0"][0] == 20 and eq5 == 20 and eq1 == 20 and par == 10 and res["G5"]["hooks_after"] == 0,
             "G6": proj <= C["compute"]["projected_run_seconds_max"] and res["weights_unchanged"],
             "G7": out("AUTHORIZATION.json").exists()}
    if not coverage_ok:
        verdict = "P2_2_BLOCKED_ONPOLICY_COVERAGE"
    elif not all(gates.values()):
        verdict = "P2_2_BLOCKED_" + "_".join(k for k, v in gates.items() if not v)
    else:
        verdict = "READY_FOR_P2_2_TRAINING"
    rec = {"schema": "reft_p2_2_preflight_v1", "utc": now(), "git": git_state(), "slurm": slurm(), "env": R.env, "cuda_libs": R.libs,
           "results": res, "coverage": cov, "gates": gates, "verdict": verdict, "wall_seconds": time.time() - t_all}
    write_json(out(f"preflight_{slurm()['SLURM_JOB_ID']}.json"), rec)
    write_json(out("coverage_train.json"), {"utc": now(), "slurm": slurm(), "coverage": cov, "coverage_ok": coverage_ok,
                                            "gate_thresholds": C["coverage_gate"], "note": "training data only; dev150 not used"})
    print(json.dumps({"verdict": verdict, "gates": gates, "coverage": {n: {k: v for k, v in c.items() if k not in ("tokens_per_event",)} for n, c in cov.items()},
                      "G4": g, "G4_goldkl": res["G4_goldkl_reuse"], "G5": res["G5"], "G6": res["G6"]}, indent=1, default=str))


def newest_preflight():
    ps = sorted(out().glob("preflight_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return ps[-1] if ps else None


# ====================================================================== train (GPU)

def run_train(seed: int):
    import torch
    from reft.adapters import make_adapter
    from reft.auto_policy import decode_frozen_lid, frozen_lid
    from reft.p2_1b_train import seed_config
    C = c22()
    assert seed in C["seeds"]
    fz = check_freeze()
    if not out("AUTHORIZATION.json").exists():
        raise SystemExit("P2-2 not authorized")
    pf = newest_preflight()
    if pf is None or read_json(pf)["verdict"] != "READY_FOR_P2_2_TRAINING":
        raise SystemExit(f"preflight not READY: {pf}")
    od = out("training", f"M2_seed{seed}")
    if od.exists():
        raise SystemExit(f"{od} exists; runs are write-once")
    b0roll = {x["id"]: x for x in read_jsonl(out("preflight_rollouts", "B0.jsonl"))}
    torch.manual_seed(seed)           # same order as P2-1b run_train: seed -> backbone -> adapter
    R = RT()
    Cs = seed_config(R.C1, seed)
    tr = Cs["training"]
    ad = make_adapter("R2", R.b.model, Cs, R.b.device)
    assert ad.n_trainable() == R.C1["methods"]["R2"]["expected_trainable_parameters"]
    ref1 = torch.load(art(f"p2_1b/training/KL_seed{seed}/epoch1/adapter.pt"))
    base_key = "rot.parametrizations.weight.0.base"
    init_pair = torch.equal(ad.state_dict()[base_key], ref1[base_key])
    if not init_pair:
        raise SystemExit("M2 initial parametrization base differs from the matched KL/CE run")
    opt = torch.optim.AdamW(ad.parameters(), lr=R.C1["methods"]["R2"]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"], weight_decay=tr["weight_decay"])
    base = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()

    def dev_decode(ep):
        res = {"beam5": [], "greedy": []}
        t0 = time.time()
        for r in R.dm:
            for reg, nb in (("beam5", 5), ("greedy", 1)):
                rec, inp = decode_frozen_lid(R.b, r["audio_path"], R.lid(r), nb, ad)
                res[reg].append({"id": r["utterance_id"], **rec})
            with ad.active(R.b.model):
                assert frozen_lid(R.b, inp["input_features"], ad)["token"] == r["frozen_lid_token"], "LID changed under adapter"
        for reg, v in res.items():
            write_jsonl(od / f"epoch{ep}" / f"dev_{reg}.jsonl", v)
        return time.time() - t0, res

    sec, res0 = dev_decode(0)
    ident = sum(x["tokens"] == base[x["id"]]["tokens"] for x in res0["beam5"])
    if ident != len(R.dm):
        raise SystemExit(f"zero-init adapter does not reproduce the frozen baseline ({ident}/{len(R.dm)})")
    hist = [{"epoch": 0, "zero_init_identity_beam5": ident, "dev_decode_seconds": sec, "init_base_equal_matched_KL": init_pair}]
    steps = 0
    for ep in range(1, tr["epochs"] + 1):
        # rollout refresh: adapter frozen at the start of the epoch, training audio only
        t0 = time.time()
        with torch.no_grad():
            hyps = {x["id"]: x for x in R.rollout(R.tm, ad)}
        t_roll = time.time() - t0
        write_jsonl(od / "rollouts" / f"epoch{ep}_hyps.jsonl", list(hyps.values()))
        nt0 = R.n_teacher
        evs, st = R.build_events(R.tm, hyps, b0roll, seed, ep)
        write_jsonl(od / "rollouts" / f"epoch{ep}_events.jsonl", evs)
        ev_by = collections.defaultdict(list)
        for e in evs:
            ev_by[e["uid"]].append(e)
        ident_roll = sum(hyps[k]["tokens"] == b0roll[k]["tokens"] for k in hyps)
        h = m2_epoch(R, ad, opt, R.tm, tr, seed, ep, ev_by, C["lambda_gold_kl"], C["lambda_op"], steps)
        steps = h["updates_total"]
        (od / f"epoch{ep}").mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), od / f"epoch{ep}" / "adapter.pt")
        s_, _ = dev_decode(ep)
        hist.append({**h, "rollout": {**st, "rollout_seconds": t_roll, "teacher_replays": R.n_teacher - nt0,
                                      "hyps_identical_to_b0": ident_roll},
                     "dev_decode_seconds": s_, "adapter_sha256": sha(od / f"epoch{ep}" / "adapter.pt")})
        print(json.dumps({k: v for k, v in hist[-1].items() if k != "rollout"} | {"rollout": {k: st[k] for k in ("located", "active", "T", "R", "U", "active_non_identity", "active_sb")}}), flush=True)
    w1 = weights_hash(R.b.model)
    lat = {reg: float(np.mean([x["seconds"] for x in read_jsonl(od / f"epoch{tr['epochs']}" / f"dev_{reg}.jsonl")])) for reg in ("beam5", "greedy")}
    write_json(od / "manifest.json", {
        "schema": "reft_p2_2_run_v1", "utc": now(), **fz, "git": git_state(), "env": R.env, "slurm": slurm(), "arm": "M2", "seed": seed,
        "lambda_gold_kl": C["lambda_gold_kl"], "lambda_op": C["lambda_op"], "method": "R2", "config": Cs["methods"]["R2"], "training": tr,
        "events_config": C["events"], "history": hist, "weights_before": R.w0, "weights_after": w1, "weights_unchanged": R.w0 == w1 == R.expected_w,
        "cuda_libs": R.libs, "preflight": pf.name,
        "efficiency": {"trainable_parameters": ad.n_trainable(), "adapter_bytes_fp32": 4 * ad.n_trainable(), "optimizer_updates": steps,
                       "train_seconds": sum(x.get("train_seconds", 0) for x in hist), "op_seconds": sum(x.get("op_seconds", 0) for x in hist),
                       "rollout_seconds": sum(x["rollout"]["rollout_seconds"] for x in hist[1:]),
                       "event_build_seconds": sum(x["rollout"]["seconds"] for x in hist[1:]),
                       "teacher_replays": R.n_teacher, "teacher_replay_steps": R.n_teacher_steps,
                       "dev_decode_seconds": sum(x.get("dev_decode_seconds", 0) for x in hist),
                       "wall_seconds": time.time() - t_all, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "mean_decode_seconds": lat}})


# ====================================================================== verify (GPU)

def run_verify():
    import torch
    from reft.adapters import make_adapter
    from reft.auto_policy import decode_frozen_lid
    from reft.p2_1b_train import seed_config
    fz = check_freeze()
    R = RT()
    res = {}
    for s in SEEDS:
        rd = out("training", f"M2_seed{s}")
        man = read_json(rd / "manifest.json")
        res[rd.name] = {}
        for h in man["history"][1:]:
            p = rd / f"epoch{h['epoch']}" / "adapter.pt"
            ad = make_adapter("R2", R.b.model, seed_config(R.C1, s), R.b.device)
            ad.load_state_dict(torch.load(p))
            rec = {x["id"]: x for x in read_jsonl(rd / f"epoch{h['epoch']}" / "dev_beam5.jsonl")}
            eq = sum(decode_frozen_lid(R.b, r["audio_path"], R.lid(r), 5, ad)[0]["tokens"] == rec[r["utterance_id"]]["tokens"] for r in R.dm[:20])
            res[rd.name][h["epoch"]] = {"sha_ok": sha(p) == h["adapter_sha256"], "reload_equal": [eq, 20]}
    w1 = weights_hash(R.b.model)
    ok = R.w0 == w1 == R.expected_w and all(v["sha_ok"] and v["reload_equal"][0] == 20 for x in res.values() for v in x.values())
    write_json(out("verify.json"), {"schema": "reft_p2_2_verify_v1", "utc": now(), **fz, "slurm": slurm(), "results": res,
                                    "weights_unchanged": R.w0 == w1, "all_ok": ok})
    print(json.dumps({"all_ok": ok, "results": res}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "preflight", "train", "verify"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        run_train(a.seed)
    else:
        {"freeze": run_freeze, "preflight": run_preflight, "verify": run_verify}[a.cmd]()


if __name__ == "__main__":
    main()
