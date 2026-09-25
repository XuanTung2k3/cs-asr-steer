#!/usr/bin/env python
"""P2-R mechanism-diagnosis runner (GPU; diagnostic namespace).

This is NOT a deployable decode path. It reads an evaluator-built frozen plan (positions,
reference-consistent target token sets, D2 targets) and measures, on the matched-baseline
trajectory:

* D1/D3 -- single-position pulses at L16 with +d, -d and one frozen random direction, all at
  the same realized edit norm e* (energy solved on the pre-edit site state, never on logits),
  each from the same unedited B-branch cache (cropped back after every arm);
* D2    -- a controlled baseline-prefix replay of the frozen P2 schedule (current placement)
  and of the energy-packet relocation (oracle placement), plus single-pulse companions.

The deployable modules (inference_cf_cached, inference_cf_p2) are imported read-only and are
never given plan, reference or target information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_p1 import direction, processed_argmax, selected_gate
from csasr.inference_cf.core_r2 import (conflict_from_logits, local_support, max_attention_window,
                                        prefix_utf8_complete)
from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, assert_no_site_hooks
from csasr.models.hooks import apply_steering
import experiments.inference_cf_cached as cached
import experiments.inference_cf_p0_r2 as r2

SCHEMA = "p2r_mechanism_diagnosis_v1"
ARMS = ("plus_d", "minus_d", "random")
TOPK = 20


# ---- frozen primitives --------------------------------------------------------------------

def random_seed(uid: str, t: int) -> int:
    return int(hashlib.sha256(f"P2R-random-v1|{uid}|{t}".encode()).hexdigest()[:16], 16)


def random_direction(uid: str, t: int, r: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Exactly one frozen Gaussian direction, orthogonal to the pre-edit site state r."""
    z = torch.from_numpy(np.random.default_rng(random_seed(uid, t)).standard_normal(r.numel()))
    rr = r.detach().double().cpu().reshape(-1)
    rhat = rr / torch.linalg.vector_norm(rr)
    v = z - torch.dot(z, rhat) * rhat
    n = float(torch.linalg.vector_norm(v))
    if n < 1e-8:
        e0 = torch.zeros_like(rr); e0[0] = 1.0
        v = e0 - torch.dot(e0, rhat) * rhat
        return v / torch.linalg.vector_norm(v), "degeneracy_fallback"
    return v / n, "ok"


def emulate_edit_norm(site_vec: torch.Tensor, v: torch.Tensor, s: float) -> float:
    """Realized ||steered - site|| of the hook for direction s*v (exact hook arithmetic)."""
    x = site_vec.reshape(1, 1, -1)
    dirs = (s * v).to(x.device, x.dtype).reshape(1, 1, -1)
    gain = torch.ones((1, 1), device=x.device, dtype=x.dtype)
    st = apply_steering(x, dirs, 1.0, 1.0, gain, True)
    return float((st - x).detach().float().norm())


def solve_scale(site_vec: torch.Tensor, v: torch.Tensor, target: float, max_eval: int = 8) -> dict:
    """Smallest s >= 0 with realized hook edit norm ~= target (analytic chord + secant in site dtype)."""
    r = site_vec.detach().double().reshape(-1)
    vv = v.detach().double().to(r.device).reshape(-1)
    vv = vv / torch.linalg.vector_norm(vv)
    rn = float(torch.linalg.vector_norm(r))
    phi = math.acos(max(-1.0, min(1.0, float(torch.dot(r, vv)) / rn)))
    if target >= 2 * rn:
        return {"status": "energy_unreachable", "s": 0.0, "phi": phi, "theta": None, "edit_norm": 0.0, "evals": 0}
    theta = 2 * math.asin(target / (2 * rn))
    if phi <= theta + 1e-9:
        return {"status": "energy_unreachable", "s": 0.0, "phi": phi, "theta": theta, "edit_norm": 0.0, "evals": 0}
    s0 = rn * math.sin(theta) / math.sin(phi - theta)
    tried = []
    def ev(s):
        e = emulate_edit_norm(site_vec, vv, s)
        tried.append((abs(e * e / (target * target) - 1.0), s, e))
        return e
    e0 = ev(s0)
    s1 = s0 * target / max(e0, 1e-12)
    a, ea, b, eb = s0, e0, s1, ev(s1)
    while len(tried) < max_eval and min(t[0] for t in tried) > 2e-3 and eb != ea:
        c = b + (target - eb) * (b - a) / (eb - ea)
        if not math.isfinite(c) or c <= 0:
            break
        a, ea, b, eb = b, eb, c, ev(c)
    err, s, e = min(tried)
    return {"status": "ok", "s": float(s), "phi": phi, "theta": theta, "edit_norm": float(e),
            "evals": len(tried), "rel_sq_err": float(err)}


def summarize(logits: torch.Tensor, step: int, suppress, begin, partition: dict,
              target_ids: list[int] | None, baseline_token: int | None,
              ref_logp: torch.Tensor | None = None) -> dict:
    """Bounded sufficient statistics of one next-token distribution."""
    raw = logits.detach().float().cpu()
    proc = cached._processed(raw, step, suppress, begin)
    lp = torch.log_softmax(proc, dim=-1)
    top = torch.topk(lp, TOPK)
    out = {"top_ids": top.indices.tolist(), "top_logp": [float(x) for x in top.values],
           "argmax": int(top.indices[0]), "top2_margin": float(proc[top.indices[0]] - proc[top.indices[1]])}
    bc = conflict_from_logits(raw, partition)
    out.update(P_E=bc["P_E"], P_M=bc["P_M"], ambiguous=bc["ambiguous_mass"])
    def mass(ids):
        return float(torch.exp(torch.logsumexp(lp[ids], 0))) if ids else 0.0
    out["P_E_processed"], out["P_M_processed"] = mass(partition["embedded_ids"]), mass(partition["matrix_ids"])
    if baseline_token is not None:
        out["logp_baseline_token"] = float(lp[baseline_token])
        out["raw_logit_baseline_token"] = float(raw[baseline_token])
    if target_ids:
        tl = lp[target_ids]
        ltgt = float(torch.logsumexp(tl, 0))
        others = lp.clone(); others[target_ids] = -float("inf")
        best_other = float(others.max())
        out.update(logp_ref=ltgt, logp_ref_members=[float(x) for x in tl],
                   rank_ref=int(1 + (lp > tl.max()).sum()),
                   margin_ref_vs_competitor=ltgt - best_other,
                   argmax_in_ref=bool(int(top.indices[0]) in target_ids),
                   raw_logsumexp_ref=float(torch.logsumexp(raw[target_ids], 0)))
        if baseline_token is not None:
            out["logodds_ref_vs_baseline"] = ltgt - float(lp[baseline_token])
    if ref_logp is not None:
        p = lp.exp()
        finite = torch.isfinite(lp) & torch.isfinite(ref_logp)
        out["kl_to_none"] = float((p[finite] * (lp[finite] - ref_logp[finite])).sum())
        out["tv_to_none"] = float(0.5 * (p - ref_logp.exp()).abs().sum())
    return out, lp


class DiagBranch(cached.Branch):
    def crop(self, n: int) -> None:
        self.cache.crop(n)
        del self.fed[n:]
        del self.positions[n:]
        if self.cache.get_seq_length() != self.length:
            raise RuntimeError(f"{self.name}: crop mismatch")


def pulse_hook(bundle, layer: int, query: int, v_fn, target: float, nfp: int, info: dict):
    """Hook editing only ``query`` with the direction v_fn(site) at realized edit norm ``target``."""
    def action_fn(q, u_source, r, abs_pos):
        n = r.shape[1]
        g = torch.zeros((1, n), device=r.device, dtype=r.dtype)
        dirs = torch.zeros((1, n, r.shape[-1]), device=r.device, dtype=r.dtype)
        hit = (abs_pos == query).nonzero().flatten()
        if hit.numel():
            k = int(hit[0])
            v, vstatus = v_fn(r[0, k])
            info["direction_status"] = vstatus
            if v is not None:
                sol = solve_scale(r[0, k], v.to(r.device), target)
                info.update(sol)
                vv = v.double().to(r.device)
                vv = vv / torch.linalg.vector_norm(vv)
                info["cos_v_state"] = float(torch.dot(vv, r[0, k].double()) / torch.linalg.vector_norm(r[0, k].double()))
                if sol["status"] == "ok":
                    g[0, k] = 1.0
                    dirs[0, k] = (sol["s"] * vv).to(r.dtype)
                    info["v"] = vv
        return g, dirs
    return DecoderPostCrossAttnInterventionHook(
        bundle, layer, None, alpha=1.0, num_forced_prefix=nfp, action_fn=action_fn,
        norm_preserve=True, record=True, record_last_only=True)


# ---- per-utterance passes -----------------------------------------------------------------

class Ctx:
    def __init__(self, bundle, cfg, partition, null_probs, language_ids, nfp):
        self.bundle, self.cfg, self.partition = bundle, cfg, partition
        self.null_probs, self.language_ids, self.nfp = null_probs, language_ids, nfp
        self.layer = int(cfg["site"]["layer"])
        self.cond = cfg["conditions"]
        self.max_new = int(cfg["site"]["max_new_tokens"])
        tok = bundle.processor.tokenizer
        self.tok, self.eos = tok, tok.eos_token_id
        gen = bundle.model.generation_config
        self.suppress, self.begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
        from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}


def continue_greedy(ctx: Ctx, br: DiagBranch, first_logits, t: int) -> tuple[list[int], str]:
    nxt = processed_argmax(first_logits, t, ctx.suppress, ctx.begin)
    cont, step = [], t
    while True:
        if nxt == ctx.eos:
            return cont, "eos"
        cont.append(int(nxt))
        step += 1
        if step >= ctx.max_new:
            return cont, "cap"
        logits, _, _ = br.step([nxt], attention=True)
        nxt = processed_argmax(logits, step, ctx.suppress, ctx.begin)


def gate_step(ctx: Ctx, waveform, tokens_prefix, heads, logits_b, lid_cache, uid):
    """Frozen R2 gate providers exactly as in cached_decode (policy ER)."""
    reasons, window, support, baseline = [], None, None, None
    if not prefix_utf8_complete(ctx.tok, tokens_prefix, ctx.byte_decoder):
        reasons.append("mid_character")
    try:
        window = max_attention_window(heads.numpy(), len(waveform))
    except Exception as exc:
        reasons.append(r2._reason(exc, "localizer_fail"))
    try:
        baseline = conflict_from_logits(logits_b, ctx.partition)
    except Exception as exc:
        reasons.append(r2._reason(exc, "baseline_provider_fail"))
    en_id, zh_id = ctx.cond["language_token_ids"]
    if not reasons:
        try:
            key = (uid, window.start_sample, window.end_sample)
            if key not in lid_cache:
                lid_cache[key] = cached.native_lid(ctx.bundle, waveform[key[1]:key[2]], ctx.language_ids)
            probs = lid_cache[key]
            support = local_support(probs[en_id], probs[zh_id], ctx.null_probs[en_id], ctx.null_probs[zh_id])
        except Exception as exc:
            reasons.append(r2._reason(exc, "local_support_fail"))
    priority = ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail",
                "ecf_provider_fail", "nonfinite_signal")
    fb = next((r for r in priority if r in reasons), None)
    g = float(selected_gate(support, baseline)) if fb is None else 0.0
    return {"fb": fb, "g": g, "window": None if window is None else
            [window.start_frame, window.end_frame, window.start_sample, window.end_sample],
            "E": None if support is None else support["E"], "R_B": None if baseline is None else baseline["R"]}


def replay(ctx: Ctx, *, encoded, waveform, uid, tokens, terminated, lid_cache, mode: str,
           plan: dict | None, record_at: dict, oracle_crop=None) -> dict:
    """Controlled baseline-prefix replay: B, E teacher-forced on the matched-baseline tokens; one S
    branch with the current (mode='current') or relocated (mode='oracle') schedule."""
    cB, cE = ctx.cond["cB"], ctx.cond["cE"]
    B, E, S = DiagBranch(ctx.bundle, encoded, cB, "B"), DiagBranch(ctx.bundle, encoded, cE, "E"), \
        DiagBranch(ctx.bundle, encoded, cB, "S")
    new = list(cB); new_e = list(cE)
    steps, alpha = [], float(ctx.cfg["d2"]["alpha"])
    n_steps = len(tokens) + (1 if terminated == "eos" else 0)
    for t in range(n_steps):
        assert_no_site_hooks(ctx.bundle)
        lb, heads, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        le, _, he = E.step(new_e, capture_layer=ctx.layer, attention=False)
        query = S.length + len(new) - 1
        gs = gate_step(ctx, waveform, tokens[:t], heads, lb, lid_cache, uid)
        dirn = direction(he, hb)
        d = dirn["d"]
        info, kind = {}, "none"
        if mode == "current" or t not in plan["moved_from"] and t not in plan["moved_to"]:
            applied = bool(alpha * gs["g"] > 0 and d is not None and query >= ctx.nfp)
            hook = cached._edit_hook(ctx.bundle, ctx.layer, query, alpha, gs["g"] if applied else 0.0,
                                     d if applied else None, ctx.nfp)
            kind = "schedule" if applied else "none"
        elif t in plan["moved_from"]:
            hook = cached._edit_hook(ctx.bundle, ctx.layer, query, alpha, 0.0, None, ctx.nfp)
            kind = "moved_away"
        else:
            target = plan["moved_to"][t]["energy"]
            hook = pulse_hook(ctx.bundle, ctx.layer, query, lambda r, d=d: (d, "ok") if d is not None else (None, dirn["status"]),
                              target, ctx.nfp, info)
            kind = "relocated_in"
        ls, _, _ = S.step(new, attention=True, hook=hook)
        assert_no_site_hooks(ctx.bundle)
        audit = hook.records[-1].to_dict() if hook.records else None
        edit_norm = audit["edit_norm"] if audit and audit["steered"] else 0.0
        st = {"t": t, "query": query, "fb": gs["fb"], "g": gs["g"], "E": gs["E"], "R_B": gs["R_B"],
              "window": gs["window"], "dir": dirn["status"], "dir_norm": dirn["norm"], "kind": kind,
              "edit": edit_norm > 0, "edit_norm": edit_norm,
              "pre_norm": audit["pre_norm"] if audit else None,
              "baseline_argmax_ok": (processed_argmax(lb, t, ctx.suppress, ctx.begin) ==
                                     (tokens[t] if t < len(tokens) else ctx.eos)),
              "s_argmax": processed_argmax(ls, t, ctx.suppress, ctx.begin)}
        if kind == "relocated_in":
            st["solver"] = {k: v for k, v in info.items() if k != "v"}
        if t in record_at:
            tgt = record_at[t]
            m0, lp0 = summarize(lb, t, ctx.suppress, ctx.begin, ctx.partition, tgt.get("target_ids"),
                                tokens[t] if t < len(tokens) else None)
            ms, _ = summarize(ls, t, ctx.suppress, ctx.begin, ctx.partition, tgt.get("target_ids"),
                              tokens[t] if t < len(tokens) else None, lp0)
            st["none"], st["arm"] = m0, ms
            if oracle_crop is not None and tgt.get("stratum") == "EN-confusion":
                st["acoustic"] = oracle_crop(t, gs)
        steps.append(st)
        if t < len(tokens):
            new, new_e = [tokens[t]], [tokens[t]]
    lineage = (B.fed[len(cB):] == tokens[:len(B.fed) - len(cB)] and S.fed == B.fed
               and E.fed[len(cE):] == B.fed[len(cB):])
    return {"steps": steps, "lineage_ok": bool(lineage),
            "total_energy": float(sum(s["edit_norm"] ** 2 for s in steps))}


def relocation_plan(current: dict, targets: list[dict]) -> dict:
    """Energy-packet relocation (frozen): donors by descending energy -> unedited targets in hash order."""
    by_t = {s["t"]: s for s in current["steps"]}
    tset = [c["t"] for c in targets]             # already in frozen hash order
    retained = [t for t in tset if by_t.get(t, {}).get("edit")]
    open_targets = [t for t in tset if t in by_t and not by_t[t]["edit"] and by_t[t]["dir"] == "ok"
                    and t >= 1]
    blocked = [t for t in tset if t in by_t and not by_t[t]["edit"] and not (by_t[t]["dir"] == "ok" and t >= 1)]
    donors = sorted([s for s in current["steps"] if s["edit"] and s["t"] not in tset],
                    key=lambda s: (-s["edit_norm"], s["t"]))
    moves = []
    for dnr, dst in zip(donors, open_targets):
        moves.append({"from": dnr["t"], "to": dst, "energy": dnr["edit_norm"],
                      "source_after_target": dnr["t"] > dst})
    return {"retained": retained, "moves": moves, "blocked_targets": blocked,
            "unmatched_donors": [s["t"] for s in donors[len(moves):]],
            "unfilled_targets": open_targets[len(moves):],
            "moved_from": {m["from"]: m for m in moves}, "moved_to": {m["to"]: m for m in moves}}


def pulse_pass(ctx: Ctx, *, encoded, uid, tokens, jobs: dict) -> dict:
    """D1/D3 (and D2 pulse companions): at each job position, run every arm from the same unedited
    B cache (cropped back after each arm), then restore and verify the unedited step bitwise."""
    cB, cE = ctx.cond["cB"], ctx.cond["cE"]
    B, E = DiagBranch(ctx.bundle, encoded, cB, "B"), DiagBranch(ctx.bundle, encoded, cE, "E")
    new, new_e = list(cB), list(cE)
    out = {}
    last = max(jobs) if jobs else -1
    for t in range(last + 1):
        L = B.length
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        le, _, he = E.step(new_e, capture_layer=ctx.layer, attention=False)
        if processed_argmax(lb, t, ctx.suppress, ctx.begin) != tokens[t]:
            out.setdefault("_baseline_mismatch", []).append(t)
        if t in jobs:
            query = L + len(new) - 1
            dirn = direction(he, hb)
            rec = {"t": t, "query": query, "dir": dirn["status"], "dir_norm": dirn["norm"],
                   "cos_d_state": None, "arms": {}}
            none, lp0 = summarize(lb, t, ctx.suppress, ctx.begin, ctx.partition, jobs[t].get("target_ids"), tokens[t])
            rec["none"] = none
            d = dirn["d"]
            if d is not None:
                rec["cos_d_state"] = float(torch.dot(d.double(), hb.double()) / torch.linalg.vector_norm(hb.double()))
            for arm_name, arm in jobs[t]["arms"].items():
                B.crop(L)
                info = {}
                if arm["dir"] == "random":
                    v_fn = lambda r: random_direction(uid, t, r)
                elif d is None:
                    v_fn = lambda r: (None, dirn["status"])
                else:
                    sign = 1.0 if arm["dir"] == "plus_d" else -1.0
                    v_fn = lambda r, sign=sign: (sign * d, "ok")
                hook = pulse_hook(ctx.bundle, ctx.layer, query, v_fn, arm["energy"], ctx.nfp, info)
                la, _, h_cons = B.step(new, capture_layer=ctx.layer, attention=True, hook=hook)
                assert_no_site_hooks(ctx.bundle)
                audit = hook.records[-1].to_dict() if hook.records else None
                ms, _ = summarize(la, t, ctx.suppress, ctx.begin, ctx.partition, jobs[t].get("target_ids"),
                                  tokens[t], lp0)
                v = info.pop("v", None)
                a = {"solver": info, "summary": ms,
                     "edit_norm": audit["edit_norm"] if audit and audit["steered"] else 0.0,
                     "pre_norm": audit["pre_norm"] if audit else None}
                if audit and audit["steered"]:
                    cons = (h_cons.double() - hb.double())
                    a["consumed_edit_norm"] = float(torch.linalg.vector_norm(cons))
                    a["cos_edit_v"] = float(torch.dot(cons, v.cpu()) / (a["consumed_edit_norm"] + 1e-30))
                    a["relative_sq_energy"] = a["edit_norm"] ** 2 / float(hb.double().norm() ** 2)
                    a["solver_matches_hook"] = abs(info.get("edit_norm", -1) - a["edit_norm"]) <= 1e-6 * max(1, a["edit_norm"])
                if arm.get("continue"):
                    a["continuation"], a["terminated"] = continue_greedy(ctx, B, la, t)
                rec["arms"][arm_name] = a
            B.crop(L)
            lb2, _, hb2 = B.step(new, capture_layer=ctx.layer, attention=True)
            rec["restore_bitwise"] = bool(torch.equal(lb2, lb) and torch.equal(hb2, hb))
            out[t] = rec
        new, new_e = [tokens[t]], [tokens[t]]
    return out


def run_utterance(ctx: Ctx, *, encoded, waveform, uid, tokens, terminated, lid_cache, d1_positions,
                  d2_targets, d2_controls, pulse_budget: list, oracle_crop=None) -> dict:
    record_at = {c["t"]: c for c in d2_controls}
    record_at.update({c["t"]: c for c in d2_targets})
    current = replay(ctx, encoded=encoded, waveform=waveform, uid=uid, tokens=tokens, terminated=terminated,
                     lid_cache=lid_cache, mode="current", plan=None, record_at=record_at, oracle_crop=oracle_crop)
    plan = relocation_plan(current, d2_targets)
    oracle = replay(ctx, encoded=encoded, waveform=waveform, uid=uid, tokens=tokens, terminated=terminated,
                    lid_cache=lid_cache, mode="oracle", plan=plan, record_at=record_at)
    e_star = float(ctx.cfg["energy"]["target_edit_norm"])
    jobs = {}
    for c in d1_positions:
        jobs[c["t"]] = {"target_ids": c["target_ids"], "stratum": c["stratum"],
                        "arms": {a: {"dir": a, "energy": e_star, "continue": bool(c["companion"])} for a in ARMS}}
    pulses = []
    for m in plan["moves"]:
        if len(pulse_budget) >= int(ctx.cfg["d2"]["pulse_companions"]):
            break
        pulse_budget.append((uid, m["to"]))
        tgt = next(c for c in d2_targets if c["t"] == m["to"])
        pulses.append(m)
        for pos, name in ((m["from"], f"pulse_current_{m['to']}"), (m["to"], f"pulse_oracle_{m['to']}")):
            j = jobs.setdefault(pos, {"target_ids": tgt["target_ids"] if pos == m["to"] else None,
                                      "stratum": None, "arms": {}})
            j["arms"][name] = {"dir": "plus_d", "energy": m["energy"], "continue": True}
    pulse = pulse_pass(ctx, encoded=encoded, uid=uid, tokens=tokens, jobs=jobs)
    serial_plan = {k: v for k, v in plan.items() if k not in ("moved_from", "moved_to")}
    return {"current": current, "oracle": oracle, "plan": serial_plan, "pulse_companions": pulses,
            "pulses": {str(k): v for k, v in pulse.items() if not isinstance(k, str)},
            "baseline_mismatch": pulse.get("_baseline_mismatch", [])}


# ---- job entry ---------------------------------------------------------------------------

def main() -> None:
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core import validated_row
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
    from csasr.utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or digest({k: v for k, v in manifest.items()
                                                 if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid P2-R manifest")
    for path, expected in manifest["sources"].items():
        if file_hash(path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != manifest["git_commit"]:
        raise ValueError("Git commit changed after P2-R manifest freeze")
    cfg = json.loads((ROOT / manifest["config"]).read_text())
    plan = json.loads((ROOT / manifest["population"]).read_text())
    if plan["population_hash"] != manifest["population_hash"]:
        raise ValueError("population hash mismatch")
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != manifest["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    null_probs = cached.native_lid(bundle, np.zeros(480000, dtype=np.float32), language_ids)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != int(cfg["site"]["num_forced_prefix"]):
        raise ValueError("forced prefix mismatch")
    ctx = Ctx(bundle, cfg, partition, null_probs, language_ids, nfp)
    base_dir = ROOT / cfg["baseline_run"]
    base_panel = json.loads((base_dir / "panel.json").read_text())["rows"]
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    base_idx = {r["utterance_id"]: i for i, r in enumerate(base_panel)}
    torch.cuda.reset_peak_memory_stats()
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running"}
    atomic_json(out / "runtime.json", runtime)
    pulse_budget: list = []
    en_id, zh_id = cfg["conditions"]["language_token_ids"]
    for i, uid in enumerate(plan["utterances"]):
        dest = out / "rows" / f"{i:03d}.json"
        base_row = json.loads((base_dir / "rows" / f"{base_idx[uid]:03d}.json").read_text())
        bsys = base_row["systems"][cfg["baseline_system"]]
        d2_targets = [c for c in plan["d2_targets"] if c["utterance_id"] == uid]
        if validated_row(dest, uid, manifest["manifest_hash"]):
            prev = json.loads(dest.read_text())
            pulse_budget.extend((uid, m["to"]) for m in prev.get("pulse_companions", []))
            continue
        t0 = time.time()
        try:
            item = audio[uid]
            waveform = load_audio(item["audio_path"], bundle.sample_rate)
            inputs = batch_model_inputs(bundle, [item["audio_path"]])
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                    input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
            heard = min(30.0, len(waveform) / bundle.sample_rate)
            lid_cache: dict = {}
            mids = {c["t"]: c["ctc_midpoint_sec"] for c in d2_targets}

            def oracle_crop(t, gs, lid_cache=lid_cache, mids=mids, waveform=waveform, heard=heard):
                mid = mids.get(t)
                if mid is None:
                    return {"status": "no_ctc_midpoint"}
                a, b = max(0.0, mid - 0.5), min(heard, mid + 0.5)
                s0, s1 = int(round(a * bundle.sample_rate)), int(round(b * bundle.sample_rate))
                probs = cached.native_lid(bundle, waveform[s0:s1], language_ids)
                sup = local_support(probs[en_id], probs[zh_id], null_probs[en_id], null_probs[zh_id])
                cur_centre = None if gs["window"] is None else (gs["window"][2] + gs["window"][3]) / 2 / bundle.sample_rate
                return {"status": "ok", "oracle_window_sec": [a, b], "E_oracle": sup["E"], "E_current": gs["E"],
                        "timing_error_sec": None if cur_centre is None else abs(cur_centre - mid)}

            with torch.inference_mode():
                res = run_utterance(ctx, encoded=enc, waveform=waveform, uid=uid, tokens=bsys["tokens"],
                                    terminated=bsys["terminated"], lid_cache=lid_cache,
                                    d1_positions=[c for s in plan["d1"].values() for c in s if c["utterance_id"] == uid],
                                    d2_targets=d2_targets,
                                    d2_controls=[c for c in plan["d2_controls"] if c["utterance_id"] == uid],
                                    pulse_budget=pulse_budget, oracle_crop=oracle_crop)
            res["status"] = "ok"
        except Exception as exc:
            import traceback
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=manifest["manifest_hash"], elapsed_sec=time.time() - t0)
        atomic_json(dest, res)
        print(f"P2-R {i + 1}/{len(plan['utterances'])} {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
    assert_no_site_hooks(bundle)
    runtime.update(end_unix=time.time(), status="completed", peak_vram_bytes=torch.cuda.max_memory_allocated())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
