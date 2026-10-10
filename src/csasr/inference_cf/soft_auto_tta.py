"""P2-TTA-A3 soft AUTO-KL episodic decoder-LN adaptation (frozen contract configs/inference_cf/p2_tta_a3.json).

Inherits the audited A2 actuator from ``episodic_tta`` unchanged (LNGuard, fp32 masters through bf16
``functional_call``, AdamW ``OPTIM``, ``STEPS`` = 2, exact reset). Only the loss differs:

    D_cond(theta) = mean_{valid t} KL( q_t || p_theta,t ),
    q_t     = softmax over generation-allowed IDs of theta0 logits under the AUTO prompt cA on the y_A prefix
              (float32, computed once before updates, detached),
    p_theta = forced-ZH (cB) student on the identical y_A prefix.

Identical-condition rule: if cA == cB (or no valid position) the objective is identically zero; exactly 2 no-op AdamW
steps on a scalar zero connected to the masters (TTA0 empty-teacher convention). No reference input here.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from csasr.inference_cf.episodic_tta import (OPTIM, STEPS, LNGuard, TTAInvalid, _l2, allowed_ids, position_terms,
                                              teacher_logits)

SOT, TRANSCRIBE, NOTIMESTAMPS = 50258, 50360, 50364


def auto_prompt(lang_id: int) -> list[int]:
    return [SOT, int(lang_id), TRANSCRIBE, NOTIMESTAMPS]


def allowed_log_probs(logits: torch.Tensor, T: int, suppress, begin) -> list[torch.Tensor]:
    """[log p over allowed IDs for content step 0 (begin-suppressed), log p for steps 1..T-1] (float32)."""
    V = logits.shape[-1]
    if not bool(torch.isfinite(logits[:T]).all()):
        raise TTAInvalid("nonfinite logits")
    x = logits.to(torch.promote_types(logits.dtype, torch.float32))      # float32 at runtime; float64 kept for tests
    out = []
    if T >= 1:
        out.append(torch.log_softmax(x[0:1].index_select(1, allowed_ids(V, 0, suppress, begin, logits.device)), dim=-1))
    if T >= 2:
        out.append(torch.log_softmax(x[1:T].index_select(1, allowed_ids(V, 1, suppress, begin, logits.device)), dim=-1))
    return out


@torch.no_grad()
def teacher_distribution(model, encoded, cA: Sequence[int], y: Sequence[int], suppress, begin) -> list[torch.Tensor]:
    """Detached float32 log q over allowed IDs on the y prefix under the AUTO prompt, at the resident (theta0) weights."""
    lq = allowed_log_probs(teacher_logits(model, encoded, cA, y, None), len(y), suppress, begin)
    return [x.detach() for x in lq]


def kl_terms(logq: list[torch.Tensor], logp: list[torch.Tensor]) -> torch.Tensor:
    """Per content position forward KL(q || p) = sum_v q (log q - log p)."""
    if not logq:
        return torch.zeros(0)
    return torch.cat([(lq.exp() * (lq - lp)).sum(-1) for lq, lp in zip(logq, logp)])


def adapt_a3(model, guard: LNGuard, encoded, cA: Sequence[int], cB: Sequence[int], y_A: Sequence[int], mask: Sequence[bool], *,
             suppress, begin, eos: int, partition: dict | None = None, counters: dict | None = None, keep_grad0: bool = False) -> dict:
    c = counters if counters is not None else {}
    if not guard.verify():
        raise TTAInvalid("resident LN != theta0 at objective start")
    identical = list(cA) == list(cB)
    T = len(y_A)
    logq = teacher_distribution(model, encoded, cA, y_A, suppress, begin)
    c["teacher_forwards"] = c.get("teacher_forwards", 0) + 1
    masters = guard.fresh_masters()
    opt = torch.optim.AdamW(list(masters.values()), **OPTIM)
    m = torch.tensor(list(mask), dtype=torch.bool)
    nvalid = int(m.sum())
    noop = identical or nvalid == 0
    log = {"kind": "A3", "losses": [], "d_cond": [], "grad_l2": [], "finite": [], "valid": nvalid, "content": T,
           "identical_condition": identical, "no_valid_content": nvalid == 0, "noop": noop, "positions": [], "diag": {}}
    grad0, local_steps = None, 0
    for k in range(STEPS + 1):
        with torch.enable_grad():
            vals = {n: mm.to(torch.bfloat16) for n, mm in masters.items()}
            logits = teacher_logits(model, encoded, cB, y_A, vals)
            c["student_forwards"] = c.get("student_forwards", 0) + 1
            logp = allowed_log_probs(logits, T, suppress, begin)
            kl = kl_terms(logq, logp)
            d_cond = kl[m.to(kl.device)].mean() if nvalid else kl.new_zeros(())
            loss = sum((p * 0).sum() for p in masters.values()) if noop else d_cond
        if not (math.isfinite(float(loss)) and math.isfinite(float(d_cond))):
            raise TTAInvalid(f"nonfinite loss step {k}")
        log["losses"].append(float(loss))
        log["d_cond"].append(float(d_cond))
        log["positions"].append({"kl": kl.detach().cpu().tolist()})
        if k in (0, STEPS):
            with torch.no_grad():
                t = position_terms(logits.detach(), y_A, suppress, begin, eos, partition)
            log["diag"]["theta0" if k == 0 else "theta2"] = {"entropy": t["entropy"].cpu().tolist(), "eos_prob": t["eos_prob"],
                                                             "P_E": t["P_E"], "P_M": t["P_M"]}
        if k < STEPS:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            c["backwards"] = c.get("backwards", 0) + 1
            grads = [masters[n].grad if masters[n].grad is not None else torch.zeros_like(masters[n]) for n in guard.names]
            g2 = _l2(grads)
            fin = math.isfinite(g2) and all(bool(torch.isfinite(g).all()) for g in grads)
            log["grad_l2"].append(g2)
            log["finite"].append(fin)
            if not fin:
                raise TTAInvalid(f"nonfinite gradient step {k}")
            if k == 0 and keep_grad0:
                grad0 = torch.cat([g.detach().flatten() for g in grads]).cpu()
            opt.step()
            c["optimizer_steps"] = c.get("optimizer_steps", 0) + 1
            local_steps += 1
            if not all(bool(torch.isfinite(mm).all()) for mm in masters.values()):
                raise TTAInvalid(f"nonfinite master step {k}")
        del logits, logp, kl, loss
    if not guard.verify():
        raise TTAInvalid("resident LN changed during adaptation")
    with torch.no_grad():
        eff = {n: masters[n].detach().to(torch.bfloat16) for n in guard.names}
        eff_delta = [eff[n].float() - guard.theta0[n].float() for n in guard.names]
        base = _l2([guard.theta0[n].float() for n in guard.names])
        log["master_delta_l2"] = _l2([masters[n].detach() - guard.theta0[n].float() for n in guard.names])
        log["master_delta_rel"] = log["master_delta_l2"] / (base + 1e-12)
        log["effective_delta_l2"] = _l2(eff_delta)
        log["effective_changed_scalars"] = int(sum(int((d != 0).sum()) for d in eff_delta))
    d0, d2 = log["d_cond"][0], log["d_cond"][-1]
    log["relative_gap_reduction"] = (d0 - d2) / max(d0, 1e-12)
    log["steps"] = local_steps
    log["loss_evaluations"] = len(log["losses"])
    del opt, logq
    return {"log": log, "effective": eff, "final_masters": {n: masters[n].detach().cpu().clone() for n in guard.names}, "grad0": grad0}
