"""P2-TTA-A4 script-preserving AUTO distillation (frozen contract configs/inference_cf/p2_tta_a4.json).

Same audited A2/A3 actuator (``episodic_tta`` LNGuard / fp32 masters / bf16 functional_call / AdamW ``OPTIM`` /
``STEPS`` = 2 / exact reset) and the A3 teacher primitives (``soft_auto_tta``). Only the teacher differs:

    q_SAFE,t = q_A,t (theta0, AUTO prompt cA)   if y_A,t in V_E (canonical ``embedded_ids``)
             = q_B,t (theta0, forced-ZH cB)     otherwise (V_M ``matrix_ids`` and OTHER),
    L_A4     = mean_{valid t} KL(q_SAFE,t || p_theta,t),  p_theta = forced-ZH student on the same y_A prefix.

Diagnostics per step: D_E (KL(q_A||p) on EN), D_M (KL(q_B||p) on MATRIX), D_ANCHOR (KL(q_B||p) on all non-EN).
Exact no-op (2 zero-gradient AdamW steps on a connected scalar zero) when cA == cB, no valid EN position, or no valid
position: there q_SAFE = q_B = p_theta0 and the objective is identically zero. No reference input here.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from csasr.inference_cf.episodic_tta import OPTIM, STEPS, LNGuard, TTAInvalid, _l2, position_terms, teacher_logits
from csasr.inference_cf.soft_auto_tta import allowed_log_probs, kl_terms, teacher_distribution


def token_classes(y: Sequence[int], partition: dict) -> list[str]:
    """'E' if target in V_E (embedded_ids), 'M' if in V_M (matrix_ids), else 'O'. Token identity only."""
    E, M = set(partition["embedded_ids"]), set(partition["matrix_ids"])
    if E & M:
        raise TTAInvalid("partition V_E/V_M not disjoint")
    return ["E" if int(t) in E else ("M" if int(t) in M else "O") for t in y]


def safe_teacher(logq_A: list[torch.Tensor], logq_B: list[torch.Tensor], classes: Sequence[str]) -> list[torch.Tensor]:
    """Row-wise selection over the allowed-ID parts [step 0, steps 1..T-1]: q_A at EN positions, q_B elsewhere."""
    T = len(classes)
    isE = torch.tensor([c == "E" for c in classes], dtype=torch.bool)
    parts, offs = [], [(0, 1), (1, T)]
    for (a, b), (lo, hi) in zip(zip(logq_A, logq_B), offs[:len(logq_A)]):
        sel = isE[lo:hi].to(a.device)[:, None]
        parts.append(torch.where(sel, a, b))
    return parts


def _masked_mean(x: torch.Tensor, m: torch.Tensor):
    return x[m.to(x.device)].mean() if int(m.sum()) else None


def adapt_a4(model, guard: LNGuard, encoded, cA: Sequence[int], cB: Sequence[int], y_A: Sequence[int], mask: Sequence[bool],
             classes: Sequence[str], *, suppress, begin, eos: int, partition: dict | None = None, counters: dict | None = None,
             keep_grad0: bool = False) -> dict:
    c = counters if counters is not None else {}
    if not guard.verify():
        raise TTAInvalid("resident LN != theta0 at objective start")
    T = len(y_A)
    if len(classes) != T or len(mask) != T:
        raise TTAInvalid("class/mask length")
    lqA = teacher_distribution(model, encoded, cA, y_A, suppress, begin)
    lqB = teacher_distribution(model, encoded, cB, y_A, suppress, begin)
    c["teacher_forwards"] = c.get("teacher_forwards", 0) + 2
    lqS = safe_teacher(lqA, lqB, classes)
    m = torch.tensor(list(mask), dtype=torch.bool)
    isE = torch.tensor([x == "E" for x in classes], dtype=torch.bool)
    isM = torch.tensor([x == "M" for x in classes], dtype=torch.bool)
    mE, mM, mNE = m & isE, m & isM, m & ~isE
    identical = list(cA) == list(cB)
    nE, nvalid = int(mE.sum()), int(m.sum())
    noop = identical or nE == 0 or nvalid == 0
    masters = guard.fresh_masters()
    opt = torch.optim.AdamW(list(masters.values()), **OPTIM)
    log = {"kind": "A4", "losses": [], "d_E": [], "d_M": [], "d_anchor": [], "grad_l2": [], "finite": [], "valid": nvalid,
           "content": T, "n_E": nE, "n_M": int(mM.sum()), "n_O": int((m & ~isE & ~isM).sum()), "n_nonE": int(mNE.sum()),
           "identical_condition": identical, "noop": noop, "positions": [], "diag": {}}
    grad0, steps = None, 0
    for k in range(STEPS + 1):
        with torch.enable_grad():
            vals = {n: mm.to(torch.bfloat16) for n, mm in masters.items()}
            logits = teacher_logits(model, encoded, cB, y_A, vals)
            c["student_forwards"] = c.get("student_forwards", 0) + 1
            logp = allowed_log_probs(logits, T, suppress, begin)
            klS, klA, klB = kl_terms(lqS, logp), kl_terms(lqA, logp), kl_terms(lqB, logp)
            obj = _masked_mean(klS, m)
            loss = sum((p * 0).sum() for p in masters.values()) if (noop or obj is None) else obj
        vals_ = {"d_E": _masked_mean(klA.detach(), mE), "d_M": _masked_mean(klB.detach(), mM), "d_anchor": _masked_mean(klB.detach(), mNE)}
        for key, v in vals_.items():
            log[key].append(None if v is None else float(v))
        if not math.isfinite(float(loss)) or any(v is not None and not math.isfinite(float(v)) for v in vals_.values()):
            raise TTAInvalid(f"nonfinite loss/diagnostic step {k}")
        log["losses"].append(float(loss))
        log["positions"].append({"kl_A": klA.detach().cpu().tolist(), "kl_B": klB.detach().cpu().tolist()})
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
            steps += 1
            if not all(bool(torch.isfinite(mm).all()) for mm in masters.values()):
                raise TTAInvalid(f"nonfinite master step {k}")
        del logits, logp, klS, klA, klB, loss
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
    dE0, dE2 = log["d_E"][0], log["d_E"][-1]
    log["R_E"] = None if dE0 is None else (dE0 - dE2) / max(dE0, 1e-12)
    log["steps"] = steps
    log["loss_evaluations"] = len(log["losses"])
    del opt, lqA, lqB, lqS
    return {"log": log, "effective": eff, "final_masters": {n: masters[n].detach().cpu().clone() for n in guard.names}, "grad0": grad0}
