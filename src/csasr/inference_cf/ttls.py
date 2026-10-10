"""TTLS-R1 test-time learned steering primitives (frozen contract docs/inference_cf/TTLS_R1_SPEC.md,
configs/inference_cf/ttls_r1.json). Exploratory ticket; core-v6 method and every historical module are unchanged.

Adaptation variables (both reset per utterance, both 2 AdamW steps, 3 loss evaluations, bf16 forward):

* ``LN``   -- the historical A2 actuator, reused unchanged from ``episodic_tta`` (194 decoder LayerNorm tensors,
  fresh fp32 masters through ``functional_call``, ``OPTIM``, exact ``LNGuard`` reset).
* ``TTLS`` -- one temporary fp32 vector ``z`` (d_model scalars) applied at the frozen DG-02 site of decoder layer 16
  (post-cross-attention residual, pre-FFN) through the canonical ``DecoderPostCrossAttnInterventionHook``:
  ``r~_t = NormPreserve(r_t + g_t z)`` (``models.hooks.apply_steering``, alpha = scale = 1, no depth rescale), with
  ``g_t`` a frozen 0/1 step mask. Forced-prefix positions (abs < 4, i.e. content step 0) are never edited.
  AdamW lr = E_STAR / (2 sqrt(d)), wd 0; after every step ``z`` is projected onto ||z|| <= E_STAR (the frozen DG-02
  L16 realized chord).

Objectives (all on the forced-ZH prompt cB):

* ``CE`` -- the historical A2 loss exactly (``episodic_tta.position_terms`` + ``objective_loss('A2')``): mean valid
  -log p(y_A[t]) on the AUTO pseudo-transcript path.
* ``AC`` -- -log p(c* | y_B[:t*]) at the first accepted acoustic-lexical candidate (``select_candidates``).
* ``+P`` -- + LAMBDA_P * mean_{t in S} KL(p_theta0,t || p_theta,t) on the frozen B0 path y_B (allowed vocabulary),
  S = ``stable_positions``.

No reference, evaluator or outcome input anywhere in this module.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np
import torch

from csasr.inference_cf.episodic_tta import (OPTIM, STEPS, LNGuard, TTAInvalid, _l2, objective_loss, position_terms,
                                              teacher_logits)
from csasr.inference_cf.soft_auto_tta import allowed_log_probs, kl_terms

LAYER = 16
PROMPT_LEN = 4
E_STAR = 1.1260757575454359
TAU = math.log(10.0)
NULL_FLOOR = math.log(1e-12)
LAMBDA_P = 1.0
STABLE_P_MIN = 0.5
SOT, EN, ZH, TRANSCRIBE, NOTIMESTAMPS = 50258, 50259, 50260, 50360, 50364
CB = [SOT, ZH, TRANSCRIBE, NOTIMESTAMPS]
CE_PROMPT = [SOT, EN, TRANSCRIBE, NOTIMESTAMPS]
ALL = "ALL"


def ttls_lr(d_model: int) -> float:
    return E_STAR / (2.0 * math.sqrt(d_model))


# ---- frozen candidate / stable-position rules (pure, CPU float64) ---------------------------------------------

def processed_logprobs(logits: torch.Tensor, suppress: Sequence[int], begin: Sequence[int]) -> np.ndarray:
    """(T+1, V) float32 full-vocabulary log-softmax of generate()-processed logits (begin-suppress at row 0 only),
    upcast to float64. Row t is the query predicting content step t."""
    z = logits.detach().float().cpu().clone()
    if suppress:
        z[:, list(suppress)] = -float("inf")
    if begin and z.shape[0]:
        z[0, list(begin)] = -float("inf")
    return torch.log_softmax(z, dim=-1).numpy().astype(np.float64)


def select_candidates(lp: dict, y_B: Sequence[int], embedded: set, utf8_ok: Sequence[bool], eos: int,
                      tau: float = TAU) -> dict:
    """Frozen AC rule on the B0 path. ``lp`` holds (T+1, V) float64 log-prob arrays for 'cB_clean', 'cB_null',
    'cE_clean', 'cE_null' on the same prefix y_B. For every content step t (1 <= t < len(y_B)):

      structural  t >= 1, UTF-8-complete prefix y_B[:t], b_t = y_B[t] not EOS and not embedded-Latin;
      proposal    e_t = argmax_v lp['cE_clean'][t] (lowest ID on ties); language-disagreement iff e_t is embedded-Latin
                  (then e_t != b_t);
      evidence    A^c(v) = l^c_clean(v) - max(l^c_null(v), log 1e-12),  E^c = A^c(e_t) - A^c(b_t), c in {cB, cE};
      accept      E^cB >= tau AND E^cE >= tau (tau = log 10), all four log-probs finite.

    Returns the per-step records, the language-disagreement steps M (acoustic-free), the accepted steps and the first
    accepted step t* (None => abstain)."""
    T = len(y_B)
    rec, M, acc = [], [], []
    for t in range(1, T):
        b = int(y_B[t])
        r = {"t": t, "b": b, "structural": bool(utf8_ok[t]) and b != eos and b not in embedded}
        if not r["structural"]:
            rec.append(r)
            continue
        row = lp["cE_clean"][t]
        best = float(np.max(row))
        e = int(np.flatnonzero(row == best).min())
        r["e"] = e
        r["proposal"] = e in embedded and e != b
        if not r["proposal"]:
            rec.append(r)
            continue
        M.append(t)
        ev = {}
        fin = True
        for c in ("cB", "cE"):
            lc, ln = lp[f"{c}_clean"][t], lp[f"{c}_null"][t]
            vals = (lc[e], ln[e], lc[b], ln[b])
            fin &= all(math.isfinite(v) for v in (lc[e], lc[b])) and not any(math.isnan(v) for v in vals)
            a_e = lc[e] - max(ln[e], NULL_FLOOR)
            a_b = lc[b] - max(ln[b], NULL_FLOOR)
            ev[c] = {"A_e": float(a_e), "A_b": float(a_b), "E": float(a_e - a_b), "l_clean_e": float(lc[e]),
                     "l_clean_b": float(lc[b]), "l_null_e": float(ln[e]), "l_null_b": float(ln[b]),
                     "null_floor_hit_e": bool(ln[e] < NULL_FLOOR)}
        r["evidence"] = ev
        r["finite"] = bool(fin)
        r["accepted"] = bool(fin and ev["cB"]["E"] >= tau and ev["cE"]["E"] >= tau)
        r["support"] = float(min(ev["cB"]["E"], ev["cE"]["E"]))
        if r["accepted"]:
            acc.append(t)
        rec.append(r)
    return {"records": rec, "M": M, "accepted": acc, "t_star": acc[0] if acc else None,
            "c_star": next((r["e"] for r in rec if r["t"] == acc[0]), None) if acc else None}


def stable_positions(p0_top: Sequence[float], valid: Sequence[bool], M: Sequence[int]) -> list[int]:
    """Frozen preservation set on the B0 path: valid content steps t (0 <= t < len(y_B)) whose theta0 forced-ZH
    probability of its own greedy token is >= 0.5 and that are not language-disagreement steps."""
    m = set(int(x) for x in M)
    return [t for t, (p, v) in enumerate(zip(p0_top, valid)) if v and p >= STABLE_P_MIN and t not in m]


# ---- TTLS edit at the canonical DG-02 site ---------------------------------------------------------------------

def ttls_hook(bundle, z: torch.Tensor, steps, *, layer: int = LAYER, mode: str = "train", record: bool = False):
    """Canonical DG-02 hook adding ``z`` (unit gain) at content steps ``steps`` (a set of ints, or ALL = every step
    t >= 1), i.e. absolute query positions PROMPT_LEN - 1 + t; NormPreserve via apply_steering."""
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook
    targets = None if steps == ALL else torch.tensor(sorted(PROMPT_LEN - 1 + int(t) for t in steps), dtype=torch.long)

    def action_fn(q, u_source, r, abs_pos):
        if targets is None:
            g = torch.ones((1, r.shape[1]), device=r.device, dtype=r.dtype)
        else:
            g = torch.isin(abs_pos, targets.to(abs_pos.device)).to(r.dtype).view(1, -1)
        return g, z
    return DecoderPostCrossAttnInterventionHook(bundle, layer, None, alpha=1.0, num_forced_prefix=PROMPT_LEN,
                                                action_fn=action_fn, norm_preserve=True, mode=mode, record=record,
                                                record_last_only=False)


# ---- one episode (2 AdamW steps, 3 loss evaluations) -----------------------------------------------------------

class Episode:
    """Fresh per-utterance variable + optimizer. ``forward(prompt, y)`` returns float32 teacher-forced logits for
    queries len(prompt)-1 .. len(prompt)-1+len(y) under the current variable value."""

    def __init__(self, bundle, guard: LNGuard | None, encoded, variable: str, steps=None, layer: int = LAYER):
        self.bundle, self.model, self.guard, self.encoded, self.variable = bundle, bundle.model, guard, encoded, variable
        self.layer = layer
        if variable == "LN":
            if not guard.verify():
                raise TTAInvalid("resident LN != theta0 at episode start")
            self.params = guard.fresh_masters()
            self.opt = torch.optim.AdamW(list(self.params.values()), **OPTIM)
            self.trainable = int(sum(p.numel() for p in self.params.values()))
        elif variable == "TTLS":
            d = int(bundle.d_model)
            dev = next(self.model.parameters()).device
            self.z = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32, device=dev))
            self.params = {"z": self.z}
            self.opt = torch.optim.AdamW([self.z], **{**OPTIM, "lr": ttls_lr(d)})
            self.steps = steps
            self.trainable = d
        else:
            raise ValueError(variable)

    def forward(self, prompt, y) -> torch.Tensor:
        if self.variable == "LN":
            vals = {n: m.to(torch.bfloat16) for n, m in self.params.items()}
            return teacher_logits(self.model, self.encoded, prompt, y, vals)
        with ttls_hook(self.bundle, self.z, self.steps, layer=self.layer, mode="train"):
            return teacher_logits(self.model, self.encoded, prompt, y, None)

    def project(self) -> bool:
        if self.variable != "TTLS":
            return False
        with torch.no_grad():
            n = float(torch.linalg.vector_norm(self.z))
            if n > E_STAR:
                self.z.mul_(E_STAR / n)
                return True
        return False

    def effective(self):
        with torch.no_grad():
            if self.variable == "LN":
                return {n: m.detach().to(torch.bfloat16) for n, m in self.params.items()}
            return self.z.detach().clone()

    def run(self, loss_fn: Callable[["Episode"], dict], counters: dict, keep_grad0: bool = False) -> dict:
        """loss_fn(ep) -> {'loss': scalar tensor, 'parts': {name: float}}. Returns the episode log."""
        log = {"variable": self.variable, "trainable_scalars": self.trainable, "losses": [], "parts": [], "grad_l2": [],
               "finite": [], "projected": [], "z_norm": []}
        grad0 = None
        t0 = time.time()
        for k in range(STEPS + 1):
            with torch.enable_grad():
                out = loss_fn(self)
                loss = out["loss"]
            counters["loss_evaluations"] = counters.get("loss_evaluations", 0) + 1
            if not math.isfinite(float(loss)):
                raise TTAInvalid(f"nonfinite loss step {k}")
            log["losses"].append(float(loss))
            log["parts"].append(out["parts"])
            if k < STEPS:
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                counters["backwards"] = counters.get("backwards", 0) + 1
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params.values()]
                g2 = _l2(grads)
                fin = math.isfinite(g2) and all(bool(torch.isfinite(g).all()) for g in grads)
                log["grad_l2"].append(g2)
                log["finite"].append(fin)
                if not fin:
                    raise TTAInvalid(f"nonfinite gradient step {k}")
                if k == 0 and keep_grad0:
                    grad0 = torch.cat([g.detach().flatten() for g in grads]).cpu()
                self.opt.step()
                counters["optimizer_steps"] = counters.get("optimizer_steps", 0) + 1
                log["projected"].append(self.project())
                if not all(bool(torch.isfinite(p).all()) for p in self.params.values()):
                    raise TTAInvalid(f"nonfinite variable step {k}")
                if self.variable == "TTLS":
                    log["z_norm"].append(float(torch.linalg.vector_norm(self.z.detach())))
            del out, loss
        log["adapt_sec"] = time.time() - t0
        with torch.no_grad():
            if self.variable == "LN":
                if not self.guard.verify():
                    raise TTAInvalid("resident LN changed during adaptation")
                deltas = [self.params[n].detach() - self.guard.theta0[n].float() for n in self.guard.names]
                eff = [self.params[n].detach().to(torch.bfloat16).float() - self.guard.theta0[n].float() for n in self.guard.names]
                log["master_delta_l2"] = _l2(deltas)
                log["effective_delta_l2"] = _l2(eff)
            else:
                log["master_delta_l2"] = float(torch.linalg.vector_norm(self.z.detach()))
                log["effective_delta_l2"] = float(torch.linalg.vector_norm(self.z.detach().to(torch.bfloat16).float()))
        log["steps"] = STEPS
        return {"log": log, "grad0": grad0}


# ---- objectives -------------------------------------------------------------------------------------------------

def ce_terms(ep: Episode, y_A, mask_A, suppress, begin, eos, prompt=CB):
    logits = ep.forward(prompt, y_A)
    terms = position_terms(logits, y_A, suppress, begin, eos)
    return logits, objective_loss("A2", terms, mask_A, ep.params)


def ac_term(logits_B: torch.Tensor, t_star: int, c_star: int, suppress, begin) -> torch.Tensor:
    from csasr.inference_cf.readout import processed_logits
    lp = torch.log_softmax(processed_logits(logits_B[t_star], t_star, suppress, begin), dim=-1)
    return -lp[int(c_star)]


def preservation_term(logits_B: torch.Tensor, logq_B: list, S: Sequence[int], T: int, suppress, begin) -> torch.Tensor:
    """mean_{t in S} KL(p_theta0,t || p_theta,t) on the B0 path (content steps; allowed vocabulary)."""
    if not S:
        return logits_B.sum() * 0.0
    kl = kl_terms(logq_B, allowed_log_probs(logits_B, T, suppress, begin))
    idx = torch.tensor(list(S), dtype=torch.long, device=kl.device)
    return kl.index_select(0, idx).mean()


# ---- decoding ----------------------------------------------------------------------------------------------------

def greedy_decode(bundle, encoded, prompt: Sequence[int], *, hook_factory=None, forced: Sequence[int] = (),
                  max_new_tokens: int = 200, capture_layer: int = LAYER) -> dict:
    """Forced-prompt greedy decode on a fresh KV cache (cached.Branch.step; processed argmax, unchanged suppression).
    ``hook_factory()`` (optional) returns a fresh DG-02 hook installed on every step. ``forced`` content tokens are fed
    verbatim before greedy continuation (AC-SUB comparator)."""
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    eos = bundle.processor.tokenizer.eos_token_id
    br = cached.Branch(bundle, encoded, list(prompt), "T")
    new, tokens, term = list(prompt), [], "cap"
    with torch.inference_mode():
        for t in range(max_new_tokens):
            assert_no_site_hooks(bundle)
            hook = hook_factory() if hook_factory is not None else None
            logits, _, _ = br.step(new, capture_layer=capture_layer, attention=True, hook=hook)   # = forced_decode numerics
            nxt = int(forced[t]) if t < len(forced) else processed_argmax(logits, t, suppress, begin)
            if nxt == eos:
                term = "eos"
                break
            tokens.append(nxt)
            new = [nxt]
    assert_no_site_hooks(bundle)
    return {"tokens": tokens, "terminated": term, "length": len(tokens),
            "text": bundle.processor.tokenizer.decode(tokens, skip_special_tokens=True)}


def cd_decode(bundle, enc_clean, enc_null, prompt: Sequence[int], legal_mask: np.ndarray, byte_decoder: dict,
              max_new_tokens: int = 200, capture_layer: int = LAYER) -> dict:
    """No-update comparator: full-sequence acoustic-contrastive decoding with the frozen DIR-SPRINT0 D3 rule
    (``dir_sprint0.d3_candidate``) at every step t >= 1 with a UTF-8-complete prefix whose clean greedy token is not
    EOS; otherwise the clean greedy token. Clean and null branches share prompt and emitted prefix."""
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.core_r2 import prefix_utf8_complete
    from csasr.inference_cf.dir_sprint0 import d3_candidate
    tok = bundle.processor.tokenizer
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    eos = tok.eos_token_id
    B = cached.Branch(bundle, enc_clean, list(prompt), "B")
    N = cached.Branch(bundle, enc_null, list(prompt), "N")
    new, tokens, term, edits = list(prompt), [], "cap", []
    with torch.inference_mode():
        for t in range(max_new_tokens):
            lb, _, _ = B.step(new, capture_layer=capture_layer, attention=True)
            ln, _, _ = N.step(new, capture_layer=capture_layer, attention=True)
            nxt = processed_argmax(lb, t, suppress, begin)
            if t >= 1 and nxt != eos and prefix_utf8_complete(tok, tokens, byte_decoder):
                d = d3_candidate(lb.numpy(), ln.numpy(), legal_mask, suppress, nxt)
                if d["status"] == "candidate":
                    edits.append({"t": t, "b": nxt, "c": d["c_AP"]})
                    nxt = int(d["c_AP"])
            if nxt == eos:
                term = "eos"
                break
            tokens.append(nxt)
            new = [nxt]
    return {"tokens": tokens, "terminated": term, "length": len(tokens), "edits": edits,
            "text": tok.decode(tokens, skip_special_tokens=True)}


# ---- reference-free displacement diagnostics ----------------------------------------------------------------------

@torch.no_grad()
def displacement(bundle, encoded, y_B, *, values=None, z=None, steps=None, base: dict | None = None,
                 suppress=(), begin=(), S: Sequence[int] = (), edit_steps: Sequence[int] = (), layer: int = LAYER,
                 prompt: Sequence[int] = CB) -> dict:
    """Teacher-forced forward on the B0 path recording the L16 site and logits. Without ``base`` returns the theta0
    reference; otherwise relative L16 site displacement (all content queries / edited steps) and mean
    KL(p_theta0 || p_adapted) over all content steps and over S."""
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    model = bundle.model
    T = len(y_B)
    ids = torch.tensor([list(prompt) + list(y_B)], device=next(model.parameters()).device, dtype=torch.long)
    kw = dict(encoder_outputs=encoded, decoder_input_ids=ids, use_cache=False, return_dict=True)
    hook = ttls_hook(bundle, z, steps, layer=layer, mode="steer") if z is not None else None
    rec = DecoderPostCrossAttnRecorder(bundle, [layer])
    ctx = [c for c in (hook, rec) if c is not None]          # hook first: the recorder sees the edited site
    for c in ctx:
        c.__enter__()
    try:
        if values is None:
            out = model(**kw)
        else:
            out = torch.func.functional_call(model, values, args=(), kwargs=kw, strict=False, tie_weights=True)
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    q0 = PROMPT_LEN - 1
    site = rec.states[layer][0, q0:q0 + T + 1].float()
    logits = out.logits[0, q0:q0 + T + 1].float()
    if base is None:
        return {"site": site, "logq": [x.detach() for x in allowed_log_probs(logits, T, suppress, begin)]}
    rel = (torch.linalg.vector_norm(site - base["site"], dim=-1) / torch.linalg.vector_norm(base["site"], dim=-1)).cpu()
    kl = kl_terms(base["logq"], allowed_log_probs(logits, T, suppress, begin)).cpu() if T else torch.zeros(0)
    es = [t for t in edit_steps if 0 <= t <= T]
    return {"site_rel_mean": float(rel[:T + 1].mean()), "site_rel_edited_mean": float(rel[es].mean()) if es else None,
            "site_rel_max": float(rel.max()), "kl_mean_all": float(kl.mean()) if T else 0.0,
            "kl_mean_stable": float(kl[list(S)].mean()) if S else None, "kl_max": float(kl.max()) if T else 0.0}
