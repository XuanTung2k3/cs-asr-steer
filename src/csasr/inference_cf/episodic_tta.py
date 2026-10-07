"""P2-TTA0 episodic decoder-LayerNorm test-time adaptation primitives (frozen contract
docs/inference_cf/P2_TTA0_SPEC.md, P2_TTA0_CODEX_DESIGN.md, configs/inference_cf/p2_tta0.json).

* Trainables: ONLY the enumerated decoder ``torch.nn.LayerNorm`` affine tensors. Resident model parameters stay
  bf16 with ``requires_grad=False``; each objective creates fresh fp32 master ``nn.Parameter``s from theta0 and a
  fresh AdamW over exactly those masters. The differentiable forward is
  ``torch.func.functional_call(model, {name: master.to(bfloat16)}, ..., strict=False, tie_weights=True)``.
* Teacher forcing: decoder input ``prompt + y`` (no EOS appended), ``use_cache=False``; content token ``t`` is
  predicted by raw logits at absolute query ``len(prompt) - 1 + t``; the final query reports EOS only.
  Allowed vocabulary = all IDs minus ``suppress_tokens`` (minus ``begin_suppress_tokens`` at t = 0), float32
  ``log_softmax`` over the gathered allowed IDs (EOS stays allowed). Valid loss positions are fixed from the
  theta0 teacher: non-special content ID whose target is allowed at its step.
* A1 GREEDY-EM: mean valid-position entropy along the theta0 forced teacher. A2 AUTO-CONSISTENCY: mean
  valid-position -log p(y_A[t]) under forced-ZH. Exactly 2 updates, 3 loss evaluations.
* ``LNGuard`` restores the original bf16 LN bytes after every objective (in ``finally``) and verifies them bitwise.

No reference, evaluator, steering, D2 or LID input anywhere in this module.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch

OPTIM = {"lr": 1e-3, "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False, "foreach": False,
         "fused": False, "maximize": False}
STEPS = 2
OBJECTIVES = ("A1", "A2")


class TTAInvalid(RuntimeError):
    """Technical failure (nonfinite, reset/path mismatch, unexpected teacher token) => stage INVALID."""


# ---- trainable enumeration / reset -----------------------------------------------------------------------

def decoder_ln_names(model) -> list[str]:
    """Affine tensor names of every torch.nn.LayerNorm inside model.model.decoder, in module order."""
    dec = model.model.decoder
    names = []
    for mname, mod in dec.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            for p in ("weight", "bias"):
                if getattr(mod, p, None) is not None:
                    names.append(f"model.decoder.{mname}.{p}")
    return names


def tensor_bytes_hash(tensors: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        x = t.detach().contiguous().cpu()
        h.update(str(x.dtype).encode() + str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes() if x.dtype == torch.bfloat16 else x.numpy().tobytes())
    return "sha256:" + h.hexdigest()


class LNGuard:
    """Immutable theta0 snapshot of the selected resident LN tensors + exact restore/verify."""

    def __init__(self, model, names: Sequence[str]):
        params = dict(model.named_parameters())
        missing = [n for n in names if n not in params]
        if missing:
            raise TTAInvalid(f"missing trainable names {missing[:3]}")
        self.model, self.names = model, list(names)
        self.params = {n: params[n] for n in self.names}
        self.theta0 = {n: p.detach().clone() for n, p in self.params.items()}
        self.theta0_hash = tensor_bytes_hash([self.theta0[n] for n in self.names])
        self.others = [p for n, p in model.named_parameters() if n not in self.params]

    def flags_ok(self) -> bool:
        return all(not p.requires_grad and p.grad is None for p in self.model.parameters())

    def other_versions(self) -> int:
        return int(sum(p._version for p in self.others))

    @torch.no_grad()
    def materialize(self, values: dict[str, torch.Tensor]) -> None:
        for n in self.names:
            v = values[n]
            if v.dtype != self.params[n].dtype or v.shape != self.params[n].shape:
                raise TTAInvalid(f"materialize dtype/shape {n}")
            self.params[n].copy_(v)

    @torch.no_grad()
    def restore(self) -> None:
        for n in self.names:
            self.params[n].copy_(self.theta0[n])
            self.params[n].grad = None
            self.params[n].requires_grad_(False)

    def verify(self) -> bool:
        return all(torch.equal(self.params[n], self.theta0[n]) for n in self.names) and self.flags_ok()

    def current_hash(self) -> str:
        return tensor_bytes_hash([self.params[n] for n in self.names])

    def fresh_masters(self) -> dict[str, torch.nn.Parameter]:
        m = {n: torch.nn.Parameter(self.theta0[n].float().clone()) for n in self.names}
        if not all(torch.equal(m[n].detach(), self.theta0[n].float()) for n in self.names):
            raise TTAInvalid("fresh masters != theta0.float")
        return m


# ---- teacher forcing ---------------------------------------------------------------------------------------

def special_ids(tokenizer) -> set[int]:
    eos = int(tokenizer.eos_token_id)
    return set(int(i) for i in tokenizer.all_special_ids) | {eos}


def is_special(tok_id: int, specials: set[int], eos: int) -> bool:
    return int(tok_id) in specials or int(tok_id) >= eos


def clean_teacher(tokens: Sequence[int], prompt: Sequence[int], eos: int, specials: set[int]) -> list[int]:
    """Strip a verified leading Whisper prompt and terminal EOS if present; interior special/control => INVALID."""
    y = [int(t) for t in tokens]
    if y and y[0] == prompt[0]:
        k = 1
        while k < len(y) and is_special(y[k], specials, eos) and y[k] != eos:
            k += 1
        y = y[k:]
    if y and y[-1] == eos:
        y = y[:-1]
    if any(is_special(t, specials, eos) for t in y):
        raise TTAInvalid("unexpected interior special/control token in teacher")
    return y


def allowed_ids(vocab: int, step: int, suppress: Sequence[int], begin: Sequence[int], device=None) -> torch.Tensor:
    keep = torch.ones(vocab, dtype=torch.bool)
    if suppress:
        keep[list(suppress)] = False
    if step == 0 and begin:
        keep[list(begin)] = False
    return torch.nonzero(keep).flatten().to(device)


def valid_mask(y: Sequence[int], suppress: Sequence[int], begin: Sequence[int], specials: set[int], eos: int) -> list[bool]:
    s, b = set(int(x) for x in suppress or []), set(int(x) for x in begin or [])
    return [not is_special(t, specials, eos) and t not in s and not (i == 0 and t in b) for i, t in enumerate(y)]


def teacher_logits(model, encoded, prompt: Sequence[int], y: Sequence[int], values: dict | None) -> torch.Tensor:
    """float32 logits for queries len(prompt)-1 .. len(prompt)-1+len(y) (content steps then the final EOS query)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([list(prompt) + list(y)], device=dev, dtype=torch.long)
    kw = dict(encoder_outputs=encoded, decoder_input_ids=ids, use_cache=False, return_dict=True)
    if values is None:
        out = model(**kw)
    else:
        out = torch.func.functional_call(model, values, args=(), kwargs=kw, strict=False, tie_weights=True)
    q0 = len(prompt) - 1
    return out.logits[0, q0:q0 + len(y) + 1].float()


def position_terms(logits: torch.Tensor, y: Sequence[int], suppress, begin, eos: int, partition: dict | None = None) -> dict:
    """Per content position: entropy over the allowed vocabulary and target log-prob (float32), plus EOS prob at
    every query (incl. final) and optional script masses. Differentiable w.r.t. ``logits``."""
    V = logits.shape[-1]
    if not bool(torch.isfinite(logits).all()):
        raise TTAInvalid("nonfinite logits")
    T = len(y)
    dev = logits.device
    out = {"eos_prob": [], "P_E": [], "P_M": []}
    ents, tgts = [], []
    for rows, step in ((slice(0, 1), 0), (slice(1, T + 1), 1)):
        x = logits[rows]
        if x.shape[0] == 0:
            continue
        idx = allowed_ids(V, step, suppress, begin, dev)
        inv = torch.full((V,), -1, dtype=torch.long, device=dev)
        inv[idx] = torch.arange(len(idx), device=dev)
        lp = torch.log_softmax(x.index_select(1, idx), dim=-1)
        p = lp.exp()
        ents.append(-(p * lp).sum(-1))
        yy = torch.tensor([int(v) for v in y[rows]], dtype=torch.long, device=dev)
        n = len(yy)                                    # content rows (the final query has no target)
        if n:
            k = inv[yy]
            g = lp[torch.arange(n, device=dev), k.clamp(min=0)]
            tgts.append(torch.where(k >= 0, g, torch.full_like(g, float("nan"))))
        with torch.no_grad():
            ke = int(inv[eos])
            out["eos_prob"] += (p[:, ke] if ke >= 0 else torch.zeros(len(p), device=dev)).tolist()
            if partition is not None:
                for key, name in (("embedded_ids", "P_E"), ("matrix_ids", "P_M")):
                    sel = inv[torch.tensor(partition[key], dtype=torch.long, device=dev)]
                    sel = sel[sel >= 0]
                    out[name] += p[:, sel].sum(-1).tolist()
    ent = torch.cat(ents)
    out["entropy"] = ent[:T]
    out["target_logprob"] = torch.cat(tgts) if tgts else logits.new_zeros(0)
    return out


def objective_loss(kind: str, terms: dict, mask: Sequence[bool], masters: dict) -> torch.Tensor:
    m = torch.tensor(list(mask), dtype=torch.bool, device=terms["entropy"].device)
    if int(m.sum()) == 0:      # empty / no-valid teacher: scalar zero connected to masters (zero gradients)
        return sum((p * 0).sum() for p in masters.values())
    if kind == "A1":
        return terms["entropy"][m].mean()
    if kind == "A2":
        return (-terms["target_logprob"][m]).mean()
    raise ValueError(kind)


def _l2(ts) -> float:
    return math.sqrt(sum(float(t.detach().double().pow(2).sum()) for t in ts))


def adapt(model, guard: LNGuard, encoded, prompt: Sequence[int], y: Sequence[int], mask: Sequence[bool], kind: str, *,
          suppress, begin, eos: int, partition: dict | None = None, counters: dict | None = None,
          keep_grad0: bool = False) -> dict:
    """Fresh masters + fresh AdamW from theta0; L0/update1, L1/update2, L2 (no update). Returns logs, the effective
    bf16 values and the fp32 master deltas. Does NOT touch resident parameters."""
    c = counters if counters is not None else {}
    if not guard.verify():
        raise TTAInvalid("resident LN != theta0 at objective start")
    masters = guard.fresh_masters()
    opt = torch.optim.AdamW(list(masters.values()), **OPTIM)
    if type(opt) is not torch.optim.AdamW or opt.defaults["lr"] != OPTIM["lr"]:
        raise TTAInvalid("optimizer")
    log = {"kind": kind, "losses": [], "grad_l2": [], "finite": [], "valid": int(sum(mask)), "content": len(y),
           "no_valid_content": int(sum(mask)) == 0, "positions": []}
    grad0 = None
    common = None
    local_steps = 0
    for k in range(STEPS + 1):
        with torch.enable_grad():
            vals = {n: m.to(torch.bfloat16) for n, m in masters.items()}
            logits = teacher_logits(model, encoded, prompt, y, vals)
            c["teacher_forwards"] = c.get("teacher_forwards", 0) + 1
            terms = position_terms(logits, y, suppress, begin, eos, partition if kind == "A1" else None)
            loss = objective_loss(kind, terms, mask, masters)
        if not math.isfinite(float(loss)):
            raise TTAInvalid(f"nonfinite loss {kind} step {k}")
        log["losses"].append(float(loss))
        log["positions"].append({"entropy": terms["entropy"].detach().cpu().tolist(),
                                 "target_logprob": [None if math.isnan(v) else v for v in terms["target_logprob"].detach().cpu().tolist()],
                                 "eos_prob": terms["eos_prob"], "P_E": terms["P_E"], "P_M": terms["P_M"]})
        if kind == "A1" and k in (0, STEPS):
            common = common or {}
            common["theta0" if k == 0 else "theta2"] = _common_summary(terms, mask)
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
                raise TTAInvalid(f"nonfinite gradient {kind} step {k}")
            if k == 0 and keep_grad0:
                grad0 = torch.cat([g.detach().flatten() for g in grads]).cpu()
            opt.step()
            c["optimizer_steps"] = c.get("optimizer_steps", 0) + 1
            local_steps += 1
            if not all(bool(torch.isfinite(m).all()) for m in masters.values()):
                raise TTAInvalid(f"nonfinite master {kind} step {k}")
        del logits, terms, loss
    if not guard.verify():
        raise TTAInvalid("resident LN changed during adaptation")
    with torch.no_grad():
        deltas = {n: (masters[n].detach() - guard.theta0[n].float()) for n in guard.names}
        eff = {n: masters[n].detach().to(torch.bfloat16) for n in guard.names}
        eff_delta = [eff[n].float() - guard.theta0[n].float() for n in guard.names]
        base = _l2([guard.theta0[n].float() for n in guard.names])
        log["master_delta_l2"] = _l2(deltas.values())
        log["master_delta_rel"] = log["master_delta_l2"] / (base + 1e-12)
        log["effective_delta_l2"] = _l2(eff_delta)
        log["effective_changed_scalars"] = int(sum(int((d != 0).sum()) for d in eff_delta))
        log["theta0_ln_l2"] = base
    log["steps"] = local_steps
    log["loss_evaluations"] = len(log["losses"])
    del opt
    return {"log": log, "effective": eff, "final_masters": {n: masters[n].detach().cpu().clone() for n in guard.names}, "grad0": grad0,
            "common_y_B": common, "masters": masters}


def _common_summary(terms: dict, mask: Sequence[bool]) -> dict:
    ent = terms["entropy"].detach().cpu()
    m = torch.tensor(list(mask), dtype=torch.bool)
    n = int(m.sum())
    return {"entropy_sum": float(ent[m].double().sum()) if n else 0.0, "valid": n,
            "eos_prob": list(terms["eos_prob"]), "P_E": list(terms["P_E"]), "P_M": list(terms["P_M"]),
            "entropy": ent.tolist()}


def common_diagnostic(model, masters: dict, encoded, prompt, y_B, mask_B, *, suppress, begin, eos, partition,
                      counters: dict | None = None) -> dict:
    """theta2 (A2) common-y_B diagnostic forward (no grad, no update)."""
    with torch.no_grad():
        vals = {n: m.detach().to(torch.bfloat16) for n, m in masters.items()}
        logits = teacher_logits(model, encoded, prompt, y_B, vals)
        if counters is not None:
            counters["teacher_forwards"] = counters.get("teacher_forwards", 0) + 1
        return _common_summary(position_terms(logits, y_B, suppress, begin, eos, partition), mask_B)


# ---- ordinary forced decode / distances ----------------------------------------------------------------------

def forced_decode(bundle, encoded, prompt: Sequence[int], *, max_new_tokens: int = 200, capture_layer: int | None = 16) -> dict:
    """Ordinary no-hook forced decode: cached.Branch.step (prompt then one token), attention=True + passive recorder,
    processed_argmax with unchanged suppression; fresh KV cache for every call."""
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    eos = bundle.processor.tokenizer.eos_token_id
    br = cached.Branch(bundle, encoded, list(prompt), "F")
    new, tokens, term = list(prompt), [], "cap"
    with torch.inference_mode():
        for t in range(max_new_tokens):
            assert_no_site_hooks(bundle)
            logits, _, _ = br.step(new, capture_layer=capture_layer, attention=True)
            nxt = processed_argmax(logits, t, suppress, begin)
            if nxt == eos:
                term = "eos"
                break
            tokens.append(nxt)
            new = [nxt]
    return {"tokens": tokens, "terminated": term, "length": len(tokens), "steps": len(tokens) + (term == "eos"),
            "text": bundle.processor.tokenizer.decode(tokens, skip_special_tokens=True)}


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def compare(tokens: Sequence[int], ref_tokens: Sequence[int]) -> dict:
    fd = next((i for i, (a, b) in enumerate(zip(tokens, ref_tokens)) if a != b), None)
    if fd is None and len(tokens) != len(ref_tokens):
        fd = min(len(tokens), len(ref_tokens))
    return {"equal": list(tokens) == list(ref_tokens), "first_difference": fd, "levenshtein": levenshtein(tokens, ref_tokens)}


def severe_truncation(base_len: int, adapted_len: int, adapted_terminated: str) -> bool:
    return base_len >= 10 and adapted_terminated == "eos" and adapted_len <= math.floor(0.5 * base_len)
