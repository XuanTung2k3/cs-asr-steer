#!/usr/bin/env python
"""P2-TTA0 engineering diagnostic (CPU only, post-hoc, NOT a scientific job): why did the first-row live A1
gradient check of run1 fail (loss |diff| 7e-7 but gradient relative diff 1.0e-2 > frozen 1e-3) while A2 matched
bitwise?

On the real bf16 Whisper-large-v3 (CPU, eager) and panel row 0's sealed theta0 teachers only, compute the theta0
step-0 master gradient for each objective with four logit-level formulas, all through the same bf16 functional_call
model backward:
  P32 primary position_terms (float32), A32 auditor live_objective_check (float32),
  P64 primary formula on float64 logits, A64 auditor formula on float64 logits.
Reports pairwise relative L2 differences. Reads no reference, evaluator output or adapted transcript; changes no
outcome, label, threshold or frozen artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.modeling_outputs import BaseModelOutput
    import csasr.inference_cf.episodic_tta as tta
    import experiments.inference_cf_p2tta0_audit as au
    from csasr.models.whisper import load_audio
    model_dir = "/mnt/data/tungnx/whisper-large-v3"
    seal = json.loads((ROOT / "results/inference_cf/p2tta0/pseudo_sealed.json").read_text())
    s = seal["rows"][0]
    proc = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True,
                                                            attn_implementation="eager").eval()
    model.requires_grad_(False)
    tok = proc.tokenizer
    sup, beg, eos = seal["suppression"]["suppress"], seal["suppression"]["begin"], seal["eos"]
    feats = proc.feature_extractor([load_audio(s["audio_path"], 16000)], sampling_rate=16000, return_tensors="pt",
                                   return_attention_mask=True)
    with torch.inference_mode():
        h = model.model.encoder(input_features=feats["input_features"].to(torch.bfloat16),
                                attention_mask=feats["attention_mask"]).last_hidden_state
    enc = BaseModelOutput(last_hidden_state=h.clone())
    names = tta.decoder_ln_names(model)
    guard = tta.LNGuard(model, names)
    CB = [50258, 50260, 50360, 50364]

    def primary(kind, y, mask, dtype):
        masters = guard.fresh_masters()
        with torch.enable_grad():
            logits = tta.teacher_logits(model, enc, CB, y, {n: m.to(torch.bfloat16) for n, m in masters.items()}).to(dtype)
            terms = tta.position_terms(logits, y, sup, beg, eos)
            loss = tta.objective_loss(kind, terms, mask, masters)
            loss.backward()
        return float(loss), torch.cat([masters[n].grad.flatten() for n in names]).double()

    def auditor(kind, y, dtype):
        if dtype == torch.float32:
            r = au.live_objective_check(model, names, enc, CB, y, kind, suppress=sup, begin=beg, tokenizer=tok)
            return r["auditor_loss"], r["_grad"].double()
        masters = [torch.nn.Parameter(guard.theta0[n].float().clone()) for n in names]
        valid = torch.tensor(au.own_valid(y, sup, beg, tok))
        with torch.enable_grad():
            out = torch.func.functional_call(model, {n: m.to(torch.bfloat16) for n, m in zip(names, masters)}, args=(),
                                             kwargs={"encoder_outputs": enc, "decoder_input_ids": torch.tensor([CB + list(y)]),
                                                     "use_cache": False}, strict=False, tie_weights=True)
            q = out.logits[0].to(dtype)[len(CB) - 1:len(CB) - 1 + len(y)]
            ban = torch.zeros_like(q, dtype=torch.bool)
            ban[:, sup] = True
            ban[0, beg] = True
            q = q.masked_fill(ban, float("-inf"))
            if kind == "A1":
                loss = torch.distributions.Categorical(logits=q[valid]).entropy().mean()
            else:
                loss = torch.nn.functional.cross_entropy(q[valid], torch.tensor(list(y))[valid])
            loss.backward()
        return float(loss), torch.cat([m.grad.flatten() for m in masters]).double()

    res = {"schema": "p2_tta0_live_audit_diagnostic_v1", "utterance_id": s["utterance_id"], "device": "cpu", "dtype_model": "bfloat16",
           "note": "engineering diagnostic only; CPU encoder/backward differ numerically from the GPU run", "objectives": {}}
    t0 = time.time()
    for kind, y, mask in (("A1", s["y_B"], s["y_B_valid_mask"]), ("A2", s["y_A"], s["y_A_valid_mask"])):
        g, L = {}, {}
        L["P32"], g["P32"] = primary(kind, y, mask, torch.float32)
        L["P32_repeat"], g["P32_repeat"] = primary(kind, y, mask, torch.float32)
        L["A32"], g["A32"] = auditor(kind, y, torch.float32)
        L["P64"], g["P64"] = primary(kind, y, mask, torch.float64)
        L["A64"], g["A64"] = auditor(kind, y, torch.float64)
        nrm = lambda x: float(torch.linalg.vector_norm(x))
        pairs = {f"{a}_vs_{b}": nrm(g[a] - g[b]) / nrm(g[b]) for a, b in (("P32_repeat", "P32"), ("A32", "P32"), ("P32", "P64"),
                                                                          ("A32", "P64"), ("A64", "P64"), ("A32", "A64"))}
        res["objectives"][kind] = {"loss": L, "grad_l2": {k: nrm(v) for k, v in g.items()}, "relative_l2_diff": pairs,
                                   "frozen_tolerance_relative": 1e-3}
        assert guard.verify()
    res["elapsed_sec"] = time.time() - t0
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("never overwrite")
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res["objectives"], indent=1))


if __name__ == "__main__":
    main()
