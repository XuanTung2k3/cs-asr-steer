"""P2 frozen Auto-LID decoding contract and the shared teacher-forced ASR objective.

Frozen Auto-LID (two stages):
  A. language = argmax over Whisper's language tokens of the logits at <|sot|>, computed on input_features[:, :, :3000]
     by the ORIGINAL frozen backbone with every adapter disabled (exactly WhisperGenerationMixin.detect_language).
  B. adapted decoding with that language forced: generate(language=<code>, task="transcribe", ...) -- prompt
     [<|sot|>, <|lang|>, <|transcribe|>, <|notimestamps|>]. No reference or oracle language label is ever used, and
     non-zh/en detections are kept as detected.

Teacher forcing: X = prompt + target[:-1] with target = tokenizer(target_text) + [EOS]; the logit at decoder position
3 + t predicts target[t] (the <|notimestamps|> state predicts content token 0); prefix positions 0..2 and right padding
carry label -100. Token-mean cross-entropy over content tokens and EOS (EOS supervised exactly once). Identical for every
method.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

PROMPT_LEN = 4
IGNORE = -100


def lang_ids(b) -> dict:
    return dict(b.model.generation_config.lang_to_id)


@contextlib.contextmanager
def maybe_disabled(adapter):
    if adapter is None:
        yield
    else:
        with adapter.disabled():
            yield


def frozen_lid(b, input_features, adapter=None) -> dict:
    """Stage A: reference-free language decision of the frozen backbone (adapter disabled if installed)."""
    gen = b.model.generation_config
    L = lang_ids(b)
    inv = {v: k for k, v in L.items()}
    ids = sorted(L.values())
    with torch.inference_mode(), maybe_disabled(adapter):
        dec = torch.full((input_features.shape[0], 1), gen.decoder_start_token_id, device=b.device, dtype=torch.long)
        z = b.model(input_features=input_features[:, :, :3000], decoder_input_ids=dec, use_cache=False).logits[:, -1].float()
    zl = z[:, ids]
    top = zl.argmax(-1).tolist()
    lp = zl.log_softmax(-1)
    out = []
    for i in range(z.shape[0]):
        o = lp[i].argsort(descending=True)[:3].tolist()
        tok = ids[top[i]]
        out.append({"token": tok, "language": inv[tok], "top3": [(inv[ids[j]], float(lp[i, j])) for j in o],
                    "argmax_over_all_lang_equals_full_vocab_masked": True})
    return out[0] if len(out) == 1 else out


def prompt_ids(b, lang_token: int) -> list[int]:
    gen = b.model.generation_config
    tok = b.processor.tokenizer
    return [gen.decoder_start_token_id, lang_token, gen.task_to_id["transcribe"], tok.convert_tokens_to_ids("<|notimestamps|>")]


def build_tf_batch(prompts: list[list[int]], targets: list[list[int]], eos: int, pad: int):
    """X [B, T] decoder inputs, Y [B, T] labels. Row i: X = prompt_i + target_i[:-1], Y[3 + t] = target_i[t]."""
    assert len(prompts) == len(targets) and prompts
    for p, y in zip(prompts, targets):
        if len(p) != PROMPT_LEN:
            raise ValueError("prompt must have 4 tokens")
        if not y or y[-1] != eos or eos in y[:-1]:
            raise ValueError("target must end with exactly one EOS")
        if any((not isinstance(v, int)) or v < 0 for v in y):
            raise ValueError("invalid target token")
    T = max(PROMPT_LEN + len(y) - 1 for y in targets)
    X = torch.full((len(targets), T), pad, dtype=torch.long)
    Y = torch.full((len(targets), T), IGNORE, dtype=torch.long)
    for i, (p, y) in enumerate(zip(prompts, targets)):
        ids = p + y[:-1]
        X[i, :len(ids)] = torch.tensor(ids)
        Y[i, PROMPT_LEN - 1:PROMPT_LEN - 1 + len(y)] = torch.tensor(y)
    return X, Y


def tf_loss(logits: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Token-mean CE over labelled positions (content tokens + EOS)."""
    valid = Y != IGNORE
    lp = F.log_softmax(logits.float(), -1)
    nll = -lp[valid].gather(-1, Y[valid][:, None])[:, 0]
    return nll.mean(), int(valid.sum())


def decode_frozen_lid(b, path: str, lang: dict, beams: int, adapter=None):
    """Stage B: decoding with the frozen language forced (adapter active if given). Returns the baseline_auto record."""
    from reft.baseline_auto import generate
    code = lang["language"].strip("<|>")
    ctx = adapter.active(b.model) if adapter is not None else contextlib.nullcontext()
    with ctx:
        rec, inp = generate(b, path, code, beams)
    assert rec["prompt"][1] == lang["token"], "decoded prompt does not carry the frozen language token"
    return rec, inp
