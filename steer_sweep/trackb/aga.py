"""B3-reimpl: Attention-Guided Adaptation, reimplemented in our stack.

Specification sources, both read, neither run:
  * bobbiaditya/Attention-Guided-Adaptation-for-Code-Switching-Speech-Recognition
    (ESPnet 202301 / PyTorch 1.13.1 / Python 3.9.16, vendored OpenAI-Whisper)
  * Attention-Guided Adaptation for Code-Switching Speech Recognition,
    ICASSP 2024, arXiv:2312.08856

The reference is NOT run. Running it would need a second environment and a
second scorer, and any B3-vs-B1/B2 difference would then be confounded by
framework rather than method. Reimplementing in our stack removes that
confound; the price is the fidelity risk itemised in `FIDELITY_NOTE`.

What the mechanism actually is (read from the reference source, and different
from a naive reading of the paper's abstract):

  1. Adaptation is bottleneck ADAPTERS -- Linear(d, d/4), GELU, Linear(d/4, d),
     residual, then LayerNorm -- placed at every encoder and decoder block, and
     they are the only trainable parameters. This is where the paper's ~5.6%
     trainable fraction comes from: for whisper-small, 24 blocks x 2 adapters at
     d/4 is 5.9% of the backbone.
  2. Head selection does NOT place parameters. It selects which decoder
     self-attention heads the auxiliary attention-guidance loss is applied to.
  3. The objective is `loss = cs_weight * loss_cs + loss_att`, where `loss_cs`
     is an MSE between the selected heads' attention on the two language prompt
     slots and a target pattern that sends Mandarin tokens to the ZH slot and
     English tokens to the EN slot with mass `c_val_attention`.
  4. Training is two-stage: stage 1 encoder adapters with `cs_weight = 0`,
     stage 2 initialised from stage 1 with decoder adapters added and the
     guidance loss switched on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .. import config as C
from . import data as TD
from .modules import AdapterStack

#: The reference selects `int(110 * head_percentage / 100)` heads out of its
#: 12 x 12 = 144. 110/144 is the fraction that generalises to another geometry.
REFERENCE_SELECTED_HEADS = 110
REFERENCE_TOTAL_HEADS = 144
HEAD_FRACTION = REFERENCE_SELECTED_HEADS / REFERENCE_TOTAL_HEADS

#: The CS prompt: the reference's `whisper_cs` layout carries BOTH language
#: tokens so the guidance has a ZH slot and an EN slot to point at.
CS_PROMPT_TOKENS = ("<|startoftranscript|>", "<|zh|>", "<|en|>",
                    "<|transcribe|>", "<|notimestamps|>")
ZH_SLOT, EN_SLOT = 1, 2

FIDELITY_NOTE: list[dict[str, str]] = [
    {"item": "framework",
     "status": "deliberate deviation",
     "detail": "reimplemented in our PyTorch/HuggingFace stack on "
               "whisper-large-v3. The reference is ESPnet 202301 on "
               "whisper-small with a vendored OpenAI-Whisper. Running the "
               "reference would confound B3-vs-B1/B2 with framework."},
    {"item": "adapter placement",
     "status": "inferred",
     "detail": "the reference places two adapters INSIDE each residual block "
               "(after self-attention and after the MLP). A forward hook sees "
               "only the block output, so both adapters are applied in "
               "sequence there. Parameter count is preserved exactly; the "
               "position of the first adapter differs by one sublayer."},
    {"item": "head-selection criterion",
     "status": "mirrored from code_util/head_selection.md and the source",
     "detail": "per decoder self-attention head and utterance, the head is "
               "counted when attention mass on prompt key positions 1:3 "
               "exceeds mass on position 0 plus positions 3:. Ranked by count "
               "descending over the training split."},
    {"item": "head-selection data",
     "status": "deliberate deviation",
     "detail": "computed on CS-Dialogue loc-train, not on SEAME, as required."},
    {"item": "released head mask is disabled upstream",
     "status": "defect in the reference, not reproduced",
     "detail": "in the released code `self.selected_heads` is commented out and "
               "replaced by a hard-coded 50% `random_onezero` 12x12 mask, so "
               "the shipped model does not use its own head selection. The "
               "documented procedure is implemented here instead."},
    {"item": "layer/head key order in the reference",
     "status": "defect in the reference, corrected here",
     "detail": "`for head, layer_dict in attention_count.items()` iterates a "
               "dict built as `attention_count[layer][head]`, then unpacks the "
               "tuple as `layer, head, num`. The transposition is invisible "
               "only because the matrix is square for whisper-small."},
    {"item": "number of selected heads",
     "status": "inferred",
     "detail": f"the reference hard-codes int(110 * pct / 100) against 144 "
               f"heads. The fraction {HEAD_FRACTION:.4f} is carried over to "
               f"this geometry rather than the literal 110."},
    {"item": "CS prompt",
     "status": "deliberate deviation, TRAINING ONLY",
     "detail": "the guidance loss needs a ZH slot and an EN slot, so B3-reimpl "
               "TRAINS under the reference's `whisper_cs` five-token prompt "
               f"{list(CS_PROMPT_TOKENS)}. Evaluation on D-dev-select uses the "
               "SAME four-token forced-<|zh|> prefix as every other arm, "
               "because 'same decode config' is one of the study's explicit "
               "invariants and mixing in a second prompt at eval time would "
               "confound the MER comparison by prompt as well as by method. "
               "The training/eval prompt mismatch is therefore B3-reimpl's own "
               "trade-off, stated here rather than silently introduced as an "
               "extra variable in the comparison."},
    {"item": "optimiser hyperparameters",
     "status": "deliberate deviation",
     "detail": "the reference uses AdamW lr 1e-3, wd 0.01, betas (0.9, 0.99), "
               "warmup 500, accum 4, 10 then 15 epochs. Track B applies the "
               "SHARED budget to every arm instead; tuning B3 on its own "
               "schedule would break matched training conditions."},
    {"item": "specaug",
     "status": "not reproduced",
     "detail": "the reference enables SpecAugment in the encoder. It is off "
               "here because no other arm uses it and it is not part of the "
               "attention-guidance mechanism."},
    {"item": "accent population",
     "status": "stated mismatch",
     "detail": "the reference repository is titled for Taiwan-accented "
               "Mandarin-English; CS-Dialogue is a different accent population."},
    {"item": "published MER",
     "status": "stated mismatch",
     "detail": "the paper's ~14.2% MER is on SEAME. We evaluate on "
               "CS-Dialogue. It is external context, not a reproduction target."},
    {"item": "license",
     "status": "recorded",
     "detail": "the reference repository has NO LICENSE file (checked "
               "2026-08-18). We reimplement rather than vendor, so the "
               "obligation is citation of arXiv:2312.08856."},
    {"item": "LID head parity",
     "status": "deliberate addition, flagged",
     "detail": "the shared LID head is added to B3-reimpl for parity with "
               "every other arm even though the attention-guidance loss "
               "already contains language-aware structure. The two terms may "
               "double-count the same signal; this is flagged rather than "
               "silently omitted or silently stacked."},
]


# ---------------------------------------------------------------------------
# head selection
# ---------------------------------------------------------------------------

@dataclass
class HeadSelection:
    counts: np.ndarray                      # (layers, heads)
    selected: np.ndarray                    # (layers, heads) 0/1
    n_selected: int
    n_total: int
    fraction: float
    utterances: int
    detail: dict[str, Any] = field(default_factory=dict)

    def mask(self, device) -> torch.Tensor:
        return torch.from_numpy(self.selected.astype(np.float32)).to(device)

    def summary(self) -> dict[str, Any]:
        by_layer = self.selected.sum(axis=1).astype(int).tolist()
        return {"n_selected": self.n_selected, "n_total": self.n_total,
                "fraction": self.fraction, "utterances_scanned": self.utterances,
                "selected_per_layer": by_layer,
                "top_layers": np.argsort(-self.counts.sum(axis=1))[:5].tolist(),
                **self.detail}


@torch.inference_mode()
def select_heads(bundle, examples: Sequence[TD.Example], *, batch_size: int = 4,
                 head_fraction: float = HEAD_FRACTION, limit: int | None = None,
                 log=None) -> HeadSelection:
    """Forward pass over the training split accumulating per-head statistics.

    No parameter is updated. For each decoder self-attention head the head is
    counted on an utterance when its attention mass on the language prompt
    slots exceeds its mass everywhere else.
    """
    from csasr.models.whisper import batch_model_inputs

    scan = list(examples[: int(limit)] if limit else examples)
    n_layers = bundle.num_decoder_layers
    n_heads = int(bundle.config.decoder_attention_heads)
    counts = np.zeros((n_layers, n_heads), dtype=np.int64)
    pad = bundle.processor.tokenizer.pad_token_id

    for start in range(0, len(scan), batch_size):
        chunk = scan[start:start + batch_size]
        inputs = batch_model_inputs(bundle, [e.audio_path for e in chunk])
        width = max(len(e.token_ids) for e in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        for i, e in enumerate(chunk):
            ids[i, : len(e.token_ids)] = torch.tensor(e.token_ids, dtype=torch.long)
        out = bundle.model(**inputs, decoder_input_ids=ids.to(bundle.device),
                           output_attentions=True, use_cache=False)
        for layer, attn in enumerate(out.decoder_attentions):
            a = attn.float()                                  # (B, H, T, T)
            prompt = a[:, :, :, ZH_SLOT:EN_SLOT + 1].sum(dim=(-1, -2))
            other = (a[:, :, :, 0].sum(dim=-1)
                     + a[:, :, :, EN_SLOT + 1:].sum(dim=(-1, -2)))
            counts[layer] += (prompt > other).sum(dim=0).cpu().numpy()
        del out
        if log and (start // batch_size) % 20 == 0:
            log.info("  head selection: %d/%d utterances", start + len(chunk), len(scan))

    total = n_layers * n_heads
    k = max(1, int(round(total * float(head_fraction))))
    flat = counts.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    selected = np.zeros(total, dtype=np.int64)
    chosen = [i for i in order[:k] if flat[i] > 0]
    selected[chosen] = 1
    return HeadSelection(
        counts=counts, selected=selected.reshape(n_layers, n_heads),
        n_selected=int(selected.sum()), n_total=total,
        fraction=float(head_fraction), utterances=len(scan),
        detail={"criterion": C.AGA["head_selection"]["criterion"],
                "computed_on": C.AGA["head_selection"]["computed_on"],
                "reference_pickle_not_used": C.AGA["head_selection"]["reference_pickle"],
                "heads_with_zero_count_excluded": int(k - len(chosen))})


# ---------------------------------------------------------------------------
# the guidance loss
# ---------------------------------------------------------------------------

def target_pattern(batch: dict[str, Any], width: int, *,
                   c_val: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Target attention on the two language slots, and the valid-query mask.

    A Mandarin query token should place `c_val` on the ZH slot and nothing on
    the EN slot; an English token the reverse. Prompt rows 1 and 2 anchor the
    slots. Rows without a language label contribute nothing.
    """
    labels = batch["lid_token_labels"][:, :width]
    device = labels.device
    pattern = torch.zeros((labels.shape[0], width, 2), dtype=torch.float32, device=device)
    valid = labels != TD.LID_IGNORE
    pattern[..., 0] = torch.where(valid & (labels == TD.LID_ZH),
                                  torch.full_like(pattern[..., 0], c_val),
                                  torch.zeros_like(pattern[..., 0]))
    pattern[..., 1] = torch.where(valid & (labels == TD.LID_EN),
                                  torch.full_like(pattern[..., 1], c_val),
                                  torch.zeros_like(pattern[..., 1]))
    mask = valid.clone()
    if width > EN_SLOT:
        pattern[:, ZH_SLOT, 0] = c_val
        pattern[:, ZH_SLOT, 1] = 0.0
        pattern[:, EN_SLOT, 0] = 0.0
        pattern[:, EN_SLOT, 1] = c_val
        mask[:, ZH_SLOT] = True
        mask[:, EN_SLOT] = True
    return pattern, mask


def guidance_loss(decoder_attentions: Sequence[torch.Tensor], batch: dict[str, Any],
                  head_mask: torch.Tensor, *, c_val: float) -> torch.Tensor:
    """MSE between selected heads' language-slot attention and the pattern."""
    if not len(decoder_attentions):
        return torch.zeros((), device=head_mask.device)
    width = decoder_attentions[0].shape[-1]
    pattern, mask = target_pattern(batch, width, c_val=c_val)
    denom = mask.float().sum().clamp(min=1.0)
    total = torch.zeros((), device=decoder_attentions[0].device, dtype=torch.float32)
    for layer, attn in enumerate(decoder_attentions):
        weights = head_mask[layer]
        if float(weights.sum()) == 0.0:
            continue
        slots = attn.float()[:, :, :, ZH_SLOT:EN_SLOT + 1]      # (B, H, T, 2)
        target = pattern.unsqueeze(1)                            # (B, 1, T, 2)
        error = ((slots - target) ** 2).sum(dim=-1)              # (B, H, T)
        error = error * mask.unsqueeze(1).float()
        per_head = error.sum(dim=-1) / denom                     # (B, H)
        total = total + (per_head * weights.unsqueeze(0)).sum(dim=-1).mean()
    return total


# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------

def parameter_fraction_check(stack: AdapterStack, backbone: int) -> dict[str, Any]:
    """~5.6% of the backbone, within a factor of two. Off by more: STOP."""
    trainable = stack.n_parameters()
    fraction = trainable / max(1, backbone)
    target = C.AGA["paper_trainable_fraction"]
    tolerance = C.AGA["fidelity_factor_tolerance"]
    ok = (target / tolerance) <= fraction <= (target * tolerance)
    return {
        "trainable_parameters": int(trainable),
        "backbone_parameters": int(backbone),
        "trainable_fraction": float(fraction),
        "paper_fraction": target,
        "within_factor": tolerance,
        "passed": bool(ok),
        "detail": (f"{trainable:,} adapter parameters = {100 * fraction:.2f}% of "
                   f"{backbone:,}; the paper reports ~{100 * target:.1f}%"),
    }


def build_cs_prompt(bundle) -> list[int]:
    tok = bundle.processor.tokenizer
    ids = tok.convert_tokens_to_ids(list(CS_PROMPT_TOKENS))
    if any(i is None for i in ids):
        raise RuntimeError(f"tokenizer lacks {CS_PROMPT_TOKENS}")
    return [int(i) for i in ids]


def retarget_examples(bundle, examples: Sequence[TD.Example]) -> list[TD.Example]:
    """Re-prefix the training examples with the five-token CS prompt."""
    prompt = build_cs_prompt(bundle)
    out: list[TD.Example] = []
    for e in examples:
        body = list(e.token_ids[C.PREFIX_WIDTH:])
        labels = [TD.LID_IGNORE] * len(prompt) + list(e.lid_token_labels[C.PREFIX_WIDTH:])
        out.append(TD.Example(
            utterance_id=e.utterance_id, dialogue_id=e.dialogue_id,
            audio_path=e.audio_path, duration_sec=e.duration_sec,
            reference=e.reference, token_ids=prompt + body,
            lid_token_labels=labels, en_spans_sec=e.en_spans_sec))
    return out
