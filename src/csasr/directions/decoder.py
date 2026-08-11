"""Per-decoder-layer directions for the DEC-ALL-GLOBAL control (section 24).

This is a confound baseline, never the principal method. One distinct direction
is built for every decoder layer; an encoder direction is never reused here and
one layer's direction is never applied to another.

Activations are collected with the *same* forward hooks used for steering, so
the direction lives in exactly the space it will later be added to (the block's
residual output, before the decoder's final layer norm).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..data.language_tags import EN, ZH, tag_unit
from ..data.normalize import normalize_text
from ..models.generation import teacher_forced_forward
from ..models.hooks import ActivationRecorder, assert_no_hooks
from ..utils.logging import get_logger
from .accumulators import DirectionAccumulatorSet

log = get_logger(__name__)


def _special_ids(processor) -> set[int]:
    tok = processor.tokenizer
    ids = set(tok.all_special_ids)
    # timestamp and language/task tokens live in the added-token range
    for t, i in tok.get_added_vocab().items():
        if t.startswith("<|") and t.endswith("|>"):
            ids.add(i)
    return ids


def token_language_labels(processor, token_ids: Sequence[int]) -> list[str]:
    """Language label per token id: EN, ZH or '' (excluded)."""
    tok = processor.tokenizer
    special = _special_ids(processor)
    labels = []
    for tid in token_ids:
        if int(tid) in special:
            labels.append("")
            continue
        piece = tok.decode([int(tid)], skip_special_tokens=True)
        norm = normalize_text(piece)
        if not norm.strip():
            labels.append("")
            continue
        tag = tag_unit(norm.strip().split()[0] if " " in norm else norm.strip())
        labels.append(tag if tag in (EN, ZH) else "")
    return labels


@torch.inference_mode()
def verify_decoder_shift(bundle, manifest: pd.DataFrame, language: str = "zh",
                         num_utterances: int = 8) -> dict:
    """Empirically confirm that decoder position j predicts token j+1.

    Required by the guide (section 24.1): the shift convention must never be
    assumed. We compare next-token accuracy under both conventions.
    """
    from ..data.alignment import build_prefix

    tok = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language=language)
    rows = manifest.head(num_utterances)
    seqs = []
    for _, r in rows.iterrows():
        ids = tok.encode(normalize_text(r["transcript_raw"]), add_special_tokens=False)[:120]
        seqs.append(prefix + list(ids) + [tok.eos_token_id])

    inputs = [s[:-1] for s in seqs]
    out, padded, mask = teacher_forced_forward(bundle, rows["audio_path"].tolist(), inputs)
    logits = out.logits.float()
    pred = logits.argmax(-1)

    next_hits = next_total = same_hits = same_total = 0
    for i, s in enumerate(seqs):
        L = len(s) - 1
        for j in range(len(prefix), L - 1):
            if pred[i, j].item() == s[j + 1]:
                next_hits += 1
            next_total += 1
            if pred[i, j].item() == s[j]:
                same_hits += 1
            same_total += 1
    report = {
        "convention": "hidden state at position j produces the logit for token j+1",
        "next_token_accuracy": next_hits / max(next_total, 1),
        "same_position_accuracy": same_hits / max(same_total, 1),
        "num_positions": next_total,
        "verified": (next_hits / max(next_total, 1)) > (same_hits / max(same_total, 1)),
    }
    if not report["verified"]:
        raise RuntimeError(f"decoder shift convention not verified: {report}")
    return report


@torch.inference_mode()
def accumulate_decoder_directions(bundle, manifest: pd.DataFrame, *,
                                  language: str = "zh", batch_size: int = 4,
                                  min_en_tokens: int = 2, min_zh_tokens: int = 2,
                                  max_text_tokens: int = 180,
                                  layers: Iterable[int] | None = None,
                                  checkpoint_path=None,
                                  checkpoint_every: int = 200) -> DirectionAccumulatorSet:
    """Teacher-forced collection of EN/ZH token states at every decoder layer."""
    from ..data.alignment import build_prefix

    layers = list(layers if layers is not None else range(bundle.num_decoder_layers))
    accs = (DirectionAccumulatorSet.load_or_new(checkpoint_path, layers, bundle.d_model)
            if checkpoint_path else DirectionAccumulatorSet(layers, bundle.d_model))

    tok = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language=language)
    todo = manifest[~manifest["utterance_id"].isin(accs.processed)]
    todo = todo.sort_values("duration_sec").reset_index(drop=True)
    n_batches = math.ceil(len(todo) / batch_size) if len(todo) else 0
    since_ckpt = 0

    for bi in tqdm(range(n_batches), desc="e2 decoder", leave=False):
        batch = todo.iloc[bi * batch_size: (bi + 1) * batch_size]
        seqs, plans = [], []
        for _, row in batch.iterrows():
            ids = tok.encode(normalize_text(row["transcript_raw"]),
                             add_special_tokens=False)[:max_text_tokens]
            full = prefix + list(ids) + [tok.eos_token_id]
            labels = token_language_labels(bundle.processor, full)
            # state at position j predicts token j+1 -> label of position j is labels[j+1]
            pos_labels = ["" ] * (len(full) - 1)
            for j in range(len(full) - 1):
                pos_labels[j] = labels[j + 1]
            en_pos = [j for j in range(len(prefix), len(full) - 1) if pos_labels[j] == EN]
            zh_pos = [j for j in range(len(prefix), len(full) - 1) if pos_labels[j] == ZH]
            if len(en_pos) < min_en_tokens or len(zh_pos) < min_zh_tokens:
                accs.mark(row["utterance_id"])
                continue
            seqs.append(full[:-1])
            plans.append((row, en_pos, zh_pos))
        if not plans:
            continue

        with ActivationRecorder(bundle, layers, module="decoder") as rec:
            teacher_forced_forward(bundle, [p[0]["audio_path"] for p in plans], seqs)
            states = {k: rec.states[k] for k in layers}
        assert_no_hooks(bundle)

        for k in layers:
            h = states[k]
            acc = accs.acc[k]
            for i, (row, en_pos, zh_pos) in enumerate(plans):
                en_t = h[i, torch.as_tensor(en_pos, device=h.device)]
                zh_t = h[i, torch.as_tensor(zh_pos, device=h.device)]
                both = torch.cat([en_t, zh_t], dim=0)
                gram = (both.T @ both).double().cpu().numpy()
                acc.add_utterance(en_t.double().cpu().numpy(), zh_t.double().cpu().numpy())
                acc.add_frames(both.double().cpu().numpy(), gram=gram)
        del states
        for row, _, _ in plans:
            accs.mark(row["utterance_id"])

        since_ckpt += len(plans)
        if checkpoint_path and since_ckpt >= checkpoint_every:
            accs.save(checkpoint_path)
            since_ckpt = 0

    if checkpoint_path:
        accs.save(checkpoint_path)
    return accs


def build_decoder_directions(bundle, accs: DirectionAccumulatorSet, *, seed: int,
                             manifest_hash: str, normalization_version: str) -> dict[int, dict]:
    from .encoder import build_direction_record

    out = {}
    for layer, acc in accs.acc.items():
        if acc.num_utterances == 0:
            log.warning("decoder layer %d has no utterances; skipped", layer)
            continue
        out[layer] = build_direction_record(
            bundle, layer, acc.mean_delta(), acc,
            direction_type="decoder_within_utterance", seed=seed,
            manifest_hash=manifest_hash, normalization_version=normalization_version,
            module="decoder",
        )
    return out
