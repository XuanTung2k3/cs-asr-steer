"""Training data and the shared LID supervision.

CE supervision is the reference transcript. LID supervision is EN-vs-ZH per
decoder token (or per encoder frame at an encoder site), derived from the CS
span alignment; positions without alignment evidence are masked with -100 so
they contribute nothing rather than a guessed label.

Splits:
  train  loc-train + util-train   (training-side roles)
  dev    router-calib             (training-side role, dialogue-disjoint from
                                   train and from D-dev-select; used only for
                                   early stopping and for the ONE lambda and
                                   ONE lr selection)

`D-dev-select` is the reporting set and is never trained or early-stopped on.
`D-dev-confirm` and `D-test` are not touched by Track B at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .. import config as C

LID_IGNORE = -100
LID_ZH, LID_EN = 0, 1

TRAIN_ROLES = (C.LOC_TRAIN, C.UTIL_TRAIN)
DEV_ROLE = "router-calib"
DEV_ROLE_NOTE = (
    "router-calib is a training-side role, dialogue-disjoint from loc-train, "
    "util-train and D-dev-select. It is used here as the training dev set for "
    "early stopping and for the single lambda_lid and lr selections, so that "
    "D-dev-select stays a pure reporting set. D-dev-confirm and D-test are "
    "never opened by Track B.")


@dataclass
class Example:
    utterance_id: str
    dialogue_id: str
    audio_path: str
    duration_sec: float
    reference: str
    token_ids: list[int]          # prefix + text + eot
    lid_token_labels: list[int]   # aligned to token_ids, LID_IGNORE where unknown
    en_spans_sec: list[tuple[float, float]]


def _role_frames(cfg: dict, roles: Sequence[str]) -> pd.DataFrame:
    from ..data import load_split
    root = Path(cfg["experiment"]["output_root"])
    candidates = pd.read_parquet(root / "alignments" / "candidates_all.parquet")
    frames = []
    for role in roles:
        manifest = load_split(role, cfg)
        sampled = set(candidates.loc[candidates["role"].astype(str) == role,
                                     "utterance_id"].astype(str))
        manifest = manifest.copy()
        manifest["role"] = role
        manifest["has_alignment"] = manifest["utterance_id"].astype(str).isin(sampled)
        frames.append(manifest)
    return pd.concat(frames, ignore_index=True)


def _en_spans(cfg: dict, roles: Sequence[str]) -> dict[str, list[tuple[float, float]]]:
    """Embedded-English span times per utterance, from the frozen alignment."""
    from csasr.experiments import v2r3_directions as D3
    from csasr.lss.align import conventions as conv

    root = Path(cfg["experiment"]["output_root"])
    candidates = pd.read_parquet(root / "alignments" / "candidates_all.parquet")
    candidates = candidates[candidates["role"].astype(str).isin(set(roles))]
    adjusted = conv.apply_convention(candidates, D3.CTC_CONVENTION,
                                     family=D3.SPAN_FAMILY)
    frame = adjusted[adjusted["aligner_family"].astype(str) == D3.SPAN_FAMILY]
    frame = frame[frame["language"].astype(str).str.upper().str.startswith("EN")]
    out: dict[str, list[tuple[float, float]]] = {}
    for utterance, group in frame.groupby("utterance_id"):
        spans = []
        for start, end in zip(pd.to_numeric(group["start_sec"], errors="coerce"),
                              pd.to_numeric(group["end_sec"], errors="coerce")):
            if np.isfinite(start) and np.isfinite(end) and end > start:
                spans.append((float(start), float(end)))
        if spans:
            out[str(utterance)] = sorted(spans)
    return out


def build_examples(bundle, cfg: dict, roles: Sequence[str], *,
                   limit: int | None = None, log=None) -> list[Example]:
    """Tokenise, and attach LID labels wherever the alignment supports them."""
    from csasr.data.alignment import build_prefix, token_char_offsets
    from csasr.data.language_tags import EN, ZH, tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.experiments import v2r3_directions as D3

    manifest = _role_frames(cfg, roles)
    if limit:
        manifest = manifest.head(int(limit))
    spans = _en_spans(cfg, roles)
    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language="zh")
    eot = tokenizer.eos_token_id

    examples: list[Example] = []
    for _, row in manifest.iterrows():
        reference = str(row["transcript_raw"])
        norm, units = normalize_and_segment(reference)
        text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
        if not text_ids:
            continue
        offsets = token_char_offsets(tokenizer, text_ids)
        char_spans = D3.unit_char_spans(norm, units)
        labels = [LID_IGNORE] * len(prefix)
        for t, (lo, hi) in enumerate(offsets):
            label = LID_IGNORE
            for index, (cs, ce) in enumerate(char_spans):
                if lo < ce and hi > cs:
                    tag = tag_unit(units[index])
                    label = LID_EN if tag == EN else (LID_ZH if tag == ZH else LID_IGNORE)
                    break
            labels.append(label)
        labels.append(LID_IGNORE)                      # eot
        examples.append(Example(
            utterance_id=str(row["utterance_id"]),
            dialogue_id=str(row["dialogue_id"]),
            audio_path=str(row["audio_path"]),
            duration_sec=float(row["duration_sec"]),
            reference=reference,
            token_ids=list(prefix) + list(text_ids) + [eot],
            lid_token_labels=labels,
            en_spans_sec=spans.get(str(row["utterance_id"]), []),
        ))
    if log:
        supervised = sum(1 for e in examples if any(l != LID_IGNORE for l in e.lid_token_labels))
        log.info("built %d examples over roles %s (%d with LID supervision)",
                 len(examples), list(roles), supervised)
    return examples


def frame_lid_labels(bundle, example: Example) -> np.ndarray:
    """Per encoder frame: EN / ZH / ignore, from the span times."""
    valid = int(bundle.valid_frames(example.duration_sec))
    labels = np.full(C.EXPECTED_ENCODER_FRAMES, LID_IGNORE, dtype=np.int64)
    if example.en_spans_sec:
        labels[:valid] = LID_ZH
        step = float(bundle.encoder_step_sec)
        for start, end in example.en_spans_sec:
            lo = max(0, min(valid, int(np.floor(start / step))))
            hi = max(lo, min(valid, int(np.ceil(end / step))))
            labels[lo:hi] = LID_EN
    return labels


class Collator:
    """Batch of audio features, decoder inputs, CE labels and LID labels."""

    def __init__(self, bundle, site: str):
        self.bundle = bundle
        self.site = site
        self.pad_id = bundle.processor.tokenizer.pad_token_id

    def __call__(self, batch: Sequence[Example]) -> dict[str, Any]:
        from csasr.models.whisper import batch_model_inputs

        inputs = batch_model_inputs(self.bundle, [e.audio_path for e in batch])
        width = max(len(e.token_ids) for e in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        ce = torch.full((len(batch), width), LID_IGNORE, dtype=torch.long)
        lid = torch.full((len(batch), width), LID_IGNORE, dtype=torch.long)
        for i, e in enumerate(batch):
            n = len(e.token_ids)
            ids[i, :n] = torch.tensor(e.token_ids, dtype=torch.long)
            # CE targets are the next token; the forced prefix is not predicted
            ce[i, : n - 1] = torch.tensor(e.token_ids[1:], dtype=torch.long)
            ce[i, : C.PREFIX_WIDTH - 1] = LID_IGNORE
            lid[i, :n] = torch.tensor(e.lid_token_labels, dtype=torch.long)
        out = {
            "input_features": inputs["input_features"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": ids.to(self.bundle.device),
            "ce_labels": ce.to(self.bundle.device),
            "lid_token_labels": lid.to(self.bundle.device),
            "examples": list(batch),
        }
        if self.site == C.SITE_ENCODER:
            frames = np.stack([frame_lid_labels(self.bundle, e) for e in batch])
            out["lid_frame_labels"] = torch.from_numpy(frames).to(self.bundle.device)
        return out


def batches(examples: Sequence[Example], batch_size: int, *, seed: int,
            shuffle: bool = True):
    """Deterministic epoch iterator."""
    order = np.arange(len(examples))
    rng = np.random.default_rng(seed)
    while True:
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            picked = order[start:start + batch_size]
            if len(picked):
                yield [examples[int(i)] for i in picked]


def subsample(examples: Sequence[Example], fraction: float, *, seed: int) -> list[Example]:
    """Dialogue-stratified subsample for the data-efficiency curve."""
    if fraction >= 1.0:
        return list(examples)
    frame = pd.DataFrame({"i": range(len(examples)),
                          "dialogue_id": [e.dialogue_id for e in examples]})
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, group in frame.groupby("dialogue_id"):
        # `to_numpy()` can return a read-only view into pandas' block
        # manager; `.copy()` guarantees `rng.shuffle` has a writable array.
        idx = group["i"].to_numpy().copy()
        rng.shuffle(idx)
        keep.extend(idx[: max(1, int(round(len(idx) * fraction)))].tolist())
    return [examples[i] for i in sorted(keep)]
