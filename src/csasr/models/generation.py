"""Deterministic batched decoding, with optional steering hooks and caching."""
from __future__ import annotations

import json
import math
import hashlib
import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm

import numpy as np
from ..data.normalize import normalize_text
from ..utils.logging import get_logger
from .hooks import MultiHook, assert_no_hooks
from .whisper import WhisperBundle, batch_features

log = get_logger(__name__)

PREDICTION_COLUMNS = [
    "utterance_id", "system", "config_id", "hypothesis_raw",
    "hypothesis_normalized", "avg_logprob", "num_tokens",
    "token_ids", "token_logprobs",
]

# Hook builders receive the batch dataframe and return a list of hook objects.
HookBuilder = Callable[[pd.DataFrame], Sequence]


def _generate_kwargs(cfg: dict, language: str | None) -> dict:
    dec = cfg["decoding"]
    kwargs = dict(
        task=dec.get("task", "transcribe"),
        do_sample=bool(dec.get("do_sample", False)),
        num_beams=int(dec.get("num_beams", 1)),
        max_new_tokens=int(dec.get("max_new_tokens", 200)),
        return_dict_in_generate=True,
        output_scores=bool(dec.get("output_scores", True)),
        return_timestamps=bool(dec.get("return_timestamps", False)),
        condition_on_prev_tokens=False,
    )
    if language is not None:
        kwargs["language"] = language
    temp = dec.get("temperature", 0.0)
    if temp:
        kwargs["temperature"] = float(temp)
    return kwargs


@torch.inference_mode()
def decode_batch(bundle: WhisperBundle, audio_paths: Sequence[str], cfg: dict,
                 language: str | None = None,
                 hooks: Sequence | None = None) -> list[dict]:
    """Decode one batch; ``hooks`` are active only for this call."""
    features = batch_features(bundle, audio_paths)
    kwargs = _generate_kwargs(cfg, language)
    with MultiHook(hooks or []):
        out = bundle.model.generate(features, **kwargs)
    assert_no_hooks(bundle)

    # With return_timestamps=True Whisper returns a plain dict (sequences +
    # per-segment results) instead of a ModelOutput; accept both shapes.
    if isinstance(out, dict) and not hasattr(out, "sequences"):
        sequences = out["sequences"]
        scores = out.get("scores")
    else:
        sequences = out.sequences
        scores = getattr(out, "scores", None)
    texts = bundle.processor.batch_decode(sequences, skip_special_tokens=True)

    avg_logprobs: list[float] = []
    num_tokens: list[int] = []
    token_ids: list[list[int]] = []
    token_logprobs: list[list[float]] = []
    if scores:
        try:
            trans = bundle.model.compute_transition_scores(
                sequences, scores, normalize_logits=True
            ).float()
            eos = bundle.processor.tokenizer.eos_token_id
            gen = sequences[:, sequences.shape[1] - trans.shape[1]:]
            mask = (gen != eos) & (gen != bundle.model.config.pad_token_id)
            for i in range(trans.shape[0]):
                m = mask[i]
                ids = gen[i][m].detach().cpu().tolist()
                lps = trans[i][m].detach().cpu().tolist()
                n = len(ids)
                avg_logprobs.append(float(np.mean(lps)) if n else float("nan"))
                num_tokens.append(n)
                token_ids.append([int(x) for x in ids])
                token_logprobs.append([float(x) for x in lps])
        except Exception as exc:  # scores are diagnostic, never fatal
            log.warning("transition scores unavailable: %s", exc)
    if not avg_logprobs:
        pad = bundle.model.config.pad_token_id
        token_ids = [[int(x) for x in s[s != pad].detach().cpu().tolist()] for s in sequences]
        token_logprobs = [[] for _ in texts]
        num_tokens = [len(x) for x in token_ids]
        avg_logprobs = [float("nan")] * len(texts)

    return [
        {
            "hypothesis_raw": texts[i].strip(),
            "hypothesis_normalized": normalize_text(texts[i]),
            "avg_logprob": avg_logprobs[i],
            "num_tokens": num_tokens[i],
            "token_ids": token_ids[i],
            "token_logprobs": token_logprobs[i],
        }
        for i in range(len(texts))
    ]


def decode_manifest(bundle: WhisperBundle, df: pd.DataFrame, cfg: dict, *,
                    system: str, config_id: str, out_path: str | Path,
                    language: str | None = None,
                    hook_builder: HookBuilder | None = None,
                    batch_size: int | None = None,
                    resume: bool = True,
                    overwrite: bool = False) -> pd.DataFrame:
    """Decode every utterance in ``df``, caching results to a parquet file.

    Resumption is at utterance granularity and guarded by provenance metadata.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    target_hash = hashlib.sha256("\n".join(sorted(df["utterance_id"].astype(str))).encode()).hexdigest()
    expected_meta = {
        "system": system, "config_id": config_id, "language": language,
        "model_revision": bundle.revision, "target_ids_sha256": target_hash,
        "num_targets": int(len(df)), "decoding": cfg["decoding"],
    }
    if out_path.exists() and not overwrite:
        if not meta_path.exists():
            raise RuntimeError(f"prediction cache lacks provenance metadata: {out_path}; use --overwrite")
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_meta != expected_meta:
            raise RuntimeError(f"prediction cache configuration/target mismatch: {out_path}; use --overwrite")
    meta_path.write_text(json.dumps(expected_meta, indent=2, default=str), encoding="utf-8")
    if not len(df):
        log.warning("nothing to decode for %s/%s (empty target set)", system, config_id)
        empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
        empty.to_parquet(out_path, index=False)
        return empty
    done: dict[str, dict] = {}
    if out_path.exists() and not overwrite:
        if not resume:
            log.info("%s exists; reusing cached predictions (%s)", out_path.name, system)
            return pd.read_parquet(out_path)
        cached = pd.read_parquet(out_path)
        done = {r["utterance_id"]: dict(r) for _, r in cached.iterrows()}

    todo = df[~df["utterance_id"].isin(done)].reset_index(drop=True)
    bs = int(batch_size or cfg["decoding"].get("batch_size", 8))
    rows = list(done.values())

    if len(todo):
        # long utterances first would spike memory; sort by duration for stable batches
        todo = todo.sort_values("duration_sec").reset_index(drop=True)
        n_batches = math.ceil(len(todo) / bs)
        t_start = time.monotonic()
        for bi in tqdm(range(n_batches), desc=f"decode {system}/{config_id}", leave=False):
            batch = todo.iloc[bi * bs: (bi + 1) * bs]
            hooks = list(hook_builder(batch)) if hook_builder is not None else None
            try:
                results = decode_batch(bundle, batch["audio_path"].tolist(), cfg,
                                       language=language, hooks=hooks)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("OOM on batch %d; retrying one utterance at a time", bi)
                results = []
                for _, row in batch.iterrows():
                    single = batch.loc[[row.name]]
                    h = list(hook_builder(single)) if hook_builder is not None else None
                    results.extend(decode_batch(bundle, [row["audio_path"]], cfg,
                                                language=language, hooks=h))
            for (_, row), res in zip(batch.iterrows(), results):
                rows.append({
                    "utterance_id": row["utterance_id"],
                    "system": system,
                    "config_id": config_id,
                    **res,
                })
            if (bi + 1) % 25 == 0 or bi == n_batches - 1:
                pd.DataFrame(rows)[PREDICTION_COLUMNS].to_parquet(out_path, index=False)
        elapsed = time.monotonic() - t_start
        log.info("decoded %d utterances for %s/%s in %.1f s (%.2f utt/s, batch_size=%d)",
                 len(todo), system, config_id, elapsed,
                 len(todo) / elapsed if elapsed > 0 else float("nan"), bs)

    out = pd.DataFrame(rows)[PREDICTION_COLUMNS]
    out.to_parquet(out_path, index=False)
    return out


@torch.inference_mode()
def teacher_forced_forward(bundle: WhisperBundle, audio_paths: Sequence[str],
                           token_ids: Sequence[Sequence[int]],
                           output_attentions: bool = False,
                           output_hidden_states: bool = False):
    """Run the model with reference tokens (used by E1 alignment and E2 decoder).

    Returns the model output plus the padded decoder input ids and pad mask.
    """
    pad_id = bundle.processor.tokenizer.pad_token_id
    max_len = max(len(t) for t in token_ids)
    padded = torch.full((len(token_ids), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(token_ids), max_len), dtype=torch.bool)
    for i, t in enumerate(token_ids):
        padded[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        mask[i, : len(t)] = True
    padded = padded.to(bundle.device)
    mask = mask.to(bundle.device)

    features = batch_features(bundle, audio_paths)
    out = bundle.model(
        input_features=features,
        decoder_input_ids=padded,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=False,
    )
    return out, padded, mask


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
