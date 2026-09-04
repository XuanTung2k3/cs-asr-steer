#!/usr/bin/env python
"""Job A: Frozen steering confirmation + SEAME transfer.

No training.  Inference only on one H100, ≤ 3 hours wall-time.

Six systems on the full CS-Dialogue dev-confirm split:
  F0  Baseline (no intervention)
  F1  Oracle-local (steer only at gold CS decoder steps)
  F2  Global, same per-position α
  F3  Global, matched total energy
  F4  Random-location local (same step count as F1)
  F5  Frozen automatic soft-local (gate calibrated on dev-select ONLY)

Optional SEAME transfer (S0/S1/S2) when data is available.

Usage:
    python experiments/job_a_frozen.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — the script is run from the repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Configuration — single block, paths and hyperparameters
# ---------------------------------------------------------------------------
DEVICE = "cuda"
CONFIG_PATH = "lss/l1b_candidates_dialogue_v2r3.yaml"
RESULTS_DIR = Path("results/job_a")
DECODER_LAYER = 24
ALPHA = 2.0
WEIGHT_NAT = 2.0
WEIGHT_PROMPT = 0.5
BEAM_SIZE = 5
BATCH_SIZE = 8
SEED = 42
PREFIX_WIDTH = 4
EPS = 1e-6
MAX_NEW_TOKENS = 200

# SEAME candidate paths (checked in order; first existing one wins)
SEAME_CANDIDATES = [
    Path("/mnt/data/tungnx/seame"),
    Path("/mnt/data/seame"),
    Path(os.environ.get("SEAME_ROOT", "/dev/null")),
]
SEAME_MAX_UTTS = 500          # speaker-balanced subset cap per split
SEAME_TIME_BUDGET_SEC = 5400  # 90 min limit for SEAME

log = logging.getLogger("job_a")


# ═══════════════════════════════════════════════════════════════════════════
# Atomic file writing
# ═══════════════════════════════════════════════════════════════════════════

def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically: tmp in same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
        os.replace(tmp, str(path))
    except BaseException:
        os.unlink(tmp)
        raise


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, str(path))
    except BaseException:
        os.unlink(tmp)
        raise


def atomic_write_csv(path: Path, df: "pd.DataFrame") -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, str(path))
    except BaseException:
        os.unlink(tmp)
        raise


def ensure_confirm_poi(bundle: Any, cfg: dict[str, Any]) -> None:
    """Materialize the evaluation-only POI table if the v2r3 input lacks it.

    The published v2r3 baseline stage deliberately stops before D-dev-confirm,
    while this job evaluates that split.  Build only the lexical evaluation
    table from this job's frozen F0 decode; it is never used for calibration,
    training, or checkpoint decisions.  The parquet promotion is atomic.
    """
    import pandas as pd
    from csasr.evaluation.pier import poi_table
    from steer_sweep import data as D

    root = Path(cfg["experiment"]["output_root"])
    poi_path = root.parent.parent / "baselines" / "generation_001" / \
        "poi_D-dev-confirm.parquet"
    if poi_path.exists():
        return

    candidate_path = root / "alignments" / "candidates_all.parquet"
    role_root = Path(cfg["v2_namespace"]["role_root"])
    candidates = pd.read_parquet(candidate_path)
    candidates = candidates[candidates["role"].astype(str) == "D-dev-confirm"]
    sampled = set(candidates["utterance_id"].astype(str))
    manifest = pd.read_parquet(role_root / "role_D-dev-confirm.parquet")
    manifest = manifest[manifest["utterance_id"].astype(str).isin(sampled)]
    manifest = manifest.sort_values("duration_sec").reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("D-dev-confirm has no candidate-backed utterances")

    log.info("D-dev-confirm POI table is absent; building evaluation-only table "
             "from frozen F0 (%d utterances)", len(manifest))
    # F0 is the frozen Whisper baseline used for all Job A comparisons.
    raw_pop = type("PopulationView", (), {"manifest": manifest})()
    baseline = decode_with_hook(bundle, raw_pop, None,
                                num_beams=BEAM_SIZE,
                                batch_size=BATCH_SIZE)
    references = [str(t) for t in manifest["transcript_raw"]]
    ids = [str(u) for u in manifest["utterance_id"]]
    table = pd.DataFrame(poi_table(ids, references, [baseline[u] for u in ids]))
    table = table.rename(columns={"poi_index": "reference_unit_index"})

    poi_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(poi_path.parent), suffix=".tmp")
    try:
        os.close(fd)
        table.to_parquet(tmp, index=False)
        os.replace(tmp, str(poi_path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    log.info("Wrote evaluation-only D-dev-confirm POI table: %s (%d rows)",
             poi_path, len(table))


# ═══════════════════════════════════════════════════════════════════════════
# NormPreserve decoder hook
# ═══════════════════════════════════════════════════════════════════════════

def _split_output(output: Any) -> tuple[torch.Tensor, Any]:
    """Separate hidden states from the rest of a layer output tuple."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rebuild_output(hidden: torch.Tensor, rest: Any) -> Any:
    if rest is not None:
        return (hidden,) + rest
    return hidden


def _cache_length(past_key_values: Any) -> int:
    """Absolute decoder KV cache length (number of positions already cached)."""
    if past_key_values is None:
        return 0
    getter = getattr(past_key_values, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return 0
    try:
        first = past_key_values[0]
        if first is None:
            return 0
        return int(first[0].shape[2])
    except (IndexError, TypeError, AttributeError):
        return 0


class NormPreserveDecoderHook:
    """Apply α · m_t · v̂ at decoder layer output with NormPreserve rescaling.

    The hook reads absolute decoder position from a `PositionTracker` and
    decides per-position whether and how much to steer.  Three modes:

    * ``mask_fn(absolute_pos, item_idx) -> bool``   — hard binary gate (F1–F4)
    * ``alpha_fn(absolute_pos, item_idx) -> float``  — per-position α  (F3)
    * ``gate_fn(hidden_1d) -> float in [0,1]``      — soft gate         (F5)

    NormPreserve: h̃ = h̄ · (‖h‖ / ‖h̄‖)  where  h̄ = h + α · m · v̂.
    """

    def __init__(
        self,
        bundle: Any,
        layer: int,
        direction: torch.Tensor,
        alpha: float,
        *,
        num_beams: int = 1,
        mask_fn: Any | None = None,
        alpha_fn: Any | None = None,
        gate_fn: Any | None = None,
    ):
        self.bundle = bundle
        self.layer = int(layer)
        self.direction = direction.detach().clone()  # (D,) unit-norm
        self.alpha = float(alpha)
        self.num_beams = max(1, int(num_beams))
        self.mask_fn = mask_fn
        self.alpha_fn = alpha_fn
        self.gate_fn = gate_fn
        self._handle: Any = None
        self._tracker_handle: Any = None
        self._abs_pos: int = 0
        self.positions_steered: int = 0
        self.gate_values: list[float] = []

    # -- position tracker (forward pre-hook on the decoder module) ----------

    def _decoder_pre_hook(self, _mod, args, kwargs):
        self._abs_pos = _cache_length(kwargs.get("past_key_values"))
        return None

    # -- steering hook on decoder layer ------------------------------------

    def _hook(self, _mod, _inp, output):
        hidden, rest = _split_output(output)
        B, T, D = hidden.shape

        # During prefill (T > 1, abs_pos == 0) never steer
        if T > 1 and self._abs_pos == 0:
            return output

        v = self.direction.to(hidden.device, hidden.dtype).view(1, 1, D)
        items = B // self.num_beams

        steered = hidden.clone()
        any_changed = False

        for t_offset in range(T):
            abs_t = self._abs_pos + t_offset
            # Never steer prefix positions
            if abs_t < PREFIX_WIDTH:
                continue

            for item in range(items):
                # Determine gating
                if self.gate_fn is not None:
                    # Soft gate: compute m_t from the pre-intervention hidden
                    # For F5: gate_fn receives hidden[item*beams, t_offset] and
                    # returns m_t in [0, 1]
                    h_vec = hidden[item * self.num_beams, t_offset]
                    m_t = float(self.gate_fn(h_vec))
                    self.gate_values.append(m_t)
                    if m_t < 1e-6:
                        continue
                    eff_alpha = self.alpha * m_t
                elif self.alpha_fn is not None:
                    eff_alpha = self.alpha_fn(abs_t, item)
                    if abs(eff_alpha) < 1e-8:
                        continue
                elif self.mask_fn is not None:
                    if not self.mask_fn(abs_t, item):
                        continue
                    eff_alpha = self.alpha
                else:
                    eff_alpha = self.alpha

                # Apply to all beams of this item
                for beam in range(self.num_beams):
                    row = item * self.num_beams + beam
                    h = hidden[row, t_offset:t_offset + 1, :]  # (1, D)
                    h_bar = h + eff_alpha * v.squeeze(0)        # (1, D)
                    orig_norm = h.norm(dim=-1, keepdim=True)
                    new_norm = h_bar.norm(dim=-1, keepdim=True)
                    h_tilde = h_bar * (orig_norm / (new_norm + EPS))
                    steered[row, t_offset] = h_tilde.squeeze(0)
                    self.positions_steered += 1
                    any_changed = True

        if not any_changed:
            return output
        return _rebuild_output(steered, rest)

    def __enter__(self) -> "NormPreserveDecoderHook":
        # Position tracker on the decoder module
        self._tracker_handle = (
            self.bundle.model.model.decoder.register_forward_pre_hook(
                self._decoder_pre_hook, with_kwargs=True
            )
        )
        # Steering hook on the target layer
        self._handle = self.bundle.decoder_layer(self.layer).register_forward_hook(
            self._hook
        )
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._tracker_handle is not None:
            self._tracker_handle.remove()
            self._tracker_handle = None
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Decoding helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def decode_with_hook(
    bundle: Any,
    pop: Any,
    hook: NormPreserveDecoderHook | None,
    *,
    language: str = "zh",
    num_beams: int = 1,
    batch_size: int = BATCH_SIZE,
) -> dict[str, str]:
    """Decode every utterance of `pop`, with an optional NormPreserve hook."""
    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe",
        "language": language,
        "do_sample": False,
        "num_beams": int(num_beams),
        "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    n_batches = math.ceil(len(manifest) / batch_size) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * batch_size : (bi + 1) * batch_size]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        if hook is None:
            out = bundle.model.generate(**inputs, **kwargs)
        else:
            with hook:
                out = bundle.model.generate(**inputs, **kwargs)
                # Reset hook for next batch but accumulate stats
            # Note: we re-enter the hook per batch so the position
            # tracker resets correctly for each generate() call.

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

        if (bi + 1) % 10 == 0:
            log.info("  decoded %d/%d utterances", min((bi + 1) * batch_size,
                                                       len(manifest)), len(manifest))
    return texts


@torch.inference_mode()
def decode_with_hook_per_batch(
    bundle: Any,
    pop: Any,
    hook_factory: Any,
    *,
    language: str = "zh",
    num_beams: int = 1,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, str], list[float]]:
    """Decode with a per-batch hook factory (for F5 where gate state accumulates).

    ``hook_factory(batch_ids)`` returns a NormPreserveDecoderHook.
    Returns (texts, gate_values).
    """
    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe",
        "language": language,
        "do_sample": False,
        "num_beams": int(num_beams),
        "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    all_gate_values: list[float] = []
    n_batches = math.ceil(len(manifest) / batch_size) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * batch_size : (bi + 1) * batch_size]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        hook = hook_factory(ids)

        with hook:
            out = bundle.model.generate(**inputs, **kwargs)
        all_gate_values.extend(hook.gate_values)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

        if (bi + 1) % 10 == 0:
            log.info("  decoded %d/%d utterances", min((bi + 1) * batch_size,
                                                       len(manifest)), len(manifest))
    return texts, all_gate_values


# ═══════════════════════════════════════════════════════════════════════════
# Metric collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_metrics(
    pop: Any,
    hypotheses: dict[str, str],
    baseline_hyps: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Collect MER, PIER, EN WER, ZH CER, corrections/corruptions."""
    import pandas as pd
    from steer_sweep import data as D
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier
    from csasr.evaluation.correction_harm import (
        correction_harm_table, outside_region_edits, summarize_correction_harm,
    )

    references = D.reference_of(pop)
    ids = [u for u in pop.utterance_ids if u in hypotheses]
    refs = [references[u] for u in ids]
    hyps = [hypotheses[u] for u in ids]

    mer_out = corpus_mer(refs, hyps)
    pier_out = pier(refs, hyps)

    out: dict[str, Any] = {
        "scored_utterances": len(ids),
        "MER": float(mer_out["mer"]),
        "PIER": float(pier_out["pier"]),
        "en_wer": float(mer_out["en_wer"]),
        "zh_cer": float(mer_out["zh_cer"]),
        "num_en_ref": int(mer_out["num_en_ref"]),
        "num_zh_ref": int(mer_out["num_zh_ref"]),
    }

    if baseline_hyps is not None:
        # Corrections / corruptions
        from csasr.experiments.v2r3_day5_expansion import outcome_rates
        from csasr.evaluation.pier import unit_status

        corrections = 0
        corruptions = 0
        outside_edits = 0
        corr_harm_rows: list[dict[str, Any]] = []

        for _, target in pop.targets.iterrows():
            uid = str(target["utterance_id"])
            if uid not in baseline_hyps or uid not in hypotheses:
                continue
            ref = references[uid]
            rates = outcome_rates(ref, baseline_hyps[uid], hypotheses[uid],
                                  int(target["reference_unit_index"]))
            corrections += int(rates["corrected"])
            corruptions += int(rates["corrupted"])
            edits = outside_region_edits(ref, baseline_hyps[uid], hypotheses[uid],
                                         int(target["reference_unit_index"]))
            outside_edits += int(edits["outside_changed"])

            # For correction_harm_table
            before = unit_status(ref, baseline_hyps[uid])
            idx = int(target["reference_unit_index"])
            corr_harm_rows.append({
                "utterance_id": uid,
                "poi_index": idx,
                "reference": ref,
                "baseline_hypothesis": baseline_hyps[uid],
                "steered_hypothesis": hypotheses[uid],
                "baseline_correct": bool(before.get(idx, (False, ""))[0]),
                "baseline_category": str(target.get("category", "")),
            })

        harm = summarize_correction_harm(correction_harm_table(corr_harm_rows))
        out["corrections"] = corrections
        out["corruptions"] = corruptions
        out["outside_region_edits"] = outside_edits
        out["correction_harm"] = harm

    return out


def per_dialogue_breakdown(
    pop: Any,
    hypotheses: dict[str, str],
) -> "pd.DataFrame":
    """Per-dialogue / per-speaker MER breakdown."""
    import pandas as pd
    from csasr.evaluation.mer import corpus_mer
    from steer_sweep import data as D

    references = D.reference_of(pop)
    rows: list[dict[str, Any]] = []
    for did, group in pop.manifest.groupby("dialogue_id"):
        uids = [str(u) for u in group["utterance_id"] if str(u) in hypotheses]
        if not uids:
            continue
        refs = [references[u] for u in uids]
        hyps = [hypotheses[u] for u in uids]
        m = corpus_mer(refs, hyps)
        rows.append({
            "dialogue_id": str(did),
            "n_utterances": len(uids),
            "MER": float(m["mer"]),
            "en_wer": float(m["en_wer"]),
            "zh_cer": float(m["zh_cer"]),
        })
    return pd.DataFrame(rows)


def hypotheses_to_jsonl(
    pop: Any,
    hypotheses: dict[str, str],
) -> list[dict[str, Any]]:
    """One row per utterance: id, reference, hypothesis."""
    from steer_sweep import data as D

    references = D.reference_of(pop)
    return [
        {"utterance_id": uid, "reference": references.get(uid, ""),
         "hypothesis": hypotheses.get(uid, "")}
        for uid in pop.utterance_ids
        if uid in hypotheses
    ]


def save_system(
    system_id: str,
    pop: Any,
    hypotheses: dict[str, str],
    baseline_hyps: dict[str, str] | None,
    wall_sec: float,
    out_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save all outputs for one system and return metrics."""
    metrics = collect_metrics(pop, hypotheses, baseline_hyps)
    metrics["inference_time_sec"] = wall_sec
    if extra:
        metrics.update(extra)

    atomic_write_json(out_dir / f"{system_id}_metrics.json", metrics)
    atomic_write_csv(out_dir / f"{system_id}_per_dialogue.csv",
                     per_dialogue_breakdown(pop, hypotheses))
    atomic_write_jsonl(out_dir / f"{system_id}_hypotheses.jsonl",
                       hypotheses_to_jsonl(pop, hypotheses))

    log.info("  %s: MER=%.4f PIER=%.4f en_wer=%.4f zh_cer=%.4f corr=%s corrupt=%s  (%.1fs)",
             system_id, metrics["MER"], metrics["PIER"],
             metrics["en_wer"], metrics["zh_cer"],
             metrics.get("corrections", "-"), metrics.get("corruptions", "-"),
             wall_sec)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# F5 gate calibration (on dev-select ONLY)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GateParams:
    """Frozen gate scalars calibrated on dev-select."""
    c1: float
    c2: float
    tau: float
    T: float
    v_nat: torch.Tensor       # (D,) unit-norm
    v_prompt: torch.Tensor    # (D,) unit-norm
    v_combined: torch.Tensor  # (D,) unit-norm


@torch.inference_mode()
def calibrate_f5_gate(
    bundle: Any,
    cfg: dict[str, Any],
    v_nat: torch.Tensor,
    v_prompt: torch.Tensor,
    v_combined: torch.Tensor,
    *,
    dry_run: bool = False,
) -> GateParams:
    """Compute gate parameters (tau, T) on dev-select using teacher-forced pass.

    Strategy: c1=1, c2=0 (gate on v_nat projection only).
    tau = median(s_t) across all decoder steps with known EN/ZH labels.
    T = std(s_t).
    """
    from steer_sweep import data as D
    from steer_sweep import config as C
    from csasr.models.hooks import ActivationRecorder
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.data.language_tags import EN, ZH, tag_unit

    log.info("Calibrating F5 gate on dev-select ...")
    pop_select = D.build_population(bundle, cfg, C.DEV_SELECT, assert_anchors=not dry_run)
    if dry_run:
        pop_select = pop_select.subsample(2)

    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language="zh")
    eot = tokenizer.eos_token_id
    # ActivationRecorder stores h_24 in float32; keep the calibration dot
    # product in that dtype instead of mixing it with the BF16 model dtype.
    v_nat_d = v_nat.to(bundle.device, torch.float32)

    all_s: list[float] = []
    manifest = pop_select.manifest.reset_index(drop=True)
    batch_size = 4  # smaller for teacher forcing

    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start:start + batch_size]
        audio_paths = batch["audio_path"].tolist()
        token_seqs: list[list[int]] = []
        lang_labels: list[list[str]] = []  # per-token language tag

        for _, row in batch.iterrows():
            norm, units = normalize_and_segment(str(row["transcript_raw"]))
            text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
            seq = list(prefix) + list(text_ids) + [eot]
            token_seqs.append(seq)

            # Build per-token language labels
            from csasr.data.alignment import token_char_offsets
            from csasr.experiments.v2r3_directions import unit_char_spans
            offsets = token_char_offsets(tokenizer, text_ids)
            char_spans = unit_char_spans(norm, units)
            tags: list[str] = ["_"] * len(prefix)  # prefix: skip
            for t_off, (lo, hi) in enumerate(offsets):
                tag = "_"
                for idx, (cs, ce) in enumerate(char_spans):
                    if lo < ce and hi > cs:
                        tag = tag_unit(units[idx])
                        break
                tags.append(tag)
            tags.append("_")  # eot
            lang_labels.append(tags)

        # Teacher-forced forward with hidden state recording at layer 24
        with ActivationRecorder(bundle, [DECODER_LAYER], module="decoder",
                                to_dtype=torch.float32) as rec:
            from csasr.models.generation import teacher_forced_forward
            teacher_forced_forward(bundle, audio_paths, token_seqs)
            h_24 = rec.states[DECODER_LAYER]  # (B, T, D)

        for b in range(h_24.shape[0]):
            tags = lang_labels[b]
            for t in range(min(h_24.shape[1], len(tags))):
                if tags[t] in (EN, ZH):
                    h_t = h_24[b, t].to(bundle.device)
                    s_t = float(torch.dot(h_t, v_nat_d))
                    all_s.append(s_t)

    if not all_s:
        raise RuntimeError("No EN/ZH steps found during F5 gate calibration")

    s_arr = np.array(all_s)
    tau = float(np.median(s_arr))
    T_val = float(np.std(s_arr))
    if T_val < 1e-8:
        T_val = 1.0  # prevent division by zero

    log.info("  F5 gate: %d EN/ZH steps, tau=%.4f, T=%.4f, "
             "s_min=%.4f, s_max=%.4f",
             len(all_s), tau, T_val, float(s_arr.min()), float(s_arr.max()))

    return GateParams(c1=1.0, c2=0.0, tau=tau, T=T_val,
                      v_nat=v_nat, v_prompt=v_prompt, v_combined=v_combined)


# ═══════════════════════════════════════════════════════════════════════════
# System runners
# ═══════════════════════════════════════════════════════════════════════════

def oracle_steps_for(pop: Any) -> dict[str, list[int]]:
    """Extract oracle (gold CS) decoder steps from the target table.

    Returns {utterance_id: [decoder_step_indices ...]} where the step index
    is the generation step (i.e., token_index, NOT absolute position).
    """
    steps: dict[str, list[int]] = {}
    for _, t in pop.targets.iterrows():
        uid = str(t["utterance_id"])
        # token_index is the index into the text token sequence (0-based)
        step = int(t["token_index"])
        steps.setdefault(uid, []).append(step)
    # Deduplicate and sort
    return {uid: sorted(set(s)) for uid, s in steps.items()}


def run_f0(bundle: Any, pop: Any, out_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """F0: Baseline — no intervention."""
    log.info("F0: Baseline (no steering)")
    t0 = time.monotonic()
    texts = decode_with_hook(bundle, pop, None,
                             num_beams=BEAM_SIZE, batch_size=BATCH_SIZE)
    wall = time.monotonic() - t0
    metrics = save_system("F0", pop, texts, None, wall, out_dir)
    return texts, metrics


def run_f1(
    bundle: Any, pop: Any, v_combined: torch.Tensor,
    baseline: dict[str, str], out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """F1: Oracle-local — steer only at gold CS decoder steps."""
    log.info("F1: Oracle-local steering")
    oracle = oracle_steps_for(pop)
    uid_set = set(oracle)

    def mask_fn(abs_pos: int, item_idx: int) -> bool:
        # Map item_idx back to utterance_id — we need to pass the current
        # batch's uid list.  This is handled below in the per-batch approach.
        return False  # placeholder; we use per-batch decoding

    # Decode per-batch with a hook that knows the batch UIDs
    t0 = time.monotonic()
    texts = _decode_with_oracle_mask(bundle, pop, v_combined, oracle,
                                     num_beams=BEAM_SIZE)
    wall = time.monotonic() - t0
    metrics = save_system("F1", pop, texts, baseline, wall, out_dir)
    return texts, metrics


@torch.inference_mode()
def _decode_with_oracle_mask(
    bundle: Any,
    pop: Any,
    direction: torch.Tensor,
    oracle_steps: dict[str, list[int]],
    *,
    num_beams: int = BEAM_SIZE,
    alpha: float = ALPHA,
) -> dict[str, str]:
    """Decode with NormPreserve steering at oracle positions only."""
    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe", "language": "zh", "do_sample": False,
        "num_beams": num_beams, "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    n_batches = math.ceil(len(manifest) / BATCH_SIZE) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        # Build the oracle mask for this batch
        batch_oracle = {i: set(oracle_steps.get(uid, []))
                        for i, uid in enumerate(ids)}

        def mask_fn(abs_pos: int, item_idx: int) -> bool:
            gen_step = abs_pos - PREFIX_WIDTH
            return gen_step in batch_oracle.get(item_idx, set())

        hook = NormPreserveDecoderHook(
            bundle, DECODER_LAYER, direction, alpha,
            num_beams=num_beams, mask_fn=mask_fn,
        )
        with hook:
            out = bundle.model.generate(**inputs, **kwargs)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

    return texts


@torch.inference_mode()
def _decode_global(
    bundle: Any,
    pop: Any,
    direction: torch.Tensor,
    *,
    alpha: float = ALPHA,
    num_beams: int = BEAM_SIZE,
    alpha_fn: Any | None = None,
) -> dict[str, str]:
    """Decode with global NormPreserve steering (all generated positions)."""
    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe", "language": "zh", "do_sample": False,
        "num_beams": num_beams, "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    n_batches = math.ceil(len(manifest) / BATCH_SIZE) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        # Global mask: all positions after prefix
        def mask_fn(abs_pos: int, item_idx: int) -> bool:
            return abs_pos >= PREFIX_WIDTH

        hook = NormPreserveDecoderHook(
            bundle, DECODER_LAYER, direction, alpha,
            num_beams=num_beams,
            mask_fn=mask_fn if alpha_fn is None else None,
            alpha_fn=alpha_fn,
        )
        with hook:
            out = bundle.model.generate(**inputs, **kwargs)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

    return texts


def run_f2(
    bundle: Any, pop: Any, v_combined: torch.Tensor,
    baseline: dict[str, str], out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """F2: Global, same per-position α=2 at every generated step."""
    log.info("F2: Global steering, α=%g", ALPHA)
    t0 = time.monotonic()
    texts = _decode_global(bundle, pop, v_combined, alpha=ALPHA,
                           num_beams=BEAM_SIZE)
    wall = time.monotonic() - t0
    metrics = save_system("F2", pop, texts, baseline, wall, out_dir)
    return texts, metrics


def run_f3(
    bundle: Any, pop: Any, v_combined: torch.Tensor,
    baseline: dict[str, str], oracle_steps: dict[str, list[int]],
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """F3: Global, matched total energy with F1.

    α_eff = α / √|S| where |S| is the number of generated positions in the
    current utterance.  The frozen baseline hypothesis supplies the per-
    utterance token count before this global decode begins.
    """
    log.info("F3: Global, matched total energy")
    # Get per-utterance generated-position counts from the already-frozen
    # baseline decode.  F3 steers every generated position, so |S| is this
    # utterance-specific count, not a global constant and not the oracle K.
    from csasr.data.normalize import normalize_and_segment

    # Count the generated text positions from the frozen baseline hypothesis.
    baseline_tokens: dict[str, int] = {}
    tokenizer = bundle.processor.tokenizer
    for uid in pop.utterance_ids:
        hyp = baseline.get(uid, "")
        norm, _ = normalize_and_segment(hyp)
        tids = tokenizer.encode(norm, add_special_tokens=False)[:220]
        baseline_tokens[uid] = max(1, len(tids))

    t0 = time.monotonic()

    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe", "language": "zh", "do_sample": False,
        "num_beams": BEAM_SIZE, "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    n_batches = math.ceil(len(manifest) / BATCH_SIZE) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        # Per-item α_eff
        batch_alpha_eff: dict[int, float] = {}
        for item_idx, uid in enumerate(ids):
            N = baseline_tokens.get(uid, 1)
            alpha_eff = ALPHA / math.sqrt(max(1, N))
            batch_alpha_eff[item_idx] = alpha_eff

        def alpha_fn(abs_pos: int, item_idx: int) -> float:
            if abs_pos < PREFIX_WIDTH:
                return 0.0
            return batch_alpha_eff.get(item_idx, 0.0)

        hook = NormPreserveDecoderHook(
            bundle, DECODER_LAYER, v_combined, ALPHA,
            num_beams=BEAM_SIZE, alpha_fn=alpha_fn,
        )
        with hook:
            out = bundle.model.generate(**inputs, **kwargs)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

    wall = time.monotonic() - t0
    metrics = save_system("F3", pop, texts, baseline, wall, out_dir)
    return texts, metrics


def run_f4(
    bundle: Any, pop: Any, v_combined: torch.Tensor,
    baseline: dict[str, str], oracle_steps: dict[str, list[int]],
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """F4: Random-location local — steer at random decoder steps.

    Same count as F1's oracle steps per utterance but at random positions.
    """
    log.info("F4: Random-location local steering")
    rng = np.random.default_rng(SEED)

    # For each utterance, pick K random generation steps (K = oracle count)
    # from the range [0, baseline_token_count)
    from csasr.data.normalize import normalize_and_segment
    from steer_sweep import data as D

    refs = D.reference_of(pop)
    tokenizer = bundle.processor.tokenizer

    random_steps: dict[str, set[int]] = {}
    for uid in pop.utterance_ids:
        K = len(oracle_steps.get(uid, []))
        if K == 0:
            random_steps[uid] = set()
            continue
        ref = refs.get(uid, "")
        norm, _ = normalize_and_segment(ref)
        tids = tokenizer.encode(norm, add_special_tokens=False)[:220]
        N = max(1, len(tids))
        chosen = rng.choice(N, size=min(K, N), replace=False)
        random_steps[uid] = set(int(c) for c in chosen)

    t0 = time.monotonic()
    texts = _decode_with_step_set(bundle, pop, v_combined, random_steps,
                                  num_beams=BEAM_SIZE)
    wall = time.monotonic() - t0
    metrics = save_system("F4", pop, texts, baseline, wall, out_dir)
    return texts, metrics


@torch.inference_mode()
def _decode_with_step_set(
    bundle: Any,
    pop: Any,
    direction: torch.Tensor,
    step_sets: dict[str, set[int]],
    *,
    num_beams: int = BEAM_SIZE,
    alpha: float = ALPHA,
) -> dict[str, str]:
    """Decode with NormPreserve steering at a fixed set of generation steps."""
    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe", "language": "zh", "do_sample": False,
        "num_beams": num_beams, "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    n_batches = math.ceil(len(manifest) / BATCH_SIZE) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        batch_steps = {i: step_sets.get(uid, set()) for i, uid in enumerate(ids)}

        def mask_fn(abs_pos: int, item_idx: int) -> bool:
            gen_step = abs_pos - PREFIX_WIDTH
            return gen_step in batch_steps.get(item_idx, set())

        hook = NormPreserveDecoderHook(
            bundle, DECODER_LAYER, direction, alpha,
            num_beams=num_beams, mask_fn=mask_fn,
        )
        with hook:
            out = bundle.model.generate(**inputs, **kwargs)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

    return texts


def run_f5(
    bundle: Any, pop: Any, gate_params: GateParams,
    baseline: dict[str, str], out_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """F5: Frozen automatic soft-local steering.

    Uses the sigmoid gate calibrated on dev-select.
    m_t = sigmoid((s_t - tau) / T)  where s_t = c1 * dot(h_24t, v_nat)
    """
    log.info("F5: Frozen automatic soft-local, tau=%.4f, T=%.4f",
             gate_params.tau, gate_params.T)
    v_nat_d = gate_params.v_nat.to(bundle.device, bundle.dtype)

    def gate_fn(h_vec: torch.Tensor) -> float:
        """Compute the sigmoid gate from pre-intervention hidden state."""
        s_t = float(torch.dot(h_vec.to(v_nat_d.dtype), v_nat_d))
        m_t = 1.0 / (1.0 + math.exp(-(s_t - gate_params.tau) / gate_params.T))
        return m_t

    t0 = time.monotonic()

    from csasr.models.whisper import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    kwargs = {
        "task": "transcribe", "language": "zh", "do_sample": False,
        "num_beams": BEAM_SIZE, "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    all_gate_values: list[float] = []
    n_batches = math.ceil(len(manifest) / BATCH_SIZE) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        hook = NormPreserveDecoderHook(
            bundle, DECODER_LAYER, gate_params.v_combined,
            ALPHA, num_beams=BEAM_SIZE, gate_fn=gate_fn,
        )
        with hook:
            out = bundle.model.generate(**inputs, **kwargs)
        all_gate_values.extend(hook.gate_values)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

    wall = time.monotonic() - t0

    # Gate statistics
    if all_gate_values:
        gv = np.array(all_gate_values)
        gate_stats = {
            "n_gate_evaluations": len(gv),
            "mean_m_t": float(gv.mean()),
            "median_m_t": float(np.median(gv)),
            "p90_m_t": float(np.percentile(gv, 90)),
            "p10_m_t": float(np.percentile(gv, 10)),
            "std_m_t": float(gv.std()),
            "frac_above_0.5": float((gv > 0.5).mean()),
            "frac_above_0.1": float((gv > 0.1).mean()),
            "histogram_bins": list(np.histogram(gv, bins=20, range=(0, 1))[0].tolist()),
            "histogram_edges": list(np.histogram(gv, bins=20, range=(0, 1))[1].tolist()),
            "gate_params": {
                "c1": gate_params.c1, "c2": gate_params.c2,
                "tau": gate_params.tau, "T": gate_params.T,
            },
        }
    else:
        gate_stats = {"n_gate_evaluations": 0}

    atomic_write_json(out_dir / "f5_gate_stats.json", gate_stats)
    metrics = save_system("F5", pop, texts, baseline, wall, out_dir,
                          extra={"gate_stats": gate_stats})
    return texts, metrics


# ═══════════════════════════════════════════════════════════════════════════
# SEAME transfer (Part A2)
# ═══════════════════════════════════════════════════════════════════════════

def find_seame_root() -> Path | None:
    """Look for SEAME data at known paths."""
    for candidate in SEAME_CANDIDATES:
        if candidate.is_dir():
            # Check for manifest / audio files
            for sub in ("data", "audio", "wavs", ""):
                check = candidate / sub
                if check.is_dir() and any(check.iterdir()):
                    log.info("SEAME data found at %s", candidate)
                    return candidate
    return None


def run_seame_transfer(
    bundle: Any,
    v_nat: torch.Tensor,
    v_combined: torch.Tensor,
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run S0/S1/S2 on SEAME dev_man and dev_sge if data is available.

    Returns summary dict or empty dict if SEAME unavailable.
    """
    seame_root = find_seame_root()
    if seame_root is None:
        log.info("SEAME data not found at any candidate path; skipping transfer.")
        return {"status": "seame_not_found",
                "candidates_checked": [str(p) for p in SEAME_CANDIDATES]}

    # Attempt to load SEAME manifests
    try:
        import pandas as pd
        results: dict[str, Any] = {"status": "completed", "root": str(seame_root)}

        for split_name in ("dev_man", "dev_sge"):
            manifest_path = seame_root / f"{split_name}.csv"
            if not manifest_path.exists():
                manifest_path = seame_root / "manifests" / f"{split_name}.csv"
            if not manifest_path.exists():
                # Try parquet
                manifest_path = seame_root / f"{split_name}.parquet"
            if not manifest_path.exists():
                log.warning("SEAME %s manifest not found; skipping", split_name)
                results[split_name] = {"status": "manifest_not_found"}
                continue

            if manifest_path.suffix == ".parquet":
                df = pd.read_parquet(manifest_path)
            else:
                df = pd.read_csv(manifest_path)

            # Subsample if needed
            if len(df) > SEAME_MAX_UTTS and not dry_run:
                # Speaker-balanced subsample
                if "speaker_id" in df.columns:
                    sampled = df.groupby("speaker_id").apply(
                        lambda g: g.sample(
                            n=min(len(g), max(1, SEAME_MAX_UTTS // df["speaker_id"].nunique())),
                            random_state=SEED,
                        )
                    ).reset_index(drop=True)
                    df = sampled.head(SEAME_MAX_UTTS)
                else:
                    df = df.sample(n=SEAME_MAX_UTTS, random_state=SEED).reset_index(drop=True)

            if dry_run:
                df = df.head(2)

            log.info("SEAME %s: %d utterances", split_name, len(df))

            # S0: Whisper baseline (greedy)
            for sys_id, direction, alpha_s in [
                ("S0", None, 0.0),
                ("S1", v_nat, ALPHA),
                ("S2", v_combined, ALPHA),
            ]:
                t0 = time.monotonic()
                texts = _decode_seame(bundle, df, direction, alpha_s)
                wall = time.monotonic() - t0

                refs = [str(r) for r in df["transcript_raw"]]
                hyps = [texts.get(str(u), "") for u in df["utterance_id"]]

                from csasr.evaluation.mer import corpus_mer
                m = corpus_mer(refs, hyps)
                entry = {
                    "system": sys_id,
                    "split": split_name,
                    "n_utterances": len(df),
                    "MER": float(m["mer"]),
                    "en_wer": float(m["en_wer"]),
                    "zh_cer": float(m["zh_cer"]),
                    "wall_sec": wall,
                }
                results.setdefault(split_name, {})[sys_id] = entry
                log.info("  SEAME %s/%s: MER=%.4f en_wer=%.4f zh_cer=%.4f (%.1fs)",
                         split_name, sys_id, entry["MER"], entry["en_wer"],
                         entry["zh_cer"], wall)

        atomic_write_json(out_dir / "seame_transfer.json", results)
        return results

    except Exception as exc:
        log.error("SEAME transfer failed: %s", exc, exc_info=True)
        return {"status": "failed", "error": str(exc)}


@torch.inference_mode()
def _decode_seame(
    bundle: Any,
    df: "pd.DataFrame",
    direction: torch.Tensor | None,
    alpha: float,
) -> dict[str, str]:
    """Decode SEAME utterances with optional global steering. Greedy only."""
    from csasr.models.whisper import batch_model_inputs

    kwargs = {
        "task": "transcribe", "language": "zh", "do_sample": False,
        "num_beams": 1, "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False,
    }
    texts: dict[str, str] = {}
    df_sorted = df.sort_values("duration_sec").reset_index(drop=True)
    n_batches = math.ceil(len(df_sorted) / BATCH_SIZE) if len(df_sorted) else 0

    for bi in range(n_batches):
        batch = df_sorted.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        if direction is None or alpha == 0.0:
            out = bundle.model.generate(**inputs, **kwargs)
        else:
            def mask_fn(abs_pos: int, item_idx: int) -> bool:
                return abs_pos >= PREFIX_WIDTH

            hook = NormPreserveDecoderHook(
                bundle, DECODER_LAYER, direction, alpha,
                num_beams=1, mask_fn=mask_fn,
            )
            with hook:
                out = bundle.model.generate(**inputs, **kwargs)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        for i, uid in enumerate(ids):
            texts[uid] = decoded[i].strip()

    return texts


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_metrics: dict[str, dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f"{'System':<28} {'MER':>8} {'PIER':>8} {'EN WER':>8} "
        f"{'ZH CER':>8} {'Corr':>6} {'Corrup':>6} {'OutEdit':>7} "
        f"{'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print("Job A: Frozen Steering Confirmation — Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for sys_id, m in all_metrics.items():
        corr = m.get("corrections", "-")
        corrup = m.get("corruptions", "-")
        out_edit = m.get("outside_region_edits", "-")
        print(
            f"{sys_id:<28} "
            f"{m.get('MER', float('nan')):8.4f} "
            f"{m.get('PIER', float('nan')):8.4f} "
            f"{m.get('en_wer', float('nan')):8.4f} "
            f"{m.get('zh_cer', float('nan')):8.4f} "
            f"{str(corr):>6} {str(corrup):>6} {str(out_edit):>7} "
            f"{m.get('inference_time_sec', 0):8.1f}"
        )

    print("=" * len(header))
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Run 2 utterances per system to verify the pipeline")
    p.add_argument("--out", default=str(RESULTS_DIR),
                   help="Output directory (default: results/job_a)")
    p.add_argument("--config", default=CONFIG_PATH,
                   help="Config YAML path (relative to configs/)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--skip-seame", action="store_true",
                   help="Skip SEAME transfer even if data is available")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dry_run = args.dry_run
    if dry_run:
        log.info("DRY RUN: 2 utterances per system")

    job_start = time.monotonic()

    # ── Load model ────────────────────────────────────────────────────
    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper

    log.info("Loading config and model ...")
    cfg = load_config(args.config)
    bundle = load_whisper(cfg)

    # Verify geometry
    assert bundle.num_decoder_layers == 32, f"Expected 32 decoder layers, got {bundle.num_decoder_layers}"
    assert bundle.d_model == 1280, f"Expected d_model=1280, got {bundle.d_model}"

    # ── Load directions ───────────────────────────────────────────────
    from steer_sweep.directions import DirectionStore

    log.info("Loading directions ...")
    store = DirectionStore(cfg, out_dir / "directions_cache", log=log)
    v_nat_spec = store.get("v_nat", "decoder", DECODER_LAYER, bundle=bundle)
    v_prompt_spec = store.get("v_prompt", "decoder", DECODER_LAYER, bundle=bundle)
    v_combined_spec, combo_detail = store.combined(
        v_nat_spec, v_prompt_spec,
        mode="weighted", weight_nat=WEIGHT_NAT, weight_prompt=WEIGHT_PROMPT,
    )
    v_nat = v_nat_spec.vector       # (D,) float32, unit-norm
    v_prompt = v_prompt_spec.vector  # (D,) float32, unit-norm
    v_combined = v_combined_spec.vector

    log.info("  v_nat source=%s, v_prompt source=%s", v_nat_spec.source, v_prompt_spec.source)
    log.info("  Combined direction: %s", combo_detail)
    log.info("  cos(v_nat, v_prompt) = %.4f", float(v_nat.numpy() @ v_prompt.numpy()))

    # Save direction metadata
    atomic_write_json(out_dir / "direction_info.json", {
        "decoder_layer": DECODER_LAYER,
        "v_nat_source": v_nat_spec.source,
        "v_nat_norm_before_unit": v_nat_spec.norm_before_unit,
        "v_prompt_source": v_prompt_spec.source,
        "v_prompt_norm_before_unit": v_prompt_spec.norm_before_unit,
        "combination": combo_detail,
        "weight_nat": WEIGHT_NAT, "weight_prompt": WEIGHT_PROMPT,
        "alpha": ALPHA, "beam_size": BEAM_SIZE,
    })

    # This is evaluation preparation only; all F5 calibration remains on
    # D-dev-select below, and Job B never reads D-dev-confirm.
    ensure_confirm_poi(bundle, cfg)

    # ── Calibrate F5 gate on dev-select ───────────────────────────────
    gate_params = calibrate_f5_gate(bundle, cfg, v_nat, v_prompt, v_combined,
                                    dry_run=dry_run)
    atomic_write_json(out_dir / "f5_gate_calibration.json", {
        "c1": gate_params.c1, "c2": gate_params.c2,
        "tau": gate_params.tau, "T": gate_params.T,
        "calibrated_on": "D-dev-select",
        "note": "NEVER calibrated on dev-confirm",
    })

    # ── Build dev-confirm population ──────────────────────────────────
    from steer_sweep import data as D

    log.info("Building dev-confirm population ...")
    pop = D.build_population(bundle, cfg, "D-dev-confirm",
                             confirm_authorized=True,
                             assert_anchors=False)
    if dry_run:
        pop = pop.subsample(2)
    log.info("  Population: %s", pop.counts)

    # ── Run F0–F5 ─────────────────────────────────────────────────────
    all_metrics: dict[str, dict[str, Any]] = {}

    # F0: Baseline
    f0_texts, f0_metrics = run_f0(bundle, pop, out_dir)
    all_metrics["F0: Baseline"] = f0_metrics
    baseline = f0_texts  # All other systems are compared against F0

    # Oracle steps (needed for F1, F3, F4)
    oracle = oracle_steps_for(pop)
    log.info("  Oracle steps: %d utterances with targets, %d total steps",
             len(oracle), sum(len(v) for v in oracle.values()))

    # F1: Oracle-local
    _, f1_metrics = run_f1(bundle, pop, v_combined, baseline, out_dir)
    all_metrics["F1: Oracle-local"] = f1_metrics

    # F2: Global, same α
    _, f2_metrics = run_f2(bundle, pop, v_combined, baseline, out_dir)
    all_metrics["F2: Global α=2"] = f2_metrics

    # F3: Global, matched energy
    _, f3_metrics = run_f3(bundle, pop, v_combined, baseline, oracle, out_dir)
    all_metrics["F3: Global matched-E"] = f3_metrics

    # F4: Random-location local
    _, f4_metrics = run_f4(bundle, pop, v_combined, baseline, oracle, out_dir)
    all_metrics["F4: Random-local"] = f4_metrics

    # F5: Frozen automatic soft-local
    _, f5_metrics = run_f5(bundle, pop, gate_params, baseline, out_dir)
    all_metrics["F5: Auto soft-local"] = f5_metrics

    # ── SEAME transfer (Part A2) ──────────────────────────────────────
    if not args.skip_seame:
        elapsed = time.monotonic() - job_start
        remaining = 10800 - elapsed  # 3-hour budget
        if remaining > 600:  # need at least 10 min
            seame_results = run_seame_transfer(
                bundle, v_nat, v_combined, out_dir, dry_run=dry_run,
            )
            if seame_results.get("status") == "completed":
                for split_name in ("dev_man", "dev_sge"):
                    split_data = seame_results.get(split_name, {})
                    if isinstance(split_data, dict):
                        for sys_id in ("S0", "S1", "S2"):
                            entry = split_data.get(sys_id)
                            if entry:
                                all_metrics[f"SEAME {split_name}/{sys_id}"] = entry
        else:
            log.warning("Insufficient time remaining (%.1f min) for SEAME; skipping",
                        remaining / 60)
    else:
        log.info("SEAME transfer skipped by --skip-seame")

    # ── Summary ───────────────────────────────────────────────────────
    total_wall = time.monotonic() - job_start
    print_summary(all_metrics)
    print(f"Total wall-clock time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")

    atomic_write_json(out_dir / "summary.json", {
        "systems": all_metrics,
        "total_wall_sec": total_wall,
        "dry_run": dry_run,
        "config": {
            "decoder_layer": DECODER_LAYER, "alpha": ALPHA,
            "weight_nat": WEIGHT_NAT, "weight_prompt": WEIGHT_PROMPT,
            "beam_size": BEAM_SIZE, "batch_size": BATCH_SIZE, "seed": SEED,
        },
    })

    log.info("All results saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
