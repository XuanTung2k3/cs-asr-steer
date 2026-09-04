"""Cross-attention pseudo-label spans: a third, independent derivation path.

Following Liu et al., IEEE TASLP 2025 (arXiv 2403.05887).  CTC token spans
exclude blank frames and a blank frame carries no language attribute, so a CTC
switch boundary is defined only up to the unowned blank run between two tokens.
That paper reports the same problem and derives language spans from decoder
cross-attention instead, finding pseudo-labels that outperform forced alignment
from a comparable-strength model.

**No DTW and no monotonic path.**  Each encoder frame is assigned the language of
the token that attends to it most -- an argmax over an averaged attention matrix
-- and contiguous same-language frames become a span.  That is the whole method,
and it is why this path can succeed where `whisper_dtw` failed: nothing here
requires a monotonic alignment to exist.

The model is frozen: `torch.inference_mode`, a single teacher-forced forward
pass over gold transcripts, no parameter update of any kind.

Every parameter below is frozen before the run and recorded in the manifest.
None was chosen by looking at results.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.alignment import build_prefix
from ..data.language_tags import EN, ZH, tag_unit
from ..data.normalize import normalize_and_segment
from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..models.generation import teacher_forced_forward
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging

STAGE = "v2r3_cross_attention_spans"
FAMILY = "cross_attention_pseudo"

# ---------------------------------------------------------------------------
# FROZEN CONFIGURATION -- set before the first run, recorded in the manifest
# ---------------------------------------------------------------------------

#: Final decoder layer.  The paper reads language identity off the deepest
#: cross-attention, where token identity is most resolved.
DECODER_LAYER_INDEX = -1

#: Every head of that layer, averaged.  Not a learned or selected subset: the
#: alignment-head subset Whisper ships is tuned for *timing* under DTW, which is
#: the mechanism this path deliberately avoids.
HEAD_SET = "all_heads_mean"

#: Mode filter over the per-frame label sequence, in frames (20 ms each).
#: 9 frames = 180 ms, about one syllable, chosen to suppress single-frame
#: flicker without merging across a real switch.
SMOOTHING_WINDOW_FRAMES = 9

#: A span shorter than this is discarded.  5 frames = 100 ms, deliberately below
#: the smallest eligibility tier (200 ms) so this step cannot pre-empt a floor.
MIN_SPAN_FRAMES = 5

#: Token classes.  Special tokens are their own class and are never assigned to
#: a language.  `neutral` is the fourth class -- punctuation and whitespace that
#: belong to no language -- and is likewise never silently folded into one.
CLASS_EN, CLASS_ZH, CLASS_NEUTRAL, CLASS_SPECIAL = "EN", "ZH", "neutral", "special"
LANGUAGE_CLASSES = (CLASS_EN, CLASS_ZH)

#: Prefix language token for the teacher-forced sequence.  The corpus is
#: Mandarin-matrix; this mirrors the existing alignment path.
PREFIX_LANGUAGE = "zh"

#: Frames beyond the utterance's own duration are encoder padding.  Whisper pads
#: every input to 30 s and applies no encoder attention mask, so the decoder can
#: and does attend into that padding; those frames are excluded by duration and
#: the attention mass landing there is measured and reported.
MAX_ENCODER_FRAMES = 1500

FROZEN_CONFIG: dict[str, Any] = {
    "decoder_layer_index": DECODER_LAYER_INDEX,
    "head_set": HEAD_SET,
    "smoothing": {"kind": "mode_filter", "window_frames": SMOOTHING_WINDOW_FRAMES,
                  "window_ms": SMOOTHING_WINDOW_FRAMES * 20},
    "min_span": {"frames": MIN_SPAN_FRAMES, "ms": MIN_SPAN_FRAMES * 20},
    "special_token_handling": (
        "special tokens form their own class and are never assigned a language; "
        "a frame whose argmax token is special is labelled 'special' and cannot "
        "join a language span. Punctuation and whitespace form a fourth "
        "'neutral' class with the same protection."),
    "prefix_language": PREFIX_LANGUAGE,
    "teacher_forced": True,
    "parameters_updated": False,
    "uses_dtw": False,
    "requires_monotonic_path": False,
    "frame_assignment": "argmax over tokens of head-averaged cross-attention",
}


# ---------------------------------------------------------------------------
# token classification
# ---------------------------------------------------------------------------

def token_classes(tokenizer, token_ids: Sequence[int]) -> list[str]:
    """One class per decoder token: EN, ZH, neutral, or special."""
    special = set(tokenizer.all_special_ids or ())
    classes: list[str] = []
    for token_id in token_ids:
        if int(token_id) in special:
            classes.append(CLASS_SPECIAL)
            continue
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        stripped = text.strip()
        if not stripped:
            classes.append(CLASS_NEUTRAL)
            continue
        if stripped.startswith("<|") and stripped.endswith("|>"):
            classes.append(CLASS_SPECIAL)
            continue
        tag = tag_unit(stripped)
        if tag == EN:
            classes.append(CLASS_EN)
        elif tag == ZH:
            classes.append(CLASS_ZH)
        else:
            classes.append(CLASS_NEUTRAL)
    return classes


def mode_filter(labels: Sequence[str], window: int) -> list[str]:
    """Majority label in a centred window.  Ties keep the original label."""
    if window <= 1 or not labels:
        return list(labels)
    half = window // 2
    out: list[str] = []
    for i, label in enumerate(labels):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        counts = Counter(labels[lo:hi])
        best = counts.most_common()
        if len(best) > 1 and best[0][1] == best[1][1]:
            out.append(label)
        else:
            out.append(best[0][0])
    return out


def spans_from_labels(labels: Sequence[str], *, step_sec: float,
                      min_frames: int = MIN_SPAN_FRAMES) -> list[dict[str, Any]]:
    """Contiguous same-language frame groups, as spans.  Never merges across a
    special or neutral stretch: only EN and ZH runs become spans."""
    spans: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i < len(labels) and labels[i] == labels[start]:
            continue
        label = labels[start]
        length = i - start
        if label in LANGUAGE_CLASSES and length >= min_frames:
            spans.append({"language": label, "start_frame": start, "end_frame": i,
                          "n_frames": length,
                          "start_sec": start * step_sec,
                          "end_sec": i * step_sec})
        start = i
    return spans


# ---------------------------------------------------------------------------
# the forward pass
# ---------------------------------------------------------------------------

@torch.inference_mode()
def pseudo_label_batch(bundle, rows: pd.DataFrame, *,
                       max_text_tokens: int = 220) -> tuple[list[dict], dict]:
    """Spans for one batch, plus the real-model observations for that batch."""
    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language=PREFIX_LANGUAGE)
    eot = tokenizer.eos_token_id

    sequences, metas = [], []
    for _, row in rows.iterrows():
        norm, _ = normalize_and_segment(row["transcript_raw"])
        text_ids = tokenizer.encode(norm, add_special_tokens=False)[:max_text_tokens]
        sequences.append(list(prefix) + list(text_ids) + [eot])
        metas.append(row)

    out, padded, mask = teacher_forced_forward(
        bundle, rows["audio_path"].tolist(), sequences, output_attentions=True)
    layer = out.cross_attentions[DECODER_LAYER_INDEX]          # (B, heads, tgt, src)
    averaged = layer.float().mean(dim=1).cpu().numpy()          # (B, tgt, src)
    n_heads = int(layer.shape[1])
    del out, layer

    step = float(bundle.encoder_step_sec)
    results: list[dict[str, Any]] = []
    observed = {"n_heads_averaged": n_heads, "batch": int(len(rows)),
                "attention_mass_beyond_valid_frames": [],
                "decoder_pad_positions": 0, "decoder_pad_attention_mass": [],
                "frames_by_class": Counter(), "encoder_frames_returned": int(averaged.shape[2])}

    for b, row in enumerate(metas):
        tokens = sequences[b]
        classes = token_classes(tokenizer, tokens)
        attention = averaged[b, : len(tokens), :]               # (tgt, src)

        duration = float(row["duration_sec"])
        valid_frames = max(1, min(MAX_ENCODER_FRAMES, int(round(duration / step))))

        # real-model check 2: the encoder has no attention mask, so measure how
        # much mass lands beyond the audio rather than assuming it is zero
        total = float(attention.sum())
        beyond = float(attention[:, valid_frames:].sum())
        observed["attention_mass_beyond_valid_frames"].append(
            beyond / total if total > 0 else 0.0)
        # and confirm the decoder pad mask excludes padded query positions
        pad_positions = int((~mask[b]).sum().item())
        observed["decoder_pad_positions"] += pad_positions
        if pad_positions:
            padded_rows = averaged[b, len(tokens):, :]
            observed["decoder_pad_attention_mass"].append(float(padded_rows.sum()))

        window = attention[:, :valid_frames]                    # (tgt, valid)
        argmax_token = window.argmax(axis=0)                    # per frame
        raw_labels = [classes[int(t)] for t in argmax_token]
        observed["frames_by_class"].update(raw_labels)
        labels = mode_filter(raw_labels, SMOOTHING_WINDOW_FRAMES)
        spans = spans_from_labels(labels, step_sec=step)
        for span in spans:
            span.update({"utterance_id": str(row["utterance_id"]),
                         "role": str(row.get("role", "")),
                         "conversation_id": str(row.get("conversation_id", "")),
                         "dialogue_id": str(row.get("dialogue_id", "")),
                         "duration_sec": duration,
                         "valid_frames": valid_frames})
        results.append({"utterance_id": str(row["utterance_id"]),
                        "spans": spans, "valid_frames": valid_frames,
                        "duration_sec": duration, "n_tokens": len(tokens),
                        "empirical_ms_per_frame": (duration * 1000.0 / valid_frames)})
    return results, observed


# ---------------------------------------------------------------------------
# comparison against the other paths
# ---------------------------------------------------------------------------

def _match_by_overlap(left: pd.DataFrame, right: pd.DataFrame) -> list[tuple[int, int]]:
    """Greedy one-to-one match of same-language runs by temporal overlap.

    Positional order would break as soon as one path finds a run the other
    misses; overlap degrades gracefully instead. Preregistered, not tuned.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, a in left.iterrows():
        for j, b in right.iterrows():
            overlap = (min(float(a["end_sec"]), float(b["end_sec"]))
                       - max(float(a["start_sec"]), float(b["start_sec"])))
            if overlap > 0:
                pairs.append((overlap, i, j))
    pairs.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    out: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        out.append((i, j))
    return out


def compare_paths(pseudo: pd.DataFrame, other: pd.DataFrame,
                  label: str) -> dict[str, Any]:
    """Boundary disagreement between the pseudo-label path and one other path."""
    rows: list[dict[str, Any]] = []
    keys = set(map(tuple, pseudo[["utterance_id", "language"]].drop_duplicates().values))
    keys |= set(map(tuple, other[["utterance_id", "language"]].drop_duplicates().values))
    for utterance, language in sorted(keys):
        a = pseudo[(pseudo["utterance_id"] == utterance)
                   & (pseudo["language"] == language)]
        b = other[(other["utterance_id"] == utterance)
                  & (other["language"] == language)]
        if not len(a) or not len(b):
            continue
        for i, j in _match_by_overlap(a, b):
            left, right = a.loc[i], b.loc[j]
            dstart = abs(float(left["start_sec"]) - float(right["start_sec"]))
            dend = abs(float(left["end_sec"]) - float(right["end_sec"]))
            rows.append({"utterance_id": utterance, "language": language,
                         "dstart_ms": dstart * 1000.0, "dend_ms": dend * 1000.0,
                         "abs_boundary_ms": max(dstart, dend) * 1000.0})
    frame = pd.DataFrame(rows)
    if not len(frame):
        return {"compared_with": label, "n": 0}
    def block(sub: pd.DataFrame) -> dict[str, Any]:
        return {"n": int(len(sub)),
                "median_boundary_ms": float(sub["abs_boundary_ms"].median()),
                "p90_boundary_ms": float(sub["abs_boundary_ms"].quantile(0.90)),
                "within_100ms": float((sub["abs_boundary_ms"] <= 100).mean()),
                "within_200ms": float((sub["abs_boundary_ms"] <= 200).mean())}
    per_language = {str(lang): block(sub) for lang, sub in frame.groupby("language")}
    en = per_language.get(EN, {}).get("median_boundary_ms")
    zh = per_language.get(ZH, {}).get("median_boundary_ms")
    return {"compared_with": label, **block(frame), "by_language": per_language,
            "en_minus_zh_median_ms": (en - zh) if en is not None and zh is not None else None,
            "measurement": "cross_aligner_disagreement",
            "note": ("estimator disagreement between two independent automatic "
                     "paths on natural speech, never error against truth"),
            "matching": "greedy one-to-one by temporal overlap within utterance and language"}


# ---------------------------------------------------------------------------
# eligibility and the driver
# ---------------------------------------------------------------------------

def eligibility_profile(spans: pd.DataFrame, *,
                        floors_ms: Sequence[float] = (400.0, 300.0, 200.0)
                        ) -> dict[str, Any]:
    """Span counts under the current rule.  No confidence condition."""
    english = spans[spans["language"] == EN].copy()
    english["duration_ms"] = (english["end_sec"] - english["start_sec"]) * 1000.0
    profile: dict[str, Any] = {
        "embedded_english_spans": int(len(english)),
        "duration_ms": {
            "min": float(english["duration_ms"].min()) if len(english) else None,
            "q1": float(english["duration_ms"].quantile(0.25)) if len(english) else None,
            "median": float(english["duration_ms"].median()) if len(english) else None,
            "q3": float(english["duration_ms"].quantile(0.75)) if len(english) else None,
            "p90": float(english["duration_ms"].quantile(0.90)) if len(english) else None,
            "max": float(english["duration_ms"].max()) if len(english) else None,
        },
        "by_role": {},
    }
    for role, group in english.groupby("role"):
        tiers = {}
        for floor in floors_ms:
            kept = group[group["duration_ms"] >= floor]
            tiers[f">={int(floor)}ms"] = {
                "spans": int(len(kept)),
                "utterances": int(kept["utterance_id"].nunique()),
                "dialogues": int(kept["dialogue_id"].nunique()),
                "speakers": int(kept["conversation_id"].nunique()),
            }
        profile["by_role"][str(role)] = {"spans": int(len(group)), "tiers": tiers}
    profile["eligibility_note"] = (
        "matrix-preceded, non-final and contiguous-index conditions are defined "
        "on reference-unit sequences; this path derives spans from frames and "
        "carries no reference-unit index, so only the duration floor is applied "
        "here and the remaining conditions are reported as not applicable")
    return profile


def resolve_output(output: str | Path) -> Path:
    target = Path(output)
    if target.is_symlink():
        raise SystemExit(f"refusing a symlink destination: {target}")
    target = target.resolve()
    if "artifacts_lss" in str(target):
        raise SystemExit(f"refusing to write inside the v1 root: {target}")
    if target.exists() or manifest_mod.manifest_path(target).exists():
        raise SystemExit(f"refusing an existing destination: {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target = resolve_output(args.output_dir)
    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"

    # byte verification only: v2r3 sidecars are legacy format and gate on the
    # combined source hash, which the tree has moved past. Bytes are the
    # property required for reading, and _parents sets the same precedent.
    verdict = manifest_mod.verify(candidate_path, cfg=None, require_identity=False)
    if not verdict["ok"]:
        raise SystemExit(f"candidate table failed byte verification: {verdict}")
    candidates, _ = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)

    role_root = Path(cfg["v2_namespace"]["role_root"])
    manifests = []
    for role in conv.DEVELOPMENT_ROLES:
        path = role_root / f"role_{role}.parquet"
        role_verdict = manifest_mod.verify(path, cfg=None, require_identity=False)
        if not role_verdict["ok"]:
            raise SystemExit(f"role manifest failed byte verification: {role_verdict}")
        manifests.append(pd.read_parquet(path))
    manifest = pd.concat(manifests, ignore_index=True)
    sampled = set(candidates["utterance_id"].astype(str))
    manifest = manifest[manifest["utterance_id"].astype(str).isin(sampled)]
    manifest = manifest.sort_values("utterance_id").reset_index(drop=True)

    plan = {"frozen_configuration": FROZEN_CONFIG, "family": FAMILY,
            "roles": list(conv.DEVELOPMENT_ROLES),
            "utterances": int(len(manifest)),
            "candidate_path": str(candidate_path),
            "candidate_sha256": verdict["manifest"]["sha256"],
            "input_identity_format": "legacy (byte-verified, identity not checked)",
            "output_identity_format": "split (code_config_sha256 + test_sha256)",
            "destination": str(target)}
    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan}, indent=2, default=str))
        return 0

    log = setup_logging(level="INFO")
    bundle = load_whisper(cfg)
    log.info("cross-attention pseudo-labels over %d utterances, layer=%s heads=%s",
             len(manifest), DECODER_LAYER_INDEX, HEAD_SET)

    spans_rows: list[dict[str, Any]] = []
    per_utterance: list[dict[str, Any]] = []
    observations = {"attention_mass_beyond_valid_frames": [],
                    "decoder_pad_positions": 0, "decoder_pad_attention_mass": [],
                    "frames_by_class": Counter(), "n_heads_averaged": None,
                    "encoder_frames_returned": None}
    for start in range(0, len(manifest), args.batch_size):
        batch = manifest.iloc[start:start + args.batch_size]
        results, observed = pseudo_label_batch(bundle, batch)
        for item in results:
            spans_rows.extend(item.pop("spans"))
            per_utterance.append(item)
        observations["attention_mass_beyond_valid_frames"].extend(
            observed["attention_mass_beyond_valid_frames"])
        observations["decoder_pad_positions"] += observed["decoder_pad_positions"]
        observations["decoder_pad_attention_mass"].extend(
            observed["decoder_pad_attention_mass"])
        observations["frames_by_class"].update(observed["frames_by_class"])
        observations["n_heads_averaged"] = observed["n_heads_averaged"]
        observations["encoder_frames_returned"] = observed["encoder_frames_returned"]
        if (start // args.batch_size) % 20 == 0:
            log.info("  %d/%d utterances", start + len(batch), len(manifest))

    spans = pd.DataFrame(spans_rows)
    utterances = pd.DataFrame(per_utterance)

    frames_total = sum(observations["frames_by_class"].values())
    ms_per_frame = utterances["empirical_ms_per_frame"]
    checks = {
        "1_frame_rate": {
            "expected_ms_per_frame": bundle.encoder_step_sec * 1000.0,
            "empirical_ms_per_frame_median": float(ms_per_frame.median()),
            "empirical_ms_per_frame_min": float(ms_per_frame.min()),
            "empirical_ms_per_frame_max": float(ms_per_frame.max()),
            "encoder_frames_returned": observations["encoder_frames_returned"],
            "note": ("empirical value is duration/valid_frames, so it equals the "
                     "expected step up to the rounding of duration onto whole "
                     "frames; the encoder always returns 1500 frames because "
                     "Whisper pads every input to 30 s"),
        },
        "2_attention_mask": {
            "encoder_attention_mask_exists": False,
            "mean_attention_mass_beyond_valid_frames": float(
                np.mean(observations["attention_mass_beyond_valid_frames"])),
            "max_attention_mass_beyond_valid_frames": float(
                np.max(observations["attention_mass_beyond_valid_frames"])),
            "decoder_pad_positions_seen": int(observations["decoder_pad_positions"]),
            "decoder_pad_attention_mass_excluded": float(
                np.sum(observations["decoder_pad_attention_mass"])),
            "observed": ("Whisper applies no encoder attention mask -- every input "
                         "is padded to 30 s and all 1500 frames are attended. The "
                         "mass landing beyond each utterance's own duration is "
                         "measured above rather than assumed to be zero, and those "
                         "frames are excluded by duration before any argmax. "
                         "Decoder padded query rows are excluded by slicing to the "
                         "true token count of each sequence."),
        },
        "3_special_token_frames": {
            "frames_by_class": {k: int(v) for k, v in
                                observations["frames_by_class"].items()},
            "fraction_special": (observations["frames_by_class"][CLASS_SPECIAL]
                                 / frames_total) if frames_total else None,
            "fraction_neutral": (observations["frames_by_class"][CLASS_NEUTRAL]
                                 / frames_total) if frames_total else None,
            "total_frames": int(frames_total),
        },
        "4_spans": eligibility_profile(spans),
    }

    comparisons = []
    # switch_transitions returns (frame, edge_counts); only the frame is the
    # transition table apply_convention expects
    transitions, _transition_edges = conv.switch_transitions(candidates)
    whisper_done = False
    for convention in conv.CONVENTIONS:
        targets, _ = conv.convention_targets(candidates, convention,
                                             transitions=transitions)
        selected = targets[targets["target_valid"].astype(bool)]
        ctc = selected[selected["aligner_family"].astype(str) == conv.CONVENTION_FAMILY]
        comparisons.append(compare_paths(spans, ctc, f"existing_ctc/{convention}"))
        if not whisper_done:
            whisper = selected[selected["aligner_family"].astype(str)
                               == conv.REFERENCE_FAMILY]
            comparisons.append(compare_paths(spans, whisper, conv.REFERENCE_FAMILY))
            whisper_done = True
    checks["5_en_minus_zh"] = {
        entry["compared_with"]: entry.get("en_minus_zh_median_ms")
        for entry in comparisons}

    payload = {"frozen_configuration": FROZEN_CONFIG, "family": FAMILY,
               "plan": plan, "real_model_checks": checks,
               "three_way_comparison": comparisons,
               "selects_no_path": True, "evaluates_no_gate": True,
               "fine_tuned": False, "uses_dtw": False}

    target.mkdir(parents=True, exist_ok=True)
    spans_path = target / "cross_attention_spans.parquet"
    spans.to_parquet(spans_path, index=False)
    utt_path = target / "cross_attention_utterances.parquet"
    utterances.to_parquet(utt_path, index=False)
    report = target / "cross_attention_report.json"
    report.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    for artifact in (spans_path, utt_path, report):
        manifest_mod.publish(artifact, stage=STAGE, cfg=cfg,
                             parents=[parent] if parent else (),
                             schema="lss_v2r3_cross_attention_spans_v1",
                             extra={"frozen_configuration": FROZEN_CONFIG,
                                    "family": FAMILY, "evaluates_no_gate": True})
    print(json.dumps({"state": "completed", "output": str(target),
                      "spans": int(len(spans)),
                      "real_model_checks": checks,
                      "comparison": [{k: v for k, v in c.items()
                                      if k in ("compared_with", "n", "median_boundary_ms",
                                               "p90_boundary_ms", "within_100ms",
                                               "within_200ms", "en_minus_zh_median_ms")}
                                     for c in comparisons]}, indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
