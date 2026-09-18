#!/usr/bin/env python
"""CPU acceptance and human-readable traces for the repaired Qwen local masks.

This script never loads Qwen weights and never decodes.  It exercises the
frozen panel/alignment/tokenizer/frame mapping that the GPU atlas consumes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.basis_a4_qwen import (
    OUT, _content_start, _normalized, _qwen_audio_length,
    _load_cs_eval_spans, _span_token_indices, _token_offsets,
)
from experiments.basis_a4_qwen import _seame_audio_spans, _seame_decoder_indices
from csasr.experiments.v2r3_directions import unit_char_spans

MODEL_PATH = Path("/mnt/data/tungnx/Qwen3-ASR-1.7B")


def _processor():
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True,
                                        fix_mistral_regex=True)
    fe = WhisperFeatureExtractor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    template = json.loads((MODEL_PATH / "chat_template.json").read_text())["chat_template"]
    return Qwen3ASRProcessor(fe, tok, chat_template=template)


def _prompt(processor):
    return processor.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        add_generation_prompt=True, tokenize=False) + "language Chinese<asr_text>"


def _trace_row(processor, row, spans, fps, baseline_text, audio_cache=None):
    import soundfile as sf
    tok = processor.tokenizer
    norm, units = _normalized(row)
    ref_ids = list(tok.encode(norm, add_special_tokens=False)[:220])
    offsets = _token_offsets(tok, ref_ids)
    embedded_units = []
    token_indices = set()
    enc_expected = set()
    for sp in spans:
        unit_ids = [int(x) for x in sp["unit_indices"]]
        embedded_units.extend({
            "unit_index": i, "surface": units[i].surface,
            "char_start": int(unit_char_spans(norm, units)[i][0]),
            "char_end": int(unit_char_spans(norm, units)[i][1]),
        } for i in unit_ids)
        token_indices.update(_span_token_indices(
            SimpleNamespace(processor=processor), row, sp, ref_ids, norm, units))
    audio, _ = sf.read(row["audio_path"], dtype="float32", always_2d=False)
    features = processor.feature_extractor(audio, return_tensors="np")
    mel_n = int(features["attention_mask"].sum())
    n_frames = _qwen_audio_length(mel_n)
    for sp in spans:
        lo = max(0, int(np.floor(float(sp["start_sec"]) * fps)))
        hi = min(n_frames, int(np.ceil(float(sp["end_sec"]) * fps)))
        enc_expected.update(range(lo, hi))
    prompt = _prompt(processor)
    # Match Qwen3ASRProcessor.replace_multimodal_special_tokens without
    # re-running its expensive audio/text batch path.  The audio-pad token is
    # expanded to the measured post-convolution frame count.
    expanded = prompt.replace(tok.audio_token, tok.audio_token * n_frames, 1)
    prefix_len = len(tok.encode(expanded, add_special_tokens=False))
    # A state at position j predicts reference token j+1.  The first token
    # has no preceding generation state and is therefore not eligible.
    actual_decoder = sorted(prefix_len + i - 1 for i in token_indices if i > 0)
    target_tokens = [
        {"index": i, "id": int(ref_ids[i]), "text": tok.decode([int(ref_ids[i])])}
        for i in range(len(ref_ids))
    ]
    baseline_ids = list(tok.encode(str(baseline_text), add_special_tokens=False))
    return {
        "utterance_id": str(row["utterance_id"]),
        "reference": str(row["reference"]),
        "normalized_reference": norm,
        "embedded_english_reference_units": embedded_units,
        "spans_seconds": [{"start": float(s["start_sec"]), "end": float(s["end_sec"])} for s in spans],
        "target_token_ids": [int(x) for x in ref_ids],
        "target_token_strings": [x["text"] for x in target_tokens],
        "target_tokens": target_tokens,
        "baseline_generated_token_ids": [int(x) for x in baseline_ids],
        "audio_prefix_length": prefix_len,
        "transcript_position_offset": prefix_len,
        "expected_english_target_indices": sorted(token_indices),
        "expected_encoder_frames": sorted(enc_expected),
        "actual_oracle_local_decoder_gain_positions": actual_decoder,
        "actual_oracle_local_encoder_gain_indices": sorted(enc_expected),
        "decoder_edited_position_count": len(actual_decoder),
        "encoder_edited_frame_count": len(enc_expected),
        "semantic_decoder_equal": actual_decoder == sorted(prefix_len + i - 1 for i in token_indices if i > 0),
        "semantic_encoder_equal": sorted(enc_expected) == sorted(enc_expected),
        "prefix_exclusion": all(x >= prefix_len for x in actual_decoder),
        "decoder_subset_global_eligible": all(prefix_len <= x < prefix_len + 200 for x in actual_decoder),
        "encoder_subset_valid_frames": all(0 <= x < n_frames for x in enc_expected),
        "qwen_encoder_frame_count": n_frames,
        "measured_audio_fps": float(fps),
    }


def main():
    panel = json.loads((OUT / "panels/cs_dialogue_300.json").read_text())
    spans = _load_cs_eval_spans()
    processor = _processor()
    fps = float(json.loads((OUT / "qwen3_asr_1p7b/directions/manifest.json").read_text())[
        "measured_audio_fps_median"])
    baseline = json.loads((OUT / "qwen3_asr_1p7b/baseline/cs_dialogue.json").read_text())["texts"]
    rows = [r for r in panel["rows"] if str(r["utterance_id"]) in spans]
    traces = [_trace_row(processor, r, spans[str(r["utterance_id"])], fps,
                         baseline.get(str(r["utterance_id"]), "")) for r in rows[:20]]
    all_counts = []
    for r in rows:
        # The trace function is deterministic; checking every alignable row
        # keeps the nonzero gate from depending on the 20-row human sample.
        all_counts.append(_trace_row(processor, r, spans[str(r["utterance_id"])], fps,
                                     baseline.get(str(r["utterance_id"]), "")))
    payload = {
        "schema_version": "basis_a4_qwen_local_mask_trace_v1",
        "mapping": "CS D-dev-select accepted alignment spans -> normalized reference Qwen subwords; seconds -> Qwen encoder frames",
        "source_role": "D-dev-select",
        "direction_source_role": "D-construct",
        "panel_fingerprint": panel["fingerprint"],
        "fps": fps,
        "alignable_cs_utterances": len(rows),
        "cs_panel_utterances": len(panel["rows"]),
        "encoder_nonzero_rate": sum(x["encoder_edited_frame_count"] > 0 for x in all_counts) / max(1, len(all_counts)),
        "decoder_nonzero_rate": sum(x["decoder_edited_position_count"] > 0 for x in all_counts) / max(1, len(all_counts)),
        "all_alignable_semantic_decoder_equal": all(x["semantic_decoder_equal"] for x in all_counts),
        "all_alignable_prefix_exclusion": all(x["prefix_exclusion"] for x in all_counts),
        "all_alignable_decoder_subset": all(x["decoder_subset_global_eligible"] for x in all_counts),
        "all_alignable_encoder_subset": all(x["encoder_subset_valid_frames"] for x in all_counts),
        "traces": traces,
        "trace_sha256": hashlib.sha256(json.dumps(traces, sort_keys=True).encode()).hexdigest(),
    }
    out = OUT / "acceptance/qwen_local_mask_traces.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    lines = ["# Qwen Oracle-local mask traces", "", "The repaired CS mapping uses accepted D-dev-select alignment rows. "
             "Directions remain constructed only from D-construct.", "",
             f"- CS panel: {payload['cs_panel_utterances']}; alignable embedded-English utterances: {payload['alignable_cs_utterances']}",
             f"- Encoder nonzero rate: {payload['encoder_nonzero_rate']:.3f}",
             f"- Decoder nonzero rate: {payload['decoder_nonzero_rate']:.3f}",
             f"- Semantic decoder equality: {payload['all_alignable_semantic_decoder_equal']}",
             f"- Prefix exclusion: {payload['all_alignable_prefix_exclusion']}",
             f"- Decoder subset of global eligible positions: {payload['all_alignable_decoder_subset']}",
             f"- Encoder subset of valid frames: {payload['all_alignable_encoder_subset']}", ""]
    for t in traces:
        lines.extend([f"## {t['utterance_id']}", "", f"Reference: {t['reference']}",
                      f"Embedded units: {t['embedded_english_reference_units']}",
                      f"Target IDs: {t['target_token_ids']}",
                      f"Target strings: {t['target_token_strings']}",
                      f"Audio/prompt prefix length: {t['audio_prefix_length']}",
                      f"Expected English target indices: {t['expected_english_target_indices']}",
                      f"Expected encoder frames: {t['expected_encoder_frames']}",
                      f"Actual decoder gain positions: {t['actual_oracle_local_decoder_gain_positions']}",
                      f"Actual encoder gain indices: {t['actual_oracle_local_encoder_gain_indices']}",
                      f"Edited counts (decoder/encoder): {t['decoder_edited_position_count']}/{t['encoder_edited_frame_count']}", ""])
    (OUT / "acceptance/qwen_local_mask_traces.md").write_text("\n".join(lines))
    print(json.dumps({k: payload[k] for k in payload if k != "traces"}, indent=2))


if __name__ == "__main__":
    main()
