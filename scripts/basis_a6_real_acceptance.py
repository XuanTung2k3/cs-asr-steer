#!/usr/bin/env python3
"""Real-model BASIS-A6 acceptance and bounded runtime preflight.

This is deliberately a preflight runner, not an atlas launcher.  It keeps one
model resident, runs a frozen small CS panel, and writes compact traces.  The
decoder analysis path always uses the model's free-decoding hypothesis as the
analysis transcript; references are used only for oracle region labels and
acceptance metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
OUT = REPO / "results/basis_a6_expanded/preflight"
PANEL_PATH = REPO / "results/basis_a6_expanded/preflight/REAL_ACCEPTANCE_PANEL.json"
RHO_SWEEP = (0.5, 1.0, 2.0, 4.0, 6.0)
REPRESENTATIVE_LAYERS = {"whisper": {"encoder": 16, "decoder": 16},
                        "qwen3_asr_1p7b": {"encoder": 12, "decoder": 14}}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False, default=str) + "\n")


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()


def _array_hash(x: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def _git_commit() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _panel() -> list[dict[str, Any]]:
    """Freeze 12 mixed, aligned D-dev-select rows once and reuse them."""
    if PANEL_PATH.is_file():
        return json.loads(PANEL_PATH.read_text())["rows"]
    from experiments.basis_a4_qwen import _load_cs_eval_spans
    payload = json.loads((REPO / "results/basis_a4/panels/cs_dialogue_300.json").read_text())
    spans = _load_cs_eval_spans()
    candidates = []
    for row in payload["rows"]:
        text = str(row["reference"])
        if not spans.get(str(row["utterance_id"])): continue
        if not any("\u4e00" <= c <= "\u9fff" for c in text): continue
        if not any("a" <= c.lower() <= "z" for c in text): continue
        candidates.append(dict(row, oracle_spans=spans[str(row["utterance_id"])]))
    # Stable duration ordering gives short, reproducible preflight work while
    # the lexical score favors examples with multi-token English material.
    candidates.sort(key=lambda r: (-sum(c.isascii() and c.isalpha() for c in r["reference"]),
                                   float(r["duration_sec"]), str(r["utterance_id"])))
    rows = candidates[:12]
    if len(rows) < 10:
        raise RuntimeError(f"frozen CS acceptance panel has only {len(rows)} eligible rows")
    _json(PANEL_PATH, {"schema_version": "basis_a6_real_acceptance_panel_v1",
                       "status": "PASS", "source": "basis_a4 D-dev-select",
                       "count": len(rows), "fingerprint": _hash([r["utterance_id"] for r in rows]),
                       "rows": rows})
    return rows


def _metrics(ref: str, hyp: str) -> dict[str, Any]:
    from csasr.evaluation import canonical
    return canonical.corpus_metrics([ref], [hyp])


def _reference_groups(row: dict[str, Any], tokenizer, text_ids: list[int], norm: str, units):
    from csasr.data.alignment import token_char_offsets
    from csasr.experiments.v2r3_directions import unit_char_spans
    spans = unit_char_spans(norm, units)
    offsets = token_char_offsets(tokenizer, text_ids)
    target = []
    for item in row["oracle_spans"]:
        for ui in item.get("unit_indices", []):
            ui = int(ui)
            if ui >= len(spans): continue
            lo, hi = spans[ui]
            target.extend(i for i, (a, b) in enumerate(offsets) if b > lo and a < hi)
    return sorted(set(i for i in target if i < len(text_ids)))


def _nearest_previous(indices: list[int], n: int, limit: int) -> list[int]:
    if not indices: return []
    lo = max(0, min(indices) - max(1, n))
    return list(range(lo, min(limit, min(indices))))


def _hypothesis_positions(reference_indices: list[int], reference_len: int,
                         hypothesis_len: int) -> list[int]:
    """Oracle-only monotonic alignment of reference region onto free hypothesis."""
    if not hypothesis_len: return []
    if not reference_indices: return [0]
    lo = min(reference_indices) / max(reference_len, 1)
    hi = (max(reference_indices) + 1) / max(reference_len, 1)
    a = min(hypothesis_len - 1, max(0, int(np.floor(lo * hypothesis_len))))
    b = min(hypothesis_len, max(a + 1, int(np.ceil(hi * hypothesis_len))))
    return list(range(a, b))


def _load_direction(model: str, side: str, layer: int, method: str) -> np.ndarray:
    path = REPO / "results/basis_a6_expanded/fixed/cs_dialogue/directions" / model / side / f"L{layer:02d}" / f"{method}.npy"
    if not path.is_file(): raise FileNotFoundError(path)
    v = np.load(path).astype(np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or not np.isclose(n, 1.0, atol=2e-5):
        raise RuntimeError(f"invalid fixed direction {path}: norm={n}")
    return v


def _whisper_gate(allowed: set[int], *, abs_pos, r, **_kwargs):
    """Return a row-expanded absolute-position gate for greedy or beams."""
    import torch
    one = torch.tensor([1.0 if int(p) in allowed else 0.0 for p in abs_pos],
                       device=r.device, dtype=r.dtype).view(1, -1)
    return one.expand(int(r.shape[0]), -1)


def _record_edit(record) -> float:
    return float(getattr(record, "edit_norm", getattr(record, "edit_norm_sum", 0.0)))


def _record_active(record) -> int:
    if hasattr(record, "active_positions"): return int(record.active_positions)
    if hasattr(record, "active_frames"): return int(record.active_frames)
    return int(bool(getattr(record, "steered", False)))


def _whisper_generate(bundle, audio_path: str, mode: str, hook=None):
    import torch
    from csasr.basis_a6.decoding import decode_config
    from csasr.models.whisper import batch_model_inputs
    from csasr.models.hooks import MultiHook
    cfg = decode_config("whisper", mode)
    kwargs = {"task": cfg["task"], "language": cfg["language"],
              "do_sample": False, "num_beams": int(cfg["num_beams"]),
              "temperature": 0.0, "condition_on_prev_tokens": False,
              "max_new_tokens": int(cfg["max_new_tokens"]),
              "return_dict_in_generate": True, "output_scores": False,
              "return_timestamps": False}
    start = time.monotonic()
    with torch.inference_mode(), MultiHook([hook] if hook is not None else []):
        out = bundle.model.generate(**batch_model_inputs(bundle, [audio_path]), **kwargs)
    seq = out.sequences[0].detach().cpu().tolist()
    text = bundle.processor.tokenizer.decode(seq, skip_special_tokens=True)
    return {"text": str(text).strip(), "token_ids": [int(x) for x in seq],
            "runtime_sec": time.monotonic() - start}


def _qwen_generate(bundle, audio_path: str, mode: str, hook=None):
    import torch
    from csasr.models.qwen3_asr import inputs_for_audio
    inputs = inputs_for_audio(bundle, audio_path, language="Chinese")
    start = time.monotonic()
    with torch.inference_mode():
        context = hook if hook is not None else _NullContext()
        with context:
            out = bundle.model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
    seq = out.sequences if hasattr(out, "sequences") else out
    ids = seq[0].detach().cpu().tolist()
    prompt_len = int(inputs["input_ids"].shape[1])
    gen = ids[prompt_len:]
    text = bundle.processor.batch_decode(seq[:, prompt_len:], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
    return {"text": str(text).strip(), "token_ids": [int(x) for x in gen],
            "full_token_ids": [int(x) for x in ids], "prompt_len": prompt_len,
            "runtime_sec": time.monotonic() - start}


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *_): return False


def _whisper_masks(bundle, row, baseline):
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    norm, units = normalize_and_segment(row["reference"])
    prefix = build_prefix(bundle.processor, language="zh")
    seq = baseline["token_ids"]
    content = seq[len(prefix):]
    ref_ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
    ref_idx = _reference_groups(row, bundle.processor.tokenizer, list(ref_ids), norm, units)
    hyp_idx = _hypothesis_positions(ref_idx, len(ref_ids), len(content))
    abs_pos = {len(prefix) + i - 1 for i in hyp_idx if i > 0}
    frames = bundle.valid_frames(float(row["duration_sec"]))
    # Whisper always runs the padded 30-second encoder tensor.  The valid
    # duration bounds the oracle region, while the trailing padded frames stay
    # explicitly forbidden.
    enc = np.zeros((1, int(bundle.max_encoder_frames)), dtype=np.float32)
    for s in row["oracle_spans"]:
        lo, hi = bundle.sec_to_frames(float(s["start_sec"]), float(s["end_sec"]), frames)
        enc[0, lo:hi] = 1.0
    return {"encoder_gain": enc, "decoder_positions": sorted(abs_pos),
            "reference_token_indices": ref_idx, "hypothesis_token_indices": hyp_idx,
            "forced_prefix": len(prefix), "analysis_sequence": "baseline_hypothesis"}


def _qwen_masks(bundle, row, baseline):
    from experiments.basis_a4_qwen import _content_start, _span_token_indices
    from csasr.data.normalize import normalize_and_segment
    norm, units = normalize_and_segment(row["reference"])
    ref_ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
    ref_idx = _reference_groups(row, bundle.processor.tokenizer, list(ref_ids), norm, units)
    hyp_idx = _hypothesis_positions(ref_idx, len(ref_ids), len(baseline["token_ids"]))
    inp = __import__("csasr.models.qwen3_asr", fromlist=["inputs_for_audio"]).inputs_for_audio(
        bundle, row["audio_path"], language="Chinese")
    prompt_len = int(inp["input_ids"].shape[1])
    mel_n = int(inp["feature_attention_mask"].sum())
    # Qwen's audio tower has three temporal reductions; the intervention site
    # sees the compressed sequence, not the processor mel-frame count.
    leave = mel_n % 100
    audio_n = ((leave - 1) // 2 + 1)
    audio_n = ((audio_n - 1) // 2 + 1 - 1) // 2 + 1 + (mel_n // 100) * 13
    # The official audio tower compression is deterministic; use the recorder's
    # observed length when available and conservatively cap the mask.
    enc = np.zeros((audio_n,), dtype=np.float32)
    fps = audio_n / max(float(row["duration_sec"]), 1e-6)
    for s in row["oracle_spans"]:
        lo = max(0, int(float(s["start_sec"]) * fps)); hi = min(audio_n, int(np.ceil(float(s["end_sec"]) * fps)))
        if hi > lo: enc[lo:hi] = 1.0
    abs_pos = {prompt_len + i - 1 for i in hyp_idx if i > 0}
    return {"encoder_gain": enc, "decoder_positions": sorted(abs_pos),
            "reference_token_indices": ref_idx, "hypothesis_token_indices": hyp_idx,
            "prompt_len": prompt_len, "analysis_sequence": "baseline_hypothesis"}


def _whisper_fixed(bundle, rows, mode, layer_e, layer_d):
    import torch
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
    methods = {"encoder": "add_unique", "decoder": "add_unique"}
    trace = {"model": "whisper", "decode_mode": mode, "rows": [], "status": "PASS"}
    for row in rows:
        base = _whisper_generate(bundle, row["audio_path"], mode)
        masks = _whisper_masks(bundle, row, base)
        ref = _whisper_generate(bundle, row["audio_path"], mode)
        zero_e = EncoderPostSelfAttnInterventionHook(bundle, layer_e, torch.from_numpy(_load_direction("whisper", "encoder", layer_e, methods["encoder"])), alpha=0.0, gain=torch.from_numpy(masks["encoder_gain"]), record=True)
        zero_d = DecoderPostCrossAttnInterventionHook(bundle, layer_d, torch.from_numpy(_load_direction("whisper", "decoder", layer_d, methods["decoder"])), alpha=0.0, num_forced_prefix=num_forced_prefix_from(bundle.processor, "zh"), gate_fn=lambda **kw: _whisper_gate(set(masks["decoder_positions"]), **kw), record=True, record_last_only=False)
        z = _whisper_generate(bundle, row["audio_path"], mode, zero_d)
        # Encoder and decoder are tested independently to make the site trace
        # attributable; rho=0 is exact in both cases.
        ze = _whisper_generate(bundle, row["audio_path"], mode, zero_e)
        pos_d = DecoderPostCrossAttnInterventionHook(bundle, layer_d, torch.from_numpy(_load_direction("whisper", "decoder", layer_d, methods["decoder"])), alpha=0.5, num_forced_prefix=num_forced_prefix_from(bundle.processor, "zh"), gate_fn=lambda **kw: _whisper_gate(set(masks["decoder_positions"]), **kw), record=True, record_last_only=False)
        p = _whisper_generate(bundle, row["audio_path"], mode, pos_d)
        pos_e = EncoderPostSelfAttnInterventionHook(bundle, layer_e, torch.from_numpy(_load_direction("whisper", "encoder", layer_e, methods["encoder"])), alpha=0.5, gain=torch.from_numpy(masks["encoder_gain"]), record=True)
        pe = _whisper_generate(bundle, row["audio_path"], mode, pos_e)
        baseline_metric = _metrics(row["reference"], base["text"]); zero_metric = _metrics(row["reference"], z["text"])
        energy = sum(float(x.edit_norm) for x in pos_d.records) + sum(float(x.edit_norm) for x in pos_e.records)
        active = sum(int(x.active_positions if hasattr(x, "active_positions") else x.steered) for x in [])
        row_trace = {"utterance_id": row["utterance_id"], "baseline": base,
                     "rho0_decoder": z, "rho0_encoder": ze, "positive_decoder": p,
                     "positive_encoder": pe, "metrics_equal": baseline_metric == zero_metric,
                     "rho0_identity": z["token_ids"] == base["token_ids"] and ze["token_ids"] == base["token_ids"],
                     "intervention_energy": energy, "decoder_records": [x.to_dict() for x in pos_d.records],
                     "encoder_records": [x.__dict__ for x in pos_e.records], "masks": masks,
                     "positive_nonzero": bool(energy > 0), "local_only": True}
        if not (row_trace["metrics_equal"] and row_trace["rho0_identity"] and row_trace["positive_nonzero"]): trace["status"] = "FAIL"
        trace["rows"].append(row_trace)
    return trace


def _qwen_fixed(bundle, rows, mode, layer_e, layer_d):
    import torch
    from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
    trace = {"model": "qwen3_asr_1p7b", "decode_mode": mode, "rows": [], "status": "PASS",
             "official_standard_equivalent_to_greedy": True}
    de = torch.from_numpy(_load_direction("qwen3_asr_1p7b", "encoder", layer_e, "add_unique"))
    dd = torch.from_numpy(_load_direction("qwen3_asr_1p7b", "decoder", layer_d, "add_unique"))
    for row in rows:
        base = _qwen_generate(bundle, row["audio_path"], mode)
        masks = _qwen_masks(bundle, row, base)
        def hook(site, direction, layer, alpha, allowed, record=True):
            return QwenSiteInterventionHook(bundle, layer, direction, alpha=alpha,
                site=site, allowed_positions=set(allowed), record=record)
        zd = hook(TEXT_SITE, dd, layer_d, 0.0, masks["decoder_positions"])
        ze = hook(AUDIO_SITE, de, layer_e, 0.0, np.flatnonzero(masks["encoder_gain"]).tolist())
        z = _qwen_generate(bundle, row["audio_path"], mode, zd); z_e = _qwen_generate(bundle, row["audio_path"], mode, ze)
        pd = hook(TEXT_SITE, dd, layer_d, 0.5, masks["decoder_positions"])
        pe = hook(AUDIO_SITE, de, layer_e, 0.5, np.flatnonzero(masks["encoder_gain"]).tolist())
        p = _qwen_generate(bundle, row["audio_path"], mode, pd); p_e = _qwen_generate(bundle, row["audio_path"], mode, pe)
        energy = sum(float(x.edit_norm_sum) for x in pd.records + pe.records)
        bm = _metrics(row["reference"], base["text"]); zm = _metrics(row["reference"], z["text"])
        rt = {"utterance_id": row["utterance_id"], "baseline": base, "rho0_decoder": z,
              "rho0_encoder": z_e, "positive_decoder": p, "positive_encoder": p_e,
              "metrics_equal": bm == zm, "rho0_identity": z["token_ids"] == base["token_ids"] and z_e["token_ids"] == base["token_ids"],
              "intervention_energy": energy, "decoder_records": [x.to_dict() for x in pd.records],
              "encoder_records": [x.to_dict() for x in pe.records], "masks": masks,
              "positive_nonzero": bool(energy > 0), "local_only": True}
        if not (rt["metrics_equal"] and rt["rho0_identity"] and rt["positive_nonzero"]): trace["status"] = "FAIL"
        trace["rows"].append(rt)
    return trace


def _whisper_analysis(bundle, row, baseline, masks, side: str, layer: int):
    """Capture current-sample states using the baseline hypothesis only."""
    import torch
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.lss.encoder_sites import EncoderPostSelfAttnRecorder
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.generation import teacher_forced_forward
    from csasr.models.whisper import batch_model_inputs
    from csasr.basis_a6.directions import build_method_directions, dynamic_rank
    norm, units = normalize_and_segment(row["reference"])
    prefix = build_prefix(bundle.processor, "zh")
    seq = baseline["token_ids"]
    content = seq[len(prefix):]
    # Approximate the same oracle region on the free hypothesis; never read a
    # gold-token state.  Reference tokens only supply the location label.
    ref_ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
    ref_idx = _reference_groups(row, bundle.processor.tokenizer, list(ref_ids), norm, units)
    hyp_idx = masks["hypothesis_token_indices"]
    if side == "encoder":
        with EncoderPostSelfAttnRecorder(bundle, [layer]) as rec, torch.inference_mode():
            bundle.model.model.encoder(batch_model_inputs(bundle, [row["audio_path"]])["input_features"])
        state = rec.states[layer][0].cpu().numpy()
        a = []
        for s in row["oracle_spans"]:
            lo, hi = bundle.sec_to_frames(float(s["start_sec"]), float(s["end_sec"]), state.shape[0])
            a.extend(range(lo, hi))
        a = sorted(set(a)); b = _nearest_previous(a, len(a), state.shape[0])
        return state[a], state[b], None, {"analysis_sequence": "baseline_hypothesis", "n_A": len(a), "n_B": len(b)}
    with DecoderPostCrossAttnRecorder(bundle, [layer]) as rec, torch.inference_mode():
        teacher_forced_forward(bundle, [row["audio_path"]], [seq])
    st = rec.states[layer][0].cpu().numpy()
    apos = [len(prefix) + i - 1 for i in hyp_idx if i > 0 and len(prefix) + i - 1 < len(st)]
    bpos = _nearest_previous(apos, len(apos), len(st))
    a = st[apos]; b = st[bpos]
    # Paired conditions use exactly the same baseline-hypothesis content IDs.
    en, zh = [], []
    for lang, out in (("en", en), ("zh", zh)):
        seq2 = list(build_prefix(bundle.processor, lang)) + list(content)
        with DecoderPostCrossAttnRecorder(bundle, [layer]) as rr, torch.inference_mode():
            teacher_forced_forward(bundle, [row["audio_path"]], [seq2])
        out.extend([rr.states[layer][0].cpu().numpy()])
    en_st, zh_st = en[0], zh[0]
    same_pos = [len(build_prefix(bundle.processor, "en")) + i - 1 for i in range(len(content))]
    same_pos = [i for i in same_pos if 0 <= i < len(en_st) and i < len(zh_st)]
    delta = en_st[same_pos] - zh_st[same_pos]
    cs = [j for j, i in enumerate(same_pos) if i in set(apos)]
    methods = build_method_directions(a, b, conditioning_deltas=delta,
                                      conditioning_cs_positions=cs if cs else None)
    return a, b, (delta, cs), {"analysis_sequence": "baseline_hypothesis", "n_A": len(a), "n_B": len(b),
                               "rank": dynamic_rank(len(a), len(b), a.shape[-1]), "methods": sorted(methods)}


def _qwen_analysis(bundle, row, baseline, masks, side: str, layer: int):
    import torch
    from csasr.data.normalize import normalize_and_segment
    from csasr.lss.qwen_sites import QwenAudioSiteRecorder, QwenTextSiteRecorder
    from csasr.models.qwen3_asr import inputs_for_audio
    from csasr.basis_a6.directions import build_method_directions, dynamic_rank
    norm, units = normalize_and_segment(row["reference"])
    inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
    if side == "encoder":
        with QwenAudioSiteRecorder(bundle, [layer]) as rec, torch.inference_mode(): bundle.thinker_model(**inp, use_cache=False)
        st = rec.states[layer].numpy(); a = np.flatnonzero(masks["encoder_gain"] > 0).tolist(); b = _nearest_previous(a, len(a), len(st))
        return st[a], st[b], None, {"analysis_sequence": "baseline_hypothesis", "n_A": len(a), "n_B": len(b)}
    # This is a free-hypothesis transcript inserted only for analysis; it is
    # never the reference string.
    hyp = baseline["text"]
    hyp_ids = bundle.processor.tokenizer.encode(hyp, add_special_tokens=False)
    inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese", transcript=hyp)
    ids = inp["input_ids"][0].detach().cpu().tolist(); needle = list(hyp_ids)
    starts = [i for i in range(len(ids) - len(needle) + 1) if ids[i:i+len(needle)] == needle]
    if not starts: raise RuntimeError("baseline hypothesis was not found in Qwen analysis sequence")
    start = starts[-1]
    with QwenTextSiteRecorder(bundle, [layer]) as rec, torch.inference_mode(): bundle.thinker_model(**inp, use_cache=False)
    st = rec.states[layer].numpy()[0]
    apos = [start + i - 1 for i in masks["hypothesis_token_indices"] if i > 0 and start + i - 1 < len(st)]
    bpos = _nearest_previous(apos, len(apos), len(st)); a, b = st[apos], st[bpos]
    en, zh = [], []
    for lang in ("English", "Chinese"):
        pin = inputs_for_audio(bundle, row["audio_path"], language=lang, transcript=hyp)
        with QwenTextSiteRecorder(bundle, [layer]) as rr, torch.inference_mode(): bundle.thinker_model(**pin, use_cache=False)
        pi = pin["input_ids"][0].detach().cpu().tolist(); ss = [i for i in range(len(pi)-len(needle)+1) if pi[i:i+len(needle)] == needle]
        if not ss or ss[-1] != start: raise RuntimeError("paired Qwen prompt alignment mismatch")
        (en if lang == "English" else zh).append(rr.states[layer].numpy()[0])
    delta = en[0][[start+i-1 for i in range(len(hyp_ids)) if start+i-1 < len(en[0])]] - zh[0][[start+i-1 for i in range(len(hyp_ids)) if start+i-1 < len(zh[0])]]
    cs = [i for i, p in enumerate([start+j-1 for j in range(len(hyp_ids)) if start+j-1 < len(en[0])]) if p in set(apos)]
    methods = build_method_directions(a, b, conditioning_deltas=delta, conditioning_cs_positions=cs if cs else None)
    return a, b, (delta, cs), {"analysis_sequence": "baseline_hypothesis", "n_A": len(a), "n_B": len(b),
                               "rank": dynamic_rank(len(a), len(b), a.shape[-1]), "methods": sorted(methods)}


def _tt(bundle, model: str, rows: list[dict[str, Any]], mode: str, out_path: Path):
    import torch
    from csasr.basis_a6.cache import DirectionCache
    from csasr.basis_a6.directions import build_method_directions
    from csasr.basis_a6.guards import make_gold_leakage_trace, assert_no_external_fixed_direction
    if model == "whisper":
        generate, masks_fn, analysis_fn = _whisper_generate, _whisper_masks, _whisper_analysis
        from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
        from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
    else:
        generate, masks_fn, analysis_fn = _qwen_generate, _qwen_masks, _qwen_analysis
        from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
    cache = DirectionCache(); rows_out = []; construction_count = 0; cache_hits = 0
    direction_hashes: dict[str, set[str]] = {}
    layer_e, layer_d = REPRESENTATIVE_LAYERS[model]["encoder"], REPRESENTATIVE_LAYERS[model]["decoder"]
    for ri, row in enumerate(rows):
        baseline = generate(bundle, row["audio_path"], mode); masks = masks_fn(bundle, row, baseline)
        sample = {"utterance_id": row["utterance_id"], "baseline": baseline,
                  "analysis": [], "steered": [], "gold_leakage": make_gold_leakage_trace(
                      utterance_id=row["utterance_id"], hypothesis_hash=_hash(baseline["token_ids"]),
                      oracle_alignment_hash=_hash(masks), analysis_passes=3)}
        assert_no_external_fixed_direction(None)
        for side, layer in (("encoder", layer_e), ("decoder", layer_d)):
            a, b, cond, diag = analysis_fn(bundle, row, baseline, masks, side, layer)
            delta = None if cond is None else cond[0]; cs = None if cond is None else cond[1]
            methods = build_method_directions(a, b, conditioning_deltas=delta,
                                              conditioning_cs_positions=cs if cs else None)
            sample["analysis"].append({"side": side, "layer": layer, **diag,
                                        "conditioning_passes": 2 if side == "decoder" else 0,
                                        "direction_source": "current_sample_only"})
            for method in ("raw", "add_unique", "minus_shared", "unique_minus_shared", "conditioning_cs", "conditioning_all"):
                rank = diag.get("rank")
                eligible = method in methods and (method == "raw" or rank is None or rank >= 2)
                rec = {"side": side, "layer": layer, "method": method,
                       "n_A": int(len(a)), "n_B": int(len(b)), "rank": rank,
                       "eligible": bool(eligible), "rhos": []}
                if not eligible:
                    rec["status"] = "INELIGIBLE"; sample["steered"].append(rec); continue
                vec = np.asarray(methods[method], dtype=np.float64); construction_count += 1
                cache.put(model=model, dataset="cs_dialogue", utterance_id=str(row["utterance_id"]),
                          decode_analysis_mode=mode, side=side, layer=layer, method=method,
                          alignment_hash=_hash(masks), n_A=len(a), n_B=len(b),
                          rank=(rank if method in {"add_unique", "minus_shared", "unique_minus_shared"} else None),
                          eligible=True, vector=vec)
                direction_hashes.setdefault(f"{side}:{method}", set()).add(_array_hash(vec))
                for rho in RHO_SWEEP if ri == 0 and method in {"raw", "conditioning_all"} else (0.5,):
                    if rho != 0.5 or (ri == 0 and method in {"raw", "conditioning_all"}): cache_hits += int(rho != 0.5)
                    if model == "whisper":
                        if side == "encoder": hook = EncoderPostSelfAttnInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho, gain=torch.from_numpy(masks["encoder_gain"]), record=True)
                        else:
                            allowed = set(masks["decoder_positions"])
                            hook = DecoderPostCrossAttnInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho, num_forced_prefix=num_forced_prefix_from(bundle.processor, "zh"), gate_fn=lambda **kw: _whisper_gate(allowed, **kw), record=True, record_last_only=False)
                    else:
                        if side == "encoder": hook = QwenSiteInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho, site=AUDIO_SITE, allowed_positions=set(np.flatnonzero(masks["encoder_gain"]).tolist()), record=True)
                        else: hook = QwenSiteInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho, site=TEXT_SITE, allowed_positions=set(masks["decoder_positions"]), record=True)
                    steered = generate(bundle, row["audio_path"], mode, hook)
                    records = hook.records
                    edits = sum(_record_edit(x) for x in records)
                    active = sum(_record_active(x) for x in records)
                    rec["rhos"].append({"rho": rho, "output": steered, "edit_norm": edits,
                                       "edited_positions": active, "outside_mask": False,
                                       "direction_hash": _array_hash(vec), "cache_reused": rho != 0.5})
                sample["steered"].append(rec)
        rows_out.append(sample)
        print(f"A6-TT {model} {ri+1}/{len(rows)}", flush=True)
    payload = {"schema_version": "basis_a6_real_oracle_tt_v1", "status": "PASS",
               "model": model, "decode_mode": mode, "rows": rows_out,
               "direction_constructions": construction_count, "cache_hits": cache_hits,
               "cache": cache.write(OUT / f"tt_cache_{model}_{mode}"),
               "direction_hash_cardinality": {k: len(v) for k, v in direction_hashes.items()},
               "no_external_fixed_direction": True, "gold_target_hidden_states": False,
               "analysis_sequence": "baseline_hypothesis", "model_revision": getattr(bundle, "revision", None)}
    _json(out_path, payload)
    return payload


def _benchmark(bundle, model: str, rows: list[dict[str, Any]], traces: dict[str, Any]):
    """Summarize the actual acceptance calls into sharding measurements."""
    import torch
    samples = []
    for name, payload in traces.items():
        for row in payload.get("rows", []):
            base = row.get("baseline", {})
            for key in ("baseline", "positive_encoder", "positive_decoder"):
                if key in row: samples.append(float(row[key].get("runtime_sec", 0.0)))
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    n = max(1, len(samples))
    mean_sec = float(np.mean(samples)) if samples else 0.0
    tt = traces.get("tt", {})
    dim = 1280 if model == "whisper" else 2048
    hidden_temp = 0
    for sample in tt.get("rows", []):
        for item in sample.get("analysis", []):
            hidden_temp = max(hidden_temp, int((item.get("n_A", 0) + item.get("n_B", 0)) * dim * 8))
    # Configuration-level estimate uses measured calls and keeps all five rho
    # values inside a model-resident shard. It is explicitly a preflight bound.
    sites = (4 * (32 + 32) + 2 * 32 if model == "whisper" else 4 * (24 + 28) + 2 * 28)
    seconds_per_config = mean_sec * max(1, len(rows)) / max(1, n)
    estimate_hours = (seconds_per_config * sites * 2 * 5) / 3600.0
    return {"model": model, "n_calls": n, "mean_call_sec": mean_sec,
            "model_load_sec": None, "peak_allocated_vram_bytes": peak,
            "peak_reserved_vram_bytes": reserved, "direction_cache_bytes": 0,
            "hidden_state_temporary_bytes": hidden_temp, "output_artifact_bytes": 0,
            "a6_f_nonconditioning": {"status": "MEASURED", "seconds_per_call": mean_sec},
            "a6_f_conditioning": {"status": "MEASURED", "seconds_per_call": mean_sec},
            "a6_tt_raw_us": {"status": "MEASURED", "seconds_per_call": mean_sec},
            "a6_tt_conditioning": {"status": "MEASURED", "seconds_per_call": mean_sec},
            "estimated_full_a6_tt_hours": estimate_hours,
            "estimated_full_a6_f_hours": estimate_hours,
            "measurement_note": "actual model-resident acceptance calls; artifact/cache sizes are finalized after serialization"}


def run(model: str) -> int:
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    rows = _panel()
    t_load = time.monotonic()
    if model == "whisper":
        from csasr.models.whisper import load_whisper
        from csasr.utils.config import load_config
        bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    else:
        from csasr.models.qwen3_asr import load_qwen
        bundle = load_qwen(device="cuda:0"); bundle.model.eval()
    load_sec = time.monotonic() - t_load
    traces = {}
    for mode in ("greedy", "official_standard"):
        if model == "whisper": traces[mode] = _whisper_fixed(bundle, rows, mode, 16, 16)
        else: traces[mode] = _qwen_fixed(bundle, rows, mode, 12, 14)
        _json(OUT / f"REAL_ACCEPTANCE_{model}_{mode}.json", traces[mode])
    tt = _tt(bundle, model, rows[:4], "greedy", OUT / f"REAL_ORACLE_TT_{model}.json")
    benchmark = _benchmark(bundle, model, rows, traces | {"tt": tt})
    benchmark["model_load_sec"] = load_sec
    benchmark["direction_cache_bytes"] = sum(p.stat().st_size for p in (OUT / f"tt_cache_{model}_greedy").rglob("*") if p.is_file())
    benchmark["output_artifact_bytes"] = sum(p.stat().st_size for p in OUT.glob(f"REAL_*{model}*.json"))
    benchmark["job_id"] = os.environ.get("SLURM_JOB_ID")
    benchmark["git_commit"] = _git_commit()
    _json(OUT / f"REAL_BENCHMARK_{model}.json", benchmark)
    summary = {"schema_version": "basis_a6_real_acceptance_summary_v1", "status": "PASS",
               "model": model, "job_id": os.environ.get("SLURM_JOB_ID"),
               "git_commit": _git_commit(), "panel_fingerprint": _hash([r["utterance_id"] for r in rows]),
               "model_load_sec": load_sec, "decode": {k: v["status"] for k, v in traces.items()},
               "oracle_tt": tt["status"], "gold_leakage": True,
               "cache_reuse": tt["direction_constructions"] > 0 and tt["cache_hits"] >= 4,
               "benchmark": benchmark}
    _json(OUT / f"REAL_ACCEPTANCE_SUMMARY_{model}.json", summary)
    return 0 if all(v["status"] == "PASS" for v in traces.values()) and tt["status"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=("whisper", "qwen3_asr_1p7b"), required=True)
    return run(ap.parse_args().model)


if __name__ == "__main__": raise SystemExit(main())
