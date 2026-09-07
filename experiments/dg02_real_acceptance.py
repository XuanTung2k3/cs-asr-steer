#!/usr/bin/env python
"""DG-02 real-Whisper acceptance check (Stage-3 Phase D).

One-utterance integration check for the exact post-cross-attention / pre-FFN
intervention site. It is **not** an experiment: it validates plumbing only and
makes no ASR-quality judgement, no layer selection, no direction/beta choice.

Run on a GPU node (or any machine that can load whisper-large-v3 -- ~3 GB; the
2 GB CPU dev box cannot). The CPU/synthetic matrix (``tests/test_dg02_site.py``)
must already be green; this script closes the real-model gates:

  D5  beta=0 decoded token IDs + transcript identical to baseline  (L16 & L24)
  D6  r = q + u_source on real states (full-sequence and a cached step)
  D7  exact-site runtime evidence (assert_site_reconstruction)
  D8  forced-prefix positions get zero edit; an eligible position can be edited
  D9  cached logical positions advance and never reset; eligibility aligned
  D10 norm preservation under a tiny non-zero test edit

Usage (single utterance, both active layers):

    python experiments/dg02_real_acceptance.py \
        --data-config configs/lss/l1b_candidates_dialogue_v2r3.yaml \
        --output results/dg02_real_acceptance.json

Writes a JSON report and prints ``DG-02 REAL ACCEPTANCE: PASS|FAIL``. Exit code
0 only on PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import torch

# The two active scientific candidates (MC §3). DG-02 exercises the plumbing at
# both; it does NOT compare them or pick a winner (that is DG-03).
LAYERS = (16, 24)
# A tiny fixed non-zero probe edit. Its only job is to prove the eligible path is
# live (D8); it is never tuned and never judged for ASR quality.
PROBE_ALPHA = 0.05
GEN_KWARGS = dict(task="transcribe", language="zh", do_sample=False, num_beams=1,
                  temperature=0.0, max_new_tokens=220, condition_on_prev_tokens=False)


def _ulp_tol(bundle, layer, base_states, floor=1e-3, ulp=8.0):
    """A dtype-justified tolerance (mirrors assert_site_reconstruction)."""
    try:
        dtype = next(bundle.decoder_layer(layer).parameters()).dtype
    except (StopIteration, AttributeError):
        dtype = getattr(bundle, "dtype", base_states.dtype)
    eps = float(torch.finfo(dtype).eps)
    scale = float(base_states.abs().max()) or 1.0
    return max(float(floor), float(ulp) * eps) * scale, eps, str(dtype)


def _check_layer(bundle, layer, features, attention_mask, num_forced_prefix, log):
    from csasr.lss.sites import (DecoderPostCrossAttnInterventionHook,
                                 DecoderPostCrossAttnRecorder,
                                 assert_no_site_hooks, assert_site_reconstruction)

    rep: dict = {"layer": int(layer)}
    feats = features
    model = bundle.model
    gen_in = dict(input_features=feats, attention_mask=attention_mask)

    # ---- D4 baseline decoding (no intervention) ---------------------------
    with torch.inference_mode():
        base = model.generate(**gen_in, **GEN_KWARGS)
    base_seq = (base if hasattr(base, "shape") else base.sequences).detach().cpu()
    base_txt = bundle.processor.batch_decode(base_seq, skip_special_tokens=True)[0].strip()
    rep["baseline_num_tokens"] = int(base_seq.shape[1])

    # ---- D5 beta=0 decoding must be token- and transcript-identical -------
    direction = torch.zeros(bundle.d_model, dtype=torch.float32)
    direction[0] = 1.0
    b0 = DecoderPostCrossAttnInterventionHook(
        bundle, layer, direction, alpha=0.0, num_forced_prefix=num_forced_prefix,
        enforce_contract_layer=True)
    with b0, torch.inference_mode():
        z = model.generate(**gen_in, **GEN_KWARGS)
    assert_no_site_hooks(bundle)
    z_seq = (z if hasattr(z, "shape") else z.sequences).detach().cpu()
    z_txt = bundle.processor.batch_decode(z_seq, skip_special_tokens=True)[0].strip()
    rep["beta0_token_identity"] = bool(z_seq.shape == base_seq.shape
                                       and torch.equal(z_seq, base_seq))
    rep["beta0_transcript_identity"] = bool(z_txt == base_txt)
    rep["beta0_steered_calls"] = int(b0.steered_calls)

    # ---- D6 r = q + u_source on real states -------------------------------
    # (a) initial / full-sequence pass over the decoded ids.
    tf_ids = base_seq.to(bundle.device)
    with DecoderPostCrossAttnRecorder(bundle, [layer]) as rec, torch.inference_mode():
        model(input_features=feats, attention_mask=attention_mask,
              decoder_input_ids=tf_ids, use_cache=False)
    q, u, r = rec.q_states[layer], rec.u_source_states[layer], rec.states[layer]
    full_err = float((r - (q + u)).abs().max())
    tol, eps, dtype = _ulp_tol(bundle, layer, r)
    # (b) a cached step: the recorder overwrites per call, so after a cached
    #     generate it holds the last (cached) step's q/u/r.
    with DecoderPostCrossAttnRecorder(bundle, [layer]) as rec2, torch.inference_mode():
        model.generate(**gen_in, **{**GEN_KWARGS, "max_new_tokens": 8})
    qc, uc, rc = rec2.q_states[layer], rec2.u_source_states[layer], rec2.states[layer]
    cached_err = float((rc - (qc + uc)).abs().max())
    rep["qur_dtype"] = dtype
    rep["qur_full_seq_max_abs_err"] = full_err
    rep["qur_cached_step_max_abs_err"] = cached_err
    rep["qur_tol"] = tol
    rep["qur_identity"] = bool(full_err <= tol and cached_err <= tol)

    # ---- D7 exact-site runtime evidence -----------------------------------
    def _forward():
        with torch.inference_mode():
            model(input_features=feats, attention_mask=attention_mask,
                  decoder_input_ids=tf_ids, use_cache=False)
    site = assert_site_reconstruction(bundle, _forward, layer)
    rep["exact_site_reconstruction_ok"] = bool(site["reconstruction_ok"])
    rep["exact_site_differs_from_block_output"] = bool(site["site_differs_from_block_output"])
    rep["exact_site_rel_err_ulp"] = float(site["rel_err_ulp"])
    rep["exact_site_err_vs_block_gap"] = float(site["err_vs_block_gap"])

    # ---- D8/D9/D10 tiny non-zero probe edit through real cached generation -
    probe = DecoderPostCrossAttnInterventionHook(
        bundle, layer, direction, alpha=PROBE_ALPHA,
        num_forced_prefix=num_forced_prefix, norm_preserve=True,
        record=True, record_last_only=False, enforce_contract_layer=True)
    with probe, torch.inference_mode():
        model.generate(**gen_in, **{**GEN_KWARGS, "max_new_tokens": 12})
    assert_no_site_hooks(bundle)
    recs = probe.records
    prefix_recs = [x for x in recs if x.is_forced_prefix]
    eligible_recs = [x for x in recs if not x.is_forced_prefix]
    steered_recs = [x for x in eligible_recs if x.edit_norm > 0.0]
    # D8
    rep["prefix_positions_zero_edit"] = bool(prefix_recs
                                             and all(x.edit_norm == 0.0 for x in prefix_recs))
    rep["eligible_position_edited"] = bool(len(steered_recs) > 0)
    # D9: positions advance, never reset, prefix boundary aligned
    positions = sorted({x.abs_pos for x in recs})
    rep["cache_positions"] = positions[:16]
    rep["cache_positions_monotonic_no_reset"] = bool(positions == sorted(set(positions))
                                                      and len(positions) > num_forced_prefix)
    rep["prefix_boundary_aligned"] = bool(
        all(x.is_forced_prefix for x in recs if x.abs_pos < num_forced_prefix)
        and all(not x.is_forced_prefix for x in recs if x.abs_pos >= num_forced_prefix))
    # D10: norm preservation under the probe edit
    if steered_recs:
        dev = max(abs(x.post_norm - x.pre_norm) / (x.pre_norm or 1.0) for x in steered_recs)
    else:
        dev = float("inf")
    rep["norm_preservation_max_rel_dev"] = dev
    rep["norm_preservation_ok"] = bool(dev <= 1e-2)

    gates = {
        "D5_beta0_token_identity": rep["beta0_token_identity"],
        "D5_beta0_transcript_identity": rep["beta0_transcript_identity"],
        "D6_qur_identity": rep["qur_identity"],
        "D7_exact_site": rep["exact_site_reconstruction_ok"]
                         and rep["exact_site_differs_from_block_output"],
        "D8_prefix_zero_edit": rep["prefix_positions_zero_edit"],
        "D8_eligible_edited": rep["eligible_position_edited"],
        "D9_cache_positions": rep["cache_positions_monotonic_no_reset"]
                              and rep["prefix_boundary_aligned"],
        "D10_norm_preservation": rep["norm_preservation_ok"],
    }
    rep["gates"] = gates
    rep["layer_pass"] = all(gates.values())
    log(f"layer {layer}: " + ("PASS" if rep["layer_pass"] else "FAIL")
        + " " + json.dumps(gates))
    return rep


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-config",
                   default="configs/lss/l1b_candidates_dialogue_v2r3.yaml")
    p.add_argument("--role", default="D-dev-select",
                   help="data role for the single debug utterance (integration/debug)")
    p.add_argument("--output", default="results/dg02_real_acceptance.json")
    args = p.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper, batch_model_inputs
    from csasr.lss.sites import num_forced_prefix_from
    from steer_sweep import data as D

    cfg = load_config(args.data_config)
    bundle = load_whisper(cfg)
    if bundle.model.training:
        bundle.model.eval()

    # One shortest development/debug utterance -> fast, minimal exposure.
    pop = D.build_population(bundle, cfg, args.role, assert_anchors=True)
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    row = manifest.iloc[0]
    uid = str(row.utterance_id)
    inputs = batch_model_inputs(bundle, [row.audio_path])
    features = inputs["input_features"]
    attention_mask = inputs.get("attention_mask")

    num_forced_prefix = num_forced_prefix_from(bundle.processor,
                                               language="zh", task="transcribe")
    log(f"utterance={uid} role={args.role} duration_sec={float(row.duration_sec):.3f} "
        f"num_forced_prefix={num_forced_prefix}")

    report = {
        "dataset": cfg.get("data", {}).get("dataset") if isinstance(cfg, dict) else None,
        "data_config": args.data_config,
        "data_role": args.role,
        "utterance_id": uid,
        "duration_sec": float(row.duration_sec),
        "num_forced_prefix": int(num_forced_prefix),
        "model": bundle.metadata(),
        "layers": {},
    }
    for layer in LAYERS:
        report["layers"][str(layer)] = _check_layer(
            bundle, layer, features, attention_mask, num_forced_prefix, log)

    passed = all(report["layers"][str(k)]["layer_pass"] for k in LAYERS)
    report["verdict"] = "PASS" if passed else "FAIL"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote {out}")
    log(f"DG-02 REAL ACCEPTANCE: {report['verdict']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
