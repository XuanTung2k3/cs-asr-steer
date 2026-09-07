#!/usr/bin/env python
"""DG-03 construction runner: build & version the v6 steering basis at L16 and L24.

One GPU pass over D-construct (both layers via one recorder). Produces, per layer:
  v_raw   = dialogue-balanced mean of exact-site (r_E - r_M) contrasts (v6 order),
  v_cond  = normalize(dialogue-balanced mean of per-utterance mean(r(c_E) - r(c_M))),
  v_local = normalize(v_raw - <v_raw,v_cond> v_cond),
  V^0     = [v_local, v_cond],
and s_l = mean site-norm on D-construct (the pre-registered diagnostic scale).

Reuses frozen exact-site primitives; assembles via csasr.directions.steering_basis
in the v6 aggregate-then-residualize order. No legacy v_nat is reused. See
docs/current/DG03_BASIS_CAUSAL_SPEC.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import pandas as pd
import torch

LAYERS = (16, 24)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_fingerprint(cfg) -> dict:
    """sha256 of every D-construct input file actually consumed (candidates + roles + POI)."""
    from csasr.lss.align import conventions as conv
    root = Path(cfg["experiment"]["output_root"])
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    files = {"candidates_all.parquet": root / "alignments" / "candidates_all.parquet"}
    for role in conv.DEVELOPMENT_ROLES:
        files[f"role_{role}.parquet"] = role_root / f"role_{role}.parquet"
        files[f"poi_{role}.parquet"] = poi_root / f"poi_{role}.parquet"
    per_file = {name: f"sha256:{_file_sha256(p)}" for name, p in sorted(files.items()) if p.is_file()}
    composite = hashlib.sha256(
        json.dumps(per_file, sort_keys=True).encode()).hexdigest()
    return {"composite": f"sha256:{composite}", "files": per_file}


def _construction_config_hash(layers) -> str:
    """sha256 over the frozen construction configuration (FROZEN_CONFIG + DG-03 params)."""
    from csasr.experiments.v2r3_directions import FROZEN_CONFIG
    payload = {
        "frozen_config": FROZEN_CONFIG,
        "layers": [int(l) for l in layers],
        "control_offset": 0,
        "conditioning_prompts": {"c_E": "en", "c_M": "zh"},
        "aggregation": "dialogue_balanced_mean",
        "order": "aggregate_then_residualize",
        "rho": 1.0,
        "schema_version": "steering_basis_v1",
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        from csasr.utils.logging import git_state
        return (git_state() or {}).get("commit")
    except Exception:
        try:
            import subprocess
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
        except Exception:
            return None


def _dialogue_balanced_mean(rows: np.ndarray, groups) -> np.ndarray:
    from csasr.experiments.v2r3_directions import dialogue_balanced_mean
    direction, _ = dialogue_balanced_mean(np.asarray(rows, dtype=float), pd.Series(list(groups)))
    return np.asarray(direction, dtype=np.float64)


def _load_construct():
    """Mirror v2r3_directions.main's D-construct loading -> (manifest, construct_spans)."""
    from csasr.utils.config import load_config
    from csasr.experiments.v2r3_directions import build_spans, attach_baseline_status
    from csasr.lss.align import conventions as conv

    cfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    candidates, _ = conv.read_development_candidates(candidate_path)
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    manifests, pois = [], []
    for role in conv.DEVELOPMENT_ROLES:
        manifests.append(pd.read_parquet(role_root / f"role_{role}.parquet"))
        pois.append(pd.read_parquet(poi_root / f"poi_{role}.parquet").rename(
            columns={"poi_index": "reference_unit_index"}))
    manifest = pd.concat(manifests, ignore_index=True)
    poi = pd.concat(pois, ignore_index=True)
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))
    spans = build_spans(candidates, dialogue_of=dialogue_of)
    spans = attach_baseline_status(spans, poi)
    construct = spans[(spans["role"] == "D-construct") & spans["all_correct"]].copy()
    return cfg, manifest, construct


def _conditioning_and_scale(bundle, manifest, construct, layers, *, limit=None, log=print):
    """Per-utterance mean(r(c_E) - r(c_M)) over content positions, and mean site-norm s_l."""
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.models.generation import teacher_forced_forward
    from csasr.lss.sites import DecoderPostCrossAttnRecorder

    tok = bundle.processor.tokenizer
    eot = tok.eos_token_id
    prefix_en = build_prefix(bundle.processor, language="en")
    prefix_zh = build_prefix(bundle.processor, language="zh")
    assert len(prefix_en) == len(prefix_zh)
    plen = len(prefix_zh)

    uids = list(dict.fromkeys(str(u) for u in construct["utterance_id"]))
    if limit:
        uids = uids[:limit]
    mrows = manifest.set_index(manifest["utterance_id"].astype(str))
    dial_of = {str(u): str(d) for u, d in zip(construct["utterance_id"], construct["dialogue_id"])}

    cond = {int(l): [] for l in layers}
    cond_groups = []
    norm_sum = {int(l): 0.0 for l in layers}
    norm_cnt = 0
    for uid in uids:
        row = mrows.loc[uid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        norm, _units = normalize_and_segment(row["transcript_raw"])
        text_ids = tok.encode(norm, add_special_tokens=False)[:220]
        if not text_ids:
            continue
        seq_en = list(prefix_en) + list(text_ids) + [eot]
        seq_zh = list(prefix_zh) + list(text_ids) + [eot]
        content = slice(plen, plen + len(text_ids))
        with DecoderPostCrossAttnRecorder(bundle, list(layers)) as rec, torch.inference_mode():
            teacher_forced_forward(bundle, [row["audio_path"]], [seq_en])
            st_en = {int(l): rec.states[int(l)][0].float().cpu().numpy() for l in layers}
        with DecoderPostCrossAttnRecorder(bundle, list(layers)) as rec, torch.inference_mode():
            teacher_forced_forward(bundle, [row["audio_path"]], [seq_zh])
            st_zh = {int(l): rec.states[int(l)][0].float().cpu().numpy() for l in layers}
        for l in layers:
            l = int(l)
            cond[l].append(st_en[l][content].mean(axis=0) - st_zh[l][content].mean(axis=0))
            norm_sum[l] += float(np.linalg.norm(st_zh[l][content], axis=1).sum())
        cond_groups.append(dial_of.get(uid, uid))
        norm_cnt += len(text_ids)
        if len(cond_groups) % 50 == 0:
            log(f"  conditioning {len(cond_groups)}/{len(uids)}")
    s_l = {int(l): (norm_sum[int(l)] / max(norm_cnt, 1)) for l in layers}
    return cond, cond_groups, s_l


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/dg03/basis")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="cap construction utterances (smoke)")
    args = p.parse_args(argv)

    def log(m): print(m, flush=True)

    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.experiments.v2r3_directions import site_d_contrasts
    from csasr.directions.steering_basis import build_basis, write_basis_artifact, diagnostic_dose

    cfg, manifest, construct = _load_construct()
    if args.limit:
        keep = set(list(dict.fromkeys(str(u) for u in construct["utterance_id"]))[:args.limit])
        construct = construct[construct["utterance_id"].astype(str).isin(keep)].copy()
    log(f"D-construct baseline-correct embedded spans: {len(construct)} "
        f"over {construct['utterance_id'].nunique()} utterances, "
        f"{construct['dialogue_id'].nunique()} dialogues")

    mcfg = load_config("model/whisper_large_v3.yaml") if Path(
        "configs/model/whisper_large_v3.yaml").exists() else cfg
    bundle = load_whisper(mcfg if "model" in mcfg else cfg)
    bundle.model.eval()

    # (1) raw language contrasts (E - M) at both layers, one pass.
    meta, extra = site_d_contrasts(bundle, manifest, construct, list(LAYERS),
                                   batch_size=args.batch_size, log=None)
    log(f"site_d_contrasts: {len(meta)} spans, skipped={extra['skipped']}")

    # (2) conditioning contrasts + mean site-norm s_l.
    cond, cond_groups, s_l = _conditioning_and_scale(
        bundle, manifest, construct, LAYERS, limit=args.limit, log=log)

    commit = _git_commit()
    dataset_fp = _dataset_fingerprint(cfg)
    config_hash = _construction_config_hash(LAYERS)
    out_dir = Path(args.output_dir)
    summary = {"layers": {}, "git_commit": commit,
               "model": bundle.metadata(), "dataset_role": "D-construct",
               "dataset_fingerprint": dataset_fp, "construction_config_hash": config_hash}
    for l in LAYERS:
        l = int(l)
        contrasts_EM = np.stack([np.asarray(v, dtype=np.float64) for v in extra["vectors"][l]])
        groups_raw = list(meta["dialogue_id"])
        cond_contrasts = np.stack(cond[l])
        # v6 aggregate-then-residualize, with dialogue-balanced aggregation.
        from csasr.directions.steering_basis import (
            raw_language_contrast, language_conditioning, conditioning_residualized_local)
        v_raw = raw_language_contrast(contrasts_EM, groups_raw)
        v_cond = language_conditioning(cond_contrasts, cond_groups)
        v_local = conditioning_residualized_local(v_raw, v_cond)
        # Package through build_basis for validation/metrics (feed the already-aggregated
        # vectors as single-row inputs so the balanced mean is the identity).
        basis = build_basis(layer=l, contrasts_EM=v_raw[None, :],
                            cond_contrasts=v_cond[None, :])
        # build_basis re-normalizes v_cond (already unit) and recomputes v_local from v_raw;
        # assert it matches the explicit v6 computation.
        assert np.allclose(basis["v_local"], v_local, atol=1e-8)
        dose = diagnostic_dose(scale=s_l[l], rho=1.0)
        prov = {
            "model_id": bundle.model_id, "model_revision": bundle.revision,
            "git_commit": commit, "site": "decoder_post_cross_attn_residual",
            "dataset_role": "D-construct",
            "dataset_fingerprint": dataset_fp,
            "construction_config_hash": config_hash,
            "n_embedded_positions": int(len(meta)),
            "n_matrix_positions": int(len(meta)),
            "n_conditioning_pairs": int(len(cond_groups)),
            "conditioning_prompts": {"c_E": "build_prefix(language='en')",
                                      "c_M": "build_prefix(language='zh')"},
            "mean_site_norm_s_l": float(s_l[l]),
            "diagnostic_dose": dose,
            "seeds": {"random_controls": [0, 1, 2]},
            "reused_cached_activations": False,
            "aggregation": "dialogue_balanced_mean",
        }
        rec = write_basis_artifact(out_dir, basis, prov)
        summary["layers"][str(l)] = {
            "artifact": rec["_json_path"], "tensor_hashes": rec["tensor_hashes"],
            "metrics": basis["metrics"], "mean_site_norm_s_l": float(s_l[l]),
            "diagnostic_dose": dose}
        log(f"L{l}: cos(raw,cond)={basis['metrics']['cos_raw_cond']:.4f} "
            f"cos(local,cond)={basis['metrics']['cos_local_cond']:.2e} "
            f"removed_frac={basis['metrics']['conditioning_fraction_removed']:.4f} "
            f"s_l={s_l[l]:.3f}")
    out = out_dir / "dg03_basis_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
