"""Day 6: invariance battery for the Day 4 headline.

The paper cannot claim boundary accuracy -- the corpus has no gold word timings
-- so the alignment evidence is invariance instead: the headline must survive the
preprocessing choices that could have manufactured it.

Headline rerun in every cell: Site D, layer 16, rho=1, teacher-forced,
conservative 400 ms subset, substitution-only stratum (G=19), paired contrast
against the label-permuted null.

Factors: boundary jitter (reported separately for English spans and Mandarin
controls), all four CTC blank conventions, both aligner paths, three pooling
schemes -- and, first, inverse-variance weighting of the direction construction.

There are **two** aligner paths, not three. The cross-attention path is a
documented negative result (`docs/V2R3_CROSS_ATTENTION_SPANS_2026-08-15.md`) and
is not a cell here. That makes `whisper_dtw` one of two rather than one of three,
so its weight in the invariance argument is correspondingly larger even though
its 17.4% invalid units are entirely Mandarin and cost 0 units in this subset.

This module selects nothing and optimises nothing. Weak cells are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging
from . import v2r3_directions as D3
from . import v2r3_oracle_screen as OS

STAGE = "v2r3_invariance"

HEADLINE = {"site": "D", "decoder_layer": 16, "rho": 1.0,
            "stratum": "substitution_only", "floor_ms": 400.0,
            "contrast": "correct vs label-permuted null", "G_expected": 19}

JITTER_MS = (50.0, 100.0, 200.0)
CONVENTIONS = conv.CONVENTIONS
ALIGNER_PATHS = ("existing_ctc", "whisper_dtw")
POOLINGS = D3.POOLING_VARIANTS

FROZEN_CONFIG: dict[str, Any] = {
    "headline": HEADLINE,
    "jitter_ms": list(JITTER_MS),
    "jitter_reported_separately_for": ["EN spans", "ZH control spans"],
    "conventions": list(CONVENTIONS),
    "aligner_paths": list(ALIGNER_PATHS),
    "aligner_path_note": ("two paths, not three; the cross-attention path is a "
                          "documented negative result and is not a cell"),
    "poolings": list(POOLINGS),
    "inverse_variance_weighting": True,
    "selects_anything": False,
    "evaluates_gate": False,
}


def inverse_variance_weights(contrasts: np.ndarray) -> np.ndarray:
    """Weight each pair by the inverse of its own squared norm.

    Site E's construction showed corr(||d_i||, duration) = -0.382: short spans
    give large contrast norms, so an unweighted mean lets short spans dominate.
    Inverse-variance weighting removes that leverage without normalising the
    contrasts themselves away.
    """
    y = np.asarray(contrasts, dtype=float)
    variance = np.maximum((y ** 2).sum(axis=1), 1e-12)
    weights = 1.0 / variance
    return weights / weights.sum()


def assemble_inverse_variance(contrasts: np.ndarray, meta: pd.DataFrame, *,
                              basis: np.ndarray | None = None) -> dict[str, Any]:
    """The Day 3 pipeline with inverse-variance instead of equal pair weights.

    Everything else is identical -- prompt projection, ridge residualisation,
    norm clipping, dialogue balancing -- so the only difference between this and
    the published direction is how much each pair contributes.
    """
    raw = np.asarray(contrasts, dtype=float)
    projected, projection_energy = (D3.project_out(raw, basis) if basis is not None
                                    else (raw, 0.0))
    residual, ridge = D3.residualise(projected, D3.nuisance_matrix(meta))
    clipped, clipping = D3.clip_norms(residual)

    pair_weights = inverse_variance_weights(clipped)
    labels = meta["dialogue_id"].astype(str).to_numpy()
    combined = np.zeros(len(clipped), dtype=float)
    for dialogue in sorted(set(labels)):
        mask = labels == dialogue
        share = pair_weights[mask]
        total = share.sum()
        combined[mask] = (share / total if total > 0 else
                          np.full(int(mask.sum()), 1.0 / max(int(mask.sum()), 1)))
        combined[mask] /= len(set(labels))
    direction = combined @ clipped

    norms = np.linalg.norm(raw, axis=1)
    duration = pd.to_numeric(meta["span_duration_sec"], errors="coerce").to_numpy()
    ok = np.isfinite(norms) & np.isfinite(duration)
    reweighted_norms = np.linalg.norm(clipped * combined[:, None] * len(clipped), axis=1)
    return {
        "direction": direction,
        "pairs": int(len(raw)),
        "dialogues": int(len(set(labels))),
        "ridge_energy_removed": ridge["energy_fraction_removed"],
        "prompt_projection_energy_removed": projection_energy,
        "clipping": clipping,
        "corr_norm_duration_before": float(np.corrcoef(norms[ok], duration[ok])[0, 1]),
        "corr_norm_duration_after": float(
            np.corrcoef(reweighted_norms[ok], duration[ok])[0, 1]),
        "weight_ratio_max_min": float(pair_weights.max() / max(pair_weights.min(), 1e-12)),
    }


def jitter_positions(positions: dict[str, list[int]], shift_steps: int
                     ) -> dict[str, list[int]]:
    """Shift every decoder target position by a fixed number of steps."""
    return {u: [max(0, int(p) + int(shift_steps)) for p in v]
            for u, v in positions.items()}


def step_shift_for(jitter_ms: float, n_tokens: int, duration_sec: float) -> int:
    """Convert a boundary jitter in ms to a decoder-step displacement.

    A Site-D target is the decoder step predicting the span's first token, and
    that step is derived from reference-unit indices -- so shifting the span's
    *times* never reaches it. The operationally meaningful perturbation is
    "the localizer is off by X ms", which for a decoder site means landing this
    many steps early or late, at the utterance's own token rate.
    """
    if duration_sec <= 0 or n_tokens <= 0:
        return 0
    tokens_per_second = float(n_tokens) / float(duration_sec)
    return int(round(float(jitter_ms) / 1000.0 * tokens_per_second))


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


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _spans_for(candidates, dialogue_of, poi, *, convention: str, family: str,
               floor_ms: float = 400.0):
    """Eligible spans under one convention/family, optionally jittered."""
    from .v2r3_headroom_diagnostic import eligible_spans, structurally_eligible

    adjusted = conv.apply_convention(candidates, convention, family=family)
    spans = eligible_spans(adjusted, family=family)
    spans = spans[structurally_eligible(spans)].copy()
    frame = adjusted[adjusted["aligner_family"].astype(str) == family]
    edges = {}
    for (utterance, index), group in frame.groupby(["utterance_id", "reference_unit_index"]):
        edges[(str(utterance), int(index))] = (
            float(pd.to_numeric(group["start_sec"], errors="coerce").min()),
            float(pd.to_numeric(group["end_sec"], errors="coerce").max()))
    starts, ends = [], []
    for utterance, indices in zip(spans["utterance_id"], spans["unit_indices"]):
        pairs = [edges[(str(utterance), int(i))] for i in indices
                 if (str(utterance), int(i)) in edges]
        if not pairs:
            starts.append(float("nan")); ends.append(float("nan")); continue
        starts.append(min(p[0] for p in pairs))
        ends.append(max(p[1] for p in pairs))
    spans["start_sec"], spans["end_sec"] = starts, ends
    spans["span_duration_sec"] = spans["end_sec"] - spans["start_sec"]
    spans["span_duration_ms"] = spans["span_duration_sec"] * 1000.0
    spans["dialogue_id"] = [str(dialogue_of.get(str(c), ""))
                            for c in spans["conversation_id"]]
    spans = spans[spans["start_sec"].notna()
                  & (spans["span_duration_ms"] >= float(floor_ms))].reset_index(drop=True)
    return D3.attach_baseline_status(spans, poi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None,
                        help="run only cells whose 'factor:detail' matches this "
                             "value; used to repair a single defective cell "
                             "without re-running the battery")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target_dir = resolve_output(args.output_dir)
    directions_root = Path(args.directions)
    if "smoke" in directions_root.name:
        raise SystemExit(f"refusing a smoke generation: {directions_root}")
    npz_path = directions_root / "directions.npz"
    side = manifest_mod.load(npz_path)
    fmt, _ = manifest_mod._identity_format(dict(side.get("identity") or {}))
    if not manifest_mod.verify(npz_path, cfg=None, require_identity=False)["ok"]:
        raise SystemExit("directions failed byte verification")
    directions = dict(np.load(npz_path))

    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    for path in [candidate_path] + [role_root / f"role_{r}.parquet" for r in conv.DEVELOPMENT_ROLES] \
            + [poi_root / f"poi_{r}.parquet" for r in conv.DEVELOPMENT_ROLES]:
        if not manifest_mod.verify(path, cfg=None, require_identity=False)["ok"]:
            raise SystemExit(f"{path.name} failed byte verification")
    candidates, verdict = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)
    manifest = pd.concat([pd.read_parquet(role_root / f"role_{r}.parquet")
                          for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    poi = pd.concat([pd.read_parquet(poi_root / f"poi_{r}.parquet").rename(
        columns={"poi_index": "reference_unit_index"})
        for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))

    plan = {"frozen_configuration": FROZEN_CONFIG,
            "directions_identity": {"classification": fmt, "identity_required": False,
                                    "byte_verified_sha256": side["sha256"]},
            "destination": str(target_dir)}
    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan}, indent=2, default=str))
        return 0

    log = setup_logging(level="INFO")
    bundle = load_whisper(cfg)
    layer, rho = int(HEADLINE["decoder_layer"]), float(HEADLINE["rho"])

    def selected(cell: dict[str, Any]) -> bool:
        """--only filters cells by 'factor:detail'; without it, everything runs."""
        if not args.only:
            return True
        return f"{cell['factor']}:{cell['detail']}" == args.only

    def run_cell(spans, direction, nulls, *, cell: dict[str, Any],
                 jitter_ms: float = 0.0) -> list[pd.DataFrame]:
        if not selected(cell):
            return []
        develop = spans[spans["role"] == "D-dev-select"]
        targets = OS.build_targets(bundle, candidates, develop, poi, manifest)
        if args.limit:
            keep = list(dict.fromkeys(targets["utterance_id"]))[: int(args.limit)]
            targets = targets[targets["utterance_id"].isin(keep)]
        if not len(targets):
            return []
        utts = manifest[manifest["utterance_id"].isin(set(targets["utterance_id"]))]
        utts = utts.drop_duplicates("utterance_id").reset_index(drop=True)
        positions = {str(u): [int(v) for v in g["query_position"]]
                     for u, g in targets.groupby("utterance_id")}
        if jitter_ms:
            shifted = {}
            for utterance, where in positions.items():
                row = manifest.loc[manifest["utterance_id"] == utterance]
                duration = float(row["duration_sec"].iloc[0]) if len(row) else 0.0
                steps = step_shift_for(jitter_ms, max(where) + 1 if where else 0,
                                       duration)
                shifted[utterance] = [max(0, int(p) + steps) for p in where]
            positions = shifted
            steps_all = [
                step_shift_for(jitter_ms, max(w) + 1 if w else 0,
                               float(manifest.loc[manifest["utterance_id"] == u,
                                                  "duration_sec"].iloc[0]))
                for u, w in positions.items()]
            cell = {**cell,
                    "frac_utterances_shifted": float(np.mean(
                        [1.0 if v else 0.0 for v in steps_all])) if steps_all else 0.0,
                    "mean_step_shift": float(np.mean([
                step_shift_for(jitter_ms, max(w) + 1 if w else 0,
                               float(manifest.loc[manifest["utterance_id"] == u,
                                                  "duration_sec"].iloc[0]))
                for u, w in positions.items()]))}
        out = []
        for kind, vec in [("correct", direction)] + [
                (f"label_permuted_{k}", n) for k, n in enumerate(nulls)]:
            per = []
            for start in range(0, len(utts), args.batch_size):
                batch = utts.iloc[start:start + args.batch_size]
                sub = targets[targets["utterance_id"].isin(
                    set(batch["utterance_id"].astype(str)))]
                per.append(OS.score_batch(bundle, batch, sub,
                                          decoder=(layer, vec, rho),
                                          decoder_positions=positions))
            frame = pd.concat(per, ignore_index=True)
            frame["kind"] = kind
            for key, value in cell.items():
                frame[key] = value
            out.append(frame)
        merged = pd.concat(out, ignore_index=True)
        return [merged.merge(targets[["utterance_id", "reference_unit_index",
                                      "dialogue_id", "category"]],
                             on=["utterance_id", "reference_unit_index"], how="left")]

    results: list[pd.DataFrame] = []
    reweighting: dict[str, Any] = {}

    # ---- PRIORITY CELL: inverse-variance weighting at both sites -------------
    base_spans = _spans_for(candidates, dialogue_of, poi,
                            convention=D3.CTC_CONVENTION, family="existing_ctc")
    construct = base_spans[(base_spans["role"] == "D-construct")
                           & base_spans["all_correct"]]
    build_manifest = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
    runs = D3.matrix_runs(candidates)
    log.info("priority cell: rebuilding both sites under inverse-variance weighting")
    e_meta, e_raw = D3.site_e_contrasts(bundle, build_manifest, construct, runs,
                                        [15, 23, 27, 31], batch_size=4, log=log)
    d_meta, d_raw = D3.site_d_contrasts(bundle, build_manifest, construct,
                                        [8, 16, 24, 31], batch_size=4, log=log)
    for site, layers, raw, meta in (("E", [15, 23, 27, 31], e_raw, e_meta),
                                    ("D", [8, 16, 24, 31], d_raw, d_meta)):
        for lyr in layers:
            key = (lyr, D3.PRIMARY_POOLING) if site == "E" else lyr
            contrasts = np.vstack(raw["vectors"][key])
            entry = assemble_inverse_variance(contrasts, meta)
            reweighting[f"site_{site.lower()}_layer{lyr}"] = {
                k: v for k, v in entry.items() if k != "direction"}
            reweighting[f"site_{site.lower()}_layer{lyr}"]["cos_with_published"] = \
                D3.cosine(entry["direction"],
                          directions[f"site_e_layer{lyr}_{D3.PRIMARY_POOLING}"]
                          if site == "E" else directions[f"site_d_layer{lyr}"])
            reweighting[f"site_{site.lower()}_layer{lyr}"]["direction"] = entry["direction"]

    d_nulls_iv = [n["direction"] for n in OS.label_permuted_directions(
        np.vstack(d_raw["vectors"][layer]), d_meta,
        draws=OS.CONTROL_LABEL_PERMUTED_DRAWS, seed=OS.CONTROL_SEED + 100 + layer)]
    results += run_cell(base_spans, reweighting[f"site_d_layer{layer}"]["direction"],
                        d_nulls_iv,
                        cell={"factor": "inverse_variance", "site": "D",
                              "layer": str(layer), "detail": "reweighted"})
    e_best = 27          # Site E's largest Day 4 point estimate; not a selection
    e_nulls_iv = [n["direction"] for n in OS.label_permuted_directions(
        np.vstack(e_raw["vectors"][(e_best, D3.PRIMARY_POOLING)]), e_meta,
        draws=OS.CONTROL_LABEL_PERMUTED_DRAWS, seed=OS.CONTROL_SEED + e_best)]
    for tag, vec, nulls in (("reweighted",
                             reweighting[f"site_e_layer{e_best}"]["direction"], e_nulls_iv),):
        develop = base_spans[base_spans["role"] == "D-dev-select"]
        targets = OS.build_targets(bundle, candidates, develop, poi, manifest)
        if args.limit:
            keep = list(dict.fromkeys(targets["utterance_id"]))[: int(args.limit)]
            targets = targets[targets["utterance_id"].isin(keep)]
        utts = manifest[manifest["utterance_id"].isin(set(targets["utterance_id"]))]
        utts = utts.drop_duplicates("utterance_id").reset_index(drop=True)
        frames = {str(t["utterance_id"]): (int(t["span_frame_lo"]), int(t["span_frame_hi"]))
                  for _, t in targets.iterrows()}
        for kind, v in [("correct", vec)] + [(f"label_permuted_{k}", n)
                                             for k, n in enumerate(nulls)]:
            per = []
            for start in range(0, len(utts), args.batch_size):
                batch = utts.iloc[start:start + args.batch_size]
                sub = targets[targets["utterance_id"].isin(
                    set(batch["utterance_id"].astype(str)))]
                per.append(OS.score_batch(bundle, batch, sub,
                                          encoder=(e_best, v, rho),
                                          encoder_frames=frames))
            frame = pd.concat(per, ignore_index=True)
            frame["kind"], frame["factor"] = kind, "inverse_variance"
            frame["site"], frame["layer"], frame["detail"] = "E", str(e_best), tag
            results.append(frame.merge(
                targets[["utterance_id", "reference_unit_index", "dialogue_id", "category"]],
                on=["utterance_id", "reference_unit_index"], how="left"))

    # ---- factor 4: pooling -----------------------------------------------------
    # Encoder frame pooling is a construction choice that can only reach a site
    # reading encoder frames, so it is exercised at Site E. It is structurally
    # inert for the Site-D headline (a decoder residual is never frame-pooled),
    # which is asserted below rather than assumed.
    pooling_meta: dict[str, Any] = {}
    for variant in POOLINGS:
        built = D3.assemble(np.vstack(e_raw["vectors"][(e_best, variant)]), e_meta)
        pooling_meta[variant] = {k: v for k, v in built.items() if k != "direction"}
        p_nulls = [n["direction"] for n in OS.label_permuted_directions(
            np.vstack(e_raw["vectors"][(e_best, variant)]), e_meta,
            draws=OS.CONTROL_LABEL_PERMUTED_DRAWS,
            seed=OS.CONTROL_SEED + 200 + POOLINGS.index(variant))]
        log.info("cell: pooling %s (Site E, layer %d)", variant, e_best)
        for kind, v in ([("correct", built["direction"])]
                        + [(f"label_permuted_{k}", n) for k, n in enumerate(p_nulls)]):
            per = []
            for start in range(0, len(utts), args.batch_size):
                batch = utts.iloc[start:start + args.batch_size]
                sub = targets[targets["utterance_id"].isin(
                    set(batch["utterance_id"].astype(str)))]
                per.append(OS.score_batch(bundle, batch, sub,
                                          encoder=(e_best, v, rho),
                                          encoder_frames=frames))
            frame = pd.concat(per, ignore_index=True)
            frame["kind"], frame["factor"] = kind, "pooling"
            frame["site"], frame["layer"], frame["detail"] = "E", str(e_best), variant
            results.append(frame.merge(
                targets[["utterance_id", "reference_unit_index", "dialogue_id", "category"]],
                on=["utterance_id", "reference_unit_index"], how="left"))

    # ---- factor cells, all at the frozen Site-D action ------------------------
    published = directions[f"site_d_layer{layer}"]
    nulls = [n["direction"] for n in OS.label_permuted_directions(
        np.vstack(d_raw["vectors"][layer]), d_meta,
        draws=OS.CONTROL_LABEL_PERMUTED_DRAWS, seed=OS.CONTROL_SEED + 100 + layer)]

    for convention in CONVENTIONS:
        spans = _spans_for(candidates, dialogue_of, poi, convention=convention,
                           family="existing_ctc")
        log.info("cell: convention %s", convention)
        results += run_cell(spans, published, nulls,
                            cell={"factor": "convention", "site": "D",
                                  "layer": str(layer), "detail": convention})
    for family in ALIGNER_PATHS:
        spans = _spans_for(candidates, dialogue_of, poi,
                           convention=D3.CTC_CONVENTION, family=family)
        log.info("cell: aligner path %s", family)
        results += run_cell(spans, published, nulls,
                            cell={"factor": "aligner_path", "site": "D",
                                  "layer": str(layer), "detail": family})
    # Corpus token rate, used to turn a boundary jitter in ms into a decoder-step
    # displacement. d_meta carries onset_position and relative_start =
    # token_index / n_tokens, so n_tokens = onset_position / relative_start.
    _rs = pd.to_numeric(d_meta["relative_start"], errors="coerce").to_numpy(float)
    _op = pd.to_numeric(d_meta["onset_position"], errors="coerce").to_numpy(float)
    _du = pd.to_numeric(d_meta["utterance_duration_sec"], errors="coerce").to_numpy(float)
    _ok = np.isfinite(_rs) & np.isfinite(_op) & np.isfinite(_du) & (_rs > 0) & (_du > 0)
    mean_tokens_per_sec = float(np.mean((_op[_ok] / _rs[_ok]) / _du[_ok])) if _ok.any() else 0.0
    log.info("corpus mean token rate: %.2f tokens/s", mean_tokens_per_sec)

    # ---- factor 1b: Mandarin CONTROL boundary jitter ---------------------------
    # A ZH-control shift cannot reach the Site-D target step (that is the EN
    # onset); it reaches the headline only by changing the contrast that builds
    # the direction. So it is applied at construction and rescored at the
    # untouched targets.
    for sign in (+1, -1):
        for jitter in JITTER_MS:
            shift_ms = sign * jitter
            offset = int(round(abs(shift_ms) / 1000.0 * mean_tokens_per_sec))
            offset = offset if sign > 0 else -offset
            zh_meta, zh_raw = D3.site_d_contrasts(
                bundle, build_manifest, construct, [layer],
                batch_size=args.batch_size, control_offset=offset, log=log)
            built = D3.assemble(np.vstack(zh_raw["vectors"][layer]), zh_meta)
            zh_nulls = [n["direction"] for n in OS.label_permuted_directions(
                np.vstack(zh_raw["vectors"][layer]), zh_meta,
                draws=OS.CONTROL_LABEL_PERMUTED_DRAWS,
                seed=OS.CONTROL_SEED + 300 + int(shift_ms))]
            log.info("cell: ZH-control jitter %+.0f ms (%+d steps)", shift_ms, offset)
            results += run_cell(base_spans, built["direction"], zh_nulls,
                                cell={"factor": "jitter_zh", "site": "D",
                                      "layer": str(layer),
                                      "detail": f"{shift_ms:+.0f}ms",
                                      "control_offset_steps": offset})

    for sign in (+1, -1):
        for jitter in JITTER_MS:
            shift = sign * jitter
            log.info("cell: EN-span jitter %+.0f ms (decoder-step displacement)", shift)
            results += run_cell(base_spans, published, nulls,
                                cell={"factor": "jitter_en", "site": "D",
                                      "layer": str(layer),
                                      "detail": f"{shift:+.0f}ms"},
                                jitter_ms=shift)
    scores = pd.concat(results, ignore_index=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(target_dir / "invariance_scores.parquet", index=False)
    np.savez(target_dir / "inverse_variance_directions.npz",
             **{k: v.pop("direction") for k, v in reweighting.items()})
    payload = {"frozen_configuration": FROZEN_CONFIG, "plan": plan,
               "inverse_variance_construction": reweighting,
               "rows": int(len(scores)),
               "pooling_construction": pooling_meta,
               "mean_tokens_per_sec": mean_tokens_per_sec,
               "pooling_note": ("pooling is a Site-E construction factor; the "
                                "Site-D headline contrasts decoder states and "
                                "does no frame pooling, so pooling is a no-op "
                                "for it by construction and is reported as such"),
               "zh_jitter_note": ("Mandarin-control jitter enters the Site-D "
                                  "headline only through the direction's control "
                                  "position, not through evaluation, because "
                                  "Site D pools no control frames"),
               "selects_anything": False, "evaluates_gate": False}
    (target_dir / "invariance_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    for name in ("invariance_scores.parquet", "inverse_variance_directions.npz",
                 "invariance_report.json"):
        manifest_mod.publish(target_dir / name, stage=STAGE, cfg=cfg,
                             parents=[parent] if parent else (),
                             schema="lss_v2r3_invariance_v1",
                             extra={"frozen_configuration": FROZEN_CONFIG,
                                    "evaluates_gate": False})
    print(json.dumps({"state": "completed", "output": str(target_dir),
                      "rows": int(len(scores)),
                      "cells": sorted(set(zip(scores.factor, scores.detail)))},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
