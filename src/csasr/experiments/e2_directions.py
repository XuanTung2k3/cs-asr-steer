"""E2 — layer-wise activation extraction and direction construction.

Run:
    python -m csasr.experiments.e2_directions \
        --config configs/experiments/e2_directions.yaml --seeds 42 43 44 45 46 --resume
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.splits import load_subset
from ..directions.decoder import (
    accumulate_decoder_directions,
    build_decoder_directions,
    verify_decoder_shift,
)
from ..directions.encoder import (
    accumulate_directions,
    control_directions,
    directions_from_accumulator,
    save_direction,
    validate_direction,
)
from ..evaluation.pier import annotate_unit_status
from ..models.generation import decode_manifest
from ..models.hook_tests import run_hook_tests
from ..models.whisper import batch_features, load_whisper, write_model_metadata
from ..utils.config import art, load_config
from ..utils.hashing import manifest_hash
from ..utils.status import StageLock, criterion, gate, require_passed
from ._common import (
    base_parser,
    finish,
    load_units,
    md_table,
    prepare,
    primary_baseline_id,
    save_report,
)

STAGE_BASE = "e2"
BASELINE_LANGUAGE = {"B0_AUTO": None, "B1_ZH": "zh"}


def ensure_training_baseline(bundle, cfg, subset: str, manifest: pd.DataFrame, log,
                             resume: bool, overwrite: bool) -> pd.DataFrame:
    """Decode exactly the manifest contributing to this direction run."""
    system = primary_baseline_id(cfg)
    out_path = art(cfg, "baselines", system, f"{subset}.parquet")
    log.info("baseline %s on %s for correct-only filtering (%d utterances)",
             system, subset, len(manifest))
    return decode_manifest(bundle, manifest, cfg, system=system, config_id=system,
                           out_path=out_path, language=BASELINE_LANGUAGE.get(system),
                           resume=resume, overwrite=overwrite)


def direction_seed_manifest(cfg, subset: str, manifest: pd.DataFrame,
                            seed: int) -> pd.DataFrame:
    """Return and persist an 80% deterministic subsample for stability."""
    if subset == "train_direction_pilot":
        seeded = load_subset(cfg, f"train_direction_seed{seed}")
        seeded = seeded[seeded["utterance_id"].isin(manifest["utterance_id"])]
    else:
        n = max(1, int(0.8 * len(manifest)))
        seeded = manifest.sample(n=n, random_state=seed).sort_values("utterance_id")
        seeded = seeded.reset_index(drop=True)
        seeded.to_parquet(
            art(cfg, "manifests", f"{subset}_seed{seed}.parquet"), index=False)
    return seeded.reset_index(drop=True)

def prepare_checkpoint(path: Path, identity: dict, overwrite: bool) -> None:
    """Guard resumable accumulators against target/configuration changes."""
    meta = path.with_suffix(path.suffix + ".meta.json")
    processed = path.with_suffix(".processed.json")
    if overwrite:
        path.unlink(missing_ok=True)
        processed.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
    if path.exists():
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8")) != identity:
            raise RuntimeError(f"accumulator provenance mismatch: {path}; use --overwrite")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(identity, indent=2), encoding="utf-8")


def annotated_units(cfg, subset: str, manifest: pd.DataFrame,
                    predictions: pd.DataFrame, log) -> pd.DataFrame:
    units = load_units(cfg, subset)
    out_path = art(cfg, "alignments", f"{subset}_annotated.parquet")
    annotated = annotate_unit_status(units, manifest, predictions)
    annotated.to_parquet(out_path, index=False)
    n_correct = int((annotated["baseline_status"] == "correct").sum())
    log.info("%s: %d/%d units baseline-correct", subset, n_correct, len(annotated))
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("E2 direction construction", "experiments/e2_directions.yaml")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--skip-decoder", action="store_true")
    args = parser.parse_args(argv)
    preview = load_config(args.config, args.overrides)
    preview_subset = preview.get("directions", {}).get("subset", "train_direction_pilot")
    stage = "e2_full" if preview_subset == "train_direction_full" else "e2_pilot"
    cfg, rdir, log = prepare(args, stage)
    root = cfg["experiment"]["output_root"]
    dcfg = cfg["directions"]
    layers = list(dcfg["encoder_layers"])
    seeds = args.seeds or list(dcfg["seeds"])
    subset = dcfg.get("subset", "train_direction_pilot")
    norm_version = cfg["experiment"]["normalization_version"]

    with StageLock(root, stage):
        if not args.force_prereq:
            require_passed(root, ["e1"])
        bundle = load_whisper(cfg)
        write_model_metadata(bundle, rdir / "model_metadata.json")
        exclude_frames = int(round((cfg["alignment"]["exclude_boundary_ms"] / 1000)
                                   / bundle.encoder_step_sec))

        manifest = load_subset(cfg, subset)
        if args.limit:
            manifest = manifest.head(args.limit)
        mhash = manifest_hash(manifest)

        # ---- hook correctness tests (gate prerequisite) ----------------
        probe = batch_features(bundle, manifest["audio_path"].head(2).tolist())
        hook_report = run_hook_tests(bundle, probe, layer=layers[-1])
        log.info("hook tests: %s", "PASSED" if hook_report["all_passed"] else "FAILED")
        if not hook_report["all_passed"] and not args.force_prereq:
            log.error("hook tests failed: %s", json.dumps(hook_report, indent=2, default=str))
        del probe

        if args.dry_run:
            (rdir / "hook_tests.json").write_text(
                json.dumps(hook_report, indent=2, default=str), encoding="utf-8")
            return 0 if hook_report["all_passed"] else 2

        # ---- baseline on the training subset + unit annotation ---------
        preds = ensure_training_baseline(
            bundle, cfg, subset, manifest, log,
            resume=args.resume or not args.overwrite, overwrite=args.overwrite)
        units = annotated_units(cfg, subset, manifest, preds, log)

        # ---- accumulate ------------------------------------------------
        dir_root = art(cfg, "directions")
        stats: dict = {"layers": layers, "seeds": seeds, "subset": subset,
                       "manifest_hash": mhash, "hook_tests": hook_report,
                       "accumulation": {}}
        primary_by_seed: dict[int, dict[int, dict]] = {}

        for seed in seeds:
            seed_manifest = direction_seed_manifest(cfg, subset, manifest, seed)
            if args.limit:
                seed_manifest = seed_manifest.head(args.limit)

            ckpt = art(cfg, "directions", "_accumulators",
                       f"correct_only_{subset}_seed{seed}.npz")
            prepare_checkpoint(ckpt, {"variant": "correct_only", "subset": subset,
                                      "seed": seed, "manifest_hash": manifest_hash(seed_manifest),
                                      "layers": layers, "model_revision": bundle.revision},
                               args.overwrite)
            accs = accumulate_directions(
                bundle, seed_manifest, units, layers,
                exclude_frames=exclude_frames, correct_only=True,
                min_en_frames=int(dcfg["min_en_frames"]),
                min_zh_frames=int(dcfg["min_zh_frames"]),
                batch_size=int(dcfg.get("batch_size", 8)),
                checkpoint_path=ckpt, checkpoint_every=int(dcfg.get("checkpoint_every", 200)),
                progress_desc=f"e2 correct_only seed{seed}",
            )
            stats["accumulation"][f"correct_only_seed{seed}"] = accs.summary()

            primary = directions_from_accumulator(
                bundle, accs, variant="within_utterance_correct_only", seed=seed,
                manifest_hash=manifest_hash(seed_manifest), normalization_version=norm_version)
            primary_by_seed[seed] = primary
            for rec in primary.values():
                validate_direction(rec)
                save_direction(rec, dir_root)

            gc = directions_from_accumulator(
                bundle, accs, variant="global_centroid_correct_only", seed=seed,
                manifest_hash=manifest_hash(seed_manifest), normalization_version=norm_version)
            for rec in gc.values():
                validate_direction(rec)
                save_direction(rec, dir_root)

            if seed == seeds[0]:
                ckpt2 = art(cfg, "directions", "_accumulators", f"all_valid_{subset}_seed{seed}.npz")
                prepare_checkpoint(ckpt2, {"variant": "all_valid", "subset": subset,
                                           "seed": seed, "manifest_hash": manifest_hash(seed_manifest),
                                           "layers": layers, "model_revision": bundle.revision},
                                   args.overwrite)
                accs_all = accumulate_directions(
                    bundle, seed_manifest, units, layers,
                    exclude_frames=exclude_frames, correct_only=False,
                    min_en_frames=int(dcfg["min_en_frames"]),
                    min_zh_frames=int(dcfg["min_zh_frames"]),
                    batch_size=int(dcfg.get("batch_size", 8)),
                    checkpoint_path=ckpt2,
                    progress_desc=f"e2 all_valid seed{seed}",
                )
                stats["accumulation"][f"all_valid_seed{seed}"] = accs_all.summary()
                av = directions_from_accumulator(
                    bundle, accs_all, variant="within_utterance_all_valid", seed=seed,
                    manifest_hash=manifest_hash(seed_manifest),
                    normalization_version=norm_version)
                for rec in av.values():
                    validate_direction(rec)
                    save_direction(rec, dir_root)

                controls = control_directions(bundle, primary, dcfg["random_seeds"],
                                              mhash, norm_version)
                for group in controls.values():
                    for rec in group.values():
                        validate_direction(rec)
                        save_direction(rec, dir_root)
                stats["controls"] = sorted(controls)

        # ---- cross-seed stability -------------------------------------
        stability = {}
        for layer in layers:
            vecs = np.stack([primary_by_seed[s][layer]["direction"].numpy() for s in seeds])
            cos = vecs @ vecs.T
            iu = np.triu_indices(len(seeds), k=1)
            stability[str(layer)] = {
                "mean_pairwise_cosine": float(cos[iu].mean()) if len(seeds) > 1 else 1.0,
                "min_pairwise_cosine": float(cos[iu].min()) if len(seeds) > 1 else 1.0,
                "projection_std_by_seed": {
                    str(s): float(primary_by_seed[s][layer]["projection_std"]) for s in seeds},
                "num_utterances": int(primary_by_seed[seeds[0]][layer]["num_utterances"]),
                "num_en_frames": int(primary_by_seed[seeds[0]][layer]["num_en_frames_or_tokens"]),
                "num_zh_frames": int(primary_by_seed[seeds[0]][layer]["num_zh_frames_or_tokens"]),
            }
        stats["stability"] = stability

        # ---- decoder baseline directions -------------------------------
        decoder_report: dict = {}
        if not args.skip_decoder and dcfg.get("build_decoder", True):
            shift = verify_decoder_shift(bundle, manifest, language="zh")
            log.info("decoder shift verified: %s", json.dumps(shift, default=str))
            dec_manifest = manifest.head(int(dcfg.get("decoder_max_utterances", 2000)))
            dec_ckpt = art(cfg, "directions", "_accumulators", f"decoder_{subset}.npz")
            prepare_checkpoint(dec_ckpt, {"variant": "decoder", "subset": subset,
                                          "manifest_hash": manifest_hash(dec_manifest),
                                          "model_revision": bundle.revision}, args.overwrite)
            dec_accs = accumulate_decoder_directions(
                bundle, dec_manifest, language="zh",
                batch_size=max(2, int(dcfg.get("batch_size", 8)) // 2),
                checkpoint_path=dec_ckpt)
            dec_dirs = build_decoder_directions(bundle, dec_accs, seed=seeds[0],
                                                manifest_hash=manifest_hash(dec_manifest),
                                                normalization_version=norm_version)
            for rec in dec_dirs.values():
                validate_direction(rec)
                save_direction(rec, dir_root)
            decoder_report = {
                "shift_verification": shift,
                "num_layers": len(dec_dirs),
                "expected_layers": bundle.num_decoder_layers,
                "summary": dec_accs.summary(),
            }
        stats["decoder"] = decoder_report

        # ---- gate ------------------------------------------------------
        min_cos = min(v["min_pairwise_cosine"] for v in stability.values())
        min_en = min(v["num_en_frames"] for v in stability.values())
        min_zh = min(v["num_zh_frames"] for v in stability.values())
        dec_ok = (not dcfg.get("build_decoder", True) or args.skip_decoder
                  or decoder_report.get("num_layers", 0) == bundle.num_decoder_layers)
        criteria = [
            criterion("directions_for_all_layers", len(primary_by_seed[seeds[0]]), len(layers),
                      len(primary_by_seed[seeds[0]]) == len(layers), "=="),
            criterion("all_directions_finite_unit_norm", 1, 1, True, "=="),
            criterion("cross_seed_min_cosine", min_cos, 0.90, min_cos >= 0.90),
            criterion("min_en_frames", min_en, 1000, min_en >= 1000),
            criterion("min_zh_frames", min_zh, 1000, min_zh >= 1000),
            criterion("decoder_directions_present", decoder_report.get("num_layers", 0),
                      bundle.num_decoder_layers, dec_ok, "=="),
            criterion("hook_tests_passed", int(hook_report["all_passed"]), 1,
                      hook_report["all_passed"], "=="),
            criterion("no_dev_or_test_contribution", int((manifest["official_split"] == "train").all()), 1, bool((manifest["official_split"] == "train").all()), "=="),
        ]
        g = gate(stage, all(c["passed"] for c in criteria), criteria,
                 "E2 does not require good separability; that is E3.")

        stats_name = f"{stage}_direction_statistics.json"
        art(cfg, "metrics", stats_name).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        stab_rows = pd.DataFrame([
            {"layer": k, **{kk: vv for kk, vv in v.items()
                            if not isinstance(vv, dict)}} for k, v in stability.items()])
        save_report(
            art(cfg, "reports", f"{stage}_direction_construction.md"),
            f"E2 — layer-wise activation extraction and direction construction ({stage})",
            [
                ("Hook convention",
                 f"`{bundle.module_report['encoder_hook_location']}` "
                 f"(`{bundle.module_report['encoder_layer_module_path']}`, "
                 f"{bundle.module_report['encoder_layer_indexing']})"),
                ("Hook tests", "```json\n" + json.dumps(hook_report, indent=2, default=str) + "\n```"),
                ("Direction stability", md_table(stab_rows)),
                ("Decoder baseline", "```json\n" + json.dumps(decoder_report, indent=2,
                                                              default=str) + "\n```"),
                ("Gate", "```json\n" + json.dumps(g, indent=2, default=str) + "\n```"),
            ],
        )
        finish(cfg, stage, rdir, stats, g, artifacts=[str(dir_root)])
        log.info("E2 gate: %s", "PASSED" if g["passed"] else "FAILED")
        return 0 if g["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
