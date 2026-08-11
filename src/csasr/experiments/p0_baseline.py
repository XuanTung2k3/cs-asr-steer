"""P0 — manifest/splits, frozen baselines and the POI audit (guide section 8).

Run:
    python -m csasr.experiments.p0_baseline --config configs/experiments/e1_alignment.yaml \
        --build-manifest --resume
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data.manifest import build_manifest, load_manifest, validate_manifest
from ..data.splits import (
    assign_internal_splits,
    build_direction_subsets,
    load_subset,
    validate_splits,
    write_split_manifests,
)
from ..evaluation.mer import corpus_mer
from ..evaluation.pier import LANGUAGE_CONFUSION, pier, poi_table
from ..models.generation import decode_manifest
from ..models.whisper import load_whisper, write_model_metadata
from ..utils.config import art
from ..utils.hashing import manifest_hash
from ..utils.status import StageLock, criterion, gate
from ._common import base_parser, finish, md_table, prepare, save_report

STAGE = "p0"
BASELINES = {"B0_AUTO": None, "B1_ZH": "zh"}
DEV_SUBSETS = ("dev_select", "dev_confirm")


def build_data_artifacts(cfg, log, overwrite: bool = False) -> dict:
    """Manifest + internal splits + direction subsets, with validation."""
    manifest_path = Path(cfg["data"]["manifest"])
    if manifest_path.exists() and not overwrite:
        log.info("manifest exists, reusing: %s", manifest_path)
        full = pd.read_parquet(manifest_path)
    else:
        log.info("building manifest from %s", cfg["data"]["index_root"])
        full = build_manifest(cfg)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(manifest_path, index=False)

    report = validate_manifest(full, prohibit_test=cfg["experiment"]["prohibit_test_split"])
    log.info("manifest validation: %s", json.dumps(report["per_split"], indent=2))

    work = full[full["official_split"] != "test"].reset_index(drop=True)
    work = assign_internal_splits(work, n_select=int(cfg["data"]["dev_select_speakers"]),
                                  seed=int(cfg["data"]["split_seed"]))
    subsets = build_direction_subsets(
        work,
        pilot_size=int(cfg["data"]["direction_pilot_utterances"]),
        seed=int(cfg["data"]["direction_pilot_seed"]),
        bootstrap_seeds=tuple(cfg["data"]["bootstrap_seeds"]),
    )
    split_report = validate_splits(work, subsets)
    write_split_manifests(art(cfg, "manifests"), work, subsets, split_report)
    log.info("splits: %s", json.dumps({k: v for k, v in split_report.items()
                                       if isinstance(v, dict)}, indent=2))
    return {"manifest": report, "splits": split_report,
            "manifest_hash": manifest_hash(work)}


def run_baselines(bundle, cfg, log, resume: bool, overwrite: bool,
                  limit: int | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    preds: dict[str, dict[str, pd.DataFrame]] = {}
    for system, language in BASELINES.items():
        preds[system] = {}
        for subset in DEV_SUBSETS:
            df = load_subset(cfg, subset)
            if limit:
                df = df.head(limit)
            out_path = art(cfg, "baselines", system, f"{subset}.parquet")
            log.info("decoding %s on %s (%d utterances)", system, subset, len(df))
            preds[system][subset] = decode_manifest(
                bundle, df, cfg, system=system, config_id=system, out_path=out_path,
                language=language, resume=resume, overwrite=overwrite,
            )
    return preds


def score(manifest: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    merged = manifest.merge(predictions, on="utterance_id", how="inner")
    m = corpus_mer(merged["transcript_raw"].tolist(), merged["hypothesis_raw"].tolist())
    p = pier(merged["transcript_raw"].tolist(), merged["hypothesis_raw"].tolist())
    return {**m, **p, "num_utterances": int(len(merged))}


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("P0 baseline and POI audit", "experiments/e1_alignment.yaml")
    parser.add_argument("--build-manifest", action="store_true",
                        help="(re)build manifest, splits and direction subsets")
    parser.add_argument("--data-only", action="store_true",
                        help="stop after data artifacts (no decoding)")
    args = parser.parse_args(argv)
    cfg, rdir, log = prepare(args, STAGE)
    root = cfg["experiment"]["output_root"]

    with StageLock(root, STAGE):
        data_report = build_data_artifacts(cfg, log, overwrite=args.build_manifest or args.overwrite)
        if args.dry_run or args.data_only:
            log.info("data artifacts complete; stopping before decoding")
            metrics = {"data": data_report}
            (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str),
                                               encoding="utf-8")
            return 0

        bundle = load_whisper(cfg)
        write_model_metadata(bundle, rdir / "model_metadata.json")
        preds = run_baselines(bundle, cfg, log, resume=args.resume or not args.overwrite,
                              overwrite=args.overwrite, limit=args.limit)

        # ---- scoring and primary-baseline selection (on dev_select only) ----
        scores: dict[str, dict] = {}
        for system in BASELINES:
            scores[system] = {s: score(load_subset(cfg, s), preds[system][s])
                              for s in DEV_SUBSETS}
        primary = min(BASELINES, key=lambda s: scores[s]["dev_select"]["mer"])
        art(cfg, "baselines", "primary_baseline.json").write_text(
            json.dumps({
                "system": primary,
                "language": BASELINES[primary],
                "selected_on": "dev_select",
                "criterion": "lowest MER",
                "scores": scores,
            }, indent=2, default=str), encoding="utf-8")
        log.info("primary baseline frozen: %s", primary)

        # ---- POI tables and error taxonomy ----
        poi_counts: dict[str, dict] = {}
        for subset in DEV_SUBSETS:
            man = load_subset(cfg, subset)
            merged = man.merge(preds[primary][subset], on="utterance_id", how="inner")
            rows = poi_table(merged["utterance_id"].tolist(),
                             merged["transcript_raw"].tolist(),
                             merged["hypothesis_raw"].tolist())
            table = pd.DataFrame(rows)
            if len(table):
                table = table.merge(man[["utterance_id", "speaker_id", "duration_sec"]],
                                    on="utterance_id", how="left")
            table.to_parquet(art(cfg, "manifests", f"poi_{subset}.parquet"), index=False)
            wrong = table[~table["correct"]] if len(table) else table
            conf = wrong[wrong["category"].isin(LANGUAGE_CONFUSION)] if len(wrong) else wrong
            poi_counts[subset] = {
                "num_poi": int(len(table)),
                "num_incorrect": int(len(wrong)),
                "num_correct": int(len(table) - len(wrong)),
                "num_language_confusion": int(len(conf)),
                "num_utterances_with_poi": int(table["utterance_id"].nunique()) if len(table) else 0,
                "per_category": (wrong["category"].value_counts().to_dict() if len(wrong) else {}),
            }
            log.info("%s POI audit: %s", subset, json.dumps(poi_counts[subset]))

        # ---- gate P0 ----
        total_wrong = sum(v["num_incorrect"] for v in poi_counts.values())
        total_conf = sum(v["num_language_confusion"] for v in poi_counts.values())
        total_correct = sum(v["num_correct"] for v in poi_counts.values())
        criteria = [
            criterion("erroneous_en_pois", total_wrong, 300, total_wrong >= 300),
            criterion("language_confusion_errors", total_conf, 100, total_conf >= 100),
            criterion("baseline_correct_en_pois", total_correct, 300, total_correct >= 300),
        ]
        g = gate("p0", all(c["passed"] for c in criteria), criteria,
                 "Counts are over dev_select + dev_confirm with the frozen primary baseline. "
                 "A failure means the dataset/scope decision must be taken by the user; "
                 "do not fabricate a balanced sample.")

        metrics = {"data": data_report, "scores": scores, "primary_baseline": primary,
                   "poi": poi_counts}
        rows = [{"system": s, "subset": sub, **{k: v for k, v in sc.items()
                                                if not isinstance(v, dict)}}
                for s, per in scores.items() for sub, sc in per.items()]
        save_report(
            art(cfg, "reports", "p0_baseline_audit.md"), "P0 — baseline and POI audit",
            [
                ("Data", md_table(pd.DataFrame(data_report["manifest"]["per_split"]).T.reset_index())),
                ("Baseline scores", md_table(pd.DataFrame(rows))),
                ("Primary baseline", f"`{primary}` (lowest MER on dev_select; frozen for E1-E5)"),
                ("POI audit", "```json\n" + json.dumps(poi_counts, indent=2) + "\n```"),
                ("Gate", "```json\n" + json.dumps(g, indent=2, default=str) + "\n```"),
            ],
        )
        finish(cfg, STAGE, rdir, metrics, g,
               artifacts=[str(art(cfg, "reports", "p0_baseline_audit.md"))])
        log.info("P0 gate: %s", "PASSED" if g["passed"] else "FAILED")
        return 0 if g["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
