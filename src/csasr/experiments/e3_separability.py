"""E3 — language-direction separability on unseen development speakers.

Diagnostic gate only: it shows whether frame language is linearly decodable
along d_l, not that intervening along d_l changes the transcript.

Run:
    python -m csasr.experiments.e3_separability \
        --config configs/experiments/e3_separability.yaml --resume
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm

from ..data.alignment import core_frames
from ..data.language_tags import EN, ZH
from ..data.splits import load_subset
from ..directions.encoder import load_direction
from ..evaluation.bootstrap import grouped_bootstrap
from ..models.hooks import ActivationRecorder, assert_no_hooks
from ..models.whisper import batch_features, load_whisper
from ..utils.config import art
from ..utils.status import StageLock, criterion, gate, require_passed
from ._common import base_parser, finish, load_units, md_table, prepare, save_report

STAGE_BASE = "e3"


@torch.inference_mode()
def collect_projection_data(bundle, manifest: pd.DataFrame, units: pd.DataFrame,
                            layers, exclude_frames: int, batch_size: int,
                            log) -> dict[int, pd.DataFrame]:
    """Per-layer frame table: activation index, label, speaker, boundary flag.

    Activations themselves are never stored; only the raw hidden vectors needed
    for projections are kept per batch and reduced immediately to a compact
    float32 matrix of eligible frames.
    """
    by_utt = {u: g for u, g in units.groupby("utterance_id")}
    manifest = manifest.sort_values("duration_sec").reset_index(drop=True)
    n_batches = math.ceil(len(manifest) / batch_size)
    acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    meta_rows: list[dict] = []

    for bi in tqdm(range(n_batches), desc="e3 collect", leave=False):
        batch = manifest.iloc[bi * batch_size: (bi + 1) * batch_size]
        plans = []
        for _, row in batch.iterrows():
            u = by_utt.get(row["utterance_id"])
            if u is None or not len(u):
                continue
            n_valid = bundle.valid_frames(row["duration_sec"])
            sel: list[tuple[int, str, bool]] = []
            for _, r in u[u["language_tag"].isin([EN, ZH]) & u["is_content"]].iterrows():
                boundary = bool(r["is_boundary_adjacent"])
                s, e = int(r["start_frame"]), int(r["end_frame"])
                if boundary:
                    cs, ce = core_frames(s, e, exclude_frames)
                else:
                    cs, ce = s, e
                for f in range(max(0, cs), min(n_valid, ce)):
                    sel.append((f, r["language_tag"], boundary))
                if boundary:
                    for f in list(range(max(0, s), max(0, cs))) + list(range(min(n_valid, ce), min(n_valid, e))):
                        sel.append((f, r["language_tag"], True))
            if sel:
                plans.append((row, sel))
        if not plans:
            continue

        features = batch_features(bundle, [p[0]["audio_path"] for p in plans])
        with ActivationRecorder(bundle, list(layers), module="encoder") as rec:
            bundle.model.model.encoder(features)
            states = {l: rec.states[l] for l in layers}
        assert_no_hooks(bundle)

        for i, (row, sel) in enumerate(plans):
            idx = torch.as_tensor([f for f, _, _ in sel], device=features.device)
            for l in layers:
                acts[l].append(states[l][i, idx].float().cpu().numpy())
            for f, tag, boundary in sel:
                meta_rows.append({
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "frame": f,
                    "label": 1 if tag == EN else 0,
                    "is_boundary": boundary,
                })
        del states, features

    meta = pd.DataFrame(meta_rows, columns=["utterance_id", "speaker_id", "frame",
                                            "label", "is_boundary"])
    out = {}
    for l in layers:
        matrix = np.concatenate(acts[l], axis=0) if acts[l] else np.zeros((0, bundle.d_model), dtype=np.float32)
        out[l] = (matrix, meta)
        labels = meta["label"].to_numpy(dtype=int)
        log.info("layer %d: %d eligible frames (%d EN / %d ZH)", l, len(matrix),
                 int(labels.sum()), int(len(labels) - labels.sum()))
    if not len(meta):
        log.warning("no eligible frames were collected; every E3 metric will be empty")
    return out


def separability_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float,
                         speakers: np.ndarray, utterances: np.ndarray,
                         bootstrap: int, seed: int) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "note": "single-class sample"}
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, labels=[0, 1],
                                                  zero_division=0)
    auroc = float(roc_auc_score(labels, scores))

    def macro_auroc(values: np.ndarray, groups: np.ndarray) -> float:
        vals = []
        for g in np.unique(groups):
            m = groups == g
            if len(np.unique(labels[m])) == 2:
                vals.append(roc_auc_score(labels[m], scores[m]))
        return float(np.mean(vals)) if vals else float("nan")

    en, zh = scores[labels == 1], scores[labels == 0]
    pooled_sd = np.sqrt(0.5 * (en.var(ddof=1) + zh.var(ddof=1))) if len(en) > 1 and len(zh) > 1 else np.nan
    idx = np.arange(len(scores), dtype=float)

    def auroc_stat(sub_idx: np.ndarray) -> float:
        i = sub_idx.astype(int)
        return float(roc_auc_score(labels[i], scores[i])) if len(np.unique(labels[i])) == 2 else np.nan

    ci = grouped_bootstrap(idx, speakers, statistic=auroc_stat,
                           n_resamples=bootstrap, seed=seed)
    return {
        "auroc": auroc,
        "auprc": float(average_precision_score(labels, scores)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "en_precision": float(p[1]), "en_recall": float(r[1]), "en_f1": float(f1[1]),
        "zh_precision": float(p[0]), "zh_recall": float(r[0]), "zh_f1": float(f1[0]),
        "balanced_accuracy": float(0.5 * (r[0] + r[1])),
        "centroid_gap": float(en.mean() - zh.mean()),
        "standardized_effect_size": float((en.mean() - zh.mean()) / pooled_sd) if pooled_sd else float("nan"),
        "utterance_macro_auroc": macro_auroc(scores, utterances),
        "speaker_macro_auroc": macro_auroc(scores, speakers),
        "speaker_bootstrap_ci_low": ci["ci_low"],
        "speaker_bootstrap_ci_high": ci["ci_high"],
        "num_frames": int(len(scores)),
        "num_en": int(labels.sum()),
        "num_zh": int(len(labels) - labels.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("E3 separability evaluation", "experiments/e3_separability.yaml")
    parser.add_argument("--phase", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args(argv)
    stage = f"e3_{args.phase}"
    cfg, rdir, log = prepare(args, stage)
    root = cfg["experiment"]["output_root"]
    ecfg = cfg["e3"]
    layers = list(cfg["directions"]["encoder_layers"])
    seeds = list(cfg["directions"]["seeds"])
    random_seeds = list(cfg["directions"]["random_seeds"])
    subset = ecfg.get("subset", "dev_select")

    with StageLock(root, stage):
        if not args.force_prereq:
            require_passed(root, ["e1", f"e2_{args.phase}"])
        bundle = load_whisper(cfg)
        exclude_frames = int(round((cfg["alignment"]["exclude_boundary_ms"] / 1000)
                                   / bundle.encoder_step_sec))
        manifest = load_subset(cfg, subset)
        manifest = manifest[manifest["contains_code_switch"]]
        cap = int(args.limit or ecfg.get("max_utterances", 1000))
        manifest = manifest.head(cap)
        units = load_units(cfg, subset)
        log.info("E3 on %s: %d utterances", subset, len(manifest))

        if args.dry_run:
            return 0

        data = collect_projection_data(bundle, manifest, units, layers, exclude_frames,
                                       int(ecfg.get("batch_size", 8)), log)
        dir_root = art(cfg, "directions")
        rows: list[dict] = []

        for layer in layers:
            matrix, meta = data[layer]
            if not len(matrix):
                continue
            labels = meta["label"].to_numpy()
            speakers = meta["speaker_id"].to_numpy()
            utts = meta["utterance_id"].to_numpy()
            non_boundary = ~meta["is_boundary"].to_numpy()

            variants = (
                [("within_utterance_correct_only", s) for s in seeds]
                + [("within_utterance_all_valid", seeds[0]),
                   ("global_centroid_correct_only", seeds[0]),
                   ("wrong_sign", seeds[0])]
                + [(f"random_s{s}", seeds[0]) for s in random_seeds]
            )
            for variant, seed in variants:
                try:
                    rec = load_direction(dir_root, "encoder", layer, variant, seed)
                except FileNotFoundError:
                    log.warning("missing direction %s L%d seed%d", variant, layer, seed)
                    continue
                d = rec["direction"].numpy().astype(np.float32)
                scores = matrix @ d
                thr = rec["midpoint_projection"]
                if not np.isfinite(thr):
                    thr = float(np.median(scores))

                for scope, mask in (("non_boundary", non_boundary),
                                    ("all_frames", np.ones(len(labels), dtype=bool))):
                    if mask.sum() == 0:
                        continue
                    m = separability_metrics(
                        scores[mask], labels[mask], float(thr), speakers[mask], utts[mask],
                        int(ecfg.get("bootstrap_resamples", 1000)),
                        int(ecfg.get("bootstrap_seed", 342)))
                    rows.append({"layer": layer, "direction": variant, "seed": seed,
                                 "scope": scope, **m})

        results = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["layer", "direction", "seed", "scope", "auroc", "auprc", "macro_f1",
                     "en_recall", "speaker_macro_auroc", "speaker_bootstrap_ci_low",
                     "speaker_bootstrap_ci_high", "num_frames"])
        results.to_parquet(art(cfg, "metrics", f"{stage}_layer_metrics.parquet"), index=False)
        results.to_parquet(art(cfg, "metrics", "e3_layer_metrics.parquet"), index=False)

        # ---- layer ranking (primary variant, non-boundary frames) ------
        prim = results[(results["direction"] == "within_utterance_correct_only")
                       & (results["scope"] == "non_boundary")]
        summary_rows = []
        for layer, grp in prim.groupby("layer"):
            rand = results[(results["layer"] == layer) & (results["scope"] == "non_boundary")
                           & results["direction"].str.startswith("random_")]
            wrong = results[(results["layer"] == layer) & (results["scope"] == "non_boundary")
                            & (results["direction"] == "wrong_sign")]
            summary_rows.append({
                "layer": int(layer),
                "auroc_mean": float(grp["auroc"].mean()),
                "auroc_seed_std": float(grp["auroc"].std(ddof=0)) if len(grp) > 1 else 0.0,
                "speaker_macro_auroc": float(grp["speaker_macro_auroc"].mean()),
                "en_recall": float(grp["en_recall"].mean()),
                "ci_low": float(grp["speaker_bootstrap_ci_low"].mean()),
                "ci_high": float(grp["speaker_bootstrap_ci_high"].mean()),
                "random_auroc_mean": float(rand["auroc"].mean()) if len(rand) else float("nan"),
                "wrong_sign_auroc": float(wrong["auroc"].mean()) if len(wrong) else float("nan"),
            })
        summary = pd.DataFrame(summary_rows, columns=[
            "layer", "auroc_mean", "auroc_seed_std", "speaker_macro_auroc", "en_recall",
            "ci_low", "ci_high", "random_auroc_mean", "wrong_sign_auroc"])
        if len(summary):
            summary = summary.sort_values(["speaker_macro_auroc", "auroc_mean"],
                                          ascending=False)

        keep = int(ecfg.get("keep_top_layers", 2))
        selected = [int(r) for r in summary["layer"].head(keep)]
        selection_payload = {
                "selected_layers": selected,
                "criterion": "speaker-macro AUROC, then frame-micro AUROC; "
                             "final layer is decided by E4 correction-versus-harm",
                "ranking": summary.to_dict(orient="records"),
                "phase": args.phase,
        }
        art(cfg, "reports", f"{stage}_selected_layers.json").write_text(
            json.dumps(selection_payload, indent=2, default=str), encoding="utf-8")
        art(cfg, "reports", "e3_selected_layers.json").write_text(
            json.dumps(selection_payload, indent=2, default=str), encoding="utf-8")

        best_row = summary.iloc[0] if len(summary) else None
        best_auroc = float(best_row["auroc_mean"]) if best_row is not None else float("nan")
        max_seed_std = float(best_row["auroc_seed_std"]) if best_row is not None else float("nan")
        beats_random = (bool(best_row["auroc_mean"] > best_row["random_auroc_mean"] + 0.05)
                        if best_row is not None and np.isfinite(best_row["random_auroc_mean"])
                        else False)
        sign_ok = (bool(abs((best_row["auroc_mean"] + best_row["wrong_sign_auroc"]) - 1.0) < 0.02)
                   if best_row is not None and np.isfinite(best_row["wrong_sign_auroc"]) else False)

        criteria = [
            criterion("best_layer_auroc", best_auroc, ecfg["pass_auroc"],
                      best_auroc >= ecfg["pass_auroc"]),
            criterion("max_seed_std", max_seed_std, ecfg["max_seed_std"],
                      max_seed_std <= ecfg["max_seed_std"], "<="),
            criterion("beats_random_controls", best_auroc,
                      float(best_row["random_auroc_mean"]) if best_row is not None else float("nan"),
                      beats_random),
            criterion("sign_consistency (auroc + wrong_sign_auroc == 1)",
                      float(best_row["auroc_mean"] + best_row["wrong_sign_auroc"])
                      if best_row is not None else float("nan"), 1.0, sign_ok, "=="),
        ]
        conditional = (ecfg["conditional_auroc"] <= best_auroc < ecfg["pass_auroc"])
        g = gate(stage, all(c["passed"] for c in criteria), criteria,
                 ("EXPLORATORY/CONDITIONAL: AUROC is in [0.70, 0.80); inspect alignment and "
                  "direction variants before E4." if conditional else
                  "AUROC < 0.70 is a failure unless a documented alignment defect explains it."
                  if best_auroc < ecfg["conditional_auroc"] else
                  "Top layers are frozen for the E4 pilot; the final layer is chosen by E4."))
        g["conditional"] = conditional
        g["selected_layers"] = selected

        art(cfg, "metrics", f"{stage}_summary.json").write_text(
            json.dumps({"summary": summary.to_dict(orient="records"),
                        "selected_layers": selected, "gate": g}, indent=2, default=str),
            encoding="utf-8")
        save_report(
            art(cfg, "reports", f"{stage}_separability.md"),
            f"E3 — language-direction separability ({args.phase})",
            [
                ("Per-layer summary (primary direction, non-boundary frames)", md_table(summary)),
                ("All directions", md_table(results[results["scope"] == "non_boundary"][
                    ["layer", "direction", "seed", "auroc", "auprc", "macro_f1",
                     "en_recall", "speaker_macro_auroc"]])),
                ("Boundary vs non-boundary", md_table(
                    results[results["direction"] == "within_utterance_correct_only"]
                    .groupby("scope")[["auroc", "macro_f1", "num_frames"]].mean().reset_index())),
                ("Selected layers", f"`{selected}` (kept for the E4 pilot)"),
                ("Gate", "```json\n" + json.dumps(g, indent=2, default=str) + "\n```"),
            ],
        )
        finish(cfg, stage, rdir, {"summary": summary.to_dict(orient="records"),
                                  "selected_layers": selected}, g)
        log.info("E3 gate: %s (best AUROC %.4f)", "PASSED" if g["passed"] else "FAILED", best_auroc)
        return 0 if g["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
