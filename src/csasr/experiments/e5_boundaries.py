"""E5 — mask and boundary robustness (guide sections 47-54).

Uses the frozen E4 layer and alpha on `dev_confirm` only. Expansion tests
uncertainty absorbed by a wider mask; jitter tests genuinely mislocalized
boundaries. They are reported separately and never conflated.

Run:
    python -m csasr.experiments.e5_boundaries \
        --config configs/experiments/e5_boundaries.yaml --resume
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..directions.encoder import load_direction
from ..models.generation import decode_manifest
from ..models.whisper import load_whisper
from ..steering.encoder_hook import EncoderLocalSteering
from ..steering.masks import build_gain, derive_jitter_seed, spec_from_kind
from ..utils.config import art
from ..utils.status import StageLock, criterion, gate, require_passed
from ._common import (
    base_parser,
    finish,
    md_table,
    prepare,
    primary_baseline_id,
    save_report,
)
from .e4_oracle import (
    BASELINE_LANGUAGE,
    build_groups,
    correction_ratio_gate_pass,
    score_system,
)

STAGE = "e5"


def bootstrap_mean(values, n_resamples: int, seed: int, alpha: float = 0.05) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    point = float(arr.mean()) if len(arr) else float("nan")
    if len(arr) < 2:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "variance": float("nan"), "n_resamples": 0, "n": int(len(arr))}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats[i] = float(sample.mean())
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "variance": float(arr.var(ddof=1)), "n_resamples": int(n_resamples),
            "n": int(len(arr))}


def retained_gain_analysis(results: pd.DataFrame, exact_mask: str, *,
                           min_exact_gain: float, min_retained_gain: float,
                           bootstrap_resamples: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Compute retained gain only when exact-span gain is positive and nontrivial."""
    out = results.copy()
    pier_base = float(out["pier_baseline"].iloc[0]) if len(out) else float("nan")
    exact_rows = out[out["mask"] == exact_mask]
    pier_exact = float(exact_rows["pier"].mean()) if len(exact_rows) else float("nan")
    exact_gain = pier_base - pier_exact
    out["pier_abs_gain"] = pier_base - out["pier"]
    out["pier_abs_delta_vs_exact"] = out["pier"] - pier_exact
    exact_ok = bool(np.isfinite(exact_gain) and exact_gain >= float(min_exact_gain))
    if exact_ok:
        out["retained_gain"] = out["pier_abs_gain"] / exact_gain
        note = (f"reference mask {exact_mask}: exact PIER gain {exact_gain:.4f} "
                f"({pier_base:.4f} -> {pier_exact:.4f})")
    else:
        out["retained_gain"] = np.nan
        note = (f"exact-mask gain is {exact_gain:.4f}; retained gain is undefined "
                f"because the required denominator is >= {float(min_exact_gain):.4f}")

    by_mask = {}
    for mask, grp in out.groupby("mask"):
        vals = grp["retained_gain"].to_numpy(dtype=float)
        by_mask[str(mask)] = bootstrap_mean(vals, bootstrap_resamples, seed)
        by_mask[str(mask)]["pier_abs_gain_mean"] = float(grp["pier_abs_gain"].mean())
        by_mask[str(mask)]["pier_abs_gain_variance"] = (
            float(grp["pier_abs_gain"].var(ddof=1)) if len(grp) > 1 else float("nan")
        )

    j100 = by_mask.get("JITTER_100", {})
    return out, {
        "pier_baseline": pier_base,
        "pier_exact": pier_exact,
        "exact_gain": exact_gain,
        "exact_gain_ok": exact_ok,
        "min_exact_gain": float(min_exact_gain),
        "min_retained_gain": float(min_retained_gain),
        "retained_gain_by_mask": by_mask,
        "retained_gain_at_100ms": j100.get("point", float("nan")),
        "retained_gain_at_100ms_ci_low": j100.get("ci_low", float("nan")),
        "retained_gain_at_100ms_ci_high": j100.get("ci_high", float("nan")),
        "jitter_100_variance": j100.get("variance", float("nan")),
        "note": note,
    }


def plot_mask_examples(cfg, bundle, targets: pd.DataFrame, masks, n: int, log) -> str:
    """Sanity plots of the gain functions on real spans (guide section 49)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = art(cfg, "audit", "e5").parent / "e5"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = targets.head(n)
    for i, (_, r) in enumerate(sample.iterrows()):
        n_valid = bundle.valid_frames(r["duration_sec"])
        fig, ax = plt.subplots(figsize=(9, 3))
        for kind in masks:
            spec = spec_from_kind(kind, bundle.encoder_step_sec,
                                  shoulder_ms=float(cfg["e5"].get("shoulder_ms", 100)),
                                  boundary_only_ms=float(cfg["e5"].get("boundary_only_ms", 100)),
                                  seed=242)
            if kind.startswith("JITTER_"):
                spec.seed = derive_jitter_seed(spec.seed or 242, str(r["utterance_id"]))
            if kind == "WHOLE_WORD_PLUS_CONTEXT":
                spec.context_span = (int(r["context_start"]), int(r["context_end"]))
            g, _ = build_gain(spec, int(r["start_frame"]), int(r["end_frame"]),
                              bundle.max_encoder_frames, n_valid)
            lo = max(0, int(r["start_frame"]) - 30)
            hi = min(n_valid, int(r["end_frame"]) + 30)
            ax.plot(np.arange(lo, hi) * bundle.encoder_step_sec, g[lo:hi], label=kind, lw=1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("gain g_t")
        ax.set_title(f"{r['utterance_id']} POI '{r['surface']}'", fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"mask_example_{i:02d}.png", dpi=110)
        plt.close(fig)
    log.info("wrote %d mask example plots to %s", len(sample), out_dir)
    return str(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("E5 mask and boundary robustness", "experiments/e5_boundaries.yaml")
    args = parser.parse_args(argv)
    cfg, rdir, log = prepare(args, STAGE)
    root = cfg["experiment"]["output_root"]
    e5 = cfg["e5"]
    e4 = cfg["e4"]
    seed = int(cfg["experiment"]["seed"])
    resume, overwrite = args.resume or not args.overwrite, args.overwrite

    with StageLock(root, STAGE):
        if not args.force_prereq:
            require_passed(root, ["e1", "e2_full", "e3_full", "e4_confirm"])
        frozen_path = art(cfg, "metrics", "e4_selected_config.json")
        if not frozen_path.exists():
            raise SystemExit(
                f"E5 needs the configuration frozen by E4, but {frozen_path} does not "
                "exist. E4 found no feasible configuration; there is no mask robustness "
                "question to ask.")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        layer, alpha = int(frozen["layer"]), float(frozen["alpha"])
        log.info("frozen E4 configuration: layer=%d alpha=%.3f mask=%s",
                 layer, alpha, frozen.get("mask"))

        subset = e5.get("subset", "dev_confirm")
        wrong, correct = build_groups(cfg, subset, e4["confirm_wrong"],
                                      e4["confirm_correct"], seed, log)
        if args.limit:          # smoke-only knob; never set in production runs
            wrong, correct = wrong.head(args.limit), correct.head(args.limit)
        targets = pd.concat([wrong, correct]).reset_index(drop=True)
        for name, df in (("wrong", wrong), ("correct", correct)):
            df.to_parquet(art(cfg, "manifests", f"e5_{name}.parquet"), index=False)
        log.info("E5 on %s: %d targets", subset, len(targets))
        if args.dry_run:
            return 0

        bundle = load_whisper(cfg)
        rec = load_direction(art(cfg, "directions"), "encoder", layer,
                             e4.get("direction_variant", "within_utterance_correct_only"),
                             int(e4.get("direction_seed", 42)))
        scale = float(rec["projection_std"])
        plot_dir = plot_mask_examples(cfg, bundle, targets, e5["masks"],
                                      int(e5.get("plot_examples", 10)), log)

        spans = {r["utterance_id"]: (int(r["start_frame"]), int(r["end_frame"]))
                 for _, r in targets.iterrows()}
        contexts = {r["utterance_id"]: (int(r["context_start"]), int(r["context_end"]))
                    for _, r in targets.iterrows()}
        language = BASELINE_LANGUAGE.get(primary_baseline_id(cfg))

        rows: list[dict] = []
        for kind in e5["masks"]:
            jitter_seeds = e5["jitter_seeds"] if kind.startswith("JITTER_") else [None]
            for js in jitter_seeds:
                spec = spec_from_kind(kind, bundle.encoder_step_sec,
                                      shoulder_ms=float(e5.get("shoulder_ms", 100)),
                                      boundary_only_ms=float(e5.get("boundary_only_ms", 100)),
                                      seed=js)
                builder = EncoderLocalSteering(bundle, layer, rec["direction"], alpha,
                                               scale, spec, spans, contexts,
                                               norm_preserve=bool(e4.get("norm_preserve", True)))
                tag = kind if js is None else f"{kind}_seed{js}"
                out_path = art(cfg, "predictions", "e5", kind, str(js or "na"),
                               f"L{layer}_a{alpha}.parquet")
                preds = decode_manifest(bundle, targets, cfg, system="ENC-LOCAL-1L",
                                        config_id=tag, out_path=out_path, language=language,
                                        hook_builder=builder, resume=resume, overwrite=overwrite)
                if builder.applied_meta:
                    pd.DataFrame(builder.applied_meta.values()).to_parquet(
                        out_path.with_name(out_path.stem + "_masks.parquet"), index=False)
                metrics, table = score_system(cfg, wrong, correct, preds,
                                              int(e5.get("bootstrap_resamples", 10000)),
                                              int(e5.get("bootstrap_seed", 342)))
                rows.append({"mask": kind, "jitter_seed": js, "layer": layer,
                             "alpha": alpha, **metrics})
                table.to_parquet(art(cfg, "predictions", "e5", "_tables",
                                     f"{tag}.parquet"), index=False)
                log.info("%s: pier=%.4f correction=%.3f corruption=%.3f",
                         tag, metrics["pier"], metrics["correction_rate"],
                         metrics["corruption_rate"])

        results = pd.DataFrame(rows)
        if "per_category_correction" in results.columns:
            results["per_category_correction"] = results["per_category_correction"].map(
                lambda v: json.dumps(v, default=str))
        results.to_parquet(art(cfg, "metrics", "e5_mask_results.parquet"), index=False)

        # ---- retained gain --------------------------------------------
        exact_mask = frozen.get("mask", "EXACT_TAPER")
        results, gain_summary = retained_gain_analysis(
            results, exact_mask,
            min_exact_gain=float(e5.get("min_exact_pier_gain_abs", 0.005)),
            min_retained_gain=float(e5["min_retained_gain_at_100ms"]),
            bootstrap_resamples=int(e5.get("bootstrap_resamples", 10000)),
            seed=int(e5.get("bootstrap_seed", 342)))
        pier_base = gain_summary["pier_baseline"]
        pier_exact = gain_summary["pier_exact"]
        gain_note = gain_summary["note"]

        agg = (results.groupby("mask")
               .agg(pier=("pier", "mean"), pier_std=("pier", "std"),
                    retained_gain=("retained_gain", "mean"),
                    pier_abs_gain=("pier_abs_gain", "mean"),
                    pier_abs_gain_variance=("pier_abs_gain", "var"),
                    correction=("correction_rate", "mean"),
                    corruption=("corruption_rate", "mean"),
                    outside_edits=("outside_edit_rate", "mean"),
                    ratio=("correction_to_corruption_ratio", "mean"),
                    n_runs=("pier", "size"))
               .reset_index())

        j100 = agg[agg["mask"] == "JITTER_100"]
        retained_100 = float(j100["retained_gain"].iloc[0]) if len(j100) else float("nan")
        retained_100_ci_low = float(gain_summary.get("retained_gain_at_100ms_ci_low",
                                                     float("nan")))
        jitter_std = float(results[results["mask"] == "JITTER_100"]["pier"].std(ddof=0)) \
            if (results["mask"] == "JITTER_100").any() else float("nan")
        hard = agg[agg["mask"] == "EXACT_HARD"]["corruption"]
        taper = agg[agg["mask"] == "EXACT_TAPER"]["corruption"]
        taper_ok = bool(len(hard) and len(taper) and taper.iloc[0] <= hard.iloc[0] + 1e-9)
        ratio_100 = float(j100["ratio"].iloc[0]) if len(j100) else float("nan")
        outside_100 = float(j100["outside_edits"].iloc[0]) if len(j100) else float("nan")
        j100_rows = results[results["mask"] == "JITTER_100"]
        ratio_100_ok = bool(len(j100_rows) and all(
            correction_ratio_gate_pass(r, e4["min_correction_corruption_ratio"])
            for _, r in j100_rows.iterrows()))

        criteria = [
            criterion("retained_gain_at_100ms_jitter", retained_100,
                      e5["min_retained_gain_at_100ms"],
                      bool(np.isfinite(retained_100)
                           and retained_100 >= e5["min_retained_gain_at_100ms"])),
            criterion("retained_gain_at_100ms_bootstrap_ci_low", retained_100_ci_low,
                      e5["min_retained_gain_at_100ms"],
                      bool(np.isfinite(retained_100_ci_low)
                           and retained_100_ci_low >= e5["min_retained_gain_at_100ms"])),
            criterion("exact_span_gain_positive_nontrivial", gain_summary["exact_gain"],
                      e5.get("min_exact_pier_gain_abs", 0.005),
                      bool(gain_summary["exact_gain_ok"])),
            criterion("correction_to_corruption_at_100ms", ratio_100,
                      e4["min_correction_corruption_ratio"],
                      ratio_100_ok, ">"),
            criterion("tapered_no_worse_than_hard",
                      float(taper.iloc[0]) if len(taper) else float("nan"),
                      float(hard.iloc[0]) if len(hard) else float("nan"), taper_ok, "<="),
            criterion("outside_region_edits_at_100ms", outside_100,
                      e4.get("max_outside_region_edit_rate",
                             e4.get("max_outside_edit_rate", 0.05)),
                      bool(np.isfinite(outside_100)
                           and outside_100 <= e4.get("max_outside_region_edit_rate",
                                                     e4.get("max_outside_edit_rate", 0.05))), "<="),
            criterion("jitter_seed_pier_std", jitter_std, 0.05,
                      bool(np.isfinite(jitter_std) and jitter_std <= 0.05), "<="),
        ]
        g = gate("e5", all(c["passed"] for c in criteria), criteria,
                 "If exact steering succeeds but +/-100 ms fails, the method is oracle-bound: "
                 "do not claim practical localization robustness. " + gain_note)

        art(cfg, "metrics", "e5_summary.json").write_text(
            json.dumps({"aggregate": agg.to_dict(orient="records"),
                        "pier_baseline": pier_base, "pier_exact": pier_exact,
                        "retained_gain": gain_summary, "gate": g},
                       indent=2, default=str), encoding="utf-8")
        save_report(
            art(cfg, "reports", "e5_boundary_robustness.md"),
            "E5 — mask and boundary robustness",
            [
                ("Frozen configuration",
                 f"layer {layer}, alpha {alpha}, evaluated on `{subset}` "
                 f"({len(wrong)} baseline-incorrect / {len(correct)} matched correct POIs)"),
                ("Per-mask results", md_table(agg)),
                ("Expansion vs boundary jitter",
                 "Expansion masks (`EXPAND_*`) test whether wider intervention "
                 "rescues boundary uncertainty. Jitter masks (`JITTER_*`) test true "
                 "boundary mislocalization. They are aggregated separately above and "
                 "must not be conflated."),
                ("Jitter seeds", md_table(results[results["mask"].str.startswith("JITTER")][
                    ["mask", "jitter_seed", "pier", "correction_rate", "corruption_rate"]])),
                ("Retained gain", "```json\n" + json.dumps(gain_summary, indent=2,
                                                           default=str) + "\n```"),
                ("Mask plots", f"`{plot_dir}`"),
                ("Gate", "```json\n" + json.dumps(g, indent=2, default=str) + "\n```"),
            ],
        )
        finish(cfg, STAGE, rdir, {"aggregate": agg.to_dict(orient="records"),
                                  "retained_gain": gain_summary}, g)
        log.info("E5 gate: %s", "PASSED" if g["passed"] else "FAILED")
        return 0 if g["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
