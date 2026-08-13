"""L1a - alignment diagnostics. Produces evidence, never a verdict.

    sbatch cs_asr_lss.sh l1a --only qwen
    sbatch cs_asr_lss.sh l1a

Whether the alignment is good enough is Gate A's decision, in l1b_valid, and it
requires compatible external lexical reference evidence. This stage exists to explain *why*
the two aligners disagree, so that Gate A is faced with a repaired aligner
rather than an unexplained one. Its gate therefore checks that every diagnostic
ran and reproduces; coverage numbers are reported and never thresholded.

The stage is also structurally barred from improving agreement: the only
correction it computes is an explicit counterfactual, marked not-applied,
because an offset fitted to inter-aligner agreement optimizes the statistic it
would then be judged by.

Two things it *does* decide, both on development-only coordinate evidence:

* it renders the synthetic **development** set (with known audio seams, not
  lexical boundaries) and publishes it authenticated, so l1b reads the same
  configured pool rather than rendering its own;
* it runs the configured `pred_start_offset` conventions over that set, scores
each relative to the constructed seam, and freezes the selected variant in
  `freeze/l1a_alignment_selection.json`. This experiment was previously deferred
  to l1b, which requires l1a -- so it could never run at all.

Selecting a convention on development data is not judging it. The gate set is
untouched by everything here, and Gate A's accuracy thresholds still have to be
cleared on data no selection has seen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.align import bias, candidates as cand, ladder as ladder_mod
from ..lss.align import qwen_probe, repair, romanization, summaries
from ..lss.artifacts import write_json, write_parquet
from ..lss.gates import check, evaluate, exit_code, reported
from ..lss.prereq import require_prerequisites
from ..lss.roles import load_role
from ..lss.seeds import seed_for
from ..utils.config import art
from ..utils.status import StageLock
from ._common import (
    base_parser,
    claim_stage,
    finish,
    last_prepare_context as _last_prepare_context,
    md_table,
    prepare,
    save_report,
    terminal_on_exception,
)

STAGE = "l1a_diag"
PARTS = ("ladder", "bias", "repair", "ctc", "uroman", "qwen", "summary")


def _load_nat5h_candidates(cfg: dict) -> pd.DataFrame:
    path = Path(str(((cfg.get("alignment") or {}).get("diagnostics") or {})
                    .get("nat5h_candidates", "")))
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _ctc_vocab(cfg: dict) -> dict | None:
    try:
        from ..data.ctc_alignment import load_ctc_aligner

        return load_ctc_aligner(cfg).vocab()
    except Exception:                       # the probe still runs without it
        return None


def _select_aligner_configuration(cfg: dict, log, rdir, taint: dict, *,
                                  qwen: dict | None = None) -> dict:
    """Build the development set, sweep the decoder-query convention, freeze it.

    Why this lives here and not in l1b
    ----------------------------------
    `bias.pred_start_sweep` existed but was never executable: it needs synthetic
    development data, which was built by l1b, which requires l1a. The experiment
    was stranded across a dependency boundary, so the ~700 ms Whisper bias could
    be described in a comment and never tested. The development set is
    prerequisite-safe -- it needs the D-construct role manifest, which L0 froze --
    so it is built here, published with a manifest, and reused by l1b for the
    operating-tolerance selection.

    Nothing in here judges lexical accuracy. Both conventions are legitimate
    readings of the same attention; the sweep says which one places predicted
    edges nearest the constructed audio seam. Gate A still requires genuine
    lexical calibration and must clear the proposal's thresholds afterwards.
    """
    from ..lss.align import devselect
    from ..models.whisper import load_whisper
    from ..nat5h.schema import RunIdentity

    root = Path(cfg["experiment"]["output_root"])
    acfg = (cfg.get("alignment") or {})
    gcfg = cfg.get("gate_a") or {}
    offsets = list(((acfg.get("dtw_variants") or {}).get("pred_start_offsets")
                    or [-1, 0]))

    dev, dev_meta = devselect.build_dev_set(
        cfg, stage=STAGE, run_dir=rdir, taint_reasons=taint["taint_reasons"])
    log.info("synthetic development set: %d pairs from %d source utterances "
             "(sha256 %s)", len(dev), len(dev_meta.get("source_utterances") or []),
             str(dev_meta.get("sha256"))[:12])
    out: dict = {"development_artifact": {
        "path": dev_meta.get("path"), "sha256": dev_meta.get("sha256"),
        "pairs": int(len(dev)), "fingerprint": dev_meta.get("fingerprint"),
        "settings": dev_meta.get("settings"),
    }, "offsets": offsets}
    if not len(dev):
        log.error("no synthetic development pairs were rendered; the pred_start "
                  "sweep has nothing to score against")
        out["whisper"] = {"pred_start_offset": None,
                          "rejected": {"*": ["no development pairs"]}}
        return out

    bundle = load_whisper(cfg)
    sweep = devselect.whisper_variant_sweep(
        bundle, dev, cfg, RunIdentity.from_cfg(cfg), offsets=offsets,
        out_dir=root / "synthetic" / "dev" / "alignments")
    sweep_path = art(cfg, "metrics", "l1a_pred_start_sweep.parquet")
    write_parquet(sweep, sweep_path)
    log.info("pred_start sweep against typed development references:\n%s",
             sweep.to_string(index=False) if len(sweep) else "(empty)")

    whisper = devselect.select_whisper_variant(
        sweep, min_boundaries=int(gcfg.get("min_pred_start_boundaries", 50)))
    log.info("selected decoder-query convention: offset=%s variant=%s "
             "(default %s, changed=%s)", whisper.get("pred_start_offset"),
             whisper.get("aligner_variant"),
             whisper.get("default_pred_start_offset"),
             whisper.get("changed_the_default"))

    payload = devselect.aligner_selection(
        whisper,
        # the probe's frozen language, recorded here so one artifact answers
        # "which variants produced this alignment" for every family
        qwen_language=((qwen or {}).get("language_diagnostics") or [{}])[0]
        .get("language") if qwen else None,
        dev_artifact=out["development_artifact"],
        seeds={"synthetic_dev": seed_for(cfg, "synthetic_dev"),
               "synthetic_gate": seed_for(cfg, "synthetic_gate")},
        thresholds={"min_pred_start_boundaries":
                    int(gcfg.get("min_pred_start_boundaries", 50))})
    selection_path = root / devselect.ALIGNER_SELECTION_FILE
    write_json(payload, selection_path)
    manifest_mod.publish(selection_path, None, stage=STAGE, cfg=cfg, run_dir=rdir,
                         taint_reasons=taint["taint_reasons"],
                         schema=devselect.ALIGNER_SELECTION_SCHEMA)
    out.update({"sweep": sweep.to_dict(orient="records"),
                "sweep_path": str(sweep_path), "whisper": whisper,
                "selection_path": str(selection_path), "payload": payload})
    return out


def _run(argv: list[str] | None = None) -> int:
    parser = base_parser("L1a - alignment diagnostics", "lss/l1a_diag.yaml")
    parser.add_argument("--only", default=None,
                        help=f"comma-separated subset of {','.join(PARTS)}")
    parser.add_argument("--sample", type=int, default=None,
                        help="override alignment.diagnostics.sample_utterances")
    args = parser.parse_args(argv)
    if args.output_dir:
        raise SystemExit("--output-dir is not supported for LSS stages")

    cfg, rdir, log = prepare(args, STAGE, claim=False)
    root = cfg["experiment"]["output_root"]
    parts = set(PARTS if not args.only else
                [p.strip() for p in args.only.split(",") if p.strip()])
    unknown = parts - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown --only part(s): {sorted(unknown)}")

    dcfg = (cfg.get("alignment") or {}).get("diagnostics") or {}
    gcfg = cfg.get("gate") or {}

    with StageLock(root, STAGE):
        claim_stage(cfg, STAGE, rdir)
        prereq = require_prerequisites(root, STAGE, force=args.force_prereq,
                                       overwrite=args.overwrite, log=log)
        metrics: dict = {"prerequisites": prereq, "parts": sorted(parts),
                         "taint": prereq["taint"]}
        criteria: list[dict] = []

        # Diagnostics run first on the candidate table the previous exploratory
        # run already produced: it costs no GPU time and can settle the
        # overlap question before any model is loaded.
        nat5h = _load_nat5h_candidates(cfg)
        metrics["nat5h_candidate_rows"] = int(len(nat5h))

        # ---- coordinate attribution ladder ---------------------------------
        if "ladder" in parts and len(nat5h):
            table, evidence = ladder_mod.run_ladder(nat5h)
            write_parquet(table, art(cfg, "metrics", "l1a_ladder.parquet"))
            write_json(evidence, art(cfg, "diagnostics", "l1a_ladder.json"))
            metrics["ladder"] = evidence
            log.info("coordinate hypothesis verdict: %s (slope %.2f ms/s)",
                     evidence["verdict"],
                     evidence["regression"].get("slope_ms_per_s", float("nan")))
            criteria += [
                check("ladder_rungs_completed", evidence["rungs_completed"],
                      int(gcfg.get("min_ladder_rungs", 4)), ">="),
                check("coordinate_hypothesis_verdict_recorded",
                      int(evidence["verdict"] in ("supported", "partially_supported",
                                                  "refuted", "inconclusive")),
                      1, "=="),
                reported("coordinate_hypothesis_verdict", evidence["verdict"]),
                reported("disagreement_theil_slope_ms_per_s",
                         evidence["regression"].get("theil_slope_ms_per_s")),
                reported("disagreement_r", evidence["regression"].get("r")),
                reported("statistic_effect_ms", evidence.get("statistic_effect_ms")),
                reported("residual_disagreement_ms",
                         evidence.get("residual_disagreement_ms")),
            ]

        # ---- signed bias ----------------------------------------------------
        if "bias" in parts and len(nat5h):
            pairs = bias.paired_edges(nat5h)
            write_parquet(pairs, art(cfg, "metrics", "l1a_paired_edges.parquet"))
            summary = bias.signed_bias_summary(pairs, by=("language",))
            write_parquet(summary, art(cfg, "metrics", "l1a_bias_summary.parquet"))
            counterfactual = bias.offset_correction_counterfactual(pairs)
            write_parquet(counterfactual,
                          art(cfg, "metrics", "l1a_offset_counterfactual.parquet"))
            metrics["bias"] = {
                "paired_units": int(len(pairs)),
                "summary": summary.to_dict(orient="records"),
                "counterfactual_note": "reported, never applied in this stage",
            }
            for _, row in summary.iterrows():
                criteria.append(reported(
                    f"median_signed_{row['edge']}_ms_{row['language']}",
                    row["median_signed_ms"]))

        # ---- adjacent-overlap inventory and repair --------------------------
        if "repair" in parts and len(nat5h):
            policy = repair.OverlapRepairPolicy.from_cfg(
                (cfg.get("alignment") or {}).get("repair"))
            inventory = repair.overlap_inventory(nat5h, policy=policy)
            write_parquet(inventory, art(cfg, "metrics", "l1a_overlap_inventory.parquet"))
            repaired, effect = repair.repair_adjacent_overlaps(nat5h, policy)
            effect["benign_fraction"] = (float(inventory["benign"].mean())
                                         if len(inventory) else float("nan"))
            effect["overlaps_found"] = int(len(inventory))
            effect["cross_language_fraction"] = (float(inventory["cross_language"].mean())
                                                 if len(inventory) else float("nan"))
            write_json(effect, art(cfg, "metrics", "l1a_repair_effect.json"))
            metrics["repair"] = effect
            log.info("overlaps: %d found, %.2f benign (<=1 frame), %d repaired",
                     effect["overlaps_found"], effect.get("benign_fraction", float("nan")),
                     effect["repaired"])
            # Why they overlap, not just how many. On the production table this
            # attributes the whole Whisper invalid rate to reference units that
            # share one Whisper token, which is not repairable by truncation --
            # see `repair.attribute_overlaps`.
            attribution, attribution_report = repair.attribute_overlaps(nat5h)
            write_parquet(attribution,
                          art(cfg, "metrics", "l1a_overlap_attribution.parquet"))
            write_json(attribution_report,
                       art(cfg, "diagnostics", "l1a_overlap_attribution.json"))
            metrics["overlap_attribution"] = attribution_report
            log.info("overlap attribution (%s): %s of %d overlapping rows; "
                     "identical spans %.3f", attribution_report["family"],
                     attribution_report["causes"],
                     attribution_report["overlaps"],
                     attribution_report["identical_span_fraction"])
            criteria += [
                check("overlap_inventory_written", 1, 1, "=="),
                check("overlap_repair_measured", 1, 1, "=="),
                check("overlap_causes_attributed",
                      int(bool(attribution_report["overlaps"] == 0
                               or attribution_report["causes"])), 1, "=="),
                reported("overlaps_found", effect["overlaps_found"]),
                reported("benign_overlap_fraction", effect.get("benign_fraction")),
                reported("overlap_causes", json.dumps(attribution_report["causes"])),
                reported("overlap_identical_span_fraction",
                         attribution_report["identical_span_fraction"]),
                reported("median_boundary_movement_ms",
                         effect.get("median_boundary_movement_ms")),
            ]

        # ---- CTC mapping failures and romanization --------------------------
        if ("ctc" in parts or "uroman" in parts) and len(nat5h):
            vocab = _ctc_vocab(cfg)
            characterized = romanization.characterize_mapping_failures(nat5h, vocab)
            write_parquet(characterized,
                          art(cfg, "metrics", "l1a_ctc_mapping_failures.parquet"))
            # uroman is slow per call; a few hundred distinct surfaces already
            # characterizes the expansion and out-of-vocabulary behaviour
            surfaces = (nat5h["reference_text"].astype(str).drop_duplicates()
                        .head(int(dcfg.get("uroman_surfaces", 400))).tolist())
            probe = romanization.probe_uroman(surfaces, vocab)
            write_parquet(probe, art(cfg, "metrics", "l1a_uroman_probe.parquet"))
            summary = romanization.romanization_summary(probe)
            causes = romanization.cause_counts(characterized)
            write_json({"summary": summary, "cause_counts": causes},
                       art(cfg, "diagnostics", "l1a_romanization.json"))
            metrics["romanization"] = {"summary": summary, "cause_counts": causes}
            explained = (1.0 if not len(characterized)
                         else float((characterized["cause"] != "unexplained").mean()))
            criteria += [
                check("mapping_failures_characterized_rate", explained,
                      float(gcfg.get("require_mapping_failure_causes", 1.0)), ">="),
                check("uroman_probe_completed", int(len(probe) > 0), 1, "=="),
                reported("mapping_failure_causes", json.dumps(causes)),
                reported("median_romanization_expansion",
                         summary.get("median_expansion_ratio")),
            ]

        # ---- per-family, per-language validity ------------------------------
        if "summary" in parts and len(nat5h):
            family = summaries.family_language_summary(nat5h)
            # l1b declares this as a prerequisite artifact, so it is published
            # with a manifest that binds it to this run and its taint
            manifest_mod.publish_frame(
                art(cfg, "metrics", "l1a_family_language_summary.parquet"), family,
                stage=STAGE, cfg=cfg, run_dir=rdir,
                taint_reasons=prereq["taint"]["taint_reasons"],
                key_columns=("aligner_family", "aligner_variant",
                             "reference_language"))
            stratified = summaries.stratified_validity(nat5h)
            write_parquet(stratified, art(cfg, "metrics",
                                          "l1a_stratified_validity.parquet"))
            # Reproduce the *candidate*-level numbers the previous run recorded.
            # Consensus counts belong to l1b: comparing candidate rows against a
            # recorded consensus-span count would compare two different things.
            observed = {}
            recorded = {}
            recorded_path = Path(str(dcfg.get("nat5h_aligner_summary", "")))
            if recorded_path.is_file():
                previous = pd.read_parquet(recorded_path)
                for _, row in previous.iterrows():
                    key = f"valid_units_{row['aligner_family']}"
                    recorded[key] = int(row["valid_units"])
                    current = family[family["aligner_family"] == row["aligner_family"]]
                    observed[key] = int(current["valid_units"].sum()) if len(current) else 0
            reproduction = summaries.reproduction_check(
                observed, recorded, float(dcfg.get("reproduction_tolerance", 0.02)))
            reproduction["compared"] = sorted(recorded)
            write_json(reproduction, art(cfg, "diagnostics",
                                         "l1a_nat5h_reproduction.json"))
            metrics["family_language_summary"] = family.to_dict(orient="records")
            metrics["nat5h_reproduction"] = reproduction
            criteria += [
                check("family_language_summary_written", int(len(family) > 0), 1, "=="),
                check("nat5h_smoke_reproduced", int(bool(reproduction["reproduced"])),
                      int(bool(gcfg.get("require_nat5h_reproduction", True))), "=="),
            ]
            for _, row in family.iterrows():
                criteria.append(reported(
                    f"valid_unit_coverage_{row['aligner_family']}_{row['reference_language']}",
                    row["valid_unit_coverage"]))

        # ---- Qwen probe (isolated, never raises) ----------------------------
        if "qwen" in parts:
            manifest = load_role(cfg, str(dcfg.get("sample_role", "D-construct")))
            sample = manifest.head(int(((cfg.get("alignment") or {}).get("qwen") or {})
                                       .get("probe_utterances", 8)))
            # its own session, so the deadline can kill the whole process group;
            # a checkpoint load stuck in native CUDA code ignores SIGALRM
            result = qwen_probe.probe_qwen_subprocess(
                cfg, sample, out_dir=Path(root) / "alignments")
            write_json(result.to_dict(), art(cfg, "diagnostics", "l1a_qwen_probe.json"))
            metrics["qwen"] = result.to_dict()
            log.info("qwen probe: %s (%s) after %.1f s of a %.0f s deadline",
                     result.state, result.reason, result.elapsed_seconds,
                     result.deadline_seconds)
            # `failed` is deliberately not a terminal state here: it means the
            # probe hit a defect in our own code rather than a missing aligner
            criteria.append(check(
                "qwen_probe_state_terminal",
                int(result.state in tuple(gcfg.get("terminal_qwen_states",
                                                   ("ok", "blocked", "completed_no_go")))),
                1, "=="))
            criteria += [
                reported("qwen_probe_state", result.state),
                reported("qwen_probe_timed_out", int(bool(result.timed_out))),
                reported("qwen_probe_elapsed_seconds", result.elapsed_seconds),
                reported("qwen_probe_deadline_seconds", result.deadline_seconds),
                reported("qwen_probe_discarded_artifacts", len(result.quarantined)),
                reported("qwen_probe_published_candidates", len(result.published)),
                reported("qwen_probe_process_group",
                         json.dumps(result.process_group or {})),
            ]

        # ---- the decoder-query convention, selected on development data -----
        if "bias" in parts and not args.dry_run and args.sample != 0:
            selection = _select_aligner_configuration(
                cfg, log, rdir, prereq["taint"], qwen=metrics.get("qwen"))
            metrics["aligner_selection"] = selection
            sweep = selection.get("sweep") or []
            criteria += [
                # The sweep used to be a log line saying it was deferred to a
                # stage that depends on this one, so the experiment could never
                # run. These are mechanical: an absent artifact is a defect, not
                # a scientific negative.
                check("pred_start_sweep_offsets_scored", len(sweep),
                      int(gcfg.get("min_pred_start_offsets", 2)), ">="),
                check("pred_start_sweep_artifact_written",
                      int(bool(selection.get("sweep_path"))), 1, "=="),
                check("whisper_variant_selected",
                      int(selection.get("whisper", {}).get("pred_start_offset")
                          is not None), 1, "=="),
                check("aligner_selection_frozen",
                      int(bool(selection.get("selection_path"))), 1, "=="),
                check("development_set_published",
                      int(bool(selection.get("development_artifact", {})
                               .get("sha256"))), 1, "=="),
                reported("selected_pred_start_offset",
                         selection.get("whisper", {}).get("pred_start_offset")),
                reported("selected_aligner_variant",
                         selection.get("whisper", {}).get("aligner_variant")),
                reported("selection_instrument",
                         selection.get("whisper", {}).get("instrument")),
                reported("pred_start_sweep", json.dumps(sweep, default=str)),
            ]

        criteria.append(check("no_scientific_claim", 1, 1, "=="))

        gate_payload, status = evaluate(
            criteria, name=STAGE,
            note=("Diagnostics only. This stage explains the disagreement between "
                  "aligner families; whether the alignment is usable is decided by "
                  "Gate A in l1b_valid against compatible external lexical "
                  "reference evidence."))

        rows = criteria_frame(criteria)
        save_report(
            art(cfg, "reports", "l1a_diagnostics.md"), "L1a - alignment diagnostics",
            [
                ("Coordinate ladder", "```json\n" + json.dumps(
                    metrics.get("ladder", {}), indent=2, default=str) + "\n```"),
                ("Signed bias", md_table(pd.DataFrame(
                    (metrics.get("bias") or {}).get("summary", [])))),
                ("Overlap repair", "```json\n" + json.dumps(
                    metrics.get("repair", {}), indent=2, default=str) + "\n```"),
                ("Romanization", "```json\n" + json.dumps(
                    metrics.get("romanization", {}), indent=2, default=str) + "\n```"),
                ("Per-family validity", md_table(pd.DataFrame(
                    metrics.get("family_language_summary", [])))),
                ("Gate", md_table(rows)),
            ],
        )
        write_parquet(rows, art(cfg, "metrics", "l1a_gate_criteria.parquet"))
        finish(cfg, STAGE, rdir, metrics, gate_payload,
               artifacts=[str(art(cfg, "reports", "l1a_diagnostics.md"))],
               status_override=status if status != "passed" else None,
               forced_prereq=bool(prereq.get("forced_run")),
               # A dry run skips the development set and the convention sweep, so
               # it has not done L1b's prerequisite work whatever its own checks
               # say. `--only` was already excluded for the same reason.
               full_stage_pass=bool(status == "passed" and not args.only
                                    and not args.dry_run),
               taint=prereq["taint"])
        log.info("L1a gate: %s", status.upper())
        return exit_code(status)


def criteria_frame(criteria: list[dict]) -> pd.DataFrame:
    """Gate criteria as a table whose cells are all strings.

    `_common.md_table` only formats columns with a float dtype; a mixed
    object column of floats and strings reaches `" ".join` and raises.
    """
    rows = []
    for c in criteria:
        rows.append({
            "group": str(c.get("group", "")),
            "name": str(c.get("name", "")),
            "value": ("" if c.get("value") is None else
                      (f"{c['value']:.4f}" if isinstance(c.get("value"), float)
                       else str(c.get("value")))),
            "comparison": str(c.get("comparison", "")),
            "threshold": str(c.get("threshold", "")),
            "passed": str(bool(c.get("passed", False))),
        })
    return pd.DataFrame(rows, columns=["group", "name", "value", "comparison",
                                       "threshold", "passed"])


def main(argv: list[str] | None = None) -> int:
    """Terminal in every path.

    An unhandled exception used to leave the status at `running`, which every
    prerequisite check reads as "a job is still working on it" -- so the stage
    was neither runnable nor reported as broken, and the pipeline stalled.
    """
    try:
        return _run(argv)
    except SystemExit:
        raise
    except BaseException as exc:                       # noqa: BLE001
        cfg, rdir, log = _last_prepare_context()
        if cfg is None:
            raise                                      # nothing to record it in
        return terminal_on_exception(cfg, STAGE, rdir, log, exc)


if __name__ == "__main__":
    raise SystemExit(main())
