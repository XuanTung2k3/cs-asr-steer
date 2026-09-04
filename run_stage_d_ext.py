#!/usr/bin/env python
"""Stage D-extended: dense (layer x alpha x beam) landscape map for Track A.

    python run_stage_d_ext.py [--out out] [--resume] [--budget-hours N]
                              [--batch-size 8] [--skip-preflight]

Runs, in order:
  1. the existing 15-check Track A preflight (unmodified) -- must still pass
  2. the section 4.1 determinism check -- must pass before ANY sweep cell runs
  3. fresh regeneration of C00 / C00_beam5, asserted against the prior anchors
  4. Phase 1: the fixed 70-cell (layer x alpha x beam) grid, checkpointed to
     out/results_D_ext.jsonl, resumable by config hash
  5. the section 4.2 small-alpha sanity gate, checked before any interpretation
  6. Phase 1 analysis: heatmaps, plateau scores, connected components, the Q2
     paired beam/steering comparison, the Q3 chance count
  7. Phase 2, ONLY if Phase 1 found a connected component of size >= 4:
     fine layer resolution (2a) and matched-energy / matched-per-layer
     multi-layer pairs (2b) -- conditioned on the map, never pre-planned
  8. out/SUMMARY_D_EXT.md

This is a landscape map, not a search: no stage narrows to "the best cell".
D-dev-confirm is never opened and no Confirm candidate is proposed here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from steer_sweep import config as C                          # noqa: E402
from steer_sweep import metrics as M                          # noqa: E402
from steer_sweep import stage_d_ext as D                      # noqa: E402
from steer_sweep.store import Budget, ResultStore             # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(C.DEFAULT_OUT))
    p.add_argument("--config", default=C.BASE_CONFIG)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--budget-hours", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--skip-preflight", action="store_true",
                   help="NOT for a real run; exists only for harness development")
    p.add_argument("--min-component-size", type=int, default=4,
                   help="section 7: minimum connected-component size to "
                        "trigger Phase 2 (spec default 4)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.budget_hours)

    from csasr.utils.config import load_config
    from csasr.utils.logging import setup_logging
    from csasr.models.whisper import load_whisper

    log = setup_logging(level="INFO")
    cfg = load_config(args.config)

    log.info("stage D-extended | resume=%s budget=%s batch_size=%d",
             args.resume, args.budget_hours, args.batch_size)

    bundle = load_whisper(cfg)
    _assert_geometry(bundle)

    from steer_sweep import data as Dd
    from steer_sweep.directions import DirectionStore

    log.info("building the evaluation population")
    pop = Dd.build_population(bundle, cfg, C.DEV_SELECT, assert_anchors=True)
    log.info("population: %s", pop.counts)

    store = DirectionStore(cfg, out_dir / "directions_cache", log=log)

    state: dict[str, Any] = {"stage": "D_ext", "population": pop.counts,
                             "started_at": time.time()}

    # ---- 1. existing Track A preflight, unmodified, must still pass -------
    if not args.skip_preflight:
        from steer_sweep.preflight_a import format_preflight, run_preflight_a
        log.info("running the existing 15-check Track A preflight")
        preflight = run_preflight_a(bundle, pop, cfg, store,
                                    batch_size=args.batch_size, smoke=False, log=log)
        print(format_preflight(preflight), flush=True)
        (out_dir / "preflight_D_ext.json").write_text(
            json.dumps(preflight, indent=2, default=str), encoding="utf-8")
        state["preflight"] = preflight
        if not preflight["passed"]:
            failed = [c for c in preflight["checks"] if not c["passed"]]
            raise SystemExit(
                "EXISTING TRACK A PREFLIGHT FAILED -- halting before the "
                "determinism check or the sweep.\n" +
                "\n".join(f"  {c['check']}: observed {c['observed']!r}, "
                          f"expected {c['expected']!r}" for c in failed))
    else:
        state["preflight"] = {"skipped": True}

    # ---- 2. section 4.1 determinism check, FIRST, before any sweep cell ---
    log.info("running the section 4.1 determinism check (layer=%d, alpha=%g)",
             D.DETERMINISM_LAYER, D.DETERMINISM_ALPHA)
    determinism = D.run_determinism_check(
        bundle, pop, store, config=args.config, out_dir=out_dir,
        batch_size=args.batch_size, log=log)
    (out_dir / "determinism_D_ext.json").write_text(
        json.dumps(determinism, indent=2, default=str), encoding="utf-8")
    state["determinism"] = determinism
    log.info("determinism check: %s", "PASSED" if determinism["passed"] else "FAILED")
    # run_determinism_check already raises SystemExit on failure; reaching
    # here means it passed.

    # ---- 3. section 3, fresh baselines, asserted against the prior anchors
    log.info("regenerating C00 and C00_beam5 fresh")
    baselines = D.regenerate_baselines(bundle, pop, batch_size=args.batch_size, log=log)
    (out_dir / "baselines_D_ext.json").write_text(
        json.dumps({k: {"metrics": v["metrics"], "decode_health": v["decode_health"],
                        "num_beams": v["num_beams"]} for k, v in baselines.items()},
                   indent=2, default=str), encoding="utf-8")
    state["baselines"] = {k: v["metrics"] for k, v in baselines.items()}

    # ---- 4. Phase 1: the fixed 70-cell grid --------------------------------
    results = ResultStore(out_dir / "results_D_ext.jsonl", track="A")
    draws = C.BOOTSTRAP_DRAWS
    phase1_rows: list[dict] = []
    cells = D.phase1_cells()
    log.info("Phase 1: %d cells (%d layers x %d alphas x %d beam settings)",
             len(cells), len(D.PHASE1_LAYERS), len(D.PHASE1_ALPHAS), len(D.PHASE1_BEAMS))
    for i, cell in enumerate(cells):
        if args.resume and results.has(cell.config_hash()):
            phase1_rows.append(results.get(cell.config_hash()))
            log.info("resume: %s already done (%d/%d)", cell.cell_id, i + 1, len(cells))
            continue
        if budget.exhausted():
            log.warning("budget exhausted at cell %d/%d; stopping Phase 1 early",
                       i + 1, len(cells))
            break
        row = D.score_cell(bundle, pop, cell, store, baselines,
                           batch_size=args.batch_size, draws=draws, log=log)
        results.append(row)
        phase1_rows.append(row)
        log.info("  [%d/%d] %s: dMER %+.5f dPIER %+.5f corr %d corrupt(units) %d "
                 "corrupt(spill) %d dialogues %d -> %s",
                 i + 1, len(cells), cell.cell_id, row["delta_mer"], row["delta_pier"],
                 row["corrections"], row["corruptions_units"],
                 row["corruptions_spillover_damaged"],
                 row["dialogues_with_correction"], row["verdict"])

    state["phase1_n_rows"] = len(phase1_rows)
    if len(phase1_rows) < len(cells):
        state["phase1_incomplete"] = True
        _write_state(out_dir, state)
        log.warning("Phase 1 incomplete (%d/%d cells); stopping before analysis. "
                   "Rerun with --resume to continue.", len(phase1_rows), len(cells))
        return 0

    # ---- 5. section 4.2, before any interpretation -------------------------
    small_alpha = D.check_small_alpha_sanity(phase1_rows, log=log)
    state["small_alpha_sanity"] = small_alpha
    if not small_alpha["passed"]:
        _write_state(out_dir, state)
        raise SystemExit(
            "SMALL-ALPHA SANITY FAILED (section 4.2): scale_mode='act_norm' is "
            f"not keeping ||delta||/mean_activation_norm below "
            f"{D.SMALL_ALPHA_RATIO_THRESHOLD} at alpha=0.25 for: "
            f"{small_alpha['failing']}. Stopping before interpreting Phase 1.")

    s_audit = D.check_s_audit(phase1_rows, log=log)
    state["s_audit"] = s_audit

    # ---- 6. Phase 1 analysis ------------------------------------------------
    analysis = _run_analysis(phase1_rows, log=log)
    state["analysis"] = analysis
    _write_state(out_dir, state)

    # ---- 7. Phase 2, conditioned on Phase 1 --------------------------------
    big_components = D.select_components_for_phase2(
        phase1_rows, min_size=args.min_component_size)
    any_big = any(big_components[b] for b in D.PHASE1_BEAMS)
    state["phase2_triggered"] = any_big
    phase2_rows: list[dict] = []
    if any_big and not budget.exhausted():
        log.info("Phase 1 found a connected component >= %d cells; running Phase 2",
                 args.min_component_size)
        phase2_rows = _run_phase2(bundle, pop, store, baselines, results,
                                  big_components, args=args, budget=budget, log=log)
        state["phase2_n_rows"] = len(phase2_rows)
    else:
        log.info("no connected component of size >= %d in Phase 1: Phase 2 is NOT "
                 "run. The null map is the finding.", args.min_component_size)
        state["phase2_n_rows"] = 0

    state["finished_at"] = time.time()
    _write_state(out_dir, state)

    # ---- 8. summary ----------------------------------------------------------
    from steer_sweep.summary_d_ext import write_summary_d_ext
    path = write_summary_d_ext(out_dir)
    log.info("wrote %s", path)
    return 0


def _write_state(out_dir: Path, state: dict) -> None:
    (out_dir / "stage_d_ext_state.json").write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8")


def _run_analysis(rows: list[dict], *, log=None) -> dict:
    heatmaps = {
        f"delta_mer_beams{b}": D.heatmap(rows, b, field="delta_mer") for b in D.PHASE1_BEAMS
    }
    heatmaps["gain_b5_minus_gain_b1"] = D.heatmap_gain_difference(rows)
    plateau = {b: D.plateau_scores(rows, b) for b in D.PHASE1_BEAMS}
    components = {b: D.connected_components(rows, b) for b in D.PHASE1_BEAMS}
    q2 = D.q2_paired_analysis(rows)
    q3 = D.q3_chance_count(rows)
    largest = None
    for b in D.PHASE1_BEAMS:
        for c in components[b]:
            if largest is None or c["size"] > largest["size"]:
                largest = {**c, "beam_setting": b}
    if log:
        log.info("Phase 1 analysis: largest component size=%s (beams=%s), "
                 "Q2=%s, Q3 observed=%d expected~%.1f",
                 largest["size"] if largest else 0,
                 largest.get("beam_setting") if largest else None,
                 q2["composition"], q3["observed_go"], q3["expected_false_positives"])
    return {"heatmaps": heatmaps, "plateau_scores": plateau,
           "connected_components": components, "largest_component": largest,
           "q2_paired_analysis": q2, "q3_chance_count": q3}


def _run_phase2(bundle, pop, store, baselines, results, big_components, *,
                args, budget, log) -> list[dict]:
    rows: list[dict] = []

    def _run_one(cell) -> dict | None:
        if args.resume and results.has(cell.config_hash()):
            log.info("resume: %s already done", cell.cell_id)
            return results.get(cell.config_hash())
        if budget.exhausted():
            log.warning("budget exhausted; stopping Phase 2 early")
            return None
        row = D.score_cell(bundle, pop, cell, store, baselines,
                           batch_size=args.batch_size, draws=C.BOOTSTRAP_DRAWS, log=log)
        results.append(row)
        log.info("  phase2 %s: dMER %+.5f dPIER %+.5f -> %s",
                 cell.cell_id, row["delta_mer"], row["delta_pier"], row["verdict"])
        return row

    for beams, components in big_components.items():
        for component in components:
            log.info("Phase 2a: fine layer resolution for component size=%d "
                     "layers=%s alphas=%s (beams=%d)", component["size"],
                     component["layer_range"], component["alpha_range"], beams)
            for cell in D.phase2a_cells(component):
                row = _run_one(cell)
                if row is None:
                    return rows
                rows.append(row)

            log.info("Phase 2b: matched-energy / matched-per-layer multi-layer "
                     "pair for component layers=%s", component["layers"])
            for matched, per_layer in D.phase2b_cells(component):
                for cell in (matched, per_layer):
                    row = _run_one(cell)
                    if row is None:
                        return rows
                    rows.append(row)
    return rows


def _assert_geometry(bundle) -> None:
    checks = [
        ("encoder layers", bundle.num_encoder_layers, C.EXPECTED_ENCODER_LAYERS),
        ("decoder layers", bundle.num_decoder_layers, C.EXPECTED_DECODER_LAYERS),
        ("d_model", bundle.d_model, C.EXPECTED_D_MODEL),
        ("encoder frames", bundle.max_encoder_frames, C.EXPECTED_ENCODER_FRAMES),
    ]
    bad = [(n, o, e) for n, o, e in checks if int(o) != int(e)]
    if bad:
        raise SystemExit(
            "model geometry does not match the study's stated context:\n"
            + "\n".join(f"  {n}: observed {o}, expected {e}" for n, o, e in bad))


if __name__ == "__main__":
    raise SystemExit(main())
