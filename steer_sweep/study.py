"""Orchestration for both tracks.

Track A pins to one GPU and runs the sweep; Track B pins to another and runs
training. They are independent processes and must not share a GPU: two training
arms (or a sweep and a training arm) on one device makes wall-clock and peak
memory meaningless.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import config as C
from . import metrics as M
from . import sweep as S
from .decode import build_utterance_plans, decode_health, decode_population, run_cell
from .directions import DirectionStore
from .store import Budget, ResultStore


# ===========================================================================
# Track A
# ===========================================================================

def _direction_for(cell: S.Cell, store: DirectionStore, bundle, pop):
    """Resolve the direction(s) a cell needs. Raises on a decoder-only kind
    requested at an encoder site -- never returns a silently substituted one."""
    encoder = decoder = None
    detail: dict[str, Any] = {}
    if cell.site in (C.SITE_ENCODER, C.SITE_BOTH):
        encoder = store.get(C.KIND_V_NAT, C.SITE_ENCODER, cell.encoder_layers[0],
                            bundle=bundle, population=pop)
        detail["encoder_direction_source"] = encoder.source
    if cell.site in (C.SITE_DECODER, C.SITE_BOTH):
        layer = cell.decoder_layers[0]
        if cell.combo_mode:
            v_nat = store.get(C.KIND_V_NAT, C.SITE_DECODER, layer, bundle=bundle,
                              population=pop)
            v_prompt = store.get(C.KIND_V_PROMPT, C.SITE_DECODER, layer,
                                 bundle=bundle, population=pop)
            decoder, combo = store.combined(v_nat, v_prompt, mode=cell.combo_mode,
                                            weight_nat=cell.weight_nat,
                                            weight_prompt=cell.weight_prompt)
            detail["combination"] = combo
        else:
            kind = (C.KIND_V_PROMPT if cell.direction_kind == C.KIND_V_PROMPT
                    else cell.direction_kind)
            decoder = store.get(kind, C.SITE_DECODER, layer, bundle=bundle,
                                population=pop)
        detail["decoder_direction_source"] = decoder.source
    return encoder, decoder, detail


def _score_cell(pop, baseline_texts: dict[str, str], result, energy,
                cell: S.Cell, *, draws: int) -> dict[str, Any]:
    health = decode_health(pop, result)
    corpus = M.corpus_metrics(pop, result.texts)
    base_corpus = M.corpus_metrics(pop, baseline_texts)
    outcomes = M.target_outcomes(pop, baseline_texts, result.texts)
    risk = M.corruption_and_retention(pop, baseline_texts, result.texts)
    delta = M.delta_pier(pop, baseline_texts, result.texts, draws=draws)
    gates = M.evaluate_gates(delta=delta, outcomes=outcomes, risk=risk, health=health)
    spillover = {
        "spillover_units_changed": int(outcomes["spillover_changed"].sum()) if len(outcomes) else 0,
        "spillover_units_improved": int(outcomes["spillover_improved"].sum()) if len(outcomes) else 0,
        "spillover_units_damaged": int(outcomes["spillover_damaged"].sum()) if len(outcomes) else 0,
        "utterances_with_spillover": int(
            outcomes.loc[outcomes["spillover_changed"] > 0, "utterance_id"].nunique())
        if len(outcomes) else 0,
    }
    return {
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "stage": cell.stage,
        "site": cell.site,
        "coverage": cell.coverage,
        "direction_kind": cell.direction_kind,
        "combo_mode": cell.combo_mode,
        "weight_nat": cell.weight_nat,
        "weight_prompt": cell.weight_prompt,
        "alpha": cell.alpha,
        "encoder_layers": list(cell.encoder_layers),
        "decoder_layers": list(cell.decoder_layers),
        "scale_mode": cell.scale_mode,
        "matched_energy": cell.matched_energy,
        "num_beams": cell.num_beams,
        "language": cell.language,
        "note": cell.note,
        "energy": energy,
        "audit": result.audit,
        "decode_health": health,
        "metrics": corpus,
        "baseline_metrics": base_corpus,
        "delta_pier": delta["delta_pier"],
        "delta_pier_ci_low": delta["ci_low"],
        "delta_pier_ci_high": delta["ci_high"],
        "delta_pier_detail": delta,
        "risk": risk,
        "zh_retention": risk["zh_retention"],
        "poi_decomposition_steered": M.poi_decomposition(pop, result.texts),
        "poi_decomposition_baseline": M.poi_decomposition(pop, baseline_texts),
        "spillover": spillover,
        **{k: v for k, v in gates.items()},
        "wall_clock_sec": result.wall_clock_sec,
    }


def run_track_a(bundle, cfg, *, out_dir: Path, smoke: bool, resume: bool,
                budget: Budget, batch_size: int, log,
                full_weighted_grid: bool = False,
                skip_preflight: bool = False) -> dict[str, Any]:
    from . import data as D
    from .preflight_a import format_preflight, run_preflight_a

    out_dir.mkdir(parents=True, exist_ok=True)
    results = ResultStore(out_dir / "results_A.jsonl", track="A")
    draws = C.SMOKE["bootstrap_draws"] if smoke else C.BOOTSTRAP_DRAWS

    log.info("building the evaluation population")
    pop = D.build_population(bundle, cfg, C.DEV_SELECT, assert_anchors=not smoke)
    log.info("population: %s", pop.counts)
    full_counts = dict(pop.counts)
    if smoke:
        pop = pop.subsample(C.SMOKE["utterances"])
        log.info("SMOKE population: %s", pop.counts)

    store = DirectionStore(cfg, out_dir / "directions_cache", log=log)

    preflight = {"skipped": True}
    if not skip_preflight:
        log.info("running Track A preflight")
        preflight = run_preflight_a(bundle, pop, cfg, store, batch_size=batch_size,
                                    smoke=smoke, log=log)
        print(format_preflight(preflight), flush=True)
        (out_dir / "preflight_A.json").write_text(
            json.dumps(preflight, indent=2, default=str), encoding="utf-8")
        if not preflight["passed"]:
            failed = [c for c in preflight["checks"] if not c["passed"]]
            raise SystemExit(
                "TRACK A PREFLIGHT FAILED -- halting before any result is produced.\n"
                + "\n".join(f"  {c['check']}: observed {c['observed']!r}, "
                            f"expected {c['expected']!r}" for c in failed))

    # ---- baselines, first: they gate everything --------------------------
    baselines: dict[str, Any] = {}
    baseline_texts: dict[str, dict[str, str]] = {}
    for cell in S.baseline_cells():
        if resume and results.has(cell.config_hash()):
            row = results.get(cell.config_hash())
            baselines[cell.name] = row
            baseline_texts[cell.name] = row.get("_texts", {})
            log.info("resume: baseline %s already done", cell.name)
            continue
        if budget.exhausted():
            log.warning("budget exhausted before baseline %s", cell.name)
            break
        log.info("baseline %s (language=%s, beams=%d)", cell.name, cell.language,
                 cell.num_beams)
        result = decode_population(bundle, pop, language=cell.language,
                                   num_beams=cell.num_beams, batch_size=batch_size,
                                   log=log)
        health = decode_health(pop, result)
        row = {
            "cell_id": cell.cell_id, "config_hash": cell.config_hash(),
            "stage": "baseline", "baseline_name": cell.name,
            "language": cell.language, "num_beams": cell.num_beams,
            "note": cell.note, "metrics": M.corpus_metrics(pop, result.texts),
            "decode_health": health,
            "poi_decomposition": M.poi_decomposition(pop, result.texts),
            "wall_clock_sec": result.wall_clock_sec,
            "_texts": result.texts,
        }
        results.append(row)
        baselines[cell.name] = row
        baseline_texts[cell.name] = result.texts

    if C.PRIMARY_BASELINE not in baseline_texts:
        raise SystemExit("the primary baseline C00 did not complete; nothing "
                         "downstream can be scored against it")
    primary = baseline_texts[C.PRIMARY_BASELINE]

    # planned_decoder_positions is seeded at zero and immediately re-derived
    # from the primary baseline's actual text lengths below; no cached token
    # count exists on a baseline row to read here.
    baseline_plans = build_utterance_plans(
        bundle, pop, baseline_tokens={u: 0 for u in pop.utterance_ids},
        utterance_ids=pop.utterance_ids)
    for utterance, text in primary.items():
        if utterance in baseline_plans:
            baseline_plans[utterance].planned_decoder_positions = max(
                1, len(bundle.processor.tokenizer.encode(text, add_special_tokens=False)))

    # ---- staged sweep ----------------------------------------------------
    trace: list[dict[str, Any]] = []

    def run_stage(name: str, cells: Sequence[S.Cell]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell in cells:
            if resume and results.has(cell.config_hash()):
                rows.append(results.get(cell.config_hash()))
                log.info("resume: %s already done", cell.cell_id)
                continue
            if budget.exhausted():
                log.warning("budget exhausted during stage %s", name)
                break
            log.info("stage %s cell %s", name, cell.cell_id)
            try:
                enc, dec, detail = _direction_for(cell, store, bundle, pop)
            except Exception as exc:                   # never silently continue
                log.error("cell %s could not resolve a direction: %s",
                          cell.cell_id, exc)
                raise
            plan = cell.to_plan(encoder_direction=enc, decoder_direction=dec)
            comparator = (C.BEAM_BASELINE if cell.num_beams > 1 else C.PRIMARY_BASELINE)
            base = baseline_texts.get(comparator, primary)
            result, energy = run_cell(bundle, pop, plan=plan,
                                      utterance_plans=baseline_plans,
                                      language=cell.language, batch_size=batch_size,
                                      log=log)
            row = _score_cell(pop, base, result, energy, cell, draws=draws)
            row["compared_against"] = comparator
            row["direction_detail"] = detail
            results.append(row)
            rows.append(row)
            log.info("  %s: dPIER %+.5f [%+.5f, %+.5f] corr %d corrupt %d "
                     "dialogues %d retention %.4f -> %s",
                     cell.cell_id, row["delta_pier"], row["delta_pier_ci_low"],
                     row["delta_pier_ci_high"], row["corrections"],
                     row["corruptions"], row["dialogues_with_correction"],
                     row["zh_retention"], row["verdict"])
        return rows

    stage_a = run_stage("A", S.stage_a_cells())
    top_a, trace_a = M.select(stage_a, top=2)
    trace.append({"stage": "A", "selected": [r["cell_id"] for r in top_a], **trace_a})
    if not top_a:
        return _finish_a(results, trace, baselines, out_dir, preflight, full_counts,
                         store, reason="no Stage A cell survived the selection rule")

    stage_b = run_stage("B", S.stage_b_cells(top_a,
                                             full_weighted_grid=full_weighted_grid))
    top_b, trace_b = M.select(stage_b + top_a, top=1)
    trace.append({"stage": "B", "selected": [r["cell_id"] for r in top_b], **trace_b})
    if not top_b:
        return _finish_a(results, trace, baselines, out_dir, preflight, full_counts,
                         store, reason="no Stage B cell survived the selection rule")

    stage_c = run_stage("C", S.stage_c_cells(top_b[0]))
    top_c, trace_c = M.select(stage_c, top=1)
    trace.append({"stage": "C", "selected": [r["cell_id"] for r in top_c], **trace_c})
    if not top_c:
        return _finish_a(results, trace, baselines, out_dir, preflight, full_counts,
                         store, reason="no Stage C cell survived the selection rule")

    stage_d = run_stage("D", S.stage_d_cells(top_c[0]))
    top_d, trace_d = M.select(stage_d, top=1)
    trace.append({"stage": "D", "selected": [r["cell_id"] for r in top_d], **trace_d})

    confirm = None
    if top_d:
        cell = S.confirm_cell(top_d[0])
        confirm = {"cell_id": cell.cell_id, "config": asdict(cell),
                   "config_hash": cell.config_hash(),
                   "status": "AWAITING_HUMAN_APPROVAL",
                   "population": C.DEV_CONFIRM,
                   "statement": (
                       "Track A Confirm is NOT run automatically. D-dev-confirm "
                       "has never been opened and has ten dialogues to spend. "
                       "This is the single configuration the fixed selection "
                       "rule chose; a human authorises the run.")}
        (out_dir / "confirm_candidate.json").write_text(
            json.dumps(confirm, indent=2, default=str), encoding="utf-8")
        print("\n=== TRACK A CONFIRM CANDIDATE (NOT RUN) ===", flush=True)
        print(json.dumps(confirm, indent=2, default=str), flush=True)

    return _finish_a(results, trace, baselines, out_dir, preflight, full_counts,
                     store, confirm=confirm)


def _finish_a(results, trace, baselines, out_dir, preflight, counts, store,
              *, reason: str | None = None, confirm=None) -> dict[str, Any]:
    payload = {
        "track": "A", "trace": trace, "baselines": list(baselines),
        "preflight": preflight, "population": counts,
        "confirm_candidate": confirm, "stopped_early_reason": reason,
        "direction_provenance": store.provenance,
        "n_rows": len(results.rows()),
    }
    (out_dir / "track_a_state.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


# ===========================================================================
# Track B
# ===========================================================================

def run_track_b(bundle, cfg, *, out_dir: Path, smoke: bool, resume: bool,
                budget: Budget, batch_size: int, log,
                skip_preflight: bool = False,
                site: str | None = None,
                layers: Sequence[int] | None = None) -> dict[str, Any]:
    from . import data as D
    from .trackb import aga as AGA
    from .trackb import arms as ARMS
    from .trackb import data as TD
    from .trackb import diagnostics as DIAG
    from .trackb.modules import backbone_parameters, freeze_backbone
    from .trackb.preflight import run_preflight_b
    from .trackb.train import evaluate, train_arm

    out_dir.mkdir(parents=True, exist_ok=True)
    results = ResultStore(out_dir / "results_B.jsonl", track="B")

    # ---- site and layers: Track A Stage D if available, else the default --
    site, layers, site_reason = _resolve_site(out_dir, site, layers, log)

    budget_cfg = dict(C.TRAIN_BUDGET)
    if smoke:
        budget_cfg.update(max_steps=C.SMOKE["train_steps"],
                          eval_every=C.SMOKE["eval_every"], max_epochs=99)

    log.info("building Track B training data")
    limit = 64 if smoke else None
    train_examples = TD.build_examples(bundle, cfg, TD.TRAIN_ROLES, limit=limit, log=log)
    dev_examples = TD.build_examples(bundle, cfg, (TD.DEV_ROLE,),
                                     limit=8 if smoke else None, log=log)
    if not smoke and len(dev_examples) > C.TRACKB_DEV_EVAL_LIMIT:
        # `router-calib` has 3,031 utterances; eval_every=250 over ~14
        # training invocations would otherwise decode the full role up to
        # ~112 times. See config.TRACKB_DEV_EVAL_LIMIT.
        before = len(dev_examples)
        fraction = C.TRACKB_DEV_EVAL_LIMIT / before
        dev_examples = TD.subsample(dev_examples, fraction,
                                    seed=C.SEEDS["dev_eval_subsample"])
        log.info("training-time dev eval subsampled from %d to %d utterances "
                 "(dialogue-stratified, seed %d); the final per-arm report "
                 "still scores the full D-dev-select population",
                 before, len(dev_examples), C.SEEDS["dev_eval_subsample"])
    freeze_backbone(bundle)
    backbone = backbone_parameters(bundle)

    preflight = {"skipped": True}
    if not skip_preflight:
        log.info("running Track B preflight")
        from .preflight_a import format_preflight
        preflight = run_preflight_b(bundle, train_examples, layers=layers)
        print(format_preflight(preflight), flush=True)
        (out_dir / "preflight_B.json").write_text(
            json.dumps(preflight, indent=2, default=str), encoding="utf-8")
        if not preflight["passed"]:
            failed = [c for c in preflight["checks"] if not c["passed"]]
            raise SystemExit(
                "TRACK B PREFLIGHT FAILED -- halting before any arm trains.\n"
                + "\n".join(f"  {c['check']}: observed {c['observed']!r}, "
                            f"expected {c['expected']!r}" for c in failed))

    variant = ARMS.choose_intervention_variant(bundle, C.DEFAULT_RANK)
    log.info("intervention variant chosen by measurement: %s (%s)",
             variant["variant"], variant["timings_sec"])

    constructed = _constructed_directions(bundle, cfg, out_dir, site, layers, log)

    last_built: dict[str, Any] = {}
    state: dict[str, Any] = {"track": "B", "site": site, "layers": list(layers),
                             "site_reason": site_reason, "preflight": preflight,
                             "intervention_variant": variant,
                             "dev_role": TD.DEV_ROLE, "dev_role_note": TD.DEV_ROLE_NOTE,
                             "dev_eval_utterances": len(dev_examples),
                             "train_utterances": len(train_examples),
                             "backbone_parameters": backbone,
                             "budget": budget_cfg, "arms": {}, "notes": []}

    def train(name: str, build, *, lr: float, lam: float, seed: int,
              tag: str, tuning: bool = False,
              examples=None) -> dict[str, Any] | None:
        key = json.dumps({"arm": name, "tag": tag, "lr": lr, "lambda": lam,
                          "seed": seed, "site": site, "layers": list(layers),
                          "smoke": smoke}, sort_keys=True)
        import hashlib
        config_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        if resume and results.has(config_hash):
            log.info("resume: %s/%s already done", name, tag)
            return results.get(config_hash)
        if budget.exhausted():
            log.warning("budget exhausted before %s/%s", name, tag)
            return None
        log.info("training %s [%s] lr=%g lambda=%g seed=%d", name, tag, lr, lam, seed)
        spec = _spec(name, build)

        def _final(built_modules: dict[str, Any]) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            if not tuning:
                payload.update(_evaluate_arm(bundle, cfg, out_dir, name,
                                             built_modules, log, smoke=smoke,
                                             batch_size=batch_size, language="zh",
                                             trained_under_cs_prompt=(name == "B3-reimpl")))
            if name in ("B0", "B1", "B2") and built_modules.get("intervention") is not None:
                payload["init_diagnostics"] = DIAG.intervention_diagnostics(
                    built_modules["intervention"], built_modules["theta_init"],
                    v_nat=constructed["v_nat"], v_prompt=constructed["v_prompt"])
            if built_modules.get("adapter_stack") is not None:
                last_built["adapter_state"] = {
                    k: v.detach().float().cpu().clone()
                    for k, v in built_modules["adapter_stack"].state_dict().items()}
            return payload

        res = train_arm(bundle, spec, train_examples=examples or train_examples,
                        dev_examples=dev_examples, site=site, layers=layers,
                        lr=lr, lambda_lid=lam, seed=seed, budget=budget_cfg,
                        log=log, smoke=smoke, final_eval=_final)
        built = res.diagnostics.pop("_built", {})
        final = built.get("final_eval", {})
        row = {
            "config_hash": config_hash, "arm": name, "tag": tag, "tuning": tuning,
            "lr": lr, "lambda_lid": lam, "seed": seed, "site": site,
            "layers": list(layers),
            "trainable_parameters": res.trainable_parameters,
            "trainable_fraction_of_backbone": res.trainable_parameters / backbone,
            "steps_run": res.steps_run, "epochs_run": res.epochs_run,
            "stage_steps": res.stage_steps,
            "best_dev_mer": res.best_dev_mer, "best_step": res.best_step,
            "early_stopped": res.early_stopped,
            "history": res.history,
            "wall_clock_sec": res.wall_clock_sec,
            "peak_memory_bytes": res.peak_memory_bytes,
            "diagnostics": res.diagnostics,
            "notes": res.notes,
        }
        row.update(final)
        for module in built.get("trained_modules", []):
            if hasattr(module, "detach"):
                module.detach()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return results.append(row)

    # ---- 1. lambda search on B1, exactly once ----------------------------
    lam_rows = []
    for lam in (C.LAMBDA_LID_GRID[:1] if smoke else C.LAMBDA_LID_GRID):
        row = train("B1", ARMS.build_b1(variant=variant["variant"],
                                        rank=C.DEFAULT_RANK, directions=None),
                    lr=C.TRAIN_BUDGET["lr_grid"][1], lam=lam,
                    seed=C.B1_SEEDS[0], tag=f"lambda_search_{lam}", tuning=True)
        if row:
            lam_rows.append(row)
    lambda_lid = (min(lam_rows, key=lambda r: r["best_dev_mer"])["lambda_lid"]
                  if lam_rows else C.LAMBDA_LID_GRID[1])
    state["lambda_lid"] = lambda_lid
    state["lambda_tuned_on"] = C.LAMBDA_TUNED_ON
    log.info("frozen lambda_lid = %g (tuned on %s)", lambda_lid, C.LAMBDA_TUNED_ON)

    # ---- 2. lr search on B1, exactly once --------------------------------
    lr_rows = []
    for lr in (C.TRAIN_BUDGET["lr_grid"][1:2] if smoke else C.TRAIN_BUDGET["lr_grid"]):
        row = train("B1", ARMS.build_b1(variant=variant["variant"],
                                        rank=C.DEFAULT_RANK, directions=None),
                    lr=lr, lam=lambda_lid, seed=C.B1_SEEDS[0],
                    tag=f"lr_search_{lr}", tuning=True)
        if row:
            lr_rows.append(row)
    lr = (min(lr_rows, key=lambda r: r["best_dev_mer"])["lr"]
          if lr_rows else C.TRAIN_BUDGET["lr_grid"][1])
    grid = list(C.TRAIN_BUDGET["lr_grid"])
    at_edge = lr in (grid[0], grid[-1])
    state["lr"] = lr
    state["lr_tuned_on"] = C.LR_TUNED_ON
    state["lr_at_grid_edge"] = bool(at_edge)
    if at_edge:
        note = (f"the selected lr {lr:g} is at an edge of the grid {grid}. The "
                f"rule is to extend the grid SYMMETRICALLY FOR EVERY ARM and "
                f"rerun -- never for one arm alone. Not done inside this "
                f"allocation; recorded as a required follow-up.")
        state["notes"].append(note)
        log.warning(note)
    log.info("frozen lr = %g (tuned on %s)", lr, C.LR_TUNED_ON)

    # ---- 3-6. the arms, strictly sequential on this GPU ------------------
    arm_rows: dict[str, list[dict[str, Any]]] = {}

    row = train("B0", ARMS.build_b0(variant=variant["variant"], rank=C.DEFAULT_RANK,
                                    directions=None),
                lr=lr, lam=0.0, seed=C.B1_SEEDS[0], tag="main")
    if row:
        arm_rows["B0"] = [row]

    b1 = []
    for seed in (C.B1_SEEDS[:1] if smoke else C.B1_SEEDS):
        row = train("B1", ARMS.build_b1(variant=variant["variant"],
                                        rank=C.DEFAULT_RANK, directions=None),
                    lr=lr, lam=lambda_lid, seed=seed, tag=f"seed_{seed}")
        if row:
            b1.append(row)
    if b1:
        arm_rows["B1"] = b1

    row = train("B2", ARMS.build_b2(variant=variant["variant"], rank=C.DEFAULT_RANK,
                                    directions=constructed),
                lr=lr, lam=lambda_lid, seed=C.B1_SEEDS[0], tag="main")
    if row:
        arm_rows["B2"] = [row]

    row = train("B4", ARMS.build_b4(rank=C.DEFAULT_RANK), lr=lr, lam=lambda_lid,
                seed=C.B1_SEEDS[0], tag="main")
    if row:
        arm_rows["B4"] = [row]

    # ---- 7. B3-reimpl: head selection, then two stages -------------------
    b3_rows, b3_note = _run_b3(bundle, cfg, out_dir, train_examples, log, train,
                               lr=lr, lam=lambda_lid, backbone=backbone,
                               budget=budget, budget_cfg=budget_cfg, smoke=smoke,
                               last_built=last_built)
    if b3_rows:
        arm_rows["B3-reimpl"] = b3_rows
    state["b3_status"] = b3_note

    state["arms"] = arm_rows
    state["fidelity_note"] = AGA.FIDELITY_NOTE
    state["convergence"] = DIAG.summarise_convergence(
        [r.get("init_diagnostics") for rows in arm_rows.values() for r in rows
         if r.get("init_diagnostics")])
    (out_dir / "track_b_state.json").write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8")
    return state


def _spec(name: str, build):
    from .trackb.train import ArmSpec
    return ArmSpec(name=name, build=build, use_lid=(name != "B0"))


def _resolve_site(out_dir: Path, site, layers, log) -> tuple[str, tuple[int, ...], str]:
    """Track A Stage D if it has finished, otherwise the documented default."""
    if site and layers:
        return site, tuple(int(l) for l in layers), "supplied on the command line"
    state_path = out_dir / "track_a_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for entry in state.get("trace", []):
                if entry.get("stage") == "D" and entry.get("selected"):
                    confirm = state.get("confirm_candidate") or {}
                    cfg = confirm.get("config") or {}
                    if cfg.get("site"):
                        chosen = (tuple(cfg.get("decoder_layers") or ())
                                  or tuple(cfg.get("encoder_layers") or ()))
                        if chosen:
                            reason = (f"Track A Stage D selected "
                                      f"{entry['selected'][0]}")
                            log.info(reason)
                            return str(cfg["site"]), chosen, reason
        except Exception as exc:
            log.warning("could not read Track A state: %s", exc)
    log.info("Track B default site/layer: %s", C.TRACKB_DEFAULT_REASON)
    return C.TRACKB_DEFAULT_SITE, tuple(C.TRACKB_DEFAULT_LAYERS), C.TRACKB_DEFAULT_REASON


def _constructed_directions(bundle, cfg, out_dir, site, layers, log) -> dict[str, np.ndarray]:
    store = DirectionStore(cfg, out_dir / "directions_cache", log=log)
    layer = int(layers[0])
    v_nat = store.get(C.KIND_V_NAT, site, layer, bundle=bundle)
    if site == C.SITE_ENCODER:
        log.warning("B2's informed init needs v_prompt, which is decoder-only; "
                    "using decoder layer %d for the prompt component and "
                    "recording the mismatch", C.MID_DECODER_LAYER)
        v_prompt = store.get(C.KIND_V_PROMPT, C.SITE_DECODER, C.MID_DECODER_LAYER,
                             bundle=bundle)
    else:
        v_prompt = store.get(C.KIND_V_PROMPT, C.SITE_DECODER, layer, bundle=bundle)
    return {"v_nat": v_nat.vector.numpy(), "v_prompt": v_prompt.vector.numpy(),
            "cos": float(v_nat.vector.numpy() @ v_prompt.vector.numpy())}


def _evaluate_arm(bundle, cfg, out_dir, name, built, log, *, smoke: bool,
                  batch_size: int, language: str = "zh",
                  trained_under_cs_prompt: bool = False) -> dict[str, Any]:
    """Score the trained arm on D-dev-select with OUR scorer only.

    Every arm, including B3-reimpl, is evaluated under the SAME decode config:
    greedy, forced <|zh|>. "Same decode config" is one of the task's explicit
    invariants and mixing in B3-reimpl's training-time five-token CS prompt at
    eval would make the MER comparison confounded by prompt as well as by
    method. `trained_under_cs_prompt` only changes the label, never the
    decode call.
    """
    from . import data as D
    from .trackb import aga as AGA

    pop = D.build_population(bundle, cfg, C.DEV_SELECT, assert_anchors=not smoke)
    if smoke:
        pop = pop.subsample(C.SMOKE["utterances"])
    result = decode_population(bundle, pop, language=language,
                               batch_size=batch_size, log=log)
    corpus = M.corpus_metrics(pop, result.texts)
    baseline_path = out_dir / "results_A.jsonl"
    baseline_texts = _load_primary_baseline(baseline_path)
    payload = {"eval_metrics": corpus,
               "eval_split": C.DEV_SELECT,
               "eval_scorer": "ours only",
               "eval_prompt": "four-token forced <|zh|> (identical across every arm)",
               "trained_under_cs_prompt": trained_under_cs_prompt,
               "decode_health": decode_health(pop, result)}
    if baseline_texts:
        shared = {u: t for u, t in baseline_texts.items() if u in result.texts}
        if shared:
            risk = M.corruption_and_retention(pop, shared, result.texts)
            outcomes = M.target_outcomes(pop, shared, result.texts)
            payload["selectivity"] = {
                "zh_retention": risk["zh_retention"],
                "zh_baseline_correct": risk["zh_baseline_correct"],
                "units_at_risk": risk["units_at_risk"],
                "units_corrupted": risk["units_corrupted"],
                "delta_en_units": int(outcomes["corrected"].sum())
                - int(outcomes["corrupted"].sum()) if len(outcomes) else 0,
                "corrections": int(outcomes["corrected"].sum()) if len(outcomes) else 0,
                "corruptions": int(outcomes["corrupted"].sum()) if len(outcomes) else 0,
                "spillover_units": int(outcomes["spillover_changed"].sum())
                if len(outcomes) else 0,
            }
    return payload


def _load_primary_baseline(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("baseline_name") == C.PRIMARY_BASELINE and row.get("_texts"):
            return dict(row["_texts"])
    return {}


def _run_b3(bundle, cfg, out_dir, train_examples, log, train, *, lr, lam,
            backbone, budget, budget_cfg, smoke,
            last_built) -> tuple[list[dict[str, Any]], dict]:
    """Head selection, then stage 1, then stage 2 initialised from stage 1."""
    from .trackb import aga as AGA
    from .trackb import arms as ARMS

    note: dict[str, Any] = {"attempted": True,
                            "time_box_days": C.AGA["time_box_days"]}
    try:
        cs_examples = AGA.retarget_examples(bundle, train_examples)
        log.info("B3-reimpl: head selection over %d utterances", len(cs_examples))
        selection = AGA.select_heads(
            bundle, cs_examples, batch_size=2,
            limit=8 if smoke else min(200, len(cs_examples)), log=log)
        note["head_selection"] = selection.summary()
        (out_dir / "b3_head_selection.json").write_text(
            json.dumps({"summary": selection.summary(),
                        "counts": selection.counts.tolist(),
                        "selected": selection.selected.tolist()},
                       indent=2, default=str), encoding="utf-8")

        # two-stage accounting: the shared single-stage total is split across
        # the two stages so B3 does not receive more optimiser steps
        total = int(budget_cfg["max_steps"])
        stage1_steps = total // 2
        stage2_steps = total - stage1_steps
        note["two_stage_split"] = {"stage1": stage1_steps, "stage2": stage2_steps,
                                   "single_stage_total": total,
                                   "rule": "total optimiser steps matched to the "
                                           "single-stage arms"}

        rows: list[dict[str, Any]] = []
        s1_budget = {**budget_cfg, "max_steps": stage1_steps}
        s2_budget = {**budget_cfg, "max_steps": stage2_steps}

        saved = dict(budget_cfg)
        budget_cfg.update(s1_budget)
        row1 = train("B3-reimpl",
                     ARMS.build_b3(stage=1, head_selection=None,
                                   backbone_params=backbone),
                     lr=lr, lam=lam, seed=C.B1_SEEDS[0], tag="stage1",
                     examples=cs_examples)
        budget_cfg.update(saved)
        if row1 is None:
            note["status"] = "TIME_BOXED_OUT_BEFORE_STAGE_1"
            return [], note
        rows.append(row1)

        stage1_state = last_built.get("adapter_state")
        note["stage1_tensors_available"] = int(len(stage1_state or {}))
        if not stage1_state:
            raise AssertionError(
                "stage 2 must be initialised from stage-1 weights, but no "
                "stage-1 adapter state was captured. Stopping rather than "
                "training stage 2 from scratch and calling it two-stage.")
        budget_cfg.update(s2_budget)
        row2 = train("B3-reimpl",
                     ARMS.build_b3(stage=2, head_selection=selection,
                                   backbone_params=backbone,
                                   stage1_state=stage1_state),
                     lr=lr, lam=lam, seed=C.B1_SEEDS[0], tag="stage2",
                     examples=cs_examples)
        budget_cfg.update(saved)
        if row2 is None:
            note["status"] = "TIME_BOXED_OUT_AFTER_STAGE_1"
            return rows, note
        rows.append(row2)
        note["status"] = "COMPLETED"
        return rows, note
    except Exception as exc:
        log.error("B3-reimpl failed: %s", exc, exc_info=True)
        note["status"] = "FAILED"
        note["error"] = str(exc)
        return [], note
