"""out/SUMMARY.md -- raw numbers before any interpretation.

The two verdict lines are computed from the stored rows, never written by hand.
VERDICT_TRACK_B is deliberately four separate claims: collapsing them into one
winner would hide exactly the outcomes the study exists to distinguish.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import config as C
from .trackb import diagnostics as DIAG


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "inf" if f > 0 else ("-inf" if f < 0 else "nan")
    return format(f, spec)


def _gate_mark(ok: Any) -> str:
    return "PASS" if ok else "fail"


def _rows(path: Path) -> list[dict[str, Any]]:
    """Read a results JSONL, deduplicated by `config_hash`, latest row wins.

    The checkpoint file is an append-only log: a cell rerun without
    `--resume` (a corrected code path re-executed, or a fresh smoke pass over
    a shared `out/` dir) appends a new row rather than overwriting the old
    one. `ResultStore` already resolves this in memory by keeping the LAST
    row per hash; this mirrors that so the summary reports current state, not
    every historical attempt stacked on top of each other. Position in the
    output follows each hash's FIRST appearance, so run order stays legible.
    """
    if not path.exists():
        return []
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    untracked: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("config_hash")
        if not key:
            untracked.append(row)
            continue
        if key not in seen:
            order.append(key)
        seen[key] = row
    return [seen[k] for k in order] + untracked


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _preflight_section(a: dict, b: dict) -> list[str]:
    lines = ["## 1. Preflight", ""]
    for label, report in (("Track A", a), ("Track B", b)):
        if not report or report.get("skipped"):
            lines += [f"**{label}** -- not run in this allocation.", ""]
            continue
        status = "PASSED" if report.get("passed") else "FAILED"
        lines += [f"**{label}: {status}** "
                  f"({report.get('n_passed')}/{report.get('n_total')})", "",
                  "| check | result | observed | expected |",
                  "|---|---|---|---|"]
        for c in report.get("checks", []):
            lines.append(f"| `{c['check']}` | {'PASS' if c['passed'] else 'FAIL'} "
                         f"| {str(c['observed'])[:90]} | {str(c['expected'])[:70]} |")
        lines.append("")
    return lines


def _baseline_section(rows: Sequence[dict]) -> list[str]:
    base = [r for r in rows if r.get("stage") == "baseline"]
    lines = ["## 2. Baselines", "",
             "`force_en` is the ceiling of Whisper's native language control and "
             "the obvious reviewer challenge to any prompt-direction result, so "
             "it is in the table rather than in a footnote. Beam-search cells "
             "are compared against `C00_beam5`, never against greedy `C00`.", "",
             "| baseline | prefix | beams | PIER | MER | WER | decode health |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in base:
        m = r.get("metrics", {})
        h = r.get("decode_health", {})
        lines.append(
            f"| `{r.get('baseline_name')}` | `<|{r.get('language')}|>` "
            f"| {r.get('num_beams')} | {_fmt(m.get('PIER'), '.5f')} "
            f"| {_fmt(m.get('MER'), '.5f')} | {_fmt(m.get('WER'), '.5f')} "
            f"| {'healthy' if h.get('healthy') else 'UNHEALTHY'} |")
    if not base:
        lines.append("| -- | -- | -- | -- | -- | -- | no baseline completed |")
    lines.append("")
    return lines


def _track_a_table(rows: Sequence[dict]) -> list[str]:
    cells = [r for r in rows if r.get("stage") in ("A", "B", "C", "D")]
    lines = ["## 3. Track A -- one row per cell", "",
             f"Evaluated on all {C.ANCHOR_DEV_SELECT_UTTERANCES} utterances of "
             f"`{C.DEV_SELECT}` over {C.ANCHOR_DEV_SELECT_DIALOGUES} dialogues, "
             f"not only the {C.ANCHOR_DEV_SELECT_TARGET_UTTERANCES} that carry a "
             "target: global steering touches everything and harm needs the "
             "correct denominator.", "",
             "A cell whose decode is unhealthy is **INVALID**, not negative. Its "
             "PIER is not interpreted and it never enters a stage ranking.", "",
             "| stage | cell | dPIER | 95% CI | corr | corrupt | ratio | "
             "dialogues | ZH ret. | G1 | G2 | G3 | G4 | verdict |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    for r in cells:
        ci = f"[{_fmt(r.get('delta_pier_ci_low'), '+.5f')}, " \
             f"{_fmt(r.get('delta_pier_ci_high'), '+.5f')}]"
        lines.append(
            f"| {r.get('stage')} | `{r.get('cell_id')}` "
            f"| {_fmt(r.get('delta_pier'), '+.5f')} | {ci} "
            f"| {r.get('corrections', 0)} | {r.get('corruptions', 0)} "
            f"| {_fmt(r.get('corrections_to_corruptions'), '.2f')} "
            f"| {r.get('dialogues_with_correction', 0)}/{C.ANCHOR_DEV_SELECT_DIALOGUES} "
            f"| {_fmt(r.get('zh_retention'), '.4f')} "
            f"| {_gate_mark(r.get('G1_delta_pier'))} "
            f"| {_gate_mark(r.get('G2_correction_ratio'))} "
            f"| {_gate_mark(r.get('G3_dialogue_coverage'))} "
            f"| {_gate_mark(r.get('G4_zh_retention'))} "
            f"| **{r.get('verdict')}** |")
    if not cells:
        lines.append("| -- | no steered cell completed | | | | | | | | | | | | |")
    lines += ["",
              "### Corpus metrics -- PIER, MER, WER, baseline vs steered", "",
              "Primary metrics per section 3.7: absolute PIER/MER/WER for the "
              "steered decode and its comparator baseline (`C00` or "
              "`C00_beam5` for beam cells), plus delta MER and delta WER "
              "alongside the already-gated delta PIER. Positive delta means "
              "the steered cell reduced error.", "",
              "| cell | comparator | PIER base -> steer | MER base -> steer "
              "| WER base -> steer | dMER | dWER |",
              "|---|---|---|---|---|---:|---:|"]
    for r in cells:
        m = r.get("metrics", {})
        b = r.get("baseline_metrics", {})
        d_mer = (b.get("MER") - m.get("MER")) if m.get("MER") is not None \
            and b.get("MER") is not None else None
        d_wer = (b.get("WER") - m.get("WER")) if m.get("WER") is not None \
            and b.get("WER") is not None else None
        lines.append(
            f"| `{r.get('cell_id')}` | {r.get('compared_against', 'C00')} "
            f"| {_fmt(b.get('PIER'), '.5f')} -> {_fmt(m.get('PIER'), '.5f')} "
            f"| {_fmt(b.get('MER'), '.5f')} -> {_fmt(m.get('MER'), '.5f')} "
            f"| {_fmt(b.get('WER'), '.5f')} -> {_fmt(m.get('WER'), '.5f')} "
            f"| {_fmt(d_mer, '+.5f')} | {_fmt(d_wer, '+.5f')} |")
    if not cells:
        lines.append("| -- | | | | | | |")
    lines += ["",
              "### Energy and coverage audit", "",
              "| cell | \\|S\\| planned | \\|S\\| observed | alpha | alpha_eff "
              "| E_total | valid frame ratio (min/mean) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in cells:
        e = r.get("energy", {})
        a = r.get("audit", {})
        lines.append(
            f"| `{r.get('cell_id')}` | {_fmt(e.get('S'))} "
            f"| {_fmt(e.get('S_observed'))} | {_fmt(e.get('alpha_nominal'), '.3g')} "
            f"| {_fmt(e.get('alpha_eff'), '.4g')} | {_fmt(e.get('E_total'), '.4g')} "
            f"| {_fmt(a.get('valid_frame_ratio_min'), '.3f')} / "
            f"{_fmt(a.get('valid_frame_ratio_mean'), '.3f')} |")
    lines.append("")
    return lines


def _poi_section(rows: Sequence[dict]) -> list[str]:
    cells = [r for r in rows if r.get("stage") in ("A", "B", "C", "D")]
    if not cells:
        return []
    lines = ["### POI decomposition and spillover on the 489 targets", "",
             "| cell | correct | deletion | wrong-language sub | other sub | "
             "spillover changed | improved | damaged |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cells:
        d = r.get("poi_decomposition_steered", {})
        s = r.get("spillover", {})
        lines.append(
            f"| `{r.get('cell_id')}` | {d.get('correct', 0)} "
            f"| {d.get('deletion', 0)} | {d.get('wrong_language_substitution', 0)} "
            f"| {d.get('other_substitution', 0)} "
            f"| {s.get('spillover_units_changed', 0)} "
            f"| {s.get('spillover_units_improved', 0)} "
            f"| {s.get('spillover_units_damaged', 0)} |")
    lines.append("")
    return lines


def _trace_section(state: dict) -> list[str]:
    lines = ["## 4. Stage A -> B -> C -> D selection trace", "",
             "The rule is fixed in `steer_sweep/metrics.select` and cannot move "
             "after a result is seen:", "",
             "1. drop runs with unhealthy decode;",
             "2. drop runs with ZH retention < 0.99;",
             "3. sort by `dialogues_with_correction` DESC, then "
             "corrections:corruptions DESC, then delta PIER DESC.", "",
             "The primary key is dialogue coverage, not delta PIER. The prior "
             "programme's seven corrections came from two of twenty dialogues, "
             "so its effective cluster count was two.", ""]
    for entry in state.get("trace", []):
        lines += [f"**Stage {entry.get('stage')}** -- "
                  f"selected `{', '.join(entry.get('selected') or ['none'])}`", "",
                  f"- cells considered: {entry.get('n_input')}",
                  f"- dropped, unhealthy decode: "
                  f"{entry.get('dropped_unhealthy_decode') or 'none'}",
                  f"- dropped, ZH retention < 0.99: "
                  f"{entry.get('dropped_zh_retention_below_0.99') or 'none'}",
                  f"- eligible after both filters: {entry.get('n_eligible')}", ""]
        ranking = entry.get("ranking") or []
        if ranking:
            lines += ["  | rank | cell | dialogues | ratio | dPIER |",
                      "  |---:|---|---:|---:|---:|"]
            for i, item in enumerate(ranking[:6], 1):
                cell, key = item[0], item[1]
                lines.append(f"  | {i} | `{cell}` | {key[0]} "
                             f"| {_fmt(key[1], '.2f')} | {_fmt(key[2], '+.5f')} |")
            lines.append("")
    if not state.get("trace"):
        lines.append("_No stage completed in this allocation._\n")
    return lines


def _confirm_section(state: dict) -> list[str]:
    confirm = state.get("confirm_candidate")
    lines = ["## 5. Confirm -- NOT RUN", ""]
    if not confirm:
        lines += ["No configuration reached the Confirm step.", ""]
        return lines
    cfg = confirm.get("config", {})
    lines += [f"`{confirm.get('cell_id')}`", "", "```json",
              json.dumps(cfg, indent=2, default=str), "```", "",
              f"**Status: {confirm.get('status')}.** {confirm.get('statement')}", ""]
    return lines


def _track_b_section(rows: Sequence[dict], state: dict) -> list[str]:
    arms = [r for r in rows if r.get("arm") and not r.get("tuning")]
    lines = ["## 6. Track B -- arms", "",
             "Arms are **under-trained by design**: the budget is a fast "
             "observation pass, not a converged result, and absolute MER here is "
             "not comparable to published numbers.", "",
             f"Site/layers: `{state.get('site')}` layers `{state.get('layers')}`. "
             f"Reason: {state.get('site_reason')}", "",
             f"Training pool: {_fmt(state.get('train_utterances'))} utterances "
             f"(`loc-train` + `util-train`). Training-time early-stopping / "
             f"lr-selection dev set: {_fmt(state.get('dev_eval_utterances'))} "
             f"utterances, dialogue-stratified from `{state.get('dev_role')}` "
             f"(full role: 3,031). The final per-arm report below always scores "
             f"the full `D-dev-select` population, once, after training.", "",
             f"Intervention variant run: "
             f"`{(state.get('intervention_variant') or {}).get('variant')}` -- "
             f"{(state.get('intervention_variant') or {}).get('reason')} "
             f"(timings {(state.get('intervention_variant') or {}).get('timings_sec')}).",
             "",
             "\"Matched budget\" means matched training CONDITIONS, not matched "
             "parameter count. The parameter gap is the finding, not a nuisance "
             "to control, and it is reported rather than equalised.", "",
             "| arm | tag | trainable params | % of 1.55 B | MER | PIER | WER | "
             "ZH ret. | spillover | dev MER | steps | wall clock | peak mem |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in arms:
        m = r.get("eval_metrics", {})
        sel = r.get("selectivity", {})
        lines.append(
            f"| {r.get('arm')} | {r.get('tag')} "
            f"| {_fmt(r.get('trainable_parameters'))} "
            f"| {_fmt(100 * (r.get('trainable_parameters', 0) / C.BACKBONE_PARAMS_NOMINAL), '.4f')}% "
            f"| {_fmt(m.get('MER'), '.5f')} | {_fmt(m.get('PIER'), '.5f')} "
            f"| {_fmt(m.get('WER'), '.5f')} "
            f"| {_fmt(sel.get('zh_retention'), '.4f')} "
            f"| {_fmt(sel.get('spillover_units'))} "
            f"| {_fmt(r.get('best_dev_mer'), '.5f')} "
            f"| {r.get('steps_run')} "
            f"| {_fmt(r.get('wall_clock_sec'), '.0f')}s "
            f"| {_fmt((r.get('peak_memory_bytes') or 0) / 2**30, '.1f')} GiB |")
    if not arms:
        lines.append("| -- | no arm completed | | | | | | | | | | | |")

    b1 = [r for r in arms if r.get("arm") == "B1"]
    if b1:
        rng = DIAG.seed_range([r.get("eval_metrics", {}).get("MER", float("nan"))
                               for r in b1])
        lines += ["", f"**B1 across {rng['n']} seeds:** MER mean "
                  f"{_fmt(rng['mean'], '.5f')}, range "
                  f"[{_fmt(rng['min'], '.5f')}, {_fmt(rng['max'], '.5f')}]. "
                  "B2 is evaluated against this range, never against a single "
                  "random draw.", ""]
    lines += ["", f"**Frozen `lambda_lid` = {state.get('lambda_lid')}**, tuned "
              f"once on {state.get('lambda_tuned_on')} over "
              f"{list(C.LAMBDA_LID_GRID)} and reused unchanged in every arm.",
              "",
              f"**Frozen `lr` = {state.get('lr')}**, tuned once on "
              f"{state.get('lr_tuned_on')} over {list(C.TRAIN_BUDGET['lr_grid'])}, "
              f"selected on dev MER with the same protocol for every arm.", ""]
    if state.get("lr_at_grid_edge"):
        lines += ["> **The selected lr is at a grid edge.** The rule is to extend "
                  "the grid symmetrically FOR EVERY ARM and rerun -- never for "
                  "one arm alone. This was not done inside this allocation and "
                  "is recorded as a required follow-up.", ""]
    for note in state.get("notes", []):
        lines += [f"> {note}", ""]
    return lines


def _diagnostics_section(rows: Sequence[dict], state: dict) -> list[str]:
    lines = ["## 7. Track B initialisation diagnostics (4.3)", "",
             "Did the optimizer converge toward the constructed directions, or "
             "away from them? Both outcomes are publishable and the "
             "implementation does not bias toward either. A random rank-2 "
             "subspace of a 1280-dimensional space sits at essentially 90 "
             "degrees, so **90 is the null, not zero**.", "",
             "| arm | seed | cos(theta, v_nat) | cos(theta, v_prompt) | "
             "\\|\\|theta - theta_0\\|\\| | principal angles (deg) |",
             "|---|---:|---:|---:|---:|---|"]
    any_row = False
    for r in rows:
        # lambda_lid/lr search calls train B1 at the same seed and log the same
        # diagnostics as its real run; only the arms actually being compared
        # belong in this table, not every hyperparameter-search side effect.
        if r.get("tuning"):
            continue
        d = r.get("init_diagnostics")
        if not d:
            continue
        any_row = True
        angles = ", ".join(_fmt(a, ".1f") for a in d.get("principal_angles_deg", []))
        lines.append(
            f"| {r.get('arm')} | {r.get('seed')} "
            f"| {_fmt(d.get('cos_theta_final_v_nat'), '+.4f')} "
            f"| {_fmt(d.get('cos_theta_final_v_prompt'), '+.4f')} "
            f"| {_fmt(d.get('theta_drift_l2'), '.4f')} | {angles} |")
    if not any_row:
        lines.append("| -- | -- | -- | -- | -- | no diagnostics recorded |")
    conv = state.get("convergence") or {}
    lines += ["", f"**Aggregate: {conv.get('verdict', 'UNDETERMINED')}** "
              f"(mean principal angle "
              f"{_fmt(conv.get('mean_principal_angle_deg'), '.1f')} deg over "
              f"{conv.get('n_runs', 0)} runs).", "",
              "- converged toward span{v_nat, v_prompt} -> the interpretability "
              "work is validated and Track A's failure is a magnitude problem, "
              "not a direction problem;",
              "- diverged -> this explains Track A's failure: the constructed "
              "direction was not the direction the model needed.", ""]
    return lines


def _b3_section(rows: Sequence[dict], state: dict) -> list[str]:
    note = state.get("b3_status") or {}
    lines = ["## 8. B3-reimpl: fidelity note and the B3-vs-B4 gate", ""]
    status = note.get("status", "NOT_ATTEMPTED")
    lines += [f"**Status: {status}.**", ""]
    if status in ("TIME_BOXED_OUT_BEFORE_STAGE_1", "TIME_BOXED_OUT_AFTER_STAGE_1",
                  "FAILED", "NOT_ATTEMPTED"):
        lines += ["The strong baseline was attempted and did not produce valid "
                  "numbers inside its time box. Track B therefore reports **B4 "
                  "(plain LoRA) as the baseline**, and this is stated rather "
                  "than omitted silently.", ""]
        if note.get("error"):
            lines += [f"Failure: `{note['error']}`", ""]

    hs = note.get("head_selection")
    if hs:
        lines += ["### Head selection", "",
                  f"- criterion: {hs.get('criterion')}",
                  f"- computed on: {hs.get('computed_on')}",
                  f"- selected {hs.get('n_selected')} of {hs.get('n_total')} heads "
                  f"(fraction {_fmt(hs.get('fraction'), '.4f')}) over "
                  f"{hs.get('utterances_scanned')} utterances",
                  f"- per-layer counts: {hs.get('selected_per_layer')}", ""]
    split = note.get("two_stage_split")
    if split:
        lines += ["### Two-stage accounting", "",
                  f"- stage 1: {split.get('stage1')} optimizer steps "
                  f"(encoder adapters, `cs_weight = 0`)",
                  f"- stage 2: {split.get('stage2')} steps, initialised from "
                  f"stage-1 weights (decoder adapters added, guidance loss on)",
                  f"- total {split.get('single_stage_total')}, matched against "
                  "the single-stage arms", ""]

    b3 = [r for r in rows if r.get("arm") == "B3-reimpl" and not r.get("tuning")]
    check = None
    for r in b3:
        rep = ((r.get("diagnostics") or {}).get("modules") or {})
        check = rep.get("parameter_fraction_check") or check
    if check:
        lines += ["### Trainable-parameter fraction", "",
                  f"- {check.get('detail')}",
                  f"- within a factor of {check.get('within_factor')} of the "
                  f"paper's ~{100 * check.get('paper_fraction', 0):.1f}%: "
                  f"**{'yes' if check.get('passed') else 'NO -- STRUCTURAL MISMATCH'}**",
                  ""]

    b4 = [r for r in rows if r.get("arm") == "B4" and not r.get("tuning")]
    gate = "UNDETERMINED"
    detail = "one of the two arms did not produce a number"
    if b3 and b4:
        b3_mer = min(r.get("eval_metrics", {}).get("MER", float("inf")) for r in b3)
        b4_mer = min(r.get("eval_metrics", {}).get("MER", float("inf")) for r in b4)
        if np.isfinite(b3_mer) and np.isfinite(b4_mer):
            gate = "PASSED" if b3_mer < b4_mer else "FAILED"
            detail = (f"B3-reimpl MER {b3_mer:.5f} vs B4 (plain LoRA) MER "
                      f"{b4_mer:.5f} under our scorer")
    lines += ["### B3-vs-B4 gate", "",
              f"**{gate}** -- {detail}.", ""]
    if gate == "FAILED":
        lines += ["B3-reimpl does not beat plain LoRA under our scorer, so it is "
                  "**not functioning as a strong baseline** and is not described "
                  "as one. Every B3 comparison in this document is therefore "
                  "marked UNRELIABLE.", ""]
    lines += ["### Fidelity note", "",
              "| item | status | detail |", "|---|---|---|"]
    for item in state.get("fidelity_note", []):
        lines.append(f"| {item['item']} | {item['status']} | {item['detail']} |")
    lines += ["",
              "Two mismatches stated rather than hidden: the reference "
              "repository is titled for Taiwan-accented Mandarin-English while "
              "CS-Dialogue is a different accent population, and the published "
              "~14.2% MER is on SEAME while we evaluate on CS-Dialogue -- "
              "external context, not a reproduction target.", ""]
    return lines


def _verdicts(a_rows: Sequence[dict], b_rows: Sequence[dict],
              a_state: dict, b_state: dict) -> list[str]:
    cells = [r for r in a_rows if r.get("stage") in ("A", "B", "C", "D")]
    go = [r for r in cells if r.get("verdict") == "GO"]
    if go:
        verdict_a = (f"GO -- {len(go)} cell(s) pass all four gates; best is "
                     f"`{go[0]['cell_id']}`")
    elif not cells:
        verdict_a = "NO-GO -- no steered cell completed in this allocation"
    else:
        counts: dict[str, int] = {}
        for r in cells:
            for g in r.get("gates_failed", []):
                counts[g] = counts.get(g, 0) + 1
        worst = max(counts, key=counts.get) if counts else "none"
        invalid = sum(1 for r in cells if r.get("verdict") == "INVALID")
        verdict_a = (f"NO-GO -- no cell passes all four gates over {len(cells)} "
                     f"cells ({invalid} INVALID on decode health). The gate that "
                     f"failed most often is {worst} "
                     f"({counts.get(worst, 0)}/{len(cells)} cells)")

    arms = [r for r in b_rows if r.get("arm") and not r.get("tuning")]
    def mer(name: str) -> float:
        vals = [r.get("eval_metrics", {}).get("MER", float("nan")) for r in arms
                if r.get("arm") == name]
        vals = [v for v in vals if np.isfinite(v)]
        return float(min(vals)) if vals else float("nan")

    b0, b2, b3, b4 = mer("B0"), mer("B2"), mer("B3-reimpl"), mer("B4")
    b1_vals = [r.get("eval_metrics", {}).get("MER", float("nan")) for r in arms
               if r.get("arm") == "B1"]
    b1_rng = DIAG.seed_range(b1_vals)

    if np.isfinite(b0) and b1_rng["n"]:
        claim_a = ("LID helped" if b1_rng["mean"] < b0 else "LID did not help")
        claim_a += f" -- B1 mean MER {b1_rng['mean']:.5f} vs B0 {b0:.5f}"
    else:
        claim_a = "UNDETERMINED -- B0 or B1 did not produce a number"

    if np.isfinite(b2) and b1_rng["n"]:
        outcome = DIAG.beats_range(b2, b1_rng)
        claim_b = (f"{outcome} -- B2 MER {b2:.5f} against the B1 seed range "
                   f"[{b1_rng['min']:.5f}, {b1_rng['max']:.5f}]")
    else:
        claim_b = "UNDETERMINED -- B2 or the B1 seed range is missing"

    gate_ok = np.isfinite(b3) and np.isfinite(b4) and b3 < b4
    if np.isfinite(b3):
        wins = []
        if b1_rng["n"] and b1_rng["mean"] < b3:
            wins.append("MER")
        b1_params = [r.get("trainable_parameters", 0) for r in arms
                     if r.get("arm") in ("B1", "B2")]
        b3_params = [r.get("trainable_parameters", 0) for r in arms
                     if r.get("arm") == "B3-reimpl"]
        if b1_params and b3_params and min(b1_params) < min(b3_params):
            wins.append("params")
        claim_c = (f"B1/B2 beat B3-reimpl on {wins}" if wins
                   else "B1/B2 beat B3-reimpl on none of "
                        "{MER, params, data efficiency, selectivity}")
        if not gate_ok:
            claim_c += " -- UNRELIABLE: the B3-vs-B4 gate did not pass"
    else:
        status = (b_state.get("b3_status") or {}).get("status", "NOT_ATTEMPTED")
        claim_c = (f"UNDETERMINED -- B3-reimpl produced no number "
                   f"(status {status}); B4 is reported as the baseline")

    conv = b_state.get("convergence") or {}
    claim_d = (f"{conv.get('verdict', 'UNDETERMINED')} -- mean principal angle "
               f"{_fmt(conv.get('mean_principal_angle_deg'), '.1f')} deg to "
               f"span(v_nat, v_prompt) against a 90 deg random null")

    return [
        "## 9. Verdicts", "",
        f"**VERDICT_TRACK_A:** {verdict_a}", "",
        "**VERDICT_TRACK_B:** four separate claims, never collapsed into one "
        "winner:", "",
        f"- **(a) did LID help (B1 vs B0)?** {claim_a}",
        f"- **(b) did informed init beat the B1 seed range (B2 vs B1)?** {claim_b}",
        f"- **(c) did B1/B2 beat B3 on any of MER, params, data efficiency, "
        f"selectivity?** {claim_c}",
        f"- **(d) did the learned subspace converge toward or away from "
        f"span(v_nat, v_prompt)?** {claim_d}", "",
        f"**B3-vs-B4 gate:** {'PASSED' if gate_ok else 'NOT PASSED'}. "
        + ("" if gate_ok else "Every B3 comparison above is marked UNRELIABLE. ")
        + "B4 (plain LoRA) is the unambiguous floor that protects the study.", "",
    ]


def _deviations_section() -> list[str]:
    lines = ["## 10. Where the specification and the repository disagree", "",
             "Recorded, not resolved silently. Nothing below was substituted "
             "without saying so.", ""]
    for d in C.SPEC_DEVIATIONS:
        lines += [f"### `{d['id']}`", "",
                  f"- **specification:** {d['spec']}",
                  f"- **repository:** {d['repository']}",
                  f"- **resolution:** {d['resolution']}", ""]
    return lines


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def write_summary(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_rows = _rows(out_dir / "results_A.jsonl")
    b_rows = _rows(out_dir / "results_B.jsonl")
    a_state = _load(out_dir / "track_a_state.json")
    b_state = _load(out_dir / "track_b_state.json")
    pre_a = _load(out_dir / "preflight_A.json")
    pre_b = _load(out_dir / "preflight_B.json")

    lines = [
        "# Two-track steering study for Mandarin-English CS-ASR", "",
        f"**Stamp:** `{C.TAINT}`. This document evaluates no production gate, "
        "freezes no spans, allocates no held-out generation, and opens neither "
        "`D-dev-confirm` nor `D-test`.", "",
        f"- model: `whisper-large-v3`, frozen unless an arm explicitly trains "
        f"something ({C.EXPECTED_ENCODER_LAYERS} encoder / "
        f"{C.EXPECTED_DECODER_LAYERS} decoder layers, d_model "
        f"{C.EXPECTED_D_MODEL}, {C.EXPECTED_ENCODER_FRAMES} encoder frames)",
        f"- population: `{C.DEV_SELECT}`, "
        f"{C.ANCHOR_DEV_SELECT_UTTERANCES} utterances / "
        f"{C.ANCHOR_DEV_SELECT_DIALOGUES} dialogues, "
        f"{C.ANCHOR_TARGETS} targets",
        f"- anchors asserted: {C.ANCHOR_TARGETS} targets, "
        f"{C.ANCHOR_BASELINE_CORRECT_UNITS:,} baseline-correct units, "
        f"{C.ANCHOR_BASELINE_CORRECT_ZH_UNITS:,} baseline-correct Mandarin units",
        f"- git commit: `{(a_rows or b_rows or [{}])[0].get('git_commit', 'unknown')}`",
        f"- seeds: `{C.SEEDS}`", "",
        "**Raw numbers first; interpretation is confined to section 9.**", "",
        "---", "",
    ]
    lines += _preflight_section(pre_a, pre_b)
    lines += _baseline_section(a_rows)
    lines += _track_a_table(a_rows)
    lines += _poi_section(a_rows)
    lines += _trace_section(a_state)
    lines += _confirm_section(a_state)
    lines += _track_b_section(b_rows, b_state)
    lines += _diagnostics_section(b_rows, b_state)
    lines += _b3_section(b_rows, b_state)
    lines += _verdicts(a_rows, b_rows, a_state, b_state)
    lines += _deviations_section()

    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
