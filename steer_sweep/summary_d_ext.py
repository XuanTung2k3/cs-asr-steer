"""out/SUMMARY_D_EXT.md -- the dense (layer x alpha x beam) landscape map.

This is not a "best cell" report. Section headers and the explicit
Q1/Q2/Q3 verdict lines are fixed by the task; nothing here ranks cells or
proposes a winner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import config as C
from . import stage_d_ext as D


def _fmt(value: Any, spec: str = ".5f") -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "inf" if f > 0 else ("-inf" if f < 0 else "nan")
    return format(f, spec)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _rows(path: Path) -> list[dict[str, Any]]:
    """Deduplicated by config_hash, latest write wins (matches ResultStore's
    own in-memory semantics; see steer_sweep.summary._rows for why)."""
    if not path.exists():
        return []
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("config_hash")
        if not key:
            continue
        if key not in seen:
            order.append(key)
        seen[key] = row
    return [seen[k] for k in order]


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def _preregistered_section() -> list[str]:
    return [
        "## 0. Preregistered questions", "",
        "Written before results, per section 1 of the task. Answered in "
        "section 9, nowhere else.", "",
        "- **Q1 -- contiguous region or isolated points?** Answered by the "
        "plateau/connected-component analysis (section 6/7), never by the "
        "single best cell.",
        "- **Q2 -- do steering and beam search compose or substitute?** "
        "`gain_b1 = PIER(C00) - PIER(cell@beams=1)`, "
        "`gain_b5 = PIER(C00_beam5) - PIER(cell@beams=5)`, paired per "
        "(layer, alpha) across the whole grid, not just the best cell.",
        "- **Q3 -- how many cells would pass the gates by chance?** 70 cells "
        "at nominal 95% gives ~3.5 expected false positives; stated against "
        "the observed count explicitly.", "",
    ]


def _determinism_section(det: dict) -> list[str]:
    lines = ["## 1. Determinism check (section 4.1) -- run first", "",
             "Two in-process decodes plus one fresh-subprocess decode of "
             f"`decoder layer {det.get('layer')}, alpha {det.get('alpha')}`, "
             "at num_beams=1 and num_beams=5. All three must be bit-identical "
             "across all 300 utterances, or nothing downstream is trustworthy.",
             ""]
    if not det:
        lines += ["_Not run in this allocation._", ""]
        return lines
    status = "PASSED" if det.get("passed") else "FAILED"
    lines += [f"**{status}**", "",
             "| beams | attempt | matched | diff (in-process pair) | "
             "diff (subprocess pair) | deterministic settings forced | elapsed |",
             "|---:|---:|---|---:|---:|---|---:|"]
    for c in det.get("checks", []):
        lines.append(
            f"| {c['num_beams']} | {c['attempt']} | "
            f"{'yes' if c['passed'] else 'NO'} | {c['n_diff_inprocess_pair']} "
            f"| {c['n_diff_subprocess_pair']} | "
            f"{'yes' if c['deterministic_settings_applied'] else 'no'} | "
            f"{_fmt(c['elapsed_sec'], '.0f')}s |")
    lines.append("")
    return lines


def _baselines_section(state: dict) -> list[str]:
    base = state.get("baselines", {})
    lines = ["## 2. Baselines -- regenerated fresh, asserted against prior anchors", "",
             "Never read from a stored value. Decoded fresh in this run and "
             "asserted to match the anchors from job 42674 within 5e-4 "
             "absolute tolerance on PIER/MER/WER; a mismatch beyond float "
             "noise would itself be a determinism finding.", "",
             "| baseline | beams | PIER | MER | WER |",
             "|---|---:|---:|---:|---:|"]
    for name in (D.baseline_for(1), D.baseline_for(5)):
        m = base.get(name, {})
        beams = 1 if name == D.baseline_for(1) else 5
        lines.append(f"| `{name}` | {beams} | {_fmt(m.get('PIER'))} "
                     f"| {_fmt(m.get('MER'))} | {_fmt(m.get('WER'))} |")
    lines.append("")
    return lines


def _checks_section(state: dict) -> list[str]:
    lines = ["## 3. Section 4.2 / 4.3 checks", ""]
    sa = state.get("small_alpha_sanity", {})
    lines += [f"**4.2 small-alpha sanity: "
             f"{'PASSED' if sa.get('passed') else 'NOT RUN / FAILED'}** "
             f"(threshold {_fmt(sa.get('threshold'), '.2f')})", "",
             "ratio = ||applied delta|| / mean activation norm at that layer "
             "= |alpha_eff| exactly, under scale_mode=\"act_norm\" (the "
             "applied edit is `alpha_eff * mean_norm(h) * unit_vector`, so "
             "its own L2 norm divided by mean_norm(h) collapses to "
             "|alpha_eff|; computed from the logged audit fields, not "
             "re-derived by assumption).", "",
             "| layer | beams | ratio | act_norm_mean | ok |",
             "|---:|---:|---:|---:|---|"]
    for e in sa.get("entries", []):
        lines.append(f"| {e['layer']} | {e['num_beams']} "
                     f"| {_fmt(e['delta_l2_over_activation_norm'], '.5f')} "
                     f"| {_fmt(e['act_norm_mean'], '.3f')} "
                     f"| {'yes' if e['ok'] else 'NO'} |")
    lines.append("")

    sd = state.get("s_audit", {})
    hits = sd.get("per_beam_utterance_hits", {})
    lines += [f"**4.3 |S| audit: {'PASSED' if sd.get('passed') else 'DEVIATIONS FOUND'}** "
             f"-- |S| observed must equal (oracle-step hits per utterance) x "
             f"(layer count) x (num_beams), out of 116 target utterances "
             f"(2 carry a CS span but no oracle decoder step at all). Hit "
             f"rate per beam setting: {hits}. beams=1 matches the original "
             "Track A run's 114/116 exactly; beams=5's 113/116 was "
             "discovered empirically by this phase -- one additional "
             "utterance's beam-search output does not reach as far as its "
             "greedy counterpart -- and confirmed uniform across all 35 "
             "beam=5 Phase 1 cells before being frozen as the expected "
             f"pattern. {sd.get('n_deviations', 0)} of {sd.get('n_cells', 0)} "
             "cells deviate from it.", ""]
    if sd.get("deviations"):
        lines += ["| cell | beams | S planned | S observed | expected |",
                 "|---|---:|---:|---:|---:|"]
        for dv in sd["deviations"][:20]:
            lines.append(f"| `{dv['cell_id']}` | {dv.get('num_beams')} "
                         f"| {dv['S_planned']} | {dv['S_observed']} "
                         f"| {dv.get('expected_observed')} |")
        lines.append("")
    return lines


def _full_table(rows: Sequence[dict[str, Any]], *, stage: str) -> list[str]:
    """The table only -- no heading. Callers own their own heading line so
    this can be reused inside a subsection (Phase 2) without duplicating or
    misnumbering the top-level section 4 header."""
    cells = [r for r in rows if r.get("stage") == stage]
    lines = ["Absolute PIER/MER/WER, deltas, CI, both corruption measures "
             "side by side (units at risk vs spillover damaged), health, and "
             "the |S| / alpha_eff / E_total audit. `baseline_used` is printed "
             "per row so beam parity is auditable, not asserted-and-hidden.", "",
             "| layer | alpha | beams | baseline | PIER | MER | WER | dPIER "
             "| dMER | dWER | dPIER CI | corr | corrupt(units) | corrupt(spill) "
             "| ratio(units) | ratio(spill) | dialogues | ZH ret | health | "
             "|S| pl/obs | alpha_eff | verdict |",
             "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"
             "---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for r in sorted(cells, key=lambda x: (x["layer"], x["alpha"], x["num_beams"])):
        ci = f"[{_fmt(r.get('delta_pier_ci_low'), '+.4f')}, {_fmt(r.get('delta_pier_ci_high'), '+.4f')}]"
        health = "ok" if r.get("decode_health", {}).get("healthy") else "UNHEALTHY"
        s = f"{r.get('S_planned')}/{r.get('S_observed')}"
        lines.append(
            f"| {r['layer']} | {r['alpha']:g} | {r['num_beams']} "
            f"| `{r['baseline_used']}` | {_fmt(r.get('PIER'))} | {_fmt(r.get('MER'))} "
            f"| {_fmt(r.get('WER'))} | {_fmt(r.get('delta_pier'), '+.5f')} "
            f"| {_fmt(r.get('delta_mer'), '+.5f')} | {_fmt(r.get('delta_wer'), '+.5f')} "
            f"| {ci} | {r.get('corrections', 0)} | {r.get('corruptions_units', 0)} "
            f"| {r.get('corruptions_spillover_damaged', 0)} "
            f"| {_fmt(r.get('ratio_units'), '.2f')} | {_fmt(r.get('ratio_spillover'), '.2f')} "
            f"| {r.get('dialogues_with_correction', 0)} "
            f"| {_fmt(r.get('zh_retention'), '.4f')} | {health} | {s} "
            f"| {_fmt(r.get('alpha_eff'), '.4g')} | **{r.get('verdict')}** |")
    if not cells:
        lines.append("| -- | no cells completed for this stage | | | | | | | "
                     "| | | | | | | | | | | | | |")
    lines.append("")
    return lines


def _heatmap_table(hm: dict, *, title: str) -> list[str]:
    lines = [f"### {title}", "",
             "| layer \\ alpha | " + " | ".join(f"{a:g}" for a in hm["alphas"]) + " |",
             "|---:|" + "---:|" * len(hm["alphas"])]
    for i, layer in enumerate(hm["layers"]):
        row = hm["values"][i]
        lines.append(f"| {layer} | " + " | ".join(_fmt(v, "+.4f") for v in row) + " |")
    lines.append("")
    return lines


def _heatmaps_section(analysis: dict) -> list[str]:
    lines = ["## 5. Heatmaps (section 6.1) -- layer x alpha, value = ΔMER", ""]
    hm = analysis.get("heatmaps", {})
    if "delta_mer_beams1" in hm:
        lines += _heatmap_table(hm["delta_mer_beams1"], title="beams = 1")
    if "delta_mer_beams5" in hm:
        lines += _heatmap_table(hm["delta_mer_beams5"], title="beams = 5")
    if "gain_b5_minus_gain_b1" in hm:
        lines += _heatmap_table(hm["gain_b5_minus_gain_b1"],
                               title="gain_b5 − gain_b1 (Q2)")
    return lines


def _components_section(analysis: dict) -> list[str]:
    lines = ["## 6. Connected-component report (section 6.3) -- the headline output", "",
             "Components of {ΔMER > 0} under 4-neighbour grid adjacency "
             "(layer ±1 step, alpha ±1 step). A size-0/1 entry is an "
             "isolated point, indistinguishable from noise; size ≥ 4 is "
             "the threshold this phase treats as structure worth a Phase 2 "
             "follow-up.", ""]
    comps = analysis.get("connected_components", {})
    for beams in D.PHASE1_BEAMS:
        cs = comps.get(beams, []) or comps.get(str(beams), [])
        lines += [f"### beams = {beams}", ""]
        if not cs:
            lines += ["No cells with ΔMER > 0 at all -- the null map, "
                     "stated as the finding.", ""]
            continue
        lines += ["| size | layer range | alpha range | mean ΔMER | members |",
                 "|---:|---|---|---:|---|"]
        for c in cs:
            members = ", ".join(f"L{m['layer']}/a{m['alpha']:g}" for m in c["members"])
            lines.append(f"| {c['size']} | {c['layer_range']} | {c['alpha_range']} "
                         f"| {_fmt(c['mean_delta_mer'], '+.5f')} | {members} |")
        lines.append("")

    lines += ["### Plateau scores (section 6.2)", "",
             "For every ΔMER > 0 cell, how many of its up-to-4 immediate "
             "grid neighbours are also ΔMER > 0. Score 0 = isolated point.", ""]
    plateau = analysis.get("plateau_scores", {})
    for beams in D.PHASE1_BEAMS:
        ps = plateau.get(beams, []) or plateau.get(str(beams), [])
        lines += [f"**beams = {beams}**: " + (
            ", ".join(f"L{p['layer']}/a{p['alpha']:g}=score {p['plateau_score']}"
                     f"{' (isolated)' if p['isolated'] else ''}" for p in ps)
            if ps else "no ΔMER > 0 cells"), ""]
    return lines


def _q2_section(analysis: dict) -> list[str]:
    q2 = analysis.get("q2_paired_analysis", {})
    lines = ["## 7. Q2 -- do steering and beam search compose or substitute? "
             "(section 6.4)", "",
             "Paired per (layer, alpha): `gain_b1 = PIER(C00) - PIER(cell@b1)`, "
             "`gain_b5 = PIER(C00_beam5) - PIER(cell@b5)`. Reported across the "
             "whole grid, not the best cell.", "",
             f"**Mean paired difference (gain_b5 − gain_b1): "
             f"{_fmt(q2.get('mean_difference'), '+.5f')}**, 95% CI "
             f"[{_fmt(q2.get('ci_low'), '+.5f')}, {_fmt(q2.get('ci_high'), '+.5f')}], "
             f"{q2.get('n_groups', '?')} layer-clustered groups, "
             f"{q2.get('n_resamples', '?')} resamples.", "",
             f"**Composition: {q2.get('composition', 'UNDETERMINED')}**", "",
             "| layer | alpha | gain_b1 | gain_b5 | difference |",
             "|---:|---:|---:|---:|---:|"]
    for p in q2.get("pairs", []):
        lines.append(f"| {p['layer']} | {p['alpha']:g} | {_fmt(p['gain_b1'], '+.5f')} "
                     f"| {_fmt(p['gain_b5'], '+.5f')} | {_fmt(p['difference'], '+.5f')} |")
    lines.append("")
    return lines


def _q3_section(analysis: dict) -> list[str]:
    q3 = analysis.get("q3_chance_count", {})
    lines = ["## 8. Q3 -- chance expectation (section 6.5)", "",
             f"At {q3.get('n_cells', 70)} cells and nominal "
             f"{100 * q3.get('nominal_alpha', 0.05):.0f}%, "
             f"**~{_fmt(q3.get('expected_false_positives'), '.1f')} cells "
             f"are expected to pass all four gates by chance alone.**", "",
             f"**Observed: {q3.get('observed_go', 0)}** "
             f"({', '.join(f'`{c}`' for c in q3.get('observed_go_cells', [])) or 'none'})",
             "", ""]
    return lines


def _phase2_section(rows: Sequence[dict[str, Any]], state: dict) -> list[str]:
    lines = ["## 9. Phase 2 (section 7)", ""]
    if not state.get("phase2_triggered"):
        lines += ["**Not run.** No connected component reached the "
                 f"size-{state.get('min_component_size', 4)} threshold in "
                 "Phase 1. The null map is the finding; section 7 is explicit "
                 "that this is the correct outcome to report, not a "
                 "shortfall.", ""]
        return lines
    lines += ["### Phase 2a -- fine layer resolution", ""]
    lines += _full_table(rows, stage=D.STAGE_P2A)
    p2b = [r for r in rows if r.get("stage") == D.STAGE_P2B]
    if p2b:
        lines += ["### Phase 2b -- multi-layer, matched-energy vs "
                 "matched-per-layer pairs", "",
                 "Under `matched_energy=true`, combining k layers divides "
                 "per-layer intensity by √k. Each pair below isolates "
                 "dilution (matched_energy=true, lower per-layer dose) from "
                 "mechanism (matched_energy=false, alpha rescaled so each "
                 "layer gets the single-layer cell's own intensity).", ""]
        lines += _full_table(p2b, stage=D.STAGE_P2B)
    return lines


def _verdicts_section(analysis: dict) -> list[str]:
    largest = analysis.get("largest_component")
    if largest and largest.get("size", 0) > 0:
        q1 = (f"size {largest['size']} at beams={largest.get('beam_setting')}, "
             f"layers {largest['layer_range']}, alphas {largest['alpha_range']}, "
             f"mean ΔMER {_fmt(largest['mean_delta_mer'], '+.5f')}")
    else:
        q1 = "NONE -- no connected component of ΔMER > 0 cells"
    q2 = analysis.get("q2_paired_analysis", {})
    q3 = analysis.get("q3_chance_count", {})
    return [
        "## 10. Verdicts on the preregistered questions", "",
        "No \"best cell\" verdict is written here. If the map is flat, the "
        "map is flat.", "",
        f"**Q1_CONTIGUITY:** {q1}",
        f"**Q2_COMPOSITION:** {q2.get('composition', 'UNDETERMINED')}, mean "
        f"paired difference {_fmt(q2.get('mean_difference'), '+.5f')}, 95% CI "
        f"[{_fmt(q2.get('ci_low'), '+.5f')}, {_fmt(q2.get('ci_high'), '+.5f')}]",
        f"**Q3_CHANCE:** {q3.get('observed_go', 0)} observed vs "
        f"~{_fmt(q3.get('expected_false_positives'), '.1f')} expected by chance",
        "",
    ]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def write_summary_d_ext(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(out_dir / "results_D_ext.jsonl")
    state = json.loads((out_dir / "stage_d_ext_state.json").read_text(encoding="utf-8")) \
        if (out_dir / "stage_d_ext_state.json").exists() else {}
    det = state.get("determinism") or _load_json(out_dir / "determinism_D_ext.json")
    analysis = state.get("analysis", {})

    lines = [
        "# Stage D-extended -- dense (layer x alpha x beam) landscape map", "",
        f"**Stamp:** `{C.TAINT}`. "
        "`D-dev-confirm` is not opened. This phase proposes no Confirm "
        "candidate.", "",
        f"- git commit: `{(rows or [{}])[0].get('git_commit', 'unknown')}`",
        f"- population: `D-dev-select`, 300 utterances / 20 dialogues",
        f"- fixed config: site=decoder, coverage=local, direction="
        f"v_nat+v_prompt weighted (2.0 / 0.5), scale_mode=act_norm, "
        f"matched_energy=true",
        f"- swept: layer in {list(D.PHASE1_LAYERS)}, alpha in "
        f"{list(D.PHASE1_ALPHAS)}, num_beams in {list(D.PHASE1_BEAMS)} "
        f"= {len(D.PHASE1_LAYERS) * len(D.PHASE1_ALPHAS) * len(D.PHASE1_BEAMS)} cells",
        "", "This is a landscape map, not a search. No stage narrows to a "
        "winner; every cell is reported.", "", "---", "",
    ]
    lines += _preregistered_section()
    lines += _determinism_section(det)
    lines += _baselines_section(state)
    lines += _checks_section(state)
    lines += [f"## 4. Full {D.STAGE} table -- every cell, no exceptions", ""]
    lines += _full_table(rows, stage=D.STAGE)
    lines += _heatmaps_section(analysis)
    lines += _components_section(analysis)
    lines += _q2_section(analysis)
    lines += _q3_section(analysis)
    lines += _phase2_section(rows, state)
    lines += _verdicts_section(analysis)

    path = out_dir / "SUMMARY_D_EXT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
