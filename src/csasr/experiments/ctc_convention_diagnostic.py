"""CTC blank-run boundary conventions, scored against cached candidates.

Diagnostic-only.  This command never invokes an aligner, writes a production
stage status, allocates or exposes a gate generation, builds or freezes spans,
or unlocks L1c.  It reads one cached candidate table, restricts it to the
development roles, and reports what the CTC-vs-Whisper boundary disagreement
would be under each of four named conventions.

It selects nothing.  Choosing a convention by which one minimises disagreement
with Whisper would be fitting to another aligner's output, and another aligner's
output is not gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..lss.artifacts import write_json, write_report
from ..utils.config import load_config
from ..utils.hashing import sha256_obj

REPORT_JSON = "ctc_boundary_conventions.json"
REPORT_MD = "ctc_boundary_conventions.md"
DISAGREEMENT_TABLE = "ctc_convention_disagreement.parquet"
TRANSITION_TABLE = "ctc_switch_transitions.parquet"

CANDIDATE_RELPATH = "alignments/candidates_all.parquet"


def evaluate_conventions(candidates: pd.DataFrame, *,
                         family: str = conv.CONVENTION_FAMILY,
                         reference_family: str = conv.REFERENCE_FAMILY,
                         declared_source_roles: Sequence[str] = (),
                         conventions: Sequence[str] | None = None
                         ) -> dict[str, Any]:
    """The payload alone, for callers that do not publish the inventory."""
    return build_diagnostic(candidates, family=family,
                            reference_family=reference_family,
                            declared_source_roles=declared_source_roles,
                            conventions=conventions)[0]


def resolve_conventions(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Which conventions to compute, always including the reference one.

    ``blank_excluded`` is the identity convention every other number is stated
    against, so it is never optional.  The order is the declared order, not the
    order asked for, so two runs of the same set are comparable.
    """
    if not requested:
        return conv.CONVENTIONS
    unknown = sorted(set(map(str, requested)) - set(conv.CONVENTIONS))
    if unknown:
        raise SystemExit(f"unknown boundary convention(s): {unknown}")
    wanted = set(map(str, requested)) | {"blank_excluded"}
    return tuple(c for c in conv.CONVENTIONS if c in wanted)


def build_diagnostic(candidates: pd.DataFrame, *,
                     family: str = conv.CONVENTION_FAMILY,
                     reference_family: str = conv.REFERENCE_FAMILY,
                     declared_source_roles: Sequence[str] = (),
                     conventions: Sequence[str] | None = None
                     ) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pure payload plus the transition inventory it was computed from.

    Returned together so the published inventory is the one the numbers came
    from, rather than a second enumeration that could differ.

    ``candidates`` must already contain only development roles.  This function
    refuses a frame that carries a held-out role rather than filtering one out:
    filtering here would mean held-out rows had already been read, and the
    guarantee would rest on this call instead of on the read.
    """
    development = candidates
    conv.assert_development_only(development)
    selected = resolve_conventions(conventions)
    transitions, edges = conv.switch_transitions(development, family=family)

    built: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    for convention in selected:
        built[convention] = conv.convention_targets(
            development, convention, family=family, transitions=transitions)

    # The variant the statistics use, read back from the production selection
    # rule, so the edge-case counts describe that same population.
    variant = conv.representative_variant(built["blank_excluded"][0], family=family)
    edge_cases = conv.edge_case_counts(transitions, edges, variant=variant)

    own = [conv.agreement_entry(targets, aggregation, convention,
                                family=family, reference_family=reference_family)
           for convention, (targets, aggregation) in built.items()]

    # One denominator for all four, so a convention cannot look better merely by
    # pairing a different set of targets.
    common: set[tuple[str, str]] | None = None
    for targets, _ in built.values():
        ids = conv.paired_target_ids(targets, [family, reference_family])
        common = ids if common is None else (common & ids)
    common = common or set()
    restricted = [
        conv.agreement_entry(targets, aggregation, convention, family=family,
                             reference_family=reference_family, restrict_to=common)
        for convention, (targets, aggregation) in built.items()]

    payload = {
        "diagnostic_only": True,
        "production_gate_evaluated": False,
        "production_spans_frozen": False,
        "l1c_unlocked": False,
        "held_out_gate_generation_consumed": False,
        "convention_selected": None,
        "conventions_computed": list(selected),
        "conventions_available": list(conv.CONVENTIONS),
        "all_conventions_computed": list(selected) == list(conv.CONVENTIONS),
        "selection_note": (
            "This diagnostic reports every convention it computed and selects "
            "none. Choosing by minimum disagreement would fit to another "
            "aligner. When conventions_computed is a subset of "
            "conventions_available the table is partial by request, not "
            "because a convention was dropped."),
        "convention_kind": (
            "blank-run reassignment, not an offset subtraction: each boundary is "
            "a pure function of the blank run bracketed by the two raw spans"),
        "convention_definitions": {
            "blank_excluded": "preceding end stays b_start, following start stays b_end",
            "blank_to_preceding": "preceding end extends to b_end",
            "blank_to_following": "following start moves back to b_start",
            "blank_midpoint": "both edges move to (b_start + b_end) / 2",
        },
        "switch_universe_source": (
            "csasr.lss.align.annotation_pack.language_runs reconstructs the "
            "runs and adjacent_run_transitions supplies the directly adjacent "
            "run pairs; every actionable transition is one of those pairs, so a "
            "convention can only move a boundary the annotation pack would also "
            "recognise"),
        "switch_universe_difference_from_annotation_pack": (
            "the inventory additionally enumerates target-language run pairs "
            "separated by a third-language run, which the pack drops silently. "
            "They are classified multiple_blank_runs, are never actionable, and "
            "exist only so the denominator is visible. The pack also keeps "
            "zero-length and overlapping gaps as eligible items, which this "
            "inventory classifies and excludes from action, so the two "
            "populations are related but not identical."),
        "role_scope": conv.role_scope(
            development, declared_source_roles=declared_source_roles),
        "family_under_convention": family,
        "family_variant_used": variant,
        "family_variants_present": sorted(
            {str(v) for v in transitions["aligner_variant"].unique()}
            if len(transitions) else set()),
        "reference_family": reference_family,
        "edge_cases": edge_cases,
        "edge_case_semantics": {
            conv.NO_BLANK_RUN:
                "the two runs overlap, so no blank run exists between them",
            conv.MULTIPLE_BLANK_RUNS:
                "one or more runs in a third language sit between them, so the "
                "interval holds several blank runs plus audio a third unit owns",
            conv.ZERO_LENGTH_BLANK_RUN:
                "the runs touch; all four conventions coincide here",
            conv.UTTERANCE_BOUNDARY_EDGE:
                "run edges with no adjacent language run (the first target run's "
                "start and the last one's end in each utterance); counted as "
                "edges, two per utterance that has any target run",
            conv.NON_CONTIGUOUS:
                "the reference-unit indices skip, so a unit absent from the "
                "table owns part of the interval",
            conv.MERGED_BY_AGGREGATION:
                "two same-language target runs separated only by a third-language "
                "run; the aggregation merges them, so no boundary exists here",
        },
        "blank_run_summary": conv.blank_run_summary(transitions, variant=variant),
        "target_edges_moved_vs_blank_excluded": conv.target_edges_moved(
            built, family=family),
        "common_paired_targets": int(len(common)),
        "per_convention_own_paired_set": own,
        "per_convention_common_paired_set": restricted,
        "encoder_frame_note": (
            "Convention edges are maintained in the second and sample domains "
            "only. start_encoder_frame/end_encoder_frame are left untouched and "
            "flagged stale, because in the cached CTC rows they already disagree "
            "with the second-domain edges."),
    }
    return payload, transitions


def convention_tables(payload: dict[str, Any]) -> pd.DataFrame:
    """Both paired-set views, one row per convention each."""
    edge_cases = payload.get("edge_cases") or {}
    moved = payload.get("target_edges_moved_vs_blank_excluded") or {}
    return pd.concat(
        [conv.convention_table(payload["per_convention_own_paired_set"],
                               edge_cases=edge_cases, moved=moved),
         conv.convention_table(payload["per_convention_common_paired_set"],
                               edge_cases=edge_cases, moved=moved)],
        ignore_index=True)


def _render_table(table: pd.DataFrame) -> str:
    """Markdown when the optional renderer is present, CSV when it is not.

    The parquet beside it is the machine-readable copy; this is for reading, so
    a missing optional dependency must not cost the report its numbers.
    """
    try:
        return table.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + table.to_csv(index=False) + "```"


def resolve_output(diagnostic_output: str | Path, root: Path) -> Path:
    """Diagnostic output must land outside the production artifacts root.

    A taint sidecar is a safety net, not permission to place a file where a
    downstream glob or an older consumer might mistake it for production.
    """
    output = Path(diagnostic_output).resolve()
    try:
        output.relative_to(Path(root).resolve())
    except ValueError:
        return output
    raise SystemExit("diagnostic output must be outside the production artifacts root")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only CTC boundary-convention development diagnostic")
    parser.add_argument("--config", default="lss/l1b_valid.yaml")
    parser.add_argument("--diagnostic-output", required=True,
                        help="diagnostic-only output directory, outside the "
                             "production artifacts root")
    parser.add_argument("--convention", dest="conventions", action="append",
                        default=[], choices=list(conv.CONVENTIONS),
                        help="compute only these conventions (repeatable); "
                             "blank_excluded is always included as the "
                             "reference. Default: all four.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / CANDIDATE_RELPATH
    # The role predicate goes into the parquet scan: held-out rows are never
    # materialised, rather than materialised and then filtered away.
    candidates, verdict = conv.read_development_candidates(candidate_path)
    if not verdict["ok"]:
        raise SystemExit(f"cached candidates are not authentic: {verdict}")

    output = resolve_output(args.diagnostic_output, root)
    output.mkdir(parents=True, exist_ok=True)

    payload, transitions = build_diagnostic(
        candidates, declared_source_roles=cfg.get("roles_to_label") or (),
        conventions=args.conventions)
    payload["config_sha256"] = sha256_obj(cfg)
    payload["source_artifacts_root"] = str(root)
    payload["source_candidate_path"] = str(candidate_path)
    payload["source_candidate_manifest"] = verdict.get("manifest") or {}
    table = convention_tables(payload)

    parent = verdict.get("manifest")
    extra = {
        "diagnostic_only": True,
        "production_gate_evaluated": False,
        "held_out_gate_generation_consumed": False,
        "production_spans_frozen": False,
        "l1c_unlocked": False,
        "convention_selected": None,
        "development_roles": list(conv.DEVELOPMENT_ROLES),
        "source_candidate_path": str(candidate_path),
    }
    for path, frame, schema in (
            (output / DISAGREEMENT_TABLE, table, conv.CONVENTION_DISAGREEMENT_SCHEMA),
            (output / TRANSITION_TABLE, transitions, conv.TRANSITION_SCHEMA)):
        manifest_mod.publish_frame(
            path, frame, stage="l1b_convention_diagnostic", cfg=cfg,
            parents=[parent] if parent else (), taint_reasons=(conv.TAINT,),
            schema=schema, extra=extra)

    report = output / REPORT_JSON
    write_json(payload, report)
    markdown = output / REPORT_MD
    write_report(
        markdown,
        "CTC boundary conventions — development diagnostic",
        [("Scope", "Diagnostic-only: no production gate was evaluated, no "
          "held-out role was read, no generation was consumed, and no spans "
          "were frozen. No convention was selected."),
         ("Conventions", _render_table(table)),
         ("Evidence", "```json\n" + json.dumps(payload, indent=2, default=str)
          + "\n```")])
    for path, schema in (
            (report, "lss_ctc_boundary_convention_diagnostic_v1"),
            (markdown, "lss_ctc_boundary_convention_diagnostic_report_v1")):
        manifest_mod.publish(
            path, None, stage="l1b_convention_diagnostic", cfg=cfg,
            parents=[parent] if parent else (), taint_reasons=(conv.TAINT,),
            schema=schema, extra=extra)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
