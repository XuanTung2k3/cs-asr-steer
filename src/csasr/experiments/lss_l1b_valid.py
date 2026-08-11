"""L1b - external boundary validation. This stage decides Gate A.

The primary path needs no human annotation:

    sbatch cs_asr_lss.sh l1b                       # prepare + evaluate, automatic
    python -m csasr.experiments.lss_l1b_valid --config configs/lss/l1b_valid.yaml \
        --prepare-audit                            # evidence only, then stop
    python -m csasr.experiments.lss_l1b_valid --config configs/lss/l1b_valid.yaml \
        --evaluate-gate                            # evaluate what is on disk

Manual boundary annotation is an *optional* second opinion, never the default:

    ... --prepare-manual-audit                     # renders clips + guide
    # humans fill audit/l1b/verdicts_raw/*.csv
    ... --evaluate-gate --mode manual

Automatic mode never reads a human verdict file; manual mode never invents one.

Automatic Gate-A evidence, and what each source is allowed to claim:

    two independent valid aligners  mechanical  a second opinion exists at all
    cross-aligner disagreement      external    how far two estimators differ
                                                -- NOT boundary error
    synthetic exact boundaries      external    true absolute error, because the
                                                boundary is known by construction
    boundary jitter +/-50/100 ms    jitter      does the mask survive being wrong
    frozen high-confidence subset   coverage    is there enough material
    coverage/validity/monotonicity  mechanical  are the spans well formed
    EN-ZH asymmetry                 external    is one language systematically worse

Synthetic scoring is not implemented yet (see
`csasr.lss.align.autoevidence.MISSING_SYNTHETIC_DESCRIPTION`), so automatic Gate
A currently blocks with `blocked_missing_synthetic_calibration` rather than
substituting agreement for accuracy. That is the honest state of the evidence.

Status, per `csasr.lss.gates`: mechanical/reporting failures are `failed`
(something is broken); external/jitter/coverage failures are `completed_no_go`
(the experiment ran and the answer is no); missing evidence is `blocked`. The
group that failed still selects the pre-registered response in
`failure_response`. Exit code 0 for every terminal outcome except `failed`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss import spans as spans_mod
from ..lss.align import autoevidence, consensus_prod, coverage as coverage_mod
from ..lss.align import jitter as jitter_mod
from ..lss.align import synthetic as synthetic_mod
from ..lss.artifacts import write_json, write_parquet
from ..lss.audit import pack as pack_mod
from ..lss.audit import verdicts as verdicts_mod
from ..lss.eligibility import unit_table
from ..lss.gates import check, evaluate, exit_code, reported
from ..lss.prereq import require_prerequisites
from ..lss.roles import load_role
from ..lss.seeds import seed_for
from ..lss.spans import REJECTED_FILE, SPANS_FILE, freeze_spans, spans_to_frames
from ..utils.config import art
from ..utils.hashing import sha256_file
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

STAGE = "l1b_valid"
PARTS = ("synthetic", "consensus", "evidence", "pack", "verdicts",
         "jitter", "coverage", "gate")

#: what each mode runs when no explicit `--only` is given
AUTOMATIC_PREPARE = ("synthetic", "consensus", "evidence", "jitter", "coverage")
AUTOMATIC_GATE = ("consensus", "evidence", "jitter", "coverage", "gate")
MANUAL_PREPARE = ("consensus", "pack")
MANUAL_GATE = ("consensus", "evidence", "verdicts", "jitter", "coverage", "gate")


CANDIDATE_KEYS = ("utterance_id", "reference_unit_index", "aligner_family",
                  "aligner_variant")

#: role -> the configured absolute-count threshold that is *about that role*.
#: Counting D-construct spans against `min_loc_train_en_spans` claimed a number
#: about loc-train without ever aligning loc-train.
ROLE_SPAN_THRESHOLDS = {
    "loc-train": "min_loc_train_en_spans",
    "D-dev-select": "min_dev_select_targets",
    "D-dev-confirm": "min_dev_confirm_targets",
}


def _role_span_counts(spans: pd.DataFrame, roles: list[str]) -> dict[str, dict]:
    """Primary spans and bilingual utterances per role."""
    out: dict[str, dict] = {}
    for role in roles:
        subset = spans[spans["role"] == role] if (len(spans) and "role" in spans) \
            else spans.iloc[0:0]
        language = subset["language"] if "language" in subset else pd.Series(dtype=str)
        out[role] = {
            "spans": int(len(subset)),
            "en_spans": int((language == "EN").sum()) if len(subset) else 0,
            "zh_spans": int((language == "ZH").sum()) if len(subset) else 0,
            "utterances": int(subset["utterance_id"].nunique()) if len(subset) else 0,
        }
    return out


def _span_schema_ok(spans: pd.DataFrame) -> bool:
    """Every column downstream stages read is present and usable."""
    required = ("utterance_id", "unit_id", "language", "consensus_start_sample",
                "consensus_end_sample", "confidence_bin", "spec_freeze_sha256")
    if any(c not in spans.columns for c in required):
        return False
    start = spans["consensus_start_sample"].astype(float)
    end = spans["consensus_end_sample"].astype(float)
    return bool((end > start).all() and (start >= 0).all()
                and spans["spec_freeze_sha256"].astype(str).str.len().gt(0).all())


def _candidates(cfg: dict, *, production: bool = True) -> tuple[pd.DataFrame, dict]:
    """Candidate spans to build consensus from, with authenticated provenance.

    Three sources, and only one of them may support a production Gate A:

    * `alignments/candidates_all.parquet` **with a valid manifest** -- produced
      by this pipeline, for this configuration, complete;
    * the same file without a manifest, or whose manifest does not authenticate
      it -- a leftover from another configuration or a partial write. Refused;
    * the recorded NAT5H table -- 20 exploratory utterances yielding 16 English
      spans. Useful for diagnostics, an order of magnitude short of what the
      gate needs, and produced by a different pipeline. Never production.
    """
    from ..lss.manifest import read_verified

    root = Path(cfg["experiment"]["output_root"])
    local = root / "alignments" / "candidates_all.parquet"
    if local.is_file():
        frame, verdict = read_verified(local, cfg=cfg, require_identity=production)
        return frame, {"source": "lss", "path": str(local),
                       "authenticated": bool(verdict["ok"]),
                       "verdict": verdict["verdict"],
                       "detail": verdict.get("detail", ""),
                       "manifest": verdict.get("manifest"),
                       "usable_for_production": bool(verdict["ok"])}

    fallback = Path(str(((cfg.get("alignment") or {}).get("diagnostics") or {})
                        .get("nat5h_candidates", "")))
    if fallback.is_file():
        return pd.read_parquet(fallback), {
            "source": "nat5h_recorded", "path": str(fallback),
            "authenticated": False, "verdict": "exploratory_artifact",
            "detail": ("recorded by the NAT5H pipeline over 20 exploratory "
                       "utterances; diagnostics only"),
            "manifest": None, "usable_for_production": False}

    return pd.DataFrame(), {"source": "none", "path": "", "authenticated": False,
                            "verdict": "missing", "detail": "",
                            "manifest": None, "usable_for_production": False}


def _expected_unit_universe(cfg: dict, roles: list[str]) -> pd.DataFrame:
    """The units the stage set out to align, per role.

    This is the denominator coverage must be measured against. Taking it from
    the candidate rows instead makes coverage 1.0 whenever an aligner silently
    drops utterances, which is the failure the threshold exists to catch.
    """
    frames = []
    for role in roles:
        try:
            manifest = load_role(cfg, role)
        except Exception:                       # a role that was never built
            continue
        units = unit_table(manifest)
        if not len(units):
            continue
        units = units.copy()
        units["role"] = role
        if "unit_id" in units and "reference_unit_index" not in units:
            units["reference_unit_index"] = units["unit_id"]
        frames.append(units)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_aligner_sweep(cfg: dict, log, *, overwrite: bool = False,
                       run_dir=None, roles: list[str] | None = None) -> dict:
    """Align the sampled role with every configured family.

    This is what turns Gate A from a toy into a measurement. The exploratory
    NAT5H table covers 20 utterances and yields 16 English spans -- an order of
    magnitude short of the 100 the audit needs -- so consensus built on it can
    only ever report a coverage failure that says nothing about the corpus.
    """
    from ..lss.align import candidates as cand
    from ..models.whisper import load_whisper
    from ..nat5h.coordinates import EncoderGeometry
    from ..nat5h.schema import RunIdentity

    root = Path(cfg["experiment"]["output_root"])
    dcfg = (cfg.get("alignment") or {}).get("diagnostics") or {}
    roles = list(roles or cfg.get("roles_to_label")
                 or [str(dcfg.get("sample_role", "D-construct"))])
    per_role = int(dcfg.get("sample_utterances", 300))

    # Every role a coverage threshold is stated about must actually be aligned,
    # or that threshold is being evaluated on somebody else's data.
    frames = []
    for role in roles:
        manifest = load_role(cfg, role)
        subset = manifest[manifest["contains_code_switch"]].head(per_role).copy()
        subset["role"] = role
        frames.append(subset)
    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("aligner sweep over %d bilingual utterances across %s",
             len(sample), roles)

    bundle = load_whisper(cfg)
    table, report = cand.run_families(
        sample, cfg, EncoderGeometry.from_bundle(bundle), RunIdentity.from_cfg(cfg),
        list((cfg.get("alignment") or {}).get("families", [])),
        bundle=bundle, out_dir=root / "alignments", overwrite=overwrite)
    report["roles"] = roles
    report["utterances"] = int(len(sample))
    report["utterances_per_role"] = {
        r: int((sample["role"] == r).sum()) for r in roles} if len(sample) else {}
    report["rows"] = int(len(table))
    if len(table):
        if "role" not in table.columns and len(sample):
            table = table.merge(sample[["utterance_id", "role"]].drop_duplicates(),
                                on="utterance_id", how="left")
        # published with a manifest, so a later run can tell this table apart
        # from a leftover of another configuration or a partial write
        report["manifest"] = manifest_mod.publish_frame(
            root / "alignments" / "candidates_all.parquet", table,
            stage=STAGE, cfg=cfg, run_dir=run_dir,
            key_columns=CANDIDATE_KEYS, schema="nat5h_candidates_v2")
    else:
        log.error("aligner sweep produced no candidates: %s", report)
    write_json(report, art(cfg, "diagnostics", "l1b_aligner_sweep.json"))
    return report



def _resolve_parts(args) -> tuple[set[str], str, str]:
    """Which parts run, in which mode, and what the run is called."""
    mode = "manual" if (args.mode == "manual" or args.prepare_manual_audit) \
        else "automatic"
    prepare_only = bool(args.prepare_audit or args.prepare_manual_audit)

    if args.only:
        parts = {p.strip() for p in args.only.split(",") if p.strip()}
        intent = "explicit"
    elif prepare_only:
        parts = set(MANUAL_PREPARE if mode == "manual" else AUTOMATIC_PREPARE)
        intent = "prepare"
    elif args.evaluate_gate:
        parts = set(MANUAL_GATE if mode == "manual" else AUTOMATIC_GATE)
        intent = "evaluate"
    else:                                   # the default: prepare and evaluate
        parts = set(MANUAL_PREPARE if mode == "manual" else AUTOMATIC_PREPARE)
        parts |= set(MANUAL_GATE if mode == "manual" else AUTOMATIC_GATE)
        intent = "prepare_and_evaluate"

    if args.evaluate_gate and intent == "explicit":
        parts |= set(MANUAL_GATE if mode == "manual" else AUTOMATIC_GATE)

    # automatic mode may never read a human verdict file, whatever was asked
    if mode == "automatic":
        parts.discard("verdicts")
    return parts, mode, intent


def _run(argv: list[str] | None = None) -> int:
    parser = base_parser("L1b - external validation and Gate A", "lss/l1b_valid.yaml")
    parser.add_argument("--mode", choices=("automatic", "manual"), default="automatic",
                        help="automatic (default) uses aligner evidence only; "
                             "manual uses human boundary verdicts")
    parser.add_argument("--only", default=None,
                        help=f"comma-separated subset of {','.join(PARTS)}")
    parser.add_argument("--prepare-audit", action="store_true",
                        help="build the audit evidence for --mode and stop")
    parser.add_argument("--prepare-manual-audit", action="store_true",
                        help="shorthand for --mode manual --prepare-audit")
    parser.add_argument("--align", action="store_true",
                        help="run the aligner families over the sampled role first, "
                             "writing alignments/candidates_all.parquet")
    parser.add_argument("--evaluate-gate", action="store_true",
                        help="evaluate Gate A from the evidence on disk")
    args = parser.parse_args(argv)
    if args.output_dir:
        raise SystemExit("--output-dir is not supported for LSS stages")

    cfg, rdir, log = prepare(args, STAGE, claim=False)
    root = Path(cfg["experiment"]["output_root"])
    parts, mode, intent = _resolve_parts(args)
    unknown = parts - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown --only part(s): {sorted(unknown)}")
    evaluating = "gate" in parts

    gcfg = cfg.get("gate_a") or {}
    ccfg = (cfg.get("alignment") or {}).get("consensus") or {}
    audit_dir = root / "audit" / "l1b"

    with StageLock(root, STAGE):
        claim_stage(cfg, STAGE, rdir)
        prereq = require_prerequisites(root, STAGE, force=args.force_prereq,
                                       overwrite=args.overwrite, log=log)
        taint = prereq["taint"]
        metrics: dict = {"prerequisites": prereq, "mode": mode, "intent": intent,
                         "parts": sorted(parts), "taint": taint}
        criteria: list[dict] = []
        natural_criteria: list[dict] = []
        blocked_reasons: list[str] = []
        log.info("mode=%s intent=%s parts=%s", mode, intent, sorted(parts))

        if args.align:
            metrics["aligner_sweep"] = _run_aligner_sweep(
                cfg, log, overwrite=args.overwrite, run_dir=rdir,
                roles=list(cfg.get("roles_to_label") or []))

        candidates, candidate_source = _candidates(cfg, production=evaluating)
        metrics["candidate_rows"] = int(len(candidates))
        metrics["candidate_source"] = candidate_source
        if not len(candidates):
            log.error("no usable candidate alignments (%s: %s); run the aligner "
                      "sweep first: l1b --align",
                      candidate_source["verdict"], candidate_source.get("detail", ""))
            blocked_reasons.append(
                autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES
                if candidate_source["source"] != "none"
                else autoevidence.BLOCKED_NO_CANDIDATES)
        elif evaluating and not candidate_source["usable_for_production"]:
            # a nonempty exploratory or unauthenticated table is the dangerous
            # case: it looks like evidence and measures like evidence
            log.error("candidate evidence is not production-grade (%s: %s)",
                      candidate_source["verdict"], candidate_source.get("detail", ""))
            blocked_reasons.append(
                autoevidence.BLOCKED_EXPLORATORY_CANDIDATES
                if candidate_source["source"] == "nat5h_recorded"
                else autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES)

        # taint bound to the bytes we are about to read, not just to statuses
        input_manifests = [candidate_source.get("manifest")]
        artifact_taint = manifest_mod.taint_of_inputs(input_manifests)
        if artifact_taint["diagnostic_only"]:
            taint = manifest_mod.merge_taint([taint, artifact_taint])
            metrics["taint"] = taint

        # Diagnostic inputs can never produce a production gate result. This is
        # checked before any measurement so a tainted run cannot pass by luck.
        if taint["diagnostic_only"]:
            log.error("inputs are diagnostic-only (%s); Gate A cannot conclude",
                      ",".join(taint["taint_reasons"]))
            blocked_reasons.append(autoevidence.BLOCKED_TAINTED_INPUTS)

        config = consensus_prod.ConsensusConfig.from_cfg(ccfg)

        # ---- synthetic ground truth ----------------------------------------
        if "synthetic" in parts and not args.dry_run:
            construct = load_role(cfg, str((cfg.get("synthetic") or {})
                                           .get("source_role", "D-construct")))
            # Two seeds give two draws from the same pool, not two disjoint
            # sets. The pool is partitioned first, then each half is rendered.
            pools = synthetic_mod.partition_sources(
                construct, seed=seed_for(cfg, "synthetic_dev"))
            dev, dev_meta = synthetic_mod.build_set(
                construct, cfg, n_pairs=int((cfg.get("synthetic") or {})
                                            .get("num_pairs_dev", 100)),
                seed=seed_for(cfg, "synthetic_dev"),
                out_dir=root / "synthetic", purpose="dev",
                sources=pools["dev"])
            gate_set, gate_meta = synthetic_mod.build_set(
                construct, cfg, n_pairs=int((cfg.get("synthetic") or {})
                                            .get("num_pairs_gate", 100)),
                seed=seed_for(cfg, "synthetic_gate"),
                out_dir=root / "synthetic", purpose="gate",
                sources=pools["gate"])
            disjoint = synthetic_mod.assert_sources_disjoint(
                {"dev": dev, "gate": gate_set})
            for frame, name in ((dev, "dev"), (gate_set, "gate")):
                if len(frame):
                    write_parquet(frame, art(cfg, "metrics",
                                             f"l1b_synthetic_{name}_items.parquet"))
            metrics["synthetic_sets"] = {"dev": dev_meta, "gate": gate_meta,
                                         "disjoint": disjoint}
            log.info("synthetic sets rendered: dev=%d gate=%d, sources disjoint=%s; "
                     "rendering is not scoring", dev_meta.get("pairs", 0),
                     gate_meta.get("pairs", 0), disjoint["disjoint"])
            criteria += [
                reported("synthetic_dev_pairs_rendered", dev_meta.get("pairs", 0)),
                reported("synthetic_gate_pairs_rendered", gate_meta.get("pairs", 0)),
                check("synthetic_sources_disjoint", int(bool(disjoint["disjoint"])),
                      1, "=="),
            ]

        # ---- consensus + tolerance sweep ------------------------------------
        spans = pd.DataFrame()
        rejected = pd.DataFrame()
        if "consensus" in parts and len(candidates):
            sweep = consensus_prod.tolerance_sweep(
                candidates, config, list(ccfg.get("tolerance_sweep_ms", [50, 100, 200])))
            write_parquet(sweep, art(cfg, "metrics", "l1b_tolerance_sweep.parquet"))
            metrics["tolerance_sweep"] = sweep.to_dict(orient="records")
            log.info("tolerance sweep:\n%s", sweep.to_string(index=False))

            spans, rejected, pairwise = consensus_prod.build(candidates, config)
            if len(spans):
                spans = spans_to_frames(spans)
                # bind every span to the spec that governs it; an empty string
                # here made the freeze unverifiable downstream
                spec_path = root / str(cfg["experiment"]["spec_freeze_path"])
                spans["spec_freeze_sha256"] = (sha256_file(spec_path)
                                               if spec_path.is_file() else "")
                manifest_mod.publish_frame(
                    root / SPANS_FILE, spans, stage=STAGE, cfg=cfg, run_dir=rdir,
                    parents=[m for m in input_manifests if m],
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("utterance_id", "unit_id"),
                    schema=spans_mod.SPAN_SCHEMA_VERSION)
            if len(rejected):
                manifest_mod.publish_frame(
                    root / REJECTED_FILE, rejected, stage=STAGE, cfg=cfg,
                    run_dir=rdir, parents=[m for m in input_manifests if m],
                    taint_reasons=taint["taint_reasons"])
            metrics["consensus"] = {
                "accepted": int(len(spans)), "rejected": int(len(rejected)),
                "tolerance_ms": config.tolerance_ms,
                "bins": (spans["confidence_bin"].value_counts().to_dict()
                         if len(spans) else {}),
            }
            metrics["consensus"]["tolerance_selected"] = config.tolerance_selected
            metrics["consensus"]["selection_rule"] = config.selection_rule
            criteria += [
                reported("consensus_accepted", int(len(spans))),
                reported("consensus_tolerance_ms", config.tolerance_ms),
                reported("consensus_tolerance_selection_rule", config.selection_rule),
                reported("consensus_estimator", config.estimator),
            ]
            if evaluating and not config.tolerance_selected:
                # The preregistered rule picks the tolerance from measured
                # external accuracy. Without that measurement the operating
                # tolerance is a default, and labelling it "selected" would
                # claim a decision that was never taken.
                log.error("the operating tolerance was never selected: "
                          "`select_primary_tolerance` needs authenticated "
                          "external accuracy, which does not exist yet")
                blocked_reasons.append(autoevidence.BLOCKED_MISSING_SYNTHETIC)
            for _, row in sweep.iterrows():
                criteria.append(reported(
                    f"en_retention_at_{int(row['tolerance_ms'])}ms", row["en_retention"]))
        elif "consensus" not in parts and (root / SPANS_FILE).is_file():
            spans = pd.read_parquet(root / SPANS_FILE)
            if (root / REJECTED_FILE).is_file():
                rejected = pd.read_parquet(root / REJECTED_FILE)

        # ---- automatic external evidence -------------------------------------
        if "evidence" in parts:
            roles_to_label = list(cfg.get("roles_to_label")
                                  or [str((cfg.get("alignment") or {})
                                          .get("diagnostics", {})
                                          .get("sample_role", "D-construct"))])
            expected_units = _expected_unit_universe(cfg, roles_to_label)
            validity = autoevidence.family_validity(candidates)
            if len(validity):
                write_parquet(validity, art(cfg, "metrics", "l1b_family_validity.parquet"))
            independence = autoevidence.independent_valid_families(
                validity,
                min_coverage=float(gcfg.get("min_family_valid_unit_coverage", 0.95)),
                max_invalid_rate=float(gcfg.get("max_invalid_rate", 0.01)),
                max_nonmonotonic_rate=float(gcfg.get("max_nonmonotonic_rate", 0.01)))
            coverage_stats = autoevidence.unit_coverage(candidates, expected_units)
            agreement = autoevidence.agreement_for_qualifying_pair(
                candidates, independence["qualifying_families"])
            calibration = autoevidence.synthetic_calibration_status(
                root, min_boundaries=int(gcfg.get("min_synthetic_boundaries", 100)),
                cfg=cfg, require_authentication=evaluating)

            metrics["automatic_evidence"] = {
                "roles_to_label": roles_to_label,
                "expected_units": int(len(expected_units)),
                "family_validity": validity.to_dict(orient="records") if len(validity) else [],
                "independence": independence,
                "unit_coverage": coverage_stats,
                "cross_aligner_agreement": agreement,
                "synthetic_calibration": calibration,
            }
            write_json(metrics["automatic_evidence"],
                       art(cfg, "diagnostics", "l1b_automatic_evidence.json"))
            log.info("independent valid aligner families: %d (%s); rejected: %s",
                     independence["n_independent_valid_families"],
                     ",".join(independence["qualifying_families"]) or "none",
                     independence["rejection_reasons"] or "none")

            n_independent = independence["n_independent_valid_families"]
            min_families = int(gcfg.get("min_valid_families", 2))
            criteria += [
                # coverage is a proposal 4.1 threshold on a measured quantity,
                # so missing it is a result, not a crash
                check("alignment_unit_coverage", coverage_stats["coverage"],
                      float(gcfg.get("min_unit_coverage", 0.95)), ">=",
                      group="coverage"),
                reported("coverage_denominator", coverage_stats["denominator"]),
                reported("expected_reference_units", coverage_stats["reference_units"]),
                reported("unaligned_expected_units", coverage_stats["missing_units"]),
                reported("independent_valid_aligner_families", n_independent),
                reported("candidate_source", candidate_source["source"]),
                reported("candidate_authenticated",
                         int(bool(candidate_source["authenticated"]))),
                reported("qualifying_aligner_families",
                         ",".join(independence["qualifying_families"]) or "none"),
                reported("rejected_aligner_families",
                         ",".join(independence.get("rejected_families", [])) or "none"),
            ]
            if coverage_stats["denominator"] != "expected_universe":
                # a coverage number whose denominator came from the candidate
                # rows cannot detect dropped utterances, so it is not evidence
                log.error("coverage has no frozen expected-unit universe for %s",
                          roles_to_label)
                blocked_reasons.append(autoevidence.BLOCKED_NO_CANDIDATES
                                       if not len(candidates)
                                       else autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES)

            # both rates gated at 0.01, independently: the 0.95 coverage floor
            # would otherwise admit 5% invalid spans against a 1% limit
            for row in (validity.to_dict(orient="records") if len(validity) else []):
                family = row["aligner_family"]
                criteria += [
                    check(f"invalid_rate_{family}", row["invalid_rate"],
                          float(gcfg.get("max_invalid_rate", 0.01)), "<=",
                          group="external"),
                    check(f"nonmonotonic_rate_{family}", row["nonmonotonic_rate"],
                          float(gcfg.get("max_nonmonotonic_rate", 0.01)), "<=",
                          group="external"),
                    reported(f"valid_unit_coverage_{family}",
                             row["valid_unit_coverage"]),
                ]

            if n_independent < min_families:
                # A second opinion that does not exist is missing infrastructure,
                # not a broken stage and not a measurement: `blocked`. This is
                # the observed state today -- existing_ctc valid, whisper_dtw
                # below the validity floor -- and it must never read as a pass.
                blocked_reasons.append(autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS)

            # natural-speech disagreement: reported and gated as *disagreement*.
            # Two families can each qualify while aligning disjoint unit sets, in
            # which case there is no paired evidence at all -- that is missing
            # evidence, not agreement, and it blocks.
            if agreement.get("n"):
                natural_criteria += [
                    check("cross_aligner_boundary_disagreement_ms",
                          agreement["cross_aligner_boundary_disagreement_ms"],
                          float(gcfg.get("max_cross_aligner_boundary_disagreement_ms",
                                         100.0)), "<=", group="external"),
                    check("cross_aligner_boundary_disagreement_p90_ms",
                          agreement["cross_aligner_boundary_disagreement_p90_ms"],
                          float(gcfg.get("max_cross_aligner_boundary_disagreement_p90_ms",
                                         200.0)), "<=", group="external"),
                    # the proposal says the EN-ZH difference must be *under* 30 ms
                    check("cross_aligner_en_zh_disagreement_diff_ms",
                          agreement["en_zh_disagreement_diff_ms"],
                          float(gcfg.get("max_cross_aligner_en_zh_diff_ms", 30.0)),
                          "<", group="external"),
                    check("paired_cross_aligner_units", agreement["n"],
                          int(gcfg.get("min_paired_units", 100)), ">=",
                          group="external"),
                    reported("cross_aligner_start_disagreement_ms",
                             agreement["cross_aligner_start_disagreement_ms"]),
                    reported("cross_aligner_end_disagreement_ms",
                             agreement["cross_aligner_end_disagreement_ms"]),
                    reported("cross_aligner_disagreement_note",
                             autoevidence.NATURAL_SPEECH_NOTE),
                ]
                # naming is the mechanism by which a reader learns what the
                # evidence supports, so it is enforced rather than reviewed
                autoevidence.assert_no_absolute_error_claims(natural_criteria)
                criteria += natural_criteria
                by_language = agreement.get("by_language") or {}
                missing_language = [lang for lang in ("EN", "ZH")
                                    if not by_language.get(lang, {}).get("n")]
                if missing_language:
                    log.error("no paired evidence for %s; the EN-ZH comparison "
                              "has nothing to compare", missing_language)
                    blocked_reasons.append(autoevidence.BLOCKED_MISSING_LANGUAGE)
                    criteria.append(reported("languages_without_paired_evidence",
                                             ",".join(missing_language)))
            else:
                log.error("no unit was aligned by two independent families: %s",
                          agreement.get("reason", "zero paired units"))
                blocked_reasons.append(autoevidence.BLOCKED_NO_PAIRED_AGREEMENT)
                criteria.append(reported("paired_cross_aligner_units", 0))

            # absolute error requires constructed or annotated boundaries
            if calibration["available"]:
                absolute = calibration["absolute_boundary_error"]
                criteria += [
                    check("synthetic_boundaries_scored", calibration["num_boundaries"],
                          int(gcfg.get("min_synthetic_boundaries", 100)), ">=",
                          group="external"),
                    check("synthetic_absolute_error_within_100ms",
                          absolute["within_100ms"],
                          float(gcfg.get("min_synthetic_within_100ms", 0.90)), ">=",
                          group="external"),
                    check("synthetic_absolute_median_error_ms",
                          absolute["median_abs_error_ms"],
                          float(gcfg.get("max_human_median_abs_error_ms", 100.0)), "<=",
                          group="external"),
                    check("synthetic_absolute_p90_error_ms",
                          absolute["p90_abs_error_ms"],
                          float(gcfg.get("max_human_p90_abs_error_ms", 200.0)), "<=",
                          group="external"),
                    check("synthetic_absolute_bias_ms", absolute["max_abs_bias_ms"],
                          float(gcfg.get("max_synthetic_abs_bias_ms", 50.0)), "<=",
                          group="external"),
                    check("synthetic_independent_families_scored",
                          len(calibration["independence_classes_scored"]),
                          int(gcfg.get("min_valid_families", 2)), ">=",
                          group="external"),
                    reported("synthetic_edges_scored",
                             ",".join(calibration["edges_scored"])),
                ]
            elif mode == "automatic":
                log.error("%s: %s", calibration["reason"],
                          calibration.get("detail") or
                          calibration["missing_implementation"])
                blocked_reasons.append(calibration["reason"])
                criteria.append(reported("synthetic_calibration",
                                         calibration["reason"]))

        # ---- manual audit pack (optional mode only) --------------------------
        if "pack" in parts and mode != "manual":
            # checked before anything is built, so the refusal does not depend
            # on whether spans happen to exist yet
            raise SystemExit(
                "the human annotation pack belongs to the manual workflow; "
                "run --prepare-manual-audit (or --mode manual --only pack)")
        if "pack" in parts and len(spans):
            spec = pack_mod.AuditPackSpec.from_cfg(cfg.get("audit"))
            manifest = load_role(cfg, "D-construct")
            summary = pack_mod.build_pack(
                spans, manifest, audit_dir, spec,
                seed=seed_for(cfg, "audit_sample"),
                blinding_seed=seed_for(cfg, "audit_blinding"),
                perturb_seed=seed_for(cfg, "audit_perturb"),
                render_audio=not args.dry_run)
            write_json(summary, art(cfg, "metrics", "l1b_audit_pack.json"))
            metrics["audit_pack"] = summary
            log.info("manual audit pack: %s items (%s EN / %s ZH, %s decoys)",
                     summary.get("items"), summary.get("en"), summary.get("zh"),
                     summary.get("decoys"))
            leaked = {"confidence_bin", "n_families", "language", "aligner_family"} \
                & set(summary.get("blinded_fields", []))
            criteria.append(check("audit_pack_is_blinded", int(not leaked), 1, "=="))

        # ---- human verdicts (manual mode only) --------------------------------
        human: dict = {}
        if "verdicts" in parts and mode == "manual":
            raw_dir = audit_dir / "verdicts_raw"
            table, ingest_report = verdicts_mod.ingest(raw_dir, audit_dir)
            metrics["verdict_ingest"] = ingest_report
            if len(table):
                unblinded = verdicts_mod.unblind(table, audit_dir / "blinding_key.json")
                human = verdicts_mod.boundary_error(unblinded)
                agree = verdicts_mod.agreement(unblinded)
                adjudication = verdicts_mod.adjudicate(unblinded)
                write_parquet(adjudication, art(cfg, "metrics", "l1b_adjudication.parquet"))
                write_json({"boundary_error": human, "agreement": agree},
                           art(cfg, "metrics", "l1b_audit_agreement.json"))
                metrics["human"] = {"boundary_error": human, "agreement": agree}
                usable = verdicts_mod.usable_fraction(unblinded)
                by_language = human.get("by_language", {})
                # these ARE absolute errors: an annotator supplied the boundary
                criteria += [
                    check("human_units_audited", human.get("n", 0),
                          int(gcfg.get("min_human_units", 200)), ">=", group="external"),
                    check("human_units_en", by_language.get("EN", {}).get("n", 0),
                          int(gcfg.get("min_human_units_per_language", 100)), ">=",
                          group="external"),
                    check("human_units_zh", by_language.get("ZH", {}).get("n", 0),
                          int(gcfg.get("min_human_units_per_language", 100)), ">=",
                          group="external"),
                    check("human_units_with_corrected_times",
                          human.get("units_with_corrected_times", 0),
                          int(gcfg.get("min_human_units_with_times", 120)), ">=",
                          group="external"),
                    check("human_usable_fraction", usable,
                          float(gcfg.get("min_human_usable_fraction", 0.90)), ">=",
                          group="external"),
                    check("human_median_absolute_boundary_error_ms",
                          human.get("median_abs_error_ms", float("nan")),
                          float(gcfg.get("max_human_median_abs_error_ms", 100.0)), "<=",
                          group="external"),
                    check("human_p90_absolute_boundary_error_ms",
                          human.get("p90_abs_error_ms", float("nan")),
                          float(gcfg.get("max_human_p90_abs_error_ms", 200.0)), "<=",
                          group="external"),
                    check("human_en_zh_median_diff_ms",
                          human.get("en_zh_median_diff_ms", float("nan")),
                          float(gcfg.get("max_human_en_zh_median_diff_ms", 30.0)), "<=",
                          group="external"),
                    check("inter_annotator_alpha", agree.get("inter_annotator_alpha", float("nan")),
                          float(gcfg.get("min_inter_annotator_alpha", 0.67)), ">=",
                          group="external"),
                    check("decoy_detection_rate", agree.get("decoy_detection_rate", float("nan")),
                          float(gcfg.get("min_decoy_detection_rate", 0.80)), ">=",
                          group="external"),
                ]
            else:
                log.warning("no human verdicts found in %s; manual Gate A cannot "
                            "conclude", raw_dir)
                blocked_reasons.append(autoevidence.BLOCKED_MISSING_VERDICTS)
                criteria.append(reported("human_verdicts", "absent"))

        # The frozen high-confidence subset is what every primary claim is made
        # on, so it is also what the robustness checks must be run on. Measuring
        # jitter over spans that are excluded from the claims answers a question
        # nobody asked.
        primary = spans_mod.high_confidence_subset(spans) if len(spans) else spans
        metrics["primary_subset"] = {
            "accepted": int(len(spans)), "primary": int(len(primary)),
            "bins": list(spans_mod.PRIMARY_BINS),
        }
        if evaluating and not len(primary):
            log.error("no accepted span reaches the primary confidence bins %s; "
                      "there is nothing for the robustness checks to run on",
                      list(spans_mod.PRIMARY_BINS))
            blocked_reasons.append(autoevidence.BLOCKED_EMPTY_CONSENSUS)

        # ---- jitter stability, on the primary subset --------------------------
        if "jitter" in parts:
            summary: dict = {}
            if len(primary):
                table, summary = jitter_mod.jitter_stability(primary, cfg)
                write_parquet(table, art(cfg, "metrics", "l1b_jitter.parquet"))
            metrics["jitter"] = summary
            metrics["jitter_population"] = "primary_high_medium_subset"
            missing_offsets = []
            for offset in (50, 100):
                measured = summary.get(f"offset_{offset}ms")
                if not measured:
                    # a robustness check that never ran is missing evidence, and
                    # omitting its criterion is how a gate passes vacuously
                    missing_offsets.append(offset)
                    criteria.append(check(f"jitter_measured_{offset}ms", 0, 1, "==",
                                          group="jitter"))
                    continue
                criteria += [
                    check(f"jitter_median_mask_iou_{offset}ms",
                          measured.get("median_mask_iou", float("nan")),
                          float(gcfg.get(f"min_jitter_mask_iou_{offset}ms",
                                         gcfg.get("min_jitter_mask_iou_100ms", 0.60))),
                          ">=", group="jitter"),
                    check(f"jitter_cross_language_contamination_{offset}ms",
                          measured.get("contaminated_rate", float("nan")),
                          float(gcfg.get(
                              f"max_jitter_cross_language_contamination_{offset}ms",
                              gcfg.get("max_jitter_cross_language_contamination_100ms",
                                       0.05))),
                          "<=", group="jitter"),
                    check(f"jitter_safe_interior_survival_{offset}ms",
                          measured.get("safe_interior_survival", float("nan")),
                          float(gcfg.get(
                              f"min_jitter_safe_interior_survival_{offset}ms",
                              gcfg.get("min_jitter_safe_interior_survival_100ms", 0.90))),
                          ">=", group="jitter"),
                ]
            if missing_offsets and evaluating:
                log.error("jitter was not measured at %s ms", missing_offsets)
                blocked_reasons.append(autoevidence.BLOCKED_MISSING_JITTER)

        # ---- coverage, partition and selection bias ---------------------------
        if "coverage" in parts:
            roles_to_label = list(cfg.get("roles_to_label") or ["D-construct"])
            units = _expected_unit_universe(cfg, roles_to_label)
            partition = coverage_mod.assert_partition(spans, rejected, units)
            label = coverage_mod.label_coverage(spans, rejected, units)
            bias_table = coverage_mod.selection_bias(spans, rejected, units)
            write_parquet(label, art(cfg, "metrics", "l1b_label_coverage.parquet"))
            write_parquet(bias_table, art(cfg, "metrics", "l1b_selection_bias.parquet"))

            # Every configured absolute count, evaluated against the role it is
            # about. Counting D-construct spans and calling them loc-train
            # claimed a number about a role that was never aligned.
            per_role = _role_span_counts(primary, roles_to_label)
            usable = autoevidence.usable_item_rate(spans)
            eligibility = autoevidence.language_eligibility(primary)
            write_json({"partition": partition,
                        "per_role_spans": per_role,
                        "usable_items": usable,
                        "language_eligibility": eligibility,
                        "selection_bias": coverage_mod.selection_bias_summary(bias_table)},
                       art(cfg, "diagnostics", "l1b_coverage.json"))
            metrics["coverage"] = {"partition": partition,
                                   "per_role_spans": per_role,
                                   "usable_items": usable,
                                   "language_eligibility": eligibility,
                                   "label_coverage": label.to_dict(orient="records")}

            criteria += [
                check("accepted_rejected_partition_exact",
                      int(bool(partition["partition_exact"])), 1, "=="),
                check("selection_bias_report_written", int(len(bias_table) > 0), 1, "=="),
                check("label_coverage_report_written", int(len(label) > 0), 1, "=="),
                check("automatic_usable_item_rate", usable["usable_rate"],
                      float(gcfg.get("min_automatic_usable_rate", 0.90)), ">=",
                      group="external"),
                reported("frozen_primary_spans", int(len(primary))),
                reported("usable_items", usable["usable"]),
            ]
            for role, threshold_key in ROLE_SPAN_THRESHOLDS.items():
                if role not in roles_to_label:
                    continue
                criteria.append(check(
                    f"{threshold_key}", per_role.get(role, {}).get("en_spans", 0),
                    int(gcfg.get(threshold_key, 0)), ">=", group="coverage"))
            construct_utterances = per_role.get("D-construct", {}).get("utterances", 0)
            criteria.append(check(
                "construct_bilingual_utterances", construct_utterances,
                int(gcfg.get("min_construct_bilingual_utterances", 500)), ">=",
                group="coverage"))
            if not eligibility["eligible"]:
                log.error("primary spans are missing language subset(s): %s",
                          eligibility["missing_languages"])
                blocked_reasons.append(autoevidence.BLOCKED_MISSING_LANGUAGE)

        # the span table must exist and be well formed whenever a gate is decided
        if evaluating:
            criteria.append(check("span_schema_valid",
                                  int(len(spans) > 0 and _span_schema_ok(spans)),
                                  1, "=="))

        # ---- gate --------------------------------------------------------------
        if not evaluating:
            # A preparation run produces evidence; it does not decide anything.
            # Emitting a gate here is what previously let `--only ...,pack`
            # report Gate A as passed on mechanical checks alone.
            return _finish_preparation(cfg, STAGE, rdir, log, metrics, mode, intent,
                                       taint, prereq, blocked_reasons)

        gate_payload, status = evaluate(
            criteria, name="gate_a", blocked_reasons=blocked_reasons,
            note=(f"Gate A, {mode} mode. Accuracy thresholds come from the "
                  "proposal and are never relaxed to continue. Cross-aligner "
                  "disagreement on natural speech is not boundary error; "
                  "absolute error comes only from constructed synthetic "
                  "boundaries or manual annotation. The failing group selects "
                  "the pre-registered response in `failure_response`."))
        metrics["gate_status"] = status
        metrics["blocked_reasons"] = blocked_reasons

        rows = _criteria_frame(criteria)
        save_report(
            art(cfg, "reports", "l1b_gate_a.md"), f"L1b - Gate A ({mode} mode)",
            [
                ("Outcome", f"**{status}** (exit {exit_code(status)})"
                 + (f"\n\nBlocked on: `{'`, `'.join(blocked_reasons)}`"
                    if blocked_reasons else "")),
                ("Evidence provenance", "```json\n" + json.dumps(
                    {"mode": mode, "candidates": metrics.get("candidate_source"),
                     "taint": taint}, indent=2, default=str) + "\n```"),
                ("Automatic evidence", "```json\n" + json.dumps(
                    metrics.get("automatic_evidence", {}), indent=2, default=str) + "\n```"),
                ("Consensus", "```json\n" + json.dumps(metrics.get("consensus", {}),
                                                       indent=2, default=str) + "\n```"),
                ("Tolerance sweep", md_table(pd.DataFrame(
                    metrics.get("tolerance_sweep", [])))),
                ("Human audit (manual mode only)", "```json\n" + json.dumps(
                    metrics.get("human", {}), indent=2, default=str) + "\n```"),
                ("Jitter", "```json\n" + json.dumps(metrics.get("jitter", {}),
                                                    indent=2, default=str) + "\n```"),
                ("Coverage", "```json\n" + json.dumps(metrics.get("coverage", {}),
                                                      indent=2, default=str) + "\n```"),
                ("Gate A", md_table(rows)),
                ("Failure response", "```json\n" + json.dumps(
                    cfg.get("failure_response", {}), indent=2) + "\n```"),
            ],
        )
        write_parquet(rows, art(cfg, "metrics", "l1b_gate_criteria.parquet"))

        # The span freeze is the artifact l1c consumes. Only a passed, untainted
        # gate may produce it: freezing spans from a blocked or no-go run would
        # hand downstream stages production labels Gate A never endorsed.
        if status == "passed" and not taint["diagnostic_only"] and len(spans):
            freeze = freeze_spans(cfg, spans, rejected,
                                  {"tolerance_ms": config.tolerance_ms,
                                   "bins": dict(config.bins or {}),
                                   "gate_a_mode": mode,
                                   "gate_a_status": status})
            metrics["span_freeze"] = freeze
            log.info("froze %d spans for downstream stages", len(spans))
        else:
            log.info("spans not frozen: Gate A is %s%s", status,
                     " (diagnostic inputs)" if taint["diagnostic_only"] else "")

        finish(cfg, STAGE, rdir, metrics, gate_payload,
               artifacts=[str(art(cfg, "reports", "l1b_gate_a.md"))],
               status_override=status if status != "passed" else None,
               forced_prereq=bool(prereq.get("forced_run")),
               mode=mode, next_action=_next_action(status, mode, blocked_reasons),
               blocked_reasons=blocked_reasons, taint=taint)
        log.info("Gate A: %s", status.upper())
        return exit_code(status)


def _next_action(status: str, mode: str, blocked_reasons: list[str]) -> str:
    if status == "passed":
        return "run l1c_labels"
    if autoevidence.BLOCKED_MISSING_SYNTHETIC in blocked_reasons:
        return ("implement synthetic exact-boundary scoring, or run the optional "
                "manual path: --prepare-manual-audit")
    if autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS in blocked_reasons:
        return ("obtain a second independent valid aligner (repair whisper_dtw or "
                "enable qwen_forced_aligner)")
    if autoevidence.BLOCKED_MISSING_VERDICTS in blocked_reasons:
        return "annotate_audit_pack"
    if autoevidence.BLOCKED_TAINTED_INPUTS in blocked_reasons:
        return "re-run the prerequisite stages without --force-prereq"
    if status == "completed_no_go":
        return "take the pre-registered failure response for the failing group"
    return "repair the stage; a mechanical or reporting criterion failed"


def _finish_preparation(cfg, stage, rdir, log, metrics, mode, intent, taint,
                        prereq, blocked_reasons) -> int:
    """Terminal, successful end of a preparation-only run.

    Producing evidence is a complete piece of work with a complete result: it
    exits 0 and says what to do next. It emits no gate, because it decided
    nothing.
    """
    if mode == "manual":
        # only the explicitly chosen manual workflow waits on people
        status, next_action = "awaiting_manual_verdicts", "annotate_audit_pack"
    else:
        status, next_action = "completed", "evaluate_gate_a"
    finish(cfg, stage, rdir, metrics, None,
           status_override=status,
           # neither preparation state decides Gate A, so neither unlocks l1c
           full_stage_pass=False,
           mode=mode, intent=intent, next_action=next_action,
           blocked_reasons=blocked_reasons,
           forced_prereq=bool(prereq.get("forced_run")),
           taint=taint)
    log.info("%s audit evidence prepared (status=%s); next: %s",
             mode, status, next_action)
    return exit_code(status)


def _criteria_frame(criteria: list[dict]) -> pd.DataFrame:
    """Gate criteria as a table whose cells are all strings."""
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
