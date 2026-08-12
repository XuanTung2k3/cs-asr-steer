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

Synthetic scoring runs in the `synthetic` part: `_score_synthetic_sets` aligns
the rendered splices with each family and writes measured absolute error to
`autoevidence.SYNTHETIC_SCORES_FILE`. If that table is absent, unauthenticated or
tainted, automatic Gate A still blocks with
`blocked_missing_synthetic_calibration` rather than substituting cross-aligner
agreement for accuracy -- agreement is not accuracy and is never used as such.

Status, per `csasr.lss.gates`: mechanical/reporting failures are `failed`
(something is broken); external/jitter/coverage failures are `completed_no_go`
(the experiment ran and the answer is no); missing evidence is `blocked`. The
group that failed still selects the pre-registered response in
`failure_response`. Exit code 0 for every terminal outcome except `failed`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss import spans as spans_mod
from ..lss.align import autoevidence, consensus_prod, coverage as coverage_mod
from ..lss.align import devselect as devselect_mod
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
    """Primary spans and bilingual utterances per role.

    `role` used to be dropped by consensus construction, so every count here was
    zero while tens of thousands of spans existed -- three per-role thresholds
    and `construct_bilingual_utterances` failed on that, not on the data.
    `consensus_prod.build` now joins the mapping back on; `role_counts_available`
    below is what makes its absence a reported defect instead of a zero.
    """
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
    """Every column downstream stages read is present and usable.

    `role` is required: `spans.SPAN_COLUMNS` declares it, `spans.load_spans`
    filters on it, and every per-role threshold is evaluated against it. A span
    table without it cannot support a role-specific claim, so its absence is a
    mechanical failure rather than a set of zero counts.
    """
    required = ("utterance_id", "unit_id", "role", "language",
                "consensus_start_sample", "consensus_end_sample",
                "confidence_bin", "spec_freeze_sha256")
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


def sweep_roles(cfg: dict, roles: list[str] | None = None) -> list[str]:
    """The roles this stage aligns and states coverage about."""
    dcfg = (cfg.get("alignment") or {}).get("diagnostics") or {}
    return list(roles or cfg.get("roles_to_label")
                or [str(dcfg.get("sample_role", "D-construct"))])


def sweep_sample(cfg: dict, roles: list[str] | None = None,
                 *, missing_ok: bool = False) -> pd.DataFrame:
    """The utterances this stage sets out to align.

    **Both** the aligner sweep and the coverage denominator must come from this
    one function. They were written twice and drifted: the sweep aligned
    `contains_code_switch].head(300)` per role, while coverage was measured
    against every unit in the *whole* role. On D-construct that is 10,167 units
    attempted against a denominator of 309,535 -- a coverage ceiling of 3.3%
    under a 0.95 threshold, so `alignment_unit_coverage` could not pass however
    good the aligners were, and Gate A would have reported `completed_no_go` on
    the coverage group for a bookkeeping mismatch rather than a measurement.

    The selection is deterministic and independent of any aligner's output, so
    it keeps the property the frozen denominator exists for: an aligner that
    silently drops utterances still shows up as missing coverage.
    """
    dcfg = (cfg.get("alignment") or {}).get("diagnostics") or {}
    per_role = int(dcfg.get("sample_utterances", 300))
    frames = []
    for role in sweep_roles(cfg, roles):
        try:
            manifest = load_role(cfg, role)
        except Exception:                       # a role that was never built
            if missing_ok:
                continue
            raise
        subset = manifest[manifest["contains_code_switch"]].head(per_role).copy()
        subset["role"] = role
        frames.append(subset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _expected_unit_universe(cfg: dict, roles: list[str]) -> pd.DataFrame:
    """The units the stage set out to align, per role.

    This is the denominator coverage must be measured against. Taking it from
    the candidate rows instead makes coverage 1.0 whenever an aligner silently
    drops utterances, which is the failure the threshold exists to catch.

    Restricted to EN/ZH units, because those are the only ones any family emits
    a candidate for (`nat5h.aligners._content_units`); counting the rest would
    charge every family for work no family attempts.
    """
    sample = sweep_sample(cfg, roles, missing_ok=True)
    if not len(sample):
        return pd.DataFrame()
    frames = []
    for role, rows in sample.groupby("role", sort=False):
        units = unit_table(rows)
        if not len(units):
            continue
        units = units.copy()
        units["role"] = role
        if "unit_id" in units and "reference_unit_index" not in units:
            units["reference_unit_index"] = units["unit_id"]
        frames.append(units)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[out["language"].isin(["EN", "ZH"])].reset_index(drop=True)


def _run_aligner_sweep(cfg: dict, log, *, overwrite: bool = False,
                       run_dir=None, roles: list[str] | None = None,
                       qwen_runner=None, parents: list | None = None,
                       taint: dict | None = None) -> dict:
    """Align the sampled role with every configured family.

    This is what turns Gate A from a toy into a measurement. The exploratory
    NAT5H table covers 20 utterances and yields 16 English spans -- an order of
    magnitude short of the 100 the audit needs -- so consensus built on it can
    only ever report a coverage failure that says nothing about the corpus.

    The decoder-query convention comes from L1a's frozen selection, so production
    alignment runs the variant the development evidence chose rather than whatever
    the default happens to be.
    """
    from ..lss.align import candidates as cand
    from ..lss.align import devselect
    from ..models.whisper import load_whisper
    from ..nat5h.coordinates import EncoderGeometry
    from ..nat5h.schema import RunIdentity

    root = Path(cfg["experiment"]["output_root"])
    # Every role a coverage threshold is stated about must actually be aligned,
    # or that threshold is being evaluated on somebody else's data. `sweep_sample`
    # is also what defines the coverage denominator, so the two cannot disagree.
    roles = sweep_roles(cfg, roles)
    sample = sweep_sample(cfg, roles)
    log.info("aligner sweep over %d bilingual utterances across %s",
             len(sample), roles)

    selection = devselect.load_aligner_selection(root)
    offset = selection.get("pred_start_offset") if selection.get("available") else None
    if not selection.get("available"):
        log.warning("no authenticated aligner selection from l1a (%s: %s); "
                    "aligning with the default decoder-query convention",
                    selection.get("reason"), selection.get("detail", ""))
    bundle = load_whisper(cfg)
    table, report = cand.run_families(
        sample, cfg, EncoderGeometry.from_bundle(bundle), RunIdentity.from_cfg(cfg),
        list((cfg.get("alignment") or {}).get("families", [])),
        bundle=bundle, out_dir=root / "alignments", overwrite=overwrite,
        pred_start_offset=offset, qwen_runner=qwen_runner, stage=STAGE,
        run_dir=run_dir, parents=[m for m in (parents or []) if m],
        taint_reasons=(taint or {}).get("taint_reasons") or (),
        purpose="natural")
    report["aligner_selection"] = selection
    report["pred_start_offset"] = offset
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
            parents=([m for m in (parents or []) if m]
                     + [m for m in (report.get("cache_manifests") or {}).values()
                        if m]),
            taint_reasons=(taint or {}).get("taint_reasons") or (),
            key_columns=CANDIDATE_KEYS, schema="nat5h_candidates_v2",
            extra={"request_manifest": report.get("request_manifest")})
    else:
        log.error("aligner sweep produced no candidates: %s", report)
    write_json(report, art(cfg, "diagnostics", "l1b_aligner_sweep.json"))
    return report



#: Families scored against synthetic ground truth.
#:
#: Every configured family, Qwen included. It used to be CTC and DTW only,
#: because `candidates.run_families` had no way to run Qwen over an arbitrary
#: manifest: it consumed whatever the L1a probe had persisted, keyed to eight
#: *corpus* utterance ids, which can never match a synthetic pair id. With
#: `csasr.lss.align.qwen_prod` there is a production path, so a family that is
#: allowed to qualify on natural speech is also scored on known boundaries --
#: which is what `_family_correspondence` then requires.
def synthetic_scored_families(cfg: dict) -> list[str]:
    return [str(f) for f in ((cfg.get("alignment") or {}).get("families") or [])]


def _score_synthetic_sets(cfg: dict, log, sets: dict, *, run_dir=None,
                          taint: dict | None = None,
                          parents: list | None = None,
                          labels: dict | None = None,
                          pred_start_offset: int | None = None,
                          qwen_runner=None,
                          gate_generation: int | None = None,
                          overwrite: bool = False) -> dict:
    """Run the aligner families over rendered synthetic audio and score them.

    This is the step whose absence made automatic Gate A block: `build_set`
    renders splices with boundaries known by construction, but until something
    aligns that audio there is no absolute error anywhere in the pipeline, and
    cross-aligner agreement is not a substitute.

    The score table is published with a manifest because
    `autoevidence.synthetic_calibration_status` authenticates it before reading:
    an unmanifested or tainted table is refused rather than believed.

    ``labels`` maps a purpose to its storage label, so a fresh gate *generation*
    writes into its own directory and can never read the cached candidate tables
    of the generation it replaces (`csasr.lss.align.exposure`).
    """
    from ..lss.align import candidates as cand
    from ..lss.align import devselect
    from ..models.whisper import load_whisper
    from ..nat5h.coordinates import EncoderGeometry
    from ..nat5h.schema import RunIdentity

    root = Path(cfg["experiment"]["output_root"])
    families = synthetic_scored_families(cfg)
    # `_candidates` holds DataFrames for the tolerance selection to read. Leading
    # underscore because the rest of this dict is written into the status file as
    # JSON, and a frame stringified into it is 4,000 characters of noise.
    report: dict = {"families_requested": families, "purposes": {},
                    "families_scored": [], "items_scored": {}, "unscorable": {},
                    "_candidates": {}}
    rendered = {name: frame for name, frame in sets.items() if len(frame)}
    if not rendered or not families:
        log.error("no synthetic set to score (sets=%s, families=%s)",
                  {k: len(v) for k, v in sets.items()}, families)
        return report

    bundle = load_whisper(cfg)
    geometry = EncoderGeometry.from_bundle(bundle)
    identity = RunIdentity.from_cfg(cfg)

    scores, per_items = [], []
    score_parents = [m for m in (parents or []) if m]
    request_fingerprints: dict[str, str] = {}
    item_fingerprints: dict[str, str] = {}
    for purpose, frame in rendered.items():
        manifest = synthetic_mod.aligner_manifest(frame)
        request = cand.request_manifest_record(manifest)
        request_fingerprints[str(purpose)] = str(request["sha256"])
        item_fingerprints[str(purpose)] = str(
            cand.request_manifest_fingerprint(frame))
        label = (labels or {}).get(purpose, purpose)
        out_dir = root / "synthetic" / label / "alignments"
        table, run_report = cand.run_families(
            manifest, cfg, geometry, identity, families, bundle=bundle,
            # a per-set directory, so the gate set's candidates can never be
            # read as the dev set's -- they are different items with the same
            # family names
            out_dir=out_dir,
            overwrite=overwrite, stage=STAGE, run_dir=run_dir,
            parents=[m for m in (parents or []) if m],
            taint_reasons=(taint or {}).get("taint_reasons") or (),
            purpose=f"synthetic_{purpose}",
            pred_start_offset=pred_start_offset,
            qwen_runner=None if qwen_runner is None
            else (lambda m, out_dir=out_dir, purpose=purpose, **kw:
                  qwen_runner(m, out_dir=out_dir, purpose=f"synthetic_{purpose}")))
        score_parents.extend(
            m for m in (run_report.get("cache_manifests") or {}).values() if m)
        score, per_item = synthetic_mod.score_rendered_set(
            frame, table, purpose=purpose)
        report["purposes"][purpose] = {
            "items": int(len(frame)), "candidate_rows": int(len(table)),
            "set_label": label,
            "request_manifest": request,
            "families": run_report.get("families", {}),
            "unscorable": synthetic_mod.unscorable_summary(per_item),
        }
        report["unscorable"][purpose] = report["purposes"][purpose]["unscorable"]
        report["_candidates"][purpose] = table
        if len(score):
            scores.append(score)
        if len(per_item):
            per_items.append(per_item)
        log.info("synthetic %s: %d items, %d candidate rows, %d score rows",
                 purpose, len(frame), len(table), len(score))

    if not scores:
        log.error("synthetic scoring produced no score rows: %s", report)
        return report

    all_scores = pd.concat(scores, ignore_index=True)
    all_items = pd.concat(per_items, ignore_index=True) if per_items else pd.DataFrame()
    report["families_scored"] = sorted(set(all_scores["family"].astype(str)))
    report["items_scored"] = {
        str(p): int(g["scorable"].sum()) for p, g in all_items.groupby("purpose")
    } if len(all_items) else {}

    manifest_mod.publish_frame(
        root / autoevidence.SYNTHETIC_SCORES_FILE, all_scores,
        stage=STAGE, cfg=cfg, run_dir=run_dir,
        parents=score_parents,
        taint_reasons=(taint or {}).get("taint_reasons"),
        key_columns=("family", "convention", "edge", "purpose"),
        schema=synthetic_mod.SCORES_SCHEMA_VERSION,
        # which gate generation these numbers describe. A later evaluation reads
        # the ledger's unexposed generation and refuses a table from an earlier
        # one, so a second run cannot quietly re-report the first one's scores.
        extra={"synthetic_gate_generation": gate_generation,
               "synthetic_set_request_sha256": request_fingerprints,
               "synthetic_gate_request_sha256": request_fingerprints.get("gate"),
               "synthetic_set_item_sha256": item_fingerprints,
               "synthetic_gate_item_sha256": item_fingerprints.get("gate"),
               "scientific_amendment": devselect.scientific_amendment(cfg)})
    if len(all_items):
        write_parquet(all_items, art(cfg, "metrics",
                                     "l1b_synthetic_boundary_error.parquet"))
        comparison = synthetic_mod.convention_comparison(all_items)
        if len(comparison):
            write_parquet(comparison, art(cfg, "metrics",
                                          "l1b_synthetic_convention_comparison.parquet"))
            report["convention_comparison"] = comparison.to_dict(orient="records")
            log.info("convention comparison (canonical vs legacy midpoint):\n%s",
                     comparison.to_string(index=False))
    log.info("synthetic scores:\n%s", all_scores.to_string(index=False))
    return report


def _qwen_runner(cfg: dict, log, rdir, taint: dict, parents: list):
    """The production Qwen path, bound to this run's provenance.

    A callable rather than a flag so `run_families` stays a pure dispatcher and a
    test can drive the *real* subprocess runner with a substituted adapter
    (`qwen_probe.ADAPTER_ENV_VAR`) instead of a fake that bypasses it.
    """
    from ..lss.align import qwen_prod

    def run(manifest, *, out_dir, purpose="natural"):
        result = qwen_prod.run_qwen_production(
            cfg, manifest, out_dir=out_dir, purpose=purpose, stage=STAGE,
            run_dir=rdir, parents=[m for m in parents if m],
            taint_reasons=taint.get("taint_reasons") or ())
        log.info("qwen production (%s): state=%s reason=%s rows=%d covered=%d/%d "
                 "chunks=%d/%d in %.1f s of %.0f s",
                 purpose, result.state, result.reason, result.candidate_rows,
                 result.covered_utterances, result.expected_utterances,
                 result.chunks_written, result.chunks_expected,
                 result.elapsed_seconds, result.deadline_seconds)
        return result

    return run


def _prepare_synthetic_evidence(cfg: dict, log, rdir, taint: dict, parents: list,
                                *, config, qwen_runner=None,
                                overwrite: bool = False) -> dict:
    """Development set, a fresh gate generation, scores, and the operating point.

    Order matters and is the whole point:

    1. the **development** set is loaded from what L1a published, not rebuilt, so
       the convention selected there and the tolerance selected here were chosen
       on the same items;
    2. the **gate** generation is rendered from sources no earlier generation has
       used, because job 38573's gate scores were read during review and a set
       that has been looked at cannot confirm anything;
    3. both are scored against their constructed boundaries;
    4. the operating tolerance is selected from the **development** rows only, by
       the preregistered rule, and frozen with the instrument recorded.

    Step 4 never sees a gate row: `devselect.assert_development_only` raises if it
    ever does.
    """
    from ..lss.align import devselect, exposure

    root = Path(cfg["experiment"]["output_root"])
    scfg = dict(cfg.get("synthetic") or {})
    gcfg = cfg.get("gate_a") or {}
    ccfg = (cfg.get("alignment") or {}).get("consensus") or {}
    out: dict = {"criteria": [], "blocked": [], "gate_generation": None}

    dev, dev_meta = devselect.load_dev_set(root)
    out["development_set"] = dev_meta
    if not len(dev):
        log.error("no authenticated synthetic development set at %s (%s: %s); "
                  "the operating tolerance cannot be selected and no aligner "
                  "configuration can be chosen", dev_meta.get("path"),
                  dev_meta.get("reason"), dev_meta.get("detail", ""))
        out["blocked"].append(autoevidence.BLOCKED_MISSING_DEVELOPMENT_SET)
        out["criteria"].append(reported("synthetic_development_set",
                                        str(dev_meta.get("reason"))))
        return out

    # ---- a gate generation no selection has seen ---------------------------
    ledger = exposure.bootstrap(root)
    exposure.save_ledger(root, ledger)
    active = [g for g in exposure.generations_for(ledger, "gate")
              if not g.get("exposed") and not g.get("abandoned")]
    if active:
        # A previous preparation did not reach the evaluation that exposes it.
        # Reusing its generation number for newly rendered items would make its
        # cached alignments and score manifest appear to describe the new set.
        # Retire the attempt, keep its sources reserved, and allocate a new,
        # immutable generation.
        log.warning("retiring %d unexposed gate generation(s) left by an "
                    "interrupted/preparation-only run: %s", len(active),
                    [int(g["generation"]) for g in active])
        ledger = exposure.abandon_unexposed(
            root, purpose="gate",
            reason="superseded by a new synthetic preparation after interruption")
    generation = exposure.next_generation(ledger, "gate")
    construct = load_role(cfg, str(scfg.get("source_role", "D-construct")))
    pools = synthetic_mod.partition_sources(
        construct, seed=seed_for(cfg, "synthetic_dev"))
    eligible, pool_report = exposure.eligible_sources(pools["gate"], ledger)
    log.info("gate generation %d: %d of %d gate-pool sources are unexposed "
             "(%d already read)", generation, pool_report["eligible"],
             pool_report["pool"], pool_report["excluded"])

    gate_set, gate_meta = synthetic_mod.build_set(
        construct, cfg, n_pairs=int(scfg.get("num_pairs_gate", 100)),
        seed=seed_for(cfg, "synthetic_gate"), out_dir=root / "synthetic",
        purpose="gate", sources=eligible, generation=generation)
    try:
        freshness = exposure.assert_unexposed(gate_set, ledger)
    except AssertionError as exc:
        # An exhausted pool is missing evidence, not a defect: a set whose scores
        # have been read cannot confirm a configuration chosen after reading them,
        # so there is nothing here to judge with. `blocked`, and the pool report
        # says how many sources are left.
        log.error("no fresh gate set could be built: %s", exc)
        out["blocked"].append(autoevidence.BLOCKED_EXPOSED_GATE_SET)
        # Reported, not a failing mechanical criterion: nothing is malformed, and
        # a mechanical failure outranks `blocked` and would report an exhausted
        # source pool as an implementation defect.
        out["criteria"] += [
            reported("synthetic_gate_set_is_fresh", 0),
            reported("synthetic_gate_pool", json.dumps(pool_report)),
        ]
        out["gate_pool"] = pool_report
        out["gate_freshness"] = {"fresh": False, "detail": str(exc)}
        return out
    from ..lss.align import candidates as candidate_cache

    gate_request = candidate_cache.request_manifest_record(
        synthetic_mod.aligner_manifest(gate_set))
    gate_item_sha256 = candidate_cache.request_manifest_fingerprint(gate_set)
    exposure.record_generation(
        root, gate_set, purpose="gate", generation=generation,
        reason="rendered for a Gate A evaluation",
        alignment_request_sha256=str(gate_request["sha256"]),
        item_set_sha256=str(gate_item_sha256))
    disjoint = synthetic_mod.assert_sources_disjoint({"dev": dev, "gate": gate_set})

    out.update({"gate_generation": generation, "gate_set": gate_meta,
                "gate_request_manifest": gate_request,
                "gate_item_sha256": gate_item_sha256,
                "gate_pool": pool_report, "gate_freshness": freshness,
                "disjoint": disjoint})
    for frame, name in ((dev, "dev"), (gate_set, "gate")):
        if len(frame):
            write_parquet(frame, art(cfg, "metrics",
                                     f"l1b_synthetic_{name}_items.parquet"))
    log.info("synthetic sets: dev=%d (reused from l1a, sha256 %s) "
             "gate=%d (generation %d), sources disjoint=%s; rendering is not "
             "scoring", len(dev), str(dev_meta.get("sha256"))[:12],
             gate_meta.get("pairs", 0), generation, disjoint["disjoint"])
    out["criteria"] += [
        reported("synthetic_dev_pairs_rendered", int(len(dev))),
        reported("synthetic_gate_pairs_rendered", gate_meta.get("pairs", 0)),
        reported("synthetic_gate_generation", generation),
        reported("synthetic_gate_sources_excluded_as_exposed",
                 pool_report["excluded"]),
        check("synthetic_sources_disjoint", int(bool(disjoint["disjoint"])), 1, "=="),
        reported("synthetic_gate_set_is_fresh", int(bool(freshness["fresh"]))),
    ]

    # ---- score both sets ---------------------------------------------------
    aligner = devselect.load_aligner_selection(root)
    offset = aligner.get("pred_start_offset") if aligner.get("available") else None
    scoring = _score_synthetic_sets(
        cfg, log, {"dev": dev, "gate": gate_set}, run_dir=rdir, taint=taint,
        parents=parents,
        labels={"dev": synthetic_mod.set_label("dev"),
                "gate": synthetic_mod.set_label("gate", generation)},
        pred_start_offset=offset, qwen_runner=qwen_runner,
        gate_generation=generation, overwrite=overwrite)
    dev_candidates = (scoring.pop("_candidates", {}) or {}).get(
        "dev", pd.DataFrame())
    out["scoring"] = scoring
    out["aligner_selection"] = aligner
    out["criteria"] += [
        reported("synthetic_families_scored",
                 ",".join(scoring.get("families_scored", [])) or "none"),
        reported("synthetic_items_scored",
                 json.dumps(scoring.get("items_scored", {}))),
        reported("synthetic_unscorable", json.dumps(scoring.get("unscorable", {}))),
        reported("aligner_selection_pred_start_offset", offset),
        reported("aligner_selection_source", str(aligner.get("reason"))),
    ]

    # ---- select the operating tolerance, on development rows only ----------
    tolerances = list(ccfg.get("tolerance_sweep_ms", [50, 100, 200]))
    evidence = devselect.tolerance_accuracy(
        dev, dev_candidates, config, tolerances,
        min_boundaries=int(gcfg.get("min_selection_boundaries", 50)))
    if len(evidence):
        write_parquet(evidence, root / devselect.SELECTION_TABLE)
        log.info("operating-tolerance selection on synthetic development "
                 "boundaries:\n%s", evidence.to_string(index=False))
    record = devselect.select_operating_tolerance(
        evidence,
        max_median_ms=float(gcfg.get("max_selection_median_abs_error_ms", 100.0)),
        max_p90_ms=float(gcfg.get("max_selection_p90_abs_error_ms", 200.0)))
    record["scientific_amendment"] = devselect.scientific_amendment(cfg)
    selected_config = devselect.apply_selection(config, record)
    payload = devselect.operating_point(
        record,
        aligner_selection=(aligner.get("payload") or {}),
        dev_artifact={k: dev_meta.get(k) for k in ("path", "sha256", "rows",
                                                  "producing_run_id")},
        seeds={"synthetic_dev": seed_for(cfg, "synthetic_dev"),
               "synthetic_gate": seed_for(cfg, "synthetic_gate")},
        config=selected_config)
    point = root / devselect.OPERATING_POINT_FILE
    write_json(payload, point)
    manifest_mod.publish(point, None, stage=STAGE, cfg=cfg, run_dir=rdir,
                         parents=[m for m in parents if m],
                         taint_reasons=taint.get("taint_reasons") or (),
                         schema=devselect.OPERATING_POINT_SCHEMA)
    out["operating_point"] = payload
    out["criteria"] += [
        reported("operating_tolerance_selected_ms",
                 record.get("selected_tolerance_ms")),
        reported("operating_tolerance_instrument", record.get("instrument")),
        reported("operating_tolerance_rejections",
                 json.dumps(record.get("rejection_reasons", {}))),
    ]
    if record.get("selected_tolerance_ms") is None:
        log.error("no swept tolerance satisfies the preregistered rule on the "
                  "synthetic development set: %s", record.get("rejection_reasons"))
    else:
        log.info("operating tolerance selected: %s ms by %s",
                 record["selected_tolerance_ms"], record["rule"])
    return out


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

        # The manifests of the bytes that unlocked this stage are parents of
        # every candidate cache and of the merged candidate table. Status alone
        # is mutable; these manifests are what make forced-prerequisite taint
        # transitive across a later retry.
        prerequisite_manifests = []
        for item in prereq.get("checks") or []:
            if item.get("kind") != "artifact" or not item.get("ok"):
                continue
            published = manifest_mod.load(root / str(item["name"]))
            if published:
                prerequisite_manifests.append(published)

        if args.align:
            metrics["aligner_sweep"] = _run_aligner_sweep(
                cfg, log, overwrite=args.overwrite, run_dir=rdir,
                roles=list(cfg.get("roles_to_label") or []),
                qwen_runner=_qwen_runner(cfg, log, rdir, taint,
                                         prerequisite_manifests),
                parents=prerequisite_manifests, taint=taint)

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

        # ---- synthetic ground truth and the operating point -----------------
        # Rendering is not scoring, and scoring is not selecting. All three
        # happen here, in that order, and the selection sees development rows
        # only.
        gate_generation = None
        if "synthetic" in parts and not args.dry_run:
            synthetic_evidence = _prepare_synthetic_evidence(
                cfg, log, rdir, taint, input_manifests, config=config,
                qwen_runner=_qwen_runner(cfg, log, rdir, taint, input_manifests),
                overwrite=args.overwrite)
            criteria += synthetic_evidence.pop("criteria", [])
            blocked_reasons += synthetic_evidence.pop("blocked", [])
            gate_generation = synthetic_evidence.get("gate_generation")
            metrics["synthetic"] = synthetic_evidence

        # ---- consensus at the selected operating point ----------------------
        operating = devselect_mod.load_operating_point(
            root, cfg=cfg, require_authentication=evaluating)
        metrics["operating_point"] = {k: v for k, v in operating.items()
                                      if k != "payload"}
        if operating["available"]:
            config = devselect_mod.apply_selection(
                config, (operating["payload"].get("selection") or {}))

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
            metrics["consensus"]["measurement_instrument"] = \
                config.measurement_instrument
            metrics["consensus"]["erosion_ms"] = config.erosion_ms
            metrics["consensus"]["union_padding_ms"] = config.union_padding_ms
            criteria += [
                reported("consensus_accepted", int(len(spans))),
                reported("consensus_tolerance_ms", config.tolerance_ms),
                reported("consensus_tolerance_selection_rule", config.selection_rule),
                reported("consensus_tolerance_instrument",
                         config.measurement_instrument),
                reported("consensus_estimator", config.estimator),
            ]
            if evaluating and not config.tolerance_selected:
                # Two different gaps, two different codes. Reporting one for the
                # other is how a run said "no synthetic calibration" while a
                # valid score table sat on disk.
                #
                # `no_qualifying` means the selection *ran* on development
                # evidence and nothing met the preregistered accuracy: a real
                # finding about the aligners, and the reason not to fall back to
                # a provisional 200 ms, because every number downstream would
                # then describe a configuration nobody chose.
                #
                # `unselected` means the selection did not run or its artifact is
                # not authentic -- missing evidence about the configuration
                # itself.
                ran = bool((operating.get("payload") or {}).get("selection", {})
                           .get("evidence"))
                if ran:
                    log.error("no swept tolerance meets the preregistered "
                              "accuracy on development boundaries; the operating "
                              "point is unselected and Gate A cannot describe a "
                              "configuration that was never chosen")
                    blocked_reasons.append(
                        autoevidence.BLOCKED_NO_QUALIFYING_TOLERANCE)
                else:
                    log.error("the operating tolerance was never selected (%s: "
                              "%s); run the `synthetic` part, which selects it "
                              "from development boundaries",
                              operating.get("reason"), operating.get("detail", ""))
                    blocked_reasons.append(
                        autoevidence.BLOCKED_UNSELECTED_TOLERANCE)
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
            min_family_coverage = float(gcfg.get("min_family_valid_unit_coverage",
                                                 0.95))
            # `valid_unit_coverage` is a rate over the rows a family produced, so
            # a family that attempted 8 utterances and got them all right scores
            # 1.0 and qualifies as a full independent aligner. That is exactly
            # what the Qwen probe writes: `probe_utterances` (8) of the swept 300.
            # An absolute floor tied to the same universe coverage is measured
            # against is what stops a probe-sized sample from counting as
            # production evidence and flipping
            # `blocked_insufficient_independent_aligners` on 3% of the units.
            min_family_units = int(math.ceil(min_family_coverage
                                             * len(expected_units)))
            independence = autoevidence.independent_valid_families(
                validity,
                min_coverage=min_family_coverage,
                max_invalid_rate=float(gcfg.get("max_invalid_rate", 0.01)),
                max_nonmonotonic_rate=float(gcfg.get("max_nonmonotonic_rate", 0.01)),
                min_units=min_family_units)
            coverage_stats = autoevidence.unit_coverage(candidates, expected_units)
            # The generation the ledger still calls unexposed. Scores from an
            # earlier one are refused however authentic the bytes are: those items
            # have been read, so they cannot confirm a configuration chosen after
            # reading them.
            from ..lss.align import exposure as exposure_mod

            ledger = exposure_mod.load_ledger(root)
            unexposed_generation = (int(gate_generation)
                                    if gate_generation is not None else
                                    exposure_mod.current_generation(ledger, "gate"))
            unexposed_entry = exposure_mod.find_generation(
                ledger, "gate", unexposed_generation) or {}
            calibration = autoevidence.synthetic_calibration_status(
                root, min_boundaries=int(gcfg.get("min_synthetic_boundaries", 100)),
                cfg=cfg, require_authentication=evaluating,
                expected_gate_generation=unexposed_generation if evaluating else None,
                expected_gate_request_sha256=(
                    unexposed_entry.get("alignment_request_sha256")
                    if evaluating else None),
                expected_gate_item_sha256=(
                    unexposed_entry.get("item_set_sha256")
                    if evaluating else None))
            # The pair Gate A speaks about has to be one pair. Disagreement is
            # measured between the families that both qualify naturally *and*
            # have synthetic accuracy evidence, so the two claims are about the
            # same aligners.
            correspondence = autoevidence.family_correspondence(
                independence["qualifying_families"],
                calibration.get("families_scored", []) if calibration["available"]
                else [],
                min_families=int(gcfg.get("min_valid_families", 2)))
            agreement = autoevidence.agreement_for_qualifying_pair(
                candidates, correspondence["corresponding_families"]
                or independence["qualifying_families"])

            metrics["automatic_evidence"] = {
                "roles_to_label": roles_to_label,
                "expected_units": int(len(expected_units)),
                "family_validity": validity.to_dict(orient="records") if len(validity) else [],
                "independence": independence,
                "unit_coverage": coverage_stats,
                "cross_aligner_agreement": agreement,
                "synthetic_calibration": calibration,
                "family_correspondence": correspondence,
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
                reported("min_valid_units_per_family", min_family_units),
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

            # The same pair on both sides, or the absolute-error claim is about
            # aligners that produced none of the spans being claimed about.
            criteria += [
                check("corresponding_qualifying_families",
                      correspondence["n_corresponding"],
                      int(gcfg.get("min_valid_families", 2)), ">=",
                      group="external"),
                reported("corresponding_aligner_families",
                         ",".join(correspondence["corresponding_families"]) or "none"),
                reported("natural_families_without_synthetic_calibration",
                         ",".join(correspondence["uncalibrated_natural_families"])
                         or "none"),
                reported("synthetically_scored_but_rejected_naturally",
                         ",".join(correspondence["scored_but_not_qualifying_classes"])
                         or "none"),
            ]
            if correspondence["uncalibrated_natural_families"] \
                    and calibration["available"]:
                # only meaningful once calibration exists at all; otherwise the
                # missing-calibration block above already says it
                log.error("qualifying natural families have no synthetic-gate "
                          "calibration: %s; a score from a family that was "
                          "rejected on natural speech cannot substitute for it",
                          correspondence["uncalibrated_natural_families"])
                blocked_reasons.append(autoevidence.BLOCKED_UNCALIBRATED_FAMILIES)

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
            log.info("primary spans per role: %s",
                     {r: v["spans"] for r, v in per_role.items()})
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

            # Spans exist but every role counts zero: that is the role column
            # having been dropped, not six empty roles. Reported as mechanical so
            # it reads as a defect instead of as a coverage result.
            role_counts_available = int(
                not len(primary)
                or any(v["spans"] for v in per_role.values()))
            criteria += [
                check("role_counts_available", role_counts_available, 1, "=="),
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
        actions = _next_actions(status, mode, blocked_reasons, criteria)
        metrics["next_actions"] = actions
        for action in actions:
            log.info("next action %d [%s]: %s", action["priority"],
                     action["kind"], action["action"])

        # Reading the gate scores exposes the generation they came from. This is
        # what makes the *next* evaluation build a fresh confirmatory set instead
        # of reporting a number from items whose results have been seen.
        if gate_generation is not None:
            from ..lss.align import exposure as exposure_mod

            exposure_mod.mark_exposed(
                root, purpose="gate", generation=gate_generation,
                reason=f"read by a Gate A evaluation that concluded {status}")

        rows = _criteria_frame(criteria)
        save_report(
            art(cfg, "reports", "l1b_gate_a.md"), f"L1b - Gate A ({mode} mode)",
            [
                ("Outcome", f"**{status}** (exit {exit_code(status)})"
                 + (f"\n\nBlocked on: `{'`, `'.join(blocked_reasons)}`"
                    if blocked_reasons else "")),
                ("Required next actions", md_table(pd.DataFrame(actions))
                 + "\n\nOrdered by dependency: an item cannot be worked on "
                   "before the items above it. `kind=manual` marks the only "
                   "entries a human annotator is the missing input for."),
                ("Evidence provenance", "```json\n" + json.dumps(
                    {"mode": mode, "candidates": metrics.get("candidate_source"),
                     "taint": taint,
                     "operating_point": metrics.get("operating_point"),
                     "gate_generation": gate_generation}, indent=2,
                    default=str) + "\n```"),
                ("Operating point (selected on development items)",
                 "```json\n" + json.dumps(
                     (metrics.get("synthetic") or {}).get("operating_point", {}),
                     indent=2, default=str) + "\n```"),
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
               mode=mode,
               next_action=str(actions[0]["action"]) if actions else "",
               next_actions=actions,
               blocked_reasons=blocked_reasons, taint=taint)
        log.info("Gate A: %s", status.upper())
        return exit_code(status)


#: Every blocker and failing criterion, and the one action that resolves it,
#: ordered by dependency: an item cannot be worked on before the items above it.
#: `manual` marks the two entries a human annotator is actually needed for --
#: recommending annotation for anything else is what made the previous
#: single-answer `_next_action` misleading, since no amount of annotation repairs
#: an aligner that cannot align.
_ACTION_LADDER: tuple[tuple[str, str, str], ...] = (
    (autoevidence.BLOCKED_TAINTED_INPUTS, "provenance",
     "re-run the prerequisite stages without --force-prereq; taint is bound to "
     "the artifact bytes and --overwrite cannot launder it"),
    (autoevidence.BLOCKED_NO_CANDIDATES, "alignment",
     "run the aligner sweep: l1b --align"),
    (autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES, "alignment",
     "re-run the aligner sweep so the candidate table is published with a "
     "manifest for this configuration"),
    (autoevidence.BLOCKED_EXPLORATORY_CANDIDATES, "alignment",
     "run the aligner sweep; only the recorded NAT5H exploratory table was "
     "available and it is 20 utterances of a different pipeline"),
    (autoevidence.BLOCKED_MISSING_DEVELOPMENT_SET, "configuration",
     "run l1a so the synthetic development set is rendered and published; every "
     "automatic configuration choice is made on it"),
    (autoevidence.BLOCKED_EXPOSED_GATE_SET, "accuracy",
     "no unexposed source audio is left for a fresh confirmatory gate; widen the "
     "synthetic source role or accept that this corpus can no longer confirm a "
     "new configuration (see synthetic/exposure_ledger.json)"),
    (autoevidence.BLOCKED_MISSING_SYNTHETIC, "accuracy",
     "run the `synthetic` part so exact-boundary scores are produced and "
     "authenticated (see autoevidence.MISSING_SYNTHETIC_DESCRIPTION)"),
    (autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC, "accuracy",
     "re-run the `synthetic` part; the score table on disk is unsigned, stale "
     "or malformed"),
    (autoevidence.BLOCKED_UNSELECTED_TOLERANCE, "configuration",
     "run the `synthetic` part, which selects the operating tolerance from "
     "development boundaries and freezes it in freeze/l1b_operating_point.json"),
    (autoevidence.BLOCKED_NO_QUALIFYING_TOLERANCE, "alignment",
     "repair alignment accuracy: no swept tolerance meets the preregistered "
     "median<=100 ms / p90<=200 ms on development boundaries, so no operating "
     "point exists to report a result at. See metrics/l1b_tolerance_selection.parquet"),
    (autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS, "alignment",
     "obtain a second independent valid aligner: repair whisper_dtw or bring "
     "qwen_forced_aligner up to full-manifest coverage"),
    (autoevidence.BLOCKED_UNCALIBRATED_FAMILIES, "accuracy",
     "score the families that qualify on natural speech against the synthetic "
     "gate set; a score from a rejected family cannot substitute"),
    (autoevidence.BLOCKED_NO_PAIRED_AGREEMENT, "alignment",
     "make two independent families align the same units; they currently align "
     "disjoint sets, so there is no paired evidence at all"),
    (autoevidence.BLOCKED_EMPTY_CONSENSUS, "alignment",
     "no accepted span reaches the primary confidence bins; repair alignment "
     "agreement before anything downstream can be measured"),
    (autoevidence.BLOCKED_MISSING_LANGUAGE, "alignment",
     "obtain primary spans in both EN and ZH; an EN-ZH comparison with one "
     "language compares nothing"),
    (autoevidence.BLOCKED_MISSING_JITTER, "robustness",
     "re-run the `jitter` part so the +/-50 and +/-100 ms checks are measured"),
    (autoevidence.BLOCKED_MISSING_VERDICTS, "manual",
     "annotate the audit pack (manual mode only): "
     "audit/l1b/verdicts_raw/<annotator-id>.csv"),
)

#: failing criterion -> the action that repairs it. Keyed by prefix, because the
#: per-family criteria carry the family name in the criterion name.
_CRITERION_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("invalid_rate_", "alignment",
     "repair the family's invalid spans; the proposal's limit is 1% and this is "
     "measured per family (see diagnostics/l1a_* for the attribution)"),
    ("nonmonotonic_rate_", "alignment",
     "repair the family's non-monotonic spans"),
    ("alignment_unit_coverage", "alignment",
     "align the units that are missing from the expected universe; coverage is "
     "measured against what the sweep set out to align, not what it produced"),
    ("synthetic_absolute_", "alignment",
     "repair boundary accuracy: absolute error against constructed boundaries "
     "is outside the proposal's thresholds. Fitting a correction is allowed only "
     "on development items and must be validated on a fresh gate set"),
    ("synthetic_boundaries_scored", "accuracy",
     "score more synthetic gate items; the boundary count is below the minimum"),
    ("corresponding_qualifying_families", "accuracy",
     "make the naturally qualifying families and the synthetically calibrated "
     "families the same pair"),
    ("jitter_", "robustness",
     "the mask does not survive the boundary error it is allowed to have; "
     "widening padding or reducing erosion changes the claim, so this is a "
     "repair-or-restrict decision from `failure_response`"),
    ("automatic_usable_item_rate", "coverage",
     "raise the fraction of spans that reach the primary confidence bins with a "
     "non-degenerate safe interior"),
    ("role_counts_available", "defect",
     "per-role counts are all zero while spans exist: the role column was "
     "dropped between consensus and coverage"),
    ("span_schema_valid", "defect",
     "the span table is missing a column downstream stages read"),
    ("min_loc_train_en_spans", "coverage",
     "not enough English spans in loc-train; the pre-registered coverage "
     "response ladder in `failure_response` applies"),
    ("min_dev_select_targets", "coverage",
     "not enough dev-select targets; see the `failure_response` ladder"),
    ("min_dev_confirm_targets", "coverage",
     "not enough dev-confirm targets; see the `failure_response` ladder"),
    ("construct_bilingual_utterances", "coverage",
     "not enough bilingual D-construct utterances reached consensus"),
)


def _next_actions(status: str, mode: str, blocked_reasons: list[str],
                  criteria: list[dict] | None = None) -> list[dict]:
    """Every action this outcome requires, ordered by dependency.

    The previous version returned the *first* matching blocker and, for the most
    common one, recommended manual annotation. That was misleading twice over: it
    hid the other blockers, and annotation does not repair a family whose invalid
    rate is 17%, an accuracy failure against known boundaries, a jitter failure or
    a coverage shortfall. Automatic repair comes first here, and `manual` appears
    only where a person is genuinely the missing input.
    """
    if status == "passed":
        return [{"priority": 1, "kind": "next_stage", "action": "run l1c_labels"}]

    out: list[dict] = []
    for reason, kind, action in _ACTION_LADDER:
        if reason in (blocked_reasons or []):
            out.append({"priority": len(out) + 1, "kind": kind,
                        "reason": reason, "action": action})
    seen: set[str] = set()
    for criterion in (criteria or []):
        if criterion.get("comparison") == "report" or criterion.get("passed"):
            continue
        name = str(criterion.get("name", ""))
        for prefix, kind, action in _CRITERION_ACTIONS:
            if name.startswith(prefix) and action not in seen:
                seen.add(action)
                out.append({"priority": len(out) + 1, "kind": kind,
                            "criterion": name, "group": criterion.get("group"),
                            "action": action})
                break
    if not out:
        out.append({"priority": 1, "kind": (
            "result" if status == "completed_no_go" else "defect"),
            "action": ("take the pre-registered failure response for the failing "
                       "group" if status == "completed_no_go"
                       else "repair the stage; a mechanical or reporting "
                            "criterion failed")})
    if mode == "automatic":
        # stated once, at the end, so it is never read as the primary action
        out.append({"priority": len(out) + 1, "kind": "optional",
                    "action": ("the optional manual audit is a second opinion on "
                               "boundary accuracy only: --prepare-manual-audit. "
                               "It repairs none of the items above except a "
                               "missing human verdict")})
    return out


def _next_action(status: str, mode: str, blocked_reasons: list[str],
                 criteria: list[dict] | None = None) -> str:
    """The single highest-priority action, for the one-line status field."""
    actions = _next_actions(status, mode, blocked_reasons, criteria)
    return str(actions[0]["action"]) if actions else "repair the stage"


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
