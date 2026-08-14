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
    lexical reference boundaries    external    true absolute lexical error
    synthetic audio splice          diagnostic  seam-relative offsets only
    boundary jitter +/-50/100 ms    jitter      does the mask survive being wrong
    frozen high-confidence subset   coverage    is there enough material
    coverage/validity/monotonicity  mechanical  are the spans well formed
    EN-ZH asymmetry                 external    is one language systematically worse

Reference scoring runs in the `synthetic` part. If no genuine lexical reference
is available, or if its table is absent, unauthenticated or tainted, automatic
Gate A blocks with
`blocked_missing_genuine_lexical_calibration` (or an authentication-specific
blocker) rather than substituting cross-aligner
agreement for accuracy -- agreement is not accuracy and is never used as such.

Status, per `csasr.lss.gates`: mechanical/reporting failures are `failed`
(something is broken); external/jitter/coverage failures are `completed_no_go`
(the experiment ran and the answer is no); missing evidence is `blocked`. The
group that failed still selects the pre-registered response in
`failure_response`. Exit code 0 for every terminal outcome except `failed`.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss import spans as spans_mod
from ..lss.align import autoevidence, consensus_prod, coverage as coverage_mod
from ..lss.align import devselect as devselect_mod
from ..lss.align import jitter as jitter_mod
from ..lss.align import synthetic as synthetic_mod
from ..lss.align import target_objects as target_mod
from ..lss.artifacts import write_json, write_parquet
from ..lss.audit import pack as pack_mod
from ..lss.audit import verdicts as verdicts_mod
from ..lss.eligibility import unit_table
from ..lss.gates import check, evaluate, exit_code, reported
from ..lss.prereq import require_prerequisites
from ..lss.roles import load_role, role_path
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

SAMPLING_REPORT_SCHEMA = "lss_dialogue_stratified_sample_v1"
SAMPLING_CONFIG_FIELDS = (
    "sample_dialogues", "utterances_per_dialogue",
    "sample_stratify_field", "sample_seed",
)


def _positive_int(value, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"alignment.diagnostics.{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"alignment.diagnostics.{field} must be a positive integer") from exc
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"alignment.diagnostics.{field} must be a positive integer")
    return parsed


def _sampling_config(cfg: dict) -> dict:
    """Validate and normalize the dialogue-stratified sampling contract.

    The legacy total-row cap is rejected even if new fields are also present.
    Otherwise a partially migrated config could appear dialogue-stratified while
    a caller continued to rely on the old, conversation-ordered cap.
    """
    dcfg = (cfg.get("alignment") or {}).get("diagnostics") or {}
    if "sample_utterances" in dcfg:
        raise ValueError(
            "legacy alignment.diagnostics.sample_utterances is forbidden; "
            "configure sample_dialogues, utterances_per_dialogue, "
            "sample_stratify_field, and sample_seed explicitly")
    missing = [field for field in SAMPLING_CONFIG_FIELDS if field not in dcfg]
    if missing:
        raise ValueError(
            "incomplete dialogue-stratified sampling configuration; missing "
            + ", ".join(f"alignment.diagnostics.{field}" for field in missing))

    requested = dcfg["sample_dialogues"]
    if isinstance(requested, str):
        if requested.strip().lower() != "all":
            raise ValueError(
                "alignment.diagnostics.sample_dialogues must be a positive "
                "integer or 'all'")
        requested = "all"
    else:
        requested = _positive_int(requested, field="sample_dialogues")

    field = str(dcfg["sample_stratify_field"]).strip()
    if not field:
        raise ValueError(
            "alignment.diagnostics.sample_stratify_field must be non-empty")
    seed = dcfg["sample_seed"]
    if isinstance(seed, bool):
        raise ValueError("alignment.diagnostics.sample_seed must be an integer")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "alignment.diagnostics.sample_seed must be an integer") from exc
    if isinstance(dcfg["sample_seed"], float) \
            and not dcfg["sample_seed"].is_integer():
        raise ValueError("alignment.diagnostics.sample_seed must be an integer")

    return {
        "sample_dialogues": requested,
        "utterances_per_dialogue": _positive_int(
            dcfg["utterances_per_dialogue"], field="utterances_per_dialogue"),
        "sample_stratify_field": field,
        "sample_seed": seed,
    }


def _derived_rng(seed: int, *labels: str) -> np.random.Generator:
    """A stable independent RNG stream for one role/dialogue decision."""
    payload = "\x1f".join([str(seed), *(str(label) for label in labels)])
    derived = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(derived)


def dialogue_stratified_sample(manifest: pd.DataFrame, design: dict, *,
                               role: str) -> tuple[pd.DataFrame, dict]:
    """Sample up to K code-switched utterances from each selected dialogue.

    Filtering precedes dialogue enumeration. Dialogue and utterance inputs are
    sorted before seeded permutations, so results do not depend on set/dict or
    input-row iteration order. Under-populated dialogues contribute what they
    contain and are never compensated for by another dialogue.
    """
    field = str(design["sample_stratify_field"])
    required = {"utterance_id", "contains_code_switch", field}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"role {role} manifest cannot support dialogue-stratified sampling; "
            f"missing column(s): {missing}")

    eligible = manifest[manifest["contains_code_switch"].fillna(False).astype(bool)].copy()
    labels = eligible[field]
    absent = labels.isna() | labels.astype(str).str.strip().eq("")
    if bool(absent.any()):
        raise ValueError(
            f"role {role} has {int(absent.sum())} eligible utterance(s) without "
            f"{field}; refusing a non-stratified fallback")
    eligible[field] = labels.astype(str)
    eligible["utterance_id"] = eligible["utterance_id"].astype(str)

    available = sorted(eligible[field].unique().tolist())
    requested = design["sample_dialogues"]
    if requested == "all":
        selected_dialogues = available
        requested_for_report = "all"
        dialogue_shortfall = 0
    else:
        wanted = int(requested)
        order = list(available)
        _derived_rng(int(design["sample_seed"]), role, "dialogues").shuffle(order)
        selected_dialogues = sorted(order[:wanted])
        requested_for_report = wanted
        dialogue_shortfall = max(0, wanted - len(available))

    k = int(design["utterances_per_dialogue"])
    frames: list[pd.DataFrame] = []
    per_dialogue: list[dict] = []
    for dialogue in selected_dialogues:
        group = (eligible[eligible[field] == dialogue]
                 .sort_values("utterance_id", kind="stable"))
        take = min(k, len(group))
        if take:
            positions = _derived_rng(
                int(design["sample_seed"]), role, dialogue, "utterances"
            ).choice(len(group), size=take, replace=False)
            chosen = group.iloc[sorted(int(i) for i in positions)].copy()
            frames.append(chosen)
        per_dialogue.append({
            "dialogue_id": dialogue,
            "eligible_utterances": int(len(group)),
            "selected_utterances": int(take),
            "utterance_shortfall": int(max(0, k - len(group))),
        })

    sample = (pd.concat(frames, ignore_index=True) if frames
              else eligible.iloc[0:0].copy())
    if len(sample):
        sample = sample.sort_values([field, "utterance_id"], kind="stable").reset_index(drop=True)
    report = {
        "schema_version": SAMPLING_REPORT_SCHEMA,
        "role": role,
        "stratify_field": field,
        "seed": int(design["sample_seed"]),
        "requested_dialogues": requested_for_report,
        "available_dialogues": int(len(available)),
        "selected_dialogues": int(len(selected_dialogues)),
        "dialogue_shortfall": int(dialogue_shortfall),
        "utterances_per_dialogue": k,
        "eligible_utterances": int(len(eligible)),
        "selected_utterances": int(len(sample)),
        "underpopulated_dialogues": int(sum(
            row["utterance_shortfall"] > 0 for row in per_dialogue)),
        "per_dialogue": per_dialogue,
    }
    return sample, report


def _load_role_evidence(cfg: dict, role: str) -> pd.DataFrame:
    """Load a role and authenticate its immutable L0 artifact when on disk."""
    frame = load_role(cfg, role)
    path = Path(role_path(cfg, role))
    if path.is_file():
        verdict = manifest_mod.verify(path, cfg=None, require_identity=False,
                                      frame=frame)
        if not verdict["ok"]:
            raise RuntimeError(
                f"role manifest {role} is not authenticated: "
                f"{verdict.get('verdict')}: {verdict.get('detail', '')}")
    return frame


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


def _full_role_data_sufficiency(cfg: dict, roles: list[str]) -> dict:
    """Count role sufficiency on complete frozen L0 manifests, never the audit.

    Alignment sampling controls how much boundary reliability is measured.  It
    cannot define whether the frozen training roles contain enough material in
    the first place.  Keeping the two universes explicit prevents a 300-item
    audit from being compared with a 500-utterance role requirement.
    """
    gcfg = cfg.get("gate_a") or {}
    sampling_design = _sampling_config(cfg)
    universe = str(gcfg.get("data_sufficiency_universe",
                            "full_l0_role_manifest"))
    if universe != "full_l0_role_manifest":
        requested_dialogues = sampling_design["sample_dialogues"]
        sample_cap = (None if requested_dialogues == "all" else
                      int(requested_dialogues)
                      * int(sampling_design["utterances_per_dialogue"]))
        requested = int(gcfg.get("min_construct_bilingual_utterances", 0))
        if (universe == "audit_sample" and sample_cap is not None
                and requested > sample_cap):
            raise ValueError(
                "unreachable Gate-A configuration: "
                f"min_construct_bilingual_utterances={requested} exceeds the "
                f"audit_sample universe cap={sample_cap}; use "
                "gate_a.data_sufficiency_universe=full_l0_role_manifest")
        raise ValueError(f"unsupported data_sufficiency_universe={universe!r}")

    per_role: dict[str, dict] = {}
    audit_sampling: dict[str, dict] = {}
    for role in roles:
        manifest = _load_role_evidence(cfg, role)
        _, sample_report = dialogue_stratified_sample(
            manifest, sampling_design, role=role)
        audit_sampling[role] = sample_report
        units = unit_table(manifest)
        if len(units):
            units = units.copy()
            if "reference_unit_index" not in units and "unit_id" in units:
                units["reference_unit_index"] = units["unit_id"]
            units["role"] = role
        target_universe = target_mod.reference_target_universe(units)
        english = (target_universe[target_universe["language"] == "EN"]
                   if len(target_universe) else target_universe)
        cs = manifest["contains_code_switch"].fillna(False).astype(bool) \
            if "contains_code_switch" in manifest else pd.Series(False, index=manifest.index)
        per_role[role] = {
            "source_manifest": str(role_path(cfg, role)),
            "manifest_utterances": int(len(manifest)),
            "bilingual_utterances": int(cs.sum()),
            "target_language_runs": int(len(target_universe)),
            "embedded_english_targets": int(len(english)),
        }

    return {
        "data_sufficiency_universe": universe,
        "count_universe": "complete authenticated role manifests frozen by L0",
        "audit_sample_size_per_role": {
            role: int(report["selected_utterances"])
            for role, report in audit_sampling.items()},
        "audit_sampling": {
            "schema_version": SAMPLING_REPORT_SCHEMA,
            "design": sampling_design,
            "roles": audit_sampling,
        },
        "per_role": per_role,
        "d_construct_bilingual_count": int(
            per_role.get("D-construct", {}).get("bilingual_utterances", 0)),
        "d_construct_required": int(
            gcfg.get("min_construct_bilingual_utterances", 500)),
    }


def _validate_gate_configuration(cfg: dict) -> None:
    """Reject mathematically unreachable evidence requirements before work."""
    gcfg = cfg.get("gate_a") or {}
    scfg = cfg.get("synthetic") or {}
    sampling_design = _sampling_config(cfg)
    required = int(gcfg.get("min_synthetic_boundaries", 100))
    for purpose in ("dev", "gate"):
        candidates = int(scfg.get(f"num_pairs_{purpose}", 0))
        if candidates <= required:
            raise ValueError(
                f"unreachable/fragile synthetic configuration: num_pairs_{purpose}="
                f"{candidates} must exceed min_synthetic_boundaries={required}; "
                "the candidate pool needs attrition margin")
    # Also validates the declared universe and detects the old 300-vs-500 shape
    # without loading role data.
    universe = str(gcfg.get("data_sufficiency_universe", "full_l0_role_manifest"))
    if universe == "audit_sample":
        requested_dialogues = sampling_design["sample_dialogues"]
        cap = (None if requested_dialogues == "all" else
               int(requested_dialogues)
               * int(sampling_design["utterances_per_dialogue"]))
        required_construct = int(gcfg.get("min_construct_bilingual_utterances", 0))
        if cap is not None and required_construct > cap:
            raise ValueError(
                "unreachable Gate-A configuration: audit sample cap "
                f"{cap} < D-construct requirement {required_construct}")
    elif universe != "full_l0_role_manifest":
        raise ValueError(f"unsupported data_sufficiency_universe={universe!r}")


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

    **Both** the aligner sweep and the coverage denominator come from this one
    function. Code-switched rows are stratified by the configured cluster field
    before at most K utterances are drawn per dialogue. This replaces the former
    conversation-ordered `.head(300)`, which concentrated a role sample in a
    handful of clusters.

    The selection is deterministic, seed-recorded, and independent of aligner
    output. Sparse dialogues are never backfilled, so their shortfall remains
    visible instead of silently increasing another cluster's weight.
    """
    design = _sampling_config(cfg)
    frames = []
    reports: dict[str, dict] = {}
    for role in sweep_roles(cfg, roles):
        try:
            manifest = _load_role_evidence(cfg, role)
        except Exception:                       # a role that was never built
            if missing_ok:
                continue
            raise
        subset, sample_report = dialogue_stratified_sample(
            manifest, design, role=role)
        subset["role"] = role
        frames.append(subset)
        reports[role] = sample_report
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.attrs["sampling_report"] = {
        "schema_version": SAMPLING_REPORT_SCHEMA,
        "design": design,
        "roles": reports,
    }
    return out


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
    sampling_report = sample.attrs.get("sampling_report", {})
    log.info("aligner sweep over %d bilingual utterances across %s; "
             "dialogue-stratified sampling=%s", len(sample), roles,
             sampling_report.get("design", {}))
    for role, role_report in (sampling_report.get("roles") or {}).items():
        if role_report.get("dialogue_shortfall"):
            log.warning(
                "role %s has %d fewer eligible dialogues than requested; "
                "using all %d and not backfilling", role,
                role_report["dialogue_shortfall"],
                role_report["selected_dialogues"])

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
    report["sampling"] = sampling_report
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
            extra={"request_manifest": report.get("request_manifest"),
                   "sampling": sampling_report})
    else:
        log.error("aligner sweep produced no candidates: %s", report)
    report_path = art(cfg, "diagnostics", "l1b_aligner_sweep.json")
    write_json(report, report_path)
    manifest_mod.publish(
        report_path, None, stage=STAGE, cfg=cfg, run_dir=run_dir,
        parents=([m for m in (parents or []) if m]
                 + [m for m in (report.get("cache_manifests") or {}).values() if m]
                 + ([report["manifest"]] if report.get("manifest") else [])),
        taint_reasons=(taint or {}).get("taint_reasons") or (),
        schema="lss_aligner_sweep_report_v2")
    return report



#: Families scored against typed synthetic references.
#:
#: Every configured family, Qwen included. It used to be CTC and DTW only,
#: because `candidates.run_families` had no way to run Qwen over an arbitrary
#: manifest: it consumed whatever the L1a probe had persisted, keyed to eight
#: *corpus* utterance ids, which can never match a synthetic pair id. With
#: `csasr.lss.align.qwen_prod` there is a production path, so a family that is
#: allowed to qualify on natural speech is also scored on compatible references --
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

    `build_set` currently renders RMS/VAD splices with exactly known audio seams,
    not lexical boundaries. Those rows provide seam-relative diagnostics only.
    Genuine lexical reference kinds remain supported for absolute-error Gate-A
    evidence, and cross-aligner agreement is never a substitute.

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

    score_extra = {"synthetic_gate_generation": gate_generation,
                   "synthetic_set_request_sha256": request_fingerprints,
                   "synthetic_gate_request_sha256": request_fingerprints.get("gate"),
                   "synthetic_set_item_sha256": item_fingerprints,
                   "synthetic_gate_item_sha256": item_fingerprints.get("gate"),
                   "scientific_amendment": devselect.scientific_amendment(cfg)}
    score_manifest = manifest_mod.publish_frame(
        root / autoevidence.SYNTHETIC_SCORES_FILE, all_scores,
        stage=STAGE, cfg=cfg, run_dir=run_dir,
        parents=score_parents,
        taint_reasons=(taint or {}).get("taint_reasons"),
        key_columns=("family", "convention", "edge", "purpose", "reference_kind"),
        schema=synthetic_mod.SCORES_SCHEMA_VERSION,
        # which gate generation these numbers describe. A later evaluation reads
        # the ledger's unexposed generation and refuses a table from an earlier
        # one, so a second run cannot quietly re-report the first one's scores.
        extra=score_extra)
    if len(all_items):
        item_manifest = manifest_mod.publish_frame(
            root / autoevidence.SYNTHETIC_ITEMS_FILE, all_items,
            stage=STAGE, cfg=cfg, run_dir=run_dir, parents=score_parents,
            taint_reasons=(taint or {}).get("taint_reasons"),
            key_columns=("pair_id", "family", "variant", "purpose"),
            schema=autoevidence.SYNTHETIC_ITEMS_SCHEMA, extra=score_extra)
        comparison = synthetic_mod.convention_comparison(all_items)
        if len(comparison):
            manifest_mod.publish_frame(
                root / "metrics/l1b_synthetic_convention_comparison.parquet",
                comparison, stage=STAGE, cfg=cfg, run_dir=run_dir,
                parents=[score_manifest, item_manifest],
                taint_reasons=(taint or {}).get("taint_reasons"),
                key_columns=("family", "purpose", "reference_kind",
                             "convention", "edge"),
                schema="lss_reference_convention_comparison_v2")
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
    3. both are scored according to their declared reference kind;
    4. the operating tolerance is selected from the **development** rows only, by
       the preregistered rule, and frozen with the instrument recorded.

    Step 4 never sees a gate row: `devselect.assert_development_only` raises if it
    ever does.
    """
    from ..lss.align import devselect, exposure

    root = Path(cfg["experiment"]["output_root"])
    scfg = dict(cfg.get("synthetic") or {})
    gcfg = cfg.get("gate_a") or {}
    _validate_gate_configuration(cfg)
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

    required_boundaries = int(gcfg.get("min_synthetic_boundaries", 100))
    if len(dev) <= required_boundaries:
        # Cross-stage identity is intentionally not required because L1a and
        # L1b resolve different stage configs. Therefore an authenticated older
        # 100-item L1a artifact can otherwise survive a new 150-item request.
        # Check the actual bytes' population, not only the live YAML, before any
        # gate generation is allocated.
        log.error("authenticated development pool has %d candidates for a %d "
                  "usable-boundary requirement; rebuild L1a with attrition margin",
                  len(dev), required_boundaries)
        out["blocked"].append(autoevidence.BLOCKED_MISSING_DEVELOPMENT_SET)
        out["criteria"] += [
            reported("synthetic_development_candidate_count", int(len(dev))),
            reported("synthetic_required_usable_boundaries", required_boundaries),
            reported("held_out_gate_generation_allocated", 0),
        ]
        return out

    reference_kinds = sorted(set(dev.get(
        "reference_kind", pd.Series([synthetic_mod.AUDIO_SPLICE] * len(dev)))
        .astype(str)))
    out["development_reference_kinds"] = reference_kinds
    if not set(reference_kinds) <= synthetic_mod.LEXICAL_REFERENCE_KINDS:
        # Stop before allocating or rendering a held-out generation.  A gate set
        # cannot repair a development instrument that measures the wrong object,
        # and exposing another generation would spend confirmatory data for no
        # possible production decision.
        log.error("development reference kinds %s do not provide genuine lexical "
                  "boundaries; refusing to allocate a held-out gate generation",
                  reference_kinds)
        out["blocked"].append(autoevidence.BLOCKED_MISSING_LEXICAL_CALIBRATION)
        out["criteria"] += [
            reported("development_reference_kinds", ",".join(reference_kinds)),
            reported("held_out_gate_generation_allocated", 0),
        ]
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
    construct = _load_role_evidence(
        cfg, str(scfg.get("source_role", "D-construct")))
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
    if len(gate_set) <= required_boundaries:
        log.error("fresh gate pool produced %d candidates for a %d usable-boundary "
                  "requirement; refusing to score or expose the undersized set",
                  len(gate_set), required_boundaries)
        out["blocked"].append(autoevidence.BLOCKED_MISSING_SYNTHETIC)
        out["criteria"] += [
            reported("synthetic_gate_candidate_count", int(len(gate_set))),
            reported("synthetic_required_usable_boundaries", required_boundaries),
        ]
        return out
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
    gate_items_path = root / "metrics/l1b_synthetic_gate_items.parquet"
    gate_items_manifest = manifest_mod.publish_frame(
        gate_items_path, gate_set, stage=STAGE, cfg=cfg, run_dir=rdir,
        parents=[m for m in parents if m],
        taint_reasons=taint.get("taint_reasons") or (),
        key_columns=("pair_id",), schema="lss_synthetic_reference_items_v2",
        extra={"synthetic_gate_generation": generation,
               "synthetic_gate_request_sha256": gate_request["sha256"],
               "synthetic_gate_item_sha256": gate_item_sha256})
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
    synthetic_parents = ([m for m in parents if m]
                         + ([dev_meta["manifest"]] if dev_meta.get("manifest") else [])
                         + [gate_items_manifest])
    scoring = _score_synthetic_sets(
        cfg, log, {"dev": dev, "gate": gate_set}, run_dir=rdir, taint=taint,
        parents=synthetic_parents,
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
        selection_parents = synthetic_parents + [
            m for m in [manifest_mod.load(root / autoevidence.SYNTHETIC_SCORES_FILE),
                        manifest_mod.load(root / autoevidence.SYNTHETIC_ITEMS_FILE)] if m]
        selection_manifest = manifest_mod.publish_frame(
            root / devselect.SELECTION_TABLE, evidence, stage=STAGE, cfg=cfg,
            run_dir=rdir, parents=selection_parents,
            taint_reasons=taint.get("taint_reasons") or (),
            key_columns=("tolerance_ms",), schema="lss_tolerance_selection_v2")
        log.info("operating-tolerance selection on synthetic development "
                 "boundaries:\n%s", evidence.to_string(index=False))
    else:
        selection_manifest = None
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
                         parents=(synthetic_parents
                                  + ([selection_manifest] if selection_manifest else [])),
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
    _validate_gate_configuration(cfg)
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

        configured_roles = list(cfg.get("roles_to_label") or ["D-construct"])
        role_manifests = [manifest_mod.load(role_path(cfg, role))
                          for role in configured_roles]
        sweep_parents = prerequisite_manifests + [m for m in role_manifests if m]

        if args.align:
            metrics["aligner_sweep"] = _run_aligner_sweep(
                cfg, log, overwrite=args.overwrite, run_dir=rdir,
                roles=configured_roles,
                qwen_runner=_qwen_runner(cfg, log, rdir, taint,
                                         sweep_parents),
                parents=sweep_parents, taint=taint)

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
        evidence_input_manifests = [m for m in input_manifests + role_manifests if m]
        artifact_taint = manifest_mod.taint_of_inputs(evidence_input_manifests)
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

        # ---- typed reference evidence and the operating point ---------------
        # Rendering is not scoring, and scoring is not selecting. All three
        # happen here, in that order, and the selection sees development rows
        # only.
        gate_generation = None
        if "synthetic" in parts and not args.dry_run:
            synthetic_evidence = _prepare_synthetic_evidence(
                cfg, log, rdir, taint, evidence_input_manifests, config=config,
                qwen_runner=_qwen_runner(cfg, log, rdir, taint,
                                         evidence_input_manifests),
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
            manifest_mod.publish_frame(
                root / "metrics/l1b_tolerance_sweep.parquet", sweep,
                stage=STAGE, cfg=cfg, run_dir=rdir,
                parents=evidence_input_manifests,
                taint_reasons=taint["taint_reasons"],
                key_columns=("tolerance_ms",), schema="lss_tolerance_sweep_v1")
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
            expected_targets = target_mod.reference_target_universe(expected_units)
            raw_validity = autoevidence.family_validity(candidates)
            raw_detail = target_mod.raw_unit_diagnostics(candidates)
            targets, language_runs, aggregation = target_mod.target_objects(candidates)
            validity = target_mod.target_family_validity(targets, expected_targets)
            evidence_parents = evidence_input_manifests
            if len(raw_validity):
                manifest_mod.publish_frame(
                    root / "metrics/l1b_raw_family_validity.parquet", raw_validity,
                    stage=STAGE, cfg=cfg, run_dir=rdir, parents=evidence_parents,
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("aligner_family",), schema="lss_raw_family_validity_v2")
            if len(raw_detail):
                manifest_mod.publish_frame(
                    root / "metrics/l1b_raw_unit_diagnostics.parquet", raw_detail,
                    stage=STAGE, cfg=cfg, run_dir=rdir, parents=evidence_parents,
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("aligner_family", "language"),
                    schema="lss_raw_unit_diagnostics_v1")
            if len(language_runs):
                manifest_mod.publish_frame(
                    root / "alignments/l1b_language_runs.parquet", language_runs,
                    stage=STAGE, cfg=cfg, run_dir=rdir, parents=evidence_parents,
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("aligner_family", "aligner_variant", "utterance_id", "run_id"),
                    schema=target_mod.LANGUAGE_RUN_SCHEMA)
            if len(targets):
                manifest_mod.publish_frame(
                    root / "alignments/l1b_target_objects.parquet", targets,
                    stage=STAGE, cfg=cfg, run_dir=rdir, parents=evidence_parents,
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("aligner_family", "aligner_variant", "target_id"),
                    schema=target_mod.TARGET_OBJECT_SCHEMA)
            if len(validity):
                manifest_mod.publish_frame(
                    root / "metrics/l1b_family_validity.parquet", validity,
                    stage=STAGE, cfg=cfg, run_dir=rdir, parents=evidence_parents,
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("aligner_family",), schema="lss_target_family_validity_v1")
            min_family_coverage = float(gcfg.get("min_family_valid_unit_coverage",
                                                 0.95))
            min_family_units = int(math.ceil(min_family_coverage
                                             * len(expected_targets)))
            independence = autoevidence.independent_valid_families(
                validity,
                min_coverage=min_family_coverage,
                max_invalid_rate=float(gcfg.get("max_invalid_rate", 0.01)),
                max_nonmonotonic_rate=float(gcfg.get("max_nonmonotonic_rate", 0.01)),
                min_units=min_family_units)
            raw_coverage = autoevidence.unit_coverage(candidates, expected_units)
            pair_priority = list(((cfg.get("alignment") or {}).get("consensus") or {})
                                 .get("aligner_pair_priority") or [])
            pair_selection = target_mod.select_pair(
                targets, independence["qualifying_families"], pair_priority,
                min_count=int(gcfg.get("min_paired_units", 100)),
                min_rate=float(gcfg.get("min_paired_target_rate", 0.90)))
            selected_pair = pair_selection.get("selected_pair", [])
            agreement = target_mod.cross_aligner_target_agreement(targets, selected_pair)
            all_pair_overlaps = target_mod.diagnostic_pair_overlaps(targets)
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
                selected_pair=selected_pair,
                cfg=cfg, require_authentication=evaluating,
                expected_gate_generation=unexposed_generation if evaluating else None,
                expected_gate_request_sha256=(
                    unexposed_entry.get("alignment_request_sha256")
                    if evaluating else None),
                expected_gate_item_sha256=(
                    unexposed_entry.get("item_set_sha256")
                    if evaluating else None))
            metrics["automatic_evidence"] = {
                "roles_to_label": roles_to_label,
                "expected_units": int(len(expected_units)),
                "expected_target_objects": int(len(expected_targets)),
                "target_alignment_objects": (
                    "language-run outer boundaries, embedded-English spans, and "
                    "EN-ZH switch edges"),
                "family_validity": validity.to_dict(orient="records") if len(validity) else [],
                "raw_family_validity": (raw_validity.to_dict(orient="records")
                                        if len(raw_validity) else []),
                "raw_unit_diagnostics": (raw_detail.to_dict(orient="records")
                                         if len(raw_detail) else []),
                "aggregation": aggregation,
                "independence": independence,
                "raw_unit_coverage": raw_coverage,
                # Compatibility alias for downstream report readers; its
                # denominator and meaning remain explicitly raw reference units.
                "unit_coverage": raw_coverage,
                "pair_selection": pair_selection,
                "diagnostic_pair_overlaps": all_pair_overlaps,
                "cross_aligner_agreement": agreement,
                "synthetic_calibration": calibration,
            }
            automatic_path = art(cfg, "diagnostics", "l1b_automatic_evidence.json")
            write_json(metrics["automatic_evidence"], automatic_path)
            manifest_mod.publish(
                automatic_path, None, stage=STAGE, cfg=cfg, run_dir=rdir,
                parents=evidence_parents, taint_reasons=taint["taint_reasons"],
                schema="lss_automatic_gate_a_evidence_v2")
            log.info("independent valid aligner families: %d (%s); rejected: %s",
                     independence["n_independent_valid_families"],
                     ",".join(independence["qualifying_families"]) or "none",
                     independence["rejection_reasons"] or "none")

            n_independent = independence["n_independent_valid_families"]
            min_families = int(gcfg.get("min_valid_families", 2))
            indexed_validity = (validity.set_index("aligner_family")
                                if len(validity) else pd.DataFrame())
            pair_coverages = [float(indexed_validity.loc[f, "valid_unit_coverage"])
                              for f in selected_pair
                              if f in getattr(indexed_validity, "index", [])]
            target_coverage = (min(pair_coverages) if len(pair_coverages) == 2
                               else (float(validity["valid_unit_coverage"].max())
                                     if len(validity) else float("nan")))
            criteria += [
                check("alignment_target_object_coverage", target_coverage,
                      float(gcfg.get("min_unit_coverage", 0.95)), ">=",
                      group="coverage"),
                reported("target_coverage_denominator", "frozen_expected_target_objects"),
                reported("expected_target_objects", int(len(expected_targets))),
                reported("raw_expected_reference_units", raw_coverage["reference_units"]),
                reported("raw_unaligned_expected_units", raw_coverage["missing_units"]),
                reported("independent_valid_aligner_families", n_independent),
                reported("min_valid_target_objects_per_family", min_family_units),
                reported("candidate_source", candidate_source["source"]),
                reported("candidate_authenticated",
                         int(bool(candidate_source["authenticated"]))),
                reported("qualifying_aligner_families",
                         ",".join(independence["qualifying_families"]) or "none"),
                reported("rejected_aligner_families",
                         ",".join(independence.get("rejected_families", [])) or "none"),
                reported("selected_aligner_pair", ",".join(selected_pair) or "none"),
                reported("pair_selection_rule", pair_selection["selection_rule"]),
            ]
            if raw_coverage["denominator"] != "expected_universe":
                # a coverage number whose denominator came from the candidate
                # rows cannot detect dropped utterances, so it is not evidence
                log.error("coverage has no frozen expected-unit universe for %s",
                          roles_to_label)
                blocked_reasons.append(autoevidence.BLOCKED_NO_CANDIDATES
                                       if not len(candidates)
                                       else autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES)

            # Qualification is on target objects.  Raw-unit failures remain in
            # the authenticated diagnostic tables but do not automatically
            # invalidate a usable enclosing same-language run.
            gating_families = set(selected_pair)
            for row in (validity.to_dict(orient="records") if len(validity) else []):
                family = row["aligner_family"]
                # Once a deterministic pair exists, only that pair supplies the
                # production evidence. A rejected third family remains fully
                # visible but cannot turn a passing pair into a no-go. If no
                # pair exists, retain thresholded rows for every family so the
                # blocker still explains which repairs are needed.
                gates = not gating_families or family in gating_families
                criteria += ([
                    check(f"target_invalid_rate_{family}", row["invalid_rate"],
                          float(gcfg.get("max_invalid_rate", 0.01)), "<=",
                          group="external"),
                    check(f"target_nonmonotonic_rate_{family}", row["nonmonotonic_rate"],
                          float(gcfg.get("max_nonmonotonic_rate", 0.01)), "<=",
                          group="external"),
                ] if gates else [
                    reported(f"target_invalid_rate_{family}", row["invalid_rate"]),
                    reported(f"target_nonmonotonic_rate_{family}",
                             row["nonmonotonic_rate"]),
                    reported(f"family_role_{family}",
                             "diagnostic_rejected_not_selected_pair"),
                ]) + [
                    reported(f"valid_target_coverage_{family}",
                             row["valid_unit_coverage"]),
                    reported(f"raw_invalid_units_retained_{family}",
                             row["raw_invalid_units_retained"]),
                ]

            if n_independent < min_families:
                # A second opinion that does not exist is missing infrastructure,
                # not a broken stage and not a measurement: `blocked`. This is
                # the observed state today -- existing_ctc valid, whisper_dtw
                # below the validity floor -- and it must never read as a pass.
                blocked_reasons.append(autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS)
            elif not pair_selection["available"]:
                log.error("two or more families qualify but no configured pair "
                          "has sufficient paired target evidence: %s",
                          pair_selection["pair_diagnostics"])
                blocked_reasons.append(autoevidence.BLOCKED_NO_PAIRED_AGREEMENT)
                criteria.append(reported("paired_cross_aligner_target_objects", 0))

            if pair_selection["available"] and agreement.get("n"):
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
                    check("paired_cross_aligner_target_objects", agreement["n"],
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

            # Absolute calibration is allowed only for the same selected pair.
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
                    reported("synthetic_selected_pair",
                             ",".join(calibration.get("selected_pair", []))),
                    reported("synthetic_edges_scored",
                             ",".join(calibration["edges_scored"])),
                ]
            elif mode == "automatic" and pair_selection["available"]:
                log.error("%s: %s", calibration["reason"],
                          calibration.get("detail") or
                          calibration["missing_implementation"])
                blocked_reasons.append(calibration["reason"])
                criteria.append(reported("synthetic_calibration",
                                         calibration["reason"]))

            if pair_selection["available"] and calibration.get("available"):
                calibrated = set(calibration.get("families_scored", []))
                missing = [f for f in selected_pair if f not in calibrated]
                if missing:
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
            manifest = _load_role_evidence(cfg, "D-construct")
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
                jitter_parents = evidence_input_manifests + [
                    m for m in [manifest_mod.load(root / SPANS_FILE)] if m]
                manifest_mod.publish_frame(
                    root / "metrics/l1b_jitter.parquet", table,
                    stage=STAGE, cfg=cfg, run_dir=rdir, parents=jitter_parents,
                    taint_reasons=taint["taint_reasons"],
                    key_columns=("offset_ms", "seed", "duration_bin"),
                    schema="lss_jitter_duration_stratified_v2")
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
            coverage_parents = evidence_input_manifests + [
                m for m in [manifest_mod.load(root / SPANS_FILE),
                            manifest_mod.load(root / REJECTED_FILE)] if m]
            manifest_mod.publish_frame(
                root / "metrics/l1b_label_coverage.parquet", label,
                stage=STAGE, cfg=cfg, run_dir=rdir, parents=coverage_parents,
                taint_reasons=taint["taint_reasons"], key_columns=("language",),
                schema="lss_label_coverage_v1")
            manifest_mod.publish_frame(
                root / "metrics/l1b_selection_bias.parquet", bias_table,
                stage=STAGE, cfg=cfg, run_dir=rdir, parents=coverage_parents,
                taint_reasons=taint["taint_reasons"], key_columns=("language",),
                schema="lss_selection_bias_v1")

            # Every configured absolute count, evaluated against the role it is
            # about. Counting D-construct spans and calling them loc-train
            # claimed a number about a role that was never aligned.
            per_role = _role_span_counts(primary, roles_to_label)
            data_sufficiency = _full_role_data_sufficiency(cfg, roles_to_label)
            log.info("primary spans per role: %s",
                     {r: v["spans"] for r, v in per_role.items()})
            log.info("full-role data sufficiency: %s",
                     {r: {"bilingual": v["bilingual_utterances"],
                          "en_targets": v["embedded_english_targets"]}
                      for r, v in data_sufficiency["per_role"].items()})
            usable = autoevidence.usable_item_rate(spans)
            eligibility = autoevidence.language_eligibility(primary)
            coverage_payload = {
                "partition": partition, "per_role_spans": per_role,
                "data_sufficiency": data_sufficiency, "usable_items": usable,
                "language_eligibility": eligibility,
                "selection_bias": coverage_mod.selection_bias_summary(bias_table)}
            coverage_path = art(cfg, "diagnostics", "l1b_coverage.json")
            write_json(coverage_payload, coverage_path)
            manifest_mod.publish(
                coverage_path, None, stage=STAGE, cfg=cfg, run_dir=rdir,
                parents=coverage_parents, taint_reasons=taint["taint_reasons"],
                schema="lss_gate_a_coverage_v2")
            metrics["coverage"] = {"partition": partition,
                                   "per_role_spans": per_role,
                                   "data_sufficiency": data_sufficiency,
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
                check("accepted_rejected_partition_structurally_valid",
                      int(bool(partition["partition_structurally_valid"])), 1, "=="),
                check("accepted_rejected_accounted_rate",
                      partition["accounted_rate"],
                      float(gcfg.get("min_unit_coverage", 0.95)), ">=",
                      group="coverage"),
                reported("accepted_rejected_partition_exact",
                         int(bool(partition["partition_exact"]))),
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
                    f"data_sufficiency_{threshold_key}",
                    data_sufficiency["per_role"].get(role, {})
                    .get("embedded_english_targets", 0),
                    int(gcfg.get(threshold_key, 0)), ">=", group="coverage"))
            construct_utterances = data_sufficiency["d_construct_bilingual_count"]
            criteria.append(check(
                "data_sufficiency_construct_bilingual_utterances", construct_utterances,
                int(gcfg.get("min_construct_bilingual_utterances", 500)), ">=",
                group="coverage"))
            criteria += [
                reported("data_sufficiency_universe",
                         data_sufficiency["data_sufficiency_universe"]),
                reported("alignment_audit_sample_size_per_role",
                         data_sufficiency["audit_sample_size_per_role"]),
            ]
            if not eligibility["eligible"]:
                log.error("primary spans are missing language subset(s): %s",
                          eligibility["missing_languages"])
                blocked_reasons.append(autoevidence.BLOCKED_MISSING_LANGUAGE)

        # A non-empty production span table must be well formed. Absence is
        # missing evidence (covered by the specific candidate/consensus blockers),
        # not a corrupt schema and therefore not an implementation `failed`.
        if evaluating:
            criteria.append(
                check("span_schema_valid", int(_span_schema_ok(spans)), 1, "==")
                if len(spans) else reported("span_schema_valid", "unavailable_no_spans"))

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
                  "absolute lexical error comes only from manual/gold lexical "
                  "edges or constructions that genuinely provide exact lexical "
                  "edges; an audio splice is insufficient. The failing group selects "
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
    (autoevidence.BLOCKED_MISSING_LEXICAL_CALIBRATION, "accuracy",
     "provide a genuine lexical reference (manual_lexical, "
     "existing_gold_lexical, or constructed_exact_lexical). The current "
     "RMS/VAD concatenation has a known audio seam, not a known lexical edge, "
     "and cannot select an operating tolerance or pass absolute-error Gate A"),
    (autoevidence.BLOCKED_EXPOSED_GATE_SET, "accuracy",
     "no unexposed source audio is left for a fresh confirmatory gate; widen the "
     "synthetic source role or accept that this corpus can no longer confirm a "
     "new configuration (see synthetic/exposure_ledger.json)"),
    (autoevidence.BLOCKED_MISSING_SYNTHETIC, "accuracy",
     "provide compatible lexical reference items, then run the `synthetic` part "
     "so exact-boundary scores are produced and authenticated (see "
     "autoevidence.MISSING_SYNTHETIC_DESCRIPTION)"),
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
     "increase eligible target-object overlap between the two naturally "
     "qualifying families; inspect the recorded raw and target-filtered pair "
     "counts rather than inferring that their input sets were disjoint"),
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
    ("target_invalid_rate_", "alignment",
     "repair the family's unusable target boundaries/switches; raw internal "
     "unit failures remain in l1b_raw_unit_diagnostics.parquet"),
    ("target_nonmonotonic_rate_", "alignment",
     "repair ordering among otherwise valid target objects"),
    ("alignment_target_object_coverage", "alignment",
     "align the language-run/span targets missing from the frozen expected "
     "universe; raw-unit coverage is reported separately"),
    ("synthetic_absolute_", "alignment",
     "repair boundary accuracy: absolute error against genuine lexical reference "
     "boundaries is outside the proposal's thresholds. Fitting a correction is "
     "allowed only on development items and must be validated on a fresh gate set"),
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
     "the complete frozen L0 D-construct manifest has too few bilingual "
     "utterances; this count is independent of the sampled alignment audit"),
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
