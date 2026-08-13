"""Configuration selection from development-only evidence.

The automatic Gate-A path requires no new human annotation, and this module is
what owns the development-only configuration choices.  It deliberately does
not turn an audio splice into lexical calibration evidence.

The conflict, and the disclosed amendment
------------------------------------------
`configs/lss/spec.yaml alignment_prereg.tolerance_selection_rule` reads

    the largest swept tolerance whose accepted spans still satisfy **human**
    median <= 100 ms and p90 <= 200 ms

so a literal reading makes the automatic path depend on annotation.  A prior
amendment proposed substituting RMS/VAD-trimmed synthetic development splices,
but those provide known audio seams rather than known lexical edges.  The
amendment is retained as an authenticated scientific record and marked
superseded.  Until a genuine lexical reference kind is supplied, automatic
tolerance selection is truthfully blocked.

The decoder-query convention may still be selected on development-only
seam-relative diagnostics: that is an implementation-coordinate choice, not a
claim of lexical accuracy.  Its artifact uses explicit splice-offset field
names.  The 100/200 ms lexical thresholds are applied only to genuine lexical
reference kinds, never to the seam diagnostic.

Two properties are enforced rather than assumed:

* **the gate set cannot select the configuration it judges.** Every function
  here takes `purpose="dev"` rows and `assert_development_only` refuses anything
  else;
* **a selection made on a handful of items is not a selection.** A tolerance
  whose accepted spans yield fewer than `min_boundaries` scorable seams cannot
  qualify, because the alternative is choosing an operating point from three
  utterances that happened to agree.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import synthetic as synthetic_mod
from .consensus_prod import ConsensusConfig, build, select_primary_tolerance

#: development instruments.  The reference kind determines which claims are
#: permitted; the name alone never licenses an absolute-error claim.
INSTRUMENT_SYNTHETIC_DEV = "synthetic_dev"
INSTRUMENT_HUMAN_AUDIT = "human_audit"

#: family name the consensus estimator's own spans are scored under. It is not an
#: aligner family and never enters `INDEPENDENCE_CLASS`: it is the *output* of
#: combining families, which is exactly what the operating tolerance governs.
CONSENSUS_FAMILY = "consensus"

#: the purpose whose items may inform a decision
DEVELOPMENT_PURPOSE = "dev"

#: artifacts this module produces
SELECTION_TABLE = "metrics/l1b_tolerance_selection.parquet"
OPERATING_POINT_FILE = "freeze/l1b_operating_point.json"
OPERATING_POINT_SCHEMA = "lss_operating_point_v1"


def scientific_amendment(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticated disclosure for the proposed instrument substitution.

    The sealed v1 text literally names a human instrument. Any future automatic
    selection on genuine exact lexical development references is therefore a
    prospective amendment, not a silent interpretation. The record also makes
    the current superseded state immutable and reviewable.
    """
    from ...utils.config import REPO_ROOT
    from ...utils.hashing import sha256_file

    block = dict(((cfg.get("gate_a") or {})
                  .get("automatic_instrument_amendment") or {}))
    relative = str(block.get("document") or "")
    path = Path(relative)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not relative or not path.is_file():
        raise FileNotFoundError(
            "automatic synthetic-dev tolerance selection requires an explicit "
            "scientific amendment document; configured path is " + str(path))
    return {
        "id": str(block.get("id") or ""),
        "status": str(block.get("status") or ""),
        "document": relative,
        "sha256": sha256_file(path),
        "effective_from_gate_generation": int(
            block.get("effective_from_gate_generation", 1)),
        "disclosure": ("post-freeze prospective amendment; not an unchanged "
                       "execution of the sealed human-instrument rule"),
    }


class DevelopmentOnlyError(AssertionError):
    """Gate-set evidence reached a code path that selects a configuration."""


def assert_development_only(frame: pd.DataFrame, *, what: str) -> None:
    """Refuse any row whose purpose is not the development set.

    This is the leakage guard. A configuration chosen with gate rows in the
    denominator is a configuration judged on the data that made it, and the
    resulting accuracy number means nothing.
    """
    if frame is None or not len(frame) or "purpose" not in frame.columns:
        return
    purposes = sorted(set(frame["purpose"].astype(str)))
    if purposes != [DEVELOPMENT_PURPOSE]:
        raise DevelopmentOnlyError(
            f"{what} contains purposes {purposes}; only "
            f"{DEVELOPMENT_PURPOSE!r} items may inform a selection, or the gate "
            "set would judge the configuration it chose")


def consensus_as_candidates(spans: pd.DataFrame, *,
                            estimator: str = "both_edges") -> pd.DataFrame:
    """Reshape consensus spans so the synthetic scorer can read them.

    `synthetic.boundary_predictions` reads a candidate-shaped table: it takes the
    end of the last valid Mandarin unit and the start of the first valid English
    unit. Consensus spans are the same quantity after combining families, so the
    seam of a *consensus* span is what a steering mask is actually built from --
    and therefore what the operating tolerance must be judged on.
    """
    if spans is None or not len(spans):
        return pd.DataFrame()
    language = spans["reference_language"] if "reference_language" in spans \
        else spans["language"]
    return pd.DataFrame({
        "utterance_id": spans["utterance_id"].astype(str),
        "reference_unit_index": spans["reference_unit_index"].astype(int),
        "reference_language": language.astype(str),
        "start_sec": spans["consensus_start_second"].astype(float),
        "end_sec": spans["consensus_end_second"].astype(float),
        "aligner_family": CONSENSUS_FAMILY,
        "aligner_variant": f"{CONSENSUS_FAMILY}/{estimator}",
        "is_valid": True,
    }).reset_index(drop=True)


def _combined_error(rendered: pd.DataFrame, spans: pd.DataFrame, *,
                    estimator: str) -> dict[str, Any]:
    """Absolute lexical-edge error of consensus spans against known references.

    The `combined` edge is used -- the worse of the two edges of each item --
    because a steering mask is wrong at whichever end is wrong, and because Gate
    A reads the worst row of the score table for the same reason.
    """
    empty = {"n": 0, "median_abs_error_ms": float("nan"),
             "p90_abs_error_ms": float("nan"),
             "median_signed_error_ms": float("nan"),
             "within_100ms": float("nan")}
    if not len(spans):
        return empty
    scores, per_item = synthetic_mod.score_rendered_set(
        rendered, consensus_as_candidates(spans, estimator=estimator),
        purpose=DEVELOPMENT_PURPOSE)
    if not len(scores):
        return empty
    combined = scores[(scores["edge"] == "combined")
                      & (scores["convention"] == synthetic_mod.CANONICAL_CONVENTION)]
    if not len(combined):
        return empty
    row = combined.iloc[0]
    scorable = int(per_item["scorable"].sum()) if len(per_item) else 0
    return {
        "n": int(row["num_boundaries"]),
        "scorable_items": scorable,
        "median_abs_error_ms": float(row["median_abs_error_ms"]),
        "p90_abs_error_ms": float(row["p90_abs_error_ms"]),
        "median_signed_error_ms": float(row["median_signed_error_ms"]),
        "within_100ms": float(row["within_100ms"]),
    }


def tolerance_accuracy(rendered: pd.DataFrame, candidates: pd.DataFrame,
                       config: ConsensusConfig,
                       tolerances_ms: Sequence[float], *,
                       min_boundaries: int = 50) -> pd.DataFrame:
    """Per-tolerance development accuracy of the spans that tolerance accepts.

    One row per swept tolerance: how many spans it accepts on the synthetic
    development set, and the absolute boundary error of those spans against the
    constructed seam. This is the table the preregistered rule is applied to.

    A tolerance with fewer than ``min_boundaries`` scorable seams is marked
    ``eligible=False``: its median would be a statistic over a handful of items
    that happened to agree, and choosing an operating point from that is worse
    than reporting that no configuration qualified.
    """
    if rendered is None or not len(rendered) or candidates is None \
            or not len(candidates):
        return pd.DataFrame()
    assert_development_only(rendered, what="the tolerance-selection item set")

    reference_kinds = set(rendered.get(
        "reference_kind", pd.Series([synthetic_mod.AUDIO_SPLICE] * len(rendered)))
        .astype(str))
    if not reference_kinds <= synthetic_mod.LEXICAL_REFERENCE_KINDS:
        # An RMS/VAD seam can diagnose systematic coordinate offsets but is not
        # the lexical reference required by the sealed selection estimand.
        return pd.DataFrame([{
            "tolerance_ms": float(tolerance),
            "purpose": DEVELOPMENT_PURPOSE,
            "instrument": "audio_splice_diagnostic",
            "reference_kinds": ",".join(sorted(reference_kinds)),
            "eligible": False,
            "accepted_spans": 0,
            "rejected_units": 0,
            "items_scored": 0,
            "min_boundaries": int(min_boundaries),
            "reason": "missing_genuine_lexical_boundaries",
            "n": 0,
            "median_abs_error_ms": float("nan"),
            "p90_abs_error_ms": float("nan"),
            "median_signed_error_ms": float("nan"),
            "within_100ms": float("nan"),
        } for tolerance in tolerances_ms])

    rows: list[dict[str, Any]] = []
    for tol in tolerances_ms:
        accepted, rejected, _ = build(candidates, config, tolerance_ms=float(tol))
        measured = _combined_error(rendered, accepted, estimator=config.estimator)
        rows.append({
            "tolerance_ms": float(tol),
            "purpose": DEVELOPMENT_PURPOSE,
            "instrument": INSTRUMENT_SYNTHETIC_DEV,
            "accepted_spans": int(len(accepted)),
            "rejected_units": int(len(rejected)),
            "items_scored": int(measured.get("scorable_items", measured["n"])),
            "min_boundaries": int(min_boundaries),
            "eligible": bool(measured["n"] >= int(min_boundaries)),
            **{k: v for k, v in measured.items() if k != "scorable_items"},
        })
    return pd.DataFrame(rows)


def select_operating_tolerance(evidence: pd.DataFrame, *,
                               max_median_ms: float = 100.0,
                               max_p90_ms: float = 200.0,
                               instrument: str = INSTRUMENT_SYNTHETIC_DEV
                               ) -> dict[str, Any]:
    """Apply the preregistered rule to development evidence.

    Returns a record, always. ``selected_tolerance_ms`` is ``None`` when nothing
    qualified, and the caller must treat that as missing evidence rather than
    falling back to a default -- a provisional 200 ms recorded as "selected"
    would claim a decision nobody took.
    """
    record: dict[str, Any] = {
        "instrument": str(instrument),
        "rule": ("largest swept tolerance whose accepted spans satisfy "
                 f"median <= {max_median_ms} ms and p90 <= {max_p90_ms} ms, "
                 f"measured on {instrument} known boundaries (canonical "
                 "unit-edge convention, worse edge per item)"),
        "thresholds": {"max_median_abs_error_ms": float(max_median_ms),
                       "max_p90_abs_error_ms": float(max_p90_ms)},
        "selected_tolerance_ms": None,
        "qualifying_tolerances_ms": [],
        "evidence": [],
        "rejection_reasons": {},
        "selected_by": "external accuracy on development items, never coverage",
    }
    if evidence is None or not len(evidence):
        record["rejection_reasons"]["*"] = ["no development evidence"]
        return record

    record["evidence"] = evidence.to_dict(orient="records")
    eligible = evidence[evidence["eligible"].astype(bool)]
    accuracy = {float(row["tolerance_ms"]): {
        "median_abs_error_ms": float(row["median_abs_error_ms"]),
        "p90_abs_error_ms": float(row["p90_abs_error_ms"]),
    } for _, row in eligible.iterrows()}
    chosen = select_primary_tolerance(evidence, accuracy,
                                      max_median_ms=max_median_ms,
                                      max_p90_ms=max_p90_ms)
    record["selected_tolerance_ms"] = chosen["selected_tolerance_ms"]
    record["qualifying_tolerances_ms"] = chosen["qualifying_tolerances_ms"]

    for _, row in evidence.iterrows():
        why: list[str] = []
        if row.get("reason"):
            why.append(str(row["reason"]))
        if not bool(row["eligible"]):
            why.append(f"items_scored={int(row['items_scored'])}"
                       f"<{int(row['min_boundaries'])}")
        median = float(row["median_abs_error_ms"])
        p90 = float(row["p90_abs_error_ms"])
        if not np.isfinite(median) or median > max_median_ms:
            why.append(f"median_abs_error_ms={median:.1f}>{max_median_ms}")
        if not np.isfinite(p90) or p90 > max_p90_ms:
            why.append(f"p90_abs_error_ms={p90:.1f}>{max_p90_ms}")
        if why:
            record["rejection_reasons"][str(int(row["tolerance_ms"]))] = why

    if record["selected_tolerance_ms"] is not None:
        row = evidence[evidence["tolerance_ms"]
                       == float(record["selected_tolerance_ms"])].iloc[0]
        record["measured_at_selected"] = {
            "median_abs_error_ms": float(row["median_abs_error_ms"]),
            "p90_abs_error_ms": float(row["p90_abs_error_ms"]),
            "items_scored": int(row["items_scored"]),
            "accepted_spans": int(row["accepted_spans"]),
        }
    return record


def apply_selection(config: ConsensusConfig, record: Mapping[str, Any]
                    ) -> ConsensusConfig:
    """The consensus configuration the selection implies.

    Erosion and union padding come from the same measurement, following
    `spec.yaml alignment_prereg.erosion_rule` with the development instrument in
    place of the human one: erosion is the measured median absolute error capped
    by a fraction of the span, and padding is the measured p90.
    """
    tolerance = record.get("selected_tolerance_ms")
    if tolerance is None:
        return config
    measured = dict(record.get("measured_at_selected") or {})
    median = measured.get("median_abs_error_ms")
    p90 = measured.get("p90_abs_error_ms")
    return dataclasses.replace(
        config,
        tolerance_ms=float(tolerance),
        tolerance_selected=True,
        selection_rule=str(record.get("rule", "")),
        measurement_instrument=str(record.get("instrument", "unmeasured")),
        erosion_ms=(None if median is None or not np.isfinite(float(median))
                    else float(median)),
        union_padding_ms=(None if p90 is None or not np.isfinite(float(p90))
                          else float(p90)),
    )


# --------------------------------------------------------------------------
# the development set, shared by L1a and L1b
# --------------------------------------------------------------------------
#: Published by L1a, read by L1b. It lives under `synthetic/` rather than under a
#: stage-prefixed metrics name because two stages depend on it: L1a selects the
#: decoder-query convention on it, L1b selects the operating tolerance on it, and
#: both must be looking at the same configured candidate pool or the two
#: decisions were made on different data.
DEV_ITEMS_FILE = "synthetic/dev_items.parquet"
DEV_ITEMS_SCHEMA = "lss_synthetic_dev_items_v1"


def build_dev_set(cfg: Mapping[str, Any], *, stage: str,
                  run_dir: str | Path | None = None,
                  taint_reasons: Sequence[str] = ()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render the synthetic development set and publish it authenticated.

    The pool is partitioned before either set is drawn, so the gate set can never
    contain a development recording however many times either is rebuilt. The
    development set is deliberately *not* filtered by the exposure ledger: it is
    meant to be reused, and a dev set that changed every run would make the
    selection unreproducible.
    """
    from ...lss import manifest as manifest_mod
    from ..roles import load_role
    from ..seeds import seed_for

    scfg = dict(cfg.get("synthetic") or {})
    root = Path(cfg["experiment"]["output_root"])
    construct = load_role(cfg, str(scfg.get("source_role", "D-construct")))
    seed = seed_for(cfg, "synthetic_dev")
    pools = synthetic_mod.partition_sources(construct, seed=seed)
    frame, meta = synthetic_mod.build_set(
        construct, cfg, n_pairs=int(scfg.get("num_pairs_dev", 100)), seed=seed,
        out_dir=root / "synthetic", purpose=DEVELOPMENT_PURPOSE,
        sources=pools[DEVELOPMENT_PURPOSE])
    if not len(frame):
        return frame, {**meta, "published": None}
    published = manifest_mod.publish_frame(
        root / DEV_ITEMS_FILE, frame, stage=stage, cfg=cfg, run_dir=run_dir,
        taint_reasons=list(taint_reasons), key_columns=("pair_id",),
        schema=DEV_ITEMS_SCHEMA)
    return frame, {**meta, "published": published,
                   "path": str(root / DEV_ITEMS_FILE),
                   "sha256": published.get("sha256")}


def load_dev_set(artifacts_root: str | Path, *,
                 require_authentication: bool = True
                 ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the authenticated development set L1a published.

    Identity is *not* required: L1a and L1b resolve different config files, so
    their `config_sha256` differ by construction and requiring a match would
    refuse every legitimate hand-off. The hash, completeness and taint of the
    bytes are still checked, which is what authenticity means here.
    """
    from ...lss.manifest import read_verified

    path = Path(artifacts_root) / DEV_ITEMS_FILE
    if not path.is_file():
        return pd.DataFrame(), {"available": False, "reason": "missing",
                                "path": str(path),
                                "detail": f"{path} does not exist; run l1a"}
    if not require_authentication:
        return pd.read_parquet(path), {"available": True, "reason": "unverified",
                                       "path": str(path)}
    frame, verdict = read_verified(path, cfg=None, require_identity=False)
    if not verdict["ok"]:
        return pd.DataFrame(), {
            "available": False, "reason": "unauthenticated", "path": str(path),
            "detail": f"{verdict['verdict']}: {verdict.get('detail') or ''}".strip(": ")}
    manifest = verdict.get("manifest") or {}
    if manifest.get("diagnostic_only"):
        return pd.DataFrame(), {"available": False, "reason": "tainted",
                                "path": str(path),
                                "detail": str(manifest.get("taint_reasons"))}
    if manifest.get("schema") != DEV_ITEMS_SCHEMA:
        return pd.DataFrame(), {"available": False, "reason": "schema_mismatch",
                                "path": str(path),
                                "detail": f"{manifest.get('schema')} != {DEV_ITEMS_SCHEMA}"}
    return frame, {"available": True, "reason": "ok", "path": str(path),
                   "sha256": manifest.get("sha256"),
                   "producing_run_id": manifest.get("producing_run_id"),
                   "rows": int(len(frame)), "manifest": manifest}


# --------------------------------------------------------------------------
# the decoder-query convention (L1a)
# --------------------------------------------------------------------------
#: artifacts the L1a development sweep produces
PRED_START_SWEEP_TABLE = "metrics/l1a_pred_start_sweep.parquet"
ALIGNER_SELECTION_FILE = "freeze/l1a_alignment_selection.json"
ALIGNER_SELECTION_SCHEMA = "lss_aligner_selection_v1"


def whisper_variant_sweep(bundle, rendered: pd.DataFrame, cfg: Mapping[str, Any],
                          identity, *, offsets: Sequence[int] = (-1, 0),
                          out_dir: str | Path | None = None) -> pd.DataFrame:
    """Score each decoder-query convention against typed development references.

    Runs through `nat5h.aligners.run_whisper_dtw` -- the same function production
    alignment uses -- so what is selected here is what will actually run. The
    older `bias.pred_start_sweep` called `align_batch` directly and scored the
    first ZH->EN switch under the legacy midpoint convention, which is not the
    quantity a steering mask consumes; this scores unit edges.  For the current
    RMS/VAD construction the output is explicitly seam-relative and cannot be
    consumed as lexical Gate-A calibration.
    """
    from ...nat5h.aligners import dtw_variant_label, run_whisper_dtw

    assert_development_only(rendered, what="the pred_start sweep item set")
    manifest = synthetic_mod.aligner_manifest(rendered)
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        table, mapping = run_whisper_dtw(bundle, manifest, dict(cfg), identity,
                                         pred_start_offset=int(offset))
        scores, per_item = synthetic_mod.score_rendered_set(
            rendered, table, purpose=DEVELOPMENT_PURPOSE)
        combined = scores[(scores["edge"] == "combined")
                          & (scores["convention"]
                             == synthetic_mod.CANONICAL_CONVENTION)] \
            if len(scores) else scores
        row: dict[str, Any] = {
            "pred_start_offset": int(offset),
            "aligner_variant": dtw_variant_label(int(offset)),
            "convention": ("query_predicting_token" if int(offset) == -1
                           else "query_at_token"),
            "purpose": DEVELOPMENT_PURPOSE,
            "instrument": INSTRUMENT_SYNTHETIC_DEV,
            "items": int(len(rendered)),
            "candidate_rows": int(len(table)),
            "valid_rows": int(table["is_valid"].sum()) if len(table)
            and "is_valid" in table else 0,
            "scorable_items": int(per_item["scorable"].sum()) if len(per_item) else 0,
            "unit_coverage": float(mapping.get("unit_coverage", float("nan"))),
        }
        reference_kinds = sorted(set(rendered.get(
            "reference_kind",
            pd.Series([synthetic_mod.AUDIO_SPLICE] * len(rendered))).astype(str)))
        row["reference_kinds"] = ",".join(reference_kinds)
        if set(reference_kinds) == {synthetic_mod.AUDIO_SPLICE}:
            seam = scores[(scores["reference_kind"] == synthetic_mod.AUDIO_SPLICE)
                          & (scores["convention"] == "audio_splice")
                          & (scores["edge"] == "seam")] if len(scores) else scores
            if len(seam):
                best = seam.iloc[0]
                row.update({
                    "metric_semantics": "audio_seam_relative_not_lexical_accuracy",
                    "selection_metric": "median_absolute_splice_edge_offset_ms",
                    "n_boundaries": int(best["num_boundaries"]),
                    "median_absolute_splice_edge_offset_ms": float(
                        best["median_absolute_splice_edge_offset_ms"]),
                    "p90_absolute_splice_edge_offset_ms": float(
                        best["p90_absolute_splice_edge_offset_ms"]),
                    "median_zh_end_minus_splice_ms": float(
                        best["median_zh_end_minus_splice_ms"]),
                    "median_en_start_minus_splice_ms": float(
                        best["median_en_start_minus_splice_ms"]),
                    "within_100ms_of_splice": float(
                        best["within_100ms_of_splice"]),
                })
            else:
                row.update({
                    "metric_semantics": "audio_seam_relative_not_lexical_accuracy",
                    "selection_metric": "median_absolute_splice_edge_offset_ms",
                    "n_boundaries": 0,
                    "median_absolute_splice_edge_offset_ms": float("nan"),
                    "p90_absolute_splice_edge_offset_ms": float("nan"),
                    "median_zh_end_minus_splice_ms": float("nan"),
                    "median_en_start_minus_splice_ms": float("nan"),
                    "within_100ms_of_splice": float("nan"),
                })
        elif set(reference_kinds) <= synthetic_mod.LEXICAL_REFERENCE_KINDS:
            if len(combined):
                best = combined.iloc[0]
                row.update({
                    "metric_semantics": "absolute_lexical_boundary_accuracy",
                    "selection_metric": "median_abs_error_ms",
                    "n_boundaries": int(best["num_boundaries"]),
                    "median_abs_error_ms": float(best["median_abs_error_ms"]),
                    "p90_abs_error_ms": float(best["p90_abs_error_ms"]),
                    "median_signed_error_ms": float(best["median_signed_error_ms"]),
                    "within_100ms": float(best["within_100ms"]),
                })
            else:
                row.update({
                    "metric_semantics": "absolute_lexical_boundary_accuracy",
                    "selection_metric": "median_abs_error_ms",
                    "n_boundaries": 0,
                    "median_abs_error_ms": float("nan"),
                    "p90_abs_error_ms": float("nan"),
                    "median_signed_error_ms": float("nan"),
                    "within_100ms": float("nan"),
                })
        else:
            raise ValueError("a pred_start sweep cannot mix incompatible "
                             f"reference kinds: {reference_kinds}")
        rows.append(row)
        if out_dir is not None and len(table):
            path = Path(out_dir) / f"candidates_whisper_dtw_pred{int(offset)}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
    return pd.DataFrame(rows)


def select_whisper_variant(sweep: pd.DataFrame, *,
                           min_boundaries: int = 50) -> dict[str, Any]:
    """Pick the convention with the smallest compatible development offset.

    Selection, not a gate: both conventions are legitimate implementations of the
    same estimator, and the question is only which one reads the attention the way
    the reference implementation does. Accuracy on a fresh gate set still has to
    clear the proposal's thresholds afterwards, and this cannot help with that.
    """
    from ...nat5h.aligners import DEFAULT_PRED_START_OFFSET, dtw_variant_label

    is_splice = (sweep is not None and "metric_semantics" in sweep.columns
                 and len(sweep)
                 and set(sweep["metric_semantics"].astype(str))
                 == {"audio_seam_relative_not_lexical_accuracy"})
    median_column = ("median_absolute_splice_edge_offset_ms" if is_splice
                     else "median_abs_error_ms")
    p90_column = ("p90_absolute_splice_edge_offset_ms" if is_splice
                  else "p90_abs_error_ms")
    record: dict[str, Any] = {
        "instrument": INSTRUMENT_SYNTHETIC_DEV,
        "metric_semantics": ("audio_seam_relative_not_lexical_accuracy"
                             if is_splice else
                             "absolute_lexical_boundary_accuracy"),
        "rule": ("smallest median absolute audio-splice edge offset on the "
                 "synthetic development set; this selects a decoder coordinate "
                 "convention and is not lexical Gate-A calibration; ties broken "
                 "by p90, then by the configured order" if is_splice else
                 "smallest median absolute lexical-boundary error on the "
                 "development set; ties broken by p90, then by the configured order"),
        "min_boundaries": int(min_boundaries),
        "default_pred_start_offset": int(DEFAULT_PRED_START_OFFSET),
        "pred_start_offset": None,
        "aligner_variant": None,
        "changed_the_default": False,
        "evidence": [],
        "rejected": {},
    }
    if sweep is None or not len(sweep):
        record["rejected"]["*"] = ["no development evidence"]
        return record
    record["evidence"] = sweep.to_dict(orient="records")
    if median_column not in sweep or p90_column not in sweep:
        record["rejected"]["*"] = [
            f"missing compatible selection columns {median_column}, {p90_column}"]
        return record
    eligible = sweep[sweep["n_boundaries"].astype(int) >= int(min_boundaries)]
    eligible = eligible[np.isfinite(eligible[median_column].to_numpy(dtype=float))]
    for _, row in sweep.iterrows():
        why: list[str] = []
        if int(row["n_boundaries"]) < int(min_boundaries):
            why.append(f"n_boundaries={int(row['n_boundaries'])}<{min_boundaries}")
        if not np.isfinite(float(row[median_column])):
            why.append(f"{median_column} is not finite")
        if why:
            record["rejected"][str(int(row["pred_start_offset"]))] = why
    if not len(eligible):
        return record
    ordered = eligible.sort_values([median_column, p90_column],
                                   kind="mergesort")
    best = ordered.iloc[0]
    offset = int(best["pred_start_offset"])
    record.update({
        "pred_start_offset": offset,
        "aligner_variant": dtw_variant_label(offset),
        "changed_the_default": offset != int(DEFAULT_PRED_START_OFFSET),
        "measured": ({
            "median_absolute_splice_edge_offset_ms": float(best[median_column]),
            "p90_absolute_splice_edge_offset_ms": float(best[p90_column]),
            "median_zh_end_minus_splice_ms": float(
                best["median_zh_end_minus_splice_ms"]),
            "median_en_start_minus_splice_ms": float(
                best["median_en_start_minus_splice_ms"]),
            "within_100ms_of_splice": float(best["within_100ms_of_splice"]),
            "n_boundaries": int(best["n_boundaries"]),
        } if is_splice else {
            "median_abs_error_ms": float(best[median_column]),
            "p90_abs_error_ms": float(best[p90_column]),
            "median_signed_error_ms": float(best["median_signed_error_ms"]),
            "within_100ms": float(best["within_100ms"]),
            "n_boundaries": int(best["n_boundaries"]),
        }),
    })
    return record


def aligner_selection(whisper: Mapping[str, Any], *,
                      qwen_language: str | None = None,
                      dev_artifact: Mapping[str, Any] | None = None,
                      seeds: Mapping[str, Any] | None = None,
                      thresholds: Mapping[str, Any] | None = None
                      ) -> dict[str, Any]:
    """The frozen aligner-configuration record L1a writes and L1b consumes."""
    return {
        "schema_version": ALIGNER_SELECTION_SCHEMA,
        "whisper_dtw": dict(whisper),
        "pred_start_offset": whisper.get("pred_start_offset"),
        "qwen_language": qwen_language,
        "qwen_language_rule": (
            "one frozen variant: qwen_asr 0.0.6 routes Chinese and English "
            "through the same tokenizer, so a second variant is a bit-identical "
            "duplicate and `_family_representatives` would discard it "
            "(see qwen_probe.PROBE_LANGUAGE)"),
        "instrument": INSTRUMENT_SYNTHETIC_DEV,
        "development_artifact": dict(dev_artifact or {}),
        "seeds": dict(seeds or {}),
        "thresholds": dict(thresholds or {}),
    }


def load_aligner_selection(artifacts_root: str | Path, *,
                           cfg: Mapping[str, Any] | None = None,
                           require_authentication: bool = True) -> dict[str, Any]:
    """Read L1a's frozen aligner selection, or say why it cannot be used."""
    import json

    from ..manifest import verify

    path = Path(artifacts_root) / ALIGNER_SELECTION_FILE
    if not path.is_file():
        return {"available": False, "reason": "missing", "path": str(path),
                "detail": f"{path} does not exist; run l1a"}
    if require_authentication:
        verdict = verify(path, cfg=cfg, require_identity=False)
        if not verdict["ok"]:
            return {"available": False, "reason": "unauthenticated",
                    "path": str(path),
                    "detail": f"{verdict['verdict']}: {verdict.get('detail') or ''}"
                    .strip(": ")}
        if (verdict.get("manifest") or {}).get("diagnostic_only"):
            return {"available": False, "reason": "tainted", "path": str(path),
                    "detail": str((verdict.get("manifest") or {})
                                  .get("taint_reasons"))}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": "unreadable", "path": str(path),
                "detail": repr(exc)}
    if payload.get("schema_version") != ALIGNER_SELECTION_SCHEMA:
        return {"available": False, "reason": "schema_mismatch", "path": str(path),
                "detail": f"{payload.get('schema_version')} != "
                          f"{ALIGNER_SELECTION_SCHEMA}"}
    return {"available": True, "reason": "ok", "path": str(path),
            "payload": payload,
            "pred_start_offset": payload.get("pred_start_offset")}


def operating_point(record: Mapping[str, Any], *,
                    aligner_selection: Mapping[str, Any] | None = None,
                    dev_artifact: Mapping[str, Any] | None = None,
                    seeds: Mapping[str, Any] | None = None,
                    config: ConsensusConfig | None = None) -> dict[str, Any]:
    """The frozen record of every configuration choice this gate rests on.

    One artifact answers "what configuration produced this Gate A result": the
    selected tolerance, the aligner variants selected upstream in L1a (by
    reference and hash), the rule and instrument, the development artifact the
    decision was made on, the thresholds and the seeds.
    """
    return {
        "schema_version": OPERATING_POINT_SCHEMA,
        "selection": dict(record),
        "selected_tolerance_ms": record.get("selected_tolerance_ms"),
        "instrument": record.get("instrument"),
        "rule": record.get("rule"),
        "thresholds": dict(record.get("thresholds") or {}),
        "aligner_selection": dict(aligner_selection or {}),
        "development_artifact": dict(dev_artifact or {}),
        "seeds": dict(seeds or {}),
        "consensus": ({} if config is None else {
            "estimator": config.estimator,
            "tolerance_ms": config.tolerance_ms,
            "min_families": config.min_families,
            "bins": dict(config.bins or {}),
            "erosion_ms": config.erosion_ms,
            "erosion_fraction": config.erosion_fraction,
            "min_safe_interior_ms": config.min_safe_interior_ms,
            "union_padding_ms": config.union_padding_ms,
            "measurement_instrument": config.measurement_instrument,
        }),
    }


def load_operating_point(artifacts_root: str | Path, *,
                         cfg: Mapping[str, Any] | None = None,
                         require_authentication: bool = True) -> dict[str, Any]:
    """Read the frozen operating point, or say why it cannot be used."""
    import json

    from ..manifest import verify

    path = Path(artifacts_root) / OPERATING_POINT_FILE
    if not path.is_file():
        return {"available": False, "reason": "missing",
                "detail": f"{path} does not exist", "path": str(path)}
    if require_authentication:
        verdict = verify(path, cfg=cfg, require_identity=cfg is not None)
        if not verdict["ok"]:
            return {"available": False, "reason": "unauthenticated",
                    "detail": f"{verdict['verdict']}: {verdict.get('detail') or ''}"
                    .strip(": "), "path": str(path)}
        if (verdict.get("manifest") or {}).get("diagnostic_only"):
            return {"available": False, "reason": "tainted",
                    "detail": str((verdict.get("manifest") or {})
                                  .get("taint_reasons")), "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": "unreadable",
                "detail": repr(exc), "path": str(path)}
    if payload.get("schema_version") != OPERATING_POINT_SCHEMA:
        return {"available": False, "reason": "schema_mismatch",
                "detail": f"{payload.get('schema_version')} != "
                          f"{OPERATING_POINT_SCHEMA}", "path": str(path)}
    return {"available": True, "reason": "ok", "path": str(path),
            "payload": payload,
            "selected_tolerance_ms": payload.get("selected_tolerance_ms"),
            "instrument": payload.get("instrument")}
