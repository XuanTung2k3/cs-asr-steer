#!/usr/bin/env python3
"""Outcome-blind A6-OTT method-specific eligibility audit.

This audit never opens a steered result.  Acoustic eligibility is taken from
the accepted D-dev-select alignment loader; decoder eligibility is based on
the frozen transcript language-region labels already present in the panel.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
ROOT = REPO / "results/a6_ott_upper_bound"
ELIG = ROOT / "eligibility"


def sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def contiguous(indices: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for i in sorted(set(map(int, indices))):
        if not out or i != out[-1][-1] + 1:
            out.append([i])
        else:
            out[-1].append(i)
    return out


def transcript_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    from csasr.data.language_tags import EN, tag_units
    from csasr.data.normalize import normalize_and_segment
    _norm, units = normalize_and_segment(str(row["reference"]))
    en = [i for i, tag in enumerate(tag_units(units)) if tag == EN]
    return [{"unit_indices": group, "language": "EN",
             "source": "frozen_reference_language_region"}
            for group in contiguous(en)]


def load_ids() -> dict[str, list[str]]:
    return json.loads((ROOT / "manifests/v2/PHASE_B_MANIFEST.json").read_text())["confirm_ids"]


def panel_rows(panel: str) -> tuple[list[dict[str, Any]], str]:
    from csasr.basis_a6.panels import load_panel
    return load_panel(panel, require_alignment=True)


def audit_dataset(name: str, panel: str, wanted: list[str], acoustic_map: dict[str, list[dict[str, Any]]] | None) -> dict[str, Any]:
    rows, panel_fp = panel_rows(panel)
    by_id = {str(row["utterance_id"]): dict(row) for row in rows}
    if set(wanted) - set(by_id):
        raise RuntimeError(f"{name}: panel missing IDs: {sorted(set(wanted) - set(by_id))[:5]}")
    entries = {}
    for uid in wanted:
        row = by_id[uid]
        acoustic = list((acoustic_map or {}).get(uid, row.get("oracle_spans") or []))
        transcript = transcript_spans(row)
        acoustic_valid = bool(acoustic) and all(
            bool(s.get("finite_span", True)) and float(s.get("end_sec", 0)) > float(s.get("start_sec", 0))
            and bool(s.get("unit_indices")) for s in acoustic)
        decoder_valid = bool(transcript)
        entries[uid] = {
            "utterance_id": uid,
            "acoustic_status": "FULLY_ELIGIBLE" if acoustic_valid else "ACOUSTIC_INELIGIBLE",
            "transcript_status": "FULLY_ELIGIBLE" if decoder_valid else "STRUCTURALLY_INELIGIBLE",
            "oracle_spans": acoustic,
            "oracle_transcript_spans": transcript,
            "acoustic_source": "accepted_D-dev-select_alignment_loader" if acoustic_valid else "none",
            "transcript_source": "frozen_reference_language_region",
            "steering_outcomes_consulted": False,
        }
    ids = set(wanted)
    enc = sorted(uid for uid, x in entries.items() if x["acoustic_status"] == "FULLY_ELIGIBLE")
    dec = sorted(uid for uid, x in entries.items() if x["transcript_status"] == "FULLY_ELIGIBLE")
    common = sorted(set(enc) & set(dec))
    methods = {
        "add_unique_encoder": {"eligible_ids": enc, "ineligible_ids": sorted(ids - set(enc)), "requires": "accepted_acoustic_oracle_span"},
        "add_unique_decoder": {"eligible_ids": dec, "ineligible_ids": sorted(ids - set(dec)), "requires": "accepted_transcript_oracle_region"},
        "conditioning_cs_decoder": {"eligible_ids": dec, "ineligible_ids": sorted(ids - set(dec)), "requires": "accepted_transcript_oracle_region"},
    }
    return {
        "dataset": name, "panel": panel, "panel_fingerprint": panel_fp,
        "total": len(wanted), "entries": entries, "methods": methods,
        "common_ids": common,
        "counts": {"FULLY_ELIGIBLE": len(common),
                   "acoustic_fully_eligible": len(enc),
                   "transcript_fully_eligible": len(dec),
                   "structurally_ineligible_acoustic": len(ids - set(enc)),
                   "structurally_ineligible_transcript": len(ids - set(dec))},
        "steering_outcomes_consulted": False,
    }


def main() -> None:
    ids = load_ids()
    from experiments.basis_a4_qwen import _load_cs_eval_spans
    accepted_cs = _load_cs_eval_spans()
    cs = audit_dataset("CS_CONFIRM_280", "cs_dialogue_dev_select", ids["cs_dialogue"], accepted_cs)
    ascend = audit_dataset("ASCEND_CONFIRM_200", "ascend_eval", ids["ascend"], None)
    # The two SEAME panels are already frozen at exactly 50 rows by their
    # panel manifests; no selection or outcome is performed here.
    seame = {}
    for name, panel in (("SEAME-man", "seame_dev_man"), ("SEAME-sge", "seame_dev_sge")):
        rows, fp = panel_rows(panel)
        seame[name] = audit_dataset(name, panel, [str(r["utterance_id"]) for r in rows], None)
    source = {
        "accepted_cs_loader": "experiments.basis_a4_qwen._load_cs_eval_spans",
        "accepted_cs_span_map_hash": sha(accepted_cs),
        "phase_b_manifest": file_sha(ROOT / "manifests/v2/PHASE_B_MANIFEST.json"),
        "outcome_artifacts_loaded": False,
    }
    for dataset, payload in (("CS_CONFIRM", cs), ("ASCEND_CONFIRM", ascend)):
        atomic(ELIG / f"{dataset}_METHODS.json", {
            "schema_version": "a6_ott_method_eligibility_v1", "status": "PASS",
            "phase": "B", "dataset": dataset, "source": source, **payload,
        })
        for method in ("add_unique_encoder", "add_unique_decoder", "conditioning_cs_decoder"):
            atomic(ELIG / f"{dataset}_{method.upper()}.json", {
                "schema_version": "a6_ott_method_manifest_v1", "status": "PASS",
                "phase": "B", "dataset": dataset, "method": method,
                "ids": payload["methods"][method]["eligible_ids"],
                "ineligible_ids": payload["methods"][method]["ineligible_ids"],
                "source_hash": sha(payload), "steering_outcomes_consulted": False,
            })
    atomic(ELIG / "CS_CONFIRM_COMMON.json", {"schema_version": "a6_ott_common_manifest_v1", "status": "PASS", "phase": "B", "dataset": "CS_CONFIRM_COMMON", "ids": cs["common_ids"], "source_hash": sha(cs), "steering_outcomes_consulted": False})
    atomic(ELIG / "ASCEND_CONFIRM_COMMON.json", {"schema_version": "a6_ott_common_manifest_v1", "status": "PASS", "phase": "B", "dataset": "ASCEND_CONFIRM_COMMON", "ids": ascend["common_ids"], "source_hash": sha(ascend), "steering_outcomes_consulted": False})
    for name, payload in seame.items():
        stem = name.replace("-", "_").upper()
        payload["phase"] = "C"
        payload["source"] = source
        atomic(ELIG / f"{stem}_METHODS.json", {"schema_version": "a6_ott_method_eligibility_v1", "status": "PASS", **payload})
        for method in ("add_unique_encoder", "add_unique_decoder", "conditioning_cs_decoder"):
            atomic(ELIG / f"{stem}_{method.upper()}.json", {"schema_version": "a6_ott_method_manifest_v1", "status": "PASS", "phase": "C", "dataset": name, "method": method, "ids": payload["methods"][method]["eligible_ids"], "ineligible_ids": payload["methods"][method]["ineligible_ids"], "source_hash": sha(payload), "steering_outcomes_consulted": False})
        atomic(ELIG / f"{stem}_COMMON.json", {"schema_version": "a6_ott_common_manifest_v1", "status": "PASS", "phase": "C", "dataset": f"{name}_COMMON", "ids": payload["common_ids"], "source_hash": sha(payload), "steering_outcomes_consulted": False})
    atomic(ELIG / "ALIGNMENT_RECOVERY.json", {
        "schema_version": "a6_ott_confirmation_alignment_recovery_v1", "status": "PASS",
        "recovered": [], "remaining_acoustic_ineligible": cs["counts"]["structurally_ineligible_acoustic"],
        "reason": "accepted D-dev-select alignment loader has no contract-valid span; no alternate authoritative local artifact exists",
        "source": source, "steering_outcomes_consulted": False,
    })
    summary = {"schema_version": "a6_ott_eligibility_summary_v1", "status": "PASS", "cs_confirm": cs["counts"], "ascend_confirm": ascend["counts"], "seame": {k: v["counts"] for k,v in seame.items()}, "source": source}
    atomic(ELIG / "CONFIRMATION_ELIGIBILITY_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
