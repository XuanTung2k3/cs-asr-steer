#!/usr/bin/env python3
"""Outcome-blind A6-OTT Search/Confirm manifest repair.

The script consumes only frozen panel metadata and the accepted alignment
procedure output.  It never opens baseline or steering outcome artifacts.
Old manifests remain immutable; v2 manifests are written atomically.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/a6_ott_upper_bound"
OUT = ROOT / "manifests/v2"
sys.path[:0] = [str(REPO), str(REPO / "src")]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def obj_hash(x) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    from experiments.basis_a4_qwen import _load_cs_eval_spans

    old_freeze_path = ROOT / "manifests/SEARCH_FREEZE.json"
    old_freeze = json.loads(old_freeze_path.read_text())
    old_ids = old_freeze["ids"]

    cs_panel_path = REPO / "results/basis_a4/panels/cs_dialogue_300.json"
    cs_panel = json.loads(cs_panel_path.read_text())["rows"]
    cs_by_id = {str(r["utterance_id"]): r for r in cs_panel}
    accepted_cs = _load_cs_eval_spans()
    accepted_cs_ids = {str(uid) for uid, spans in accepted_cs.items() if spans}

    ascend_manifest_path = REPO / "results/basis_a6_expanded/ascend/ASCEND_EVAL_MANIFEST.json"
    ascend_alignment_path = REPO / "results/basis_a6_expanded/ascend/ASCEND_EVAL_ALIGNMENT.json"
    ascend_manifest = json.loads(ascend_manifest_path.read_text())
    ascend_alignment = json.loads(ascend_alignment_path.read_text())["rows"]
    ascend_pool = [str(x) for x in ascend_manifest["utterance_ids"]]
    accepted_ascend_ids = {
        uid for uid in ascend_pool
        if ascend_alignment.get(uid, {}).get("units")
        and any(str(u.get("language_tag")) == "EN" for u in ascend_alignment[uid]["units"])
    }

    # This is the original frozen ordering: the same seed, dataset label, and
    # utterance ID hash used by freeze_search_ids.  No ASR outcome is read.
    ordered_cs = sorted(cs_by_id, key=lambda uid: hashlib.sha256(
        f"{old_freeze['seed']}\0cs_dialogue\0{uid}".encode()).hexdigest())
    eligible_order = [uid for uid in ordered_cs if uid in accepted_cs_ids]
    new_cs = eligible_order[:20]
    new_ascend = [uid for uid in old_ids["ascend"] if uid in accepted_ascend_ids]
    if len(new_cs) != 20 or len(new_ascend) != 20:
        raise RuntimeError(f"v2 Search eligibility failed: CS={len(new_cs)} ASCEND={len(new_ascend)}")

    removed = [uid for uid in old_ids["cs_dialogue"] if uid not in new_cs]
    added = [uid for uid in new_cs if uid not in old_ids["cs_dialogue"]]
    replacements = [{"removed_id": old, "replacement_id": new, "reason": "ORACLE_ALIGNMENT_INELIGIBLE",
                     "steering_outcomes_consulted": False,
                     "removed_status": "STRUCTURALLY_INELIGIBLE",
                     "replacement_status": "FULLY_ELIGIBLE"}
                    for old, new in zip(removed, added)]

    cs_pool = sorted(cs_by_id)
    asc_pool = sorted(ascend_pool)
    confirm_cs = [uid for uid in cs_pool if uid not in new_cs]
    confirm_asc = [uid for uid in asc_pool if uid not in new_ascend]
    cs_confirm_eligible = [uid for uid in confirm_cs if uid in accepted_cs_ids]
    asc_confirm_eligible = [uid for uid in confirm_asc if uid in accepted_ascend_ids]

    source = {
        "cs_panel": {"path": str(cs_panel_path.relative_to(REPO)), "sha256": sha(cs_panel_path), "pool_count": len(cs_pool)},
        "cs_accepted_alignment_procedure": {
            "procedure": "experiments.basis_a4_qwen._load_cs_eval_spans",
            "inputs": [
                {"path": "artifacts_lss/alignments/candidates_all.parquet", "sha256": sha(Path("/mnt/data/tungnx/cs-asr-steer/artifacts_lss/alignments/candidates_all.parquet"))},
                {"path": "artifacts_lss/manifests/roles/role_D-dev-select.parquet", "sha256": sha(Path("/mnt/data/tungnx/cs-asr-steer/artifacts_lss/manifests/roles/role_D-dev-select.parquet"))},
                {"path": "artifacts/manifests/poi_dev_select.parquet", "sha256": sha(Path("/mnt/data/tungnx/cs-asr-steer/artifacts/manifests/poi_dev_select.parquet"))},
            ],
            "accepted_id_count": len(accepted_cs_ids),
        },
        "ascend_manifest": {"path": str(ascend_manifest_path.relative_to(REPO)), "sha256": sha(ascend_manifest_path), "pool_count": len(asc_pool)},
        "ascend_alignment": {"path": str(ascend_alignment_path.relative_to(REPO)), "sha256": sha(ascend_alignment_path), "accepted_id_count": len(accepted_ascend_ids)},
    }
    freeze = {
        "schema_version": "a6_ott_search_freeze_v2",
        "phase": "A", "seed": old_freeze["seed"], "target_per_dataset": 20,
        "selection_rule": "sha256(seed,dataset,utterance_id), first 20 FULLY_ELIGIBLE",
        "ids": {"cs_dialogue": new_cs, "ascend": new_ascend},
        "prior_manifest": str(old_freeze_path.relative_to(REPO)),
        "prior_manifest_sha256": sha(old_freeze_path),
        "source": source, "replacements": replacements,
        "steering_outcomes_consulted": False,
    }
    freeze["freeze_hash"] = obj_hash(freeze)
    phase_a = {
        "schema_version": "a6_ott_phase_manifest_v2", "status": "FROZEN", "phase": "A",
        "search_freeze": "results/a6_ott_upper_bound/manifests/v2/SEARCH_FREEZE.json",
        "search_freeze_hash": freeze["freeze_hash"], "search_ids": freeze["ids"],
        "allowed_artifact_prefixes": ["results/a6_ott_upper_bound/phase_a/", "results/basis_a6_expanded/baselines/*/*/greedy.json"],
        "forbidden_path_terms": ["confirm", "transfer", "seame", "d-test", "phase_b", "phase_c"],
        "steering_outcomes_consulted": False,
    }
    phase_b = {
        "schema_version": "a6_ott_phase_manifest_v2", "status": "LOCKED_UNTIL_PREVIOUS_PHASE", "phase": "B",
        "search_freeze": "results/a6_ott_upper_bound/manifests/v2/SEARCH_FREEZE.json",
        "search_freeze_hash": freeze["freeze_hash"], "search_ids": freeze["ids"],
        "confirm_ids": {"cs_dialogue": confirm_cs, "ascend": confirm_asc},
        "union_checks": {"cs_count": len(confirm_cs), "ascend_count": len(confirm_asc),
                          "cs_union_original_300": len(set(new_cs) | set(confirm_cs)) == 300,
                          "ascend_union_original_220": len(set(new_ascend) | set(confirm_asc)) == 220,
                          "search_confirm_disjoint": not (set(new_cs) & set(confirm_cs)) and not (set(new_ascend) & set(confirm_asc))},
        "confirmation_alignment_audit": {
            "cs_dialogue": {"FULLY_ELIGIBLE": len(cs_confirm_eligible), "STRUCTURALLY_INELIGIBLE": len(confirm_cs) - len(cs_confirm_eligible)},
            "ascend": {"FULLY_ELIGIBLE": len(asc_confirm_eligible), "STRUCTURALLY_INELIGIBLE": len(confirm_asc) - len(asc_confirm_eligible)},
        },
        "allowed_artifact_prefixes": ["results/a6_ott_upper_bound/phase_a/", "results/a6_ott_upper_bound/phase_b/", "results/basis_a6_expanded/baselines/"],
        "forbidden_path_terms": ["transfer", "seame", "d-test", "phase_c"],
    }
    write(OUT / "SEARCH_FREEZE.json", freeze)
    write(OUT / "PHASE_A_MANIFEST.json", phase_a)
    write(OUT / "PHASE_B_MANIFEST.json", phase_b)
    write(OUT / "CONFIRMATION_POOL.json", {"schema_version": "a6_ott_confirmation_pool_v2", "phase": "B",
        "search_freeze_hash": freeze["freeze_hash"], "confirm_ids": phase_b["confirm_ids"],
        "union_checks": phase_b["union_checks"], "alignment_counts": phase_b["confirmation_alignment_audit"],
        "steering_outcomes_consulted": False})
    audit = {
        "schema_version": "a6_ott_search_alignment_audit_v2", "status": "PASS", "phase": "A", "regime": "oracle_tt",
        "steering_outcomes_consulted": False, "source": source, "search": {
            "cs_dialogue": {"FULLY_ELIGIBLE": 20, "RECOVERABLE_ALIGNMENT": 0, "STRUCTURALLY_INELIGIBLE": 0},
            "ascend": {"FULLY_ELIGIBLE": 20, "RECOVERABLE_ALIGNMENT": 0, "STRUCTURALLY_INELIGIBLE": 0}},
        "removed_structurally_ineligible": removed, "replacements": replacements,
        "confirmation": phase_b["confirmation_alignment_audit"], "search_ids": freeze["ids"],
        "manifest_hash": freeze["freeze_hash"],
    }
    write(ROOT / "preflight/SEARCH_ALIGNMENT_AUDIT_V2.json", audit)
    print(json.dumps({"status": "PASS", "search": freeze["ids"], "replacements": replacements,
                      "confirmation": phase_b["confirmation_alignment_audit"], "freeze_hash": freeze["freeze_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
