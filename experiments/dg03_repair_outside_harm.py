#!/usr/bin/env python
"""Populate canonical DG-03 outside harm from stored screen transcripts.

This is a CPU-only post-processing repair.  It never loads Whisper and never
changes a decoded transcript, condition, dose, or selection input.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.evaluation.dg03_outside_harm import (  # noqa: E402
    baseline_outside_harm,
    corpus_outside_harm,
    population_unit_sets,
)
from csasr.evaluation.result_schema import from_json, validate  # noqa: E402
from csasr.evaluation.normalization import normalize_text, segment_units  # noqa: E402


def _git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return None


def _read(path: Path, columns: list[str], *, role: str | None = None):
    filters = [("role", "==", role)] if role is not None else None
    return pq.read_table(path, columns=columns, filters=filters).to_pandas()


def _runs(group: Any) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for _, row in group.sort_values("reference_unit_index").iterrows():
        language = str(row["reference_language"])
        if runs and runs[-1]["language"] == language:
            runs[-1]["rows"].append(row)
        else:
            runs.append({"language": language, "rows": [row]})
    return runs


def _frozen_candidates(candidates, poi, references: dict[str, str], *, sample_rate=16000):
    """Rebuild eligible existing_ctc EN runs under blank_to_preceding."""
    errors = {(str(u), int(i)): not bool(ok)
              for u, i, ok in zip(poi["utterance_id"],
                                  poi["reference_unit_index"], poi["correct"])}
    units_by_utt: dict[str, list[dict[str, Any]]] = {}
    candidates_by_utt: dict[str, list[dict[str, Any]]] = {}
    for uid, group in candidates.groupby("utterance_id", sort=False):
        uid = str(uid)
        rows = []
        for _, row in group.sort_values("reference_unit_index").iterrows():
            start, end = float(row["start_sec"]), float(row["end_sec"])
            if not (start >= 0 and end > start):
                continue
            rows.append({
                "unit_id": int(row["reference_unit_index"]),
                "start_sample": start * sample_rate,
                "end_sample": end * sample_rate,
                "language": str(row["reference_language"]),
            })
        units_by_utt[uid] = rows
        runs = _runs(group)
        for pos, run in enumerate(runs):
            indices = [int(row["reference_unit_index"]) for row in run["rows"]]
            if str(run["language"]) != "EN" or pos == 0 or pos == len(runs) - 1:
                continue
            if str(runs[pos - 1]["language"]) != "ZH":
                continue
            if indices != list(range(indices[0], indices[-1] + 1)):
                continue
            start = min(float(row["start_sec"]) for row in run["rows"])
            # blank_to_preceding assigns the blank after the EN run to it.
            following = runs[pos + 1]["rows"][0]
            end = float(following["start_sec"])
            if (end - start) * 1000.0 < 400.0:
                continue
            if not any(errors.get((uid, index), False) for index in indices):
                continue
            candidates_by_utt.setdefault(uid, []).append({
                "start_sample": start * sample_rate,
                "end_sample": end * sample_rate,
                "unit_indices": indices,
            })

    sets = []
    languages = []
    diagnostics = {"target_candidates": 0, "inside_units": 0,
                   "outside_units": 0, "unknown_units": 0}
    for uid, reference in references.items():
        unit_sets, lang = population_unit_sets(
            reference, units_by_utt.get(uid, []),
            candidates_by_utt.get(uid, []),
        )
        sets.append(unit_sets)
        languages.append(lang)
        diagnostics["target_candidates"] += len(candidates_by_utt.get(uid, []))
        diagnostics["inside_units"] += len(unit_sets.inside)
        diagnostics["outside_units"] += len(unit_sets.outside)
        diagnostics["unknown_units"] += len(unit_sets.unknown)
    return sets, languages, diagnostics


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def repair(screen_path: Path, data_root: Path) -> dict[str, Any]:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    role_path = data_root / "manifests/roles/role_D-dev-select.parquet"
    candidate_path = data_root / "candidate_generations/generation_001/alignments/candidates_all.parquet"
    poi_path = data_root / "baselines/generation_001/poi_D-dev-select.parquet"
    role = _read(role_path, ["utterance_id", "transcript_raw"], role="D-dev-select")
    ids = list(screen["texts"]["C0_baseline"])
    references = dict(zip(role["utterance_id"].astype(str), role["transcript_raw"].astype(str)))
    missing = [uid for uid in ids if uid not in references]
    if missing:
        raise ValueError(f"screen ids missing from D-dev-select role manifest: {missing[:3]}")
    references = {uid: references[uid] for uid in ids}
    candidates = _read(candidate_path, [
        "utterance_id", "reference_unit_index", "reference_language",
        "aligner_family", "start_sec", "end_sec", "role",
    ], role="D-dev-select")
    candidates = candidates[candidates["aligner_family"].astype(str) == "existing_ctc"]
    poi = _read(poi_path, ["utterance_id", "poi_index", "correct"],
                role="D-dev-select").rename(columns={"poi_index": "reference_unit_index"})
    sets, languages, diagnostics = _frozen_candidates(candidates, poi, references)
    base = [screen["texts"]["C0_baseline"][uid] for uid in ids]
    provenance = {
        "schema": "dg03_canonical_outside_harm_v1",
        "repair_git_commit": _git_commit(),
        "source": "offline reconstruction from stored DG-03 transcripts",
        "data_role": "D-dev-select",
        "alignment_family": "existing_ctc",
        "boundary_convention": "blank_to_preceding",
        "min_overlap_ratio": 0.5,
        "insertions_counted_as_harm": False,
        "population": diagnostics,
        "definition": (
            "n_corrupted_outside: baseline-correct to method-wrong trusted "
            "reference units outside the union of frozen eligible embedded-English "
            "target candidates; unknown/untrusted units are excluded; insertions "
            "are excluded per csasr.lss.outcomes.COUNT_INSERTIONS_AS_HARM"
        ),
    }

    def add(result: dict[str, Any], method: list[str]) -> None:
        summary = corpus_outside_harm(
            [references[uid] for uid in ids], base, method, sets, languages)
        result["metrics"]["outside_harm"] = int(summary["outside_harm"])
        result["metrics"]["outside_harm_accounting"] = summary
        result["metrics"]["candidate_utility"] = float(summary["utility"])
        result["provenance"]["outside_harm_note"] = (
            "canonical n_corrupted_outside reconstructed offline from the frozen "
            "D-dev-select existing_ctc candidate population; correctness-flip harm, "
            "not transcript edit count")
        result["provenance"]["outside_harm_accounting"] = provenance
        validate(result)
        # Exercise the canonical result round-trip before writing.
        from_json(from_json(json.dumps(result, allow_nan=False)).to_json())

    add(screen["baseline_result_v1"], base)
    for name, entry in screen["conditions"].items():
        add(entry["result_v1"], [screen["texts"][name][uid] for uid in ids])
    screen["outside_harm_repair"] = {
        **provenance,
        "result_schema": "result_v1",
        "selection_inputs_unchanged": True,
        "gpu_rerun_required": False,
    }
    _atomic_json(screen_path, screen)
    return {"path": str(screen_path), **diagnostics,
            "conditions": {
                "C0_baseline": screen["baseline_result_v1"]["metrics"]["outside_harm"],
                **{name: entry["result_v1"]["metrics"]["outside_harm"]
                   for name, entry in screen["conditions"].items()},
            }}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="append", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    for screen in args.screen:
        print(json.dumps(repair(Path(screen), Path(args.data_root)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
