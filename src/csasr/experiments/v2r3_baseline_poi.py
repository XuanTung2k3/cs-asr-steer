"""B0_AUTO hypotheses and lexical POI tables for the v2r3 dialogue-atomic sample.

Why this exists as a separate driver rather than a call into `p0_baseline`: that
module is a *stage*, not a callable scoring path.  Its POI loop is hard-coded to
`DEV_SUBSETS = ("dev_select", "dev_confirm")`, it loads v1 split manifests
through `load_subset`, it writes `poi_{subset}.parquet` into the v1 artifacts
root -- over the immutable tables -- and it selects a primary baseline and
evaluates the P0 gate.  Reaching D-construct on the v2r3 sample through it would
require changing all four behaviours.

The *scoring* is not reimplemented here.  `pier.evaluate_pois` and
`pier.poi_table` are imported and called unmodified; this module only chooses
which utterances to decode, where to write, and what provenance to record.  It
evaluates no gate, selects no baseline, and writes no `primary_baseline.json`.

Roles are restricted to `D-construct` and `D-dev-select` by name.  `D-dev-confirm`
and `D-test` are refused rather than silently skipped: a held-out split must fail
loudly when requested, never quietly produce nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..evaluation.pier import poi_table          # used unmodified
from ..lss import manifest as manifest_mod
from ..lss.v2_namespace import configured_paths, validate_namespace
from ..models.generation import decode_manifest
from ..models.whisper import load_whisper
from ..nat5h.statusing import atomic_write_json
from ..utils.config import load_config
from ..utils.hashing import sha256_file
from ..utils.logging import setup_logging

STAGE = "v2r3_baseline_poi"

#: The only roles this driver may touch.  Everything else is refused by name.
PERMITTED_ROLES = ("D-construct", "D-dev-select")

#: B0_AUTO is hard-bound.  With a single baseline there is nothing to select, and
#: a selection step would write the `primary_baseline.json` this driver must not
#: produce.
SYSTEM = "B0_AUTO"
LANGUAGE = None

COMPLETE_MARKER = "baseline_poi_complete.manifest.json"


class BaselinePoiError(RuntimeError):
    """The request or the destination cannot support this run."""


def resolve_roles(requested: list[str]) -> tuple[str, ...]:
    """Accept only the two permitted roles, naming any refusal explicitly."""
    if not requested:
        raise BaselinePoiError("at least one role must be requested")
    refused = [r for r in requested if r not in PERMITTED_ROLES]
    if refused:
        raise BaselinePoiError(
            f"role(s) {refused} are not permitted here; this driver generates "
            f"baselines only for {list(PERMITTED_ROLES)}. D-dev-confirm and "
            "D-test are held out and must not receive baseline labelling.")
    seen: list[str] = []
    for role in requested:
        if role not in seen:
            seen.append(role)
    return tuple(seen)


def resolve_output(output: str | Path) -> Path:
    """Refuse a destination that already exists, in whole or in part."""
    target = Path(output)
    if target.is_symlink():
        raise BaselinePoiError(f"refusing a symlink destination: {target}")
    target = target.resolve()
    if target.exists() or manifest_mod.manifest_path(target).exists():
        raise BaselinePoiError(
            f"refusing to overwrite an existing destination: {target}; "
            "baseline generations are write-once")
    return target


def sampled_utterances(candidate_path: Path, role: str) -> set[str]:
    """The utterances this role actually contributed to the candidate table."""
    frame = pd.read_parquet(candidate_path, columns=["utterance_id", "role"])
    return set(frame.loc[frame["role"] == role, "utterance_id"].astype(str))


def role_frame(role_root: Path, role: str, keep: set[str]) -> pd.DataFrame:
    """Role manifest rows for exactly the sampled utterances, in stable order."""
    path = role_root / f"role_{role}.parquet"
    if not path.is_file():
        raise BaselinePoiError(f"role manifest missing: {path}")
    frame = pd.read_parquet(path)
    frame = frame[frame["utterance_id"].astype(str).isin(keep)]
    missing = keep - set(frame["utterance_id"].astype(str))
    if missing:
        raise BaselinePoiError(
            f"{len(missing)} sampled utterance(s) are absent from {path.name}: "
            f"{sorted(missing)[:3]}")
    return frame.sort_values("utterance_id").reset_index(drop=True)


def build_poi(manifest: pd.DataFrame, predictions: pd.DataFrame,
              role: str) -> pd.DataFrame:
    """Per-POI records via `pier.poi_table`, unmodified, plus row provenance."""
    merged = manifest.merge(predictions, on="utterance_id", how="inner")
    if len(merged) != len(manifest):
        raise BaselinePoiError(
            f"{role}: {len(manifest) - len(merged)} utterance(s) received no "
            "hypothesis; every sampled utterance must be decoded")
    rows = poi_table(merged["utterance_id"].tolist(),
                     merged["transcript_raw"].tolist(),
                     merged["hypothesis_raw"].tolist())
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    columns = [c for c in ("utterance_id", "speaker_id", "conversation_id",
                           "dialogue_id", "duration_sec") if c in manifest.columns]
    table = table.merge(manifest[columns], on="utterance_id", how="left")
    table["role"] = role
    return table


def summarise(table: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    """Counts only.  No threshold is applied and no gate is evaluated."""
    if not len(table):
        return {"lexical_units": 0, "correct": 0, "baseline_error": 0,
                "utterances_with_units": 0, "utterances": int(len(manifest)),
                "dialogues": int(manifest["dialogue_id"].nunique())
                if "dialogue_id" in manifest else 0,
                "unmatched_units": 0, "per_category": {}}
    correct = table["correct"].astype(bool)
    unmatched = int(table["category"].isna().sum()) if "category" in table else 0
    return {
        "lexical_units": int(len(table)),
        "correct": int(correct.sum()),
        "baseline_error": int((~correct).sum()),
        "utterances_with_units": int(table["utterance_id"].nunique()),
        "utterances": int(len(manifest)),
        "dialogues": int(manifest["dialogue_id"].nunique())
        if "dialogue_id" in manifest else 0,
        "unmatched_units": unmatched,
        "per_category": {str(k): int(v) for k, v in
                         table.loc[~correct, "category"].value_counts().items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--roles", nargs="+", default=list(PERMITTED_ROLES))
    parser.add_argument("--output-dir", required=True,
                        help="new write-once directory under the v2r3 namespace")
    parser.add_argument("--execute", action="store_true",
                        help="without this flag no model is loaded and nothing is written")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    roles = resolve_roles(list(args.roles))
    isolation = validate_namespace(cfg, require_freeze=True)
    paths = configured_paths(cfg)

    generation = paths.candidate_generation_root / "generation_complete.manifest.json"
    if not generation.is_file():
        raise BaselinePoiError(
            f"the v2r3 candidate generation is not complete: {generation} is "
            "missing; a generation without its completion manifest is absent")
    generation_payload = json.loads(generation.read_text(encoding="utf-8"))
    if not generation_payload.get("complete"):
        raise BaselinePoiError(f"candidate generation is not complete: {generation}")

    target = resolve_output(args.output_dir)
    plan = {
        "roles": list(roles),
        "system": SYSTEM,
        "language": LANGUAGE,
        "decoding": cfg["decoding"],
        "model": {"hub_id": (cfg.get("model") or {}).get("hub_id"),
                  "revision": (cfg.get("model") or {}).get("revision")},
        "candidate_generation": str(paths.candidate_generation_root),
        "candidate_sha256": generation_payload.get("candidate_sha256"),
        "role_root": str(paths.role_root),
        "destination": str(target),
        "evaluates_gate": False,
        "selects_primary_baseline": False,
    }
    counts = {role: len(sampled_utterances(paths.candidate_path, role))
              for role in roles}
    plan["sampled_utterances"] = counts

    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan,
                          "isolation": isolation}, indent=2, default=str))
        return 0

    log = setup_logging(level=str((cfg.get("runtime") or {}).get("log_level", "INFO")))
    parents: list[dict[str, Any]] = []
    for artifact in (paths.candidate_path, paths.freeze_path):
        published = manifest_mod.load(artifact)
        if not published:
            raise BaselinePoiError(f"required parent has no manifest: {artifact}")
        parents.append(published)
    for role in roles:
        published = manifest_mod.load(paths.role_root / f"role_{role}.parquet")
        if not published:
            raise BaselinePoiError(f"role manifest has no sidecar: {role}")
        parents.append(published)

    bundle = load_whisper(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{target.name}.attempt-",
                                    dir=str(target.parent)))
    promoted = False
    try:
        summary: dict[str, Any] = {}
        for role in roles:
            keep = sampled_utterances(paths.candidate_path, role)
            manifest = role_frame(paths.role_root, role, keep)
            log.info("decoding %s on %s (%d utterances, language=%r)",
                     SYSTEM, role, len(manifest), LANGUAGE)
            predictions = decode_manifest(
                bundle, manifest, cfg, system=SYSTEM, config_id=SYSTEM,
                out_path=attempt / "baselines" / SYSTEM / f"{role}.parquet",
                language=LANGUAGE, resume=False, overwrite=True)
            table = build_poi(manifest, predictions, role)
            table.to_parquet(attempt / f"poi_{role}.parquet", index=False)
            summary[role] = summarise(table, manifest)
            log.info("%s POI audit: %s", role, json.dumps(summary[role]))

        inventory = []
        for artifact in sorted(p for p in attempt.rglob("*") if p.is_file()):
            inventory.append({"relative_path": str(artifact.relative_to(attempt)),
                              "path": str(target / artifact.relative_to(attempt)),
                              "sha256": sha256_file(artifact),
                              "bytes": int(artifact.stat().st_size)})
        atomic_write_json(attempt / COMPLETE_MARKER, {
            "schema": "lss_v2r3_baseline_poi_v1",
            "state": "completed", "complete": True, "published": True,
            "plan": plan,
            "counts": summary,
            "parents": [{"path": p.get("path"), "sha256": p.get("sha256")}
                        for p in parents],
            "artifacts": inventory,
            "evaluates_gate": False,
            "selects_primary_baseline": False,
        })
        os.rename(attempt, target)
        promoted = True
    finally:
        if not promoted and attempt.exists():
            shutil.rmtree(attempt)

    for role in roles:
        manifest_mod.publish(
            target / f"poi_{role}.parquet", stage=STAGE, cfg=cfg, parents=parents,
            schema="lss_v2r3_poi_lexical_units_v1",
            extra={"role": role, "system": SYSTEM, "language": LANGUAGE,
                   "decoding": cfg["decoding"], "counts": summary[role],
                   "evaluates_gate": False})
        manifest_mod.publish(
            target / "baselines" / SYSTEM / f"{role}.parquet", stage=STAGE, cfg=cfg,
            parents=parents, schema="lss_v2r3_b0_auto_hypotheses_v1",
            extra={"role": role, "system": SYSTEM, "language": LANGUAGE,
                   "decoding": cfg["decoding"]})

    print(json.dumps({"state": "completed", "output": str(target),
                      "counts": summary, "plan": plan}, indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
