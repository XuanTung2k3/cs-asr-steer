"""Attach `dialogue_id` to the utterance manifest, with the evidence for it.

CS-Dialogue stores each participant's channel as its own recording, so the
manifest's `conversation_id` and `speaker_id` both name one *side* of a
two-party session.  This stage adds the missing level -- which two participants
shared one session -- and writes the result to a new path.

The grouping is inferred, not read: the release ships no session id.  Every
artifact written here therefore carries `dialogue_pairing_evidence`, which
reports each agreement statistic beside the value a random matching produces.
If any structural check fails the stage refuses to write, because a wrong
dialogue grouping is worse than none: it would silently merge two independent
speakers into one bootstrap cluster.

Read-only against its inputs.  It never rewrites the source manifest, never
touches a status, freeze, exposure, or role artifact, and selects nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.manifest import (DialoguePairingError, MANIFEST_COLUMNS,
                             _read_information_index, derive_dialogue_ids,
                             dialogue_pairing_evidence)
from ..lss import manifest as manifest_mod
from ..utils.config import load_config
from ..utils.hashing import sha256_file

STAGE = "dialogue_manifest"
MANIFEST_NAME = "cs_dialogue_with_dialogue_id.parquet"
EVIDENCE_NAME = "dialogue_pairing_evidence.json"

#: Never write inside these, whatever the caller passes.
PROTECTED = ("/artifacts_lss/status", "/artifacts_lss/freeze",
             "/artifacts_lss/synthetic", "/artifacts_lss/alignments",
             "/artifacts_lss/manifests")

#: Columns that must survive byte-identical from the source manifest.  The whole
#: point of a new artifact is that it adds a column and changes nothing else.
PRESERVED = ("utterance_id", "conversation_id", "speaker_id", "official_split")


def resolve_output(output: str | Path, source: Path) -> Path:
    """Refuse an output that would overwrite or land in a protected tree."""
    target = Path(output).resolve()
    for guard in PROTECTED:
        if guard in str(target):
            raise SystemExit(f"refusing to write inside a protected tree: {target}")
    if target == source.resolve():
        raise SystemExit("refusing to overwrite the source manifest; "
                         "existing artifacts are immutable")
    if target.exists():
        raise SystemExit(f"refusing to overwrite an existing artifact: {target}")
    return target


def attach_dialogue_ids(manifest: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Add `dialogue_id` and `corpus_speaker_id`, leaving every other value alone."""
    if "utterance_id" not in info.columns:
        raise DialoguePairingError("information index has no utterance_id")
    covered = info.set_index("utterance_id")["corpus_speaker_id"]
    missing = int((~manifest["utterance_id"].isin(covered.index)).sum())
    if missing:
        raise DialoguePairingError(
            f"{missing} manifest utterance(s) are absent from the information "
            "index, so their dialogue cannot be derived")

    frame = manifest.copy()
    frame["corpus_speaker_id"] = frame["utterance_id"].map(covered)
    pairs = derive_dialogue_ids(frame)
    frame = frame.drop(columns=["corpus_speaker_id"]).merge(
        pairs, on="conversation_id", how="left", validate="many_to_one")

    for column in PRESERVED:
        if column in manifest.columns and not frame[column].equals(manifest[column]):
            raise DialoguePairingError(
                f"`{column}` changed while attaching dialogue_id; the new manifest "
                "must differ from its source by added columns only")
    if frame["dialogue_id"].isna().any():
        raise DialoguePairingError("some rows received no dialogue_id")
    ordered = [c for c in MANIFEST_COLUMNS if c in frame.columns]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/base.yaml")
    parser.add_argument("--source-manifest", default=None,
                        help="defaults to data.manifest from the config")
    parser.add_argument("--information-index", default=None,
                        help="defaults to data.information_index from the config")
    parser.add_argument("--output-dir", required=True,
                        help="new directory for the dialogue-aware manifest")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    source = Path(args.source_manifest or cfg["data"]["manifest"])
    index = Path(args.information_index or cfg["data"]["information_index"])
    if not source.is_file():
        raise SystemExit(f"source manifest not found: {source}")
    if not index.is_file():
        raise SystemExit(f"information index not found: {index}")

    out_dir = Path(args.output_dir)
    target = resolve_output(out_dir / MANIFEST_NAME, source)
    evidence_path = resolve_output(out_dir / EVIDENCE_NAME, source)

    manifest = pd.read_parquet(source)
    info = _read_information_index(index)
    frame = attach_dialogue_ids(manifest, info)
    evidence = dialogue_pairing_evidence(frame)

    if not evidence["all_checks_pass"]:
        print(json.dumps(evidence, indent=2, default=str))
        raise SystemExit("dialogue pairing evidence did not pass; nothing was written")

    out_dir.mkdir(parents=True, exist_ok=True)
    # the source manifest predates artifact sidecars, so hash it here rather than
    # record a parent with a null sha256: the derivation must be pinned to the
    # exact bytes it read
    source_manifest = manifest_mod.load(source) or {
        "path": str(source), "sha256": sha256_file(source)}
    published = manifest_mod.publish_frame(
        target, frame, stage=STAGE, cfg=cfg,
        parents=[source_manifest],
        key_columns=["utterance_id"],
        schema="cs_dialogue_manifest_with_dialogue_id_v1",
        extra={"dialogue_pairing_evidence": evidence,
               "source_manifest": str(source),
               "information_index": str(index),
               "derivation": "csasr.data.manifest.derive_dialogue_ids",
               "inferred_columns": ["dialogue_id", "corpus_speaker_id"]})
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str),
                             encoding="utf-8")
    manifest_mod.publish(evidence_path, stage=STAGE, cfg=cfg,
                         parents=[published], schema="dialogue_pairing_evidence_v1")

    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "conversations": int(frame["conversation_id"].nunique()),
        "dialogues": int(frame["dialogue_id"].nunique()),
        "manifest": str(target),
        "evidence": str(evidence_path),
        "sha256": published["sha256"],
        "checks_pass": evidence["all_checks_pass"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
