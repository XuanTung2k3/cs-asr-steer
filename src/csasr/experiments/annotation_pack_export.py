"""Export code-switch boundary clips for human annotation.

Read-only against cached D-dev-select candidates and their source audio; every
write lands under ``--output``.  No aligner is invoked, no production status is
written, no gate generation is allocated or consumed, and no boundary value is
ever placed in an annotation tier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from ..lss import manifest as manifest_mod
from ..lss.align import annotation_pack as pack_mod
from ..utils.config import load_config

PRIMARY_DIR = "clips"
DOUBLE_DIR = "double_annotation"
MANIFEST_CSV = "manifest.csv"
MANIFEST_JSON = "manifest.json"
README = "README.md"

#: Columns the analyst needs.  ``_provisional_center_sec`` is deliberately not
#: among them: the pack must not ship a machine boundary in any form.
MANIFEST_COLUMNS = [
    "item_id", "fingerprint", "utterance_id", "speaker_id", "conversation_id",
    "clip_start_offset_sec", "clip_duration_sec", "sample_rate",
    "pad_head_sec", "pad_tail_sec", "transcript",
    "word_before_switch", "word_after_switch",
    "language_before_switch", "language_after_switch",
    "switch_direction", "embedded_span_duration_sec", "duration_stratum",
    "stratum", "double_annotation", "seed", "clip_path", "textgrid_path",
    "source_audio_path", "left_reference_unit_index", "right_reference_unit_index",
]


def _transcripts(candidates: pd.DataFrame) -> dict[str, str]:
    """Reference transcript per utterance, rebuilt from its ordered units."""
    out: dict[str, str] = {}
    for utterance, group in candidates.groupby("utterance_id", sort=False):
        ordered = group.sort_values("reference_unit_index")
        pieces = [str(s) for s in ordered["surface"].tolist() if str(s)]
        out[str(utterance)] = " ".join(pieces)
    return out


def build_pack(candidates: pd.DataFrame, output: Path, *, seed: int,
               n_primary: int, n_double: int,
               write_audio: bool = True) -> dict[str, Any]:
    """Sample, cut, and write the pack.  Returns a report plus every path written."""
    universe, excluded = pack_mod.switch_universe(candidates)
    if not len(universe):
        raise pack_mod.AnnotationPackError("no eligible code-switch boundaries")
    selected = pack_mod.sample_pack(universe, seed=seed, n_primary=n_primary,
                                    n_double=n_double)
    transcripts = _transcripts(
        candidates[candidates["aligner_family"].astype(str)
                   == pack_mod.PROVISIONAL_FAMILY])

    output = Path(output)
    (output / PRIMARY_DIR).mkdir(parents=True, exist_ok=True)
    (output / DOUBLE_DIR).mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for position, item in enumerate(selected.to_dict("records")):
        source = Path(str(item["audio_path"]))
        try:
            if str(source) not in audio_cache:
                samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
                if getattr(samples, "ndim", 1) > 1:
                    samples = samples[:, 0]
                audio_cache[str(source)] = (samples, int(rate))
            samples, rate = audio_cache[str(source)]
        except Exception as error:                      # unreadable source audio
            failures.append({"utterance_id": str(item["utterance_id"]),
                             "switch_direction": str(item["switch_direction"]),
                             "reason": f"audio unreadable: {error}"})
            continue

        start, _ = pack_mod.clip_bounds(float(item["_provisional_center_sec"]))
        clip, pad_head, pad_tail = pack_mod.extract_clip(samples, rate, start)
        record = {
            **{key: item[key] for key in (
                "utterance_id", "conversation_id", "speaker_id",
                "switch_direction", "word_before_switch", "word_after_switch",
                "language_before_switch", "language_after_switch",
                "embedded_span_duration_sec", "duration_stratum", "stratum",
                "left_reference_unit_index", "right_reference_unit_index")},
            "speaker_id": str(item["speaker_id"]),
            "clip_start_offset_sec": float(start),
            "clip_duration_sec": float(pack_mod.CLIP_DURATION_SEC),
            "sample_rate": int(rate),
            "pad_head_sec": float(pad_head),
            "pad_tail_sec": float(pad_tail),
            "transcript": transcripts.get(str(item["utterance_id"]), ""),
            "double_annotation": bool(item["double_annotation"]),
            "seed": int(seed),
            "source_audio_path": str(source),
        }
        record["fingerprint"] = pack_mod.fingerprint(record)
        record["item_id"] = f"csb_{position + 1:04d}"
        folder = DOUBLE_DIR if record["double_annotation"] else PRIMARY_DIR
        clip_path = output / folder / f"{record['fingerprint']}.wav"
        grid_path = output / folder / f"{record['fingerprint']}.TextGrid"
        record["clip_path"] = str(clip_path.relative_to(output))
        record["textgrid_path"] = str(grid_path.relative_to(output))

        if write_audio:
            sf.write(str(clip_path), clip, rate, subtype="PCM_16")
            grid_path.write_text(pack_mod.textgrid_text(), encoding="utf-8")
            written += [str(clip_path), str(grid_path)]
        rows.append(record)

    frame = pd.DataFrame(rows)[MANIFEST_COLUMNS]
    csv_path, json_path = output / MANIFEST_CSV, output / MANIFEST_JSON
    frame.to_csv(csv_path, index=False)
    payload = {
        "schema_version": pack_mod.MANIFEST_SCHEMA,
        "diagnostic_only": True,
        "boundaries_populated": False,
        "source_role": pack_mod.SOURCE_ROLE,
        "provisional_boundary_family": pack_mod.PROVISIONAL_FAMILY,
        "provisional_boundary_use": "clip centring only; never written to a tier",
        "seed": int(seed),
        "requested_primary": int(n_primary),
        "requested_double": int(n_double),
        "clip_duration_sec": pack_mod.CLIP_DURATION_SEC,
        "tier_names": list(pack_mod.TIER_NAMES),
        "tercile_cuts_sec": [float(selected["tercile_lower_sec"].iloc[0]),
                             float(selected["tercile_upper_sec"].iloc[0])],
        "universe_boundaries": int(len(universe)),
        "universe_utterances": int(universe["utterance_id"].nunique()),
        "excluded_from_universe": excluded,
        "export_failures": failures,
        "items": frame.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    readme_path = output / README
    readme_path.write_text(
        pack_mod.ANNOTATOR_README.format(
            n_primary=int((~frame["double_annotation"]).sum()),
            n_double=int(frame["double_annotation"].sum())),
        encoding="utf-8")
    written += [str(csv_path), str(json_path), str(readme_path)]

    # Annotator-facing index: identity, language, and words only. Timing is
    # withheld so nothing beside the clips can be inverted into a boundary.
    for folder, is_double in ((PRIMARY_DIR, False), (DOUBLE_DIR, True)):
        subset = frame[frame["double_annotation"] == is_double]
        items_path = output / folder / "items.csv"
        subset[list(pack_mod.ANNOTATOR_COLUMNS)].to_csv(items_path, index=False)
        written.append(str(items_path))
    ambiguity_path = output / "ambiguity_notes.csv"
    ambiguity_path.write_text(",".join(pack_mod.AMBIGUITY_HEADER) + "\n",
                              encoding="utf-8")
    written.append(str(ambiguity_path))
    return {"report": payload, "frame": frame, "written": written,
            "universe": universe, "selected": selected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export code-switch boundary clips for human annotation")
    parser.add_argument("--config", default="lss/l1b_valid.yaml")
    parser.add_argument("--output", required=True,
                        help="pack directory; must be outside the artifacts root")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--primary", type=int, default=150)
    parser.add_argument("--double", type=int, default=40)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["experiment"]["output_root"])
    output = Path(args.output).resolve()
    try:
        output.relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("annotation pack must be written outside the artifacts root")

    candidates, verdict = manifest_mod.read_verified(
        root / "alignments/candidates_all.parquet", cfg=None, require_identity=False)
    if not verdict["ok"]:
        raise SystemExit(f"cached candidates are not authentic: {verdict}")

    result = build_pack(candidates, output, seed=args.seed,
                        n_primary=args.primary, n_double=args.double)
    print(json.dumps({"output": str(output),
                      "items": len(result["frame"]),
                      "written": len(result["written"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
