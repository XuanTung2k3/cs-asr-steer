"""Frozen evaluation-panel loading for the executable BASIS-A6 atlas.

This module is intentionally the only place where the four panel formats are
adapted to the model runners.  It never reads ASCEND ``test`` and it refuses
to construct a panel without a saved oracle alignment for ASCEND.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ASCEND_ROOT = REPO / "data/external/ASCEND"
EXPANDED = REPO / "results/basis_a6_expanded"


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode()).hexdigest()


def _seame_audio_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path("/mnt/data/tungnx/seame_accent_stress_pilot") / "audio/segments" / str(row["utterance_id"]) / "segments.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        silence = float(payload.get("silence_seconds", 0.08))
        cursor, spans = 0.0, []
        for item in payload.get("segments", []):
            duration = float(item["audio"]["duration"])
            if bool(item.get("target", False)):
                spans.append({"start_sec": cursor, "end_sec": cursor + duration})
            cursor += duration + silence
        if spans:
            return spans
    bounds = row.get("target_span_indices", [])
    return ([{"start_sec": float(bounds[0]) / 100.0,
              "end_sec": float(bounds[1]) / 100.0}]
            if len(bounds) == 2 else [])


def _char_units(norm: str, units: list[Any], needle: str, ordinal: int) -> list[int]:
    folded = norm.casefold()
    needle = needle.replace("*", "").strip().casefold()
    starts, at = [], 0
    while needle:
        at = folded.find(needle, at)
        if at < 0:
            break
        starts.append(at)
        at += max(1, len(needle))
    if not starts:
        return []
    lo = starts[min(ordinal, len(starts) - 1)]
    hi = lo + len(needle)
    return [i for i, u in enumerate(units) if u.char_end > lo and u.char_start < hi]


def _seame_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    from ..data.normalize import normalize_and_segment
    norm, units = normalize_and_segment(str(row["reference"]))
    bounds = row.get("target_span_indices", [])
    runs = [r for r in row.get("language_runs", []) if str(r.get("kind")) == "en"]
    ordinal = next((i for i, r in enumerate(runs)
                    if len(bounds) == 2 and int(r.get("start", -1)) == int(bounds[0])
                    and int(r.get("end", -1)) == int(bounds[1])), 0)
    indices = _char_units(norm, units, str(row.get("target_english_text", "")), ordinal)
    spans = _seame_audio_spans(row)
    if not indices or not spans:
        raise RuntimeError(f"SEAME local alignment is empty for {row['utterance_id']}")
    return [{**spans[0], "unit_indices": indices, "all_correct": True,
             "utterance_id": str(row["utterance_id"])}]


def _materialize_audio(value: Any, sample_rate: int | None, uid: str) -> str:
    if isinstance(value, (str, Path)) and Path(value).is_file():
        return str(value)
    import soundfile as sf
    root = ASCEND_ROOT / "runtime_audio"
    root.mkdir(parents=True, exist_ok=True)
    out = root / (uid.replace("/", "_") + ".wav")
    if not out.is_file():
        if isinstance(value, (bytes, bytearray)):
            audio, sr = sf.read(io.BytesIO(value), dtype="float32", always_2d=False)
        else:
            audio, sr = np.asarray(value, dtype=np.float32), int(sample_rate or 16000)
        if np.asarray(audio).ndim > 1:
            audio = np.asarray(audio).mean(axis=1)
        sf.write(out, np.asarray(audio, dtype=np.float32), int(sr or sample_rate or 16000))
    return str(out)


def _ascend_rows(*, require_alignment: bool = True) -> dict[str, dict[str, Any]]:
    from .data import load_ascend_split
    manifest = json.loads((EXPANDED / "ascend/ASCEND_EVAL_MANIFEST.json").read_text())
    wanted = set(map(str, manifest["utterance_ids"]))
    rows = {}
    for item in load_ascend_split(ASCEND_ROOT / "dataset", "validation"):
        if item.utterance_id not in wanted:
            continue
        rows[item.utterance_id] = {
            "utterance_id": item.utterance_id,
            "audio_path": _materialize_audio(item.audio, item.sample_rate, item.utterance_id),
            "reference": item.transcript_raw,
            "duration_sec": float(item.duration_sec or 0.0),
            "dialogue_id": f"ASCEND:{item.session_id}",
            "data_role": "ASCEND-eval",
        }
    if set(rows) != wanted:
        raise RuntimeError(f"ASCEND eval materialization mismatch: {len(rows)} vs {len(wanted)}")
    alignment = EXPANDED / "ascend/ASCEND_EVAL_ALIGNMENT.json"
    if not alignment.is_file() and require_alignment:
        raise RuntimeError("ASCEND_EVAL_ALIGNMENT.json is required before scientific execution")
    if not alignment.is_file():
        for row in rows.values():
            row["oracle_spans"] = []
        return rows
    payload = json.loads(alignment.read_text())
    for uid, row in rows.items():
        units = payload.get("rows", {}).get(uid, {}).get("units", [])
        eng = [i for i, u in enumerate(units) if u.get("language_tag") == "EN"]
        if not eng:
            raise RuntimeError(f"ASCEND eval alignment has no English span for {uid}")
        # Preserve contiguous English runs as separate oracle spans.
        groups: list[list[int]] = []
        for i in eng:
            if not groups or i != groups[-1][-1] + 1:
                groups.append([i])
            else:
                groups[-1].append(i)
        row["oracle_spans"] = [{"start_sec": float(units[g[0]]["start_sec"]),
                                "end_sec": float(units[g[-1]]["end_sec"]),
                                "unit_indices": g, "all_correct": True,
                                "utterance_id": uid} for g in groups]
    return rows


def load_panel(dataset: str, *, require_alignment: bool = True) -> tuple[list[dict[str, Any]], str]:
    """Load one frozen panel and attach the accepted oracle-local spans."""
    if dataset == "ascend_eval":
        rows = list(_ascend_rows(require_alignment=require_alignment).values())
        fp = json.loads((EXPANDED / "ascend/ASCEND_EVAL_MANIFEST.json").read_text())["fingerprint"]
        return rows, fp
    from experiments.basis_a4_qwen import _load_cs_eval_spans
    names = {"cs_dialogue_dev_select": "cs_dialogue_300",
             "seame_dev_man": "seame_dev_man_300",
             "seame_dev_sge": "seame_dev_sge_300"}
    path = REPO / "results/basis_a4/panels" / f"{names[dataset]}.json"
    payload = json.loads(path.read_text())
    rows = [dict(row) for row in payload["rows"]]
    if dataset == "cs_dialogue_dev_select":
        spans = _load_cs_eval_spans()
        for row in rows:
            # The frozen D-dev-select panel is the full 300-row evaluation
            # panel.  Rows without an accepted A4 alignment remain in the
            # panel with an empty local mask; they are never silently dropped
            # or replaced by another utterance.
            row["oracle_spans"] = spans.get(str(row["utterance_id"]), [])
    else:
        for row in rows:
            row["oracle_spans"] = _seame_spans(row)
    return rows, payload["fingerprint"]
