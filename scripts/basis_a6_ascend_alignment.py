#!/usr/bin/env python3
"""Prepare train-only ASCEND unit spans for the frozen A6 construction pass.

ASCEND supplies transcript language units but no word timestamps. The project
canonical independent CTC aligner is used to map those same canonical units to
audio time; no evaluation or test split is read.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from csasr.basis_a6.data import load_ascend_split
from csasr.data.ctc_alignment import align_units, load_ctc_aligner
from csasr.models.qwen3_asr import load_audio
from csasr.utils.config import load_config


def _audio_array(value):
    if isinstance(value, (bytes, bytearray)):
        import soundfile as sf
        audio, sample_rate = sf.read(io.BytesIO(value), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False), 16000
    return load_audio(str(value), 16000), 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/external/ASCEND/dataset")
    ap.add_argument("--manifest", default="results/basis_a6_expanded/ascend/ASCEND_CONSTRUCT_MANIFEST.json")
    ap.add_argument("--output", default="results/basis_a6_expanded/ascend/ASCEND_CONSTRUCT_ALIGNMENT.json")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    wanted = set(manifest["utterance_ids"])
    rows = [row for row in load_ascend_split(args.dataset_dir, "train") if row.utterance_id in wanted]
    cfg = load_config("lss/align.yaml")
    cfg["model"] = {"device": "cpu"}
    aligner = load_ctc_aligner(cfg)
    aligned = {}
    failures = []
    for number, row in enumerate(rows, 1):
        try:
            audio, sample_rate = _audio_array(row.audio)
            spans = align_units(aligner, audio, [u["surface"] for u in row.units], sample_rate)
            by_unit = {int(span["unit_id"]): span for span in spans}
            content = []
            for index, unit in enumerate(row.units):
                if unit["language_tag"] not in {"EN", "ZH"}:
                    continue
                span = by_unit.get(index)
                if span is None or span["end_sec"] <= span["start_sec"]:
                    continue
                content.append({"unit_index": index, "language_tag": unit["language_tag"],
                                "surface": unit["surface"], "start_sec": span["start_sec"],
                                "end_sec": span["end_sec"]})
            if not any(x["language_tag"] == "EN" for x in content) or not any(x["language_tag"] == "ZH" for x in content):
                failures.append({"utterance_id": row.utterance_id, "reason": "aligned_content_not_mixed"})
                continue
            aligned[row.utterance_id] = {"duration_sec": row.duration_sec, "units": content}
        except Exception as exc:
            failures.append({"utterance_id": row.utterance_id, "reason": repr(exc)})
        if number % 10 == 0:
            print(f"ASCEND CTC alignment {number}/{len(rows)}", flush=True)
    body = {"manifest_fingerprint": manifest["fingerprint"], "aligner": "/mnt/data/tungnx/models/mms-fa",
            "alignment_method": "csasr.data.ctc_alignment.align_units", "source_split": "train",
            "rows": aligned, "failures": failures}
    body["fingerprint"] = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["status"] = "PASS" if not failures and len(aligned) == len(rows) else "BLOCKED"
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "rows": len(aligned), "failures": len(failures), "fingerprint": body["fingerprint"]}, indent=2))
    return 0 if body["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
