#!/usr/bin/env python3
"""Create the frozen ASCEND-validation oracle spans used by A6 evaluation.

Only the validation split named by ASCEND_EVAL_MANIFEST is loaded.  The
alignment is the project's canonical independent CTC unit aligner; the output
contains locations only and is never used to supply a target transcript to a
decoder analysis pass.
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
    import soundfile as sf
    if isinstance(value, (bytes, bytearray)):
        audio, sr = sf.read(io.BytesIO(value), dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False), 16000
    return load_audio(str(value), 16000), 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/external/ASCEND/dataset")
    ap.add_argument("--manifest", default="results/basis_a6_expanded/ascend/ASCEND_EVAL_MANIFEST.json")
    ap.add_argument("--output", default="results/basis_a6_expanded/ascend/ASCEND_EVAL_ALIGNMENT.json")
    args = ap.parse_args()
    wanted = set(json.loads(Path(args.manifest).read_text())["utterance_ids"])
    rows = [r for r in load_ascend_split(args.dataset_dir, "validation") if r.utterance_id in wanted]
    cfg = load_config("lss/align.yaml")
    cfg["model"] = {"device": "cpu"}
    aligner = load_ctc_aligner(cfg)
    aligned, failures = {}, []
    for n, row in enumerate(rows, 1):
        try:
            audio, sr = _audio_array(row.audio)
            spans = align_units(aligner, audio, [u["surface"] for u in row.units], sr)
            by_unit = {int(s["unit_id"]): s for s in spans}
            content = []
            for i, unit in enumerate(row.units):
                if unit["language_tag"] not in {"EN", "ZH"}:
                    continue
                s = by_unit.get(i)
                if s is not None and float(s["end_sec"]) > float(s["start_sec"]):
                    content.append({"unit_index": i, "language_tag": unit["language_tag"],
                                    "surface": unit["surface"], "start_sec": float(s["start_sec"]),
                                    "end_sec": float(s["end_sec"])})
            if not any(x["language_tag"] == "EN" for x in content) or not any(x["language_tag"] == "ZH" for x in content):
                raise RuntimeError("aligned content is not mixed")
            aligned[row.utterance_id] = {"duration_sec": row.duration_sec, "units": content}
        except Exception as exc:
            failures.append({"utterance_id": row.utterance_id, "error": repr(exc)})
        if n % 10 == 0:
            print(f"ASCEND validation CTC alignment {n}/{len(rows)}", flush=True)
    body = {"schema_version": "basis_a6_ascend_eval_alignment_v1",
            "source_split": "validation", "manifest": str(Path(args.manifest)),
            "alignment_method": "csasr.data.ctc_alignment.align_units",
            "aligner": "/mnt/data/tungnx/models/mms-fa", "rows": aligned,
            "failures": failures}
    body["fingerprint"] = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    body["status"] = "PASS" if not failures and len(aligned) == len(rows) else "BLOCKED"
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": body["status"], "rows": len(aligned), "failures": len(failures), "fingerprint": body["fingerprint"]}, indent=2))
    return 0 if body["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
