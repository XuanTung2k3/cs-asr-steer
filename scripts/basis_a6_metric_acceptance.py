#!/usr/bin/env python3
"""Stage-4 CPU metric/segmentation acceptance on ASCEND fixtures and real rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from csasr.basis_a6.data import load_ascend_split
from csasr.data.language_tags import tag_units
from csasr.data.normalize import normalize_and_segment
from csasr.evaluation.canonical import corpus_metrics


def main() -> int:
    out = REPO / "results/basis_a6_expanded/ascend"
    rows = load_ascend_split(REPO / "data/external/ASCEND/dataset", "validation")
    refs = ["你好 hello", "I like 北京"]
    fixture = corpus_metrics(refs, refs)
    real = rows[: min(8, len(rows))]
    real_metrics = corpus_metrics([x.transcript_raw for x in real], [x.transcript_raw for x in real])
    trace_rows = []
    for item in ([rows[0]] if rows else []) + real[:2]:
        norm, units = normalize_and_segment(item.transcript_raw)
        trace_rows.append({"utterance_id": item.utterance_id, "raw": item.transcript_raw,
                           "normalized": norm, "units": [u.surface for u in units],
                           "tags": tag_units(units), "en_units": item.en_units, "zh_units": item.zh_units,
                           "eligible": item.cs_eligible})
    checks = {"fixture_mer_zero": fixture["mer"] == 0.0, "fixture_pier_zero": fixture["pier"] == 0.0,
              "fixture_en_wer_zero": fixture["en_wer"] == 0.0, "fixture_matrix_cer_zero": fixture["zh_cer"] == 0.0,
              "real_rows_nonempty": bool(real), "real_identity_mer_zero": real_metrics["mer"] == 0.0,
              "trace_has_en_zh": any("EN" in x["tags"] and "ZH" in x["tags"] for x in trace_rows)}
    payload = {"schema_version": "basis_a6_metric_acceptance_v1", "status": "PASS" if all(checks.values()) else "FAIL",
               "metrics": {"fixture": fixture, "real_identity": real_metrics}, "checks": checks,
               "test_usage": 0, "segmentation_trace": "ASCEND_SEGMENTATION_TRACE.json"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ASCEND_METRIC_ACCEPTANCE.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (out / "ASCEND_SEGMENTATION_TRACE.json").write_text(json.dumps(trace_rows, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
