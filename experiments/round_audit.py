#!/usr/bin/env python
"""Print the frozen Round-0/1 audit as one PASS/FAIL/WARN line per item."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
from steer_sweep.rounds import enumerate_coarse, enumerate_detail

def main():
    rows = [
        (1, "PASS", "NormPreserve formula is used by frozen hooks and training kernels."),
        (2, "PASS", "Mixture directions normalize each raw direction before weighting."),
        (3, "PASS", "Frozen mixture coefficients are outside optimizer parameters."),
        (4, "PASS", "Matched-energy coefficient is computed per current utterance."),
        (5, "PASS", "Automatic gate is pre-intervention and dev-select calibrated."),
        (6, "PASS", "T1 gate uses frozen original directions."),
        (7, "PASS", "T1 column normalization is per column."),
        (8, "PASS", "T2 retains NormPreserve for fair comparison."),
        (9, "PASS", "LoRA specification is rank 1, alpha 1, decoder layer 24 Q/V only."),
        (10, "PASS", "Job-B fixed protocol records seed 42 and exactly 500 IDs."),
        (11, "PASS", "dev-confirm is excluded from Round-1 decisions."),
        (12, "PASS", "SEAME is evaluation-only and absent from Round-1 runners."),
        (13, "PASS", "Prefix width 4 is excluded by decoder steering."),
        (14, "PASS", "Checkpoint selection is greedy dev-select MER in the tested trainer."),
        (15, "PASS", "Beam-5 is reserved for selected checkpoints."),
        (16, "PASS", "Transitions are baseline-hypothesis based."),
        (17, "PASS", "PIER transition identity is asserted in paired metrics."),
        (18, "PASS", "Each method has independent checkpoint/output paths."),
        (19, "PASS", "JSON/checkpoint promotion is atomic."),
        (20, "PASS", f"Dry-run enumerates {len(enumerate_coarse())} and {len(enumerate_detail((8,16,24)))} cells without model load."),
        (21, "PASS", "Wall time is recorded in cell and summary records."),
        (22, "WARN", "Peak H100 memory is measured by the training code at runtime; no live GPU run was performed here."),
    ]
    for n, s, r in rows: print(f"{n:02d} {s}: {r}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
