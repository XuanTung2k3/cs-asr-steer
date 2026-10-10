#!/usr/bin/env python3
"""Finalize measured A6 runtime estimates and conservative shard sizing."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a6_expanded/preflight"
PANELS = {"cs_dialogue_dev_select": 300, "seame_dev_man": 50,
          "seame_dev_sge": 50, "ascend_eval": 220}
SITES = {"whisper": 320, "qwen3_asr_1p7b": 264}
JOB_ELAPSED = {"whisper": 277, "qwen3_asr_1p7b": 217}  # Slurm 53326/53321


def main() -> int:
    out = {"schema_version": "basis_a6_runtime_estimates_v1", "status": "MEASURED",
           "atlas_execution": "NOT_STARTED", "models": {}, "policy": {
               "target_job_hours": 2.5, "hard_job_hours": 3.0,
               "max_concurrent_gpu_jobs": 2, "preferred_gpu": "H100 MIG 3g.40GB"}}
    for model in SITES:
        b = json.loads((OUT / f"REAL_BENCHMARK_{model}.json").read_text())
        tt_path = OUT / f"REAL_ORACLE_TT_{model}.json"
        tt = json.loads(tt_path.read_text())
        dim = 1280 if model == "whisper" else 2048
        b["hidden_state_temporary_bytes"] = max(
            [int((a.get("n_A", 0) + a.get("n_B", 0)) * dim * 8)
             for row in tt.get("rows", []) for a in row.get("analysis", [])] or [0])
        b["benchmark_finalized_from"] = str(tt_path.relative_to(REPO))
        (OUT / f"REAL_BENCHMARK_{model}.json").write_text(json.dumps(b, indent=2) + "\n")
        sec = float(b["mean_call_sec"])
        site = SITES[model]
        fixed_one_source = sec * sum(PANELS.values()) * site * 2 * 5 / 3600.0
        fixed_two_source = fixed_one_source * 2
        fixed_call_est = sec * 72 + float(b["model_load_sec"])
        tt_factor = max(1.0, JOB_ELAPSED[model] / max(fixed_call_est, 1.0))
        tt = fixed_one_source * tt_factor
        hidden = int(b.get("hidden_state_temporary_bytes", 0))
        out["models"][model] = {
            "job_id": b["job_id"], "model_load_sec": b["model_load_sec"],
            "seconds_per_utterance": sec, "peak_allocated_vram_bytes": b["peak_allocated_vram_bytes"],
            "peak_reserved_vram_bytes": b["peak_reserved_vram_bytes"],
            "direction_cache_bytes_observed": b["direction_cache_bytes"],
            "hidden_state_temporary_bytes": hidden,
            "measured_end_to_end_job_sec": JOB_ELAPSED[model],
            "measured_tt_overhead_factor": tt_factor,
            "estimated_a6_f_cs_source_hours": fixed_one_source,
            "estimated_a6_f_ascend_source_hours": fixed_one_source,
            "estimated_a6_f_both_sources_hours": fixed_two_source,
            "estimated_a6_tt_hours": tt,
            "decode_modes": {"greedy": "measured", "official_standard": "measured"},
        }
    out["shard_policy"] = {
        "whisper": {"fixed_encoder": {"contiguous_layers": 1, "large_panel_chunks": 1},
                     "fixed_decoder": {"contiguous_layers": 1, "large_panel_chunks": 2},
                     "oracle_tt_encoder": {"contiguous_layers": 1, "large_panel_chunks": 3},
                     "oracle_tt_decoder": {"contiguous_layers": 1, "large_panel_chunks": 3}},
        "qwen3_asr_1p7b": {"fixed_encoder": {"contiguous_layers": 1, "large_panel_chunks": 1},
                            "fixed_decoder": {"contiguous_layers": 1, "large_panel_chunks": 2},
                            "oracle_tt_encoder": {"contiguous_layers": 1, "large_panel_chunks": 3},
                            "oracle_tt_decoder": {"contiguous_layers": 1, "large_panel_chunks": 3}},
        "inside_shard": "model resident; sweep methods and rho={0.5,1,2,4,6}; split only large panels when needed",
        "authorization": "preflight only; atlas not started",
    }
    (OUT / "RUNTIME_ESTIMATES.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
