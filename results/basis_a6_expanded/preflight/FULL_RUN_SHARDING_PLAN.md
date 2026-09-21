# BASIS-A6 Full-Run Sharding Plan

This plan is recorded before any atlas execution. It is not an authorization to run it.

- Shard by model × evaluation panel, keeping one model resident per worker.
- Within a worker, sweep layers and all five rho values; do not create one job per cell.
- A6-TT performs baseline analysis and direction extraction once per utterance/decode mode, then reuses the cache across rho.
- Qwen `official_standard` reuses the greedy hypothesis only when the recorded hypothesis/config hashes match; both matrix rows remain enumerated.
- Submit at most two `sbatch` jobs concurrently, preferring MIG ≤40 GB after measured model preflight.
- Choose contiguous layer blocks only from measured preflight memory/runtime; no block size is authorized yet.
- Resume by the complete scientific key and reject duplicate or empty-provenance rows.

## Measured construction jobs

These are construction-only measurements and do not authorize atlas execution:

| Job | Model | Work | Elapsed | GPU | Result |
|---:|---|---|---:|---|---|
| 53312 | Whisper-large-v3 | CS Conditioning-CS + ASCEND 320 entries | 00:07:44 | H100 MIG 3g.40GB | PASS |
| 53315 | Qwen3-ASR-1.7B | CS Conditioning-CS + ASCEND 264 entries | 00:06:27 | H100 MIG 3g.40GB | PASS |

Slurm accounting did not expose MaxRSS for these completed jobs, so no VRAM
estimate is inferred from them. Decode walltime/utterance, peak VRAM, cache
size, disk size, and safe contiguous layer blocks remain unmeasured and must be
obtained from representative real A6-F and A6-TT runners before submission.

Current state: **BLOCKED** pending model-resident A6-F/A6-TT decode benchmark,
end-to-end greedy/official-standard acceptance, and Oracle-TT causal acceptance.
