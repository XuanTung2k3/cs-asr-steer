# BASIS-A6 Full-Run Sharding Plan

This plan is recorded before any atlas execution. It is not an authorization to run it.

- Shard by model × evaluation panel, keeping one model resident per worker.
- Within a worker, sweep layers and all five rho values; do not create one job per cell.
- A6-TT performs baseline analysis and direction extraction once per utterance/decode mode, then reuses the cache across rho.
- Qwen `official_standard` reuses the greedy hypothesis only when the recorded hypothesis/config hashes match; both matrix rows remain enumerated.
- Submit at most two `sbatch` jobs concurrently, preferring MIG ≤40 GB after measured model preflight.
- Measured safe policy: one contiguous layer per shard. Split the 300-row
  D-dev-select panel into 2 chunks for fixed decoder shards and 3 chunks for
  Oracle-TT shards; smaller 220/50-row panels fit in one fixed chunk.
- Resume by the complete scientific key and reject duplicate or empty-provenance rows.

## Measured real-model acceptance jobs

| Job | Model | Work | Elapsed | Peak allocated / reserved VRAM | Result |
|---:|---|---|---:|---:|---|
| 53321 | Qwen3-ASR-1.7B | 4 decode modes + 4-row Oracle-TT + cache benchmark | 00:03:37 | 4.70 / 4.78 GB | PASS |
| 53326 | Whisper-large-v3 | 4 decode modes + 4-row Oracle-TT + cache benchmark | 00:04:37 | 4.98 / 5.34 GB | PASS |

Measured per-utterance decode time is 1.067 s for Qwen and 1.353 s for
Whisper. Full-run estimates and the derived layer-block policy are in
`RUNTIME_ESTIMATES.json`. Persisted TT vectors use float32 storage with
canonical float64 hashes; hidden states are not persisted.

## Measured construction jobs

These are construction-only measurements and do not authorize atlas execution:

| Job | Model | Work | Elapsed | GPU | Result |
|---:|---|---|---:|---|---|
| 53312 | Whisper-large-v3 | CS Conditioning-CS + ASCEND 320 entries | 00:07:44 | H100 MIG 3g.40GB | PASS |
| 53315 | Qwen3-ASR-1.7B | CS Conditioning-CS + ASCEND 264 entries | 00:06:27 | H100 MIG 3g.40GB | PASS |

Slurm accounting did not expose MaxRSS for the construction jobs, but real
model decode VRAM and elapsed measurements are recorded above. The current
filesystem has approximately 816 GB free; the planned float32 TT cache and
conservative diagnostics estimate are approximately 4.47 GB. No atlas job is
authorized by this document.

Current state: **MEASURED / FROZEN FOR REVIEW**. Full atlas execution remains
unstarted.
