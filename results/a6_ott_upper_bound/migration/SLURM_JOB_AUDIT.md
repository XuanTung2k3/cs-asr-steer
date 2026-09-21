# A6-OTT Migration — Slurm Job Audit

Recorded: 2026-09-21. Session task: supersede the large BASIS-A6 (70,080-cell A6-F/A6-TT
atlas) with the compact **A6-OTT** (Oracle Per-Sample Test-Time Steering upper-bound search).
This file records the pre-cancellation state of every active job, the classification, and the
preserved artifacts. **No new scientific GPU jobs were submitted in this session.**

Repo HEAD at audit: `b7cf331` (branch `learned-expansion`). Only user `tungnx` jobs, all named
`basis-a6*`; **no unrelated or unknown jobs were present**.

## Active jobs at audit (pre-cancel snapshot)

| Job | Name | Partition | State | RunTime | Command | Log |
|---|---|---|---|---|---|---|
| **53354** | basis-a6-waves | main | RUNNING | 02:27:44 | `slurm/basis_a6_wave_coordinator.sbatch` | `results/basis_a6_expanded/slurm/waves-53354.out` |
| **53355** (array base) | basis-a6-atlas | mig | mixed | — | `slurm/basis_a6_atlas_array.sbatch` | `results/basis_a6_expanded/slurm/atlas-53355_*.out` |
| 53355_0 | basis-a6-atlas | mig | COMPLETED | 00:59:17 | " | atlas-53355_0.out |
| 53355_1 | basis-a6-atlas | mig | COMPLETED | 01:00:48 | " | atlas-53355_1.out |
| 53355_2 | basis-a6-atlas | mig | COMPLETED | 00:36:08 | " | atlas-53355_2.out |
| 53355_3 | basis-a6-atlas | mig | COMPLETED | 01:03:28 | " | atlas-53355_3.out |
| 53355_4 (53380) | basis-a6-atlas | mig | RUNNING | ~00:53 | " | atlas-53355_4.out |
| 53355_5 (53384) | basis-a6-atlas | mig | RUNNING | ~00:24 | " | atlas-53355_5.out |
| 53355_[6-9%2] | basis-a6-atlas | mig | PENDING (JobArrayTaskLimit) | 00:00 | " | — |

WorkDir for all: `/home/tungnx/cs-asr-steer`.

## Classification

| Job | Class | Rationale |
|---|---|---|
| 53354 (waves coordinator) | **CANCEL_SUPERSEDED_A6** | Orchestrates the full 70,080-cell A6-F/A6-TT atlas (2,784 shards). Keeps launching new atlas waves; A6-OTT replaces this atlas. |
| 53355 atlas array (running 4,5 + pending 6-9) | **CANCEL_SUPERSEDED_A6** | Each task decodes shards of the 70,080-cell matrix — overwhelmingly **fixed corpus-level** cells (task-4 log: `fixed whisper ascend … encoder … unique_minus_shared rho=…`) that A6-OTT does not need. |
| (none) | UNRELATED_USER_JOB | No non-A6 jobs present. |
| (none) | UNKNOWN | None. |

**No KEEP_NEW_A6_OTT jobs**: no currently-active job is executing the compact A6-OTT protocol.

## Progress at cancel (from `FULL_RUN_EXECUTION_STATUS.json`)

- Status `IN_PROGRESS`; `implementation_commit b7cf331`.
- Completed rows: **a6_f = 1, a6_tt = 0, total = 1** of expected 70,080. The atlas had barely
  started decoding result rows; the running tasks were on fixed corpus-level cells.
- `shard_count = 2784`, `max_concurrent_gpu_jobs = 2`.

## Preserved artifacts (ON DISK — NOT DELETED)

All under `results/basis_a6_expanded/` (kept intact per policy §2; superseded for execution, not
erased). Cancellation does not touch on-disk artifacts.

- **Baselines (complete, 16 files):** `baselines/{whisper,qwen3_asr_1p7b}/{cs_dialogue_dev_select,
  ascend_eval,seame_dev_man,seame_dev_sge}/{greedy,official_standard}.json` — unsteered
  free-decode; directly reusable by A6-OTT.
- **Oracle-TT per-sample caches + runs (partial):**
  `oracle_tt/cache/whisper/cs_dialogue_dev_select/greedy/encoder/{L00,L01}/` (DIRECTION_CACHE.json
  + per-utterance `vectors/*.npy`, 1,190 files) and
  `oracle_tt/runs/whisper/cs_dialogue_dev_select/greedy/encoder/{L00,L01}.{jsonl,manifest.json}`.
  Methods present: `add_unique`, `minus_shared`, `raw`, `unique_minus_shared` (encoder). These
  are the only completed oracle-TT layers.
- **ASCEND manifests:** `ascend/ASCEND_{CONSTRUCT,EVAL}_MANIFEST.json`, eval alignment/eligibility
  audits, role freeze, segmentation trace, metric acceptance.
- **Fixed corpus-level direction inventories/materializations** (superseded for A6-OTT execution;
  retained as evidence): `fixed/{cs_dialogue,ascend}/…`, `fixed/geometry/…`.
- **Preflight + GPU acceptance + runtime estimates:** `preflight/…`.
- **Manifests:** `manifests/{A6_IMPLEMENTATION_MANIFEST,A6_MATRIX_ENUMERATION.csv,
  FULL_RUN_EXECUTION_STATUS,FULL_RUN_SHARDS,DIRECTION_AUDIT,FIXED_DIRECTION_INVENTORY}.json`.
- **Slurm logs:** `results/basis_a6_expanded/slurm/*.out` (kept).

No partial output required preservation *action* beyond leaving `results/basis_a6_expanded/`
untouched — all completed cells were already flushed to disk before cancellation; only in-flight
(uncounted) cells in tasks 53355_4/_5 were discarded by scancel.

## Cancellation actions (this session)

Order: cancel the coordinator first (stops new waves), then the atlas array (running + pending).

```
scancel 53354        # basis-a6-waves coordinator
scancel 53355        # basis-a6-atlas array (tasks 4,5 running; 6-9 pending)
```

Post-cancel verification recorded below (appended after execution).

## Post-cancel state

Executed 2026-09-21T08:45:37 (`scancel 53354`, `scancel 53355`). Result:

- `53354` (waves coordinator): **CANCELLED**.
- `53355_4`, `53355_5`, `53355_[6-9%2]`: **CANCELLED** (were running/pending).
- `53355_0..3`: already **COMPLETED** before cancel (untouched).
- `squeue -u tungnx`: **0 live jobs**.

On-disk artifacts verified intact after cancel: baselines **16** files, oracle_tt cache+runs
**1,190** files; `results/basis_a6_expanded/` subtree unchanged. Nothing was deleted.

