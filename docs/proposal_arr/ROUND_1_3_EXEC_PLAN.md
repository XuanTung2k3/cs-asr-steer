# Round 1–3 execution plan

Status: implementation and validation in progress. Round 1 is the only round
eligible for Slurm submission in this task. Round 2 and Round 3 are code-only.

## Milestones

1. Round-0 audit and metric/test infrastructure — implemented.
2. Shared round configuration, deterministic cell manifests, atomic outputs,
   lazy backend boundary and explicit Round-2 lock — implemented.
3. Round-1 Job A frozen runner and geometry primitives — implemented; GPU
   execution is scheduler-only.
4. Round-1 Job B fixed-protocol runner — implemented by adapting the existing,
   tested representation trainer; output root is isolated per job.
5. Tests, dry-runs and two-utterance smoke checks — in progress.
6. Submit exactly two independent Round-1 Slurm jobs with `sbatch --parsable` —
   accepted as Job A `46528` and Job B `46529`.

## Frozen choices

- D-dev-select is the only Round-1 selection/calibration split.
- D-dev-confirm, SEAME and D-test-lock are not consumed by Round-1 runners.
- Coarse frozen manifest is 7 × 3 × 2 × 2 = 84 cells.
- Detailed frozen manifest is 3 × 7 × 2 × 3 = 126 cells.
- Job B records seed 42, 500 training utterances, three epochs, micro-batch
  8, accumulation 2, AdamW `(0.9, 0.99)`, epsilon `1e-6`, weight decay 0,
  gradient clip 1.0 and greedy dev-select MER selection.
- Round-1 Slurm requests inherit the working cluster convention: `main`, one
  H100, 8 CPUs, 96G RAM, and a three-hour limit.

## Validation commands

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_round1_3_infrastructure.py
python experiments/round1_frozen.py --dry-run
python experiments/round1_training.py --dry-run
python experiments/round_audit.py
```

The two dry-runs printed 84 and 126 for Job A and did not load Whisper or Qwen.
The repository-wide test suite passed. GPU smoke/execution is performed only
inside Slurm.

## Blockers and decisions

- The repository has no explicit `role_D-test-lock.parquet` manifest. It is
  therefore recorded as a missing artifact for Round 2 and is not silently
  substituted with the existing D-test material.
- Qwen weights are not initialized or downloaded. The backend is lazy and the
  tests use a mock backend.
- Existing prior `results/job_a` and `results/job_b` are preserved. New jobs
  write to `results/round1/job_a` and `results/round1/job_b`.

## Submission record

Accepted 2026-08-26 UTC:

- Job A: `46528`, RUNNING at acceptance,
  `/home/tungnx/cs-asr-steer/logs/cs_asr_round1_frozen_46528.log`
- Job B: `46529`, PENDING for resources at acceptance,
  `/home/tungnx/cs-asr-steer/logs/cs_asr_round1_training_46529.log`
- Both had `Dependency=(null)` and were submitted consecutively with
  `sbatch --parsable`; both subsequently failed during configuration loading
  before model initialization. Replacement jobs `46544` and `46545` were then
  submitted independently; Job B completed, while Job A failed its strict
  paired PIER identity check. No Round-2 or Round-3 job was submitted.
- Repair: runners now resolve `data_config` to the existing nested model/data
  configuration. Replacement independent submissions accepted:
  Job A `46544` and Job B `46545`, both pending resources at acceptance,
  with no dependency.
