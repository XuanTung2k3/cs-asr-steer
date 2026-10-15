# P2-2 amendment r2 — per-job preflight output paths

**Status.** This is an engineering repair, recorded in `p2_2/freeze_r2.json`, which chains to `freeze_r1.json` and
`freeze.json`.

**Unchanged:** amendment r1 and every scientific element (construction, targets, gates, thresholds, loss, training,
evaluation, rules).

## 1. Failure

The amendment-r1 preflight rerun (job 59353) would have written its rollouts and coverage report to the same
write-once paths as the first, blocked preflight (job 59346):
- `p2_2/preflight_rollouts/{B0,CE42,KL42}{,_events}.jsonl`;
- `p2_2/coverage_train.json`.

It would therefore have crashed at its first write. I cancelled it after 15 seconds, before it wrote anything. The
first preflight's artifacts are preserved and unmodified.

## 2. Repair (`reft/p2_2_train.py`)

- The preflight writes its rollouts and events to `p2_2/preflight_rollouts/<job>/`, and its coverage report to
  `p2_2/coverage_train_<job>.json`.
- The preflight record stores `rollouts_dir`.
- `run_train` reads the B0 training rollout from the `rollouts_dir` of the newest preflight, which must be
  `READY_FOR_P2_2_TRAINING`.
- The freeze chain gains `freeze-r2`, and `slurm/reft_p2_2_cpu.sbatch` gains the step.

## 3. Rerun

The CPU tests, then `freeze-r2`, then the complete P2-2A GPU preflight.
