# Round 1–3 implementation status

Updated: 2026-08-26 UTC.

## Implemented

- Shared round constants and deterministic manifests in `steer_sweep/rounds.py`.
- Normalized direction mixtures, safe near-degenerate handling and atomic JSON.
- NormPreserve option in shared hooks and the existing frozen/training kernels.
- Paired MER/PIER metrics with substitutions, deletions, insertions, baseline
  transitions and unique-utterance outside-edit aggregation.
- Teacher-forced loss/top-1/gold probability-margin-rank diagnostic.
- Counterfactual state names `base_00`, `natural_10`, `prompt_01`,
  `combined_11`; orthogonal axes with a collinearity fallback; synergy,
  response, downstream-interaction and joint-PCA primitives; stable record
  schema.
- Lazy Qwen backend and guarded Round-3 runners; no Qwen initialization.
- Explicit Round-2 lock serializer and guarded transfer/combination runners.
- Round-1 config files, independent Slurm scripts and resumable output paths.

## Validation record

- Focused tests: passed (`tests/test_round1_3_infrastructure.py`, existing hook
  and PIER tests).
- Full repository test suite: passed (100%, warnings only).
- `git diff --check`, Python compilation and Slurm shell syntax: passed.
- Job A dry-run: passed; coarse 84, detailed 126.
- Job B dry-run: passed; fixed settings printed and no model load.
- Counterfactual CPU primitives: passed in focused tests.
- H100 peak memory: not measured in this login-node validation; Job B retains
  runtime peak-memory instrumentation. The fixed BF16 micro-batch is 8 with
  accumulation 2, under the requested effective batch 16.

## Not run/submitted

- Round 2: implemented, not submitted; lock is explicit and D-test-lock is
  currently a missing artifact.
- Round 3: implemented with lazy/mock loading, not initialized and not
  submitted.
- Round-1 Job A `46528`: FAILED after 7 seconds, exit code 1.
- Round-1 Job B `46529`: FAILED after 6 seconds, exit code 1.
- Root cause for both: the round overlay YAML was passed directly to the
  existing Whisper loader, which requires the nested legacy model/data config
  and raised `KeyError: 'model'`. The runners now resolve their declared
  `data_config`; dry-runs and diff checks pass after repair.
- Replacement submissions after repair: Job A `46544` failed during paired
  PIER validation; Job B `46545` completed successfully.
- PIER repair: transition POIs now use normalized references, matching the
  existing corpus PIER denominator. Focused tests and dry-runs pass.
- Corrected Job A resubmitted as Slurm `46551`; it was RUNNING on `worker-1`
  at the latest check, with no dependency.
- Failed-attempt logs were `logs/cs_asr_round1_frozen_46528.log` and
  `logs/cs_asr_round1_training_46529.log`. Replacement logs are
  `logs/cs_asr_round1_frozen_46544.log` and
  `logs/cs_asr_round1_training_46545.log` (stderr uses the same stem with
  `.err`). Job B metrics/checkpoints exist; Job A has only baselines and
  direction caches because its first cell failed validation.
- Current Job-A logs are `logs/cs_asr_round1_frozen_46551.log` and
  `logs/cs_asr_round1_frozen_46551.err`.

## Known scope boundary

The new runners extend the existing package and preserve prior results. The
existing representation trainer is reused for the fixed T1/T2/T3 kernels;
its method checkpoints are redirected to the isolated Round-1 Job-B root.

The submitted wrapper currently delegates the prior fixed layer-24 trainer;
the seven-layer T1/GlobalDecoder/SALSA-E/LoRA enumeration and the complete
post-screen geometry/report production remain an implementation gap to repair
before treating the Slurm outputs as scientifically complete. This is recorded
explicitly rather than labeling partial output as completed.
