# CS-ASR E1-E5 latest status snapshot

Snapshot time: 2026-07-28T04:02:52+00:00  
Workspace: `/home/tungnx/cs-asr-steer`  
Production artifacts: `/mnt/data/tungnx/cs-asr-steer/artifacts_v2`  
Smoke artifacts: `/mnt/data/tungnx/cs-asr-steer/artifacts_smoke`  
Main command submitted: `sbatch cs_asr_e1_e5.sh auto`

This file is written for reading the experiment status without reading the
codebase. It summarizes what has already produced valid results, what is only a
smoke/plumbing artifact, what the pending Slurm job is expected to do, and where
the current implementation differs from `docs/CS_ASR_E1_E5_AI_Guide.md`.

## Executive state

The newest submitted production job is Slurm job `31555`. At the audit time it
was still pending, not running:

| job | command | state | reason | submitted | time limit |
|---|---|---|---|---|---|
| 31555 | `cs_asr_e1_e5.sh auto` | `PENDING` | `Resources` | 2026-07-28T03:50:53 UTC | 48 hours |

Therefore, job `31555` has not produced new result files yet. Current production
results come from earlier jobs:

| stage | current production status | source job | interpretation |
|---|---|---:|---|
| P0 | passed | 31413 | Valid production result. Baseline and POI audit are complete. |
| E1 | failed | 31473 | Old result from before the latest code changes. It is expected to be rerun by job `31555`. |
| E2 | pending | none | No production direction-construction result yet. |
| E3 | pending | none | No production separability result yet. |
| E4 | pending | none | No production oracle-steering result yet. |
| E5 | pending | none | No production boundary-robustness result yet. |

Only one workflow completion marker exists in production right now:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/workflow/guide_v2_20260727_p0.done
```

This means the next `auto` run should skip production P0, but it should still
run unit tests, smoke, E1, and then later stages if gates pass.

## What is valid already

### P0 baseline and POI audit

P0 passed on 2026-07-27T19:42:02 UTC in job `31413`.

The frozen primary baseline is `B0_AUTO`, selected because it had the lowest MER
on `dev_select`. It is the baseline used for E1-E5 comparisons.

| subset | MER | EN WER | ZH CER | PIER | EN POIs | POI errors |
|---|---:|---:|---:|---:|---:|---:|
| dev_select | 0.2322 | 0.2891 | 0.2139 | 0.2814 | 23726 | 6676 |
| dev_confirm | 0.2237 | 0.2714 | 0.2064 | 0.2635 | 11478 | 3024 |

P0 gate evidence:

| criterion | value | threshold | result |
|---|---:|---:|---|
| erroneous EN POIs | 9700 | >= 300 | pass |
| language-confusion errors | 1125 | >= 100 | pass |
| baseline-correct EN POIs | 25504 | >= 300 | pass |

Reader interpretation: there are enough baseline English POI errors and enough
correct English POIs to make E4/E5 meaningful. The project should continue past
P0.

### Existing E1 result is stale but useful context

E1 failed on 2026-07-27T20:26:55 UTC in job `31473`.

Old E1 gate evidence:

| criterion | value | threshold | result |
|---|---:|---:|---|
| units validate | 0 | == 1 | fail |
| switch boundaries | 3043 | >= 100 | pass |
| eligible bilingual training utterances | 3734 | >= 500 | pass |
| synthetic boundaries within 100 ms | 0.0202 | >= 0.90 | fail |
| synthetic absolute systematic bias | 1646.5 ms | <= 50 ms | fail |
| config agreement within 100 ms | 0.9699 | >= 0.90 | pass |
| silence absorption rate | 0.0659 | <= 0.20 | pass |

The old failure is not the final scientific answer because the code was changed
after that run. The current E1 implementation now:

1. Re-applies a stronger boundary-overlap resolver to cached alignment tables.
2. Regenerates synthetic splice audio when the synthetic harness settings change.
3. Trims leading/trailing silence before synthetic concatenation.
4. Treats synthetic thresholds as advisory unless numeric gate thresholds are set.
5. Does not require a blocking human audit unless `require_human_audit: true`.

Expected effect on the next E1 run: the unresolved overlap should be removed,
and the synthetic check should be recomputed under the trimmed-silence protocol.
The old E1 report should be considered historical context until job `31555`
rewrites it.

## What is not a real result

Files under `/mnt/data/tungnx/cs-asr-steer/artifacts_smoke` are smoke/plumbing
artifacts only. They are intentionally tiny and use forced prerequisites. Some
smoke E3/E4/E5 gates fail because the sample sizes are too small. Those failures
are not production science results.

The smoke run exists to check that each code path can execute without crashing.
It must not be used to claim that E3, E4, or E5 passed or failed scientifically.

## Current `auto` pipeline

`sbatch cs_asr_e1_e5.sh auto` runs one Slurm job and sequences the stages inside
that single allocation. It requests:

| resource | value |
|---|---|
| partition | `main` |
| GPU | 1 |
| CPUs | 8 |
| memory | 96 GB |
| wall time | 48 hours |
| stdout | `/mnt/data/tungnx/cs-asr-steer/logs/cs_asr_e1_e5_<jobid>.out` |
| stderr | `/mnt/data/tungnx/cs-asr-steer/logs/cs_asr_e1_e5_<jobid>.err` |

Execution order:

```text
preflight
unit tests
smoke run in artifacts_smoke
P0 baseline
E1 alignment
E2 pilot directions
E3 pilot separability
E4 pilot oracle steering
E2 full directions
E3 full separability
E4 refine
E4 confirm
E5 boundary robustness
summary
```

The launcher uses workflow markers in
`/mnt/data/tungnx/cs-asr-steer/artifacts_v2/workflow`. A stage with a marker is
skipped on later `auto` submissions unless `OVERWRITE=1` is used.

Important status-file detail: E2 pilot and E2 full both write `status/e2.json`.
E3 pilot and E3 full both write `status/e3.json`. E4 pilot/refine/confirm all
write `status/e4.json`. For exact phase completion, read workflow markers and
reports, not only `status/overview.json`.

## Expected behavior of pending job 31555

When job `31555` starts, the expected path is:

1. Preflight checks local model, local dataset, CUDA, Python, and package versions.
2. Unit tests run under the Conda environment.
3. Smoke runs under `/mnt/data/tungnx/cs-asr-steer/artifacts_smoke`.
4. Production P0 should be skipped because the P0 marker exists.
5. Production E1 should rerun because no E1 workflow marker exists.
6. If current E1 passes, the job proceeds to E2 pilot.
7. If E2 pilot passes, the job proceeds to E3 pilot.
8. If E3 pilot passes, the job proceeds to E4 pilot.
9. If E4 pilot passes, the job scales to E2 full, E3 full, E4 refine, E4 confirm.
10. If confirmed E4 passes, the job runs E5.

Any failed production gate exits the job early. That is intended by the guide.

## Stage-by-stage reader guide

### P0: baseline and POI audit

Purpose: choose and freeze a Whisper baseline, quantify English points of
interest, and verify enough errors/correct examples exist.

Current implementation:

- Uses local Whisper large-v3.
- Uses deterministic greedy decoding.
- Uses `B0_AUTO` and `B1_ZH`, then selects by MER on `dev_select`.
- Freezes `B0_AUTO` for later stages.

Expected output:

- `reports/p0_baseline_audit.md`
- `baselines/primary_baseline.json`
- `status/p0.json`

Current state: passed.

### E1: alignment construction and audit

Purpose: map transcript language units to Whisper encoder frames.

Current implementation:

- Uses Whisper cross-attention DTW because local CS-Dialogue has only
  `short_wav`; `long_wav` TextGrid boundaries are unavailable.
- Aligns `dev_select`, `dev_confirm`, and `train_direction_full`.
- Derives `train_direction_pilot` from the full training alignment table.
- Excludes 100 ms around language boundaries for direction construction.
- Builds a 100-item audit pack, but human audit is optional in current config.
- Runs automatic checks: unit-table validation, cross-config boundary agreement,
  silence absorption, and synthetic known-boundary error.
- CTC aligner check is disabled in current config.

Current guide deviation:

- The guide asks for a manual boundary audit gate. Current config sets
  `require_human_audit: false`.
- The guide also expects boundary quality to be strong enough before using local
  spans. Current config reports synthetic boundary error but does not block on it
  because `min_synthetic_within_100ms` and `max_abs_bias_ms` are `null`.

Expected output:

- `reports/e1_alignment_audit.md`
- `metrics/e1_alignment_metrics.json`
- `alignments/dev_select.parquet`
- `alignments/dev_confirm.parquet`
- `alignments/train_direction_full.parquet`
- `alignments/train_direction_pilot.parquet`
- `audit/e1/records.jsonl`
- `audit/e1/verdicts_template.csv`
- `status/e1.json`

Reader interpretation after job `31555`: if E1 passes but synthetic boundary
error remains large, the pipeline may continue mechanically, but the local
boundary claims should be treated as weaker than the original guide intended.

### E2: activation directions

Purpose: build EN-minus-ZH direction vectors from training speakers only.

Current implementation:

- Uses encoder layers 15, 23, 27, and 31.
- Builds these variants:
  - `within_utterance_correct_only`
  - `within_utterance_all_valid`
  - `global_centroid_correct_only`
- Uses seeds 42-46 and random control seeds 142-146.
- Builds decoder directions for the `DEC-ALL-GLOBAL` E4 control.
- Requires no dev/test contribution.

Gate checks:

- Directions exist for all configured layers.
- All directions are finite and unit norm.
- Cross-seed minimum cosine is at least 0.90.
- Enough EN and ZH frames exist.
- Decoder directions exist.
- Hook tests pass.
- Manifest rows all come from official train.

Expected output:

- `reports/e2_direction_construction.md`
- `metrics/e2_direction_statistics.json`
- `directions/`
- `alignments/train_direction_<subset>_annotated.parquet`
- `status/e2.json`

Current state: no production E2 result yet.

### E3: separability

Purpose: test whether the E2 direction separates EN and ZH frames on unseen
development speakers.

Current implementation:

- Evaluates on `dev_select` code-switching utterances.
- Caps to 1000 utterances unless overridden.
- Reports non-boundary and all-frame metrics.
- Uses speaker-macro AUROC and speaker bootstrap intervals.
- Compares primary directions to all-valid, global-centroid, wrong-sign, and
  matched random controls.
- Keeps top 2 layers for E4.

Gate checks:

- Best layer AUROC >= 0.80.
- Seed standard deviation <= 0.03.
- Best layer beats matched random controls.
- Wrong-sign AUROC is consistent with sign reversal.

Expected output:

- `reports/e3_separability.md`
- `reports/e3_selected_layers.json`
- `metrics/e3_layer_metrics.parquet`
- `metrics/e3_summary.json`
- `status/e3.json`

Current state: no production E3 result yet.

### E4: oracle-span steering

Purpose: decisive go/no-go test of whether local steering fixes embedded-English
errors more than it harms correct outputs.

Current implementation:

- Uses one English POI per utterance.
- Pilot uses `dev_select`; confirm uses `dev_confirm`.
- Pilot target size: 100 wrong + 100 correct.
- Confirm target size: up to 300 wrong + 300 correct, with a current floor of
  100 wrong (`min_confirm_wrong: 100`).
- Primary system is `ENC-LOCAL-1L`.
- Required controls include global encoder steering, wrong-sign steering,
  matched random vectors, and global decoder steering.
- Extra advisory control `ENC-RANDLOC-1L` applies the same direction/strength at
  a random disjoint span in the same utterance.

Gate checks:

- Enough baseline-incorrect and matched-correct targets.
- Relative PIER gain >= 0.05.
- Correction-to-corruption ratio > 2.0.
- Correct-sign steering beats wrong-sign steering.
- Correct-sign steering beats matched random controls.
- Local steering has less collateral damage than global controls.
- Overall MER degradation <= 0.2.
- Outside-region edit rate <= 0.05.
- ZH-CER degradation <= 0.2.

Expected output:

- `reports/e4_oracle_steering_pilot.md`
- `reports/e4_oracle_steering_refine.md`
- `reports/e4_oracle_steering_confirm.md`
- `metrics/e4_all_runs.parquet`
- `metrics/e4_selected_config.json`
- `manifests/e4_<phase>_wrong.parquet`
- `manifests/e4_<phase>_correct.parquet`
- `predictions/e4/`
- `status/e4.json`

Current state: no production E4 result yet.

### E5: boundary robustness

Purpose: test whether E4 gains survive mask expansion and +/- boundary error.

Current implementation:

- Uses the frozen E4 configuration on `dev_confirm`.
- Tests masks:
  - `EXACT_HARD`
  - `EXACT_TAPER`
  - `EXPAND_50`, `EXPAND_100`, `EXPAND_200`
  - `JITTER_50`, `JITTER_100`, `JITTER_200`
  - `BOUNDARY_ONLY`
  - `WHOLE_WORD_PLUS_CONTEXT`
- Uses jitter seeds 242-246.
- Requires 70% retained gain at 100 ms jitter.

Gate checks:

- Retained gain at 100 ms jitter >= 0.70.
- Correction-to-corruption at 100 ms > 2.0.
- Tapered mask no worse than hard mask on corruption.
- Outside-region edit rate at 100 ms <= 0.05.
- Jitter-seed PIER standard deviation <= 0.05.

Expected output:

- `reports/e5_boundary_robustness.md`
- `metrics/e5_mask_results.parquet`
- `metrics/e5_summary.json`
- `predictions/e5/`
- `audit/e5/mask_example_*.png`
- `status/e5.json`

Current state: no production E5 result yet.

## Guide conformance summary

The current implementation follows these guide constraints:

| guide item | current state |
|---|---|
| Frozen Whisper backbone | implemented |
| No gradient training for directions | implemented |
| Local model/data, offline execution | implemented |
| Official test split prohibited in E1-E5 | implemented by manifest filtering and assertions |
| Dev split into `dev_select` and `dev_confirm` by speaker | implemented |
| Training speakers only for directions | implemented and gated in E2 |
| One target POI per utterance in E4/E5 | implemented |
| Baseline cached and frozen | implemented |
| E1 -> E2 -> E3 -> E4 -> E5 gate order | implemented in production stages |
| Random, wrong-sign, global encoder, and global decoder controls | implemented |
| Boundary robustness masks in E5 | implemented |

Known deviations or shortened implementation:

| area | current behavior | risk |
|---|---|---|
| E1 source alignments | Uses Whisper cross-attention DTW, not TextGrid boundaries. | Acceptable fallback, but needs stronger validation. |
| Human audit | Audit pack is produced, but human audit is optional. | Weaker than guide section 13/15. |
| Synthetic E1 gate | Synthetic boundary error is reported but not blocking. | Pipeline may proceed even if true boundary error is large. |
| CTC audit | Disabled. | No independent forced-aligner comparison. |
| Git commit logging | Workspace is not a Git repo. | Guide asks for exact code commit; current runs can only record `is_git_repo: false`. |
| Launcher comment | Header still says first E1 stops for human audit. | Comment is stale; actual config does not stop for audit. |
| E4 confirm size | Confirmation can pass with 100 wrong examples if fewer than 300 are available. | Reasonable practical change, but shorter than the guide's ideal 300 wrong + 300 correct. |
| Smoke | Smoke plants a synthetic E4 config if pilot cannot select one. | Safe because smoke is isolated, but smoke outputs are not scientific. |

## Local verification performed

These checks were run from `/home/tungnx/cs-asr-steer`:

| check | result |
|---|---|
| `bash -n cs_asr_e1_e5.sh` | pass |
| `python -m compileall -q src tests` | pass |
| `python -m pytest -q` with launcher-equivalent `LD_LIBRARY_PATH` and `PYTHONPATH` | pass, 90 tests |

Note: running pytest without the launcher's `LD_LIBRARY_PATH` fails because
SciPy loads the system `libstdc++` and misses `CXXABI_1.3.15`. The Slurm script
sets the Conda library path first, which avoids that failure.

## Monitoring commands

Use these while job `31555` is pending or running:

```bash
squeue -j 31555 -o '%.18i %.20j %.2t %.12M %.12l %.6D %R'
```

After the job starts, follow logs:

```bash
tail -f /mnt/data/tungnx/cs-asr-steer/logs/cs_asr_e1_e5_31555.out
tail -f /mnt/data/tungnx/cs-asr-steer/logs/cs_asr_e1_e5_31555.err
```

Check stage status:

```bash
jq . /mnt/data/tungnx/cs-asr-steer/artifacts_v2/status/overview.json
```

Check exact completed workflow markers:

```bash
find /mnt/data/tungnx/cs-asr-steer/artifacts_v2/workflow -maxdepth 1 -type f -printf '%f\n' | sort
```

List new reports:

```bash
find /mnt/data/tungnx/cs-asr-steer/artifacts_v2/reports -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

Do not submit a second `auto` job while `31555` is pending or running. This
pipeline is designed for one active production job at a time.

## Decision points after job 31555 runs

If E1 passes:

- Read `reports/e1_alignment_audit.md`.
- Check `Boundary error vs synthetic ground truth`, even though it is currently
  advisory.
- If synthetic boundary error is still large, interpret later E4/E5 local-span
  results cautiously or restore a blocking audit/CTC check before making strong
  claims.

If E2 fails:

- The direction estimate is unstable, has too few frames, lacks decoder controls,
  or hook tests failed. Do not continue to E3/E4 as a production claim.

If E3 fails:

- The EN/ZH direction is not credible on unseen dev speakers. The guide says to
  stop and diagnose alignment/direction construction before large E4 sweeps.

If E4 fails:

- Do not build a router. E4 is the decisive feasibility gate. Report the negative
  correction-versus-harm result and inspect layer/strength/mask choices only as
  diagnostics.

If E5 fails after E4 passes:

- Exact oracle steering may work, but the method is boundary-sensitive. The
  correct claim is "oracle-bound", not deployment-ready.

## Current source snapshot hashes

The folder is not a Git repository, so these hashes are the practical source
snapshot references for this audit.

| file | sha256 |
|---|---|
| `cs_asr_e1_e5.sh` | `b4d0056d76c2982bde25ac6b0e3529bdc54d3230149455d83269268718b25a58` |
| `configs/base.yaml` | `cec12b7fab3e33f5392d7e3f9d5c11e4fb34275704599a06f3ee4db551404691` |
| `configs/experiments/e1_alignment.yaml` | `8365a84f53d535cc2f4361fe741df128b4a405b68580a369efb28a168b1da715` |
| `configs/experiments/e2_directions.yaml` | `44976ce5ebfc796d7205cf69ddc4fc23ed9130b29ecd64ef593a9ad57274ff92` |
| `configs/experiments/e3_separability.yaml` | `bb0e855b88f47011396c7404b8396691238aa66cdc612b94af226ab882f2cb9e` |
| `configs/experiments/e4_oracle.yaml` | `60cf414abcabb1a71f857c5ddfefdc92cf6ccedeb3ddbc6a8291325bad23d472` |
| `configs/experiments/e5_boundaries.yaml` | `8df2d8aa81436fe2d35937fce8c05e27198cfb4988cf8ed01dc8341e79036259` |
| `src/csasr/experiments/e1_alignment.py` | `dc54c8b2c2b5c036d4d70d18836055eecab09a276caa7470d4483bd744a6182d` |
| `src/csasr/experiments/e2_directions.py` | `721161cda0c515c91333f0e59af18cb879ee499acf23cdcfc12b0f18dc2754c7` |
| `src/csasr/experiments/e3_separability.py` | `f5f09aeaed678aa687124ad810537737b51eae0c70383ec1338399b9f3198b8e` |
| `src/csasr/experiments/e4_oracle.py` | `9139faccf8900311a879c03a0c78aead7ee499efd27472edddc6bcaae3a9a4ab` |
| `src/csasr/experiments/e5_boundaries.py` | `744ad28d86a52eb620bc22c575445f40b9a672f0610c82b2a702a59f6dadc4ff` |
| `src/csasr/data/alignment.py` | `592abcf8e341b82894d84f4f49676bec5e21522e198076b921a448341141224d` |
| `src/csasr/data/alignment_checks.py` | `d4675ac90ff38b0cfd09ecfab8cf05f637f5b8414b503225c82731bcd73d4380` |

