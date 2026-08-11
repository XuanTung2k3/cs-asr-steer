# CS-ASR E1–E5 current hardened pipeline runbook

Snapshot time: 2026-07-28  
Workspace: `/home/tungnx/cs-asr-steer`  
Production artifacts: `/mnt/data/tungnx/cs-asr-steer/artifacts_v2`  
Smoke artifacts: `/mnt/data/tungnx/cs-asr-steer/artifacts_smoke`

This file describes the current implementation after the hardening update. It
replaces the operational flow described in `CS_ASR_E1_E5_Latest_Status.md`,
which is now a historical snapshot from before the E1/E4/E5 gate changes.

## Current state

Production E1 job `31574` completed on 2026-07-28T10:46:20 UTC and failed its
scientific gate. See:

```text
docs/CS_ASR_E1_E5_Job31574_Result_Analysis.md
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/status/e1.json
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/reports/e1_alignment_audit.md
```

The immediate next task is E1 alignment diagnosis/repair. Do not submit
`e2_pilot`, `auto`, or any downstream production phase until E1 status becomes
`passed`.

## Main rule

Run one production Slurm job at a time. Each phase is independently resumable.
Do not proceed to the next phase unless the previous phase status is `passed`.

Use this monitor command after every job:

```bash
squeue -j <JOBID> -o '%.18i %.20j %.2t %.12M %.12l %.6D %R'
```

After a job ends, check accounting:

```bash
sacct -j <JOBID> --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End -P
```

Check pipeline status:

```bash
/home/tungnx/miniconda3/envs/acl1/bin/python -m csasr.experiments.pipeline \
  --config configs/base.yaml \
  --set experiment.output_root=/mnt/data/tungnx/cs-asr-steer/artifacts_v2 \
  --status
```

## Current launcher behavior

`auto` is no longer a full E1→E5 run. It is now safe/conservative:

```text
preflight
unit tests
optional smoke
P0
E1
stop for inspection
```

The intended production flow is explicit phase-by-phase submission:

```text
E1
→ inspect/pass
E2 pilot
→ E3 pilot
→ inspect/pass
E4 pilot
→ inspect/pass
E2 full
→ E3 full
→ E4 refine
→ E4 confirm
→ inspect/pass
E5
→ final summary
```

## Exact phase commands

Submit only one command at a time.

```bash
sbatch cs_asr_e1_e5.sh e1
sbatch cs_asr_e1_e5.sh e2_pilot
sbatch cs_asr_e1_e5.sh e3_pilot
sbatch cs_asr_e1_e5.sh e4_pilot
sbatch cs_asr_e1_e5.sh e2_full
sbatch cs_asr_e1_e5.sh e3_full
sbatch cs_asr_e1_e5.sh e4_refine
sbatch cs_asr_e1_e5.sh e4_confirm
sbatch cs_asr_e1_e5.sh e5
sbatch cs_asr_e1_e5.sh summary
```

If a job times out or is interrupted, rerun the same phase command. The stage
uses resumable checkpoints where possible.

## Status files used by current implementation

The current implementation uses phase-specific statuses:

```text
status/p0.json
status/e1.json
status/e2_pilot.json
status/e2_full.json
status/e3_pilot.json
status/e3_full.json
status/e4_pilot.json
status/e4_refine.json
status/e4_confirm.json
status/e5.json
status/overview.json
```

Do not use old shared `status/e2.json`, `status/e3.json`, or `status/e4.json`
as current phase evidence.

## Step 1: E1 alignment and audit

Current running command:

```bash
sbatch cs_asr_e1_e5.sh e1
```

Expected outputs:

```text
reports/e1_alignment_audit.md
metrics/e1_alignment_metrics.json
metrics/e1_synthetic_boundary_error.parquet
metrics/e1_ctc_agreement.parquet
alignments/dev_select.parquet
alignments/dev_confirm.parquet
alignments/train_direction_full.parquet
alignments/train_direction_pilot.parquet
audit/e1/records.jsonl
audit/e1/verdicts_template.csv
status/e1.json
```

E1 production pass now requires all of these gates:

| gate | threshold |
|---|---:|
| unit table validates | required |
| human audit verdicts | 100 items |
| human usable fraction | >= 0.90 |
| synthetic boundaries scored | >= 100 |
| synthetic boundaries within 100 ms | >= 0.90 |
| absolute synthetic systematic bias | <= 50 ms |
| independent CTC audit boundaries | >= 200 |
| independent CTC within 100 ms | >= 0.90 |
| independent CTC median absolute disagreement | <= 100 ms |

### E1 decision tree

After E1 ends, inspect:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/status/e1.json
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/reports/e1_alignment_audit.md
```

If `status/e1.json` says `passed`:

```bash
sbatch cs_asr_e1_e5.sh e2_pilot
```

If `status/e1.json` says `blocked`, usually the 100-item human audit is not
finished. Fill this file:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/verdicts_template.csv
```

Save the completed version as:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/verdicts.csv
```

Then rerun only E1:

```bash
sbatch cs_asr_e1_e5.sh e1
```

If `status/e1.json` says `failed`, do not continue. Read the E1 report and fix
the failed gate. Common failure causes are:

1. invalid unit table;
2. synthetic known-boundary result still biased/incorrect;
3. independent CTC audit unavailable or disagrees too much;
4. not enough boundaries;
5. silence absorption too high.

Only after E1 is `passed` may E2 start.

## Step 2: E2 pilot direction construction

Run:

```bash
sbatch cs_asr_e1_e5.sh e2_pilot
```

Purpose:

- Build pilot EN-minus-ZH directions from training speakers only.
- Build required control directions.
- Build decoder directions needed by `DEC-ALL-GLOBAL`.

Expected outputs:

```text
reports/e2_pilot_direction_construction.md
metrics/e2_pilot_direction_statistics.json
directions/
status/e2_pilot.json
```

If passed, continue:

```bash
sbatch cs_asr_e1_e5.sh e3_pilot
```

If failed, stop. Do not run E3/E4.

## Step 3: E3 pilot separability

Run:

```bash
sbatch cs_asr_e1_e5.sh e3_pilot
```

Purpose:

- Test whether pilot directions separate EN and ZH frames on unseen
  `dev_select` speakers.
- Select top layers for E4 pilot.

Expected outputs:

```text
reports/e3_pilot_separability.md
reports/e3_pilot_selected_layers.json
metrics/e3_pilot_layer_metrics.parquet
metrics/e3_pilot_summary.json
status/e3_pilot.json
```

If passed, continue:

```bash
sbatch cs_asr_e1_e5.sh e4_pilot
```

If failed, stop and diagnose directions/alignment before E4.

## Step 4: E4 pilot oracle steering

Run:

```bash
sbatch cs_asr_e1_e5.sh e4_pilot
```

Purpose:

- Small decisive pilot on `dev_select`.
- Test whether exact local encoder steering fixes English POI errors more than
  it harms correct outputs.
- Compare against wrong-sign, random-vector, random-location, global encoder,
  and global decoder controls.

Expected outputs:

```text
reports/e4_oracle_steering_pilot.md
metrics/e4_all_runs.parquet
metrics/e4_selected_config.json
manifests/e4_pilot_wrong.parquet
manifests/e4_pilot_correct.parquet
predictions/e4/
status/e4_pilot.json
```

Pilot gates use fraction-scale degradation limits:

| gate | threshold |
|---|---:|
| relative PIER gain | >= 0.05 |
| correction/corruption evidence | > 2.0 ratio, or zero-corruption with raw corrections |
| MER degradation | <= 0.005 |
| ZH-CER degradation | <= 0.005 |
| outside-region edit rate | <= 0.05 |

If passed, inspect the report. Then continue to full-scale directions:

```bash
sbatch cs_asr_e1_e5.sh e2_full
```

If failed, stop. Do not scale to full E2/E3/E4.

## Step 5: E2 full direction construction

Run:

```bash
sbatch cs_asr_e1_e5.sh e2_full
```

Purpose:

- Rebuild directions using `train_direction_full`.
- Preserve same frozen model, dataset, official splits, and no-training
  constraint.

Expected outputs:

```text
reports/e2_full_direction_construction.md
metrics/e2_full_direction_statistics.json
directions/
status/e2_full.json
```

If passed:

```bash
sbatch cs_asr_e1_e5.sh e3_full
```

## Step 6: E3 full separability

Run:

```bash
sbatch cs_asr_e1_e5.sh e3_full
```

Purpose:

- Validate full directions before confirmation/refinement.
- Freeze layer candidates for E4 refine/confirm.

Expected outputs:

```text
reports/e3_full_separability.md
reports/e3_full_selected_layers.json
metrics/e3_full_layer_metrics.parquet
metrics/e3_full_summary.json
status/e3_full.json
```

If passed:

```bash
sbatch cs_asr_e1_e5.sh e4_refine
```

## Step 7: E4 refine

Run:

```bash
sbatch cs_asr_e1_e5.sh e4_refine
```

Purpose:

- Use full directions.
- Refine alpha around the selected pilot configuration.
- Keep same E4 systems and controls.

Expected outputs:

```text
reports/e4_oracle_steering_refine.md
metrics/e4_all_runs.parquet
metrics/e4_selected_config.json
status/e4_refine.json
```

Refine uses confirmation-strength degradation limits:

| gate | threshold |
|---|---:|
| MER degradation | <= 0.002 |
| ZH-CER degradation | <= 0.002 |
| outside-region edit rate | <= 0.05 |

If passed:

```bash
sbatch cs_asr_e1_e5.sh e4_confirm
```

## Step 8: E4 confirm

Run:

```bash
sbatch cs_asr_e1_e5.sh e4_confirm
```

Purpose:

- Confirm the selected E4 configuration on `dev_confirm`.
- This is the main production go/no-go result for local activation steering.

Current confirmation target:

```yaml
confirm_wrong: 300
confirm_correct: 300
min_confirm_wrong: 300
```

If one-POI-per-utterance and matching leave fewer than 300 wrong POIs, the
report must state the exact reason and use every eligible example. It should not
silently reduce the target.

Expected outputs:

```text
reports/e4_oracle_steering_confirm.md
metrics/e4_all_runs.parquet
manifests/e4_confirm_wrong.parquet
manifests/e4_confirm_correct.parquet
predictions/e4/
status/e4_confirm.json
```

If `e4_confirm` passes, continue:

```bash
sbatch cs_asr_e1_e5.sh e5
```

If `e4_confirm` fails, stop. Do not claim the steering method works. Do not run
E5 as a production claim.

## Step 9: E5 boundary robustness

Run:

```bash
sbatch cs_asr_e1_e5.sh e5
```

Purpose:

- Test whether the confirmed exact-span E4 gain survives boundary expansion and
  true boundary jitter.
- Report absolute PIER differences, retained-gain bootstrap, mean/variance
  across jitter seeds.

Expected outputs:

```text
reports/e5_boundary_robustness.md
metrics/e5_mask_results.parquet
metrics/e5_summary.json
predictions/e5/
audit/e5/
status/e5.json
```

E5 gates:

| gate | threshold |
|---|---:|
| exact-span PIER gain | >= 0.005 |
| retained gain at 100 ms jitter | >= 0.70 |
| retained-gain bootstrap lower bound | >= 0.70 |
| correction/corruption at 100 ms | > 2.0 evidence |
| outside-region edit rate at 100 ms | <= 0.05 |
| jitter-seed PIER std | <= 0.05 |

If E5 fails after E4 passes, the correct interpretation is:

```text
Exact oracle steering may work, but it is boundary-sensitive / oracle-bound.
```

Do not call it deployment-ready.

## Final summary

After E5, run:

```bash
sbatch cs_asr_e1_e5.sh summary
```

Then read:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/status/overview.json
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/reports/
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/metrics/
```

## Important interpretation rules

1. Smoke artifacts are not scientific results.
2. Old failed E1 artifacts are historical only.
3. Any downstream stage may consume upstream artifacts only when the upstream
   phase status is `passed`.
4. Do not proceed past E1 unless the human, synthetic, unit, and independent CTC
   gates pass.
5. Do not proceed past E4 confirm unless correction gain clearly beats harm and
   controls.
6. Do not weaken a gate to make the pipeline continue.
