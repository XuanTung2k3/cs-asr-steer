# CS-ASR E1 job 31574 result analysis

Snapshot time: 2026-07-28 after Slurm job `31574` completed  
Workspace: `/home/tungnx/cs-asr-steer`  
Production artifacts: `/mnt/data/tungnx/cs-asr-steer/artifacts_v2`

## Executive result

Job `31574` completed, but production E1 failed.

| job | state | exit code | elapsed | start | end |
|---:|---|---:|---:|---|---|
| 31574 | FAILED | 2:0 | 06:09:44 | 2026-07-28T04:36:36 | 2026-07-28T10:46:20 |

Current pipeline status:

| phase | status |
|---|---|
| P0 | passed |
| E1 | failed |
| E2 pilot | pending |
| E2 full | pending |
| E3 pilot | pending |
| E3 full | pending |
| E4 pilot | pending |
| E4 refine | pending |
| E4 confirm | pending |
| E5 | pending |

Conclusion: do not run E2/E3/E4/E5 yet. The current E1 alignments exist on
disk, but they are diagnostic only because `status/e1.json` is failed and
`complete` is false.

## What worked

The hardened E1 fixed the old unit-table blocker. Unit validation now passes.

Passed E1 gates:

| gate | value | threshold | result |
|---|---:|---:|---|
| unit table validates | 1 | == 1 | pass |
| switch boundaries | 3043 | >= 100 | pass |
| eligible bilingual training utterances | 3734 | >= 500 | pass |
| DTW cross-config agreement within 100 ms | 0.9699 | >= 0.90 | pass |
| silence absorption rate | 0.0659 | <= 0.20 | pass |
| synthetic boundaries scored | 100 | >= 100 | pass |
| independent CTC audit status | ok | ok | pass |
| independent CTC boundaries compared | 853 | >= 200 | pass |
| audit pack items | 100 | >= 100 | pass |

Subset summary:

| subset | utterances | bilingual utterances | units | unit validation |
|---|---:|---:|---:|---|
| dev_select | 3360 | 708 | 116172 | pass |
| dev_confirm | 2826 | 420 | 65197 | pass |
| train_direction_full | 4596 | 4596 | 178033 | pass |
| train_direction_pilot | 3734 | 3734 | 149258 | pass |

## What failed

E1 failed because boundary timing is not reliable enough for local-span science.

Failed E1 gates:

| gate | value | threshold | result |
|---|---:|---:|---|
| synthetic boundaries within 100 ms | 0.0600 | >= 0.90 | fail |
| absolute synthetic systematic bias | 548.6 ms | <= 50 ms | fail |
| independent CTC within 100 ms | 0.0703 | >= 0.90 | fail |
| independent CTC median absolute disagreement | 370.0 ms | <= 100 ms | fail |
| human audit verdicts | 0 | >= 100 | fail |
| human usable fraction | not provided | >= 0.90 | fail |

The missing human audit is not the main blocker. Even if 100 human verdicts were
completed now, E1 would still fail because both automatic boundary-validity
checks fail by a large margin.

## Synthetic known-boundary result

| metric | value |
|---|---:|
| scored boundaries | 100 |
| within 50 ms | 0.0000 |
| within 100 ms | 0.0600 |
| within 150 ms | 0.1200 |
| within 200 ms | 0.2100 |
| within 300 ms | 0.3400 |
| within 500 ms | 0.5100 |
| mean signed error | -548.6 ms |
| median signed error | -490.0 ms |
| median absolute error | 490.0 ms |
| p95 absolute error | 1120.0 ms |
| silence trimmed | true |

Interpretation:

- Whisper-DTW predicts the synthetic ZH→EN boundary about 0.55 s too early on
  average.
- The configured 100 ms boundary exclusion covers only 6% of measured synthetic
  boundary error.
- Even a 500 ms tolerance covers only 51% of synthetic boundaries.
- This is a hard scientific failure for exact local boundary claims.

## Independent CTC audit result

Overall CTC-vs-DTW comparison:

| metric | value |
|---|---:|
| comparable boundaries | 853 |
| within 50 ms | 0.0117 |
| within 100 ms | 0.0703 |
| median absolute disagreement | 370.0 ms |
| systematic offset | +1353.9 ms |

By switch direction:

| direction | boundaries | within 50 ms | within 100 ms | median abs disagreement | systematic offset |
|---|---:|---:|---:|---:|---:|
| EN→ZH | 417 | 0.0072 | 0.0408 | 410.0 ms | +1364.3 ms |
| ZH→EN | 436 | 0.0161 | 0.0986 | 310.0 ms | +1343.9 ms |

Interpretation:

- The independent CTC audit ran successfully, but it strongly disagrees with
  Whisper-DTW.
- Same-method DTW cross-configuration agreement is high, but it does not prove
  absolute boundary correctness.
- The CTC systematic offset is large and consistent across EN→ZH and ZH→EN.
  Either DTW, CTC, or the coordinate mapping between them is systematically
  wrong. This must be diagnosed before downstream stages.

## Current scientific insights

1. The hardened gate is working: it blocked an unsafe E1 instead of allowing E2
   to consume poor boundaries.
2. The old unit-validation failure is resolved.
3. The remaining blocker is boundary localization, not sample size.
4. Whisper-DTW is internally consistent across two configurations, but external
   validation fails.
5. The current 100 ms exclusion window is not enough for the measured alignment
   error.
6. E2/E3/E4/E5 would be scientifically invalid if started now.

## Useful produced evidence

Read these first:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/status/e1.json
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/reports/e1_alignment_audit.md
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/metrics/e1_alignment_metrics.json
```

Detailed metric tables:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/metrics/e1_synthetic_boundary_error.parquet
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/metrics/e1_ctc_agreement.parquet
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/metrics/e1_boundary_agreement.parquet
```

Human audit pack:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/records.jsonl
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/verdicts_template.csv
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/clips/
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/figures/
```

Diagnostic alignment tables:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/alignments/dev_select.parquet
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/alignments/dev_confirm.parquet
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/alignments/train_direction_full.parquet
/mnt/data/tungnx/cs-asr-steer/artifacts_v2/alignments/train_direction_pilot.parquet
```

These alignment tables must not be treated as passed production upstream
artifacts until E1 passes.

## What to do now

Do not run:

```bash
sbatch cs_asr_e1_e5.sh e2_pilot
sbatch cs_asr_e1_e5.sh auto
```

Recommended next work:

1. Diagnose the E1 boundary coordinate/timing failure.
   - Inspect worst synthetic examples from `e1_synthetic_boundary_error.parquet`.
   - Compare true splice time, predicted DTW boundary, waveform/spectrogram
     transition, and source clip trim offsets.
   - Check whether E1 is selecting the intended first synthetic ZH→EN boundary
     or an internal tagging/alignment artifact.
   - Inspect CTC-vs-DTW rows in `e1_ctc_agreement.parquet`.
   - Verify whether the CTC aligner has a time-coordinate offset or is unsuitable
     for this mixed-script audit.

2. Fix the alignment path.
   - Possible fixes: corrected DTW boundary extraction, corrected CTC coordinate
     transform, calibrated offset only if scientifically justified, a better
     multilingual forced aligner, or dataset TextGrid/long-wav boundaries if
     available.
   - Do not weaken the gates to continue.

3. Rerun only E1 after the alignment fix.

```bash
sbatch cs_asr_e1_e5.sh e1
```

4. Complete the 100-item human audit after the automatic boundary issue is
   understood. It is useful evidence, but it cannot make E1 pass while synthetic
   and CTC gates fail.

5. Proceed to E2 pilot only after E1 status is `passed`.

```bash
sbatch cs_asr_e1_e5.sh e2_pilot
```

## Bottom line

The result is not good enough to continue. This is a scientifically useful
failure: it shows that the current DTW boundary localization is not reliable
enough for exact local steering claims. The next task is E1 alignment diagnosis
and repair, not downstream execution.

