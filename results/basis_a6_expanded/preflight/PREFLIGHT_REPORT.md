# BASIS-A6 Expanded Preflight

Status: **READY_FOR_FULL_RUN**

The 70,080-cell atlas was not executed.

## Gate summary

| Gate | Status |
|---|---|
| matrix_counts | PASS |
| direction_finite_unit | PASS |
| unique_shared_gate | PASS |
| cache_bundle | PASS |
| ascend_download_exact_splits | PASS |
| ascend_manifests_frozen | PASS |
| ascend_construct_target | PASS |
| ascend_eval_target_or_all | PASS |
| metric_acceptance | PASS |
| local_mask_acceptance | PASS |
| gpu_acceptance | PASS |
| fixed_directions_complete | PASS |
| cross_source_geometry_complete | PASS |
| oracle_tt_model_acceptance | PASS |
| decoding_acceptance_both_models | PASS |
| runtime_vram_preflight | PASS |
| real_model_dynamic_rank | PASS |
| real_model_cache_reuse | PASS |
| runtime_sharding_measured | PASS |
| real_acceptance_artifact_validation | PASS |

## Blocking gates



## Matrix dry run

A6-F = 46,720; A6-TT = 23,360; total = 70,080; baselines = 16.

## ASCEND

Download and subset freeze PASS: train 9,869, validation 1,130, test 1,315; construct N=125; eval N=220; test usage=0. Validation eligibility audit: total=1,130, mixed-eligible=220, selected=220 (all eligible below the 300 target).

## Direction status

CS fixed directions: 584/584. ASCEND fixed directions: 584/584. Cross-source geometry: complete (584 cross-source rows; 120 within-source conditioning rows).

## Oracle-TT / leakage

Real-model Whisper and Qwen Oracle-TT PASS. Decoder traces identify `baseline_hypothesis`; gold hidden states/logits/target directions are false. Dynamic rank and per-sample eligibility are recorded in the TT artifacts.

## Local steering / decoding

Whisper greedy PASS; Whisper official-standard PASS with beam traces. Qwen greedy PASS; Qwen official-standard PASS and recorded equivalent to greedy. rho=0 identities and positive local edits pass on all 12 frozen panel rows per condition.

## Runtime / sharding

Real model-resident measurements are in `REAL_BENCHMARK_whisper.json` and `REAL_BENCHMARK_qwen3_asr_1p7b.json`; storage/capacity is in `DISK_CAPACITY.json`; derived runtime and contiguous layer-block sizes are in `RUNTIME_ESTIMATES.json` and `FULL_RUN_SHARDING_PLAN.md`.

Measured jobs: Whisper 53326, Qwen 53321. Peak allocated/reserved VRAM: Whisper 4.64/4.97 GiB; Qwen 4.38/4.45 GiB. Free disk 759.9 GiB; estimated planned cache/result total 4.32 GiB.
