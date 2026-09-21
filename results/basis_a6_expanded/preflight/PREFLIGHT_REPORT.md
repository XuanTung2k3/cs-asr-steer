# BASIS-A6 Expanded Preflight

Status: **BLOCKED**

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
| oracle_tt_model_acceptance | BLOCKED |
| decoding_acceptance_both_models | BLOCKED |
| runtime_vram_preflight | BLOCKED |

## Blocking gates

- `oracle_tt_model_acceptance`
- `decoding_acceptance_both_models`
- `runtime_vram_preflight`

## Matrix dry run

A6-F = 46,720; A6-TT = 23,360; total = 70,080; baselines = 16.

## ASCEND

Download and subset freeze PASS: train 9,869, validation 1,130, test 1,315; construct N=125; eval N=220; test usage=0. Validation eligibility audit: total=1,130, mixed-eligible=220, selected=220 (all eligible below the 300 target).

## Direction status

CS fixed directions: 584/584. ASCEND fixed directions: 584/584. Cross-source geometry: complete (584 cross-source rows; 120 within-source conditioning rows).

## Oracle-TT / leakage

CPU phase orchestration, dynamic-rank rules, cache keys/bundle hashing, local-mask guard, and gold-token trace PASS. Model-resident causal acceptance remains BLOCKED.

## Local steering / decoding

Accepted A4 Whisper/Qwen local site and mask hashes PASS. Direction construction jobs 53312 (Whisper) and 53315 (Qwen) PASS; 53313 was a repaired manifest bookkeeping failure. CUDA/environment acceptance PASS via 53316 (H100 MIG 3g.40gb, acl1 bootstrap). End-to-end Whisper/Qwen greedy and official-standard acceptance remains BLOCKED.

## Runtime / sharding

No representative decode benchmark was run; runtime, VRAM, cache-size, and disk-size measurements are null. The recorded sharding plan is `FULL_RUN_SHARDING_PLAN.md`; atlas authorization is false.
