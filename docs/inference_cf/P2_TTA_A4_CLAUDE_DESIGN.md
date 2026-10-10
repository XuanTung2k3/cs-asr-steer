# P2-TTA-A4 — implementation design (pre-outcome)

This design extends the A3 code with no changes to historical modules.

- **`src/csasr/inference_cf/script_safe_tta.py`:** token classes from the canonical partition; per-position safe-teacher selection with `torch.where` over the A3 `allowed_log_probs` parts of q_A and q_B; and an `adapt_a4` mirror of `adapt_a3`. It logs D_E, D_M and D_ANCHOR per step, applies the exact no-op rule, and keeps the A2/A3 actuator unchanged.
- **`experiments/inference_cf_p2tta_a4{,_analyze,_audit}.py`:** the runner, the guarded evaluator and the independent auditor (live check and post-audit).
- **`slurm/inference_cf_p2tta_a4.sbatch`** and **`tests/test_inference_cf_p2tta_a4.py`**.
- **Comparators:** B0, AUTO and A2 come from the A3 plan seal; A3 outputs come from the A3 run1 rows, hash-checked against the A3 output seal.
- **Gates:** freeze, then `PASS_TO_P2_TTA_A4`, then the manifest is pushed, then one sbatch job, then the output seal, then evaluation, then `P2_TTA_A4_AUDIT: PASS`.
