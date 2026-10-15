# P2-1c amendment r1 — float32 re-specification of the T1 cached-vs-full check

**Status:** an engineering repair of one T1 tolerance. It was written after the full replay (job 59321) and **before
any P2-1c scientific output was opened**. The pre-registered analysis of job 59322 (`p2_1c/analysis.json`) is kept
unchanged and unread at the time of this amendment. Its verdict under the frozen rules is `P2_1C_ENGINEERING_BLOCKED`,
and it is reported as such.

The amendment is recorded in `p2_1c/freeze_r1.json`, which chains to `freeze.json` through its SHA-256 and also records
the hashes of the outputs that existed at amendment time.

**Unchanged:**
- the population, anchors, replay engine and continuations;
- candidate scores;
- every other T1 check;
- the flags, cells, verdict and recommendation rules.

## 1. Failure

T1 check 4 compares incremental KV-cached decoding with one full forward. It requires the absolute log-probability
difference of every observed token to be at most 0.10 nats.

In the full replay (job 59321, `integrity.json`), this failed once: case d1, model CE42, 0.1234 nats at one position.

**Everything else passed:**
- argmax agreement wherever the top-2 margin was ≥ 0.2;
- `cache_position` correct;
- finite logits;
- replay parity 70/70;
- all 700 generated beams within 0.018 of their teacher-forced beam-equivalent scores (tolerance 0.05);
- recorded P2-1b scores reproduced (≤ 1e-6);
- no cross-case leakage;
- weights and adapter hashes unchanged.

`integrity.all_ok` was false only because of this check.

## 2. Diagnosis (engineering only; no outcome field read)

- **Script and output:** `p2_1c/posthoc/p21c_cvf_diag.py` and `p2_1c/posthoc/cvf_diag_59325.json`. GPU job 59325
  completed. Jobs 59323 and 59324 failed on script errors (an unbound shell argument, and the encoder being called on
  BF16 features after the cast) before computing anything.
- **Setup:** reference firewall on. Cases d1 and d0. All seven systems. Each comparison was run with the BF16 backbone
  and then in float32 (the model cast in memory, with the encoder recomputed in float32).

| | max \|Δ log p\| over observed tokens | where | argmax |
|---|---|---|---|
| BF16, d1 | 0.009–0.123 (CE42 0.123) | low-probability tokens (log p ≈ −4.9 for the maximum) | equal wherever the margin is ≥ 0.2 |
| BF16, d0 | 0.031–0.124 (CE44 0.124; this model was not part of the replay's check) | log p ≈ −5.1 | equal |
| **float32, d1 and d0** | **≤ 2 × 10⁻⁵ for all 7 models** | — | equal everywhere |

**Conclusion.**
- In float32, the cached and full paths agree to rounding, so caching, absolute positions and the adapter hooks are
  semantically exact.
- The BF16 difference is rounding, which can reach about 0.12 nats on low-probability tokens.
- The 0.10 BF16 bound was set too tight. The failure is not an engineering defect.
- The failed case had already passed every other replay check.

## 3. Repair

**T1 check 4 is re-specified:**
- **Semantic check, gating.** In float32, for **all 10 cases × 7 systems**, |Δ log p| ≤ 0.001 for every observed token,
  with positions correct, finite logits, and argmax agreement where the margin is ≥ 0.2.
- **BF16 check.** Positions, finiteness and argmax-where-sure still gate. Its |Δ log p| is reported descriptively.

**Implementation.**
- `reft/p2_1c_continuation_audit.py` adds:
  - `cvf32`, a GPU job, write-once `p2_1c/t1_cached_fp32_r1.json`;
  - `t1_ok_amended` and `amended_integrity`;
  - `analyze-r1`, which writes `analysis_r1.json`, `case_analysis_private_r1.json`, `figures_r1/` and
    `manifest_r1.json`;
  - the freeze chain (`freeze-r1`, `check_freeze` walking the chain).
- `configs/reft/p2_1c_frozen.yaml` adds `tolerance.incremental_vs_full_fp32_max_abs_logp: 0.001`.
- `tests/reft/test_p2_1c.py` adds the regression test `test_t1_amended_uses_fp32_semantic_check`.
- The Slurm scripts add the steps.

**Unchanged outputs.**
- No replay or analysis output is regenerated: `analysis_r1` re-reads the same replay outputs.
- The float32 cast is in memory, inside the `cvf32` job only. The BF16 weights hash is verified before the cast.

## 4. Rerun

The CPU tests and `freeze-r1`, then `cvf32` (GPU), then `analyze-r1` (CPU), as `afterok` jobs with
`--kill-on-invalid-dep`.

**Reporting.** The result reports both:
- the pre-registered verdict, `P2_1C_ENGINEERING_BLOCKED`, from `analysis.json`;
- the amended verdict from `analysis_r1.json`.
