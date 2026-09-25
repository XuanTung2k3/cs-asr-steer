# P2-A attempt 1 (jobs 54803/54804) — `P2_BLOCKED_INVALID_EXPERIMENT` (baseline execution mismatch)

**Runs.**

| Job | Layer | Status | Output |
|---|---|---|---|
| 54803 | L16 | COMPLETED, 45 min | `results/inference_cf/p2_A_L16/` |
| 54804 | L24 | COMPLETED, 45 min | `results/inference_cf/p2_A_L24/` |

Commit `339896b`; manifests `04c9dc93…` and `9090708b…`. Both have 300/300 `ok`. The artifacts are
preserved, and the evaluator summary is kept as `results/inference_cf/p2_A_summary.json`. That
evaluator output selected L16 α=2.0, but **this result is invalid and must not be used**. The
selection file is kept only as `p2_A_attempt1_selection_INVALID.json`.

## What the audit found

**The baseline did not match the method's execution.** The P2 spec defined B0 as the standalone
`cached_greedy` decode. The steered systems run inside the three-branch `cached_decode` loop.

- **Where divergences start:** in **52 of the 60** utterances where a steered configuration
  diverges from B0 (L16, α=0.5), the method's *own unsteered* B-branch argmax already differs
  from B0 at the divergence step.
- **No edit involved:** several of these diverge with **no edit** at or before the divergence.
  For example, `ZH-CN_U0011_S0_91` diverges at t=0: B0 gives '真的…', while the B branch
  **and** ordinary Whisper `generate()` (R2) both give '那真的…'.
- **The largest damage source is not a steering effect:** `ZH-CN_U1003_S0_141` reaches the
  200-token cap in every configuration (+87 ZH errors). It starts at t=53, where the unsteered B
  branch already chooses '这' but B0 chooses '相'.
- **What the deltas mix:** steering effects and baseline-execution differences. Both code paths
  are internally deterministic: B0 was identical across the two jobs, and the U1003 B-branch
  choice was identical across all six configurations. They are nonetheless different bf16
  computations, and near-tie argmax flips separate them. The CE population (10 utterances) did
  not contain such a flip.
- **Rejected explanation:** `generate()` does not mutate `generation_config` (verified).

## Handling

- **Record:** attempt 1 is `P2_BLOCKED_INVALID_EXPERIMENT` and is not used for selection.
- **Fix (P2 spec v1.1), implementation only, no rule change:**
  - The matched baseline for each layer is the **α=0 run of the identical `cached_decode`
    path**, which CE1 proved bitwise equal to its B branch.
  - `cached_greedy` stays only as a reported diagnostic.
- **New experiment-validity check (frozen before the rerun):** every method-vs-baseline
  divergence must occur at or after an applied edit. Any divergence without a prior edit
  invalidates the experiment.
- **Unchanged:** grids, dose maps, layers, detector, thresholds and the selection/tie rule.
- **Rerun:** versioned directories `p2_A_r1_L16` and `p2_A_r1_L24`.
