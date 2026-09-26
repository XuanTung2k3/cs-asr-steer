# P1 causal-acceptance report

**P1 verdict: `P1_CAUSAL_ACCEPTANCE_PASS`** (spec v1.1, job 54781).
**POST_P1_AUDIT: PASS** (`P1_POST_RUN_AUDIT.md`).
**Project state: `READY_FOR_P2_COMPACT_DEVELOPMENT`.**

What passing means: the gate-driven, same-prefix counterfactual hidden-state intervention is
computationally and causally valid in the frozen Whisper model. It says **nothing** about ASR
improvement, optimal layer or alpha, or generalization.

## Frozen method exercised

- **Gate:** the R2-selected `g_old = E·R_B` (R2 `R2_CF_FEASIBLE_NOT_PREFERRED`, job 54758, manifest
  `sha256:e8869f0c…`), with max 1.0 s alignment-head window, LS-B, BC-B, partition
  `sha256:7daa5009…` and R2 fallbacks, all unchanged.
- **Direction:** `d = (hE − hB)/(‖hE − hB‖ + 1e-6)`, where `hB = hM` (forced zh). It comes from
  unsteered same-prefix full replays; the tiny/nonfinite fallback threshold is 1e-4.
- **Site and edit:** Whisper-large-v3 decoder **layer 24**, post-cross-attention residual
  `r = q + u_source` (input to the FFN), edited by the canonical DG-02 hook:
  - `h' = NormPreserve(hB + α·g·d)`;
  - exact bypass at zero dose;
  - positions < 4 never edited.
- **Doses:** α ∈ {0, 1.0}; 1.0 is an engineering value frozen in spec v1 before any run.
- **Decode:** greedy with `generate()` suppression. Stored edits are re-applied at their own
  positions each step, which equals KV-cached steering.
- **Population:** 10 D-dev-select utterances (R2 panel positions 0, 30, …, 270).

## Runs

| Attempt | Job | Manifest | Commit | Result |
|---|---|---|---|---|
| 1 (spec v1) | 54770, COMPLETED, 51 s | `sha256:4e9d12fa…` | `c272606` | `P1_BLOCKED`: A6 instrument compared bf16 edits with a float64 reference; 2 sub-resolution edits failed. Preserved in `results/inference_cf/p1/` and `P1_ATTEMPT1_INSTRUMENT_DEFECT.md`. |
| **2 (spec v1.1)** | **54781**, COMPLETED 0:0, 53 s wall, 37.0 s program, peak 3.42 GB, H100 MIG 3g.40gb | `sha256:6e595613…` | `d704db9` | **`P1_CAUSAL_ACCEPTANCE_PASS`**; `results/inference_cf/p1_r1/` |

Both attempts produced identical per-step statistics, so the GPU path is deterministic across
jobs. Total P1 GPU time is about 0.03 GPU-h.

## Acceptance (v1.1)

| # | Criterion | Evidence |
|---|---|---|
| A1 | α = 0 identity | F3 (hook installed) logits **bitwise equal** to unsteered F1 at all 190 α = 0 steps; α = 0 tokens equal the unsteered argmax |
| A2 | Finite, varying direction | 374/374 steps valid, `‖hE−hB‖` 1.37–8.15; minimum consecutive cosine 0.095 |
| A3 | Tiny-direction fallback | Unit-tested; 0 runtime occurrences |
| A4 | Real edit | 20 edited steps (23 `g>0` steps at α = 1; 3 blocked at the forced-prefix query); g 0.004–0.890 |
| A5 | Site | Every hook audit record is layer 24 at the current query position |
| A6 | Sign and specified edit | `cos(h'−hB, d)` ≥ 0.872 on all edits; hook edit equals the site-precision replica exactly (max relative error 0.0); float64 reference within 1.1% above bf16 resolution |
| A7 | Gate | Logged `g` equals `E·R_B` recomputed with `core_r2` on every step; 20 `mid_character` fallback steps have g = 0 and no edit |
| A8 | NormPreserve | Maximum relative norm change 0.38% (bf16); non-edited queries have zero edit |
| A9 | Prefix integrity | Every forward input is `prompt + emitted prefix`; no position < 4 edited |
| A10 | Isolation | No site hook present around the unsteered branches at any step (asserted); none leaked after the run |
| A11 | Continuation | All 10 α = 1 decodes terminated normally with valid text |
| A12 | Serialization | 20/20 records bound to the manifest, with full per-step records |
| A13 | Cost | Runtime and VRAM captured |

**Edit magnitudes.** Realized edit norm 0.017–0.860, i.e. 0.18–9.6% of the site norm.

## Diagnostics (no pass rule, no recognition claim)

- **α = 0 vs `generate()`:** α = 0 tokens equal the R2 `generate()` tokens on 9/10 utterances. The
  exception (`ZH-CN_U0029_S0_1085`) diverges at t = 6, exactly where R2's own full replay of the
  `generate()` sequence also disagreed with `generate()`. That is cached-vs-full-replay bf16
  numerics at a near-tie, independent of the hook.
- **α = 1 changes:** α = 1 changed the token sequence of 1/10 utterances (the same one). The
  change is at t = 6, an edited step (g = 0.419, six earlier edits in that utterance), where the
  steered argmax differs from the unsteered argmax. This is a causal effect of the edit; its
  correctness is not assessed.

## Limitations carried forward to P2

1. **First token not steerable.** It is predicted from the final forced-prompt position.
2. **Gate reach.** The gate targets confusion only and does not reach deletions (R2), and its
   localization on EN-confusion is weak (46% within 0.5 s).
3. **Engineering values.** Layer 24 and α = 1.0 are engineering choices; layer and dose selection
   belong to P2 under a separately frozen budget.
4. **Cost.** The O(T²) full-replay decode suits acceptance but not deployment; a KV-cached
   implementation must be shown equivalent before P2 cost claims.
5. **Baseline.** B0_AUTO remains the stronger unsteered baseline.
