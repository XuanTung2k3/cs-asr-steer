# ST-PROMPT-R1-A — decoder layer × relative dose × sign single-pulse screen (report, terminal)

**Verdict: `ST_PROMPT_R1_A_MARGIN_ONLY`.**
- Independent audits: `PASS_TO_ST_PROMPT_R1_A`, then `ST_PROMPT_R1_A_AUDIT: PASS (PRIMARY)` before references, then
  `ST_PROMPT_R1_A_AUDIT: PASS (FULL)`. The auditor reproduced the label and the empty selection independently.
- Two arms have both adjusted lower bounds above zero, against NONE and against the matched random direction:
  L16 η = 0.45 minus and L24 η = 0.30 minus.
- No arm meets the full benefit gate of ≥ 3 corrections across ≥ 3 dialogues. The best arm makes 1 correction.
- Selection is empty, so **R1-B is NOT authorized and was not run.**

| Step | Commit / job | Result |
|---|---|---|
| Freeze | `e965a7f` | pre-outcome |
| Implementation | `0ad5374` | 11 focused tests; contract + ST-LOC0/DG-02/LSS/P2-DIR/P2-R/R0 regression 131/131 |
| Plan | `bff8fcf` | 13/13 prepare checks |
| Pre-run audit | `0bc533b` | `PASS_TO_ST_PROMPT_R1_A` (46 checks) |
| Manifest (plumbing child; job ran at `0bc533b`) | `b8a82e2` | pushed before the job |
| Run | Slurm 58158 | 6 min 20 s: capture 90 s → geometry gate → barrier 65 s → pulses 205 s |
| Amendment A1 (user-authorized, before references) | `03a2289` | see below |
| Output seal (5 batches) | `3cb167d` … `3072361` | pushed before references |
| Primary-phase audit | `9ff9e74` | PASS (24 checks) |
| Reference evaluation | `343ebac` | MARGIN_ONLY |
| Full audit | `69fdbd6` | PASS (11 checks) |

## Setup

**Direction.** `core_p1.direction(h_E, h_M)`, built from the same audio, the same B0M_L16 content prefix and the same query.
The only difference is the forced language token (50259 vs 50260).
- Float64 subtraction, then float32 unit vector.
- The ± signs negate the same vector.
- No corpus-derived, gold-informed or learned direction.

**Site and layers.** DG-02 post-cross-attention / pre-FFN residual r, at zero-based decoder layers 3, 8, 16 and 24.

**Dose.** Target ‖Δ‖ = η·‖float64(native r)‖ with η ∈ {0.15, 0.30, 0.45}, solved from the query's own site with the
unchanged P2-R solver.
- NormPreserve repair via the frozen `apply_steering`; at most 8 evaluations.
- The edit is not α = η.

**Matrix and controls.** 24 primary arms, each a single pulse from the pristine pre-query cache. Controls:
- NONE (180 cells).
- 12 fixed PCG64 random directions, one per layer and dose, matched to the same relative target.
- Historical D2 L16 at e* = 1.12608 (apparatus only).
- Zero dose at all four layers (720 cells).

**Population.** The frozen panel `sha256:41168f57…`: 180 positions (60 EN-confusion / 60 EN-correct / 60 ZH-correct) from
80 utterances and 20 dialogues. Already-exposed D-dev-select only.

**Firewall.** The runner received only `runtime_queries` and `utterances`. The geometry-coverage gate ran as a separate
process and read offline strata only to count reachability. P2-RJ reference sets and fixed competitors were opened only after
the seal was pushed and the PRIMARY audit had passed.

## Historical reproduction — PASS

- **B0:** L16 forced-ZH and forced-EN states, and the NONE packed logits, are bitwise equal to the sealed ST-LOC0
  calibration on all 180 positions.
- **D0:** all 180 L16 v_prompt vectors are hash-identical to the historical D0 records.
- **Original L16 pulses:** ST-LOC0 v_prompt L16 plus and minus, and D2, at the historical absolute e* reproduce bitwise on
  540 of 540 cells. This covers both the logits and the consumed states. The solver scale is within 1e-9 and edit norms are
  equal.
- **Zero dose:** bitwise baseline at L3/L8/L16/L24 on 720 of 720 cells.
- **Cache and restore:** the prefix-cache fingerprint and positions are unchanged after every one of the 6,480 cells.
  Restore and clean replay are bitwise. Weights are unchanged, grads are None, and no hooks leaked.

## Dose validity and amendment A1

The pre-pulse CPU geometry ledger found all 36 arms reachable on 180/180 cells (gate passed).

In the run, all 6,480 primary and random cells were steered; there were 0 unreachable or no-edit cells. Realized η lay within
[0.1497, 0.1503], [0.2996, 0.3003] and [0.4495, 0.4504] for the three doses. Every frozen criterion passed on every cell:
- η and squared-energy relative error ≤ 0.02;
- solver vs hook ≤ 1e-6;
- consumed vs proposed ≤ 0.005;
- same-target pairwise squared energy ≤ 0.02.

**Amendment A1.** I had added a check to the runner that is not in the frozen config: |‖r′‖ − ‖r‖| ≤ 0.5%·‖r‖.
- It flagged 189 of 6,480 cells (maximum 0.68%, median among flagged 0.55%), and the runner exited with Slurm state FAILED.
- The same deviation is already present in the frozen `apply_steering` NormPreserve output itself, at native bf16 precision.
  It is not a hook or consumed-energy violation.
- After the job and **before any reference access**, the user explicitly chose to treat this flag as a recorded diagnostic.
  Validity now means "steered and all frozen checks", and pairwise energy is recomputed over those cells. The record is
  `results/inference_cf/st_prompt_r1/amendment_A1.json`.
- The raw runner records are unchanged.
- The independent auditor recomputed the diagnostic (189 cells above 0.005, maximum 0.0068) from the saved native arrays.

## Results — all 24 primary arms

- **EN Δm:** EN-confusion dialogue-macro Δmargin, m = logsumexp(z[Y_ref]) − z[c_fixed] on processed logits, with the
  adjusted vs-NONE interval (Bonferroni family of 48, quantiles 0.00052 / 0.99948).
- **vs random:** paired candidate − matched-random contrast, with its adjusted lower bound.
- **Validity:** every arm had 60/60/60 valid cells.

| Layer | η | Sign | EN Δm [adj. CI] | vs random (adj. lower) | EN corrections (dialogues) | top-1 changes | EN-correct corruptions | ZH-correct corruptions | KL conf / ZH |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.15 | + | +0.010 [−0.063, +0.106] | +0.005 (−0.082) | 0 (0) | 0 | 0 | 0 | 0.004 / 0.000 |
| 3 | 0.15 | − | +0.089 [−0.078, +0.401] | +0.083 (−0.085) | 0 (0) | 2 | 1 | 0 | 0.005 / 0.000 |
| 3 | 0.30 | + | +0.034 [−0.128, +0.279] | −0.009 (−0.171) | 0 (0) | 4 | 0 | 0 | 0.026 / 0.001 |
| 3 | 0.30 | − | +0.263 [+0.041, +0.862] | +0.219 (−0.062) | 0 (0) | 4 | 1 | 1 | 0.148 / 0.001 |
| 3 | 0.45 | + | +0.114 [−0.193, +0.610] | −0.059 (−0.296) | 0 (0) | 5 | 0 | 1 | 0.102 / 0.001 |
| 3 | 0.45 | − | +0.535 [+0.119, +1.234] | +0.363 (−0.074) | 0 (0) | 8 | 2 | 1 | 0.322 / 0.003 |
| 8 | 0.15 | + | −0.016 [−0.081, +0.064] | −0.001 (−0.089) | 0 (0) | 1 | 0 | 0 | 0.004 / 0.000 |
| 8 | 0.15 | − | +0.068 [+0.008, +0.139] | +0.083 (−0.013) | 0 (0) | 5 | 0 | 0 | 0.003 / 0.000 |
| 8 | 0.30 | + | −0.010 [−0.163, +0.170] | −0.126 (−0.333) | 0 (0) | 4 | 0 | 0 | 0.019 / 0.001 |
| 8 | 0.30 | − | +0.258 [+0.074, +0.523] | +0.142 (−0.064) | 0 (0) | 5 | 2 | 1 | 0.022 / 0.001 |
| 8 | 0.45 | + | +0.054 [−0.197, +0.319] | −0.132 (−0.363) | 0 (0) | 4 | 1 | 1 | 0.052 / 0.002 |
| 8 | 0.45 | − | +0.561 [+0.184, +1.217] | +0.375 (−0.025) | 0 (0) | 8 | 2 | 1 | 0.147 / 0.002 |
| 16 | 0.15 | + | −0.009 [−0.090, +0.069] | −0.054 (−0.191) | 0 (0) | 1 | 1 | 0 | 0.006 / 0.001 |
| 16 | 0.15 | − | +0.110 [+0.037, +0.197] | +0.065 (−0.019) | 0 (0) | 3 | 0 | 0 | 0.009 / 0.001 |
| 16 | 0.30 | + | +0.044 [−0.129, +0.235] | −0.218 (−0.417) | 0 (0) | 3 | 1 | 1 | 0.024 / 0.003 |
| 16 | 0.30 | − | +0.289 [+0.147, +0.452] | +0.027 (−0.208) | 0 (0) | 5 | 0 | 1 | 0.050 / 0.002 |
| 16 | 0.45 | + | +0.216 [−0.077, +0.535] | −0.026 (−0.340) | 0 (0) | 4 | 3 | 2 | 0.062 / 0.007 |
| **16** | **0.45** | **−** | **+0.632 [+0.404, +0.884]** | **+0.390 (+0.073)** | 0 (0) | 8 | 2 | 1 | 0.160 / 0.005 |
| 24 | 0.15 | + | −0.038 [−0.247, +0.143] | −0.159 (−0.400) | 0 (0) | 2 | 2 | 1 | 0.017 / 0.007 |
| 24 | 0.15 | − | +0.270 [+0.106, +0.495] | +0.149 (−0.079) | 0 (0) | 6 | 1 | 1 | 0.024 / 0.005 |
| 24 | 0.30 | + | +0.137 [−0.342, +0.543] | −0.181 (−0.586) | 0 (0) | 5 | 2 | 4 | 0.079 / 0.031 |
| **24** | **0.30** | **−** | **+0.750 [+0.427, +1.183]** | **+0.433 (+0.100)** | 1 (1) | 11 | 1 | 2 | 0.107 / 0.019 |
| 24 | 0.45 | + | +0.572 [−0.164, +1.230] | −0.442 (−1.195) | 1 (1) | 10 | 3 | 5 | 0.216 / 0.083 |
| 24 | 0.45 | − | +1.479 [+0.958, +2.125] | +0.466 (−0.262) | 1 (1) | 15 | 5 | 3 | 0.279 / 0.046 |

**Controls.**
- **NONE:** Δ = 0, and zero dose is bitwise baseline.
- **Matched random** (EN-confusion macro Δm, pointwise 95%), by layer at η 0.15 / 0.30 / 0.45:
  - L3: +0.006 / +0.043 / +0.173.
  - L8: −0.015 / +0.116 / +0.185.
  - L16: +0.045 / +0.262 / +0.242.
  - L24: +0.122 / +0.317 / **+1.014**. L24 η 0.45 random also corrects q44 (CSD0532).
- **D2** (historical apparatus): reproduces its sealed ST-LOC0 cells bitwise.

**Oracle union across all 24 primary arms** (reference-dependent, not deployable): **2 corrected positions in 2 dialogues.**
- q25 (CSD0501), corrected by L24 η 0.30 − and L24 η 0.45 −.
- q45 (CSD0540), corrected by L24 η 0.45 +.

**Statistics.**
- Dialogue-cluster bootstrap: B = 10000, seed 240924, 20 sorted dialogues shared across all endpoints. All endpoints had
  10,000 usable draws.
- Bonferroni family of 48 (24 arms × {vs NONE, vs random}).
- The independent auditor matched every estimate and adjusted bound to ≤ 1e-9, and matched all correction and corruption
  counts.

**Gates.**

| Gate | Result |
|---|---|
| ≥ 3 corrections | 0/24 arms (maximum 1) |
| ≥ 3 correction dialogues | 0/24 |
| Macro ≥ 0.50 nat | 6 arms: L3 0.45−, L8 0.45−, L16 0.45−, L24 0.30−, L24 0.45+ (+0.572, but adjusted lower < 0), L24 0.45− |
| Adjusted lower > 0 vs NONE | 11 arms |
| Random point > 0 and adjusted lower > 0 | 2 arms (L16 0.45−, L24 0.30−) |
| Corruption budget (≤ 5 total, ZH ≤ 2) | fails for L24 0.30+, L24 0.45+ and L24 0.45− |

Precedence: no arm passes benefit; two arms have both adjusted lower bounds > 0. Result: **MARGIN_ONLY**.

## Interpretation

1. **Material causal lexical repair? No.**
   - v_prompt shifts the fixed-competitor margin, and does so beyond matched-energy random at L16 η 0.45 − and L24 η 0.30 −.
   - It almost never flips the top-1 to a reference token: at most 1 correction per arm and a 2-position union.
2. **Does strength explain the earlier weak ST-LOC0 results? Partly, for the margin only.**
   - The margin grows monotonically with dose for the minus sign at every layer. At L24 it goes +0.27 → +0.75 → +1.48 nat.
   - Corrections stay at 0–1 even at η 0.45, about three times the historical relative edit.
   - Collateral damage grows with dose: KL rises, and EN-correct and ZH-correct corruptions reach 5 and 3 at L24 η 0.45 −.
3. **Layer dependence. Yes.** Effects are larger at L24 > L16 ≈ L8 ≈ L3. Matched random effects also grow at L24, with
   +1.01 nat at η 0.45, so much of the large-dose L24 shift is non-specific. Only L24 η 0.30 − and L16 η 0.45 − exceed random
   with adjusted confidence.
4. **Sign.** The informative sign is consistently minus, i.e. toward the forced-ZH prompt state. This matches ST-LOC0, where
   v_prompt minus was the positive sign. The plus arms are flat or negative relative to random.
5. **Power only with damage?** The largest margin (L24 η 0.45 −, +1.48) comes with 8 correct-state corruptions and still makes
   only 1 correction. There is no correction power even at the cost of damage.
6. **Sequence transfer:** not tested. R1-B was not authorized.

Limitations:
- This is exposed development data with selection bias.
- The evaluation is single-step on teacher-forced baseline prefixes.
- The oracle union depends on references.
- Amendment A1 is disclosed above.

METHOD_CONTRACT/v6 and all historical verdicts (ST-LOC0 LOCAL_EFFECT_ONLY, R0 ORACLE_CONSTRUCTION_INSUFFICIENT,
P2-A2-MECH0) are unchanged.

## Compute

- One A job, Slurm 58158, on an H100 MIG 3g.40gb: 6 min 20 s wall time.
  - Capture: 89.5 s, 6,318 forwards.
  - Pulse phase: 280 s, 14,418 forwards.
- Pulse calls: 7,740 (6,480 primary/random + 540 barrier + 720 zero dose); 14,141 solver evaluations; 6,840 cache
  fingerprints.
- Peak VRAM: 3.97 GB allocated / 4.37 GB reserved.
- No LID calls, no autograd, no training.

Artifacts: `results/inference_cf/st_prompt_r1/` (plan, prerun, `runA/`, `amendment_A1.json`, primary analysis, `output_seal_A`,
primary audit, `secondary_analysis_A`, `final_audit_A`).
