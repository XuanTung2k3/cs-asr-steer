# S0/S1 — pre-registered decision rules

These rules were written before any R3 or M1 training outcome existed. They are implemented in
`reft/arch_s0s1_evaluate.py` (`gain`, `guard`, `corrections`, `strong`, `improvement_over_r2`, `decide`, `RECOMMEND`)
and unit-tested in `tests/reft/test_arch_s0s1.py`. The constants are in `configs/reft/arch_s0s1_frozen.yaml`
(`decision`).

## Inputs

**Checkpoints:**
- R3_s and M1_s: selected by the unchanged P2-1 rule;
- R2_s: the historical P2-1b `KL_seed{s}` selected checkpoints, recomputed with the same code. Recomputation must
  reproduce the P2-1b evaluation, and values are read from the artifacts, never hard-coded.

**Evaluation:** dev150, frozen-LID beam-5, against the pinned B0. s ∈ {42, 43, 44}.

**Definitions** (P2-1b / P2-2 code):

| Term | Definition |
|---|---|
| `zh_new` | new Mandarin errors (B0-correct Mandarin reference units now wrong); retention = 1 − zh_new / 4950 |
| `sb` | long switch-back dropped units (P2-1D/P2-1b definition: runs ≥ 5) |
| `en_net` | substantive English net corrections against B0 |
| `cs_mer` | genuine-CS MER |
| `worst_dlg` | the largest per-dialogue increase in mixed errors against B0 |
| loops, nonterm | from the canonical scorer |

mean(·) is the mean over the three seeds. All thresholds are internal screening effect sizes, not significance tests.

## A — M1 architectural advantage over R3 (all required)

| Condition | Requirement |
|---|---|
| A1 | mean zh_new(M1) ≤ mean zh_new(R3) − 5 |
| A2 | zh_new(M1_s) < zh_new(R3_s) in ≥ 2 of 3 seeds |
| A3 | mean sb(M1) ≤ mean sb(R3) − 5 |
| A4 | sb(M1_s) < sb(R3_s) in ≥ 2 of 3 seeds |

Absolute and relative changes are reported.

## B — existing-best guard against R2-GoldKL (all required)

| Condition | Requirement |
|---|---|
| B1 | mean zh_new(M1) ≤ mean zh_new(R2) |
| B2 | mean sb(M1) ≤ mean sb(R2) |
| B3 | Σ loops(M1) ≤ Σ loops(R2) + 1 and Σ nonterm(M1) ≤ Σ nonterm(R2) + 1 |

## C — English correction preservation (all required)

| Condition | Requirement |
|---|---|
| C1 | en_net(M1_s) > 0 in ≥ 2 seeds |
| C2 | mean en_net(M1) / mean en_net(R2) ≥ 0.75 (undefined if the R2 mean is ≤ 0 → fail) |
| C3 | mean cs_mer(M1) − mean cs_mer(R2) ≤ 0.005 **and** mean cs_mer(M1) − mean cs_mer(R3) ≤ 0.005 |
| C4 | for every seed, worst_dlg(M1_s) ≤ worst_dlg(R3_s) + 10 |

## D — strong preservation (reported, outside the verdict)

zh_new ≤ 49, i.e. retention ≥ 0.99, in ≥ 2 seeds. It is reported separately for M1, R3 and R2. It never selects
checkpoints.

## Meaningful improvement over R2-GoldKL (G_R3 for R3, G_M1 for M1)

The same conventions as A and C, with R2 as the reference and the comparison arm. For X ∈ {R3, M1}, all of the
following are required:

| Part | Requirement |
|---|---|
| Gain | mean zh_new(X) ≤ mean zh_new(R2) − 5; zh_new(X_s) < zh_new(R2_s) in ≥ 2 seeds; mean sb(X) ≤ mean sb(R2) − 5; sb(X_s) < sb(R2_s) in ≥ 2 seeds |
| No correction loss | en_net(X_s) > 0 in ≥ 2 seeds; mean en_net(X) / mean en_net(R2) ≥ 0.75; mean cs_mer(X) − mean cs_mer(R2) ≤ 0.005; worst_dlg(X_s) ≤ worst_dlg(R2_s) + 10 for every seed |
| No loop worsening | Σ loops(X) ≤ Σ loops(R2) + 1 and Σ nonterm(X) ≤ Σ nonterm(R2) + 1 |

## Integrity (all required, else ENGINEERING_BLOCKED)

- reload verification all ok (6 runs × 3 epochs, 20/20 each, hashes and parameter counts);
- six runs with epochs 1–3 and 96 updates each;
- epoch-0 identity 150/150 for every run;
- backbone weights unchanged;
- trainable parameters 7,683 (R3) and 7,682 (M1);
- R2-GoldKL recomputation reproduces the P2-1b evaluation (selected epoch, zh_new, sb, worst dialogue, EN net, CS MER,
  MER) for all three seeds;
- per epoch, supervised tokens, GoldKL mask positions and update counts equal the matched R2-GoldKL run;
- M1's initial `base` equals R2-GoldKL of the same seed;
- orthogonality error ≤ 1e-5 at every step.

## Terminal verdict (strict precedence; no overlap)

1. `ARCH_S1_ENGINEERING_BLOCKED`: integrity fails.
2. `ARCH_S1_M1_PROMISING`: A ∧ B ∧ C.
3. `ARCH_S1_M1_BEATS_R3_ONLY`: A ∧ C ∧ ¬B.
4. `ARCH_S1_R3_CAPACITY_GAIN`: ¬(A ∧ C), i.e. M1 fails its gain test, and G_R3.
5. `ARCH_S1_NO_ARCHITECTURAL_GAIN`: ¬(A ∧ C) ∧ ¬G_R3 ∧ ¬G_M1. Neither architecture shows a meaningful, stable
   improvement over R2.
6. `ARCH_S1_MIXED_OR_INCONCLUSIVE`: otherwise. The only case left is ¬(A ∧ C) ∧ ¬G_R3 ∧ G_M1: M1 improves over R2 but is
   not separable from R3, while R3 does not improve over R2.

Every component is reported. If R3 improves and M1 is mixed, both findings stay in the component report. The verdict is
never upgraded by hand, and thresholds are never changed after outputs are seen.

## Recommendation (one; it is not authorization)

| Verdict | Recommendation |
|---|---|
| M1_PROMISING | `GO_M1_MECHANISM_AND_INDEPENDENT_VALIDATION`: controlled acoustic-dependence checks, readout ablations, a strong optimized LoRA baseline, independent dialogue-disjoint evaluation, compute comparison |
| M1_BEATS_R3_ONLY | `NO_NEW_BEST_METHOD`: keep the architectural observation; R2-GoldKL is not replaced |
| R3_CAPACITY_GAIN | `CAPACITY_BOTTLENECK_MORE_PLAUSIBLE`: no cross-attention-conditioned benefit is claimed; further architecture work needs a new explicit hypothesis, not automatic rank expansion |
| NO_ARCHITECTURAL_GAIN | `STOP_GENERIC_LOREFT_ARCHITECTURE_EXPANSION`: no rank-4/8, layer sweeps or E1+D2 by default; consider a separately scoped direction (phonetic/acoustic discrimination or decoder-prefix formatting robustness) |
| MIXED_OR_INCONCLUSIVE | `HOLD_ARCHITECTURE_CLAIMS`: explain the source of uncertainty (seed variability, 20 dialogues, optimization differences, conflicting metrics) |
| ENGINEERING_BLOCKED | `HOLD_FOR_ENGINEERING_REPAIR` |

**Interpretation guards:**
- M1 beating R2 but not R3 gives no support for a cross-attention-conditioned advantage beyond capacity.
- M1 beating R3 but losing to R2 is an architecture comparison, not a replacement.
- R3 beating M1 rejects this M1 parameterization under the frozen recipe, not acoustic conditioning in general.

The mechanism checks (contribution audit, P2-1c cases) are exploratory and do not enter the verdict.
