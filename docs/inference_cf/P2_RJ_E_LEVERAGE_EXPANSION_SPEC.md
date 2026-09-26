# P2-RJ-E leverage expansion — frozen spec v1

**Stage:** P2-RJ-E is the **final diagnostic stage** of the inference-time steering programme.

- It is a sample-size expansion of **one** already-frozen P2-RJ statistic, the leverage fraction
  Q50.
- It is not method development and creates no second endpoint.
- **After it, the diagnostic programme ends, whatever the outcome.**
  - No P2-RJ-F.
  - No further Q50 expansion.
  - No new Jacobian, random-control or localization diagnostic.

**Frozen before any P2-RJ-E GPU outcome:**

| Artifact | Path |
|---|---|
| Config | `configs/inference_cf/p2_rj_e_leverage_expansion.json` |
| Code | `experiments/inference_cf_p2rje{,_analyze,_audit}.py`. It reuses the P2-RJ runner and P2-R primitives read-only, and edits no frozen file. |
| Positions | `results/inference_cf/p2rje/positions.json` |
| Slurm | `slurm/inference_cf_p2rje.sbatch` |
| Tests | `tests/test_inference_cf_p2rje.py` |

## 1. Fixed history (not rewritten)

| Stage | Standing |
|---|---|
| P2 | `P2_VALID_CONFIGURATION_SELECTED`, `P2_AUDIT: PASS` |
| P2-R | `P2_R_MECHANISM_STILL_AMBIGUOUS`, `P2_R_AUDIT: PASS` |
| P2-RJ | `P2_RJ_STILL_AMBIGUOUS`, `P2_RJ_AUDIT: PASS`: alignment `MISALIGNED`, leverage `UNRESOLVED` |

**H1 is established** under every admissible leverage class: d = norm(hE − hB) captures κ₊ ≈ −1% of
the available first-order lexical leverage. H1 is **not** retested here.

The only open question is whether L16 leverage at the fixed P2 budget e\* is **SUBSTANTIAL** or
**WEAK** (H4).

## 2. Primary quantity (unchanged P2-RJ definition)

```math
Q_{50}=P_{\text{dialogue-weighted}}\!\left[\rho\ge 0.5\right],\qquad
\rho=\frac{\|\nabla m_{tan}\|\,e^*}{\text{gap}},\qquad \text{gap}=-m(r_0).
```

- **Margin:** m = logsumexp z(Y_ref) − z(c\*) on processed logits.
- **Gradient:** g = ∂m/∂r at the exact unedited L16 DG-02 state, read by the zero-valued
  `SiteGradientProbe`.
- **Tangent:** g_tan = g − r̂⟨r̂, g⟩.
- **Energy:** e\* = 1.1260757575454359.
- **Bootstrap:** dialogue resampling with 10,000 replicates and seed 240924. Positions are
  averaged within each dialogue, then across dialogues. Intervals are percentile intervals.
- **Multiplicity:** each bound at α = 0.025, as in the P2-RJ Bonferroni decision family of 2.

**Leverage class (P2-RJ `leverage_class`, unchanged):**

| Class | Rule |
|---|---|
| WEAK | hi(Q50) < 0.5 |
| SUBSTANTIAL | lo(Q50) > 0.5 |
| UNRESOLVED | otherwise |

## 3. Population (complete, frozen)

**Rule.** Take **all** structurally valid EN-confusion positions of the 300-utterance
D-dev-select R2/P2 panel. Eligibility is the unchanged P2-R code:
`inference_cf_p2r_population.build`, candidates *before* the round-robin selection.

- Enumeration intercepts `select` and does not modify the frozen P2-R module.
- Rebuilding must reproduce the frozen P2-R population hash (`sha256:6a880d2b…`); verified.

**Resulting population:**

| Count | Value |
|---|---|
| Eligible EN-confusion positions | **227**: the 60 original P2-RJ positions (all contained, recomputed here) + **167 new** |
| Utterances / dialogues | 43 / **17** |
| Duplicates | 0 |
| Selection | none; every eligible position enters the estimator exactly once |

**Exclusions** (371 candidate units; P2-R ledger reasons):

| Reason | Units |
|---|---|
| mid-character prefix | 45 |
| not a token start | 36 |
| multiple reference units at the position | 30 |
| R2-unalignable | 22 |
| forced-prefix first content token | 11 |

No position is dropped by gap, gradient, gate or any outcome.

**Pre-outcome limitation (disclosed).** All eligible positions lie in the **same 17 dialogues**
as the original 60, because EN-confusion units in this panel occur only in these dialogues. The
bootstrap unit is the dialogue. The expansion therefore sharpens within-dialogue estimates but
cannot add dialogues, and the interval may remain wide. If it still contains 0.5, the frozen
terminal state is `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`.

## 4. Competitor for new positions (same rule, no change)

- **The rule.** P2-RJ fixed c\* as the first non-Y_ref token in the P2-R saved unedited top-20
  list, i.e. `torch.topk(log_softmax(processed logits), 20)` order.
- **New positions.** No P2-R list exists, so the **identical rule** is applied to the identical
  unedited B-branch logits in a no-grad pre-pass, before any gradient.
- **Original 60.** The frozen c\* is used and must equal the rule's value.
- **Ties.** A tie between c\* and another non-reference token is recorded. At a tie, m(r₀) is
  unchanged (P2-RJ §12).

## 5. Validity (P2-RJ V1–V7, instantiated for this population)

**Run validity (V1–V5).** A failure here is an implementation or infrastructure invalidity: no
diagnosis is issued, and at most one versioned rerun is allowed, preserving the failed attempt.

- **V1:** manifest, config, positions and source hashes, and the Git commit.
- **V2:** every row ok in both passes; all 227 positions present.
- **V3, all positions:**
  - the decode-rule argmax equals the baseline token;
  - grad-step logits are bitwise equal to the unedited step;
  - the probe site equals the recorder site;
  - the best non-reference token equals c\*, or ties exactly;
  - the gradient-path margin equals the summary-path margin within 1e-3.
- **V3, the 60 original positions, additionally:**
  - the P2-RJ `state_checks` against P2-R run3;
  - **overlap reproduction against P2-RJ run1 bf16:** A, ‖g_tan‖ and gap within 1e-6 relative,
    and the margin-gradient cosine ≥ 0.9999.
  - More than 5% of positions mismatching means the run is invalid.
- **V4:** gradients are finite and nonzero.
- **V5:** at least 40 valid positions over at least 12 dialogues.

**Carried over and precision (V6, V7):**

- **V6:** first-order validity is carried over from P2-RJ, where r = 0.918 and slope 1.03 at the
  same site, energy and protocol. No perturbation is run.
- **V7:** the float32 replicate must give a median g_tan cosine ≥ 0.9 **and** the same leverage
  class. Otherwise the result is `SITE_UNRESOLVED` (terminal); this is not a rerun trigger.

## 6. Final diagnosis (exactly one)

| State | Condition | Interpretation |
|---|---|---|
| `P2_RJ_E_DIRECTION_ISSUE` | valid, SUBSTANTIAL (bf16 = fp32) | L16 has meaningful bounded leverage at e\*, but hE − hB is nearly orthogonal to it. The revision should target direction construction. |
| `P2_RJ_E_DIRECTION_AND_SITE_ISSUE` | valid, WEAK (bf16 = fp32) | The direction is bad **and** L16 at e\* lacks broad lexical leverage. The actuator needs a coherent direction/site or representation redesign. |
| `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED` | valid, and UNRESOLVED or precision-fragile | **Terminal.** H1 is established; H4 cannot be resolved with the current exposed role. Redesign proceeds with site uncertainty recorded. |

**Descriptive only (H1 is not reopened):**

- median gap, ‖g‖, ‖g_tan‖, A and ρ;
- the fractions with ρ ≥ 0.1, 0.5 and 1;
- κ(+d), κ(−d), κ(random) and cos(g_tan, d) on the expanded population;
- the original-60 versus new-167 split.

## 7. Constraints

- **The gradient is evaluator-only.** It is never applied as a steering direction, and no model
  forward carries a nonzero edit.
- **No other runs:** no free decoding, WER/MER, gate ablation, direction intervention, new site,
  layer or α.
- **Data role:** D-dev-select only. There is no D-dev-confirm, D-test, P3 or transfer access, and
  `router-calib` is not used.
- **Compute:** one Slurm job on MIG 3g.40gb with a 1-hour limit; bf16 then fp32 passes. At most
  one versioned rerun, and only for recorded invalidity. There is never a rerun because a result
  is unfavourable.
