# P2-RJ Jacobian H1-vs-H4 disambiguation — frozen spec v1

**Stage:** P2-RJ is a versioned, evaluator-side **diagnostic** stage. It is the single follow-up
that P2-R prescribed for its ambiguous branch (`P2_R_REPORT.md`, "Single most informative
follow-up"; Codex contract, "If ambiguous"). It is not a method-selection stage.

**Project state while it runs:** `P2_R_MECHANISM_STILL_AMBIGUOUS`, **P3 HELD**.

**Frozen before any P2-RJ outcome.** No gradient, Jacobian norm or alignment has been computed on
any position when this file is committed.

| Artifact | Path |
|---|---|
| Config | `configs/inference_cf/p2_rj_jacobian_diagnosis.json` |
| Population | `results/inference_cf/p2r/population.json` (P2-R, unchanged, `sha256:6a880d2b…`) |
| Code | `experiments/inference_cf_p2rj{,_prepare,_analyze,_audit}.py` |
| Slurm | `slurm/inference_cf_p2rj.sbatch` |
| Tests | `tests/test_inference_cf_p2rj.py` |

## 1. What P2-RJ does not change

- **P2 stays as it is:** `P2_VALID_CONFIGURATION_SELECTED`, `P2_AUDIT: PASS`. The selected
  configuration stays L16, α = 2, φ_id, g = E·R_B, +d.
- **P2-R stays as it is:** `P2_R_MECHANISM_STILL_AMBIGUOUS`, `P2_R_AUDIT: PASS`. P2-RJ reuses its
  population, states, target sets, directions and energy. It reruns no P2-R arm and relabels no
  P2-R result.
- **No steering experiment.** P2-RJ changes no gate, direction, site, layer, α, dose map or
  localizer, and it applies **no new edit** to the model.
- **The gradient is evaluator-only.** It is never applied as a steering direction, never used in a
  hooked forward with a nonzero edit, and never exposed to a deployable module.

## 2. Scientific question (only this)

At the exact frozen P2-R states and the P2 edit-energy budget e\*:

1. **Leverage (H4).** Could an optimally aligned NormPreserve edit of size e\* at L16 move the
   reference lexical margin enough, to first order?
2. **Alignment (H1).** If it could, does d = norm(hE − hB) point along that useful direction?

Out of scope: which α, layer, gate or random direction is "best".

## 3. Data roles (firewall; unchanged from P2-R §2)

| Role | P2-RJ use |
|---|---|
| `D-dev-select`: the 80 P2-R utterances, 20 dialogues (already exposed) | **Diagnostic only.** |
| `router-calib` | **Not used.** |
| `D-dev-confirm`, `D-test`, P3 populations, transfer corpora | **No access.** |

References enter only as the frozen P2-R target-token sets, which are evaluator-only. No fresh
repair-validation role exists. Any repair hypothesis frozen after P2-RJ is therefore
`REPAIR_NOT_VALIDATED_NO_FRESH_ROLE` (§10).

## 4. Population, state and site (identical to P2-R D1)

- **Positions:** all 180 P2-R D1 positions from `population.json` `d1`:
  - 60 EN-confusion (37 utterances, 17 dialogues);
  - 60 EN-correct;
  - 60 ZH-correct.
  - No position is added, removed or re-selected.
- **State:** r = q + u_source at the query that predicts token t. This is the exact DG-02
  post-cross-attention / pre-FFN site at **L16** of the unedited forced-ZH B branch. The branch is
  teacher-forced on the matched-baseline `B0M_L16` tokens with the P2-R cached-branch semantics:
  - eager attention;
  - `generate()` suppression;
  - `num_forced_prefix` = 4.
  - The forced-EN E branch runs alongside, as in P2-R, to recompute hE and d.
- **Primary precision:** bf16. These are the exact P2-R states.
- **State identity (V3)** is verified per position against the saved P2-R run3 records:
  - baseline processed argmax;
  - log p(ref) and the margin;
  - ‖r‖;
  - cos(d, r);
  - the competitor token;
  - the recomputed solver scales for all three arms.

## 5. Margin (frozen)

The reference margin at state r is normalizer-free, on processed logits z (suppression applied):

```math
m(r) = \log\sum_{y\in Y_{ref}} e^{z(y)} \;-\; z(c^*) .
```

- **Y_ref:** the frozen P2-R `target_ids` of the position. For English strata this is the
  reference-consistent first-token set; for ZH-correct it is the baseline token.
- **c\*:** the first token in the saved unedited P2-R top-20 list (`none.top_ids`) that is not in
  Y_ref.
  - For EN-confusion it is the baseline argmax. The competitor identity is fixed per position.
  - m(r₀) therefore **equals the P2-R `margin_ref_vs_competitor`**. P2-R measured the median
    EN-confusion value at −10.6 nats.
- **Gap:** gap = −m(r₀) for EN-confusion (all 60 are > 0 in P2-R). For EN-correct and ZH-correct,
  m(r₀) > 0 is a **surplus**.
- **Secondary targets** (descriptive; gradients computed in the same pass):
  - log p(ref), the P2-R L quantity;
  - log P_E and log P_M: the processed masses of the frozen Latin and Han token partitions. These
    test whether d encodes script preference rather than lexical identity.

## 6. Gradient and geometry (frozen)

**Gradient.** A zero-valued float32 probe δ is cast to the site dtype and added to u_source at the
query only. The forward is therefore value-identical to the unedited step: its logits must be
bitwise equal to the no-grad step's logits (V3). Then g = ∂m/∂δ at δ = 0, which equals ∂m/∂r.

- The step runs on a deep copy of the branch cache, so the frozen trajectory is never touched.
- It is followed by a no-grad step on the real cache.

**Tangent.** The NormPreserve edit lies on the ‖r‖ sphere, so to first order it is tangent:

```math
g_{tan} = g - \hat r\langle\hat r,g\rangle,\qquad \hat r=r/\|r\|.
```

- The **primary bound uses g_tan**.
- ‖g‖ and the radial component are co-reported.

**First-order maximum achievable change.** Under an e\*-size edit, with e\* = 1.1260757575454359
exactly as frozen in P2-R:

```math
A = \|g_{tan}\|\,e^*, \qquad \rho = A/\text{gap}\quad(\text{gap}\le 0 \Rightarrow \rho=+\infty).
```

**Arm directions (P2-R, recomputed exactly; no new edit applied):**

| Arm | Direction |
|---|---|
| `+d` | d = norm(hE − hB) |
| `−d` | −d |
| random | the frozen P2-R vector: seed `P2R-random-v1\|uid\|t`, Gaussian, projected orthogonal to r |

**Realized arm edit.** δ_a = apply_steering(r, s_a·v_a) − r, in the site dtype, where
s_a = `inference_cf_p2r.solve_scale(r, v_a, e*)`.

- s_a must equal the saved P2-R s bitwise, within a relative 1e-9.
- This is geometry on the state vector only. The model never sees δ_a.

**Per-arm quantities:**

- cos(g_tan, v_a).
- pred_a = ⟨g, δ_a⟩: the first-order predicted margin change of the realized P2-R edit.
- κ_a = ⟨g_tan, δ_a⟩ / A: the fraction of the first-order optimum that arm a captures. It lies in
  about [−1, 1]; a random direction gives about ±1/√1279 ≈ ±0.03.

## 7. Analysis (frozen; implemented in `inference_cf_p2rj_analyze.decide`)

**Bootstrap:** dialogue resampling with 10,000 replicates and seed 240924. Positions are averaged
within each dialogue, then across dialogues. Intervals are percentile intervals.

**Decision family:** Q50 and K₊, with Bonferroni-simultaneous two-sided 95% intervals (each at
α = 0.025, i.e. percentiles 0.0125 and 0.9875).

- **Q50:** the dialogue-weighted fraction of valid EN-confusion positions with ρ ≥ 0.5.
- **K₊:** the dialogue-weighted mean of κ₊ over valid EN-confusion positions.

**Leverage (H4 axis):**

| Class | Rule | Reading |
|---|---|---|
| `WEAK` | hi(Q50) < 0.5 | Most positions cannot close half the gap, even with the optimal e\*-edit. |
| `SUBSTANTIAL` | lo(Q50) > 0.5 | Most positions could close at least half the gap. |
| `UNRESOLVED` | otherwise | |

**Alignment (H1 axis):**

| Class | Rule | Reading |
|---|---|---|
| `MISALIGNED` | hi(K₊) < 0.10 | d captures < 10% of the available first-order leverage. This includes anti-alignment. |
| `ALIGNED` | lo(K₊) ≥ 0.25 | d captures at least a quarter of the first-order optimum. |
| `PARTIAL` | otherwise | |

**Diagnosis (exactly one):**

| Label | Requires |
|---|---|
| `P2_RJ_DIRECTION_ISSUE` | valid, SUBSTANTIAL and MISALIGNED |
| `P2_RJ_DIRECTION_AND_SITE_ISSUE` | valid, WEAK and MISALIGNED |
| `P2_RJ_SITE_OR_SENSITIVITY_ISSUE` | valid, WEAK, and PARTIAL or ALIGNED |
| `P2_RJ_STILL_AMBIGUOUS` | everything else |

`STILL_AMBIGUOUS` covers any validity failure and UNRESOLVED leverage. It also covers SUBSTANTIAL
with PARTIAL or ALIGNED, which would contradict P2-R's inertness at first order.

**Validity** (any failure means `P2_RJ_STILL_AMBIGUOUS`):

- **V1 — provenance.** Manifest, config, population, source hashes and Git commit verify.
- **V2 — completeness.** Every row is ok and all 180 positions are present.
- **V3 — state identity** (§4 list, with the tolerances in the config), plus the bitwise-equal
  grad-step logits.
  - Mismatching positions are excluded and counted.
  - More than 9 of 180 (5%) means the run is invalid, an implementation defect.
- **V4 — finite gradients.** Every gradient is finite and nonzero.
- **V5 — minimum population.** At least 40 valid EN-confusion positions across at least 12
  dialogues.
- **V6 — first-order sanity (finite-difference check against saved P2-R outcomes; no new edits).**
  - For every valid (position, arm), the observed change is
    Δm_obs = [log p_arm(ref) − log p_arm(c\*)] − [log p_none(ref) − log p_none(c\*)].
    It is taken from the saved P2-R arm summaries, with c\* looked up in the arm's saved top-20;
    a pair is missing if c\* is absent.
  - The pooled Pearson r(pred_a, Δm_obs) over all strata and arms needs a dialogue-bootstrap
    lower 95% bound > 0.
  - This catches sign or scale defects. Slope and r per arm and stratum are descriptive, because
    the observed P2-R changes are ~0.1 nat, near the bf16 logit resolution.
- **V7 — precision robustness.** The whole computation is repeated with the same bf16 weights
  cast to float32, on the same prefixes. The float32 pass uses analytic float64 chord edits of
  length e\* for the arms. Both conditions must hold:
  - the median over valid positions of cos(g_tan,bf16, g_tan,fp32) is ≥ 0.9;
  - the float32 diagnosis label equals the bf16 label.

**Descriptive (pre-registered, not decision inputs):**

- **By stratum:** median gap/surplus, ‖g‖, ‖g_tan‖, A and ρ, and the fractions with ρ ≥ 0.1, 0.5
  and 1.
- **Arm alignment:** cos(g_tan, ±d/random) and pred/κ per arm, with pointwise intervals;
  K₊ − K_random.
- **Span split:** span-initial vs within-span EN-confusion.
- **Correct strata (safety):** A/surplus, i.e. how much of the surplus the worst-case e\*-edit
  could remove.
- **Secondary targets:** the same quantities, plus cos(d, ∇log P_E,tan), cos(d, ∇log P_M,tan) and
  cos(g_tan, ∇log P_E,tan). These characterize what d encodes: script preference versus lexical
  margin.
- **Consistency with P2-R:** predicted versus observed log p(ref) change per arm, to explain the
  "+d weak/negative, −d and random slightly better" pattern.

## 8. Implementation constraints

- **New module:** `experiments/inference_cf_p2rj.py`, in the diagnostic namespace.
  - It imports the P2-R primitives (`random_direction`, `solve_scale`, `scaled_direction`,
    `DiagBranch`) read-only.
  - The deployable modules (`inference_cf_cached.py`, `inference_cf_p2.py`, `core*`, `sites.py`,
    `hooks.py`) are **byte-unchanged**.
- **Gradient probe:** a dedicated site probe (`SiteGradientProbe`), installed on the same
  submodules as the canonical DG-02 hook.
  - At install time and at forward time it asserts that the probe δ is exactly zero.
  - It exposes no API that accepts a direction.
  - No model forward in P2-RJ carries a nonzero edit.
- **Saved outputs:** compact per-position statistics plus bounded float32 vectors
  (r, g for the four targets, d, the random direction, and δ_a) in per-utterance `.npz` files.
  The audit recomputes every norm, cosine, ratio, κ, interval and the decision from these without
  another model run.
- **Manifest:** resolved config, source hashes, Git commit, population hash, the P2-R run3
  manifest hash, environment and completion status.

## 9. Compute and stopping

- **Compute:** one Slurm job on H100 MIG 3g.40gb, batch 1, with a 1-hour limit. It runs the bf16
  pass, then the float32 pass. The row ledger is resumable.
- **Reruns** happen only for recorded infrastructure or implementation invalidity. They go into
  versioned directories and preserve the failed attempt. There is never a rerun because a result
  is unfavourable.
- **No full decode** and no 300-utterance run.

## 10. After the diagnosis (conditional; at most one hypothesis, never executed here)

| Diagnosis | Allowed |
|---|---|
| DIRECTION | Characterize what hE − hB encodes, using the §7 secondary-target cosines. Then freeze at most one direction-construction hypothesis with a mathematical rationale. Never `−d` by default, never a family of directions, and **never the gradient itself**. |
| SITE_OR_SENSITIVITY | Freeze at most one site/representation hypothesis: where can a bounded intervention causally reach the margin? It must be restricted to a tiny, theoretically justified set, with no layer/α sweep. The gate/localizer does not change first. |
| DIRECTION_AND_SITE | Record that the current repair actuator requires substantial redesign. Freeze nothing unless one coherent representation-level redesign follows directly from the evidence. |
| STILL_AMBIGUOUS | Freeze nothing. Name the single most informative remaining question. |

- **No repair validation runs.** No authorized fresh role exists, so any hypothesis is
  `REPAIR_NOT_VALIDATED_NO_FRESH_ROLE`. `router-calib` is not used automatically.
- **Never** use `D-dev-confirm`, `D-test`, P3 or transfer data.

## 11. Disclosed design choices (made before outcomes)

1. **Fixed-competitor margin.** m uses a fixed c\* so that it is differentiable at r₀ and equals
   the P2-R gap exactly. The P2-R "best other" margin has the same value at r₀ and the same
   gradient wherever the argmax does not switch.
2. **Tangent as primary.** The P2 edit is NormPreserve with ‖δ‖ = e\* ≈ 0.15‖r‖. Its radial part is
   second order (≈ e\*²/(2‖r‖) ≈ 0.085), so only tangent leverage is reachable to first order.
3. **Threshold 0.5 on ρ.** "A meaningful fraction" is operationalized as half the gap. A site
   where the best e\*-edit cannot close half the gap at most positions has insufficient leverage
   *at the P2 energy*. This is a narrow claim: it says nothing about larger energies or other
   layers.
4. **K₊ cut-offs 0.10 and 0.25.** These sit roughly 3.5 and 9 random-direction standard deviations
   (1/√1279) above zero.
5. **First-order only.** No finite perturbation along g is ever run, because the gradient may not
   be used as an edit. First-order validity is checked only against saved P2-R arm outcomes (V6).
6. **Precision (V7).** bf16 gradients are coarse, so the diagnosis must survive a float32
   recomputation. A label change counts as ambiguity.
