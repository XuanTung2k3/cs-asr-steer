# P2-RJ Jacobian H1-vs-H4 disambiguation — report

**Primary diagnosis: `P2_RJ_STILL_AMBIGUOUS`.** It comes from the frozen rules of
`P2_RJ_JACOBIAN_DIAGNOSIS_SPEC.md` §7 (v1 + v1.1): the **leverage axis is `UNRESOLVED`** and the
**alignment axis is `MISALIGNED`**.

- **Audit:** `P2_RJ_AUDIT: PASS` (`P2_RJ_POST_RUN_AUDIT.md`).
- **Repair hypothesis:** none. The ambiguous branch forbids one.
- **P3:** **HELD.**

**What stays unchanged:**

- **P2:** `P2_VALID_CONFIGURATION_SELECTED`, `P2_AUDIT: PASS`; L16 α = 2 φ_id g = E·R_B +d.
- **P2-R:** `P2_R_MECHANISM_STILL_AMBIGUOUS`, `P2_R_AUDIT: PASS`.

**The ambiguity is now narrow.**

- **H1 (direction) is supported under every admissible leverage class.** SUBSTANTIAL leverage would
  give `DIRECTION_ISSUE`; WEAK leverage would give `DIRECTION_AND_SITE_ISSUE`.
- **What stays open is only whether H4 co-occurs.** The leverage statistic sits on the frozen 0.5
  boundary.

## Run

| Item | Value |
|---|---|
| Job | 54998 (MIG 3g.40gb): bf16 pass 91 s, fp32 pass 104 s, peak 8.1 GB |
| Commit / manifest | `a93aa37` / `sha256:c020cf62…` |
| Rows | 80/80 utterances × 2 passes ok; **180/180 positions in both passes** |
| Attempts | No failed attempt; nothing rerun |

**Validity (all pass):**

- **State identity:** 180/180. Decode-rule argmax, log p(ref), m(r₀) = the P2-R margin, the
  hook-audit ‖r‖, cos(d, r), the competitor, and the solver scale s for all 3 arms reproduce P2-R
  run3. Grad-step logits and site are bitwise equal to the unedited step.
- **Finiteness:** every gradient is finite and nonzero.
- **Population:** 60 EN-confusion positions over 17 dialogues.
- **First-order sanity (V6):** pooled Pearson r = **0.918 [0.892, 0.936]**, slope **1.03**, over 540
  (position, arm) pairs.
- **Precision (V7):** median cos(g_tan,bf16, g_tan,fp32) = **0.9998**, and the fp32 label is
  identical.

## Margin definition

- **m(r):** `m(r) = logsumexp z(Y_ref) − z(c*)` on processed logits.
  - Y_ref is the P2-R reference-consistent first-token set.
  - c\* is the fixed P2-R competitor. At all 60 EN-confusion positions it is the baseline argmax.
- **EN-confusion baseline deficit (gap = −m(r₀)):**
  - median **10.61 nats** (IQR 8.0–12.3; range 0.29–15.7);
  - identical to P2-R.

## Site sensitivity (H4 axis) — EN-confusion, n = 60

| Quantity | Median (IQR) |
|---|---|
| ‖∇m‖ | 4.51 (3.81–6.32) nats per unit state norm |
| ‖∇m_tan‖ | 4.50 (3.81–6.31). The gradient is ~100% tangent: median \|radial\| is 0.09. |
| e\* | 1.1261 (frozen P2-R) |
| **A = ‖∇m_tan‖·e\*** (first-order maximum achievable Δm) | **5.07 nats** (4.29–7.11) |
| **ρ = A/gap** | **0.58** (0.43–0.72) |

**Fraction of positions whose gap an optimally aligned e\*-edit could close, to first order:**

| Closes | Positions | Dialogue-weighted |
|---|---|---|
| ≥ 10% of the gap | 100% | 1.00 |
| ≥ 50% of the gap | 63% | **Q50 = 0.614**, Bonferroni CI [0.442, 0.761] |
| ≥ 100% of the gap | 13% | 0.147 [0.049, 0.255] |

**Leverage class: `UNRESOLVED`.** The Q50 interval contains 0.5.

## Direction alignment (H1 axis) — EN-confusion

| Arm | cos(g_tan, v): mean [95%] | median \|cos\| | κ = captured fraction of the optimum | Predicted Δm (first order) | Observed Δm (P2-R) |
|---|---|---|---|---|---|
| +d | **−0.0107** [−0.020, −0.002] | 0.023 | **−0.011** (Bonferroni [−0.022, −0.001]) | −0.058 [−0.102, −0.015] | −0.020 [−0.062, 0.024] |
| −d | +0.0107 [0.002, 0.020] | 0.023 | +0.011 | +0.063 [0.019, 0.109] | +0.114 [0.063, 0.168] |
| random | +0.0027 [−0.007, 0.011] | 0.022 | +0.002 | +0.041 [−0.014, 0.097] | +0.078 [0.026, 0.134] |
| optimum (evaluator-only, **never applied**) | 1 | — | 1 | +5.07 | — |

- **Alignment class: `MISALIGNED`**, with an upper bound of −0.001 < 0.10.
- **Captured leverage.** d captures about **−1%** of the available first-order leverage. Its
  |cosine| with the useful direction equals that of a random direction (0.023 vs 0.022) in 1,280
  dimensions.
- **Sign.** The sign is slightly wrong: κ₊ − κ_random = −0.013 [−0.022, −0.004].
- **Span split.** The anti-alignment is concentrated in within-span positions:
  - within-span (n = 47): κ₊ = −0.015 [−0.025, −0.006];
  - span-initial (n = 13): +0.004 [−0.013, +0.023].

## Mechanistic explanation

**1. P2-R's pattern is first-order geometry.**

- The Jacobian predicts the realized P2-R arm changes with r = 0.92 and slope ≈ 1 across all strata
  and arms.
- It reproduces the sign ordering: +d < random < −d.
- At e\* the L16 margin response is essentially linear. The P2-R "inertness" is therefore **not** a
  saturated or flat site.

**2. The site has moderate leverage, and d does not use it.**

- An e\*-size edit aligned with ∇m_tan would move the margin by a median of ~5 nats. That is about
  **50–100×** what ±d or random achieved (≤ 0.1 nat).
- d = norm(hE − hB) is, to within noise, **orthogonal** to the useful lexical direction, with a
  small anti-aligned bias. The reason P2/P2-R saw no repair is predominantly the direction (H1).

**3. What d encodes.** These secondary-target cosines are descriptive:

- **The useful direction is largely a script-readout direction.** At EN-confusion positions,
  cos(∇m_tan, ∇log P_E,tan) = **0.80** [0.76, 0.83] and cos(∇m_tan, ∇log P_M,tan) = −0.78. It
  mostly means "raise Latin-script mass / lower Han mass".
- **d is not that direction either.** cos(d, ∇log P_E,tan) = −0.006 [−0.015, 0.003].
- **Reading.** The forced-EN vs forced-ZH prompt-conditioning displacement at L16 is
  near-orthogonal to both the lexical and the script sensitivities of the downstream readout at
  these tokens.
  - It is **not** a script-preference direction. It is not a sign-flipped repair direction either:
    −d captures only +1%.
  - The measured quantity is a prompt/condition displacement that the downstream decision is
    locally insensitive to.
- **In correct strata.** The useful direction is less script-like: EN-correct 0.46; ZH-correct
  −0.55, i.e. there raising Latin mass *lowers* the ZH margin. Any script-like repair direction
  would therefore need selective gating.

**4. Why H4 is not resolved.** Even a perfectly aligned edit at the P2 budget would:

- close a median of 58% of the gap;
- fully close it at only ~13% of positions.

Whether "most positions can close half the gap" is 0.614 [0.44, 0.76], right at the frozen 0.5
boundary. At e\*, L16 therefore has neither clearly sufficient nor clearly insufficient leverage.
This is a narrow statement about e\* at L16 only.

## Correct strata (safety, descriptive)

| Stratum | Median A | Median A/surplus | Worst-case e\*-edit could erase the surplus | +d effect |
|---|---|---|---|---|
| EN-correct | 2.89 | 0.86 | 37% of positions | +d is near-neutral (κ₊ = +0.012) |
| ZH-correct | 3.06 | 0.68 | 27% of positions | +d is slightly harmful (κ₊ = −0.010) and raises predicted log P_E (+0.040), consistent with P2-R's small ZH-correct damage signal (−0.007) |

## Repair and validation

- **No repair hypothesis** was created. The frozen branch is `STILL_AMBIGUOUS`, which forbids one.
  - The gradient was never applied as a steering direction and is not proposed as one.
  - The script-readout alignment in §3 is a descriptive observation, not a frozen hypothesis.
- **Validation run:** NO. No authorized fresh role exists; `router-calib` was not used.

## Single most information-efficient remaining question (not run; needs human authorization)

**Is the L16 leverage at e\* SUBSTANTIAL or WEAK (Q50 above or below 0.5)?**

- **Why it matters.** It is the only open axis. It decides between two outcomes, and each fixes a
  different next step under the §10 repair policy:
  - `DIRECTION_ISSUE`: freeze one direction-construction hypothesis.
  - `DIRECTION_AND_SITE_ISSUE`: the current actuator needs redesign.
- **Why sample size is the lever.** Uncertainty alone dominates this axis: the point estimate is
  0.61 and the CI is [0.44, 0.76]. So the Codex-prescribed form applies: a **predeclared sample
  expansion for that single contrast from the already-exposed diagnostic role, with no changed
  conditions.**
  - **Protocol:** identical to P2-RJ, on all remaining structurally valid EN-confusion positions
    of the 300-utterance D-dev-select panel (P2-R §3 eligibility, without the 60-position cap).
  - **Cost:** about 5 GPU-minutes.
- **What does not change.** H1 stands whatever that expansion shows.
