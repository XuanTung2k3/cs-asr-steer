# P2-R mechanism diagnosis — report

**Primary diagnosis: `MECHANISM_STILL_AMBIGUOUS`.** This is `P2_R_MECHANISM_STILL_AMBIGUOUS`,
produced by the frozen rules of `P2_R_MECHANISM_DIAGNOSIS_SPEC.md` §7. No diagnosis profile is
satisfied.

- **Audit:** `P2_R_AUDIT: PASS` (`P2_R_POST_RUN_AUDIT.md`).
- **Repair hypothesis:** none created. The ambiguous branch forbids inventing one.
- **P3:** **HELD.**

**P2 is unchanged:** `P2_VALID_CONFIGURATION_SELECTED`, `P2_AUDIT: PASS`, selected L16 α = 2 φ_id
g = E·R_B +d.

## Runs

| Run | Job | Commit / manifest | Status |
|---|---|---|---|
| run1 | 54855 | `4e9adf2` / `90efa6b2…` | SIGABRT from fp64 kernels on the GPU inside the solver. No rows. Preserved. See `P2_R_RUN1_ABORT.md`. |
| run2 | 54871 | `7f6837a` / `65b3b6c2…` | 79/80 rows; an EOS-step donor pulse raised `IndexError`. Invalid; no metric computed. Preserved. See `P2_R_RUN2_INVALID.md`. |
| **run3** | **54874** | **`d3c1014` / `33a99f29…`** | **80/80 rows ok, 13.8 min, peak 4.0 GiB. The run analyzed here.** |

**Validity (all pass):**

- Replay baseline-argmax identity and lineage in both D2 passes.
- The gate schedule is identical across the current and oracle passes.
- The **current-placement replay reproduces the saved P2 selected run on 4,028/4,028 compared
  steps** (g, edit, edit norm).
- Pulse restores are bitwise; 0 baseline mismatches.
- The solver equals the hook on 100% of arms.
- **180/180 D1 positions are energy-matched.** Realized edit norms are within 2% of
  e\* = 1.126. The consumed-state edit is within 0.1% of the audited edit.
- D2 total realized energy, oracle over current, is **1.000002**.

## Setting

- **Population:** 60 EN-confusion positions (17 dialogues, 13 span-initial), 60 EN-correct
  (20 dialogues), 60 ZH-correct (20 dialogues). 80 D-dev-select utterances in total.
- **Energy:** a single L16 pulse at matched realized energy e\* = 1.126. This is about 2.2%
  relative squared energy, or about 15% of ‖h‖ ≈ 7.5.
- **Baseline at EN-confusion positions:**
  - The reference first-token set has median log p = −11.0 and median rank 353.
  - It sits a median **10.6 nats** below the best competitor.
  - P_M = 0.94 and P_E = 0.03.
- **Direction geometry:** the median cos(d, h) is −0.17, and the median ‖hE − hB‖ is 2.3.

## D1 — direction semantics (EN-confusion; simultaneous 95% CI, Bonferroni over 11)

| Contrast | Estimate | CI |
|---|---|---|
| L(+d): Δ log p(ref) | −0.026 | [−0.080, 0.025] |
| L(−d) | **+0.098** | **[0.031, 0.177]** |
| L(+d) − L(−d) | **−0.124** | **[−0.252, −0.012]** |
| Lex(+d) − Lex(−d): lexical specificity | −0.068 | [−0.160, 0.018] |

**Pointwise, descriptive:**

- **Script and lexical movement:**
  - ΔP_E(+d) is +0.0002 [−0.0013, 0.002]: **+d does not increase English mass.**
  - ΔP_M(+d) is +0.0014.
  - −d lowers P_M by −0.0038 [−0.0061, −0.0018].
  - Lex(+d) is −0.029 [−0.058, −0.0002], so +d moves the reference *down* relative to Latin mass.
  - Lex(−d) is +0.039 [0.004, 0.075].
- **Top-1 and rank:**
  - Top-1 corrections to the reference: **0/60** for +d, −d and random alike.
  - Median reference rank after the pulse: 372 (+d), 326 (−d), 357 (random).
- **Companions** (20 per stratum, free continuation):
  - 0 target words recovered under any arm.
  - Transcript changes are rare: 1–2 per stratum per arm.
- **Correct strata:**
  - ZH-correct: L(+d) −0.007 [−0.014, −0.002], a small damage signal.
  - EN-correct: unchanged.
- **Span split:**
  - Within-span (n = 47): +d −0.056 [−0.104, −0.010] vs −d +0.116 [0.056, 0.182].
  - Span-initial (n = 13, 9 dialogues): all intervals are wide; +d is +0.079 [−0.050, 0.231].

**D1 verdict: `DIRECTION_UNSTABLE`.**

- Opposite signs differ significantly in reference log-probability, in favour of −d.
- The lexical-specificity contrast does not exclude 0, so the frozen `SIGN_CONTRADICTION` profile
  is not met.
- `−d` is **not** adopted.

## D3 — generic perturbation (same positions, one frozen random direction)

| Contrast | Estimate | CI |
|---|---|---|
| L(+d) − L(random) | **−0.085** | **[−0.187, −0.012]** |
| L(−d) − L(random) | +0.039 | [−0.061, 0.145] |
| P(+d) − P(random) | −0.00006 | [−0.0015, 0.0016] (inside ±δ_p = 0.0167) |
| P(−d) − P(random) | −0.0006 | [−0.0038, 0.0015] (inside ±δ_p) |

**Pointwise:** the random direction alone raises reference log p (+0.059 [0.015, 0.106]) and P_E
(+0.0019), and lowers P_M.

**D3 verdict: `GENERIC_NOT_ESTABLISHED`.** The components:

| Component | Holds? | Why |
|---|---|---|
| G1: random changes outputs | yes | 3 changed companions |
| G2: no specific advantage | yes | — |
| G3: equivalence | **no** | Probability effects are equivalent within δ_p, but there is sign-specific log-probability evidence: +d is significantly *below* random and below −d. |

## D2 — localization (209 EN-confusion targets; energy-packet relocation)

**Relocation:**

- Current placement edits 163 of 209 targets.
- The oracle moves 38 donor packets onto unedited targets, which leaves 201 targets edited.
- Total energy is matched (ratio 1.000002).

| Contrast | Estimate | CI |
|---|---|---|
| L(oracle − current) | +0.011 | [−0.016, 0.044] |
| L(oracle − none) | −0.036 | [−0.099, 0.014] |
| P(oracle − current) | +0.0005 | [−0.0005, **0.0016**], upper bound below δ_p,D2 = 0.0048 |

**Pointwise, descriptive:**

- **Current placement against no edit:** L = −0.048 [−0.089, −0.010]. The P2 schedule slightly
  *lowers* reference evidence at the targets.
- **Correct answers:** 0 targets are correct under the current, oracle or no-edit arms.
- **Corrupted controls:** 6 under current and 7 under oracle, out of 2,507.
- **Pulse companions:** 0/20 recovered, for both current and oracle placement.
- **Acoustic annotation** (209 targets): the median timing error of the current window is
  0.74 s. The evaluator-centred window raises mean E from 0.53 to 0.62. The crop is imperfect, but
  better placement does not change the outcome.

**D2 verdict: `D2-B_NO_MATERIAL_ORACLE_RESCUE`.** Localization alone does not explain the P2
failure, at the declared resolution.

## Mechanistic conclusion

**1. The intervention is inert at the lexical decision, at this site and energy.**

- A matched 15%-norm pulse at L16 moves reference log-probability by ≤ 0.1 nat and changes the
  distribution by KL ≤ 0.007.
- The gap it would need to close is **10.6 nats**. This holds for every direction tested, the
  random one included.
- That is why P2's end-to-end effect was about 2/2,268. Neither current nor oracle placement can
  change that.

**2. Within that inert regime, +d is the worst direction tested.**

- It does not raise English mass.
- It lowers reference evidence relative to Latin mass.
- It is significantly below both −d and a random orthogonal direction.
- This is direction evidence *against* reading d = norm(hE − hB) as an English-repair direction at
  L16. It is small in absolute terms and not lexically established, so it is not proof of a sign
  error.

**3. Why the frozen rules return ambiguity.** The data support elements of both remaining
hypotheses, and the rules forbid choosing between them:

- **H1, direction semantics:** +d is wrong-signed relative to −d and random.
- **H4, site/sensitivity:** every direction is roughly 100× too weak to reach the decision.

**4. What is excluded:**

- A localization-only explanation (D2-B).
- A purely generic explanation (G3 fails because the direction effect is sign-specific).
- The current mechanism as proposed.

## Single most informative follow-up (not run; requires a human decision)

This is Codex's prescribed H1-vs-H4 follow-up. Nothing new is tuned.

- **What:** at the same 180 frozen positions and pre-edit L16 states, compute the
  **evaluator-only Jacobian of the reference margin** with respect to the L16 site state.
- **Report:**
  - its tangent norm, which bounds the first-order margin change any e\*-size edit can produce;
  - the achievable change, ‖∇m‖·e\*, against the 10.6-nat gap;
  - the alignment cos(∇m, d) against cos(∇m, random).
- **Reading:**
  - If ‖∇m‖·e\* is far below the gap, the primary issue is site/sensitivity.
  - If the gradient is large but +d is orthogonal or anti-aligned, the primary issue is direction.
- **Constraint:** the gradient must never be applied as a steering direction.
- **Cost:** about 180 backward passes, well under one GPU-hour.

## Repair and validation

- **No repair hypothesis** was created (ambiguous branch).
- **No fresh validation role exists** (§2 of the spec). Any future repair could only be
  `REPAIR_NOT_VALIDATED_NO_FRESH_ROLE` without a human decision about `router-calib` or new data.
