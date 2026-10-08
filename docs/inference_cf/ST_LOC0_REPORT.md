# ST-LOC0 — frozen-direction multi-site causal feasibility (report, terminal)

**Terminal verdict: `ST_LOC0_LOCAL_EFFECT_ONLY`.** Independent audit `ST_LOC0_AUDIT: PASS`: once on the primary phase before
any reference was opened, then in full. The auditor's label and winner (none) match the analysis.

**Decision:** "Partial causal evidence; insufficient lexical correction power for automatic progression."
- No LOC1, no expansion of LOC0 and no new direction in this session.
- This is a local, single-step, teacher-forced-prefix causal result on exposed development positions. It is not a full-ASR or
  paper-level performance claim.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, panel, config, contract tests) | `39ff5e9` | pre-outcome |
| Implementation | `cd28f06` | 10 focused tests; contract + DG-02/P2-DIR/P2-R/P2-RJ/LSS regression 138/138 |
| Sealed reference-free plan | `b09ab9a` | 14/14 prepare checks |
| Auditor fixture fix (pre-run) | `f36105d` | see *Audit notes* |
| Pre-run audit | `58dc232` | `PASS_TO_ST_LOC0` (65 checks) |
| Manifest (plumbing child; job ran at `58dc232`) | `ac735c4` | pushed before the job |
| Calibration seal (pushed from the login node while the job waited for GO) | `9998e5c` | pushed before any pulse |
| Run | Slurm 58109 | 6 min 32 s, 0 failures |
| Output seal (4 batches `08dcf54`, `fc58f9a`, `5a66613`, `48638ce`) | `48638ce` | pushed before any reference |
| Auditor fixture fix (post-seal, pre-reference) | `b589a44` | see *Audit notes* |
| Primary-phase audit | `25b31d0` | `ST_LOC0_AUDIT: PASS` (25 checks) |
| Secondary analysis + full audit | `19f8d37` + `4b3ed55` | LOCAL_EFFECT_ONLY; `ST_LOC0_AUDIT: PASS` (13 checks) |


## Population and data roles

- **Panel:** the frozen ST-LOC0 panel (identity `sha256:fa9d33c9…`): 180 positions from 80 utterances in 20 dialogues.
  - 60 EN-confusion, 60 EN-correct and 60 ZH-correct positions.
  - EN-confusion positions span 17 dialogues; the other two strata span all 20.
- **Data:** already-exposed D-dev-select only (the P2-R/P2-RJ/P2-DIR positions). There is no fresh validation.
- **Baseline:** B0M_L16 tokens (`results/inference_cf/p2_A_r1_L16`).
  - Query index is 4 + t − 1.
  - Prompts: c_M = forced-ZH and c_E = forced-EN, with the identical baseline content prefix.
- **Pulse runner inputs:** only the runtime projection (`runtime_queries` + `utterances`) and the sealed vectors.
- **Membership labels:** construction membership (EN-correct = A, ZH-correct = B) was used only by the offline V2 calibration
  builder, under whole-dialogue exclusion.
- **Evaluator references:** the targets and competitor from `p2rj/positions.json` were opened only after the output seal was
  pushed and the primary audit had passed.

## Historical reproduction (barrier before new-site pulses) — PASS

- **B0:** on 180/180 queries, the four-site passive replay reproduced the historical L16 DG-02 B/E states and the NONE packed
  logits bitwise, with matching processed argmax.
- **D0 = v_prompt L16 cross plus:** all 180 per-query vectors are hash-identical to the sealed exp1 D0.
- **D1 = v_unq L16 cross plus:** the 20 fold refits equal the historical vectors exactly (max abs 0), with matching
  status/index.
  - The pulses use the historical fold files (hash-verified), as the spec requires; the refit is provenance only.
- **D2:** the 180 sealed vectors are hash-verified. Provenance rests on the historical independent spot audit and
  `P2_DIR_AUDIT: PASS`; no new autograd was run.
- **Barrier pulses:** D0/D1/D2 logits and consumed L16 states are bitwise equal to historical exp1 on 180/180. The solver
  scale is within 1e-9 and the edit norms are equal.
- **Zero dose:** bitwise baseline at all 4 sites, for all 180 queries.
- **Restore and cache:** restore is bitwise after every query. The self-attention cache prefix hash and positions were
  unchanged after every one of the 3780 arms.
- **Post-seal evaluator:**
  - D0/D1/D2 Δmargin match the historical P2-DIR analysis exactly (max abs 0 in the analysis, 1.8e-15 in the auditor).
  - Top-1 matches on 0 mismatching positions.

## Directions

**v_prompt (V1).** `core_p1.direction(h_E, h_M)`: float64 subtraction, ‖·‖ + 1e-6, float32 serialisation, computed per
query at each physical site from the same-audio, same-prefix E/M replay.
- Both signs are frozen arms; minus = toward the forced-ZH state.
- 0/720 vectors were degenerate.
- Median ‖h_E − h_M‖: 2.48 (L16) and 2.59–2.60 (L24).
- Median cosine between the self and cross sites of the same layer: 0.998 (L16) and 0.995 (L24). At a fixed layer, V1 is
  essentially the same direction at both sites.

**v_unq (V2).** `unique.fit_fold` (NEW_P2_DIR_CROSSFIT_V1, rank 32, uncentred moments, all guards), fitted independently at
each site, with 20 leave-one-dialogue-out folds per site.
- A (EN-correct) has 55–58 rows per fold; B (ZH-correct) has 56–59.
- Moment hashes, eigen spectra, principal cosines, scores, index, sign and vector hashes are sealed in
  `run1/calibration/folds_*`.
- **Fold validity:** 80/80; no fallback was used.

Per-site detail:

| Site | Winning index | Winning principal cosine σ, median [range] | Min score gap | Sign flips |
|---|---|---|---|---|
| L16 cross | 0 in 20/20 folds | 0.868 | 0.128 | 8/20 |
| L24 self | 0 in 20/20 | 0.870 | 2.72 | 11/20 |
| L24 cross | 0 in 20/20 | 0.866 | 4.23 | 11/20 |
| L16 self | scattered (0 ×8; 12–31 ×12) | 0.355 [0.007, 0.875] | 0.010 | 5/20 |

- **L16 self:** the guards passed, but this construction is weakly identified.
- **Cross-site similarity of V2** (median fold-wise cosine):
  - L24 self vs L24 cross: 0.985.
  - L16 cross vs L24 cross: 0.544.
  - L16 self vs L16 cross: 0.013.
- **V1 vs V2 at the same site:** cosine within ±0.04 (near-orthogonal).

## Causal experiment

- **Arms:** 21 pulse arms, all at the fixed P2-R chord e* = 1.12608 with NormPreserve and no depth rescale.
  - 16 primary arms: V1/V2 × L16/L24 × post-self/post-cross × sign.
  - 4 same-site random controls (hash-pinned).
  - The historical D2 READOUT control at L16 cross (never selectable).
- **Matrix:** 180 queries × 21 arms = 3780 single pulses from the identical pre-query cache. No sequential accumulation.
- **Validity:** 3780/3780 valid matched-energy pulses; 0 no-edit cells; 0 integrity failures; pairwise energy within 0.02 on
  every query.
  - The independent auditor recomputed proposed/consumed edit norms from the saved native states and checked the solver
    direction against the sealed vector for every cell.
- **Relative squared edit energy** ‖Δ‖²/‖h‖², median [min, max]:

  | Site | Median [min, max] | Relative magnitude |
  |---|---|---|
  | L16 self | 0.0249 [0.0155, 0.0341] | ≈ 0.158 |
  | L16 cross | 0.0227 [0.0149, 0.0297] | ≈ 0.151 |
  | L24 self | 0.0173 [0.0115, 0.0252] | ≈ 0.131 |
  | L24 cross | 0.0158 [0.0109, 0.0225] | ≈ 0.126 |

  The same fixed chord is therefore relatively weaker at L24, because L24 states have larger norms.
- **Eligibility:** all 21 arms are eligible (60/60 valid in every stratum; 17/20/20 dialogues).

## Results — all 16 primary arms

The **margin** is m = logsumexp(z[Y_ref]) − z[c_fixed], using the fixed historical evaluator target set and competitor on
processed logits.
- **EN-confusion Δ** is the dialogue-macro estimate (the gate estimand) with a **Bonferroni-adjusted** interval (family of 16,
  quantiles .0015625/.9984375).
- **Correct-strata Δ** are dialogue-macro estimates with pointwise 95% intervals (descriptive).
- **Corrections:** baseline top-1 ∉ Y_ref → edited top-1 ∈ Y_ref.
- **Corruptions:** the reverse.
- **Validity:** valid cells per stratum, out of 60.

| Direction | Layer | Site | Sign | EN-conf Δm macro [adj. CI] | EN-conf corrections (dialogues) | top-1 changes | EN-correct Δm / corruptions | ZH-correct Δm / corruptions | Validity |
|---|---|---|---|---|---|---|---|---|---|
| v_prompt | 16 | self | + | −0.011 [−0.077, +0.060] | 0 (0) | 1 | +0.037 / 1 | −0.031 / 0 | 60/60/60 |
| v_prompt | 16 | self | − | **+0.103 [+0.026, +0.192]** | 0 (0) | 2 | −0.053 / 0 | +0.034 / 0 | 60/60/60 |
| v_prompt | 16 | cross | + | −0.020 [−0.082, +0.046] | 0 (0) | 2 | +0.027 / 1 | −0.047 / 0 | 60/60/60 |
| v_prompt | 16 | cross | − | **+0.114 [+0.036, +0.199]** | 0 (0) | 3 | −0.046 / 0 | +0.035 / 0 | 60/60/60 |
| v_prompt | 24 | self | + | −0.056 [−0.222, +0.077] | 0 (0) | 1 | +0.204 / 2 | −0.039 / 1 | 60/60/60 |
| v_prompt | 24 | self | − | **+0.219 [+0.095, +0.405]** | 0 (0) | 6 | −0.161 / 1 | +0.036 / 1 | 60/60/60 |
| v_prompt | 24 | cross | + | −0.048 [−0.212, +0.091] | 0 (0) | 1 | +0.200 / 2 | −0.044 / 1 | 60/60/60 |
| v_prompt | 24 | cross | − | **+0.228 [+0.106, +0.402]** | 0 (0) | 6 | −0.162 / 1 | +0.034 / 1 | 60/60/60 |
| v_unq | 16 | self | + | **+0.191 [+0.044, +0.404]** | 0 (0) | 0 | +0.066 / 0 | −0.001 / 0 | 60/60/60 |
| v_unq | 16 | self | − | −0.045 [−0.181, +0.075] | 0 (0) | 2 | −0.086 / 1 | +0.018 / 1 | 60/60/60 |
| v_unq | 16 | cross | + | **+0.227 [+0.074, +0.436]** | 0 (0) | 1 | −0.034 / 0 | −0.040 / 0 | 60/60/60 |
| v_unq | 16 | cross | − | −0.038 [−0.245, +0.086] | 0 (0) | 2 | −0.030 / 2 | +0.041 / 2 | 60/60/60 |
| v_unq | 24 | self | + | **+0.572 [+0.347, +0.831]** | 1 (1) | 4 | −0.078 / 0 | −0.101 / 0 | 60/60/60 |
| v_unq | 24 | self | − | −0.217 [−0.455, +0.010] | 0 (0) | 4 | +0.076 / 3 | +0.109 / 1 | 60/60/60 |
| v_unq | 24 | cross | + | **+0.562 [+0.306, +0.832]** | 1 (1) | 4 | −0.038 / 0 | −0.105 / 0 | 60/60/60 |
| v_unq | 24 | cross | − | −0.154 [−0.419, +0.113] | 0 (0) | 5 | +0.030 / 3 | +0.100 / 2 | 60/60/60 |

Bold marks the 8 arms whose adjusted lower bound is above 0.

**Supporting margin detail (EN-confusion):**
- **Position-mean Δm** agrees in sign with the macro on every arm. Examples: v_unq L24 self + is +0.546, and L24 cross + is
  +0.538.
- **Reference-rank improvement:** 52/60 and 49/60 positions for the two v_unq L24 plus arms, against 17–34/60 for the other
  arms.
- **Script mass:** the v_unq L24 plus arms move it toward English (mean ΔP_E +0.010, ΔP_M −0.016).

**Paired comparisons** (EN-confusion, dialogue-macro, pointwise 95%):
- **vs same-site random:**
  - v_unq L24 self +: +0.485 [+0.332, +0.653].
  - v_unq L24 cross +: +0.467 [+0.283, +0.644].
  - v_unq L16 cross +: +0.207 [+0.095, +0.341].
  - v_unq L16 self +: +0.193 [+0.061, +0.355].
  - v_prompt minus arms: +0.094 to +0.133, all excluding 0.
  - v_prompt plus arms: −0.143 to −0.009.
- **V1 − V2 at the same site/sign:**
  - plus: −0.20 to −0.63, all excluding 0 (V2 stronger).
  - minus: +0.15 to +0.44 (V1 minus beats V2 minus, whose sign is the wrong one).
- **vs D2 (historical positive control):** every primary arm is lower, by −3.93 to −4.72 nats.
- **vs historical D0/D1:** the L16 cross plus arms are the historical D0/D1 cells, reproduced exactly.
  - v_prompt L16 cross +: −0.020.
  - v_unq L16 cross +: +0.227.

## Controls

- **NONE:** Δ = 0 by construction. Zero dose and restore were bitwise baseline at all sites.
- **Random, same site, matched energy** (EN-confusion macro, pointwise 95%):

  | Site | Macro [95% CI] |
  |---|---|
  | L16 self | −0.002 [−0.056, +0.052] |
  | L16 cross | +0.020 [−0.029, +0.063] |
  | L24 self | +0.087 [+0.028, +0.149] |
  | L24 cross | +0.095 [+0.039, +0.153] |

  Random controls produce 0 corrections and 0–2 correct-state corruptions. A non-specific matched-energy perturbation at L24
  alone moves the margin by about +0.09 nat; the v_unq L24 plus effect exceeds it by about +0.47–0.49 nat.
- **D2 READOUT (historical positive control, L16 cross, not selectable):**
  - +4.499 [+4.043, +4.963] on EN-confusion, with 5 corrections across 5 dialogues and 43/60 top-1 changes.
  - EN-correct +1.535 with 1 corruption; ZH-correct −2.011 with 8 corruptions.
  - This reproduces the historical P2-DIR outcome exactly.

## Oracle envelope — REFERENCE-DEPENDENT UPPER BOUND. NOT DEPLOYABLE.

Computed post-seal over the 16 primary arms only; D2 and random are excluded.
- All 60 EN-confusion positions are baseline-incorrect.
- **V1 correctable:** 0/60.
- **V2 correctable:** 1/60. Position q44 (ZH-CN_U1063_S0_17, CSD0532, t = 43), via v_unq L24 self + and L24 cross +.
- **Union:** 1/60 (1.7%), covering 1 dialogue. One utterance and one dialogue hold 100% of the union. The best single arm
  makes 1 correction.
- **Correct-state damage exposure:** 7 of the 120 correct-state positions are corrupted by at least one of the 16 arms. Arm
  corruptions total 27; the maximum is 5 per arm, on the negative-sign v_unq L24 arms.
- **Comparison with D2:** D2 corrects 5 positions, q44 among them. The tested V1/V2 family at this energy reaches only 1 of
  D2's 5.

No location policy is built from the oracle.

## Statistical checks and gates

**Bootstrap:**
- Paired dialogue-cluster bootstrap, B = 10000, seed 240924.
- Sorted dialogue ids and shared draws across all arms and strata.
- Mean over positions within each dialogue, then mean over represented dialogues; no calibration refit inside the bootstrap.
- Bonferroni: family of 16 (all primary EN-confusion comparisons, including any ineligible arm), alpha 0.05/16.
- The independent auditor recomputed every estimate and interval from its own draws: max difference 3.5e-16.

**Frozen gates**, applied to eligible primary arms:

| Gate | Result |
|---|---|
| Adjusted lower bound > 0 | 8/16 arms (bold in the table) |
| Dialogue-macro Δm ≥ +0.50 nat | 2/16 arms (v_unq L24 self +: +0.572; L24 cross +: +0.562) |
| ≥ 3 top-1 reference corrections | 0/16 (max 1) |
| ≥ 3 correction dialogues | 0/16 (max 1) |
| Matched-energy / integrity | PASS (3780/3780 valid; primary audit and full audit PASS) |

Precedence: not INVALID; at least one eligible arm has an adjusted lower bound above 0, so not NO_FIXED_DIRECTION_LEVER. No
arm passes every gate, so the result is **`ST_LOC0_LOCAL_EFFECT_ONLY`**. There is no qualifying arm and no winner.

## Localization implications (descriptive; no new candidate is created)

1. **Site.** A matched-energy, frozen-direction pulse moves the fixed EN-confusion margin in a direction-specific way at every
   tested site. The specific part (beyond same-site random) is largest for v_unq at L24, with self ≈ cross (their V2 cosine
   is 0.985). L24 shows more effect than L16 even though its edit is relatively smaller.
2. **Sign and family.** V2 works with its frozen orientation (plus). V1 works only with the minus sign, i.e. toward the
   forced-ZH prompt state. Its plus orientation (toward the forced-EN prompt state) does not raise the margin; at L24 it
   raises EN-correct margins instead (+0.20).
3. **Magnitude.** The largest V1/V2 effect is about one eighth of D2's margin shift at L16. It converts 1/60 confusions,
   against D2's 5/60. Margin movement at this frozen energy rarely crosses the top-1 boundary. This is the
   lexical-correction-power gap the frozen gates were designed to detect.
4. **Collateral effects.** The positive arms carry small, mostly non-significant correct-state movements. The v_unq L24 plus
   arms lower the ZH-correct margin by about −0.10 with 0 corruptions. No safety claim is made either way.

## Limitations and exposure

- This is a single-step causal probe on a teacher-forced baseline prefix at a fixed P2-R chord. It is not free decoding, not
  full ASR, and not fresh validation.
- All 180 positions are already-exposed D-dev-select. No D-dev-confirm, D-test, new router-calib role, P3 or transfer data
  was used.
- EN-confusion covers 17 dialogues, and the single correction sits in one dialogue.
- The oracle depends on references and is non-deployable.
- V2 at L16 self is weakly identified (scattered winning index, small score gap) but passed every frozen guard.
- The result concerns these tested directions, sites and energy only. It is not evidence that representation steering in
  general is impossible.
- METHOD_CONTRACT/v6 and all historical conclusions (P2-DIR, P2-R, A2/MECH0) are unchanged.

## Compute and audit

- **Job:** one Slurm job, 58109, on an H100 MIG 3g.40gb.
  - Wall time 6 min 32 s (runtime 385 s): phase A 71 s, fits 45 s, GO wait 60 s, barrier 66 s, pulses 130 s.
  - Peak VRAM: 4.23 GB allocated / 4.65 GB reserved.
- **Counters:**
  - 17,496 decoder forwards; 80 encoder passes.
  - 4500 pulse calls (3780 cells + 720 zero-dose) and 7674 solver emulations.
  - 4500 cache fingerprints; 360 restore checks.
  - 0 autograd calls; 0 LID calls.
- **Outputs:** 371 MB in `results/inference_cf/st_loc0/run1`, plus the plan, analyses, seal and audits.

**Audit notes** (both fixes were made before any reference or outcome was visible):
- **Pre-run:** the first pre-run audit BLOCKed on one check. A substring rule for the analysis constant `POSITIONS` matched
  the reference-free panel key `construction_positions`. The rule was changed to an exact-name match. The uncommitted BLOCK
  file was discarded, and the re-run gave `PASS_TO_ST_LOC0`.
- **Primary phase:** the first primary-phase audit FAILed on one check. The push check looked up `origin/<local branch>`, but
  the tracked upstream is `origin/cs-asr-steer-inf`. It now uses `@{u}`. The uncommitted FAIL file was discarded, and the
  re-run gave `ST_LOC0_AUDIT: PASS`.

Artifacts: `results/inference_cf/st_loc0/` (`plan_sealed.json`, `prerun_audit.json`, `run1/`, `primary_analysis.json`,
`output_seal.json`, `primary_audit.json`, `secondary_analysis.json`, `final_audit.json`).
