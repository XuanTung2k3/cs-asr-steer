# Results inventory — Days 4–8 complete

**Updated 17 August after Day 8. Gate B does not pass. The preregistered fallback is active — and Day 8 removed the explanation that fallback was going to rest on.** An inventory of claims, with the evidence behind each and the numbers you can quote.

> **Day 8 headline, stated once at the top so it is not read as a footnote:** giving transport the English token it was missing — even by handing it the entire gold transcript — recovers **no** corrections. **0/489 in all three arms.** The 81.1% no-English-in-hypothesis rate is a **correlate of the failure, not its demonstrated cause.** §4 has the numbers and the limits of that null; §7 has what it does to the paper.

> **Two numbers in this document were re-checked against the artifacts on 17 August and changed status.** (1) All 7 oracle corrections come from **2 utterances in 2 dialogues** — 18 of 20 clusters contribute zero, so the effective G behind the only positive transcript result is 2, not 20 (§2). (2) The Day 6 per-cell effects in §1.6 **do not reproduce** from the stored score rows under any estimator in this repository; the invariance *conclusion* holds, the per-cell numbers are not quotable yet (§1.6).
>
> Full provenance, the stage-by-stage record, the defect register, and the discrepancy register are in **`docs/STEERING_EXPERIMENT_FULL_RECORD_2026-08-17.md`**. That document is the underlying record; this one is the claim-by-claim inventory for writing.

---

## 1. The mechanism — established (Days 3–6)

### 1.1 A language direction at decoder layer 16 has causal leverage

Difference-in-means direction, teacher-forced, oracle-localized, substitution-only stratum:

| Outcome | Effect | 95% CI | G |
|---|---:|---|---:|
| Δ gold-token logprob | **+0.1057** | [+0.0466, +0.1649] | 19 |
| Margin | +0.119 | [+0.036, +0.202] | 19 |
| Gold rank | +42.6 positions | does not resolve | 19 |

Beats **all four** controls in paired contrasts: label-permuted null, matched-norm random (≥5 draws), opposite sign, wrong location. Null norm ratios 0.175–0.619 confirm the nulls were genuinely rebuilt through the pipeline — a sign flip would give exactly 1.000.

### 1.2 The effect is not the language-token mechanism

`cos(v_nat, U_prompt) = 0.010–0.030`; the prompt subspace removes only 0.03–0.10% of contrast energy. The switch-onset contrast is nearly orthogonal to the prompt direction.

This answers the circularity objection empirically rather than by construction, and it is the single most defensible claim in the project. **Day 8 does not touch it** — Day 8 is a transcript-space result and leaves every representation-level finding standing.

### 1.3 Encoder evidence shows no causal leverage — CONFIRMED, a genuine negative result

Site E beats no control at any layer or strength. Best case +0.038 [−0.013, +0.089]; layer 31 at ρ=2 is *worse* than its own null (−0.453).

**Day 6 settled the construction-artifact question.** Under inverse-variance weighting:

| Site | Effect | CI | Excludes 0 |
|---|---:|---|---|
| D (layer 16) | +0.1150 | [+0.0367, +0.1932] | **yes** |
| E (layer 27) | +0.0164 | [−0.0335, +0.0662] | no |

The reweighting demonstrably worked. `corr(‖d‖, duration)` at site_e_layer15 moved **−0.386 → −0.019** — the coupling essentially removed — and Site E remained null. Three pooling schemes agree independently (Hann +0.0234, central-60 +0.0194, uniform +0.0230; every CI includes zero; bootstrap cosine 0.975–0.978).

**Site E's null is a property of the site, not of how the direction was built.** This is now a defensible negative localization result.

*One caveat to record:* weight ratios at Site E layers 23/27/31 are extreme (322–2709×), so a handful of pairs carry those directions. It does not rescue the site — layer 15, where the ratio is a benign 6.5, is null too — but it belongs in the write-up.

### 1.4 The dose-response saturates and then declines — the anti-artifact argument closes

Day 5 extended the grid. Site D, correct direction vs label-permuted null, substitution-only, G=19:

| Layer | ρ=0.25 | ρ=0.5 | ρ=1 | ρ=2 | ρ=3 | ρ=4 |
|---|---:|---:|---:|---:|---:|---:|
| 8 | +0.008* | +0.017* | +0.031* | +0.066* | +0.076 | +0.043 |
| 16 | +0.024* | +0.054* | **+0.106*** | +0.178* | +0.170* | +0.079 |

`*` = interval excludes zero. Layer 16 turns between ρ=2 and ρ=3, then falls to +0.079 with the interval crossing zero [−0.099, +0.256]. Layer 8 peaks at ρ=3 and falls at ρ=4.

**This is a stronger result than Day 4 suggested.** An effect that kept growing without bound would be hard to separate from a generic magnitude perturbation; the observed rise-peak-decline is the shape a genuine directional mechanism predicts. The frozen ρ=1 sits on the rising limb well below the turn, and was not revisited on this curve.

### 1.5 Coverage is insensitive to the duration floor

Relaxing 400 → 200 ms adds 26 error units (+5.3%) and 8 substitution units (+8.0%). Effect is flat within noise: +0.106 / +0.110 / +0.104 at 400/300/200 ms, every interval excluding zero, G=20 at every tier.

Report as "coverage is insensitive to the floor in this corpus," not as a three-point curve. **Consequence:** relaxing the floor is not available as a route to statistical power. Day 7 confirmed this in transcript space — the non-oracle correction rate is 0.0000 at all three tiers.

### 1.6 The result is invariant to alignment choices

> **Status change, 17 August.** The *conclusion* below holds and was re-verified: sign preserved in every valid cell, Site E null under reweighting and all three poolings. The **per-cell effect sizes do not reproduce** from `invariance/generation_001` under either the plain per-unit paired mean used by `paired_contrast()` or the dialogue-balanced mean the construction pipeline uses. Unit counts match exactly in every cell, so the populations are right and only the aggregation differs; the worst divergence is `jitter_zh +200 ms` (+0.1642 circulated, +0.1043/+0.1190 recomputed), which is the top of the magnitude band quoted below. **Root cause: Day 6 is the one stage with no dated result document, so its analysis path was never written down.** Quote the conclusion, not the cell numbers, until the estimator is recovered. Full comparison: `STEERING_EXPERIMENT_FULL_RECORD_2026-08-17.md` §5.7.5 and §10.1.

Day 6, 23 cells, substitution-only stratum, G=19, each paired against 5 label-permuted nulls:

| Factor | Cell | Units | Effect | CI |
|---|---|---:|---:|---|
| baseline | `existing_ctc` / `blank_to_preceding` | 199 | +0.1192 | [+0.0364, +0.2020] |
| aligner | `whisper_dtw` | 187 | +0.0951 | [+0.0160, +0.1741] |
| convention | `blank_midpoint` | 205 | +0.1222 | [+0.0400, +0.2044] |
| convention | `blank_to_following` | 203 | +0.1167 | [+0.0331, +0.2003] |
| convention | `blank_excluded` | 184 | +0.0783 | [+0.0105, +0.1460] |
| jitter ZH | +200 ms | 199 | +0.1642 | [+0.0917, +0.2366] |
| jitter EN | −200 ms | 199 | +0.0986 | [+0.0132, +0.1840] |
| jitter EN | +200 ms | 199 | +0.0692 | [−0.0096, +0.1481] |
| inverse-variance | reweighted | 199 | +0.1150 | [+0.0367, +0.1932] |

**Sign preserved in every valid cell.** Magnitudes span +0.069 to +0.164, centred near the +0.106 Day 4 headline. Weakest genuine cell +0.0783 (`blank_excluded`); only zero-crossing is EN jitter +200 ms.

**Two cell classes are weaker evidence than the matrix suggests — say so in the paper:**

*The EN-jitter row is largely inert.* At the corpus token rate of 3.38 tokens/s, ±50 and ±100 ms round to **0 decoder steps**, so four of six EN-jitter cells are the baseline by construction, identical to four decimal places. Only ±200 ms displaces anything, and only for ~27% of utterances. Sub-token boundary error cannot move a decoder-site intervention — which is itself a finding about why Site D is robust, but it is not six independent confirmations.

*`jitter_zh −200 ms` is degenerate and must be discarded.* A sign-mapping defect collapsed the control step onto the onset, making the contrast identically zero (+0.0000 [0, 0], correct and null margins bit-identical). That is an implementation defect, not a measurement. It was repaired on Day 7 — see §3.5.

*Useful by-product:* the four ZH cells at ±50/±100 ms have `control_offset = 0`, so they are identical constructions differing only in null-draw seed. They span +0.0827 to +0.1144, which **calibrates null-draw noise at roughly ±0.03**. That also explains the baseline reproducing at +0.1192 against Day 4's +0.1057 — a +0.0135 difference, well inside that noise. **This ±0.03 figure is the yardstick for every arm-to-arm difference in §4** — all of them are far below it.

### 1.7 Supporting diagnostics

C11 near-additive at ρ=1 (−0.022 to +0.003). Direction stability: dialogue-bootstrap p05 rising 0.868 → 0.944 with depth. Residualization mild — `cos(Δ_raw, Δ_resid)` 0.994–0.999, ridge removes 2.0–5.1%. Layer 31 peaks at ρ=1 and turns down; layer 24 collapses at ρ=2.

---

## 2. The oracle free-decoding result — read this carefully

Day 5's first transcript-space measurement, **oracle-localized** (gold step supplied). This is an upper bound, not the practical result.

| Metric | Baseline | Steered |
|---|---:|---:|
| PIER | 0.5249 | 0.5189 |
| MER | 0.2850 | 0.2780 |
| WER | 0.8684 | 0.8674 |

**7 corrections over 489 baseline-error targets** — a 1.4% correction rate.

> **Day 5's "0 corruptions" was wrong, and Day 7 corrected it.** That count was taken only inside a target set consisting entirely of baseline errors, where corruption is impossible by construction. Counted properly over baseline-correct units per utterance, the oracle arm produces **7 corrections against 13 corruptions** at 400 ms (7 : 19 at 200 ms). The oracle intervention is **net harmful**. Every reading of Day 5 that treated zero corruptions as a favourable sign — including mine — was mistaken.

### The three things this establishes

**The effect crosses into transcript space, but barely, and net negatively.** A logprob shift of +0.106 produces 7 corrections and 13 corruptions. 1.4% correction at the oracle upper bound, with more damage than repair.

**Spillover dominates, and the localized-edit claim is in trouble.** 85 units changed outside the target (72 improved, 13 damaged), concentrated in 5 of 116 utterances. That is **~12× the targeted effect**. Net favourable, but a single-step intervention changing 85 units elsewhere is not a localized edit, and "selectivity" was going to be the paper's headline argument.

**The deletion prediction: mechanism confirmed, conclusion refuted.**

| Category | Targets | Hook fired | Corrected (of fired) |
|---|---:|---:|---:|
| deletion | 290 | 207 (71.4%) | 6 / 207 = 2.90% |
| substitution | 199 | 198 (99.5%) | 1 / 198 = 0.51% |

For **28.6% of deletion targets the oracle step did not exist** — the decode terminated before reaching it — against 0.5% for substitutions. That is the predicted mechanism, measured, with no analogue under teacher forcing.

But conditional on the step existing, deletions still correct ~6× more often — the same direction as teacher forcing (+0.157 vs +0.106). The advantage did not collapse.

**Do not over-read this.** The entire comparison rests on 7 corrections (6 deletion, 1 substitution). At those counts the ratio is not distinguishable from chance and no bootstrap interval on it would be meaningful. Report the 28.6% step-unavailability rate — that number is solid — and report the correction ratio as descriptive with its denominator visible.

### The corrections are concentrated in two dialogues — verified 17 August

| Effect | Utterances | Dialogues |
|---|---:|---:|
| raw text changed at all | 10 / 116 | 8 / 20 |
| any scored effect after normalization | 5 / 116 | 5 / 20 |
| **all 7 corrections** | **2 / 116** | **2 / 20** |
| all 13 corruptions | 4 / 116 | 4 / 20 |
| all 85 spillover units | 5 / 116 | 5 / 20 |

Six of the seven corrections are deletions in a single utterance (`ZH-CN_U1003_S0_56`, `CSD0502`); the seventh is a same-language substitution in `ZH-CN_U0027_S0_114` (`CSD0014`). Two of the four corrupting utterances are the two correcting ones.

**Consequence for the statistics.** `+0.0143 [+0.0033, +0.0254], G=20` is computed over 20 clusters of which **18 contribute exactly zero**. The wild bootstrap is essentially resampling the presence of two clusters, so the interval's width says nothing useful about the effect's precision. State the effective cluster count — **2** — alongside the nominal G.

**Consequence for the interpretation.** The intervention is not doing a little good in many places and a little harm elsewhere. It does nothing at all in 111 of 116 utterances and everything — corrections, corruptions, and spillover together — in a handful. That strengthens the abstention argument in §3.3 and complicates it at the same time: a router that declines those utterances declines the only ones that ever correct.

**One number from this section carries into §4.8 and matters more than it looks:** conditional on the step existing at all, the oracle arm corrects at **7 / 405 = 1.73%**. That is the largest per-opportunity correction rate this project has ever measured, and it sets the scale any downstream arm must be powered against.

---

## 3. Gate B — the practical claim fails, and the reason is structural

**Non-oracle correction rate is exactly zero at every tier.** The entire oracle effect is lost.

| Tier | Oracle | Non-oracle | Gap | Surviving |
|---|---:|---:|---|---:|
| 400 ms | 0.0143 (7/489) | **0.0000** (0/489) | +0.0143 [+0.0033, +0.0254] G=20 | **0.000** |
| 300 ms | 0.0139 | 0.0000 | +0.0139 [+0.0031, +0.0248] | 0.000 |
| 200 ms | 0.0136 | 0.0000 | +0.0136 [+0.0028, +0.0244] | 0.000 |

PIER gap −0.0060. The gap's CI excludes zero because the non-oracle arm is exactly zero, not because 7 corrections is many — the oracle rate is descriptive.

Monolingual material is not the casualty: over 5,711 baseline-correct Mandarin units the oracle arm retains 99.79% and the non-oracle arm 100.00%. The damage the oracle arm does is small and concentrated, not broad.

### 3.1 The measured correlate: the error destroys the evidence needed to repair it

> **Read this subsection together with §4.** Everything below was measured correctly and reproduces exactly. What changed on Day 8 is its *status*: it is a correlate of the failure, and the causal probe designed to promote it to a cause came back negative.

Two failure modes, and the new one dominates.

| Mode | Rate |
|---|---:|
| Target has no decoder step at all (Day 5's mode, reproduced exactly) | 28.6% of deletions |
| **Step exists, but the first-pass hypothesis contains no English at all** | **81.1% of wrong-language substitutions, 37.7% of deletions** |
| Overall abstention | **88.3%** |
| Localized | 11.7% (57/489) |
| Transport hit | 1.4% (7/489) |

Examples: *social skills* → 社會技巧; *improving* → 進步; *Exercise your strength … generally* dropped entirely.

**This is the circularity, measured.** Transport needs to attend to an English token in the hypothesis. When the baseline error *is* the disappearance of the English, there is nothing to attend to. The intervention requires as input precisely the thing whose absence defines the error.

The prediction was made before the experiment; the number is 81.1%. **Whether that absence is what actually blocks the correction is a separate question, and §4 answers it: no.**

### 3.2 It is a localization null, not a dose null

| Arm | Correct direction | Label-permuted null | Matched-energy random |
|---|---:|---:|---:|
| Non-oracle, 400 ms | 0 / 57 | 0 / 285 | 0 / 285 |

Paired contrasts +0.0000 [0, 0], G=11. `E_eff` matched exactly at 17.4889 for both correct and matched-energy arms.

Critically, the non-oracle arm delivers **more** energy where it acts than the oracle arm (17.49 vs 4.83). The failure is not insufficient intervention strength — it is that the intervention lands in the wrong place, or nowhere. Day 8 reinforces this from the other side: arms with *more* delivered energy and *more* coverage still correct nothing (§4.3).

### 3.3 Spillover concentrates decisively

Per-utterance counts: **41, 32, 10, 1, 1**. Top 2 of 116 utterances account for **85.9%**; top 3 for **97.6%**.

That settles the interpretation raised at Day 5: spillover is not diffuse churn, it is a small set of pathological utterances. An abstention rule is the appropriate response, and that is a method observation rather than a defect.

The non-oracle arm has zero spillover, so the cross-arm overlap question is trivially answered — but see §4.6, where the oracle *input* conditions break that zero.

### 3.4 Two things that constrain any future claim

**Corruptions exceed corrections even at the oracle bound.** 7 : 13 at 400 ms, 7 : 19 at 200 ms. See the correction in §2.

**D-test is underpowered for effects of this size.** MDE 0.0608 at G=15 — roughly 4× the largest oracle effect measured anywhere in this project. Even a positive development result could not have been confirmed on test. The reference-only D-test counts are 6,257 utterances, 15 dialogues, 176,345 reference units, 35,921 embedded-English units (20.37%); the C00 baseline puts 27.12% of the error mass on embedded-English units.

### 3.5 The degenerate Day 6 cell, repaired during the Day 7 run

`jitter_zh −200 ms`: +0.0000 [0, 0] → **+0.0915 [+0.0156, +0.1674]**, G=19. `control_step_for()` now clamps control strictly before onset under every offset sign; 11 regression tests with a failing fixture. Nothing else moved — verified deterministically and empirically, all 11,736 rows bit-identical to `generation_001`.

*Caveat to record:* the clamp maps offset −1 onto the offset-0 construction, so the repaired cell is a null-seed replicate rather than an independent jitter probe. A Mandarin control cannot sit later than the step before onset.

---

## 4. Day 8 — the causal probe: the 81.1% does not survive as a cause

**Slurm 41736, COMPLETED, exit 0, 5 min 11 s. Full destination `artifacts_dialogue_v2r3/hypothesis_substitution/generation_001`. Source document: `docs/V2R3_HYPOTHESIS_SUBSTITUTION_2026-08-17.md`.** Nothing was selected, no gate was evaluated, no threshold moved, and `D-dev-confirm` and `D-test` were neither read nor touched.

> **Arms B and C are oracle conditions built to establish causation. They are not practical methods, were never eligible for selection, and must not be written up as proposed mitigations.**

### 4.1 The design, and the one thing that changed

Same 489 baseline-error targets, 116 utterances, 20 dialogues, 400 ms conservative tier, `D-dev-select`. Same frozen action — **Site D, decoder layer 16, ρ=1**, recorded rather than re-derived. Same transport formula, same final-layer all-head attention, same non-renormalized weights, same parameter-free enrichment abstention rule, same ±1-step transport-hit tolerance.

**Only the hypothesis token sequence that `g_t` and its cross-attention are derived from changed.**

| Arm | Transport input | Status |
|---|---|---|
| A — baseline | the generated first-pass hypothesis, exactly as Day 7 | reference; reproduces Day 7 |
| B — gold-substituted | the complete gold transcript | oracle causal condition |
| C — partial substitution | first-pass output, minimally edited to contain the reference English unit | oracle causal condition |

Arm A reproduced Day 7 **exactly on all 489 paired rows** — correction, abstention, selected step, transport hit, effective energy, and steered text. The reference arm is a genuine reproduction, not a re-estimate.

### 4.2 How Arm C was built, and how often it was well defined

Per utterance, for every target reference unit in it: normalize and segment reference and hypothesis under the frozen MER/PIER unitization; take the existing deterministic Levenshtein backtrace; insert the reference English unit immediately before its aligned hypothesis unit (for a substitution) or at the alignment cursor (for a deletion); map that unit boundary back to a raw character boundary and insert with separating spaces; **preserve every original character of the first-pass output — the wrong token is not replaced or deleted**; then re-normalize and require the result to equal the original units plus exactly the requested insertions, or mark the target skipped with a reason.

**Well defined for 489/489 targets (100%). Skipped: 0.** By category: deletion 290/290, wrong-language substitution 90/90, same-language substitution 77/77, phonetic/transliteration 17/17, boundary error 12/12, other 3/3.

There is no construction-failure escape hatch in this result. Arm C is exactly what it claims to be.

### 4.3 The headline: nothing corrects, in any arm

Corrections are over the 489 baseline-error targets. Corruptions are counted once per utterance over the **6,264 reference units that were correct in the first pass**. PIER/MER/WER are corpus micro-aggregates over the 116 utterances. `E_eff` is the mean over all 489 target rows; "active" conditions on non-abstention.

| Arm | Corrections / 489 | Corruptions / 6,264 | Corrected : corrupted | PIER | MER | WER | E_eff all | Active; E_eff active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | **0** | 0 | 0 : 0 | 0.524914 | 0.285039 | 0.868394 | 2.0386 | 57; 17.4889 |
| B gold | **0** | 1 (0.0160%) | 0 : 1 | 0.524914 | 0.285154 | 0.868394 | 3.9999 | 147; 13.3058 |
| C partial | **0** | 2 (0.0319%) | 0 : 2 | 0.524914 | 0.285269 | 0.867876 | 2.6805 | 96; 13.6537 |

Wild cluster bootstrap, Rademacher weights, 10,000 draws, cluster `dialogue_id`, t(G−1) critical values, seed 20260817:

- **Correction rate: 0.0000 [0.0000, 0.0000] in A, B, and C. G=20, n=489.**
- Corruption rate: A 0.000000 [0.000000, 0.000000]; B 0.000160 [−0.000058, +0.000377]; C 0.000319 [−0.000081, +0.000720]. G=20, n=6,264.
- Mean per-utterance PIER change vs baseline: 0.000000 [0.000000, 0.000000] in all three arms. G=20, n=116.
- Mean per-utterance MER change: A 0.000000 [0, 0]; B +0.000059 [−0.000021, +0.000138]; C +0.000238 [−0.000305, +0.000781]. G=20, n=116.
- Mean per-utterance WER change: A and B 0.000000 [0, 0]; C −0.000575 [−0.001355, +0.000206]. G=20, n=116.

The zero-correction intervals are **degenerate** — every observed outcome is zero, so the bootstrap has nothing to resample. They are not binomial upper bounds and they do not establish an exactly zero population rate. Every B−A and C−A difference above is one to three orders of magnitude below the **±0.03 null-draw noise** calibrated on Day 6 (§1.6) and must not be read as a distinction between arms.

### 4.4 The substitution did exactly what it was supposed to do

The first five columns partition the 489 targets. "Delivered despite no gold step" is separate because gold-substituted transport can localize an earlier generated query even when the baseline decode terminated before the target's gold step.

| Arm | No step at all | Step exists, no English | English below 400 ms | Region found, not enriched | Localized | Delivered despite no gold step | Total delivered |
|---|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 84 | **161** | 122 | 65 | 57 | 0 | 57 (11.7%) |
| B gold | 84 | **0** | 162 | 135 | 108 | 39 | **147 (30.1%)** |
| C partial | 84 | **0** | 214 | 95 | 96 | 0 | **96 (19.6%)** |

On the two categories that motivated the run:

- **wrong-language substitution**: A has 73/90 = **81.1%** step-existing targets with no English in the hypothesis. B and C: **0/90**.
- **deletion**: excluding the 83 targets with no gold step, A has 78/207 = **37.7%**. B and C: **0/207**.

The diagnosed input absence was eliminated completely, and delivered coverage rose from 57 to 147 (B) and 96 (C). **Transport hits — the selected step landing within ±1 of the gold step — rose from 7 to 25 (B) and 12 (C).** Delivered energy rose. Corrections did not move off zero.

`E_eff` rules out an energy explanation from either side. B and C deliver on 2.6× and 1.7× more targets than A and still correct nothing; where they act they deliver 13.3 and 13.7 mean energy against the oracle arm's 4.83. In B, the 108 gold-step-localized deliveries average 1.93 and the 39 no-gold-step deliveries average 44.81 — the energy is concentrated in exactly the deliveries that cannot correct a target by construction.

### 4.5 Stratified by error category

Counts read: no step / step-no-English / below-floor / not-enriched / localized. G is the number of dialogues contributing to that category.

| Arm | Category (n; G) | Partition | Delivered |
|---|---|---:|---:|
| A | deletion (290; G=19) | 83 / 78 / 70 / 29 / 30 | 30 |
| A | wrong-language sub. (90; G=12) | 0 / 73 / 12 / 0 / 5 | 5 |
| A | same-language sub. (77; G=17) | 0 / 0 / 31 / 28 / 18 | 18 |
| B | deletion (290; G=19) | 83 / 0 / 71 / 61 / 75 | 114, incl. 39 no-step |
| B | wrong-language sub. (90; G=12) | 0 / 0 / 38 / 47 / 5 | 5 |
| B | same-language sub. (77; G=17) | 0 / 0 / 38 / 19 / 20 | 20 |
| C | deletion (290; G=19) | 83 / 0 / 106 / 46 / 55 | 55 |
| C | wrong-language sub. (90; G=12) | 0 / 0 / 55 / 24 / 11 | 11 |
| C | same-language sub. (77; G=17) | 0 / 0 / 37 / 19 / 21 | 21 |

Category utterance sets overlap, so the denominators and corruptions below are **not additive**.

| Arm | Category | Corrections | Corruptions / at risk | E_eff all; active | PIER | MER | WER |
|---|---|---:|---:|---:|---:|---:|---:|
| A | deletion | 0/290, G=19 | 0/4,210 | 2.0758; 20.0659 | 0.550499 | 0.336144 | 0.862340 |
| A | wrong-language sub. | 0/90, G=12 | 0/867 | 0.0213; 0.3832 | 0.781513 | 0.389051 | 0.935484 |
| A | same-language sub. | 0/77, G=17 | 0/3,520 | 4.4960; 19.2331 | 0.335017 | 0.169789 | 0.802495 |
| B | deletion | 0/290, G=19 | 1/4,210 | 6.5385; 16.6331 | 0.550499 | 0.336303 | 0.862340 |
| B | wrong-language sub. | 0/90, G=12 | 0/867 | 0.0174; 0.3137 | 0.781513 | 0.389051 | 0.935484 |
| B | same-language sub. | 0/77, G=17 | 0/3,520 | 0.5963; 2.2957 | 0.335017 | 0.169789 | 0.802495 |
| C | deletion | 0/290, G=19 | 1/4,210 | 2.4251; 12.7869 | 0.550499 | 0.336303 | 0.861626 |
| C | wrong-language sub. | 0/90, G=12 | 0/867 | 0.0255; 0.2087 | 0.781513 | 0.389051 | 0.935484 |
| C | same-language sub. | 0/77, G=17 | 1/3,520 | 6.3370; 23.2357 | 0.335017 | 0.170026 | 0.802495 |

Every category correction rate is 0.0000 [0.0000, 0.0000] — G=19 deletion, G=12 wrong-language substitution, G=17 same-language substitution — and every one of those intervals is degenerate for the reason in §4.3. The remaining 32 targets (17 phonetic/transliteration, 12 boundary, 3 other) also correct zero in every arm; their full metrics are in the authenticated summary artifact.

### 4.6 Spillover, and the one thing that is not zero

| Arm | Outside-target units changed | Improved | Damaged | Utterances with any | Per-utterance distribution |
|---|---:|---:|---:|---:|---|
| A | 0 | 0 | 0 | 0/116 | 116 zeros |
| B | 1 | 0 | 1 | 1/116 | 1, then 115 zeros |
| C | 3 | 1 | 2 | 3/116 | 1, 1, 1, then 113 zeros |

Affected utterances: B — `ZH-CN_U1004_S0_221` (damaged); C — `ZH-CN_U0028_S0_120` (damaged), `ZH-CN_U1021_S0_68` (improved), `ZH-CN_U2004_S0_281` (damaged).

Unlike Day 7's non-oracle arm, spillover is **not** uniformly zero once transport is given better input. It remains sparse, concentrated, and — crucially — it never includes a target correction. Four changed units across two arms is descriptive and should be quoted with its denominator.

### 4.7 Controls: correctly not run

No arm produced a non-zero correction rate, so neither the label-permuted null nor the matched-energy random control was warranted under the rule declared before the run: when the treated direction produces no target correction, a direction control cannot distinguish an absent transcript effect from a mis-specified one. **Zero control rows were generated.** That decision was made mechanically after all three arms completed; no result was used to tune the action, transport, abstention, or target set.

### 4.8 How far this null actually reaches — state this in the paper

This is the part that must not be overstated in either direction.

**What the run establishes:** the diagnosed state — a decoder step exists but the hypothesis has no English to attend to — was eliminated completely (81.1% → 0%, 37.7% → 0%), delivered coverage more than doubled (57 → 147), delivered energy rose, and **the correction count stayed at exactly zero**. Putting English into the transport input is **not sufficient** to recover transcript corrections under the frozen mechanism. The 81.1% cannot be presented as the demonstrated cause of Gate B's failure.

**What the run does not establish**, and the arithmetic is unforgiving:

| Arm | Delivered | Transport hits (±1 step) | Hit precision | Expected corrections at the oracle rate 1.73% |
|---|---:|---:|---:|---:|
| A | 57 | 7 | 12.3% | 0.12 |
| B | 147 | 25 | 17.0% (23.1% of the 108 gold-step deliveries) | 0.43 |
| C | 96 | 12 | 12.5% | 0.21 |
| **all three** | 300 | **44** | — | **0.76** |

If only interventions landing within ±1 of the gold step can correct, and they correct at the oracle arm's per-opportunity rate of **7/405 = 1.73%** (§2), then all three arms together predict **0.76 corrections**, and observing zero has probability **≈0.47**. A coin-flip. **This design cannot distinguish "hypothesis content is irrelevant" from "the effect is real and the run had ~44 chances to see something that happens 1.7% of the time."**

So the honest claim is the narrow one: *removing the diagnosed input absence did not restore corrections, and the burden is now on anyone asserting the 81.1% is causal.* The broad claim — *transport input never matters* — is not supported, and neither is any statement that the downstream blocker has been identified. It has not been.

The binding constraint that this run does surface is **step selection**: even with the gold transcript in hand, transport lands within ±1 of the gold step in only 23.1% of its gold-step deliveries. That is the number a follow-up should attack, and it is measurable without any oracle.

### 4.9 Execution and immutability

The single job wrote `smoke_001` first (8 targets, 3 utterances), authenticated it, and recorded `smoke_used_as_input=false`; the smoke was never promoted or consumed. Frozen identity matched before smoke, between smoke and full run, and after: Git HEAD `fdd441a7…`, code/config snapshot `b9b950be…`, test snapshot `61f8ba27…`. Every full artifact passed identity-required verification. No existing artifact was overwritten, deleted, or modified — not production `status/`, `freeze/`, the exposure ledger, existing candidate/role/POI artifacts, or any prior v2r3 generation. No gate generation was allocated or consumed. Production Gate A is unchanged.

New artifacts under `hypothesis_substitution/generation_001`: `hypothesis_substitution_targets.parquet`, `hypothesis_substitution_utterances.parquet`, `partial_construction.parquet`, `hypothesis_substitution_summary.json`, plus an authentication sidecar each. There is no controls parquet because controls were not warranted.

*Verification caveat carried from the source document:* 43 focused v2r3 tests passed before submission and the new module passed Ruff and compilation, but the **full suite (701 collected) reached the 900-second verification timeout at 71% with no failures observed. It is incomplete and is not a full-suite pass.**

---

## 5. The claim ladder — updated after Day 8

| # | Claim | Status |
|---|---|---|
| 1 | Language information is linearly available at the decoder | **Established** |
| 2 | Site-specific causal leverage over the prediction | **Established**, and not the prompt mechanism |
| 3 | Correctability is asymmetric across sites | **Established** — Site E null survives reweighting and three pooling schemes |
| 4 | A subset of errors remains correctable in transcript space | **Established but net harmful** — 7 corrections vs 13 corruptions at the oracle bound |
| 5 | Practical inference-time mitigation | **REFUTED** — 0/489 under inference-available localization, at every tier |
| 5a | *The failure is caused by the hypothesis lacking the English token* | **NOT ESTABLISHED** — the state was removed in two oracle arms and corrections stayed at 0/489. Correlate, not demonstrated cause (§4) |
| 5b | *The mechanism blocking transcript correction is identified* | **Open.** Day 8 removed the leading candidate without supplying a replacement |
| 6 | Geometry predicts causal response | Not attempted |
| 7 | Generality across models/corpora | Not attempted |

Rungs 1–3 are solid. Rung 4 is real but negative in sign. Rung 5 fails. **Rung 5a is the one that changed on 17 August, and it changed against us** — the failure is now diagnosed less well than it was on 16 August, not more.

---

## 6. What you explicitly will not have

State these plainly in Limitations rather than letting a reviewer find them:

- **One model, one corpus, one language pair.** Whisper-large-v3, CS-Dialogue, Mandarin–English.
- **No localizer.** W3. Days 7–8 use transport, not a learned localizer.
- **No utility selector.** W4–W5.
- **No PEFT comparison.** LoRA is W3; you cannot yet say what fraction of a fine-tune's gain a frozen rank-1 edit recovers.
- **No estimator comparison.** LDA and SVM ablations are implemented-and-tested but not run. Your claim is "this direction works," not "difference-in-means is the right estimator."
- **Rank 1 only.** Nothing bounds what higher rank could achieve.
- **K=1 per utterance** caps transcript-level gain by design.
- **Two aligner paths, not three** — cross-attention is a documented negative result.
- **Construction used wholly-correct spans**, costing 44% of eligible spans (233 of 419).
- **G ≈ 20** clusters; wild bootstrap with t(G−1) throughout. At G=12 the floor p-value is ~2.4×10⁻⁴.
- **No identified cause for the transcript-space null.** Day 8 eliminated the leading candidate. You can describe the failure precisely and you cannot yet explain it.
- **The Day 8 arms are underpowered against an oracle-size effect** — 44 near-gold deliveries across three arms, expected yield 0.76 corrections (§4.8).
- **The only positive transcript result has effective G = 2.** All 7 oracle corrections sit in 2 dialogues; 18 of 20 clusters contribute zero (§2).
- **Day 6's per-cell numbers are not currently reproducible** and Day 6 has no result document. The invariance conclusion stands; the table does not, yet (§1.6).
- **No current full-CPU-suite pass is on record.** 701 tests collect cleanly; the last full run timed out at 71% with no failures observed.

---

## 7. The paper — revised on 17 August

**Gate B does not pass on either arm.** Non-oracle: 0 corrections. Oracle: corrections do not exceed corruptions. The preregistered fallback is active, and this is the shape planned for since v5 of the proposal.

**What changed today:** the fallback paper was going to be built on the 81.1% as an explanation. That sentence is no longer available. The finding survives as a *measurement* of the localization problem's structure; it does not survive as the *cause* of the intervention's failure.

### The paper you have

*Code-switching errors remain correctable in a frozen model's decoder representations, and that correctability does not survive contact with free decoding — not because the intervention is too weak, and not, as we predicted and tested, because the first-pass hypothesis lacks the English token to localize against.*

Every component is measured:

| Finding | Evidence | Status |
|---|---|---|
| Causal leverage at decoder integration | +0.106 [+0.047, +0.165], G=19, beats four controls | Established |
| Not the language-token mechanism | `cos(v_nat, U_prompt)` = 0.010–0.030; prompt subspace removes 0.03–0.10% | Established |
| **No** leverage at encoder evidence | Null under inverse-variance reweighting and three pooling schemes | Established |
| Dose-response rises, peaks, declines | Peak ρ=2, zero-crossing by ρ=4 | Established |
| Invariant to alignment choices | Sign preserved in every valid cell, 2 paths × 4 conventions × reweighting | Established |
| Crosses into transcript space only under oracle localization, and net harmfully | 7 corrections : 13 corruptions | Established |
| Does not convert to transcript improvement in practice | 0/489 non-oracle at every tier | Established |
| The hypothesis usually contains no English to localize against | 81.1% of wrong-language substitutions, 37.7% of deletions | Established **as a measurement** |
| That absence is what blocks the correction | **Tested and not supported** — 0/489 in both oracle-input arms | **Refuted as a sufficient cause** |
| Not a dose problem | Non-oracle delivers more energy (17.49 vs 4.83) and corrects nothing; B/C deliver more coverage and correct nothing | Established |
| Spillover is concentrated, not diffuse | Top 2 of 116 utterances = 85.9% | Established |
| Transport lands near the gold step rarely, even given gold text | 23.1% of B's gold-step deliveries | New, and the best lead |

**The honest framing is now the negative-result framing, and it is a stronger paper than a just-so story would have been.** The 81.1% stays in as a structural measurement of why first-pass-conditioned localization is hard — it is real, predicted in advance, and reproduces exactly. What must go is the causal sentence. We ran the experiment that would have promoted it, preregistered all three outcomes, got the third, and are reporting it.

A reviewer who asks "did you check whether the missing token was the cause?" now gets an experiment rather than an argument. That is worth more than the claim we lost.

### Venue

Interpretability-first: BlackboxNLP, or an *ACL Interpretability and Analysis track (Findings a realistic outcome). ARR October cycle, 12 Oct.

### What this changes about the plan

W3–W5 as scoped are **no longer needed**. The localizer, utility selector, PEFT baselines, and selectivity frontier were the method paper. The core experiments for the analysis paper are complete as of today, 17 August, against a 12 October deadline.

---

## 8. What is worth adding, with eight weeks free

Reordered after Day 8. The former item 1 is done and came back negative.

**1. Step-selection accuracy, on its own terms.** §4.8's 23.1% is the new leading candidate and needs no oracle to measure. Transport picks a step; how often is it right, what does the distribution of `selected_step − oracle_step` look like, and does the enrichment rule concentrate or scatter it? If the answer is that transport is near-random in step choice, that explains the null without any appeal to hypothesis content, and it is a clean, honest mechanism. **Highest value, cheap, and it uses artifacts already on disk.**

**2. Power the correction measurement, or stop reporting it as an effect.** Every transcript-space claim in this project rests on ≤7 events. Either enlarge the target population until an oracle-size effect is detectable, or state throughout that transcript-level correction is descriptive and the representation-level result is the finding. The second is free and should happen regardless.

**3. SEAME transfer.** Generality was optional for a method paper; for an analysis paper claiming a structural property of CS-ASR intervention, replication on a second corpus is worth a great deal. Does the decoder-leverage / encoder-null asymmetry hold? Does the no-English-in-hypothesis rate reproduce? Licensing permitting.

**4. Estimator ablation.** LDA and SVM are implemented and unit-tested but never run. One run each. Lets you say covariance-aware estimation does not change the conclusion — small, clean, closes an obvious reviewer question.

**5. Abstention-rule characterisation.** Given spillover concentrates in 2 of 116 utterances, characterise what those utterances have in common. If they are identifiable in advance, that is a concrete recommendation rather than a caveat.

Not worth adding: higher rank, learned directions, SAEs, multi-span correction. The paper's strength is that one variable is isolated cleanly.

---

## 9. Process notes for the write-up

**Four defects were caught: two before a run, two after.**

The abstention threshold τ=0.5 was scale-wrong — Whisper's padded attention keeps `r_q` near 0.1, so it would have abstained always and made the arm vacuous by construction. Replaced with a parameter-free rule **before any effect was known**, with `r_max` recorded so any threshold can be read off post hoc. That ordering matters and should be stated.

`tag_unit(str(unit))` passed the repr rather than the surface, so nothing was ever tagged English — caught because D-test reported 0 embedded-English units across 6,257 utterances. It affects only monolingual retention and D-test headroom; correction, corruption, spillover, transport, and control results never call it. PIER independently confirms the corrected counts (num_poi 35,921, errors 10,458 — exact match).

**The Day 5 corruption-counting error is the one to disclose in the paper**, since it inverted the sign of the oracle result. Count corruption over baseline-correct units, not within a target set composed entirely of baseline errors.

`control_step_for()` mapped a negative offset onto the onset itself, making `jitter_zh −200 ms` identically zero. Repaired on Day 7; the repaired cell is a null-seed replicate, not an independent probe (§3.5).

**Day 6's ±50/±100 ms jitter cells are inert by construction** at 3.38 tokens/s — sub-token boundary error cannot displace a decoder-site intervention. Present this as a finding about why Site D is robust, not as six confirmations.

**The one process failure worth fixing is documentary.** Every stage except Day 6 produced a dated result document at the time of the run. Day 6 did not, and the consequence surfaced only when its cell effects were checked against the artifact eight days later and failed to reproduce — see §1.6. The artifacts were fine; the analysis path was simply never written down. Write `docs/V2R3_INVARIANCE_2026-08-16.md` before drafting the invariance section.

**Day 8 is the methodological set-piece.** Three outcomes were written down before the run, the negative one landed, and it was reported without adjustment. Arm A reproduced Day 7 bit-for-bit on all 489 rows, so the reference is a reproduction rather than a re-estimate; Arm C's construction was well defined on 489/489 targets, so there is no skip-rate escape hatch; and controls were withheld by a rule fixed in advance rather than by judgement after seeing zeros. Say all of that in the paper — it is the difference between a null result and an uninformative one.

---

### Provenance of this document

| Section | Source |
|---|---|
| §1 | Days 3–6 direction, invariance, and dose-response runs |
| §2 | `docs/V2R3_DAY5_EXPANSION_2026-08-16.md` |
| §3 | `docs/V2R3_GATE_B_2026-08-16.md` (Slurm 41162) |
| §4 | `docs/V2R3_HYPOTHESIS_SUBSTITUTION_2026-08-17.md` (Slurm 41736), re-verified against `hypothesis_substitution/generation_001` |
| §4.8 | derived here from §2's 7/405 oracle rate and the Day 8 transport-hit counts; assumptions stated in place |

No number in this document is a production Gate-A result. Production Gate A remains as recorded in `docs/RUNBOOK_V2_2026-08-14.md` and is unchanged by Days 4–8.
