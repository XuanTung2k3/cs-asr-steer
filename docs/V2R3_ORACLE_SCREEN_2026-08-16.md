# Day 4 — the teacher-forced oracle screen

**Date:** 2026-08-16
**Job:** Slurm 40953, `COMPLETED`, exit 0, **24 m 37 s**
**Destination:** `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/oracle_screen/generation_001`
**Scale:** 449 conditions × 489 targets = **219,561 scored rows**

**Headline: Site D shows a positive, dose-responsive effect that beats all four
controls at three of four layers. Site E beats no control anywhere, and at the
deepest layer is actively harmful.** Numbers below; no gate is declared and no
layer or strength is selected.

---

## 1. Purpose and boundary

Teacher-forced scoring only — one forward pass per utterance per condition, gold
transcripts fixed. **No free decoding**; that is Day 7. This session evaluates no
gate, declares no readiness, and selects neither layer nor strength.
`D-dev-confirm` and `D-test` were never opened.

---

## 2. Provenance — a correction to the instruction

The Day 4 instruction required verifying the Day 3 directions through the split
identity branch. That check passed at pre-flight and then **failed**, with
`differs in ['code_config_sha256']`. The cause was writing
`v2r3_oracle_screen.py` itself: the consuming module changed
`code_config_sha256`, so the artifact it was written to read no longer matched
the running tree. The check as specified was self-defeating — the screen must
exist before it can run.

Resolution (Option 1, authorised): the directions are **classified `split`,
byte-verified against the recorded sha256, and identity is not required**, with
that reason recorded in the screen's manifest. This is the standard
`lss_v2_candidates._parents` already applies to role manifests. The split gate
remains fully in force for the artifacts this session **publishes**.

The general rule is now in the runbook's identity section: identity applies to
what a session publishes and to artifacts consumed by pre-existing code, never
to artifacts consumed by newly written code. This was the fourth occurrence of
the pattern and the first where the consuming module was the cause.

| Input | Format | Check |
|---|---|---|
| `directions.npz` and Day 3 siblings | split | byte-verified, identity not required |
| candidate table `cdc0ff4e…` | legacy | bytes |
| role manifests `d61c305e…`, `c0f3ad12…` | legacy | bytes |
| POI tables `334feed9…`, `5cebe272…` | legacy | bytes |
| `directions/smoke_001` | — | **excluded**; the driver refuses any path containing "smoke" |

All three published artifacts are **split** format.

---

## 3. Subset — confirmed, and the strata reconciled

482 spans, **489 error units**, **20 dialogues**, 116 utterances — matching Day 3
exactly. Span duration: min 400 ms, median 2,281 ms, P90 5,782 ms, max 11,404 ms.
Every target is baseline-error.

### The strata discrepancy, fully explained

Day 3 reported substitution-only as 256 units and wrong-language as 108. The
correct unit-level counts are 199 and 90. The objection that a span-level
over-count should also inflate all-errors was right, and the resolution is
structural:

1. **All 489 units satisfy every eligibility condition individually** — duration
   floor 489/489, matrix-preceded 489/489, not-utterance-final 489/489,
   contiguous indices 489/489. Eligibility is a span property and units inherit
   it; nothing is excluded by eligibility anywhere in this analysis.
2. **The 57-unit difference is entirely deletions** — `{'deletion': 57}` — that
   share a span with a substitution. None fails any eligibility condition. The
   exclusion is **by category, not by eligibility**. Wrong-language: 108 − 90 =
   18 co-located non-wrong-language errors.
3. **All-errors is unaffected because its filter and its summand agree.** The
   filter "span contains ≥1 error" selects exactly the spans holding the units
   being summed, so sum = count (489 = 489, verified). Inflation appears only
   when a span-level filter is paired with a unit-level sum over a *different*
   category — i.e. only in sub-strata.

**Mechanism for units in all-errors but not substitution-only:** deletions
co-located in a span that also contains a substitution. Correctly errors,
correctly not substitutions.

| Stratum | Units | **G** | Utterances |
|---|---:|---:|---:|
| all errors | 489 | **20** | 116 |
| substitution-only | 199 | **19** | 84 |
| wrong_language_substitution (descriptive) | 90 | **12** | 20 |

Deletions 290/489 = **59.3%**.

---

## 4. Results, in the order requested

All contrasts are **paired** — per-unit difference between the correct direction
and the control — bootstrapped by wild cluster bootstrap on `dialogue_id` with
Rademacher weights and t(G−1) critical values. `*` marks an interval excluding
zero. Outcome is gold-token logprob unless stated.

### 4.1 Does the correct direction beat the LABEL-PERMUTED null?

The strongest null: the direction rebuilt from labels shuffled within
nuisance-matched strata, through the identical pipeline including ridge
residualisation.

**Site D — yes, at layers 8, 16 and 31, at every ρ, in every stratum.**
**Site E — no, nowhere.**

Substitution-only, G = 19:

| Site | Layer | ρ=0.25 | ρ=0.5 | ρ=1.0 | ρ=2.0 |
|---|---|---|---|---|---|
| E | 15 | +0.002 | +0.002 | +0.003 | +0.009 |
| E | 23 | +0.019 | +0.036 | +0.028 | −0.004 |
| E | 27 | +0.006 | +0.012 | +0.038 | +0.035 |
| E | 31 | −0.004 | −0.018 | −0.069 | **−0.453\*** |
| D | 8 | **+0.008\*** | **+0.017\*** | **+0.031\*** | **+0.066\*** |
| D | 16 | **+0.024\*** | **+0.054\*** | **+0.106\*** | **+0.178\*** |
| D | 24 | +0.027 | +0.033 | −0.009 | **−0.350\*** |
| D | 31 | **+0.027\*** | **+0.050\*** | **+0.085\*** | +0.069 |

Strongest single result: **Site D layer 16, ρ=1, +0.1057 [+0.0466, +0.1649],
G=19**. No Site-E interval excludes zero in the positive direction at any layer
or ρ. Site E layer 31 at ρ=2 excludes zero **negatively** — the direction is
worse than its own null.

**The other three controls agree.** At Site D layers 8/16/31 the correct
direction beats wrong-location, matched-norm random, and opposite-sign at every
ρ, all intervals excluding zero. At Site E no control is beaten at any layer or
ρ, with two exceptions that are artefacts of harm rather than benefit: at ρ=2 the
opposite-sign contrast is large and positive at layers 23 and 27 (+0.554, +0.508)
because the flipped direction is severely damaging, not because the correct one
helps.

Null norm ratios ‖Δ_null‖/‖Δ_real‖ span **0.175–0.619** (Site D median 0.313,
Site E 0.377). A sign-flipped copy would give exactly 1.000, so the nulls were
genuinely rebuilt through the pipeline, not negated.

### 4.2 Is the dose-response monotone and saturating?

Mean Δ gold logprob against C00, correct direction, substitution-only:

| Site | Layer | ρ=0.25 | ρ=0.5 | ρ=1.0 | ρ=2.0 | Shape |
|---|---|---|---|---|---|---|
| E | 15 | +0.003 | +0.003 | +0.008 | +0.020 | monotone, negligible |
| E | 23 | +0.021 | +0.044 | +0.051 | +0.039 | non-monotone |
| E | 27 | +0.004 | +0.011 | +0.037 | +0.038 | monotone, saturating |
| E | 31 | −0.007 | −0.020 | −0.072 | −0.463 | monotone **negative** |
| D | 8 | +0.006 | +0.015 | +0.028 | +0.061 | **monotone, not yet saturating** |
| D | 16 | +0.020 | +0.045 | +0.089 | +0.147 | **monotone, not yet saturating** |
| D | 24 | +0.029 | +0.040 | +0.007 | −0.322 | non-monotone, collapses |
| D | 31 | +0.025 | +0.046 | +0.081 | +0.064 | non-monotone, peaks at ρ=1 |

**Partially as predicted.** Site D layers 8 and 16 are cleanly monotone and still
rising at ρ=2 — monotone but **not saturating within the tested grid**. Layer 31
peaks at ρ=1 and turns down, the closest thing to saturation observed. Layer 24
collapses at ρ=2. The expected "monotone and saturating" shape is realised in its
monotone half only; the grid does not reach saturation for the two strongest
layers.

### 4.3 Site E versus Site D on substitutions, G = 19

Unambiguous: **Site D carries the effect; Site E does not.** At ρ=1 against the
label-permuted null, Site D layer 16 gives +0.106 [+0.047, +0.165]\* while the
best Site E layer (27) gives +0.038 [−0.013, +0.089], interval including zero.
Every Site-E interval at every layer and ρ includes zero except the harmful ones
at layer 31.

Secondary outcomes at Site D layer 16, ρ=1, substitution-only: **margin +0.119
[+0.036, +0.202]\*** confirms the logprob result. Gold rank improves by 42.6
positions on average but the interval includes zero (−85.3, +0.14) — rank is
heavy-tailed and G=19 does not resolve it. Entropy is flat, +0.010 [−0.021,
+0.040].

### 4.4 wrong_language_substitution, G = 12 — DESCRIPTIVE ONLY

At G=12 the wild bootstrap admits 2¹² = 4,096 sign vectors, so the smallest
attainable p-value is ≈2.4 × 10⁻⁴. That is adequate to **detect** an effect and
close to uninformative as a null; a null here is not evidence against the
hypothesis.

Site D layer 16 is positive and interval-excluding-zero at every ρ: +0.021,
+0.044, +0.075, +0.114 (all \*). Layer 31 likewise at ρ ≤ 1. Site E is null
everywhere except the harmful layer-31/ρ=2 cell. The pattern **matches** the
substitution-only stratum in direction and ordering, which is the most that can
be claimed at this cluster count.

---

## 5. Deletion versus substitution — and a correction to the prediction

At ρ=1, correct versus label-permuted:

| Site | Layer | Substitution (G=19) | Deletion (G=19) |
|---|---|---|---|
| E | 15 | +0.003 | +0.002 |
| E | 23 | +0.028 | −0.004 |
| E | 27 | +0.038 | **−0.013\*** |
| E | 31 | −0.069 | **−0.082\*** |
| D | 8 | **+0.031\*** | **+0.046\*** |
| D | 16 | **+0.106\*** | **+0.157\*** |
| D | 24 | −0.009 | +0.114 |
| D | 31 | **+0.085\*** | **+0.130\*** |

**Site D moves deletion-category units at least as much as substitutions — the
opposite of the predicted by-construction null.** The prediction is not wrong; it
does not apply here. "Site D cannot act at a decoder step that does not exist" is
a property of **free decoding**, where an omitted word has no step. Under
**teacher forcing** the gold sequence is supplied, so every gold token has a
decoder step regardless of what the baseline produced, and Site D can act at all
of them. The by-construction null becomes testable at Day 7, not here.

This matters for interpretation: the Day 4 deletion result **cannot** be read as
evidence that Site D will repair deletions in free decoding.

---

## 6. C11 — both sites, and the off-distribution check

Δ^D was estimated on unintervened decoder states, so in C11 it is applied off the
distribution it was fit on. Reported separately from C10 and C01, as required.

| Pair | ρ | C11 vs wrong-location | C11 Δ | C10 Δ | C01 Δ | Interaction (C11 − additive) |
|---|---|---|---|---|---|---|
| 15+8 | 1.0 | +0.029 [−0.009, +0.067] | +0.039 | +0.008 | +0.028 | **+0.003** |
| 23+16 | 1.0 | **+0.136 [+0.039, +0.234]\*** | +0.125 | +0.051 | +0.089 | **−0.015** |
| 27+24 | 1.0 | +0.033 [−0.100, +0.167] | +0.022 | +0.037 | +0.007 | −0.022 |
| 31+31 | 1.0 | −0.010 [−0.109, +0.090] | −0.004 | −0.072 | +0.081 | −0.014 |
| 23+16 | 2.0 | **+0.195 [+0.053, +0.338]\*** | +0.198 | +0.039 | +0.147 | +0.012 |
| 27+24 | 2.0 | **−0.322 [−0.527, −0.118]\*** | −0.353 | +0.038 | −0.322 | −0.069 |
| 31+31 | 2.0 | **−0.417 [−0.626, −0.207]\*** | −0.509 | −0.463 | +0.064 | −0.111 |

**The interaction is small at ρ=1** (−0.022 to +0.003) — C11 is close to additive
where both components behave. It becomes materially negative only at ρ=2 on the
layer pairs that are already collapsing individually (27+24: −0.069; 31+31:
−0.111). Reported as **descriptive**: the off-distribution penalty is not
detectable at moderate strength and is confounded at ρ=2 with the single-site
collapse at those layers.

---

## 7. Statistics

Wild cluster bootstrap, Rademacher weights, clustered on `dialogue_id`, 2,000
draws, seed 20260817, t(G−1) critical values, 95%. **G is stated with every
interval: 20 / 19 / 12 by stratum**, all below the ~30 at which the ordinary
cluster bootstrap is reliable — which is why the wild variant is used.

Every comparison is a **paired** contrast computed per unit, never two marginal
intervals compared by eye.

**Pair permutation was not implemented**, by design: for a paired mean
difference, Σᵢ(h⁺ᵢ − h⁻_π(i))/N = mean(h⁺) − mean(h⁻) for every π, so it is
algebraically incapable of being a null. Labels were permuted within
nuisance-matched strata instead.

---

## 8. Verification

| # | Check | Result |
|---|---|---|
| 1 | Subset composition | 489 targets, 20 dialogues, 116 utterances, **all baseline-error**; duration min 400 / median 2,281 / max 11,404 ms |
| 2 | Bootstrap by `dialogue_id` | confirmed: Rademacher weights drawn over groups; `paired_contrast` merges `dialogue_id`; the word `utterance_id` never appears in the resampling code |
| 3 | C11 ordering | structural, not scheduled — the encoder hook fires inside the encoder forward, cross-attention consumes the steered states unmodified, the decoder hook fires on the post-cross-attention residual |
| 4 | Label-permuted null used the full pipeline | calls `D3.assemble`, which performs ridge residualisation; norm ratios 0.175–0.619, where a mere sign flip would give exactly 1.000 |
| 5 | `smoke_001` excluded | directions source is `directions/generation_001`; the driver refuses any path containing "smoke"; both screen smoke destinations were cleared |
| 6 | Immutability | `artifacts_lss` 2026-08-12 02:41:33, `artifacts_v2` 2026-07-28 10:46:20, `artifacts_dialogue_v2` 2026-08-14 17:00:10, `v2r2` 2026-08-14 19:41:13; exposure ledger **1786502490** |
| 7 | Code freeze | `9d200923f2446aef79dc2ef6227ef8701173ae0089b6ee642c5d88474a7e2fb7` identical before publication, inside job 40953, and after completion |

### Artifacts

| sha256 | Bytes | Artifact | Format |
|---|---:|---|---|
| `7f0e48de9e8de24f475f2aac3be7bdc9…` | 3,709,239 | `oracle_scores.parquet` | split |
| `4521795658e76cd636c232083b042fcc…` | 19,574 | `oracle_targets.parquet` | split |
| `a1445dec2ce6df18c10f2e6197736ef4…` | 4,188 | `oracle_screen_report.json` | split |

Two smoke runs preceded the screen: 40946 failed at the parquet write (the
`layer` column mixes ints for C10/C01 with `"enc+dec"` strings for C11) and 40952
was clean at 449 conditions. Both destinations were cleared deliberately before
the real run.

---

## 9. Open items

1. **The dose-response does not saturate within ρ ≤ 2** at the two strongest Site
   D layers. Whether the curve turns over beyond ρ=2 is untested.
2. **Layer 24 behaves differently from 8, 16 and 31** at both sites — positive at
   low ρ, collapsing at ρ=2. Unexplained.
3. **Gold rank does not resolve at G=19** despite a mean improvement of 42.6
   positions; the outcome is heavy-tailed.
4. **The deletion result cannot transfer to free decoding** (§5). Day 7 is where
   the by-construction prediction becomes testable.
5. **Site E is null or harmful throughout.** Whether that reflects the encoder
   site, the span-level pooling, or the direction construction is not
   distinguished by this screen.

**No gate was evaluated. No layer or strength was selected. Production Gate A is
unchanged.**
