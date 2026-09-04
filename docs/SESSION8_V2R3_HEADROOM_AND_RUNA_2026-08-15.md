# Session 8 — headroom and Run A regenerated on the v2r3 population

**Date:** 2026-08-15
**Diagnostic root:** `/mnt/data/tungnx/cs-asr-steer/diagnostics/session8_v2r3_2026-08-15`
**Population:** v2r3 candidate generation `cdc0ff4e…`, v2r3 baseline POI tables,
partition fingerprint `b6c1af90…`, development roles only.

---

## 1. Purpose and scientific boundary

Two analyses went stale when the sample and the independence unit changed: the
headroom estimate and the Run A convention diagnostic, both computed on the v1
300-per-role sample under conversation-level clustering. This session regenerates
both on the v2r3 dialogue-atomic population, characterises `whisper_dtw`'s
invalid units, and states the realized cluster counts.

**This run selects no convention, sets no eligibility floor, evaluates no gate,
and declares no readiness.** It recommends nothing. Every number is reported so
the decisions remain open.

Read-only with respect to production: no production execution path was modified,
no production artifact was written or read for writing, and every artifact
produced carries `development_only_diagnostic` taint in its manifest. Roles are
restricted to `D-construct` and `D-dev-select`; `D-dev-confirm`, `D-test`, and
every gate generation were untouched.

Run A used `src/csasr/experiments/ctc_convention_diagnostic.py` and
`src/csasr/lss/align/conventions.py` **unmodified**. One new read-only module,
`src/csasr/experiments/v2r3_headroom_diagnostic.py`, computes the headroom and
`whisper_dtw` sections; it writes only to the diagnostic root passed as an
argument.

The eligibility rule was applied exactly as specified, with **no confidence
condition**: embedded-English lexical unit, aligned span duration above the
floor, matrix-language preceding context, not utterance-final, contiguous
reference-unit indices. Span durations come from `existing_ctc`, the provisional
family the annotation pack and Run A already used — recorded here, not chosen
here, and the only family with a 0% invalid rate.

---

## 2. Part 1 — headroom

### 1a / 1b — structural spans and baseline-error units

**D-dev-select**

| Floor | Spans | Lexical units | Utterances | **Dialogues** | Speakers | Baseline-error | Baseline-correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥400 ms | 409 | 1,152 | 197 | **20** | 40 | 455 | 697 |
| ≥300 ms | 487 | 1,234 | 211 | **20** | 40 | 489 | 745 |
| ≥200 ms | 526 | 1,275 | 221 | **20** | 40 | 506 | 769 |

**D-construct**

| Floor | Spans | Lexical units | Utterances | **Dialogues** | Speakers | Baseline-error | Baseline-correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥400 ms | 356 | 1,032 | 165 | **20** | 39 | 372 | 660 |
| ≥300 ms | 416 | 1,096 | 182 | **20** | 40 | 394 | 702 |
| ≥200 ms | 446 | 1,129 | 193 | **20** | 40 | 408 | 721 |

**All 20 dialogues per role survive at every floor**, against the ~20 expected.
The floor costs utterances and units, never clusters. Structural totals before
the floor: D-dev-select 729 embedded-English runs, 556 structurally eligible;
D-construct 682 and 464.

### 1c — baseline-error units by category

Units / dialogues / speakers.

**D-dev-select**

| Category | ≥400 ms | ≥300 ms | ≥200 ms |
|---|---|---|---|
| `deletion` | 271 / 18 / 30 | 286 / 19 / 32 | 295 / 20 / 33 |
| `wrong_language_substitution` | 89 / 11 / 14 | 92 / 13 / 16 | 94 / 13 / 16 |
| `same_language_substitution` | 68 / 17 / 24 | 78 / 17 / 25 | 82 / 17 / 26 |
| `boundary_error` | 12 / 8 / 9 | 12 / 8 / 9 | 12 / 8 / 9 |
| `phonetic_transliteration_or_script` | 13 / 7 / 9 | 18 / 9 / 12 | 20 / 10 / 13 |
| `other` | 2 / 1 / 1 | 3 / 2 / 2 | 3 / 2 / 2 |

**D-construct**

| Category | ≥400 ms | ≥300 ms | ≥200 ms |
|---|---|---|---|
| `deletion` | 203 / 18 / 24 | 212 / 18 / 25 | 219 / 18 / 25 |
| `wrong_language_substitution` | 74 / 14 / 19 | 77 / 15 / 20 | 79 / 15 / 20 |
| `same_language_substitution` | 60 / 16 / 27 | 65 / 16 / 27 | 69 / 17 / 28 |
| `boundary_error` | 19 / 8 / 11 | 20 / 9 / 12 | 20 / 9 / 12 |
| `phonetic_transliteration_or_script` | 15 / 11 / 12 | 19 / 12 / 13 | 20 / 12 / 13 |
| `other` | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |

### The three derived subsets — D-dev-select

Units / dialogues / speakers.

| Floor | All baseline-error | Substitution-only (excl. deletion) | `wrong_language_substitution` only |
|---|---|---|---|
| ≥400 ms | **455 / 20 / 39** | **184 / 19 / 33** | **89 / 11 / 14** |
| ≥300 ms | 489 / 20 / 39 | 203 / 19 / 34 | 92 / 13 / 16 |
| ≥200 ms | 506 / 20 / 39 | 211 / 19 / 35 | 94 / 13 / 16 |

Deletion is the largest single category — 271 of 455 errors at ≥400 ms, 60% —
and Site D cannot reach any of them by construction, since there is no decoder
step for a token that was never emitted. The substitution-only count is
therefore the governing size for the Site-D arm, and it is under half the
headline. The sharpest test, `wrong_language_substitution`, is 89 units from
**11 dialogues** at ≥400 ms — the smallest cluster count anywhere in this
document.

### 1d — D-construct baseline-correct pool

| Floor | Correct units | Dialogues | Speakers |
|---|---:|---:|---:|
| ≥400 ms | 660 | 20 | 36 |
| ≥300 ms | 702 | 20 | 37 |
| ≥200 ms | 721 | 20 | 37 |

This is the pool direction construction draws from. It spans all 20 dialogues at
every floor.

### 1e — funnel

| Stage | D-construct | D-dev-select |
|---|---:|---:|
| Sampled utterances | 300 | 300 |
| ≥1 reference unit | 300 | 300 |
| ≥2 language runs | 300 | 300 |
| ≥1 eligible embedded-English span (any floor) | 199 | 232 |

Every sampled utterance carries reference units and at least two language runs —
the sampler drew from bilingual utterances by construction. The attrition is
entirely at the last stage: 66% and 77% of utterances contribute an eligible
embedded-English span, the rest being filtered by the matrix-preceding,
non-final, and contiguity conditions.

---

## 3. Part 2 — Run A on the v2r3 population

Existing code, unmodified. Development roles only. Paired count **3,031**
(v1: 3,028 — the sample changed completely; the near-identical size is because
both populations are 600 utterances, 300 per development role).

| Convention | Median | P90 | ≤100 ms | ≤200 ms | EN med | ZH med | EN−ZH | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `blank_excluded` | 458.4 | 1378.5 | 8.64% | 26.86% | 253.4 | 680.7 | −427.3 | 3,031 |
| `blank_to_preceding` | **340.1** | 1280.7 | **13.26%** | **34.48%** | 318.7 | 390.4 | **−71.7** | 3,031 |
| `blank_to_following` | 694.2 | 1560.4 | 1.75% | 9.11% | 606.9 | 721.0 | −114.2 | 3,031 |
| `blank_midpoint` | 467.3 | **1118.4** | 2.87% | 13.96% | 408.6 | 538.5 | −129.9 | 3,031 |

**Utterance-final split** (median ms, n):

| Convention | Utterance-final | n | Non-final | n |
|---|---:|---:|---:|---:|
| `blank_excluded` | 735.5 | 600 | 378.9 | 2,431 |
| `blank_to_preceding` | 735.5 | 600 | **261.6** | 2,431 |
| `blank_to_following` | 795.6 | 600 | 655.2 | 2,431 |
| `blank_midpoint` | 736.0 | 600 | 419.1 | 2,431 |

**Edge cases:** single blank run 2,369 (actionable); zero-length blank run 41;
non-contiguous reference-unit index 24; transitions total 2,434; transitions at
an utterance boundary 1,200; target language runs 3,034; utterances with target
runs 600. No multiple-blank-runs, no unusable edges, no non-finite boundaries, no
cross-language overlap. Blank runs: median 240.1 ms, P90 1001.2 ms, total 984.1 s.

### v1 findings, confirmed or refuted by name

**F1 — `blank_to_preceding` is best on median and both coverage bands.
HOLDS.**
Median 458.4 → 340.1 ms, the lowest of the four. Within-100 ms 8.64% → 13.26%,
the highest. Within-200 ms 26.86% → 34.48%, the highest. v1 was 440 → 340,
8.22% → 10.40%, 23.94% → 30.45%. The ordering and every direction reproduce; the
coverage gains are larger on v2r3.

**F2 — `blank_to_preceding` reduces the EN−ZH asymmetry roughly threefold.
CHANGED — the reduction is larger, about sixfold.**
−427.3 → −71.7 ms, a 5.96× reduction. v1 was −440 → −140, 3.1×. The finding's
direction and mechanism hold and are stronger here; only the "roughly threefold"
magnitude no longer describes it.

**F3 — utterance-final targets are unaffected by every convention.
CHANGED — unaffected by three of four, not all four.**
`blank_excluded` 735.5, `blank_to_preceding` 735.5, `blank_midpoint` 736.0 — flat
to within 0.5 ms. `blank_to_following` moves them to 795.6 ms, +60.1 ms. The
share is 600 of 3,031 = **19.80%**, against v1's 600 of 3,028 = 19.8% — an
almost exact reproduction of the proportion on a different sample. The v1 median
was 700 ms; here it is 735.5 ms.

**F4 — median blank run at language transitions is around 220 ms. HOLDS.**
240.1 ms median (v1: 220), P90 1001.2 ms (v1: 940). Both within 10% of v1 on a
disjoint sample. The finding that matters is unchanged: the median blank run is
far below the ~800 ms that the seam-relative synthetic measurement suggested, so
blank reassignment remains a term rather than the term.

**F5 — excluding utterance-final under `blank_to_preceding` gives a working
median around 280 ms. HOLDS.**
261.6 ms (v1: ~280). Combined with F1, the best convention plus utterance-final
exclusion takes the median from 458.4 ms to 261.6 ms on this population.

---

## 4. Part 3 — `whisper_dtw` invalid units

### 3a — the condition

`src/csasr/nat5h/schema.py:257-262` marks a unit invalid with
`failure_code="adjacent_overlap"` when its `start_sec` precedes the previous
canonical unit's `end_sec` by more than `overlap_epsilon_sec`. It is a condition
**between two adjacent units**, not a property of one interval — which is exactly
why `invalid_duration` and `nonmonotonic` are both zero. Those describe a single
interval's length and the ordering of starts; this describes an overlap between
neighbours. A start preceding the previous *start* would be `non_monotonic_span`,
a distinct code that did not fire.

Failure codes over development roles: `adjacent_overlap` **5,118**,
`timestamp_out_of_bounds` 1. Overall invalid rate 0.17392.

`csasr.lss.align.target_objects._edge_supported` explicitly admits an
`adjacent_overlap` unit as a run edge, because same-language tokenizer overlap
stays meaningful after union. Run A's aggregation reports
`same_language_overlap_merged: 5118` and `cross_language_overlap_invalid: 0` —
the same 5,118 units, all absorbed at run level.

### 3b — by span duration decile

| Decile | Range (ms) | Units | Invalid | Rate |
|---:|---|---:|---:|---:|
| 1 | 20–60 | 2,956 | 389 | 0.1316 |
| 2 | 60–120 | 3,245 | 143 | 0.0441 |
| 3 | 140–160 | 3,105 | 336 | 0.1082 |
| 4 | 160–200 | 2,501 | 379 | 0.1515 |
| 5 | 200–240 | 3,536 | 688 | 0.1946 |
| 6 | 240–280 | 2,543 | 558 | 0.2194 |
| 7 | 300–360 | 2,962 | 683 | 0.2306 |
| 8 | 360–460 | 2,718 | 634 | 0.2333 |
| 9 | 480–780 | 2,944 | 651 | 0.2211 |
| 10 | 780–3,860 | 2,923 | 658 | 0.2251 |

Invalidity is **not** concentrated in short spans. The rate rises with duration
from 4.4–13% in the shortest deciles to a 22–23% plateau above ~300 ms. Longer
units have more opportunity to overlap a neighbour.

### 3c — by POI error category

**Empty. Zero POI units are affected.** POIs are embedded-English units, and no
English unit is invalid (3d), so no invalid unit is a POI and the breakdown is
zero by construction rather than by coincidence.

### 3d — by language class

| Class | Units | Invalid | Rate |
|---|---:|---:|---:|
| EN | 4,576 | **0** | **0.0000** |
| ZH | 24,857 | 5,119 | 0.2059 |

The 17.4% invalid rate is **entirely Mandarin**. Not a single English unit is
invalid.

### 3e — cost to the conservative subset

Conservative subset (≥400 ms, non-final, matrix-preceded), both development
roles: 765 spans, 2,184 lexical units.

| Quantity | Value |
|---|---:|
| Units lost to invalid | **0** |
| Spans touching an invalid unit | **0** |
| Fraction of units lost | **0.0%** |

`whisper_dtw` loses **nothing** from the conservative subset. Its invalid units
are ZH same-language tokenizer overlaps, which the target-object layer merges,
and which never touch an embedded-English span.

---

## 5. Part 4 — v1 versus v2r3

Computed with identical code on both populations. v1 rows use the v1 candidate
table under the old conversation-level roles.

| Quantity | D-construct v1 | D-construct v2r3 | D-dev-select v1 | D-dev-select v2r3 |
|---|---:|---:|---:|---:|
| Sampled utterances | 300 | 300 | 300 | 300 |
| Conversations | 4 | **40** | 7 | **40** |
| **Dialogues** | **3** | **20** | **6** | **20** |
| Structural spans ≥400 ms | 313 | 356 | 446 | 409 |
| Lexical units in those spans | 516 | 1,032 | 1,508 | 1,152 |
| Baseline-error units ≥400 ms | *no v1 POI table exists* | 372 | 711 | 455 |

Corpus-level, both roles pooled:

| Quantity | v1 | v2r3 |
|---|---:|---:|
| Paired targets | 3,028 | 3,031 |
| Utterance-final share | 600 / 3,028 = 19.8% | 600 / 3,031 = 19.80% |
| `blank_excluded` median | 440 | 458.4 |
| `blank_to_preceding` median | 340 | 340.1 |
| `blank_to_following` median | 680 | 694.2 |
| `blank_midpoint` median | 480 | 467.3 |

**The headline of this table is the dialogue row.** The v1 300-utterance sample
drew from 4 and 7 conversations — 3 and 6 dialogues — because it sampled
utterances without dialogue stratification. The v2r3 sample is stratified at 15
utterances per dialogue, so all 20 dialogues per role are represented. Independent
clusters rose from 3 to 20 in D-construct and from 6 to 20 in D-dev-select.

The v1 D-dev-select baseline-error count (711) is larger than v2r3's (455)
because its 300 utterances came from only 6 dialogues and 1,508 units cleared the
floor, against 1,152 here. More units, far fewer clusters.

The four convention medians reproduce closely on a disjoint sample: three of four
within 4%, `blank_to_preceding` within 0.03%.

---

## 6. Part 5 — bootstrap adequacy

Independent clusters available for the Day 4 paired contrasts, D-dev-select at
the ≥400 ms floor:

| Subset | Units | **G (dialogues)** | Speakers |
|---|---:|---:|---:|
| All baseline-error (Site E) | 455 | **20** | 39 |
| Substitution-only (Site D) | 184 | **19** | 33 |
| `wrong_language_substitution` only | 89 | **11** | 14 |

`dialogue_id` is the independence unit, so G is the dialogue count, not the
speaker or unit count. Speakers are shown only to make the 2:1 relationship
visible; they are not independent of each other within a dialogue.

**On the choice of bootstrap.** At G = 11–20 the standard nonparametric cluster
bootstrap is known to under-cover. Cluster-robust variance estimators are biased
downward when the number of clusters is small, and resampling 11–20 clusters with
replacement produces too coarse an empirical distribution to approximate the
sampling distribution of the statistic. Normal critical values compound this by
ignoring that the variance is itself estimated from few clusters. The wild
cluster bootstrap-t with Rademacher weights, imposing the null, is the standard
remedy in that regime (Cameron, Gelbach & Miller 2008), and t(G−1) critical
values account for the estimated variance. On these numbers, that reasoning
applies to all three subsets and most sharply to the third. The runbook §12A
already commits to this design and to always reporting G.

One hard constraint follows from the numbers rather than from any preference: at
G = 11, Rademacher weights admit only 2^11 = 2,048 distinct sign vectors, so the
smallest attainable p-value is 1/2048 ≈ 4.9 × 10⁻⁴, and the resolution of any
interval built from them is bounded accordingly. At G = 19–20 the constraint is
not binding (2^19 ≈ 5.2 × 10⁵).

Nothing was implemented. The choice is yours.

---

## 7. Verification

| Check | Result |
|---|---|
| Run A code unmodified | `ctc_convention_diagnostic.py` and `conventions.py` untouched; run via `--config lss/l1b_candidates_dialogue_v2r3.yaml` |
| Development roles only | `roles_permitted: [D-construct, D-dev-select]`; withheld: `D-dev-confirm, loc-train, router-calib, util-train`; 88,433 rows read, D-test never opened |
| POI join integrity | `poi_index == reference_unit_index` verified surface-by-surface: **2,268 / 2,268 match, 0 unmatched, 100% EN** |
| Taint on every artifact | all 6 manifests carry `taint_reasons: ['development_only_diagnostic']`, `diagnostic_only: true` |
| No production write | nothing modified under `artifacts_lss`, `artifacts_v2`, `artifacts_dialogue_v2`, `v2r2`, or `v2r3` during this session; newest mtimes 2026-08-12 02:41, 2026-07-28 10:46, 08-14 17:00, 08-14 19:41, 08-15 04:15 (job 40486, before this session) |
| Exposure ledger | mtime `1786502490`, unchanged; no gate generation allocated, exposed, or consumed |
| Compile and focused tests | `compileall` clean; `test_lss_ctc_conventions.py` + `test_lss_dialogue_roles.py` pass (75 tests) |

The full CPU suite was **not** re-run: this session added one read-only
diagnostic module and modified no existing code path. `compileall` plus the two
test files covering the reused code were run instead.

---

## 8. Open items

1. **No test covers the new headroom module.** It is a read-only diagnostic whose
   outputs are reported here, but its eligibility logic is unverified by any
   regression test. If any of these numbers are to be relied on beyond this
   document, that gap should be closed first.
2. **The `existing_ctc` span source is inherited, not justified here.** Span
   durations, and therefore which spans clear a floor, depend on it. A different
   family would give different tier counts. No sensitivity analysis was run.
3. **`wrong_language_substitution` at ≥400 ms rests on 11 dialogues.** The
   sharpest test of the language-direction hypothesis has the fewest clusters.
4. **v2r3 artifacts already fail execution-time identity again.** They were
   published under source snapshot `64346ca6…`; adding the baseline driver and
   its tests moved the tree to `4fff0bcf…`, and this session's new module moves
   it again. Nothing here depended on identity verification, but the next run
   that does will refuse, exactly as job 40373 did. This is the third instance of
   the pattern recorded in runbook §13.
5. Carried forward: the partition fingerprint is not a literal field in the
   candidate generation manifest.

**No convention was selected. No eligibility floor was set. No gate was
evaluated. No readiness was declared. Production Gate A is unchanged.**
