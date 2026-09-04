# Day 3 — Site-E and Site-D direction construction on v2r3

**Date:** 2026-08-16
**Jobs:** Slurm 40918 (smoke, `COMPLETED`, 31 s), **40919** (full, `COMPLETED`, exit 0, **66 s**)
**Destination:** `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/directions/generation_001`

Directions were constructed at four encoder and four decoder layers, on
`D-construct` only, with every estimator parameter frozen before the run. **No
layer and no strength was selected.** No gate was evaluated.

---

## 1. Purpose and scientific boundary

This session builds the fixed directions Day 4's oracle screen will steer with,
and reports the conservative subset Day 4 will evaluate on. It fits nothing on
`D-dev-select`; that split is measured and reported only. `D-dev-confirm` and
`D-test` were never opened — a regression test asserts neither string appears in
the module.

Decided inputs, taken as given and not re-derived: provisional aligner path
`existing_ctc`, CTC convention `blank_to_preceding`.

### Why the third path is absent

The cross-attention path was implemented and evaluated
(`docs/V2R3_CROSS_ATTENTION_SPANS_2026-08-15.md`). In the preregistered
configuration it did not recover language spans: median disagreement ≈3 s against
both other paths, 93.2% of frames selecting the special-token class, 79.8% of
cross-attention mass on padding, and only 3 spans ≥400 ms in `D-dev-select`
against CTC's 409 across 20 dialogues. It is **excluded as a span source** and
stands as a reported negative result, not a failed step to retry. Its artifacts
exist as the negative-result record and were never read for span derivation here.

The three-way selection rule therefore collapses to a two-path comparison.
`blank_to_preceding` is the provisional path because it is characterised across
four conventions, has the smallest EN−ZH asymmetry (−71.7 ms), and both of its
error mechanisms are identified — blank ownership partly, utterance-final
entirely. Whisper DTW becomes the Day 6 independence check rather than a
co-primary; its 17.4% invalid units are entirely Mandarin and cost **0 units** in
the conservative subset.

---

## 2. Identity handling — and the first live use of the split branch

**Inputs read — LEGACY format, byte-verified, identity never attempted.**

| Input | Recorded sha256 | Byte check |
|---|---|---|
| candidate table | `cdc0ff4e…` | ok |
| `role_D-construct.parquet` | `d61c305e…` | ok |
| `role_D-dev-select.parquet` | `c0f3ad12…` | ok |
| `poi_D-construct.parquet` | `334feed9…` | ok |
| `poi_D-dev-select.parquet` | `5cebe272…` | ok |

**Artifacts published — SPLIT format**, all four carrying `code_config_sha256`
and `test_sha256`.

### The split branch, exercised on real artifacts

Day 2's artifacts were read back first — the second live exercise — and then the
narrowed binding was tested in the way that actually matters. After publication,
**13 new tests were added**, changing `test_sha256` from `21983432…` to
`4c8de0f4…`. Re-verifying then gives:

| Artifact | Format | `identity_ok` | Detail |
|---|---|---|---|
| Day 3 `directions_report.json` | split | **True** | — |
| Day 3 `directions.npz` | split | **True** | — |
| Day 2 `cross_attention_report.json` | split | False | `differs in ['code_config_sha256']` |
| Day 2 `cross_attention_spans.parquet` | split | False | `differs in ['code_config_sha256']` |
| v2r3 `candidates_all.parquet` | legacy | False | `differs in ['source_sha256']` |

Three things are established, all against real artifacts on disk rather than
fixtures:

1. **A test-only edit no longer invalidates a split-format artifact.** Day 3's
   artifacts verify `ok` with a `test_sha256` that has changed since publication.
   This is the property the narrowing exists for, and it now has live evidence.
2. **A code edit still does.** Day 2 fails — and fails on
   `code_config_sha256` **only**, not on `test_sha256`, even though both changed.
   Adding `v2r3_directions.py` to `src/` moved `code_config_sha256` from
   `407952ce…` to `793c0d03…`, and the gate caught it. The split branch gates
   code and configuration exactly as intended and ignores tests exactly as
   intended.
3. **Legacy semantics are untouched.** The candidate table still gates
   `source_sha256`, grandfathered behaviour preserved.

Open item 3 of `docs/IDENTITY_BINDING_NARROWED_2026-08-15.md` — "the split branch
has been exercised only by the regression suite, never against a real artifact"
— is now closed.

---

## 3. Frozen estimator configuration

| Parameter | Value |
|---|---|
| Span family / convention | `existing_ctc` / `blank_to_preceding` |
| Floor | 400 ms |
| Pooling | Hann taper over the aligned interval, normalised to sum 1 (primary); `central60` and `uniform` retained for Day 6 |
| Control | matrix-language frames immediately preceding the span, matched one-for-one in frame count |
| Nuisance vector | span duration, frame count, relative start, utterance duration, unit count — standardised |
| Ridge | multi-output, α = 1.0, **fitted on D-construct only**, intercept retained |
| Individual contrasts normalised | **No** |
| Norm clipping | 99th percentile, direction preserved |
| Aggregation | dialogue-balanced weighted mean, weight 1/(G·n_g) |
| Prompt subspace | `Orth[Mean(p_j)]`, r = 1, projected out of **each** contrast before aggregation |
| Bootstrap | 2000 draws, seed 20260815, resampling **dialogues** |
| Encoder / decoder layers | [15, 23, 27, 31] / [8, 16, 24, 31] (from `configs/lss/spec.yaml`) |

"Intercept retained" is implemented as `residual = y − (x − x̄)β`: the sample mean
survives on every axis and only the nuisance-predicted *variation* is removed. A
regression test pins exactly that.

---

## 4. Subsets, and a deviation from Session 8

### The deviation, decomposed

Session 8's figures were computed on **unadjusted** spans. Day 3 applies the
decided convention first, which moves boundaries and changes which spans clear
400 ms. Running both ways through identical code:

| Variant | Role | Spans | Error units | Correct units | Dialogues |
|---|---|---:|---:|---:|---:|
| No convention (Session 8) | D-dev-select | **409** | **455** | 697 | **20** |
| No convention (Session 8) | D-construct | 356 | 372 | **660** | **20** |
| `blank_to_preceding` (Day 3) | D-dev-select | 482 | 489 | 740 | 20 |
| `blank_to_preceding` (Day 3) | D-construct | 419 | 400 | 699 | 20 |

**The no-convention row reproduces Session 8 exactly** — 409 spans, 455 error
units, 20 dialogues, and 660 correct units. The deviation is entirely and only
the convention, which is a decided input, not a discrepancy.

### The conservative subset (D-dev-select, ≥400 ms, `blank_to_preceding`)

482 spans, 211 utterances, 489 baseline-error units, **20 dialogues**. The three
strata Day 4 reports on separately:

| Stratum | Spans | Units | **Dialogues** |
|---|---:|---:|---:|
| all errors | 209 | 489 | **20** |
| substitution-only (excludes deletion) | 116 | 256 | **19** |
| `wrong_language_substitution` only | 31 | 108 | **12** |

### The construction set (D-construct, baseline-correct, ≥400 ms)

**233 spans, 436 correct units, 20 dialogues, 125 utterances.**

This is smaller than the ~660 units expected, for a reason worth stating
plainly. Pooling is defined over *the aligned interval* of a span, so a span is
the pooling unit. Of the 419 eligible D-construct spans holding 699 correct
units, only **233 have every unit baseline-correct**. Spans with at least one
correct unit number 300 and hold all 699 units, but pooling those would mix
baseline-correct and baseline-error material inside a single positive vector and
contaminate the premise the construction rests on.

I used the strict definition — every unit in the span baseline-correct. It is the
conservative reading and cannot contaminate. If you prefer the 699-unit
definition, that is a re-run with one filter changed, and the decomposition above
is the number you would be choosing between.

---

## 5. Results

Zero pairs were skipped at either site: `no_control` 0, `too_short` 0,
`no_token` 0. All 233 spans across all 20 dialogues contributed at every layer.

### Site E — encoder

| Layer / pooling | Pairs | G | cos(Δ_raw, Δ_resid) | Ridge energy removed | Bootstrap cos p05 | mean | ‖Δ‖ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 / hann | 233 | 20 | 0.9962 | 0.0504 | 0.8697 | 0.9290 | 0.815 |
| 15 / central60 | 233 | 20 | 0.9959 | 0.0497 | 0.8682 | 0.9280 | 0.822 |
| 15 / uniform | 233 | 20 | 0.9968 | 0.0514 | 0.8806 | 0.9389 | 0.746 |
| 23 / hann | 233 | 20 | 0.9959 | 0.0334 | **0.9406** | 0.9827 | 3.424 |
| 23 / central60 | 233 | 20 | 0.9951 | 0.0324 | 0.9290 | 0.9790 | 3.186 |
| 23 / uniform | 233 | 20 | 0.9968 | 0.0239 | 0.9270 | 0.9801 | 3.270 |
| 27 / hann | 233 | 20 | 0.9947 | 0.0334 | 0.9298 | 0.9783 | 3.592 |
| 27 / central60 | 233 | 20 | 0.9938 | 0.0325 | 0.9192 | 0.9746 | 3.373 |
| 27 / uniform | 233 | 20 | 0.9957 | 0.0240 | 0.9130 | 0.9751 | 3.404 |
| 31 / hann | 233 | 20 | 0.9950 | 0.0362 | **0.9441** | 0.9763 | 5.544 |
| 31 / central60 | 233 | 20 | 0.9950 | 0.0361 | 0.9435 | 0.9751 | 5.432 |
| 31 / uniform | 233 | 20 | 0.9951 | 0.0277 | 0.9321 | 0.9728 | 4.996 |

### Site D — decoder

| Layer | Pairs | G | cos(Δ_raw, Δ_resid) | Ridge removed | U_prompt energy removed | cos(v_nat, U_prompt) | Bootstrap cos p05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 233 | 20 | 0.9971 | 0.0199 | 0.0003 | 0.0102 | 0.8630 |
| 16 | 233 | 20 | 0.9979 | 0.0282 | 0.0005 | 0.0300 | 0.9169 |
| 24 | 233 | 20 | 0.9986 | 0.0423 | 0.0004 | 0.0222 | **0.9630** |
| 31 | 233 | 20 | 0.9973 | 0.0315 | 0.0010 | 0.0277 | 0.9258 |

### Reading these numbers

- **Residualisation barely rotates the direction** (cos 0.994–0.999) while
  removing 2–5% of contrast energy. The nuisance vector explains a small,
  non-zero share, and what it explains is close to orthogonal to the mean effect.
- **The prompt subspace is nearly irrelevant at Site D.** `cos(v_nat, U_prompt)`
  is 0.010–0.030 and projection removes 0.03–0.10% of energy. The natural
  contrast is almost orthogonal to the prompt mean, so the projection is cheap
  insurance rather than a large correction.
- **Dialogue-bootstrap stability is high but not uniform**: p05 ranges 0.863
  (Site D layer 8) to 0.964 (Site D layer 24). Deeper layers are more stable at
  both sites. G = 20 everywhere.
- **`corr(‖d_i‖, duration)` = −0.382 at Site E**, −0.032 at Site D. The Site-E
  magnitude is moderately anti-correlated with span duration — longer spans give
  smaller pooled contrasts, as normalised pooling over more frames would predict.
  This is reported, not corrected; duration is already in the nuisance vector.
- Norm clipping bound 3 of 233 pairs at each site.

### `corr(‖d_i‖, confidence)` — not computed, and why

There is **no frozen alignment-confidence measure in this protocol**. Runbook §2
records that no tolerance qualified, so `selected_tolerance_ms` is null and no
valid confidence threshold exists; the eligibility rule deliberately has no
confidence condition. Substituting a decoding-confidence proxy such as B0_AUTO
`avg_logprob` would introduce a measure that is not part of the frozen
configuration, so I did not. Adding it is a one-field change to the pair
metadata plus a re-run, if you want it.

### LDA ablation — not run

`--lda-ablation` was not passed, so `cos(Δ_LDA, Δ_mean)` is not reported. The
numerical requirement is implemented and unit-tested regardless: `shrinkage_lda`
forms, shrinks, and inverts the covariance **inside an orthonormal basis of the
complement of U_prompt** and maps back, and a regression test confirms a removed
axis receives no weight after mapping back even when it carries by far the
largest raw separation.

---

## 6. Artifacts

| sha256 | Bytes | Artifact | Format |
|---|---:|---|---|
| `813e84a54ec2d8f8d2721e2b2b83b1f1…` | 168,204 | `directions.npz` | split |
| `d5e015ecd10ad60d20c8e1682c5bb8a5…` | 19,581 | `directions_report.json` | split |
| `e8295ebc05e2ed8e8c0613b51ff2a1e7…` | 14,245 | `site_e_pairs.parquet` | split |
| `64b341e06c2df69ed72c5d450658c2c6…` | 13,237 | `site_d_pairs.parquet` | split |

`directions.npz` holds 16 arrays — 12 Site-E (4 layers × 3 pooling variants) and
4 Site-D — keyed `site_e_layer{L}_{variant}` and `site_d_layer{L}`. Parent:
`candidates_all.parquet`. `taint_reasons: []`.

**Also present:** `directions/smoke_001`, an 8-utterance smoke generation from
job 40918 used to validate the GPU path before committing the full run. It is
not the Day 3 artifact and nothing should consume it.

### Code freeze

| Checkpoint | `code_config_sha256` |
|---|---|
| At publication (printed by the launcher inside job 40919) | `793c0d03c6ce30c2499cc1987df239afede43f6470ec727c8e13211648b93196` |
| Now, after adding 13 tests | identical |

The freeze covered `src/` and `configs/` only. `tests/` was deliberately edited
after publication — that is the narrowing working as designed, and §2 documents
the live proof.

### Immutability

| Root | Newest mtime |
|---|---|
| `artifacts_lss` | 2026-08-12 02:41:33 |
| `artifacts_v2` | 2026-07-28 10:46:20 |
| `artifacts_dialogue_v2` | 2026-08-14 17:00:10 |
| `artifacts_dialogue_v2r2` | 2026-08-14 19:41:13 |

Exposure ledger mtime `1786502490`, sha256 `46849f5caed62201…` — unchanged. No
gate generation allocated, exposed, or consumed.

---

## 7. Open items

1. **The construction set is 233 spans / 436 units, not ~660 units**, because
   pooling is span-level and only 233 spans are wholly baseline-correct (§4). The
   alternative definition and its counts are given; the choice is yours.
2. **`corr(‖d_i‖, confidence)` is not reported** — no frozen alignment-confidence
   measure exists in this protocol (§5).
3. **The LDA ablation was not run.** The complement-basis requirement is
   implemented and tested.
4. **`wrong_language_substitution` rests on 12 dialogues** in the conservative
   subset at ≥400 ms — the smallest cluster count Day 4 will face.
5. Site-E contrast magnitude is anti-correlated with span duration (−0.382);
   duration is in the nuisance vector, but the residual correlation was not
   re-measured after residualisation.

**No layer or strength was selected. No gate was evaluated. Production Gate A is
unchanged.**
