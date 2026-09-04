# Selective test-time steering for Mandarin–English CS-ASR — complete experimental record

**Compiled:** 2026-08-17
**Programme:** v2r3 namespace, 14–17 August 2026, nine GPU stages
**Population:** CS-Dialogue, `D-construct` + `D-dev-select` only
**Status:** Gate B does not pass on either arm. No gate was ever declared, nothing
was selected, `D-dev-confirm` and `D-test` were never used as intervention
populations, and production Gate A is unchanged throughout.

---

## 0. How to read this document

This is the consolidated record of the steering experiment from the v2r3
candidate generation (14 Aug) to the hypothesis-substitution causal probe
(17 Aug). It exists because the per-day documents are individually complete but
collectively hard to reason across, and because one stage — Day 6, the invariance
battery — never received a dated document at all.

Every quantity carries a provenance tag:

| Tag | Meaning |
|---|---|
| **[A]** | read directly from an authenticated artifact during this compilation |
| **[C]** | computed during this compilation from artifact rows; the estimator is stated in place |
| **[D]** | taken from the stage's dated document and **not** re-derived here |

Where **[C]** disagrees with **[D]**, both are shown and the disagreement is
listed in §10. Nothing was reconciled by choosing the more convenient number.

**Authority.** This document does not supersede the dated stage documents; it
indexes and cross-checks them. On conflict, the artifact wins, then the dated
document, then this file. It authorises nothing — no run, no selection, no gate.

---

## 1. Executive summary

### 1.1 What is established

1. **A language direction at decoder layer 16 has causal leverage on the model's
   prediction.** Teacher-forced, oracle-localized, substitution-only:
   Δ gold-token logprob **+0.1057 [+0.0466, +0.1649], G=19**, beating all four
   controls — label-permuted null, matched-norm random, opposite sign, wrong
   location. [D]
2. **The effect is not the language-token (prompt) mechanism.**
   `cos(v_nat, U_prompt)` = 0.0102–0.0300 across the four decoder layers, and
   projecting out the prompt subspace removes 0.03–0.10% of contrast energy. [A]
3. **The encoder site carries no leverage.** Site E beats no control at any layer
   or strength; at layer 31, ρ=2 it is worse than its own null (−0.453). The null
   survives inverse-variance reweighting that demonstrably removed the
   duration–magnitude coupling (`corr(‖d‖, duration)` at layer 15:
   **−0.3857 → −0.0192**) and survives all three pooling schemes. [A][D]
4. **The dose-response rises, peaks, and declines.** Layer 16: +0.024 / +0.054 /
   +0.106 / +0.178 / +0.170 / +0.079 at ρ = 0.25 / 0.5 / 1 / 2 / 3 / 4, with the
   turn between ρ=2 and ρ=3 and the interval crossing zero at ρ=4. [D]
5. **The conclusion is invariant to alignment choices** in sign, across two
   aligner paths, four CTC blank conventions, boundary jitter, and three pooling
   schemes — with two important qualifications in §5.7 and §10.
6. **A third aligner path was built and failed.** Cross-attention pseudo-labels
   in the preregistered configuration assign **93.20%** of valid frames to the
   special-token class and disagree with both other paths by ~3 s median. Reported
   as a negative result, not tuned. [D]

### 1.2 What is refuted

7. **Practical, inference-time mitigation does not work.** Under
   inference-available (transport) localization the correction rate is **0/489 at
   400 ms, 0/502 at 300 ms, 0/515 at 200 ms**. [A]
8. **Even the oracle upper bound is net harmful.** With the gold decoder step
   supplied, 7 corrections against **13 corruptions** at 400 ms (7 : 19 at
   200 ms). [A]
9. **The leading explanation for (7) is not supported.** Day 7 diagnosed that
   81.1% of wrong-language substitutions and 37.7% of deletions have a decoder
   step but no English in the first-pass hypothesis. Day 8 removed that state
   entirely — in one arm by substituting the gold transcript, in another by
   minimally inserting the reference English unit — and the correction count
   stayed at **0/489 in all three arms**. The 81.1% is a **correlate**, not a
   demonstrated cause. [A]

### 1.3 What this compilation adds

10. **The one positive transcript result rests on two dialogues.** All 7 oracle
    corrections come from **2 utterances in 2 dialogues** (`CSD0014`, `CSD0502`)
    out of 116 utterances and 20 dialogues. Eighteen of twenty clusters
    contribute exactly zero. The reported interval `+0.0143 [+0.0033, +0.0254],
    G=20` is therefore driven by whether two clusters are resampled. **[C]** —
    §7. This is a stronger caveat than "7 events is descriptive" and it is not
    stated in any existing document.
11. **The Day 8 null is underpowered against an oracle-size effect.** Across all
    three arms, 44 interventions landed within ±1 step of the gold step. At the
    oracle arm's per-opportunity rate of 7/405 = 1.73%, that predicts **0.76**
    corrections and P(0) ≈ **0.47**. **[C]** — §8.
12. **Day 6 is reconstructed from its artifacts**, since no dated document exists
    for it. The structural claims verify exactly; the cell effect sizes in
    circulation do **not** reproduce under any estimator implemented in this
    repository — §5.7 and §10.1.
13. **The non-oracle arm is not literally inert.** It changes one utterance's raw
    text (`ZH-CN_U0027_S0_163`); the change is punctuation only and normalizes
    away. **[C]** — §6.4.

---

## 2. The system under test

### 2.1 Model and action

| Component | Value | Tag |
|---|---|---|
| Model | `openai/whisper-large-v3`, bfloat16, frozen | [D] |
| Model revision | `sha256:a8e94b859…` | [D] |
| Hardware | `worker-0`, NVIDIA H100 80GB HBM3, torch 2.10.0+cu128 | [D] |
| Repository SHA for every GPU stage | `fdd441a75b0d66f56d8e5efec62f01649f7b9ae9` | [A] |
| **Frozen action** | **Site D, decoder layer 16, ρ = 1** | [A] |
| Direction norm ‖Δ^D‖ | **2.414639416389413** | [A] |
| Action energy per intervened step | ρ²‖Δ‖² = **5.8305** | [A] |
| Decoding | `transcribe`, `language=None`, temperature 0, greedy (`num_beams=1`), `max_new_tokens=200`, batch 16 | [D] |

The action was decided by a human before Day 5 and recorded, not re-derived, in
every subsequent stage's manifest. Its recorded rationale: *"layer 16 beat the
label-permuted null at every rho in every stratum; its dose-response is clean and
monotone; layer 31 turns down at rho=2 and layer 24 collapses. rho=1 is one
centroid displacement — the interpretable unit — and sits mid-grid rather than at
an edge."* [A]

### 2.2 The two sites

- **Site D (decoder):** the post-cross-attention residual at decoder layer *L*,
  at a single decoder step. The intervention adds ρ·Δ^D at that step.
- **Site E (encoder):** encoder hidden states at layer *L*, pooled over the
  aligned span interval. Three pooling schemes: Hann taper (primary), central-60%,
  uniform.

Direction construction (Day 3, frozen before the run): difference-in-means
between embedded-English span frames and matched matrix-language control frames
immediately preceding the span; nuisance-residualised by multi-output ridge
(α=1.0, fitted on `D-construct` only, intercept retained); prompt subspace
`Orth[Mean(p_j)]`, r=1, projected out of each contrast before aggregation;
99th-percentile norm clipping; dialogue-balanced aggregation with weight
1/(G·n_g). [D]

### 2.3 The two localization regimes

| Regime | How the decoder step is chosen | Available at inference? |
|---|---|---|
| **Oracle** | the gold decoder step is supplied | **No** |
| **Non-oracle (transport)** | `r_q = Σ_t Ā[q,t]·g_t` over final-decoder-layer all-head-mean cross-attention; the step maximising `r_q` is chosen | Yes |

Transport abstains when there is no candidate English region, or when the best
step's attention on the region does not exceed uniform chance
(`enrichment = r_max / (|region| / |valid frames|) ≤ 1`, ties abstain). The rule
is parameter-free. `r_q` is deliberately **not** renormalised: each attention row
sums to 1 over frames, so `r_q ∈ [0,1]` is a meaningful fraction. [A]

`E_eff = ρ²‖Δ^D‖²·Σ_q r_q²`. A "transport hit" is the selected step landing
within **±1 step** of the gold step. [A]

### 2.4 Outcome spaces

Three, and the distinction carries the whole story:

1. **Token-level (teacher-forced):** Δ gold-token logprob, margin, gold rank,
   entropy. Days 3–6.
2. **Transcript-level, oracle-localized:** corrections, corruptions, spillover,
   PIER/MER/WER under free decoding with the gold step supplied. Day 5, Day 7.
3. **Transcript-level, inference-available:** the same, with transport
   localization. Day 7, Day 8.

The effect is solid in (1), marginal and net negative in (2), and absent in (3).

---

## 3. Data, roles, and the independence unit

### 3.1 Corpus and partition

CS-Dialogue: **100 two-party dialogues represented as 200 speaker sides.** The
independent cluster is **`dialogue_id`** — `conversation_id` and `speaker_id`
identify one side of a dialogue and must never be counted as independent. Roles
are dialogue-atomic. [D]

The v2r3 sample is dialogue-stratified at **K=15 utterances per dialogue**,
sampler seed 303, stratify field `dialogue_id`. [A]

| Role | Dialogues | Utterances | Used for | Read by this programme? |
|---|---:|---:|---|---|
| `D-construct` | 20 | 300 | direction construction | yes |
| `D-dev-select` | 20 | 300 | development evaluation | yes |
| `loc-train` | 15 | 225 | localizer training (W3) | candidates only |
| `util-train` | 12 | 180 | utility labels / selector | candidates only |
| `router-calib` | 8 | 120 | abstention calibration | candidates only |
| `D-dev-confirm` | 10 | 150 | one final dev confirmation | **never opened** |
| `D-test` | — | 6,257 | locked test | reference counts + C00 only |
| **Total swept** | **85** | **1,275** | | |

Realized counts match the read-only expectation exactly: min 15 and max 15
utterances per dialogue, **total shortfall 0** across all 85 dialogues. [D]

The partition was verified bit-identical to the superseded v2r2 allocation before
publication: assignment hash `313ae124…`, per-role dialogue sets identical
dialogue-by-dialogue over 7 roles and 100 dialogues, four disjointness assertions
true, gated TV 0.1279 ≤ 0.15, max |SMD| 0.2240 ≤ 0.25, zero balance failures,
`D-test` exactly the official test split. [D]

### 3.2 Why the cluster count is what it is

This is a corpus property, not a fixable defect. Per-stratum cluster counts at the
400 ms tier, conservative subset:

| Stratum | Units | **G (dialogues)** | Speakers |
|---|---:|---:|---:|
| all baseline errors | 489 | **20** | 39 |
| substitution-only (excludes deletion) | 199 | **19** | 33 |
| `wrong_language_substitution` only | 90 | **12** | 14 |

At G=12 the Rademacher wild bootstrap admits 2¹² = 4,096 sign vectors, so the
smallest attainable p-value is ≈2.4×10⁻⁴ — adequate to detect an effect, close to
uninformative as a null. The `wrong_language_substitution` stratum is therefore
reported as **descriptive** throughout. [D]

Every interval in this programme uses a **wild cluster bootstrap with Rademacher
weights, clustered on `dialogue_id`, with t(G−1) critical values**, and G is
stated with every estimate. Draws: 2,000 for Days 4–6, 10,000 for Days 7–8. [A]

### 3.3 The v1 → v2r3 population change

The v1 sample drew 300 utterances per role **without** dialogue stratification,
so `D-construct` came from **3 dialogues** and `D-dev-select` from **6**. Under
v2r3 both have **20**. Independent clusters rose from 3→20 and 6→20 while unit
counts fell (v1 `D-dev-select` had 1,508 units above 400 ms against 1,152 here):
more units, far fewer clusters. Every v1-sample analysis was recomputed rather
than carried forward. [D]

Evidence the re-partition is genuine: of the 14 dialogues in v1's `dev_select`,
only **4** remain in v2r3's `D-dev-select`; 5 moved to `D-construct`, 2 to
`loc-train`, 2 to `util-train`, 1 to `router-calib`. Utterance overlap is
54/300 = 18.0%. [D]

---

## 4. Execution ledger

Every Slurm job in the programme, from `sacct`. [A]

| Job | Name | State | Elapsed | Stage |
|---|---|---|---|---|
| 40373 | `v2r2_candidates` | **FAILED** (2 s) | 00:00:05 | refused on identity mismatch — see §9.1 |
| 40374 | `v2r3_candidates` | COMPLETED | 00:07:10 | Stage 0 — candidate generation |
| 40486 | `v2r3_baseline_poi` | COMPLETED | 00:01:59 | Stage 1 — baseline + POI |
| 40703 | `v2r3_cross_attention` | **FAILED** | 00:01:01 | tuple-unpack defect, nothing written |
| 40711 | `v2r3_cross_attention` | COMPLETED | 00:02:23 | Day 2 — cross-attention (negative) |
| 40918 | `v2r3_directions` | COMPLETED | 00:00:31 | Day 3 smoke |
| 40919 | `v2r3_directions` | COMPLETED | 00:01:06 | **Day 3 — directions** |
| 40946 | `v2r3_oracle` | **FAILED** | 00:02:41 | parquet mixed-type column |
| 40952 | `v2r3_oracle` | COMPLETED | 00:02:41 | Day 4 smoke |
| 40953 | `v2r3_oracle` | COMPLETED | 00:24:37 | **Day 4 — oracle screen** |
| 41005 | `v2r3_day5` | COMPLETED | 00:02:25 | Day 5 smoke |
| 41007 | `v2r3_day5` | COMPLETED | 00:07:29 | **Day 5 — dose + first free decoding** |
| 41014/41015/41019 | `v2r3_invariance` | COMPLETED | 2–5 min | Day 6 smokes |
| 41020 | `v2r3_invariance` | COMPLETED | 00:11:28 | **Day 6 — invariance battery** |
| 41142 | `v2r3_invariance` | COMPLETED | 00:05:49 | Day 6 ZH-control repair |
| 41143/41146 | `v2r3_gate_b` | **FAILED** | ~2 min | pre-run defects, §9 |
| 41147/41148/41149/41155 | `v2r3_gate_b` | COMPLETED | 2–3 min | Day 7 smokes |
| 41162 | `v2r3_gate_b` | COMPLETED | 00:29:44 | **Day 7 — Gate B** |
| 41736 | `v2r3_hyp_sub` | COMPLETED | 00:05:11 | **Day 8 — hypothesis substitution** |

Total GPU wall clock for the nine published stages: **≈1 h 32 m**.

### 4.1 Artifact roots

All under `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/`. [A]

```text
candidate_generations/generation_001   candidate tables, 3 aligner families
baselines/generation_001               B0_AUTO hypotheses + POI tables
cross_attention/generation_001         the negative-result spans
directions/generation_001              directions.npz (16 arrays) + pairs
  directions/smoke_001                 8-utterance smoke; never consumed
oracle_screen/generation_001           449 conditions × 489 targets
day5/generation_001                    dose extension + oracle free decoding
invariance/generation_001              Day 6, 66,924 rows
invariance/generation_002_zh_repair    ZH-control repair, 14,670 rows
gate_b/generation_001                  oracle vs non-oracle, 3 tiers
  gate_b/smoke_001                     refused by name as an input
hypothesis_substitution/generation_001 Day 8, three arms
  hypothesis_substitution/smoke_001    smoke_used_as_input = false
freeze/spec_freeze_v2.json             sealed spec e6c38a3e…
manifests/roles/                       7 role manifests, fingerprint b6c1af90…
```

Every artifact carries a published manifest sidecar. Smoke generations are
refused **by name** as inputs by every downstream driver. [D]

### 4.2 Immutability across the programme

| Root | Newest mtime, unchanged throughout | Tag |
|---|---|---|
| `artifacts_lss` (v1 production) | 2026-08-12 02:41:33 | [D] |
| `artifacts_v2` | 2026-07-28 10:46:20 | [D] |
| `artifacts_dialogue_v2` | 2026-08-14 17:00:10 | [D] |
| `artifacts_dialogue_v2r2` | 2026-08-14 19:41:13 | [D] |
| Exposure ledger | mtime `1786502490`, sha256 `46849f5c…` | [D] |

**Gate generation 3 was never allocated, exposed, or consumed.** No production
status or freeze artifact was written at any point. L1c remains locked. [D]

---

## 5. Stage-by-stage record

### 5.0 Stage 0 — candidate generation (job 40374, 14 Aug)

**Purpose.** Produce the natural candidate table for the dialogue-atomic roles.

**Result.** 178,884 rows, 38 columns, three aligner families, each covering all
1,275 utterances: `existing_ctc` 59,651 rows, `qwen_forced_aligner` 59,651,
`whisper_dtw` 59,582. [D]

Raw-unit validity over development roles only (600 utterances):

| Family | Independence class | Units | Coverage | Invalid | Invalid-duration | Nonmonotonic |
|---|---|---:|---:|---:|---:|---:|
| `existing_ctc` | `ctc_forced_alignment` | 29,500 | 600/600 | 0.0000 | 0.0000 | 0.0000 |
| `qwen_forced_aligner` | `qwen_forced_alignment` | 29,500 | 600/600 | 0.0329 | 0.0328 | 0.0000 |
| `whisper_dtw` | `whisper_cross_attention_dtw` | 29,433 | 600/600 | **0.1739** | 0.0000 | 0.0000 |

`invalid_duration` and `nonmonotonic` keep separate denominators. These are raw
validity statistics, **not** alignment accuracy; nothing here compares an aligner
to truth. [D]

**Why the namespace is v2r3 and not v2r2.** Job 40373 refused in 2 seconds with
`identity_mismatch differs in ['source_sha256']` and wrote nothing. `manifest.identity()`
bound artifacts to a snapshot covering `src/`, `configs/` **and `tests/`**; six
regression tests added after the v2r2 publish moved the tree hash, so every v2r2
artifact failed the execution-time check. **This was a correct refusal** — the
artifacts were not produced by the tree present at execution time, and the guard
said so before spending GPU time. v2r3 republishes the same allocation under the
current tree; nothing about the science changed. [D]

**Publication mechanism differs by stage and consumers must not assume otherwise:**
role generation uses an attempt-private directory with atomic rename; candidate
generation writes directly into the destination and publishes the completion
manifest last. **A candidate generation lacking `generation_complete.manifest.json`
must be treated as absent, never as a resumable partial cache.** [D]

### 5.1 Stage 1 — baseline and POI labelling (job 40486, 15 Aug)

**Purpose.** What the frozen baseline transcribes, and whether each embedded-English
lexical unit is reproduced correctly — for `D-construct` (which supplies
baseline-**correct** units to construction) and `D-dev-select` (which supplies
baseline-**error** units to evaluation).

A new driver was required: `p0_baseline.py` iterates `("dev_select", "dev_confirm")`,
would have generated held-out `D-dev-confirm`, writes into the v1 root over the
immutable `poi_dev_select.parquet`, selects a primary baseline, and evaluates the
P0 gate. The replacement driver does none of those and **reuses the scoring
functions verbatim** — a regression test asserts `driver.poi_table is pier.poi_table`
and that the module defines no local `poi_table`, `evaluate_pois`, or `_categorize`.
The driver refuses any role outside `{D-construct, D-dev-select}` **by name**. [D]

| Role | Lexical units | Correct | Baseline-error | Utterances | Dialogues |
|---|---:|---:|---:|---:|---:|
| D-construct | 2,310 | 1,408 | 902 | 300 | 20 |
| D-dev-select | 2,268 | 1,384 | 884 | 300 | 20 |

Error categories:

| Category | D-construct | D-dev-select |
|---|---:|---:|
| `deletion` | 456 | 499 |
| `wrong_language_substitution` | 224 | 175 |
| `same_language_substitution` | 125 | 138 |
| `boundary_error` | 50 | 34 |
| `phonetic_transliteration_or_script` | 42 | 30 |
| `other` | 5 | 8 |

Zero null categories, zero null `correct` flags, correct + error = rows exactly.
Decoding settings for the two roles are identical **including the model revision
hash**, so the construction/evaluation contrast is sound. [D]

### 5.2 Stage 2 — headroom and Run A on v2r3 (diagnostic, 15 Aug)

Read-only, `development_only_diagnostic` taint on every artifact, written outside
`$ARTIFACTS`. Selected nothing.

**Headroom, `D-dev-select`** (spans / lexical units / utterances / **dialogues**):

| Floor | Spans | Units | Utts | **Dialogues** | Baseline-error | Baseline-correct |
|---|---:|---:|---:|---:|---:|---:|
| ≥400 ms | 409 | 1,152 | 197 | **20** | 455 | 697 |
| ≥300 ms | 487 | 1,234 | 211 | **20** | 489 | 745 |
| ≥200 ms | 526 | 1,275 | 221 | **20** | 506 | 769 |

**All 20 dialogues survive at every floor — the floor costs utterances and units,
never clusters.** [D]

Deletion is 271 of 455 errors (60%) at ≥400 ms and Site D cannot reach any of them
under free decoding by construction, so the substitution-only count (184 units,
19 dialogues before the convention is applied) is the governing size. [D]

**Run A reproduced on the new population** (paired count 3,031; v1 was 3,028):

| Convention | Median | P90 | ≤100 ms | ≤200 ms | EN med | ZH med | EN−ZH |
|---|---:|---:|---:|---:|---:|---:|---:|
| `blank_excluded` | 458.4 | 1378.5 | 8.64% | 26.86% | 253.4 | 680.7 | −427.3 |
| `blank_to_preceding` | **340.1** | 1280.7 | **13.26%** | **34.48%** | 318.7 | 390.4 | **−71.7** |
| `blank_to_following` | 694.2 | 1560.4 | 1.75% | 9.11% | 606.9 | 721.0 | −114.2 |
| `blank_midpoint` | 467.3 | **1118.4** | 2.87% | 13.96% | 408.6 | 538.5 | −129.9 |

Five named v1 findings were checked by name: F1 (to-preceding best) **holds**;
F2 (EN−ZH asymmetry reduction) **changed — larger, ~6× not ~3×**; F3
(utterance-final unaffected) **changed — three of four conventions, not all
four**; F4 (median blank run ~220 ms) **holds** at 240.1 ms; F5 (working median
~280 ms after exclusion) **holds** at 261.6 ms. The utterance-final share
reproduces almost exactly on a disjoint sample: 600/3,031 = 19.80% against
600/3,028 = 19.8%. [D]

**`whisper_dtw`'s 17.4% invalid units are entirely Mandarin.** EN units:
4,576 with **0** invalid. ZH: 24,857 with 5,119 invalid, all
`adjacent_overlap` — a between-neighbour condition, which is why
`invalid_duration` and `nonmonotonic` are both zero. Cost to the conservative
subset (765 spans, 2,184 units): **0 units, 0 spans, 0.0%**. [D]

### 5.3 Day 2 — cross-attention pseudo-labels (job 40711, 15 Aug) — **negative result**

Following Liu et al., IEEE TASLP 2025 (arXiv 2403.05887). Frozen configuration set
in source before the first run: final decoder layer (−1 of 32), all 20 heads
averaged, 9-frame (180 ms) mode filter, 5-frame (100 ms) minimum span, special
tokens as their own protected class, per-frame argmax over head-averaged
cross-attention, no DTW and no monotonic path. [D]

**Outcome: the configuration does not recover language spans.**

| Class | Frames | Fraction |
|---|---:|---:|
| `special` | 436,155 | **93.20%** |
| `ZH` | 14,221 | 3.04% |
| `EN` | 12,741 | 2.72% |
| `neutral` | 4,874 | 1.04% |

**This is not a token-mapping bug** — audited on CPU over 50 utterances, exactly
5 special tokens per utterance (4 prefix + EOT) against a median 23 tokens, i.e.
21.24% of *tokens* are special yet they win the argmax on 93% of *frames*. It is
the attention-sink phenomenon, visible on both axes: Whisper applies **no encoder
attention mask**, so a mean of **79.8%** of cross-attention mass falls on padding
beyond the audio (max 94.2%). [D]

Spans: 442 survived, median duration **140 ms**. At ≥400 ms in `D-dev-select`:
**3 spans across 2 dialogues**, against CTC's **409 across 20**. Disagreement
against the other paths is **2,768–3,207 ms median**, with ≤100 ms agreement
never exceeding 0.67%. [D]

The escalation trigger (">300 ms median disagreement against both paths") fired at
roughly 10× its threshold. The path is **excluded, not repaired in place**. Four
candidate corrections — text-token renormalisation, excluding padded frames before
normalisation, a different layer or independently-chosen head subset, and
attention-sink correction — are each a different method requiring its own
preregistration. Tuning the frozen parameters after seeing the result is exactly
the move the protocol forbids, and was not made. [D]

**Consequence:** the invariance battery runs over **two** aligner paths, not three.
These remain two genuinely independent families — different acoustic model,
different alignment mechanism — so the two-family requirement is not weakened;
what narrows is the breadth of path-axis evidence. [D]

### 5.4 Day 3 — direction construction (job 40919, 16 Aug)

`D-construct` only. Every estimator parameter frozen before the run. No layer and
no strength selected. Encoder layers [15, 23, 27, 31], decoder layers [8, 16, 24,
31]. [D]

**The construction set is smaller than headroom suggests, and the reason matters.**
Pooling is defined over a span's aligned interval, so the span is the pooling
unit. Of 419 eligible `D-construct` spans holding 699 baseline-correct units, only
**233 have every unit baseline-correct**. The strict definition was used:
**233 spans, 436 correct units, 20 dialogues, 125 utterances.** Pooling the
300-span / 699-unit alternative would mix baseline-correct and baseline-error
material inside a single positive vector and contaminate the premise. Zero pairs
were skipped at either site (`no_control` 0, `too_short` 0, `no_token` 0). [D]

**The conservative evaluation subset** (`D-dev-select`, ≥400 ms, `blank_to_preceding`):
482 spans, 211 utterances, **489 baseline-error units, 20 dialogues**. Applying the
convention *before* the floor is what moves this from Session 8's 409 spans / 455
units — the no-convention variant reproduces Session 8 exactly (409 / 455 / 20 /
660), so the difference is entirely the decided convention, not a discrepancy. [D]

Site D construction diagnostics [A]:

| Layer | Pairs | G | cos(Δ_raw, Δ_resid) | Ridge removed | U_prompt removed | cos(v_nat, U_prompt) | Bootstrap cos p05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 233 | 20 | 0.9971 | 0.0199 | 0.0003 | 0.0102 | 0.8630 |
| 16 | 233 | 20 | 0.9979 | 0.0282 | 0.0005 | 0.0300 | 0.9169 |
| 24 | 233 | 20 | 0.9986 | 0.0423 | 0.0004 | 0.0222 | **0.9630** |
| 31 | 233 | 20 | 0.9973 | 0.0315 | 0.0010 | 0.0277 | 0.9258 |

Residualisation barely rotates the direction (cos 0.994–0.999) while removing
2–5% of contrast energy. **The prompt subspace is nearly irrelevant at Site D** —
this is the empirical answer to the circularity objection, and the most defensible
single claim in the project. `corr(‖d_i‖, duration)` is −0.382 at Site E against
−0.032 at Site D; norm clipping bound 3 of 233 pairs at each site. [D]

Two things were implemented, tested, and **not run**: the LDA ablation
(`cos(Δ_LDA, Δ_mean)` unreported) and `corr(‖d_i‖, confidence)` — the latter
because no frozen alignment-confidence measure exists in this protocol, and
substituting a decoding-confidence proxy would introduce an unfrozen measure. [D]

### 5.5 Day 4 — teacher-forced oracle screen (job 40953, 16 Aug)

**449 conditions × 489 targets = 219,561 scored rows.** Teacher-forced only, no
free decoding. Controls: label-permuted null, matched-norm random, opposite sign,
wrong location — 5 draws each, control seed 20260816. [A]

**Site D beats the label-permuted null at layers 8, 16 and 31, at every ρ, in
every stratum. Site E beats no control anywhere.** Substitution-only, G=19: [D]

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

`*` = interval excludes zero. Strongest cell: **Site D layer 16, ρ=1, +0.1057
[+0.0466, +0.1649], G=19.** Secondary outcomes agree: margin **+0.119 [+0.036,
+0.202]\***; gold rank improves 42.6 positions but does not resolve at G=19
(heavy-tailed); entropy flat. [D]

**The nulls were genuinely rebuilt, not negated.** Norm ratios ‖Δ_null‖/‖Δ_real‖
span **0.175–0.619** (Site D median 0.313, Site E 0.377); a sign-flipped copy
would give exactly 1.000. Pair permutation was deliberately **not** implemented —
for a paired mean difference, Σᵢ(h⁺ᵢ − h⁻_π(i))/N = mean(h⁺) − mean(h⁻) for every
π, so it is algebraically incapable of being a null. [D]

**The strata discrepancy from Day 3 was fully resolved.** Day 3's span-level
counts (256 substitution-only, 108 wrong-language) become 199 and 90 at unit
level. All 489 units satisfy every eligibility condition individually
(489/489 on duration, matrix-preceded, non-final, contiguous). The 57-unit
difference is **entirely deletions co-located in a span that also holds a
substitution** — excluded by category, never by eligibility. All-errors is
unaffected because its span-level filter and unit-level sum agree (489 = 489,
verified). [D]

Span duration over the **489 target units**: min 400.1, median **2,280.5**, P90
**5,782.2**, max 11,403.9 ms **[C]**; over the **482 spans**: median 1,000.9, P90
3,199.5 ms **[A]**. Both are correct on their own denominator — see §10.2.

**A correction to the prediction.** Site D moves deletion-category units *at least
as much* as substitutions under teacher forcing (+0.157 vs +0.106 at ρ=1) — the
opposite of the predicted by-construction null. The prediction is not wrong; it
does not apply here. "Site D cannot act at a step that does not exist" is a
property of **free decoding**. Under teacher forcing the gold sequence is
supplied, so every gold token has a step. **The Day 4 deletion result cannot be
read as evidence that Site D will repair deletions in free decoding.** [D]

C11 (both sites simultaneously) is near-additive at ρ=1 (interaction −0.022 to
+0.003) and becomes materially negative only at ρ=2 on layer pairs already
collapsing individually. Reported as descriptive. [D]

### 5.6 Day 5 — dose extension, tiers, first free decoding (job 41007, 16 Aug)

**The dose curve saturates and then declines** — the anti-artifact argument that
Day 4 could not close: [D]

| Layer | ρ=0.25 | ρ=0.5 | ρ=1 | ρ=2 | **ρ=3** | **ρ=4** |
|---|---|---|---|---|---|---|
| 8 | +0.008\* | +0.017\* | +0.031\* | +0.066\* | **+0.076** | **+0.043** |
| 16 | +0.024\* | +0.054\* | +0.106\* | +0.178\* | **+0.170\*** | **+0.079** |

Layer 16 ρ=4 is +0.0786 [−0.0990, +0.2563] — the interval crosses zero. An effect
that grew without bound would be hard to separate from a generic magnitude
perturbation; the observed rise-peak-decline is the shape a genuine directional
mechanism predicts. **The frozen ρ=1 sits on the rising limb and was not revisited
on this curve.** [D]

**Eligibility expansion buys almost nothing.** 400 → 200 ms adds 26 error units
(489 → 515, +5.3%) and 8 substitution units (+8.0%); the effect is flat within
noise (+0.106 / +0.110 / +0.104), every interval excluding zero, G=20 at every
tier. **Consequence: relaxing the floor is not available as a route to power.** [D]

**First free decoding, oracle-localized** (upper bound, not the practical result) [A]:

| Metric | Baseline | Steered |
|---|---:|---:|
| PIER | 0.5249140893 | **0.5189003436** |
| MER | 0.2850386434 | **0.2780020764** |
| WER | 0.8683937824 | **0.8673575130** |
| EN WER | 0.5292096220 | 0.5231958763 |
| ZH CER | 0.2448515646 | 0.2376303825 |

**7 corrections over 489 targets = 1.43%.** [A]

**The POI category shift is worth stating and is not in any document.** The
oracle arm's net POI gain is 611 → 604 errors, decomposed as: deletion
**365 → 353 (−12)**, wrong-language substitution **114 → 120 (+6)**,
same-language substitution 90 → 91 (+1), phonetic 21 → 19 (−2), boundary 18 → 18,
other 3 → 3. **[A]** The entire net gain is deletions, and the intervention makes
wrong-language substitutions *worse* — the category the language direction was
built to fix.

**Hook firing under free decoding** [A]:

| Category | Targets | Hook fired | Corrected (of fired) |
|---|---:|---:|---:|
| deletion | 290 | **207 (71.4%)** | 6 / 207 = **2.90%** |
| substitution/other | 199 | **198 (99.5%)** | 1 / 198 = **0.51%** |

For **28.6% of deletion targets the oracle step did not exist** — the decode
terminated before reaching it — against 0.5% for substitutions. That is the
predicted mechanism, measured, with no analogue under teacher forcing. But
conditional on the step existing, deletions still correct ~6× more often. **The
prediction's mechanism is confirmed and its conclusion is not observed**; at 7
events the ratio is not distinguishable from chance. [D]

**Day 5's "0 corruptions" was wrong**, and Day 7 corrected it — see §9.3.

### 5.7 Day 6 — invariance battery (jobs 41020 and 41142, 16 Aug) — **reconstructed**

**There is no dated document for Day 6.** This section is reconstructed from
`invariance/generation_001` (66,924 rows) and `invariance/generation_002_zh_repair`
(14,670 rows).

**Frozen configuration** [A]: headline cell Site D layer 16, ρ=1,
substitution-only, 400 ms, correct vs label-permuted null, G expected 19; jitter
{50, 100, 200} ms reported separately for EN spans and ZH control spans; four
conventions; two aligner paths (`existing_ctc`, `whisper_dtw`, with the recorded
note *"two paths, not three; the cross-attention path is a documented negative
result and is not a cell"*); three poolings; inverse-variance weighting;
`selects_anything: false`, `evaluates_gate: false`.

#### 5.7.1 Inverse-variance reweighting — the construction-artifact question, settled

The concern was that Site E's null might be an artifact of how its direction was
built, because pooled contrast magnitude is anti-correlated with span duration.
Reweighting demonstrably removed the coupling and Site E stayed null [A]:

| Direction | corr(‖d‖, dur) before | after | weight ratio max/min | cos with published |
|---|---:|---:|---:|---:|
| site_e_layer15 | **−0.3857** | **−0.0192** | 6.47 | 0.9948 |
| site_e_layer23 | −0.0077 | −0.0856 | **2709.4** | 0.9052 |
| site_e_layer27 | −0.0117 | −0.0830 | **1587.3** | 0.8991 |
| site_e_layer31 | −0.0197 | −0.0623 | **322.8** | 0.9503 |
| site_d_layer8 | −0.0319 | −0.0117 | 7.46 | 0.9969 |
| site_d_layer16 | +0.0384 | −0.0208 | 3.64 | 0.9975 |
| site_d_layer24 | +0.0844 | −0.0294 | 7.56 | 0.9964 |
| site_d_layer31 | +0.0572 | −0.0376 | 12.68 | 0.9849 |

Two things must be said together. The coupling at layer 15 was real and is now
gone, and layer 15 is still null — so the null is a property of the site, not of
the construction. **But the weight ratios at Site E layers 23/27/31 are extreme
(322–2709×)**, meaning a handful of pairs carry those reweighted directions, and
their cosine with the published direction drops to 0.899–0.950. Layer 15, whose
ratio is a benign 6.5, is the cell that carries the argument.

#### 5.7.2 The EN-jitter row is inert by construction — verified exactly

The corpus token rate is **3.383095 tokens/s** [A]. At that rate ±50 and ±100 ms
round to zero decoder steps. The artifact records this directly [A]:

| EN jitter | fraction of utterances displaced | mean step shift |
|---|---:|---:|
| ±50 ms | **0.0000** | 0.000 |
| ±100 ms | **0.0000** | 0.000 |
| +200 ms | **0.2759** | +0.276 |
| −200 ms | **0.2500** | −0.250 |

Four of six EN-jitter cells are therefore the baseline by construction, and my
recomputation confirms it: those four cells return an effect identical to the
baseline cell to four decimal places (+0.1057 each) **[C]**. Only ±200 ms
displaces anything, and only for a quarter of utterances.

**This is a finding about why Site D is robust — sub-token boundary error cannot
move a decoder-step intervention — not six independent confirmations.** It must be
presented that way.

ZH control offsets are likewise recorded: **0 steps at ±50 and ±100 ms**, +1 at
+200 ms, −1 at −200 ms [A]. So the four ZH cells at ±50/±100 ms are identical
constructions differing only in null-draw seed, which makes their spread a direct
read of null-draw noise.

#### 5.7.3 Null-draw noise, measured directly

Per-null-draw paired contrasts for the baseline cell (`existing_ctc` /
`blank_to_preceding`, substitution-only, n=199) **[C]**:

| Draw | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Effect | +0.1187 | +0.1051 | +0.0855 | +0.1249 | +0.0945 |

Spread **0.0855 – 0.1249**, i.e. roughly **±0.02** around the mean of +0.1057 from
a single seed change. The ±0.03 figure used as the noise yardstick throughout the
programme is, if anything, conservative. **Any arm-to-arm difference smaller than
this is not a distinction.**

#### 5.7.4 The degenerate cell and its repair

`jitter_zh −200 ms` returned exactly **+0.0000** because the sign mapping produced
`control = onset − 1 − (−1) = onset`, making the contrast identically zero — a
zero-vector "direction", with correct and null margins bit-identical. **[C]
confirms the artifact value is exactly 0.0000.**

The repair clamps the control strictly before onset under every offset sign:
`min(onset − 1, onset − 1 − control_offset)`. Repaired value **[C]**: **+0.0854**
(unit mean) / **+0.0969** (dialogue-balanced), against the circulated
**+0.0915 [+0.0156, +0.1674]** [D].

**"Nothing else moved" is verified exactly.** Comparing every row the repair job
recomputed against `generation_001`, excluding the repaired cell: **11,736 rows,
max absolute difference 0.0 on `gold_logprob`, `margin`, `entropy`, and
`gold_rank`.** **[C]** The repaired cell itself differs, as it must (max |diff|
3.697).

*Caveat that must be recorded:* the clamp maps offset −1 onto the offset-0
construction, so the repaired cell is a **null-seed replicate of the baseline
construction, not an independent jitter probe**. A Mandarin control cannot sit
later than the step immediately before onset, so positive-direction ZH jitter is
structurally clamped.

#### 5.7.5 Cell effects — and a reconciliation failure

Recomputed from `invariance_scores.parquet`: paired per-unit difference between
the correct arm and the mean of the five label-permuted draws, substitution-only
stratum (`category != deletion`), G=19 for every Site-D cell. Two aggregations are
shown because the project's construction convention is dialogue-balanced while
`paired_contrast()` in the codebase takes a plain per-unit mean. **[C]**

| Factor | Cell | Units | Unit mean | Dialogue-balanced | Circulated [D] |
|---|---|---:|---:|---:|---:|
| baseline | `existing_ctc` / `blank_to_preceding` | 199 | +0.1057 | +0.1221 | +0.1192 |
| aligner | `whisper_dtw` | 187 | +0.0803 | +0.0873 | +0.0951 |
| convention | `blank_midpoint` | 205 | +0.1068 | +0.1096 | +0.1222 |
| convention | `blank_to_following` | 203 | +0.1052 | +0.1138 | +0.1167 |
| convention | `blank_excluded` | 184 | +0.0631 | +0.0891 | +0.0783 |
| jitter ZH | +200 ms | 199 | +0.1043 | +0.1190 | +0.1642 |
| jitter EN | −200 ms | 199 | +0.0881 | +0.1149 | +0.0986 |
| jitter EN | +200 ms | 199 | +0.0729 | +0.1014 | +0.0692 |
| inverse-variance | Site D reweighted | 199 | +0.1042 | +0.1234 | +0.1150 |
| inverse-variance | Site E layer 27 reweighted | 199 | +0.0336 | +0.0498 | +0.0164 |
| pooling | Site E hann | 199 | +0.0346 | +0.0530 | +0.0234 |
| pooling | Site E central60 | 199 | +0.0281 | +0.0486 | +0.0194 |
| pooling | Site E uniform | 199 | +0.0288 | +0.0407 | +0.0230 |

**The unit counts match the circulated table exactly in every cell** (199 / 187 /
205 / 203 / 184 / …), so the populations and strata are the same. **The effect
sizes do not reproduce under either estimator, and no estimator implemented in
this repository produces the circulated values.** See §10.1 — this is an open
discrepancy, not a resolved one.

**What is safe to state about Day 6 today:**

- the **sign is preserved in every valid cell** — this holds under both of my
  aggregations and under the circulated values;
- the **magnitude band is roughly +0.06 to +0.13** under recomputation (the
  circulated band, +0.069 to +0.164, is wider mainly because of the ZH +200 ms
  cell that fails to reproduce);
- **Site E remains null under reweighting and all three poolings** under every
  aggregation;
- the **structural findings — inert EN jitter, ZH offsets, the degenerate cell,
  the 11,736 bit-identical rows, the reweighting diagnostics — verify exactly.**

The invariance *claim* stands. The specific per-cell numbers should not be quoted
in a paper until the Day 6 estimator is recovered.

### 5.8 Day 7 — Gate B (job 41162, 16 Aug)

The first stage to ask whether the frozen action produces a net practical
improvement when the step must be found from the model's own output.

**Task 1 — oracle vs non-oracle, all tiers** [A]:

| Tier | Arm | Targets | Abstained | Corrections | Transport hits |
|---|---|---:|---:|---:|---:|
| 400 ms | oracle | 489 | 0 | **7** | 7 |
| 400 ms | non-oracle | 489 | **432 (88.3%)** | **0** | 7 |
| 300 ms | oracle | 502 | 0 | 7 | 7 |
| 300 ms | non-oracle | 502 | 434 (86.5%) | 0 | 7 |
| 200 ms | oracle | 515 | 0 | 7 | 11 |
| 200 ms | non-oracle | 515 | 434 (84.3%) | 0 | 11 |

The oracle arm reproduces Day 5 exactly. Relaxing the floor adds targets and the
oracle correction count stays at exactly 7 at all three tiers.

**Corruption, with a proper denominator** [A]:

| Tier | Arm | Corrections | Units at risk | Corrupted | Ratio |
|---|---|---:|---:|---:|---|
| 400 ms | oracle | 7 | 6,264 | **13** | **7 : 13** |
| 400 ms | non-oracle | 0 | 6,264 | 0 | 0 : 0 |
| 300 ms | oracle | 7 | 6,391 | 13 | 7 : 13 |
| 300 ms | non-oracle | 0 | 6,391 | 2 | 0 : 2 |
| 200 ms | oracle | 7 | 6,792 | **19** | **7 : 19** |
| 200 ms | non-oracle | 0 | 6,792 | 0 | 0 : 0 |

**Corruptions exceed corrections in the oracle arm at every tier.**

**The gap** [D]: +0.0143 [+0.0033, +0.0254] at 400 ms, +0.0139 and +0.0136 at 300
and 200 ms, G=20, every interval excluding zero. **Fraction of the oracle effect
surviving inference-available localization: 0.000 at every tier.** PIER gap
−0.0060. The interval excludes zero because the non-oracle arm is exactly zero,
not because 7 corrections is many.

**Transport abstention, decomposed** — the diagnosis that Day 8 then tested [D]:

| Category | Targets | No step | **No English in hypothesis** | EN below floor | Not enriched | Localized | Hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| deletion | 290 | **28.6%** | 37.7% | 33.8% | 14.0% | 10.3% | 1.0% |
| wrong-language sub. | 90 | 0% | **81.1%** | 13.3% | 0% | 5.6% | 3.3% |
| same-language sub. | 77 | 0% | 0% | 40.3% | 36.4% | 23.4% | 1.3% |
| phonetic/translit. | 17 | 5.9% | 62.5% | 25.0% | 12.5% | 0% | 0% |
| boundary error | 12 | 0% | 0% | 33.3% | 33.3% | 33.3% | 0% |
| other | 3 | 0% | 0% | 33.3% | 66.7% | 0% | 0% |

Overall: no step 17.2%, abstained 88.3%, localized 11.7%, transport hit 1.4%.
Concrete cases: *social skills* → 社會技巧, *improving* → 進步 (translated away);
*Exercise your strength … generally* (dropped entirely).

Abstention as a function of a fixed cut on `max_q r_q`, so the rule can be re-read
without another run: 0.707 / 0.716 / 0.810 / 0.905 / 0.940 / 0.983 at cuts of
0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.5. [D]

**It is a localization null, not a dose null** [A]:

| Condition | Corrections | E_eff | G | n |
|---|---:|---:|---:|---:|
| correct direction | 0 | 17.4889 | 11 | 57 |
| label-permuted null (5 draws) | 0 | 1.5941 (range 1.348–1.951) | 11 | 285 |
| matched-energy random (5 draws) | 0 | **17.4889** (all five exactly) | 11 | 285 |

The matched-energy control's energy equals the correct direction's to four
decimals because it reuses the same `r_q` and ρ with a direction of identical
norm. **Where transport acts it delivers more energy than the oracle arm — 17.49
against 4.8289 — and still corrects nothing.** [A]

**Spillover concentrates decisively** [A]: 85 units changed outside the target (72
improved, 13 damaged) in **5 of 116 utterances**, per-utterance counts
**41, 32, 10, 1, 1** — top 2 = 85.9%, top 3 = 97.6%. The non-oracle arm's
spillover is zero.

**D-test headroom, reference-only** [A]: 6,257 utterances, **G=15**, 16.638 h,
176,345 reference units, 35,921 embedded-English units (20.37%). C00 baseline:
38,555 error units, 10,458 embedded-English error units, **27.12% of error mass**,
PIER 0.29114, MER 0.22548 (EN WER 0.30105, ZH CER 0.20183). **MDE 0.0608 at
G=15** [D] — roughly 4× the largest oracle effect measured anywhere in the
project. *No intervened condition was run or scored on `D-test`.*

### 5.9 Day 8 — hypothesis substitution (job 41736, 17 Aug)

> **Arms B and C are oracle conditions built to establish causation. They are not
> practical methods, were never eligible for selection, and must not be written up
> as proposed mitigations.**

Same 489 targets, 116 utterances, 20 dialogues, same frozen action, same transport
mechanism, same abstention rule. **Only the hypothesis token sequence that `g_t`
and its cross-attention derive from changed.** Arm A reproduced Day 7 exactly on
all 489 paired rows. [A][D]

**Arm C construction:** normalize and segment reference and hypothesis under the
frozen MER/PIER unitization; take the deterministic Levenshtein backtrace; insert
the reference English unit immediately before its aligned hypothesis unit (for a
substitution) or at the alignment cursor (for a deletion); map back to a raw
character boundary and insert with separating spaces; **preserve every original
character — the wrong token is not replaced or deleted**; re-normalize and require
the result to equal the original units plus exactly the requested insertions.
**Well defined for 489/489 (100%), skipped 0.** [A]

**Headline** [A]:

| Arm | Corrections / 489 | Corruptions / 6,264 | PIER | MER | WER | E_eff all | Delivered; E_eff active |
|---|---:|---:|---:|---:|---:|---:|---:|
| A baseline | **0** | 0 | 0.524914 | 0.285039 | 0.868394 | 2.0386 | 57; 17.4889 |
| B gold | **0** | 1 | 0.524914 | 0.285154 | 0.868394 | 3.9999 | 147; 13.3058 |
| C partial | **0** | 2 | 0.524914 | 0.285269 | 0.867876 | 2.6805 | 96; 13.6537 |

Correction rate 0.0000 [0.0000, 0.0000] in every arm, G=20, n=489 — **degenerate
intervals**, since every observed outcome is zero. They are not binomial upper
bounds. Corruption rates: B 0.000160 [−0.000058, +0.000377], C 0.000319
[−0.000081, +0.000720]. Mean per-utterance PIER change 0.000000 in all three arms.
Every B−A and C−A difference is one to three orders of magnitude below the ±0.03
null-draw noise. [A]

**The diagnosed state was eliminated, and coverage rose** [A]:

| Arm | No step | Step, no English | EN below floor | Not enriched | Localized | Delivered w/o gold step | Delivered |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 84 | **161** | 122 | 65 | 57 | 0 | 57 (11.7%) |
| B | 84 | **0** | 162 | 135 | 108 | 39 | **147 (30.1%)** |
| C | 84 | **0** | 214 | 95 | 96 | 0 | **96 (19.6%)** |

Wrong-language substitution: 73/90 = 81.1% in A → **0/90** in B and C. Deletion
(excluding 83 with no gold step): 78/207 = 37.7% → **0/207**. Transport hits rose
7 → 25 (B) → 12 (C). [A]

**Energy is not the explanation from either direction.** In B, the 108
gold-step-localized deliveries average `E_eff` 1.93 while the 39 no-gold-step
deliveries average **44.81** — the energy is concentrated in exactly the
deliveries that cannot correct a target by construction, and all 25 of B's
transport hits are among the 108. [C]

**Spillover** [A]: A changes 0 units outside the target; B changes 1 (damaged);
C changes 3 (1 improved, 2 damaged), in `ZH-CN_U1004_S0_221` (B) and
`ZH-CN_U0028_S0_120`, `ZH-CN_U1021_S0_68`, `ZH-CN_U2004_S0_281` (C).

**Controls were correctly not run.** No arm produced a non-zero correction rate,
so under the rule declared before the run a direction control cannot distinguish
an absent transcript effect from a mis-specified one. Zero control rows generated;
the decision was mechanical, after all three arms completed. [A]

---

## 6. Where the effect dies — cross-cutting synthesis

### 6.1 The funnel, in one table

489 baseline-error targets, 116 utterances, 20 dialogues, 400 ms tier, frozen
action. **[A]/[C]**

| Stage | Oracle | Non-oracle (A) | Gold-substituted (B) | Partial (C) |
|---|---:|---:|---:|---:|
| targets | 489 | 489 | 489 | 489 |
| a decoder step exists | 405 (82.8%) | 405 | 405 | 405 |
| intervention delivered | **489** (by construction) | 57 (11.7%) | 147 (30.1%) | 96 (19.6%) |
| lands within ±1 of gold step | 489 | 7 | 25 | 12 |
| **transcript corrections** | **7** | **0** | **0** | **0** |
| corruptions | 13 | 0 | 1 | 2 |

Reading down the columns: the oracle column is the mechanism's ceiling — 1.43% of
targets, 1.73% of the targets where a step exists, and net negative once
corruption has a denominator. Reading across the rows: substituting the hypothesis
buys coverage and step accuracy but no corrections.

### 6.2 Three ceilings stack, and only two are diagnosed

1. **No decoder step exists** for 28.6% of deletions (84/489 overall) — the decode
   terminates before reaching the target. Diagnosed, reproduces exactly across
   Days 5, 7, 8. Structural; no localizer can fix it.
2. **No English in the first-pass hypothesis** for 81.1% of wrong-language
   substitutions and 37.7% of deletions. Diagnosed, and **tested and not supported
   as the cause** (Day 8).
3. **The intervention does not convert a delivered, on-target edit into a
   correction.** Undiagnosed. This is where the programme now sits.

### 6.3 Step selection is the surviving lead

Even handed the complete gold transcript, transport lands within ±1 of the gold
step in only **25/108 = 23.1%** of its gold-step deliveries; in arms A and C the
figure is 12.3% and 12.5% of deliveries. **[C]** That is measurable without any
oracle, needs no new GPU run to characterise from the stored `selected_step` and
`oracle_step` columns, and would explain the null without any appeal to hypothesis
content.

### 6.4 What the intervention actually does when it lands

Three concrete cases, all verified from stored text. **[C]**

**It perturbs the English region without steering it to the reference.**
`ZH-CN_U1003_S0_22`: reference `… south say cities …`, baseline `… south cities …`.
In Arm B, all three targets in this utterance are `localized` **and all three are
transport hits** — the intervention landed on the right step with energy
delivered. The output became `… south sea cities …`; in Arm C, `… South Sea
cities …`. The model inserted an English token adjacent to the reference token and
still scored as no correction. This is the clearest single observation of the
mechanism operating in transcript space under non-oracle localization, and its
outcome is a *different* wrong token rather than the right one. It is one case and
is descriptive.

**Its one favourable off-target edit is Mandarin, not English.**
`ZH-CN_U1021_S0_68`, Arm C: inserted 那 to give 那就肯定是要去, matching the
reference. The single "improved" spillover unit in all of Day 8 is a Mandarin
function word.

**Gold-substituted transport can act where no gold step exists.**
`ZH-CN_U1004_S0_221`, Arm B: all 28 targets are `no_step_at_all`, yet transport
localized an earlier generated query, delivered energy, and deleted 这 from
其实这对 — one damaged unit and zero possible corrections. This is what the "39
delivered despite no gold step" column means in practice.

### 6.5 The non-oracle arm is not literally inert

At raw-text level the Day 7 non-oracle arm changes **1 of 116 utterances**
(`ZH-CN_U0027_S0_163`) and the Day 8 Arm A changes the same one. The change is
comma insertion, which normalization removes — hence zero corrections, zero
corruptions, zero spillover. **[C]** The oracle arm changes **10 of 116** at raw
text level but only **5** survive normalization as scored effects.

The statement "the non-oracle arm changes no transcript" should be "changes no
transcript after normalization."

---

## 7. Concentration: the positive result rests on two dialogues

This is the most consequential thing this compilation found, and it is not stated
in any existing document. All figures **[C]**, oracle arm, 400 ms tier.

| Effect | Utterances | Dialogues | Identifiers |
|---|---:|---:|---|
| raw text changed | 10 / 116 | 8 / 20 | — |
| any scored effect after normalization | 5 / 116 | 5 / 20 | CSD0014, CSD0020, CSD0502, CSD0532, CSD0541 |
| **all 7 corrections** | **2 / 116** | **2 / 20** | `ZH-CN_U0027_S0_114` (CSD0014), `ZH-CN_U1003_S0_56` (CSD0502) |
| all 13 corruptions | 4 / 116 | 4 / 20 | U0027_S0_114 (1), U0040_S0_149 (3), U1003_S0_56 (2), U1082_S0_60 (7) |
| all 85 spillover units | 5 / 116 | 5 / 20 | 41 + 32 + 10 + 1 + 1 |

The 7 corrections decompose as **6 deletions in one utterance** (`ZH-CN_U1003_S0_56`,
reference unit indices 36–39, 88–89) and **1 same-language substitution** in
another (`ZH-CN_U0027_S0_114`, index 44).

**Consequences that must appear in any write-up:**

- The correction-rate interval `+0.0143 [+0.0033, +0.0254], G=20` is computed over
  20 clusters of which **18 contribute exactly zero**. A wild cluster bootstrap in
  that configuration is essentially resampling the presence of two clusters. The
  interval is not wrong, but its width is not informative about the effect's
  precision.
- Corrections and corruptions **co-occur in the same utterances**: 2 of the 4
  corrupting utterances are the 2 correcting ones. The intervention is not doing
  a little good in many places and a little harm elsewhere; it is doing everything
  — good and bad — in a handful of utterances and nothing at all in 111 of 116.
- This aligns with, and strengthens, the spillover concentration finding. It is
  the same phenomenon measured on the target axis rather than the off-target axis.
- It also reframes the abstention argument favourably: if those utterances are
  identifiable before decoding, a router could decline them; but it would then
  decline the only utterances that ever produce a correction.

### 7.1 Effective cluster count

The nominal G is 20. For the correction outcome the effective count is **2**. Any
statement of the form "G=20" attached to the correction rate should be read with
that alongside it.

---

## 8. Power arithmetic — how far the Day 8 null reaches

Derived here from two verified quantities: the oracle arm's per-opportunity
correction rate (7/405 = **1.7284%**, §5.6 [A]) and the Day 8 transport-hit counts
(§5.9 [A]). Assumption stated plainly: **only interventions landing within ±1 of
the gold step can correct, and they correct at the oracle rate.** **[C]**

| Arm | Delivered | Hits (±1 step) | Hit precision | Expected corrections | P(0 observed) |
|---|---:|---:|---:|---:|---:|
| A | 57 | 7 | 12.3% | 0.121 | 0.886 |
| B | 147 | 25 | 17.0% (23.1% of 108 gold-step deliveries) | 0.432 | 0.649 |
| C | 96 | 12 | 12.5% | 0.207 | 0.813 |
| **all three** | 300 | **44** | — | **0.760** | **0.467** |

Poisson and exact binomial agree to three decimals (binomial P(0) = 0.464).

**Therefore:** observing zero corrections across all three arms is a coin-flip
outcome under the hypothesis that the effect is unchanged. The run **does** show
that eliminating the no-English state is not sufficient to restore corrections; it
**does not** show that hypothesis content is irrelevant, and it does not identify
the blocker. The correct claim is the narrow one, and the burden now sits on
anyone asserting the 81.1% is causal.

The same arithmetic applies retrospectively to Day 7: the non-oracle arm had 7
near-gold deliveries, expected yield 0.12 corrections. **Day 7's 0/489 was, on its
own, close to uninformative about the mechanism** — what made it informative was
the 88.3% abstention rate, which is a coverage measurement rather than an effect
measurement.

---

## 9. Defect register

Every defect found in the programme, what it would have cost, and whether it was
caught before or after the affected run.

### 9.1 Identity binding covered `tests/` — caught before GPU spend

Job 40373 refused in 2 s because six regression tests added after the v2r2 publish
moved `source_snapshot_hash`. Correct refusal, zero cost beyond a namespace
republish. Later narrowed to a **split** identity carrying `code_config_sha256`
and `test_sha256` separately, and the narrowing was proved live on real artifacts:
Day 3's artifacts verify `ok` after 13 tests were added, while Day 2's fail on
`code_config_sha256` **only** — code and configuration still gated, tests
correctly ignored. Legacy artifacts retain `source_sha256` semantics. [D]

A follow-on rule was needed and recorded: **identity applies to what a session
publishes and to artifacts consumed by pre-existing code, never to artifacts
consumed by newly written code** — writing the consuming module is itself what
changes the hash, so the check would be self-defeating. Day 4 was the fourth
occurrence and the first where the consuming module was the cause. [D]

### 9.2 Cross-attention tuple unpack — caught, nothing written

Job 40703 failed after all 600 utterances had been processed:
`conv.switch_transitions()` returns `tuple[DataFrame, dict]` and the whole tuple
was passed as `transitions=`. The destination did not exist, so no partial
publication occurred. Before resubmitting, the failed path was exercised end to
end on CPU against the real candidate table, and an injected +50 ms start shift
returned a 50.0 ms median — validating the disagreement arithmetic independently
of the GPU. [D]

### 9.3 Day 5's "0 corruptions" — caught after the run, and it inverted a sign

The count was taken inside a target set consisting **entirely of baseline errors**,
where corruption is impossible by construction. Counted properly over
baseline-correct units per utterance, the oracle arm produces **7 corrections
against 13 corruptions**. The oracle intervention is **net harmful**, and every
reading of Day 5 that treated zero corruptions as favourable was wrong. **This is
the defect to disclose in the paper.** [D][A]

A related trap: per-target spillover counting inflates 85 to **1,034**, because
one utterance can hold many targets. Both numbers are in the artifact; the
per-utterance one is correct. [A]

### 9.4 Abstention threshold τ=0.5 was scale-wrong — caught before the run

Whisper pads to 1500 frames and attends into the padding, so `r_q` never
approaches 0.5 — the smoke reached 0.114. A fixed cut of 0.5 would have abstained
always and made the non-oracle arm **vacuous by construction**. Replaced with the
parameter-free enrichment rule **before the full run and before any effect was
known**, with `r_max` recorded so any threshold can be read off post hoc. **The
ordering matters and should be stated in the paper.** [D]

### 9.5 Enrichment baseline was apples-to-oranges — caught before the run

`r_max` is a share of the whole attention row (padding included) while chance was
computed inside the audio, so no step could clear it. Fixed to put both on the
same denominator. Ties now abstain, because a region covering the whole utterance
gives enrichment exactly 1 and float noise was otherwise deciding the outcome. [D]

### 9.6 `tag_unit(str(unit))` tagged nothing as English — caught after the run

`str(Unit(...))` is the repr, not the surface, so every embedded-English count
silently became zero. Found by noticing D-test reported **0 embedded-English units
across 6,257 utterances**. **The stored Gate B artifact still carries the
uncorrected values** — `embedded_english_units: 0`,
`embedded_share_of_error_mass: 0.0` **[A]** — and the corrected values were
recomputed from stored text. The recomputation is independently confirmed by PIER,
which uses its own reference-POI logic and records `num_poi` = **35,921** and
`num_poi_errors` = **10,458** in the same artifact — an exact match. **[A]**

Affected: monolingual retention and D-test headroom only. **Correction,
corruption, spillover, transport, and control results never call `tag_unit`.** [D]

Anyone reading `gate_b_report.json` directly must know that the
`reference_only.embedded_english_units` and
`c00_baseline.embedded_english_error_units` fields are stale zeros and the
authoritative values are the PIER fields beside them.

### 9.7 Boundary jitter would have been vacuous — caught on Day 6

The Site-D target derives from unit indices, so shifting span *times* never
reaches it. Jitter now displaces the decoder step at the utterance's own token
rate — which is precisely why §5.7.2 shows four of six EN cells at zero
displacement. [D]

### 9.8 `control_step_for()` collapsed the control onto the onset

`jitter_zh −200 ms` returned identically zero. Repaired; 11 regression tests with
a failing fixture; under the pre-fix expression offsets −1, −2, −3 all violate
`control_step < onset`. See §5.7.4 for the verified "nothing else moved" check. [D]

### 9.9 Two smaller ones, caught before Day 5's real run

A walrus expression in `FROZEN_CONFIG` that assigned `None` and always took the
else branch; and `corpus_word_error_rate` reading keys
`substitutions`/`deletions`/`insertions` while `csasr.evaluation.mer.error_rate`
returns `sub`/`del`/`ins` — with `.get(..., 0)` this would have reported
**WER = 0.0000 for both arms**. Found by printing the function's actual return
keys and unit-checked against a 1-edit / 5-token case returning 0.2. [D]

### 9.10 Summary

Nine distinct defects. **Six were caught before the affected run** (9.1, 9.2, 9.4,
9.5, 9.7, 9.9), **three after** (9.3, 9.6, 9.8). Of the three caught after, one
inverted the sign of a headline result (9.3), one affected only two reported
quantities and was independently cross-checked (9.6), and one produced a
degenerate cell that was repaired with a bit-identical verification of everything
else (9.8). No defect is known to affect the correction, corruption, transport, or
control results.

---

## 10. Discrepancy register

Things that do not reconcile. Each is stated rather than resolved by preference.

### 10.1 Day 6 cell effects do not reproduce — **open, and it matters**

The circulated Day 6 table (§5.7.5, "Circulated" column) cannot be reproduced from
`invariance_scores.parquet` by either the plain per-unit paired mean used by
`paired_contrast()` in `src/csasr/experiments/v2r3_oracle_screen.py:374` or by the
dialogue-balanced mean the construction pipeline uses. Unit counts match exactly
in every cell, so the population and stratum are right; only the aggregation
differs. The largest divergence is `jitter_zh +200 ms`: circulated **+0.1642**,
recomputed **+0.1043** (unit) / **+0.1190** (dialogue-balanced) — and that cell is
the top of the circulated "+0.069 to +0.164" magnitude band.

**Root cause: Day 6 has no dated result document.** Days 1–5 and 7–8 all have one;
the invariance battery does not, so its analysis path was never written down.

**Recommended resolution before any paper draft:** recover the Day 6 analysis
code path, re-derive the 23 cells with the estimator stated explicitly, and write
the missing `docs/V2R3_INVARIANCE_2026-08-16.md`. Until then, quote the invariance
*conclusion* (sign preserved in every valid cell; Site E null under reweighting and
all three poolings), not the per-cell numbers.

### 10.2 Span duration median — reconciled

`oracle_screen_report.json` records median 1,000.9 ms / P90 3,199.5 ms; the Day 4
document says median 2,281 ms / P90 5,782 ms. **Both are correct**: the artifact
figure is over the **482 spans**, the document figure is over the **489 target
units** (confirmed: units median 2,280.5, P90 5,782.2 **[C]**). Min 400.1 and max
11,403.9 agree. No action needed beyond stating the denominator.

### 10.3 200 ms tier count: 517 vs 515 — cosmetic

`day5_report.json`'s plan block records 517 error units at the 200 ms tier
**[A]**; every analysis and both documents use **515**, which is what
`day5_scores.parquet` actually contains **[C]**. The plan-side count is stale. No
result depends on it.

### 10.4 Gate B stored vs recomputed embedded-English fields

See §9.6. The artifact holds uncorrected zeros; the documents hold the recomputed
values. Both are on disk and a naive reader of the JSON gets the wrong number.

### 10.5 Day 7 abstention rate stated two ways

The Day 7 document's §4 reports overall abstention **88.3%** (= 432/489, verified
**[C]**) and then, in the threshold table, "the parameter-free enrichment rule
actually abstains at **0.836**". The two denominators differ — 0.836 excludes
targets with no candidate region. Only 88.3% is comparable with the tier table.

### 10.6 The `blank_to_preceding` convention is provisional, not selected

It is used as the span convention from Day 3 onward. It was never selected by a
frozen procedure; it is a **decided input**, chosen on development evidence for
having the smallest EN−ZH asymmetry and two identified error mechanisms. Every
downstream number inherits that choice, and Day 6 exists to show the conclusion
does not depend on it. This is not an error — it is a dependency that must be
disclosed.

---

## 11. Verification performed during this compilation

All read-only. No artifact was written, modified, or deleted.

| Check | Result |
|---|---|
| Slurm ledger for every stage | 19 jobs recovered via `sacct`; states and elapsed times as tabulated in §4 |
| Day 8 headline numbers vs `hypothesis_substitution_summary.json` and the three parquets | **all match** the dated document: corrections, corruptions, abstention partitions, E_eff, CIs, spillover IDs, construction counts |
| 81.1% and 37.7% recomputed from Arm A rows | **0.8111** (73/90) and **0.3768** (78/207) — exact |
| Day 6 "11,736 rows bit-identical" | **verified**: max abs diff **0.0** on `gold_logprob`, `margin`, `entropy`, `gold_rank` |
| Day 6 EN-jitter inertness | **verified**: `frac_utterances_shifted` = 0.0000 at ±50/±100 ms; 0.2759 / 0.2500 at ±200 ms |
| Day 6 ZH control offsets | **verified**: 0 / 0 / 0 / 0 steps at ±50/±100; +1 / −1 at ±200 |
| Day 6 cell effects | **do not reproduce** — §10.1 |
| Day 5 oracle free decoding | **verified**: 7 corrections, hook_fired 405 (207 + 198), spillover 85 per-utterance / 1,034 per-target, 116 utterances |
| Day 5 PIER category decomposition | **verified** from `day5_report.json`; reported in §5.6 |
| Day 7 controls | **verified**: 5 + 5 draws, n=57 each, G=11, 0 corrections, matched-energy E_eff exactly 17.4889 ×5 |
| Day 7 corruption / retention / spillover | **verified**: 13 / 6,264, 85 units in 5 utterances, per-utterance 41-32-10-1-1 |
| Correction concentration | **new finding**: 7 corrections in 2 utterances / 2 dialogues — §7 |
| D-test C00 baseline | **verified** against artifact; embedded-English fields stale, PIER fields authoritative — §9.6 |
| Test suite collection | **701 tests across 47 files** collected cleanly |

**Not run:** the full CPU suite (the Day 8 session's attempt reached 71% with no
failures before a 900 s timeout, so there is no current full-suite pass on
record); any GPU work; any re-derivation of a bootstrap interval.

---

## 12. Scientific boundaries maintained throughout

| Boundary | Status |
|---|---|
| Production Gate A | **unchanged** at every stage |
| Span freeze / L1c | no freeze written; **L1c remains locked** |
| Gate generation 3 | **never allocated, exposed, or consumed**; ledger mtime `1786502490` unchanged |
| `D-dev-confirm` | **never opened** |
| `D-test` | reference-only counts and a C00 baseline; **0 intervened conditions evaluated** |
| Layer, strength, floor, tier, convention, path | **none selected** by any run; all frozen or decided by a human beforehand |
| Gate B | **not declared** passed or failed by any run; numbers reported against the criterion for a human decision |
| Thresholds / criteria | none lowered, widened, removed, or reinterpreted |
| Smoke generations | refused **by name** as inputs; `smoke_used_as_input=false` recorded |
| Diagnostic taint | carried on every diagnostic artifact (Session 8) |

Against the Gate B criterion, numbers only [D]:

| Criterion clause | Oracle (400 ms) | Non-oracle (400 ms) |
|---|---|---|
| net utility positive, 95% CI excluding zero | +0.0143 [+0.0033, +0.0254]; PIER −0.0060 | +0.0000 [0, 0]; PIER unchanged |
| corrections exceeding corruptions | **7 : 13 — fails** | 0 : 0 — vacuous |
| correct direction beats every required null | not run in this arm | **0/57 vs 0/285 and 0/285 — beats neither** |

---

## 13. Consolidated open items

**Scientific**

1. **The blocker between a delivered on-target edit and a transcript correction is
   unidentified.** Day 8 removed the leading candidate without supplying a
   replacement. (§6.2)
2. **Step-selection accuracy is the surviving lead** — 23.1% even with the gold
   transcript. Characterisable from stored columns without a GPU run. (§6.3)
3. **Every transcript-space claim rests on ≤7 events concentrated in 2 dialogues.**
   Either enlarge the target population until an oracle-size effect is detectable,
   or state throughout that transcript-level correction is descriptive and the
   representation-level result is the finding. (§7, §8)
4. **`D-test` at G=15 has MDE 0.0608**, ~4× the largest oracle effect measured. A
   confirmatory test on this split is underpowered independent of everything above.
5. **The third aligner path does not exist.** Four candidate corrections to the
   cross-attention method are identified and each needs its own preregistration.
6. **Site E's reweighted directions at layers 23/27/31 rest on extreme weight
   ratios** (322–2709×). Layer 15 carries the argument; the others should be
   quoted with the ratio.
7. **`wrong_language_substitution` rests on 12 dialogues** — the sharpest test of
   the hypothesis has the fewest clusters.
8. **Whether the 5 pathological utterances are identifiable before decoding**
   decides whether an abstention rule is a method contribution or a post-hoc
   filter — and note they are also the only utterances that ever correct. (§7)
9. **`router_calib_split` remains unset** — how 8 dialogues divide between
   probability calibration and threshold selection is a scientific decision.
10. **The `existing_ctc` span source is inherited, not justified.** No sensitivity
    analysis over the span family was run.

**Documentation and code**

11. **Day 6 has no result document and its cell estimator is unrecovered.** (§10.1)
    Highest-priority documentation gap.
12. **`gate_b_report.json` carries stale zero embedded-English fields.** (§9.6)
13. **The partition fingerprint is not a literal field** in the candidate
    generation manifest — the binding is real but indirect through parent hashes.
14. **No test covers the Session 8 headroom module.** Its eligibility logic is
    unverified by any regression test.
15. **No current full-CPU-suite pass is on record** — 701 tests collect; the last
    attempt timed out at 71% with no failures.

---

## 14. What the evidence supports as a claim

**Supported, at representation level:**

> A single language direction at Whisper's decoder layer 16 has causal leverage
> over the model's next-token prediction on code-switched speech. It beats four
> controls, its dose-response rises, peaks, and declines, it is nearly orthogonal
> to the prompt/language-token subspace, and the corresponding encoder site is
> null — a null that survives reweighting which demonstrably removed the
> confound it was designed to test.

**Supported, at transcript level:**

> That leverage does not convert into transcript corrections under free decoding.
> With an oracle decoder step it produces 7 corrections against 13 corruptions,
> concentrated in two dialogues. With inference-available localization it produces
> none, at any eligibility tier, because transport abstains on 88.3% of targets.

**Supported as a measurement, not as a cause:**

> For 81.1% of wrong-language substitutions and 37.7% of deletions the first-pass
> hypothesis contains no English token to localize against. This is a real,
> predicted-in-advance structural property of first-pass-conditioned localization
> for CS-ASR.

**Explicitly not supported:**

> That the missing English token is what blocks the correction. Two oracle-input
> arms removed that state completely, raised delivered coverage from 57 to 147
> targets and transport hits from 7 to 25, and still corrected nothing — though
> with only 44 near-gold deliveries in total, the run had a coin-flip chance of
> observing zero even if the effect were unchanged.

**The honest shape of the paper** is a negative result with a diagnosed
mechanism and one explicitly undiagnosed step, supported by a preregistered causal
probe that returned its third declared outcome and was reported without
adjustment. A reviewer asking "did you check whether the missing token was the
cause?" gets an experiment rather than an argument. That is worth more than the
claim it cost.

---

### Document provenance

| Section | Primary source |
|---|---|
| §3, §5.0 | `docs/V2R3_CANDIDATE_GENERATION_2026-08-14.md` (job 40374) |
| §5.1 | `docs/V2R3_BASELINE_POI_2026-08-15.md` (job 40486) |
| §5.2 | `docs/SESSION8_V2R3_HEADROOM_AND_RUNA_2026-08-15.md` |
| §5.3 | `docs/V2R3_CROSS_ATTENTION_SPANS_2026-08-15.md` (job 40711) |
| §5.4 | `docs/V2R3_DIRECTIONS_2026-08-16.md` (job 40919) + `directions_report` |
| §5.5 | `docs/V2R3_ORACLE_SCREEN_2026-08-16.md` (job 40953) |
| §5.6 | `docs/V2R3_DAY5_EXPANSION_2026-08-16.md` (job 41007) + `day5_report.json` |
| §5.7 | **no document exists** — reconstructed from `invariance/generation_001` and `generation_002_zh_repair` (jobs 41020, 41142) |
| §5.8 | `docs/V2R3_GATE_B_2026-08-16.md` (job 41162) + `gate_b_report.json` |
| §5.9 | `docs/V2R3_HYPOTHESIS_SUBSTITUTION_2026-08-17.md` (job 41736) |
| §6.4, §7, §8 | derived here from stored artifact rows; estimators stated in place |
| §12 | `docs/RUNBOOK_V2_2026-08-14.md` §9, §11, §12A |

Companion: `docs/RESULTS_RUN_7DAYS.md` is the shorter claim-by-claim inventory for
writing; this document is the underlying record with provenance.

**No number in this document is a production Gate-A result. Production Gate A
remains as recorded in `docs/RUNBOOK_V2_2026-08-14.md` and is unchanged by
Days 4–8.**
