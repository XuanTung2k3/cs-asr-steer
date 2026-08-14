# Runbook v2 — alignment to Gate B

**Date:** 2026-08-14
**Status:** Active. This document is the authority for the current alignment and Gate-B criterion.
**Supersedes:** the Gate-A criterion in `GATE_A_CURRENT_STATUS_2026-08-13.md`, and the human-annotation plan in `ANNOTATION_PACK_IMPLEMENTATION_AND_RESULTS_2026-08-13.md`. Both are retained as historical evidence; their bodies remain accurate as records of what was believed on 13 August.

---

## 0. What changed, and why

Two decisions taken on 13–14 August restructure the alignment work.

**Human lexical annotation is dropped.** The claim moves from *accuracy* ("boundaries are within X ms of truth") to *invariance* ("conclusions hold across jitter, conventions, and independent aligner paths"). This is scientifically legitimate, fully automatic, and transfers to any corpus at zero marginal cost. The annotation pack built on 13 August is superseded, not failed; its switch-universe enumeration is reused.

**A third aligner path is added.** Cross-attention pseudo-labels, following the language-alignment method of Liu et al., IEEE TASLP 2025 (arXiv 2403.05887). That paper explicitly rejects CTC for frame-to-token alignment because blank frames carry no language attribute, and reports that cross-attention pseudo-labels outperform forced alignment computed from a comparable-strength model.

Net effect: the decisive experiment (oracle screen) moves from Day 7 to **Day 4**.

---

## 1. Gate A, redefined

**Gate A is a development measurement, not a held-out confirmatory gate.**

Exposure ledgers, immutable gate generations, and held-out accounting exist to protect a preregistered claim evaluated once. That is `D-test`. An alignment audit is a preprocessing measurement reported in Experimental Setup.

### What Gate A no longer requires

- Absolute lexical boundary accuracy against gold. The corpus has no gold word timings, human annotation was deliberately dropped, and no absolute-accuracy claim is made.
- A human `manual_lexical` reference.
- Allocation or consumption of a gate generation.
- The 100 ms / 200 ms precision targets, which were inherited from a superseded design that pooled encoder frames over exact spans.
- The 30 ms EN−ZH criterion **as applied to cross-aligner disagreement**. That bound was specified for signed lexical error against gold. Cross-aligner disagreement is a different quantity with much larger natural variance, and is reported descriptively.

### What Gate A now requires

1. A **provisional aligner configuration**, frozen on development evidence, with a recorded config hash.
2. An **invariance battery** (Day 6) showing that the headline conclusion is preserved across boundary jitter, blank conventions, independent aligner paths, and pooling schemes.
3. Boundary statistics **reported by language class** throughout, as a descriptive confound check.

Gate A passes when a provisional configuration is frozen and the invariance battery shows no conclusion flipping. It does not certify accuracy, and the paper does not claim it.

### Aligner paths in scope

| Path | Status |
|---|---|
| CTC forced alignment under a named blank convention | Characterised (Day 1) |
| Whisper DTW | Reference for disagreement statistics |
| Cross-attention pseudo-labels | To be built (Day 2) |

---

## 2. Eligibility rule (current)

An eligible unit is an embedded-English lexical unit with:

- aligned span duration above the tier floor (≥400 ms conservative; ≥300 ms and ≥200 ms as expansion tiers);
- alignment confidence above the frozen threshold;
- matrix-language preceding context;
- **not utterance-final** (added 14 August — see Day 1 findings);
- non-contiguous reference-unit indices excluded.

Pooling is a Hann taper over the aligned interval, normalised to sum 1, applied identically to positive and matched control spans. Central-60% and uniform full-span are retained as Day 6 sensitivity variants.

---

## 3. Day 1 — CTC blank conventions · **COMPLETE (13–14 Aug)**

Four named conventions recomputed over cached candidates, development roles only (`D-construct`, `D-dev-select`), 3,028 paired targets.

### Results

| Convention | Median | P90 | ≤100 ms | ≤200 ms | EN med | ZH med | EN−ZH |
|---|---:|---:|---:|---:|---:|---:|---:|
| `blank_excluded` | 440 | 1,260 | 8.22% | 23.94% | 240 | 680 | −440 |
| `blank_to_preceding` | **340** | 1,220 | **10.40%** | **30.45%** | 280 | 420 | **−140** |
| `blank_to_following` | 680 | 1,460 | 1.75% | 8.26% | 560 | 720 | −160 |
| `blank_midpoint` | 480 | **1,000** | 3.01% | 12.25% | 400 | 590 | −190 |

### Findings

**The blank-ownership hypothesis is partially confirmed.** Median blank run at language switches on natural speech is 220 ms — not the ~800 ms figure from `GATE_A_CURRENT_STATUS_2026-08-13.md` §5.1, which was a seam-relative measurement on *synthetic spliced* items and therefore a different population. Blank reassignment accounts for roughly 100 ms of a 440 ms median. It is a term, not the term.

**`blank_to_preceding` reduces the class asymmetry 3×**, from EN−ZH −440 ms to −140 ms. This matters more than the median improvement: class-asymmetric boundary error enters the paired contrast `d_i = x⁺ − x⁻` directly and contaminates the direction itself.

**Utterance-final targets are a separate, unaddressable mechanism.** 600 of 3,028 targets (19.8%) are utterance-final, disagree at 700 ms median, and are **unchanged by every convention** — consistent with DTW absorbing trailing silence into the final token, which involves no between-language blank run. Excluding them takes the working median from 440 ms to 280 ms under `blank_to_preceding`.

**Switch universe is clean.** Of 2,428 transitions: 2,347 (96.7%) actionable single-blank-run, 56 zero-length, 25 non-contiguous. No overlaps, no unusable edges, no non-finite boundaries, no third-language interruptions.

### Consequences

- Utterance-final exclusion added to the eligibility rule (§2).
- `blank_to_preceding` is the **provisional** CTC convention, pending three-way comparison. Not frozen.
- Residual 280 ms median after best convention and utterance-final exclusion remains unexplained. This is a reason not to over-invest in repairing CTC.

### Reproduction criterion (re-baselined 14 Aug)

The original criterion mixed an all-role paired count (9,728) and P90 with a diagnostic prohibited from reading four of six roles — arithmetically incompatible. Option 1 was chosen: keep development-only scope, re-baseline to that population. Median, EN median, ZH median, and EN−ZH must match documented values exactly; paired count must equal the development-role target count. P90 removed from the criterion. All conditions hold.

---

## 4. Day 2 — Cross-attention pseudo-labels (real model, GPU)

The new path, and simultaneously the real-model validation.

Run frozen Whisper-large-v3 with teacher forcing on gold transcripts over `D-dev-select` and `D-construct`. Average cross-attention over heads in the final decoder layer, assign each encoder frame the language of its argmax token, group contiguous same-language frames into spans.

No DTW. No monotonic path. Argmax over an averaged attention matrix — which is why this may succeed where `whisper_dtw` failed.

**Freeze in advance:** decoder layer, head set, smoothing window, minimum span length, special-token handling (a third class, never silently assigned to a language).

**Compare against all four CTC conventions and Whisper DTW**, not only the provisional convention. Three-way agreement is required for the Day 3 selection rule, and the comparison is nearly free once spans exist.

**Real-model checks (this run doubles as validation):** empirical ms-per-frame vs expected frame rate; attention mask genuinely masking padded regions, verified on a real batch; special-token frame fraction plausible; diagnostic taint on all artifacts; nothing written to production paths.

**Watch for:** if cross-attention agrees well with Whisper DTW *and* shows low EN−ZH asymmetry, CTC may drop out of the pipeline entirely. Two clean paths beat three where one has two unmodelled error mechanisms.

---

## 5. Day 3 — Freeze config, build subset, construct directions

**Selection rule, fixed in advance:** prefer the aligner path with highest agreement to the other two independent paths, breaking ties by lower boundary variance within span-duration strata. This is a provisional choice supporting an invariance claim, not a certified-accuracy claim. Record the config hash.

**Conservative oracle subset** from `D-dev-select` under §2 eligibility at the ≥400 ms tier, baseline-error units only, central-mass pooling. Low coverage is expected and correct.

**Construct directions** on `D-construct`: `Δ^E` at candidate encoder layers; `Δ^{D⊥}` at candidate decoder layers with the prompt subspace projected out of each contrast before aggregation.

Report: `cos(Δ_raw, Δ_residualized)`; ridge energy fraction removed; speaker-bootstrap cosine stability; `corr(‖d_i‖, duration)` and `corr(‖d_i‖, confidence)`; for Site D, `cos(v_nat, U_prompt)` and projection energy removed.

**Numerical requirement:** if shrinkage LDA is run as the estimator ablation, perform covariance estimation and inversion inside an orthonormal basis of the complement of `U_prompt`, then map back.

---

## 6. Day 4 — Oracle screen · **decisive**

Teacher-forced only. C00 / C10 / C01 / C11 across candidate layers and ρ ∈ {0.25, 0.5, 1, 2}.

**Controls, same batch, not deferred:**

| Control | Answers |
|---|---|
| Wrong location (random position, same utterance) | Is the span meaningful? |
| Matched-norm random direction, ≥5 draws | Direction or perturbation? |
| Opposite sign | Is the axis directional? |
| Label-permuted, ≥5 draws | Does the language pairing carry the effect? |

**Do not implement pair permutation.** For a paired mean difference, permuting which positive pairs with which negative leaves the direction algebraically unchanged: `Σᵢ(h⁺ᵢ − h⁻_π(i))/N = mean(h⁺) − mean(h⁻)`. Permute **labels** within nuisance-matched strata and rebuild through the full pipeline including residualisation. Report `‖Δ_null‖/‖Δ_real‖` per draw.

**Report:** Δ gold-token logprob, gold rank, margin, entropy, with conversation-block bootstrap CIs and **paired** contrasts against each control. Dose–response over ρ per site. Natural-overlap / Mahalanobis displacement for **C11 separately**, since `Δ^D` was estimated on unintervened decoder states.

**Wrong-location does double duty.** If steering the CS span beats steering a random position in the same utterance, the spans localise something real — without any gold reference. This is what replaces annotation, and it is the honest substitute.

---

## 7. Day 5 — Decision, or offset sweep

**If the effect is present** (correct direction beats all four controls with non-overlapping paired CIs, monotone saturating dose–response): expand eligibility 400 → 300 → 200 ms, measure effect against coverage, then run a first free-decoding check on the conservative subset.

**If absent:** run the offset sweep before concluding. Same direction and strength at −300, −150, 0, +150, +300 ms, reported separately for English and Mandarin spans.

| Result | Diagnosis | Action |
|---|---|---|
| Peak at non-zero offset | Systematic span displacement | Correct offset, rerun Day 4. The peak location estimates the displacement — measured without gold. |
| Flat curve | No causal leverage at these sites | Gate B fails. Activate the preregistered fallback. Do not attempt further alignment repair. |

---

## 8. Day 6 — Invariance battery

Rerun the Day 4 headline result under:

1. Boundary jitter ±50 / ±100 / ±200 ms, **separately for English and Mandarin spans**
2. All four CTC conventions
3. All three aligner paths
4. Pooling: Hann taper, central-60%, uniform full-span

**Pass condition:** sign and approximate magnitude preserved in every cell; no conclusion flips.

Also report the EN−ZH signed boundary difference per path, descriptively. This section is the paper's alignment evidence, replacing an accuracy table.

---

## 9. Day 7 — Gate B

Free decoding on the conservative subset and expansion tiers.

**Pass condition:** net utility positive with conversation-block bootstrap 95% CI excluding zero; corrections exceeding corruptions; correct direction beating every required null in paired bootstrap contrasts.

**Required stratification: deletion vs substitution.** Site D cannot act at a decoder step that does not exist, so a deleted word is structurally beyond its reach; a pooled Site-D result averages a mechanism that cannot fire with one that can.

**Headroom:** eligible units, baseline errors, cluster count, embedded share of error mass, bootstrap MDE — on `D-test` reference-only counts and C00 baselines (permitted; these are not intervention outcomes) as well as `D-dev-select`.

---

## 10. What the paper says about alignment

Draft into Experimental Setup once Day 6 completes:

> Word-level timing annotations are unavailable for this corpus, and forced alignment on code-switched speech is known to be unreliable (Liu et al., 2025). We therefore make no claim about absolute boundary accuracy. Instead we derive spans from three independent paths — CTC forced alignment under an explicit blank-frame convention, Whisper DTW, and cross-attention pseudo-labels following Liu et al. (2025) — and show that our conclusions are invariant to the choice among them, to boundary jitter of up to ±200 ms, and to the pooling scheme. Boundary statistics are reported by language class throughout.

Defensible, requires no annotation, transfers unchanged to any corpus added later.

---

## 11. Rules

- Do not touch `D-dev-confirm` or `D-test` except for reference-only counts and C00 baselines.
- Do not tune to improve a number. A bad result is a finding; the response is the pre-committed fallback.
- Log every run: command, git SHA, config hash, output path, wall clock.
- Fix code, not criteria.
- Selection of conventions, paths, thresholds, and eligibility floors is a human decision on development evidence. Diagnostics report; they do not choose.

## 12. Escalation triggers

Stop and reassess within 48 h if:

- Day 2 pseudo-labels disagree with **both** other paths by more than ~300 ms median
- Day 4 shows no effect **and** the Day 5 offset sweep is also flat
- Day 6 shows conclusions flipping across paths or conventions
- Day 3 arrives with no provisional configuration frozen

## 13. Outstanding

- **Headroom estimate** — conservative subset size under the revised eligibility rule (≥400 ms, non-final, baseline-error). Determines whether Day 4 can resolve a paired contrast against four controls. If under ~150 units, relax to ≥300 ms before running Day 4.
- **Residual 280 ms** median disagreement after best convention and utterance-final exclusion is unexplained. Not blocking; noted.