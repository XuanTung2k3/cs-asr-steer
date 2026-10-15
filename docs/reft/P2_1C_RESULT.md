# P2-1c — on-policy continuation and switch-back failure audit (result)

**Verdicts.**
- **Pre-registered:** **`P2_1C_ENGINEERING_BLOCKED`** (`p2_1c/analysis.json`). One T1 tolerance check failed on BF16
  rounding; see B.
- **Under the documented amendment r1:** **`P2_1C_DIAGNOSTIC_COMPLETE`** (`p2_1c/analysis_r1.json`). The case rows are
  identical in both files; only the integrity input differs.

**Research recommendation (not authorization): `GO_P2_2_ON_POLICY_PRESERVATION_DESIGN`.**

**Scope.**
- No training, no adapter change, no new data or cases.
- No P2-2 or P3 was executed.
- Ten fixed P2-1b cases on the exposed dev150 panel, with reference-informed anchors: exploratory and non-deployable.
- Protocol: `P2_1C_PROTOCOL.md`. Rules: `P2_1C_DECISION_RULES.md`. Amendment: `P2_1C_AMENDMENT_R1.md`.

## A. Executive conclusion

**The persistent switch-back omissions follow the systems' own decoder history; the switch-back itself is not lost
locally.**

**Under B0's prefix.** All six adapted checkpoints (CE and KL, seeds 42/43/44) recover the persistent spans d0, d3 and
d4 in 17 of 18 cases (the other is partial) when they continue from **B0's own generated prefix**, ending right after
the English.

**Under their own prefix.** Restarted from **their own generated prefix** at the same switch-back location, they drop
the span in 18 of 18 cases.

**Local preference under B0's prefix:**
- B0's switch-back token stays the adapted model's top-1 in 25 of 30 damaged case × system rows;
- the largest local drop is −1.36 nats;
- premature EOS is not favored.

**Mechanism flags.** H1 (prefix history) holds in 20 of the 24 case × system rows that originally drop the span.

| | H1 cells | H2 | H3 | Unresolved rows |
|---|---|---|---|---|
| Result | 7 of 8 seed-consistent cells (4 of 5 damaged cases, both arms) | 1 row | 1 cell (d2/CE, 2 rows) | 1 |

**Two forms of history dependence:**

| Form | Cases | What differs in the history | Frozen B0 at the system's prefix |
|---|---|---|---|
| **Adapter × history** | d0, d1 (lexically identical context), d3 (4/6) | d0/d1: only formatting/tokenization, i.e. the same canonical words | continues the span |
| **History itself steers away from Mandarin** | d4 (6/6), d3 (2/6) | the adapted model's **English** differs lexically. In 3 of 6 d3 rows it is *more correct* (5/5 units vs B0's 3/5) | drops too |

**What KL changed.**
- For the KL-recovered span d1, KL continues the Mandarin even from CE's own prefix (3/3), so it removed the
  adapter × history sensitivity there.
- For the persistent spans, KL did not change the own-prefix failure (9/9 drop).
- d2 is different: a near-tie at the sequence level (margins of 0.00–0.08 nats per token) that fails for CE even under
  B0's prefix.

**Interpretation.** This explains P2-1b. Its KL term was applied on gold teacher-forced prefixes, while the failure
appears only on the model's own free-decoding prefixes.

**Limitations.** Five failure cases. Reference-informed anchors. The pre-registered verdict is ENGINEERING_BLOCKED.
"Prefix history" is an operational label; this audit does not identify which earlier token causes the omission.

## B. Integrity and reproducibility

**Freeze chain and code.**
- `p2_1c/freeze.json` `cfc503f6…` → `freeze_r1.json` `d8e5f3ed…`.
- P2-1b chain head `24942fab…`; P2-0 `9b77ecba…` (verified by the CPU steps).
- The code was committed **before execution** in `6250c21`.

**Inputs and environment.**
- All 6 adapters match their recorded `adapter_sha256`.
- Backbone weights equal the pinned hash before and after.
- Canonical CUDA 12.8 libraries.
- Reference firewall on in every GPU job.

**Preflight (job 59320): `READY_FOR_P2_1C_EXECUTION`** (G1–G7 all pass). The estimate was 6.5 GPU-minutes; the full
replay took 8.7.

**Full replay (job 59321).** T1 results:

| Check | Result |
|---|---|
| Empty-prefix replay parity with the recorded original beam-5 decodes | **70/70** |
| All 700 generated beams vs their teacher-forced beam-equivalent score | ≤ 0.018 (tolerance 0.05) |
| P2-1b recorded scores | reproduced to ≤ 2e-6 |
| Positions; early positions bitwise | correct |
| Determinism | yes |
| No cross-case leakage (first case re-run identical) | yes |
| Hooks cleared; finite | yes |
| Cached-vs-full | **1 failure** (below) |

**Amendment r1.** One check failed: cached-vs-full |Δ log p| for d1 / CE42 was 0.123 nats against the declared 0.10.
- **Diagnosis** (job 59325; engineering only; no outcome opened):
  - in **float32**, cached and full forwards agree to ≤ 2 × 10⁻⁵ nats for all 7 systems;
  - in BF16, the differences reach about 0.12 nats only on low-probability tokens (log p ≈ −5).
- **Conclusion.** BF16 rounding, not a caching or position defect.
- **Repair.** Amendment r1, frozen before any outcome was read, re-specified the semantic check in float32 for all
  10 cases × 7 systems (job 59328). The maximum was **1.96 × 10⁻⁵** (tolerance 1e-3); all pass. The BF16 maximum of
  0.123 is kept as descriptive.
- `analysis.json` (BLOCKED, job 59322) and every replay output are unchanged (hashes recorded in `freeze_r1.json`).
  `analysis_r1.json` (job 59329) re-reads the same outputs.

**Tests.**
- 84/84 CPU tests: 72 earlier + 12 for P2-1c, including the r1 regression test.
- The canonical text is reproduced from the output tokens for all 70 original decodes.

**Pre-freeze engineering dry run.**
- Jobs 59314–59318: anchors, one GPU case and the analysis code, in scratch.
- It found that the switch-back anchor had to be the first span unit, as the protocol table specified, not the P2-1b
  `sb_unit` (B0-wrong in d2 and d3).
- It also found that `early_zh` needed a clean in-span boundary.
- Both were fixed before the freeze. The dry-run outputs were deleted without any outcome field being read.

## C. Anchor validity

| Case | Group | Span units | B0 span correct | `pre_en` | `post_en` | `early_zh` | Own anchors valid |
|---|---|---|---|---|---|---|---|
| d0 | persistent | 13 | 13/13 | valid (= start; English opens the utterance) | valid | valid | 6/6 (lexically matched to B0) |
| d1 | KL-recovered | 15 | 15/15 | valid (= start) | valid | valid | 6/6 (CE lexically matched) |
| d2 | KL-recovered | 6 | 6/6 | valid | valid | valid | 6/6 (KL43 token-identical to B0 → comparison void) |
| d3 | persistent | 9 | 9/9 | valid | valid | valid | 6/6 (lexically different) |
| d4 | persistent | 9 | 9/9 | valid | valid | valid | 6/6 (lexically different) |
| s0 | safe | 3 | 3/3 | invalid (B0 deletes the first EN unit) | valid | none (multi-character token) | 6/6 |
| s1 | safe | 5 | 5/5 | valid | valid | valid | 6/6 |
| s2 | safe | 5 | 3/5 | invalid | invalid (B0 substitutes the first span unit) | invalid | 2/6 |
| s3, s4 | safe | 5 | 5/5 | valid | valid | valid | 6/6 |

- **Damaged cases:** all 5 have exact, valid generated-prefix anchors at every tested location.
- **Safe controls:** s2 is unmappable for precise claims; it is kept in the inventory and excluded from precise-anchor
  claims.
- No case was added or replaced.

## D. T1–T3: divergences and B0-prefix versus own-prefix continuation

### Divergence from B0 (30 damaged case × system rows)

- **First token divergence before the switch-back:** 26/30. In 20 of the 30 rows, the first divergence is format or
  tokenization only (it precedes the first lexical difference).
- **First substantive lexical divergence before the switch-back:** 17/30.
  - d0 and CE on d1: the first *lexical* difference is the omission itself (unit 0 relative to the anchor), after an
    earlier format divergence.
  - d3 and d4: the English region differs lexically.
  - d2: divergence 28 units earlier.

Figure F1: `p2_1c/figures_r1/F1_divergence_timeline.png`.

### Table B — span status (rec = recovered ≥ 0.8; drop ≤ 0.2; part = partial)

| Case | Systems | Original | From B0 prefix (T2) | From own prefix (T3) | B0 model at system's prefix | Counterpart at system's prefix | Label |
|---|---|---|---|---|---|---|---|
| d0 | CE42/43/44 | drop ×3 | rec ×3 | drop ×3 | rec ×3 | drop ×3 | H1 ×3 (adapter × history; matched context) |
| d0 | KL42/43/44 | drop ×3 | rec ×3 | drop ×3 | rec ×3 | drop ×3 | H1 ×3 (same) |
| d1 | CE42/43/44 | drop ×3 | rec ×3 | drop ×3 | rec ×3 | **KL: rec ×3** | H1 ×3 (adapter × history; matched context) |
| d1 | KL42/43/44 | rec ×3 | rec ×3 | rec ×3 | rec ×3 | rec ×3 | no original omission |
| d2 | CE42/43/44 | drop ×3 | **drop ×3** | drop ×3 | rec ×3 | drop ×3 | CE42 H2; CE43, CE44 H3 |
| d2 | KL42/43/44 | rec ×3 | rec, rec, **drop** (KL44) | rec ×3 | rec ×3 | rec, drop, rec | no original omission |
| d3 | CE42/43/44 | drop ×3 | rec ×3 | drop ×3 | rec, **drop** (CE43), rec | drop ×3 | H1 ×3 |
| d3 | KL42/43/44 | drop ×3 | rec ×3 | drop ×3 | rec, rec, **drop** (KL44) | drop ×3 | H1 ×3 |
| d4 | CE42/43/44 | drop ×3 | rec, **part** (CE43), rec | drop ×3 | **drop ×3** | drop ×3 | H1 ×2; CE43 unresolved |
| d4 | KL42/43/44 | drop ×3 | rec ×3 | drop ×3 | **drop ×3** | drop ×3 | H1 ×3 |

**Denominators:**
- 30 damaged rows; 24 originally drop the span.
- H1 20, H2 1, H3 2, unresolved 1.
- Unmappable damaged cases: 0.

**Safe controls** (s0–s4): every system retains the span in every condition. The only partial results are on s2 (9 of
46 continuations), whose B0 span is itself 3/5.

### Next-token behavior at the system's own prefix

This is descriptive. For d3 and d4 the contexts differ lexically, so it is a different-context comparison.

| Case | B0 switch-back token, under S | Same token, under B0 at S's prefix | Top-1 |
|---|---|---|---|
| d0 | rank 5–15 (log p −3.9 to −5.4); top-1 Latin | rank 6, yet B0's beam continues the span | — |
| d1 (CE) | rank 13–30; EOS log p −1.0 to −2.1 | rank 40–45, yet B0 continues | — |
| d3 | rank 123–533 | rank 270–1734 | Latin |
| d4 | rank 436–2308 | rank 219–838 | Latin; B0 also drops |

The own-prefix drops frequently end the utterance early (premature EOS in 25/60 own-prefix continuations by adapted
models) or loop (d1 CE43, d3).

## E. T4: forced-prefix rescue (only the newly generated suffix is scored)

| Case | Rescue boundary (earliest tested anchor with the span recovered) | Note |
|---|---|---|
| d0 | `post_en` 6/6 | `pre_en` = original (English opens the utterance) |
| d1 (CE) | `post_en` 3/3 | |
| d2 (CE) | **`early_zh` 3/3** | only once the first Mandarin units are supplied |
| d3 | `post_en` 6/6 | From B0's `pre_en`, the adapted models regenerate their own English (5/5 correct in 5 of 6 rows vs B0's 3/5), then drop the span |
| d4 | **`pre_en` 5/6**; CE43 at `early_zh` | From B0's pre-English history the adapted models generate B0-quality English (6/9) and keep the span |

**Other results:**
- No `early_zh` continuation introduces a new error (0 new errors in every row).
- `post_en` continuations add new errors only in d2/CE (6 = the span) and d4/CE43 (4).
- These are rescue boundaries under forced prefixes, not original commitment points.
- Figure F2: `p2_1c/figures_r1/F2_forced_prefix_recovery.png`.

## F. T5: sequence-level preference (B0 `post_en` pool; same prefix, no splicing)

### Table C — at the B0 post-English prefix

Δ is log p_S − log p_B0 of B0's switch-back token.

| Case | B0: log p (rank) | CE Δ (rank) | KL Δ (rank) | EOS log p | Pool: retain / omit | Omission preferred (beam-equivalent) | Top beam keeps span |
|---|---|---|---|---|---|---|---|
| d0 | −1.12 (2) | −0.99 / −1.32 / −1.36 (1, 2, 2) | −0.46 / −0.12 / −0.81 (1) | ≈ −5 | 10 / 1 | no (retain by +0.04…+0.16) | 6/6 |
| d1 | −0.01 (1) | −0.01…−0.02 (1) | ≈ 0 (1) | ≈ −12 | 13 / 0 | undefined | 6/6 |
| d2 | −0.51 (1) | −0.47 / −0.25 / −0.24 (2, 1, 1) | −0.08 / −0.06 / −0.13 (1) | ≈ −8 | 2 / 2 | **yes, 6/6 adapted** (CE −0.04…−0.08; KL −0.00…−0.02); B0 retain +0.03 | CE 0/3; KL 2/3 |
| d3 | −0.01 (1) | +0.01 (1) | +0.01 (1) | ≈ −13 | 17 / 0 | undefined | 6/6 |
| d4 | −1.03 (1) | −0.10 / −0.87 / −0.35 (1, 3, 2) | +0.16 / +0.12 / +0.14 (1) | ≈ −10 | 10 / 0 | undefined | 5/6 (+1 partial) |

**Under B0's prefix, neither local nor sequence-level omission preference occurs** on the persistent spans:
- every recovered row keeps B0's continuation as its top beam;
- no omission candidate even arises in d3 or d4.

**d2 is the only sequence-level case.** It is a near-tie: omission paths outscore the span-retaining path by only
0.00–0.08 nats per token under every adapted model.

**KL versus CE.**
- KL moved the local switch-back preferences toward B0: smaller drops on d0 and d2, and positive on d4.
- It moved d2's sequence margins to about 0.

**Safe controls.** Every adapted model raises the switch-back token's probability (Δ +0.02 to +0.26 on the four
mappable controls).

Figure F3: `p2_1c/figures_r1/F3_local_vs_sequence.png`.

## G. T6: recovered (d1, d2) versus persistent (d0, d3, d4)

These are descriptive only: five spans, no classifier or threshold.

**Persistent spans (d0, d3, d4).**
- **Seed-consistent in both arms:** recovery under B0's prefix (17/18) and failure under the system's own prefix
  (18/18).
- **KL does not change the own-prefix failure.**
- **Per case:**
  - d0: the context is lexically identical to B0's, differing only in formatting/tokenization of the English. B0
    continues from it; the adapted models of both arms do not.
  - d3 and d4: the adapted English differs lexically. In d3 it is more correct in 3 of 6 rows.
  - With d4's adapted prefixes, frozen B0 drops as well (6/6).
- **Counterexample:** d4/CE43 is partial under B0's prefix.

**Recovered span d1.**
- The CE failure is the same adapter × history pattern as d0, on a lexically matched context.
- **KL continues the span from CE's own prefix (3/3).**
- So KL removed the history sensitivity for this span.

**Recovered span d2.**
- It is fragile rather than repaired: a sequence-level near-tie, with a lexical divergence 28 units before the switch.
- CE fails even under B0's prefix and is rescued only at `early_zh`.
- KL44 also drops from B0's prefix, while its own path keeps the span.

**What separates the two groups:**
- whether the switch-back context reached on the model's own path still licenses the Mandarin continuation for that
  model;
- not whether the local switch-back token is preferred under a B0-like context. It is, in all groups.

Figure F4: `p2_1c/figures_r1/F4_recovered_vs_persistent.png`.

## H. Mechanism assessment

### H1 — prefix history: supported, dominant

- **Extent:** 20 of 24 omission rows; 7 of 8 cells; 4 of 5 cases; both arms; ≥ 2 seeds in every supporting cell.
- **Counterexamples:** d4/CE43 (unresolved) and all of d2.
- **Caveats:**
  - The label means only that **B0's actual tokens rescue** the span while the system's own tokens do not.
  - For d0/d1 the operative difference is token-level form, not words.
  - For d3/d4 it includes the adapter's own English, partly its *corrections* (d3). This couples English correction
    to the later omission in the rows where it is more correct.
  - Not established: which earlier token is responsible, or an exact commitment point.

### H2 — local switch-back: weakly supported

- Under B0's prefix, the switch-back token stays top-1 in 25/30 rows.
- Only d2/CE42 fits H2.
- Local disfavor without a failed continuation (d0 CE43/44, d4 CE44) does **not** stop the span from being produced.

### H3 — sequence/path: supported only for d2

- d2 is a near-tie in CE (2 rows) and is fragile in KL.
- It is not a general mechanism here.

### H4 — mixed or unresolved

- d2 is mixed (H2/H3).
- d4/CE43 is unresolved.
- Everything else is classified.

### Out of scope

- This says nothing about acoustics: the encoder was unchanged, and recovering under B0's prefix does not show
  acoustic grounding.

## I. Research decision

**Recommendation: `GO_P2_2_ON_POLICY_PRESERVATION_DESIGN`.** This follows from the frozen rule (H1 dominant under
DIAGNOSTIC_COMPLETE) and from the evidence. It recommends designing the next stage; it does not authorize it.

**Proposed bounded P2-2 design** (for separate approval):

- **Arms.** R2 LoReFT, the same budget, 3 seeds, and fixed weights (no sweep):

  | Arm | Description |
  |---|---|
  | CE | control |
  | CE + gold-prefix KL | the P2-1b arm |
  | **CE + on-policy preservation** | below |

- **On-policy preservation:**
  - Training prefixes are the adapter's own decoded hypotheses, refreshed per epoch on training data only.
  - At every on-policy continuation position, the student is matched to the frozen teacher on that same own prefix.
  - Where the teacher itself fails on the own prefix (the d4 pattern), the reference continuation aligned after the
    own prefix is used instead.
- **Pre-declared outcomes:**
  - own-prefix span retention on these fixed cases;
  - the P2-1b metrics: new Mandarin errors, SB units, English net and CS MER.

**Why the alternatives are less informative:**

| Alternative | Why not |
|---|---|
| Switch-back-aware adaptation | Under a B0-like context the switch-back is already handled (top-1 in 25/30). A gate keyed to the switch-back would act where the failure is not. |
| Sequence-level stability | Only d2, a near-tie. |
| Acoustic architecture screening | Nothing here implicates missing acoustic conditioning: the same adapters produce every persistent span once given B0's history. It stays an option after on-policy preservation. |
| Hold | The replay is valid (parity 70/70; float32 agreement 2e-5). |
| Stop | English corrections are seed-stable (P2-1/P2-1b), and every omission is recoverable under a valid prefix. |

## J. Resources and artifacts

**GPU time:** 12.6 MIG-minutes allocated in total.

| Job | Purpose | Time |
|---|---|---|
| 59315 | dry run | 0.9 min |
| 59320 | smoke | 1.1 min |
| 59321 | replay | 8.7 min |
| 59323, 59324 | failed diagnosis scripts | 0.4 min |
| 59325 | diagnosis | 0.4 min |
| 59328 | float32 T1 | 1.2 min |

**CPU time:** 59313, 59314, 59316–59319, 59322, 59327, 59329 and 59330 (about 2 min).

**Model forwards:**
- Replay: 589 beam-5 `generate` calls (21,252 beam steps) and 2,708 teacher-forced forwards (453 next-token rows, 453
  continuations, 266 candidates × 7 models).
- Smoke: 64 `generate` calls and 250 teacher-forced forwards.
- Float32 check: 70 cached-vs-full comparisons.

**Tests:** 84/84 CPU tests.

**Commits:** `6250c21` (protocol, code and tests, before execution), then this result with amendment r1. Both pushed
to `cs-asr-reft`.

**Hashes (sha256, prefix):**

| Group | Files |
|---|---|
| Freeze | `freeze.json` `cfc503f6`, `freeze_r1.json` `d8e5f3ed` |
| Anchors and preflight | `anchors_private.json` `78a3d545`, `anchors_inference.json` `90bf8094`, `preflight_59320.json` `dc3595f0` |
| Replay outputs | `integrity.json` `797d542d`, `replay_private.jsonl` `cee5b1e4`, `continuation_private.jsonl` `43004baf`, `candidate_scores_private.jsonl` `505aa696` |
| Analysis | `t1_cached_fp32_r1.json` `1961310d`, `analysis.json` `7bd00368`, `analysis_r1.json` `61b7769a`, `case_analysis_private_r1.json` `07acdff0`, `manifest_r1.json` `550f59e3` |
| Post-hoc | `posthoc/cvf_diag_59325.json` `e37aa48d`, `posthoc/describe_59330.json` `726ebd1f` (descriptive, outside the verdict) |

**Artifact paths:** everything is under `/mnt/data/tungnx/cs-asr-reft/p2_1c/` (private). The figures are in
`figures/` (pre-registered run) and `figures_r1/`.

**Confirmations:**
- No training or optimizer step.
- No adapter modification.
- No new dataset or case.
- No locked role opened.
- No P2-2 or P3 executed.
- Historical P2-0, P2-1, P2-1D and P2-1b freezes and artifacts are unchanged; the chains verify.
