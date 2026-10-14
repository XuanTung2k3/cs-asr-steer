# P2-1b — paired CE vs KL-preservation feasibility: result

**Terminal verdict: `P2_1B_INCONCLUSIVE`.** It was assigned by the deterministic reporting script
(`reft/p2_1b_evaluate.py report`) under `P2_1B_DECISION_RULES.md`, which were frozen before training.

**Verdict path:** integrity passes → P fails (P1) → U false → D and R both true → none of SUPPRESSES / NOT_EFFECTIVE
applies → `INCONCLUSIVE`.

In the protocol's words, this is a **partial preservation success**:
- **Damage is lower under KL.** Frozen-backbone KL (λ = 1.0) cut new Mandarin errors by 40% and long switch-back
  dropped units by 45%, in every matched seed.
- **No measurable cost to the English corrections.**
- **The retention floor is still not cleared.** It is ≥ 0.99, i.e. ≤ 49 new errors. It was required in at least two of
  three KL seeds, and only one seed cleared it: 53 / 48 / 50 new errors.
- **Three of the five fixed damaged switch-back spans remain fully dropped by every system.**

**Scope.** dev150 is exposed exploratory data that also selected every checkpoint. The intervals below are descriptive.
Three seeds on the same 20 dialogues do not add independent dialogues.

---

## A. Executive verdict

**What KL fixed, relative to the matched CE run in each seed (seeds 42 / 43 / 44, each arm at its P2-1-rule
checkpoint):**
- New Mandarin errors: 73 / 86 / 93 → 53 / 48 / 50.
- Long switch-back dropped units (SB): 52 / 52 / 56 → 30 / 31 / 26.
- Mandarin damage far from a language switch (distance > 7 units): 8 / 18 / 17 → 0 / 3 / 1 errors, out of 2,732
  B0-correct units.

**What KL kept:**
- Net substantive English corrections: 37 / 34 / 47 → 34 / 41 / 35, a ratio of 0.93.
- Mean genuine-CS MER: identical (0.1110 vs 0.1110).

**What KL did not fix:**
- **The retention floor.** Mandarin retention reached 0.99 in only one KL seed. The other two missed by 1 and 4 errors.
- **The three remaining spans.** Of the 5 fixed damaged switch-back spans from P2-1D, KL fully recovers the same 2 in
  all three seeds (21 of 52 units). The same 3 spans (31 units) stay dropped in all six runs.
- **What the prefix audit can show.** It finds no gold-prefix degradation (Type A = 0 in every system), but the probe is
  confounded (§F).
  - The damaged outputs leave B0's output very early, at the first few tokens of the utterance, and the adapted models
    prefer their own damaged outputs under rescoring.
  - The remaining failure therefore lies in free decoding, which teacher-forced, gold-prefix KL does not observe.

## B. Preflight and integrity

**Preflight.**
- **First preflight (job 59254):** `P2_1B_BLOCKED_PREFLIGHT_G3_kl_objective`. The record is kept.
  - G3's initial-gradient check compared two adapters built at different global-RNG states.
  - PyTorch's trivialized Householder orthogonal parametrization draws the completion `base` of R from the global RNG.
    So the gradient on the stored parameter was the same gradient expressed in different coordinates.
  - Everything that does not depend on those coordinates was bitwise equal: dL/dR, the W and b gradients, and the loss.
    The KL gradient at initialization was 4e-8.
- **Repair, amendment r1** (`P2_1B_AMENDMENT_R1.md`, commit `8d91c26`): both adapters in the check are seeded
  identically, and the check also requires their initial state including `base` to be equal. The design is unchanged.
- **Preflight r1 (job 59264): `READY_FOR_P2_1B_EXECUTION`.**

**Preflight r1 gates:**

| Gate | Result |
|---|---|
| G1 provenance | pass |
| G2 CE equivalence | one new CE epoch at seed 42 reproduces P2-1 `R2/epoch1/adapter.pt` **bitwise** → historical CE_42 reused |
| G3 KL objective | all items pass; initial gradient max diff 5.6e-9; teacher = frozen forward; zero-init student = teacher; initial KL 0; perturbed KL 2.5e-4; KL gradients reach R, W, b |
| Mask coverage (training only) | 16,737 / 18,616 positions teacher-correct (0.899); switch-back 470 / 613 (0.77 ≥ 0.10); EOS 463 / 500; ZH 0.93, EN 0.79; unmappable 0 |
| G4 decoding | zero-init beam-5 20/20; cached greedy vs teacher-forced 0 mismatches |
| G5 resources | teacher overhead +8.3%; projected job about 25 min (< 2.5 h) |
| G6, G7 | pass |

**Pairing check (post-hoc).** `base` is a buffer the optimizer never touches.
- In each seed it is bitwise identical between the CE_s and KL_s checkpoints.
- It differs across seeds.
- So the arms shared their initial state.

**Amendment r2** (`P2_1B_AMENDMENT_R2.md`):
- Evaluate job 59271 failed while writing its output: integer seed keys alongside `"seed_mean"` cannot be sorted by the
  deterministic JSON writer.
- The fix is serialization only. Nothing was written or inspected before the fix.
- Dependent jobs had been cancelled automatically, and were rerun after the fix.

**Freeze chain:**
- `freeze.json` (`d0937f86…`) → `freeze_r1.json` (`cc0f25e6…`) → `freeze_r2.json` (`24942fab…`).
- The P2-0 chain head `9b77ecba…` is unchanged.

**Integrity in the report:**
- all 6 runs are present;
- zero-init reproduces B0 on 150/150 utterances in every run;
- backbone weights are unchanged;
- reload verification: 15/15 epoch adapters of the new runs re-decode 20/20 utterances as recorded, with matching
  hashes. The historical CE_42 was verified in P2-1.

**Jobs (all `COMPLETED` unless noted):**

| Step | Job(s) | Wall time |
|---|---|---|
| tests | 59262, 59287 | — |
| freeze r1 / r2 | 59263 / 59288 | — |
| smoke | 59254 (blocked), 59264 (ready) | 2:02 each |
| train CE_43, CE_44, KL_42, KL_43, KL_44 | 59265–59269 | 19:16–19:59 |
| select | 59270 | — |
| evaluate | 59271 (failed, r2), 59289 | 1:03 |
| verify | 59290 | 6:13 |
| prefix audit | 59273 | — |
| report | 59291 | — |

Jobs 59272 and 59274 were cancelled because they depended on 59271.

## C. Full paired results (beam-5, selected checkpoints; B0: MER 0.1199, CS MER 0.1227)

Checkpoints are selected by the P2-1 rule, applied per run.

| Seed | Epoch CE / KL | CE CS MER | KL CS MER | CE ZH retention | KL ZH retention | CE EN net | KL EN net | CE SB units | KL SB units | CE new ZH | KL new ZH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2 (P2-1) / 3 | 0.1099 | 0.1124 | 0.9853 | 0.9893 | 37 | 34 | 52 | 30 | 73 | 53 |
| 43 | 3 / 2 | 0.1127 | 0.1105 | 0.9826 | **0.9903** | 34 | 41 | 52 | 31 | 86 | **48** |
| 44 | 2 / 3 | 0.1105 | 0.1103 | 0.9812 | 0.9899 | 47 | 35 | 56 | 26 | 93 | 50 |
| mean | | 0.1110 | 0.1110 | 0.9830 | 0.9898 | 39.3 | 36.7 | 53.3 | 29.0 | 84.0 | 50.3 |

**Matched KL − CE.** Dialogue-cluster bootstrap with 2,000 resamples, using the same dialogue draws for both arms; the
seed-mean interval is pooled.

| Metric | Seed 42 | Seed 43 | Seed 44 | Seed mean [95% CI] (range) |
|---|---:|---:|---:|---|
| New ZH errors | −20 | −38 | −43 | **−33.7 [−73.0, −5.7]** (−43 … −20) |
| SB units | −22 | −21 | −30 | −24.3 [−63.7, 0.0] (−30 … −21) |
| EN net | −3 | +7 | −12 | −2.7 [−9.7, +4.3] (−12 … +7) |
| CS MER | +0.0024 | −0.0023 | −0.0002 | +0.0000 [−0.0039, +0.0036] |
| MER | +0.0027 | −0.0003 | +0.0000 | +0.0008 [−0.0014, +0.0030] |

**Epoch-matched KL − CE (secondary).** In all 9 seed × epoch pairs:
- KL has fewer new ZH errors, by 19 to 47;
- KL has fewer SB units, by 17 to 38.

**Other selected-checkpoint metrics:**

| Metric | CE seeds 42 / 43 / 44 | KL seeds 42 / 43 / 44 |
|---|---|---|
| MER | 0.1077 / 0.1098 / 0.1089 | 0.1104 / 0.1095 / 0.1089 |
| English WER | 0.180 / 0.181 / 0.171 | 0.186 / 0.182 / 0.183 |
| PIER | 0.164 / 0.166 / 0.156 | 0.168 / 0.161 / 0.166 |
| Mandarin CER | 0.0889 / 0.0909 / 0.0920 | 0.0905 / 0.0902 / 0.0891 |
| English corrections / corruptions | 62/25, 56/22, 66/19 | 54/20, 56/15, 59/24 |
| Mandarin repairs | 103 / 102 / 109 | 94 / 90 / 100 |
| Outside-target damage (of 4,971) | 76 / 89 / 97 | 56 / 51 / 52 |
| Improved / worsened utterances | 47/22, 44/22, 47/22 | 38/21, 35/16, 38/19 |
| Dialogues with corrections | 14 / 14 / 14 | 13 / 14 / 13 |
| Loops | 0 everywhere | 0 everywhere |
| Not terminated (B0: 1) | 2 / 2 / 4 | 1 / 1 / 2 |

**Against B0** (P2-1 rules, ΔCS-MER CIs all include 0):
- no run is "useful";
- only KL_43 meets the P2-1 preservation rule.

**Greedy (secondary; B0 greedy MER 0.1493, CS 0.1366):**

| | CE seeds 42 / 43 / 44 | KL seeds 42 / 43 / 44 |
|---|---|---|
| MER | 0.133 / 0.135 / 0.132 | 0.136 / 0.137 / 0.136 |
| CS MER | 0.138 / 0.140 / 0.136 | 0.141 / 0.144 / 0.142 |
| New ZH errors | 122 / 121 / 116 | 107 / 95 / 104 |
| EN net | 20 / 3 / 22 | 4 / 3 / 10 |

Under greedy decoding KL also reduces Mandarin damage, but it keeps less of the English benefit.

**Rule table:**

| Criterion | Result | Value |
|---|---|---|
| P1 (retention ≥ 0.99 in ≥ 2 KL seeds) | **fail** | only seed 43 |
| P2 (mean new ZH errors lower under KL) | pass | 50.3 < 84.0 |
| P3 (mean SB reduction ≥ 30% and KL < CE in ≥ 2 seeds) | pass | 45.4%, 3/3 seeds |
| P4 (loops / non-terminations not up by more than 1) | pass | loops 0 vs 0; non-terminations 4 vs 8 |
| D | true | |
| R1 (EN net > 0 in ≥ 2 KL seeds) | pass | 3/3 |
| R2 (KL/CE EN-net ratio ≥ 0.75) | pass | 0.932 |
| R3 (ΔCS MER ≤ 0.005) | pass | +0.0000 |
| R4 (worst dialogue ≤ CE + 10) | pass | 32/32/32 vs 30/29/30 |
| U (seed instability) | false | |

## D. Preservation analysis

**KL substantially reduces the P2-1D failure pattern, but does not remove it.**

- **Deletions and runs.** New Mandarin deletions fall from 63 / 75 / 83 to 43 / 38 / 37. Long damaged runs fall from
  7 / 7 / 8 to 5 / 4 / 4.
- **Where the remaining damage sits.** Almost all of it is near switches:
  - near-switch new errors: CE 65 / 68 / 76 vs KL 53 / 45 / 49, out of 2,218 B0-correct near units;
  - far from switches: CE 8 / 18 / 17 vs KL 0 / 3 / 1, out of 2,732.
  - KL removes almost all far damage and about 30% of the near-switch damage.
- **Damage in utterances with no English repair:** CE 51 / 83 / 69, KL 50 / 28 / 47.

**Fixed-case span recovery** (post-hoc, descriptive; the 5 damaged and 5 safe cases frozen before training):
- **CE.** CE_43 and CE_44 drop exactly the same 5 spans as P2-1's CE_42 (52 of 52 units, all deletions).
- **KL, recovered spans.** Every KL seed fully recovers the same two spans, 15 and 6 units.
- **KL, remaining spans.** Every KL seed still drops the same three spans (13, 9 and 9 units). In two seeds one 9-unit
  span becomes 5 deletions plus 4 substitutions.
- **Safe cases.** All systems leave the safe cases unchanged (1/23 units wrong in each, vs 2/23 for B0).

The KL effect on actual long-span omissions is real and seed-consistent, but partial. Some switch-back omissions are
insensitive to this KL.

## E. English lexical behavior

**Category transitions** (B0 category → system, selected checkpoints; B0 has 89 omission, 28 translation-like and 39
lexical-substitution English errors):

| Transition | CE seeds 42 / 43 / 44 | KL seeds 42 / 43 / 44 |
|---|---|---|
| omission → correct | 33 / 32 / 36 | 29 / 32 / 33 |
| translation-like → correct | 15 / 15 / 16 | 13 / 14 / 14 |
| lexical substitution → correct | 5 / 2 / 4 | 2 / 4 / 2 |
| correct → wrong | 25 / 22 / 19 | 20 / 15 / 24 |
| wrong → a different wrong category | 16 / 17 / 14 | 14 / 13 / 18 |

- **Omission and translation repairs survive KL almost intact.** They account for 77–84% of gross corrections in both
  arms.
- **Fine English-to-English lexical substitutions are not improved by either arm.** 34–37 of 39 stay lexical errors, and
  2–5 correct words become substitutions. This repeats P2-1D: the adapter fixes which language and whether a word is
  present, not lexical precision.

## F. Prefix diagnostic

**Setup.** 5 damaged and 5 safe switch-back cases, fixed before training, all mappable. The systems are B0, CE_42 at
epoch 2 (shown twice in F5 under two labels, being the same reused checkpoint), and the selected CE_s and KL_s.

**Gold prefix.**
- Every system **raises** log p of the first Mandarin token after English relative to B0:
  - damaged cases: +0.86 / +0.92 / +0.92 nats for CE, +0.80 / +0.66 / +1.01 nats for KL;
  - safe cases: +0.08 to +0.19 nats.
- No system loses the gold argmax, so there is no Type A case anywhere.
- **Confound.** In 4 of the 5 damaged cases, B0 itself gives the gold token only −5.1 to −6.4 nats and does not rank it
  first, although B0 decodes these spans correctly. The gold prefix is the normalized reference target format. It
  differs from Whisper's natural output, for example in punctuation and spacing tokens; B0's next token at the
  divergence point is often a non-letter "other" token.
- So the positive gold-prefix shifts largely reflect adaptation to the target format. **The gold-prefix probe cannot
  discriminate H1 from H3 here.**

**Generated prefix.**
- **CE runs.** The damaged outputs leave B0's output at the start of the utterance: position fraction 0.00–0.05 in 4 of
  5 cases. The remaining case diverges at 0.25 and is the one Type B case.
- **Format-token divergences.** In the two cases where B0's next token is a non-letter "other" token:
  - the CE model gives its own Latin-script token log p ≈ −0.01, and B0's token −4.2 to −5.5;
  - the KL models either keep B0's token at −1.0 to −1.2, or leave B0's output much later (fraction 0.55, Han script).
    The second of these is one of the two spans KL recovers.
- **Latin-token divergences.** Where both next tokens are Latin, both models give moderate probability to both tokens.
- **Whole-sequence rescoring** (summed and per token): the adapted models prefer their own damaged outputs in 4–5 of 5
  cases. The
  omission is the model's preference, not a beam-search accident.

**Typing:**

| System | Type A / B / C |
|---|---|
| Every CE | 0 / 1 / 4 |
| KL_42 | 0 / 1 / 4 |
| KL_43 | 0 / 0 / 5 |
| KL_44 | 0 / 1 / 4 |

The dominant type is **C (mixed / unresolved)** in every system.

**Interpretation.**
- *What the evidence is consistent with:* the remaining omissions arise along generated prefixes in free decoding, after
  an early divergence from B0, rather than through a weakened gold-prefix continuation.
- *What it does not establish:*
  - a causal mechanism;
  - acoustic involvement;
  - that gold-prefix KL cannot in principle reach these cases.

In the protocol's words: **gold-prefix KL reduced, but did not resolve, the autoregressive failure.** KL's 2/5 recovery
coincides with KL staying closer to B0 at the early divergence point.

## G. Training and efficiency

| | CE arm (seeds 42 / 43 / 44) | KL arm (seeds 42 / 43 / 44) |
|---|---|---|
| Trainable parameters / adapter size | 5,122 / 20,488 B | same |
| Updates / supervised tokens | 96 / 55,848 | same |
| Train CE per token, epochs 1 → 3 | 0.396–0.398 → 0.347 | 0.402–0.404 → 0.361–0.363 |
| Train KL per token on the mask, epochs 1 → 3 | — | 0.0075–0.0081 → 0.0155–0.0163 |
| Mean gradient norm, epochs 1 → 3 | 0.33–0.36 → 0.36–0.39 | 0.21–0.22 → 0.26–0.30 |
| Pure training time | 81 / 68 / 77 s | 77 / 85 / 85 s |
| Teacher forward, total | — | 7.7–7.9 s (about 10% of training compute) |
| Peak allocated / reserved VRAM | 5.51 / 7.29 GB | 5.73 / 8.02 GB |
| Job wall time (including 4 × dev150 beam-5 + greedy) | 19:26 / 19:59 (new) | 19:16–19:57 |
| Mean beam-5 decode time per utterance | 1.04–1.25 s | 1.02–1.07 s |

- **Decode timing.** Decode time varies with the node and contention; the B0 reference is 1.16 s per utterance.
- **Total GPU use for P2-1b** on H100 MIG 3g.40gb: about 1.85 slice-hours.
  - training: 5 × about 19.6 min, i.e. 1.64 h;
  - two smokes, verify, the prefix audit and six bounded G3 diagnostic jobs make up the rest.
- **Mask.** No micro-batch had an empty mask, and the mask covers 0.899 of positions.
- Figures F1–F4 are the frontier, switch-back counts, MER and retention, and training curves. F5 is the gold-prefix
  shift. All are under `p2_1b/figures/`: anonymized aggregates, kept local.

## H. Research implications

- **Does global KL regularization suffice?** No.
  - It is a cheap, seed-stable improvement: −40% new Mandarin errors at no measurable cost to corrections, at about 10%
    extra training compute.
  - It misses the 0.99 floor in 2 of 3 seeds, and leaves the same 3 of 5 damaged switch-back spans dropped in every
    seed.
  - It is not a reason to sweep λ, which is outside the authorized scope.
- **Is selective intervention still necessary?** Probably.
  - After KL, damage is almost entirely within 7 units of a language switch (≤ 3 far errors per seed).
  - So any further gain must come from what happens around switches.
  - This is localization, not evidence that a deployable gate can find those positions.
- **Is acoustic-conditioned ReFT still motivated?** P2-1b neither supports nor undermines it. Nothing here tested
  acoustics, and the remaining failure shows up as a decoding-path divergence, not a measured acoustic deficit.
- **Is free-decoding continuation the more important unsolved problem?** The evidence leans that way.
  - The damaged outputs diverge from B0 at the very beginning.
  - The adapted models prefer their own omitting outputs.
  - The gold-prefix probe, the only teacher-forced view, is confounded by target format.
  - The exact locus is unresolved: an early-token divergence vs a local switch-back decision under the system's own
    prefix.

## I. Next-stage recommendation (not authorized)

**Recommendation: `GO_AUTOREGRESSIVE_CONTINUATION_DIAGNOSTIC`.** This is a bounded diagnostic with no training, using
the existing B0, CE and KL checkpoints and the fixed cases.

**Measurements:**
- Replay B0's own decoded prefix (not the reference format) through each adapted model, and measure at the switch-back
  point:
  - continuation vs skip, EOS and English-continuation probabilities;
  - forced-continuation recovery.
- Locate where each system's own path first commits to the omission:
  - at the early format/script divergence, or
  - at the switch-back itself.
- Compare the 2 KL-recovered spans with the 3 never-recovered spans.

**Expected information gained.** It decides the locus of the remaining failure, which in turn decides the next
adaptation design:
- if the failure is local to the switch-back, a selective or switch-back-aware adaptation design;
- if it is induced earlier, preservation on the model's own (on-policy) prefixes rather than on gold prefixes.

It also fixes the confounded gold-prefix probe.

**Why the alternatives are less valuable now:**
- **Selectivity / switch-back design now:** it risks targeting the wrong position if the omission is committed earlier.
- **P2-2 acoustic screening:** nothing here implicates acoustics, and it adds architecture variance before the failure
  is localized.
- **Optimization or evaluation hold:** optimization, integrity and scoring are sound; the one weak instrument is the
  prefix probe, which the diagnostic replaces.
- **Stopping the ReFT line:** English corrections are seed-stable (net +34 to +47 in all 6 runs, in 13–14 dialogues),
  and KL showed the damage is malleable, so stopping is premature.

## J. Reproducibility

- **Commits** on branch `cs-asr-reft`, pushed:

  | Commit | Content |
  |---|---|
  | `2c909d2` | design, code and tests |
  | `8d91c26` | amendment r1 |
  | (this commit) | amendment r2 sources, result and STATUS |

- **Seeds.** 42, 43 and 44 serve as the init seed, global seed and epoch order (`default_rng(s + epoch)`). λ_KL = 1.0.
- **Full training runs completed:** 5 new (CE_43, CE_44, KL_42, KL_43, KL_44), plus the historical CE_42 reused after a
  bitwise equivalence check. This is within the limit of 6.
- **Tests:**
  - CPU: 72/72 pass (`sbatch slurm/reft_p2_1b_cpu.sbatch test`, job 59287).
  - GPU: smoke r1 READY (59264), verify 15/15 (59290).
- **Private outputs** under `/mnt/data/tungnx/cs-asr-reft/p2_1b/` (sha256 prefixes):

  | File | sha256 |
  |---|---|
  | `evaluation.json` | `9547c7dd…` |
  | `report.json` | `6548b825…` |
  | `verify.json` | `62df3dd5…` |
  | `prefix_audit/summary.json` | `9ea80425…` |
  | `prefix_audit/results_private.jsonl` | `004fff1c…` |
  | `prefix_audit/cases_private.json` | `806cd20d…` |
  | `preflight_59254.json` | `ff7ef425…` |
  | `preflight_59264.json` | `dd4a51c2…` |
  | `AUTHORIZATION.json` | `a4c4d1d7…` |

  Also there: `training/{CE_seed43, CE_seed44, KL_seed42, KL_seed43, KL_seed44}/` (adapters, decodes, manifests,
  selections) and `figures/`.
- **Post-hoc diagnostics.** These are descriptive, were added after the result, and do not enter the verdict. They are
  stored with their scripts under `posthoc/`:
  - G3 diagnosis: GPU jobs 59256–59261.
  - Span recovery: CPU job 59293, `9b040e2f…`.
  - Pairing `base` check: CPU job 59296, `c5aa5572…`.
- **Not executed:** no P2-2, M1, encoder adaptation, LoRA retuning, λ/seed/method sweep or locked-role access (D-test,
  D-dev-confirm, router-calib).
- **Not pushed:** no transcripts, identifiers, adapters or raw outputs.
