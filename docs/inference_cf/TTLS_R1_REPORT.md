# TTLS-R1 — test-time learned steering vs decoder-LN TTA (initial report, exploratory)

**Exploratory label (frozen rule): `TTLS_R1_MIXED`.** It is triggered by T1 (TTLS-CE) alone.

**Substantive reading:** test-time learned steering does **not** correct genuine embedded-English lexical errors on this
panel.
- T1's three qualifying "genuine-substitution" corrections are all English **word-boundary repairs** (`peoplethey` →
  `people they`, `withlike` → `with like`, `somerelationship` → `some relationship`). The canonical evaluator classes
  these as same-language substitutions.
- No TTLS arm corrected a Mandarin-for-English confusion except through the acoustic candidate. On that single row
  (`烧烤` → `cooking`), the no-update candidate substitution does exactly the same.

The robust findings are about **preservation**, not correction:
- TTLS-CE introduces **0** new Mandarin errors (A2: 26).
- Adding KL preservation to A2 cuts its new Mandarin errors from **26 to 3** while keeping 30/31 POI corrections.

This is exposed development data (the fixed-100 panel), so no confirmation or significance claim is made.

| Item | Value |
|---|---|
| Branch | `feature/ttls-r1` (from `c69038ad`) |
| Freeze | `770ee86e`: spec, config, module, runner, evaluator, sbatch, 14 tests |
| Plan | `13f955f2` |
| Manifest | `8a048c1f` (r2; attempt 1 superseded because of an unsatisfiable HEAD check, see §8) |
| Job | Slurm **58385**, one job, COMPLETED, 8 min 50 s (runtime 508 s), MIG 3g.40gb |
| Output seal | `42e6848f`, pushed before any reference was opened |
| Evaluation | `results/inference_cf/ttls_r1/evaluation.json`; 15/15 integrity checks |

## 1. What was run

- **Population:** the fixed-100 D-dev-select panel (20 dialogues × 5), already exposed.
- **Subgroups:** the A3 24-panel, D (12 rows where AUTO ≠ forced) and A (88), all reported separately.
- **Model:** Whisper-large-v3, BF16, greedy decoding, forced-ZH final decode.
- **Steps:** every adaptation arm takes 2 AdamW steps with a fresh variable and an exact per-utterance reset.

| Arm | Variable (trainable scalars) | Objective | Mask |
|---|---|---|---|
| B0 / B1 | — | forced-ZH / AUTO (reused, reproduced in-job 100/100) | — |
| B2 | LN (248,320) | historical A2 AUTO pseudo-CE (reused; archived masters decoded in-job, 100/100 identical) | — |
| T1 / T2 | TTLS z ∈ R^1280 (1,280) | CE / CE + P | all steps t ≥ 1 |
| T2A | LN | CE + P (added so P is not a TTLS-only advantage) | — |
| T3 / T5 | LN | AC / AC + P | — |
| T4 / T6 | TTLS | AC / AC + P | {t\*} |
| CD | — | DIR-SPRINT0 D3 rule at every step (no update) | — |
| ACSUB | — | accepted candidate substituted at t\*, then greedy (no update) | — |

**TTLS.**
- **Edit:** `r~_t = NormPreserve(r_t + g_t z)` at the L16 DG-02 post-cross-attention / pre-FFN residual, through the
  canonical `DecoderPostCrossAttnInterventionHook` + `apply_steering`.
- **Optimizer:** AdamW with lr = E\*/(2√1280).
- **Budget:** ‖z‖ ≤ E\* = 1.126, the frozen L16 chord (about 15% of ‖r‖). It was never binding: 0 projections, median
  final ‖z‖ ≈ 1.05.

**AC candidate rule** (frozen, reference-free, computed before adaptation):
- **Proposal:** on the B0 path, where the same-prefix `<|en|>` branch's argmax is an embedded-Latin token and the B0 token
  is not.
- **Acceptance:** prompt-robust clean-versus-null evidence, E^cB ≥ log 10 **and** E^cE ≥ log 10.
- **Target:** the first accepted step only.

**P (preservation):** KL(p_θ0 ‖ p_θ) on stable B0 positions (p_θ0(b_t) ≥ 0.5 and not a language-disagreement step),
with λ = 1.

### Deviation from the proposed localization (design time, reference-free, recorded before the run)

Forced-ZH and AUTO differ on only 12/100 rows, and on **12/12** the first differing token is at content step 0. The cause
is either a leading-space artifact (`The` vs ` The`) or a whole-utterance translation, including Malay on U2004. Step 0
is a DG-02 forced-prefix position that is never edited, so the "first AUTO/forced disagreement region" is empty or
uneditable everywhere. CE arms therefore use the all-steps mask, which is the closest analogue of A2's shared
parameters. AC arms are localized by the candidate rule.

## 2. Primary results (free decoding, 100 utterances)

| System | PIER | MER | EN-WER | ZH-CER | POI err | S / D / I |
|---|---:|---:|---:|---:|---:|---|
| B0 forced-ZH | 0.4957 | 0.2570 | 0.5029 | 0.2193 | 345 | 339 / 1026 / 118 |
| **B1 AUTO** | **0.4080** | 0.2591 | **0.4152** | 0.2336 | **284** | 324 / 1053 / 118 |
| B2 A2-CE | 0.4540 | 0.2537 | 0.4626 | 0.2211 | 316 | 351 / 985 / 128 |
| T1 TTLS-CE | 0.4842 | 0.2526 | 0.4928 | **0.2157** | 337 | 333 / 1007 / 118 |
| T2 TTLS-CE+P | 0.4871 | 0.2530 | 0.4957 | **0.2157** | 339 | 334 / 1008 / 118 |
| T2A A2-CE+P | 0.4555 | **0.2495** | 0.4641 | 0.2161 | 317 | 331 / 983 / 126 |
| T3 A2-AC | 0.4899 | 0.2611 | 0.4971 | 0.2248 | 341 | 333 / 1057 / 117 |
| T4 TTLS-AC | 0.4914 | 0.2606 | 0.4986 | 0.2240 | 342 | 334 / 1053 / 117 |
| T5 A2-AC+P | 0.4914 | 0.2618 | 0.4986 | 0.2254 | 342 | 334 / 1060 / 117 |
| T6 TTLS-AC+P | 0.4914 | 0.2606 | 0.4986 | 0.2240 | 342 | 334 / 1053 / 117 |
| CD (no update) | 0.4828 | 0.2746 | 0.4856 | 0.2419 | 336 | 312 / 1252 / 21 |
| ACSUB (no update) | 0.4914 | 0.2606 | 0.4986 | 0.2240 | 342 | 334 / 1053 / 117 |

B0, B1 and B2 reproduce TTA1 exactly.

### Transitions against B0

POI transitions:

| System | Corrections | Genuine-substitution | Deletion-type | In EOS-recovery rows | Corruptions |
|---|---:|---:|---:|---:|---:|
| B1 | 63 | 39 | 24 | 0 | 2 |
| B2 | 31 | 17 | 14 | 12 | 2 |
| **T1** | 8 | 3 (all word-boundary) | 5 | 2 | **0** |
| T2 | 6 | 2 | 4 | 2 | 0 |
| T2A | 30 | 17 | 13 | 12 | 2 |
| T3 | 4 | 1 | 3 | 0 | 0 |
| T4 / T6 / ACSUB | 3 | 1 | 2 | 0 | 0 |
| T5 | 3 | 1 | 2 | 0 | 0 |
| CD | 58 | 38 | 20 | 0 | **49** |

Mandarin and sequence events:

| System | New ZH errors | ZH repairs | Changed rows | EOS-recovery rows | Active-edit harm | Safe (TTA1 bounds) |
|---|---:|---:|---:|---:|---|---|
| B1 | 72 | 0 | 12 | 0 | 5/12 | yes |
| B2 | 26 | 26 | 14 | 4 | 3/14 | yes |
| **T1** | **0** | 17 | 11 | 4 | **0/11** | yes |
| T2 | 0 | 17 | 10 | 4 | 0/10 | yes |
| T2A | **3** | 26 | 11 | 4 | 1/11 | yes |
| T3 | 30 | 1 | 6 | 0 | 2/6 | yes |
| T4 / T6 / ACSUB | 25 | 0 | 5 / 5 / 6 | 0 | 2/5 | yes |
| T5 | 32 | 0 | 6 | 0 | 2/6 | yes |
| CD | 254 | 46 | 84 | 0 | 46/84 | **no** |

Further notes on these tables:
- **Premature-EOS rows and new severe truncations** are 0 for every system except CD, which has 1 severe truncation.
- **Prefix preservation** before the editable region holds for TTLS on 100/100 rows (CE) and 6/6 (AC). TTLS outside-POI
  harm is 0.000 (CE) and 0.006 (AC).
- **Matrix-ZH retention:** T1/T2 1.000, T2A 0.999, B2 0.994, T4 0.994, B1 0.982, CD 0.937.

### Dialogue-cluster bootstrap (2,000 draws; descriptive)

| Contrast | ΔPIER | ΔMER | ΔZH-CER |
|---|---|---|---|
| T1 − B0 | −0.012 [−0.023, −0.002] | −0.004 [−0.008, −0.001] | −0.004 [−0.007, −0.0003] |
| T1 − B1 (AUTO) | **+0.076 [+0.008, +0.153]** | −0.006 [−0.017, +0.005] | −0.018 [−0.037, −0.004] |
| T1 − B2 | +0.030 [−0.010, +0.082] | −0.001 [−0.012, +0.008] | −0.005 [−0.017, +0.002] |
| T2A − B2 (effect of P on A2) | +0.001 [0, +0.004] | −0.004 [−0.013, 0] | −0.005 [−0.016, 0] |
| T2 − T1 (effect of P on TTLS) | +0.003 [0, +0.010] | +0.0003 [0, +0.001] | 0 |
| T4 − T3 | +0.001 [0, +0.005] | −0.0005 [−0.003, +0.001] | −0.001 [−0.003, +0.001] |
| T4 − ACSUB | 0 | 0 | 0 |
| B2 − B1 | +0.046 [−0.026, +0.126] | −0.005 [−0.020, +0.012] | −0.012 [−0.025, −0.002] |

### Subgroups (descriptive; the A3 24-panel is enriched and is not a population estimate)

- **D12** (AUTO ≠ forced):
  - B2 buys 17 genuine corrections with 23 new ZH errors (ZH-CER 0.150 → 0.417), because it follows AUTO's translation
    from step 0.
  - T2A keeps the 17 corrections with 3 new ZH errors (ZH-CER 0.233).
  - T1 changes 4 rows: 3 corrections, all boundary/deletion type, and 0 ZH damage (ZH-CER 0.150).
  - AC arms abstain on all 12 rows.
- **A88** (AUTO = forced): every CE gain is EOS continuation or boundary repair.

  | System | Corrections | ZH repairs | New ZH errors |
  |---|---:|---:|---:|
  | B2 | 13 | 26 | 3 |
  | T2A | 12 | 26 | 0 |
  | T1 | 5 | 17 | 0 |

- **A3 24-panel:**

  | System | PIER | MER | ZH-CER |
  |---|---:|---:|---:|
  | B0 | 0.586 | 0.286 | 0.185 |
  | B1 | 0.341 | 0.298 | 0.279 |
  | B2 | 0.514 | 0.292 | 0.215 |
  | T2A | 0.514 | 0.270 | 0.186 |
  | T1 | 0.566 | 0.273 | 0.173 |
  | T4 | 0.582 | 0.284 | 0.184 |

  On the A3 panel, T2A is the only arm that keeps A2's POI gain (128 errors) without A2's Mandarin damage.

## 3. Candidates, optimization and compute

### AC candidates (frozen rule)

**Coverage.**
- 98/100 rows have structural steps (3,570 steps).
- Only **12 proposals in 11 rows** (all in A rows).
- 6 accepted: **94% row abstention**; acceptance 50% of proposals.
- Median support: accepted 7.4 nats, rejected 0.4.
- The `<|en|>` branch almost always continues in Mandarin once the prefix is Mandarin, so proposals are rare.

**Correctness.** The accepted token is a prefix of an English word in the reference on 6/6 rows. This is a weak check,
because ` I`, `c` and ` or` are short. The outcomes per row:

| Row | Substitution | Outcome (TTLS = ACSUB) |
|---|---|---|
| U1034_S0_53 | `烧烤` → `cooking` | genuine correction |
| U2011_S0_99 | inserts `of course` | 2 deletion-type corrections |
| U0029_S0_204 | `!` → ` Twilight` | B0 already had the word; no change |
| U0012_S0_103 | `,` → ` I` | punctuation only |
| U0086_S0_183 | `这个` → `system` | +2 errors (A2-AC +7) |
| **U1003_S0_63** | `,` → ` or` | the decode closes the sentence early: **+23 mixed errors** (all 25 new ZH errors of T4) |

### Optimization (means over active rows)

| Arm | L0 → L1 → L2 | Rows with loss ↓ | L16 site displacement (rel., edited) | KL to θ0 (all / stable) |
|---|---|---|---|---|
| T1 | 0.414 → 0.343 → 0.294 | 100/100 | 0.137 | 0.034 / 0.020 |
| T2 | 0.414 → 0.349 → 0.309 (P 0 → 0.006 → 0.014) | 100/100 | 0.131 | 0.030 / 0.014 |
| B2 (archived A2) | 0.413 → … → (TTA1) | — | 0.098 (global) | 0.093 / 0.051 |
| T2A | 0.413 → 0.272 → 0.214 (P 0 → 0.022 → 0.028) | 100/100 | 0.085 (global) | 0.074 / 0.028 |
| T3 | 1.874 → 0.473 → 0.207 | 6/6, target rank-1 6/6 | 0.090 (global) | 0.145 / 0.023 |
| T4 | 1.848 → 1.114 → 0.678 | 6/6, target rank-1 5/6 | 0.143 at t\* only (0.008 overall) | 0.034 / 0.0001 |

As expected, P is exactly 0 at step 0, so it acts only through the second update.

### Compute (job total 508 s)

| Arm | Adapt (s) | Decode (s) | Backward passes | Trainable scalars | Peak GB |
|---|---:|---:|---:|---:|---:|
| T1 | 9.5 | 54.1 | 200 | 1,280 | 3.79 |
| T2 | 10.3 | 54.4 | 200 | 1,280 | 3.87 |
| T2A | 19.6 | 54.0 | 200 | 248,320 | 4.33 |
| T3 / T5 | 1.1 / 1.1 | 2.1 / 2.0 | 12 / 12 | 248,320 | 3.75 / 3.81 |
| T4 / T6 | 0.5 / 0.5 | 2.1 / 2.1 | 12 / 12 | 1,280 | 3.47 / 3.53 |

Other phases:

| Phase | Time (s) |
|---|---:|
| Baseline B0 + B1 replay decodes | 107 |
| B2 replay | 54 |
| Frozen AC branches | 13 |
| CD decodes (314 edits) | 99 |

TTLS adaptation costs about half of LN adaptation per step. Decoding dominates both.

## 4. Answers

1. **Does TTLS correct genuine embedded-English lexical errors?**
   - **No, not with CE.** T1 makes 8 corrections and 0 corruptions, but the corrections are word-boundary repairs (3),
     deletions (3) and EOS continuations (2). It makes **0** Mandarin-for-English corrections.
   - **With AC, only through the candidate.** TTLS-AC flips the candidate on 5/6 accepted rows and yields exactly the
     no-update substitution outcome (T4 − ACSUB = 0 on every metric): 1 genuine correction (`烧烤` → `cooking`).
   - Part of TTLS's safety on D rows is structural: step 0 cannot be edited, so TTLS cannot follow AUTO's
     whole-utterance language switch.
2. **Does TTLS preserve Mandarin better than A2 under the same objective?**
   - **Under CE, yes:** 0 vs 26 new ZH errors and harm rate 0/11 vs 3/14. It also delivers far less benefit: 8 vs 31
     POI corrections, PIER +0.030 [−0.010, +0.082] worse than A2.
   - **Under CE+P, the gap almost vanishes:** 0 vs 3 new ZH errors, with T2A keeping 30 corrections.
   - **Under AC, roughly equal:** TTLS 25 vs A2 30 new ZH errors, dominated by one sequence-level failure
     (U1003_S0_63).
   - Representation-level locality reduces damage by doing less, not by a better trade-off at matched benefit.
3. **Is preservation regularization useful?**
   - **For A2, clearly yes**, the strongest finding of the round. B2 → T2A: new ZH errors 26 → 3, corrections 31 → 30,
     MER 0.2537 → 0.2495 (best of all systems), ZH-CER 0.2211 → 0.2161. On D12, ZH-CER goes 0.417 → 0.233 with the same
     17 genuine corrections.
   - **For TTLS, no:** it is redundant (T2 ≈ T1, with 2 fewer corrections), because the edit already barely moves the
     stable positions (KL 0.02).
   - **For AC, no effect:** the damage is sequence-level, which teacher-forced KL on the B0 path does not see. This
     matches the A4 finding.
   - P acts only through step 2, so its A2 effect comes from a single regularized update.
4. **Does acoustic supervision help either adaptation method?**
   - **No.** The frozen prompt-robust rule is very conservative (94% abstention). Its few accepted candidates mostly
     point at real English words, but substituting them can derail the continuation (+23 errors on one row).
   - Net, AC is worse than CE for both variables (MER +0.004 vs B0), and adaptation adds nothing beyond direct
     substitution (T4 = ACSUB).
   - Full-sequence acoustic contrastive decoding (CD) finds more English (38 genuine corrections) but is unsafe:
     49 corruptions, 254 new ZH errors, harm on 46/84 changed rows.
5. **Does TTLS improve on ordinary AUTO?**
   - **No on English:** T1 − B1 PIER +0.076 [+0.008, +0.153].
   - **Yes on Mandarin:** ZH-CER −0.018 [−0.037, −0.004].
   - **No reliable MER difference:** −0.006 [−0.017, +0.005].
   - No practical method here beats AUTO on PIER; the best is T2A at +0.047.
6. **Does the evidence justify further work on TTLS?**
   - **Not as a lexical-correction method in this form:** the supervision, not the variable, is the bottleneck. AUTO-CE
     carries a lexical signal on only 12/100 rows, mostly as translation from step 0. The acoustic rule is too sparse and
     its substitutions are sequence-fragile.
   - **What the data do support:**
     - (a) TTLS is a near-zero-damage actuator: 0 new ZH errors and 0 corruptions over 100 adapted utterances, at half
       the adaptation cost.
     - (b) Preservation-regularized A2 (T2A) is the best practical arm. It merits a separately frozen exposure-safe
       confirmation before anything new on TTLS.
   - Any TTLS follow-up should first solve candidate supervision; changing the adaptation variable will not fix it.

## 5. Limitations

- **Exposed panel and small n.** The panel is already exposed, and n is small: 12 D rows and 6 accepted candidates.
- **Bootstrap.** The bootstrap is descriptive only.
- **The MIXED label.** It rests on evaluator-categorized word-boundary repairs. Read strictly as Mandarin-for-English
  correction, the outcome is NOT_SUPPORTED. Per policy, the label is reported as frozen and not amended.
- **Localization deviation.** The AUTO-region mask was not used (§1). TTLS-CE is therefore a global edit at L16, and its
  locality claims apply to AC only.
- **Single first-candidate AC.** One τ, one layer and 2 steps were used, with no sweeps by design. P's two-step weakness
  is inherent to KL-to-θ0.
- **Approximate realized edit size.** The chord was not measured directly. The L16 relative site displacement (0.13–0.14)
  is the realized proxy.

## 6. Issues for independent audit

1. **Localization deviation.** Is it justified by the reference-free step-0 finding (§1; `configs/inference_cf/ttls_r1.json:masks`)?
2. **Candidate rule.** Confirm that it uses only θ0 teacher-forced branches computed before adaptation (`ttls.select_candidates`; run rows `candidates.records`).
3. **Hook semantics.** Confirm that the TTLS hook reproduces canonical DG-02 semantics. Two checks to recheck:
   - an empty mask decodes bitwise to B0;
   - a zero edit at all steps gives a max logit difference of 0.125 from BF16 renormalization rounding (`validated.clean_identity`).
4. **Correction categories.** Check the genuine/deletion split. The "same_language_substitution" corrections of T1 are
   word-boundary repairs, which affects the frozen MIXED label.
5. **Matched-comparator fairness.** T2A was added to the user matrix. Check that P is identical across variables.
6. **TTLS lr and budget rule.** lr = E\*/(2√d) with projection was frozen without an outcome sweep. Check that it is
   not tuned.
7. **Manifest r2 runner change.** It touches only the ancestor check (§8). Check that the attempt-1 manifest is
   preserved.

## 7. Reproduction

```bash
PY=/home/tungnx/miniconda3/envs/acl1/bin/python; export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH PYTHONPATH=src:.
$PY -m pytest -q tests/test_ttls_r1.py                                     # 14 tests
$PY experiments/inference_cf_ttls_r1.py prepare                             # plan_sealed.json
$PY experiments/inference_cf_ttls_r1.py manifest --out results/inference_cf/ttls_r1/run1
OUT=results/inference_cf/ttls_r1/run1; sbatch --output=$OUT/slurm-%j.out --error=$OUT/slurm-%j.err --export=ALL,OUT=$OUT slurm/inference_cf_ttls_r1.sbatch
$PY experiments/inference_cf_ttls_r1.py seal --run results/inference_cf/ttls_r1/run1      # commit + push before evaluating
$PY experiments/inference_cf_ttls_r1_evaluate.py --run results/inference_cf/ttls_r1/run1 --out results/inference_cf/ttls_r1/evaluation.json
```

**Artifacts** (`results/inference_cf/ttls_r1/`):

| Artifact | Contents |
|---|---|
| `plan_sealed.json` | Inputs, hashes, UTF-8 flags, A2 archive pointers |
| `run1/manifest.json` | Commit, sources, model files, seed |
| `run1/rows/NNN.json` | Per utterance: all arm transcripts and tokens, loss/gradient/z trajectories, displacement, candidate records, timing, peak memory, effective z |
| `run1/runtime.json` | Counters, validations, reset and non-LN hashes |
| `output_seal.json` | Hashes of every run output |
| `evaluation.json` | Metrics, transitions with per-POI events, ZH changes, safety, bootstrap, subgroups, candidate and optimization descriptives, changed utterances, label |

## 8. Process notes

**Manifest attempt 1** (`51e40546`) was superseded before any job ran:
- The runner required HEAD to equal the manifest's source commit, which committing the manifest itself makes impossible.
- The check now requires the source commit to be an ancestor of HEAD, with only `run1/manifest.json` changed since.
- The old manifest is kept as `manifest_attempt1_superseded.json`.

**Other facts about the round:**
- No outcome-dependent change was made.
- One GPU job ran, with no smoke job or rerun.
- No existing module was modified.
