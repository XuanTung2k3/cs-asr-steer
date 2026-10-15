# P2-2 — on-policy continuation preservation (result)

**Verdict: `P2_2_NO_INCREMENTAL_GAIN`** (pre-registered rules; `p2_2/report.json`).

**Recommendation (not authorization): `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING`.**

**Scope.**
- Three new runs (M2, seeds 42/43/44) against the reused P2-1b CE and GoldKL controls.
- The exposed dev150 panel (20 dialogues) selects checkpoints and evaluates them.
- No P2-3, P3, M1, encoder adaptation, LoRA or sweeps.

**Documents.** Protocol: `P2_2_PROTOCOL.md`. Rules: `P2_2_DECISION_RULES.md`. Amendments: `P2_2_AMENDMENT_R1.md` (label
checker) and `P2_2_AMENDMENT_R2.md` (per-job preflight paths).

## A. Executive verdict

**On-policy preservation adds no Mandarin-preservation value over GoldKL in this pilot.** The supervision itself was
valid and substantial: in epochs 2–3, 611–665 non-identity own-prefix events per seed contributed gradient.

**Against matched GoldKL:**

| Measure | GoldKL | M2 | Change |
|---|---|---|---|
| New Mandarin errors | 53 / 48 / 50 (mean 50.3) | 51 / 70 / 62 (mean 61.0) | **+21%** |
| Long switch-back dropped units | 30 / 31 / 26 | 23 / 37 / 24 | mean −3.9%; fewer in 2 of 3 seeds |
| Mandarin retention ≥ 0.99 | 1 of 3 seeds | 1 of 3 seeds | — |

**The main gain is English.**
- English net corrections rose from 34 / 41 / 35 to 49 / 38 / 46 (ratio 1.21).
- Correct-to-wrong English fell from 20 / 15 / 24 to 12 / 19 / 15.
- CS MER is unchanged (+0.001).
- Loops and non-terminations went down.

**Decision components.**

| Component | Result |
|---|---|
| A preservation | fails (A1, A2 fail) |
| B correction retention | passes |
| C strong preservation | fails |
| D on-policy exposure | passes (3 of 3 seeds) |
| Integrity | passes |

Because A fails with mean reductions below 10%, the rules give NO_INCREMENTAL_GAIN.

**Mechanism.** The exploratory P2-1c cases show the same split:
- M2 repairs the formatting-only history case (d0) in 2 of 3 seeds.
- It does not repair the lexical-history cases (d3, d4).
- It loses the near-tie span d2, which GoldKL had recovered.

**Interpretation.** On-policy exposure alone does not deliver aggregate preservation. The teacher-invalid fallback
(reference CE, more than half of all events, at 1.5–1.9 nats/token) coincides with new damage **far** from switches.
That is a plausible objective-specific cost, not a demonstrated cause.

## B. P2-2A feasibility (training data only; B0, CE42 and KL42 rollouts; preflight 59355)

Each source has 1,078 chosen events, one per category per utterance where an opportunity exists. Opportunities are
SB 613, PRE 613 and ZH 744.

| | B0 | CE42 | KL42 |
|---|---|---|---|
| Located | 767 | 796 | 794 |
| T (teacher-valid) / R (reference) / U | 345 / 422 / 311 | 365 / 431 / 282 | 360 / 434 / 284 |
| Valid events: utterances / dialogues | 384 / 27 | 385 / 27 | 387 / 27 |
| Valid SB events | 231 | 252 | 245 |
| Non-identity prefixes (format-only / lexical / B0-unlocated) | 0 | 331 (114 / 162 / 55) | 294 (98 / 148 / 48) |
| Label mismatches | 0 | 0 | 0 |

**Exclusions.**
- Every U is a location failure: either the context before r is not an exact unique match, or there is no clean
  boundary.
- Every located event received a target, so the valid share of located events is 1.0.

**Teacher reliability.**
- Most teacher-invalid cases had a wrong continuation (328–338), meaning the frozen teacher does **not** reproduce the
  next ≤ 8 reference units; the rest were 15–18 incomplete UTF-8 characters at the 32-token cap and about 80 scope
  boundaries.
- B0 itself drops the Mandarin under the own prefix in 15 of about 245 located SB events.

**Event size.** Mostly 3–5 tokens. The median prefix is 9 tokens.

**Coverage gate.** Passed on CE42 and KL42. The SB share of all switch-back opportunities is 0.38–0.41.

**First preflight (59346): `P2_2_BLOCKED_ONPOLICY_COVERAGE`.** It failed solely on one label mismatch (the same event in
all sources). The target was correct, but the cap cut a Han character's UTF-8 bytes and the checker was too strict.
- **Amendment r1** repaired the checker, keeping the 0-mismatch gate.
- **Amendment r2** separated preflight output paths after a colliding rerun was cancelled within 15 s, before writing
  anything.
- **The rerun** (59355) reproduced the same coverage counts and the same rollouts (tokens identical 500/500 per source)
  with 0 mismatches.

## C. Engineering gates

**Preflight 59355: `READY_FOR_P2_2_TRAINING`.**

| Gate | Result |
|---|---|
| G1 | branch, chains, control hashes, pinned baseline and weights: all pass |
| G2/G3 | coverage and label consistency: pass |
| G4 | identity OP-T KL exactly 0; OP-R CE finite; perturbed KL > 0; gradients reach R, W, b; no backbone gradient; batched vs single within BF16 tolerance; hooks cleared; **the M2 loop at λ_OP = 0 reproduces P2-1b KL_seed42 epoch 1 bitwise** (max diff 0); initial `base` equals KL42's |
| G5 | identity training rollouts = B0 (20/20); dev beam-5 and greedy = pinned baseline (20/20 each); replay parity 10/10 |
| G6 | projected 68 min per run (actual 61–63 min) |
| G7 | authorization record, frozen rules |

**Runs.**
- Zero-init identity 150/150 (each run).
- Initial `base` equals the matched KL run's (each run).
- Epoch-1 rollouts identical to B0, 500/500 (each run).
- Weights unchanged.

**Reload.** Job 59369 reproduced the recorded decodes for all 9 epoch checkpoints (20/20 each).

**Controls.** Recomputed C0/C1 metrics reproduce the P2-1b evaluation exactly (6/6).

**Freeze chain:** `freeze.json` `f3786318` → `freeze_r1` `d7e93742` → `freeze_r2` `a456b22b`.

## D. Matched results (selected checkpoints; beam-5; M2 selected epoch 2 for all seeds)

| Seed | GoldKL CS MER | M2 CS MER | GoldKL ZH retention | M2 retention | GoldKL EN net | M2 EN net | GoldKL SB units | M2 SB units |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.1124 | 0.1114 | 0.9893 | 0.9897 | 34 | 49 | 30 | 23 |
| 43 | 0.1105 | 0.1144 | 0.9903 | 0.9859 | 41 | 38 | 31 | 37 |
| 44 | 0.1103 | 0.1109 | 0.9899 | 0.9875 | 35 | 46 | 26 | 24 |
| CE (ref.) | 0.1099 / 0.1127 / 0.1105 | — | 0.9853 / 0.9826 / 0.9812 | — | 37 / 34 / 47 | — | 52 / 52 / 56 | — |

### Paired M2 − GoldKL (shared dialogue-bootstrap draws)

| | Seed mean [95% CI] | Per seed |
|---|---|---|
| New Mandarin errors | **+10.7** [−17.3, +37.3] | −2 / +22 / +12 |
| SB units | −1.0 [−27.3, +25.7] | −7 / +6 / −2 |
| EN net | **+7.7** [−0.3, +17.3] | +15 / −3 / +11 |
| CS MER | +0.0012 [−0.0064, +0.0072] | |
| MER | +0.0022 [−0.0044, +0.0074] | |

**M2 − CE:** new errors −23.0, SB −25.3 and EN net +5.0. M2 still clearly improves on plain CE.

**Epoch-matched M2 − GoldKL** (secondary): new Mandarin errors are higher in 8 of 9 seed × epoch pairs (−3 to +30).

### Failure profile (M2 vs GoldKL)

| Measure | GoldKL | M2 |
|---|---|---|
| New deletions | 43 / 38 / 37 | 34 / 58 / 45 |
| Near-switch damage | 53 / 45 / 49 | 47 / 59 / 46 |
| **Far-from-switch damage** | 0 / 3 / 1 | **4 / 11 / 16** |
| Outside-target damage | 56 / 51 / 52 | 55 / 74 / 66 |
| Switch-back spans dropped | 7 / 6 / 6 | 5 / 6 / 5 |
| Loops | 0 everywhere | 0 everywhere |
| Non-terminations | 1 / 1 / 2 | 0 / 0 / 1 |
| Improved / worsened utterances | 38/21, 35/16, 38/19 | 33/23, 33/23, 32/23 |

M2 seed 43 regresses sharply at epoch 3 (CS MER 0.169, one loop, two non-terminations). Epoch 3 was not selected.

## E. On-policy mechanism

### Aggregate (training)

| Measure | Value |
|---|---|
| Events per epoch | 767–796 (T about 350, R about 420) |
| Non-identity prefixes, epoch 1 | 0 (identity rollout) |
| Non-identity prefixes, epochs 2–3 | 298–334 per epoch (format-only about 95, lexical about 175, B0-unlocated about 45) |
| Rollout hypotheses identical to B0, epochs 2–3 | 185–219 of 500 |
| OP gradient norm | 0.8–1.2, vs total 1.2–1.9: a large share of each update |
| OP-T KL | 0.02 → 0.07 |
| OP-R CE | 1.9 → 1.5 nats/token, vs gold CE 0.39–0.41 |

So M2 did train on its own histories. The aggregate dev result nevertheless shows no switch-back benefit and more
Mandarin damage elsewhere.

### Ten fixed P2-1c cases (exploratory; the GoldKL rows reproduce P2-1c exactly)

| Case | Original (M2) | B0 prefix → M2 | Own prefix → M2 | GoldKL |
|---|---|---|---|---|
| d0 (format-only history) | **kept 2/3** | kept 3/3 | **kept 2/3** | dropped 3/3, own prefix 0/3 |
| d1 | kept 3/3 | kept 3/3 | kept 3/3 | kept 3/3 |
| d2 (near-tie) | **dropped 3/3** | dropped 3/3 (switch token rank 2, Δ −0.8 to −0.9) | dropped 2/3 | kept 3/3 |
| d3 (lexical history) | dropped 3/3 | kept 3/3 | dropped 3/3 | same as M2 |
| d4 (lexical history; B0 also drops) | dropped 3/3 | kept 3/3 | dropped 3/3 | same as M2 |
| safe s0–s4 | kept | kept | kept | kept |

**The model is less sensitive to formatting-only variation in its own history** (d0), which is the adapter × history
form found in P2-1c. **It is not repaired for lexically different histories** (d3, d4). There, the frozen teacher
itself fails on the own prefix, so supervision falls back to reference CE.

**This does not generalize to the aggregate.** These are five failure cases, and the aggregate switch-back damage did
not improve.

## F. English lexical behavior (B0 → system fates; selected checkpoints)

| Transition | GoldKL | M2 |
|---|---|---|
| Omission → correct | 29 / 32 / 33 | 35 / 31 / 34 |
| Translation-like → correct | 13 / 14 / 14 | 14 / 14 / 13 |
| EN→EN lexical substitution → correct | 2 / 4 / 2 | 4 / 2 / 4 |
| Correct → wrong | 20 / 15 / 24 | 12 / 19 / 15 |

**Corrections:** 54 / 56 / 59 for GoldKL vs 61 / 57 / 61 for M2.

M2's English gain comes mostly from fewer corruptions and slightly more omission recovery. Lexical substitutions
remain largely unrepaired.

## G. Compute and efficiency

**Unchanged:** 5,122 trainable parameters, a 20 KB adapter, and identical inference cost (beam-5 about 1.0 s per
utterance, greedy about 0.65 s, as for R2).

**Per M2 run** (wall 61–63 min, vs about 19.5 min for a P2-1b KL run, so **about 3.2×**):

| Phase | Time |
|---|---|
| Rollouts (3 × 500 beam-5) | 23 min |
| Event construction with about 2,340 B0 teacher replays (about 46,700 decoder steps) | 18.5 min |
| Gold training | 2.6–2.8 min |
| OP forward/backward | 1.3–1.4 min |
| Dev decodes | 17 min |

Peak VRAM was 5.73 GB allocated and 8.24 GB reserved.

**GPU allocation for P2-2: 298 MIG-minutes ≈ 4.97 slice-hours.**

| Item | Slice-hours |
|---|---|
| Three runs | 3.12 |
| Two preflights | 1.43 |
| Cancelled rerun | 15 s |
| Dry runs | 0.36 |
| Verify and mechanism replay | 0.16 |

CPU time was about 6 min.

M2 is not cheaper. Its parameter count is unchanged, but its training compute is about 3× that of GoldKL.

## H. Limitations

- **Exposed panel.** dev150 is exposed and also selects checkpoints, with only 20 evaluation dialogues. The three seeds
  share those dialogues, and every interval is wide (the new-error CI spans −17 to +37).
- **Reference-informed history.** The P2-1c cases and anchors were reference-informed; their analysis here is
  exploratory.
- **Teacher-validity selection.** It used training references, and is strict (exact next ≤ 8 units). As a result,
  reference CE carries more than half of the OP events. Its targets follow the training-target format (no
  punctuation, no spaces next to Han), which may conflict with Whisper's own output style after a generated prefix.
  This is a hypothesis, not tested.
- **Attribution.** There is no matched extra-supervision or off-policy-prefix control. The English gain and the
  Mandarin cost cannot be attributed to on-policy *specificity* as opposed to added supervision.
- **Other gaps:**
  - no optimized LoRA control;
  - no independent test set;
  - no acoustic-causal experiment;
  - a single λ_OP = 1.0 pilot, untuned by design.

## I. Research decision: `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING`

**This is the frozen mapping for NO_INCREMENTAL_GAIN with valid exposure.**
- The on-policy objective was exercised with substantial, valid, non-identity supervision.
- It still did not reduce aggregate switch-back omissions beyond GoldKL, and it added Mandarin damage away from
  switches.
- This failure is specific to this objective (teacher KL + reference-CE fallback at λ = 1). It is **not** evidence
  against representation adaptation in general:
  - M2 still clearly beats CE (−23 new Mandarin errors, −25 SB units);
  - it keeps or improves English corrections;
  - it repaired the formatting-only history case.

**Next step.** The smallest informative experiment is the pre-planned, parameter-matched architecture screen: R3 vs M1
vs E1+D2, on GoldKL. It tests whether conditioning the intervention, rather than changing its training histories,
moves the correction–preservation frontier. It is not authorized here.

**Why the alternatives are less informative:**

| Alternative | Why not |
|---|---|
| `GO_P2_3_MATCHED_SUPERVISION_CONTROL` | There is no gain to attribute. |
| `GO_P2_3_ONPOLICY_MECHANISM_VALIDATION` | Another diagnostic of a method that did not improve the target metric. The d0-specific effect is recorded for later. |
| `HOLD` | Every integrity check passed. |
| `STOP` | Not supported: R2 + GoldKL / M2 still improve English and reduce CE's damage. |

## J. Reproducibility

**Commits** (all pushed to `cs-asr-reft`):
- `981a875`: protocol, code and tests, before execution;
- `6c94cce`: amendments r1 and r2, before training;
- this result.

**Jobs:**

| Purpose | Jobs |
|---|---|
| Tests | 59331, 59339 |
| Pre-freeze dry runs and debugging (scratch, deleted) | 59332–59344 |
| Freeze | 59345 |
| Preflight (blocked) | 59346 |
| Freeze-r1 | 59352 |
| Cancelled rerun | 59353 |
| Freeze-r2 | 59354 |
| Preflight (READY) | 59355 |
| Training | 59364 / 59365 / 59366 |
| Select | 59367 |
| Evaluate | 59368 |
| Verify | 59369 |
| Mechanism anchors | 59370 |
| Mechanism replay | 59371 |
| Report | 59372 |

**Tests:** 93/93 CPU tests: 72 earlier + 12 for P2-1c + 9 for P2-2, including the r1 regression test.

**Seeds:** 42, 43, 44. **Training runs:** 3 new (M2 only); C0/C1 reused.

**Hashes (sha256, prefix):**
- Preflights: `preflight_59346` `a784fe65`, `preflight_59355` `f3a8e1d6`, `coverage_train_59355` `d4090f77`.
- Outputs: `evaluation.json` `6b086326`, `report.json` `a450837d`, `verify.json` `71493099`, `mech_integrity.json`
  `62af39e2`, `manifest.json` `426813b0`.
- Run manifests: `fe27ea7c` / `fbe6d1eb` / `d32bc323`. Selections: `20d535c0` / `d4102082` / `00dc6643`.

**Artifacts:** `/mnt/data/tungnx/cs-asr-reft/p2_2/` (private). It holds:
- rollouts and events per epoch in `training/M2_seed*/rollouts/`;
- the two preflights with their per-job rollouts;
- the mechanism audit (`mech_*`);
- figures F1–F5 in `figures/`.

**Confirmations:**
- No P2-3, P3, M1, encoder adapter or LoRA work.
- No sweep and no extra run.
- No locked-role data.
- No dev data used in training or event construction.
- Historical P2-0 to P2-1c freezes and artifacts unchanged.
