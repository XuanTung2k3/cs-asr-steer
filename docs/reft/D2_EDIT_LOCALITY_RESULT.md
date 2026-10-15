# D2 — representation edit-locality pilot (result)

**Verdict: `D2_NO_INCREMENTAL_GAIN`.** Recommendation: `STOP_GOLD_PREFIX_LOCALITY_OBJECTIVE_EXPANSION` (not
authorization).

This is a negative, exploratory result on the exposed dev150 panel; every integrity check passed. D2 achieved its
representation-level target (much smaller, partly selective Mandarin edits), but free-decoding Mandarin damage moved the
wrong way: new Mandarin errors were higher than GoldKL in all three matched seeds.

Protocol: `D2_EDIT_LOCALITY_PROTOCOL.md`. Rules: `D2_EDIT_LOCALITY_DECISION_RULES.md`. Private artifacts:
`/mnt/data/tungnx/cs-asr-reft/d2_edit_locality/`.

## 1. Engineering

**Edit mask (training references only):**
- 500/500 utterances mappable; targets equal the manifest; 18,616 supervised positions; 16,215 substantive Mandarin and
  3,001 substantive English units.
- 14,225 Mandarin-positive tokens (0 ambiguous; 343 `other` tokens and 1 unmapped token excluded); 3,547 English tokens.
  The switch-back subset has 2,825 tokens, equal to D1's region.
- Teacher-correct coverage: **13,289/14,225 = 93.4%** (switch-back 2,557/2,825; non-switch-back 10,732/11,400); the edit
  mask is 79.4% of all 16,737 teacher-correct positions; English 2,831/3,547; EOS 463/500.
- Mandarin-positive tokens in 460 utterances and all 27 dialogues (top-3 dialogues 17.9% of edit positions).
- In all nine seed × epoch training orders every micro-batch has ≥ 45 teacher-correct edit positions (median ≈ 205);
  manifest order has 5/63 empty micro-batches (reported only; see protocol §12). Mapping takes 0.3 s on CPU.

**D2.0 calibration (job 59643): `D2_CALIBRATION_OK`.**

| Quantity | Value |
|---|---|
| State | historical R2-GoldKL seed-42 epoch-1 (sha verified) |
| Batches | 8 × 8 training rows, sha256 order; 134–287 edit positions each |
| median ‖g_base‖ / median ‖g_edit‖ | 0.2462 / 0.0698 |
| **μ** | **0.70512459** (both passes bitwise identical; equal to the dry-run value) |
| Ratio spread (limit 10) | 1.95 |
| Scaled ratios μ‖g_edit‖/‖g_base‖ (limit 1.0) | 0.136–0.266 |
| cos(g_base, g_edit) | −0.27 to +0.12 (7 of 8 negative) |
| μ·L_edit / (CE + KL) | 0.33–0.58% |
| Identity | L_edit = 0 and ∇L_edit = 0 exactly |

**Preflight 59644: `D2_READY_FOR_TRAINING`, G1–G7 pass** (head `c16394d`).
- G2: one observer record and one adapter call per forward; logits bitwise unchanged by observing; single application
  bitwise against the P1-site injection of the manually recomputed delta; observer energy bitwise equal to the manual
  delta's energy; rank-2 identity ‖Δh‖² = ‖a‖² to 1.9e-7 relative; L_edit gradient reaches R, W and b; R_M ⊆
  teacher-correct; EOS and prompt excluded; BF16-cast energy differs from the float32 energy by 9e-5 (median, relative).
- G4: μ = 0 with the observer installed equals P2-1b `kl_step` bitwise (zero init and perturbed: loss, CE, KL, mask,
  logits, gradients); the full seed-42 epoch-1 replay reproduces `KL_seed42/epoch1/adapter.pt` bitwise with equal train
  KL, CE, mask positions, tokens and 32 updates.
- G5: identity zero value and gradient; on a separate batch at the calibration state the scaled ratio is 0.122.
- G6: 5,122 parameters; zero-init dev 20/20 beam-5 and greedy; LID 20/20; cache positions; float32 cached-vs-full
  ≤ 3.6e-6 (BF16 up to 0.053, descriptive); micro-batch D2 0.432 s vs GoldKL 0.439 s vs CE 0.419 s (no measurable cost).

## 2. Integrity

All report integrity checks are true: calibration OK and μ frozen in every run; preflight READY; verify 9 × 20/20 with
matching hashes; 3 runs × 3 epochs × 96 updates; seed-specific identity init with `base` equal to GoldKL; zero init
150/150 for all six runs; backbone unchanged; GoldKL recomputation reproduces P2-1b at every epoch; per-epoch mask
positions, supervised tokens and updates equal the matched GoldKL run; orthogonality ≤ 6.0e-7; frozen mask hash; edit
term active every epoch (13,288–13,290 edit positions, no empty micro-batch); constant lr 5e-4.

## 3. Matched results (beam-5, dev150, selected checkpoints)

Selected epochs: GoldKL 3/2/3, D2 3/2/3. TW: 222 windows, 1,270 eligible units, 1,156 B0-correct.

| Seed | New ZH (GoldKL / D2) | TW new | Long-SB units | EN net | CS MER |
|---|---|---|---|---|---|
| 42 | 53 / 54 | 37 / 37 | 30 / 35 | 34 / 37 | .1124 / .1116 |
| 43 | 48 / 56 | 31 / 39 | 31 / 35 | 41 / 38 | .1105 / .1137 |
| 44 | 50 / 58 | 33 / 39 | 26 / 30 | 35 / 29 | .1103 / .1148 |
| Mean | 50.3 / 56.0 | 33.7 / 38.3 | 29.0 / 33.3 | Σ110 / Σ104 | .1111 / .1134 |

**Rules:**
- A fails: mean +5.7 (wrong direction), D2 lower in 0/3 seeds (per seed +1/+8/+8).
- B passes: 104/110 = 0.945.
- C fails: +0.0023.
- Components: TW +4.7 and SB +4.3, 0/3 seeds lower → fail. TW support: not supported. E (zh_new ≤ 49): D2 0/3, GoldKL
  1/3.

**Paired D2 − GoldKL, seed mean [dialogue bootstrap 95%, 2,000 shared draws]:** zh_new +5.7 [−0.7, +14.7]; TW +4.7
[0, +13.3]; SB +4.3 [0, +11.3]; EN net −2.0 [−6.7, +1.7]; CS MER +0.0023 [+0.0006, +0.0048]; overall MER +0.0016
[+0.0002, +0.0036].

**Stability:** seeds 43 and 44 are +8 each and survive leave-one-dialogue-out (ranges +2…+10 and +2…+9; largest single
dialogue 75% of the seed total); seed 42 is +1 (range −1…+2). The worsening is not one dialogue.

## 4. Correction–preservation detail (GoldKL / D2 per seed)

| Measure | GoldKL | D2 |
|---|---|---|
| New Mandarin deletions | 43/38/37 | 47/47/42 |
| TW deletions / substitutions | 30/26/25 / 7/5/8 | 32/33/31 / 5/6/8 |
| Near-switch (≤ 7) new errors, of 2,218 | 53/45/49 | 54/55/54 |
| Far new errors, of 2,732 | 0/3/1 | 0/1/4 |
| Outside-TW new errors | 16/17/17 | 17/17/19 |
| Dropped switch-back spans | 7/6/6 | 7/7/7 |
| Mandarin repairs | 94/90/100 | 98/85/87 |
| Retention | .9893/.9903/.9899 | .9891/.9887/.9883 |
| Worst-dialogue regression | 32/32/32 | 34/34/32 |
| New errors in utterances without an English repair | 50/28/47 | 34/54/56 |
| EN corrections / corruptions | 54/56/59 / 20/15/24 | 55/54/54 / 18/16/25 |
| Omission → correct | 29/32/33 | 30/32/32 |
| Translation → correct | 13/14/14 | 13/14/13 |
| Correct → wrong | 20/15/24 | 18/16/25 |
| Overall MER | .1104/.1095/.1089 | .1095/.1115/.1125 |
| Mandarin CER | .0905/.0902/.0891 | .0898/.0922/.0924 |
| Loops / non-terminations | 0 / 1/1/2 | 0 / 2/1/2 |

The extra damage is mostly near-switch Mandarin deletions. English corrections are essentially retained (the −6 comes
from seed 44's extra corruptions and fewer corrections). Greedy (secondary): CS MER .141/.144/.142 vs .170/.144/.142;
greedy new Mandarin errors 107/95/104 vs 101/91/91.

## 5. Selection sensitivity

Fixed epoch 3: new Mandarin errors 53/59/50 vs 54/56/58 (mean +2.0, D2 lower in 1/3); TW 37/39/33 vs 37/39/39 (+2.0);
SB 30/35/26 vs 35/35/30 (+3.0); EN net Σ103 vs Σ99 (0.961); CS MER +0.0021. Epoch-matched new-Mandarin differences range
−12 to +8 (seed 43 epoch 1 is −12; every selected-epoch difference is ≥ 0). No view shows a consistent D2 gain.

## 6. Training dynamics

| Epoch 3 (seeds 42/43/44) | GoldKL | D2 |
|---|---|---|
| Train CE | .3620/.3628/.3612 | .3650/.3646/.3614 |
| Train GoldKL | .0163/.0159/.0155 | .0142/.0142/.0141 |
| L_edit (teacher-correct Mandarin, /1280) | — | .00233/.00245/.00261 |
| μ·L_edit | — | .00164/.00172/.00184 |
| Teacher-correct English edit energy (/1280) | — | .0088/.0081/.0096 |
| Pre-clip gradient norm (mean) | .286/.263/.296 | .242/.231/.271 |

L_edit rose from ≈ 0.0014 (epoch 1) to ≈ 0.0025 (epoch 3) as the adapter grew; within D2 runs, English edit energy was
≈ 3.3–3.8× the Mandarin edit energy at epoch 3. The penalty is ≈ 0.5% of the loss value but, by construction, ≈ 20% of
the gradient scale at calibration. Train GoldKL fell 11–13% relative to GoldKL; CE was ≤ 0.003 higher.

## 7. Representation selectivity (training utterances, gold prefixes, teacher-correct positions; descriptive)

| D2 / GoldKL, selected checkpoints | Seeds 42/43/44 |
|---|---|
| ‖Δh‖ ratio, Mandarin (non-switch-back) | 0.59/0.65/0.63 |
| ‖Δh‖ ratio, Mandarin (switch-back) | 0.63/0.70/0.67 |
| ‖Δh‖ ratio, English | 0.83/0.87/0.85 |
| Edit-energy ratio, Mandarin / English | 0.40–0.58 / 0.72–0.78 |
| **Selectivity** (Mandarin ratio / English ratio; 1 = uniform shrinkage) | **0.72/0.76/0.75** |
| Per-position edit cosine, Mandarin / English | 0.89–0.97 / 0.98–0.99 |
| Teacher-forced KL ratio, Mandarin non-SB / SB / English | 0.81–0.84 / 0.90–0.92 / 0.89–0.97 |
| R row-space principal cosines | 0.986–0.999 |

‖Δh‖/‖h‖ at Mandarin positions fell from 0.12–0.17 to 0.08–0.10. So D2 is more than generic shrinkage: Mandarin edits
shrank about twice as much as English edits, with some rotation at Mandarin positions (epoch-3 seed-43 cosine p10 0.50),
inside nearly the same 2-D subspace. Teacher-forced Mandarin KL fell. None of this produced better free decoding.

## 8. Same-prefix diagnostic (40 training utterances; 320/320 rows eligible; descriptive)

Han-script positions (frozen top-1), adapter off vs on on the identical prefix, D2/GoldKL ratio of mean KL(p_off ‖ p_on):

| Prefix | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| Gold | 0.87 | 0.91 | 0.88 |
| B0 beam-5 output | 0.87 | 0.92 | 0.88 |
| Own beam-5 output | 0.88 | 0.86 | 0.88 |
| Transfer (B0 / gold; own / gold) | 1.01; 1.02 | 1.01; 0.95 | 1.00; 1.00 |

Han-to-non-Han top-1 changes were rare in both arms (0.15–0.71% of Han positions) and equal or lower for D2; changes to EOS
were ≈ 0. B0 matched the gold sequence in 1/40 cases; systems matched B0 in 12–14/40. On training audio the D2 next-token
protection transfers fully from gold to generated prefixes. The dev free-decoding worsening is therefore not explained by a
gold-to-generated next-token gap on training audio. It points instead to sequence-level effects of small distribution
changes, or to dev-audio generalization, which this bounded diagnostic does not test.

## 9. Efficiency

5,122 parameters (20,488 bytes; checkpoint 6,576,349 bytes including the orthogonal-parametrization buffer); training
84–85 s per run plus 7.7 s teacher forward; wall 1,109–1,124 s per job; peak 5.73 GB allocated / 8.02 GB reserved; beam-5
1.00–1.04 s/utt, greedy 0.62–0.63 s/utt; edit capture adds no measurable time.

**GPU: 86.7 MIG-minutes ≈ 1.44 slice-hours.**

| Component | MIG-minutes |
|---|---|
| Formal runs | 56.9 |
| Preflights (3 dry + 1 official) | 10.4 |
| Pre-freeze smoke test (mechanism + prefix) | 7.2 |
| Prefix diagnostic | 5.1 |
| Verify | 3.8 |
| Mechanism | 2.2 |
| Calibration (dry + official) | 1.1 |

## 10. Interpretation

1. **Did D2 reduce actual Mandarin damage?** No. +5.7 new errors on average, higher in 3/3 seeds, mostly near-switch
   deletions.
2. **Stable across seeds and dialogues?** The worsening is (seeds 43/44 robust to leaving out any dialogue); there is no
   gain to be stable.
3. **English corrections retained?** Yes within the guard: 104/110 = 0.945.
4. **Genuine-CS MER competitive?** No: +0.0023, the bootstrap interval excludes 0.
5. **More selective than reducing all edit magnitudes?** At the representation level, yes (selectivity 0.72–0.76,
   Mandarin edits ≈ −35 to −41%). Behaviourally this selectivity did not help.
6. **Did gold-prefix improvements translate into generated-prefix behaviour?** At the next-token level on training audio,
   yes (transfer ≈ 1.0). In dev free decoding, no.
7. **What would justify continuing?** A locality-type objective would need (a) a pre-registered free-decoding gain (≥ 5
   units, ≥ 2 seeds, guards held) and (b) evidence that the regularized quantity predicts held-out free-decoding damage.
   D2 shows the opposite: a strong reduction of the regularized quantity alongside more damage.

**Hypotheses:**

| Hypothesis | Status |
|---|---|
| H1 (selective Mandarin edit suppression improves the trade-off) | Not supported (wrong direction) |
| H2 (generic attenuation that sacrifices corrections) | Not supported as stated: suppression was partly selective and English held within the guard (−6) |
| H3 (edit norms fall, free-decoding errors remain) | Supported, and stronger: errors increased |
| H4 (generated-prefix sensitivity dominates) | Not supported by the bounded diagnostic: next-token protection transferred to generated prefixes on training audio |
| H5 (seed/dialogue variation dominates) | Partly: seed 42 ≈ 0; seeds 43/44 consistent and dialogue-robust |

**Alternative explanations (not tested):** with a rank-2 intervention shared across positions, shrinking Mandarin edits
changes the low-rank map the English corrections also use. The ≈ 20% gradient share may act mainly as a different
optimization path rather than as preservation. Dev audio may generalize differently from training audio. One μ and one
mask definition were tested; no claim is made about other coefficients.

## 11. Limitations

dev150 is exposed (20 dialogues) and used for both selection and evaluation; three seeds share those dialogues; one μ; no
optimized LoRA baseline; no independent test; the representation and prefix diagnostics are training-only and
descriptive; the prefix diagnostic covers 40 utterances and next-token distributions, not whole sequences.

## 12. Next step (recommendation only)

`STOP_GOLD_PREFIX_LOCALITY_OBJECTIVE_EXPANSION`. Do not add another weighted KL, locality coefficient or architecture
variant by default. The next direction should be chosen jointly with a competitive, independently protocolled LoRA
benchmark and the prefix-history findings (P2-1c, and this report's §8). The locality-specificity control of protocol §10
applies only after a positive D2 signal, so it is not recommended. Nothing was started.

## 13. Provenance

- Commits: `c16394d` (protocol, rules, code, tests; frozen head) and this result commit.
- Freeze `freeze.json` sha256 `272d85b9…` (chains to D1 `affa9460…`, which verifies S0/S1, P2-0, P2-1b, P2-1c and P2-2);
  `mu_freeze.json` `66c713be…`; calibration `27a056f3…`; mask `a1859d1f…`; mask static `2daef522…`; preflight_59644
  `c889a94e…`; mask_coverage_59644 `4807ec62…`; evaluation `5a2fc0e3…`; verify `9b903679…`; mechanism `8b9d2ed8…`;
  prefix diagnostic `f970be1d…`; report `776928b6…`; manifest `763813ac…`.
- Jobs: tests 59625 (one test fixed), 59627, 59633, 59634; dry runs 59628–59630, 59632, 59635; smoke 59637–59639; mask
  59641; freeze 59642; calibration 59643; preflight 59644; training 59647/59648/59649; select 59650; evaluate 59651; verify
  59652; mechanism 59653; prefix 59654; report 59655.
- Exactly three formal D2 runs, none retried; historical GoldKL controls not retrained; no locked role opened; no D1+D2,
  μ sweep, LoRA, architecture change or independent evaluation.
