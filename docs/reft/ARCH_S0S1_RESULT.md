# S0/S1 — parameter-matched R3 vs M1 under GoldKL (result)

**Verdict (pre-registered rules, `ARCH_S0S1_DECISION_RULES.md`): `ARCH_S1_NO_ARCHITECTURAL_GAIN`**
→ recommendation `STOP_GENERIC_LOREFT_ARCHITECTURE_EXPANSION`. The recommendation is not authorization.

- **S0:** `ARCH_S0_READY_FOR_S1`, with all gates G1–G7 passing in preflight 59509 under freeze `ca2eab0c…`.
- **S1:** exactly six runs (R3 and M1 × seeds 42/43/44). All integrity checks pass.
- **Exposure:** dev150 is exposed and used for both selection and evaluation, with 20 dialogues shared across seeds.
  Everything below is exploratory.

## 1. Selected checkpoints (beam-5, dev150; unchanged P2-1 selection rule)

R2 = historical P2-1b R2-GoldKL, recomputed here; it reproduces the P2-1b evaluation exactly for 3/3 seeds.

| Seed | R2 new ZH | R3 new ZH | M1 new ZH | R2 SB | R3 SB | M1 SB | R2 EN net | R3 EN net | M1 EN net | R2 CS MER | R3 CS MER | M1 CS MER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 53 | 58 | 54 | 30 | 30 | 35 | 34 | 34 | 32 | 0.1124 | 0.1137 | 0.1154 |
| 43 | 48 | 56 | 56 | 31 | 30 | 35 | 41 | 21 | 38 | 0.1105 | 0.1174 | 0.1118 |
| 44 | 50 | 46 | 56 | 26 | 27 | 30 | 35 | 39 | 40 | 0.1103 | 0.1056 | 0.1101 |
| **Mean** | **50.3** | **53.3** | **55.3** | **29.0** | **29.0** | **33.3** | **36.7** | **31.3** | **36.7** | **0.1111** | **0.1122** | **0.1124** |

- **Selected epochs:** R2 3/2/3; R3 3/2/3; M1 2/2/3.
- **Mandarin retention:** R2 0.9893/0.9903/0.9899; R3 0.9883/0.9887/0.9907; M1 0.9891/0.9887/0.9887.
- **Loops:** 0 everywhere.
- **Non-terminations:** R2 1/1/2, R3 1/2/2, M1 1/1/2.

**Paired differences** (seed mean, with the per-seed range and a dialogue-bootstrap 95% CI: 2,000 draws, seed 42, shared
across arms):

| Contrast | New ZH | SB units | EN net | CS MER |
|---|---|---|---|---|
| M1 − R3 (primary) | +2.0 [−4.0, +9.7] (−4/0/+10) | +4.3 [0.0, +11.7] (+5/+5/+3) | +5.3 [−2.0, +15.7] (−2/+17/+1) | +0.0002 [−0.005, +0.005] |
| M1 − R2 | +5.0 [0.0, +13.3] (+1/+8/+6) | +4.3 [0.0, +11.3] (+5/+4/+4) | 0.0 [−6.3, +5.7] (−2/−3/+5) | +0.0014 [−0.001, +0.004] |
| R3 − R2 | +3.0 [−1.0, +7.3] (+5/+8/−4) | 0.0 [−4, +4] (0/−1/+1) | −5.3 [−20.7, +5.7] (0/−20/+4) | +0.0012 [−0.004, +0.007] |

Positive new-ZH / SB values are worse. CS-only Mandarin and SB counts match the overall pattern; nearly all switch-back
damage is in CS utterances. Several per-seed differences are concentrated in one dialogue:
- M1 − R3, seed 44: 9 of the +10 new-ZH errors come from one dialogue;
- R3 − R2, seed 43: 15 of the −20 EN net come from one dialogue.

## 2. Decision components

| Component | Result | Detail |
|---|---|---|
| A — M1 beats R3 (all required) | **fail** | A1 mean new ZH −5: +2.0 ✗; A2 seeds lower: 1/3 ✗; A3 mean SB −5: +4.3 ✗; A4 seeds lower: 0/3 ✗ |
| B — guard vs R2 | **fail** | B1 55.3 > 50.3 ✗; B2 33.3 > 29.0 ✗; B3 loops/non-termination ✓ |
| C — corrections (M1) | **fail** | C1 EN net > 0 in 3/3 ✓; C2 ratio 1.00 ✓; C3 CS MER +0.0014 vs R2 and +0.0002 vs R3 ✓; C4 ✗: worst dialogue M1_44 32 > R3_44 14 + 10 |
| D — retention ≥ 0.99 in ≥ 2 seeds | not met by any arm | R2 1/3, R3 1/3, M1 0/3 |
| G_R3 — R3 meaningfully better than R2 | **fail** | new ZH +3.0 (1/3 seeds lower), SB 0.0 (1/3); corrections kept (ratio 0.85, CS MER +0.0012) |
| G_M1 — M1 meaningfully better than R2 | **fail** | new ZH +5.0 (0/3 seeds lower), SB +4.3 (0/3); corrections kept (ratio 1.00) |

The precedence is ¬(A ∧ C), ¬G_R3, ¬G_M1, which gives `ARCH_S1_NO_ARCHITECTURAL_GAIN`. Neither larger architecture
improves on the 5,122-parameter R2-GoldKL. Both match it on English corrections and CS MER within the tolerances, and
both are numerically slightly worse on Mandarin preservation.

## 3. Correction–preservation detail (selected checkpoints, seeds 42/43/44)

| | R2 | R3 | M1 |
|---|---|---|---|
| EN corrections / corruptions | 54/20, 56/15, 59/24 | 53/19, 47/26, 53/14 | 59/27, 55/17, 61/21 |
| Omission → correct | 29/32/33 | 29/25/28 | 30/31/33 |
| Translation → correct | 13/14/14 | 13/10/14 | 13/14/15 |
| Lexical substitution → correct | 2/4/2 | 3/2/2 | 4/4/2 |
| Correct → wrong | 20/15/24 | 19/26/14 | 27/17/21 |
| New Mandarin deletions | 43/38/37 | 43/43/39 | 47/45/43 |
| Damage near switch (≤ 7 units) / far | 53/45/49 · 0/3/1 | 56/55/46 · 2/1/0 | 54/53/55 · 0/3/1 |
| Switch-back spans dropped | 7/6/6 | 7/7/4 | 7/7/7 |
| New ZH errors in utterances without an EN repair | 50/28/47 | 55/53/42 | 33/36/53 |
| Worst-dialogue Δ mixed errors | 32/32/32 | 32/32/14 | 32/32/32 |
| Dialogues improved / worsened | 14/5 each | 13/5, 14/5, 14/2 | 14/5, 15/4, 14/4 |

English lexical substitutions remain essentially unrepaired by every arm (2–4 per run). The English gains of all three
architectures still come from omission and translation recovery.

**Damage location.** Mandarin damage stays almost entirely within 7 units of a switch. The persistent switch-back
omission pattern of P2-1D/P2-1b is unchanged by either architecture.

**Greedy (secondary)** is consistent: CS MER 0.136–0.144 for all runs except R3_44 at 0.175 (189 new ZH errors under
greedy).

**All epochs** (CS MER / new ZH / SB) show the same shape as P2-1b:
- Epoch 1 checkpoints have the least Mandarin damage (new ZH 34–48, SB 14–24) but higher CS MER.
- Two epoch-3 regressions occurred but were not selected: R3_43 (CS MER 0.138, 3 non-terminations) and M1_42 (CS MER
  0.132, 2 non-terminations).

## 4. Architectural attribution

**Rank/capacity (R3 vs R2).** Adding a third LoReFT direction (+50% parameters) did not reduce Mandarin damage or
switch-back omissions: +3.0 new ZH, 0.0 SB, with seeds split. It cost 5.3 EN net on average, driven by seed 43. H1, that
rank-2 capacity is the bottleneck, is not supported under this recipe.

**Cross-attention readout (M1 vs R3, M1 vs R2).** M1 is no better than R3 (+2.0 new ZH, +4.3 SB) and slightly worse
than R2 on preservation (+5.0, +4.3), with equal English net. H2 is not supported.

**W_u was trained but carried little of the edit.**
- **Growth:** ‖W_u‖ grew 0 → ≈0.21 → 0.42 → 0.61 over the three epochs. Its Adam step sizes were comparable to W_h's.
- **Share of the edit:** on the selected checkpoints' own dev outputs, ‖W_u u‖ averaged 0.17–0.24, against
  ‖(W_h − R)h‖ of 2.3–2.9. W_u u carried 0.5–2.4% of the squared pre-projection edit, and ‖e_u‖ > ‖e_h‖ at < 0.2% of
  positions.
- **Collinearity:** W_u u was nearly collinear with the h-branch (cos 0.86–0.98).
- **Scale:** u has norm ≈1.2, while h is about 15–20× larger. Under Adam, equal per-element step sizes therefore give
  the u-branch a much smaller output effect.

So M1 behaved like R2 plus a weak, largely redundant u-branch. This matches the near-zero epoch-matched M1 − R2
differences (e.g. seed 42 epochs 1–2: new ZH +2/0, SB 0/0). The result rejects *this* parameterization under the frozen
recipe. It is weak evidence about cross-attention conditioning in general, and it says nothing about acoustic
dependence.

**H3 is supported:** neither architectural change resolves the correction–preservation bottleneck. Retention ≥ 0.99 in
≥ 2 seeds is reached by no arm.

**H4 is partly supported:**
- seed-level swings (e.g. R3 EN net 21–39, new ZH 46–58) are as large as the between-architecture differences;
- several contrasts are dominated by one dialogue;
- every bootstrap CI includes or touches 0.

## 5. Bounded mechanism checks (exploratory; outside the verdict)

**13.1 contribution audit.**

| Arm | ‖update‖ (mean) | ‖update‖ / ‖h‖ | EN vs ZH positions |
|---|---|---|---|
| R2 | 2.5–3.2 | 0.14–0.18 | larger at EN-token positions (EN 3.6–4.6, ZH 1.8–2.6) |
| R3 | 2.8–3.2 | 0.16–0.18 | same EN > ZH pattern |
| M1 | 2.5–3.1 | 0.14–0.17 | same EN > ZH pattern |

For M1, the W_u u share was slightly larger at ZH positions (0.6–2.4%) than at EN positions (0.5–1.0%).

The audit's internal checks hold:
- the isometry ‖eR‖ = ‖e‖ holds exactly;
- orthogonality is ≤ 1e-6 everywhere.

**13.2 ten fixed P2-1c cases.** The R2 rows reproduce P2-1c.
- **d0, d3, d4 (persistent omissions):** dropped by every R3 and M1 run, as by R2. The damage is recovered from B0's
  post-English prefix and dropped again from the systems' own prefix; R3_44 keeps d4 partially.
- **d2 (near-tie):** lost from B0's prefix by all R3 and M1 runs. In the original decode it is lost by R3_44 only.
- **d1 and the five safe cases:** unchanged.

The architectures did not change the prefix-history dependence identified in P2-1c.

## 6. Training dynamics

All runs have 96 updates, and their train CE and GoldKL are near-identical across arms:
- **Train CE:** 0.40 → 0.35–0.36;
- **GoldKL:** 0.007–0.008 → 0.016–0.017;
- **Mean pre-clip gradient norm:** 0.21 → 0.26–0.33.

R3 reaches marginally lower train CE (0.348–0.357 at epoch 3, vs 0.361–0.363 for R2 and M1).

**Pairing held exactly for every epoch and seed:**
- GoldKL mask positions equal R2's (16,739 / 16,739 / 16,737);
- 18,616 supervised tokens per epoch;
- M1's initial `base` equals R2's;
- orthogonality error ≤ 1e-6.

## 7. Efficiency

| | R2 | R3 | M1 |
|---|---|---|---|
| Trainable parameters | 5,122 | 7,683 | 7,682 |
| Parameter bytes (FP32) / + Adam state | 20.5 KB / 61 KB | 30.7 KB / 92 KB | 30.7 KB / 92 KB |
| Checkpoint file (includes the 1280×1280 parametrization `base`) | 6.58 MB | 6.59 MB | 6.59 MB |
| Train seconds per run (3 epochs) | 77–85 | 83–84 | 75–83 |
| Teacher-forward seconds per run | 7.7–7.9 | 7.6–7.7 | 7.7 |
| Micro-batch forward + backward (preflight) | 0.429 s | 0.437 s | 0.435 s |
| Job wall time | 1,134–1,172 s | 1,106–1,132 s | 1,115–1,171 s |
| Peak allocated / reserved VRAM | 5.73 / 8.02 GB | 5.73 / 8.02 GB | 5.73 / 8.02 GB |
| Mean decode time, beam-5 / greedy (epoch 3) | 1.02–1.07 / 0.64–0.67 s | 1.00–1.06 / 0.61–0.62 s | 1.04–1.07 / 0.64–0.68 s |

Training compute is essentially identical across the three architectures. Dev decoding (4 × 150 × beam-5 + greedy)
dominates the wall time.

**GPU use for this stage:** 145.7 MIG-minutes ≈ 2.43 slice-hours.

| Component | MIG-minutes |
|---|---|
| Six runs | 115.6 |
| Two pre-freeze dry-run preflights and one official preflight | 12.9 |
| Verify | 7.2 |
| Mechanism | 10.0 |

## 8. Limitations

- The exposed dev150 panel has only 20 dialogues. It is used for both checkpoint selection and evaluation, and its
  dialogues are shared across seeds, so the intervals are descriptive.
- There are only three seeds. Seed and dialogue variance is of the same order as the architecture contrasts.
- There is no optimized LoRA baseline and no independent or dialogue-disjoint test confirmation.
- There is no acoustic-causal evidence. M1 reads the query-dependent cross-attention output, which is already part of
  h.
- M1 and R3 differ in output rank (2 vs 3) despite matched parameter counts.
- Only one frozen recipe was tested: λ = 1, lr 5e-4, 96 updates, layer 31. In particular, the u-branch's small input
  scale under Adam is a property of this parameterization and recipe. The result does not cover a rescaled or normalized
  readout, which was not run and is not authorized.
- The P2-1c cases are reference-informed and exploratory.

## 9. Reproducibility

**Commit before execution:** `fc02be8` (protocol, rules, code, tests).

**Chain:**
- freeze `arch_s0s1/freeze.json` `ca2eab0c…`, which verifies the P2-0 → P2-1b (via P2-2), P2-1c and P2-2 chains and the
  input hashes;
- `AUTHORIZATION.json`;
- preflight `preflight_59509.json` `40e55d02…`.

**Jobs:**

| Purpose | Job IDs |
|---|---|
| Tests | 59487 (one test failure, fixed in the test only: weight decay), 59495 (112/112) |
| Pre-freeze dry-run preflights (scratch outputs deleted) | 59496, 59498 |
| Freeze | 59508 |
| Preflight | 59509 |
| Train | 59513–59518 |
| Select | 59519 |
| Evaluate | 59520 |
| Verify | 59521 |
| Mech-anchors | 59522 |
| Mech-gpu | 59523 |
| Report | 59524 |

The second dry run followed one pre-freeze gate-design fix: G5's leakage check now uses an utterance whose decode the
perturbed adapter changes.

**Output hashes:**

| File | Hash prefix |
|---|---|
| `evaluation.json` | `46c34961` |
| `verify.json` | `95578496` (all_ok; 18 checkpoints × 20/20) |
| `report.json` | `d70f117a` |
| `manifest.json` | `df7120a9` |

**Environment:** canonical CUDA 12.8 (pip `nvidia/cublas` 12.8 libraries), `OMP_NUM_THREADS=2`, frozen BF16
Whisper-large-v3 with expected weights hash.

**Private artifacts** (local only): `/mnt/data/tungnx/cs-asr-reft/arch_s0s1/`. They include the training runs, step
diagnostics, decodes, mechanism files and figures F1–F5.

**Not run:** E1+D2, P2-3, P3, LoRA, sweeps, and any additional run or data. No locked-role data was opened, and
historical freezes and artifacts are unchanged.
