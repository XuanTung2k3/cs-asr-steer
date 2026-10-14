# P2-1 — supervised adaptation feasibility pilot: result

**Terminal verdict: `P2_1_CORRECTION_PRESERVATION_FAILURE`.** It follows from the rules in `P2_1_DECISION_RULES.md`,
which were fixed and pushed before submission (commit `4a21738`).

**R2 (LoReFT)** is the first P2 method to correct English under operational Whisper Auto at scale. At its selected
epoch:
- 62 substantive English corrections against 25 corruptions, across 14 of 20 dialogues;
- genuine-CS MER 0.1227 → 0.1099 and overall MER 0.1199 → 0.1077 under beam-5.

But:
- it introduces 73 new Mandarin errors (Mandarin retention 0.985, below the frozen 0.99 floor) while repairing 103;
- its CS-MER gain is not established at dialogue level (95% CI [−0.032, +0.011]).

**L1 (LoRA q/v, rank 1, learning rate 1e-4)** barely moved from its zero-initialized update:
- training CE stayed flat, 0.4277 → 0.4270;
- 5 corrections, 0 corruptions, ΔCS-MER −0.0013 [−0.004, +0.001].

**Scope.** dev150 is exposed exploratory development data that also chose each checkpoint. No untouched-test claim is
made, and no P2-2 is authorized.

**Run provenance.** All runs used the frozen P2-1 design (`configs/reft/p2_1_frozen.yaml`, sha256 `fca2d7d2…`) under the
P2-0 freeze chain `p2_0.json → p2_0_r1.json` (`9b77ecba…`), in the canonical CUDA 12.8 environment, against the pinned
baseline B0. The authorization record is `p2_1/AUTHORIZATION.json` (`5edd8a9f…`).

**Slurm chain.** Every dependent job used `afterok` with `--kill-on-invalid-dep`.

| Step | Job | Wall time |
|---|---|---:|
| train R2 | 59232 | 22:50 |
| train L1 | 59233 | 22:45 |
| select R2 / select L1 | 59234 / 59235 | 0:08 each |
| evaluate | 59236 | 0:08 |
| reload verification | 59237 | 2:38 |
| report | 59238 | 0:26 |

All seven jobs completed.

## 1. Integrity: all checks pass

- **Pinned baseline:** hashes match; the canonical CUDA libraries were enforced inside each training job.
- **Zero-init reproduction:** with the zero-initialized adapter, each arm reproduced the pinned beam-5 baseline on
  **150/150** utterances before training.
- **Backbone:** weights are unchanged in both training jobs and in verification, and equal the P0 anchor.
- **Checkpoints:** all six epoch adapters reload with hashes matching their manifests, and re-decode the first 20 dev
  utterances token for token as recorded during training (20/20 each).
- **Language ID:** re-checked for every dev utterance at every epoch with the adapter disabled; it never changed.

## 2. Training and efficiency

| | R2 LoReFT (layer 31, rank 2) | L1 LoRA q/v (layer 31, rank 1) |
|---|---|---|
| Trainable parameters / adapter size | 5,122 / 20,488 B (fp32) | 5,120 / 20,480 B |
| Optimizer updates / supervised tokens | 96 / 55,848 | 96 / 55,848 |
| Training CE, epochs 1 / 2 / 3 | 0.396 / 0.360 / 0.347 | 0.4277 / 0.4275 / 0.4270 |
| Mean gradient norm, epochs 1 / 2 / 3 | 0.355 / 0.382 / 0.378 | 0.0025 / 0.0084 / 0.0155 |
| Pure training compute | 81 s | 81 s |
| Job wall time (including 4 × dev150 beam-5 + greedy decodes) | 22.8 min | 22.8 min |
| Peak VRAM | 5.5 GB | 5.5 GB |
| Mean beam-5 decode time per utterance (B0: 1.157 s, measured in smoke 59208) | 1.250 s (+8%) | 1.214 s (+5%) |

**GPU use.** The P2-1 total is about 48 min on H100 MIG 3g.40gb slices, about 0.8 slice-hours. The two training jobs ran
in parallel.

**Optimization.** L1's gradient is tiny, because rank-1 LoRA starts with B = 0. At the frozen lr 1e-4 and 96 updates,
it is effectively under-optimized: the pre-declared risk materialized. **L1 is therefore not an informative
weight-space PEFT baseline**, and R2 vs L1 says nothing about representation vs weight-space efficiency.

## 3. Dev150, all epochs (B0 = canonical P2 baseline)

**Beam-5 (primary).** ΔMER and ΔCS-MER CIs are dialogue-cluster bootstrap 95% intervals.

| System / epoch | MER | CS MER | English WER | PIER | Mandarin CER | English corrections / corruptions | New Mandarin errors / Mandarin repairs (retention) | ΔMER [CI] | ΔCS-MER [CI] | Loops |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|
| B0 | 0.1199 | 0.1227 | 0.2109 | 0.1966 | 0.0962 | — | — | — | — | 0 |
| R2 epoch 1 | 0.1133 | 0.1158 | 0.1739 | 0.1571 | 0.0966 | 63 / 19 | 77 / 67 (0.984) | −0.007 [−0.022, +0.011] | −0.007 [−0.025, +0.015] | 0 |
| **R2 epoch 2 (selected)** | **0.1077** | **0.1099** | 0.1798 | 0.1639 | **0.0889** | **62 / 25** | **73 / 103 (0.985)** | −0.012 [−0.028, +0.005] | **−0.013 [−0.032, +0.011]** | 0 |
| R2 epoch 3 | 0.1379 | 0.1114 | 0.1773 | 0.1613 | 0.1263 | 60 / 20 | 83 / 103 (0.983) | +0.018 [−0.018, +0.079] | −0.011 [−0.031, +0.012] | 1 |
| L1 epoch 1 | 0.1196 | 0.1222 | 0.2109 | 0.1966 | 0.0959 | 0 / 0 | 3 / 5 | −0.000 | −0.001 | 0 |
| L1 epoch 2 | 0.1196 | 0.1225 | 0.2101 | 0.1958 | 0.0960 | 1 / 0 | 8 / 10 | −0.000 | −0.000 | 0 |
| **L1 epoch 3 (selected)** | 0.1191 | 0.1214 | 0.2067 | 0.1924 | 0.0962 | 5 / 0 | 5 / 6 (0.999) | −0.001 [−0.003, +0.001] | −0.001 [−0.004, +0.001] | 0 |

**Greedy (secondary).**

| System | Greedy MER | Greedy CS MER |
|---|---|---|
| B0 | 0.1493 | 0.1366 |
| R2, epochs 1 / 2 / 3 | 0.162 / 0.133 / 0.131 | 0.173 / 0.138 / 0.135 |
| L1 | about 0.150 throughout | about 0.137 throughout |

R2's greedy gains are smaller and noisier than its beam-5 gains. At epoch 1, greedy is worse than B0.

**R2 epoch 3 derailment.** One utterance began looping, which pushed overall MER to 0.138 even though CS MER still
improved. The frozen rule selected epoch 2.

## 4. What R2 changes (selected epoch 2, beam-5)

**Remaining substantive English error kinds**, B0 → R2:

| Kind | B0 | R2 |
|---|---:|---:|
| translation-like | 28 | **6** |
| omission | 89 | **63** |
| English → English lexical substitution | 39 | 45 |
| segmentation artifact | 37 | 47 |
| boundary | 26 | 23 |
| total substantive English errors | 222 | 185 |

- The gain is mostly undoing Auto's remaining translations and recovering omitted English.
- English → English lexical errors do **not** improve; they grow slightly. As with ACL-0, this is a language/lexicon
  adaptation, not a fine lexical repair mechanism.

**By stratum:**

| Stratum | B0 | R2 |
|---|---|---|
| CS (97 utterances), MER | 0.1227 | 0.1099 |
| CS English WER / PIER | 0.248 / 0.233 | 0.206 / 0.188 |
| CS Mandarin CER | 0.0968 | 0.0897 |
| English-only (18), MER | 0.1355 | 0.1305 |
| Mandarin-only (22), MER | 0.0878 | 0.0760 |

**Mandarin preservation.**
- Of 4,950 B0-correct non-filler Mandarin units, 73 become wrong (1.47% > 1% allowed) while 103 wrong units are repaired.
- Net Mandarin improves (Mandarin CER −0.007), but the edit is not conservative.
- Outside-target damage: 76 of 4,971 B0-correct non-English units.

**Spread over dialogues.**
- 15 of 20 dialogues improved, 2 worsened and 3 were tied. The largest single dialogue holds 20% of the improvement, and
  the largest single utterance 22% of the English net.
- The two worsened dialogues lose 21 and 30 mixed units. The +30 dialogue has no substantive English change, so its
  loss comes from non-English units (Mandarin errors and insertions).
- Utterances: 47 improved, 22 worsened.

**R2 − L1** (both selected, beam-5): ΔMER −0.011 [−0.028, +0.007]; ΔCS-MER −0.011 [−0.031, +0.013]. This is descriptive
only, given that L1 is under-optimized.

## 5. Why this verdict (rules fixed in advance)

**Useful(R2) fails** on two conditions:
- the ΔCS-MER CI upper bound is +0.011, not < 0;
- Mandarin retention is 0.985 < 0.99.

Every other condition holds:
- ΔCS-MER −0.013 ≤ −0.005;
- ΔMER ≤ 0;
- net +37 in 14 dialogues;
- largest share 0.22;
- Mandarin CER not above B0;
- no new loops.

**Useful(L1) fails:** ΔCS-MER is −0.001, not ≤ −0.005, and corrections fall in only 2 dialogues.

**Correction-but-damage(R2) holds.** There is correction evidence (62 gross corrections; ΔCS-MER ≤ −0.005), and
preservation fails on Mandarin retention.

**→ `P2_1_CORRECTION_PRESERVATION_FAILURE`.** The rule order puts this ahead of `INCONCLUSIVE`.

**What this verdict does and does not say.**
- The preservation failure is narrow (1.47% new-error rate against a 1% floor), and net Mandarin improved.
- The CI on the CS gain still includes 0 at 20 dialogues.
- The evidence is a real but noisy and not conservative adaptation signal from a 5,122-parameter LoReFT. It is not an
  established improvement.

## 6. Limitations

- **Single seed.** The 500 training utterances include 109 former P1 audit utterances.
- **Exposed dev150.** It served both for selection (3 candidates per arm) and for reporting, and has only 20 dialogues.
- **Weak LoRA control.** The single-layer rank-1 LoRA at lr 1e-4 was under-optimized, so no representation-vs-weight
  conclusion is possible.
- **First run in the canonical environment.** The B0 decode time was measured in a different job (smoke 59208).

## 7. Recommended next action (one; not authorized)

Pre-register, before any P2-2, a **P2-1b preservation-and-variance pilot for R2 only**:
- the identical R2 configuration plus the ACL-0 KL-to-frozen preservation term on matched-prefix (baseline-correct)
  positions, with its weight fixed in advance;
- run with 3 seeds;
- judged by the same decision rules.

It answers the two open questions:
- whether LoReFT's English gains survive a Mandarin-preservation constraint;
- whether the CS gain is stable across seeds.

A fair weight-space comparison would separately need a LoRA arm whose learning rate or scaling is fixed by a
training-split criterion, never by dev150. That is not part of this recommendation.

**Artifacts** (local only; not pushed):
- `p2_1/report.json` (`ddbea144…`), `p2_1/evaluation.json` (`11576664…`), `p2_1/verify.json` (`37157b5e…`);
- `p2_1/{R2,L1}/manifest.json` (`81be453d…` / `e360c131…`);
- per-epoch decodes and adapters under `p2_1/{R2,L1}/epoch*/`.
