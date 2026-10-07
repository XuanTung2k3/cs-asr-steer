# P2-TTA1 — selected-objective replication on the fixed-100 panel (report, terminal)

**Verdict: `P2_TTA1_SUPPORTED`** against matched forced-ZH (B0-FORCED). Independent post-audit
`P2_TTA1_AUDIT: PASS`. The method is still **below ordinary Whisper AUTO on PIER**. This is exposed-development
replication: no fresh validation, no full-300, no P3.

| Step | Commit / job | Result |
|---|---|---|
| Selection seal (TTA0-R, A2) | `336647f` | `selected_objective.json`, A2 AUTO-CONSISTENCY |
| Implementation + plan seal | `4cb6efb` parent | runner/evaluator/auditor/sbatch/tests; reuse plan sealed |
| Pre-TTA1 audit | `4cb6efb` | `PASS_TO_P2_TTA1` |
| Resolved manifest pushed | `4a8b1f4` | `sha256:91b01505…`, before the job |
| Run | Slurm 57871 | 80 new A2 episodes, 2 m 16 s; outputs sealed and pushed (`882c8c4`) before references |
| Evaluation + post-audit | this commit | `P2_TTA1_SUPPORTED`, `P2_TTA1_AUDIT: PASS` |

## Setup

- **Panel:** fixed 100 (`P2_SEL_MINI_PANEL.json`, sha256 `266ea7ea…`), 20 dialogues × 5.
- **Systems:** B0-FORCED (P2-SEQ S0), B0-AUTO (historical AUTO) and SELECTED-TTA (A2). Both baselines are
  hash-anchored and reused.
- **A2:** the inherited settings exactly:
  - teacher = theta0 AUTO greedy content IDs; student under forced-ZH;
  - loss = mean valid-position −log p(y_A[t]);
  - trainables = 194 decoder-LN tensors;
  - fresh fp32 masters + AdamW (lr 1e-3, wd 0), 2 steps, bf16 forward;
  - exact reset per utterance;
  - final forced-ZH greedy decode.
- **Episode origin:** 20 reused independently audited TTA0-R A2 ancestors (every compatible row, with no outcome
  filtering) and 80 new.
- **Integrity:** the theta0 forced decode reproduced S0 for 80/80 new IDs. The live first-row independent check was
  bitwise (loss and gradient difference 0; tolerance 2e-2).
- **Compute:** 160 optimizer steps, 161 backward passes, 401 teacher forwards. Wall 117 s (setup 13 s, adapt 21 s,
  decode 42 s, theta0 decode 41 s). Peak VRAM 3.87 GB allocated, 4.22 GB reserved.

## Results (100 utterances)

| System | PIER | MER | EN-WER | ZH-CER | POI err | S / D / I | caps |
|---|---|---|---|---|---|---|---|
| B0-FORCED | 0.4957 | 0.2570 | 0.5029 | 0.2193 | 345/696 | 339 / 1026 / 118 | 1 |
| B0-AUTO | 0.4080 | 0.2591 | 0.4152 | 0.2336 | 284 | 324 / 1053 / 118 | 1 |
| SELECTED-TTA (A2) | 0.4540 | 0.2537 | 0.4626 | 0.2211 | 316 | 351 / 985 / 128 | 1 |

### TTA vs B0-FORCED (causal baseline)

| Quantity | Point | 95% dialogue bootstrap | Frozen rule | Pass |
|---|---|---|---|---|
| Net POI error reduction | +29 (31 corrections / 2 corruptions) | — | ≥ 5 | yes |
| ΔPIER | −0.0417 | [−0.091, −0.002] | decrease ≥ 0.005 | yes |
| ΔMER | −0.0033 | [−0.014, +0.008] | ≤ +0.01 | yes |
| ΔZH-CER | +0.0018 | [−0.008, +0.015] | ≤ +0.015 | yes |
| ΔEN-WER | −0.0402 | [−0.089, −0.001] | — | — |
| Matrix-ZH retention | 0.9936 | [0.980, 1.000] | ≥ 0.98 | yes |
| Embedded-EN retention | 0.9943 | [0.980, 1.000] | ≥ 0.95 | yes |
| POI corruption rate | 0.0057 (2/351) | [0, 0.020] | ≤ 0.05 | yes |
| Outside-POI lexical harm | 0.0064 (26/4055) | [0, 0.020] | ≤ 0.03 | yes |
| Added caps / new severe truncations | 0 / 0 | — | ≤ 1 / 0 | yes |

Every safety bound and both benefit rules pass, so the label is **`P2_TTA1_SUPPORTED`**. Bootstrap: 2000 paired draws,
seed 240924, 2000/2000 valid; descriptive, not a significance gate.

### TTA vs B0-AUTO (practical reference)

| Quantity | TTA − AUTO | 95% CI |
|---|---|---|
| PIER | **+0.0460** | [−0.026, +0.126] |
| MER | −0.0054 | [−0.020, +0.012] |
| ZH-CER | −0.0125 | [−0.025, −0.002] |
| EN-WER | +0.0474 | [−0.026, +0.129] |

On PIER, the primary practical metric, TTA is **still below AUTO** (+0.046). It recovers about 47% of the forced→AUTO
PIER gap (0.042 of 0.088). It is slightly better than AUTO on MER and ZH-CER. **Beating forced-ZH is not beating
ordinary Whisper AUTO.**

## Mechanism and behaviour

- **NLL:** fell 0.413 → 0.250 → 0.179 (mean), in 100/100 utterances. Master update L2 0.90 (0.59% relative);
  effective bf16 L2 0.93.
- **Outputs changed vs B0-FORCED:** 14/100. Mixed errors improved in 6 utterances and worsened in 3. Mean length
  46.26 → 46.87; deletions 1026 → 985.
- **Disagreement group** (12 where forced ≠ AUTO): 6 changed; 3 moved closer to AUTO and 2 farther. Summed token
  distance to AUTO 149 → 146, so there is little movement toward AUTO. Net POI +16, but net mixed errors −17 (worse).
- **Agreement group** (88 where forced = AUTO, so the teacher equals the forced output): 8 changed, which is
  self-consistency on the forced path. Net POI +13, net mixed errors +36.
- **Concentration:** two utterances account for 25 of the 29 net POI corrections. `U0021_S0_513` (D) goes 15 → 0 POI
  errors; `U1004_S0_221` (A) goes 29 → 19. The supported verdict therefore rests on a small number of large utterance
  changes. The dialogue-bootstrap PIER interval vs forced only just excludes zero (upper −0.002).

## Next action

Stop, as the frozen contract requires regardless of the verdict. No full-300, P3, transfer, fresh validation or
further LR/steps/objective/subset exploration. Any confirmation of A2 episodic TTA, or an attempt to close the
remaining gap to AUTO, needs a new human decision and a separately frozen contract.
