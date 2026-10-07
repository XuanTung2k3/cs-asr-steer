# P2-TTA-A3 — soft AUTO-KL conditioning-gap debug (report, terminal)

**Verdict: `P2_TTA_A3_TEACHER_UNSAFE`.** Independent post-audit `P2_TTA_A3_AUDIT: PASS` (first attempt).
Exposed-development debug on an enriched, reference-free 24-utterance panel. It is not a population estimate, and no
significance claim is made.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, config, panel) | `bd41bc5` | before any A3 output |
| Implementation + plan seal | `eab6c64` | `soft_auto_tta.py`, runner, analysis, auditor, sbatch, 8 tests |
| Pre-run audit | `90dbe96` | `PASS_TO_P2_TTA_A3` |
| Resolved manifest pushed | `d802297` | before the job |
| Run | Slurm 57876 | 24 rows, 57 s allocation (39 s runtime) |
| Output seal | `0d29d35` | pushed before references |
| Evaluation + post-audit | this commit | `TEACHER_UNSAFE`, `P2_TTA_A3_AUDIT: PASS` |

## Panel

The panel has 24 utterances in 20 dialogues (sha256 `6422741b…`):
- **D (12):** all fixed-100 rows where the stored theta0 AUTO decoded text differs from FORCED. This was reproduced
  independently and matches the funnel split.
- **A (12):** controls taken first from the 11 dialogues with no D row, then 1 from CSD0011.

No references were used in selection.

## Contract (inherited A2 actuator)

| Setting | Value |
|---|---|
| Trainables | 194 decoder LayerNorm affine tensors / 248,320 scalars |
| Optimizer | fresh fp32 masters + AdamW (lr 1e-3, wd 0, betas (0.9, 0.999), eps 1e-8), 2 steps, bf16 forward, exact reset per utterance |
| Teacher | theta0 AUTO distribution q on the historical AUTO path y_A, under the AUTO prompt recovered by the historical `detect_language(input_features)` call |
| Student | forced-ZH on the same y_A prefix |
| Loss | **forward KL(q_AUTO ‖ p_FORCED)** over the generation-allowed vocabulary, float32 |
| Final decode | forced-ZH greedy |

AUTO conditions: D rows detected `<|en|>` (11) and `<|ms|>` (1). All 12 A controls detected `<|zh|>`, which equals the
forced prompt, so they fall under the pre-registered identical-condition rule: exact no-op, output = B0-FORCED.

Integrity:
- theta0 forced decode = S0, 24/24;
- theta0 AUTO replay under the recovered prompt = historical AUTO text, 24/24;
- live independent KL check on the first row: loss diff 3e-8, gradient rel 0.75% (≤ 2%);
- resets bitwise; non-LN parameters unchanged;
- 48 optimizer steps, 49 backward passes;
- peak VRAM 4.41 GB allocated / 4.77 GB reserved.

## Condition gap (D group)

| | D_cond0 | D_cond1 | D_cond2 |
|---|---|---|---|
| D, token-weighted (nats) | 0.668 | 0.421 | 0.306 |
| D, row mean | 1.306 | 0.859 | 0.619 |
| A controls | 0 | 0 | 0 (identical condition) |

D_cond decreased in **12/12** D rows, with a median relative reduction of **0.514** (range 0.41–0.75). Required: ≥ 9
rows and ≥ 0.25. **GAP_CLOSED: PASS.** The soft objective couples strongly to the decoder-LN actuator, roughly halving
the AUTO–forced KL gap in 2 steps.

## Sequence movement (D group)

| | Σ distance to AUTO | Closer / same / farther | R_dist |
|---|---|---|---|
| B0-FORCED | 149 | — | — |
| A2 | 146 | 3 / 7 / 2 | 0.020 |
| A3 | 132 | 4 / 7 / 1 | 0.114 |

**MOVEMENT passes, narrowly, via criterion (a):** exactly 4 rows are closer and only 1 is farther. The pooled
reduction (0.114) is below the 0.15 alternative. 7/12 D transcripts changed. Five rows show no transcript change
despite a 41–75% gap reduction (`U0029_S0_472`, `U0039_S0_80`, `U1003_S0_61`, `U1079_S0_23`, `U2004_S0_104`). So a
large teacher-forced KL reduction often does not reach free decoding.

## ASR (24-panel, enriched; not a population estimate)

| System | PIER | MER | EN-WER | ZH-CER | POI err | S / D / I |
|---|---|---|---|---|---|---|
| B0-FORCED | 0.5863 | 0.2861 | 0.5863 | 0.1848 | 146/249 | 73 / 215 / 3 |
| B0-AUTO | 0.3414 | 0.2979 | 0.3414 | 0.2792 | 85 | 58 / 242 / 3 |
| A2 (TTA1, reused) | 0.5141 | 0.2920 | 0.5181 | 0.2149 | 128 | 81 / 203 / 13 |
| A3 | 0.5141 | 0.2930 | 0.5181 | 0.2163 | 128 | 83 / 211 / 4 |

A3 vs B0-FORCED, against the frozen TTA0/TTA1 safety bounds:

| Quantity | A3 | Bound | Pass |
|---|---|---|---|
| POI corrections / corruptions | 18 / 0 (net +18) | — | — |
| ΔPIER | −0.072 | — | — |
| ΔMER | +0.0069 | ≤ 0.01 | yes |
| ΔZH-CER | **+0.0315** | ≤ 0.015 | **no** |
| Matrix-ZH retention | **0.9616** | ≥ 0.98 | **no** |
| Embedded-EN retention | 1.000 | ≥ 0.95 | yes |
| Outside-POI lexical harm | **0.0383** (24/626) | ≤ 0.03 | **no** |
| POI corruption | 0.000 | ≤ 0.05 | yes |
| Added caps / severe truncations | 0 / 0 | ≤ 1 / 0 | yes |

A3 vs A2: POI errors 0 difference (128 = 128), PIER 0 difference, MER +0.0010, ZH-CER +0.0013. The A2 comparison
condition itself passes, but the label is decided earlier by safety.

Gap, movement and the first safety stages pass in order, and safety then fails on three bounds, so the frozen
precedence gives **`P2_TTA_A3_TEACHER_UNSAFE`**.

Descriptive context (changes no label; bootstrap is descriptive):
- On this enriched panel, **A2 also breaches the same three bounds**: ΔZH-CER +0.030, ZH retention 0.963, outside
  harm 0.037. On the representative 100-panel A2 passed them in TTA1.
- B0-AUTO itself is much worse on Mandarin here: ZH-CER 0.279 vs 0.185.
- By group:
  - D group POI errors: B0 119, AUTO 58, A2 103, A3 101.
  - A group: A3 is identical to B0 (27), while A2 reaches 25.
  - D-group mixed errors: B0 137, A3 144, A2 154. So A3 does less mixed-error damage than A2 on D rows, but more than
    forced.
- Descriptive bootstrap: A3−B0 ZH-CER +0.032 [0.000, +0.099]; A3−A2 PIER 0.000 [−0.024, +0.023].

## Interpretation

| Question | Answer |
|---|---|
| Objective optimized? | Yes. D_cond fell in 12/12 D rows, median 51%. |
| Conditioning gap closed? | Yes, by the frozen criterion. |
| Sequence leverage? | Partial. Movement passes at the minimum (4 closer / 1 farther), the pooled reduction is only 11%, and 5/12 rows with a halved gap did not change at all. |
| Movement toward AUTO? | Yes, more than A2: Σ distance 132 vs 146; 1 farther vs 2. |
| Teacher safety? | **No.** Moving toward AUTO imports AUTO's Mandarin degradation on these code-switched utterances (ZH-CER, matrix-ZH retention, outside harm). |
| Sequence stability? | No caps, no truncations, insertions 3 → 4. |
| A3 vs A2? | Equal on POI and PIER; marginally worse MER/ZH-CER (+0.001); fewer corruptions (0 vs 2); and it acts only where AUTO ≠ forced. |

**Core bottleneck: the AUTO teacher.** The objective/actuator coupling works: the gap closes. Free-decode leverage is
limited but present. But the place A3 is sent, AUTO's language-switched behaviour on disagreement utterances, buys
English POI corrections at the cost of Mandarin accuracy. The soft AUTO target recovers no more ASR value than the
hard A2 target. So the remaining gap to AUTO is not closable by AUTO-matching alone without the frozen safety
trade-off.

## Next action

Stop. No tuning of LR, steps, trainables or objective, and no A3-on-100 run. Any follow-up needs a new human decision
and a separately frozen contract, for example a teacher that is safe for the matrix language. Data: exposed
D-dev-select only. No fresh validation, full-100 A3, full-300 or P3.
