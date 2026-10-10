# P2-SEQ — sequence-level viability screen (report, terminal)

**Terminal: `P2_SEQ_SEQUENCE_DAMAGE`.** Independent post-run audit `P2_SEQ_AUDIT: PASS` (own metrics, transitions,
retention, outside-POI harm, bootstrap and label recomputed without importing the primary analysis).

| Step | Commit / job | Result |
|---|---|---|
| Freeze | `660a619` | spec/design/config |
| Implementation | `ed9c355` | D2 sequence driver, analysis, independent auditor, Slurm, tests |
| Reuse proof | `14a2339` | AUTO reuse 100/100 (`REUSE_AUTO_ALL_100`); attempt-1 hash-definition bug preserved |
| Pre-run audit | `9cb1d75` | `PASS_TO_P2_SEQ` |
| Run | Slurm 57867 (`c0f942c`) | 100/100 ok, 607 s elapsed on one MIG 3g.40gb |
| Post audit | `c0f942c` | `P2_SEQ_AUDIT: PASS`, label agrees |

Panel: fixed `docs/inference_cf/P2_SEL_MINI_PANEL.json` (100 utterances, 20 dialogues × 5, sha256 `266ea7ea…`),
D-dev-select only. Manifest `sha256:3493014108003bee…`. Outputs `results/inference_cf/p2seq/run1/`,
`run1_analysis.json`, `run1_audit.json`.

## Systems

- **S0 B0-FORCED**: new driver, forced-ZH prompt, alpha 0. Bitwise equal to the clean B branch at every step.
- **S1 B0-AUTO**: historical `model.generate` (language=None) outputs, reused after the sealed pre-outcome proof.
- **S2 STEER**: same driver, D2 READOUT direction, L16 DG-02 site, gate E·R_B, alpha 2, NormPreserve.

## Results (corpus, 100 utterances)

| System | PIER | MER | EN-WER | ZH-CER | POI errors | caps |
|---|---|---|---|---|---|---|
| S0 B0-FORCED | 0.4957 | 0.2570 | 0.5029 | 0.2193 | 345 / 696 | 1 |
| S1 B0-AUTO | 0.4080 | 0.2591 | 0.4152 | 0.2336 | 284 / 696 | 1 |
| S2 STEER | 0.4871 | 0.2852 | 0.4986 | 0.2522 | 339 / 696 | 0 |

### STEER vs B0-FORCED (matched)

| Quantity | Point | 95% dialogue bootstrap (S2−S0) | Frozen rule | Pass |
|---|---|---|---|---|
| ΔPIER | −0.0086 (gain 0.0086) | [−0.0606, +0.0304] | gain ≥ 0.005 | yes |
| Net POI error reduction | 6 (20 corrections, 14 corruptions) | — | ≥ 5 | yes |
| ΔMER | +0.0282 | [+0.0062, +0.0540] | ≤ +0.005 | **no** |
| ΔZH-CER | +0.0329 | [+0.0098, +0.0596] | ≤ +0.005 | **no** |
| ΔEN-WER | −0.0043 | [−0.0440, +0.0300] | (reported) | — |
| Matrix ZH retention | 0.9398 (3810/4054) | [0.851, 0.991] | ≥ 0.99 | **no** |
| Embedded EN retention | 0.9601 (337/351) | [0.914, 0.992] | ≥ 0.95 | yes |
| POI corruption rate | 0.0399 (14/351) | [0.008, 0.086] | ≤ 0.05 | yes |
| Lexical outside-POI harm | 0.0602 (244/4055) | [0.009, 0.149] | ≤ 0.01 | **no** |
| Cap change | −1 | — | ≤ +1 | yes |

Bootstrap: 2000 draws, seed 240924, 20 dialogues resampled; 2000/2000 valid. Both benefit rules pass but four safety rules
fail, so the frozen precedence (INVALID → DAMAGE → PROMISING → NO_GAIN) gives `P2_SEQ_SEQUENCE_DAMAGE`.

**Does STEER improve matched forced-ZH?** On the POI metric only, and only by a point estimate: PIER −0.0086, net +6 POI,
with the 95% interval spanning zero. It worsens MER and ZH-CER, and both 95% intervals exclude zero.

### STEER vs B0-AUTO

| Quantity | S2−S1 point | 95% bootstrap |
|---|---|---|
| PIER | +0.0790 | [+0.0125, +0.1517] |
| MER | +0.0262 | [+0.0004, +0.0549] |
| ZH-CER | +0.0186 | [−0.0135, +0.0498] |
| EN-WER | +0.0833 | [+0.0159, +0.1556] |

STEER is **below AUTO on the primary practical metric**. PIER STEER−AUTO is +0.079 and the 95% lower bound is above 0, so
on this panel STEER is clearly worse than AUTO. MER is also worse (lower bound +0.0004). Steering closes only about 10% of
the forced→AUTO PIER gap (0.0086 of 0.0877).

## Steering behaviour

- 4459 decoded steps, 4005 eligible. The gate was positive on 237 steps, which got 237 D2 calls, 237 autograd calls,
  237 valid directions and 237 actual edits, across 69/100 utterances.
- Mean gate 0.027 over all steps, max 0.994. Realized edit energy 281.9, relative energy 0.0213.
- 41 transcripts changed vs S0. Of these, 22 have more mixed errors, 6 have fewer and 13 are equal. At POI level, 5 are
  worse and 4 better.
- Damage is mostly deletion: corpus deletions go from 1026 to 1299 and insertions from 118 to 53. Changed transcripts lose
  288 characters net. Two utterances account for 133 of the 163 added mixed errors. In both, an edit is followed by an
  early end of transcript (e.g. `U1003_S0_63`: a long utterance decoded as "是的,是的。").
- Runtime (one MIG slice): S0 wall 394.7 s, S2 206.5 s. S2 reused the shared LID cache (247 new calls, 3858 hits), so the
  wall times are not a cost comparison. Peak allocated memory was 4.26 GB (S0) and 4.53 GB (S2).

## Integrity (all true; auditor re-derived)

S0 bitwise to clean B at every step and no D2. S2 bitwise until the first edit. D2 scratch logits/site bitwise to B on all
237 calls. Cache lineage and distinct caches hold. Zero-gate steps are site-identical. Every token divergence is
attributed to a preceding edit. Exact bf16 edit-energy emulation is within 1e-5. Canonical POI transition identity
holds, and the gate equals E·R_B with a unit D2 on every edit.

## Interpretation (descriptive; does not change the label)

In the matched sequence setting, the existing E·R_B·D2 controller reaches the per-edit benefit predicted by earlier
token-level screens (net POI +6). It pays for it with sequence damage the token screens could not see: Mandarin deletions
and early termination after edits. It also sits far below simply letting Whisper choose its own language (AUTO), which
improves PIER by 0.088 over forced-ZH at a MER cost of only +0.002 (ZH-CER +0.014). Together with the terminal selectivity stages
(P2-SEL, -E, -T, -XA, -LAC), this closes the current local-gate inference-steering line on this panel.

## Next action

Stop. Per the frozen contract, `P2_SEQ_SEQUENCE_DAMAGE` → evidence-only handoff `docs/inference_cf/P2_SEQ_TTA_HANDOFF.md`
for a separately frozen P2-TTA0 design. No TTA was designed, implemented or run here. No full-300 run, no fresh
validation, P3 HELD.
