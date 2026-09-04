# Two-track steering study for Mandarin-English CS-ASR

**Stamp:** `development_only_diagnostic`. This document evaluates no production gate, freezes no spans, allocates no held-out generation, and opens neither `D-dev-confirm` nor `D-test`.

- model: `whisper-large-v3`, frozen unless an arm explicitly trains something (32 encoder / 32 decoder layers, d_model 1280, 1500 encoder frames)
- population: `D-dev-select`, 300 utterances / 20 dialogues, 489 targets
- anchors asserted: 489 targets, 6,264 baseline-correct units, 5,711 baseline-correct Mandarin units
- git commit: `fdd441a75b0d66f56d8e5efec62f01649f7b9ae9`
- seeds: `{'global': 20260818, 'bootstrap': 20260818, 'direction_random_init': 20260821, 'data_subsample': 20260822, 'dev_eval_subsample': 20260823}`

**Raw numbers first; interpretation is confined to section 9.**

---

## 1. Preflight

**Track A: PASSED** (15/15)

| check | result | observed | expected |
|---|---|---|---|
| `6.matched_energy_invariance` | PASS | {'1': 4.0, '10': 4.0, '100': 4.0, '1000': 4.0} | {'1': 4.0, '10': 4.0, '100': 4.0, '1000': 4.0} |
| `4.encoder_padding.frame_count` | PASS | 0 mismatches | 0 mismatches |
| `4.encoder_padding.short_utterances_masked` | PASS | 273/300 utterances below 1500 frames | >0 utterances below 1500 frames |
| `1.6.prompt_at_encoder_raises` | PASS | PromptDirectionAtEncoderError | PromptDirectionAtEncoderError |
| `1.alpha_zero[encoder/global]` | PASS | 0 utterances differ | 0 utterances differ |
| `1.alpha_zero[encoder/local]` | PASS | 0 utterances differ | 0 utterances differ |
| `1.alpha_zero[decoder/global]` | PASS | 0 utterances differ | 0 utterances differ |
| `1.alpha_zero[decoder/local]` | PASS | 0 utterances differ | 0 utterances differ |
| `1.alpha_zero[encoder+decoder/global]` | PASS | 0 utterances differ | 0 utterances differ |
| `1.alpha_zero[encoder+decoder/local]` | PASS | 0 utterances differ | 0 utterances differ |
| `3.hook_fires.encoder` | PASS | 2 | 2 |
| `3.hook_fires.decoder` | PASS | 123 | 123 |
| `5.beam_coverage` | PASS | {'steered_positions': 800, 'num_beams': 5, 'remainder': 0} | steered_positions > 0 and divisible by num_beams |
| `2.reproduce_prior.rows` | PASS | 489 | 489 |
| `2.reproduce_prior.corrections` | PASS | 7 | 7 |

**Track B: PASSED** (7/7)

| check | result | observed | expected |
|---|---|---|---|
| `B.params.intervention[loreft]` | PASS | 10244 | 10244 |
| `B.params.intervention[additive]` | PASS | 10240 | 10240 |
| `B.params.lid_head` | PASS | 2562 | 2562 |
| `B.params.aga_adapters` | PASS | 105390080 | 105390080 |
| `B.params.aga_fraction_within_factor_2` | PASS | 6.83% | 5.6% within a factor of 2.0 |
| `B.grad.backbone_is_zero` | PASS | 0.0 | 0.0 |
| `B.grad.intervention_receives_gradient` | PASS | 0.38041032435449856 | > 0.0 |

## 2. Baselines

`force_en` is the ceiling of Whisper's native language control and the obvious reviewer challenge to any prompt-direction result, so it is in the table rather than in a footnote. Beam-search cells are compared against `C00_beam5`, never against greedy `C00`.

| baseline | prefix | beams | PIER | MER | WER | decode health |
|---|---|---:|---:|---:|---:|---|
| `C00` | `<|zh|>` | 1 | 0.47310 | 0.25818 | 0.85545 | healthy |
| `C00_beam5` | `<|zh|>` | 5 | 0.42460 | 0.20210 | 0.84175 | healthy |
| `force_en` | `<|en|>` | 1 | 0.32055 | 0.79433 | 1.55669 | UNHEALTHY |
| `force_zh` | `<|zh|>` | 1 | 0.47310 | 0.25818 | 0.85545 | healthy |

## 3. Track A -- one row per cell

Evaluated on all 300 utterances of `D-dev-select` over 20 dialogues, not only the 116 that carry a target: global steering touches everything and harm needs the correct denominator.

A cell whose decode is unhealthy is **INVALID**, not negative. Its PIER is not interpreted and it never enters a stage ranking.

| stage | cell | dPIER | 95% CI | corr | corrupt | ratio | dialogues | ZH ret. | G1 | G2 | G3 | G4 | verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| A | `A:encoder:global:v_nat:a1:E15` | -0.00661 | [-0.01634, +0.00131] | 0 | 97 | 0.00 | 0/20 | 0.9933 | fail | fail | fail | PASS | **NO-GO** |
| A | `A:encoder:local:v_nat:a1:E15` | +0.00000 | [-0.00703, +0.00851] | 6 | 51 | 0.12 | 1/20 | 0.9961 | fail | fail | fail | PASS | **NO-GO** |
| A | `A:decoder:global:v_nat:a1:D16` | -0.00088 | [-0.00685, +0.00405] | 1 | 41 | 0.02 | 1/20 | 0.9970 | fail | fail | fail | PASS | **NO-GO** |
| A | `A:decoder:local:v_nat:a1:D16` | +0.00353 | [+0.00045, +0.00771] | 7 | 20 | 0.35 | 4/20 | 0.9983 | PASS | fail | fail | PASS | **NO-GO** |
| A | `A:encoder+decoder:global:v_nat:a1:E15+D16` | -0.00265 | [-0.01119, +0.00614] | 6 | 63 | 0.10 | 1/20 | 0.9959 | fail | fail | fail | PASS | **NO-GO** |
| A | `A:encoder+decoder:local:v_nat:a1:E15+D16` | -0.00265 | [-0.00812, +0.00094] | 0 | 53 | 0.00 | 0/20 | 0.9960 | fail | fail | fail | PASS | **NO-GO** |
| B | `B:decoder:global:v_prompt:a1:D16` | +0.00000 | [-0.00127, +0.00128] | 0 | 24 | 0.00 | 0/20 | 0.9980 | fail | fail | fail | PASS | **NO-GO** |
| B | `B:decoder:local:v_prompt:a1:D16` | -0.00265 | [-0.00788, +0.00000] | 0 | 40 | 0.00 | 0/20 | 0.9971 | fail | fail | fail | PASS | **NO-GO** |
| B | `B:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | +0.00309 | [+0.00000, +0.00750] | 7 | 1 | 7.00 | 4/20 | 1.0000 | fail | PASS | fail | PASS | **NO-GO** |
| B | `B:decoder:local:v_nat+v_prompt_weighted_n1p1:a1:D16` | -0.00044 | [-0.00688, +0.00564] | 5 | 43 | 0.12 | 2/20 | 0.9968 | fail | fail | fail | PASS | **NO-GO** |
| B | `B:decoder:local:v_nat+v_prompt_weighted_n0.5p2:a1:D16` | -0.00265 | [-0.00788, +0.00000] | 0 | 40 | 0.00 | 0/20 | 0.9971 | fail | fail | fail | PASS | **NO-GO** |
| B | `B:decoder:local:v_nat+v_prompt_rank2:a1:D16` | -0.00265 | [-0.00788, +0.00000] | 0 | 54 | 0.00 | 0/20 | 0.9959 | fail | fail | fail | PASS | **NO-GO** |
| C | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.25:D16` | -0.00265 | [-0.00788, +0.00000] | 0 | 47 | 0.00 | 0/20 | 0.9965 | fail | fail | fail | PASS | **NO-GO** |
| C | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.5:D16` | -0.00088 | [-0.00729, +0.00526] | 4 | 50 | 0.08 | 1/20 | 0.9962 | fail | fail | fail | PASS | **NO-GO** |
| C | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | +0.00309 | [+0.00000, +0.00750] | 7 | 1 | 7.00 | 4/20 | 1.0000 | fail | PASS | fail | PASS | **NO-GO** |
| C | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a2:D16` | +0.00353 | [+0.00045, +0.00771] | 7 | 20 | 0.35 | 4/20 | 0.9983 | PASS | fail | fail | PASS | **NO-GO** |
| C | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a3:D16` | +0.00309 | [+0.00000, +0.00750] | 7 | 14 | 0.50 | 4/20 | 0.9989 | fail | fail | fail | PASS | **NO-GO** |
| D | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8` | -0.00265 | [-0.00788, +0.00000] | 0 | 40 | 0.00 | 0/20 | 0.9971 | fail | fail | fail | PASS | **NO-GO** |
| D | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | +0.00309 | [+0.00000, +0.00750] | 7 | 1 | 7.00 | 4/20 | 1.0000 | fail | PASS | fail | PASS | **NO-GO** |
| D | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D24` | +0.00088 | [-0.00574, +0.00712] | 7 | 30 | 0.23 | 4/20 | 0.9979 | fail | fail | fail | PASS | **NO-GO** |
| D | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28` | +0.00485 | [+0.00097, +0.00998] | 8 | 3 | 2.67 | 5/20 | 0.9997 | PASS | PASS | PASS | PASS | **GO** |
| D | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16-20` | +0.00309 | [+0.00000, +0.00726] | 6 | 20 | 0.30 | 3/20 | 0.9983 | fail | fail | fail | PASS | **NO-GO** |
| D | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8-16-24-28` | +0.00000 | [-0.00648, +0.00607] | 5 | 47 | 0.11 | 2/20 | 0.9965 | fail | fail | fail | PASS | **NO-GO** |

### Corpus metrics -- PIER, MER, WER, baseline vs steered

Primary metrics per section 3.7: absolute PIER/MER/WER for the steered decode and its comparator baseline (`C00` or `C00_beam5` for beam cells), plus delta MER and delta WER alongside the already-gated delta PIER. Positive delta means the steered cell reduced error.

| cell | comparator | PIER base -> steer | MER base -> steer | WER base -> steer | dMER | dWER |
|---|---|---|---|---|---:|---:|
| `A:encoder:global:v_nat:a1:E15` | C00 | 0.47310 -> 0.47972 | 0.25818 -> 0.26442 | 0.85545 -> 0.85733 | -0.00624 | -0.00188 |
| `A:encoder:local:v_nat:a1:E15` | C00 | 0.47310 -> 0.47310 | 0.25818 -> 0.25919 | 0.85545 -> 0.85599 | -0.00101 | -0.00054 |
| `A:decoder:global:v_nat:a1:D16` | C00 | 0.47310 -> 0.47399 | 0.25818 -> 0.25943 | 0.85545 -> 0.85626 | -0.00125 | -0.00081 |
| `A:decoder:local:v_nat:a1:D16` | C00 | 0.47310 -> 0.46958 | 0.25818 -> 0.26062 | 0.85545 -> 0.85492 | -0.00244 | +0.00054 |
| `A:encoder+decoder:global:v_nat:a1:E15+D16` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.25284 | 0.85545 -> 0.85626 | +0.00535 | -0.00081 |
| `A:encoder+decoder:local:v_nat:a1:E15+D16` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.26507 | 0.85545 -> 0.85599 | -0.00689 | -0.00054 |
| `B:decoder:global:v_prompt:a1:D16` | C00 | 0.47310 -> 0.47310 | 0.25818 -> 0.26323 | 0.85545 -> 0.85545 | -0.00505 | +0.00000 |
| `B:decoder:local:v_prompt:a1:D16` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.26502 | 0.85545 -> 0.85599 | -0.00683 | -0.00054 |
| `B:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | C00 | 0.47310 -> 0.47002 | 0.25818 -> 0.25765 | 0.85545 -> 0.85492 | +0.00053 | +0.00054 |
| `B:decoder:local:v_nat+v_prompt_weighted_n1p1:a1:D16` | C00 | 0.47310 -> 0.47354 | 0.25818 -> 0.26264 | 0.85545 -> 0.85545 | -0.00446 | +0.00000 |
| `B:decoder:local:v_nat+v_prompt_weighted_n0.5p2:a1:D16` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.26502 | 0.85545 -> 0.85599 | -0.00683 | -0.00054 |
| `B:decoder:local:v_nat+v_prompt_rank2:a1:D16` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.26513 | 0.85545 -> 0.85599 | -0.00695 | -0.00054 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.25:D16` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.26490 | 0.85545 -> 0.85599 | -0.00671 | -0.00054 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.5:D16` | C00 | 0.47310 -> 0.47399 | 0.25818 -> 0.26288 | 0.85545 -> 0.85545 | -0.00469 | +0.00000 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | C00 | 0.47310 -> 0.47002 | 0.25818 -> 0.25765 | 0.85545 -> 0.85492 | +0.00053 | +0.00054 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a2:D16` | C00 | 0.47310 -> 0.46958 | 0.25818 -> 0.26080 | 0.85545 -> 0.85492 | -0.00261 | +0.00054 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a3:D16` | C00 | 0.47310 -> 0.47002 | 0.25818 -> 0.26008 | 0.85545 -> 0.85492 | -0.00190 | +0.00054 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8` | C00 | 0.47310 -> 0.47575 | 0.25818 -> 0.26478 | 0.85545 -> 0.85599 | -0.00659 | -0.00054 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | C00 | 0.47310 -> 0.47002 | 0.25818 -> 0.25765 | 0.85545 -> 0.85492 | +0.00053 | +0.00054 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D24` | C00 | 0.47310 -> 0.47222 | 0.25818 -> 0.25925 | 0.85545 -> 0.85545 | -0.00107 | +0.00000 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28` | C00 | 0.47310 -> 0.46825 | 0.25818 -> 0.25521 | 0.85545 -> 0.85492 | +0.00297 | +0.00054 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16-20` | C00 | 0.47310 -> 0.47002 | 0.25818 -> 0.26062 | 0.85545 -> 0.85492 | -0.00244 | +0.00054 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8-16-24-28` | C00 | 0.47310 -> 0.47310 | 0.25818 -> 0.26454 | 0.85545 -> 0.85545 | -0.00636 | +0.00000 |

### Energy and coverage audit

| cell | \|S\| planned | \|S\| observed | alpha | alpha_eff | E_total | valid frame ratio (min/mean) |
|---|---:|---:|---:|---:|---:|---|
| `A:encoder:global:v_nat:a1:E15` | 245,535 | 245,535 | 1 | 0.002018 | 1 | 0.085 / 0.546 |
| `A:encoder:local:v_nat:a1:E15` | 17,943 | 17,943 | 1 | 0.007465 | 1 | 0.085 / 0.546 |
| `A:decoder:global:v_nat:a1:D16` | 13,482 | 20,648 | 1 | 0.008612 | 1 | 0.085 / 0.546 |
| `A:decoder:local:v_nat:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `A:encoder+decoder:global:v_nat:a1:E15+D16` | 259,017 | 266,207 | 1 | 0.001965 | 1 | 0.085 / 0.546 |
| `A:encoder+decoder:local:v_nat:a1:E15+D16` | 18,059 | 18,057 | 1 | 0.007441 | 1 | 0.085 / 0.546 |
| `B:decoder:global:v_prompt:a1:D16` | 13,482 | 21,380 | 1 | 0.008612 | 1 | 0.085 / 0.546 |
| `B:decoder:local:v_prompt:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `B:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `B:decoder:local:v_nat+v_prompt_weighted_n1p1:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `B:decoder:local:v_nat+v_prompt_weighted_n0.5p2:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `B:decoder:local:v_nat+v_prompt_rank2:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.25:D16` | 116 | 114 | 0.25 | 0.02321 | 0.0625 | 0.085 / 0.546 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.5:D16` | 116 | 114 | 0.5 | 0.04642 | 0.25 | 0.085 / 0.546 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a2:D16` | 116 | 114 | 2 | 0.1857 | 4 | 0.085 / 0.546 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a3:D16` | 116 | 114 | 3 | 0.2785 | 9 | 0.085 / 0.546 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D24` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28` | 116 | 114 | 1 | 0.09285 | 1 | 0.085 / 0.546 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16-20` | 232 | 228 | 1 | 0.06565 | 1 | 0.085 / 0.546 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8-16-24-28` | 464 | 456 | 1 | 0.04642 | 1 | 0.085 / 0.546 |

### POI decomposition and spillover on the 489 targets

| cell | correct | deletion | wrong-language sub | other sub | spillover changed | improved | damaged |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A:encoder:global:v_nat:a1:E15` | 0 | 278 | 119 | 89 | 1244 | 558 | 686 |
| `A:encoder:local:v_nat:a1:E15` | 6 | 285 | 107 | 88 | 1355 | 744 | 611 |
| `A:decoder:global:v_nat:a1:D16` | 1 | 293 | 104 | 88 | 380 | 26 | 354 |
| `A:decoder:local:v_nat:a1:D16` | 13 | 276 | 110 | 87 | 823 | 560 | 263 |
| `A:encoder+decoder:global:v_nat:a1:E15+D16` | 6 | 275 | 118 | 87 | 1664 | 1212 | 452 |
| `A:encoder+decoder:local:v_nat:a1:E15+D16` | 0 | 291 | 108 | 87 | 697 | 62 | 635 |
| `B:decoder:global:v_prompt:a1:D16` | 6 | 280 | 111 | 89 | 156 | 6 | 150 |
| `B:decoder:local:v_prompt:a1:D16` | 0 | 286 | 112 | 88 | 518 | 24 | 494 |
| `B:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 13 | 288 | 97 | 88 | 26 | 24 | 2 |
| `B:decoder:local:v_nat+v_prompt_weighted_n1p1:a1:D16` | 5 | 274 | 118 | 89 | 1115 | 582 | 533 |
| `B:decoder:local:v_nat+v_prompt_weighted_n0.5p2:a1:D16` | 0 | 286 | 112 | 88 | 518 | 24 | 494 |
| `B:decoder:local:v_nat+v_prompt_rank2:a1:D16` | 0 | 290 | 107 | 89 | 648 | 63 | 585 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.25:D16` | 0 | 290 | 107 | 89 | 640 | 62 | 578 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.5:D16` | 4 | 278 | 115 | 89 | 1170 | 553 | 617 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 13 | 288 | 97 | 88 | 26 | 24 | 2 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a2:D16` | 13 | 275 | 110 | 88 | 817 | 554 | 263 |
| `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a3:D16` | 13 | 269 | 116 | 88 | 923 | 742 | 181 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8` | 0 | 286 | 112 | 88 | 578 | 84 | 494 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 13 | 288 | 97 | 88 | 26 | 24 | 2 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D24` | 7 | 293 | 98 | 88 | 402 | 48 | 354 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28` | 14 | 275 | 108 | 89 | 559 | 520 | 39 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16-20` | 12 | 275 | 110 | 89 | 866 | 603 | 263 |
| `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8-16-24-28` | 5 | 292 | 101 | 88 | 659 | 81 | 578 |

## 4. Stage A -> B -> C -> D selection trace

The rule is fixed in `steer_sweep/metrics.select` and cannot move after a result is seen:

1. drop runs with unhealthy decode;
2. drop runs with ZH retention < 0.99;
3. sort by `dialogues_with_correction` DESC, then corrections:corruptions DESC, then delta PIER DESC.

The primary key is dialogue coverage, not delta PIER. The prior programme's seven corrections came from two of twenty dialogues, so its effective cluster count was two.

**Stage A** -- selected `A:decoder:local:v_nat:a1:D16, A:encoder:local:v_nat:a1:E15`

- cells considered: 6
- dropped, unhealthy decode: none
- dropped, ZH retention < 0.99: none
- eligible after both filters: 6

  | rank | cell | dialogues | ratio | dPIER |
  |---:|---|---:|---:|---:|
  | 1 | `A:decoder:local:v_nat:a1:D16` | 4 | 0.35 | +0.00353 |
  | 2 | `A:encoder:local:v_nat:a1:E15` | 1 | 0.12 | -inf |
  | 3 | `A:encoder+decoder:global:v_nat:a1:E15+D16` | 1 | 0.10 | -0.00265 |
  | 4 | `A:decoder:global:v_nat:a1:D16` | 1 | 0.02 | -0.00088 |
  | 5 | `A:encoder+decoder:local:v_nat:a1:E15+D16` | 0 | 0.00 | -0.00265 |
  | 6 | `A:encoder:global:v_nat:a1:E15` | 0 | 0.00 | -0.00661 |

**Stage B** -- selected `B:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16`

- cells considered: 8
- dropped, unhealthy decode: none
- dropped, ZH retention < 0.99: none
- eligible after both filters: 8

  | rank | cell | dialogues | ratio | dPIER |
  |---:|---|---:|---:|---:|
  | 1 | `B:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 4 | 7.00 | +0.00309 |
  | 2 | `A:decoder:local:v_nat:a1:D16` | 4 | 0.35 | +0.00353 |
  | 3 | `B:decoder:local:v_nat+v_prompt_weighted_n1p1:a1:D16` | 2 | 0.12 | -0.00044 |
  | 4 | `A:encoder:local:v_nat:a1:E15` | 1 | 0.12 | -inf |
  | 5 | `B:decoder:local:v_prompt:a1:D16` | 0 | 0.00 | -0.00265 |
  | 6 | `B:decoder:local:v_nat+v_prompt_weighted_n0.5p2:a1:D16` | 0 | 0.00 | -0.00265 |

**Stage C** -- selected `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16`

- cells considered: 5
- dropped, unhealthy decode: none
- dropped, ZH retention < 0.99: none
- eligible after both filters: 5

  | rank | cell | dialogues | ratio | dPIER |
  |---:|---|---:|---:|---:|
  | 1 | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 4 | 7.00 | +0.00309 |
  | 2 | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a3:D16` | 4 | 0.50 | +0.00309 |
  | 3 | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a2:D16` | 4 | 0.35 | +0.00353 |
  | 4 | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.5:D16` | 1 | 0.08 | -0.00088 |
  | 5 | `C:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a0.25:D16` | 0 | 0.00 | -0.00265 |

**Stage D** -- selected `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28`

- cells considered: 6
- dropped, unhealthy decode: none
- dropped, ZH retention < 0.99: none
- eligible after both filters: 6

  | rank | cell | dialogues | ratio | dPIER |
  |---:|---|---:|---:|---:|
  | 1 | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28` | 5 | 2.67 | +0.00485 |
  | 2 | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16` | 4 | 7.00 | +0.00309 |
  | 3 | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D24` | 4 | 0.23 | +0.00088 |
  | 4 | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D16-20` | 3 | 0.30 | +0.00309 |
  | 5 | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8-16-24-28` | 2 | 0.11 | -inf |
  | 6 | `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D8` | 0 | 0.00 | -0.00265 |

## 5. Confirm -- NOT RUN

`Confirm:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28`

```json
{
  "stage": "Confirm",
  "site": "decoder",
  "coverage": "local",
  "direction_kind": "v_nat+v_prompt",
  "alpha": 1.0,
  "encoder_layers": [],
  "decoder_layers": [
    28
  ],
  "scale_mode": "act_norm",
  "matched_energy": true,
  "num_beams": 1,
  "language": "zh",
  "combo_mode": "weighted",
  "weight_nat": 2.0,
  "weight_prompt": 0.5,
  "note": "the ONE configuration proposed for D-dev-confirm; requires explicit human approval"
}
```

**Status: AWAITING_HUMAN_APPROVAL.** Track A Confirm is NOT run automatically. D-dev-confirm has never been opened and has ten dialogues to spend. This is the single configuration the fixed selection rule chose; a human authorises the run.

## 6. Track B -- arms

Arms are **under-trained by design**: the budget is a fast observation pass, not a converged result, and absolute MER here is not comparable to published numbers.

Site/layers: `decoder` layers `[8]`. Reason: Track A Stage D selected D:decoder:global:v_prompt:a0.25:D8

Training pool: 10,768 utterances (`loc-train` + `util-train`). Training-time early-stopping / lr-selection dev set: 150 utterances, dialogue-stratified from `router-calib` (full role: 3,031). The final per-arm report below always scores the full `D-dev-select` population, once, after training.

Intervention variant run: `additive` -- measured forward+backward time; the cheaper variant runs (timings {'loreft': 0.0008089259266853332, 'additive': 0.00031695812940597535}).

"Matched budget" means matched training CONDITIONS, not matched parameter count. The parameter gap is the finding, not a nuisance to control, and it is reported rather than equalised.

| arm | tag | trainable params | % of 1.55 B | MER | PIER | WER | ZH ret. | spillover | dev MER | steps | wall clock | peak mem |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | main | 10,240 | 0.0007% | 0.13058 | 0.12390 | 0.75228 | 0.9897 | 11,319 | 0.12482 | 2000 | 1159s | 9.0 GiB |
| B1 | seed_20260818 | 12,802 | 0.0008% | 0.13545 | 0.12743 | 0.76599 | 0.9858 | 11,493 | 0.12460 | 2000 | 1165s | 9.0 GiB |
| B1 | seed_20260819 | 12,802 | 0.0008% | 0.13373 | 0.12566 | 0.76599 | 0.9876 | 11,569 | 0.12967 | 2000 | 1146s | 9.0 GiB |
| B1 | seed_20260820 | 12,802 | 0.0008% | 0.13592 | 0.14506 | 0.76491 | 0.9878 | 11,423 | 0.13221 | 1250 | 713s | 9.0 GiB |
| B2 | main | 12,802 | 0.0008% | 0.13176 | 0.12743 | 0.76787 | 0.9905 | 11,261 | 0.12439 | 2000 | 1117s | 9.0 GiB |
| B4 | main | 1,968,642 | 0.1270% | 0.18084 | 0.10802 | 0.84068 | 0.9913 | 12,232 | 0.13326 | 1000 | 1209s | 47.1 GiB |
| B3-reimpl | stage1 | 52,697,602 | 3.3998% | 0.53181 | 0.56746 | 1.24073 | 0.7199 | 15,536 | 0.49039 | 1000 | 1270s | 49.9 GiB |
| B3-reimpl | stage2 | 105,392,642 | 6.7995% | 0.53187 | 0.50353 | 1.28211 | 0.7031 | 16,404 | 0.46336 | 1000 | 1376s | 52.6 GiB |

**B1 across 3 seeds:** MER mean 0.13503, range [0.13373, 0.13592]. B2 is evaluated against this range, never against a single random draw.


**Frozen `lambda_lid` = 0.5**, tuned once on B1 over [0.1, 0.3, 0.5] and reused unchanged in every arm.

**Frozen `lr` = 0.0001**, tuned once on B1 over [0.0001, 0.0005, 0.001], selected on dev MER with the same protocol for every arm.

> **The selected lr is at a grid edge.** The rule is to extend the grid symmetrically FOR EVERY ARM and rerun -- never for one arm alone. This was not done inside this allocation and is recorded as a required follow-up.

> the selected lr 0.0001 is at an edge of the grid [0.0001, 0.0005, 0.001]. The rule is to extend the grid SYMMETRICALLY FOR EVERY ARM and rerun -- never for one arm alone. Not done inside this allocation; recorded as a required follow-up.

## 7. Track B initialisation diagnostics (4.3)

Did the optimizer converge toward the constructed directions, or away from them? Both outcomes are publishable and the implementation does not bias toward either. A random rank-2 subspace of a 1280-dimensional space sits at essentially 90 degrees, so **90 is the null, not zero**.

| arm | seed | cos(theta, v_nat) | cos(theta, v_prompt) | \|\|theta - theta_0\|\| | principal angles (deg) |
|---|---:|---:|---:|---:|---|
| B0 | 20260818 | +0.0516 | +0.0257 | 2.3697 | 81.5, 87.2 |
| B1 | 20260818 | +0.0514 | +0.0258 | 2.3695 | 81.5, 87.2 |
| B1 | 20260819 | +0.0169 | -0.0395 | 2.2827 | 85.4, 87.3 |
| B1 | 20260820 | -0.0678 | +0.0113 | 2.1450 | 84.6, 88.9 |
| B2 | 20260818 | +0.7493 | -0.0033 | 2.2585 | 18.5, 38.4 |

**Aggregate: INTERMEDIATE** (mean principal angle 74.0 deg over 5 runs).

- converged toward span{v_nat, v_prompt} -> the interpretability work is validated and Track A's failure is a magnitude problem, not a direction problem;
- diverged -> this explains Track A's failure: the constructed direction was not the direction the model needed.

## 8. B3-reimpl: fidelity note and the B3-vs-B4 gate

**Status: COMPLETED.**

### Head selection

- criterion: per decoder self-attention head, per utterance: sum of attention mass on prompt key positions 1:3 versus mass on position 0 plus positions 3:; the head is counted when the former exceeds the latter
- computed on: CS-Dialogue loc-train (NOT SEAME)
- selected 331 of 640 heads (fraction 0.7639) over 200 utterances
- per-layer counts: [1, 2, 0, 2, 4, 0, 4, 1, 7, 6, 8, 6, 6, 8, 9, 14, 12, 16, 15, 17, 15, 14, 15, 17, 16, 15, 15, 14, 16, 19, 18, 19]

### Two-stage accounting

- stage 1: 1000 optimizer steps (encoder adapters, `cs_weight = 0`)
- stage 2: 1000 steps, initialised from stage-1 weights (decoder adapters added, guidance loss on)
- total 2000, matched against the single-stage arms

### Trainable-parameter fraction

- 105,390,080 adapter parameters = 6.83% of 1,543,490,560; the paper reports ~5.6%
- within a factor of 2.0 of the paper's ~5.6%: **yes**

### B3-vs-B4 gate

**FAILED** -- B3-reimpl MER 0.53181 vs B4 (plain LoRA) MER 0.18084 under our scorer.

B3-reimpl does not beat plain LoRA under our scorer, so it is **not functioning as a strong baseline** and is not described as one. Every B3 comparison in this document is therefore marked UNRELIABLE.

### Fidelity note

| item | status | detail |
|---|---|---|
| framework | deliberate deviation | reimplemented in our PyTorch/HuggingFace stack on whisper-large-v3. The reference is ESPnet 202301 on whisper-small with a vendored OpenAI-Whisper. Running the reference would confound B3-vs-B1/B2 with framework. |
| adapter placement | inferred | the reference places two adapters INSIDE each residual block (after self-attention and after the MLP). A forward hook sees only the block output, so both adapters are applied in sequence there. Parameter count is preserved exactly; the position of the first adapter differs by one sublayer. |
| head-selection criterion | mirrored from code_util/head_selection.md and the source | per decoder self-attention head and utterance, the head is counted when attention mass on prompt key positions 1:3 exceeds mass on position 0 plus positions 3:. Ranked by count descending over the training split. |
| head-selection data | deliberate deviation | computed on CS-Dialogue loc-train, not on SEAME, as required. |
| released head mask is disabled upstream | defect in the reference, not reproduced | in the released code `self.selected_heads` is commented out and replaced by a hard-coded 50% `random_onezero` 12x12 mask, so the shipped model does not use its own head selection. The documented procedure is implemented here instead. |
| layer/head key order in the reference | defect in the reference, corrected here | `for head, layer_dict in attention_count.items()` iterates a dict built as `attention_count[layer][head]`, then unpacks the tuple as `layer, head, num`. The transposition is invisible only because the matrix is square for whisper-small. |
| number of selected heads | inferred | the reference hard-codes int(110 * pct / 100) against 144 heads. The fraction 0.7639 is carried over to this geometry rather than the literal 110. |
| CS prompt | deliberate deviation, TRAINING ONLY | the guidance loss needs a ZH slot and an EN slot, so B3-reimpl TRAINS under the reference's `whisper_cs` five-token prompt ['<|startoftranscript|>', '<|zh|>', '<|en|>', '<|transcribe|>', '<|notimestamps|>']. Evaluation on D-dev-select uses the SAME four-token forced-<|zh|> prefix as every other arm, because 'same decode config' is one of the study's explicit invariants and mixing in a second prompt at eval time would confound the MER comparison by prompt as well as by method. The training/eval prompt mismatch is therefore B3-reimpl's own trade-off, stated here rather than silently introduced as an extra variable in the comparison. |
| optimiser hyperparameters | deliberate deviation | the reference uses AdamW lr 1e-3, wd 0.01, betas (0.9, 0.99), warmup 500, accum 4, 10 then 15 epochs. Track B applies the SHARED budget to every arm instead; tuning B3 on its own schedule would break matched training conditions. |
| specaug | not reproduced | the reference enables SpecAugment in the encoder. It is off here because no other arm uses it and it is not part of the attention-guidance mechanism. |
| accent population | stated mismatch | the reference repository is titled for Taiwan-accented Mandarin-English; CS-Dialogue is a different accent population. |
| published MER | stated mismatch | the paper's ~14.2% MER is on SEAME. We evaluate on CS-Dialogue. It is external context, not a reproduction target. |
| license | recorded | the reference repository has NO LICENSE file (checked 2026-08-18). We reimplement rather than vendor, so the obligation is citation of arXiv:2312.08856. |
| LID head parity | deliberate addition, flagged | the shared LID head is added to B3-reimpl for parity with every other arm even though the attention-guidance loss already contains language-aware structure. The two terms may double-count the same signal; this is flagged rather than silently omitted or silently stacked. |

Two mismatches stated rather than hidden: the reference repository is titled for Taiwan-accented Mandarin-English while CS-Dialogue is a different accent population, and the published ~14.2% MER is on SEAME while we evaluate on CS-Dialogue -- external context, not a reproduction target.

## 9. Verdicts

**VERDICT_TRACK_A:** GO -- 1 cell(s) pass all four gates; best is `D:decoder:local:v_nat+v_prompt_weighted_n2p0.5:a1:D28`

**VERDICT_TRACK_B:** four separate claims, never collapsed into one winner:

- **(a) did LID help (B1 vs B0)?** LID did not help -- B1 mean MER 0.13503 vs B0 0.13058
- **(b) did informed init beat the B1 seed range (B2 vs B1)?** BEATS_RANGE -- B2 MER 0.13176 against the B1 seed range [0.13373, 0.13592]
- **(c) did B1/B2 beat B3 on any of MER, params, data efficiency, selectivity?** B1/B2 beat B3-reimpl on ['MER', 'params'] -- UNRELIABLE: the B3-vs-B4 gate did not pass
- **(d) did the learned subspace converge toward or away from span(v_nat, v_prompt)?** INTERMEDIATE -- mean principal angle 74.0 deg to span(v_nat, v_prompt) against a 90 deg random null

**B3-vs-B4 gate:** NOT PASSED. Every B3 comparison above is marked UNRELIABLE. B4 (plain LoRA) is the unambiguous floor that protects the study.

## 10. Where the specification and the repository disagree

Recorded, not resolved silently. Nothing below was substituted without saying so.

### `no_steer_sweep_directory`

- **specification:** a partial harness exists at steer_sweep/ (adapters, hooks, metrics, sweep, preflight, run.py)
- **repository:** no such directory exists anywhere on this filesystem. The real partial harness is src/csasr/steering/ (encoder_hook, decoder_hook, masks), src/csasr/models/hooks.py, src/csasr/lss/sites.py and the frozen v2r3 experiment drivers.
- **resolution:** this package is written at steer_sweep/ as asked and builds on the existing modules; none of them is rewritten.

### `direction_store_lacks_v_prompt`

- **specification:** v_prompt is loaded from the existing direction store
- **repository:** directions.npz holds only site_e_layer{15,23,27,31}_{hann,central60,uniform} and site_d_layer{8,16,24,31}. There is no prompt direction: the rank-1 prompt subspace was PROJECTED OUT of every v_nat contrast at construction (prompt_subspace_rank=1).
- **resolution:** v_prompt is constructed here by the specified method (difference-in-means over prompt language-token states, <|en|> prefix minus <|zh|> prefix) and stored as a NEW artifact under its own kind. It is never conflated with v_nat.

### `layer_grid_not_in_store`

- **specification:** encoder singles {8,15,20,27}, decoder singles {8,16,24,28}
- **repository:** the store holds encoder {15,23,27,31} and decoder {8,16,24,31}. Only encoder 15/27 and decoder 8/16/24 overlap the requested grid.
- **resolution:** missing layers are constructed on D-construct through the frozen Day-3 pipeline (v2r3_directions) with the identical recipe, and every direction records whether it was loaded or reconstructed.

### `directions_are_not_unit_norm`

- **specification:** unit-norm float32, loaded from the existing direction store
- **repository:** stored directions are float64 with norms 0.75-5.54 (||site_d_layer16|| = 2.414639416389413). The prior runs used the RAW norm with rho=1 and norm-preserving renormalisation.
- **resolution:** the sweep unit-normalises to float32 and carries the dose in alpha under scale_mode. Preflight check 2 instead reproduces the prior convention exactly, because its job is to reproduce the prior experiment.

### `prior_run_did_not_force_zh`

- **specification:** decoder prefix is 4 forced tokens including <|zh|>
- **repository:** every v2r3 free-decoding stage ran with language=None, so Whisper chose the language token itself and the prefill width was MEASURED rather than assumed.
- **resolution:** preflight check 2 replicates language=None because it must reproduce the prior 7/489. The study's own baselines and every steered cell force the language token as specified, and C00 is the comparator.

### `aga_is_a_loss_not_a_parameter_attachment`

- **specification:** attach trainable parameters to the selected heads; match the paper's ~5.6% trainable-parameter fraction
- **repository:** the reference implementation attaches bottleneck adapters (Linear d->d/4 -> GELU -> Linear d/4->d, residual, + LayerNorm) at EVERY encoder and decoder block, and uses head selection to mask an AUXILIARY attention-guidance MSE loss (loss = cs_weight * loss_cs + loss_att), not to place parameters. Adapters at d/4 over 24 blocks of whisper-small reproduce ~5.9% -- which is where the ~5.6% figure comes from.
- **resolution:** B3-reimpl follows the reference mechanism: adapters everywhere, head selection masks the guidance loss. The parameter-fraction check is applied to the adapter set. Recorded in the fidelity note.

### `reference_repo_disables_its_own_head_selection`

- **specification:** mirror the head-selection procedure
- **repository:** in the released code the selected-head mask is commented out and replaced by a hard-coded 50% `random_onezero` 12x12 mask; the flattening loop also swaps the layer and head keys, which is invisible only because the matrix is square for whisper-small.
- **resolution:** the procedure documented in code_util/head_selection.md is implemented (not the disabled code path), computed on CS-Dialogue, with the key-order defect corrected. Both facts are in the fidelity note.

### `reference_repo_has_no_license`

- **specification:** check the repository for a LICENSE file
- **repository:** no LICENSE file is present at the repository root (checked 2026-08-18).
- **resolution:** recorded; we reimplement rather than vendor, so the obligation is citation of arXiv:2312.08856, which is discharged in the summary.

### `gate_b_artifact_carries_pre_fix_retention`

- **specification:** denominator 5,711 baseline-correct Mandarin units
- **repository:** the stored gate_b parquet carries 6,264 / 6,251 / 13 for monolingual retention because it predates the Unit-tagging repair; recomputing from the stored text with the CURRENT source gives 5,711 / 5,699 / 12 (99.79%), matching the dated document.
- **resolution:** G4 uses 5,711 and the harness recomputes retention with current code rather than reading the stale field.

