# P0-R1 report — K=3 short-continuation evidence

**Final status: `P0_BLOCKED_EVIDENCE`. P1 has not started.**

The revised K=3 evidence definition (frozen in `P0_R1_EVIDENCE_SPEC.md`, committed before any GPU
result) was run once (Slurm **54712**, H100 MIG `3g.40gb`, COMPLETED, 69 s). Every scalar below was
independently recomputed from the immutable row shards (raw per-step log-probs, raw candidate
tokens) with a script that imports no `csasr` code; it reproduces the committed `summary.json`
exactly. G1 and G2 are unchanged; **G3 remains `WEAK/BLOCKED` under the frozen thresholds, which
were not weakened.**

## Gate table

| Gate | Verdict | Quantitative evidence (independently recomputed) |
|---|---|---|
| G1 | `DEGENERATE_BY_CONSTRUCTION` | `c0==cM` token-identical; `‖h0−hM‖ = 0` on all 57 rows; `‖hE−hM‖` median 2.749 (identical to P0 — the diagnostic-position states are the same forward). The matrix-collapse need factor stays non-identifiable. |
| G2 | `PASS` | 57/60 retained, 3 deterministic `frozen_baseline_text_mismatch` skips (same identities as P0). 0 alignment violations; continuation-scoring masks all-ones, cache positions `0..T−1`, `use_cache=false`, greedy lineage, verified across **all three** continuation query positions. |
| G3 | `WEAK/BLOCKED` | See below. Two frozen sub-criteria fail. |

## Original K=1 vs revised K=3

| Metric | K=1 (job 54703) | K=3 (job 54712) | Frozen G3 requirement | K=3 meets? |
|---|---|---|---|---|
| Finite retained coverage | 57/60 = 95% | 57/60 = 95% | ≥ 90% | ✅ |
| Candidate collision rate | 50/57 = 87.7% | 42/57 = **73.7%** | ≤ 50% | ❌ |
| Usable (non-colliding) rows | 7 | **15** | — | improved |
| Usable wrong-English | 3 | **7** | ≥ 10 | ❌ |
| Usable correct-English | 3 | 5 | reported separately | (n/a) |
| Usable Mandarin | 1 | **3** | ≥ 10 | ❌ |
| AUROC wrong-Eng vs Mandarin | not estimable | not estimable | ≥ 0.60 | ❌ (not estimable) |
| Audio-control abs Δq_cross | mean 1.994 (n=7) | mean **0.485** (n=15) | ≥ 0.05 over ≥ 10 rows | ✅ |
| GPU job | 54703, 45.8 s | 54712, 69 s | one physical job | ✅ |

q_cross by stratum (usable rows, K=3): wrong-English median 1.013 (mean 1.538, n=7); correct-English
median 0.292 (mean 1.314, n=5); Mandarin median 0.325 (mean 0.991, n=3). Candidate length median 3
(min 2, EOS-free); longest-common-prefix median 3 (i.e., most retained rows are full or near-full
collisions).

## Why K=3 helped but did not pass

K=3 broke roughly a seventh of the K=1 collisions (87.7% → 73.7%) and doubled usable rows
(7 → 15), and its panel-wide audio-control statistic now clears the frozen ≥0.05/≥10 bar — a
genuine audio dependence signal. But the collision rate is still above 50%, and the collisions are
**concentrated in the Mandarin stratum** (17/20 collide → only 3 usable), so the AUROC
wrong-English-vs-Mandarin contrast remains not estimable and the ≥10-per-class requirement fails.
This matches the audited mechanism: at Mandarin (and deep mid-content) positions the acoustic
content is Chinese, so flipping the language prompt `<|en|>`↔`<|zh|>` rarely changes even a
three-token greedy continuation; the two conditions only diverge reliably at early / language-
ambiguous positions.

Deterministic representative rows (real audio):

| Stratum | Row | logical | yE / yM | q_cross real / shuffled |
|---|---|---|---|---|
| Wrong-English | `english_wrong:ZH-CN_U0011_S0_111:0` | 0 | ` Okay.` / `所以你用` | 3.934 / 2.038 |
| Correct-English (collision) | `english_correct:ZH-CN_U0011_S0_265:27` | 22 | `�,但` / `�,但` | null / null |
| Correct-English (usable) | `english_correct:ZH-CN_U0091_S0_104:2` | 2 | `这地方是` / `它是一` | 0.085 / −0.182 |
| Mandarin (usable) | `mandarin:ZH-CN_U0011_S0_166:18` | 14 | `的。我` / `的我看` | 0.325 / 0.028 |

## Decision and consequence

Per the frozen decision rule, the outcome is **`P0_BLOCKED_EVIDENCE`**: the candidate-continuation
support formulation (K=1 and its one authorized K=3 revision) has **failed P0**. Thresholds are not
weakened after seeing the result, and per spec §8 no K=2/K=4/top-k/alternative-scorer variant is
attempted in this session. Any further attempt must be designed as a **new, separately versioned
support experiment** (e.g., a support definition that does not depend on the language prompt
changing the greedy continuation at Chinese-dominant positions).

Frozen consequences carried forward (do **not** let anything consume these as valid P1 signals):

- The matrix-collapse / `rho` / `c` geometry need factor is **non-identifiable** (`c0≡cM`) and is
  permanently removed.
- The direction `d = normalize(hE − hM)` at L24 exists and is well-defined, but **no validated
  evidence gate exists**: G3 is blocked, so P1 must not begin on this evidence. If a future,
  separately versioned experiment validates a `q_cross`-like support signal, only then may P1 pair
  `d = normalize(hE − hM)` with an evidence-only gate.

**P1 has not started.**
