# D1 — region-weighted GoldKL pilot (result)

**Verdict: `D1_NO_INCREMENTAL_GAIN`.** Recommendation: `STOP_FIXED_SWITCHBACK_WEIGHTING` (not authorization).

This is a negative, exploratory result on the exposed dev150 panel. Every integrity check passed.

Protocol: `D1_REGIONKL_PROTOCOL.md`. Rules: `D1_REGIONKL_DECISION_RULES.md`. Private artifacts:
`/mnt/data/tungnx/cs-asr-reft/d1_regionkl/`.

## 1. Engineering (preflight 59556: `D1_READY_FOR_TRAINING`, G1–G7 pass)

**Training region (training references only):**
- 500/500 utterances mappable; targets equal the manifest; 18,616 supervised positions.
- 3,001 substantive English and 16,215 substantive Mandarin units.
- **613 EN→ZH switch-backs**, giving 3,419 window units. 381 windows reach 7 units; 232 are shorter.
- 2,901 tokens touch window units:
  - 2,825 are unambiguous region tokens;
  - 76 are ambiguous, all `window_end` (a token spanning the 7th and 8th unit). They keep weight 1.
  - 82 window units have no unambiguous token.
- No language-boundary, filler or multi-window ambiguities occurred; one token is unmapped; EOS is never weighted.
- Region tokens appear in 338 utterances and all 27 training dialogues.
- Mapping takes 0.25 s on CPU.

**Teacher-correct coverage:**
- region: **2,557/2,825 = 90.5%** (floor 10%), in all 27 dialogues; the top-3 dialogues hold 20.1%;
- region share of teacher-correct positions: 2,557/16,737 = 15.3%;
- first-switch-back-token coverage, P2-1b's measure: 470/613 = 76.7%;
- EOS: 463/500.

Global effective per-token factors versus GoldKL are 1.73 for the region and 0.87 elsewhere. Per training micro-batch,
2,557–2,558 masked region positions per epoch; median 37–41 per micro-batch.

**G3 GoldKL equivalence:**
- With all weights 1, the D1 step equals P2-1b `kl_step` bitwise at zero init and with a perturbed adapter: loss, CE,
  KL, mask, teacher and student logits, and all gradients (max difference 0).
- The seed-42 epoch-1 replay reproduces `KL_seed42/epoch1/adapter.pt` **bitwise**, together with train KL, CE, mask
  positions (16,739), tokens and the 32 updates.
- The historical GoldKL runs were therefore reused as matched controls and not retrained.

**G4 weighting:**
- region active; denominator equal to N + K;
- exact aggregation on the same float32 per-token KL matches to ~1e-9 relative;
- float64 cross-check 7.6e-5 relative, against 2.7e-5 for unweighted GoldKL (float32 precision);
- teacher-incorrect positions excluded, and weights placed there have no effect;
- empty mask gives a differentiable zero; KL ≥ 0; teacher and weights detached;
- gradients differ from GoldKL by about 1.1% relative (perturbed adapter).

**G5 and G6:**
- G5 model and decoding integrity: 5,122 parameters; zero-init and disabled adapter equal frozen; zero-init dev 20/20
  beam-5 and greedy; LID 20/20; cache positions; no leakage; reload bitwise; float32 cached-vs-full ≤ 3.6e-6 nats
  (BF16 up to 0.053, descriptive).
- G6 resources: D1 micro-batch 0.407 s vs GoldKL 0.411 s and CE 0.368 s, so region weighting adds no measurable cost;
  weight construction takes 0.3 ms; projected job 1,591 s.

**Before the freeze:** two dry runs (59552, 59553). One engineering tolerance in G4 was corrected (protocol §10); there
was no scientific change.

## 2. Integrity

All checks pass:
- preflight READY;
- verify: 3 runs × 3 epochs reload and re-decode 20/20, with hashes and 5,122 parameters;
- three runs with 96 updates each; zero-init 150/150; backbone unchanged;
- GoldKL recomputation reproduces P2-1b at every epoch;
- per-epoch D1 mask positions, tokens and updates equal the GoldKL run;
- initial `base` equal; orthogonality ≤ 6.0e-7;
- frozen region hash used; region weights active in every epoch; constant lr 5e-4.

## 3. Matched results (beam-5, dev150, selected checkpoints)

**Selected epochs:** GoldKL 3/2/3, D1 3/3/2 (seeds 42/43/44).

**Target window (TW) denominators** (dev references): 222 switch-back windows, 1,270 eligible units, 1,156 of them
correct under B0.

| Seed | GoldKL new ZH | D1 new ZH | GoldKL TW new | D1 TW new | GoldKL SB | D1 SB | GoldKL EN net | D1 EN net | GoldKL CS MER | D1 CS MER |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 53 | 53 | 37 | 37 | 30 | 30 | 34 | 35 | 0.1124 | 0.1127 |
| 43 | 48 | 49 | 31 | 31 | 31 | 31 | 41 | 37 | 0.1105 | 0.1105 |
| 44 | 50 | 39 | 33 | 30 | 26 | 22 | 35 | 37 | 0.1103 | 0.1105 |
| Mean | 50.3 | 47.0 | 33.7 | 32.7 | 29.0 | 27.7 | 36.7 (Σ 110) | 36.3 (Σ 109) | 0.1111 | 0.1112 |

D1 − GoldKL, seed mean [95% dialogue bootstrap, shared draws]:

| Metric | Point | 95% CI | Per-seed range |
|---|---|---|---|
| New Mandarin errors | −3.3 | [−13.0, +4.0] | −11 … +1 |
| TW new | −1.0 | [−7.7, +5.3] | |
| Long SB units | −1.3 | [−12.7, +6.3] | |
| EN net | −0.3 | [−4.3, +3.7] | |
| CS MER | +0.0002 | [−0.0028, +0.0027] | |

**Rules:**
- **A fails:** mean −3.3 (needs ≤ −5); lower in 1/3 seeds (needs 2).
- **B passes:** 109/110 = 0.99.
- **C fails:** mean CS MER +0.00019. Seed 42 is +0.0004, seed 43 equal, seed 44 +0.0002.
- **Components:** TW −1.0 and SB −1.3, each lower in 1/3 seeds, so neither passes.
- **D (TW support):** not supported.
- **E (retention ≥ 0.99):** D1 in seeds 43 and 44 (2/3); GoldKL in seed 43 only.

**Concentration (H5):**
- Seed 42: D1 matches GoldKL exactly on every preservation count.
- Seed 43: +1 new Mandarin error, −4 EN net.
- Seed 44 carries the whole aggregate gain (−11). One dialogue contributes 13 units, 118% of that seed's difference;
  leave-one-dialogue-out range is −17 … +2.

## 4. Correction–preservation detail

| Measure | GoldKL (seeds 42/43/44) | D1 (seeds 42/43/44) |
|---|---|---|
| Omission → correct | 29/32/33 | 32/31/31 |
| Translation → correct | 13/14/14 | 13/14/14 |
| Lexical → correct | 2/4/2 | 2/4/4 |
| Correct → wrong | 20/15/24 | 23/18/20 |
| Wrong → other wrong | 14/13/18 | 17/13/13 |
| English corrections / corruptions | 54/20, 56/15, 59/24 | 58/23, 55/18, 57/20 |
| Near-switch (≤ 7) new errors, of 2,218 B0-correct | 53/45/49 | 53/48/38 |
| Far-from-switch new errors, of 2,732 | 0/3/1 | 0/1/1 |
| Mandarin new errors outside TW | 16/17/17 | 16/18/9 |
| New Mandarin deletions | 43/38/37 | 43/41/31 |
| Dropped switch-back spans | 7/6/6 | 7/6/6 |
| Worst-dialogue mixed regression | 32/32/32 | 32/34/20 |

- **Where the damage is:** TW new errors make up about 70% of all new Mandarin errors in both arms. This confirms
  where the damage lies, but D1 did not move it.
- **Loops and non-terminations:** loops 0 everywhere; non-terminations 1/1/2 (GoldKL) and 2/1/1 (D1).
- **Greedy (secondary):**
  - CS MER: GoldKL 0.141/0.144/0.142; D1 0.142/0.142/0.167. D1 seed 44 epoch 2 is a greedy outlier.
  - Greedy new Mandarin errors: 107/95/104 (GoldKL) and 91/103/86 (D1).

## 5. Selection sensitivity

| Seed | GoldKL ep3 zh | D1 ep3 zh | GoldKL ep3 TW | D1 ep3 TW | GoldKL ep3 EN | D1 ep3 EN | GoldKL ep3 CS | D1 ep3 CS |
|---|---|---|---|---|---|---|---|---|
| 42 | 53 | 53 | 37 | 37 | 34 | 35 | 0.1124 | 0.1127 |
| 43 | 59 | 49 | 39 | 31 | 34 | 37 | 0.1118 | 0.1105 |
| 44 | 50 | 56 | 33 | 39 | 35 | 33 | 0.1103 | 0.1120 |

- At fixed epoch 3 the mean new-Mandarin difference is −1.3 (−10 … +6); TW −0.7, SB 0.0, CS MER +0.00025, Σ EN net
  105/103. A and C fail.
- The sign of the seed-43 and seed-44 differences flips between the selected and fixed-epoch views, and between epochs.
  For example, seed 44 D1 is 39 at epoch 2 and 56 at epoch 3.
- Epoch-matched D1 − GoldKL differences range from −18 to +6.

## 6. Training dynamics

- **Train CE:** D1 0.403 → 0.363, GoldKL 0.402 → 0.362; +0.001 for D1 at every epoch.
- **RegionKL (weighted):** 0.008 → 0.016.
- **Diagnostic unweighted KL:** 0.0068–0.0072 → 0.0145–0.0152, 6–11% below the historical GoldKL (0.0075–0.0081 →
  0.0155–0.0163).
- **Region positions:**
  - per-token KL about twice the rest (≈ 0.024 vs 0.015 at epoch 3);
  - share of the weighted KL mass 0.45–0.46 in epoch 1, falling to 0.37–0.38 by epoch 3;
  - weight sum per epoch 19,294–19,297.
- **Gradient norm:** 0.20–0.22 → 0.26–0.30, as in GoldKL.
- **Orthogonality** ≤ 6e-7; lr constant.

## 7. Representation diagnostics (training utterances, gold prefixes, teacher-correct positions; descriptive)

**At matched epoch 3**, D1 and GoldKL learned almost the same intervention:
- principal-angle cosines of the R row spaces 0.9985–0.9999;
- per-position edit cosine ≈ 0.998 in every token class;
- ‖b‖ and ‖W − R‖ within 1%.

**D1 shrinks region edits slightly:**

| Class (epoch 3) | D1 / GoldKL mean ‖Δh‖ |
|---|---|
| Switch-back region | 0.935–0.982 |
| Other Mandarin | 0.958–1.003 |
| English | 0.980–0.998 |
| Selectivity (region vs English) | 0.95–0.98 |

**Teacher-forced token KL on teacher-correct region positions** fell from 0.027–0.028 to 0.022–0.023 (−17 to −21%).
Other Mandarin fell by 7–11%; English was essentially unchanged (0.021–0.024 vs 0.021–0.025).

So on gold prefixes D1 did protect the target tokens, mildly and somewhat selectively. That did not translate into
free-decoding preservation. Selected-checkpoint comparisons mix in epoch differences (seeds 43 and 44), so §5 is the
cleaner view.

## 8. Efficiency

| Item | Value |
|---|---|
| Trainable parameters | 5,122 (20,488 bytes float32) |
| Checkpoint file | 6,576,349 bytes (includes the 1280×1280 parametrization `base`) |
| Training per run | 76–83 s training, 7.6–7.7 s teacher forward, ~1,110 s wall (dev decoding dominates) |
| Peak memory | 5.73 GB allocated, 8.02 GB reserved |
| Decode latency | beam-5 1.00–1.04 s/utt, greedy 0.62–0.63 s/utt (unchanged) |
| Region overhead | none measurable |

**GPU total for D1: 70.5 MIG-min ≈ 1.18 slice-h.**

| Component | MIG-minutes |
|---|---|
| Three runs | 56.7 |
| Three preflights (2 dry + official) | 8.0 |
| Verify | 3.7 |
| Mechanism | 2.1 |

CPU jobs took under 3 min in total.

## 9. Interpretation

| Hypothesis | Assessment |
|---|---|
| H1 (region pressure improves the trade-off) | **Not supported.** The mean gain (−3.3) is below the margin, comes from one seed and one dialogue, and reverses at fixed epoch 3 |
| H2 (reweighting suppresses English corrections) | **Not supported.** EN net 109 vs 110; correction types unchanged |
| H3 (gold-prefix preservation cannot fix generated-prefix damage) | **Consistent with the data.** Teacher-forced region KL dropped about 20% while free-decoding TW damage did not move (§7) |
| H4 (damage-critical positions poorly covered by the mask) | **Not supported as the explanation.** Region coverage 90.5%; first switch-back token 76.7% |
| H5 (dialogues or seeds dominate) | **Supported** for the observed differences |

- **Specificity:** D1 is not distinguishable from generic GoldKL. Decodes are 128–144/150 token-identical per epoch, and
  the parameters nearly coincide.
- **Unexplained by gold-prefix training:** the omission decision under the system's own generated prefix (P2-1c).

## 10. Limitations

- **Evaluation data:** dev150 is exposed, has 20 dialogues, and is used for both selection and evaluation. Seeds share
  the dialogues.
- **Design:** three seeds; one weight (2) and one window (7 units); no matched non-switch-region control (not needed
  given no gain).
- **Missing comparisons:** no optimized LoRA baseline and no independent test.
- **Diagnostics:** representation diagnostics are teacher-forced on training data and descriptive.
- **Coverage:** the training-region definition is reference-based; coverage is high, so low coverage does not explain
  the null.

## 11. Next step (recommendation only)

`STOP_FIXED_SWITCHBACK_WEIGHTING`. Re-weighting gold-prefix KL toward switch-back regions does not change free-decoding
behaviour; the evidence again points to generated-prefix history.

D2 (representation edit-locality) remains a separate candidate and has not been started. A strong optimized LoRA
benchmark should be prepared as an independent protocol.

## 12. Provenance

| Item | Value |
|---|---|
| Commits | `6ee441a` (before execution; frozen head), plus this result commit |
| Hashes (sha256 prefixes) | freeze `affa9460`, preflight `9af3a3dd`, regions `1e253623`, region_static `9b1dd82d`, coverage `0a61d915`, evaluation `3c442fd5`, verify `3672bf73`, mechanism `f0a6f512`, report `1ab44283`, manifest `31771e20` |
| Tests | 59550 (all `tests/reft` pass) |
| Dry runs | 59551/59552, 59553 |
| Regions / freeze / preflight | 59554 / 59555 / 59556 |
| Training | 59557/59558/59559 |
| Select / evaluate / verify / mechanism / report | 59560 / 59561 / 59562 / 59563 / 59564 |

Exactly three formal D1 runs were executed. No locked data was opened, and earlier freezes and artifacts are unchanged
(verified by the S0/S1 → P2-2/P2-1c → P2-1b → P2-0 chain checks in every job).
