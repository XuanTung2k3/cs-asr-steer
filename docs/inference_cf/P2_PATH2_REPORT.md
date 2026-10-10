# P2-PATH2 — online consensus guard + short theta0 handoff (report, terminal)

**Verdict: `P2_PATH2_BRANCH_CONTROL_SUFFICIENT`.** G1 (CONSENSUS-1) passes all frozen criteria. G2 also passes but is
token-identical to G1 on every row, so it has no material advantage. Independent audit `P2_PATH2_AUDIT: PASS`
(primary-phase before references, then full).

Scope: an exposed-development controller diagnostic on the 12 PATH0 D rows (9 dialogues), an enriched selected cohort.
There is no population or significance claim, and **the Mandarin rescue comes from a single utterance** (see below).

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, panel, config) | `e1c6ee7` | panel sha256 `ade7567c…` |
| Implementation + plan seal | `5fa259d` | `consensus_guard.py`, runner, analysis, auditor, sbatch, tests |
| Pre-run audit | `155aa40` | `PASS_TO_P2_PATH2` |
| Manifest pushed | `e525ae7` | before the job |
| Run | Slurm 58013 | 54 s allocation, 26 s runtime, two resident instances |
| Output seal + primary | `e1198aa` | before references; primary audit PASS next commit |
| Secondary + full audit + report | this commit | BRANCH_CONTROL_SUFFICIENT; `P2_PATH2_AUDIT: PASS` |

## Validity (all checks true)

- **Barrier 12/12:** theta0 decode = B0, the effective bf16 LN state equals the checkpoint, and G0 = sealed A4 in tokens
  and termination.
- **Instances:** two non-aliased resident instances with identical non-LN weights. Theta0 was never modified; the A4
  instance was reset exactly after every row. Peak VRAM 6.83 GB allocated / 7.41 GB reserved.
- **Trigger:** the online detector (first any-token processed-argmax disagreement, single event) fired on exactly the 5
  historical C rows at k = 0, 5, 15, 0, 10. It never fired on the 7 U rows, which give G1 = G2 = G0 = A4 exactly. C/U
  and the historical sites were used only in audit and analysis, never by the controller (source allowlist check).
- **Branches and scores:** the live b0/b4 branches equal the sealed B0/A4 `[k:k+3]`. All 4 per-token log-prob lists,
  the scores, the margins and the winners reproduce PATH1 exactly. The same decision hash is shared by G1 and G2. When
  A4 wins, G1 = G2 = G0.
- **Ownership:** every detector, rollout, score and G path used a fresh owner cache with the LN hash locked. G2's
  theta0 handoff was replayed on a fresh A4 cache (prompt plus full history, token by token).
- **Counts:** 24 A4 updates plus 1 live-check backward; 5 triggers, 10 rollouts, 20 score paths, 5 G1 decodes and 3 G2
  replay decodes.

## Adjudication (triggered rows)

| Row | k | S_θ0(b0) | S_θ0(b4) | S_A4(b0) | S_A4(b4) | Winner | Margin |
|---|---|---|---|---|---|---|---|
| U0021_S0_513 | 0 | −1.594 | −1.271 | −2.482 | −0.334 | A4 | −1.236 |
| U0023_S0_666 | 5 | −0.333 | −0.388 | −0.502 | −0.220 | A4 | −0.114 |
| U0023_S0_87 | 15 | −0.158 | −0.534 | −0.495 | −0.120 | theta0 | +0.0005 |
| U0023_S0_664 | 0 | −0.283 | −2.057 | −0.790 | −0.987 | theta0 | +0.985 |
| U0092_S0_101 | 10 | −0.901 | −0.970 | −0.937 | −0.951 | theta0 | +0.042 |

The winners are theta0 ×3 and A4 ×2. G1 and G2 both change 3 transcripts relative to A4 (the theta0 winners).

Summed decision-stream distance is identical for G1 and G2: 31 to B0, 22 to A4, 139 to AUTO. G2 = G1 on all three
theta0-winner rows: after one selected theta0 token, A4 itself continues with the next two theta0 candidate tokens.

## ASR (all 12 rows)

| System | ZH errors | POI errors | Mixed errors | PIER | MER | EN-WER | ZH-CER | ZH ret. vs B0 | Outside harm vs B0 |
|---|---|---|---|---|---|---|---|---|---|
| B0-FORCED | 18 | 119 | 137 | 0.758 | 0.493 | 0.758 | 0.150 | — | — |
| B0-AUTO | 90 | 58 | 149 | 0.369 | 0.536 | 0.369 | 0.750 | 0.294 | 0.709 |
| A2 | 50 | 103 | 154 | 0.656 | 0.554 | 0.662 | 0.417 | 0.775 | 0.223 |
| A4 = G0 | 41 | 101 | 143 | 0.643 | 0.514 | 0.650 | 0.342 | 0.775 | 0.223 |
| **G1** | **21** | **101** | **123** | 0.643 | **0.442** | 0.650 | **0.175** | 0.971 | 0.029 |
| **G2** | 21 | 101 | 123 | 0.643 | 0.442 | 0.650 | 0.175 | 0.971 | 0.029 |

- **Versus B0:** G1 has 18 POI corrections and 0 corruptions, embedded-EN retention 1.0, no caps and no severe
  truncations. S / D / I: G1 36 / 86 / 1 vs A4 56 / 86 / 1.
- **C rows (5):** ZH / POI / mixed errors are B0 4/50/54, A4 27/32/60, **G1 7/32/40**, AUTO 66/4/70.
- **U rows (7):** G1 equals A4 equals B0 at 14/69/83.

**Criteria** (both G1 and G2): ZH 21 ≤ 29 ✓; POI 101 ≤ 105 ✓; mixed 123 ≤ 143 ✓; caps 0 and severe 0 ✓ → **FULL PASS**.
G2 − G1 differences are ZH 0, POI 0, mixed 0, so there is no raw material advantage. Under the frozen precedence, G1
passes and the G2 extra is not supported → **`P2_PATH2_BRANCH_CONTROL_SUFFICIENT`**.

## Row-level analysis

- **Harmful A4 branch, U0023_S0_664:** consensus clearly chooses theta0 (+0.985). One theta0 token and then A4 repairs
  the Mandarin damage: ZH errors 22 → 2, mixed 39 → 19, POI unchanged at 17. **This single row accounts for all 20 ZH
  errors rescued over the 12 rows**, and for 20 of the 20 mixed-error reductions vs A4. It matches PATH0's RESCUE-1
  result on the same row (R = 0.90).
- **Beneficial A4 branches, U0021_S0_513 and U0023_S0_666:** consensus chooses A4, and its POI gains are retained
  (15 → 0 and 3 → 0 POI errors vs B0).
- **Near-tie theta0 wins, U0023_S0_87 (+0.0005) and U0092_S0_101 (+0.04):** the output changes by 1 token from A4, but
  ZH, POI and mixed counts are unchanged.
- **No-trigger controls (7 U rows):** exact no-op.

## Interpretation

| Question | Answer |
|---|---|
| Is branch choice sufficient? | **On this cohort, yes.** One consensus-selected token at the first theta0/A4 disagreement both keeps A4's useful switches and rejects the harmful one. |
| Is a short theta0 guard needed? | **No.** G2's three-token handoff never changed an output relative to G1. |
| Does consensus preserve beneficial A4 switches? | Yes, for both beneficial C rows. |
| Does it reject harmful switches? | Yes, for the one materially harmful C row, with the largest margin. |
| Current bottleneck | Evidence breadth. The controller's net benefit here rests on one decisive utterance and two near-tie decisions, among only 5 triggers in 9 dialogues. |

The A4 forced-ZH PIER (0.643) is unchanged, and it remains far behind AUTO (0.369). The guard repairs A4's Mandarin
safety without improving English further.

## Next action

Stop. The single justified next stage is a **separately frozen, broader development test of G1 (CONSENSUS-1)** on
more exposed development utterances, with its own pre-registered criteria. Nothing is implemented or run here: no G2
tuning, H change, repeated guard, threshold or weight change, and no full-24/100/300 or P3.

**Data:** exposed D-dev-select 12 rows only. No fresh validation.
