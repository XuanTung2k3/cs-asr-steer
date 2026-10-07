# P2-PATH1 — branch adjudication and state-vs-prefix causal decomposition (report, terminal)

**Primary verdict: `P2_PATH1_MIXED`.** `CONSENSUS_REJECTS_AUTO`: PASS. Secondary `ASR_STATE_ALIGNMENT`: MIXED.
Independent audit `P2_PATH1_AUDIT: PASS`, primary-phase (before references) and full.

Scope: an exposed-development causal diagnostic on the 7 PATH0 U rows (6 eligible), plus 5 C rows that were scored
only. No significance claim.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, sites, config) | `5a159f2` | sites sha256 `57db0b0f…` |
| Implementation + plan seal | `3f378be` | `branch_adjudication.py`, runner, analysis, auditor, sbatch, tests |
| Pre-run audit | `cea9410` | `PASS_TO_P2_PATH1` (including the no-cross-state-cache check) |
| Manifest pushed | `30a6dfb` | before the job |
| Run | Slurm 58011 | 43 s allocation, 23 s runtime |
| Output seal + primary | `2bf92bc` | before references |
| Auditor mechanical fix | `8fd56ce` | primary-audit attempt 1 crashed (KeyError on factorial path records) before writing any output; fixed with a test; no science changed |
| Primary audit / secondary / full audit / report | following commits | MIXED; `P2_PATH1_AUDIT: PASS` |

## Validity

- **Sites:** recomputed independently from raw arrays. All 12 PATH0-row fingerprints, sites, prefixes, c_B/c_A, L1
  eligibility, H=3 content donors and A4A targets match. U = 7 with k = 0 throughout; Σd_BA = 48.
- **Reconstruction:** A4 rebuilt 12/12 through the unchanged `adapt_a4`. That is 24 updates plus 1 live-check backward
  (gradient 0.89%). The effective bf16 state equals the sealed checkpoint, FREE-A4 equals the sealed output, and the
  theta0 decode equals B0.
- **Factorial sanity:** T0B = B0 7/7, A4B = B0 7/7, and A4A equals PATH0 INDUCE-1 tokens and termination 7/7. A4A
  re-passes the inherited INDUCE_1 with identical pooled value 0.771.
- **No cross-model cache:** every path and score ran on a fresh `cached.Branch` created after state selection. The LN
  hash was locked for each whole path (owner hash verified before and after); 28 factorial decodes and 48 score paths
  (136 scored terms).
- **Resets and VRAM:** exact; non-LN weights unchanged. Peak 3.74 GB allocated / 4.05 GB reserved.

## Factorial decomposition (r = k+1, forced token excluded)

| Row | Dialogue | d_BA | T0A d_A / d_B | I_θ0 | A4A d_A / d_B | I_A4 | Δ_state | τ | Class |
|---|---|---|---|---|---|---|---|---|---|
| U0029_S0_472 | CSD0015 | 3 | 1 / 2 | 0.667 | 0 / 3 | 1.000 | 0.333 | 0.333 | prefix success; boundary amplification (Δ = τ) |
| U0039_S0_80 | CSD0020 | 2 | 1 / 1 | 0.500 | 1 / 1 | 0.500 | 0 | 0.5 | prefix (θ0 ≈ A4) |
| U0091_S0_68 | CSD0046 | 3 | 0 / 3 | 1.000 | 0 / 3 | 1.000 | 0 | 0.333 | prefix (θ0 ≈ A4) |
| U0102_S0_80 | CSD0051 | 29 | 12 / 25 | 0.586 | 2 / 28 | 0.931 | +0.345 | 0.25 | prefix success, **state-amplified** |
| U1003_S0_61 | CSD0502 | 1 | 1 / 1 | 0.000 | 1 / 1 | 0.000 | 0 | 1.0 | no induction either way (θ0 returns to B0) |
| U2004_S0_104 | CSD1002 | 10 | 16 / 16 | −0.600 | 7 / 9 | 0.300 | +0.900 | 0.25 | **state-required**: θ0 leaves to a third path |
| U1079_S0_23 | CSD0540 | 0 | — | — | — | — | — | — | DIRECT_ONLY |

**Inherited INDUCE_1.** Both conditions pass and both have 4 successes across 4 dialogues:

| Condition | Pooled | Pass |
|---|---|---|
| A4A (prefix + adapted state) | 0.771 | **yes** |
| T0A (prefix only) | 0.354 | **yes** |

**State contribution:**
- Δ_state pooled = **+0.417**.
- Materially A4-stronger rows: 3 (U0029 at the boundary, U0102, U2004).
- θ0 ≈ A4 rows: 4. The boundary row U0029 is counted in both lists, by the inclusive and strict rules respectively.
- θ0-stronger rows: 0. θ0 returns toward B0 in 1 row (U1003).

**Precedence:**
- PREFIX_DOMINANT fails: Δ_pooled 0.417 > 0.20, and 3 material rows exceeds the limit of 2.
- ADAPTED_STATE_DOMINANT fails: T0A passes.
- MIXED holds: T0A passes, and the meaningful-prefix predicate also holds.
- → **`P2_PATH1_MIXED`.**

## Branch adjudication (H=3, mean donor log-prob, S_cons = 0.5·S_θ0 + 0.5·S_A4)

**U rows:** S_cons chooses B0 in **7/7**, and in all 6 eligible rows as strict wins across 6 dialogues; AUTO is chosen
0 times. Margins range from +1.0 to +7.8 (U2004 +7.8). → **CONSENSUS_REJECTS_AUTO: PASS.**
- Both models individually prefer B0 at these sites. A4 alone also prefers B0 in every U row, just less strongly.

**C rows** (descriptive): B0 is chosen in 3 and A4-alt in 2.

| Row | Choice | Margin | Descriptive note (secondary) |
|---|---|---|---|
| U0021 | A4-alt | −1.24 | A4 removed 15 POI errors |
| U0023_S0_666 | A4-alt | −0.11 | A4 removed 3 POI errors |
| U0023_S0_664 | B0 | +0.99 | the row where A4 added 20 ZH errors |
| U0023_S0_87 | B0 | +0.0005 | near tie |
| U0092 | B0 | +0.04 | direct-only |

These are scored donor branches, not simulated adjudicated transcripts; the alignment is not a claim of benefit.

## Secondary reference evaluation (U rows, after the seal; does not change the primary label)

| System | ZH errors | POI errors | Mixed errors | S / D / I |
|---|---|---|---|---|
| B0 | 14 | 69 | 83 | 23 / 60 / 0 |
| AUTO | 24 | 54 | 79 | 15 / 64 / 0 |
| T0A (AUTO token, then θ0) | 20 | 54 | 75 | 17 / 58 / 0 |
| A4A (AUTO token, then A4) | 24 | 54 | 79 | 17 / 62 / 0 |

All outputs terminate with EOS and none hits the cap.

- **ZH-error excess vs B0** (eligible rows): T0A +6 (2 rows), A4A +10 (3 rows). Both meet the harm condition, and the
  difference is 4 (> 2) → **ASR_STATE_ALIGNMENT = MIXED.**
- **POI:** T0A and A4A both reach AUTO's full POI benefit (−15 vs B0).
- **Answers:**
  - One AUTO token followed by θ0 reproduces AUTO's full POI benefit and about 60% of its Mandarin harm (+6 of +10).
  - A4 continuation adds the remaining harm: +4 ZH errors, mainly U2004 (θ0: 6, A4: 8) and U0102 (θ0: 4, A4: 6).
  - A4 adds no POI benefit beyond θ0.

## Interpretation

| Question | Answer |
|---|---|
| Is the AUTO token / history alone sufficient? | On this cohort, **yes for basin entry**. T0A alone passes the inherited induction criterion in 4 rows / 4 dialogues (CSD0015, CSD0020, CSD0046, CSD0051) and brings the full POI change and most of the ZH harm. |
| Is persistent A4 state necessary? | **Not in general.** It is necessary only for U2004 (CSD1002), where θ0 goes elsewhere after the AUTO token. |
| Does A4 amplify the basin? | **Yes.** Pooled attraction 0.354 → 0.771, materially stronger in U0102 and U2004 (U0029 at the boundary), adding +4 ZH errors (+10 vs +6) without extra POI benefit. |
| Can cross-model consensus reject unsafe branch entry? | **Yes, on this cohort.** Short H=3 consensus scores reject the AUTO branch at all 7 U sites with clear margins. On C rows they choose B0 at the strongly Mandarin-damaged site. This is a reference-free signal, not a validated controller. |

**Most likely current bottleneck:** a mixed mechanism. Entering the AUTO branch at the first decision is enough to set
the basin, and A4's persistent state then deepens it in some rows. Prefix-driven: CSD0015, CSD0020, CSD0046, CSD0051
(entry). State-amplified: CSD0051 (U0102) and CSD1002 (U2004).

## Next action

Stop. No controller is implemented here. The justified next class is a **separately frozen guarded-switch diagnostic**
combining two parts, both pre-registered:
1. Short-horizon cross-model branch adjudication at the first language-switch decision.
2. Temporary A4 rollback (θ0 continuation) inside the guarded switch region.

A new human decision is required.

**Data:** the exposed D-dev-select 12 rows only. No fresh validation, no full-24 extension, no full-100/300, no P3.
