# P2-SEL — readout-direction selectivity debug: report

Frozen contract `P2_SEL_SPEC.md` / `P2_SEL_CODEX_DESIGN.md` / `configs/inference_cf/p2_sel.json` (`b4602b8`).
Pre-run audit `PASS_TO_P2_SEL_S1` (`P2_SEL_PRE_RUN_AUDIT.md`). Development diagnostic on already-exposed
D-dev-select states; teacher-forced mechanism screen, not a transcript claim.

**S1 frozen label: `P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE`. `P2_SEL_AUDIT: PASS (S1)`.
S2 NOT RUN (only `P2_SEL_GATE_RESCUES_D2` authorizes it). Final session audit `P2_SEL_AUDIT: PASS`.**

## Execution

| Step | Commit / job | Notes |
|---|---|---|
| Implementation | `6c7644a` | runner/analysis/auditor/Slurm/tests; reuse-first |
| Pre-run audit | `c8ec788` | `PASS_TO_P2_SEL_S1` |
| S1 | Slurm 57805, manifest at `c8ec788` | 143 s total (pass A 93 s, pass B 50 s), peak VRAM 3.69 GB allocated / 4.00 GB reserved, 0 autograd / 0 readout calls |
| S1 audit attempt 1 | — | `P2_SEL_AUDIT: BLOCK`, preserved as `results/inference_cf/p2sel/s1_run1_audit_attempt1_BLOCK.json`: the auditor compared bf16 hook energy to a float64 ideal edit, which is invalid when g is tiny (6/94 gated rows, e.g. g ≈ 5e-7: ideal 1e-6, bf16 hook 0.0188). Mechanical auditor fix `a9606da` (exact bf16 hook emulation, reproduces all 94 rows ≤ 1.6e-7; test added). No run, analysis, threshold or label change. |
| S1 audit attempt 2 | `2a51faf` | `P2_SEL_AUDIT: PASS (S1)`; statistics agree to 8.9e-16; label reproduced independently |

Reused (hash-verified, not recomputed): P2-RJ 180 positions; P2-DIR extracted h_B/h_E, sealed D0/D2 vectors,
NONE logits, Exp-1 per-position ungated-D2 Δm; P2-R run3 E/R_B/g (exact (utterance, query) join 180/180).
All 180 live states, NONE logits and D0 vectors matched P2-DIR bitwise; R_B recomputed from live logits within
1e-6; zero-gain rows bitwise equal to NONE; crop/restore bitwise. Only C1/C2/C3 pulses were new GPU work.

## S1 results (family-12 Bonferroni, α = 0.05/12; 10000 shared dialogue draws, seed 240924)

| Arm | EN-confusion Δm | EN-correct Δm | ZH-correct Δm | EN corruption | ZH corruption |
|---|---|---|---|---|---|
| C0 NONE | 0 | 0 | 0 | 0 | 0 |
| C1 OLD-GATED (descriptive, pointwise) | −0.011 [−0.051, +0.030] | 0 (no edits) | −0.005 [−0.021, +0.006] | 0 | 0 |
| C2 D2-GATED | **+3.520 [+1.914, +4.956]** | 0.000 [0, 0] | **−0.195 [−0.542, 0.000]** | 0.000 [0, 0] | **0.054 [0, 0.158]** |
| C3 D2-BROAD | +3.062 [+2.584, +3.543] | +1.044 [+0.696, +1.467] | −1.358 [−1.668, −1.064] | 0.017 [0, 0.083] | 0.071 [0, 0.176] |
| ungated D2 (P2-DIR, reused) | +4.499 | +1.535 | −2.011 | 0.017 | 0.137 |

Paired C2 − C3: EN-confusion +0.458 [−1.303, +1.956]; ZH-correct +1.163 [+0.633, +1.597].
TOST 90%: confusion [−0.492, +1.368], ZH [+0.872, +1.428] (not equivalent).
Ratios vs ungated D2 (matched denominators 4.49899 / −2.01146; fixed 4.499 / −2.011):
**r_benefit = 0.782, r_harm = 0.097** (C3: 0.681 / 0.675).
C3 energy: Q2 = 100.982, e_broad = 0.7490, Q3 = 100.988 (mismatch +6.3e-5), 0 unreachable, 180/180 edited.

Frozen checks for C2: benefit passes (point 3.52 ≥ 0.5, lower 1.91 > 0); EN-correct safety passes; ZH margin lower
−0.542 < −0.25 FAIL; ZH observed corruption 0.054 > 0.05 FAIL; ZH corruption upper 0.158 > 0.05 FAIL.
Selective vs BROAD: ZH lower +0.633 ≥ +0.10 passes, confusion lower −1.303 < −0.25 FAILS. Equivalence: no.
Precedence -> not ADDS_LITTLE (C3 fails safety, not equivalent), not RESCUES (C2 fails ZH safety and confusion
non-inferiority), not TOO_CONSERVATIVE (C2 fails safety) -> **`P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE`**. Stop.

## Prescribed descriptive gate/dose table (existing S1 scalars, by stratum)

| Stratum | E median [IQR], frac>0 | R_B median [IQR], frac>0 | g median [IQR], frac>0 | C2 dose α·g mean | C2 realized ‖edit‖² mean (frac>0) |
|---|---|---|---|---|---|
| EN-confusion | 0.703 [0.000, 0.939], 0.70 | 0.907 [0.818, 0.985], 1.00 | 0.615 [0.000, 0.849], 0.70 | 0.962 | 1.472 (0.70) |
| EN-correct | 0.792 [0.000, 0.962], 0.60 | 0.000 [0, 0], 0.00 | 0 [0, 0], 0.00 | 0 | 0 (0) |
| ZH-correct | 0.000 [0, 0], 0.08 | 0.998 [0.980, 1.000], 1.00 | 0 [0, 0], 0.08 | 0.127 | 0.211 (0.08) |

The C3 realized ‖edit‖² is the constant packet 0.561 at every row.

Description, using existing scalars only (no new diagnostic was run):
* **R_B does not separate EN-confusion from ZH-correct.** It is near 1 in both strata (0.91 vs 1.00), because a
  confusion position is, by definition, a Mandarin-dominant prediction. R_B does fully suppress EN-correct
  (R_B = 0 there).
* **E is the only discriminating factor.** It is > 0 at 70% of EN-confusion positions and 8% (5/60) of ZH-correct
  positions. All residual ZH harm comes from those 5 positions, where local LID support fires strongly
  (E 0.75–0.98; g 0.23–0.96). Three of them are corrupted (Δm −0.19 to −4.0); OLD-GATED at the same gates
  corrupts none.
* On the benefit side, 18/60 EN-confusion positions have E = 0 (g = 0), which accounts for the 22% lost benefit.
  The C2 − C3 confusion interval is wide (lower bound −1.30) because gated doses are concentrated on 70% of the
  rows, while BROAD spreads equal packets over every row.

## Bottleneck update (descriptive; does not change the label)

D2's direction power survives gating: 78% of the benefit is retained and 90% of the margin harm is removed. The
remaining failure is gate selectivity at the Mandarin-correct tail. The R_B factor cannot distinguish
EN-confusion from ZH-correct, so selectivity rests entirely on E, and E false positives on a few Mandarin
positions are enough to break the 5% corruption budget. Gate/localizer selectivity, specifically the E
(local LID) component, is now the apparent bottleneck for D2 deployment.

## Data exposure / firewall

Only the already-exposed 180 P2-R/P2-RJ D-dev-select positions (80 utterances, 20 dialogues). The S2
100-utterance mini-panel was frozen but **not decoded**. No router-calib, D-dev-confirm, D-test, P3 or
transfer data. P3 HELD.

## Next action

The stage is terminal; no S2, gate repair, localizer change, oracle localization or further GPU work is
authorized. Any next step needs a new human decision and a separately frozen pre-registration. One candidate
is a design pass on gate selectivity for D2, e.g. E reliability on Mandarin-correct positions. Such a stage
would be post-hoc and motivated by these exposed results.
