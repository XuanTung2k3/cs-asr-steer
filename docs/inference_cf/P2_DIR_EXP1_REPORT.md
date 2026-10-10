# P2-DIR Exp-1 — direction-only causal screen: report and post-run audit

Frozen spec `P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md` (section 6), config
`configs/inference_cf/p2_dir_direction_identification.json`, pre-run audit `P2_DIR_PRE_RUN_AUDIT.md`
(`PASS_TO_P2_DIR_EXP1`). Teacher-forced mechanism screen on already-exposed D-dev-select states; not a
transcript claim.

**Frozen result: `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`** (supplementary `P2_DIR_CAUSAL_POWER_WITH_DAMAGE:D2`).
**Independent audit: `P2_DIR_AUDIT: PASS` (stage EXP1).** Exp-2 and Exp-3 were not run (frozen stop rule).

## Execution

| Step | Job / commit | Outcome |
|---|---|---|
| Implementation | `1802441` | providers, runner, analysis, auditor, tests |
| Pre-run audit | `00dd8b2` | `PASS_TO_P2_DIR_EXP1` |
| Extraction (B/E states only) | Slurm 57789, manifest at `00dd8b2` | 80/80 utterances; 180/180 states reproduce P2-R argmax / site norm (1e-6) / cos_d_state (1e-6); 86 s, 3.7 GB peak |
| D1 folds (CPU) | `results/inference_cf/p2dir/folds_run1` | 20/20 leave-one-dialogue-out folds pass every guard (A 55-58, B 56-59 observations) |
| Exp-1 pulses + post-seal evaluator pass + D2 GPU spot audit | Slurm 57792, manifest at `2876d96` | 180/180 positions, 3 arms each, 113 s (+55 s evaluator), 4.0 GB peak; spot audit 30/30 |
| Analysis / audit | `2b2a566` | one serialization-only wrapper added after the first analysis call failed on a numpy bool (decision code unchanged, test added) |

One attempt per stage; no rerun, refill or replacement. Raw lossless bf16 logits stay local
(`rows/*_logits.npz`, sha256 in the committed rows and `directions_sealed.json`).

## Engineering validity (all pass)

Rows complete; directions sealed before the evaluator pass; every bitwise check holds at all 180
positions (extraction state, D2 scratch logits and site vs clean B step, crop-restore, evaluator logits/site);
0/180 historical state-identity mismatches (P2-RJ V3 tie-aware rules; D0 solver s equals P2-R `+d` s within
1e-9 at every position, the D0 historical regression); every arm valid at 180/180 (direction, solver, hook,
target and pairwise squared energy within 2%); no D2 fallback; 180 readout autograd calls (one per
position; median 0.039 s, max 0.27 s); matched three-way population 60/60/60 positions over 17/20/20
dialogues; 10000/10000 valid draws for every family statistic.

## Primary family (Bonferroni 0.05/10, dialogue bootstrap 10000, seed 240924; matched rows)

All values are Δ fixed-competitor margin m = lse z'(Y_ref) − z'(c*) in nats (arm − no-edit), except corruption.

| Candidate | EN-confusion Δm | paired vs D0 | EN-correct Δm | ZH-correct Δm | pooled correct corruption | qualifies |
|---|---|---|---|---|---|---|
| D1 UNIQUE | **+0.227** [+0.079, +0.421] | +0.247 [+0.107, +0.457] | −0.034 [−0.165, +0.103] | −0.040 [−0.099, +0.017] | 0.000 [0, 0] | **NO** (point < 0.5) |
| D2 READOUT | **+4.499** [+3.829, +5.168] | +4.519 [+3.874, +5.169] | +1.535 [+1.042, +2.136] | **−2.011** [−2.466, −1.584] | **0.077** [0.013, 0.163] | **NO** (ZH margin, corruption) |

Observed per-stratum corruption (dialogue-weighted, pointwise 95%): D1 EN 0.000, ZH 0.000;
D2 EN 0.017 [0, 0.05], **ZH 0.137 [0.054, 0.233]** (8/60 ZH-correct top-1 flips; 1/60 EN-correct).
Tie contrast D2 − D1 (not used; neither qualifies): +4.27 [+3.83, +4.74].

D1 benefit (lower > 0, paired > 0) is real but fails the frozen 0.5-nat materiality screen; D1 passes
every safety rule. D2 meets every benefit rule (point ≥ 0.5, lower > 0, paired > 0) and fails safety
(ZH margin lower −2.47 < −0.25; pooled corruption upper 0.163 > 0.05; ZH observed 0.137 > 0.05) ->
causal power with damage. No candidate qualifies -> no selection -> STOP.

## Mechanism and alignment (pointwise 95%, matched rows)

| Stratum / arm | Δm | Δ log p(ref) | ΔJ (readout objective) | Δ log P_E | Δ log P_M | cos(d, g_ref,tan) | κ |
|---|---|---|---|---|---|---|---|
| EN-confusion D0 | −0.020 | −0.026 | +0.001 | +0.003 | +0.001 | −0.011 | −0.011 |
| EN-confusion D1 | +0.227 | +0.195 | +0.099 | +0.088 | −0.011 | +0.020 | +0.021 |
| EN-confusion D2 | +4.499 | +2.146 | +2.348 | +1.989 | −0.358 | **+0.799** | **+0.797** |
| EN-correct D0 | +0.027 | +0.002 | −0.074 | +0.001 | +0.075 | +0.012 | +0.012 |
| EN-correct D1 | −0.034 | +0.002 | −0.141 | +0.005 | +0.146 | −0.002 | −0.001 |
| EN-correct D2 | +1.535 | +0.101 | +2.213 | +0.045 | −2.168 | +0.459 | +0.457 |
| ZH-correct D0 | −0.047 | −0.007 | +0.057 | +0.056 | −0.002 | −0.010 | −0.010 |
| ZH-correct D1 | −0.040 | +0.002 | +0.038 | +0.037 | −0.001 | −0.007 | −0.007 |
| ZH-correct D2 | −2.011 | −0.427 | +2.283 | +2.168 | −0.114 | **−0.584** | −0.582 |

Same realized edit for all arms: relative edit ≈ 0.148-0.152 of ‖h‖ (e* = 1.12608). Available first-order
leverage at confusion states A = ‖g_tan‖ e* = 5.72 [5.18, 6.30] nats; D2 captures κ ≈ 0.80 of it (D0 ≈ 0,
D1 ≈ 0.02). Confusion top-1 corrections: D0 0/60, D1 0/60, D2 5/60 (43/60 top-1 changes). Mutual
cosines: D1 and D2 are near-orthogonal to D0 (|cos| ≤ 0.05) and to each other at confusion states.

## Interpretation (within the frozen screen)

* **D0** reproduces the historical inert result exactly (Δm ≈ 0, κ ≈ 0): the old direction remains the defective
  actuator, consistent with `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`.
* **D1** (development-fitted unique EN-correct vs ZH-correct subspace mode) is a safe but weak lexical lever:
  kappa ≈ 0.02 (Δm ≈ 4% of A); its representational contrast is not the readout-relevant axis.
* **D2** shows that the L16 DG-02 site has large, well-aligned leverage when the direction is the reference-free
  readout tangent: +4.5 nats at e* with κ ≈ 0.80 against the evaluator gradient it never saw. Its failure is
  selectivity, not power: applied without any gate it pushes ZH-correct states toward English (κ −0.58,
  13.7% corruption). Exp-1 deliberately removes gating, and the frozen rule does not let a candidate that
  fails safety progress to the gate-coupled screen. The result therefore stops the stage. It does not test
  whether the current gate would supply the missing selectivity, and no claim about that is made.
* This is conditional development evidence on exposed states (no independent confirmation, no transcript claim).

## Post-run independent audit (`results/inference_cf/p2dir/exp1_run1_audit.json`)

`experiments/inference_cf_p2dir_audit.py exp1`, which does not import the analysis module:

* provenance: manifest self-hash, config hash, every source at the manifest commit, positions/construction
  hashes, sealed-file hashes;
* population: 180 frozen keys, order identical to the construction projection;
* D1: all 20 folds independently refit from the extraction states (own eigh/SVD/score/sign code): same
  status, same selected index, vectors within 2e-6;
* D0: independent float64 recomputation from saved h_E/h_B within 1e-6 at all positions;
* D2: tangent/unit recomputed from the saved raw readout gradient (≤ 2e-6), J recomputed from saved
  logits (≤ 1e-4), and the GPU spot check of 30 positions with an independent no-grad cache, probe and
  objective. All 30 match: site bitwise, J ≤ 1e-5, direction ≤ 2e-6, no parameter gradients;
* energy: realized edit norms recomputed from the saved before/after site states within the frozen 2% bound
  and within 5e-3 of the hook norm;
* statistics: all ten family estimates/intervals agree within 8.9e-16; valid-draw counts, matched set,
  valid rates, qualification and the terminal label/selection agree exactly.

Verdict: **`P2_DIR_AUDIT: PASS`** (EXP1), label `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`, selected none.
