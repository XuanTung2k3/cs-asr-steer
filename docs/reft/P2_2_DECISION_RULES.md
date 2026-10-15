# P2-2 — pre-registered decision rules

These rules were written before any P2-2 coverage statistic or training outcome existed. They are implemented in
`reft/p2_2_evaluate.py` (`preservation`, `retention`, `strong`, `exposure`, `decide`, `RECOMMEND`) and
`reft/p2_2_train.py` (`coverage_gate`), and unit-tested in `tests/reft/test_p2_2.py`.

## Inputs

- **Checkpoints.** The selected checkpoints (unchanged P2-1 rule) of GoldKL_s, CE_s and M2_s, for s ∈ {42, 43, 44}.
- **Evaluation.** dev150, frozen-LID beam-5, against the pinned B0.

**Definitions** (the P2-1b code):

| Term | Definition |
|---|---|
| `zh_new` | new Mandarin errors |
| retention | 1 − zh_new / 4950 |
| SB | long switch-back dropped units |
| EN_net | English net corrections against B0 |
| loops, non-terminations | from the canonical scorer |
| `worst_dlg` | the largest per-dialogue increase in mixed errors |

## Preflight

1. **Coverage gate.** Separately on the CE42 and KL42 training rollouts, all of the following:
   - ≥ 30 valid events;
   - ≥ 20 utterances;
   - ≥ 6 dialogues;
   - ≥ 20 valid SB events;
   - ≥ 10 valid non-identity-prefix events;
   - 0 label mismatches.

   Failure → `P2_2_BLOCKED_ONPOLICY_COVERAGE`. No training; recommendation `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING`.
2. **Engineering gates G1, G4, G5, G6, G7** (protocol section 5). Failure → `P2_2_BLOCKED_<gates>`. No training;
   recommendation `HOLD_FOR_ENGINEERING_REPAIR`.

## Component decisions, M2 vs matched GoldKL

**A — preservation improvement** (all required):

| Condition | Requirement |
|---|---|
| A1 | mean zh_new(M2) < mean zh_new(GoldKL) |
| A2 | mean over seeds with SB_KL > 0 of (SB_KL − SB_M2) / SB_KL ≥ 0.20 (undefined → fail) |
| A3 | SB_M2 < SB_KL in ≥ 2 seeds |
| A4 | Σ loops(M2) ≤ Σ loops(KL) + 1 and Σ non-terminations(M2) ≤ Σ non-terminations(KL) + 1 |

**B — correction retention** (all required):

| Condition | Requirement |
|---|---|
| B1 | EN_net(M2) > 0 in ≥ 2 seeds |
| B2 | mean EN_net(M2) / mean EN_net(KL) ≥ 0.75 (undefined if the KL mean is ≤ 0 → fail) |
| B3 | mean CS MER(M2) − mean CS MER(KL) ≤ 0.005 |
| B4 | for every seed, worst_dlg(M2_s) ≤ worst_dlg(KL_s) + 10 |

**C — strong preservation.** zh_new(M2) ≤ 49 (retention ≥ 0.99) in ≥ 2 seeds. It is reported separately.

**D — on-policy validity**, summed over epochs 2–3 of each run. A seed passes D if:
- at least **30** active on-policy events whose prefix is non-identity contributed gradient; **and**
- the valid share (T + R) / located is at least **0.5**.

D holds if at least 2 seeds pass.

## Integrity (all required, else ENGINEERING_BLOCKED)

- reload verification all ok;
- 9 runs present;
- epoch-0 identity 150/150;
- weights unchanged;
- C0 and C1 reproduce the P2-1b evaluation;
- the mechanism audit is present;
- the epoch-1 rollouts are identical to B0 (500/500);
- 0 label mismatches in every training epoch.

## Terminal verdict (strict precedence)

1. `P2_2_ENGINEERING_BLOCKED`: integrity fails.
2. `P2_2_ONPOLICY_EXPOSURE_INSUFFICIENT`: D fails. This is never hidden by a metric gain.
3. `P2_2_ONPOLICY_FEASIBLE`: A and B hold. Substatus `P2_2_PRESERVATION_RESCUED` if C also holds.
4. `P2_2_CORRECTIONS_SUPPRESSED`: A holds and B fails.
5. `P2_2_NO_INCREMENTAL_GAIN`: A fails, the mean zh_new reduction is < 10%, **and** the mean SB reduction is < 10%
   (or undefined).
6. `P2_2_INCONCLUSIVE`: otherwise, i.e. a partial or unstable benefit.

All components are reported. The verdict is never upgraded by hand.

## Recommendation (one; it is not authorization)

| Verdict | Recommendation |
|---|---|
| FEASIBLE (± RESCUED) | `GO_P2_3_MATCHED_SUPERVISION_CONTROL`: attribution to on-policy specificity first, then independent confirmation |
| CORRECTIONS_SUPPRESSED | `GO_P2_3_ONPOLICY_MECHANISM_VALIDATION` |
| INCONCLUSIVE | `GO_P2_3_ONPOLICY_MECHANISM_VALIDATION` |
| NO_INCREMENTAL_GAIN | `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING`: objective-specific failure with valid coverage |
| EXPOSURE_INSUFFICIENT, or blocked by coverage | `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING`: the rollout mechanism did not produce informative supervision |
| ENGINEERING_BLOCKED, or other preflight blocks | `HOLD_FOR_ENGINEERING_REPAIR` |

`STOP_CURRENT_REFT_LINE` is not selected by these rules. This pilot tests one objective, not representation
adaptation as a whole.

The mechanism audit on the ten P2-1c cases is exploratory and does not enter the verdict.
