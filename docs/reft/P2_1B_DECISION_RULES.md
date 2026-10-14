# P2-1b — pre-registered decision rules

These rules were written before any P2-1b training or KL outcome existed. They are implemented in
`reft/p2_1b_evaluate.py` (`decide`, `preservation_success`, `correction_retained`, …) and unit-tested in
`tests/reft/test_p2_1b.py`.

## Inputs

**Runs.**
- For each seed s ∈ {42, 43, 44}, the **selected** checkpoint of R2-CE_s and R2-KL_s, each chosen by the unchanged P2-1
  rule.
- Evaluation: dev150, frozen-LID beam-5, paired against the pinned P2 baseline B0.
- R2-CE_42 is the historical P2-1 run if the equivalence gate passes; otherwise it is the fresh control.

**Definitions** (identical code for every run).
- `zh_new`: B0-correct non-filler Mandarin reference units that become wrong.
- `retention = 1 − zh_new / 4950`.
- `SB`: the **long switch-back dropped units**. These are newly wrong Mandarin units that the system deletes, lying in a
  maximal run of ≥ 5 consecutive system-deleted reference units whose preceding reference unit is a substantive English
  unit.
- `EN_net`: substantive English corrections − corruptions against B0.
- Loops and non-terminations: from the canonical scorer.
- `worst_dlg`: the largest per-dialogue increase in mixed errors against B0.

## Integrity → `P2_1B_ENGINEERING_BLOCKED`

The verdict is ENGINEERING_BLOCKED if any of the following fails:
- training, selection, evaluation or verification failed;
- zero-init reproduction < 150/150;
- weights changed;
- reload ≠ recorded decodes;
- a freeze or baseline hash mismatched;
- a run was missing.

## Preservation success `P`, all required

| Condition | Requirement |
|---|---|
| P1 | retention ≥ 0.99 (zh_new ≤ 49) in ≥ 2 of 3 KL seeds |
| P2 | mean zh_new(KL) < mean zh_new(CE) |
| P3 | mean over seeds of (SB_CE − SB_KL) / SB_CE ≥ 0.30, using seeds with SB_CE > 0 only (undefined, and therefore failed, if none), and SB_KL < SB_CE in ≥ 2 seeds |
| P4 | Σ loops(KL) ≤ Σ loops(CE) + 1 and Σ non-terminations(KL) ≤ Σ non-terminations(CE) + 1 |

## Damage improvement `D`, a weaker condition used for the verdict order

- P2 holds, **and**
- either P3 holds, or the mean zh_new reduction is ≥ 30% of mean zh_new(CE).

## Correction retention `R`, all required

| Condition | Requirement |
|---|---|
| R1 | EN_net(KL) > 0 in ≥ 2 KL seeds |
| R2 | ratio = mean EN_net(KL) / mean EN_net(CE) ≥ 0.75. Defined only if mean EN_net(CE) > 0; otherwise R2 fails and the ratio is reported as undefined. |
| R3 | mean CS MER(KL) − mean CS MER(CE) ≤ 0.005 |
| R4 | for every seed, worst_dlg(KL_s) ≤ worst_dlg(CE_s) + 10 |

## Seed instability `U`

U holds if the matched differences Δ_s = zh_new(KL_s) − zh_new(CE_s), **or** Δ_s on SB, contain both a value ≤ −10 and
a value ≥ +10 across the three seeds.

## Terminal verdict (first matching rule)

1. `P2_1B_ENGINEERING_BLOCKED` if integrity fails.
2. `P2_1B_PRESERVATION_RESCUED` if P and R.
3. `P2_1B_SEED_UNSTABLE` if U.
4. `P2_1B_KL_SUPPRESSES_CORRECTIONS` if D and not R.
5. `P2_1B_KL_NOT_EFFECTIVE` if not D: little or no damage reduction. This covers mean zh_new reduction < 30% with P3
   failing, or P2 failing.
6. `P2_1B_INCONCLUSIVE` otherwise. For example, D and R hold but P fails, which counts as **partial preservation**.

**Partial success.** Every component (P1–P4, R1–R4, D, U) is reported. When P and R disagree, the components are shown
and no verdict is overridden.

**Separate reporting.** Each run against B0 is also reported with the P2-1 rules (`useful`, `preservation_ok`). That
does not change the P2-1b verdict.

## Prefix-diagnostic classification (post hoc, descriptive; never part of the verdict)

Cases are fixed in `p2_1b/prefix_audit/cases_private.json` before training. For each damaged case and system S against
B0:

| Type | Rule |
|---|---|
| **A** (gold-prefix degradation) | at the gold prefix of the first Mandarin token after the English span, log p_S(gold) ≤ log p_B0(gold) − 1.0 nat, or the gold token is B0's argmax but not S's |
| **B** (generated-prefix instability) | not A, and at the first divergence of S's beam-5 output from B0's, S's next token is EOS, a Latin-script token or an English-continuation token while B0's is Han |
| **C** | otherwise (mixed, or unresolved alignment) |

A type is called dominant when it covers ≥ 60% of the classifiable damaged cases.
