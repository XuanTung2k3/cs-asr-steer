# P2-OPP0-ABSTAIN — derived A2+G1A on fixed100 (post-hoc development diagnostic, no GPU)

**What this is.** A deterministic counterfactual built from already-sealed PATH4-R1 artifacts. It is not new validation,
it ran no new decode, and it is never used to tune G1A.

**G1A (abstaining consensus guard).** At the first theta0/A2 disagreement, content/content runs the original frozen G1;
any EOS-boundary disagreement abstains to A2; no disagreement leaves the output as A2.

**Derivation** (`experiments/inference_cf_p2opp0_abstain.py derive`):
- Inputs are the committed PATH4 output seal and the unchanged sealed files; both PATH4 audits PASSed.
- Each row's output is chosen only from its sealed live dispatch mode:
  - CONTENT_G1 → the sealed PATH4 G output. This is bit-identical to the original G1, as verified by PATH4's
    direct-original comparator.
  - EOS_BOUNDARY or NO_TRIGGER → the sealed A2 output.
- All 100 IDs are accounted for: 86 A2 no-trigger, 10 PATH4 content G1, 4 A2 EOS-abstain.
- No manual transcript editing, and no reference-dependent selection.
- `results/inference_cf/p2opp0/derived_g1a.json` (hash `sha256:33bcd466…`) was committed (`7745e5b`) **before**
  references were opened for `evaluate`. These fixed100 references were already exposed.

## Fixed100 results

| System | PIER | MER | EN-WER | ZH-CER | ZH err | POI err | Mixed err |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-FORCED | 0.4957 | 0.2570 | 0.5029 | 0.2193 | 1107 | 345 | 1483 |
| B0-AUTO | 0.4080 | 0.2591 | 0.4152 | 0.2336 | 1179 | 284 | 1495 |
| A2 | 0.4540 | 0.2537 | 0.4626 | 0.2211 | 1116 | 316 | 1464 |
| PATH4-R1 G | 0.4655 | 0.2537 | 0.4741 | 0.2195 | 1108 | 324 | 1464 |
| **A2+G1A (derived)** | **0.4511** | **0.2494** | **0.4598** | **0.2165** | **1093** | **314** | **1439** |

A2+G1A vs B0 has 31 POI corrections and 0 corruptions. ZH retention is 0.9985, EN retention 1.0, outside harm 0.0015,
caps 1 (B0 also has 1), and 0 severe truncations.

## Phase-0 questions

| Question | Answer |
|---|---|
| 1. Does abstention eliminate the PATH4 EOS harm? | **Yes.** All 4 EOS rows have zero ZH/POI/mixed delta vs A2. U1004_S0_221's +14 ZH / +10 POI / +24 mixed disappear, as does U0091_S0_196's +1 ZH. |
| 2. Is every content/content G1 result preserved? | **Yes.** All 10 content rows equal the sealed PATH4 G exactly. |
| 3. Does U1004_S0_221 revert exactly to A2? | **Yes.** |
| 4. Does any EOS-boundary row differ from A2 under G1A? | **No.** |
| 5. Is fixed100 positive rescue still concentrated in the old PATH2 rows? | **Yes.** R_plus is 23 with R_minus 0. The two rescue rows are U0023_S0_664 (−20 ZH, −20 mixed) and U0091_S0_68 (−3 ZH, −2 POI, −5 mixed), both in PATH2_12, so the PATH2_12 share is 1.0. |

## Conclusion

On fixed100, abstention is exactly the intended safety repair: it removes all PATH4 EOS damage and keeps every content
G1 decision. A2+G1A is better than A2 on every reported metric. However, its Mandarin rescue still comes entirely from
the two old PATH2 diagnostic rows, so fixed100 cannot show breadth. Whether G1A has broad value is tested once, frozen,
on NEW200 (`P2_PATH5_SPEC.md`). No fixed100 result here is used to change G1A.
