# P2-PATH4-R1 — frozen G1 transfer to A2 on fixed100, EOS_BOUNDARY_V1 (report, terminal)

**Verdict: `P2_PATH4_OVERCONSERVATIVE`.** A2+G (PATH2 CONSENSUS-1 with the R1 EOS-boundary extension) passes
ABSOLUTE_SAFETY and AGGREGATE_RESCUE. It fails BENEFIT_RETENTION: POI retention is 21/29 = 0.724 (< 0.75) and PIER is
+0.0115 above A2 (> 0.01). BREADTH also fails, but precedence stops at OVERCONSERVATIVE. Independent audit
`P2_PATH4_AUDIT: PASS` (primary phase before references, then full); the audit's own label agrees.

**Contract revision:** `PATH4_R1_EOS_BOUNDARY_V1`, human-authorized after the original freeze was blocked pre-run
(`P2_PATH4_BLOCKED_EOS_FIRST_BRANCH`, commit `29661e9`). That provenance is preserved unchanged: the archived config
`p2_path4_blocked_v0.json`, the BLOCKED panel status, and the verbatim revision-0 spec/design bodies. Only the EOS-first
restriction changed. Content/content G1, A2, thresholds, partitions and precedence are revision 0.

**Scope:** breadth-transfer DEVELOPMENT on already-exposed D-dev-select fixed100. It is not fresh validation and makes
no generalization or significance claim.

| Step | Commit / job | Result |
|---|---|---|
| Revision-0 freeze, blocked pre-run | `29661e9` | 4 theta0-EOS / A2-content first disagreements |
| R1 amendment + freeze contract audit | `490ec20` | `PASS_TO_P2_PATH4_R1` (freeze scope) |
| Implementation + sealed plan | `bd5db02` | runner/analyzer/auditor/sbatch; 9 focused tests; required CPU suites 71/71 |
| Runnable pre-run audit | `316b727` | `PASS_TO_P2_PATH4_R1` (manifest/code/environment/reconstruction/cache) |
| Manifest pushed | `c57936a` | before the job |
| Run | Slurm 58035 | 7 min 31 s allocation (426 s runtime; phase 1 154 s, phase 2 266 s) |
| Output seal + primary analysis | `4781605` | pushed before any reference access |
| Primary-phase audit | `fcf000f` | `P2_PATH4_AUDIT: PASS` before references |
| Secondary + full audit + report | this commit | `P2_PATH4_OVERCONSERVATIVE`; `P2_PATH4_AUDIT: PASS` |

Re-check before implementation: the required CPU suites (TTA0, funnel, PATH1/2/3, PATH4-R1) passed 71/71. The freeze
contract auditor re-run passed every check except `scope_no_runner_or_scientific_output`. That check is by design: it
certifies that the freeze session had no runner, and this session's runner now exists.

## Reconstruction barrier (all 100, before any G1)

100/100 rows passed, all exact: theta0 decode = B0, free A2 = sealed A2 (tokens, termination, text), all 194 fp32 final
masters = the historical archive arrays, bf16 effective state =, and exact reset. These used the unchanged TTA1
`run_objective` (2 AdamW updates per row, 200 total). The independent live check on row 0 gave a loss difference of
6.0e−8 and a relative gradient difference of 0.

## Reference-free controller

The single logical detector fired on exactly the 14 A2_DELTA rows. Dispatch by live action types gave
**10 CONTENT_G1 + 4 EOS_BOUNDARY**; the 86 no-trigger rows equal A2 exactly. On every content and no-trigger row the
engineering-only direct-original comparator (PATH3 `online_g1` from the same saved A2 state) matched bit-for-bit. On all
4 boundary rows the original G1 refused as EOS-first, reproducing the revision-0 block.

| | FULL100 | DEV24 | NOVEL76 (= A2_DELTA_OUTSIDE_DEV24 triggers) | PATH2_12 |
|---|---:|---:|---:|---:|
| Triggers (content / EOS) | 14 (10 / 4) | 7 (6 / 1) | 7 (4 / 3) | 6 (6 / 0) |
| theta0 / A2 winners | 6 / 8 | 3 / 4 | 3 / 4 | 3 / 3 |
| G ≠ A2 | 6 | 3 | 3 | 3 |
| Trigger dialogues | 9 | 4 | 6 | 3 |

Margin quantiles [0, .25, .5, .75, 1] (theta0 − A2, triggers only):
- FULL100: −1.095, −0.116, −0.026, 0.044, 0.922.
- NOVEL76: −0.938, −0.105, −0.077, 0.022, 0.125.

**EOS_BOUNDARY (H=1, before references).** All 4 have the theta0-EOS orientation.

| Row | k | Margin | Winner | Action | G length / A2 / B0 |
|---|---:|---:|---|---|---|
| U0027_S0_116 (DEV24) | 75 | −0.375 | A2 | content | 85 / 85 / 75 |
| U0086_S0_222 | 35 | −0.938 | A2 | content | 36 / 36 / 35 |
| U0091_S0_196 | 84 | +0.031 | theta0 | EOS | 84 / 86 / 84 |
| U1004_S0_221 | 50 | +0.125 | theta0 | EOS | 50 / 84 / 50 |

Every boundary row terminated with EOS, with no new severe truncation.

## FULL100 ASR

| System | ZH err | POI err | Mixed err | PIER | MER | EN-WER | ZH-CER | S / D / I | Caps |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| B0-FORCED | 1107 | 345 | 1483 | 0.4957 | 0.2570 | 0.5029 | 0.2193 | 339 / 1026 / 118 | 1 |
| B0-AUTO | 1179 | 284 | 1495 | 0.4080 | 0.2591 | 0.4152 | 0.2336 | 324 / 1053 / 118 | 1 |
| A2 | 1116 | 316 | 1464 | 0.4540 | 0.2537 | 0.4626 | 0.2211 | 351 / 985 / 128 | 1 |
| **A2+G** | **1108** | 324 | 1464 | 0.4655 | 0.2537 | 0.4741 | **0.2195** | 324 / 1015 / 125 | 1 |
| PATH3 A4+G1 (sealed, display) | 1110 | 327 | 1469 | 0.4698 | 0.2545 | 0.4784 | 0.2199 | 324 / 1026 / 119 | 1 |

- The historical aggregates reproduce exactly: B0 1107/345/1483, A2 1116/316/1464, POI denominator 696.
- G vs B0: 21 POI corrections, 0 corruptions. ZH retention 0.9985, EN retention 1.0, outside harm 0.0015, added caps
  0, severe truncations 0.
- G − AUTO: ZH −71, mixed −31, but POI +40 and PIER +0.057.

## Frozen gates

| Gate | Values | Result |
|---|---|---|
| ABSOLUTE_SAFETY (G vs B0) | ΔMER −0.0033; ΔZH-CER +0.0002; ZH ret 0.9985; EN ret 1.0; outside harm 0.0015; corruption 0; added caps 0; severe 0 | **PASS** |
| AGGREGATE_RESCUE | X_ZH 9, required 5, R_ZH 8 (Z_G 1108 ≤ 1111) | **PASS** |
| BENEFIT_RETENTION | I_A2 29, I_G 21, retention 0.724 (< 0.75 ✗); PIER_G − PIER_A2 +0.0115 (> 0.01 ✗); MER +0.0000 ✓; mixed 1464 ≤ 1464 ✓; POI 324 > cutoff 322 | **FAIL** |
| BREADTH (FULL100 + NOVEL76) | rescue rows 2 (≥ 3 ✗), dialogues 2 (≥ 3 ✗), C_utt = C_dlg 0.870 (✗); NOVEL76 net −15 (≥ 1 ✗), NOVEL76 rescue rows 0 (≥ 1 ✗) | FAIL (not reached by precedence) |

Precedence INVALID → DAMAGE → NO_ADDED_VALUE → OVERCONSERVATIVE gives **`P2_PATH4_OVERCONSERVATIVE`**.

## Rows, concentration, LODO, bootstrap

- **Rescues (2 rows, R_plus 23):**
  - U0023_S0_664 (CSD0012): ZH −20, mixed −20 (theta0 content win).
  - U0091_S0_68 (CSD0046): ZH −3, POI −2, mixed −5 (theta0 content win).
  - Both are in PATH2_12. The max row and max dialogue share are both 0.870.
- **New harms (2 rows, R_minus 15, both NOVEL76 EOS wins):**
  - **U1004_S0_221 (CSD0502): ZH +14, POI +10, mixed +24.** theta0 EOS won by +0.125 and dropped A2's 34-token
    continuation, reverting the row to B0.
  - U0091_S0_196: ZH +1, mixed +1.
  - The two EOS-content wins were exact A2 identity (no change).
- **Other theta0 content wins:** U0039_S0_380 and U0092_S0_101 changed one token with no count change.
- **LODO (descriptive):**
  - All 20 dialogues: net rescue min −12, median 8, max 22; 19/20 positive.
  - The 4 changed dialogues (CSD0012/0020/0046/0502): min −12, median 7, max 22; 3/4 positive.
- **Bootstrap (2000 paired dialogue blocks, seed 240924, all draws valid; descriptive):**
  - G − A2: PIER +0.0115 [−0.0078, 0.0438], MER 0 [−0.0120, 0.0110], ZH-CER −0.0016 [−0.0134, 0.0071], POI +8 [−6, 30].
  - G − B0: PIER −0.030 [−0.076, −0.003], MER −0.0033 [−0.0071, 0], ZH-CER +0.0002 [−0.0045, 0.0056], POI −21 [−53, −2].

## Scientific interpretation

- **A2 gives G1 real opportunities.** Unlike A4 in PATH3, A2 triggers on 7 NOVEL76 rows, and the transfer runs safely.
  Content/content G1 behaves exactly as in PATH2. It still recovers U0023_S0_664 and adds U0091_S0_68.
- **The EOS boundary is where value is lost.** On U1004_S0_221, H=1 next-action consensus preferred theta0's
  early EOS by a small margin. That threw away a 34-token A2 continuation that the references score as substantially
  better than B0's early stop. That single action costs 10 POI errors, which flips retention below 0.75 and
  PIER above the 0.01 guard, and it produces almost all of the NOVEL76 net ZH harm.
- **The G1 rescue signal is still concentrated in PATH2_12.** No NOVEL76 row was rescued.
- **These observations are descriptive only.** They do not authorize an EOS-rule or threshold change after outcomes.
  Any redesign of end-of-utterance arbitration would need a separately frozen contract.

## Next action

STOP. No G1/A2/EOS-rule/threshold change, G2, full300, P3 or transfer. A further revision of the boundary arbitration
would require a new, separately frozen human-authorized contract.

**Data:**
- PATH4 used only the already-exposed D-dev-select fixed100 and sealed B0/AUTO/A2/PATH3 artifacts.
- The 7 NOVEL76 trigger rows are TTA1-exposed, not fresh validation.
- No D-dev-confirm, D-test, router-calib, full300, P3 or transfer data.
