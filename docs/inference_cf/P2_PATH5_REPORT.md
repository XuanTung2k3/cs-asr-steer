# P2-PATH5 — frozen A2 + G1A on NEW200 (report, terminal)

**Verdict: `P2_PATH5_SAFE_NO_ADDED_VALUE`.** On the broader development population NEW200, A2+G1A passes absolute safety
and strong A2 retention (1.00). It produces **no positive Mandarin rescue**: R_plus 0, R_minus 2, R_net −2. Independent
audit `P2_PATH5_AUDIT: PASS` (primary phase before references, then full); the audit's own label agrees.

**Main-method decision (frozen hard-stop rule):**
- The G1 branch-controller line is **closed** as a main-method direction.
- The **main adaptation candidate returns to A2.**
- No further controller revision: no EOS, H, weight or threshold tuning, no retrigger, no G2, no row selection.

**Scope:** development/breadth decision stage on already-exposed D-dev-select data. It is not fresh validation and
makes no significance claim.

| Step | Commit / job | Result |
|---|---|---|
| OPP0 derived outputs (no GPU) | `7745e5b` | committed before references |
| OPP0 evaluation + PATH5 freeze | `d1d057d` | spec/design/panel/config pushed before any PATH5 outcome |
| Implementation + plan | `14bd563` | runner/analyzer/auditor/sbatch; 11 focused tests; regression 91/91 |
| Pre-run audit | `01d57b8` | `PASS_TO_P2_PATH5` |
| Manifest | `48f0ace` | pushed before the job |
| Run | Slurm 58074 | 13 min 55 s allocation (809 s runtime) |
| Output seal + primary analysis | `101ca00` | pushed before any NEW200 reference |
| Primary-phase audit | `e1086f2` | `P2_PATH5_AUDIT: PASS`; opportunity gate open |
| Secondary + full audit + report | this commit | `P2_PATH5_SAFE_NO_ADDED_VALUE`; `P2_PATH5_AUDIT: PASS` |

## Phase 0 — OPP0-ABSTAIN (fixed100, derived, no GPU)

| System | ZH | POI | Mixed | PIER | MER |
|---|---:|---:|---:|---:|---:|
| A2 | 1116 | 316 | 1464 | 0.4540 | 0.2537 |
| PATH4-R1 G | 1108 | 324 | 1464 | 0.4655 | 0.2537 |
| A2+G1A | 1093 | 314 | 1439 | 0.4511 | 0.2494 |

- Abstention removes all PATH4 EOS harm: every EOS row equals A2 exactly, including U1004_S0_221.
- All 10 content G1 results are preserved.
- The fixed100 rescue (23) is 100% in PATH2_12. See `P2_OPP0_ABSTAIN_REPORT.md`.

## Population

**FULL300** is the canonical P0-R2 / P2-A r1 D-dev-select panel (20 dialogues × 15). It is the byte-pinned parent of
fixed100, with role and dialogue verified from ID/dialogue/role columns only.
- **NEW200** = FULL300 − fixed100: 200 rows, 20 dialogues × 10, membership hash `e23a8ce5…`.
- **Exposure:** NEW200 was never used in the TTA/A2/PATH chain. Its references were previously used in DG-04/06/07, P0-R2
  and P2 selection, so it is not fresh validation.
- **B0 on NEW200:** the live canonical theta0 cached forced decode. P2-A `systems.B0` differs from the canonical S0 on
  19/100 fixed100 rows and was not used.

## Fixed100 barrier (inside the one job, before NEW200)

**PASS.**
- A2 reconstruction is exact on 100/100: tokens, termination and text equal sealed A2; 194 fp32 masters equal the
  archives; bf16 effective state matches; theta0 equals sealed B0.
- Live A2+G1A equals the OPP0 derived target on 100/100.
- The 10 content rows reproduce the sealed PATH4 detector, decision, score paths and G1 exactly.
- The 4 EOS rows and 86 no-trigger rows equal A2.
- Live check: loss difference 6.0e−8, relative gradient difference 0.

## Reference-free NEW200

- **Rows:** 200, with 30 first disagreements and 170 no-trigger rows.
- **Content/content (22):** theta0 won 7 and A2 won 15.
- **EOS abstentions (8):** all theta0-EOS orientation.
- **G1A-changed transcripts:** 7, in 6 dialogues: U0011_S0_265, U0021_S0_118, U0023_S0_91, U0085_S0_108, U0085_S0_99,
  U1063_S0_56, U2004_S0_118.
- **Opportunity gate (frozen minimum 3 rows and 3 dialogues):** observed 7 / 6 → **PASS**, so references were opened
  after the pushed seal and the primary audit.

## NEW200 ASR

| System | PIER | MER | EN-WER | ZH-CER | ZH | POI | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-FORCED | 0.4637 | 0.2688 | 0.4656 | 0.2330 | 2202 | 729 | 2973 |
| B0-AUTO | 0.3836 | 0.2608 | 0.3880 | 0.2367 | 2237 | 603 | 2885 |
| A2 | 0.4427 | 0.2406 | 0.4466 | 0.2033 | 1921 | 696 | 2662 |
| **A2+G1A** | 0.4427 | 0.2408 | 0.4466 | 0.2035 | 1923 | 696 | 2664 |

- **S/D/I:** B0 748/1986/239, A2 728/1694/240, G1A 730/1692/242.
- **Caps:** 1 for every system.
- **Severe truncations:** 0.
- **FULL300 (secondary, ZH/POI/mixed):** B0 3309/1074/4456, A2 3037/1012/4126, G1A 3016/1010/4103.

## Gates (NEW200)

| Gate | Values | Result |
|---|---|---|
| Absolute safety (G1A vs B0) | ΔMER −0.0279; ΔZH-CER −0.0295; ZH ret 0.9988; EN ret 0.9976; outside harm 0.0012; corruption 0.0024; added caps 0; severe 0 | **PASS** |
| Rescue | R_plus 0, R_minus 2, R_net −2 | **R_net ≤ 0 → SAFE_NO_ADDED_VALUE** |
| Strong A2 retention (not reached) | I_A2 33, I_G 33, retention 1.00; ΔPIER 0; ΔMER +0.0002 | pass |
| Breadth (not reached) | 0 rescue rows / dialogues; C_utt = C_dlg = 1 | fail |

Precedence: INVALID → OPPORTUNITY_SPARSE → DAMAGE → **SAFE_NO_ADDED_VALUE**.

**All rescue rows:** none.

**All harm rows:**
- U0023_S0_91 (CSD0012): POI +1, mixed +1.
- U0085_S0_108 (CSD0043): ZH +2, mixed +2.

Of the other theta0-win changes, U0021_S0_118 gains POI −1 / mixed −1, and the remaining four are count-neutral.

**Opportunity-outcome map (NEW200; deltas are G1A − A2, plus A2 − B0 for context):**

| Class | Rows | ΔZH | ΔPOI | ΔMixed | A2 − B0 ZH / POI |
|---|---:|---:|---:|---:|---|
| NO_TRIGGER | 170 | 0 | 0 | 0 | 0 / 0 |
| EOS_ABSTAIN | 8 | 0 | 0 | 0 | **−274 / −11** |
| CONTENT_A2_WIN | 15 | 0 | 0 | 0 | −5 / −22 |
| CONTENT_THETA0_WIN_NO_CHANGE | 0 | — | — | — | — |
| CONTENT_THETA0_WIN_CHANGED | 7 | +2 | 0 | +2 | −2 / 0 |

**LODO (descriptive):** removing any of the 20 dialogues leaves net rescue between −2 and 0; 0/20 positive.

**Bootstrap (2000 paired dialogue blocks, seed 240924; descriptive):**
- G1A − A2: MER +0.0002 [−0.0002, 0.0007], PIER 0 [−0.0019, 0.0018], ZH-CER +0.0002 [0, 0.0007], POI 0 [−3, 3].
- G1A − B0: MER −0.0279 [−0.0545, −0.0055], PIER −0.0210 [−0.0404, −0.0061], ZH-CER −0.0295 [−0.0607, −0.0040],
  POI −33 [−61, −10].

## Interpretation and method decision

- **The controller has opportunity but no value.** G1A acted on 22 content disagreements and changed 7 transcripts in 6
  dialogues. None of those changes rescued Mandarin, and two caused small harm. The PATH2/fixed100 rescues
  (U0023_S0_664, U0091_S0_68) do not recur on new rows. The bottleneck is **no added value**, not sparsity, safety or
  concentration.
- **Abstention is what protects A2.** NEW200's large A2 gain over B0 (ZH −281, MER −0.028) comes almost entirely from
  the 8 EOS-abstain rows (ZH −274). There, theta0/B0 stops very early (for example 7 vs 112 tokens, 4 vs 91 tokens) and
  A2 recovers the speech. G1A keeps these exactly as A2, whereas PATH4-R1's EOS arbitration could have discarded them.
  This is descriptive, not a new claim.
- **Decision:** close the G1 controller line and keep **A2** as the main adaptation candidate. The next research
  question is upstream adaptation/objective work, not another G1 patch. Promoting A2 itself would require its own
  separately frozen independent confirmation.

**Data:**
- Used: already-exposed D-dev-select FULL300 only (fixed100 + NEW200).
- Not used: fresh validation, D-dev-confirm, D-test, router-calib (new role), P3, transfer.

**Compute:**
- One job: 13 min 55 s (809 s runtime); peak 7.53 GB allocated / 8.16 GB reserved.
- A2: 600 updates, 600 + 1 backward passes.
- Controller: 300 detector runs, 44 triggers (32 content G1 events, 12 EOS abstentions), 64 rollouts, 128 score paths,
  32 G1 decodes; 0 boundary scores, 0 G2, 0 A4.
