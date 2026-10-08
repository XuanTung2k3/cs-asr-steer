# Current separate inference-time ticket — P2-PATH5 (2026-10-08): terminal `P2_PATH5_SAFE_NO_ADDED_VALUE` — G1 controller line CLOSED; main candidate A2

OPP0 (no GPU): derived A2+G1A on fixed100 ZH 1093 / POI 314 / mixed 1439 (EOS harm removed; rescue only PATH2_12). PATH5 freeze `d1d057d`;
`PASS_TO_P2_PATH5` (`01d57b8`); manifest `48f0ace`; Slurm 58074 (13.9 min); seal `101ca00` pushed before NEW200 references; `P2_PATH5_AUDIT:
PASS` (primary + full). FIXED100 barrier exact (A2 100/100, G1A == derived 100/100). NEW200 (FULL300 - fixed100, already-exposed, not fresh):
30 disagreements (22 content, 8 EOS abstain), 7 G1A-changed rows / 6 dialogues -> gate open; safety PASS, retention 1.00, but R_plus 0,
R_net -2 -> SAFE_NO_ADDED_VALUE. Hard stop: no further controller revision; main adaptation candidate returns to A2 (A2 vs B0 on NEW200:
ZH 2202 -> 1921, MER 0.269 -> 0.241, mostly from 8 early-EOS rows that A2 continues). Report `docs/inference_cf/P2_PATH5_REPORT.md`. Core v6 unchanged.

<!-- superseded header (P2-PATH4-R1 result) -->
## (Previous header) Current separate inference-time ticket — P2-PATH4-R1 (2026-10-07): terminal `P2_PATH4_OVERCONSERVATIVE`

Revision `PATH4_R1_EOS_BOUNDARY_V1` (freeze `490ec20`; original `P2_PATH4_BLOCKED_EOS_FIRST_BRANCH` `29661e9` preserved). Runnable
`PASS_TO_P2_PATH4_R1` (`316b727`); manifest `c57936a`; Slurm 58035 (7.5 min); seal `4781605` pushed before references;
`P2_PATH4_AUDIT: PASS` (primary + full). All-100 A2 reconstruction exact; 14 triggers = A2_DELTA (10 CONTENT_G1 bit-identical to original
G1, 4 EOS_BOUNDARY H=1: 2 EOS wins, 2 content wins). A2+G vs B0: safety PASS; aggregate rescue PASS (R_ZH 8 >= 5, Z_G 1108);
**benefit retention FAIL** (I_G 21/29 = 0.724; PIER_G - PIER_A2 = +0.0115 > 0.01; POI 324 > 322); breadth also fails (NOVEL76 net -15).
The decisive loss is one EOS-boundary theta0 win (U1004_S0_221: G = B0, +14 ZH/+10 POI vs A2). Exposed development only, NOT fresh
validation. Report `docs/inference_cf/P2_PATH4_REPORT.md`. STOP; no EOS-rule/threshold change without a new frozen contract. Core v6 unchanged.

<!-- superseded header (P2-PATH3 result) -->
## (Previous header) Current separate inference-time ticket — P2-PATH3 (2026-10-07): terminal `P2_PATH3_SIGNAL_CONCENTRATED`

Freeze `5853a8d`; `PASS_TO_P2_PATH3` (`71a51a1`); manifest `cd93f3d`; Slurm 58015 (215 s, two resident instances); seal `bceb341` pushed before
references; `P2_PATH3_AUDIT: PASS` (primary + full). PATH2 replay barrier 12/12 exact, then the remaining 88. Unchanged A4+G1 on fixed100:
safety PASS, aggregate rescue PASS (R_ZH 20 >= 12), POI retention 1.0 (I_A4 = I_G1 = 18), useful gain PASS; **NOVEL76 breadth FAIL**:
0 triggers and 0 rescue, because native LID = zh on all 76 NOVEL76 and 12 other DEV24 rows -> frozen exact A4 no-op (A4 = B0 = G1;
AUTO = B0 there). All fixed100 rescue = U0023_S0_664 (CSD0012). Exposed development only, NOT fresh validation. Report
`docs/inference_cf/P2_PATH3_REPORT.md`. STOP; any further G1 test needs a separately frozen contract on rows where A4 activates plus a
human data-role decision. Core v6 unchanged.

<!-- superseded header (P2-PATH2 result) -->
## (Previous header) Current separate inference-time ticket — P2-PATH2 (2026-10-07): terminal `P2_PATH2_BRANCH_CONTROL_SUFFICIENT`

Freeze `e1c6ee7`; `PASS_TO_P2_PATH2` (`155aa40`); manifest `e525ae7`; Slurm 58013 (54 s, two resident instances); seal `e1198aa` before
references; `P2_PATH2_AUDIT: PASS` (primary + full). Online first-disagreement consensus guard: 5 triggers (historical C sites), 7 exact
no-op U rows, PATH1 scores/winners reproduced. G1 (one consensus token, then A4) on all 12: ZH 21 / POI 101 / mixed 123 (A4 41/101/143,
B0 18/119/137) -> passes 29/105/143/0; G2 identical to G1 (no material advantage). Mandarin rescue rests on one utterance (U0023_S0_664).
Report `docs/inference_cf/P2_PATH2_REPORT.md`. STOP; next: separately frozen broader development test of G1. Core v6 unchanged.

<!-- superseded header (P2-PATH1 result) -->
## (Previous header) Current separate inference-time ticket — P2-PATH1 (2026-10-07): terminal `P2_PATH1_MIXED`

Freeze `5a159f2`; `PASS_TO_P2_PATH1` (`cea9410`); manifest `30a6dfb`; Slurm 58011 (43 s); seal `2bf92bc` before references;
`P2_PATH1_AUDIT: PASS` (primary + full; one mechanical auditor crash fixed with test, no output change). Clean 2x2 state x branch
factorial (fresh state-owned caches): T0A (AUTO token then theta0) passes inherited INDUCE_1 (I 0.354), A4A 0.771, Delta_state +0.417
(U0102/U2004 amplified, U0029 boundary). CONSENSUS_REJECTS_AUTO PASS (B0 chosen 7/7 U, 6/6 strict). ASR_STATE_ALIGNMENT MIXED (ZH +6
theta0 vs +10 A4; full POI gain already from the token). Report `docs/inference_cf/P2_PATH1_REPORT.md`. STOP; next class: separately
frozen guarded-switch diagnostic (branch adjudication + temporary A4 rollback). Core v6 unchanged.

<!-- superseded header (P2-PATH0 result) -->
## (Previous header) Current separate inference-time ticket — P2-PATH0 (2026-10-07): terminal `P2_PATH0_ASYMMETRIC`

Freeze `fc3602d`; `PASS_TO_P2_PATH0` (`ce9ff03`); manifest `a15a8c8`; Slurm 58004 (45 s); output seal `ca4a92e` before references;
`P2_PATH0_AUDIT: PASS` (primary + full). A4 reconstructed exactly 12/12, SELF==FREE 12/12. INDUCE_1/3 pass (pooled 0.77/0.875: forcing
AUTO's first token sends the suffix into the AUTO basin), RESCUE_1/3 fail (2 successes in one dialogue). ASR_HARM_ALIGNMENT INDUCTION_ONLY
(induced rows reproduce AUTO's errors: ZH +10, POI -15); rescue removes most ZH damage (27->7/4) at a POI cost. Report
`docs/inference_cf/P2_PATH0_REPORT.md`. STOP; next class: separately frozen branch-aware path-control diagnostic. Core v6 unchanged.

<!-- superseded header (P2-TTA-A4 result) -->
## (Previous header) Current separate inference-time ticket — P2-TTA-A4 (2026-10-07): terminal `P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED`

Freeze `37441eb`; implementation `9488a69`; `PASS_TO_P2_TTA_A4` (`d32676b`); manifest `b349f12`; Slurm 57937 (24 rows, 53 s);
seal `e506e74`; `P2_TTA_A4_AUDIT: PASS`. Script-preserving safe teacher (q_AUTO at V_E positions, q_FORCED elsewhere) on the A3
24-panel: English transfer passes (11/11, median R_E 0.465), teacher-forced anchor preserved (rho 0.152, D_M2 0.003), but free-decode
Mandarin safety not rescued (ZH-CER +0.030, ZH retention 0.963, outside harm 0.037; <=19% of A3 gaps closed). POI/MER equal to A2.
Bottleneck: sequence/free-decoding path selection (AUTO Latin rendering of Mandarin on D rows), not teacher-forced LN interference.
Report `docs/inference_cf/P2_TTA_A4_REPORT.md`. STOP: no tuning, no A4-on-100, no full300/P3. Core v6 METHOD_CONTRACT unchanged.

<!-- superseded header (P2-TTA-A3 result) -->
## (Previous header) Current separate inference-time ticket — P2-TTA-A3 (2026-10-07): terminal `P2_TTA_A3_TEACHER_UNSAFE`

Freeze `bd41bc5`; implementation `eab6c64`; `PASS_TO_P2_TTA_A3` (`90dbe96`); manifest `d802297`; Slurm 57876 (24 rows, 57 s);
seal `0d29d35`; `P2_TTA_A3_AUDIT: PASS`. Enriched reference-free 24-panel (12 AUTO!=FORCED + 12 controls). Soft AUTO forward KL
on the inherited A2 actuator closed the gap (D_cond down 12/12, median 51%), moved D transcripts toward AUTO only minimally
(4 closer / 1 farther; summed distance 149 -> 132), but failed frozen safety vs forced (ZH-CER +0.032, ZH retention 0.962,
outside harm 0.038); POI/PIER equal to A2. Bottleneck: AUTO is not a Mandarin-safe teacher. Report
`docs/inference_cf/P2_TTA_A3_REPORT.md`. STOP: no tuning, no A3-on-100, no full300/P3. Core v6 METHOD_CONTRACT unchanged.

<!-- superseded header (P2-TTA-FUNNEL result) -->
## (Previous header) Current separate inference-time ticket — P2-TTA-FUNNEL (2026-10-07): TTA0-R -> A2 selected -> TTA1 `P2_TTA1_SUPPORTED`

Funnel freeze `ff75e1a`. TTA0-R (CPU, sealed run1, gradient tolerance 2e-2 only): `PASS_TO_P2_TTA0_R_EVALUATION`,
`P2_TTA0_R_AUDIT: PASS`, A1 `TTA0_EM_CONFIRMATION_BIAS` (1 severe truncation), A2 `TTA0_AC_VIABLE` -> selected; MAP skipped.
TTA1 (Slurm 57871, fixed100, A2, 20 reused + 80 new episodes): `PASS_TO_P2_TTA1`, `P2_TTA1_AUDIT: PASS`. vs forced-ZH
PIER .4957->.4540 (net POI +29), MER .2570->.2537, all safety bounds pass -> SUPPORTED; vs AUTO PIER still +.046 worse
(MER -.005). 14/100 outputs changed; 2 utterances carry 25/29 net POI. Reports `docs/inference_cf/P2_TTA_FUNNEL_REPORT.md`,
`P2_TTA0_R_REPORT.md`, `P2_TTA1_REPORT.md`. STOP: no full300/P3/transfer/fresh validation. Core v6 METHOD_CONTRACT unchanged.

<!-- superseded header (P2-TTA0 result) -->
## (Previous header) Current separate inference-time ticket — P2-TTA0 (2026-10-07): terminal `P2_TTA0_INVALID`

Freeze `eb0da2a`; implementation + pseudo seal `503c874`; `PASS_TO_P2_TTA0` (`6d408d2`); manifest pushed (`6a954f5`);
Slurm 57868 (20/20 x A1/A2, 68 s); `P2_TTA0_AUDIT: PASS` (INVALID confirmed, `2b57f8a`). Frozen first-row live A1
gradient check failed (rel 1.02e-2 > 1e-3; loss agrees 7e-7; A2 bitwise). CPU diagnostic: formulas bitwise equal in
float64, float32/bf16-backward precision ~1e-2 for both objectives -> tolerance unattainable, not a code bug. Sealed
adapted outputs NOT evaluated. No objective selected, no TTA1 handoff. New human decision required. Report
`docs/inference_cf/P2_TTA0_REPORT.md`. No TTA1/full100/300/P3/fresh validation. Core v6 METHOD_CONTRACT unchanged.

<!-- superseded header (P2-SEQ result) -->
# Current separate inference-time ticket — P2-TTA0 (2026-10-07)

Pre-outcome objective viability design frozen: `P2_TTA0_SPEC.md`, `P2_TTA0_CODEX_DESIGN.md`,
`configs/inference_cf/p2_tta0.json`, `P2_TTA0_PANEL20.json`. First saved ID/dialogue from fixed100;
20 episodic utterances, decoder-LN-only,2 AdamW steps, A1 entropy/A2 AUTO consistency, no steering.
P2-SEQ damage and prior local-selectivity terminals remain closed. Implementation/preaudit pending;
no adaptation outcome/job in design session. One allocation only after pushed PASS_TO_P2_TTA0;
no100/300/P3/fresh validation. Core v6 method unchanged.

## (Previous header) Current separate inference-time ticket — P2-SEQ (2026-10-07): terminal `P2_SEQ_SEQUENCE_DAMAGE`

Freeze `660a619`; implementation `ed9c355`; AUTO reuse proof `14a2339`; `PASS_TO_P2_SEQ` (`9cb1d75`); Slurm 57867
(100/100); `P2_SEQ_AUDIT: PASS`. Matched forced-ZH S0 PIER .4957 / MER .2570; AUTO .4080 / .2591; STEER (E*R_B*D2, L16,
alpha2) .4871 / .2852. Benefit passes (net POI +6, PIER gain .0086, CI spans 0) but safety fails (MER +.028, ZH-CER +.033,
ZH retention .940, outside harm .060). STEER is clearly worse than AUTO on PIER (+.079, lower95 +.0125). Evidence-only
handoff `docs/inference_cf/P2_SEQ_TTA_HANDOFF.md` (no TTA designed/run). Report `docs/inference_cf/P2_SEQ_REPORT.md`.
No300/P3/TTA/fresh validation. Core v6 METHOD_CONTRACT unchanged.

<!-- superseded header (P2-SEQ design freeze) -->
## (Previous header) Current separate inference-time ticket — P2-SEQ (2026-10-07)

Pre-outcome design freeze in `docs/inference_cf/P2_SEQ_SPEC.md`, `P2_SEQ_CODEX_DESIGN.md` and
`configs/inference_cf/p2_seq.json`. One fixed100 sequence screen, exactly matched forced-ZH/AUTO/
E*R_B*D2 at L16 alpha2. E/T/XA/LAC terminal labels preserved. Runner pending; no scientific job
in this design session. Next: additive implementation, focused tests, independent PASS_TO_P2_SEQ
and pushed resolved reuse/manifest before one GPU allocation. No300/P3/TTA/fresh validation.
Core v6 METHOD_CONTRACT unchanged.

# STATUS

**Newest bounded inference-time stage (2026-10-06): P2-SEL-LAC design frozen; implementation/pre-LAC0 audit pending.** Accept terminal E ambiguous, T token-LID-not-discriminative and XA INVALID. Contract: `docs/inference_cf/P2_SEL_LAC_SPEC.md`, `P2_SEL_LAC_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel_lac.json`. Unmasked candidate seal + exact T-window waveform-zero counterfactual, whole masked encoder/prefix replay; only conditional tanh(S_lex/2) precision factor. No candidate extraction/masked outcome/job in design; core method unchanged, P3 HELD.

**Newest bounded inference-time stage (2026-10-06): P2-SEL-XA design frozen; implementation/pre-XA0 audit pending.** Accept terminal P2-SEL-E ambiguous and P2-SEL-T token-LID-not-discriminative (audited PASS). New contract: `docs/inference_cf/P2_SEL_XA_SPEC.md`, `P2_SEL_XA_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel_xa.json`. Raw D2-gradient/source dot diagnostic, fixed0.95 scratch validation, only conditional clip(S,0,1) gate factor. No XA outcomes/jobs in design; core method unchanged, P3 HELD.

**Newest bounded inference-time development stage (2026-10-06): P2-SEL-T design frozen, implementation/pre-run audit pending.** Parent P2-SEL-E remains terminal `P2_SEL_E_DIAGNOSIS_AMBIGUOUS` (audited PASS). See `docs/inference_cf/P2_SEL_T_SPEC.md`, `P2_SEL_T_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel_t.json`: fixed L/C/R query-attention diagnostic; only conditional E_tok repair; no outcomes in design session. P3 HELD; no core-method revision or fresh validation.

**Separate inference-time steering proposal — current state (2026-09-26): DIAGNOSTIC PROGRAM ENDED — P2-RJ-E `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED` (terminal); **P3 HELD**; next = one independent Codex/Astra actuator-design pass.** P2-RJ-E (spec `docs/inference_cf/P2_RJ_E_LEVERAGE_EXPANSION_SPEC.md`, job 55002, `P2_RJ_E_AUDIT: PASS`, report `docs/inference_cf/P2_RJ_E_REPORT.md`) re-measured the frozen P2-RJ leverage statistic on all 227 eligible EN-confusion positions of D-dev-select (60 original, bitwise-reproduced, + 167 new; 17 dialogues — no more exist): Q50 = 0.539 [0.406, 0.665] (fp32 0.519) → leverage UNRESOLVED; median gap 10.56 nats, median first-order optimum A 4.90 nats (ρ 0.52; 12% fully closable). H1 confirmed on the full population: κ(+d) = −0.0098 [−0.019, −0.002], |cos| at random level; −d (+1%) not validated. No P2-RJ-F / further diagnostics authorized. Handoff for the redesign: `docs/inference_cf/P2_RJ_E_ACTUATOR_REDESIGN_HANDOFF.md` (direction-first replacement for norm(hE−hB) at L16 with a predeclared site-failure criterion; evaluator gradient forbidden at inference; no fresh validation role authorized). No replacement actuator may be implemented before that independent design pass.

*(Previous state, 2026-09-26 — P2-RJ; unchanged history:)* **Separate inference-time steering proposal — current state (2026-09-26): P2-RJ COMPLETE — `P2_RJ_STILL_AMBIGUOUS`; **P3 HELD** (not started; needs a human decision).** P2-RJ (spec `docs/inference_cf/P2_RJ_JACOBIAN_DIAGNOSIS_SPEC.md` v1+v1.1, job 54998, `P2_RJ_AUDIT: PASS`, report `docs/inference_cf/P2_RJ_REPORT.md`): evaluator-only reference-margin Jacobian at the 180 frozen P2-R states (L16, e*=1.126; gradient never applied as steering). First-order model validated (predicted vs observed P2-R arm Δm r=0.92, slope 1.03; fp32 gradient cos 0.9998). **Alignment MISALIGNED:** d=norm(hE−hB) captures κ₊=−1.1% [−2.2, −0.1] of the available first-order leverage (|cos| 0.023 = random 0.022), so H1 (direction) holds under every leverage class. **Leverage UNRESOLVED:** an optimally aligned e*-edit would move the margin by median 5.1 nats vs a 10.6-nat gap (ρ median 0.58; Q50=0.61 [0.44, 0.76] straddles the frozen 0.5; only 13% fully closable) — H4 co-occurrence open. The useful direction is largely a script-readout direction (cos with ∇log P_E 0.80) that d also misses (−0.006). No repair frozen (ambiguous branch); no validation role. Recommended single follow-up (needs authorization): predeclared EN-confusion sample expansion of the identical P2-RJ protocol within D-dev-select to resolve Q50 only. Start-of-stage cleanup: an accidental uncommitted 63-line truncation of the canonical plan was restored to HEAD and preserved as `docs/inference_cf/patches/plan_uncommitted_truncation_2026-09-25.patch`.

*(Previous state, 2026-09-25 — P2-R; unchanged history:)* **Separate inference-time steering proposal — current state (2026-09-25): P2-R COMPLETE — `P2_R_MECHANISM_STILL_AMBIGUOUS`; **P3 HELD** (not started; needs a human decision).** P2-R (spec `docs/inference_cf/P2_R_MECHANISM_DIAGNOSIS_SPEC.md`, job 54874, `P2_R_AUDIT: PASS`, report `docs/inference_cf/P2_R_REPORT.md`): at L16 and the P2 energy every tested direction is ~inert at the lexical decision (Δlog p(ref) ≤ 0.1 nat vs a 10.6-nat gap; 0/60 corrections); within that regime +d is the worst direction (below −d and a matched random direction); oracle relocation gives no material rescue (D2-B). No repair hypothesis frozen; recommended single follow-up: evaluator-only reference-margin Jacobian at the same positions (H1 vs H4). P2 unchanged: `P2_VALID_CONFIGURATION_SELECTED`, `P2_AUDIT: PASS` (L16, α=2.0, φ_id, g=E·R_B, +d; ΔPIER +2/2268; B0_AUTO 8.2 PIER points better). Earlier this day: Pre-P2 `CACHED_EQUIVALENCE: PASS` (job 54802); P2-A attempt 1 (jobs 54803/54804) invalid (baseline mismatch); spec v1.1 P2-A r1 54821/54822, P2-B 54824, P2-C 54825; P2-R attempts run1 54855 (abort) and run2 54871 (1 failed row) preserved, unanalyzed.
Main method, frozen from P1 onward:
- **Detector:** `g = E·R_B`. `E` = LS-B null-corrected local support on the 1.0 s max-attention window; `R_B` = BC-B baseline matrix-script conflict.
- **Direction:** `d = norm(hE − hB)`.
- **Edit:** NormPreserve at the decoder post-cross-attention / pre-FFN site.

The R2 counterfactual gate `g_cf = E·[R_B − R_Ecf]_+` was rejected (not preferred) and is kept only as an ablation.

- **R2** (job 54758, manifest `e8869f0c…`, commit `fe960e4`): `R2_CF_FEASIBLE_NOT_PREFERRED`, `POST_R2_AUDIT: PASS`.
  - ALIGN 0.962; LOCALIZER 0.890 (EN-confusion only 0.464).
  - LS AUROC 0.820 (mismatched audio 0.503).
  - g_cf 0.710/0.726 vs g=E·R_B 0.820/0.836 (ΔAUROC −0.11, CI<0).
  - See `docs/inference_cf/P0_R2_REPORT.md`, `P0_R2_POST_RUN_AUDIT.md`.
- **P1** (job 54781, spec v1.1): `P1_CAUSAL_ACCEPTANCE_PASS`, `POST_P1_AUDIT: PASS`.
  - Attempt 1 (job 54770) was blocked by an A6 instrument defect and is preserved.
  - Verified: bitwise α=0 identity; 20 real L24 edits; isolation.
  - See `docs/inference_cf/P1_REPORT.md`.
- **Baselines:** B0 forced-ZH and B0_AUTO (stronger: PIER 0.390 vs 0.474) remain mandatory.
- **Claims:** no ASR-improvement claim yet. The post-R2 runner patch is preserved unapplied (`docs/inference_cf/patches/`).

*History (pre-run R2 design, kept for provenance):* `P0_R2_REPAIRABILITY_SPEC.md` froze the max-50-frame window, LS-B, BC-B, a same-prefix forced-English conflict, and `g_cf` as the R2 primary candidate, with `E·R_B` as its ablation. R2 then selected `E·R_B`.

**Separate inference-time counterfactual P0 (2026-09-24): `P0_BLOCKED_EVIDENCE`.**
Worktree `feature/inference-cf-steering` completed the pre-registered 60-row D-dev-select
diagnostic. Frozen spec/implementation commit `55faf76`; parser-only correction `cba7784`;
Slurm jobs 54702 (preserved invalid first attempt) and 54703 (corrected K=1 run). G1 is
`DEGENERATE_BY_CONSTRUCTION` (`c0=cM`, 57 zero state residuals); G2 is `PASS` on 57 retained rows;
K=1 G3 was `WEAK/BLOCKED` (50/57 candidate collisions, AUROC unestimable). An independent audit
(`P0_INDEPENDENT_AUDIT.md`) reproduced every number and confirmed the block is a genuine K=1
argmax-collision property, not a defect. The one authorized repair (K=1→K=3, spec
`docs/inference_cf/P0_R1_EVIDENCE_SPEC.md`, report `docs/inference_cf/P0_R1_REPORT.md`, job 54712,
manifest `e4514a8…` superseding `0287abb…`) lowered collision to 73.7% and doubled usable rows to
15 but **still fails frozen G3** (collision >50%; Mandarin usable 3; AUROC unestimable) →
`P0_BLOCKED_EVIDENCE`. Thresholds not weakened; no further K/scorer variant attempted. The
matrix-collapse need factor is non-identifiable and removed; `d=normalize(hE−hM)` has no validated
gate. See `FEASIBILITY.md`, `results/inference_cf/{p0_retry,p0_r1}/`. Separate proposal; does not
alter the frozen DG-03R/DG-08 method. (Historical as of P0: P1 had not started then.)

## Stage roll-up (2026-09-08)

- **DG-00 — COMPLETE.** Scientific contract & guardrails.
- **DG-01 — COMPLETE.** Canonical metrics (`metrics_v1`) + result schema (`result_v1`) + adapter.
- **DG-02 — COMPLETE / FROZEN.** Exact post-cross-attention/pre-FFN site; real-model acceptance PASS
  at L16 & L24 (Slurm job 50369). Committed `db365d8`.
- **DG-03R — COMPLETE / FROZEN.** The reconciled documents define **contrastive basis → adaptive
  controller → damage-aware optimization** and the older disagreement / temporal-localizer /
  utility-selector / factorized-gate roadmap is **superseded for the core paper** (`LEGACY DESIGN`).
  **Prior blocker (now resolved):** the updated method existed only in the contract docs / the DG-03R
  ticket, with no versioned proposal artifact, while `AGENTS.md`/`CLAUDE.md` still pointed to v5.
  **Resolution:** added the versioned proposal
  `docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v6.md` as the scientific source of
  record; repointed `AGENTS.md`, `CLAUDE.md`, and `METHOD_CONTRACT.md` to v6; marked v5 (and
  Implementation_Plan/ROUND_1_3/GATE_A) **SUPERSEDED / HISTORICAL**; and confirmed no active
  experiment gate requires the localizer / utility selector / disagreement / factorized gate. No
  code/tests/configs changed; no experiment run. Not yet `COMPLETE` — awaiting the independent final
  audit. See v6, `METHOD_CONTRACT.md` §4–§8/§10, `EXPERIMENT_MATRIX.md`, `CODE_MAP.md`.

- **DG-03 — COMPLETE / FROZEN.** Canonical v6 steering basis built & versioned at L16 and L24 on
  Whisper-large-v3 × CS-Dialogue (`steering_basis_v1`, `results/dg03/basis/`; construction Slurm jobs
  50452 then **50470** with complete provenance). Free-decoding causal screen on D-dev-select (final
  jobs **50471/50472**) selected **L24** by the pre-registered rule (`DG03_BASIS_CAUSAL_SPEC.md` §8,
  §11): intended `v_local` at oracle embedded steps gives U=+33 (61 corrections vs 28 corruptions,
  PIER gain +0.0146, matrix-CER 0.2287 vs baseline 0.2261), beating sign-reversed (−43),
  matched-random (−7), and count-matched wrong-location (+10); L16 showed no useful headroom (U=−6).
  Dose fixed pre-run at ρ=1×s_ℓ (no tuning). **Audit blockers resolved** (repair commits
  `5465e04`/`12f0cac`; final artifact freeze `212a06f`): basis
  carries a real `dataset_fingerprint`/`construction_config_hash`; every screen condition is a
  complete validated `result_v1` (metrics+gains+POI transitions+3 retention populations+reserved
  gate_coverage) with per-utterance texts; per-condition edit-count/total-energy recorded and C4
  count-matched to C1. Canonical `outside_harm = n_corrupted_outside` is now populated in every
  C0–C4 result from the stored hypotheses and frozen D-dev-select candidate spans by CPU-only
  post-processing; it is correctness-flip harm, not outside transcript edits. No GPU rerun was
  needed. The added candidate utility is diagnostic only; the frozen selection U remains POI
  corrections − corruptions. Pre-run commit `749759b`; GPU jobs 50470–50472 completed; focused
  DG-03/DG-01 validation: **34 passed**.

- **DG-04 — COMPLETE / FROZEN.** Frozen exact-site baselines B0/B1/B2/B3 ran on the same 300-utterance
  D-dev-select population at L24 with the pre-registered `ρ={0.5,1.0,2.0}` grid. Jobs **50483** and
  **50484** completed on `mig`; all 10 frontier points validate as `result_v1` with canonical outside
  harm and realized-energy accounting. B3 calibration used 300 shortest `router-calib` utterances
  (τ=1.2037518, T=1.9582404) and no runtime reference information. The frozen POI rule selects
  **B1 ρ=0.5** (`U=38`, 135 corrections / 97 corruptions, PIER gain +0.01675); B3 ρ=0.5 is the
  other positive point (`U=33`). No adaptive/controller/training work was run. Freeze artifacts:
  `results/dg04/frontier.json`, `results/dg04/reference.json`; runner commit `33799da`, audit
  hardening commit `a4f14be`; final freeze commit `2ed6cf1`; focused DG-04 tests: **7 passed**.

- **DG-05 — COMPLETE / FROZEN.** DG-05B trained the fixed L24 controller once with
  seed 42 in job **50507** (`mig`, H100 3g/40GB) after three failed execution attempts
  (50489/50505/50506). Training used only `loc-train ∪ util-train`, correction-only CE on a
  validated 51,227-position `C_E`, and no retention/anchor/gate losses. All three checkpoints
  were free-decoded on the same 300-utterance D-dev-select population as DG-04; epoch 3 was
  selected by the frozen utility → PIER gain → energy rule. A2 epoch 3 has utility +91 (135
  corrections / 44 corruptions), PIER gain +0.04012, embedded WER gain +0.03219, matrix CER
  gain −0.02745, outside harm 1,142, embedded retention 0.9634, matrix retention 0.9040, and
  total energy 63,926.3 (18,812 realized edits; mean energy 3.398). It improves the DG-04 B1 rho=0.5 point (utility +38; 135/97;
  PIER gain +0.01675; energy 82,007.1; matrix retention 0.8930) while showing a present damage
  signal that motivates DG-06. Classification: **VALID ADAPTIVE SIGNAL WITH DAMAGE**. The
  accepted run executed at source commit `6f9c7506d2c39b0723070537c9403f5be9cd935a`, an
  ancestor of audit HEAD `bd533039733971be1860fd75d967cfd26cbd09e4`. Complete
  artifacts/provenance are in `results/dg05/controller/`; selected checkpoint hash is
  `sha256:6d9390dee19097110bbe3f9ed6707ee72e83e93722cf448bd075c8c3f40dfbcc`.

- **DG-06 — COMPLETE / FROZEN.** Damage-aware training added on
  top of the frozen DG-05 controller (`src/csasr/steering/dg06_losses.py`,
  `experiments/dg06_damage_aware.py`; spec `DG06_DAMAGE_AWARE_SPEC.md`). Two `mig` GPU jobs from
  pre-run commit `16f4578` trained the fixed L24 controller from the **identical** reconstructed
  DG-05/D0 init (`sha256:25ebe8e4…156f`, D1==D2), seed 42, β fixed, `λ_M=λ_E=1.0`, C_E reused
  verbatim from DG-05, retention populations `R_E` (11,748 pos) / `R_M` (151,556 pos) built from the
  reused frozen baseline. Checkpoints selected by free decoding on the same 300-utt D-dev-select
  population; both selected epoch 1.
  - **D0** (reused DG-05 epoch 3): net 91 (135 corr / 44 corrupt), PIER gain +0.04012, **MER gain
    −0.01919**, **matrix-CER gain −0.02745**, outside harm 1,142, embed ret 0.9634, matrix ret
    0.9040, energy 63,926 (gate mean 0.823).
  - **D1** correction+matrix retention (job **50558**): net **85** (116/31), PIER gain +0.0375,
    **MER gain +0.0254**, **matrix-CER gain +0.0239**, outside harm **316**, embed ret 0.9742,
    matrix ret **0.9745**, energy 35,575 (gate mean 0.446).
  - **D2** full damage-aware (job **50559**): net 68 (83/15), PIER gain +0.0300, MER gain +0.0250,
    matrix-CER gain +0.0243, outside harm **295**, embed ret **0.9875**, matrix ret 0.9756, energy
    35,920 (gate mean 0.451).
  Both retention variants **flip MER and matrix-CER from degrading (D0) to improving**, cut outside
  harm ~72–74% and corruptions, and roughly halve gate strength/energy. D2's embedded-retention
  objective further raises embedded retention (0.9742→0.9875) and lowers corruption/outside harm but
  sacrifices corrections (116→83), lowering net utility below D1. Frozen selection (highest utility)
  ⇒ **`SELECT D1`** under the predeclared utility → matrix-retention → embedded-retention → energy
  rule. D2 adds measurable embedded and matrix retention benefit while retaining positive utility;
  outcome **`FULL DAMAGE-AWARE SUCCESS`**. Gate penalty and basis refinement remain deferred
  (DG-06 core forbids them). Focused tests: **27 passed** (15 DG-06 + 12 DG-05, CPU), plus the
  exact-site/site-infrastructure checks passed in the audit environment. No D-dev-confirm / D-test
  read; no post-hoc λ/β/LR tuning.

- **DG-07A — COMPLETE / FROZEN PRE-RUN.** The required comparison matrix is frozen in
  `docs/current/DG07_BASELINES_ABLATIONS_SPEC.md`: LB1 exact-site SALSA global, LB2 matched-budget
  Q/V LoRA, A1 local-only, A2 conditioning-only, and A3 fixed-mixture gate. All new methods share
  the selected DG-06 D1 objective (`L_corr(C_E) + λ_M L_ret,M`, λM=1), seed 42, three-epoch budget,
  and greedy `D-dev-select` checkpoint selection. A5 refined basis is deferred because no λA was
  predeclared. Implementations, CPU focused tests, and `mig`-only launch batches are committed;
  no DG-07 GPU job has been submitted.

- **DG-07B — COMPLETE / FROZEN.** The frozen five-run matrix completed on `mig` with
  one development seed and greedy D-dev-select decoding. Accepted jobs are **50592** (LB1),
  **50604** (LB2 exact-config rerun after a mechanical N/A-energy tie-break fix), **50612** (A1),
  **50613** (A2), and **50642** (A3); failed job **50593** was the pre-fix LB2 software attempt.
  All runs used `loc-train ∪ util-train`, seed 42, 2,019 optimizer updates, and the frozen DG-06
  D1 objective. A3 produced valid per-epoch `result_v1` records but no selected checkpoint because
  all three utilities were non-positive under the predeclared positive-utility rule. The compact
  canonical summary is `results/dg07/summary_v1.json`; attempts and full per-method artifacts are
  in `results/dg07/`. Independent audit: required matrix, exact-site/baseline fairness, canonical
  metrics, efficiency accounting, provenance, Slurm partition, and data-role separation **PASS**.
  Focused DG-07 tests: **17 passed**. No D-dev-confirm or D-test data was read.

- **DG-08 — COMPLETE / FROZEN.** Locked Whisper core evaluation on
  Whisper-large-v3 × CS-Dialogue only. Protocol frozen + committed (`f1dc4a8`); hard D-test lock
  committed BEFORE any D-test decode (`1fb2c3`, `results/dg08/DG08_TEST_LOCK.json`). Finalists F0
  frozen Whisper, F1 DG-04 frozen steering (B1 ρ=0.5), F2 SALSA global (1,280 params), F3 matched
  LoRA (rank 9, 46,080), F4 **M\*** = DG-06 D1 adaptive controller (43,651). Learned finalists trained
  at **seeds [13,42,73]** — seed 42 reused hash-verified from DG-06/DG-07; seeds 13/73 newly trained
  on `mig` (SALSA `50704`/`50705`, LoRA `50713`/`50714`, M* `50718`/`50719`, all COMPLETED, only the
  seed varied via a backward-compatible `--seed`). D-test (6,257 utts / 15 dialogues) decoded greedy
  (jobs `50725`/`50733`/`50734`) and beam-5 (`50760`/`50767`/`50768`/`51100`/`51101`/`51102`), all
  `mig`, never `main`; 2000-rep dialogue-block bootstrap (seed 42) offline.
  - **Greedy (mean/3 seeds):** MER — F0 0.361, F1 0.407, SALSA **0.299** (best), LoRA 1.289, M* 0.347;
    PIER — F0 0.831, SALSA 0.518, LoRA 0.145, M* 0.790; matrix-ret — M* **0.977** (best among
    interventions; F0=1.000 is the trivial no-op).
  - **Beam-5:** M* has the **lowest MER (0.320) of any system** (marginally below SALSA 0.321) and the
    best matrix retention (0.988); M* en-WER 0.689 is second to F1's 0.677, and M* is not the lowest-
    en-WER system in either regime (SALSA 0.609 lowest greedy). LoRA MER catastrophe worsens (1.91,
    en-WER 8.2).
  - **Bootstrap:** M* significantly beats frozen Whisper on PIER, MER, and utility in **both**
    regimes (all 95% CIs exclude zero); SALSA beats M* on PIER/utility (MER tied under beam); M*
    significantly beats LoRA on MER. **Beam robustness: PERSISTS** (M* edge over F0 grows under beam).
  - **Efficiency:** M* 43,651 params (0.0028% backbone), train 46.2±1.9 min, peak 5.51 GB, inference
    ≈ parity with F0; LoRA inference 2.63× overhead; SALSA smallest (1,280 params).
  - **Verdict:** RQ1 adaptive gives a better correction–**damage** trade-off than fixed steering;
    RQ2 M* **COMPETITIVE** (best matrix retention among interventions + lowest beam-5 MER + inference
    parity; does not dominate SALSA on raw correction or en-WER; LoRA non-viable). Outside-harm not computable on D-test (no candidate/POI
    alignments) — documented, text metrics complete. No SOTA claim; no post-test tuning. Focused
    tests: 13 DG-08 (+ DG-06/07 regression) pass. Full report `results/dg08/DG08_RESULTS_SUMMARY.md`.
    **Do not begin SEAME/ViMedCSS/Qwen cross-dataset/model expansion.**
  - **Independent audit & freeze (2026-09-09, CPU-only).** All blocking gates PASS: chronology
    protocol(`f1dc4a8`)→lock(`1fb2c37`, checkpoints + lock JSON, no D-test decode)→result(`e94339d`,
    D-test decodes) verified; D-test manifest fingerprint recomputed == lock
    (`sha256:dfbf53bc…e1a42`); all 9 selected-checkpoint SHAs recomputed == lock; seed-42 finalists
    hash-match DG-06/DG-07; basis hashes == frozen DG-03 L24 (`local 2459a6…`, `cond 319951b…`);
    three seeds [13,42,73] complete for F2/F3/F4 (none dropped/replaced); F0–F4 identities correct;
    M* keeps `v_cond` (not redesigned after DG-07); training config identical across seeds (seed is
    sole variation); all mean/std recomputed == tables; utility == corr−corrupt; bootstrap 2000
    reps / dialogue unit / seed 42 / 9 arms / CIs reproduced; param counts verified analytically
    (SALSA 1,280 / LoRA 46,080 / M* 43,651); LoRA over-generation confirmed (insertions ~26× F0
    across all 3 seeds); 13 focused DG-08 tests pass; post-lock commits are throughput/mechanical/doc
    only (batch-size flag, stats path, basis-ablation workstream) — no scientific config changed.
    **Audit corrections (CPU-only, wording):** removed an overclaim — M* is NOT lowest en-WER in
    either regime (SALSA 0.609 greedy, F1 0.677 beam) and is lowest MER only under beam-5; corrected
    in `DG08_RESULTS_SUMMARY.md` and this file. Non-blocking notes: M* training manifests record
    `seed=42` (the shared D0-init reconstruction seed); the effective training seed 13/73 is in each
    `summary.json` top-level `seed` and is confirmed by divergent checkpoints/metrics — not edited to
    avoid mutating locked artifacts. `table_C_efficiency.json` F4 `checkpoint_size_bytes=null` while
    the summary shows 189 KB (cosmetic). Freeze artifacts: lock `results/dg08/DG08_TEST_LOCK.json`;
    greedy `results/dg08/dtest/greedy/`; beam-5 `results/dg08/dtest/beam5/`; bootstrap
    `results/dg08/stats/bootstrap_{greedy,beam5}.json`; efficiency `results/dg08/tables/table_C_efficiency.json`.
    Verdict: **PASS — DG-08 locked Whisper core evaluation COMPLETE / FROZEN.**

**Current ticket (historical):** `DG-08 — locked Whisper core evaluation` — implementation,
evaluation, and independent audit/freeze complete (2026-09-09). DG-00–DG-08 method selection is
closed; proceed only to supporting generalization/reporting work.

**BASIS-A — frozen direction-construction ablation COMPLETE.** Additive D-dev-select-only
comparison at exact L24 post-cross-attention/pre-FFN used one completed `main` job (`51107`) for
new R/C/RC conditions; DG-04 B0/B1/B2 were reused after strict compatibility checks. The required
batch-32 preflight changed Local transcripts, so the correctness-preserving batch-1 path was
frozen before the successful rerun. PCA and projection distributions were deferred because no
compatible cached representation states were available. Results and geometry are under
`results/basis_ablation_frozen/`; no D-dev-confirm or D-test output was read or produced. This
addendum does not reopen DG-07 or DG-08 and does not begin learned/adaptive basis construction.

---

**Stage 2 / DG-01 — COMPLETE (canonical metrics + result artifacts; freeze commit pending).**
`metrics_v1` + `result_v1` are frozen. Canonical modules: `src/csasr/evaluation/canonical.py`,
`retention.py`, `result_schema.py`, `legacy_adapter.py`, and the emission path `result_emit.py`.
Regression validated against the committed `results/job_a/summary.json` and
`results/job_b/summary.json` (read-only; MER/PIER reproduced exactly, gains re-signed to
`baseline − method`, legacy blended retention and outside-edits preserved as descriptive-only).
All 31 focused/regression tests pass (`tests/test_canonical_metrics.py`, `test_retention.py`,
`test_result_schema.py`, `test_legacy_adapter.py`, `test_dg01_regression.py`). Gate-coverage
denominator stays a **non-blocking deferred** item (spec §4 / MC §6; resolved with the gate,
DG-03+). **The DG-01 deliverables are currently uncommitted (untracked) in the working tree**; the
working tree must be committed before any hash is treated as the frozen DG-01 state — no freeze
commit is fabricated here (`HEAD` is still the pre-DG-01 code baseline `0975244…`). Next ticket:
**DG-02 — exact post-cross-attention, pre-FFN intervention hook.**

**Stage 1 / DG-00 — COMPLETE (documents finalized; freeze commit pending).** Scientific contract
finalized 2026-09-07 against repository **code baseline** `0975244bfb9563cdb123c4572eca697259ed399a`
(current `HEAD`). That baseline is the code state the contract audited; it does **not** contain the
Stage-1 deliverables. The deliverables themselves — `AGENTS.md`, `CLAUDE.md`, and `docs/current/*` —
are **currently uncommitted (untracked) in the working tree**. The freeze is therefore not yet
captured in an immutable commit; committing these files is a required mechanical step before the
contract can be relied on as locked. No fabricated freeze commit is recorded here. Authority:
`METHOD_CONTRACT.md` (MC). See also `CODE_MAP.md`, `EXPERIMENT_MATRIX.md`, and `DATA_EXPOSURE.md`.

This stage changed documentation only. Existing runnable results are not evidence for the
finalized method unless CODE_MAP classifies the relevant path as canonical.

## Confirmed decisions

1. Intervention site = decoder post-cross-attention residual, pre-FFN
   (`csasr.lss.sites.DECODER_TENSOR`); whole decoder-block output is rejected.
2. No depth rescale at the frozen site; norm-preserving repair is required.
3. `q`, `u_source`, pre-intervention `r`, repaired `r̃`, and effective `u^S = r̃-r` are defined in
   MC §1–§2 and §7; zero-gain positions must be bit-identical and `β=0` is a no-op.
4. Decoder candidate layers are `{16, 24}`.
5. The frozen basis is `V^0 = [v_local, v_cond]`; legacy directions (including codebase `v_nat`)
   remain labeled legacy and must not be silently relabeled.
6. The inference-safe feature allowlist is canonical; the factorized gate is superseded `LEGACY
   DESIGN`, not a current contract component.
7. Headline evidence requires free decoding; teacher-forced results are screening only.
8. `D-test` has permitted reference/C00 exposure but no steered exposure found.
9. Encoder–decoder disagreement is **optional supporting analysis** only (MC §5), not a decision
   rule or fallback gate.
10. Correction, English retention, Mandarin retention, and monolingual retention populations are
    distinguished (MC §8).
11. Metric sign convention: `PIER_gain = PIER_baseline − PIER_method` (positive = improvement).
    Legacy artifacts with opposite signs must be converted before comparison.

## Implementation gaps

> **DG-03R note:** this list predates the DG-03R reconciliation and is retained as historical
> context. **G1 is resolved** (exact-site hook FROZEN, DG-02). **G5 is superseded** — the temporal
> localizer / utility selector / abstention / factorized gate are `LEGACY DESIGN`, replaced by the
> adaptive controller. The authoritative post-DG-02 gaps are `METHOD_CONTRACT.md` §13 and
> `CODE_MAP.md` §2 (basis builder, controller, damage-aware losses, scientific exact-site runner).

- **G1 — exact-site integration:** the reusable exact-site recorder/hook exists, but no current
  free-decoding runner wires it through cache position, forced prefix, and beam expansion. Current
  Job-A/Job-B and Round-1 execution paths intervene post-FFN.
- **G2 — layer sets:** `configs/lss/spec.yaml` lists `[8,16,24,31]`; Round-1 frozen executes seven
  layers; the contract permits decoder layers `{16,24}`.
- **G3 — scaling:** contract scale is construction-time projection standard deviation. Job A/B
  use unit scale; sweep code defaults to activation-norm scale; one legacy hook also divides by
  `sqrt(num_layers)`.
- **G4 — directions:** exact-site v2r3 assembly residualizes each contrast before aggregation,
  whereas MC specifies aggregate raw direction then project it. The dedicated class named by
  `configs/lss/spec.yaml` (`DirectionAccumulator`) does not exist; `LayerAccumulator` does.
- **G5 — gate:** no temporal localizer, outcome-supervised utility selector, abstention component,
  or factorized gate is implemented. Job-A F5 and Job-B T1 are monolithic projection gates.
- **G6 — training:** the current wrapper delegates the fixed-layer legacy trainer. Its T1 is
  rank-2, its T2 spans layers 24/25, and its CE covers every non-prefix token rather than only the
  correction set. Retention is measured, not represented by an explicit loss.
- **G7 — metrics:** implementations use different candidate populations/accounting; three
  outside-edit fields have three non-harm meanings; and delta signs conflict:
  `steer_sweep.metrics.delta_pier` is positive-better while Round-1 `delta_PIER`/`delta_MER` are
  negative-better.
- **G8 — calibration split:** dialogue-v2 `router-calib` has no frozen
  `calib-prob`/`calib-thresh` subdivision.
- **G9 — provenance:** current Round-1 result directories do not contain the complete resolved
  config/environment/git/model/hash manifest required by `AGENTS.md`.

## Historical open scientific decisions (pre-DG-03R; retained for provenance only)

> **DG-03R note:** superseded by `METHOD_CONTRACT.md` §12 (reconciled). Items about the
> disagreement gate, factorized-gate algebra, and conditioning-subspace rank are no longer open for
> the core paper; the live open decisions are layer selection (DG-03 screen), controller
> architecture (DG-05), and training weights/schedule (DG-06).

1. Practical deployable site: E-only, decoder, or E+D (MC §1).
2. Conditioning subspace rank/estimator (MC §4).
3. Trained direction rank: 1 vs 2 (MC §4).
4. Feature-to-score aggregation (MC §5).
5. Exact factorized-gate algebra and transport composition (MC §6).
6. Whether an explicit retention loss and its weight (MC §8).
7. Staged training order: correction-only-first vs joint (MC §8).
8. Dialogue-v2 calibration subdivision (MC §6; DATA_EXPOSURE).

None of these decisions blocks metric canonicalization.

## Artifact state

- `results/job_a/summary.json` and `results/job_b/summary.json` are completed legacy pilots.
- `results/round1/job_a/summary.json` is complete at 84/84 cells, but remains a seven-layer,
  post-FFN current-wrapper result.
- `results/round1/job_b/training_status.json` records successful completion of the delegated legacy
  trainer; `results/round1/job_b/summary.json` contains that trainer's result summary.
- The dialogue-v2 locked test artifact exists as `locked/role_D-test.parquet` with its sidecar.
- `D-test` remains underpowered (MDE approximately 0.0608 at 15 dialogue blocks); this is a
  scientific limitation rather than an implementation blocker.

## Current ticket

**DG-01 — canonical metric and result-schema implementation: COMPLETE.** The spec is frozen at
`docs/current/DG01_METRICS_RESULTS_SPEC.md` (§13 implementation status; §14 regression + emission
completion). Metric schema `metrics_v1`, result schema `result_v1`. The regression-validation pass
ran: the read-only adapter round-trips the committed Job-A and Job-B summaries with MER/PIER
reproduced exactly and deltas re-signed to `baseline − method`; a fixed `adapt_job_b_summary` now
reads the flat `results_table` block (it previously dropped every Job-B method system by reading a
nonexistent `systems` key); and a current emission path (`src/csasr/evaluation/result_emit.py`)
emits `result_v1` with a real reused-infrastructure manifest (git HEAD, `resolved_config_hash`,
dataset fingerprint, schema versions; unavailable provenance stays `null`). No second canonical
metric/schema/gain-sign definition exists (Phase-D scan). Regression fixtures checked:
`results/job_a/summary.json`, `results/job_b/summary.json` (both byte-untouched). Non-blocking
carry-forward: gate-coverage denominator deferred to the gate ticket (DG-03+); the *scientific*
free-decoding runner that would call `result_emit` is DG-02+ (G1/G5) — the emitter exists and is
tested but no exact-site free-decoding runner exists yet to feed it. **Deliverables uncommitted:**
the DG-01 files are untracked; commit the working tree before treating any hash as the frozen DG-01
state (no freeze commit fabricated).

Historical DG-01 build record (retained): the canonical modules exist and pass CPU-only unit tests:
- `src/csasr/evaluation/canonical.py` (`SCHEMA_VERSION = "metrics_v1"`) — reuse facade for MER,
  PIER, EN-WER, ZH-CER, `*_gain` (baseline − method, positive = improvement), POI
  correction/corruption, candidate outside harm; asserts the PIER transition identity.
- `src/csasr/evaluation/retention.py` — three distinct MC §8 retention populations.
- `src/csasr/evaluation/result_schema.py` (`RESULT_SCHEMA_VERSION = "result_v1"`) — deterministic
  JSON, `gate_coverage` reserved/null with a guard.
- `src/csasr/evaluation/legacy_adapter.py` — read-only re-signing adapter (`legacy-derived`,
  `production_artifact=False`).
- Tests: `tests/test_canonical_metrics.py`, `tests/test_retention.py`,
  `tests/test_result_schema.py`, `tests/test_legacy_adapter.py`.

No existing Round-1 number may be merged or treated as comparable until it passes through the
adapter. The regression-validation pass has now run (spec §14); DG-01 is **COMPLETE**. The legacy
execution scripts are intentionally **not** rewired to emit `result_v1` (that is DG-02+ scientific-
runner work, not a DG-01 blocker); the canonical emission helper `result_emit.py` is the current
`result_v1` producer and is validated.

Scoped, non-blocking deferral inside DG-01: **gate coverage denominator** is not fixed by Stage 1
and the gate does not yet exist (MC §6). The canonical schema reserves the `gate_coverage` field
but does not compute it in DG-01; its denominator resolves with the gate (DG-03+). This is recorded
as a blocker on that one metric only, not on the metric/schema/manifest work that proceeds now.

**Reconcile in the implementation pass:** `src/csasr/evaluation/{pier,mer,correction_harm}.py`,
`steer_sweep/metrics.py`, `steer_sweep/round1_metrics.py`, and `src/csasr/lss/outcomes.py` to one
versioned denominator, transition convention, sign convention, and artifact schema, via a reuse
facade (`src/csasr/evaluation/canonical.py`) — not by reimplementing scoring.

## Current ticket (Stage 3)

**DG-02 — exact post-cross-attention/pre-FFN intervention site: COMPLETE / FROZEN.** Implementation
+ CPU/synthetic validation green, and the real-model acceptance passed at **both L16 and L24** on
whisper-large-v3 (Slurm job **50369**, `mig` H100 3g.40gb, COMPLETED exit 0; artifact
`results/dg02_real_acceptance.json`, verdict PASS). Spec at
`docs/current/DG02_INTERVENTION_SITE_SPEC.md` (§17 delivered implementation, §18 acceptance
evidence). Implementation commit `4e10399`.

Real-model acceptance summary (utterance `ZH-CN_U0091_S0_68`, `D-dev-select`, 2.525 s, bf16):
β=0 token+transcript identity (steered_calls=0), `r=q+u` at bf16 rounding scale (L16 ≤3.1e-3, L24
≤7.8e-3, both ≪ tol), exact-site `err_vs_block_gap ≈ 0.008 ≪ 1` (site not block output),
forced-prefix zero-edit + eligible edit, cache positions `[0..8]` monotonic/no-reset/boundary-aligned,
norm preservation rel dev ≤ 4.6e-4. No layer/direction/β selection (DG-03).

Delivered (implementation pass, working tree, uncommitted):
- `src/csasr/lss/sites.py` — new canonical `DecoderPostCrossAttnInterventionHook` (exact site
  `r=q+u_source`; FFN consumes repaired `r̃`; layer `cache_position` pre-hook; dynamic
  `num_forced_prefix`; per-row `gate_fn`/`gain`; `steer`/`train` modes; detached `AuditRecord`
  emitter; `CONTRACT_DECODER_LAYERS=(16,24)` guard). Recorder now exposes `q`/`u_source`/`r`
  separately. Legacy `DecoderPostCrossAttnSteeringHook`, `apply_steering`, and F5/T1 paths
  untouched. `apply_steering` (`models/hooks.py`) reused verbatim — no `sqrt(num_layers)` rescale.
- `tests/test_dg02_site.py` — spec §13 matrix, **19 tests**; with `tests/test_lss_sites.py` (11)
  → **30 passed** CPU. DG-01 suite (31) re-run **unchanged**.
- `experiments/dg02_real_acceptance.py` + `sbatch/cs_asr_dg02_real_acceptance.sh` — the §14 rung-1
  one-utterance real-Whisper acceptance (β=0 identity, `r=q+u`, exact site, prefix/cache/norm at
  L16 & L24). **Must run on a GPU/compute node to close the real-model gates before freeze.**

Freeze evidence: the real-model acceptance was executed on a GPU node (Slurm job 50369, `mig`
partition) after the CPU dev box proved too small to load the model; it reported `PASS` at both
layers, so DG-02 is frozen. No DG-02 bug surfaced on the real model — no code changed between the
CPU-green state and acceptance.

Confirmed architecture facts (installed source + repository, verified this pass):

Confirmed architecture facts (installed source + repository, verified this pass):
- Model is **Whisper-large-v3, 32 decoder layers**; `bundle.decoder_layer(k)` →
  `model.model.decoder.layers[k]`, **0-indexed and direct** — scientific **L16→index 16, L24→index
  24**, both interior, no off-by-one.
- `transformers` `WhisperDecoderLayer.forward` (installed): the **site is line 528**
  `residual + encoder_attn(LN(residual))` (post-cross-attn, pre-FFN); block output (line 537,
  post-FFN) is the **rejected** site. Dropouts (511/527) are identity under `eval()`.
- Operational capture: `q` = `encoder_attn_layer_norm` pre-hook `args[0]` (post-self-attn state);
  `u_source` = `encoder_attn` output[0] (source cross-attn contribution); `r = q + u_source` = the
  site (exact to dtype ULP; validated by `assert_site_reconstruction`). No LayerNorm sits on the
  `q→r` residual path.
- `sites.py` (`DecoderPostCrossAttnRecorder` / `DecoderPostCrossAttnSteeringHook`) is the
  **CANONICAL/REUSABLE** base; it omits the `sqrt(num_layers)` rescale and reuses
  `apply_steering` norm preservation (preserves per-token L2 norm of the site `r`, `EPS=1e-6`).
- Cache position available via layer `cache_position` kwarg and `cache_length(past_key_values)`
  (`steer_sweep/hooks.py:46`); forced prefix = `build_prefix` length (4 here), to be read
  dynamically not hard-coded; per-beam gate is row-local (batch = `batch*num_beams`).

The additive infra `IMPLEMENTATION GAP`s are now **implemented** in `sites.py`: q/u_source/r exposed
separately; decoder-layer pre-hook for `cache_position`; dynamic `num_forced_prefix`; per-row
`gate_fn` + trainable direction/gain + `steer`/`train` mode + detached `AuditRecord`. Still deferred
by design (row-local interface provided, transport later): G-e source-item→beam gate transport;
G-f first-content-token/final-prefill steering choice (kept at the legacy "skip all prefill"
default via position-based exclusion). DG-02 is **not** marked complete until the real-model
acceptance passes. Layer selection (L16 vs L24) and directions are DG-03.

## Next tickets

Current post-DG-06 ticket: **DG-07 — learned baselines and core ablations.** The
historical roll-up below is retained for provenance.

DG-02 is frozen (real-model acceptance PASS, job 50369). Next ticket:

**DG-03 — steering basis construction and causal validation: COMPLETE / FROZEN.** Basis
construction and both free-decoding causal screens completed; the independent-audit blockers
(basis provenance, complete `result_v1`+retention, C4 edit-count/energy) are resolved (commit
`eac2e37`) and re-run (construction 50470; screens 50471/50472). **SELECT L24.** Controller and
training work remain separate DG-04+ tickets.

**BASIS-A2 — frozen all-layer steering response atlas: COMPLETE (exploratory/additive).** The
32-layer, five-direction, four-dose D-dev-select micro-panel atlas, teacher-forced token/
representation diagnostics, fixed lambda response, dialogue-disjoint linear probe, geometry,
projection summaries, and figures are recorded under `results/basis_frozen_layer_atlas/`. This
does not reopen DG-00…DG-08, does not validate a final layer, and does not authorize D-dev-confirm,
D-test, or learned/adaptive basis work.

## Current separate inference-time ticket — P2-DIR (2026-10-06)

Independent local-first design complete; spec/config/handoff frozen before outcomes.
`docs/inference_cf/P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md` defines OLD/UNIQUE/READOUT at
L16 and exact P2 energy, conditional gate-coupled/free-decoding screens, safety/selection/audit
rules. Implementation pending; no GPU jobs launched. Diagnostic program remains ended and
P3 held. Historical A5 recovered but not reused unchanged; D1 is a new dialogue-cross-fitted
definition. Next: Claude implementation with required tests and pre-run independent audit.

## P2-DIR result (2026-10-06) — terminal: `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`

Implemented (`1802441`), pre-run audit `PASS_TO_P2_DIR_EXP1` (`00dd8b2`), extraction (Slurm 57789) and
Exp-1 (Slurm 57792) executed once each; Exp-1 `P2_DIR_AUDIT: PASS`; final session audit PASS. At L16 DG-02,
matched e*, ungated single pulses on the frozen 180 states: D1 UNIQUE +0.227 nat [0.079, 0.421] (safe,
below 0.5 materiality); D2 READOUT +4.50 [3.83, 5.17], kappa about 0.80, but ZH-correct margin −2.01
[−2.47, −1.58] and pooled corruption 0.077 [0.013, 0.163] -> supplementary
`P2_DIR_CAUSAL_POWER_WITH_DAMAGE:D2`. No selection; Exp-2/Exp-3 NOT run (stop rule). P3 HELD.
Reports: `docs/inference_cf/P2_DIR_EXP1_REPORT.md`, `P2_DIR_FINAL_REPORT.md`. Next: human decision on
any new separately pre-registered stage; nothing authorized.

## P2-SEL result (2026-10-06) — terminal: `P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE`

Contract `b4602b8`; implementation `6c7644a`; pre-run `PASS_TO_P2_SEL_S1`; S1 Slurm 57805 (143 s, no autograd);
`P2_SEL_AUDIT: PASS (S1)` (attempt-1 BLOCK from an auditor tolerance defect preserved; mechanical fix `a9606da`).
D2-GATED (E·R_B, alpha 2, L16): EN-confusion +3.52 [1.91, 4.96] (78% of ungated), ZH-correct −0.195
[−0.54, 0.00] (10% of ungated harm), but ZH corruption 0.054 [0, 0.158] > 0.05 and C2−C3 confusion
lower −1.30 < −0.25. S2 NOT run. R_B does not separate EN-confusion from ZH-correct (≈1 in both); E false
positives on 5/60 ZH positions carry all residual harm. P3 HELD. Report `docs/inference_cf/P2_SEL_REPORT.md`.
Next: human decision on any new separately frozen stage; nothing authorized.

## P2-SEL-E original freeze (2026-10-06) — BLOCKED before E0 (`BLOCK_BEFORE_P2_SEL_E_E0`)

IMPLEMENTATION GAP: the frozen contract (`11f4e63`) assumes `native_lid` is a two-token softmax (Q_local = 1) and
invalidates E0 for Q outside [0.999999, 1.000001]. The frozen provider is a 100-language softmax; already-exposed
P0-R2 pair-mass values have median 0.957 and min 0.0008, and 100% fall outside the interval. E0 would therefore
be INVALID by construction, and H_E3's ineligibility rests on a false premise. No job, no outcome. Needs a revised
freeze. Details: `docs/inference_cf/P2_SEL_E_PRE_RUN_AUDIT.md`. P3 HELD.

### P2-SEL-E Q/H_E3 contract repair (pre-outcome)

The original block narrative and `docs/inference_cf/P2_SEL_E_PRE_RUN_AUDIT.md` are retained as provenance. The audit
blocked before E0; **no E0 outcome existed**. The pre-outcome contract revision corrects Q to the absolute EN/ZH
pair mass from the actual 100-language softmax, replaces the invalid Q≈1 integrity condition with frozen provider
map/code identity plus current-E reproduction, and activates H_E3 with a fixed global P0-R2 q10 cutoff and exact
`R3: E_new=E*Q`. D2, R_B, site, alpha, localizer, groups, other hypotheses/repairs, E1/E2 rules, panel, firewall,
and compute bounds remain unchanged. No runner or scientific job is authorized by this status update. Run a new
pre-E0 audit against the revised freeze before any E0 job; preserve the prior BLOCK artifact.

## P2-SEL-E result (2026-10-06) — terminal: `P2_SEL_E_DIAGNOSIS_AMBIGUOUS`

Revised freeze `b693b8d`; implementation `0703523`; pre-E0 r1 `PASS_TO_P2_SEL_E_E0`; E0 Slurm 57809 (no steering);
`P2_SEL_E_AUDIT: PASS (E0)`. Groups 42/18/55/5 reproduced, E exact vs P2-R. H_E1 (scale / oracle-FN / short-FN),
H_E2 (spikes), H_E3 (low Q) and H_E4 (null) all fail their frozen criteria, so no repair and no E1/E2. The 5 ZH FPs
show strong persistent English LID evidence (ell_local 4.75–7.44, Q ~0.9), so they are not artifacts. P3 HELD.
Reports `docs/inference_cf/P2_SEL_E_{E0_REPORT,REPORT}.md`. Next: human decision; nothing authorized.

## P2-SEL-T result (2026-10-06) — terminal: `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE`

Freeze `fdace84`; implementation `d36d7f6`; pre-T0 PASS (`50004bd`); T0 Slurm 57838 (no steering, 358 L/R LID calls,
E0 CENTER reused); `P2_SEL_T_AUDIT: PASS (T0)`. Attention-selected 0.5 s E_tok: FP suppression 1/5, TP evidence 36/42,
E_tok TP−FP +0.010 (lower −0.167); contrast and recall branches also fail. No R_TOK, T1/T2 not run. P3 HELD.
Reports `docs/inference_cf/P2_SEL_T_{T0_REPORT,REPORT}.md`. Next: human decision (e.g. phonetic/lexical compatibility).

## P2-SEL-XA result (2026-10-06) — terminal: `P2_SEL_XA_INVALID`

Freeze `595e174`; implementation `83845ee`; pre-XA0 PASS (`e6cf957`); XA0 Slurm 57844 (sealed raw gradients reused;
0 backward/LID/steering); `P2_SEL_XA_AUDIT: PASS (XA0)` (attempt-1 auditor-tolerance BLOCK preserved, fixed in `02d9128`).
The frozen λ=0.95 finite-difference material subset holds only 4 rows / 2 dialogues (needs 30 / 10): S_src = g_J·u_source is
tiny (|cos| ~0.02, max |S| 0.49), so INVALID. No gate; XA1/XA2 not run. P3 HELD. Reports `docs/inference_cf/P2_SEL_XA_{XA0_REPORT,REPORT}.md`.

## P2-SEL-LAC result (2026-10-07) — terminal: `P2_SEL_LAC_NOT_DISCRIMINATIVE`

Freeze `be460ce`; implementation `4e6062c`; candidate seal pushed before masking; pre-LAC0 PASS (`92b81c6`); LAC0 Slurm
57866 (forward-only, 180 counterfactuals); `P2_SEL_LAC_AUDIT: PASS (LAC0)`. EN-TP strong 6/42 (< 34), ZH-FP weak 4/5 (pass),
S TP−FP +0.95 with lower80 −0.022; recall 4/18. No F_lex; LAC1/LAC2 not run. Occlusion is strong on correct states
(ZH −4.0, EN +2.3) but ≈0 on EN-confusion TP and FP. P3 HELD. Reports `docs/inference_cf/P2_SEL_LAC_{LAC0_REPORT,REPORT}.md`.

## P2-SEQ result (2026-10-07) — terminal: `P2_SEQ_SEQUENCE_DAMAGE`

Freeze `660a619`; implementation `ed9c355`; AUTO reuse 100/100 sealed (`14a2339`, attempt-1 hash-definition seal
preserved); `PASS_TO_P2_SEQ` (`9cb1d75`); Slurm 57867, one MIG allocation, 607 s; `P2_SEQ_AUDIT: PASS`. 237 edits across
69/100 utterances; 41 transcripts changed. Damage is mostly deletion/early termination (deletions 1026→1299). Benefit
rules pass, but MER, ZH-CER, matrix-ZH retention and outside-POI harm fail. STEER is below AUTO. Next: separately frozen
P2-TTA0 design from the evidence-only handoff. P3 HELD.

## P2-TTA0 result (2026-10-07) — terminal: `P2_TTA0_INVALID`

One allocation (Slurm 57868) completed all 20 x {A1 GREEDY-EM, A2 AUTO-CONSISTENCY} episodes with exact resets and
theta0 = P2-SEQ S0 20/20, but the frozen first-row independent gradient check failed for A1 (1.02e-2 vs 1e-3). The CPU
diagnostic shows this is intrinsic float32-logit/bf16-backward precision (~1e-2, A2 too), not an implementation error.
Reference-free mechanics only: both losses fell 20/20; A1 entropy -53%, 10/20 transcripts changed, 1 severe truncation;
A2 changed 3/20 (all where AUTO = forced) and never moved toward AUTO. No reference evaluation. P3 HELD.


## P2-TTA-FUNNEL master freeze

Next separate inference_cf development ticket: P2-TTA-FUNNEL, design frozen only. TTA0-R must pass repaired sealed-run audit before reference evaluation; MAP only after audited no-viable; TTA1 only after exactly one audited selection. No outcomes or jobs in this design session. See ../inference_cf/P2_TTA_FUNNEL_SPEC.md. Historical P2_TTA0_INVALID and closed steering diagnostics preserved.

## P2-TTA-FUNNEL result (2026-10-07)

Path: TTA0-R `P2_TTA0_R_OBJECTIVE_SELECTED` (A2) -> MAP not run -> TTA1 `P2_TTA1_SUPPORTED` (vs matched forced-ZH; below AUTO
on PIER). One new GPU job (57871). Preserved engineering attempts: R gate/post-audit attempt 1 (test-file source-scope bug).
Exposed-development evidence only; new human decision required for any confirmation.

## P2-TTA-A3 result (2026-10-07) — terminal: `P2_TTA_A3_TEACHER_UNSAFE`

Gap closed (12/12, median 0.514), movement passed at the minimum (criterion a; R_dist 0.114), safety failed (ZH-CER,
matrix-ZH retention, outside harm). A3 = A2 on POI (128) and PIER on the enriched panel; A2 also breaches the same bounds there.
One GPU job (57876). Exposed development only; new human decision required.

## P2-TTA-A4 result (2026-10-07) — terminal: `P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED`

EN transfer and anchor preservation pass; full and partial matrix rescue fail; benefit vs A2 retained (equal POI/MER). One GPU job (57937).
Exposed development only; new human decision required.


## P2-PATH0 pre-outcome freeze

Next authorized separate development design: P2-PATH0, first-divergence rescue/induction causal diagnostic, spec/config/sites frozen only.12 D rows; unchanged A4 reconstruction must reproduce12/12 before clamps. PASS_TO_P2_PATH0 required before later execution. No scientific outcomes in this session; core v6 and historical terminal stages unchanged.

## P2-PATH0 result (2026-10-07) — terminal: `P2_PATH0_ASYMMETRIC`

Induction (single and three-token) is causal for the AUTO basin under fixed A4 weights; rescue is not established (one dialogue).
One GPU job (58004). Exposed development only; new human decision required.


## P2-PATH1 pre-outcome freeze

Next separate exposed-development diagnostic: P2-PATH1 pre-outcome spec/config/fingerprinted sites frozen only. New theta0-A counterfactual separates prefix from persistent A4 state; exact parent criterion and reproduction required. Fixed3-content-token consensus score is secondary only. PASS_TO_P2_PATH1 before later run; no scientific outcomes here. Core v6 unchanged.

## P2-PATH1 result (2026-10-07) — terminal: `P2_PATH1_MIXED`

AUTO-token history alone establishes the basin on this cohort; persistent A4 state amplifies it in some rows (required only for U2004).
Short-horizon theta0/A4 consensus rejects the AUTO branch at all U sites. One GPU job (58011). Exposed development only.



## P2-PATH2 pre-outcome freeze

Next separate exposed-development ticket: P2-PATH2 pre-outcome spec/config/fingerprinted panel frozen only. Online one-event guard, G1 versus G2; audited reconstruction/PATH1-score/no-trigger identity required. PASS_TO_P2_PATH2 before later job; no scientific job/outcome here. Core v6 and historical terminal results unchanged. P3 HELD.

## P2-PATH2 result (2026-10-07) — terminal: `P2_PATH2_BRANCH_CONTROL_SUFFICIENT`

One consensus-selected token at the first theta0/A4 disagreement keeps A4's beneficial switches and rejects the harmful one on this cohort;
G2 three-token handoff adds nothing. One GPU job (58013). Exposed development only; broader frozen test required.



## P2-PATH3 pre-outcome freeze

Next separate exposed-development ticket: P2-PATH3 design only. Frozen PATH2 G1 fixed100 breadth confirmation, 12-row exact replay barrier then88; authoritative TTA1 safety and NOVEL76 breadth gates. PASS_TO_P2_PATH3 before later run. No outcome/job here; historical stages/core v6 unchanged. NOVEL76 is NOT fresh validation. P3 HELD.

## P2-PATH4 pre-run design — BLOCKED

After audited PATH3 SIGNAL_CONCENTRATED, intended G1 transfer to A2 is documented/fingerprinted only. Four sealed first disagreements are EOS-first on theta0: U0027_S0_116, U0086_S0_222, U0091_S0_196, U1004_S0_221. Exact frozen G1 refuses empty content candidates; therefore PASS_TO_P2_PATH4 cannot be issued. No runner/outcome/job. Requires separately authorized contract-boundary decision, not a silent fallback or panel change. See ../inference_cf/P2_PATH4_SPEC.md. P3 HELD.

## P2-PATH4-R1 — authorized EOS-boundary extension

Original P2_PATH4_BLOCKED_EOS_FIRST_BRANCH (29661e9) preserved. Human authorized PATH4_R1_EOS_BOUNDARY_V1 before any scientific run/PATH4 reference outcome. Content/content G1 unchanged; first EOS/content disagreement uses equal-horizon1 next-action .5/.5 consensus, inherited tie and one-action A2 execution. Fixed100/A2/thresholds/firewall unchanged. Additive eos_boundary helper, CPU tests and independent freeze auditor; no runner/scientific outcomes/GPU. See docs/inference_cf/P2_PATH4_EOS_AMENDMENT.md. PASS_TO_P2_PATH4_R1 required before later execution; runnable manifest/code audit remains mandatory. P3 HELD.

## P2-OPP0-ABSTAIN + P2-PATH5 pre-outcome freeze (2026-10-08)

OPP0 derived diagnostic (no GPU): A2+G1A fixed100 ZH 1093 / POI 314 / mixed 1439; abstention removes the PATH4 EOS harm; rescue still only
PATH2_12. PATH5 contract frozen (spec/design/panel/config) before any NEW200 decode or reference: frozen A2+G1A on NEW200 (200 rows, 20
dialogues), opportunity gate, 0.90 retention, hard stop rule. Implementation/run pending `PASS_TO_P2_PATH5`.
