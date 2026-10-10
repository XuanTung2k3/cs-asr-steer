# P2-PATH5 — Frozen A2 + G1A (abstaining consensus guard) on the broader D-dev-select population

**Status: PRE-OUTCOME FREEZE.** Starting local = remote HEAD `2491255`, clean tree, empty queue. Normative artifacts:
this spec, `P2_PATH5_CLAUDE_DESIGN.md`, `P2_PATH5_PANEL.json`, `configs/inference_cf/p2_path5.json`. No PATH5 decode,
adaptation, or NEW200 reference evaluation happened before this freeze was committed and pushed. Core v6 is unchanged.

## Why this stage

PATH4-R1 (`P2_PATH4_OVERCONSERVATIVE`) showed two things. A2 creates broader first-disagreement opportunities (14 on
fixed100, 7 outside DEV24). EOS-boundary H=1 arbitration is unsafe: on U1004_S0_221 a theta0 EOS discarded A2's
continuation. The only authorized robustness revision is the domain restriction below. **OPP0-ABSTAIN**
(`P2_OPP0_ABSTAIN_REPORT.md`) is a no-GPU diagnostic derived from the sealed PATH4 rows. It shows that abstention removes
the EOS harm on fixed100 (A2+G1A ZH 1093 / POI 314 / mixed 1439 vs A2 1116/316/1464), but that all positive rescue there
is still in the two PATH2_12 rows. PATH5 tests the frozen method once on rows never used in the A2/G1 line, then takes a
hard method decision.

## Candidate method (frozen, no tuning)

**A2** is the exact historical TTA1 objective:
- Teacher: the sealed AUTO greedy y_A; forced-ZH student.
- Trainables: 194 decoder-LN affine tensors (fp32 masters, bf16 forward).
- Optimizer: AdamW, lr 1e-3, wd 0, exactly 2 updates.
- Episodic reset; ordinary cached forced-ZH decode.
- Code path: unchanged `inference_cf_p2tta0.run_objective`.

**G1A — abstaining consensus guard.** Two resident model instances (theta0, A2). One logical detector
(`consensus_guard.lockstep_detect`) runs at the first processed-argmax disagreement on the live common prefix.
Dispatch is on the two live proposals' action types (`eos_boundary.action_mode`):

| First disagreement | Action |
|---|---|
| content/content | exact original G1 (`inference_cf_p2path3.online_g1`, A2 in its legacy A4 slot): H=3 live b0/b2 rollouts, four fresh-cache score paths, S_cons = 0.5·S_theta0 + 0.5·S_A2, tie → theta0 if diff ≥ −1e-12; emit one winning token under A2, then ordinary A2 |
| theta0 EOS / A2 content | **ABSTAIN**: ordinary A2 output; no EOS scoring, no H=1 arbitration, no new decoder path |
| theta0 content / A2 EOS | **ABSTAIN**: ordinary A2 EOS termination |
| both EOS / no disagreement | NO_TRIGGER: ordinary A2 |

At most one decision event per utterance; after an abstention the controller is disabled. Eligibility depends only on
live token/action type. There are no learned thresholds, no row IDs and no reference input.

**Rationale.** The H=3 branch scorer has a valid comparison domain only when both alternatives have nonempty content
continuations. EOS lies outside that domain, so unsupported cases preserve the adapted actor A2.

## Population (proved from IDs/dialogues/fingerprints; no references read)

FULL300 is the canonical P0-R2 / P2-A r1 D-dev-select panel, `results/inference_cf/p2_A_r1_L16/panel.json` (20
dialogues × 15). It is the byte-pinned `parent_panel` of the historical fixed100 (`P2_SEL_MINI_PANEL.json`). The role
manifest (ID/dialogue/role columns only) confirms 300 D-dev-select rows.

| Partition | Rows | Dialogues | Membership hash |
|---|---:|---:|---|
| FULL300 | 300 | 20 | `bc806239…` |
| FIXED100 | 100 | 20 | `15c0c82e…` (same IDs/order as PATH3/PATH4 FULL100 `3560be21…`) |
| **NEW200 = FULL300 − FIXED100** | **200** | 20 (× 10) | `e23a8ce5…` |

Execution order is FIXED100 (fixed100 order), then NEW200 (FULL300 order); hash `4b748d93…`.

**Exposure.** FULL300 is the same 300 already-exposed DG-04 B0 D-dev-select utterances. Their references were used in
DG-04/06/07 selection, P0-R2, P2 compact selection and P2-R/RJ/RJ-E. NEW200 was **never used in the TTA/A2/A3/A4/PATH
chain** (no A2 episode, no G1, no PATH reference evaluation), but it is **not fresh validation**. PATH5 is a
development/breadth decision stage.

**Primary population: NEW200.** FIXED100 serves only as the reconstruction/regression barrier and as historical
context. FULL300 aggregates are secondary. No FIXED100 rescue counts toward any NEW200 gate.

## Systems and baselines

- **B0-FORCED.**
  - FIXED100: the sealed P2SEQ S0; the live theta0 decode must equal it.
  - NEW200: the live theta0 canonical cached forced decode (the same path as P2SEQ S0 and the TTA1 y_B), recorded and
    sealed before references.
  - P2-A r1 `systems.B0` differs from the canonical S0 on 19/100 fixed100 rows and is not used.
- **B0-AUTO:** sealed P2-A r1 `systems.B0_AUTO`, the same source as the TTA1 teachers.
- **A2.** FIXED100 is reconstructed (must equal sealed); NEW200 gets new episodes. The NEW200 teacher is y_A =
  `clean_teacher(B0_AUTO)`; all 200 are clean.
- **A2+G1A:** new on all 300.
- No A4, A3, G2 or EOS arbitration.

## One job (≤ 1 sbatch; target < 30 min, hard 3 h)

1. **FIXED100 barrier, before any NEW200 encoder pass.**
   - Phase 1: exact A2 reconstruction 100/100. Tokens, termination and text equal the sealed A2; 194 fp32 master arrays
     equal the archives; the bf16 effective state matches; theta0 equals the sealed B0. Independent live check on the
     first row (loss 1e−5, gradient 2%).
   - Phase 2: live A2+G1A equals the OPP0 derived target (tokens and termination) on all 100.
   - The 10 content rows reproduce the sealed PATH4 detector, decision, score paths and G1 exactly.
   - The 4 EOS rows equal A2; the 86 no-trigger rows equal A2.
   - Any failure → `P2_PATH5_INVALID` and abort.
2. **NEW200.** Per row:
   - theta0 decode (B0);
   - exact A2 episode;
   - materialize, then the G1A detector and dispatch;
   - exact reset.
   No-trigger, abstain and A2-winner outputs must equal the live ordinary A2 output.
3. **Budget:** 600 A2 updates; ≤ 601 backward passes (including one live check); controller is forward-only.

## Reference-free seal and opportunity gate

Seal and push all 300 outputs and logs before any NEW200 reference. The seal covers: IDs/partitions, B0/A2/G1A outputs,
disagreement types, candidates, scores, winners, abstentions, termination, and config/source hashes. The independent
primary audit must PASS. Then, on NEW200:

| Quantity | Definition |
|---|---|
| N_DISAGREE | rows with a first theta0/A2 disagreement |
| N_CONTENT_ELIGIBLE | rows whose first disagreement is content/content |
| N_EOS_ABSTAIN | rows whose first disagreement involves EOS |
| N_G1A_CHANGED | rows where the G1A output ≠ A2 |
| N_CHANGED_DLG | distinct dialogues among the G1A-changed rows |

Breadth later needs ≥ 3 rescue rows across ≥ 3 dialogues. So if **N_G1A_CHANGED < 3 or N_CHANGED_DLG < 3**, broad
rescue is impossible, and the terminal label is **`P2_PATH5_OPPORTUNITY_SPARSE`**. In that case NEW200 references are
never opened, the final report is written, and the controller line stops.

## Post-seal gates (NEW200, only if the opportunity gate passes)

**Absolute safety** (A2+G1A vs B0-FORCED), the authoritative TTA1 thresholds:
- ΔMER ≤ 0.01; ΔZH-CER ≤ 0.015;
- matrix-ZH retention ≥ 0.98; embedded-EN retention ≥ 0.95;
- outside-POI harm ≤ 0.03; POI corruption ≤ 0.05;
- added caps (row-wise) ≤ 1; new severe truncations = 0.

These use a 1e−12 float guard and the TTA1 None/not-assessable semantics. The NEW200 POI/ZH/EN reference populations
must be nonempty.

**Strong A2 retention** (the safety-layer standard). I_A2 = POI_B0 − POI_A2 and I_G = POI_B0 − POI_G1A.
- If I_A2 > 0, require I_G / I_A2 ≥ 0.90.
- If I_A2 ≤ 0, require POI_G1A ≤ POI_A2.
- In both cases also require PIER_G − PIER_A2 ≤ +0.005 and MER_G − MER_A2 ≤ +0.005.
- The old 0.75 rule is reported only for continuity.

**Rescue and breadth.**
- Per row: r_i = ZH_A2(i) − ZH_G1A(i), p_i = max(r_i, 0), h_i = max(−r_i, 0).
- R_plus, R_minus, R_net; P_d = sum of p_i within dialogue d.
- Rescue utterances = count(r_i ≥ 1); rescue dialogues = count(P_d > 0).
- C_utt = max p_i / R_plus; C_dlg = max P_d / R_plus; both are 1 if R_plus = 0.
- All bounds are inclusive.

## Exhaustive terminal precedence (first applicable)

1. `P2_PATH5_INVALID` — any panel/provenance, fixed100 barrier, A2 reconstruction, G1A semantics, cache ownership,
   reference leakage, evaluator, budget or audit failure.
2. `P2_PATH5_OPPORTUNITY_SPARSE` — N_G1A_CHANGED < 3 or N_CHANGED_DLG < 3 on NEW200 (references not opened).
3. `P2_PATH5_SEQUENCE_DAMAGE` — absolute safety fails (this includes any new severe truncation).
4. `P2_PATH5_SAFE_NO_ADDED_VALUE` — R_net ≤ 0.
5. `P2_PATH5_OVERCONSERVATIVE` — strong retention fails, or ΔPIER > 0.005, or ΔMER > 0.005.
6. `P2_PATH5_SIGNAL_CONCENTRATED` — rescue utterances < 3, rescue dialogues < 3, C_utt > 0.50 or C_dlg > 0.60.
7. `P2_PATH5_G1A_SUPPORTED` — every gate passes and the independent post-audit passes.

**Descriptive only:**
- NEW200 LODO over 20 dialogues.
- Paired dialogue-block bootstrap: B = 2000, seed 240924, existing P2SEQ `draws`; G1A−A2 and G1A−B0 for PIER, MER,
  ZH-CER and POI count.
- Opportunity-outcome map: NO_TRIGGER, EOS_ABSTAIN, CONTENT_A2_WIN, CONTENT_THETA0_WIN_NO_CHANGE,
  CONTENT_THETA0_WIN_CHANGED.
- FULL300 secondary aggregate.
- All rescue and harm rows.

None of these can change the label.

## Hard stop and method decision

- **`P2_PATH5_G1A_SUPPORTED`:** the main candidate is A2 + G1A. Freeze it and recommend one separately frozen
  independent confirmation. No further tuning.
- **Any other valid label:** the G1 branch-controller line is closed as a main-method direction, and the main
  adaptation candidate returns to **A2**. No EOS, H, weight, threshold, retrigger, G2, row-selection or other controller
  revision is allowed; the next question is upstream adaptation/objective work.

## Audit, firewall, GitHub

- **Audits:** pre-run `PASS_TO_P2_PATH5`; post-seal primary `P2_PATH5_AUDIT: PASS`; full `P2_PATH5_AUDIT: PASS` (only if
  the gate passes). The independent auditor never imports runner or analyzer decision functions, `consensus_guard`,
  `branch_adjudication`, `path_decode` or `eos_boundary`.
- **Allowed data:** already-exposed D-dev-select FULL300 and sealed artifacts.
- **Forbidden:** fresh validation, D-dev-confirm, D-test, a router-calib new role, a new panel or split, P3, SEAME,
  CS-FLEURS, ViMedCSS, ASCEND and transfer corpora.
- **GitHub, in order:** freeze, implementation + `PASS_TO_P2_PATH5`, manifest, job, seal, primary audit, then
  (if allowed) evaluation, full audit and report. Each step runs check → diff → commit → push to
  `origin HEAD:cs-asr-steer-inf`. No force push, no merge to main, no PR.
