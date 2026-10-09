# SRC-CF0-P — final scientific report

**Terminal verdict: `SRC_CF0_PILOT_CAUSAL_INSUFFICIENT`** (independent FULL audit: identical label, no selected
configuration). The acoustic counterfactual direction v_AC can be constructed reliably, and it is geometrically distinct
from equal-duration off-target masking. Pulsing it at the DG-02 site, however, produced **zero** actual English
first-token corrections in every one of the eight frozen arms. No arm reaches the margin-only bar either.
Expanded Mandarin-safety study: **not recommended**. SRC-CF1: **NO**. TTO-CF: **NO**.

## Experiment identity

| Item | Value |
|---|---|
| Freeze (Codex) | `2d235742d5fe59915eb8cc3ba6ad8511f0fa1c9c` (config `sha256:f1881050…51aa87`) |
| Implementation | `f0300f5`; notes `docs/inference_cf/SRC_CF0_PILOT_IMPLEMENTATION_NOTES.md` (interpretations fixed pre-execution) |
| Runtime projection | `run1/runtime.json`, runtime hash `sha256:63d62e8a…332fb8`, projection hash `sha256:70386353…7a20d6` |
| Construct manifest | `3de4605`, `sha256:0c52dc14…a396c0` |
| Pre-run audit | `f91d256` — `PASS_TO_SRC_CF0_PILOT` (first attempt) |
| P-A job | Slurm **58236** (3 min 30 s wall, 184 s in-process) |
| Direction seal | `fe25502`, `sha256:344e55b6…0413f7` (166 files, pushed before any audit / strata) |
| Construction audit | `8eeaf4d` — `SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)` 31/31 (first attempt) |
| Gate P-A + authorization | `9b35961` — PASS, qualified layers [16, 24]; authorization `sha256:b7178e52…65f31` |
| P-B launch failure (preserved) | Slurm 58238: launcher grep only, 1 s, no Python / model / forward (`run1/pulse_attempt1_LAUNCH_FAIL.json`, `manifest_pulse_attempt1_LAUNCH_FAIL.json`) |
| Pulse manifest | `b9dcb57`, `sha256:344bfb94…41618e` |
| P-B job | Slurm **58239** (2 min 14 s wall, 109 s in-process) |
| Pulse seal | `f32e7e5`, `sha256:a3249e9e…7de79` (247 files, pushed before any lexical reference) |
| PRIMARY audit | `628992e` — `SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)` 26/26 (first attempt) |
| Evaluation | `4770612` — `SRC_CF0_PILOT_CAUSAL_INSUFFICIENT` |
| FULL audit | `025af48` — `SRC_CF0_PILOT_AUDIT: PASS (FULL)`, independent label identical, independent selection none |

Model: Whisper-large-v3 (`model.safetensors sha256:a8e94b85…fd95`), BF16, eval, eager attention, torch 2.10.0+cu128,
transformers 4.57.6 (`WhisperDecoderLayer.forward sha256:78267a12…74aa6`), CUDA 12.8, mig 3g.40gb (historical GPU type).
Panel identity `sha256:2f66af05…464d371`: 180 queries / 80 utterances / 20 dialogues, 60/60/60. Site: DG-02
post-cross-attention / pre-FFN residual, layers 16 and 24 (zero-based). Outputs: `results/inference_cf/src_cf0_pilot/run1/`.

## P-A — reference-free direction construction

Integrity:
- Two independent cold sweeps over 80 originals and 124 distinct S1 masks: 408 encoders, 13,318 cached steps. Both
  counts equal the frozen bound exactly.
- The two sweeps are bitwise identical: every logit, both layers' states and the attention heads. The auditor
  re-verified this from the stored arrays.
- Clean raw logits are bitwise equal to S1 M and to ST-LOC0 NONE (180/180). Clean attention heads are bitwise equal
  to S1's.
- L16/L24 clean states are bitwise equal to the ST-LOC0 and ST-PROMPT-R1 archives (180/180).
- Masked logits are bitwise equal to S1's (132/132).
- The recorder site equals the tensor the FFN actually consumes, at every readout.
- 0 pulses, 0 autograd or optimizer calls; weights byte-identical before and after.
- Regions in the projection were re-derived from S1's sealed heads plus the R0 heard intervals, both by the provider
  and by the auditor's exact-Fraction route (180/180). All 124 masks are byte-identical to S1's.

| Per layer (identical at L16 and L24) | Count |
|---|---|
| Numerically valid targets (of 73 raw) | **73** (≥ 66 needed) |
| Valid off-target pairs (of 59 raw) | **59** (≥ 54) |
| EN-confusion valid targets | 28 in 12 dialogues (≥ 20 / 8) |
| EN-confusion joint-reachable pairs | 20 in 9 dialogues (≥ 15 / 6) |
| EN-correct joint-reachable pairs | 35 in 16 dialogues (≥ 20 / 10) |
| ZH-correct targets / joint pairs | 6 (6 dialogues) / 4 (4 dialogues) — no Mandarin minimum, safety unresolved |
| Degenerate (raw norm < 1e-4) / nonfinite | 0 / 0 |

Geometry. Each cell shows L16 / L24, as median and range where given.

| Quantity | Target v_AC | Off-target | Random |
|---|---|---|---|
| Raw ‖h_clean − h_mask‖ | 4.07 (1.06–6.12) / 5.84 (1.15–8.91) | 0.57 (0.18–2.11) / 0.61 (0.23–3.69) | — |
| Serialized ‖v‖ (historical ε) | 0.9999998 / 0.9999998 | 0.9999982 / 0.9999984 | 1.0 |
| cos(v, h_clean) | +0.30 (−0.10…+0.51) / +0.32 (−0.05…+0.53) | +0.06 / +0.03 | ≈ 0 |
| Tangent ratio ‖v⊥‖/‖v‖ | 0.954 (min 0.862) / 0.948 (min 0.845) | 0.997 / 0.999 | 0.9998 / 0.9999 |
| Raw tangent norm | 3.84 / 5.32 | 0.56 / 0.61 | — |

Target-vs-off-target distinction:
- Specificity cohort: 20 EN-confusion pairs in 9 dialogues.
- Median |cos(v_target, v_off)|: **0.122 at L16** (max 0.49), **0.109 at L24** (max 0.38). The bar is ≤ 0.90.
- Median raw tangent-norm ratio: **5.98 at L16** (range 2.65–12.7), **8.20 at L24** (4.50–14.3). The bar is ≥ 1.25.
- Over all 59 pairs: median |cos| 0.068 / 0.095, median ratio 6.41 / 8.03.
- Masking the predicted-English region moves the decoder residual about 7–10× more than masking an equal-duration
  low-attention segment of matched energy. The two directions are nearly orthogonal.

Reachability:
- All (target, random) × (±) × (η 0.15, 0.30) cells are reachable on 73/73 valid targets, and the off-target cells
  on 59/59.
- Joint prediction (tangent ≥ 0.25 and paired energy ≤ 2 %) holds for 59/59 pairs at both layers. No pair failed for
  any reason.

**Gate P-A: PASS at both layers (coverage, geometry and construction specificity); qualified layers [16, 24].** This
establishes construction distinction only, not lexical grounding.

## P-B — conditional causal pulse screen

Integrity:
- The complete 180 × 24 matrix was recorded: 8 v_AC, 8 off-target and 8 random arms. There were 1,640 actual pulse
  forwards; the other cells are explicit no-edits for ineligible rows.
- NONE was bitwise equal to the sealed P-A clean readout (180/180), and the zero dose was bitwise at L16 and L24
  (360/360).
- The L16 historical apparatus (ST-LOC0 D0 ±, sealed D2 at e* = 1.1260757575454359) reproduced old logits and consumed
  states bitwise, with solver scale within 1e-9 (540/540).
- Every steered cell passed the four frozen checks: η error ≤ 2 %, squared-energy error ≤ 2 %, solver vs hook ≤ 1e-6,
  consumed vs proposed ≤ 0.5 %. All of them were re-computed independently.
- Realized η stayed within 0.1499–0.3003 of the requested 0.15 / 0.30.
- 0 integrity failures. Paired cohort retention was 59/59 = 1.00 in every arm.

Population, identical in every arm:
- Target-active rows: EN-confusion 28, EN-correct 39, ZH-correct 6.
- Paired (joint-valid) rows: 20 / 35 / 4.
- Abstentions on the full 60-row denominators: 32 / 21 / 54.

| Arm | Paired EN-conf (dial.) | Corrections (dial.) | Net vs random | Net vs off | EN-corr. corruptions | ZH-corr. corruptions | EOS | Δmargin macro [adj. 99.375 %] | mean Δrank ref | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| L16 η.15 + | 20 (9) | 0 (0) | 0 | 0 | 0 | 0 | 0 | +0.166 [−0.008, +0.374] | −35 | fail |
| L16 η.15 − | 20 (9) | 0 (0) | 0 | 0 | 0 | 0 | 0 | −0.101 [−0.355, +0.093] | +73 | fail |
| L16 η.30 + | 20 (9) | 0 (0) | 0 | 0 | 1 | 1 | 0 | +0.352 [+0.016, +0.704] | −42 | fail |
| L16 η.30 − | 20 (9) | 0 (0) | 0 | 0 | 2 | 0 | 0 | −0.102 [−0.557, +0.268] | +231 | fail |
| L24 η.15 + | 20 (9) | 0 (0) | 0 | 0 | 1 | 1 | 0 | +0.148 [−0.066, +0.386] | −102 | fail |
| L24 η.15 − | 20 (9) | 0 (0) | 0 | 0 | 2 | 1 | 0 | −0.040 [−0.318, +0.217] | +307 | fail |
| L24 η.30 + | 20 (9) | 0 (0) | 0 | 0 | 1 | 1 | 0 | **+0.467** [+0.062, +0.909] | −118 | fail |
| L24 η.30 − | 20 (9) | 0 (0) | 0 | 0 | 3 | 2 | 0 | +0.029 [−0.602, +0.627] | +825 | fail |

How to read the table:
- **Δmargin** is the fixed-competitor margin change on the paired EN-confusion cohort: dialogue macro, family-8
  adjusted interval. **Δrank** is the position mean on the same cohort.
- **Energy validity:** every arm is energy-valid, all arms belong to qualified layers, and retention is 1.00.
- **Controls:** the matched random and off-target controls also made 0 corrections in every arm, on the cohort and on
  all 60 EN-confusion rows.
- **No oracle union:** the union of corrected positions across all arms (an oracle, not deployable) is **empty**.
- **Unconditional (of 60):** EN-confusion corrections are 0 in every arm.
- **Descriptive target-only cohort (28 rows):** 0 target and 0 random corrections in every arm.

Damage:
- Correct-state EOS promotions: 0 in every arm.
- Maximum per-arm corruptions: 3 EN-correct and 2 ZH-correct (L24 η.30 −).
- The global serious-damage stop (≥ 12 / ≥ 3 / ≥ 2) is not met.
- Every edited Mandarin change is enumerated in `evaluation.json` (`mandarin_active_changes`). They occur at ZH-correct
  queries j = 135 (CSD0538), 140 (CSD0006) and 176 (CSD1006), all on the 6 target-active ZH rows.
- The off-target and random controls cause comparable damage: 0–1 EN and 0–1 ZH per control arm.

**Gate P-B:**
- No arm passes the power gate: 0 < 3 corrections, net advantage 0 < 2 against both controls, macro advantage
  0 < 0.10, and LODO cannot be positive.
- No arm meets MARGIN_ONLY, which needs macro ≥ 0.50 nat and adjusted lower bound > 0. The closest is L24 η.30 +,
  at macro +0.467 with lower +0.062. L16 η.30 + has lower +0.016 but macro only +0.352.

## Statistical analysis

- 10,000 paired dialogue-cluster draws (seed 240924, 20 sorted dialogues, shared indices); 10,000/10,000 finite draws
  for every reported interval.
- The 16 primary contrasts (8 arms × random / off-target corrected-indicator difference) are exactly 0 with adjusted
  interval [0, 0], because no system corrected anything. LODO net advantage is 0 for every removal.
- Denominators:
  - Unconditional: 60/60/60.
  - Target-active: 28/39/6.
  - Joint-valid paired: 20/35/4.
- The intervals are descriptive by design (frozen); the negative verdict follows from the integer gates.
- Limitations:
  - The paired EN-confusion cohort is only 20 rows in 9 dialogues.
  - Only 6 target-active (4 paired) ZH-correct opportunities exist. They cannot bound active Mandarin harm; a
    zero-event or 1–2-event count on 6 opportunities is uninformative.
  - S1 eligibility depends on attention-based region association, which restricts which positions are reachable at all.

## Audits

| Stage | Verdict | Attempts |
|---|---|---|
| Pre-run | `PASS_TO_SRC_CF0_PILOT` | 1. Includes independent region re-derivation, mask bytes, firewall static scan, 60-case synthetic helper-vs-independent agreement, committed CPU engineering smoke, 51 pilot tests |
| Construction | `SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)` 31/31 | 1. Independent float64 re-derivation of every v_AC (byte-equal), random vectors, geometry, reach ledger, repeats, historical identities, open-file firewall (0 forbidden) |
| PRIMARY | `SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)` 26/26 | 1. Independent energy checks on all 1,640 steered cells, direction-used hashes, top-1, paired energy, eligibility, NONE / zero / apparatus / restore, authorization minimality |
| FULL | `SRC_CF0_PILOT_AUDIT: PASS (FULL)` | 1. Independent Gate P-A, cohorts, corrections, nets, LODO, damage, bootstrap, margin, precedence, selection: label identical, 0 arm differences |

Failed attempts preserved: P-B launch 58238. It was a launcher precondition `grep` that expected a space in compact JSON.
No Python started, so there was no model load and no forward. It was fixed in the launcher only and recorded in
`run1/pulse_attempt1_LAUNCH_FAIL.json`; the original pulse manifest is kept as
`manifest_pulse_attempt1_LAUNCH_FAIL.json`. No audit attempt failed.

## Scientific interpretation

1. **Can v_AC be constructed reliably?** Yes. 73/73 targets and 59/59 pairs were valid at both layers, with
   bitwise-repeatable states, logits and vectors. The masked-minus-clean residual is large: median 4.1 at L16 and 5.8
   at L24, against a clean state norm of about 7–9. It is mostly tangent to the state (ratio about 0.95).
2. **Is it distinguishable from off-target acoustic changes?** Geometrically yes. It is about 7–10× larger than the
   off-target change, nearly orthogonal to the off-target direction (median |cos| about 0.1), and specificity passes
   with a wide margin. This is construction distinction, not evidence of lexical or phonetic content.
3. **Does it correct actual English tokens?** No. There were 0 corrections in all 8 arms, both signs, both doses and
   both layers, on 20 paired and 28 target-active EN-confusion queries.
4. **Does it outperform matched controls?** Not on corrections: all controls are also at 0. On margins, the + sign at
   η 0.30 raises the fixed-competitor margin more than NONE (L24 +0.47, L16 +0.35 nat macro) and improves the
   reference rank by roughly 40–120 positions on average. The − sign tends to lower it. So the direction carries a
   sign-consistent push toward the reference, but it is far too weak to flip a top-1. Controls were not compared on
   margins by a frozen gate.
5. **Is observed damage acceptable?** Within the frozen limits: at most 3 EN-correct and 2 ZH-correct corruptions per
   arm, 0 EOS promotions, and no global stop. Controls do similar damage.
6. **What remains unknown about Mandarin safety?** Almost everything. Only 6 ZH-correct queries were ever target-active
   (4 paired), and 1–2 of them changed top-1 in several arms. Active Mandarin harm risk is unbounded by this pilot.
   FULL300 feasibility remains `FULL300_SAFETY_FEASIBILITY_UNKNOWN`.
7. **Does the evidence justify further work on this direction?** Not on the frozen rules. The direction is real and
   specific as a representation change, but it has no correction power at the frozen doses. The margin effect is below
   the frozen 0.50-nat diagnostic bar. A further study would need a new, separately frozen hypothesis, such as larger
   doses or multi-position or sequence application. Nothing here authorizes one, and none was run.

## Final decision

- Terminal verdict: **`SRC_CF0_PILOT_CAUSAL_INSUFFICIENT`** (FULL audit PASS, independent label identical).
- Recommend an expanded Mandarin-safety study: **NO**. The pilot produced no correction signal whose safety would
  need establishing.
- Proceed to SRC-CF1: **NO**.
- Proceed to TTO-CF: **NO**.
- The FULL300 plan status stays `FULL300_SAFETY_FEASIBILITY_UNKNOWN`. No census or expanded-safety experiment was
  implemented or run.

## Compute

| Job | Phase | Wall | In-process | Peak VRAM alloc / reserved | Forwards |
|---|---|---|---|---|---|
| 58236 | P-A | 3 min 30 s | 184 s | 3.42 / 3.74 GB | 408 encoders, 13,318 cached decoder steps, 4,320 native-BF16 solver ledger cells, 0 pulses |
| 58238 | P-B launch (failed precondition) | 1 s | — | — | 0 |
| 58239 | P-B | 2 min 14 s | 109 s | 3.42 / 3.74 GB | 80 encoders, 3,159 prefix steps (incl. 180 NONE), 360 zero, 540 apparatus, 1,640 matrix pulses |

In-allocation throughput forecast after the first utterance: P-A 365 s, P-B 479 s, both far below 3 h. There were no
LID, autograd or optimizer calls, and no continuation decoding.
