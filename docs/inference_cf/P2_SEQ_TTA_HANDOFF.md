# P2-SEQ → P2-TTA0 evidence handoff (evidence only)

**Status:** written after terminal `P2_SEQ_SEQUENCE_DAMAGE` (`P2_SEQ_AUDIT: PASS`), as the frozen P2-SEQ contract
requires. **This is not a TTA contract.** It does not pick a learning rate, number of steps, trainable layers or
objective. It implements nothing and runs no adaptation, and no TTA outcome has been looked at. A P2-TTA0 design
must be separately frozen by an independent design session before any code or GPU job.

Repository `XuanTung2k3/cs-asr-steer`, branch `cs-asr-steer-inf`. Authoritative detail: `P2_SEQ_REPORT.md`, plus
the P2-SEL / -E / -T / -XA / -LAC reports in `docs/inference_cf/`.

## A. Evidence motivating a test-time adaptation (TTA) line

1. **Inference steering does not survive the sequence.** On the fixed 100-utterance D-dev-select panel, the existing
   E·R_B·D2 controller (L16 DG-02, alpha 2, NormPreserve) gives net +6 POI and PIER −0.0086 (95% CI spans 0). It also
   gives MER +0.028 and ZH-CER +0.033 (both CIs exclude 0), matrix-ZH retention 0.940 and lexical outside-POI harm
   0.060. Damage is mostly Mandarin deletion and early end of transcript after an edit: deletions go from 1026 to 1299.
2. **The forced-language decision is the larger lever.** AUTO (Whisper choosing its language) beats forced-ZH by
   0.088 PIER at a MER cost of only +0.002
   (ZH-CER +0.014). STEER is clearly worse than AUTO on PIER (+0.079, 95% lower bound +0.0125).
   Most of the recoverable error therefore sits in the conditioning/prompt regime, not in sparse per-token edits.
3. **One-direction, per-token edits are a narrow actuator.** 237 edits over 4005 eligible steps (69 utterances), mean
   gate 0.027. Each edit is a single rank-one push at one site, and when it misfires the decoder carries the damage
   forward through the rest of the sequence. An adaptation that changes the model consistently over a whole
   utterance is a different, untested hypothesis class. This is the motivation for an episodic TTA line, not evidence
   that it will work.

## B. Why further local gate / LID / acoustic heuristics are not prioritized

Every audited, reference-free local selectivity signal has failed its frozen separation criteria on the exposed
positions:

| Stage | Signal | Terminal |
|---|---|---|
| P2-SEL | E·R_B gate | `GATE_INSUFFICIENTLY_SELECTIVE` |
| P2-SEL-E | window LID evidence diagnosis | `DIAGNOSIS_AMBIGUOUS` |
| P2-SEL-T | token-aligned LID | `TOKEN_LID_NOT_DISCRIMINATIVE` |
| P2-SEL-XA | cross-attention source gradient | `INVALID` (material subset 4/30 rows) |
| P2-SEL-LAC | local acoustic occlusion | `NOT_DISCRIMINATIVE` (EN-TP strong 6/42) |
| P2-SEQ | sequence screen of the existing gate+D2 | `SEQUENCE_DAMAGE` |

LAC showed why. At EN-confusion points, the English candidate's deficit does not depend on the attended local audio
(median S ≈ +0.08). True and false positives look the same, so a sharper local detector has little to work with.
These failures do not prove the information is absent. They do mean that another local heuristic has a low expected
return compared with a sequence-level change.

## C. Frozen constraints a future P2-TTA0 design must honour

- **Reference-free adaptation.** No transcript, reference, POI label or evaluator output may enter adaptation or
  runtime construction.
- **Episodic first.** Adapt within one utterance and reset to the original weights before the next. Persistent or
  continual adaptation would need its own separate justification later.
- **Tiny parameter subset.** Adapt only a small, explicitly enumerated set of parameters; which set is left to the design
  session. No full-model tuning.
- **Entropy minimization is a baseline, not the assumed winner.** It must be compared, never presumed.
- **Problem-specific candidate.** Investigate prompt / condition-consistency objectives, e.g. agreement between the
  forced-ZH and AUTO/forced-EN conditioned branches. These follow from evidence A.2.
- **Same exposed development firewall.** Only already-exposed D-dev-select material (the fixed 100 panel and existing
  `results/inference_cf/` artifacts). No router-calib new role, D-dev-confirm, D-test, SEAME, CS-FLEURS, ViMedCSS,
  ASCEND, transfer corpora or fresh validation. No full-300 run. P3 stays HELD.
- **Same safety vocabulary.** MER / ZH-CER increase, matrix-ZH and embedded-EN retention, POI corruption, lexical
  outside-POI harm and caps against matched forced-ZH. AUTO is reported as the practical reference. A pre-registered
  label precedence must be in place before any outcome.
- **Process.** Freeze before implementation, independent pre-run audit, at most bounded GPU jobs, independent post
  audit, and the core v6 `METHOD_CONTRACT` unchanged.

Left open for the design session (not chosen here): objective, learning rate, step count, trainable subset,
optimizer, reset granularity, stopping rule, compute budget.

## D. Data and results that may be reused

- Panel `docs/inference_cf/P2_SEL_MINI_PANEL.json` (sha256 `266ea7ea…`, 100 utts / 20 dialogues).
- P2-SEQ run `results/inference_cf/p2seq/run1/`: per-step S0/S2 rows, edit vectors, LID/gate traces and manifest.
  `run1_analysis.json` / `run1_audit.json` hold the S0, S1 and S2 baselines.
- Historical AUTO rows `results/inference_cf/p2_A_r1_L16/rows/*.json`, re-usable under the same sealed proof
  (`results/inference_cf/p2seq/reuse_audit.json`).
- Analysis and audit code: canonical metrics/bootstrap in `experiments/inference_cf_p2seq_analyze.py` and the
  independent auditor `experiments/inference_cf_p2seq_audit.py` (as templates). Matched no-edit driver
  `experiments/inference_cf_p2seq.py::seq_decode` (alpha 0 is bitwise the clean forced-ZH branch).
- Exposed-position evidence from P2-R / P2-RJ / P2-SEL-* (180 positions, 80 utterances), for diagnosis only.
