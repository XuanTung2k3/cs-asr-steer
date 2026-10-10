# P2-DIR — final report (terminal for this stage)

**Terminal frozen state: `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`**
(supplementary, reported alongside per spec section 9: `P2_DIR_CAUSAL_POWER_WITH_DAMAGE:D2`).
Exp-1 audit `P2_DIR_AUDIT: PASS` (EXP1); final session audit `P2_DIR_AUDIT: PASS`
(`results/inference_cf/p2dir/final_audit.json`). Exp-2 (gate coupling) and Exp-3 (free decoding) were
**not run**: under the frozen stop rule, Exp-1 selected no candidate.

## Stages

| Stage | Run | Result |
|---|---|---|
| Implementation + tests | `1802441` | 3 providers (D0/D1/D2) behind one interface; 55 new focused tests + existing suites pass |
| Pre-run audit | `00dd8b2` | `PASS_TO_P2_DIR_EXP1` |
| Extraction | Slurm 57789 | 180/180 states reproduce P2-R |
| D1 folds | CPU | 20/20 valid |
| Exp-1 | Slurm 57792 | valid; no candidate qualifies |
| Exp-2 | — | NOT RUN (stop rule) |
| Exp-3 | — | NOT RUN (stop rule) |

Compute: about 6 min of MIG GPU time in 2 jobs (target ≤ 4 h). D2 readout: one autograd call per position (180),
median 0.039 s; peak VRAM 4.0 GB allocated / 4.3 GB reserved.

## Exp-1 outcome (family 10, Bonferroni; details in `P2_DIR_EXP1_REPORT.md`)

| | D0 OLD | D1 UNIQUE | D2 READOUT |
|---|---|---|---|
| EN-confusion Δm (nats) | −0.020 (pointwise) | +0.227 [+0.079, +0.421] | +4.499 [+3.829, +5.168] |
| paired vs D0 | — | +0.247 [+0.107, +0.457] | +4.519 [+3.874, +5.169] |
| EN-correct Δm | +0.027 | −0.034 [−0.165, +0.103] | +1.535 [+1.042, +2.136] |
| ZH-correct Δm | −0.047 | −0.040 [−0.099, +0.017] | −2.011 [−2.466, −1.584] |
| pooled correct corruption | 0 | 0.000 [0, 0] | 0.077 [0.013, 0.163] |
| κ vs evaluator tangent (confusion) | −0.011 | +0.021 | +0.797 |
| verdict | comparator | fails materiality (0.5 nat) | fails safety (causal power with damage) |

## Bottleneck update

Frozen label: **direction construction still unresolved** for a deployable, safe actuator under this screen.
Supporting observations (descriptive; they do not change the label):

* The old direction (D0) is confirmed inert again at matched energy (κ ≈ 0).
* The L16 DG-02 site is not leverage-limited for a well-chosen direction. At the same e*, the
  reference-free readout tangent (D2) moves the confusion margin by +4.5 nats and captures about 80% of the
  first-order leverage (A ≈ 5.7 nats). This is descriptive evidence against "site has too little leverage"
  for single-step lexical moves, though the spec defines no site label for Exp-1.
* The binding problem for D2 is selectivity: an ungated push toward English damages correct Mandarin states
  (κ −0.58, 13.7% ZH-correct top-1 corruption). Whether the frozen gate/localizer would supply that
  selectivity was **not tested**, because the frozen rule does not let a safety-failing candidate progress.
* The development-fitted unique subspace (D1) is safe but captures only about 2% of the available leverage.

## Data exposure

Consumed only already-exposed D-dev-select P2-R/P2-RJ states (80 utterances, 20 dialogues, 180 positions).
No new data role, no fresh validation, no router-calib/D-dev-confirm/D-test/P3/transfer use. P3 HELD.

## Exact next action

The stage is terminal. The next step is a human decision, not something to continue in this session. If
pursued, it needs a new, separately frozen pre-registration before any outcome, for example an independent
design pass that decides whether to pre-register a gate-coupled screen of the readout direction under new
safety rules. Such a stage would be a post-Exp-1 redesign, would be motivated by these exposed results, and
would count only as development evidence. No redesign, gate change, P3 or fresh confirmation is authorized
or started here.
