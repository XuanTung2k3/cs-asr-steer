# SRC-CF0-P — implementation notes (pre-execution, outcome-blind)

Written and committed with the implementation, BEFORE the pre-run audit, any GPU allocation, any new model forward on
panel audio, any direction outcome and any lexical reference. They record how the frozen SPEC / DESIGN / FIREWALL /
HANDOFF / config (freeze `2d23574`) were mapped to code. No threshold, sample, mask, direction equation, control, dose,
statistical test or stopping rule is changed. Where the frozen text admits more than one literal reading, the reading
below is fixed now and is never revisited after outcomes.

## Files

| Path | Role |
|---|---|
| `src/csasr/inference_cf/src_cf0_pilot.py` | reference-free helpers: allowlisted runtime projection, v_AC (same arithmetic as the frozen `cf_pilot_contract.direction`), geometry, random controls, P-A reachability ledger, frozen pulse checks, pre-FFN probe |
| `experiments/inference_cf_src_cf0_pilot.py` | runner: `prepare`, `manifest`, `construct`, `seal-a`, `pulse`, `seal-b` (+ CPU `smoke` on synthetic audio, engineering only) |
| `experiments/inference_cf_src_cf0_pilot_evaluate.py` | isolated evaluator: `gate-a`, `evaluate-b` (+ `negative` after a failed Gate P-A) |
| `experiments/inference_cf_src_cf0_pilot_audit.py` | independent auditor: `pre`, `construction`, `primary`, `full` |
| `slurm/inference_cf_src_cf0_pilot.sbatch` | PHASE=construct / pulse; one allocation per phase; <=3 h; mig 3g.40gb (historical GPU type) |
| `tests/test_src_cf0_pilot_impl.py` | focused CPU tests (synthetic arrays, sealed metadata, tiny random Whisper) |

Outputs: `results/inference_cf/src_cf0_pilot/run1/` (runtime projection, manifests, construction rows, seals, audits,
gate, authorization, pulse rows, evaluation). Failed attempts are preserved under distinct names, never overwritten.

## Reuse (unchanged, hash-pinned)

S1 `lexical_compatibility.hard_mask` / `waveform_model_inputs`; S1 sealed region records (re-derived in `prepare` from
the sealed S1 cached-query heads + R0 PRIMARY `track_heard_intervals`, and independently by the auditor's exact-Fraction
route); `inference_cf_p2r.DiagBranch` / `solve_scale` / `scaled_direction`; `loc0_sites.pulse_action` /
`cross_pulse_hook` / `Composite` / `cache_fingerprint`; `prompt_r1.relative_action` / `cell_checks` (the relative-dose
pulse of ST-PROMPT-R1, i.e. target = eta x float64 norm of the live native BF16 site); `models.hooks.apply_steering`
(NormPreserve, no depth rescale); `lss.sites.DecoderPostCrossAttnRecorder`. No R0 full-replay state, no S1 masked logit
is used as a representation (S1 masked logits are only bitwise identity comparators).

## Interpretations fixed before execution

1. **Runtime projection.** Exactly `config.firewall` keys: queries (6 keys), utterances (7 keys; content tokens truncated
   at each utterance's largest query t), regions (`target`, `off_target`, `paired_available` verbatim from the sealed S1
   records). Waveform / mask byte hashes and sealed comparator file hashes are separate top-level sections of
   `runtime.json`. GPU phases never open `SRC_CF0_PANEL.json` (not even for hashing); `prepare` alone does.
2. **P-A sweeps.** Two complete independent sweeps (sweep 1 over all 80 utterances, then sweep 2), each input with a cold
   encoder and a cold model-owned forced-ZH cache; both layers and the pre-FFN input recorded in the same passive pass.
   Masks are grouped by exact (utterance, bounds); one cold cache per mask through the largest member t (= S1 grouping,
   so the frozen bound 408 encoders / 13,318 steps is met exactly). Sweep-2 arrays are stored in full for the auditor.
3. **Numerical validity.** A target (off-target) direction is "numerically valid" when its status is `OK`
   (raw delta norm >= 1e-4). Valid off-target PAIR = target and off-target both `OK` on a sealed paired row.
4. **P-A joint reachability (per query, per layer).** Target and off-target valid; target / off / random tangent ratio
   >= 0.25; for every (family, sign in {+,-}, eta in {0.15, 0.30}) the native-BF16 emulation of the unchanged P2-R
   solver reaches the target chord (solver `ok`, emulated relative squared error <= 0.02, realized eta error <= 0.02);
   and for each (sign, eta) the three families' emulated squared energies differ by <= 2 % (max/min - 1). The emulation
   runs on the GPU on the exact native state (P-A has no forward with an edit).
5. **Specificity cohort.** All EN-confusion rows with valid target AND off-target directions at that layer (before any
   energy filter) whose off-target RAW tangent norm is > 0 (zero would abstain, never infinity). Medians over that cohort.
6. **Random control eligibility.** The random arm mirrors its target arm: it is pulsed at a position iff the target
   direction at that layer is numerically valid ("compared only on matching target eligibility"); otherwise explicit
   no-edit. The off-target arm is pulsed iff the off-target direction is valid. Base random vector: one per UID/t/layer,
   shared by both signs and doses (sign multiplies it).
7. **P-B paired cohort of an arm (L, eta, sign).** Positions in the SEALED P-A joint-reachable set of layer L whose
   actual target, off-target and random cells of that configuration are all valid (four frozen checks below) with paired
   realized squared energies within 2 %. No row outside the sealed set is ever added. Retention = cohort / sealed set
   (all strata); an arm is energy-valid iff retention >= 0.90 and none of its 24 x 180 cells has an integrity failure.
8. **Frozen per-cell validity.** eta relative error <= 0.02; squared-energy relative error <= 0.02; solver-vs-hook
   |edit norm difference| <= 1e-6 * max(1, edit); consumed-vs-proposed relative error <= 0.005. A consumed-vs-proposed or
   solver-vs-hook failure, a no-edit cell that is not bitwise baseline, a changed upstream layer, a broken cache
   fingerprint, nonfinite logits or an unexpected no-edit reason is a critical integrity failure (INVALID).
   BF16 norm rounding and proposed-vs-hook equality are recorded as descriptive only (ST-PROMPT-R1 amendment A1).
9. **NONE / zero / apparatus ordering.** At each query: zero dose at L16 and L24, the L16 historical apparatus
   (ST-LOC0 D0 +/- vectors and sealed D2 at e* = 1.1260757575454359, bitwise vs ST-LOC0 logits / consumed states, solver
   scale within 1e-9*max(1,s)), then the 24 matrix cells, each from the cropped pristine cache; finally the live NONE,
   which must be bitwise the sealed P-A clean readout and also advances the pristine cache. Zero / no-edit cells are
   compared bitwise against the sealed clean readout. This keeps the pulse job inside the frozen forward bound
   (NONE+zero 540, apparatus 540, matrix <= 4,320, 80 encoders, 3,159 prefix steps).
10. **Correction / damage.** Acceptable sets = P2-RJ `target_ids` (< EOS); processed float32-from-BF16 logits, top-1 by
    the lowest token ID on ties. Corrections and control corrections are counted on the arm's paired EN-confusion cohort.
    Arm damage (EN-correct / ZH-correct corruptions, correct-state EOS promotions) is counted on ALL target-active rows
    (every actually edited target cell, including target-only rows); the global serious-damage stop uses the same counts.
11. **Margin diagnostic.** Fixed-competitor margin (log-sum-exp over the acceptable set minus the P2-RJ competitor logit,
    as in ST-PROMPT-R1), edited minus NONE, on the arm's paired EN-confusion cohort; equal-dialogue macro;
    family-8 quantiles .003125 / .996875; >= 9,900 finite draws or undefined.
12. **Bootstrap.** 10,000 draws, seed 240924, 20 sorted dialogue IDs sampled with replacement (shared indices), dialogue
    means weighted by multiplicity; draws with no represented dialogue are undefined. 16 primary contrasts
    (8 arms x {random, off-target} corrected-indicator differences), quantiles .0015625 / .9984375; descriptive only.
13. **Decisions.** The evaluator applies the Codex-frozen `cf_pilot_contract.construction_layer`, `power_pass`,
    `observed_safety_pass` and `terminal`; the auditor re-implements them independently. Selection order exactly as frozen.
14. **Compute guard.** After the first utterance of P-A, the measured throughput forecasts both jobs; a forecast above
    2.5 h stops the job for an explicit compute revision (no matrix or population reduction).

## Engineering smoke

`smoke` runs the exact capture / direction / pulse code paths with the real frozen model on CPU on SYNTHETIC audio with
injected regions (no panel audio, no outcome). It is engineering evidence for the pre-run audit, not scientific inference.
