# P2-SEL — Readout-Direction Selectivity Debug

**Status:** pre-outcome development freeze, 2026-10-06. This stage was motivated by the exposed
P2-DIR result and can produce development evidence only. It does not reopen direction search or
authorize a core-method claim, P3, confirmation data, or a full 300-utterance rerun. The active
core-paper method in `docs/current/METHOD_CONTRACT.md` remains authoritative; P2-SEL is a bounded
legacy inference-time diagnostic and must not be presented as a contract-conformant core method.

The authoritative machine-readable values and hashes are in
`configs/inference_cf/p2_sel.json`. The exact 100-utterance S2 panel is frozen in
`P2_SEL_MINI_PANEL.json`. If prose and config disagree, stop before any outcome and report the
conflict; do not infer a resolution.

## 1. Question and starting evidence

Question: **Can the existing gate/localizer selectively deploy the powerful D2 readout direction
on EN-confusion positions while suppressing its known damage on correct Mandarin positions?**

P2-DIR Exp-1 is complete and audited. On its exposed 180 positions, ungated D2 gave EN-confusion
`Δm_ref = +4.499 nat` (95% simultaneous-family interval `[+3.829,+5.168]`, `κ≈+0.797`) and
ZH-correct `Δm_correct = −2.011 nat` (`[−2.466,−1.584]`, `κ≈−0.582`), with 8/60 correct-ZH
top-1 corruptions (13.3% in the position counts; P2-DIR reports dialogue-weighted 13.7%). Its
terminal P2-DIR label was `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`; the supplementary D2 result was
causal power with damage. P2-SEL is a separately frozen post-outcome development stage. It does
not revise P2-DIR's stop decision or claim that P2-DIR selected D2.

## 2. Frozen intervention

Use D2 `p2dir_readout_v1` exactly as in the sealed P2-DIR Exp-1 artifacts:

`J(h)=log(P_E(h)+1e-12)−log(P_M(h)+1e-12)`, with the frozen
`whisper_han_ascii_v1` partition, generate-suppressed float32 logits at temperature 1, one
`autograd.grad` through a zero-valued float32 leaf at the current L16 B-branch DG-02 site, then
float64 tangent projection against `h/||h||` and epsilon normalization (`1e-12`), returned as
float32. The P2-DIR provider's tiny/nonfinite fallbacks, scratch-cache isolation, parameter-grad
isolation, and site/logit identity checks remain in force. S1 reuses the sealed direction rows;
it does not recompute ungated directions or the ungated outcome. S2 calls the same provider at
runtime using only the current B prefix/cache and reference-free inputs. No transcript, alignment,
reference token set, evaluator margin/Jacobian, future token, or outcome may construct or orient D2.

Frozen deployment components:

| Component | Value |
|---|---|
| Decoder layer | L16, zero-based |
| Site | DG-02 decoder post-cross-attention residual, pre-FFN |
| Gate | `g=E*R_B`, from unchanged R2 `local_support` and baseline conflict providers |
| Dose | `alpha=2`, `phi(g)=g` |
| Repair | NormPreserve ON; zero gain is exact no-edit |
| Localizer | Existing `max_attention_window` 50-frame (1.0 s) window, unchanged |
| Depth scaling | None |
| Decode | Existing forced-ZH B/E/S semantics, greedy, bf16/eager, 200-token cap |

`R_B` means the existing baseline conflict `R=max(0,P_M−P_E)`; `E` is the existing R2 local
support signal. No gate, localizer, direction, layer, alpha, or dose-map search is allowed.

## 3. S1 — 180-position gate-coupled mechanism screen

### Population and estimands

Use exactly the ordered keys and strata in `results/inference_cf/p2rj/positions.json`, anchored
by the P2-R/P2-RJ hashes in the config: 60 EN-confusion, 60 EN-correct, and 60 ZH-correct
positions, on the same 20 diagnostic dialogues and identical unedited B states/prefixes. Do not
refill, replace, relabel, or add positions. `E`, `R_B`, and `g` are joined from the audited
P2-R run3 current-gate records by `(utterance_id, absolute query)`; require exact state/query
identity to the P2-DIR extraction. Missing or mismatched gate rows invalidate S1.

Use the P2-DIR frozen evaluator reference set, fixed competitor, suppression, and margin
definition. The endpoint is the arm-minus-NONE fixed-competitor reference margin at the same
state. References and evaluator quantities are read only after direction/gate construction is
sealed.

### Arms

All are independent single pulses from the same unedited state at the frozen query; no arm has
earlier edits.

| Arm | Action |
|---|---|
| C0 NONE | Exact zero-gain control; reuse the bitwise-matched P2-DIR/P2-RJ baseline logits and state when their hashes and identities pass audit. |
| C1 OLD-GATED | `alpha=2`, `g=E*R_B`, `phi(g)=g`, unchanged NormPreserve, D0 `core_p1.direction(h_E,h_B)` unchanged. Run the pulse on the S1 population; old P2 free-decode outputs are not a substitute for these single-pulse rows. |
| C2 D2-GATED | `alpha=2`, `g=E*R_B`, `phi(g)=g`, unchanged NormPreserve, sealed per-position D2 direction from P2-DIR. |
| C3 D2-BROAD | Same D2 direction and repair with `g=1`; set one pooled energy target from C2 as specified below. No stratum labels or evaluator quantities enter allocation. |

### C3 energy match

Let `Q2 = Σ_i ||h'_C2,i−h_i||²` using the **actual** post-NormPreserve state edit for each of
the 180 valid frozen query rows. Let `N=180`, the fixed count of runtime-eligible diagnostic
queries (all rows are eligible regardless of stratum or C2 gate value). Set the single common
target chord norm `e_broad=sqrt(Q2/N)`. At every C3 query, use the unchanged P2-R `solve_scale`
procedure with the C3 site's unit D2 axis to solve for its effective scalar `λ_i` that reaches
`e_broad`. Apply with the frozen `alpha=2` and gate exactly `g=1`, passing the collinear
direction tensor `(λ_i/2)*d_readout`; this keeps the D2 axis, deployment alpha, and BROAD gate
fixed while the scalar direction amplitude supplies only the predeclared energy match. Verify
the actual hook output after site-dtype rounding. A solver-unreachable row realizes zero and is
retained in the aggregate. If `Q2=0`, set the target to zero, C3 is exact no-edit, and zero
matches zero. Otherwise C3 is energy-valid only when
`abs(Σ_i ||edit_C3,i||²/Q2−1)≤0.02`; otherwise S1 is invalid. This is one pooled rule, fixed
before outcomes, with no label-, reference-, or margin-dependent allocation. Report per-row
and aggregate energy plus unreachable count. The P2-R solver cap is eight evaluations.

### S1 measurements

Primary and safety quantities, with no exploratory metric family:

- EN-confusion mean `Δm_ref` for C2 and C3, and paired C2−C3;
- EN-correct and ZH-correct mean `Δm_correct` for C2 and C3;
- top-1 corruption rate on each correct stratum;
- report the same margin and corruption summaries for C1 OLD-GATED as a historical comparator;
  C1 is descriptive and never enters S1 pass/label decisions;
- raw `E`, `R_B`, `g=E*R_B`, and realized edit energy, by stratum as median and IQR, plus
  fraction `>0` for gate and realized edit;
- descriptive ratios against the frozen ungated D2 P2-DIR rows on the identical positions:
  `r_benefit = mean(Δm_conf,C2)/mean(Δm_conf,ungated D2)` and
  `r_harm = |mean(Δm_ZH,C2)|/|mean(Δm_ZH,ungated D2)|`.

The P2-DIR denominators are fixed at `+4.499 nat` and `−2.011 nat`; calculate ratios from the
matched per-dialogue estimates, and report both the ratios and denominators. Ratios are
descriptive and do not independently decide the label. A missing/nonfinite denominator or a
population mismatch is invalid.

Margin definition, top-1 tie rule, and correct-state reference membership are exactly those in
P2-DIR: generate-suppressed float logits, fixed evaluator reference tokens and fixed competitor;
first token ID wins an argmax tie. `Δm` is arm minus NONE. A correct-state corruption is a
baseline-correct position whose post-edit argmax is outside its frozen reference set.

### S1 uncertainty and frozen thresholds

For each endpoint/stratum, average positions within each dialogue that has at least one row in
that stratum, then take an equal-weight mean across those represented dialogues. The bootstrap
universe is always the 20 frozen dialogue IDs. Generate 10,000 **shared paired dialogue
bootstrap** draws using `numpy.random.default_rng(240924)`, sampling 20 dialogue IDs with
replacement; for each endpoint, omit sampled dialogues with no rows in that stratum from that
draw's numerator and denominator. Use percentile intervals and require at least 9,900 valid
draws for every decision quantity. For the 12 decision quantities below, use Bonferroni
simultaneous 95% intervals (`alpha=0.05/12` per two-sided interval): C2 and C3 EN-confusion
`Δm`; C2 and C3 EN-correct and ZH-correct `Δm`; paired C2−C3 EN-confusion and ZH-correct `Δm`;
and EN-correct and ZH-correct corruption rates for both C2 and C3. Report pointwise 95%
intervals for descriptive quantities only. Compute corruption intervals by the same dialogue
bootstrap over within-dialogue proportions.

C2's frozen standalone material-benefit and safety requirements are all mandatory:

1. EN-confusion C2 `Δm` point estimate `≥ +0.50 nat` and its simultaneous lower bound `>0`.
2. EN-correct and ZH-correct C2 `Δm` simultaneous lower bounds each `≥ −0.25 nat`.
3. For each correct stratum, observed corruption `≤0.05` and its simultaneous upper bound
   `≤0.05`.
4. All 180 frozen keys are emitted once per arm, and every state/gate join and direction
   provenance check passes. A predeclared D0 tiny/nonfinite no-edit fallback remains a measured
   no-edit row; it is never dropped or refilled. Any missing/duplicate row or state identity,
   gate join, D2 validity, or audit failure invalidates S1.

The +0.50 nat threshold is the frozen P2-DIR materiality anchor (roughly 11% of D2's known
+4.499 nat effect and well above the historical 0.05–0.1 nat near-inert range). The −0.25 nat
and 5% corruption bounds are inherited unchanged from the P2-DIR safety contract.

To call the gate selectively useful against equal-energy broad steering, additionally require:

- paired C2−C3 EN-confusion lower bound `≥−0.25 nat` (non-inferior benefit); and
- paired C2−C3 ZH-correct margin lower bound `≥+0.10 nat` (material Mandarin-harm reduction).

“C2 adds little versus BROAD” is an equivalence label, not a failure to reject a difference:
C3 must independently meet the same standalone material-benefit and safety requirements, and
the shared-bootstrap 90% intervals for C2−C3 EN-confusion and ZH-correct margins must lie wholly
inside `[-0.25,+0.25] nat` and `[-0.10,+0.10] nat`, respectively. These intervals are pointwise
90% two-one-sided-test intervals and are used only for that label.

### S1 label precedence and stop

Apply exactly this precedence:

1. `P2_SEL_INVALID` if any provenance/population/state/direction/energy check fails, any required
   interval has fewer than 9,900 valid draws, or an arm lacks any of the 180 frozen row records.
2. `P2_SEL_GATE_ADDS_LITTLE_VS_BROAD` if C2 and C3 both meet standalone requirements and both
   prespecified equivalence intervals fall wholly within their margins.
3. `P2_SEL_GATE_RESCUES_D2` if C2 meets standalone requirements and both C2−C3 selective-use
   bounds pass.
4. `P2_SEL_GATE_TOO_CONSERVATIVE` if C2 meets both correct-state safety requirements but fails
   the EN-confusion material-benefit requirement, regardless of C3.
5. `P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE` in every other valid case.

Only `P2_SEL_GATE_RESCUES_D2` authorizes S2. `ADDS_LITTLE_VS_BROAD`, `TOO_CONSERVATIVE`,
`INSUFFICIENTLY_SELECTIVE`, and `INVALID` all stop P2-SEL after S1. If stopped, provide only a
descriptive table of existing S1 `E`, `R_B`, `g`, and realized-dose distributions by stratum;
do not run another diagnostic, oracle localization, gate repair, or GPU experiment.

## 4. S2 — conditional 100-utterance mini free-decoding screen

S2 runs only after `P2_SEL_GATE_RESCUES_D2` and `P2_SEL_AUDIT: PASS (S1)`. Use the exact 100 rows
in `P2_SEL_MINI_PANEL.json`; do not regenerate, refill, or filter by labels/outcomes. The role is
already exposed `D-dev-select`: its existing 20-dialogue universe is fixed by the parent P2-A
300-utterance panel and role manifest. In each dialogue, eligible IDs are the role-mapped
parent-panel IDs recorded in the JSON. The five selected IDs are the first five under ascending
`(hex(SHA256(UTF8("P2SEL-mini100-v1|" + utterance_id))), utterance_id)`. Emit dialogues in
lexicographic order and rows in that hash order. The JSON records every eligible ID, selected
stable ID, per-row hash, parent panel/mapping hashes, and its canonical payload digest. Selection
uses no audio, transcripts, labels, baseline correctness, or outcomes.

Run exactly:

| System | Definition |
|---|---|
| B0 | Matched no-edit forced-ZH cached baseline, generated through the same S2 path as interventions. Reuse only if exact utterance IDs, model/tokenizer, prompt, generation settings, and cached-path identity match; otherwise generate in the one S2 job. |
| OLD | Current gate `E*R_B`, `alpha=2`, `phi(g)=g`, L16 D0, DG-02, NormPreserve. |
| NEW | Same frozen gate/site/dose, D2 readout provider unchanged. Compute a direction only when the row is runtime-eligible and the gate is positive, from the pre-step B cache/prefix and a scratch forward that cannot mutate B/E/S caches. Apply it to S's current DG-02 state. |
| BROAD | `g=1`, D2 direction, using the existing `BroadBudget` rule per utterance: seal `Q_u=Σ||edit_NEW||²` without labels/references; let `N_u` be B0's UTF8-complete eligible content queries (`t≥1`); packet norm `sqrt(Q_u/N_u)`; allocate sequentially, cap by remaining energy, debit actual realized squared energy, clamp rounding overshoot to zero, and spend nothing on invalid direction/unreachable edits. If `Q_u=0`, BROAD is exact no-edit for that utterance; if `Q_u>0` and `N_u=0`, invalidate S2. If every `Q_u=0`, zero-versus-zero BROAD energy match is valid. Otherwise require aggregate energy mismatch ≤2% and no more than 5% of positive-budget utterances individually outside 2%. |
| AUTO | Reuse an exactly compatible cached `B0_AUTO` output on all 100 IDs if one exists with verified source/model/generation hashes. Otherwise omit AUTO and record `AUTO=NOT_AVAILABLE`; do not run Whisper AUTO just for P2-SEL. |

Use existing canonical evaluators. Report corpus PIER, MER, EN-WER, ZH-CER; POI corrections and
corruptions; embedded-English and matrix-Mandarin retention against B0; outside harm under the
canonical candidate-unit definition; gate coverage, valid-direction rate, edit coverage, total
and relative realized energy; wall runtime, D2 autograd-call count, and peak allocated/reserved
VRAM. Coverage denominators are all runtime-eligible content queries (`t≥1`, UTF8-complete):
gate coverage is fraction with `g>0`, edit coverage is fraction with a nonzero realized
post-NormPreserve edit, and direction-valid rate is valid D2 direction calls divided by
positive-gate eligible queries. Realized energy is `Σ||edit||²`; relative energy divides this by
`Σ||h_pre||²` over the same eligible queries. Runtime is wall seconds per utterance; autograd
count is calls to `readout_direction`'s `autograd.grad`; peak VRAM reports maximum allocated and
reserved bytes. For each paired draw, sample the 20 dialogue clusters with replacement and
recompute each corpus ratio from the summed utterance-level numerators and denominators in that
draw, following `experiments/inference_cf_p2_evaluate.py:paired_bootstrap`; use 10,000 draws,
seed 240924, and two-sided pointwise percentile 95% intervals. For retention and outside-harm
rates, recompute numerator/denominator from the sampled utterance rows with repeated dialogue
clusters retained. This 20-dialogue/100-row panel is a
**development mini-screen**, never a final ASR result or confirmation.

S2 label precedence: `P2_SEL_MINI_INVALID` for any ID/hash/manifest/firewall/lineage/metric or
energy-audit failure; else `P2_SEL_MINI_DAMAGE_UNRESOLVED` if NEW's two-sided percentile 95%
upper bound exceeds any of these NEW-minus-B0 damage limits: MER `+0.005`, ZH-CER `+0.005`, EN-WER `+0.010`,
English retention loss `+0.010`, Mandarin retention loss `+0.010`, or outside-harm rate `0.005`;
else `P2_SEL_MINI_PROMISING` if PIER gain (B0 PIER minus NEW PIER) is at least `+0.005` and its
paired dialogue-bootstrap lower bound is `>0` versus both B0 and OLD; otherwise
`P2_SEL_MINI_INCONCLUSIVE`. These thresholds are development triage only. Even a promising
result ends this stage and can only recommend a separately frozen full 300-utterance development
confirmation; it does not authorize that run.

## 5. Provenance, audits, firewall, and compute

Before S1, a CPU pre-run audit must verify the config/spec hashes, all source hashes, the 180 exact
ordered keys and strata, P2-DIR D2 sealing/audit, P2-R gate-row joins and state identities,
reusable NONE/ungated references, C3 pooled energy formula, all thresholds/labels, and the
absence of outcomes. It emits `PASS_TO_P2_SEL_S1` or stops. It must not run model inference.

After each executed stage, an independent auditor must not import the primary analysis/decision
module. It independently recomputes the population, joined E/R_B/g, direction provenance, realized
edit energy and C3 matching, margins/corruptions, dialogue-bootstrap intervals, label precedence,
and—for S2—canonical metrics, retention, outside harm, coverage/runtime counters, and S2 label.
No scientific interpretation is released without `P2_SEL_AUDIT: PASS (S1)` or
`P2_SEL_AUDIT: PASS (S2)`. Store compact row-level scalars and hashes; do not duplicate full
vocabulary logits or high-dimensional traces already sealed by P2-DIR.

Allowed data: existing P2-R/P2-RJ/P2-DIR artifacts and the frozen D-dev-select mini panel.
Forbidden: new router-calib role, D-dev-confirm, D-test, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND,
or any transfer corpus. P3 remains held. Do not edit the core method contract or relabel P2-DIR
or historical P2 outcomes.

Use Slurm `sbatch` on the existing H100 MIG environment, batch size 1, and at most two pending or
running GPU jobs across the project. Plan one S1 job; run one S2 job only after audited S1 rescue.
Reuse frozen states, gates, directions, baselines, and canonical evaluation artifacts wherever
their hashes and semantics match. Target total P2-SEL GPU use is at most two GPU-hours. No GPU
job is part of this design freeze.

Every run writes a provenance manifest through the repository's provenance/spec-freeze helpers,
including resolved config, environment, git state, model/tokenizer/generation metadata, source
and input hashes, data role, status, counters, and audit result. Preserve all existing artifacts.
For every intentional repository edit: check, inspect the diff, commit, then push to
`origin HEAD:cs-asr-steer-inf`. Never force-push, merge `main`, or create a PR.
