# P2-SEL-E — Local English Evidence Selectivity Diagnosis and Repair

**Status:** revised pre-E0 freeze, 2026-10-06. The original freeze was blocked before E0 by
`docs/inference_cf/P2_SEL_E_PRE_RUN_AUDIT.md`; that audit remains unchanged as provenance. No E0
outcome existed before this revision. This is a bounded, exposed-development diagnostic after
P2-DIR/P2-SEL. It is not the active core method in `docs/current/METHOD_CONTRACT.md`, does not
revise that contract, and cannot support a core-method claim, confirmation result, P3, or test-set
claim. No P2-SEL-E outcome exists at freeze. Do not start E0 before this spec, config, panel
reference, and implementation pre-run audit are committed and pushed.

The question is: **Why does the frozen local-English-evidence term E produce a small damaging
ZH-correct false-positive tail while missing EN-confusion positions, and can one preselected,
reference-free repair improve that precision/recall tradeoff?** The only eligible repair is the
single branch selected by the E0 decision tree below. No direction, layer, alpha, R_B, or baseline
localizer search is allowed.

## 1. Frozen baseline and implementation facts

The starting stage is the audited P2-SEL terminal result `P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE`.
On the identical 180 positions, D2-GATED had EN-confusion `Δm=+3.520 nat`, versus ungated D2
`+4.499` (`r_benefit=0.782`); ZH-correct `Δm=-0.195` versus ungated `-2.011`
(`r_harm=0.097`). P2-SEL reported 3/60 ZH top-1 corruptions (5.4% under its dialogue-weighted
rate), all among five ZH rows with `E>0`; 18/60 EN-confusion rows had `E=0`. `R_B` is high on
both EN-confusion and ZH-correct rows, so the current selectivity bottleneck is E. These are
already-exposed P2-SEL outcomes, not P2-SEL-E outcomes.

The following remain byte/semantic-compatible with `p2dir_readout_v1` and P2-SEL:

| Component | Frozen value |
|---|---|
| Direction | D2 READOUT, `p2dir_readout_v1`, unchanged |
| Layer/site | L16, DG-02 decoder post-cross-attention residual / pre-FFN |
| Dose | `alpha=2`, `phi(g)=g` |
| Repair | NormPreserve ON; exact no-edit at zero effective gate |
| `R_B` | Existing P2-R conflict provider `R=max(0,P_M-P_E)`, unchanged |
| Baseline localizer | Existing `max_attention_window`: current-query frozen alignment-head attention, maximum-mass 50-frame (1.0 s) window, earliest tie |
| Decode/state | Existing forced-ZH B/E/S semantics and matched baseline prefixes |

`d_readout` is the existing reference-free tangent-projected gradient of
`log(P_E+1e-12)-log(P_M+1e-12)` at the current L16 B state, with the frozen tokenizer partition,
float32 logits/leaf, float64 tangent projection and normalization, and existing fallback/cache
isolation. No evaluator gradient, transcript, reference token, CTC alignment, or future token may
construct or orient D2 or `E_new`.

The frozen `native_lid` provider softmaxes over the full 100-token Whisper language vocabulary and
returns selected probabilities from that distribution. Thus `P_E` and `P_M` are absolute
100-way probabilities, `Q_local=P_E+P_M` is the total mass assigned to the relevant EN/ZH pair,
and `P_E/P_M` expresses relative EN-vs-ZH preference. Q is allowed anywhere in `[0,1]`; it is not
expected to equal 1. The already-exposed P0-R2 global pair-mass distribution (5,082 rows) has
10th percentile `0.21689272671937943`, median `0.957465011626482`, and minimum
`0.0008186694176401943`; this is the frozen scale reference for H_E3, not a statistic fitted on
the five ZH false positives.

Provider integrity is checked by: the exact frozen Whisper `generation_config.json` SHA256
`fbdfa70135de9b1d31553393f14e80aaeb1936ea36576b2ba864055943c09d23`; exactly 100 unique
`lang_to_id` entries; the canonical sorted `(language token, integer ID)` mapping SHA256
`639cd6d6fbdb9cdf9bc09708371676fbc47f98c35f068399d23b3accd9c06731`; `<|en|>`/`<|zh|>` IDs
`[50259,50260]`; and unchanged provider implementation/code hashes recorded in the config. The
returned 100 probabilities must be finite, nonnegative, and sum to 1 within absolute `1e-6`.
The same zero-waveform null input, current selected window, `epsilon=1e-12`, and `native_lid`
calls must be used. Independently recomputed current E must match P2-R/P2-SEL E within absolute
`1e-6`. No integrity check requires Q to be near 1.

## 2. E0 — diagnostic only, 180 positions

Use exactly the ordered 180 keys and frozen labels in
`results/inference_cf/p2rj/positions.json`: 60 EN-confusion, 60 EN-correct, 60 ZH-correct on its
fixed dialogue universe. Never refill, reclassify, stratify anew, or add a role. E0 applies no
intervention and computes no evaluator outcome.

### Quantities and artifact reuse

For each key, recover or recompute from the existing P2-R/P2-DIR/P2-SEL inputs:

* `pi_local(E)`, `pi_local(M)`, `pi_null(E)`, `pi_null(M)`, using the existing `native_lid` and
  exact zero-waveform null provider;
* with `eps=1e-12`, `ell_local=log((pi_local(E)+eps)/(pi_local(M)+eps))`,
  `ell_null=log((pi_null(E)+eps)/(pi_null(M)+eps))`, `A=ell_local-ell_null`, and
  `E=max(0,tanh(A/2))`; also `Q_local=pi_local(E)+pi_local(M)`;
* current 1-second localizer window start/end samples and frames, attention mass, maximum-attention
  frame, and center; record existing evaluator alignment timing error where available;
* `E_short` from exactly one diagnostic narrower crop: 0.5 s (8,000 samples at 16 kHz), centered
  on the already-selected 1-second window center. For selected bounds `[s0,s1)`, use
  `a=clamp(floor((s0+s1-8000)/2),0,heard_samples-8000)` and `[a,a+8000)`; for waveforms shorter
  than 8,000 samples use `[0,heard_samples)`. Recompute local LID and E with the unchanged null
  correction. This is diagnostic in E0, not a steering candidate;
* causal `E_(t-1)` and `E_(t-2)` from the complete existing P2-R run3 current traces, joined by
  utterance and absolute query. Missing history is zero. No future step may be read;
* existing P2-R run3 `E_oracle` and `timing_error_sec` for EN-confusion targets only. This is
  evaluator-centered diagnosis using already-exposed CTC midpoints. Do not calculate a new
  alignment. These fields are diagnostic only and must never enter `E_new` at inference.

The existing P2-R traces already store every current-step `E`, `R_B`, `g`, and 1-second window;
P2-R also stores `E_oracle`/timing error for its EN-confusion targets. They do **not** store the
four local/null probabilities, attention mass, or maximum-attention frame. Therefore E0 may use
one small GPU extraction job for those missing fields, replaying only the frozen baseline prefixes
needed for the 180 keys and reusing cached audio/state inputs where verified. Do not repeat P2-R,
P2-DIR, or P2-SEL arms. The 1-second recomputed E must match the P2-R run3 E within absolute
`1e-6`; otherwise invalidate E0. Persist compact per-position rows and source hashes only.

### Failure groups and reporting

Partition using the frozen P2-SEL scalar `E>0`:

| Group | Definition | Expected count to reproduce |
|---|---|---:|
| EN-confusion TP | EN-confusion and `E>0` | 42/60 |
| EN-confusion FN | EN-confusion and `E=0` | 18/60 |
| ZH-correct TN | ZH-correct and `E=0` | 55/60 |
| ZH-correct FP | ZH-correct and `E>0` | 5/60 |
| EN-correct | safety/reference stratum | 60 |

Any count mismatch is investigated against exact P2-SEL/P2-R key joins before diagnosis; mismatch,
duplicate keys, or missing rows is `P2_SEL_E_INVALID`. Report exact counts, the distinct dialogue
IDs and per-dialogue counts for every group, plus per-stratum/per-group median and IQR. Descriptive
uncertainty uses 10,000 shared paired dialogue bootstrap draws over the fixed 20-dialogue universe,
seed `240924`, percentile 90% intervals; resample dialogues, retaining all positions from a drawn
dialogue. For rates, compute each drawn-dialogue rate over that dialogue's rows and average
represented drawn dialogues equally; omit empty stratum cells and require 9,900 valid draws. H_E2
uses the same draws and denominator rule for its persistent-rate difference. These intervals
describe the exposed diagnostic and do not substitute for the frozen classification criteria.

### Frozen diagnostic hypotheses and numerical criteria

All criteria are applied once, in the precedence below. Minimum support means affected rows span
at least three distinct frozen dialogues. The small FP group is never used for a fitted threshold.

**H_E1 — localization/window contamination.** Define:

1. `static_scale_pattern`: at least 4/5 ZH FP have `E_short<=0.10` and
   `E_long-E_short>=0.25`, while at least 34/42 EN TP have `E_short>=0.20`; both affected groups
   span at least three dialogues.
2. `oracle_FN_pattern`: at least 9/18 EN FN have existing `E_oracle>=0.20` and
   `E_oracle-E_long>=0.20`, spanning at least three dialogues.
3. `short_only_FN_pattern`: at least 9/18 EN FN have `E_short>=0.20` while `E_long=0`, spanning
   at least three dialogues.

H_E1 passes if any one pattern passes. Report timing error on the available EN-confusion rows.
If `oracle_FN_pattern` or `short_only_FN_pattern` passes, a same-center two-scale agreement cannot
recover the zero-long-scale misses, so choose `P2_SEL_E_LOCALIZER_PRIMARY` and stop. Otherwise,
if `static_scale_pattern` passes, select the static multi-scale repair R2. H_E1 is evaluated before
all other hypotheses.

**H_E2 — temporal instability/spikes.** Let `p_t=max(E_(t-1),E_(t-2))`. A ZH FP is isolated when
`E_t>=0.50` and `p_t<=0.10`; an EN TP is persistent when `p_t>=0.25`. H_E2 passes only if at least
4/5 ZH FP are isolated across at least three dialogues, at least 26/42 EN TP are persistent
across at least three dialogues, and the one-sided 80% dialogue-bootstrap lower bound for the
paired difference (EN-TP persistent rate minus ZH-FP persistent rate) is above zero. Use the
fixed dialogue universe and seed above. The interval is exploratory support, not a confirmatory
claim.

**H_E3 — low EN/ZH pair-confidence artifact.** Freeze the low-pair-mass cutoff at the already
exposed P0-R2 global 10th percentile `q_low=0.21689272671937943`. Define a low-Q row as
`Q_local<=q_low`. H_E3 passes only if (i) at least 4/5 ZH-correct FP rows are low-Q, (ii) no more
than 10/42 EN-confusion TP rows are low-Q, (iii) the dialogue-equal low-Q prevalence difference
`rate_ZH_FP-rate_EN_TP` is at least `0.50`, and (iv) the low-Q ZH-FP rows span at least three
dialogues. In addition, its one-sided 80% dialogue-bootstrap lower bound for that prevalence
difference must be above zero. For each of 10,000 draws, sample 20 dialogue IDs with replacement
from the fixed universe using `numpy.random.default_rng(240924)`; reuse each draw for both groups.
Within each sampled draw, average each group's low-Q proportion equally over represented dialogues,
omitting a dialogue with no rows in that group. The one-sided 80% lower bound is the 20th
percentile of the shared paired draw differences; require at least 9,900 valid draws. The fixed
percentile cutoff comes only from the 5,082 already
exposed P0-R2 rows; the H_E3 threshold is not recalculated from E0 and is not fitted to these five
ZH-FP rows. Variation in global Q alone is insufficient: the FP-vs-TP enrichment and dialogue
support requirements must all pass.

**H_E4 — null-correction pathology.** For each ZH FP, define `null_share=(-ell_null)/A` when `A>0`.
H_E4 passes only if at least 4/5 ZH FP have `ell_local<=0`, `A>0`, and `null_share>=0.80`, the
affected rows span at least three dialogues, and the median `ell_local` among EN TP is positive.
Report all component distributions; do not fit a share threshold.

### Deterministic diagnosis and repair choice

1. Any integrity, provider, population, E-reproduction, or required-field failure:
   `P2_SEL_E_INVALID`, stop.
2. If H_E1 passes, apply its internal rule above: localizer-primary stop for either FN pattern;
   otherwise static-scale pattern selects R2 and no other repair.
3. Else if H_E2 passes, select R1 and no other repair.
4. Else if H_E3 passes, select R3 and no other repair.
5. Else if H_E4 passes, select R4 and no other repair.
6. Else `P2_SEL_E_DIAGNOSIS_AMBIGUOUS`, stop.

Exactly one repair may be selected. A stopped or ambiguous diagnosis launches no E1 GPU job and
no extra diagnostic, alignment, or localizer experiment. If a single repair is selected, the
following formula is frozen for that branch:

| Selected branch | Exact `E_new,t` | Notes |
|---|---|---|
| H_E1 static scale → R2 | `sqrt(E_long,t * E_short,t)` | Uses only current-query support and the two frozen same-center windows. |
| H_E2 → R1 | `E_t * max(E_(t-1),E_(t-2))` | Missing causal history is zero; no future evidence. |
| H_E3 → R3 | `E_new,t=E_t*Q_local,t` | Exact reference-free pair-mass weighting; no threshold or fitted coefficient. |
| H_E4 → R4 | `max(0,tanh((ell_local-0.5*ell_null)/2))` | Fixed 50% shrink of the current null subtraction; no coefficient fitting. |
| H_E1 FN pattern → R5 | no formula; `P2_SEL_E_LOCALIZER_PRIMARY`, stop | A future localizer stage requires its own freeze. |
| Ambiguous | no formula; `P2_SEL_E_DIAGNOSIS_AMBIGUOUS`, stop | No fallback repair. |

All `E_new` inputs are reference-free runtime quantities. For R3, `Q_local` is the sum of the
current 100-way probabilities at the frozen EN/ZH language IDs. Since `0<=Q_local<=1` and
`0<=E<=1`, require `0<=E_new<=E<=1` within `1e-7`; any violation is an integrity failure. Oracle
timing, strata, references, evaluator margins, labels, and outcomes are prohibited from formula
construction.

## 3. E1 — one repair mechanism screen

E1 runs only after E0 selects exactly one of R1/R2/R4 and the pre-E1 audit passes. Use the same
180 keys, frozen B states/prefixes, D2, R_B, alpha, layer, site, and NormPreserve. Arms are exactly
`C0 NONE`, `C_old=E_old*R_B*D2`, and `C_new=E_new*R_B*D2`. Reuse P2-SEL C0 and C_old only when their
source manifests, per-key state identities, direction hashes, hook semantics, and evaluator
definitions pass exact audit; otherwise C_old may be rerun in the same single E1 job. Execute only
the C_new pulse when reusable rows pass. Do not run another direction, BROAD, gate, or layer arm.

### Metrics and uncertainty

Primary/safety metrics are dialogue-equal mean `Δm` on EN-confusion, EN-correct, and ZH-correct;
top-1 corruption on both correct strata; per-stratum E_new-positive rate and realized squared
post-NormPreserve edit energy; and paired ratios
`benefit_retention=Δm_conf,new/Δm_conf,old` and
`harm_ratio=abs(Δm_ZH,new)/abs(Δm_ZH,old)`. Also report outcomes on the historical 5 ZH-FP and
18 EN-FN rows, descriptively only. A zero/nonfinite old denominator invalidates E1.

Use 10,000 shared paired dialogue-bootstrap draws, 20 frozen dialogue clusters with replacement,
seed `240924`, percentile intervals, and at least 9,900 valid draws. The six decision quantities
are new EN-confusion margin, new EN-correct margin, new ZH-correct margin, paired new-minus-old
ZH margin, EN-correct corruption rate, and ZH-correct corruption rate. Use two-sided 90%
Bonferroni simultaneous intervals (`alpha=0.10/6`). This is an exposed-development screen, not a
confirmatory claim.

### Frozen E1 thresholds and labels

Define `old_conf=+3.520 nat` and `old_zh=-0.195 nat` from the audited matched P2-SEL C2 rows.
E1 is supported only if all of these hold:

1. Benefit retention point ratio is at least `0.70` (new EN-confusion `Δm>=+2.464 nat`) and its
   simultaneous lower confidence bound is above zero.
2. ZH harm ratio is at most `0.50` and paired `Δm_ZH,new-Δm_ZH,old` is at least `+0.10 nat`,
   with simultaneous lower confidence bound above zero.
3. For each correct stratum, observed corruption is at most `0.05` and simultaneous upper bound
   at most `0.10`; EN-correct and ZH-correct margin lower bounds are each at least `-0.25 nat`.
4. All 180 keys, frozen-provider identities, zero-gate exact no-edit rows, and independent audit
   are valid.

The observed corruption cap remains the P2-SEL 5% cap. The 90% familywise uncertainty screen
uses a 10% upper bound, explicitly as a development-only tolerance for 60 positions clustered
within dialogues; it is not a claim that 10% damage is acceptable at deployment or confirmation.
No bound is changed after outcomes.

Apply exact precedence:

1. Integrity, invalid denominator, fewer than 9,900 draws, or audit failure:
   `P2_SEL_E_INVALID`.
2. Either correct-state margin/corruption safety rule fails:
   `P2_SEL_E_REPAIR_STILL_UNSAFE`.
3. Safety passes but benefit-retention point ratio is below `0.70`:
   `P2_SEL_E_REPAIR_TOO_CONSERVATIVE`.
4. Safety and point benefit retention pass, but any benefit-CI, harm-ratio, paired-ZH-improvement,
   or paired-CI criterion fails: `P2_SEL_E_NO_MATERIAL_GAIN`.
5. All four success rules pass: `P2_SEL_E_REPAIR_SUPPORTED`.

Only `P2_SEL_E_REPAIR_SUPPORTED` permits the conditional mini decode. Every other E1 label stops.

## 4. E2 — conditional development mini decode

E2 runs only after `P2_SEL_E_REPAIR_SUPPORTED` and `P2_SEL_E_AUDIT: PASS (E1)`. Reuse exactly the
100 stable IDs in `docs/inference_cf/P2_SEL_MINI_PANEL.json`, panel SHA256
`266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`; do not reseat, refill, or
select another panel. Selection remains the already-frozen P2-SEL deterministic hash algorithm.
Only already-exposed D-dev-select is allowed.

Systems:

| System | Definition |
|---|---|
| B0 | Matched no-edit forced-ZH cached baseline; reuse only with exact IDs/model/prompt/generation/path hashes. |
| OLD-E + D2 | Existing `E_old*R_B` gate and unchanged D2 READOUT, alpha 2, frozen site/repair. |
| NEW-E + D2 | Selected E_new formula, same R_B/D2/alpha/site/repair. |

No OLD direction arm is added. AUTO is omitted unless an exactly compatible cached result exists;
do not run AUTO solely for this stage. No BROAD system is required unless the existing frozen
P2-SEL implementation has it essentially free without new decode work; it is excluded from the
primary E2 comparison.

Report canonical PIER, MER, EN-WER, ZH-CER, corrections/corruptions, EN/ZH retention, outside harm,
gate/edit coverage, realized and relative energy, runtime, and peak VRAM. Use the existing
dialogue bootstrap and canonical evaluators. E2 is descriptive development evidence only: it has
no pass label, cannot authorize full 300-utterance decoding, and even a favorable result stops
this stage and recommends a separately frozen confirmation plan.

## 5. Engineering, tests, audit, data, and compute

Reuse `src/csasr/inference_cf/core_r2.py` (`max_attention_window`, `local_support`,
`conflict_from_logits`), `experiments/inference_cf_p2r.py` current trace/oracle fields and pulse
mechanics, `experiments/inference_cf_p2dir.py`/sealed P2-DIR direction and evaluator artifacts,
`experiments/inference_cf_p2sel.py` population and state joins, P2-SEL bootstrap/auditor patterns,
`experiments/inference_cf_cached.py` B/E/S cache path, and canonical metric functions. Do not fork
or modify the historical R2 formulas.

Focused test contract: exact old E/D2/R_B regression; decomposition math; frozen 100-way provider
ID/mapping identity; Q=`P_E+P_M` with values allowed throughout `[0,1]`; current E reproduces
P2-R/P2-SEL within `1e-6`; exact R3 `E_new=E*Q` and `0<=E_new<=E`; causal-neighbor indexing/no-
future access; exact same-center short-window bounds and agreement formula; null-shrink formula;
reference/oracle inputs absent from every E_new provider; zero E_new is exact no-edit; the 180-key
identity; deterministic H_E3 criterion and diagnosis precedence H_E1→H_E2→H_E3→H_E4; exactly-one/
no-fallback branch; source/panel hashes; E1 label precedence; reuse compatibility; conditional
mini-panel identity.
The independent auditor must not import the primary analysis/decision module. Pre-E0 and pre-E1
audits verify frozen inputs/hashes/population/firewall before permitting any job. Post-E0 audit
independently recomputes decomposition, E reproduction, group counts, causal joins, hypotheses,
precedence, and selected formula. Post-E1 audit independently recomputes pulse energy, margins,
corruption, bootstrap, and label. No conclusion is reportable without `P2_SEL_E_AUDIT: PASS`
for that stage.

Data allowed: already-exposed D-dev-select, P2-R/P2-RJ/P2-DIR/P2-SEL artifacts, and the existing
P2-SEL mini panel. Forbidden: router-calib as a new role, D-dev-confirm, D-test, P3, SEAME,
CS-FLEURS, ViMedCSS, ASCEND, Qwen transfer, or any fresh validation. Evaluator/CTC timing is
diagnostic-only in E0; it never enters runtime E_new.

Compute ceiling: E0 is CPU/artifact analysis plus at most one small MIG GPU job to extract missing
local/null probabilities and attention scalars; E1 at most one MIG GPU job; E2 at most one
conditional MIG GPU job. Submit via `sbatch`, at most two pending/running jobs total. No jobs
launch in the design session. Reuse validated rows; do not persist high-dimensional tensors when
compact scalar rows plus hashes suffice.

## 6. Required Claude execution order and stopping rules

1. Verify this freeze/config, source hashes, P2-SEL terminal report/audits, zero E-stage outcome,
   exact 180 keys, and mini-panel hash. Run CPU pre-E0 auditor; stop unless it passes.
2. Run E0 diagnostics only. Run independent E0 auditor and apply the one frozen precedence.
   `INVALID`, `LOCALIZER_PRIMARY`, or `DIAGNOSIS_AMBIGUOUS` stops the stage; no repair GPU job.
3. If exactly R1/R2/R4 was selected, run only its C_new pulse plus reusable exact C0/C_old arms.
   Run independent E1 audit and frozen label once. Stop unless `REPAIR_SUPPORTED`.
4. Only after E1 support and audit pass, run one E2 mini decode on the frozen 100 IDs and report
   development metrics. Stop regardless of result. Do not launch full 300, P3, or confirmation.

Any implementation mismatch against the active `METHOD_CONTRACT.md`, protected source identity,
or data firewall is an implementation gap: record it and stop before outcomes. No project method
contract, historical P2 result, P2-DIR result, or P2-SEL result is edited by this stage.
