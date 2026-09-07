# METHOD CONTRACT — CS-ASR Selective Test-Time Steering

**Status:** Stage 1 contract; repository-consistency audit completed 2026-09-07. This file is the scientific authority for
implementation. Where it conflicts with existing code, **the code is wrong until this
file is revised by a human**, not the reverse.

**Scientific source of record:** `docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`
(the method-first proposal, v5, 10 Aug 2026), refined by the finalized-contract vocabulary
frozen below. Planning detail:
`docs/proposal_arr/CS_ASR_ARR_October_2026_Implementation_Plan_v2.md`; latest execution framing:
`docs/proposal_arr/ROUND_1_3_EXEC_PLAN.md` and
`docs/proposal_arr/ROUND_1_3_IMPLEMENTATION_STATUS.md`.

Legend:
- **`OPEN DECISION`** — a choice this contract does not yet fix; needs human judgment.
- **`IMPLEMENTATION GAP`** — the desired method here differs from what the code currently does.

The proposal is **not reproduced** here. This file locks only the invariants downstream
code must not silently break.

---

## 1. Intervention site (LOCKED)

The single frozen intervention site is the **decoder post-cross-attention residual, before
the feed-forward (FFN) block**.

In `transformers` `WhisperDecoderLayer.forward`, the site is the tensor named `site` below:

```
residual = hidden_states
hidden_states = self.encoder_attn_layer_norm(hidden_states)
hidden_states, cross_attn_weights = self.encoder_attn(...)
hidden_states = dropout(hidden_states)            # identity under eval()
site = residual + hidden_states                   # <-- THE SITE (pre-FFN)
```

- Canonical implementation: `src/csasr/lss/sites.py`
  (`DECODER_TENSOR = "decoder_post_cross_attn_residual"`,
  `DecoderPostCrossAttnRecorder`, `DecoderPostCrossAttnSteeringHook`).
- The site is reconstructed from two hooks: a forward **pre-hook** on `encoder_attn_layer_norm`
  (captures `residual`) and a forward hook on `encoder_attn` (captures the attention output).
  Intervening from the `encoder_attn` hook returns `attn_out + (steer(site) − site)`, so the
  layer itself recomputes `steer(site)` with no assumption about downstream FFN behaviour.
- The site identity holds **only under `model.eval()` with dropout inactive**
  (`assert_dropout_disabled`). This is a required precondition, not an assumption.
- **Rejected site:** the whole-`WhisperDecoderLayer` output (post-FFN),
  `DECODER_REJECTED_TENSOR = "decoder_block_output"`, used by the legacy
  `csasr.models.hooks.DecoderSteeringHook`. Constructing a direction in one space and adding
  it in another is a scientific error, not a naming difference.
- **No depth rescale.** The frozen hook does **not** divide strength by `sqrt(num_layers)`.
  The legacy `DecoderSteeringHook` does (`hooks.py:170`, `eff_alpha = alpha / num_layers**0.5`);
  that makes `ρ=1` mean different things at different depths. `IMPLEMENTATION GAP`: any pipeline
  still calling `DecoderSteeringHook` is on the rejected site with the rejected scaling.

`IMPLEMENTATION GAP`: the exact-site recorder and hook exist, but no current free-decoding
Round-1 runner uses them. `experiments/round1_frozen.py` imports
`experiments.job_a_frozen.NormPreserveDecoderHook`, and Job B uses `T1Hook` / `T2Hook`; all attach
to the output of a whole decoder layer (post-FFN). `steer_sweep.hooks.DecoderSteering` does the
same. Their cache, beam, and forced-prefix handling therefore does **not** establish those
properties for `DecoderPostCrossAttnSteeringHook`.

Required free-decoding semantics for the future exact-site integration: derive the absolute decode
position from the KV-cache length; assign zero gain to all forced-prefix positions; apply each
source item's gate to every expanded beam; and keep the same layer index/direction/scale across
prefill, cached steps, and beam branches. The existing post-FFN runners implement versions of this
bookkeeping, but it has not been implemented or tested at the exact site.

**Encoder site scope.** The proposal also defines an encoder-frame steering path (`Δ^E`,
`src/csasr/models/hooks.py:EncoderSteeringHook`, re-exported by
`csasr.steering.encoder_hook`) and names E-only as the *safe practical
default* (v5 §2.8). This contract locks the **decoder post-cross-attention site** as the site
for the tensors, directions, scores, and gate defined below. `OPEN DECISION`: whether the
deployable practical system is E-only (v5 default), decoder-site, or E+D — resolve on
development evidence, not by which code exists.

---

## 2. Site tensors (LOCKED notation)

At decoder layer `ℓ` and decode step `t`, on the site of §1:

| Symbol | Meaning |
|---|---|
| `q_{ℓ,t}` | the **pre-intervention** site residual (`residual_in + attn_out`), the query/state the FFN would otherwise consume |
| `\widetilde u^S_{ℓ,t}` | the **nominal rank-one update** before repair: `\widetilde u^S_{ℓ,t} = α · s_ℓ · g_{ℓ,t} · d_ℓ` (see §4–§6) |
| `r_{ℓ,t}` | the **steered residual after repair**: `r_{ℓ,t} = Repair(q_{ℓ,t}, q_{ℓ,t} + \widetilde u^S_{ℓ,t})` (§7) |
| `u^S_{ℓ,t}` | the **effective update after repair**, `u^S_{ℓ,t} = r_{ℓ,t} - q_{ℓ,t}`, so the identity `r_{ℓ,t} = q_{ℓ,t} + u^S_{ℓ,t}` remains exact |

`s_ℓ` is the per-layer activation scale (from the accumulated site statistics), `α` the
strength, `g_{ℓ,t}` the gate, `d_ℓ` the unit-norm direction at layer `ℓ`.

Invariants:
- Zero-gain positions must return **bit-identical** `q_{ℓ,t}` (verified: `apply_steering` returns
  the input tensor where `gain == 0`).
- `α = 0` is a hard no-op (returns the input object).
- The direction is added in the **same space it was constructed in** (§1).

---

## 3. Candidate layers (LOCKED)

The **only** candidate decoder layers for the site of §1 are:

> **layer 16 and layer 24** (decoder, 0-indexed).

`IMPLEMENTATION GAP`: `configs/lss/spec.yaml` still lists `decoder_candidate_layers: [8, 16, 24, 31]`
(and `encoder_candidate_layers: [15, 23, 27, 31]`). The contract narrows the decoder set to
`{16, 24}`. Layers 8 and 31 remain in code as historical candidates; they are **not** candidate
layers under this contract. One layer is selected per site on `D-dev-select` only (§8, §11).
The current Round-1 wrapper is further out of contract:
`steer_sweep.rounds.ROUND1_LAYERS = (8, 12, 16, 20, 22, 24, 26)` and
`configs/rounds/round1_frozen.yaml` repeats those seven layers. The delegated Job-B trainer is
fixed at layer 24 rather than enumerating `{16, 24}`.

---

## 4. Directions (LOCKED definitions)

All directions live at the site of §1, at a candidate layer of §3, are **rank one**, and are
frozen before any confirmatory run. Construction data is `D-construct` only (§ DATA_EXPOSURE).

| Name | Definition | Notes |
|---|---|---|
| **Raw direction** `Δ^raw_ℓ` | paired mean of within-utterance contrasts `d_i = z^{postCA}(first EN BPE after ZH prefix) − z^{postCA}(matched ZH continuation)`, under the same normal prompt `p₀`, over baseline-correct English construction units | v5 §2.3 `d^{D,nat}` / `Δ`; nuisance-residualized, conversation-balanced, before conditioning removal |
| **Conditioning direction / subspace** `U^cond_ℓ` | the low-rank **language-prompt-induced** subspace estimated from forced-English vs forced-Mandarin runs | ≡ v5's prompt subspace `U^{prompt}`; "conditioning" = the utterance-level language prompt's effect |
| **Conditioning-residualized direction** `Δ^⊥_ℓ` | `Δ^⊥_ℓ = (I − U^cond_ℓ U^{cond⊤}_ℓ) Δ^raw_ℓ`, then unit-normalized | v5 §2.3 `Δ^{D⊥}`; the **primary** steering direction. Prevents a local decoder effect from being explained as another global language-prompt shift |
| **Conditioning-only direction** `U^cond_ℓ[:,1]` | the leading conditioning/prompt-subspace component itself, not residualized | Ablation baseline for §B-E2; measures the prompt-shift effect in isolation |
| **Legacy direction** `Δ^legacy` (codebase: `v_nat`) | any direction constructed for / added at the **rejected post-FFN site** (`decoder_block_output`) or via the depth-rescaled `DecoderSteeringHook`, and the pre-narrowing encoder `Δ^E` | Preserved, not deleted; **not** on this contract's critical path. The legacy `v_nat` must not be silently relabeled as the conditioning-residualized local direction `Δ^⊥`; they differ in site, conditioning treatment, and construction details |

Repository mapping: exact-site decoder contrasts are implemented by
`src/csasr/experiments/v2r3_directions.py:site_d_contrasts` using
`DecoderPostCrossAttnRecorder`. `src/csasr/directions/decoder.py` instead records whole decoder
block outputs and is legacy for this contract. The v2r3 implementation also provides
`orthonormal_basis`, `project_out`, and `assemble`; there is no dedicated canonical
conditioning-residualization module.

`IMPLEMENTATION GAP`: the v2r3 assembly projects the conditioning basis out of each contrast
**before** nuisance residualization, clipping, and dialogue-balanced averaging. The frozen
definition above residualizes/aggregates `Δ^raw` and then projects that direction. These orders
must not be treated as equivalent without an explicit scientific decision.

Report at freeze time: `cos(Δ^raw, Δ^⊥)`, energy fraction removed by conditioning
residualization, and speaker/dialogue-bootstrap cosine stability
(observed historically ≈0.99 cosine, 2–5% energy removed — `docs/RESULTS_RUN_7DAYS.md`).

`OPEN DECISION`: rank of `U^cond` (prompt-subspace dimension) and the estimator for it.
`OPEN DECISION`: whether the primary trained direction (Job B, T1) is rank-1 or rank-2 — the
proposal fixes rank-one; Job B currently trains **rank-2 local steering** (`IMPLEMENTATION GAP`
vs v5 §2.3 "rank one only"). Resolve explicitly.

---

## 5. Candidate scores and disagreement hypothesis (LOCKED roles; exact algebra partly open)

Three scores describe a candidate region/position. They are computed from
**inference-available** signals only — the frozen allowlist is
`csasr.lss.features_contract.FEATURE_ALLOWLIST` (`ALLOWLIST_VERSION = "lss_features_v1"`),
guarded by `assert_inference_safe`, which rejects reference/alignment-derived columns.

| Score | What it measures | Backing features (allowlist) |
|---|---|---|
| **Source score** | acoustic/**encoder-source** evidence that this region is embedded English | `localizer_score_max`, `localizer_score_mean`, `encoder_en_margin` (projection of pooled span states on the frozen direction — defined even for deletions), `cross_attention_concentration` |
| **Decoder score** | the **decoder's own** (un)certainty about the first-pass token here | `token_confidence_min`, `token_entropy_max`, `token_top2_margin_min`, `first_pass_token_is_latin` (all missing for deletions → `NaN` + `_missing` flag) |
| **Disagreement score** | whether the source and the decoder **disagree** about the language | `encoder_decoder_disagreement` (encoder margin sign vs decoded token language) |

`IMPLEMENTATION GAP`: the code exposes these as **individual selector features**, not as three
named aggregate scores. The named source/decoder/disagreement scores are the contract's grouping;
the aggregation function of features → each score is `OPEN DECISION`.

### Disagreement hypothesis (conditional)

The **encoder–decoder disagreement** is a conditional mechanistic hypothesis, not an established
fact. Define:

- `s_t^S = logit P(E | u_t^S)` — source-side English evidence from encoder/acoustic features,
- `s_t^Q = logit P(E | q_t)` — decoder-side English evidence from decoder state,
- `δ_t = s_t^S − s_t^Q` — signed disagreement.

The hypothesis is that `δ_t > 0` (encoder says English, decoder does not) identifies positions
where steering has corrective leverage beyond what decoder uncertainty alone predicts.

**Decision rule.** If disagreement provides held-out incremental repairability-prediction value
beyond decoder-uncertainty features alone (measured on `router-calib` as AUROC/calibration lift
in the utility selector), retain it as a gating input. Otherwise, report the mechanistic
negative result and fall back to a retention-aware gate that uses only source and decoder scores
without the disagreement factor (§6 fallback).

---

## 6. Factorized gate (LOCKED structure; exact form open)

The gate `g_{ℓ,t} ∈ [0,1]` is **factorized** — a product of independent factors, not one
monolithic score — so each factor can be ablated and audited separately. The proposal's operating
form (v5 §2.6) is:

```
g_t = 1[t ∈ S_{k*}] · m_t · 1[ p̂_{k*} ≥ τ ]
```

- `1[t ∈ S_{k*}]` — the selected candidate span (localization),
- `m_t` — the soft localizer score inside the span (§5 source side),
- `1[p̂ ≥ τ]` — the **utility-selector acceptance** at abstention threshold `τ`, where
  `p̂ = f_util(source, decoder, disagreement scores …) ≈ P(net utility > 0)`.

Contract: the gate is the product of a **localization factor**, a **source factor**, and a
**utility/acceptance factor** built from the §5 scores; steering fires only when all factors are
nonzero; abstention keeps the first-pass transcript.

`OPEN DECISION`: the exact functional form of the factorized gate — whether the decoder and
disagreement scores enter multiplicatively as their own factors or only through `p̂`, and how the
acoustic→decoder transport `r_q = Σ_t Ā_{q,t} g_t` (v5 §2.7) composes with the factors for the
decoder site. `OPEN DECISION`: `τ` is set on `router-calib` only, then swept for the frontier.

`IMPLEMENTATION GAP`: no current runner implements this factorized gate or the outcome-supervised
utility selector. The prior Job-A F5 gate is one sigmoid of a post-FFN `v_nat` projection, with
its median/scale estimated from teacher-forced reference language labels on `D-dev-select`.
Job-B's logistic regression is a diagnostic EN/ZH probe, not `f_util`; its T1 gate is likewise a
monolithic two-projection sigmoid. A trained temporal localizer is also absent.

**Legacy `v_nat` identity.** The codebase name `v_nat` (`steer_sweep/trackb/`,
`experiments/job_b_training.py`, `experiments/round1_frozen.py`) refers to the "natural"
within-utterance decoder direction constructed at the **rejected post-FFN site** under legacy
procedures. It is **not** the same as the conditioning-residualized direction `Δ^⊥` defined in
§4, which lives at the exact post-cross-attention site and has had conditioning projected out.
Code that reads `v_nat` from the direction store is legacy.

**Fallback.** If the disagreement hypothesis (§5) fails on held-out data, the gate falls back
to a retention-aware design using only source and decoder scores — a product of the localization
factor and a simplified acceptance factor `1[p̂_{no-disagree} ≥ τ]` that omits the disagreement
input. This fallback preserves the factorized structure and the ablation requirement.

---

## 7. Norm-preserving repair (LOCKED)

After adding the update, the steered residual is renormalized to the pre-intervention norm:

```
\bar r_{ℓ,t} = q_{ℓ,t} + \widetilde u^S_{ℓ,t}
if norm_preserve:  r_{ℓ,t} = \bar r_{ℓ,t} · ( ‖q_{ℓ,t}‖ / (‖\bar r_{ℓ,t}‖ + EPS) )
else:              r_{ℓ,t} = \bar r_{ℓ,t}
u^S_{ℓ,t} = r_{ℓ,t} - q_{ℓ,t}
```

- Canonical implementation: `csasr.models.hooks.apply_steering(..., norm_preserve=True)`
  (`hooks.py:72`), used by both `sites.DecoderPostCrossAttnSteeringHook` and
  `steering.EncoderSteeringHook`.
- Invariants (§2): zero-gain positions bit-identical; `α=0` no-op.
- The frozen decoder-site hook (`sites.py`) applies norm preservation **without** any
  `sqrt(num_layers)` rescale (§1). `NormPreserve` must be applied identically across the proposed
  system and every steering control (matched-energy, random, sign-flip) so comparisons change the
  direction/location, not the energy.
- `IMPLEMENTATION GAP`: Round-1 frozen/training runners reimplement norm repair at the rejected
  post-FFN site and do not use the contract scale `s_ℓ = projection_std`; the frozen wrapper's
  reusable hook defaults to mean activation norm, while the prior Job-A/B drivers use unit scale.

---

## 8. Training sets and losses (LOCKED roles; loss algebra partly open)

Applies to the trained decoder-site direction (Job B / T1, `experiments/job_b_training.py`).

- **Correction set:** baseline-**incorrect** embedded-English lexical units on training data —
  the units steering must repair. `E_EN = { u : Ŷ_u ≠ Y*_u }` (v5 §3.4).
- **Retention set:** baseline-**correct** content that must be preserved — correct embedded-English
  units `C_EN = { u : Ŷ_u = Y*_u }`, neighbouring Mandarin, and monolingual material.
- **Correction loss (contract):** token cross-entropy toward the reference on the correction set.
  `IMPLEMENTATION GAP`: `experiments/job_b_training.py:1016` applies `F.cross_entropy` to every
  non-prefix reference token in its selected utterances. `cs_ce_val` is diagnostic only; neither
  is the contract's correction-set-only objective.
- **Retention populations.** Three retention groups are distinguished:
  - **English retention:** baseline-correct embedded-English units `C_EN` — corruption here is
    the primary harm measure.
  - **Mandarin retention:** matrix-language (Mandarin) material near the intervention site —
    degradation measured as new Mandarin errors.
  - **Monolingual retention:** monolingual Mandarin and English utterances outside any CS region.
- **Retention enforcement:** currently enforced by (a) **norm-preserving repair** (§7) and (b)
  measurement via `steer_sweep.metrics.corruption_and_retention` (`zh_retention`, corruption
  counts), **not** by an explicit retention loss term.
- `OPEN DECISION`: **staged training order.** The proposal does not prescribe a fixed training
  curriculum. A staged order (1. correction only → 2. correction + Mandarin retention →
  3. correction + Mandarin + English retention → 4. gate-budget penalty if coverage exceeds a
  ceiling) is one candidate schedule. An alternative is joint correction+retention from the start
  with loss weighting. This choice is deferred to the training ticket; it does not block DG-01
  metric canonicalization.
- `IMPLEMENTATION GAP` / `OPEN DECISION`: whether an explicit retention/anchor loss (e.g.
  KL-to-baseline on the retention set) is part of the contract, and its weight relative to the
  correction loss.
- **Data matching:** trained directions and any LoRA baseline are trained on **exactly**
  `loc-train ∪ util-train` (`lora-router-matched` view in `configs/lss/roles.yaml`), never on
  `D-dev-confirm` or `D-test` (v5 §6-Exp4).

`IMPLEMENTATION GAP`: the submitted Job-B wrapper delegates the **prior fixed layer-24 trainer**;
the full seven-layer / T1 / GlobalDecoder / SALSA-E / LoRA enumeration is not yet wired
(`docs/proposal_arr/ROUND_1_3_IMPLEMENTATION_STATUS.md`). Under §3 only layers 16 and 24 are
candidates regardless.

---

## 9. Teacher-forced vs free-decoding claims (LOCKED boundary)

These are **different evidence classes** and must never be conflated.

- **Teacher-forced** (gold tokens fixed): fast diagnostic screening only — gold-token logprob
  margin, top-1, rank. `steer_sweep.teacher_forced_sanity`, Job-A/B diagnostics.
- **Free-decoding** (the model generates): the only evidence for **transcript** claims — PIER, MER,
  correction/corruption, retention.

Rules:
- The **action `a*` is selected from free-decoding utility**, never from teacher-forced margin
  (v5 §5.1).
- A teacher-forced gold-token improvement is **not** a transcript improvement. Historically a
  +0.106 teacher-forced logprob shift produced net-negative transcript change (7 corrections /
  13 corruptions — `docs/RESULTS_RUN_7DAYS.md`).
- Gate B (§10) is evaluated on **free-decoding net utility**.

---

## 10. Scientific decision gates (LOCKED)

From the proposal (v5 §8.1). A gate that fails is a **result**, not a licence to relax the
threshold. Code status semantics: `passed` / `completed_no_go` (ran, answer is no) / `blocked`
(evidence unavailable) / `failed` (implementation defect) — `csasr.lss.gates`,
`csasr.utils.status`. Only `passed` satisfies a downstream prerequisite.

| Gate | Question | Pass condition |
|---|---|---|
| **A — Alignment** | Are local frame labels trustworthy? | coverage ≥0.95, invalid/nonmonotonic ≤0.01, ≥90% units usable, median boundary err ≤100 ms, p90 ≤200 ms, class-asymmetry & ±50/±100 ms jitter stable |
| **B — Oracle headroom** | Does a fixed local action repair errors? | **free-decoding** net utility >0 with conversation-block bootstrap 95% CI excluding 0; corrections > corruptions; correct direction beats label-permuted, matched-random, sign-flip, wrong-location nulls in paired bootstrap |
| **T — Decoder transport** | Can the acoustic gate reach decoder steps? | ≥60% practical coverage with useful target-step concentration; else use E-only |
| **C — Localizer recall** | Do candidates include repairable spans? | ≥70% recall of oracle-correctable spans at manageable candidate load |
| **D — Utility value** | Does outcome supervision beat steer-all / uncertainty? | better net utility at nontrivial coverage on development data |
| **E — Automatic method** | Does predicted steering retain oracle gain at low harm? | positive development PIER gain; corrections > corruptions |
| **F — Method comparison** | Competitive with LoRA / global steering? | non-dominated on correction-vs-harm or preservation |

Utility (v5 §2.5): `U_k(a*) = N_corrected − η·N_local_harm − κ·N_outside_harm`, primary
`η = κ = 1`; 0.5 and 2 reported as sensitivity only, never retuned after seeing outcomes.
Harm is a **correctness flip** (correct→incorrect) against the reference, not any transcript
difference (`csasr.lss.outcomes`). Under the canonical candidate accounting, correction is an
incorrect→correct flip on a reference unit inside the candidate; local harm is a correct→incorrect
flip inside; outside harm is a correct→incorrect flip on a trusted unit outside. Inside/outside is
assigned by acoustic overlap (default unit-overlap threshold 0.5), untrusted boundaries are
`unknown`, and new insertions are reported but excluded from primary utility by default.

**Metric sign convention.** PIER, MER, corruption, and harm rates are error quantities (lower is
better). Baseline-to-method transcript edit fields measure change magnitude only: smaller means
fewer changes, not necessarily a better transcript. The contract's improvement quantity is
`PIER_gain = PIER_baseline − PIER_method` (positive is better), matching
`steer_sweep.metrics.delta_pier`. `IMPLEMENTATION GAP`:
`steer_sweep.round1_metrics.paired_metric_report` emits `delta_PIER` and `delta_MER` as
`method − baseline` (negative is better). These fields must not be combined until Ticket 2
renames/re-signs them. Corrections are incorrect→correct; corruptions/local harm/outside harm are
correct→incorrect. `evaluation.correction_harm.outside_region_edits` counts baseline-to-method
projection differences outside a ±1-reference-unit target radius; Round-1 `outside_edit_rate`
instead stores whole-transcript baseline-to-method edit distance per utterance. Neither field is
outside harm.

---

## 11. Safe-claim boundaries (LOCKED)

The paper / results must **not** claim:
- that the system is training-free (backbone frozen, but localizer & selector are trained);
- that the direction is a pure or unique "language variable";
- that every embedded-English span needs steering;
- that a frame-level LID gain is ASR correction;
- that a teacher-forced gold-token improvement is a transcript improvement (§9);
- that successful intervention proves the unmodified model normally uses this direction;
- multilingual/architecture generality from one model and one language pair.

Positive framing that is safe:
- **Oracle-span steering is an upper bound**, not a deployable system.
- Development results choose the primary vs fallback claim; the pipeline is frozen **before**
  `D-test`; the primary claim is never chosen after seeing test outcomes.

---

## 12. Open decisions requiring human judgment (index)

1. Practical deployable site: E-only vs decoder-site vs E+D (§1).
2. Rank of the conditioning subspace `U^cond` and its estimator (§4).
3. Rank of the primary trained direction: rank-1 (proposal) vs rank-2 (current Job B) (§4, §8).
4. Feature→score aggregation for source/decoder/disagreement (§5).
5. Exact factorized-gate algebra and transport composition for the decoder site (§6).
6. Whether an explicit retention loss term exists and its weight (§8).
7. Staged training order: correction-only first vs joint correction+retention (§8).
8. How dialogue-v2 `router-calib` is divided into disjoint probability-calibration and
   threshold-selection roles; the current dialogue-v2 config does not define that subdivision.

None of these open decisions blocks DG-01 metric canonicalization. Items 4–8 are deferred to the
training and gate tickets (DG-03+). Items 1–3 are resolved on development evidence before
confirmation.

## 13. Implementation gaps (index)

1. Candidate-layer configurations and Round-1 enumeration disagree with `{16,24}` (§3).
2. Current free-decoding runners steer post-FFN block outputs; no exact-site runner yet carries
   absolute cache position, forced-prefix exclusion, and beam expansion (§1).
3. Round-1 runners do not use the contract's `projection_std` scale (§2, §7).
4. Conditioning residualization exists only in reusable v2r3 code and its operation order differs
   from the frozen definition (§4).
5. The factorized gate, trained temporal localizer, and outcome-supervised utility selector are
   not implemented (§5–§6).
6. Job-B delegates a fixed layer-24, rank-2, post-FFN trainer and uses all-token CE rather than
   correction-set-only CE; the full method/baseline enumeration is not wired (§4, §8).
7. Retention is measured and norm-repaired but has no explicit loss (§8).
8. Metric surfaces use different denominators/accounting and opposite delta signs (this section;
   CODE_MAP §3) — canonicalization is the next ticket.
9. Current Round-1 result directories contain cell/protocol records but not the full
   config/environment/git/model/hash manifest required by AGENTS.md.
10. Dialogue-v2 `router-calib` exists, but no `calib-prob`/`calib-thresh` sub-roles are configured
    for the finalized gate (§6; DATA_EXPOSURE).
