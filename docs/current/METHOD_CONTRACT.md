# METHOD CONTRACT — CS-ASR Selective Test-Time Steering

**Status:** Stage-1 contract, **reconciled 2026-09-07 (DG-03R)** to the updated research proposal.
This file is the scientific authority for implementation. Where it conflicts with existing code,
**the code is wrong until this file is revised by a human**, not the reverse.

**Scientific source of record (post-DG-02):**
`docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v6.md` — the updated method-first
proposal (**contrastive steering basis → adaptive controller → damage-aware optimization**). **v6 is
the scientific authority; this file is its implementation/operationalization** and must not silently
contradict it. v6 supersedes, for the core paper, the disagreement/temporal-localizer/outcome-selector/
factorized-gate roadmap of `CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`, which — with
`Implementation_Plan_v1/v2.md`, `ROUND_1_3_*`, and `GATE_A_*` — is **SUPERSEDED / HISTORICAL** and
must not override v6.

Legend:
- **`OPEN DECISION`** — a choice this contract does not yet fix; needs human judgment.
- **`IMPLEMENTATION GAP`** — the desired method here differs from what the code currently does.
- **`LEGACY DESIGN`** — superseded for the core paper; preserved, off the critical path.
- **`OPTIONAL SUPPORTING ANALYSIS`** / **`NOT IN CURRENT CORE SCOPE`** — may inform, does not block.

The proposal is **not reproduced** here. This file locks only the invariants downstream
code must not silently break.

## DG-03R reconciliation summary (what changed 2026-09-07)

The core method after the frozen DG-02 site is now a **three-stage pipeline**:

1. a **contrastive steering basis** `V_ℓ^0 = [v_ℓ^{local}, v_ℓ^{cond}]` at each candidate layer (§4);
2. an **adaptive controller** `f_θ(LN(r_{ℓ,t})) → (g_t, π_t)` predicting per-token strength and a
   mixture over the basis, with **no oracle CS location at inference** (§6);
3. **damage-aware optimization** — correction CE plus KL retention on baseline-correct
   embedded/matrix positions, optional basis-refinement anchor (§8).

**Superseded for the core paper** (now `LEGACY DESIGN` / `NOT IN CURRENT CORE SCOPE`): the
separately-engineered source/decoder/disagreement scores, the encoder–decoder **disagreement
decision gate**, the temporal localizer, the outcome-supervised utility selector, the abstention
selector, and the factorized gate (§5, §6). DG-01 metrics and the DG-02 site are **unchanged**.

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

**DG-02 FROZEN.** The exact-site hook `csasr.lss.sites.DecoderPostCrossAttnInterventionHook` is
implemented, tested (CPU/synthetic matrix), and passed **real-model acceptance at L16 and L24** on
Whisper-large-v3 (Slurm job 50369; `results/dg02_real_acceptance.json`; `DG02_INTERVENTION_SITE_SPEC.md`
§18). The site, `r = q + u_source`, NormPreserve, forced-prefix exclusion, cache-position tracking,
and the per-row/per-token intervention interface are **frozen and must not be reopened**.

`IMPLEMENTATION GAP`: the legacy free-decoding runners (`experiments/round1_frozen.py` →
`job_a_frozen.NormPreserveDecoderHook`; Job B `T1Hook`/`T2Hook`; `steer_sweep.hooks.DecoderSteering`)
still attach post-FFN and are `LEGACY DESIGN`. No **scientific** exact-site free-decoding runner
exists yet for DG-03+; building one on the frozen hook is a DG-03+ implementation gap (CODE_MAP).

Required free-decoding semantics for the exact-site scientific runner: derive the absolute decode
position from `cache_position` (primary) / KV-cache length (fallback); assign zero effective edit to
all forced-prefix positions; and keep the same layer index and basis across prefill, cached steps,
and beam branches. The controller (§6) computes `(g_t, π_t)` **per row from that row's own
`r_{ℓ,t}`** — no oracle CS location and no source-item→beam gate transport are required at
inference (that transport is `NOT IN CURRENT CORE SCOPE`). The DG-02 hook already validates this
row-local interface synthetically and on the real model.

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

**Naming note.** The DG-02 spec calls the pre-intervention site `r_{ℓ,t}` (`= q + u_source`, line
528); the controller (§6) and NormPreserve read exactly that tensor. Below, `r_{ℓ,t}` is that
**pre-intervention site** and `r̃_{ℓ,t}` is the **repaired steered site**, matching the updated
proposal. (The older §2 used `q` for the pre-intervention site and `r` for the repaired one; the
tensors are identical, only the letters changed. `DG02_INTERVENTION_SITE_SPEC.md` §2 records both
conventions.)

| Symbol | Meaning |
|---|---|
| `r_{ℓ,t}` | the **pre-intervention site** (`q + u_source`, DG-02), the state the FFN would otherwise consume; the controller input is `LN(r_{ℓ,t})` |
| `d_{ℓ,t}` | the **position-dependent unit direction** `d_{ℓ,t} = normalize(V_ℓ^0 π_t)`, a mixture of the frozen basis columns (§4, §6) — rank-one per position |
| `\widetilde r_{ℓ,t}` (pre-repair) | the **nominal update** `r_{ℓ,t} + β·g_t·d_{ℓ,t}` before repair (§6) |
| `r̃_{ℓ,t}` | the **repaired steered site** `r̃_{ℓ,t} = NormPreserve(r_{ℓ,t}, r_{ℓ,t} + β·g_t·d_{ℓ,t})` (§7) |
| `u^S_{ℓ,t}` | the **effective update after repair**, `u^S_{ℓ,t} = r̃_{ℓ,t} - r_{ℓ,t}` |

`β` is the (optionally trainable) global strength, `g_t ∈ [0,1]` the controller gate, `π_t` the
controller mixture over the basis columns, `V_ℓ^0` the frozen basis (§4). The older per-layer
scalar scale `s_ℓ` and single direction `d_ℓ` are subsumed: `β` carries the strength and `d_{ℓ,t}`
the (position-dependent) direction. `apply_steering` remains the frozen update kernel; the caller
supplies `β·g_t` as the gain and `d_{ℓ,t}` as the direction.

Invariants (unchanged, DG-02):
- Zero-gain positions (`g_t = 0`) return **bit-identical** `r_{ℓ,t}` (`apply_steering` returns the
  input tensor where `gain == 0`).
- `β = 0` is a hard no-op (returns the input object) — verified on the real model (β=0 identity).
- The direction is added in the **same space it was constructed in** (§1), i.e. the exact site.

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

## 4. Canonical steering basis (LOCKED definitions — updated proposal)

All basis vectors live at the exact site of §1, at a candidate layer of §3, are constructed on
`D-construct` only, and are **versioned/hashed and frozen** before any confirmatory run. They are
read from `r_{ℓ,t}` (the pre-intervention site). Each vector is finite and unit-norm.

For each candidate layer `ℓ`:

**Raw language contrast** — difference of mean site states at baseline-correct positions:
```
v_ℓ^{raw} = μ_{ℓ,E}^{correct} − μ_{ℓ,M}^{correct}
```
`μ_{ℓ,E}^{correct}` = mean `r_{ℓ,t}` over baseline-correct **embedded-language** positions;
`μ_{ℓ,M}^{correct}` = mean over baseline-correct **matrix-language** positions (per the updated
proposal's position definitions).

**Language-conditioning direction** — same audio/reference prefix decoded under embedded- vs
matrix-language conditioning `c_E` / `c_M`:
```
v_ℓ^{cond} = normalize( E[ r_{ℓ,t}(c_E) − r_{ℓ,t}(c_M) ] )
```

**Conditioning-residualized local-language direction** — remove the conditioning component from
the raw contrast:
```
v_ℓ^{local} = normalize( v_ℓ^{raw} − ⟨v_ℓ^{raw}, v_ℓ^{cond}⟩ v_ℓ^{cond} )
```

**Initial basis** (frozen; the controller of §6 mixes its columns):
```
V_ℓ^0 = [ v_ℓ^{local}, v_ℓ^{cond} ]
```

### Terminology (conservative — LOCKED claim boundary)

- Call `v_ℓ^{local}` the **conditioning-residualized local-language direction**.
- Call `v_ℓ^{cond}` the **language-conditioning direction**.
- Do **not** call either vector a *pure acoustic direction*, a *universal language axis*, or a
  *causal language representation* (§11).
- **Do not silently relabel legacy `v_nat` as `v_local`.** Legacy `v_nat` was constructed at the
  **rejected post-FFN site** under legacy procedures; it differs in site, conditioning treatment,
  and construction. Legacy direction artifacts stay explicitly labeled `LEGACY DESIGN`.

### Repository mapping (reuse vs gap)

- Reusable exact-site primitives exist: `src/csasr/experiments/v2r3_directions.py`
  (`site_d_contrasts`, `orthonormal_basis`, `project_out`, `assemble`) over
  `DecoderPostCrossAttnRecorder`; `src/csasr/directions/controls.py` (`random_direction`,
  `wrong_sign`). `src/csasr/directions/decoder.py` records whole post-FFN block outputs and is
  `LEGACY DESIGN`.
- `IMPLEMENTATION GAP`: no canonical builder yet emits `v_raw`/`v_cond`/`v_local`/`V^0` in the
  **exact order above** (raw difference-of-means at baseline-correct positions → residualize
  against `v_cond` → assemble `V^0`). The v2r3 `assemble` projects before aggregating and must not
  be treated as equivalent. Building this canonical basis builder is DG-03.

Report at freeze time (DG-03): `cos(v_raw, v_local)`, energy fraction removed by conditioning
residualization, and dialogue-bootstrap cosine stability of each vector.

`OPEN DECISION`: the exact baseline-correct embedded/matrix position sets and any
nuisance/dialogue-balancing applied to the means (fix in DG-03 against the updated proposal).
`NOT IN CURRENT CORE SCOPE`: a wider conditioning **subspace** (rank >1) — the core basis is the
two named directions; the *per-position* steering direction `d_{ℓ,t}` is rank-one (§6).

---

## 5. Scores and disagreement — `NOT IN CURRENT CORE SCOPE` (reclassified DG-03R)

The separately-engineered **source score**, **decoder score**, and **encoder–decoder disagreement
score**, and the **disagreement repairability hypothesis**, are **removed from the core-paper
critical path**. The updated method replaces hand-designed positional scores with the learned
controller (§6), which reads only `LN(r_{ℓ,t})`.

- Encoder–decoder disagreement may still be reported as **`OPTIONAL SUPPORTING ANALYSIS`** (a
  mechanistic observation). It is **no longer a decision gate** before controller training, and no
  disagreement result blocks DG-03+.
- The temporal localizer, outcome-supervised utility selector, abstention selector, and factorized
  gate are **`LEGACY DESIGN`** (see §6). They are not prerequisites for the core method.

Preserved infrastructure (still valid, reusable): the inference-safe feature allowlist
`csasr.lss.features_contract.FEATURE_ALLOWLIST` (`ALLOWLIST_VERSION = "lss_features_v1"`) and its
`assert_inference_safe` guard remain the leakage guard for **any** inference-time signal. The core
controller trivially satisfies it — its only input is `LN(r_{ℓ,t})`, an inference-available site
state with no reference/alignment/oracle-location dependence.

---

## 6. Adaptive controller (LOCKED structure — updated proposal)

The core method replaces the factorized gate with a small **adaptive controller** that reads the
exact-site state and predicts both intervention strength and a mixture over the frozen basis:

```
(g_t, π_t) = f_θ( LN(r_{ℓ,t}) )
```

- `g_t ∈ [0,1]` — per-token intervention **strength gate** (e.g. sigmoid output);
- `π_t` — per-token **mixture** over the columns of `V_ℓ^0` (e.g. softmax / bounded weights);
- position-dependent direction: `d_{ℓ,t} = normalize(V_ℓ^0 π_t)` (rank-one per position);
- intervention: `r̃_{ℓ,t} = NormPreserve( r_{ℓ,t} + β g_t d_{ℓ,t} )` (§7).

Contract:
- `f_θ` is a small **bottleneck controller**; `LN` is a layer-norm on the site state.
- **No oracle CS location at inference.** The controller's only input is the inference-available
  site state `LN(r_{ℓ,t})`; it does not consume reference, alignment, or oracle-span signals.
- The controller runs **per row, per token** on the frozen DG-02 interface; a row's `(g_t, π_t)`
  come from that row's own `r_{ℓ,t}` (beam-safe by construction).
- `β` (global strength) may be a fixed hyperparameter or a single trainable scalar (§8).
- Invariants of §2 hold: `g_t = 0` ⇒ bit-identical `r_{ℓ,t}`; `β = 0` ⇒ no-op.

`DG-05A RESOLUTION:` the development controller is fixed as
`LayerNorm(1280) → Linear(1280,32) → GELU → Linear(32,3)`, with sigmoid `g_t`, softmax rank-2
`π_t`, and fixed `β` equal to the DG-04 reference operating point. The complete pre-training
specification is `docs/current/DG05_ADAPTIVE_CONTROLLER_SPEC.md`; no architecture/beta sweep is
permitted. This resolves the prior architecture/mixture/beta open decision for DG-05A only;
DG-06 may not silently change these inputs.

The DG-05A controller implementation is now present in `src/csasr/steering/controller.py` and
uses the exact-site dynamic action path. Legacy Job-A F5 (single post-FFN `v_nat`-projection
sigmoid) and Job-B T1 (monolithic two-projection sigmoid) remain **`LEGACY DESIGN`** and **must not**
be described as this controller. The end-to-end trained/free-decoding evidence remains pending
DG-05B.

### Reclassified as `LEGACY DESIGN` / `NOT IN CURRENT CORE SCOPE`

The following are **not prerequisites** for the core method and are off the critical path:
factorized gate; temporal localizer; source/decoder/disagreement score modules; outcome-supervised
utility selector; abstention selector; the disagreement repairability study (§5); and the
acoustic→decoder gate-transport `r_q = Σ_t Ā_{q,t} g_t`. They are preserved as history and may
reappear only as optional supporting analysis, never as a gate before controller training.

**Legacy `v_nat` identity (unchanged).** Codebase `v_nat` (`steer_sweep/trackb/`,
`experiments/job_b_training.py`, `experiments/round1_frozen.py`) is the post-FFN legacy direction;
it is **not** `v_ℓ^{local}` (§4). Code reading `v_nat` from the direction store is `LEGACY DESIGN`.

---

## 7. Norm-preserving repair (LOCKED)

After adding the update, the steered residual is renormalized to the pre-intervention norm. In the
updated-proposal notation (§2, `r` = pre-intervention site, `r̃` = repaired site):

```
\bar r_{ℓ,t} = r_{ℓ,t} + β g_t d_{ℓ,t}
if norm_preserve:  r̃_{ℓ,t} = \bar r_{ℓ,t} · ( ‖r_{ℓ,t}‖ / (‖\bar r_{ℓ,t}‖ + EPS) )
else:              r̃_{ℓ,t} = \bar r_{ℓ,t}
u^S_{ℓ,t} = r̃_{ℓ,t} - r_{ℓ,t}
```

- **FROZEN (DG-02).** Canonical implementation: `csasr.models.hooks.apply_steering(..., norm_preserve=True)`
  (`hooks.py:72`), used by `sites.DecoderPostCrossAttnInterventionHook` (and the legacy hooks /
  `steering.EncoderSteeringHook`). Real-model acceptance verified norm preservation at L16/L24.
- Invariants (§2): zero-gain positions bit-identical; `β=0` no-op.
- The frozen decoder-site hook (`sites.py`) applies norm preservation **without** any
  `sqrt(num_layers)` rescale (§1). `NormPreserve` must be applied identically across the proposed
  system and every steering control (matched-energy, random, sign-flip) so comparisons change the
  direction/location, not the energy.
- `IMPLEMENTATION GAP`: Round-1 frozen/training runners reimplement norm repair at the rejected
  post-FFN site and do not use the contract scale `s_ℓ = projection_std`; the frozen wrapper's
  reusable hook defaults to mean activation norm, while the prior Job-A/B drivers use unit scale.

---

## 8. Damage-aware training contract (LOCKED — updated proposal)

Trains **only** the controller `f_θ` (§6) and optionally the global scalar `β`; the backbone stays
frozen and the basis `V_ℓ^0` stays fixed (except the basis-refinement ablation below).

**Position sets** (on training data; baseline = frozen backbone, no intervention):
- **Correction set** `𝒞_E` = baseline-**wrong** embedded-language positions (what steering must repair).
- **Embedded retention set** `ℛ_E` = baseline-**correct** embedded-language positions.
- **Matrix retention set** `ℛ_M` = baseline-**correct** matrix-language positions.

**Objectives:**
```
𝓛_corr  = (1/|𝒞_E|) Σ_{t∈𝒞_E}  − log p_θ(y_t | x, y_{<t})
𝓛_ret,E = (1/|ℛ_E|) Σ_{t∈ℛ_E}  D_KL( p_{0,t} ‖ p_{θ,t} )
𝓛_ret,M = (1/|ℛ_M|) Σ_{t∈ℛ_M}  D_KL( p_{0,t} ‖ p_{θ,t} )
𝓛       = 𝓛_corr + λ_E 𝓛_ret,E + λ_M 𝓛_ret,M + λ_A 𝓛_anchor
```
- `p_{0,t}` = frozen-baseline next-token distribution; `p_{θ,t}` = distribution with the controller
  active. Retention is now an **explicit KL-to-baseline loss**, not only norm-preserve + measurement.
- `𝓛_corr` is **correction-set-only** cross-entropy (`IMPLEMENTATION GAP` vs Job-B's all-token CE at
  `experiments/job_b_training.py:1016`, which is `LEGACY DESIGN`).
- `𝓛_anchor = ‖ΔV_ℓ‖_F²` is active **only** for the basis-refinement ablation (below); for the main
  method `λ_A` is inactive.

**Controller variants:**
- **Main method:** `V_ℓ^0` fixed; train the bottleneck controller (gate + mixture outputs), optionally `β`.
- **Constrained-refinement ablation:** `V_ℓ = V_ℓ^0 + ΔV_ℓ` with anchor `𝓛_anchor = ‖ΔV_ℓ‖_F²`.
  This is an **ablation/variant, not the default**.
- **SALSA-style learned global vector:** an unconstrained learned global direction is a **baseline**,
  not the proposed method (§10 / EXPERIMENT_MATRIX).

**Optimization vs claims:**
- **Teacher forcing** is used for **optimization** (the losses above).
- **Free decoding** is used for **checkpoint selection and practical ASR claims** (MC §9).

**Staged training order** (DG-06): (1) correction-only → (2) + matrix retention → (3) + embedded
retention → (4) optional gate-coverage penalty *only if* measured controller coverage is
excessively broad → (5) optional constrained basis refinement. Do **not** claim sparsity unless
measured coverage is genuinely sparse.

**Data matching:** the controller (and any LoRA/SALSA baseline) is trained on the existing
controller-training pool `loc-train ∪ util-train` (physical role names preserved), never on
`D-dev-confirm` or `D-test` (DATA_EXPOSURE).

`OPEN DECISION` (DG-06): the weights `λ_E`, `λ_M` (and `λ_A` for refinement); whether a gate-coverage
penalty is added; staged vs joint schedule.

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

A gate that fails is a **result**, not a licence to relax the threshold. Code status semantics:
`passed` / `completed_no_go` (ran, answer is no) / `blocked` (evidence unavailable) / `failed`
(implementation defect) — `csasr.lss.gates`, `csasr.utils.status`. Only `passed` satisfies a
downstream prerequisite. The gate set is reconciled to the post-DG-02 sequence (DG-03…DG-08;
EXPERIMENT_MATRIX §A). The **main scientific comparison is the correction–retention–efficiency
trade-off**, visualized as the **correction–damage frontier** over steering strength / controller
operating points.

| Gate | Stage | Question | Pass condition |
|---|---|---|---|
| **A — Alignment** | supporting | Are local frame labels trustworthy? | coverage ≥0.95, invalid/nonmonotonic ≤0.01, ≥90% units usable, median boundary err ≤100 ms, p90 ≤200 ms, jitter stable |
| **B — Basis causal screen** | DG-03 | Does the frozen basis steer at the exact site? | intended-direction effect present at L16 and/or L24, and it **beats** sign-reversed, matched-norm random, and wrong-location controls; used to select L16 vs L24 and to confirm headroom |
| **RQ1 — Frozen-steering frontier** | DG-04 | Does any fixed local/mixture steering repair errors at acceptable damage? | a favorable point on the free-decoding correction–damage frontier; corrections > corruptions with dialogue-block bootstrap CI |
| **RQ2 — Adaptive controller** | DG-05/06 | Does the learned controller beat fixed steering on the frontier? | dominates or matches the best fixed operating point at equal/higher correction with lower damage; free-decoding |
| **RQ3 — Method comparison** | DG-07/08 | Competitive with SALSA-global / matched LoRA? | non-dominated on the correction–retention–efficiency trade-off |

`LEGACY DESIGN` / `NOT IN CURRENT CORE SCOPE` (removed as core gates): **T — decoder transport**,
**C — localizer recall**, **D — utility-selector value**, and the disagreement decision gate.
They may run only as optional supporting analysis and never block controller training.

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
- that the system is training-free (backbone frozen, but the **controller** is trained);
- that the direction is a pure or unique "language variable";
- that `v_local` / `v_cond` (§4) is a **pure acoustic direction**, a **universal language axis**, or
  a **causal language representation** — use "conditioning-residualized local-language direction" and
  "language-conditioning direction" only;
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

## 12. Open decisions requiring human judgment (index — reconciled DG-03R)

1. **Layer selection L16 vs L24** — resolved by the DG-03 basis causal screen (§10 Gate B), not
   pre-decided.
2. **Basis construction detail (§4):** the exact baseline-correct embedded/matrix position sets and
   any nuisance/dialogue-balancing of the means; resolve in DG-03 against the updated proposal.
3. **Controller (§6):** architecture/bottleneck width, `π_t` parameterization (simplex vs bounded),
   and whether `β` is fixed or trained; resolve in DG-05.
4. **Training weights (§8):** `λ_E`, `λ_M`, `λ_A`; whether a gate-coverage penalty is added; staged
   vs joint schedule; resolve in DG-06.
5. **Calibration split:** whether the controller needs a `router-calib` subdivision; only if a
   calibration/threshold step is introduced (DATA_EXPOSURE).

Resolved / superseded (no longer open for the core paper): practical deployable site (decoder exact
site is FROZEN, DG-02); conditioning-subspace rank (core basis is the two named directions, §4);
rank-1 vs rank-2 primary direction (per-position `d_{ℓ,t}` is rank-one; the basis is 2-column, §4);
feature→score aggregation and factorized-gate algebra (`LEGACY DESIGN`, §5–§6).

## 13. Implementation gaps (index — reconciled DG-03R)

1. `configs/lss/spec.yaml` still lists decoder layers `[8,16,24,31]`; contract permits `{16,24}` (§3).
2. No **canonical steering-basis builder** emits `v_raw`/`v_cond`/`v_local`/`V^0` in the frozen §4
   order at the exact site (v2r3 `assemble` projects before aggregating — not equivalent). [DG-03]
3. No **adaptive controller** `f_θ(LN(r)) → (g_t, π_t)` exists; legacy F5/T1 gates are `LEGACY DESIGN`
   and must not be relabeled as it (§6). [DG-05]
4. No **damage-aware training**: correction-set-only CE, explicit KL retention (`𝓛_ret,E`/`𝓛_ret,M`),
   and optional basis-refinement anchor are unimplemented; Job-B all-token CE is legacy (§8). [DG-06]
5. No **scientific exact-site free-decoding runner** for DG-03+ built on the frozen DG-02 hook; the
   legacy runners steer post-FFN (§1). [DG-03+]
6. Gate-coverage denominator remains deferred until legitimately defined by the controller (§12 of
   DG-01 spec / this §6); do not compute it before then.
7. Round-1 result directories lack the full config/environment/git/model/hash manifest (AGENTS.md);
   new DG-03+ runs must write one.

**Done / not gaps:** DG-01 canonical metrics/result schema/sign convention (§10) and the DG-02 exact
site + NormPreserve + hook are COMPLETE/FROZEN.
